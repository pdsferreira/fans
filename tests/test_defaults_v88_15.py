"""Numerical and launch regressions for the v88.15 scientific defaults."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import shlex
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
from astropy.cosmology import Planck15, Planck18

from spirality import carron_proxy_spirality as carron
from spirality import sconce_node_spirality as sc


ROOT = Path(__file__).resolve().parents[1]
SHELLS = ((0.110123456, 0.125987654), (0.280123456, 0.295987654),
          (0.480123456, 0.495987654))


def auxiliary_points(native="planck15"):
    """Small tangent-vector samples with deliberately nondefault distances."""
    frames = []
    ra = np.array([150.0, 150.08, 150.16, 150.0, 150.08, 150.16])
    dec = np.array([5.0, 5.0, 5.0, 5.08, 5.08, 5.08])
    for lo, hi in SHELLS:
        distance = np.asarray(Planck15.comoving_distance([lo, hi]).value)
        if native == "shifted":
            distance = np.asarray(Planck18.comoving_distance([lo, hi]).value) + 20.0
        frames.append(pd.DataFrame({
            "ra": ra, "dec": dec, "z_low": lo, "z_high": hi,
            "z_mid": 0.5 * (lo + hi),
            "comov_dist_low": distance[0], "comov_dist_high": distance[1],
            "grad_Dir1": -np.sin(np.deg2rad(ra)),
            "grad_Dir2": np.cos(np.deg2rad(ra)), "grad_Dir3": 0.0,
            "density": [1.0, 2.0, 3.0, 1.0, 2.0, 3.0],
            "unc_meas": 0.001, "bw": 0.01, "is_knot": False,
        }))
    return pd.concat(frames, ignore_index=True)


def carron_points(auxiliary):
    points = auxiliary[["ra", "dec", "z_low", "z_high", "density", "unc_meas"]].copy()
    points["z_low_key"] = points.z_low.round(8)
    points["z_high_key"] = points.z_high.round(8)
    points["angle"] = 0.0
    points["block"] = 1
    return points


def match_shells(auxiliary, ridges):
    return carron.match_carron_to_sconce_shells(
        auxiliary, ridges, min_overlap_fraction=0.5, max_center_offset_mpc=5.0)


@pytest.mark.parametrize("native", ["planck15", "shifted"])
def test_default_planck18_ignores_native_distances_and_legacy_parameters(native):
    points = auxiliary_points(native)
    original = points.copy(deep=True)
    default = sc.Config()
    assert default.distance_source == "planck18"
    # Legacy-mode knobs must not silently override the fixed Planck18 mode.
    legacy_knobs = replace(default, h=0.5, omega_m=0.1, fits_distance_units="mpc_h")
    for (lo, hi), shell in points.groupby(["z_low", "z_high"]):
        expected = np.asarray(Planck18.comoving_distance([lo, hi]).value)
        native_bounds = shell[["comov_dist_low", "comov_dist_high"]].iloc[0].to_numpy()
        assert not np.allclose(native_bounds, expected, rtol=1e-10, atol=1e-10)
        for cfg in (default, legacy_knobs):
            np.testing.assert_allclose(sc.sconce_shell_distance_bounds_mpc(shell, cfg),
                                       expected, rtol=1e-13, atol=1e-10)
            assert sc.slice_chi_from_sconce(shell, cfg) == pytest.approx(
                expected.mean() * Planck18.h, rel=1e-13, abs=1e-10)
    pd.testing.assert_frame_equal(points, original)


def test_default_auxiliary_and_carron_geometry_and_shell_matching_agree():
    points = auxiliary_points("shifted")
    ridges = carron_points(points)
    cfg = sc.Config(edge_mode="none")
    aux_shells = carron.build_sconce_shell_table(sc, points, cfg)
    ridge_shells = carron.build_carron_shell_table(ridges)
    columns = ["chi_low_mpc", "chi_high_mpc", "chi_center_mpc", "chi_center_mpc_h"]
    np.testing.assert_allclose(aux_shells[columns], ridge_shells[columns],
                               rtol=1e-13, atol=1e-10)
    matches = match_shells(aux_shells, ridge_shells)
    assert matches.accepted.all()
    np.testing.assert_array_equal(matches.sconce_shell_id, matches.carron_shell_id)
    np.testing.assert_allclose(matches.abs_delta_chi_center_mpc, 0.0, atol=1e-10)
    np.testing.assert_allclose(matches.overlap_fraction_of_narrower, 1.0, atol=1e-12)

    aux_slices = sc.build_sconce_slices(points, cfg, verbose=False)
    ridge_slices = carron.build_carron_slices_for_sconce_keys(
        sc, ridges, matches, cfg, verbose=False)
    assert set(aux_slices) == set(ridge_slices) == {round(lo, 5) for lo, _ in SHELLS}
    for key, auxiliary in aux_slices.items():
        ridge = ridge_slices[key]
        assert auxiliary.chi_mpc_h == pytest.approx(ridge.chi_mpc_h, rel=1e-13)
        np.testing.assert_allclose(auxiliary.x_mpc_h, ridge.x_mpc_h, rtol=1e-13, atol=1e-12)
        np.testing.assert_allclose(auxiliary.y_mpc_h, ridge.y_mpc_h, rtol=1e-13, atol=1e-12)


def test_explicit_fits_mode_respects_native_distances_and_mpc_h_conversion():
    points_mpc = auxiliary_points("shifted")
    cfg_mpc = sc.Config(distance_source="fits", fits_distance_units="mpc", h=0.63)
    cfg_h = replace(cfg_mpc, fits_distance_units="mpc_h")
    points_h = points_mpc.copy(deep=True)
    points_h[["comov_dist_low", "comov_dist_high"]] *= cfg_mpc.h
    for (lo, hi), shell_mpc in points_mpc.groupby(["z_low", "z_high"]):
        shell_h = points_h.loc[points_h.z_low.eq(lo)]
        native = shell_mpc[["comov_dist_low", "comov_dist_high"]].iloc[0].to_numpy()
        for shell, cfg in ((shell_mpc, cfg_mpc), (shell_h, cfg_h)):
            np.testing.assert_allclose(sc.sconce_shell_distance_bounds_mpc(shell, cfg),
                                       native, rtol=1e-13, atol=1e-10)
            assert sc.slice_chi_from_sconce(shell, cfg) == pytest.approx(
                native.mean() * cfg_mpc.h, rel=1e-13)
    shells_mpc = carron.build_sconce_shell_table(sc, points_mpc, cfg_mpc)
    shells_h = carron.build_sconce_shell_table(sc, points_h, cfg_h)
    pd.testing.assert_frame_equal(shells_mpc, shells_h, check_exact=False,
                                  rtol=1e-13, atol=1e-10)
    ridges = carron.build_carron_shell_table(carron_points(points_mpc))
    matches_mpc = match_shells(shells_mpc, ridges)
    matches_h = match_shells(shells_h, ridges)
    pd.testing.assert_frame_equal(matches_mpc, matches_h, check_exact=False,
                                  rtol=1e-13, atol=1e-10)
    # The native +20 Mpc offset is retained: overlap is adequate, but centres fail.
    assert (matches_mpc.overlap_fraction_of_narrower >= 0.5).all()
    assert not matches_mpc.accepted.any()
    np.testing.assert_allclose(matches_mpc.abs_delta_chi_center_mpc, 20.0, atol=1e-10)


def test_cosmology_modes_preserve_original_redshift_boundaries_and_shell_keys():
    points = auxiliary_points()
    original = points.copy(deep=True)
    expected = pd.DataFrame({
        "z_low": [lo for lo, _ in SHELLS],
        "z_high": [hi for _, hi in SHELLS],
        "z_mid": [0.5 * (lo + hi) for lo, hi in SHELLS],
        "sconce_z_key": [round(lo, 5) for lo, _ in SHELLS],
    })
    for mode in ("planck18", "fits", "zmid"):
        shells = carron.build_sconce_shell_table(sc, points, sc.Config(distance_source=mode))
        pd.testing.assert_frame_equal(shells[list(expected.columns)], expected)
    pd.testing.assert_frame_equal(points, original)


@pytest.mark.parametrize("module", [sc, carron])
def test_node_command_line_defaults_use_planck18_and_1000_bootstraps(module, monkeypatch):
    argv = ["program", "--sconce-fits", "unused.fits", "--outdir", "unused-output"]
    if module is carron:
        argv += ["--carron-files", "Block1.csv", "Block2.csv", "Block3.csv"]
    monkeypatch.setattr(sys, "argv", argv)
    args = module.parse_args()
    assert args.distance_source == "planck18"
    assert args.fits_distance_units == "mpc"
    assert args.h == pytest.approx(Planck18.h)
    assert args.omega_m == pytest.approx(Planck18.Om0)
    assert args.bootstrap_B == 1000


@pytest.mark.parametrize("mode", ["fiducial", "grid"])
def test_dry_run_propagates_1000_bootstraps_and_skips_duplicate_grid_draws(mode, tmp_path):
    outroot = tmp_path / "results"
    missing_data = tmp_path / "unavailable-inputs"
    stage = "all" if mode == "fiducial" else "projected"
    proc = subprocess.run([
        sys.executable, str(ROOT / "scripts/run_referee_revision.py"), mode,
        "--outroot", str(outroot), "--data-dir", str(missing_data),
        "--stage", stage, "--dry-run", "--skip-fits-export", "--skip-paper-products",
    ], cwd=ROOT, text=True, capture_output=True, check=True, timeout=30)
    commands = [shlex.split(line) for line in proc.stdout.splitlines()
                if "scripts/" in line]

    def arguments_for(script):
        return [args for args in commands if len(args) > 1 and args[1] == f"scripts/{script}"]

    def option(args, name):
        return args[args.index(name) + 1]

    reconstruction = arguments_for("run_spirality_pipeline.py")
    multiplicity = arguments_for("analyze_fans_multiplicity.py")
    extrinsic = arguments_for("run_extrinsic_spirality.py")
    expected_runs = 1 if mode == "fiducial" else 27
    assert len(reconstruction) == len(multiplicity) == expected_runs
    assert all(option(args, "--bootstrap-B") == "1000" for args in multiplicity)
    assert all(option(args, "--null-B") == "1000" for args in multiplicity)
    preliminary_b = "1000" if mode == "fiducial" else "0"
    assert all(option(args, "--bootstrap-B") == preliminary_b for args in reconstruction)
    if mode == "fiducial":
        assert len(extrinsic) == 1
        assert option(extrinsic[0], "--bootstrap-B") == "1000"
    else:
        assert not extrinsic
        assert all(option(args, "--signflip-B") == "0" for args in reconstruction)
    assert not outroot.exists()
    assert not missing_data.exists()
