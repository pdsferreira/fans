#!/usr/bin/env python3
"""
sconce_carron_proxy_spirality.py

Hybrid consistency test:
  - Use SCONCE/DR17 knot positions to define nodes/intersections.
  - Measure filament-arm projected winding around the same nodes using:
      (A) SCONCE ridge points/tangents
      (B) Carrón ridge points/angles, matched shell-by-shell in comoving distance.

This is intended to test whether the tomographic handedness/spirality signal is
specific to the SCONCE ridge/tangent reconstruction or is also recovered using
the Carrón filament ridges around the same SCONCE-defined nodes.

Key consistency choices:
  * The same SCONCE knot-clustered node catalogue is used for both measurements.
  * Carrón shells are matched to SCONCE shells by comoving-distance overlap.
  * Nodes are processed only in SCONCE shells with an accepted Carrón-shell match.
  * Local arm geometry is measured with the same node-link radius, arm-opening
    angle, radial segments, and point weighting.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import time
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd

try:
    from astropy.cosmology import Planck18
except ImportError:
    Planck18 = None



# -----------------------------------------------------------------------------
# Dynamic import of the SCONCE spirality pipeline helpers
# -----------------------------------------------------------------------------

def import_sconce_pipeline(path: Optional[str]):
    if path is None:
        path = str(Path(__file__).resolve().parent / "sconce_node_spirality.py")
    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(
            f"Could not find SCONCE pipeline helper: {p}\n"
            "Put sconce_node_spirality.py in the same directory as this script, "
            "or pass --sconce-pipeline-script /full/path/to/sconce_node_spirality.py"
        )
    spec = importlib.util.spec_from_file_location("sconce_spirality_helpers", str(p))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import helper module from {p}")
    mod = importlib.util.module_from_spec(spec)
    # dataclasses needs the module to be registered in sys.modules during
    # execution, otherwise @dataclass can fail when imported dynamically.
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def cosmo_h(cosmo) -> float:
    return float(cosmo.H0.value) / 100.0


def comoving_mpc(cosmo, z: np.ndarray) -> np.ndarray:
    return np.asarray(cosmo.comoving_distance(np.asarray(z, dtype=float)).value, dtype=float)


# -----------------------------------------------------------------------------
# Carrón catalogue I/O and shell matching
# -----------------------------------------------------------------------------

def _norm_col(c: str) -> str:
    return str(c).strip().strip('"').strip("'").lower()


def read_carron_blocks(paths: Iterable[str | Path], *, z_min: Optional[float], z_max: Optional[float]) -> pd.DataFrame:
    frames = []
    use = {"ra", "dec", "dens", "unc", "grad_ra", "grad_dec", "angle", "z_low", "z_high", "ini_dens"}
    for ib, p in enumerate(paths, start=1):
        p = Path(p).expanduser()
        if not p.exists():
            raise FileNotFoundError(p)
        df = pd.read_csv(p)
        colmap = {c: _norm_col(c) for c in df.columns}
        inv = {v: k for k, v in colmap.items()}
        missing = [c for c in ["ra", "dec", "angle", "z_low", "z_high"] if c not in inv]
        if missing:
            raise ValueError(f"Carrón file {p} is missing required columns {missing}. Columns={list(df.columns)}")
        keep_cols = [inv[c] for c in use if c in inv]
        sub = df[keep_cols].rename(columns={inv[c]: c for c in use if c in inv}).copy()
        sub["block"] = ib
        sub["source_file"] = str(p)
        frames.append(sub)
    out = pd.concat(frames, ignore_index=True)
    for c in ["ra", "dec", "angle", "z_low", "z_high"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    if "dens" in out.columns:
        out["density"] = pd.to_numeric(out["dens"], errors="coerce")
    else:
        out["density"] = 1.0
    if "unc" in out.columns:
        out["unc_meas"] = pd.to_numeric(out["unc"], errors="coerce")
    else:
        out["unc_meas"] = np.nan

    ok = np.isfinite(out["ra"]) & np.isfinite(out["dec"]) & np.isfinite(out["angle"]) & np.isfinite(out["z_low"]) & np.isfinite(out["z_high"])
    if z_min is not None:
        ok &= out["z_high"].to_numpy(float) > float(z_min)
    if z_max is not None:
        ok &= out["z_low"].to_numpy(float) < float(z_max)
    out = out.loc[ok].reset_index(drop=True)
    out["z_low_key"] = out["z_low"].round(8)
    out["z_high_key"] = out["z_high"].round(8)
    return out


def build_sconce_shell_table(sc, fmaps: pd.DataFrame, cfg) -> pd.DataFrame:
    rows = []
    for z_raw, sub0 in fmaps.groupby(fmaps["z_low"].map(sc.zkey)):
        sub = sub0.copy()
        z_low = float(np.nanmedian(sub["z_low"].to_numpy(float)))
        z_high = float(np.nanmedian(sub["z_high"].to_numpy(float))) if "z_high" in sub.columns else np.nan
        z_mid = float(np.nanmedian(sub["z_mid"].to_numpy(float))) if "z_mid" in sub.columns else 0.5 * (z_low + z_high)
        chi_low_mpc, chi_high_mpc = sc.sconce_shell_distance_bounds_mpc(sub, cfg)
        rows.append({
            "sconce_z_key": float(z_raw),
            "z_low": z_low,
            "z_high": z_high,
            "z_mid": z_mid,
            "chi_low_mpc": chi_low_mpc,
            "chi_high_mpc": chi_high_mpc,
            "chi_center_mpc": 0.5 * (chi_low_mpc + chi_high_mpc),
            "chi_width_mpc": chi_high_mpc - chi_low_mpc,
            "chi_center_mpc_h": float(sc.slice_chi_from_sconce(sub, cfg)),
            "n_sconce_points": int(len(sub)),
        })
    sh = pd.DataFrame(rows).sort_values("chi_center_mpc").reset_index(drop=True)
    sh.insert(0, "sconce_shell_id", np.arange(len(sh), dtype=int))
    return sh


def build_carron_shell_table(carron: pd.DataFrame, *, cosmo=Planck18) -> pd.DataFrame:
    rows = []
    h = cosmo_h(cosmo)
    for (zlo_key, zhi_key), sub in carron.groupby(["z_low_key", "z_high_key"]):
        zlo = float(np.nanmedian(sub["z_low"].to_numpy(float)))
        zhi = float(np.nanmedian(sub["z_high"].to_numpy(float)))
        chi_lo = float(cosmo.comoving_distance(zlo).value)
        chi_hi = float(cosmo.comoving_distance(zhi).value)
        rows.append({
            "carron_z_low_key": float(zlo_key),
            "carron_z_high_key": float(zhi_key),
            "z_low": zlo,
            "z_high": zhi,
            "z_mid": 0.5 * (zlo + zhi),
            "chi_low_mpc": chi_lo,
            "chi_high_mpc": chi_hi,
            "chi_center_mpc": 0.5 * (chi_lo + chi_hi),
            "chi_width_mpc": chi_hi - chi_lo,
            "chi_center_mpc_h": 0.5 * (chi_lo + chi_hi) * h,
            "n_carron_points": int(len(sub)),
            "blocks": ",".join(map(str, sorted(set(sub["block"].astype(int))))),
        })
    sh = pd.DataFrame(rows).sort_values("chi_center_mpc").reset_index(drop=True)
    sh.insert(0, "carron_shell_id", np.arange(len(sh), dtype=int))
    return sh


def match_carron_to_sconce_shells(
    sconce_shells: pd.DataFrame,
    carron_shells: pd.DataFrame,
    *,
    min_overlap_fraction: float,
    max_center_offset_mpc: float,
) -> pd.DataFrame:
    rows = []
    c_cent = carron_shells["chi_center_mpc"].to_numpy(float)
    for _, s in sconce_shells.iterrows():
        j = int(np.nanargmin(np.abs(c_cent - float(s["chi_center_mpc"]))))
        c = carron_shells.iloc[j]
        overlap = max(
            0.0,
            min(float(s["chi_high_mpc"]), float(c["chi_high_mpc"])) -
            max(float(s["chi_low_mpc"]), float(c["chi_low_mpc"]))
        )
        min_width = min(float(s["chi_width_mpc"]), float(c["chi_width_mpc"]))
        frac = overlap / min_width if min_width > 0 else np.nan
        dchi = float(c["chi_center_mpc"]) - float(s["chi_center_mpc"])
        accept = bool(
            np.isfinite(frac) and
            frac >= float(min_overlap_fraction) and
            abs(dchi) <= float(max_center_offset_mpc)
        )
        rows.append({
            "accepted": accept,
            "sconce_shell_id": int(s["sconce_shell_id"]),
            "sconce_z_key": float(s["sconce_z_key"]),
            "sconce_z_low": float(s["z_low"]),
            "sconce_z_high": float(s["z_high"]),
            "sconce_chi_low_mpc": float(s["chi_low_mpc"]),
            "sconce_chi_high_mpc": float(s["chi_high_mpc"]),
            "sconce_chi_center_mpc": float(s["chi_center_mpc"]),
            "sconce_n_points": int(s["n_sconce_points"]),
            "carron_shell_id": int(c["carron_shell_id"]),
            "carron_z_low_key": float(c["carron_z_low_key"]),
            "carron_z_high_key": float(c["carron_z_high_key"]),
            "carron_z_low": float(c["z_low"]),
            "carron_z_high": float(c["z_high"]),
            "carron_chi_low_mpc": float(c["chi_low_mpc"]),
            "carron_chi_high_mpc": float(c["chi_high_mpc"]),
            "carron_chi_center_mpc": float(c["chi_center_mpc"]),
            "carron_n_points": int(c["n_carron_points"]),
            "carron_blocks": str(c.get("blocks", "")),
            "delta_chi_center_mpc_carron_minus_sconce": dchi,
            "abs_delta_chi_center_mpc": abs(dchi),
            "overlap_mpc": overlap,
            "overlap_fraction_of_narrower": frac,
        })
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Carrón slice construction
# -----------------------------------------------------------------------------

def carron_EN_from_angle(angle_deg: np.ndarray, *, angle_offset_deg: float = 0.0, flip_sign: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """Convert Carrón 'angle with parallels, counter-clockwise' to East/North components.

    We interpret the parallel direction as local East. Thus angle=0 means along
    increasing RA/east, and positive angles rotate toward local North.
    """
    ang = (np.asarray(angle_deg, dtype=float) + float(angle_offset_deg)) * np.pi / 180.0
    E = np.cos(ang)
    N = np.sin(ang)
    if flip_sign:
        E = -E
        N = -N
    norm = np.hypot(E, N)
    ok = norm > 0
    return np.where(ok, E / norm, np.nan), np.where(ok, N / norm, np.nan)


def build_carron_slices_for_sconce_keys(
    sc,
    carron: pd.DataFrame,
    matches: pd.DataFrame,
    cfg,
    *,
    angle_offset_deg: float = 0.0,
    flip_angle_sign: bool = False,
    verbose: bool = True,
) -> Dict[float, object]:
    """Build SliceData dictionaries keyed by SCONCE z_key, but using Carrón points."""
    out: Dict[float, object] = {}
    rng = np.random.default_rng(cfg.seed + 913)
    accepted = matches[matches["accepted"].astype(bool)].copy()
    h18 = cosmo_h(Planck18)
    for _, m in accepted.iterrows():
        s_key = float(m["sconce_z_key"])
        zlo_key = float(m["carron_z_low_key"])
        zhi_key = float(m["carron_z_high_key"])
        mask = np.isclose(carron["z_low_key"].to_numpy(float), zlo_key, atol=5e-9) & np.isclose(carron["z_high_key"].to_numpy(float), zhi_key, atol=5e-9)
        sub = carron.loc[mask].copy().reset_index(drop=True)
        if len(sub) == 0:
            continue

        chi_mpc_h = float(m["carron_chi_center_mpc"]) * h18
        ra0 = float(np.nanmedian(sub["ra"].to_numpy(float)))
        dec0 = float(np.nanmedian(sub["dec"].to_numpy(float)))
        x, y = sc.project_slice_xy_mpc_h(
            sub["ra"].to_numpy(float),
            sub["dec"].to_numpy(float),
            ra0_deg=ra0,
            dec0_deg=dec0,
            chi_mpc_h=chi_mpc_h,
        )
        E, N = carron_EN_from_angle(
            sub["angle"].to_numpy(float),
            angle_offset_deg=angle_offset_deg,
            flip_sign=flip_angle_sign,
        )
        density = sub["density"].to_numpy(float)
        uncertainty = sub["unc_meas"].to_numpy(float) if "unc_meas" in sub.columns else np.full(len(sub), np.nan, dtype=float)
        good = np.isfinite(x) & np.isfinite(y) & np.isfinite(E) & np.isfinite(N)
        sub = sub.loc[good].reset_index(drop=True)
        x = x[good]; y = y[good]; E = E[good]; N = N[good]; density = density[good]; uncertainty = uncertainty[good]
        if len(sub) == 0:
            continue
        tree = sc.cKDTree(np.column_stack([x, y]))
        hull_eq = None
        if cfg.edge_mode == "convexhull":
            hull_eq = sc.build_hull_equations(x, y, rng=rng)
        out[s_key] = sc.SliceData(
            z_key=s_key,
            z_value=float(m["sconce_z_low"]),
            chi_mpc_h=chi_mpc_h,
            ra0_deg=ra0,
            dec0_deg=dec0,
            points=sub,
            x_mpc_h=x,
            y_mpc_h=y,
            e_hat=E,
            n_hat=N,
            density=density,
            uncertainty=uncertainty,
            tree=tree,
            hull_equations=hull_eq,
        )
        if verbose:
            print(
                f"[Carron slice] SCONCE z=[{m['sconce_z_low']:.5f},{m['sconce_z_high']:.5f}] "
                f"<- Carron z=[{m['carron_z_low']:.5f},{m['carron_z_high']:.5f}], "
                f"points={len(sub):,}, dchi={m['delta_chi_center_mpc_carron_minus_sconce']:.2f} Mpc, "
                f"overlap={m['overlap_fraction_of_narrower']:.3f}"
            )
    return out


# -----------------------------------------------------------------------------
# Node processing and output
# -----------------------------------------------------------------------------

def process_nodes_with_slices(
    sc,
    nodes: pd.DataFrame,
    slices: Dict[float, object],
    cfg,
    *,
    n_jobs: int,
    timing_interval: int,
    catalog_label: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if "z_key" not in nodes.columns:
        nodes = nodes.copy()
        nodes["z_key"] = nodes["z_low"].map(sc.zkey)
    has_slice = nodes["z_key"].isin(slices.keys())
    if not np.all(has_slice):
        print(f"[{catalog_label}] marking {np.count_nonzero(~has_slice):,} nodes as unassessed (no matched slice)")
    work = nodes.loc[has_slice].reset_index(drop=True)
    print(f"[{catalog_label}] processing {len(work):,} nodes with n_jobs={n_jobs}")

    node_rows: List[dict] = []
    for _, missing in nodes.loc[~has_slice].iterrows():
        row = missing.to_dict()
        row.update(status="no_matched_slice", n_arms_valid=np.nan,
                   audit_evaluated=False, audit_trace_evaluated=False,
                   audit_m_candidates=np.nan, audit_m_valid=np.nan,
                   audit_reason="no_matched_slice", ridge_catalog=catalog_label)
        node_rows.append(row)
    arm_rows: List[dict] = []
    seg_rows: List[dict] = []

    def run_one(i: int):
        t0 = time.perf_counter()
        row = work.iloc[i]
        sd = slices[float(row["z_key"])]
        nr, ar, sr = sc.process_one_node(row, sd, cfg)
        nr["node_wall_time_s"] = float(time.perf_counter() - t0)
        nr["ridge_catalog"] = catalog_label
        for col in [
            "n_knot_points_clustered", "cluster_radius_mpc_h", "cluster_rms_mpc_h",
            "mean_knot_density", "mean_unc_meas", "z_mid", "z_high",
            "gal_l_deg", "gal_b_deg", "gal_hemi",
        ]:
            if col in row.index:
                nr[col] = row[col]
        for a in ar:
            a["ridge_catalog"] = catalog_label
        for s in sr:
            s["ridge_catalog"] = catalog_label
        return nr, ar, sr

    t0 = time.perf_counter()
    if int(n_jobs) <= 1:
        for i in range(len(work)):
            nr, ar, sr = run_one(i)
            node_rows.append(nr); arm_rows.extend(ar); seg_rows.extend(sr)
            if timing_interval and (i + 1) % int(timing_interval) == 0:
                elapsed = time.perf_counter() - t0
                print(f"  [{catalog_label}] processed {i+1:,}/{len(work):,} | {elapsed/60:.2f} min")
    else:
        with ThreadPoolExecutor(max_workers=int(n_jobs)) as ex:
            futs = {ex.submit(run_one, i): i for i in range(len(work))}
            done = 0
            for fut in as_completed(futs):
                nr, ar, sr = fut.result()
                node_rows.append(nr); arm_rows.extend(ar); seg_rows.extend(sr)
                done += 1
                if timing_interval and done % int(timing_interval) == 0:
                    elapsed = time.perf_counter() - t0
                    rate = done / elapsed if elapsed > 0 else np.nan
                    eta = (len(work) - done) / rate if rate > 0 else np.nan
                    print(f"  [{catalog_label}] processed {done:,}/{len(work):,} | {elapsed/60:.2f} min | ETA={eta/60:.2f} min")

    node_df = pd.DataFrame(node_rows).sort_values("node_id").reset_index(drop=True) if node_rows else pd.DataFrame()
    arm_df = pd.DataFrame(arm_rows).sort_values(["node_id", "arm_local_id"]).reset_index(drop=True) if arm_rows else pd.DataFrame()
    seg_df = pd.DataFrame(seg_rows).sort_values(["node_id", "arm_local_id", "arm_segment_index"]).reset_index(drop=True) if seg_rows else pd.DataFrame()
    return node_df, arm_df, seg_df


def make_product_args(args: argparse.Namespace, *, label: str, comparison_node_summary: Optional[str] = None, comparison_arm_summary: Optional[str] = None, comparison_label: str = "") -> argparse.Namespace:
    return argparse.Namespace(
        zbin_edges=args.zbin_edges,
        zbin_nbins=0,
        zbin_min=None,
        zbin_max=None,
        zbin_plot_format=args.plot_format,
        plot_format=args.plot_format,
        dr12_node_zbin_summary=comparison_node_summary,
        dr12_arm_zbin_summary=comparison_arm_summary,
        dr17_label=label,
        dr12_label=comparison_label or "Comparison",
        bootstrap_B=args.bootstrap_B,
    )


def write_catalog_outputs(
    sc,
    *,
    outdir: Path,
    label: str,
    node_df: pd.DataFrame,
    arm_df: pd.DataFrame,
    seg_df: pd.DataFrame,
    cfg,
    args: argparse.Namespace,
) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    from spirality.selection import write_construction_diagnostics
    write_construction_diagnostics(node_df, outdir)
    node_df.to_csv(outdir / "node_spirality.csv", index=False)
    arm_df.to_csv(outdir / "arm_spirality.csv", index=False)
    seg_df.to_csv(outdir / "arm_segments.csv", index=False)
    print(f"[write] {outdir / 'node_spirality.csv'}")
    print(f"[write] {outdir / 'arm_spirality.csv'}")
    print(f"[write] {outdir / 'arm_segments.csv'}")

    pargs = make_product_args(args, label=label)
    sc.write_node_status_summaries(node_df, pargs, outdir)

    summary = sc.build_global_summary(node_df, cfg)
    with open(outdir / "global_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    pd.DataFrame([summary]).to_csv(outdir / "global_summary.csv", index=False)

    ghemi = sc.build_global_summary_by_gal_hemi(node_df, cfg)
    ghemi.to_csv(outdir / "global_summary_by_gal_hemi.csv", index=False)

    sc.write_zbin_products(node_df, cfg, pargs, outdir)
    sc.write_redshift_handedness_trend_summary(node_df, cfg, pargs, outdir, label="nodes")

    if len(arm_df) > 0:
        sc.write_arm_level_products(arm_df, node_df, cfg, pargs, outdir)
        arm_tab = sc.make_arm_level_table(arm_df, node_df)
        sc.write_redshift_handedness_trend_summary(arm_tab, cfg, pargs, outdir, label="arms")
        sc.write_arm_direction_count_products(arm_tab, pargs, outdir)


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--sconce-pipeline-script", default=None,
                   help="Path to sconce_node_spirality.py. Default: same directory as this script.")
    p.add_argument("--sconce-fits", required=True)
    p.add_argument("--carron-files", nargs="+", required=True)
    p.add_argument("--outdir", required=True)

    p.add_argument("--z-min", type=float, default=0.1)
    p.add_argument("--z-max", type=float, default=0.6)
    p.add_argument("--z-selection", default="mid", choices=["mid", "low", "high"])
    p.add_argument("--zbin-edges", default="0.1,0.225,0.35,0.475,0.6")

    p.add_argument("--segment-length-mpc-h", type=float, default=10.0)
    p.add_argument("--n-arm-segments", type=int, default=3)
    p.add_argument("--node-link-radius-mpc-h", type=float, default=8.0)
    p.add_argument("--inner-exclusion-mpc-h", type=float, default=0.0)
    p.add_argument("--arm-opening-deg", type=float, default=35.0)
    p.add_argument("--cluster-gap-deg", type=float, default=35.0)
    p.add_argument("--seed-max-tangent-angle-deg", type=float, default=75.0)
    p.add_argument("--min-seed-points", type=int, default=2)
    p.add_argument("--min-points-per-arm-segment", type=int, default=3)
    p.add_argument("--min-valid-arm-segments", type=int, default=2)
    p.add_argument("--min-arms-node", type=int, default=2)
    p.add_argument("--max-arms-node", type=int, default=12)

    p.add_argument("--edge-margin-mpc-h", type=float, default=70.0)
    p.add_argument("--edge-mode", default="convexhull", choices=["convexhull", "none"])

    p.add_argument("--point-weight", default="density", choices=["uniform", "density", "sqrt_density", "log_density", "inv_unc2", "density_inv_unc2"])
    p.add_argument("--point-unc-floor", type=float, default=1e-3, help="Floor added in quadrature to Carrón unc for inv_unc2 point weighting")
    p.add_argument("--point-unc-max-weight", type=float, default=1e6, help="Maximum inverse-uncertainty point weight")
    p.add_argument("--node-weight", default="uniform", choices=["uniform", "mean_density", "sqrt_mean_density", "n_segments", "sqrt_points", "inv_var"])
    p.add_argument("--sconce-gradient-mode", default="tangent", choices=["tangent", "perp"])

    p.add_argument("--distance-source", default="planck18", choices=["planck18", "zmid", "fits"],
                   help="Default: recompute auxiliary shell-boundary distances with Astropy Planck18; zmid/fits are explicit legacy modes.")
    p.add_argument("--fits-distance-units", default="mpc", choices=["mpc", "mpc_h"],
                   help="SCONCE comov_dist column units. Catalogue native is Mpc.")
    p.add_argument("--h", type=float, default=float(cosmo_h(Planck18)), help="Distance conversion for legacy zmid/fits modes; ignored in fixed Planck18 mode")
    p.add_argument("--omega-m", type=float, default=float(Planck18.Om0), help="Matter density for legacy zmid mode; ignored in fixed Planck18 mode")

    p.add_argument("--cluster-knots", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--knot-cluster-radius-mpc-h", type=float, default=5.0)
    p.add_argument("--knot-cluster-weight", default="density", choices=["uniform", "density", "inv_unc2"])
    p.add_argument("--min-knot-points-per-node", type=int, default=1)

    p.add_argument("--match-min-overlap", type=float, default=0.5)
    p.add_argument("--match-max-dchi-mpc", type=float, default=5.0)
    p.add_argument("--carron-angle-offset-deg", type=float, default=0.0,
                   help="Diagnostic rotation applied to Carrón angles before measuring arms.")
    p.add_argument("--carron-flip-angle-sign", action="store_true",
                   help="Diagnostic: multiply Carrón tangent vector by -1 before outward orientation. Should not affect axial results.")

    p.add_argument("--bootstrap-B", type=int, default=1000)
    p.add_argument("--signflip-B", type=int, default=2000)
    p.add_argument("--entropy-bins", type=int, default=31)
    p.add_argument("--arm-angle-entropy-bins", type=int, default=8)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--n-jobs", type=int, default=32)
    p.add_argument("--timing-interval", type=int, default=1000)
    p.add_argument("--plot-format", default="pdf", choices=["pdf", "png", "jpg"])

    p.add_argument("--make-gallery", action="store_true", help="Save native Carrón-proxy optimal/typical node galleries using the same in-memory slice/KDTree logic as SCONCE.")
    p.add_argument("--gallery-seed", type=int, default=None, help="Random seed for gallery selection; defaults to --seed")
    p.add_argument("--gallery-ncols", type=int, default=8)
    p.add_argument("--gallery-nrows", type=int, default=12)
    p.add_argument("--gallery-optimal-pool-frac", type=float, default=0.02)
    p.add_argument("--gallery-optimal-arms", type=int, default=0)
    p.add_argument("--gallery-optimal-min-arms", type=int, default=3)
    p.add_argument("--gallery-allow-mixed-arm-sign", action="store_true")
    p.add_argument("--gallery-filename", default="carron_node_spiral_gallery")
    p.add_argument("--make-random-fans-gallery", action="store_true",
                   help="Save individual random FANS PNGs using the native gallery plotter.")
    p.add_argument("--random-fans-gallery-outdir", default="random_fans_gallery")
    p.add_argument("--random-fans-gallery-catalog-key", default="carron")
    p.add_argument("--random-fans-gallery-n-per-arm-count", type=int, default=1000)
    p.add_argument("--random-fans-gallery-arm-counts", nargs="+", type=int, default=[2, 3])
    p.add_argument("--skip-sconce-ridge-measurement", action="store_true",
                   help="Only run Carrón ridges around SCONCE nodes; comparison plots require previous SCONCE outputs.")
    return p.parse_args()


def main() -> None:
    t0 = time.perf_counter()
    args = parse_args()
    if Planck18 is None:
        raise SystemExit("Carrón shell matching requires astropy; install requirements.txt")
    sc = import_sconce_pipeline(args.sconce_pipeline_script)

    outdir = Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    cfg = sc.Config(
        segment_length_mpc_h=args.segment_length_mpc_h,
        n_arm_segments=args.n_arm_segments,
        node_link_radius_mpc_h=args.node_link_radius_mpc_h,
        inner_exclusion_mpc_h=args.inner_exclusion_mpc_h,
        arm_opening_deg=args.arm_opening_deg,
        cluster_gap_deg=args.cluster_gap_deg,
        seed_max_tangent_angle_deg=args.seed_max_tangent_angle_deg,
        min_seed_points=args.min_seed_points,
        min_points_per_arm_segment=args.min_points_per_arm_segment,
        min_valid_arm_segments=args.min_valid_arm_segments,
        min_arms_node=args.min_arms_node,
        max_arms_node=args.max_arms_node,
        edge_margin_mpc_h=args.edge_margin_mpc_h,
        edge_mode=args.edge_mode,
        omega_m=args.omega_m,
        bootstrap_B=args.bootstrap_B,
        signflip_B=args.signflip_B,
        node_weight=args.node_weight,
        point_weight=args.point_weight,
        point_unc_floor=args.point_unc_floor,
        point_unc_max_weight=args.point_unc_max_weight,
        entropy_bins=args.entropy_bins,
        arm_angle_entropy_bins=args.arm_angle_entropy_bins,
        sconce_gradient_mode=args.sconce_gradient_mode,
        distance_source=args.distance_source,
        fits_distance_units=args.fits_distance_units,
        h=args.h,
        cluster_knots=bool(args.cluster_knots),
        knot_cluster_radius_mpc_h=args.knot_cluster_radius_mpc_h,
        knot_cluster_weight=args.knot_cluster_weight,
        min_knot_points_per_node=args.min_knot_points_per_node,
        seed=args.seed,
    )

    with open(outdir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump({"args": vars(args), "config": asdict(cfg)}, f, indent=2, ensure_ascii=False)

    print("[read] SCONCE:", args.sconce_fits)
    fmaps, raw_knots, read_meta = sc.read_sconce_fits(args.sconce_fits, z_min=args.z_min, z_max=args.z_max, z_selection=args.z_selection)
    with open(outdir / "sconce_read_summary.json", "w", encoding="utf-8") as f:
        json.dump(read_meta, f, indent=2, ensure_ascii=False)
    print(f"[data] SCONCE selected points={len(fmaps):,}, raw knots={len(raw_knots):,}")

    print("[slices] building SCONCE ridge slices")
    sconce_slices = sc.build_sconce_slices(fmaps, cfg, verbose=True)
    if not sconce_slices:
        raise RuntimeError("No SCONCE slices built.")

    print("[nodes] building clustered SCONCE nodes")
    nodes_all = sc.build_nodes_from_sconce_knots(raw_knots, sconce_slices, cfg)
    if len(nodes_all) == 0:
        raise RuntimeError("No SCONCE nodes built.")
    nodes_all.to_csv(outdir / "sconce_nodes_all_before_carron_match.csv", index=False)
    print(f"[nodes] SCONCE clustered nodes before Carrón matching: {len(nodes_all):,}")

    print("[read] Carrón blocks")
    carron = read_carron_blocks(args.carron_files, z_min=args.z_min, z_max=args.z_max)
    print(f"[data] Carrón selected points={len(carron):,}")

    sconce_shells = build_sconce_shell_table(sc, fmaps, cfg)
    carron_shells = build_carron_shell_table(carron, cosmo=Planck18)
    matches = match_carron_to_sconce_shells(
        sconce_shells,
        carron_shells,
        min_overlap_fraction=args.match_min_overlap,
        max_center_offset_mpc=args.match_max_dchi_mpc,
    )
    matches.to_csv(outdir / "sconce_carron_shell_matches.csv", index=False)
    sconce_shells.to_csv(outdir / "sconce_shells_used.csv", index=False)
    carron_shells.to_csv(outdir / "carron_shells_used.csv", index=False)

    accepted = matches[matches["accepted"].astype(bool)].copy()
    print(f"[match] accepted shell matches: {len(accepted):,}/{len(matches):,}")
    print(f"[match] median |dchi| = {accepted['abs_delta_chi_center_mpc'].median():.3f} Mpc")
    print(f"[match] max |dchi| = {accepted['abs_delta_chi_center_mpc'].max():.3f} Mpc")
    print(f"[match] median overlap fraction = {accepted['overlap_fraction_of_narrower'].median():.3f}")
    print("[match] rejected shell matches:")
    rej = matches[~matches["accepted"].astype(bool)].copy()
    if len(rej):
        print(rej[["sconce_z_low", "sconce_z_high", "carron_z_low", "carron_z_high",
                   "abs_delta_chi_center_mpc", "overlap_fraction_of_narrower"]].tail(10).to_string(index=False))
    else:
        print("  none")

    carron_slices = build_carron_slices_for_sconce_keys(
        sc,
        carron,
        matches,
        cfg,
        angle_offset_deg=args.carron_angle_offset_deg,
        flip_angle_sign=args.carron_flip_angle_sign,
        verbose=True,
    )
    if not carron_slices:
        raise RuntimeError("No Carrón proxy slices built after matching.")

    accepted_keys = set(carron_slices.keys())
    nodes = nodes_all[nodes_all["z_key"].isin(accepted_keys)].copy().reset_index(drop=True)
    nodes.to_csv(outdir / "sconce_nodes_used_for_proxy.csv", index=False)
    coverage = nodes_all[[c for c in ["node_id", "ra", "dec", "z_low", "z_key"] if c in nodes_all]].copy()
    coverage["has_ridge_slice"] = nodes_all["z_key"].isin(accepted_keys).to_numpy()
    coverage.to_csv(outdir / "input_node_coverage.csv", index=False)
    print(f"[nodes] nodes in accepted SCONCE/Carrón matched shells: {len(nodes):,}")

    # Use exactly the same SCONCE-node set for the SCONCE-ridge and Carrón-ridge measurements.
    sconce_out = outdir / "sconce_ridges"
    carron_out = outdir / "carron_proxy_ridges"

    if not args.skip_sconce_ridge_measurement:
        s_node, s_arm, s_seg = process_nodes_with_slices(
            sc, nodes, sconce_slices, cfg,
            n_jobs=args.n_jobs,
            timing_interval=args.timing_interval,
            catalog_label="SCONCE",
        )
        write_catalog_outputs(
            sc,
            outdir=sconce_out,
            label="SCONCE ridges",
            node_df=s_node,
            arm_df=s_arm,
            seg_df=s_seg,
            cfg=cfg,
            args=args,
        )

    c_node, c_arm, c_seg = process_nodes_with_slices(
        sc, nodes_all, carron_slices, cfg,
        n_jobs=args.n_jobs,
        timing_interval=args.timing_interval,
        catalog_label="Carron",
    )
    write_catalog_outputs(
        sc,
        outdir=carron_out,
        label="Carrón ridges",
        node_df=c_node,
        arm_df=c_arm,
        seg_df=c_seg,
        cfg=cfg,
        args=args,
    )

    if args.make_gallery and len(c_seg) > 0:
        gallery_dir = carron_out / "gallery"
        print(f"[gallery] writing native Carrón-proxy gallery to {gallery_dir}")
        sc.make_node_gallery(c_node, c_seg, carron_slices, cfg, args, gallery_dir)

    dt = time.perf_counter() - t0
    print(f"[done] total wall time = {dt/60:.2f} min")
    print(f"[done] outputs: {outdir}")


if __name__ == "__main__":
    main()
