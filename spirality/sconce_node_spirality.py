#!/usr/bin/env python3
r"""
Measure projected filament-arm winding / handedness around SCONCE-SCMS knots.

This is the SCONCE/DirSCMS analogue of chen_node_spirality_v4.py.
Version v4 adds asymmetric bootstrap error bars and per-plot CSV exports.
It uses one FITS catalogue such as

  nodes_scms.fits

which contains filamentary points, Riemannian gradient vectors and knot labels.
Rows with knot_label == "Knot" are used as node/intersection candidates.  Since
SCONCE labels filamentary *points* as knots, nearby knot-labeled points can be
clustered into one physical node per redshift slice before measuring the arm
winding statistics.

Main outputs are the same as in the Chen script:
  - node_spirality.csv
  - arm_spirality.csv
  - arm_segments.csv
  - global_summary.json / .csv
  - optional zbin_summary.csv / zbin_summary_by_gal_hemi.csv / zbin_plots/
  - optional per-node spiral plots

Statistic
---------
For each knot/node in a redshift slice, the code identifies radial filament arms
from nearby SCONCE filament points, traces each arm through fixed comoving radial
bins, orients the headless filament tangent outward from the node, and computes

    S_node = < t_phi >,

where t_phi is the tangential component around the node in the local sky tangent
plane. Positive S corresponds to counter-clockwise projected winding as radius
increases; negative S corresponds to clockwise projected winding.

The SCONCE gradient convention
-----------------------------
The public catalogue provides grad_Dir1/2/3, described as the Riemannian gradient
of the estimated density field.  For a 1D ridge on the sphere, the gradient lies
in the ridge/tangent subspace, so the default assumption is to use the projected
gradient as the local filament tangent.  Use

    --sconce-gradient-mode perp

as a diagnostic if you want to rotate the gradient by 90 degrees in the local
tangent plane. If C_radial collapses or results change qualitatively, this is an
important convention/systematics check.

Caveat
------
This is a projected/tomographic handedness statistic.  A parity/cosmology claim
requires mocks, mirror/null tests, footprint tests, RSD/projection tests and
robustness against the knot clustering and gradient convention.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
try:
    from astropy.io import fits
    from astropy.cosmology import Planck18
except ImportError:
    fits = None  # Pure geometry functions remain importable for unit tests.
    Planck18 = None
from scipy.spatial import ConvexHull, QhullError, cKDTree
from scipy.stats import chi2 as chi2_dist, norm as norm_dist
from concurrent.futures import ThreadPoolExecutor, as_completed

DEG2RAD = np.pi / 180.0
RAD2DEG = 180.0 / np.pi
C_OVER_H0_MPC_H = 2997.92458
PLANCK18_H = float(Planck18.h) if Planck18 is not None else 0.6766
PLANCK18_OM0 = float(Planck18.Om0) if Planck18 is not None else 0.30966


# -----------------------------------------------------------------------------
# Configuration containers
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class Config:
    segment_length_mpc_h: float = 10.0
    n_arm_segments: int = 3
    node_link_radius_mpc_h: float = 8.0
    inner_exclusion_mpc_h: float = 0.0
    arm_opening_deg: float = 35.0
    cluster_gap_deg: float = 35.0
    seed_max_tangent_angle_deg: float = 75.0
    min_seed_points: int = 2
    min_points_per_arm_segment: int = 3
    min_valid_arm_segments: int = 2
    min_arms_node: int = 2
    max_arms_node: int = 12
    edge_margin_mpc_h: float = 35.0
    edge_mode: str = "convexhull"
    omega_m: float = PLANCK18_OM0
    bootstrap_B: int = 1000
    signflip_B: int = 2000
    node_weight: str = "uniform"
    point_weight: str = "density"
    point_unc_floor: float = 1e-3
    point_unc_max_weight: float = 1e6
    entropy_bins: int = 31
    arm_angle_entropy_bins: int = 8
    chen_direction_convention: str = "physical"  # kept for compatibility with shared helpers
    sconce_gradient_mode: str = "tangent"
    distance_source: str = "planck18"
    fits_distance_units: str = "mpc"
    h: float = PLANCK18_H
    cluster_knots: bool = True
    knot_cluster_radius_mpc_h: float = 5.0
    knot_cluster_weight: str = "density"
    min_knot_points_per_node: int = 1
    seed: int = 2026

    @property
    def arm_radius_mpc_h(self) -> float:
        return float(self.segment_length_mpc_h) * int(self.n_arm_segments)


@dataclass
class SliceData:
    z_key: float
    z_value: float
    chi_mpc_h: float
    ra0_deg: float
    dec0_deg: float
    points: pd.DataFrame
    x_mpc_h: np.ndarray
    y_mpc_h: np.ndarray
    e_hat: np.ndarray
    n_hat: np.ndarray
    density: np.ndarray
    uncertainty: np.ndarray
    tree: cKDTree
    hull_equations: Optional[np.ndarray]


# -----------------------------------------------------------------------------
# I/O and coordinate helpers
# -----------------------------------------------------------------------------

def normalize_name(name: str) -> str:
    s = str(name).strip().strip('"').strip("'").lower().replace("-", "_")
    for ch in "()[]{}":
        s = s.replace(ch, "")
    s = s.replace(".", "").replace(" ", "_")
    return s


def rename_chen_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip().strip('"').strip("'") for c in out.columns]
    mapping: Dict[str, str] = {}
    for col in out.columns:
        n = normalize_name(col)
        if n in {"alpha2000", "ra", "ra2000", "ra_deg"}:
            mapping[col] = "ra"
        elif n in {"delta2000", "dec", "declination", "dec_deg"}:
            mapping[col] = "dec"
        elif n in {"zlo", "z_low", "zlow"}:
            mapping[col] = "z_low"
        elif n == "density":
            mapping[col] = "density"
        elif n == "h":
            mapping[col] = "H"
        elif n == "um":
            mapping[col] = "UM"
        elif n in {"vra", "v_ra"}:
            mapping[col] = "v_ra"
        elif n in {"vdec", "v_dec"}:
            mapping[col] = "v_dec"
        else:
            mapping[col] = str(col)
    out = out.rename(columns=mapping)
    out = out.loc[:, ~out.columns.duplicated()].copy()
    return out


def read_chen_fmaps(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    required = ["ra", "dec", "z_low", "v_ra", "v_dec"]
    tried: List[object] = []

    for kwargs in [
        dict(sep=",", comment="#"),
        dict(sep=None, engine="python", comment="#"),
        dict(sep=r"\s+", engine="python", comment="#"),
    ]:
        try:
            df = pd.read_csv(path, **kwargs)
            df = rename_chen_columns(df)
            if all(c in df.columns for c in required):
                break
            tried.append(list(df.columns))
        except Exception as exc:
            tried.append(repr(exc))
    else:
        cols = ["ra", "dec", "z_low", "density", "H", "UM", "v_ra", "v_dec"]
        for kwargs in [
            dict(sep=",", comment="#", header=None),
            dict(sep=None, engine="python", comment="#", header=None),
            dict(sep=r"\s+", engine="python", comment="#", header=None),
        ]:
            try:
                df = pd.read_csv(path, **kwargs)
                if df.shape[1] >= 8:
                    df = df.iloc[:, :8].copy()
                    df.columns = cols
                    break
                tried.append(f"headerless shape={df.shape}")
            except Exception as exc:
                tried.append(repr(exc))
        else:
            raise ValueError(f"Could not parse {path}. Tried: {tried}")

    df = rename_chen_columns(df)
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{path} missing required columns: {missing}")

    for c in ["ra", "dec", "z_low", "v_ra", "v_dec", "density", "H", "UM"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df[np.isfinite(df["ra"]) & np.isfinite(df["dec"]) & np.isfinite(df["z_low"])]
    df = df[np.isfinite(df["v_ra"]) & np.isfinite(df["v_dec"])]
    subset = [c for c in ["ra", "dec", "z_low", "v_ra", "v_dec"] if c in df.columns]
    df = df.drop_duplicates(subset=subset, keep="first").reset_index(drop=True)
    return df


def read_intersections(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    tried: List[object] = []
    for kwargs in [
        dict(sep=",", comment="#"),
        dict(sep=None, engine="python", comment="#"),
        dict(sep=r"\s+", engine="python", comment="#"),
    ]:
        try:
            df = pd.read_csv(path, **kwargs)
            df = rename_chen_columns(df)
            if all(c in df.columns for c in ["ra", "dec", "z_low"]):
                break
            tried.append(list(df.columns))
        except Exception as exc:
            tried.append(repr(exc))
    else:
        for kwargs in [
            dict(sep=",", comment="#", header=None),
            dict(sep=None, engine="python", comment="#", header=None),
            dict(sep=r"\s+", engine="python", comment="#", header=None),
        ]:
            try:
                df = pd.read_csv(path, **kwargs)
                if df.shape[1] >= 3:
                    df = df.iloc[:, :3].copy()
                    df.columns = ["ra", "dec", "z_low"]
                    break
                tried.append(f"headerless shape={df.shape}")
            except Exception as exc:
                tried.append(repr(exc))
        else:
            raise ValueError(f"Could not parse {path}. Tried: {tried}")

    df = rename_chen_columns(df)
    missing = [c for c in ["ra", "dec", "z_low"] if c not in df.columns]
    if missing:
        raise ValueError(f"{path} missing required columns: {missing}")
    for c in ["ra", "dec", "z_low"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df[np.isfinite(df["ra"]) & np.isfinite(df["dec"]) & np.isfinite(df["z_low"])]
    df = df.drop_duplicates(subset=["ra", "dec", "z_low"], keep="first").reset_index(drop=True)
    df.insert(0, "node_id", np.arange(len(df), dtype=np.int64))
    return df


def zkey(z: float, ndigits: int = 5) -> float:
    return float(np.round(float(z), ndigits))


def circular_mean_rad(phi: np.ndarray, weights: Optional[np.ndarray] = None) -> float:
    phi = np.asarray(phi, dtype=float)
    if weights is None:
        weights = np.ones_like(phi)
    weights = np.asarray(weights, dtype=float)
    ok = np.isfinite(phi) & np.isfinite(weights) & (weights >= 0)
    if not np.any(ok):
        return np.nan
    s = np.sum(weights[ok] * np.sin(phi[ok]))
    c = np.sum(weights[ok] * np.cos(phi[ok]))
    return float(np.arctan2(s, c))


def angular_diff_rad(a: np.ndarray | float, b: np.ndarray | float) -> np.ndarray:
    return (np.asarray(a) - np.asarray(b) + np.pi) % (2.0 * np.pi) - np.pi


def wrap_ra_delta_deg(ra_deg: np.ndarray, ra0_deg: float) -> np.ndarray:
    return (np.asarray(ra_deg, dtype=float) - float(ra0_deg) + 180.0) % 360.0 - 180.0


def comoving_distance_mpc_h(z: float, omega_m: float = 0.3153, ngrid: int = 2048) -> float:
    z = float(z)
    if not np.isfinite(z) or z <= 0:
        return 0.0
    zz = np.linspace(0.0, z, ngrid)
    invE = 1.0 / np.sqrt(omega_m * (1.0 + zz) ** 3 + (1.0 - omega_m))
    # NumPy 2.x removed np.trapz. Do not use getattr(..., np.trapz)
    # because the default argument is evaluated eagerly.
    trap = getattr(np, "trapezoid", None)
    if trap is None:
        trap = getattr(np, "trapz", None)
    if trap is not None:
        integral = trap(invE, zz)
    else:
        # Very small compatibility fallback.
        integral = np.sum(0.5 * (invE[1:] + invE[:-1]) * np.diff(zz))
    return float(C_OVER_H0_MPC_H * integral)


def project_slice_xy_mpc_h(
    ra_deg: np.ndarray,
    dec_deg: np.ndarray,
    *,
    ra0_deg: float,
    dec0_deg: float,
    chi_mpc_h: float,
) -> Tuple[np.ndarray, np.ndarray]:
    dra = wrap_ra_delta_deg(ra_deg, ra0_deg)
    x = chi_mpc_h * DEG2RAD * dra * np.cos(float(dec0_deg) * DEG2RAD)
    y = chi_mpc_h * DEG2RAD * (np.asarray(dec_deg, dtype=float) - float(dec0_deg))
    return x, y


def local_xy_from_node_mpc_h(
    ra_deg: np.ndarray,
    dec_deg: np.ndarray,
    *,
    node_ra_deg: float,
    node_dec_deg: float,
    chi_mpc_h: float,
) -> Tuple[np.ndarray, np.ndarray]:
    dra = wrap_ra_delta_deg(ra_deg, node_ra_deg)
    x = chi_mpc_h * DEG2RAD * dra * np.cos(float(node_dec_deg) * DEG2RAD)
    y = chi_mpc_h * DEG2RAD * (np.asarray(dec_deg, dtype=float) - float(node_dec_deg))
    return x, y


def direction_EN_from_v(
    ra_deg: np.ndarray,
    dec_deg: np.ndarray,
    v_ra: np.ndarray,
    v_dec: np.ndarray,
    *,
    convention: str = "physical",
) -> Tuple[np.ndarray, np.ndarray]:
    """Return unit filament direction in the local physical tangent basis.

    Chen stores local direction components in the (RA, Dec) chart.  A displacement
    dRA corresponds to the physical tangent displacement cos(dec) dRA.  Therefore
    the default convention used here and in the LAIA Chen converter is

        E = v_ra * cos(dec),  N = v_dec.

    Set convention="raw" only as a diagnostic to reproduce the uncorrected
    chart-vector convention.
    """
    convention = str(convention).lower().strip()
    if convention not in {"physical", "raw"}:
        raise ValueError("chen_direction_convention must be 'physical' or 'raw'")
    if convention == "physical":
        E = np.asarray(v_ra, dtype=float) * np.cos(np.asarray(dec_deg, dtype=float) * DEG2RAD)
    else:
        E = np.asarray(v_ra, dtype=float)
    N = np.asarray(v_dec, dtype=float)
    norm = np.hypot(E, N)
    ok = norm > 0
    E = np.where(ok, E / norm, np.nan)
    N = np.where(ok, N / norm, np.nan)
    return E, N




# -----------------------------------------------------------------------------
# SCONCE-SCMS FITS I/O and orientation helpers
# -----------------------------------------------------------------------------

def _decode_str_array(arr: np.ndarray) -> np.ndarray:
    """Decode FITS byte/string arrays into stripped unicode strings."""
    a = np.asarray(arr)
    if a.dtype.kind == "S":
        out = np.char.decode(a, "utf-8", errors="ignore")
    else:
        out = a.astype(str)
    return np.char.strip(out)


def _sconce_label_is_knot(labels: np.ndarray) -> np.ndarray:
    lab = np.char.lower(_decode_str_array(labels))
    # Important: do not use substring matching because "Non-Knot" contains "Knot".
    return lab == "knot"


def read_sconce_fits(path: str | Path, *, z_min: Optional[float] = None, z_max: Optional[float] = None, z_selection: str = "mid") -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, object]]:
    """Read the public SCONCE/DirSCMS filament FITS catalogue.

    Returns
    -------
    fmaps : DataFrame
        All selected filamentary points.
    knots : DataFrame
        Subset with knot_label == 'Knot'. These are raw knot-labeled points,
        not necessarily de-duplicated physical nodes.
    meta : dict
        Small metadata summary.
    """
    path = Path(path)
    z_selection = str(z_selection).lower().strip()
    if z_selection not in {"mid", "low", "high"}:
        raise ValueError("z_selection must be one of: mid, low, high")

    if fits is None:
        raise ImportError("Reading SCONCE FITS requires astropy; install requirements.txt")
    with fits.open(path, memmap=True) as hdul:
        data = hdul[1].data
        names = set(data.names)
        required = {
            "RA", "DEC", "z_low", "z_high", "comov_dist_low", "comov_dist_high",
            "bw", "unc_meas", "density", "grad_Dir1", "grad_Dir2", "grad_Dir3", "knot_label",
        }
        missing = sorted(required - names)
        if missing:
            raise ValueError(f"{path} missing required SCONCE columns: {missing}")

        ra = np.asarray(data["RA"], dtype=float)
        dec = np.asarray(data["DEC"], dtype=float)
        z_low = np.asarray(data["z_low"], dtype=float)
        z_high = np.asarray(data["z_high"], dtype=float)
        z_mid = 0.5 * (z_low + z_high)
        if z_selection == "low":
            zsel = z_low
        elif z_selection == "high":
            zsel = z_high
        else:
            zsel = z_mid

        mask = np.isfinite(ra) & np.isfinite(dec) & np.isfinite(z_low) & np.isfinite(z_high)
        mask &= np.isfinite(np.asarray(data["grad_Dir1"], dtype=float))
        mask &= np.isfinite(np.asarray(data["grad_Dir2"], dtype=float))
        mask &= np.isfinite(np.asarray(data["grad_Dir3"], dtype=float))
        if z_min is not None:
            mask &= zsel >= float(z_min)
        if z_max is not None:
            mask &= zsel <= float(z_max)

        idx = np.flatnonzero(mask)
        labels = _decode_str_array(data["knot_label"][idx])
        is_knot = _sconce_label_is_knot(data["knot_label"][idx])

        cols = {
            "ra": ra[idx],
            "dec": dec[idx],
            "z_low": z_low[idx],
            "z_high": z_high[idx],
            "z_mid": z_mid[idx],
            "comov_dist_low": np.asarray(data["comov_dist_low"], dtype=float)[idx],
            "comov_dist_high": np.asarray(data["comov_dist_high"], dtype=float)[idx],
            "bw": np.asarray(data["bw"], dtype=float)[idx],
            "unc_meas": np.asarray(data["unc_meas"], dtype=float)[idx],
            "density": np.asarray(data["density"], dtype=float)[idx],
            "grad_Dir1": np.asarray(data["grad_Dir1"], dtype=float)[idx],
            "grad_Dir2": np.asarray(data["grad_Dir2"], dtype=float)[idx],
            "grad_Dir3": np.asarray(data["grad_Dir3"], dtype=float)[idx],
            "knot_label": labels,
            "is_knot": is_knot,
        }
    fmaps = pd.DataFrame(cols)
    fmaps = fmaps.drop_duplicates(subset=["ra", "dec", "z_low", "z_high", "grad_Dir1", "grad_Dir2", "grad_Dir3"], keep="first").reset_index(drop=True)
    knots = fmaps.loc[fmaps["is_knot"].to_numpy(bool)].copy().reset_index(drop=True)
    meta = {
        "path": str(path),
        "n_selected_rows": int(len(fmaps)),
        "n_raw_knot_rows": int(len(knots)),
        "z_selection": z_selection,
        "z_min": None if z_min is None else float(z_min),
        "z_max": None if z_max is None else float(z_max),
    }
    return fmaps, knots, meta


def radec_unit_xyz(ra_deg: np.ndarray, dec_deg: np.ndarray) -> np.ndarray:
    ra = np.asarray(ra_deg, dtype=float) * DEG2RAD
    dec = np.asarray(dec_deg, dtype=float) * DEG2RAD
    c = np.cos(dec)
    return np.column_stack([c * np.cos(ra), c * np.sin(ra), np.sin(dec)])


def local_east_north_basis(ra_deg: np.ndarray, dec_deg: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    ra = np.asarray(ra_deg, dtype=float) * DEG2RAD
    dec = np.asarray(dec_deg, dtype=float) * DEG2RAD
    east = np.column_stack([-np.sin(ra), np.cos(ra), np.zeros_like(ra)])
    north = np.column_stack([-np.sin(dec) * np.cos(ra), -np.sin(dec) * np.sin(ra), np.cos(dec)])
    return east, north


def radec_to_galactic_deg(ra_deg: np.ndarray, dec_deg: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Convert ICRS RA/Dec to Galactic longitude/latitude in degrees.

    Uses the standard J2000 ICRS-to-Galactic rotation matrix. This avoids an
    astropy.coordinates dependency in the hot path and is accurate enough for
    the NGC/SGC split.
    """
    n_icrs = radec_unit_xyz(np.asarray(ra_deg, dtype=float), np.asarray(dec_deg, dtype=float))
    rot = np.array([
        [-0.0548755604162154, -0.8734370902348850, -0.4838350155487132],
        [ 0.4941094278755837, -0.4448296299600112,  0.7469822444972189],
        [-0.8676661490190047, -0.1980763734312015,  0.4559837761750669],
    ], dtype=float)
    g = n_icrs @ rot.T
    glon = (np.degrees(np.arctan2(g[:, 1], g[:, 0])) + 360.0) % 360.0
    glat = np.degrees(np.arcsin(np.clip(g[:, 2], -1.0, 1.0)))
    return glon, glat


def add_galactic_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add gal_l_deg, gal_b_deg and gal_hemi north/south columns."""
    if len(df) == 0 or not {"ra", "dec"}.issubset(df.columns):
        return df
    out = df.copy()
    glon, glat = radec_to_galactic_deg(out["ra"].to_numpy(float), out["dec"].to_numpy(float))
    out["gal_l_deg"] = glon
    out["gal_b_deg"] = glat
    out["gal_hemi"] = np.where(np.isfinite(glat) & (glat >= 0.0), "north", "south")
    return out


def direction_EN_from_sconce_grad(
    ra_deg: np.ndarray,
    dec_deg: np.ndarray,
    gx: np.ndarray,
    gy: np.ndarray,
    gz: np.ndarray,
    *,
    mode: str = "tangent",
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert SCONCE grad_Dir1/2/3 to local physical tangent components.

    Default mode='tangent' assumes the Riemannian gradient is tangent to the
    1D ridge/filament.  mode='perp' rotates by +90 degrees in the tangent plane
    as a convention/systematics diagnostic.
    """
    mode = str(mode).lower().strip()
    if mode not in {"tangent", "perp"}:
        raise ValueError("sconce_gradient_mode must be 'tangent' or 'perp'")
    nvec = radec_unit_xyz(ra_deg, dec_deg)
    east, north = local_east_north_basis(ra_deg, dec_deg)
    g = np.column_stack([np.asarray(gx, dtype=float), np.asarray(gy, dtype=float), np.asarray(gz, dtype=float)])
    # Enforce tangency numerically.
    g = g - np.sum(g * nvec, axis=1, keepdims=True) * nvec
    E = np.sum(g * east, axis=1)
    N = np.sum(g * north, axis=1)
    if mode == "perp":
        E, N = -N, E
    norm = np.hypot(E, N)
    ok = norm > 0
    E = np.where(ok, E / norm, np.nan)
    N = np.where(ok, N / norm, np.nan)
    return E, N


def sconce_shell_distance_bounds_mpc(sub: pd.DataFrame, cfg: Config) -> Tuple[float, float]:
    """Shell boundaries in Mpc under the same policy as the node geometry.

    The default recomputes distances from the original redshift boundaries,
    so native FITS distances from a different cosmology cannot enter silently.
    The explicit legacy modes retain configurable flat-LCDM or native distances.
    """
    source = str(cfg.distance_source).lower().strip()
    if source == "fits":
        bounds = np.array([
            np.nanmedian(sub["comov_dist_low"].to_numpy(float)),
            np.nanmedian(sub["comov_dist_high"].to_numpy(float)),
        ], dtype=float)
        units = str(cfg.fits_distance_units).lower().strip()
        if units in {"mpc_h", "mpc/h", "hmpc"}:
            bounds /= float(cfg.h)
        elif units not in {"mpc", "m"}:
            raise ValueError("fits_distance_units must be 'mpc_h' or 'mpc'")
    elif source in {"planck18", "zmid"}:
        redshifts = np.array([
            np.nanmedian(sub["z_low"].to_numpy(float)),
            np.nanmedian(sub["z_high"].to_numpy(float)),
        ], dtype=float)
        if source == "planck18":
            if Planck18 is None:
                raise ImportError("Planck18 node geometry requires astropy; install requirements.txt")
            bounds = np.asarray(Planck18.comoving_distance(redshifts).value, dtype=float)
        else:
            bounds = np.array([comoving_distance_mpc_h(z, cfg.omega_m) / float(cfg.h) for z in redshifts])
    else:
        raise ValueError("distance_source must be 'planck18', 'zmid' or 'fits'")
    return float(bounds[0]), float(bounds[1])


def slice_chi_from_sconce(sub: pd.DataFrame, cfg: Config) -> float:
    source = str(cfg.distance_source).lower().strip()
    if source == "zmid":
        zmid = float(np.nanmedian(sub["z_mid"].to_numpy(float))) if "z_mid" in sub.columns else float(np.nanmedian(sub["z_low"].to_numpy(float)))
        return comoving_distance_mpc_h(zmid, cfg.omega_m)
    chi_low, chi_high = sconce_shell_distance_bounds_mpc(sub, cfg)
    # Match the Carron convention: mean boundary distance, in h^-1 Mpc.
    h = PLANCK18_H if source == "planck18" else float(cfg.h)
    return 0.5 * (chi_low + chi_high) * h


def build_sconce_slices(fmaps: pd.DataFrame, cfg: Config, *, verbose: bool = True) -> Dict[float, SliceData]:
    out: Dict[float, SliceData] = {}
    rng = np.random.default_rng(cfg.seed)
    for z_raw, sub0 in fmaps.groupby(fmaps["z_low"].map(zkey)):
        sub = sub0.copy().reset_index(drop=True)
        z_val = float(np.nanmedian(sub["z_mid"].to_numpy(float))) if "z_mid" in sub.columns else float(np.nanmedian(sub["z_low"].to_numpy(float)))
        chi = slice_chi_from_sconce(sub, cfg)
        if chi <= 0 or not np.isfinite(chi):
            continue
        ra0 = float(np.nanmedian(sub["ra"].to_numpy(float)))
        dec0 = float(np.nanmedian(sub["dec"].to_numpy(float)))
        x, y = project_slice_xy_mpc_h(sub["ra"].to_numpy(float), sub["dec"].to_numpy(float), ra0_deg=ra0, dec0_deg=dec0, chi_mpc_h=chi)
        E, N = direction_EN_from_sconce_grad(
            sub["ra"].to_numpy(float), sub["dec"].to_numpy(float),
            sub["grad_Dir1"].to_numpy(float), sub["grad_Dir2"].to_numpy(float), sub["grad_Dir3"].to_numpy(float),
            mode=cfg.sconce_gradient_mode,
        )
        density = sub["density"].to_numpy(float) if "density" in sub.columns else np.ones(len(sub), dtype=float)
        uncertainty = sub["unc_meas"].to_numpy(float) if "unc_meas" in sub.columns else np.full(len(sub), np.nan, dtype=float)
        good = np.isfinite(x) & np.isfinite(y) & np.isfinite(E) & np.isfinite(N)
        sub = sub.loc[good].reset_index(drop=True)
        x = x[good]
        y = y[good]
        E = E[good]
        N = N[good]
        density = density[good]
        uncertainty = uncertainty[good]
        if len(sub) == 0:
            continue
        tree = cKDTree(np.column_stack([x, y]))
        hull_eq = None
        if cfg.edge_mode == "convexhull":
            hull_eq = build_hull_equations(x, y, rng=rng)
        out[float(z_raw)] = SliceData(
            z_key=float(z_raw), z_value=z_val, chi_mpc_h=chi, ra0_deg=ra0, dec0_deg=dec0,
            points=sub, x_mpc_h=x, y_mpc_h=y, e_hat=E, n_hat=N, density=density, uncertainty=uncertainty,
            tree=tree, hull_equations=hull_eq,
        )
        if verbose:
            nk = int(np.count_nonzero(sub.get("is_knot", pd.Series(False, index=sub.index)).to_numpy(bool))) if "is_knot" in sub.columns else 0
            print(f"[slice z={z_val:.5f}] points={len(sub):,} raw_knot_points={nk:,} chi={chi:.1f} Mpc/h hull={'yes' if hull_eq is not None else 'no'}")
    return out


class _DSU:
    def __init__(self, n: int):
        self.parent = np.arange(n, dtype=np.int64)
        self.size = np.ones(n, dtype=np.int64)
    def find(self, x: int) -> int:
        p = self.parent[x]
        while p != self.parent[p]:
            p = self.parent[p]
        while x != p:
            nxt = self.parent[x]
            self.parent[x] = p
            x = nxt
        return int(p)
    def union(self, a: int, b: int) -> None:
        ra = self.find(int(a)); rb = self.find(int(b))
        if ra == rb:
            return
        if self.size[ra] < self.size[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.size[ra] += self.size[rb]


def invert_slice_xy_to_radec(x: float, y: float, *, ra0_deg: float, dec0_deg: float, chi_mpc_h: float) -> Tuple[float, float]:
    dec = float(dec0_deg) + (float(y) / max(float(chi_mpc_h), 1e-12)) * RAD2DEG
    denom = max(float(chi_mpc_h) * DEG2RAD * np.cos(float(dec0_deg) * DEG2RAD), 1e-12)
    dra = float(x) / denom
    ra = (float(ra0_deg) + dra) % 360.0
    return ra, dec


def knot_cluster_weights(sub: pd.DataFrame, mode: str) -> np.ndarray:
    mode = str(mode).lower().strip()
    if mode == "uniform":
        return np.ones(len(sub), dtype=float)
    if mode == "density" and "density" in sub.columns:
        w = sub["density"].to_numpy(float)
        return np.where(np.isfinite(w) & (w > 0), w, 1.0)
    if mode == "inv_unc2" and "unc_meas" in sub.columns:
        u = sub["unc_meas"].to_numpy(float)
        return np.where(np.isfinite(u) & (u > 0), 1.0 / u**2, 1.0)
    raise ValueError("knot_cluster_weight must be one of: uniform, density, inv_unc2")


def build_nodes_from_sconce_knots(knots: pd.DataFrame, slices: Dict[float, SliceData], cfg: Config) -> pd.DataFrame:
    """Build a de-duplicated node catalogue from SCONCE knot-labeled points."""
    rows: List[Dict[str, object]] = []
    raw_id_counter = 0
    for z_raw, sub0 in knots.groupby(knots["z_low"].map(zkey)):
        sd = slices.get(float(z_raw))
        if sd is None:
            continue
        sub = sub0.copy().reset_index(drop=True)
        if len(sub) == 0:
            continue
        x, y = project_slice_xy_mpc_h(sub["ra"].to_numpy(float), sub["dec"].to_numpy(float), ra0_deg=sd.ra0_deg, dec0_deg=sd.dec0_deg, chi_mpc_h=sd.chi_mpc_h)
        good = np.isfinite(x) & np.isfinite(y)
        sub = sub.loc[good].reset_index(drop=True)
        x = x[good]; y = y[good]
        if len(sub) == 0:
            continue

        if not cfg.cluster_knots:
            for j in range(len(sub)):
                rows.append({
                    "node_id": raw_id_counter,
                    "ra": float(sub.loc[j, "ra"]),
                    "dec": float(sub.loc[j, "dec"]),
                    "z_low": float(sub.loc[j, "z_low"]),
                    "z_high": float(sub.loc[j, "z_high"]),
                    "z_mid": float(sub.loc[j, "z_mid"]),
                    "z_key": float(z_raw),
                    "n_knot_points_clustered": 1,
                    "node_x_slice_mpc_h": float(x[j]),
                    "node_y_slice_mpc_h": float(y[j]),
                    "cluster_radius_mpc_h": 0.0,
                })
                raw_id_counter += 1
            continue

        tree = cKDTree(np.column_stack([x, y]))
        pairs = tree.query_pairs(r=float(cfg.knot_cluster_radius_mpc_h), output_type="ndarray")
        dsu = _DSU(len(sub))
        for a, b in pairs:
            dsu.union(int(a), int(b))
        roots = np.array([dsu.find(i) for i in range(len(sub))], dtype=np.int64)
        for root in np.unique(roots):
            idx = np.flatnonzero(roots == root)
            if idx.size < int(cfg.min_knot_points_per_node):
                continue
            comp = sub.iloc[idx].copy()
            w = knot_cluster_weights(comp, cfg.knot_cluster_weight)
            xx = float(np.sum(w * x[idx]) / np.sum(w))
            yy = float(np.sum(w * y[idx]) / np.sum(w))
            rr = np.hypot(x[idx] - xx, y[idx] - yy)
            ra_c, dec_c = invert_slice_xy_to_radec(xx, yy, ra0_deg=sd.ra0_deg, dec0_deg=sd.dec0_deg, chi_mpc_h=sd.chi_mpc_h)
            rows.append({
                "node_id": raw_id_counter,
                "ra": float(ra_c),
                "dec": float(dec_c),
                "z_low": float(np.nanmedian(comp["z_low"].to_numpy(float))),
                "z_high": float(np.nanmedian(comp["z_high"].to_numpy(float))),
                "z_mid": float(np.nanmedian(comp["z_mid"].to_numpy(float))),
                "z_key": float(z_raw),
                "n_knot_points_clustered": int(idx.size),
                "node_x_slice_mpc_h": xx,
                "node_y_slice_mpc_h": yy,
                "cluster_radius_mpc_h": float(np.nanmax(rr)) if len(rr) else 0.0,
                "cluster_rms_mpc_h": float(np.sqrt(np.nanmean(rr**2))) if len(rr) else 0.0,
                "mean_knot_density": float(np.nanmean(comp["density"].to_numpy(float))) if "density" in comp.columns else np.nan,
                "mean_unc_meas": float(np.nanmean(comp["unc_meas"].to_numpy(float))) if "unc_meas" in comp.columns else np.nan,
            })
            raw_id_counter += 1
    if not rows:
        return pd.DataFrame(columns=["node_id", "ra", "dec", "z_low", "z_key"])
    nodes = pd.DataFrame(rows).sort_values(["z_low", "node_id"]).reset_index(drop=True)
    nodes["node_id"] = np.arange(len(nodes), dtype=np.int64)
    nodes = add_galactic_columns(nodes)
    return nodes

# -----------------------------------------------------------------------------
# Footprint / slice construction
# -----------------------------------------------------------------------------

def build_hull_equations(x: np.ndarray, y: np.ndarray, *, max_hull_points: int = 250_000, rng: Optional[np.random.Generator] = None) -> Optional[np.ndarray]:
    pts = np.column_stack([x, y])
    ok = np.isfinite(pts).all(axis=1)
    pts = pts[ok]
    if len(pts) < 4:
        return None
    if len(pts) > max_hull_points:
        rng = np.random.default_rng(12345) if rng is None else rng
        idx = rng.choice(len(pts), size=max_hull_points, replace=False)
        pts = pts[idx]
    try:
        hull = ConvexHull(pts)
    except QhullError:
        return None
    return np.asarray(hull.equations, dtype=float)


def hull_distance_mpc_h(x: float, y: float, equations: Optional[np.ndarray]) -> Tuple[bool, float]:
    if equations is None or len(equations) == 0:
        return True, np.inf
    a = equations[:, 0]
    b = equations[:, 1]
    c = equations[:, 2]
    vals = a * float(x) + b * float(y) + c
    den = np.hypot(a, b)
    signed_inside_dist = -vals / den
    inside = bool(np.all(vals <= 1e-9))
    return inside, float(np.nanmin(signed_inside_dist))


def build_slices(fmaps: pd.DataFrame, cfg: Config, *, verbose: bool = True) -> Dict[float, SliceData]:
    out: Dict[float, SliceData] = {}
    rng = np.random.default_rng(cfg.seed)
    for z_raw, sub0 in fmaps.groupby(fmaps["z_low"].map(zkey)):
        sub = sub0.copy().reset_index(drop=True)
        z_val = float(np.nanmedian(sub["z_low"].to_numpy(float)))
        chi = comoving_distance_mpc_h(z_val, cfg.omega_m)
        if chi <= 0:
            continue
        # Use a global tangent plane per slice for KDTree / hull.
        ra0 = float(np.nanmedian(sub["ra"].to_numpy(float)))
        dec0 = float(np.nanmedian(sub["dec"].to_numpy(float)))
        x, y = project_slice_xy_mpc_h(sub["ra"].to_numpy(float), sub["dec"].to_numpy(float), ra0_deg=ra0, dec0_deg=dec0, chi_mpc_h=chi)
        E, N = direction_EN_from_v(sub["ra"].to_numpy(float), sub["dec"].to_numpy(float), sub["v_ra"].to_numpy(float), sub["v_dec"].to_numpy(float), convention=cfg.chen_direction_convention)
        density = sub["density"].to_numpy(float) if "density" in sub.columns else np.ones(len(sub), dtype=float)
        good = np.isfinite(x) & np.isfinite(y) & np.isfinite(E) & np.isfinite(N)
        sub = sub.loc[good].reset_index(drop=True)
        x = x[good]
        y = y[good]
        E = E[good]
        N = N[good]
        density = density[good]
        if len(sub) == 0:
            continue
        tree = cKDTree(np.column_stack([x, y]))
        hull_eq = None
        if cfg.edge_mode == "convexhull":
            hull_eq = build_hull_equations(x, y, rng=rng)
        out[float(z_raw)] = SliceData(
            z_key=float(z_raw),
            z_value=z_val,
            chi_mpc_h=chi,
            ra0_deg=ra0,
            dec0_deg=dec0,
            points=sub,
            x_mpc_h=x,
            y_mpc_h=y,
            e_hat=E,
            n_hat=N,
            density=density,
            tree=tree,
            hull_equations=hull_eq,
        )
        if verbose:
            print(f"[slice z={z_val:.5f}] points={len(sub):,} chi={chi:.1f} Mpc/h hull={'yes' if hull_eq is not None else 'no'}")
    return out


# -----------------------------------------------------------------------------
# Arm detection and metric fitting
# -----------------------------------------------------------------------------

def cluster_seed_angles(phi: np.ndarray, *, gap_deg: float, min_points: int, max_clusters: int) -> List[np.ndarray]:
    """Cluster circular seed angles by cutting gaps larger than gap_deg."""
    phi = np.asarray(phi, dtype=float)
    ok = np.isfinite(phi)
    idx_all = np.flatnonzero(ok)
    if idx_all.size == 0:
        return []
    vals = phi[idx_all] % (2.0 * np.pi)
    order = np.argsort(vals)
    idx_sorted = idx_all[order]
    vals_sorted = vals[order]

    if len(vals_sorted) == 1:
        return [idx_sorted] if len(idx_sorted) >= min_points else []

    gaps = np.diff(np.r_[vals_sorted, vals_sorted[0] + 2.0 * np.pi])
    # Rotate after the largest gap to avoid splitting across 0/2pi.
    start = int((np.argmax(gaps) + 1) % len(vals_sorted))
    vals_rot = np.r_[vals_sorted[start:], vals_sorted[:start] + 2.0 * np.pi]
    idx_rot = np.r_[idx_sorted[start:], idx_sorted[:start]]
    gaps_rot = np.diff(vals_rot)
    cut = np.where(gaps_rot > gap_deg * DEG2RAD)[0]

    clusters: List[np.ndarray] = []
    lo = 0
    for c in cut:
        hi = int(c) + 1
        if hi - lo >= min_points:
            clusters.append(idx_rot[lo:hi])
        lo = hi
    if len(idx_rot) - lo >= min_points:
        clusters.append(idx_rot[lo:])

    clusters = sorted(clusters, key=len, reverse=True)
    return clusters[:max_clusters]


def weighted_mean_direction(E: np.ndarray, N: np.ndarray, weights: np.ndarray) -> Tuple[float, float]:
    ok = np.isfinite(E) & np.isfinite(N) & np.isfinite(weights) & (weights >= 0)
    if not np.any(ok):
        return np.nan, np.nan
    e = float(np.sum(weights[ok] * E[ok]))
    n = float(np.sum(weights[ok] * N[ok]))
    norm = math.hypot(e, n)
    if norm <= 0 or not np.isfinite(norm):
        return np.nan, np.nan
    return e / norm, n / norm


def fit_line_wls(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> Tuple[float, float, float, float, int]:
    """Weighted y = a + b x. Returns a,b,err_a,err_b,dof."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    w = np.asarray(w, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(w) & (w > 0)
    if np.count_nonzero(ok) < 2:
        return np.nan, np.nan, np.nan, np.nan, 0
    X = np.column_stack([np.ones(np.count_nonzero(ok)), x[ok]])
    yw = y[ok]
    ww = w[ok]
    sw = np.sqrt(ww)
    Xw = X * sw[:, None]
    yw2 = yw * sw
    try:
        beta, *_ = np.linalg.lstsq(Xw, yw2, rcond=None)
        resid = yw - X @ beta
        dof = max(0, len(yw) - 2)
        if dof > 0:
            sigma2 = float(np.sum(ww * resid**2) / dof)
            cov = sigma2 * np.linalg.pinv(X.T @ (ww[:, None] * X))
            err = np.sqrt(np.diag(cov))
        else:
            err = np.array([np.nan, np.nan])
        return float(beta[0]), float(beta[1]), float(err[0]), float(err[1]), int(dof)
    except np.linalg.LinAlgError:
        return np.nan, np.nan, np.nan, np.nan, 0


def fit_common_beta(segments: pd.DataFrame) -> Tuple[float, float, Dict[int, float], int]:
    """Fit phi = alpha_arm + beta ln(r/r_ref) with one common beta."""
    if len(segments) == 0:
        return np.nan, np.nan, {}, 0
    work = segments.copy()
    work = work[np.isfinite(work["phi_unwrapped_rad"]) & np.isfinite(work["rho_mpc_h"]) & (work["rho_mpc_h"] > 0)]
    if len(work) < 3 or work["arm_local_id"].nunique() < 1:
        return np.nan, np.nan, {}, 0
    r_ref = float(np.nanmedian(work["rho_mpc_h"]))
    work["xlog"] = np.log(work["rho_mpc_h"] / r_ref)
    arms = sorted(int(a) for a in work["arm_local_id"].unique())
    arm_index = {a: i for i, a in enumerate(arms)}
    X = np.zeros((len(work), 1 + len(arms)), dtype=float)
    X[:, 0] = work["xlog"].to_numpy(float)
    for i, arm in enumerate(work["arm_local_id"].to_numpy(int)):
        X[i, 1 + arm_index[int(arm)]] = 1.0
    y = work["phi_unwrapped_rad"].to_numpy(float)
    if "weight_sum" in work.columns:
        w = work["weight_sum"].to_numpy(float)
    else:
        w = work["n_points"].to_numpy(float)
    w = np.where(np.isfinite(w) & (w > 0), w, 1.0)
    sw = np.sqrt(w)
    try:
        coeff, *_ = np.linalg.lstsq(X * sw[:, None], y * sw, rcond=None)
        resid = y - X @ coeff
        dof = max(0, len(y) - X.shape[1])
        if dof > 0:
            sigma2 = float(np.sum(w * resid**2) / dof)
            cov = sigma2 * np.linalg.pinv(X.T @ (w[:, None] * X))
            beta_err = float(np.sqrt(cov[0, 0]))
        else:
            beta_err = np.nan
        intercepts = {arm: float(coeff[1 + arm_index[arm]]) for arm in arms}
        return float(coeff[0]), beta_err, intercepts, int(dof)
    except np.linalg.LinAlgError:
        return np.nan, np.nan, {}, 0


def effective_weighted_mean(values: np.ndarray, weights: np.ndarray) -> Tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    ok = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not np.any(ok):
        return np.nan, np.nan, 0.0
    v = values[ok]
    w = weights[ok]
    mean = float(np.sum(w * v) / np.sum(w))
    neff = float(np.sum(w) ** 2 / np.sum(w**2)) if np.sum(w**2) > 0 else 0.0
    if neff > 1:
        var = float(np.sum(w * (v - mean) ** 2) / np.sum(w))
        err = math.sqrt(max(var, 0.0) / neff)
    else:
        err = np.nan
    return mean, err, neff



def _safe_weights(arr: np.ndarray, mode: str) -> np.ndarray:
    """Build non-negative point weights from a single density-like array."""
    return _safe_point_weights(arr, None, mode)


def _safe_point_weights(
    density: np.ndarray,
    uncertainty: Optional[np.ndarray],
    mode: str,
    *,
    unc_floor: float = 1e-3,
    max_weight: float = 1e6,
) -> np.ndarray:
    """Build non-negative local point weights.

    Supported modes
    ---------------
    uniform
        Every filament/ridge point contributes equally to a radial segment.

    density, sqrt_density, log_density
        Use the catalogue ridge density as the local weight.

    inv_unc2
        Use inverse-square catalogue point uncertainty, 1/(unc^2+floor^2).
        The uncertainty is the native uncertainty column of each catalogue
        (SCONCE unc_meas, Carrón unc, Chen UM).  It is used only within each
        catalogue and should not be compared across catalogues as an absolute
        calibration.

    density_inv_unc2
        Hybrid local weight density/(unc^2+floor^2).
    """
    mode = str(mode).lower().strip()
    den = np.asarray(density, dtype=float)
    den = np.where(np.isfinite(den) & (den > 0), den, 1.0)

    if uncertainty is None:
        unc = np.full_like(den, np.nan, dtype=float)
    else:
        unc = np.asarray(uncertainty, dtype=float)
        if unc.shape != den.shape:
            unc = np.full_like(den, np.nan, dtype=float)
    good_unc = np.isfinite(unc) & (unc > 0)
    floor = float(unc_floor)
    floor = floor if np.isfinite(floor) and floor > 0 else 1e-3
    invu = np.where(good_unc, 1.0 / (unc * unc + floor * floor), 1.0)
    maxw = float(max_weight)
    if np.isfinite(maxw) and maxw > 0:
        invu = np.minimum(invu, maxw)

    if mode == "uniform":
        w = np.ones_like(den, dtype=float)
    elif mode == "density":
        w = den
    elif mode == "sqrt_density":
        w = np.sqrt(np.maximum(den, 0.0))
    elif mode == "log_density":
        pos = den[np.isfinite(den) & (den > 0)]
        scale = float(np.nanmedian(pos)) if len(pos) else 1.0
        scale = scale if np.isfinite(scale) and scale > 0 else 1.0
        w = np.log1p(np.maximum(den, 0.0) / scale)
    elif mode == "inv_unc2":
        w = invu
    elif mode == "density_inv_unc2":
        w = den * invu
    else:
        raise ValueError("point_weight must be one of: uniform, density, sqrt_density, log_density, inv_unc2, density_inv_unc2")
    return np.where(np.isfinite(w) & (w > 0), w, 1.0)


def weighted_circular_variance(phi: np.ndarray, weights: np.ndarray) -> float:
    phi = np.asarray(phi, dtype=float)
    w = np.asarray(weights, dtype=float)
    ok = np.isfinite(phi) & np.isfinite(w) & (w > 0)
    if not np.any(ok):
        return np.nan
    z = np.sum(w[ok] * np.exp(1j * phi[ok])) / np.sum(w[ok])
    return float(1.0 - np.abs(z))


def line_fit_diagnostics(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> Dict[str, float]:
    """Diagnostics for weighted y=a+b*x against beta=0 model.

    The absolute chi2 assumes sigma_phi ~ 1/sqrt(weight). Treat reduced chi2 as
    a weighted residual scatter diagnostic, not as a rigorous likelihood unless
    this noise model is independently calibrated. The delta-chi2 between the
    free-beta and beta=0 models is more meaningful as an internal goodness-of-fit
    / spiral-improvement diagnostic.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    w = np.asarray(w, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(w) & (w > 0)
    out = {
        "fit_n": int(np.count_nonzero(ok)),
        "fit_chi2": np.nan,
        "fit_dof": 0,
        "fit_redchi2": np.nan,
        "fit_wrms_rad": np.nan,
        "fit_wrms_deg": np.nan,
        "fit_null_chi2": np.nan,
        "fit_null_dof": 0,
        "fit_null_redchi2": np.nan,
        "fit_delta_chi2_vs_radial": np.nan,
        "fit_delta_aic_vs_radial": np.nan,
        "fit_delta_bic_vs_radial": np.nan,
        "fit_p_improve_vs_radial": np.nan,
        "fit_abs_resid_p50_deg": np.nan,
        "fit_abs_resid_p68_deg": np.nan,
        "fit_abs_resid_p90_deg": np.nan,
    }
    if np.count_nonzero(ok) < 2:
        return out
    x = x[ok]
    y = y[ok]
    w = w[ok]
    X = np.column_stack([np.ones(len(x)), x])
    try:
        beta, *_ = np.linalg.lstsq(X * np.sqrt(w)[:, None], y * np.sqrt(w), rcond=None)
    except np.linalg.LinAlgError:
        return out
    resid = y - X @ beta
    chi2 = float(np.sum(w * resid**2))
    dof = max(0, len(y) - 2)
    y0 = float(np.sum(w * y) / np.sum(w))
    resid0 = y - y0
    chi20 = float(np.sum(w * resid0**2))
    dof0 = max(0, len(y) - 1)
    delta = chi20 - chi2
    k_sp = 2
    k_null = 1
    n = len(y)
    aic_sp = chi2 + 2 * k_sp
    aic_null = chi20 + 2 * k_null
    bic_sp = chi2 + k_sp * math.log(max(n, 1))
    bic_null = chi20 + k_null * math.log(max(n, 1))
    absres = np.abs(resid) * RAD2DEG
    out.update({
        "fit_chi2": chi2,
        "fit_dof": int(dof),
        "fit_redchi2": float(chi2 / dof) if dof > 0 else np.nan,
        "fit_wrms_rad": float(math.sqrt(chi2 / np.sum(w))) if np.sum(w) > 0 else np.nan,
        "fit_wrms_deg": float(math.sqrt(chi2 / np.sum(w)) * RAD2DEG) if np.sum(w) > 0 else np.nan,
        "fit_null_chi2": chi20,
        "fit_null_dof": int(dof0),
        "fit_null_redchi2": float(chi20 / dof0) if dof0 > 0 else np.nan,
        "fit_delta_chi2_vs_radial": float(delta),
        "fit_delta_aic_vs_radial": float(aic_null - aic_sp),
        "fit_delta_bic_vs_radial": float(bic_null - bic_sp),
        "fit_p_improve_vs_radial": float(chi2_dist.sf(max(delta, 0.0), 1)),
        "fit_abs_resid_p50_deg": float(np.nanpercentile(absres, 50.0)) if len(absres) else np.nan,
        "fit_abs_resid_p68_deg": float(np.nanpercentile(absres, 68.0)) if len(absres) else np.nan,
        "fit_abs_resid_p90_deg": float(np.nanpercentile(absres, 90.0)) if len(absres) else np.nan,
    })
    return out


def common_beta_fit_diagnostics(segments: pd.DataFrame) -> Dict[str, float]:
    """Diagnostics for phi = alpha_arm + beta ln(r/r_ref) versus beta=0."""
    out = {
        "beta_common_fit_n": 0,
        "beta_common_fit_chi2": np.nan,
        "beta_common_fit_dof": 0,
        "beta_common_fit_redchi2": np.nan,
        "beta_common_fit_wrms_rad": np.nan,
        "beta_common_fit_wrms_deg": np.nan,
        "beta_common_null_chi2": np.nan,
        "beta_common_null_dof": 0,
        "beta_common_null_redchi2": np.nan,
        "beta_common_delta_chi2_vs_radial": np.nan,
        "beta_common_delta_aic_vs_radial": np.nan,
        "beta_common_delta_bic_vs_radial": np.nan,
        "beta_common_p_improve_vs_radial": np.nan,
        "beta_common_abs_resid_p50_deg": np.nan,
        "beta_common_abs_resid_p68_deg": np.nan,
        "beta_common_abs_resid_p90_deg": np.nan,
    }
    if len(segments) == 0:
        return out
    work = segments.copy()
    work = work[np.isfinite(work["phi_unwrapped_rad"]) & np.isfinite(work["rho_mpc_h"]) & (work["rho_mpc_h"] > 0)]
    if len(work) < 3 or work["arm_local_id"].nunique() < 1:
        return out
    r_ref = float(np.nanmedian(work["rho_mpc_h"]))
    work["xlog"] = np.log(work["rho_mpc_h"] / r_ref)
    arms = sorted(int(a) for a in work["arm_local_id"].unique())
    arm_index = {a: i for i, a in enumerate(arms)}
    y = work["phi_unwrapped_rad"].to_numpy(float)
    if "weight_sum" in work.columns:
        w = work["weight_sum"].to_numpy(float)
    else:
        w = work["n_points"].to_numpy(float)
    w = np.where(np.isfinite(w) & (w > 0), w, 1.0)
    n = len(work)

    X = np.zeros((n, 1 + len(arms)), dtype=float)
    X[:, 0] = work["xlog"].to_numpy(float)
    for i, arm in enumerate(work["arm_local_id"].to_numpy(int)):
        X[i, 1 + arm_index[int(arm)]] = 1.0
    X0 = np.zeros((n, len(arms)), dtype=float)
    for i, arm in enumerate(work["arm_local_id"].to_numpy(int)):
        X0[i, arm_index[int(arm)]] = 1.0
    try:
        coeff, *_ = np.linalg.lstsq(X * np.sqrt(w)[:, None], y * np.sqrt(w), rcond=None)
        coeff0, *_ = np.linalg.lstsq(X0 * np.sqrt(w)[:, None], y * np.sqrt(w), rcond=None)
    except np.linalg.LinAlgError:
        return out
    resid = y - X @ coeff
    resid0 = y - X0 @ coeff0
    chi2 = float(np.sum(w * resid**2))
    chi20 = float(np.sum(w * resid0**2))
    dof = max(0, n - X.shape[1])
    dof0 = max(0, n - X0.shape[1])
    delta = chi20 - chi2
    k_sp = X.shape[1]
    k_null = X0.shape[1]
    aic_sp = chi2 + 2 * k_sp
    aic_null = chi20 + 2 * k_null
    bic_sp = chi2 + k_sp * math.log(max(n, 1))
    bic_null = chi20 + k_null * math.log(max(n, 1))
    absres = np.abs(resid) * RAD2DEG
    out.update({
        "beta_common_fit_n": int(n),
        "beta_common_fit_chi2": chi2,
        "beta_common_fit_dof": int(dof),
        "beta_common_fit_redchi2": float(chi2 / dof) if dof > 0 else np.nan,
        "beta_common_fit_wrms_rad": float(math.sqrt(chi2 / np.sum(w))) if np.sum(w) > 0 else np.nan,
        "beta_common_fit_wrms_deg": float(math.sqrt(chi2 / np.sum(w)) * RAD2DEG) if np.sum(w) > 0 else np.nan,
        "beta_common_null_chi2": chi20,
        "beta_common_null_dof": int(dof0),
        "beta_common_null_redchi2": float(chi20 / dof0) if dof0 > 0 else np.nan,
        "beta_common_delta_chi2_vs_radial": float(delta),
        "beta_common_delta_aic_vs_radial": float(aic_null - aic_sp),
        "beta_common_delta_bic_vs_radial": float(bic_null - bic_sp),
        "beta_common_p_improve_vs_radial": float(chi2_dist.sf(max(delta, 0.0), 1)),
        "beta_common_abs_resid_p50_deg": float(np.nanpercentile(absres, 50.0)) if len(absres) else np.nan,
        "beta_common_abs_resid_p68_deg": float(np.nanpercentile(absres, 68.0)) if len(absres) else np.nan,
        "beta_common_abs_resid_p90_deg": float(np.nanpercentile(absres, 90.0)) if len(absres) else np.nan,
    })

    return out


def _common_model_design(work: pd.DataFrame, model: str, *, power_p: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[int], List[str]]:
    """Build y, weights and design matrix for common arm-geometry models.

    The fitted quantity is the unwrapped angular position of the arm segments,
    phi_{a,k}.  Every model has one intercept alpha_a per arm; the model
    parameters describe a common radial drift shared by all arms of the node.

    Models:
      radial      : phi = alpha_a
      log         : phi = alpha_a + b log(r/r_ref)
      arch        : phi = alpha_a + b (r/r_ref - 1)
      poly2       : phi = alpha_a + b1 x + b2 x^2, x=r/r_ref-1
      power       : phi = alpha_a + b [(r/r_ref)^p - 1]
    """
    model = str(model).lower().strip()
    r = work["rho_mpc_h"].to_numpy(float)
    y = work["phi_unwrapped_rad"].to_numpy(float)
    if "weight_sum" in work.columns:
        w = work["weight_sum"].to_numpy(float)
    elif "n_points" in work.columns:
        w = work["n_points"].to_numpy(float)
    else:
        w = np.ones(len(work), dtype=float)
    w = np.where(np.isfinite(w) & (w > 0), w, 1.0)
    r_ref = float(np.nanmedian(r[np.isfinite(r) & (r > 0)]))
    if not np.isfinite(r_ref) or r_ref <= 0:
        r_ref = 1.0
    rr = r / r_ref
    x = rr - 1.0

    arms = sorted(int(a) for a in work["arm_local_id"].unique())
    arm_index = {a: i for i, a in enumerate(arms)}
    cols: List[np.ndarray] = []
    names: List[str] = []
    # Arm intercepts.
    for arm in arms:
        cols.append((work["arm_local_id"].to_numpy(int) == int(arm)).astype(float))
        names.append(f"alpha_arm{arm}")

    if model == "radial":
        pass
    elif model == "log":
        cols.insert(0, np.log(np.maximum(rr, 1e-12)))
        names.insert(0, "b_log")
    elif model == "arch":
        cols.insert(0, x)
        names.insert(0, "b_arch")
    elif model == "poly2":
        cols.insert(0, x**2)
        names.insert(0, "b2_poly")
        cols.insert(0, x)
        names.insert(0, "b1_poly")
    elif model == "power":
        pwr = 1.0 if power_p is None else float(power_p)
        cols.insert(0, np.power(np.maximum(rr, 1e-12), pwr) - 1.0)
        names.insert(0, "b_power")
    else:
        raise ValueError(f"Unknown arm model: {model}")
    X = np.column_stack(cols) if cols else np.empty((len(work), 0), dtype=float)
    return y, w, X, arms, names


def _wls_chi2(y: np.ndarray, w: np.ndarray, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float, int, np.ndarray]:
    """Weighted least squares returning coeff, resid, chi2, dof, covariance."""
    ok = np.isfinite(y) & np.isfinite(w) & (w > 0) & np.isfinite(X).all(axis=1)
    if np.count_nonzero(ok) < max(1, X.shape[1]):
        return np.full(X.shape[1], np.nan), np.full(len(y), np.nan), np.nan, 0, np.full((X.shape[1], X.shape[1]), np.nan)
    yy = y[ok]
    ww = w[ok]
    XX = X[ok]
    sw = np.sqrt(ww)
    try:
        coeff, *_ = np.linalg.lstsq(XX * sw[:, None], yy * sw, rcond=None)
        resid_ok = yy - XX @ coeff
        chi2 = float(np.sum(ww * resid_ok**2))
        dof = max(0, len(yy) - X.shape[1])
        if dof > 0:
            sigma2 = chi2 / dof
            cov = sigma2 * np.linalg.pinv(XX.T @ (ww[:, None] * XX))
        else:
            cov = np.full((X.shape[1], X.shape[1]), np.nan)
        resid = np.full(len(y), np.nan, dtype=float)
        resid[ok] = resid_ok
        return coeff, resid, chi2, int(dof), cov
    except np.linalg.LinAlgError:
        return np.full(X.shape[1], np.nan), np.full(len(y), np.nan), np.nan, 0, np.full((X.shape[1], X.shape[1]), np.nan)


def common_arm_model_fit_diagnostics(segments: pd.DataFrame, model: str) -> Dict[str, float]:
    """Compare a common arm-winding model to the radial-arm null.

    Fits are performed on the density-weighted radial segment centroids, not on
    every raw SCONCE point.  This is deliberate: the segment centroids are the
    actual measurements entering S_phi and avoid giving arbitrary leverage to
    regions where SCONCE produced a denser sampling along the same ridge.  If
    --point-weight density is used, the centroids and the segment weights already
    encode the local density information.
    """
    prefix = f"fit_{str(model).lower().strip()}"
    out = {
        f"{prefix}_n": 0,
        f"{prefix}_chi2": np.nan,
        f"{prefix}_dof": 0,
        f"{prefix}_redchi2": np.nan,
        f"{prefix}_wrms_deg": np.nan,
        f"{prefix}_null_chi2": np.nan,
        f"{prefix}_null_dof": 0,
        f"{prefix}_null_wrms_deg": np.nan,
        f"{prefix}_delta_chi2_vs_radial": np.nan,
        f"{prefix}_delta_aic_vs_radial": np.nan,
        f"{prefix}_delta_bic_vs_radial": np.nan,
        f"{prefix}_p_improve_vs_radial": np.nan,
        f"{prefix}_abs_resid_p68_deg": np.nan,
        f"{prefix}_abs_resid_p90_deg": np.nan,
        f"{prefix}_param1": np.nan,
        f"{prefix}_param2": np.nan,
        f"{prefix}_param_power_p": np.nan,
    }
    if len(segments) == 0:
        return out
    work = segments.copy()
    work = work[np.isfinite(work["phi_unwrapped_rad"]) & np.isfinite(work["rho_mpc_h"]) & (work["rho_mpc_h"] > 0)]
    if len(work) < 3 or work["arm_local_id"].nunique() < 1:
        return out

    y0, w0, X0, arms, names0 = _common_model_design(work, "radial")
    coeff0, resid0, chi20, dof0, cov0 = _wls_chi2(y0, w0, X0)
    if not np.isfinite(chi20):
        return out
    model_l = str(model).lower().strip()
    power_grid = [None]
    if model_l == "power":
        # A small fixed grid keeps the fit fast and avoids fragile nonlinear
        # optimization on nodes with only a few radial segments.
        power_grid = list(np.linspace(0.25, 3.0, 23))

    best = None
    for pwr in power_grid:
        y, w, X, arms2, names = _common_model_design(work, model_l, power_p=pwr)
        coeff, resid, chi2, dof, cov = _wls_chi2(y, w, X)
        if not np.isfinite(chi2):
            continue
        # AIC/BIC with respect to the radial null. For the power-law model, the
        # scanned exponent counts as one extra model parameter.
        k_model = X.shape[1] + (1 if model_l == "power" else 0)
        n = int(np.count_nonzero(np.isfinite(resid)))
        aic = chi2 + 2 * k_model
        bic = chi2 + k_model * math.log(max(n, 1))
        if best is None or aic < best["aic"]:
            best = dict(coeff=coeff, resid=resid, chi2=chi2, dof=dof, cov=cov, X=X, pwr=pwr, aic=aic, bic=bic, n=n)
    if best is None:
        return out

    n0 = int(np.count_nonzero(np.isfinite(resid0)))
    k0 = X0.shape[1]
    aic0 = chi20 + 2 * k0
    bic0 = chi20 + k0 * math.log(max(n0, 1))
    chi2 = float(best["chi2"])
    dof = int(best["dof"])
    resid = np.asarray(best["resid"], dtype=float)
    delta = float(chi20 - chi2)
    # Number of common-winding parameters beyond the radial null.
    delta_dof = max(1, int(best["X"].shape[1] - X0.shape[1]) + (1 if model_l == "power" else 0))
    absres = np.abs(resid[np.isfinite(resid)]) * RAD2DEG
    coeff = np.asarray(best["coeff"], dtype=float)
    wsum = float(np.nansum(w0[np.isfinite(w0) & (w0 > 0)]))
    null_wrms = math.sqrt(chi20 / wsum) * RAD2DEG if wsum > 0 else np.nan
    fit_wrms = math.sqrt(chi2 / wsum) * RAD2DEG if wsum > 0 else np.nan
    out.update({
        f"{prefix}_n": int(best["n"]),
        f"{prefix}_chi2": chi2,
        f"{prefix}_dof": dof,
        f"{prefix}_redchi2": float(chi2 / dof) if dof > 0 else np.nan,
        f"{prefix}_wrms_deg": float(fit_wrms),
        f"{prefix}_null_chi2": float(chi20),
        f"{prefix}_null_dof": int(dof0),
        f"{prefix}_null_wrms_deg": float(null_wrms),
        f"{prefix}_delta_chi2_vs_radial": delta,
        f"{prefix}_delta_aic_vs_radial": float(aic0 - float(best["aic"])),
        f"{prefix}_delta_bic_vs_radial": float(bic0 - float(best["bic"])),
        f"{prefix}_p_improve_vs_radial": float(chi2_dist.sf(max(delta, 0.0), delta_dof)),
        f"{prefix}_abs_resid_p68_deg": float(np.nanpercentile(absres, 68.0)) if len(absres) else np.nan,
        f"{prefix}_abs_resid_p90_deg": float(np.nanpercentile(absres, 90.0)) if len(absres) else np.nan,
        f"{prefix}_param1": float(coeff[0]) if len(coeff) > 0 and np.isfinite(coeff[0]) else np.nan,
        f"{prefix}_param2": float(coeff[1]) if len(coeff) > 1 and np.isfinite(coeff[1]) and model_l == "poly2" else np.nan,
        f"{prefix}_param_power_p": float(best["pwr"]) if model_l == "power" and best["pwr"] is not None else np.nan,
    })
    return out


def fit_model_summary_for_node(segments: pd.DataFrame) -> Dict[str, float]:
    """Diagnostics for radial vs log/Archimedean/power-law/poly2 arm models."""
    out: Dict[str, float] = {}
    for model in ["log", "arch", "power", "poly2"]:
        out.update(common_arm_model_fit_diagnostics(segments, model))
    # Record the best model by AIC among the tested non-radial models.
    best_name = "none"
    best_daic = -np.inf
    for model in ["log", "arch", "power", "poly2"]:
        key = f"fit_{model}_delta_aic_vs_radial"
        val = float(out.get(key, np.nan))
        if np.isfinite(val) and val > best_daic:
            best_daic = val
            best_name = model
    out["fit_best_model_by_aic"] = best_name
    out["fit_best_delta_aic_vs_radial"] = float(best_daic) if np.isfinite(best_daic) else np.nan
    return out


def segment_radial_null_diagnostics(segments: pd.DataFrame) -> Dict[str, float]:
    """Model-free radial-null diagnostics based on the measured 10 Mpc/h segments.

    For each arm we fit a straight radial ray, i.e. a constant polar angle
    phi = alpha_arm, and measure how strongly the segment centroids deviate from
    that radial null.  This uses the same segment measurements that enter S_phi,
    so it avoids depending on any specific spiral parametrization.
    """
    out = {
        "segment_radial_n": 0,
        "segment_radial_chi2": np.nan,
        "segment_radial_dof": 0,
        "segment_radial_redchi2": np.nan,
        "segment_radial_wrms_deg": np.nan,
        "segment_radial_p_radial": np.nan,
        "segment_radial_reject_sigma": np.nan,
        "segment_radial_abs_resid_p68_deg": np.nan,
        "segment_radial_abs_resid_p90_deg": np.nan,
    }
    if len(segments) == 0:
        return out
    work = segments.copy()
    work = work[np.isfinite(work["phi_rad"]) & np.isfinite(work["rho_mpc_h"]) & (work["rho_mpc_h"] > 0)]
    if len(work) < 3 or work["arm_local_id"].nunique() < 1:
        return out
    alphas = {}
    resid_all = []
    w_all = []
    for arm_id, sub in work.groupby("arm_local_id"):
        sub = sub.sort_values("rho_mpc_h").copy()
        phi = np.unwrap(sub["phi_rad"].to_numpy(float))
        w = sub["n_points"].to_numpy(float) if "n_points" in sub.columns else np.ones(len(sub), dtype=float)
        w = np.where(np.isfinite(w) & (w > 0), w, 1.0)
        alpha = float(np.sum(w * phi) / np.sum(w))
        alphas[int(arm_id)] = alpha
        resid = angular_diff_rad(phi, alpha)
        resid_all.extend(resid.tolist())
        w_all.extend(w.tolist())
    if len(resid_all) < 3:
        return out
    resid = np.asarray(resid_all, dtype=float)
    w = np.asarray(w_all, dtype=float)
    chi2 = float(np.sum(w * resid**2))
    dof = max(0, len(resid) - len(alphas))
    wsum = float(np.sum(w))
    wrms_deg = math.sqrt(chi2 / wsum) * RAD2DEG if wsum > 0 else np.nan
    p = float(chi2_dist.sf(max(chi2, 0.0), max(dof, 1))) if dof > 0 else np.nan
    sigma = float(norm_dist.isf(p / 2.0)) if np.isfinite(p) and p > 0 and p < 1 else np.nan
    absres = np.abs(resid) * RAD2DEG
    out.update({
        "segment_radial_n": int(len(resid)),
        "segment_radial_chi2": chi2,
        "segment_radial_dof": int(dof),
        "segment_radial_redchi2": float(chi2 / dof) if dof > 0 else np.nan,
        "segment_radial_wrms_deg": float(wrms_deg),
        "segment_radial_p_radial": p,
        "segment_radial_reject_sigma": sigma,
        "segment_radial_abs_resid_p68_deg": float(np.nanpercentile(absres, 68.0)) if len(absres) else np.nan,
        "segment_radial_abs_resid_p90_deg": float(np.nanpercentile(absres, 90.0)) if len(absres) else np.nan,
    })
    return out


def normalized_shannon_entropy_from_probs(p: np.ndarray) -> float:
    p = np.asarray(p, dtype=float)
    p = p[np.isfinite(p) & (p > 0)]
    if len(p) <= 1:
        return 0.0
    h = -float(np.sum(p * np.log(p)))
    return float(h / math.log(len(p))) if len(p) > 1 else np.nan


def weighted_sign_entropy(values: np.ndarray, weights: np.ndarray) -> Tuple[float, float, float, float]:
    """Return normalized sign entropy, p_plus, p_minus, sign asymmetry."""
    v = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    ok = np.isfinite(v) & np.isfinite(w) & (w > 0) & (v != 0)
    if not np.any(ok):
        return np.nan, np.nan, np.nan, np.nan
    wp = float(np.sum(w[ok & (v > 0)]))
    wm = float(np.sum(w[ok & (v < 0)]))
    tot = wp + wm
    if tot <= 0:
        return np.nan, np.nan, np.nan, np.nan
    pp = wp / tot
    pm = wm / tot
    h = normalized_shannon_entropy_from_probs(np.array([pp, pm]))
    asym = (pp - pm)
    return h, pp, pm, asym


def weighted_binned_entropy(values: np.ndarray, weights: np.ndarray, *, nbins: int, vmin: float = -1.0, vmax: float = 1.0) -> float:
    v = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    ok = np.isfinite(v) & np.isfinite(w) & (w > 0)
    if not np.any(ok):
        return np.nan
    counts, _ = np.histogram(v[ok], bins=int(nbins), range=(float(vmin), float(vmax)), weights=w[ok])
    tot = float(np.sum(counts))
    if tot <= 0:
        return np.nan
    p = counts / tot
    h = -float(np.sum(p[p > 0] * np.log(p[p > 0])))
    return float(h / math.log(int(nbins))) if int(nbins) > 1 else np.nan


def arm_angle_entropy(seed_phi_deg: np.ndarray, *, nbins: int = 8) -> float:
    phi = np.asarray(seed_phi_deg, dtype=float) * DEG2RAD
    phi = phi[np.isfinite(phi)] % (2.0 * np.pi)
    if len(phi) <= 1:
        return 0.0
    counts, _ = np.histogram(phi, bins=int(nbins), range=(0.0, 2.0 * np.pi))
    nonzero = counts[counts > 0]
    if len(nonzero) <= 1:
        return 0.0
    p = nonzero / np.sum(nonzero)
    # Normalize by log(min(nbins, n_arms)) so a node with few arms can reach H=1.
    denom = math.log(min(int(nbins), len(phi)))
    return float((-np.sum(p * np.log(p))) / denom) if denom > 0 else 0.0


def bootstrap_statistic(values: np.ndarray, weights: np.ndarray, func, *, B: int, rng: np.random.Generator) -> Tuple[float, float, float, float]:
    v = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    ok = np.isfinite(v) & np.isfinite(w) & (w > 0)
    v = v[ok]
    w = w[ok]
    if len(v) == 0:
        return np.nan, np.nan, np.nan, np.nan
    obs = float(func(v, w))
    if B <= 0 or len(v) < 2:
        return obs, np.nan, np.nan, np.nan
    boots = np.empty(B, dtype=float)
    for b in range(B):
        ii = rng.integers(0, len(v), size=len(v))
        boots[b] = float(func(v[ii], w[ii]))
    return obs, float(np.nanstd(boots, ddof=1)), float(np.nanpercentile(boots, 16.0)), float(np.nanpercentile(boots, 84.0))




def _pack_gallery_float_array(arr, *, max_n: int = 350) -> str:
    """Compact semicolon-separated local coordinates for per-FANS PNG diagnostics."""
    try:
        a = np.asarray(arr, dtype=float)
        a = a[np.isfinite(a)]
        if a.size == 0:
            return ""
        if a.size > int(max_n):
            # Deterministic thinning keeps the PNG/lightweight CSV size bounded.
            ii = np.linspace(0, a.size - 1, int(max_n)).round().astype(int)
            a = a[ii]
        return ";".join(f"{x:.5g}" for x in a)
    except Exception:
        return ""

def trace_arm_segments(
    sd: SliceData,
    *,
    node_x: float,
    node_y: float,
    all_idx: np.ndarray,
    seed_phi: float,
    arm_local_id: int,
    cfg: Config,
    audit: Optional[dict] = None,
) -> List[Dict[str, float]]:
    """Trace one arm through radial bins, allowing phi to drift slowly."""
    rows: List[Dict[str, float]] = []
    if audit is not None:
        audit.update(stop_reason="completed", stop_segment=None, n_points_at_stop=None)
    current_phi = float(seed_phi)
    opening = float(cfg.arm_opening_deg) * DEG2RAD
    max_r = cfg.arm_radius_mpc_h
    seg_len = cfg.segment_length_mpc_h

    dx_all = sd.x_mpc_h[all_idx] - node_x
    dy_all = sd.y_mpc_h[all_idx] - node_y
    rho_all = np.hypot(dx_all, dy_all)
    phi_all = np.arctan2(dy_all, dx_all)
    E_all = sd.e_hat[all_idx]
    N_all = sd.n_hat[all_idx]
    den_all = sd.density[all_idx]
    den_all = np.where(np.isfinite(den_all) & (den_all > 0), den_all, 1.0)
    unc_all = sd.uncertainty[all_idx] if hasattr(sd, "uncertainty") else np.full_like(den_all, np.nan, dtype=float)
    point_w_all = _safe_point_weights(den_all, unc_all, cfg.point_weight, unc_floor=cfg.point_unc_floor, max_weight=cfg.point_unc_max_weight)

    for k in range(int(cfg.n_arm_segments)):
        r_lo = max(float(cfg.inner_exclusion_mpc_h), k * seg_len)
        r_hi = (k + 1) * seg_len
        if r_lo >= max_r:
            if audit is not None:
                audit.update(stop_reason="radial_limit", stop_segment=k)
            break
        dphi = np.abs(angular_diff_rad(phi_all, current_phi))
        mask = (rho_all >= r_lo) & (rho_all < r_hi) & (dphi <= opening)
        if np.count_nonzero(mask) < int(cfg.min_points_per_arm_segment):
            if audit is not None:
                audit.update(stop_reason="too_few_points", stop_segment=k, n_points_at_stop=int(np.count_nonzero(mask)))
            break

        idx_loc = np.flatnonzero(mask)
        w = point_w_all[idx_loc]
        if not np.any(np.isfinite(w) & (w > 0)):
            w = np.ones_like(idx_loc, dtype=float)

        x_cent = float(np.sum(w * dx_all[idx_loc]) / np.sum(w))
        y_cent = float(np.sum(w * dy_all[idx_loc]) / np.sum(w))
        rho = float(math.hypot(x_cent, y_cent))
        phi = float(math.atan2(y_cent, x_cent))
        if not np.isfinite(rho) or rho <= 0:
            if audit is not None:
                audit.update(stop_reason="invalid_centroid", stop_segment=k, n_points_at_stop=int(len(idx_loc)))
            break

        # Orient all point tangents outward from the node before averaging.
        rx = dx_all[idx_loc] / np.maximum(rho_all[idx_loc], 1e-12)
        ry = dy_all[idx_loc] / np.maximum(rho_all[idx_loc], 1e-12)
        dot = E_all[idx_loc] * rx + N_all[idx_loc] * ry
        sign = np.where(dot >= 0, 1.0, -1.0)
        Eout = E_all[idx_loc] * sign
        Nout = N_all[idx_loc] * sign
        Emean, Nmean = weighted_mean_direction(Eout, Nout, w)
        if not np.isfinite(Emean) or not np.isfinite(Nmean):
            if audit is not None:
                audit.update(stop_reason="invalid_tangent", stop_segment=k, n_points_at_stop=int(len(idx_loc)))
            break

        rhat_x = x_cent / rho
        rhat_y = y_cent / rho
        phihat_x = -rhat_y
        phihat_y = rhat_x
        t_r = float(Emean * rhat_x + Nmean * rhat_y)
        t_phi = float(Emean * phihat_x + Nmean * phihat_y)
        psi = float(math.atan2(Nmean, Emean))

        rows.append({
            "arm_local_id": int(arm_local_id),
            "arm_segment_index": int(k),
            "n_points": int(len(idx_loc)),
            "weight_sum": float(np.nansum(w)),
            "rho_mpc_h": rho,
            "phi_rad": phi,
            "phi_deg": phi * RAD2DEG,
            "x_mpc_h": x_cent,
            "y_mpc_h": y_cent,
            "raw_x_mpc_h": _pack_gallery_float_array(dx_all[idx_loc]),
            "raw_y_mpc_h": _pack_gallery_float_array(dy_all[idx_loc]),
            "E_out": Emean,
            "N_out": Nmean,
            "psi_rad": psi,
            "psi_deg": psi * RAD2DEG,
            "t_radial": t_r,
            "t_phi": t_phi,
            "mean_density": float(np.nanmean(den_all[idx_loc])),
        })
        current_phi = phi
    return rows



def collect_arm_fit_points(
    sd: SliceData,
    *,
    node_x: float,
    node_y: float,
    all_idx: np.ndarray,
    seed_phi: float,
    arm_local_id: int,
    cfg: Config,
) -> List[Dict[str, float]]:
    """Collect all SCONCE points assigned to an arm for model-fit diagnostics.

    This follows the same radial-bin tracking logic used for the measured
    segments, but keeps every assigned point with its density/point weight.  The
    geometric model comparison can therefore be done on the raw SCONCE points,
    while S_phi itself remains based on the density-weighted 10 Mpc/h segment
    measurements.
    """
    rows: List[Dict[str, float]] = []
    current_phi = float(seed_phi)
    opening = float(cfg.arm_opening_deg) * DEG2RAD
    max_r = cfg.arm_radius_mpc_h
    seg_len = cfg.segment_length_mpc_h
    dx_all = sd.x_mpc_h[all_idx] - node_x
    dy_all = sd.y_mpc_h[all_idx] - node_y
    rho_all = np.hypot(dx_all, dy_all)
    phi_all = np.arctan2(dy_all, dx_all)
    den_all = sd.density[all_idx]
    den_all = np.where(np.isfinite(den_all) & (den_all > 0), den_all, 1.0)
    unc_all = sd.uncertainty[all_idx] if hasattr(sd, "uncertainty") else np.full_like(den_all, np.nan, dtype=float)
    point_w_all = _safe_point_weights(den_all, unc_all, cfg.point_weight, unc_floor=cfg.point_unc_floor, max_weight=cfg.point_unc_max_weight)

    for k in range(int(cfg.n_arm_segments)):
        r_lo = max(float(cfg.inner_exclusion_mpc_h), k * seg_len)
        r_hi = (k + 1) * seg_len
        if r_lo >= max_r:
            break
        dphi = np.abs(angular_diff_rad(phi_all, current_phi))
        mask = (rho_all >= r_lo) & (rho_all < r_hi) & (dphi <= opening)
        if np.count_nonzero(mask) < int(cfg.min_points_per_arm_segment):
            break
        idx_loc = np.flatnonzero(mask)
        w = point_w_all[idx_loc]
        # Update the arm center angle exactly as trace_arm_segments does.
        x_cent = float(np.sum(w * dx_all[idx_loc]) / np.sum(w))
        y_cent = float(np.sum(w * dy_all[idx_loc]) / np.sum(w))
        rho_cent = float(math.hypot(x_cent, y_cent))
        phi_cent = float(math.atan2(y_cent, x_cent))
        if not np.isfinite(rho_cent) or rho_cent <= 0:
            break
        for jj in idx_loc:
            rows.append({
                "arm_local_id": int(arm_local_id),
                "arm_segment_index": int(k),
                "rho_mpc_h": float(rho_all[jj]),
                "phi_rad": float(phi_all[jj]),
                "density": float(den_all[jj]),
                "weight_sum": float(point_w_all[jj]),
                "n_points": 1,
            })
        current_phi = phi_cent
    return rows

def _process_one_node_science(node: pd.Series, sd: SliceData, cfg: Config) -> Tuple[Dict[str, object], List[Dict[str, object]], List[Dict[str, object]]]:
    node_id = int(node["node_id"])
    ra = float(node["ra"])
    dec = float(node["dec"])
    z = float(node["z_low"])
    node_x_slice, node_y_slice = project_slice_xy_mpc_h(
        np.array([ra]), np.array([dec]), ra0_deg=sd.ra0_deg, dec0_deg=sd.dec0_deg, chi_mpc_h=sd.chi_mpc_h
    )
    nx = float(node_x_slice[0])
    ny = float(node_y_slice[0])

    inside, edge_dist = hull_distance_mpc_h(nx, ny, sd.hull_equations)
    base: Dict[str, object] = {
        "node_id": node_id,
        "ra": ra,
        "dec": dec,
        "z_low": z,
        "z_key": sd.z_key,
        "chi_mpc_h": sd.chi_mpc_h,
        "gal_l_deg": float(node.get("gal_l_deg", np.nan)),
        "gal_b_deg": float(node.get("gal_b_deg", np.nan)),
        "gal_hemi": str(node.get("gal_hemi", "north" if float(node.get("gal_b_deg", np.nan)) >= 0.0 else "south")),
        "edge_inside_hull": bool(inside),
        "edge_distance_mpc_h": edge_dist,
        "edge_keep": bool(inside and (edge_dist >= cfg.edge_margin_mpc_h or cfg.edge_mode == "none")),
        "status": "ok",
    }

    if cfg.edge_mode != "none" and (not inside or edge_dist < cfg.edge_margin_mpc_h):
        base.update({"status": "edge_reject"})
        return base, [], []

    # Query all potentially useful points. Convert distances in the global slice plane;
    # local per-node curvature is then measured relative to the node below.
    max_query = cfg.arm_radius_mpc_h + cfg.node_link_radius_mpc_h
    all_idx = np.array(sd.tree.query_ball_point([nx, ny], r=max_query), dtype=np.int64)
    if all_idx.size == 0:
        base.update({"status": "no_points_near_node"})
        return base, [], []

    dx = sd.x_mpc_h[all_idx] - nx
    dy = sd.y_mpc_h[all_idx] - ny
    rho = np.hypot(dx, dy)
    phi = np.arctan2(dy, dx)
    seed_mask = (rho >= max(float(cfg.inner_exclusion_mpc_h), 1e-6)) & (rho <= float(cfg.node_link_radius_mpc_h))
    if np.count_nonzero(seed_mask) < int(cfg.min_seed_points):
        base.update({"status": "too_few_seed_points", "n_near_points": int(all_idx.size), "n_seed_points": int(np.count_nonzero(seed_mask))})
        return base, [], []

    seed_local = np.flatnonzero(seed_mask)
    # Keep only seed points whose local tangent is not almost tangential.
    rx = dx[seed_local] / np.maximum(rho[seed_local], 1e-12)
    ry = dy[seed_local] / np.maximum(rho[seed_local], 1e-12)
    dot_abs = np.abs(sd.e_hat[all_idx[seed_local]] * rx + sd.n_hat[all_idx[seed_local]] * ry)
    keep_orient = dot_abs >= math.cos(float(cfg.seed_max_tangent_angle_deg) * DEG2RAD)
    seed_local = seed_local[keep_orient]
    if seed_local.size < int(cfg.min_seed_points):
        base.update({"status": "too_few_radial_seed_points", "n_near_points": int(all_idx.size), "n_seed_points": int(seed_local.size)})
        return base, [], []

    clusters_loc = cluster_seed_angles(phi[seed_local], gap_deg=cfg.cluster_gap_deg, min_points=cfg.min_seed_points, max_clusters=cfg.max_arms_node)
    # cluster_seed_angles returns indices local to seed_local array, so map back.
    clusters = [seed_local[c] for c in clusters_loc]
    if len(clusters) < int(cfg.min_arms_node):
        base.update({"status": "too_few_arms", "n_detected_seed_clusters": int(len(clusters)), "n_seed_points": int(seed_local.size)})
        return base, [], []

    arm_rows: List[Dict[str, object]] = []
    seg_rows: List[Dict[str, object]] = []
    fit_point_rows: List[Dict[str, object]] = []

    for arm_id, cloc in enumerate(clusters):
        seed_phi = circular_mean_rad(phi[cloc])
        segs = trace_arm_segments(sd, node_x=nx, node_y=ny, all_idx=all_idx, seed_phi=seed_phi, arm_local_id=arm_id, cfg=cfg)
        if len(segs) < int(cfg.min_valid_arm_segments):
            continue
        fit_pts = collect_arm_fit_points(sd, node_x=nx, node_y=ny, all_idx=all_idx, seed_phi=seed_phi, arm_local_id=arm_id, cfg=cfg)
        # unwrap phi along this arm and fit log-spiral slope.
        ph = np.unwrap(np.array([r["phi_rad"] for r in segs], dtype=float))
        rr = np.array([r["rho_mpc_h"] for r in segs], dtype=float)
        ww = np.array([r["n_points"] for r in segs], dtype=float)
        for j, r in enumerate(segs):
            r.update({
                "node_id": node_id,
                "ra_node": ra,
                "dec_node": dec,
                "z_low": z,
                "spiral_id": f"z{sd.z_key:.5f}_node{node_id:07d}_arm{arm_id:02d}",
                "phi_unwrapped_rad": float(ph[j]),
                "phi_unwrapped_deg": float(ph[j] * RAD2DEG),
            })
        rref = float(np.nanmedian(rr))
        xlog_arm = np.log(rr / rref)
        _, beta, _, beta_err, dof = fit_line_wls(xlog_arm, ph, ww)
        arm_fit_diag = line_fit_diagnostics(xlog_arm, ph, ww)
        tphi = np.array([r["t_phi"] for r in segs], dtype=float)
        trad = np.array([r["t_radial"] for r in segs], dtype=float)
        mean_tphi, err_tphi, neff = effective_weighted_mean(tphi, ww)
        mean_trad, _, _ = effective_weighted_mean(trad, ww)
        arm_mean_density = float(np.average(np.array([r.get("mean_density", np.nan) for r in segs], dtype=float), weights=np.maximum(ww, 1.0))) if len(segs) else np.nan
        arm_rows.append({
            "node_id": node_id,
            "arm_local_id": int(arm_id),
            "spiral_id": f"z{sd.z_key:.5f}_node{node_id:07d}_arm{arm_id:02d}",
            "ra_node": ra,
            "dec_node": dec,
            "z_low": z,
            "n_arm_segments_valid": int(len(segs)),
            "n_points_total": int(np.sum(ww)),
            "mean_arm_density": arm_mean_density,
            "seed_phi_deg": float(seed_phi * RAD2DEG),
            "delta_phi_deg": float((ph[-1] - ph[0]) * RAD2DEG),
            "beta_logspiral_rad": float(beta),
            "beta_logspiral_err_rad": float(beta_err),
            "beta_logspiral_deg": float(beta * RAD2DEG) if np.isfinite(beta) else np.nan,
            "beta_logspiral_err_deg": float(beta_err * RAD2DEG) if np.isfinite(beta_err) else np.nan,
            "beta_dof": int(dof),
            "beta_fit_chi2": arm_fit_diag.get("fit_chi2", np.nan),
            "beta_fit_dof": arm_fit_diag.get("fit_dof", 0),
            "beta_fit_redchi2": arm_fit_diag.get("fit_redchi2", np.nan),
            "beta_fit_wrms_deg": arm_fit_diag.get("fit_wrms_deg", np.nan),
            "beta_null_chi2": arm_fit_diag.get("fit_null_chi2", np.nan),
            "beta_delta_chi2_vs_radial": arm_fit_diag.get("fit_delta_chi2_vs_radial", np.nan),
            "beta_delta_aic_vs_radial": arm_fit_diag.get("fit_delta_aic_vs_radial", np.nan),
            "beta_delta_bic_vs_radial": arm_fit_diag.get("fit_delta_bic_vs_radial", np.nan),
            "beta_p_improve_vs_radial": arm_fit_diag.get("fit_p_improve_vs_radial", np.nan),
            "mean_t_phi": mean_tphi,
            "mean_t_phi_err": err_tphi,
            "mean_t_radial": mean_trad,
            "neff_segments": neff,
        })
        seg_rows.extend(segs)
        fit_point_rows.extend(fit_pts)

    if len(arm_rows) < int(cfg.min_arms_node):
        base.update({
            "status": "too_few_valid_arms",
            "n_arms_valid": int(len(arm_rows)),
            "n_near_points": int(all_idx.size),
            "n_seed_points": int(seed_local.size),
        })
        return base, arm_rows, seg_rows

    seg_df = pd.DataFrame(seg_rows)
    radial_seg_diag = segment_radial_null_diagnostics(seg_df)
    beta_node, beta_node_err, intercepts, beta_dof = fit_common_beta(seg_df)
    common_fit_diag = common_beta_fit_diagnostics(seg_df)
    # Model-comparison diagnostics are computed on all assigned SCONCE points
    # when available, using density/point weights. This is independent of the
    # 10 Mpc/h segmentation used to define S_phi.
    fit_df = pd.DataFrame(fit_point_rows) if fit_point_rows else pd.DataFrame()
    if len(fit_df) > 0:
        unwrapped_rows = []
        for arm_id, arm_sub in fit_df.groupby("arm_local_id"):
            arm_sub = arm_sub.sort_values("rho_mpc_h").copy()
            arm_sub["phi_unwrapped_rad"] = np.unwrap(arm_sub["phi_rad"].to_numpy(float))
            unwrapped_rows.append(arm_sub)
        fit_df = pd.concat(unwrapped_rows, ignore_index=True) if unwrapped_rows else fit_df
        model_fit_diag = fit_model_summary_for_node(fit_df)
        model_fit_diag["fit_model_input"] = "points"
        model_fit_diag["fit_model_n_raw_points"] = int(len(fit_df))
    else:
        model_fit_diag = fit_model_summary_for_node(seg_df)
        model_fit_diag["fit_model_input"] = "segments"
        model_fit_diag["fit_model_n_raw_points"] = 0
    tphi_all = seg_df["t_phi"].to_numpy(float)
    trad_all = seg_df["t_radial"].to_numpy(float)
    wseg = seg_df["n_points"].to_numpy(float)
    S_node, S_node_err, neff = effective_weighted_mean(tphi_all, wseg)
    R_node, _, _ = effective_weighted_mean(trad_all, wseg)
    arm_beta = np.array([a["beta_logspiral_rad"] for a in arm_rows], dtype=float)
    arm_tphi = np.array([a["mean_t_phi"] for a in arm_rows], dtype=float)
    arm_seed_phi = np.array([a.get("seed_phi_deg", np.nan) for a in arm_rows], dtype=float)
    arm_pos = int(np.count_nonzero(np.isfinite(arm_tphi) & (arm_tphi > 0)))
    arm_neg = int(np.count_nonzero(np.isfinite(arm_tphi) & (arm_tphi < 0)))
    arm_angle_H = arm_angle_entropy(arm_seed_phi, nbins=cfg.arm_angle_entropy_bins)
    mean_seg_density = float(np.nanmean(seg_df["mean_density"].to_numpy(float))) if "mean_density" in seg_df.columns and len(seg_df) else np.nan

    base.update({
        "status": "ok",
        "n_near_points": int(all_idx.size),
        "n_seed_points": int(seed_local.size),
        "n_arms_valid": int(len(arm_rows)),
        "n_arm_segments_total": int(len(seg_rows)),
        "n_points_used_total": int(np.nansum(wseg)),
        "S_tangential": float(S_node),
        "S_tangential_err": float(S_node_err) if np.isfinite(S_node_err) else np.nan,
        "S_tangential_abs": float(abs(S_node)) if np.isfinite(S_node) else np.nan,
        "S_tangential_sq": float(S_node**2) if np.isfinite(S_node) else np.nan,
        "mean_t_radial": float(R_node),
        "beta_common_rad": float(beta_node),
        "beta_common_err_rad": float(beta_node_err),
        "beta_common_deg": float(beta_node * RAD2DEG) if np.isfinite(beta_node) else np.nan,
        "beta_common_err_deg": float(beta_node_err * RAD2DEG) if np.isfinite(beta_node_err) else np.nan,
        "beta_common_dof": int(beta_dof),
        "beta_common_fit_chi2": common_fit_diag.get("beta_common_fit_chi2", np.nan),
        "beta_common_fit_dof": common_fit_diag.get("beta_common_fit_dof", 0),
        "beta_common_fit_redchi2": common_fit_diag.get("beta_common_fit_redchi2", np.nan),
        "beta_common_fit_wrms_deg": common_fit_diag.get("beta_common_fit_wrms_deg", np.nan),
        "beta_common_null_chi2": common_fit_diag.get("beta_common_null_chi2", np.nan),
        "beta_common_null_redchi2": common_fit_diag.get("beta_common_null_redchi2", np.nan),
        "beta_common_delta_chi2_vs_radial": common_fit_diag.get("beta_common_delta_chi2_vs_radial", np.nan),
        "beta_common_delta_aic_vs_radial": common_fit_diag.get("beta_common_delta_aic_vs_radial", np.nan),
        "beta_common_delta_bic_vs_radial": common_fit_diag.get("beta_common_delta_bic_vs_radial", np.nan),
        "beta_common_p_improve_vs_radial": common_fit_diag.get("beta_common_p_improve_vs_radial", np.nan),
        "beta_common_abs_resid_p50_deg": common_fit_diag.get("beta_common_abs_resid_p50_deg", np.nan),
        "beta_common_abs_resid_p68_deg": common_fit_diag.get("beta_common_abs_resid_p68_deg", np.nan),
        "beta_common_abs_resid_p90_deg": common_fit_diag.get("beta_common_abs_resid_p90_deg", np.nan),
        "beta_arm_mean_rad": float(np.nanmean(arm_beta)) if np.any(np.isfinite(arm_beta)) else np.nan,
        "S_arm_mean": float(np.nanmean(arm_tphi)) if np.any(np.isfinite(arm_tphi)) else np.nan,
        "arm_angle_entropy_norm": float(arm_angle_H),
        "mean_segment_density": mean_seg_density,
        "neff_segments": float(neff),
        "n_arms_positive_tphi": arm_pos,
        "n_arms_negative_tphi": arm_neg,
        "frac_arms_positive_tphi": float(arm_pos / max(arm_pos + arm_neg, 1)),
        "frac_arms_negative_tphi": float(arm_neg / max(arm_pos + arm_neg, 1)),
    })
    base.update(radial_seg_diag)
    base.update(model_fit_diag)
    # Store fit intercepts on segment rows for plotting.
    for r in seg_rows:
        arm = int(r["arm_local_id"])
        r["beta_common_rad"] = float(beta_node)
        r["beta_common_err_rad"] = float(beta_node_err) if np.isfinite(beta_node_err) else np.nan
        r["fit_intercept_arm_rad"] = float(intercepts.get(arm, np.nan))
    return base, arm_rows, seg_rows


# -----------------------------------------------------------------------------
# Global estimator and nulls
# -----------------------------------------------------------------------------

def process_one_node(node: pd.Series, sd: SliceData, cfg: Config):
    """Unchanged science geometry plus a separate before/after selection audit."""
    from spirality.construction_audit import audit_node
    result, arms, segments = _process_one_node_science(node, sd, cfg)
    diagnostics = audit_node(node, sd, cfg, project=project_slice_xy_mpc_h,
                             cluster=cluster_seed_angles, circular_mean=circular_mean_rad,
                             trace=trace_arm_segments)
    if result.get("status") in {"ok", "too_few_valid_arms"}:
        if int(result["n_arms_valid"]) != diagnostics["audit_m_valid"]:
            raise RuntimeError("Independent construction audit disagrees with science arm count")
    result.update(diagnostics)
    # Early rejections may have no science multiplicity. Keep it unknown rather
    # than replacing it with a counterfactual trace count (audit_m_valid).
    result.setdefault("n_arms_valid", np.nan)
    return result, arms, segments


def node_weights(node_df: pd.DataFrame, mode: str) -> np.ndarray:
    mode = str(mode).lower()
    if mode == "uniform":
        return np.ones(len(node_df), dtype=float)
    if mode == "n_segments":
        return node_df["n_arm_segments_total"].to_numpy(float)
    if mode == "sqrt_points":
        return np.sqrt(np.maximum(node_df["n_points_used_total"].to_numpy(float), 0.0))
    if mode == "inv_var":
        err = node_df["S_tangential_err"].to_numpy(float)
        return np.where(np.isfinite(err) & (err > 0), 1.0 / err**2, 0.0)
    if mode == "mean_density":
        den = node_df["mean_segment_density"].to_numpy(float) if "mean_segment_density" in node_df.columns else np.ones(len(node_df), dtype=float)
        return np.where(np.isfinite(den) & (den > 0), den, 0.0)
    if mode == "sqrt_mean_density":
        den = node_df["mean_segment_density"].to_numpy(float) if "mean_segment_density" in node_df.columns else np.ones(len(node_df), dtype=float)
        return np.where(np.isfinite(den) & (den > 0), np.sqrt(den), 0.0)
    raise ValueError(f"Unknown node weight mode: {mode}")


def weighted_global_mean(values: np.ndarray, weights: np.ndarray) -> float:
    ok = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not np.any(ok):
        return np.nan
    return float(np.sum(weights[ok] * values[ok]) / np.sum(weights[ok]))


def bootstrap_global(values: np.ndarray, weights: np.ndarray, *, B: int, rng: np.random.Generator) -> Tuple[float, float, float, np.ndarray]:
    ok = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    v = values[ok]
    w = weights[ok]
    n = len(v)
    if n == 0:
        return np.nan, np.nan, np.nan, np.array([], dtype=float)
    obs = weighted_global_mean(v, w)
    if B <= 0 or n < 2:
        return obs, np.nan, np.nan, np.array([], dtype=float)
    boots = np.empty(B, dtype=float)
    for b in range(B):
        ii = rng.integers(0, n, size=n)
        boots[b] = weighted_global_mean(v[ii], w[ii])
    err = float(np.nanstd(boots, ddof=1))
    z = float(obs / err) if np.isfinite(err) and err > 0 else np.nan
    return obs, err, z, boots


def signflip_null(values: np.ndarray, weights: np.ndarray, *, B: int, rng: np.random.Generator) -> Tuple[float, float, np.ndarray]:
    ok = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    v = values[ok]
    w = weights[ok]
    if len(v) == 0 or B <= 0:
        return np.nan, np.nan, np.array([], dtype=float)
    obs = weighted_global_mean(v, w)
    null = np.empty(B, dtype=float)
    for b in range(B):
        signs = rng.choice(np.array([-1.0, 1.0]), size=len(v), replace=True)
        null[b] = weighted_global_mean(signs * v, w)
    p = float((1.0 + np.count_nonzero(np.abs(null) >= abs(obs))) / (B + 1.0))
    z_null = float(obs / np.nanstd(null, ddof=1)) if np.nanstd(null, ddof=1) > 0 else np.nan
    return p, z_null, null


def bootstrap_weighted_mean(values: np.ndarray, weights: np.ndarray, *, B: int, rng: np.random.Generator) -> Tuple[float, float, float, float, np.ndarray]:
    """Bootstrap a weighted mean and return mean, err, q16, q84, samples."""
    ok = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    v = np.asarray(values, dtype=float)[ok]
    w = np.asarray(weights, dtype=float)[ok]
    n = len(v)
    if n == 0:
        return np.nan, np.nan, np.nan, np.nan, np.array([], dtype=float)
    obs = weighted_global_mean(v, w)
    if B <= 0 or n < 2:
        return obs, np.nan, np.nan, np.nan, np.array([], dtype=float)
    boots = np.empty(B, dtype=float)
    for b in range(B):
        ii = rng.integers(0, n, size=n)
        boots[b] = weighted_global_mean(v[ii], w[ii])
    err = float(np.nanstd(boots, ddof=1))
    q16 = float(np.nanpercentile(boots, 16.0))
    q84 = float(np.nanpercentile(boots, 84.0))
    return obs, err, q16, q84, boots


def build_global_summary(node_df: pd.DataFrame, cfg: Config, *, seed_offset: int = 987) -> Dict[str, object]:
    ok = node_df["status"].eq("ok") if "status" in node_df.columns else np.ones(len(node_df), dtype=bool)
    nd = node_df.loc[ok].copy()
    rng = np.random.default_rng(cfg.seed + int(seed_offset))
    out: Dict[str, object] = {
        "n_nodes_total_rows": int(len(node_df)),
        "n_nodes_ok": int(len(nd)),
        "n_nodes_edge_reject": int(np.count_nonzero(node_df.get("status", pd.Series([], dtype=str)).eq("edge_reject"))) if len(node_df) else 0,
        "config": asdict(cfg),
    }
    if len(nd) == 0:
        return out
    w = node_weights(nd, cfg.node_weight)

    # Parity-odd signed handedness estimators.
    for col, label in [("S_tangential", "S"), ("beta_common_rad", "beta")]:
        vals = nd[col].to_numpy(float) if col in nd.columns else np.full(len(nd), np.nan)
        obs, err, z, boots = bootstrap_global(vals, w, B=cfg.bootstrap_B, rng=rng)
        p, zsf, null = signflip_null(vals, w, B=cfg.signflip_B, rng=rng)
        out.update({
            f"H_{label}_weighted_mean": obs,
            f"H_{label}_bootstrap_err": err,
            f"H_{label}_bootstrap_z": z,
            f"H_{label}_signflip_p_two_sided": p,
            f"H_{label}_signflip_z": zsf,
        })
        if label == "beta" and np.isfinite(obs):
            out["H_beta_weighted_mean_deg"] = float(obs * RAD2DEG)
        if label == "beta" and np.isfinite(err):
            out["H_beta_bootstrap_err_deg"] = float(err * RAD2DEG)
        if len(boots):
            out[f"H_{label}_bootstrap_q16"] = float(np.nanpercentile(boots, 16.0))
            out[f"H_{label}_bootstrap_q84"] = float(np.nanpercentile(boots, 84.0))
            if label == "beta":
                out["H_beta_bootstrap_q16_deg"] = float(np.nanpercentile(boots, 16.0) * RAD2DEG)
                out["H_beta_bootstrap_q84_deg"] = float(np.nanpercentile(boots, 84.0) * RAD2DEG)
        if len(null):
            out[f"H_{label}_signflip_q16"] = float(np.nanpercentile(null, 16.0))
            out[f"H_{label}_signflip_q84"] = float(np.nanpercentile(null, 84.0))

    # Parity-even / quality diagnostics, now also with bootstrap errors.
    parity_even_cols = [
        ("S_tangential_abs", "mean_abs_projected_twist"),
        ("S_tangential_sq", "mean_transverse_power"),
        ("mean_t_radial", "mean_radial_coherence"),
    ]
    for col, label in parity_even_cols:
        if col in nd.columns:
            vals = nd[col].to_numpy(float)
            obs, err, q16, q84, _ = bootstrap_weighted_mean(vals, w, B=cfg.bootstrap_B, rng=rng)
            out[label] = obs
            out[f"{label}_bootstrap_err"] = err
            out[f"{label}_bootstrap_q16"] = q16
            out[f"{label}_bootstrap_q84"] = q84

    # Entropy / information diagnostics of the node-handedness distribution.
    if "S_tangential" in nd.columns:
        vals = nd["S_tangential"].to_numpy(float)
        hsign, pplus, pminus, asym = weighted_sign_entropy(vals, w)
        out["S_sign_entropy_norm"] = hsign
        out["S_sign_p_plus"] = pplus
        out["S_sign_p_minus"] = pminus
        out["S_sign_asymmetry"] = asym
        if cfg.bootstrap_B > 0 and len(vals) > 1:
            okv = np.isfinite(vals) & np.isfinite(w) & (w > 0)
            vv = vals[okv]
            ww = w[okv]
            if len(vv) > 1:
                bh = np.empty(cfg.bootstrap_B, dtype=float)
                ba = np.empty(cfg.bootstrap_B, dtype=float)
                for b in range(cfg.bootstrap_B):
                    ii = rng.integers(0, len(vv), size=len(vv))
                    hh, _, _, aa = weighted_sign_entropy(vv[ii], ww[ii])
                    bh[b] = hh
                    ba[b] = aa
                out["S_sign_entropy_bootstrap_err"] = float(np.nanstd(bh, ddof=1))
                out["S_sign_entropy_bootstrap_q16"] = float(np.nanpercentile(bh, 16.0))
                out["S_sign_entropy_bootstrap_q84"] = float(np.nanpercentile(bh, 84.0))
                out["S_sign_asymmetry_bootstrap_err"] = float(np.nanstd(ba, ddof=1))
                out["S_sign_asymmetry_bootstrap_q16"] = float(np.nanpercentile(ba, 16.0))
                out["S_sign_asymmetry_bootstrap_q84"] = float(np.nanpercentile(ba, 84.0))
        Hdist, Herr, Hq16, Hq84 = bootstrap_statistic(
            vals, w, lambda v, ww: weighted_binned_entropy(v, ww, nbins=cfg.entropy_bins, vmin=-1.0, vmax=1.0),
            B=cfg.bootstrap_B, rng=rng,
        )
        out["S_distribution_entropy_norm"] = Hdist
        out["S_distribution_entropy_bootstrap_err"] = Herr
        out["S_distribution_entropy_bootstrap_q16"] = Hq16
        out["S_distribution_entropy_bootstrap_q84"] = Hq84

    if "arm_angle_entropy_norm" in nd.columns:
        vals = nd["arm_angle_entropy_norm"].to_numpy(float)
        obs, err, q16, q84, _ = bootstrap_weighted_mean(vals, w, B=cfg.bootstrap_B, rng=rng)
        out["mean_arm_angle_entropy_norm"] = obs
        out["mean_arm_angle_entropy_bootstrap_err"] = err
        out["mean_arm_angle_entropy_bootstrap_q16"] = q16
        out["mean_arm_angle_entropy_bootstrap_q84"] = q84

    # Spiral-fit quality diagnostics. These are not absolute likelihood tests unless
    # the sigma_phi ~ 1/sqrt(n_points) model is calibrated, but they are useful
    # to show that the log-spiral model is geometrically meaningful and to compare
    # against the beta=0 radial-arm null.
    fit_cols = [
        ("beta_common_fit_wrms_deg", "mean_beta_common_fit_wrms_deg"),
        ("beta_common_fit_redchi2", "mean_beta_common_fit_redchi2"),
        ("beta_common_delta_chi2_vs_radial", "mean_beta_common_delta_chi2_vs_radial"),
        ("beta_common_delta_aic_vs_radial", "mean_beta_common_delta_aic_vs_radial"),
        ("beta_common_delta_bic_vs_radial", "mean_beta_common_delta_bic_vs_radial"),
        ("beta_common_abs_resid_p68_deg", "mean_beta_common_abs_resid_p68_deg"),
        ("beta_common_abs_resid_p90_deg", "mean_beta_common_abs_resid_p90_deg"),
        ("fit_best_delta_aic_vs_radial", "mean_fit_best_delta_aic_vs_radial"),
    ]
    fit_cols.extend([
        ("segment_radial_wrms_deg", "mean_segment_radial_wrms_deg"),
        ("segment_radial_redchi2", "mean_segment_radial_redchi2"),
        ("segment_radial_reject_sigma", "mean_segment_radial_reject_sigma"),
    ])
    for _m in ["log", "arch", "power", "poly2"]:
        fit_cols.extend([
            (f"fit_{_m}_wrms_deg", f"mean_fit_{_m}_wrms_deg"),
            (f"fit_{_m}_delta_aic_vs_radial", f"mean_fit_{_m}_delta_aic_vs_radial"),
            (f"fit_{_m}_delta_bic_vs_radial", f"mean_fit_{_m}_delta_bic_vs_radial"),
            (f"fit_{_m}_p_improve_vs_radial", f"mean_fit_{_m}_p_improve_vs_radial"),
        ])
    for col, label in fit_cols:
        if col in nd.columns:
            vals = nd[col].to_numpy(float)
            obs, err, q16, q84, _ = bootstrap_weighted_mean(vals, w, B=cfg.bootstrap_B, rng=rng)
            out[label] = obs
            out[f"{label}_bootstrap_err"] = err
            out[f"{label}_bootstrap_q16"] = q16
            out[f"{label}_bootstrap_q84"] = q84
    if "beta_common_p_improve_vs_radial" in nd.columns:
        pvals = nd["beta_common_p_improve_vs_radial"].to_numpy(float)
        okp = np.isfinite(pvals) & np.isfinite(w) & (w > 0)
        if np.any(okp):
            out["frac_nodes_spiral_improves_p_lt_0p05"] = float(np.sum(w[okp] * (pvals[okp] < 0.05)) / np.sum(w[okp]))
            out["frac_nodes_spiral_improves_p_lt_0p10"] = float(np.sum(w[okp] * (pvals[okp] < 0.10)) / np.sum(w[okp]))
    return out


def parse_zbin_edges(args: argparse.Namespace, node_df: pd.DataFrame) -> Optional[np.ndarray]:
    """Return redshift bin edges or None if z-bin products are disabled."""
    if getattr(args, "zbin_edges", None):
        edges = np.array([float(x) for x in str(args.zbin_edges).replace(";", ",").split(",") if str(x).strip() != ""], dtype=float)
        if len(edges) < 2:
            raise ValueError("--zbin-edges must contain at least two comma-separated values")
        if not np.all(np.diff(edges) > 0):
            raise ValueError("--zbin-edges must be strictly increasing")
        return edges

    nb = int(getattr(args, "zbin_nbins", 0) or 0)
    if nb <= 0:
        return None
    zbin_col = "z_mid" if "z_mid" in node_df.columns else "z_low"
    zcol = node_df[zbin_col].to_numpy(float) if zbin_col in node_df.columns else np.array([], dtype=float)
    finite = zcol[np.isfinite(zcol)]
    if len(finite) == 0:
        raise ValueError("Cannot build z bins: node_spirality table has no finite z_low values")
    zmin = getattr(args, "zbin_min", None)
    zmax = getattr(args, "zbin_max", None)
    if zmin is None:
        zmin = args.z_min if getattr(args, "z_min", None) is not None else float(np.nanmin(finite))
    if zmax is None:
        zmax = args.z_max if getattr(args, "z_max", None) is not None else float(np.nanmax(finite))
    zmin = float(zmin)
    zmax = float(zmax)
    if not (np.isfinite(zmin) and np.isfinite(zmax) and zmax > zmin):
        raise ValueError(f"Invalid z-bin range: zmin={zmin}, zmax={zmax}")
    return np.linspace(zmin, zmax, nb + 1)


def build_zbin_summary(node_df: pd.DataFrame, cfg: Config, edges: np.ndarray) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    zbin_col = "z_mid" if "z_mid" in node_df.columns else "z_low"
    z = node_df[zbin_col].to_numpy(float)
    for ib in range(len(edges) - 1):
        z0 = float(edges[ib])
        z1 = float(edges[ib + 1])
        if ib == len(edges) - 2:
            mask = np.isfinite(z) & (z >= z0) & (z <= z1)
        else:
            mask = np.isfinite(z) & (z >= z0) & (z < z1)
        sub = node_df.loc[mask].copy()
        summ = build_global_summary(sub, cfg, seed_offset=20000 + ib * 137)
        ok_sub = sub[sub["status"].eq("ok")].copy() if "status" in sub.columns else sub.copy()
        row: Dict[str, object] = {
            "zbin_id": ib,
            "z_min": z0,
            "z_max": z1,
            "z_center": 0.5 * (z0 + z1),
            "z_width": z1 - z0,
            "z_mean_ok": float(np.nanmean(ok_sub[zbin_col].to_numpy(float))) if len(ok_sub) else np.nan,
            "z_median_ok": float(np.nanmedian(ok_sub[zbin_col].to_numpy(float))) if len(ok_sub) else np.nan,
            "zbin_coord": zbin_col,
        }
        # Avoid dumping a nested config into each CSV row.
        summ = {k: v for k, v in summ.items() if k != "config"}
        row.update(summ)
        rows.append(row)
    return pd.DataFrame(rows)


def build_zbin_summary_by_gal_hemi(node_df: pd.DataFrame, cfg: Config, edges: np.ndarray) -> pd.DataFrame:
    """Build z-bin summaries for total, Galactic north and Galactic south samples."""
    if "gal_b_deg" not in node_df.columns:
        node_df = add_galactic_columns(node_df)
    specs = [
        ("all", "Total", np.ones(len(node_df), dtype=bool)),
        ("north", "Galactic north", node_df["gal_b_deg"].to_numpy(float) >= 0.0),
        ("south", "Galactic south", node_df["gal_b_deg"].to_numpy(float) < 0.0),
    ]
    rows: List[pd.DataFrame] = []
    for sample, label, mask in specs:
        sub = node_df.loc[np.asarray(mask, dtype=bool)].copy()
        zdf = build_zbin_summary(sub, cfg, edges)
        zdf.insert(0, "sample", sample)
        zdf.insert(1, "sample_label", label)
        rows.append(zdf)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def build_global_summary_by_gal_hemi(node_df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Build global summaries for total, Galactic north and Galactic south samples."""
    if "gal_b_deg" not in node_df.columns:
        node_df = add_galactic_columns(node_df)
    specs = [
        ("all", "Total", np.ones(len(node_df), dtype=bool)),
        ("north", "Galactic north", node_df["gal_b_deg"].to_numpy(float) >= 0.0),
        ("south", "Galactic south", node_df["gal_b_deg"].to_numpy(float) < 0.0),
    ]
    rows: List[Dict[str, object]] = []
    for i, (sample, label, mask) in enumerate(specs):
        summ = build_global_summary(node_df.loc[np.asarray(mask, dtype=bool)].copy(), cfg, seed_offset=50000 + 997 * i)
        summ = {k: v for k, v in summ.items() if k != "config"}
        row = {"sample": sample, "sample_label": label}
        row.update(summ)
        rows.append(row)
    return pd.DataFrame(rows)



def _bootstrap_quantile_cols_for_ycol(ycol: str) -> Tuple[Optional[str], Optional[str]]:
    """Return q16/q84 columns corresponding to a plotted y column."""
    explicit = {
        "H_S_weighted_mean": ("H_S_bootstrap_q16", "H_S_bootstrap_q84"),
        "H_beta_weighted_mean": ("H_beta_bootstrap_q16", "H_beta_bootstrap_q84"),
        "H_beta_weighted_mean_deg": ("H_beta_bootstrap_q16_deg", "H_beta_bootstrap_q84_deg"),
        "S_sign_entropy_norm": ("S_sign_entropy_bootstrap_q16", "S_sign_entropy_bootstrap_q84"),
        "S_sign_asymmetry": ("S_sign_asymmetry_bootstrap_q16", "S_sign_asymmetry_bootstrap_q84"),
        "S_distribution_entropy_norm": ("S_distribution_entropy_bootstrap_q16", "S_distribution_entropy_bootstrap_q84"),
        "mean_arm_angle_entropy_norm": ("mean_arm_angle_entropy_bootstrap_q16", "mean_arm_angle_entropy_bootstrap_q84"),
    }
    if ycol in explicit:
        return explicit[ycol]
    return f"{ycol}_bootstrap_q16", f"{ycol}_bootstrap_q84"


def _asymmetric_yerr(df: pd.DataFrame, ycol: str, errcol: Optional[str] = None) -> Optional[np.ndarray]:
    """Prefer bootstrap q16/q84 asymmetric errors; fall back to symmetric errcol."""
    if ycol not in df.columns:
        return None
    y = df[ycol].to_numpy(float)
    q16_col, q84_col = _bootstrap_quantile_cols_for_ycol(ycol)
    if q16_col in df.columns and q84_col in df.columns:
        q16 = df[q16_col].to_numpy(float)
        q84 = df[q84_col].to_numpy(float)
        lo = y - q16
        hi = q84 - y
        bad = (~np.isfinite(lo)) | (~np.isfinite(hi)) | (lo < 0) | (hi < 0)
        lo = np.where(bad, np.nan, lo)
        hi = np.where(bad, np.nan, hi)
        if np.any(np.isfinite(lo) | np.isfinite(hi)):
            return np.vstack([lo, hi])
    if errcol and errcol in df.columns:
        e = df[errcol].to_numpy(float)
        return e
    return None


def _write_plot_data_csv(
    df: pd.DataFrame,
    ycol: str,
    errcol: Optional[str],
    out: Path,
    *,
    sample: Optional[str] = None,
    dx_frac: float = 0.0,
) -> None:
    """Write the numerical data actually used by a z-bin plot next to the figure."""
    if ycol not in df.columns:
        return
    d = df.copy()
    if sample is not None and "sample" in d.columns:
        d = d[d["sample"].eq(sample)].copy()
    if len(d) == 0:
        return
    y = d[ycol].to_numpy(float)
    x = d["z_center"].to_numpy(float) + dx_frac * d["z_width"].to_numpy(float)
    xerr = 0.5 * d["z_width"].to_numpy(float)
    yerr = _asymmetric_yerr(d, ycol, errcol)
    if yerr is None:
        ylo = np.full(len(d), np.nan)
        yhi = np.full(len(d), np.nan)
    elif np.ndim(yerr) == 2:
        ylo = yerr[0]
        yhi = yerr[1]
    else:
        ylo = np.asarray(yerr, dtype=float)
        yhi = np.asarray(yerr, dtype=float)
    q16_col, q84_col = _bootstrap_quantile_cols_for_ycol(ycol)
    out_df = pd.DataFrame({
        "sample": d["sample"].to_numpy(str) if "sample" in d.columns else (sample if sample is not None else "all"),
        "zbin_id": d["zbin_id"].to_numpy(int) if "zbin_id" in d.columns else np.arange(len(d)),
        "z_min": d["z_min"].to_numpy(float),
        "z_max": d["z_max"].to_numpy(float),
        "z_center_plot": x,
        "z_err_low": xerr,
        "z_err_high": xerr,
        ycol: y,
        "y_err_low": ylo,
        "y_err_high": yhi,
    })
    if errcol and errcol in d.columns:
        out_df[errcol] = d[errcol].to_numpy(float)
    if q16_col in d.columns:
        out_df[q16_col] = d[q16_col].to_numpy(float)
    if q84_col in d.columns:
        out_df[q84_col] = d[q84_col].to_numpy(float)
    csv = out.with_suffix(".csv")
    csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(csv, index=False)


def _plot_zbin_one_by_gal_hemi(zdf: pd.DataFrame, ycol: str, errcol: Optional[str], ylabel: str, out: Path, *, zero_line: bool = True) -> None:
    setup_matplotlib()
    fig, ax = plt.subplots(figsize=(6.5, 4.25))
    specs = [
        ("all", "Total", "black", "o", 0.0),
        ("north", "Galactic north", "tab:blue", "^", -0.012),
        ("south", "Galactic south", "tab:orange", "v", 0.012),
    ]
    csv_rows = []
    for sample, label, color, marker, dx_frac in specs:
        sub = zdf[zdf["sample"].eq(sample)].copy()
        if len(sub) == 0 or ycol not in sub.columns:
            continue
        x = sub["z_center"].to_numpy(float) + dx_frac * sub["z_width"].to_numpy(float)
        xerr = 0.5 * sub["z_width"].to_numpy(float)
        y = sub[ycol].to_numpy(float)
        yerr = _asymmetric_yerr(sub, ycol, errcol)
        ax.errorbar(x, y, xerr=xerr, yerr=yerr, fmt=marker, capsize=3.0, lw=1.05, ms=4.3,
                    color=color, ecolor=color, label=label)
        # append plot data
        temp_out = out.with_name(out.stem + f"_{sample}" + out.suffix)
        _write_plot_data_csv(sub, ycol, errcol, temp_out, sample=sample, dx_frac=dx_frac)
        csv_path = temp_out.with_suffix(".csv")
        if csv_path.exists():
            try:
                csv_rows.append(pd.read_csv(csv_path))
                csv_path.unlink()
            except Exception:
                pass
    if zero_line:
        ax.axhline(0.0, color="0.35", lw=0.8, ls="--")
    ax.set_xlabel(r"redshift $z$")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.22)
    ax.legend(loc="best", fontsize=8.0, frameon=False)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    if csv_rows:
        pd.concat(csv_rows, ignore_index=True).to_csv(out.with_suffix(".csv"), index=False)



def _plot_zbin_panels(zdf: pd.DataFrame, out: Path) -> None:
    setup_matplotlib()
    x = zdf["z_center"].to_numpy(float)
    xerr = 0.5 * zdf["z_width"].to_numpy(float)
    panels = [
        ("H_S_weighted_mean", "H_S_bootstrap_err", r"$H_S=\langle S_\phi\rangle$", True),
        ("H_beta_weighted_mean_deg", "H_beta_bootstrap_err_deg", r"$H_\beta$ [deg / $\ln r$]", True),
        ("mean_abs_projected_twist", "mean_abs_projected_twist_bootstrap_err", r"$\langle |S_\phi|\rangle$", False),
        ("mean_radial_coherence", "mean_radial_coherence_bootstrap_err", r"$\langle C_{\rm radial}\rangle$", False),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(8.2, 6.2), sharex=True)
    csv_rows = []
    for ax, (ycol, ecol, ylabel, zline) in zip(axes.ravel(), panels):
        y = zdf[ycol].to_numpy(float)
        yerr = _asymmetric_yerr(zdf, ycol, ecol)
        ax.errorbar(x, y, xerr=xerr, yerr=yerr, fmt="o", capsize=3.0, lw=1.05, ms=4.0)
        if zline:
            ax.axhline(0.0, color="0.35", lw=0.8, ls="--")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.22)
        temp_out = out.with_name(out.stem + f"_{ycol}" + out.suffix)
        _write_plot_data_csv(zdf, ycol, ecol, temp_out)
        csv_path = temp_out.with_suffix(".csv")
        if csv_path.exists():
            try:
                d = pd.read_csv(csv_path)
                d.insert(0, "panel_ycol", ycol)
                csv_rows.append(d)
                csv_path.unlink()
            except Exception:
                pass
    axes[1, 0].set_xlabel(r"redshift $z$")
    axes[1, 1].set_xlabel(r"redshift $z$")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    if csv_rows:
        pd.concat(csv_rows, ignore_index=True).to_csv(out.with_suffix(".csv"), index=False)


def _plot_zbin_fit_quality_panels(zdf: pd.DataFrame, out: Path) -> None:
    """Two-panel fit-quality diagnostic: absolute WRMS and AIC improvement."""
    setup_matplotlib()
    x = zdf["z_center"].to_numpy(float)
    xerr = 0.5 * zdf["z_width"].to_numpy(float)
    panels = [
        ("mean_beta_common_fit_wrms_deg", "mean_beta_common_fit_wrms_deg_bootstrap_err", r"log-spiral WRMS [deg]", False),
        ("mean_beta_common_delta_aic_vs_radial", "mean_beta_common_delta_aic_vs_radial_bootstrap_err", r"$\Delta$AIC: radial $-$ spiral", True),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8), sharex=True)
    csv_rows = []
    for ax, (ycol, ecol, ylabel, zline) in zip(axes.ravel(), panels):
        if ycol not in zdf.columns:
            continue
        y = zdf[ycol].to_numpy(float)
        yerr = _asymmetric_yerr(zdf, ycol, ecol)
        ax.errorbar(x, y, xerr=xerr, yerr=yerr, fmt="o", capsize=3.0, lw=1.05, ms=4.0)
        if zline:
            ax.axhline(0.0, color="0.35", lw=0.8, ls="--")
        ax.set_xlabel(r"redshift $z$")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.22)
        temp_out = out.with_name(out.stem + f"_{ycol}" + out.suffix)
        _write_plot_data_csv(zdf, ycol, ecol, temp_out)
        csv_path = temp_out.with_suffix(".csv")
        if csv_path.exists():
            try:
                d = pd.read_csv(csv_path)
                d.insert(0, "panel_ycol", ycol)
                csv_rows.append(d)
                csv_path.unlink()
            except Exception:
                pass
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    if csv_rows:
        pd.concat(csv_rows, ignore_index=True).to_csv(out.with_suffix(".csv"), index=False)


def _plot_zbin_fit_quality_panels_by_gal_hemi(zdf: pd.DataFrame, out: Path) -> None:
    setup_matplotlib()
    panels = [
        ("mean_beta_common_fit_wrms_deg", "mean_beta_common_fit_wrms_deg_bootstrap_err", r"log-spiral WRMS [deg]", False),
        ("mean_beta_common_delta_aic_vs_radial", "mean_beta_common_delta_aic_vs_radial_bootstrap_err", r"$\Delta$AIC: radial $-$ spiral", True),
    ]
    specs = [
        ("all", "Total", "black", "o", 0.0),
        ("north", "Galactic north", "tab:blue", "^", -0.012),
        ("south", "Galactic south", "tab:orange", "v", 0.012),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(9.7, 3.9), sharex=True)
    csv_rows = []
    for ax, (ycol, ecol, ylabel, zline) in zip(axes.ravel(), panels):
        for sample, label, color, marker, dx_frac in specs:
            sub = zdf[zdf["sample"].eq(sample)].copy()
            if len(sub) == 0 or ycol not in sub.columns:
                continue
            x = sub["z_center"].to_numpy(float) + dx_frac * sub["z_width"].to_numpy(float)
            xerr = 0.5 * sub["z_width"].to_numpy(float)
            y = sub[ycol].to_numpy(float)
            yerr = _asymmetric_yerr(sub, ycol, ecol)
            ax.errorbar(x, y, xerr=xerr, yerr=yerr, fmt=marker, capsize=2.8, lw=0.95, ms=3.8,
                        color=color, ecolor=color, label=label)
            temp_out = out.with_name(out.stem + f"_{sample}_{ycol}" + out.suffix)
            _write_plot_data_csv(sub, ycol, ecol, temp_out, sample=sample, dx_frac=dx_frac)
            csv_path = temp_out.with_suffix(".csv")
            if csv_path.exists():
                try:
                    d = pd.read_csv(csv_path)
                    d.insert(0, "panel_ycol", ycol)
                    csv_rows.append(d)
                    csv_path.unlink()
                except Exception:
                    pass
        if zline:
            ax.axhline(0.0, color="0.35", lw=0.8, ls="--")
        ax.set_xlabel(r"redshift $z$")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.22)
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, fontsize=8.5, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    if csv_rows:
        pd.concat(csv_rows, ignore_index=True).to_csv(out.with_suffix(".csv"), index=False)


def _plot_zbin_one(zdf: pd.DataFrame, ycol: str, errcol: Optional[str], ylabel: str, out: Path, *, zero_line: bool = True) -> None:
    """Single-sample z-bin plot with asymmetric vertical error bars and CSV export."""
    setup_matplotlib()
    fig, ax = plt.subplots(figsize=(6.5, 4.25))
    if ycol not in zdf.columns:
        return
    x = zdf["z_center"].to_numpy(float)
    y = zdf[ycol].to_numpy(float)
    yerr = _asymmetric_yerr(zdf, ycol, errcol)
    ax.errorbar(x, y, yerr=yerr, fmt="o", capsize=3.0, lw=1.05, ms=4.0, color="black", ecolor="black")
    if zero_line:
        ax.axhline(0.0, color="0.35", lw=0.8, ls="--")
    ax.set_xlabel(r"redshift $z$")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.22)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    _write_plot_data_csv(zdf, ycol, errcol, out)


def _angle_deg_from_s(sval: np.ndarray) -> np.ndarray:
    sval = np.asarray(sval, dtype=float)
    return np.degrees(np.arcsin(np.clip(sval, -1.0, 1.0)))


def _plot_main_S_panels_by_gal_hemi(zdf: pd.DataFrame, out: Path, *, angle_version: bool = False) -> None:
    """Clean 2x2 main panel: H_S, <|S_phi|>, C_radial, <S_phi^2>.

    Total is emphasized in black. Galactic north/south are shown as darker
    gray filled triangles. Only vertical error bars are drawn. Each panel also
    contains a compact table with the per-bin and inverse-variance combined
    significance for the total sample.
    """
    setup_matplotlib()
    try:
        from matplotlib.lines import Line2D
    except Exception:
        Line2D = None

    specs = [
        # sample, label, color, marker, alpha, ealpha, markersize, z-offset
        ("all", "Total", "black", "o", 1.00, 1.00, 7.2, 0.000),
        ("north", "Gal. north", "0.20", "^", 0.92, 0.75, 6.8, +0.030),
        ("south", "Gal. south", "0.34", "v", 0.92, 0.75, 6.8, -0.030),
    ]

    if angle_version:
        panels = [
            ("H_S_weighted_mean", "H_S_bootstrap_err", r"$\theta_{H_S}=\sin^{-1}H_S$ [deg]", True, True, 0.0, "Handedness"),
            ("mean_abs_projected_twist", "mean_abs_projected_twist_bootstrap_err", r"$\sin^{-1}\langle |S_\phi|\rangle$ [deg]", False, True, 0.0, "Spirality/Torsion"),
            ("mean_radial_coherence", "mean_radial_coherence_bootstrap_err", r"$\langle C_{\rm radial}\rangle$", False, False, 1.0, "Radial coherence"),
            ("mean_transverse_power", "mean_transverse_power_bootstrap_err", r"$\langle S_\phi^2\rangle$", False, False, 0.0, "Transverse power"),
        ]
    else:
        panels = [
            ("H_S_weighted_mean", "H_S_bootstrap_err", r"$H_S=\langle S_\phi\rangle$", True, False, 0.0, "Handedness"),
            ("mean_abs_projected_twist", "mean_abs_projected_twist_bootstrap_err", r"$\langle |S_\phi|\rangle$", False, False, 0.0, "Spirality/Torsion"),
            ("mean_radial_coherence", "mean_radial_coherence_bootstrap_err", r"$\langle C_{\rm radial}\rangle$", False, False, 1.0, "Radial coherence"),
            ("mean_transverse_power", "mean_transverse_power_bootstrap_err", r"$\langle S_\phi^2\rangle$", False, False, 0.0, "Transverse power"),
        ]

    def _err1d_for_sig(sub: pd.DataFrame, ycol: str, ecol: Optional[str]) -> np.ndarray:
        ye = _asymmetric_yerr(sub, ycol, ecol)
        if ye is None:
            return np.full(len(sub), np.nan)
        arr = np.asarray(ye, dtype=float)
        if arr.ndim == 2:
            return 0.5 * (arr[0] + arr[1])
        return arr

    def _fmt_z(z: float) -> str:
        if not np.isfinite(z):
            return "..."
        if abs(z) >= 10:
            return f"{z:+.0f}"
        return f"{z:+.1f}"

    def _significance_table_text(zdf_in: pd.DataFrame, ycol: str, ecol: Optional[str], ref: float) -> str:
        sub = zdf_in[zdf_in["sample"].eq("all")].copy()
        if len(sub) == 0 or ycol not in sub.columns:
            return ""
        y = sub[ycol].to_numpy(float)
        e = _err1d_for_sig(sub, ycol, ecol)
        zsig = (y - ref) / e
        lines = [r"$Z$"]
        for _, row in sub.iterrows():
            yy = float(row[ycol])
            ee = _err1d_for_sig(pd.DataFrame([row]), ycol, ecol)[0]
            zz = (yy - ref) / ee if np.isfinite(ee) and ee > 0 else np.nan
            lines.append(f"{float(row['z_max']):.1f}: {_fmt_z(zz)}")
        good = np.isfinite(y) & np.isfinite(e) & (e > 0)
        if np.any(good):
            w = 1.0 / e[good]**2
            ytot = float(np.sum(w * y[good]) / np.sum(w))
            etot = float(np.sqrt(1.0 / np.sum(w)))
            ztot = (ytot - ref) / etot if etot > 0 else np.nan
            lines.append(f"all: {_fmt_z(ztot)}")
        return "\n".join(lines)

    fig, axes = plt.subplots(2, 2, figsize=(15.2, 11.0), sharex=True)
    csv_rows = []
    for pi, (ax, (ycol, ecol, ylabel, zline, to_angle, ref, panel_name)) in enumerate(zip(axes.ravel(), panels)):
        for sample, label, color, marker, alpha, ealpha, ms, xoff in specs:
            sub = zdf[zdf["sample"].eq(sample)].copy()
            if len(sub) == 0 or ycol not in sub.columns:
                continue
            x = sub["z_center"].to_numpy(float)
            x_plot = x + float(xoff)
            y = sub[ycol].to_numpy(float)
            yerr = _asymmetric_yerr(sub, ycol, ecol)
            if to_angle:
                y_plot = _angle_deg_from_s(y)
                if yerr is not None and np.ndim(yerr) == 2:
                    ylo_val = y - yerr[0]
                    yhi_val = y + yerr[1]
                    lo = y_plot - _angle_deg_from_s(ylo_val)
                    hi = _angle_deg_from_s(yhi_val) - y_plot
                    yerr_plot = np.vstack([lo, hi])
                elif yerr is not None:
                    yy = np.asarray(yerr, dtype=float)
                    yerr_plot = 0.5 * (_angle_deg_from_s(y + yy) - _angle_deg_from_s(y - yy))
                else:
                    yerr_plot = None
                y_use = y_plot
            else:
                y_use = y
                yerr_plot = yerr
            if yerr_plot is not None and not np.any(np.isfinite(np.asarray(yerr_plot, dtype=float))):
                yerr_plot = None
            ax.errorbar(
                x_plot, y_use, yerr=yerr_plot, fmt=marker, capsize=3.2,
                lw=1.25 if sample == "all" else 1.0,
                ms=ms, color=color, ecolor=color, alpha=alpha,
                elinewidth=1.2 if sample == "all" else 0.85,
                markerfacecolor=color, markeredgecolor=color, markeredgewidth=1.0,
                label=None,
            )
            if sample == "all":
                ax.plot(x_plot, y_use, color="black", ls="--", lw=1.05, alpha=0.85)
            temp_out = out.with_name(out.stem + f"_{sample}_{ycol}" + out.suffix)
            _write_plot_data_csv(sub, ycol, ecol, temp_out, sample=sample, dx_frac=0.0)
            csv_path = temp_out.with_suffix(".csv")
            if csv_path.exists():
                try:
                    d = pd.read_csv(csv_path)
                    d.insert(0, "panel_ycol", ycol)
                    d.insert(1, "angle_version", bool(angle_version))
                    csv_rows.append(d)
                    csv_path.unlink()
                except Exception:
                    pass
        if zline:
            ax.axhline(0.0, color="0.35", lw=0.8, ls="--")
        ax.set_ylabel(ylabel, fontsize=20)
        ax.grid(alpha=0.22)
        ax.tick_params(labelsize=18, length=5.0, width=1.0)
        ax.text(0.50, 0.965, panel_name, transform=ax.transAxes, ha="center", va="top",
                fontsize=22, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.70))
        if pi == 0:
            # Place handedness labels away from the significance table.
            ax.text(0.50, 0.84, "Left-handed", transform=ax.transAxes, ha="center", va="center",
                    fontsize=17, color="0.35", alpha=0.82)
            ax.text(0.50, 0.16, "Right-handed", transform=ax.transAxes, ha="center", va="center",
                    fontsize=17, color="0.35", alpha=0.82)
            if Line2D is not None:
                handles = [Line2D([0], [0], marker=m, color=c, lw=0, markersize=7.0,
                                  markerfacecolor=c, markeredgecolor=c, label=lab)
                           for _, lab, c, m, *_ in specs]
                ax.legend(handles=handles, loc="lower left", fontsize=16, frameon=False, handletextpad=0.55, borderpad=0.15)
        # Significances are saved in the source CSVs, not drawn in the panel.
    axes[1, 0].set_xlabel(r"redshift $z$", fontsize=20)
    axes[1, 1].set_xlabel(r"redshift $z$", fontsize=20)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=260, bbox_inches="tight")
    plt.close(fig)
    if csv_rows:
        pd.concat(csv_rows, ignore_index=True).to_csv(out.with_suffix(".csv"), index=False)




def _prepare_catalog_zbin_summary(zdf: pd.DataFrame, *, catalog: str, label: str) -> pd.DataFrame:
    """Normalize a z-bin summary table for catalog-comparison plots."""
    d = zdf.copy()
    if "sample" in d.columns:
        d = d[d["sample"].astype(str).eq("all")].copy()
    if len(d) == 0:
        return d
    if "z_center" not in d.columns and {"z_min", "z_max"}.issubset(d.columns):
        d["z_center"] = 0.5 * (d["z_min"].astype(float) + d["z_max"].astype(float))
    if "z_width" not in d.columns and {"z_min", "z_max"}.issubset(d.columns):
        d["z_width"] = d["z_max"].astype(float) - d["z_min"].astype(float)
    d["catalog"] = str(catalog)
    d["catalog_label"] = str(label)
    return d


def _read_optional_catalog_zbin_summary(path: Optional[str], *, catalog: str, label: str) -> Optional[pd.DataFrame]:
    if path is None or str(path).strip() == "":
        return None
    p = Path(path).expanduser()
    if not p.exists():
        print(f"[warn] comparison z-bin summary not found: {p}")
        return None
    try:
        return _prepare_catalog_zbin_summary(pd.read_csv(p), catalog=catalog, label=label)
    except Exception as exc:
        print(f"[warn] could not read comparison z-bin summary {p}: {exc}")
        return None


def _panel_y_and_err(sub: pd.DataFrame, ycol: str, ecol: Optional[str], *, to_angle: bool) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    y = sub[ycol].to_numpy(float)
    yerr = _asymmetric_yerr(sub, ycol, ecol)
    if not to_angle:
        return y, yerr
    y_plot = _angle_deg_from_s(y)
    if yerr is not None and np.ndim(yerr) == 2:
        ylo_val = y - yerr[0]
        yhi_val = y + yerr[1]
        lo = y_plot - _angle_deg_from_s(ylo_val)
        hi = _angle_deg_from_s(yhi_val) - y_plot
        return y_plot, np.vstack([lo, hi])
    if yerr is not None:
        yy = np.asarray(yerr, dtype=float)
        return y_plot, 0.5 * (_angle_deg_from_s(y + yy) - _angle_deg_from_s(y - yy))
    return y_plot, None


def _plot_catalog_band(
    ax,
    sub: pd.DataFrame,
    *,
    ycol: str,
    ecol: Optional[str],
    to_angle: bool,
    label: str,
    kind: str,
) -> Optional[pd.DataFrame]:
    """Draw one catalogue in the DR17/DR12 comparison style."""
    if len(sub) == 0 or ycol not in sub.columns:
        return None
    sub = sub.sort_values("z_center").copy()
    x = sub["z_center"].to_numpy(float)
    y, yerr = _panel_y_and_err(sub, ycol, ecol, to_angle=to_angle)
    if yerr is None:
        lo = np.full_like(y, np.nan, dtype=float)
        hi = np.full_like(y, np.nan, dtype=float)
    else:
        yy = np.asarray(yerr, dtype=float)
        if yy.ndim == 2:
            lo = y - yy[0]
            hi = y + yy[1]
        else:
            lo = y - yy
            hi = y + yy
    finite_band = np.isfinite(x) & np.isfinite(lo) & np.isfinite(hi)
    if kind == "dr17":
        line_color = "orangered"
        fill_color = "lightsalmon"
        if np.count_nonzero(finite_band) >= 2:
            ax.fill_between(x[finite_band], lo[finite_band], hi[finite_band],
                            color=fill_color, alpha=0.38, lw=0, zorder=1)
            ax.plot(x[finite_band], lo[finite_band], color=line_color, lw=1.65, ls="-", zorder=2)
            ax.plot(x[finite_band], hi[finite_band], color=line_color, lw=1.65, ls="-", zorder=2)
        ax.plot(x, y, ls="none", marker="o", ms=8.6, color=line_color,
                markerfacecolor=line_color, markeredgecolor=line_color, markeredgewidth=1.25,
                label=label, zorder=5)
    else:
        line_color = "royalblue"
        fill_color = "lightblue"
        if np.count_nonzero(finite_band) >= 2:
            ax.fill_between(x[finite_band], lo[finite_band], hi[finite_band],
                            color=fill_color, alpha=0.40, lw=0, zorder=0)
            ax.plot(x[finite_band], lo[finite_band], color=line_color, lw=1.55, ls="-", zorder=3)
            ax.plot(x[finite_band], hi[finite_band], color=line_color, lw=1.55, ls="-", zorder=3)
        ax.plot(x, y, ls="none", marker="o", ms=8.6, color=line_color,
                markerfacecolor=line_color, markeredgecolor=line_color, markeredgewidth=1.25,
                label=label, zorder=6)
    out = pd.DataFrame({
        "catalog_label": label,
        "catalog_kind": kind,
        "zbin_id": sub["zbin_id"].to_numpy(int) if "zbin_id" in sub.columns else np.arange(len(sub)),
        "z_min": sub["z_min"].to_numpy(float) if "z_min" in sub.columns else np.nan,
        "z_max": sub["z_max"].to_numpy(float) if "z_max" in sub.columns else np.nan,
        "z_center": x,
        ycol: y,
        "y_low": lo,
        "y_high": hi,
    })
    return out


def _plot_main_S_panels_catalog_compare(
    zdf_dr17: pd.DataFrame,
    out: Path,
    *,
    angle_version: bool = False,
    zdf_dr12: Optional[pd.DataFrame] = None,
    dr17_label: str = "DR17 SCONCE",
    dr12_label: str = "DR12 Chen",
) -> None:
    """Paper-style 2x2 panel comparing DR17 SCONCE to Chen DR12.

    DR17: orangered filled circles plus a lightsalmon bootstrap band.
    DR12: royalblue filled circles plus a lightblue bootstrap band.
    The full-footprint rows are used; no north/south split is drawn.
    """
    setup_matplotlib()
    dr17 = _prepare_catalog_zbin_summary(zdf_dr17, catalog="DR17", label=dr17_label)
    dr12 = zdf_dr12.copy() if zdf_dr12 is not None else None
    if dr12 is not None and len(dr12) > 0:
        dr12 = _prepare_catalog_zbin_summary(dr12, catalog="DR12", label=dr12_label)

    if angle_version:
        panels = [
            ("H_S_weighted_mean", "H_S_bootstrap_err", r"$\theta_{H_S}=\sin^{-1}H_S$ [deg]", True, True, 0.0, "Handedness"),
            ("mean_abs_projected_twist", "mean_abs_projected_twist_bootstrap_err", r"$\sin^{-1}\langle |S_\phi|\rangle$ [deg]", False, True, 0.0, "Spirality/Torsion"),
            ("mean_radial_coherence", "mean_radial_coherence_bootstrap_err", r"$\langle C_{\rm radial}\rangle$", False, False, 1.0, "Radial coherence"),
            ("mean_transverse_power", "mean_transverse_power_bootstrap_err", r"$\langle S_\phi^2\rangle$", False, False, 0.0, "Transverse power"),
        ]
    else:
        panels = [
            ("H_S_weighted_mean", "H_S_bootstrap_err", r"$H_S=\langle S_\phi\rangle$", True, False, 0.0, "Handedness"),
            ("mean_abs_projected_twist", "mean_abs_projected_twist_bootstrap_err", r"$\langle |S_\phi|\rangle$", False, False, 0.0, "Spirality/Torsion"),
            ("mean_radial_coherence", "mean_radial_coherence_bootstrap_err", r"$\langle C_{\rm radial}\rangle$", False, False, 1.0, "Radial coherence"),
            ("mean_transverse_power", "mean_transverse_power_bootstrap_err", r"$\langle S_\phi^2\rangle$", False, False, 0.0, "Transverse power"),
        ]

    fig, axes = plt.subplots(2, 2, figsize=(15.2, 11.0), sharex=True)
    csv_rows = []
    for pi, (ax, (ycol, ecol, ylabel, zline, to_angle, ref, panel_name)) in enumerate(zip(axes.ravel(), panels)):
        r = _plot_catalog_band(ax, dr17, ycol=ycol, ecol=ecol, to_angle=to_angle, label=dr17_label, kind="dr17")
        if r is not None:
            r.insert(0, "panel_ycol", ycol); r.insert(1, "angle_version", bool(angle_version)); csv_rows.append(r)
        if dr12 is not None:
            r = _plot_catalog_band(ax, dr12, ycol=ycol, ecol=ecol, to_angle=to_angle, label=dr12_label, kind="dr12")
            if r is not None:
                r.insert(0, "panel_ycol", ycol); r.insert(1, "angle_version", bool(angle_version)); csv_rows.append(r)
        if zline:
            ax.axhline(0.0, color="0.35", lw=1.1, ls="--", zorder=0)
        ax.set_ylabel(ylabel, fontsize=20)
        ax.grid(alpha=0.22)
        ax.tick_params(labelsize=18, length=5.0, width=1.0)
        ax.text(0.50, 0.965, panel_name, transform=ax.transAxes, ha="center", va="top",
                fontsize=22, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.70))
        if pi == 0:
            ax.text(0.50, 0.84, "Left-handed", transform=ax.transAxes, ha="center", va="center",
                    fontsize=17, color="0.35", alpha=0.82)
            ax.text(0.50, 0.16, "Right-handed", transform=ax.transAxes, ha="center", va="center",
                    fontsize=17, color="0.35", alpha=0.82)
            ax.legend(loc="lower left", fontsize=16, frameon=False, handletextpad=0.55, borderpad=0.15)
    axes[1, 0].set_xlabel(r"redshift $z$", fontsize=20)
    axes[1, 1].set_xlabel(r"redshift $z$", fontsize=20)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=260, bbox_inches="tight")
    plt.close(fig)
    if csv_rows:
        pd.concat(csv_rows, ignore_index=True).to_csv(out.with_suffix(".csv"), index=False)


def _plot_model_fit_comparison(zdf: pd.DataFrame, out: Path) -> None:
    """Compare WRMS and AIC improvement for log/arch/power/poly2 models."""
    setup_matplotlib()
    models = [("log", "log"), ("arch", "Arch."), ("power", "Power"), ("poly2", "Poly2")]
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.2), sharex=True)
    csv_rows = []
    for ax, metric, ylabel, zline in [
        (axes[0], "wrms_deg", "fit WRMS [deg]", False),
        (axes[1], "delta_aic_vs_radial", r"$\Delta$AIC: radial $-$ model", True),
    ]:
        for j, (m, label) in enumerate(models):
            ycol = f"mean_fit_{m}_{metric}"
            ecol = f"{ycol}_bootstrap_err"
            if ycol not in zdf.columns:
                continue
            x = zdf["z_center"].to_numpy(float) + (j - 1.5) * 0.006
            y = zdf[ycol].to_numpy(float)
            yerr = _asymmetric_yerr(zdf, ycol, ecol)
            ax.errorbar(x, y, yerr=yerr, fmt="o", ms=4.8, capsize=2.5, lw=0.95, label=label)
            temp_out = out.with_name(out.stem + f"_{m}_{metric}" + out.suffix)
            _write_plot_data_csv(zdf, ycol, ecol, temp_out)
            csv_path = temp_out.with_suffix(".csv")
            if csv_path.exists():
                try:
                    d = pd.read_csv(csv_path)
                    d.insert(0, "model", m)
                    d.insert(1, "metric", metric)
                    csv_rows.append(d)
                    csv_path.unlink()
                except Exception:
                    pass
        if zline:
            ax.axhline(0.0, color="0.35", lw=0.8, ls="--")
        ax.set_xlabel(r"redshift $z$")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.22)
        ax.legend(fontsize=8.2, frameon=False)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=240, bbox_inches="tight")
    plt.close(fig)
    if csv_rows:
        pd.concat(csv_rows, ignore_index=True).to_csv(out.with_suffix(".csv"), index=False)


def _plot_zbin_nnodes(zdf: pd.DataFrame, out: Path) -> None:
    """Plot the number of accepted nodes per redshift bin and save CSV."""
    setup_matplotlib()
    fig, ax = plt.subplots(figsize=(6.5, 4.25))
    x = zdf["z_center"].to_numpy(float)
    y = zdf["n_nodes_ok"].to_numpy(float) if "n_nodes_ok" in zdf.columns else np.full(len(zdf), np.nan)
    ax.plot(x, y, marker="o", ls="--", color="black", ms=6.4, lw=1.0)
    ax.set_xlabel(r"redshift $z$")
    ax.set_ylabel(r"accepted nodes")
    ax.grid(alpha=0.22)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=240, bbox_inches="tight")
    plt.close(fig)
    pd.DataFrame({
        "zbin_id": zdf["zbin_id"].to_numpy(int) if "zbin_id" in zdf.columns else np.arange(len(zdf)),
        "z_min": zdf["z_min"].to_numpy(float),
        "z_max": zdf["z_max"].to_numpy(float),
        "z_center_plot": x,
        "n_nodes_ok": y,
    }).to_csv(out.with_suffix(".csv"), index=False)


def _plot_zbin_panels_by_gal_hemi(zdf: pd.DataFrame, out: Path) -> None:
    """Legacy 2x2 tomographic panel split by Galactic hemisphere.

    Kept for backward compatibility with older output products. The preferred
    paper-ready panel is _plot_main_S_panels_by_gal_hemi.
    """
    setup_matplotlib()
    panels = [
        ("H_S_weighted_mean", "H_S_bootstrap_err", r"$H_S=\langle S_\phi\rangle$", True),
        ("H_beta_weighted_mean_deg", "H_beta_bootstrap_err_deg", r"$H_\beta$ [deg / $\ln r$]", True),
        ("mean_abs_projected_twist", "mean_abs_projected_twist_bootstrap_err", r"$\langle |S_\phi|\rangle$", False),
        ("mean_radial_coherence", "mean_radial_coherence_bootstrap_err", r"$\langle C_{\rm radial}\rangle$", False),
    ]
    specs = [
        ("all", "Total", "black", "o", 1.0, True),
        ("north", "Galactic north", "0.62", "^", 0.45, False),
        ("south", "Galactic south", "0.78", "v", 0.35, False),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(8.6, 6.4), sharex=True)
    csv_rows = []
    for pi, (ax, (ycol, ecol, ylabel, zline)) in enumerate(zip(axes.ravel(), panels)):
        for sample, label, color, marker, alpha, filled in specs:
            sub = zdf[zdf["sample"].eq(sample)].copy() if "sample" in zdf.columns else zdf.copy()
            if len(sub) == 0 or ycol not in sub.columns:
                continue
            x = sub["z_center"].to_numpy(float)
            y = sub[ycol].to_numpy(float)
            yerr = _asymmetric_yerr(sub, ycol, ecol)
            mfc = color if filled else "none"
            ax.errorbar(
                x, y, yerr=yerr, fmt=marker, capsize=3.0,
                lw=1.1 if sample == "all" else 0.85,
                ms=6.0 if sample == "all" else 5.2,
                color=color, ecolor=color, alpha=alpha,
                markerfacecolor=mfc, markeredgecolor=color, markeredgewidth=1.0,
                label=label if pi == 0 else None,
            )
            if sample == "all":
                ax.plot(x, y, color="black", ls="--", lw=1.0, alpha=0.85)
            temp_out = out.with_name(out.stem + f"_{sample}_{ycol}" + out.suffix)
            _write_plot_data_csv(sub, ycol, ecol, temp_out, sample=sample, dx_frac=0.0)
            csv_path = temp_out.with_suffix(".csv")
            if csv_path.exists():
                try:
                    d = pd.read_csv(csv_path)
                    d.insert(0, "panel_ycol", ycol)
                    csv_rows.append(d)
                    csv_path.unlink()
                except Exception:
                    pass
        if zline:
            ax.axhline(0.0, color="0.35", lw=0.8, ls="--")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.22)
        if pi == 0:
            ax.legend(loc="lower left", fontsize=8.3, frameon=False)
    axes[1, 0].set_xlabel(r"redshift $z$")
    axes[1, 1].set_xlabel(r"redshift $z$")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=240, bbox_inches="tight")
    plt.close(fig)
    if csv_rows:
        pd.concat(csv_rows, ignore_index=True).to_csv(out.with_suffix(".csv"), index=False)

def write_zbin_products(node_df: pd.DataFrame, cfg: Config, args: argparse.Namespace, outdir: Path) -> None:
    edges = parse_zbin_edges(args, node_df)
    if edges is None:
        return
    zdf = build_zbin_summary(node_df, cfg, edges)
    zcsv = outdir / "zbin_summary.csv"
    zjson = outdir / "zbin_summary.json"
    zdf.to_csv(zcsv, index=False)
    payload = {"zbin_edges": edges.tolist(), "rows": zdf.replace({np.nan: None}).to_dict(orient="records")}
    with open(zjson, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print("[write]", zcsv)
    print("[write]", zjson)

    plot_dir = outdir / "zbin_plots"
    fmt = str(getattr(args, "zbin_plot_format", "pdf"))
    _plot_zbin_one(zdf, "H_S_weighted_mean", "H_S_bootstrap_err", r"$H_S=\langle S_\phi\rangle$", plot_dir / f"zbin_H_S.{fmt}", zero_line=True)
    _plot_zbin_one(zdf, "H_beta_weighted_mean_deg", "H_beta_bootstrap_err_deg", r"$H_\beta$ [deg / $\ln r$]", plot_dir / f"zbin_H_beta.{fmt}", zero_line=True)
    _plot_zbin_one(zdf, "mean_abs_projected_twist", "mean_abs_projected_twist_bootstrap_err", r"$\langle |S_\phi|\rangle$", plot_dir / f"zbin_abs_S.{fmt}", zero_line=False)
    _plot_zbin_one(zdf, "mean_radial_coherence", "mean_radial_coherence_bootstrap_err", r"$\langle C_{\rm radial}\rangle$", plot_dir / f"zbin_C_radial.{fmt}", zero_line=False)
    _plot_zbin_nnodes(zdf, plot_dir / f"zbin_n_nodes_ok.{fmt}")
    _plot_zbin_panels(zdf, plot_dir / f"zbin_evolution_panels.{fmt}")
    # Entropy and fit-quality diagnostics.
    if "S_sign_entropy_norm" in zdf.columns:
        _plot_zbin_one(zdf, "S_sign_entropy_norm", "S_sign_entropy_bootstrap_err", r"sign entropy of $S_\phi$", plot_dir / f"zbin_S_sign_entropy.{fmt}", zero_line=False)
    if "S_sign_asymmetry" in zdf.columns:
        _plot_zbin_one(zdf, "S_sign_asymmetry", "S_sign_asymmetry_bootstrap_err", r"$P(S_\phi>0)-P(S_\phi<0)$", plot_dir / f"zbin_S_sign_asymmetry.{fmt}", zero_line=True)
    if "S_distribution_entropy_norm" in zdf.columns:
        _plot_zbin_one(zdf, "S_distribution_entropy_norm", "S_distribution_entropy_bootstrap_err", r"binned entropy of $P(S_\phi)$", plot_dir / f"zbin_S_distribution_entropy.{fmt}", zero_line=False)
    if "mean_arm_angle_entropy_norm" in zdf.columns:
        _plot_zbin_one(zdf, "mean_arm_angle_entropy_norm", "mean_arm_angle_entropy_bootstrap_err", r"mean arm-angle entropy", plot_dir / f"zbin_arm_angle_entropy.{fmt}", zero_line=False)
    if "mean_beta_common_fit_wrms_deg" in zdf.columns:
        _plot_zbin_one(zdf, "mean_beta_common_fit_wrms_deg", "mean_beta_common_fit_wrms_deg_bootstrap_err", r"log-spiral fit WRMS [deg]", plot_dir / f"zbin_beta_fit_wrms.{fmt}", zero_line=False)
    if "mean_beta_common_delta_aic_vs_radial" in zdf.columns:
        _plot_zbin_one(zdf, "mean_beta_common_delta_aic_vs_radial", "mean_beta_common_delta_aic_vs_radial_bootstrap_err", r"$\Delta$AIC: radial $-$ spiral", plot_dir / f"zbin_beta_delta_aic.{fmt}", zero_line=True)
    if "mean_beta_common_fit_wrms_deg" in zdf.columns and "mean_beta_common_delta_aic_vs_radial" in zdf.columns:
        _plot_zbin_fit_quality_panels(zdf, plot_dir / f"zbin_beta_fit_quality_panels.{fmt}")
    if "mean_fit_log_wrms_deg" in zdf.columns:
        _plot_model_fit_comparison(zdf, plot_dir / f"zbin_spiral_model_fit_comparison.{fmt}")

    # Galactic north/south split products. These are written in addition to the
    # total-only products above, so old workflows remain unchanged.
    zdf_hemi = build_zbin_summary_by_gal_hemi(node_df, cfg, edges)
    zcsv_hemi = outdir / "zbin_summary_by_gal_hemi.csv"
    zjson_hemi = outdir / "zbin_summary_by_gal_hemi.json"
    zdf_hemi.to_csv(zcsv_hemi, index=False)
    payload_hemi = {"zbin_edges": edges.tolist(), "rows": zdf_hemi.replace({np.nan: None}).to_dict(orient="records")}
    with open(zjson_hemi, "w", encoding="utf-8") as f:
        json.dump(payload_hemi, f, indent=2, ensure_ascii=False)
    dr12_node_zdf = _read_optional_catalog_zbin_summary(
        getattr(args, "dr12_node_zbin_summary", None),
        catalog="DR12",
        label=str(getattr(args, "dr12_label", "DR12 Chen")),
    )
    _plot_main_S_panels_catalog_compare(
        zdf,
        plot_dir / f"zbin_main_S_panels_catalogs.{fmt}",
        angle_version=False,
        zdf_dr12=dr12_node_zdf,
        dr17_label=str(getattr(args, "dr17_label", "DR17 SCONCE")),
        dr12_label=str(getattr(args, "dr12_label", "DR12 Chen")),
    )
    _plot_main_S_panels_catalog_compare(
        zdf,
        plot_dir / f"zbin_main_S_angle_panels_catalogs.{fmt}",
        angle_version=True,
        zdf_dr12=dr12_node_zdf,
        dr17_label=str(getattr(args, "dr17_label", "DR17 SCONCE")),
        dr12_label=str(getattr(args, "dr12_label", "DR12 Chen")),
    )
    _plot_main_S_panels_by_gal_hemi(zdf_hemi, plot_dir / f"zbin_main_S_panels_by_gal_hemi.{fmt}", angle_version=False)
    _plot_main_S_panels_by_gal_hemi(zdf_hemi, plot_dir / f"zbin_main_S_angle_panels_by_gal_hemi.{fmt}", angle_version=True)
    _plot_zbin_one_by_gal_hemi(zdf_hemi, "H_S_weighted_mean", "H_S_bootstrap_err", r"$H_S=\langle S_\phi\rangle$", plot_dir / f"zbin_H_S_by_gal_hemi.{fmt}", zero_line=True)
    _plot_zbin_one_by_gal_hemi(zdf_hemi, "H_beta_weighted_mean_deg", "H_beta_bootstrap_err_deg", r"$H_\beta$ [deg / $\ln r$]", plot_dir / f"zbin_H_beta_by_gal_hemi.{fmt}", zero_line=True)
    _plot_zbin_one_by_gal_hemi(zdf_hemi, "mean_abs_projected_twist", "mean_abs_projected_twist_bootstrap_err", r"$\langle |S_\phi|\rangle$", plot_dir / f"zbin_abs_S_by_gal_hemi.{fmt}", zero_line=False)
    _plot_zbin_one_by_gal_hemi(zdf_hemi, "mean_radial_coherence", "mean_radial_coherence_bootstrap_err", r"$\langle C_{\rm radial}\rangle$", plot_dir / f"zbin_C_radial_by_gal_hemi.{fmt}", zero_line=False)
    if "S_sign_entropy_norm" in zdf_hemi.columns:
        _plot_zbin_one_by_gal_hemi(zdf_hemi, "S_sign_entropy_norm", "S_sign_entropy_bootstrap_err", r"sign entropy of $S_\phi$", plot_dir / f"zbin_S_sign_entropy_by_gal_hemi.{fmt}", zero_line=False)
    if "S_sign_asymmetry" in zdf_hemi.columns:
        _plot_zbin_one_by_gal_hemi(zdf_hemi, "S_sign_asymmetry", "S_sign_asymmetry_bootstrap_err", r"$P(S_\phi>0)-P(S_\phi<0)$", plot_dir / f"zbin_S_sign_asymmetry_by_gal_hemi.{fmt}", zero_line=True)
    if "S_distribution_entropy_norm" in zdf_hemi.columns:
        _plot_zbin_one_by_gal_hemi(zdf_hemi, "S_distribution_entropy_norm", "S_distribution_entropy_bootstrap_err", r"binned entropy of $P(S_\phi)$", plot_dir / f"zbin_S_distribution_entropy_by_gal_hemi.{fmt}", zero_line=False)
    if "mean_arm_angle_entropy_norm" in zdf_hemi.columns:
        _plot_zbin_one_by_gal_hemi(zdf_hemi, "mean_arm_angle_entropy_norm", "mean_arm_angle_entropy_bootstrap_err", r"mean arm-angle entropy", plot_dir / f"zbin_arm_angle_entropy_by_gal_hemi.{fmt}", zero_line=False)
    if "mean_beta_common_fit_wrms_deg" in zdf_hemi.columns:
        _plot_zbin_one_by_gal_hemi(zdf_hemi, "mean_beta_common_fit_wrms_deg", "mean_beta_common_fit_wrms_deg_bootstrap_err", r"log-spiral fit WRMS [deg]", plot_dir / f"zbin_beta_fit_wrms_by_gal_hemi.{fmt}", zero_line=False)
    if "mean_beta_common_delta_aic_vs_radial" in zdf_hemi.columns:
        _plot_zbin_one_by_gal_hemi(zdf_hemi, "mean_beta_common_delta_aic_vs_radial", "mean_beta_common_delta_aic_vs_radial_bootstrap_err", r"$\Delta$AIC: radial $-$ spiral", plot_dir / f"zbin_beta_delta_aic_by_gal_hemi.{fmt}", zero_line=True)
    if "mean_beta_common_fit_wrms_deg" in zdf_hemi.columns and "mean_beta_common_delta_aic_vs_radial" in zdf_hemi.columns:
        _plot_zbin_fit_quality_panels_by_gal_hemi(zdf_hemi, plot_dir / f"zbin_beta_fit_quality_panels_by_gal_hemi.{fmt}")
    _plot_zbin_panels_by_gal_hemi(zdf_hemi, plot_dir / f"zbin_evolution_panels_by_gal_hemi.{fmt}")
    print("[write]", zcsv_hemi)
    print("[write]", zjson_hemi)
    print("[write]", plot_dir)


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------

def setup_matplotlib() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Nimbus Roman", "Liberation Serif", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "axes.linewidth": 0.9,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def plot_one_node(
    node_row: pd.Series,
    seg_df: pd.DataFrame,
    sd: SliceData,
    cfg: Config,
    outdir: Path,
    fmt: str = "pdf",
) -> Optional[Path]:
    if node_row.get("status", "") != "ok":
        return None
    node_id = int(node_row["node_id"])
    ra = float(node_row["ra"])
    dec = float(node_row["dec"])
    node_x_slice, node_y_slice = project_slice_xy_mpc_h(
        np.array([ra]), np.array([dec]), ra0_deg=sd.ra0_deg, dec0_deg=sd.dec0_deg, chi_mpc_h=sd.chi_mpc_h
    )
    nx = float(node_x_slice[0])
    ny = float(node_y_slice[0])
    max_query = cfg.arm_radius_mpc_h + cfg.node_link_radius_mpc_h
    idx = np.array(sd.tree.query_ball_point([nx, ny], r=max_query), dtype=np.int64)
    if len(idx) == 0:
        return None
    xloc = sd.x_mpc_h[idx] - nx
    yloc = sd.y_mpc_h[idx] - ny
    subseg = seg_df[seg_df["node_id"].astype(int) == node_id].copy()
    if len(subseg) == 0:
        return None

    fig, ax = plt.subplots(figsize=(6.8, 6.2))
    ax.scatter(xloc, yloc, s=4, color="0.72", alpha=0.45, rasterized=True, label="SCONCE filament points")
    ax.scatter([0.0], [0.0], marker="*", s=95, color="k", zorder=6, label="intersection")

    # draw arm measured centroids + fitted common spiral
    beta = float(node_row.get("beta_common_rad", np.nan))
    beta_err = float(node_row.get("beta_common_err_rad", np.nan))
    for arm_id, arm_sub in subseg.groupby("arm_local_id"):
        arm_sub = arm_sub.sort_values("arm_segment_index")
        x = arm_sub["x_mpc_h"].to_numpy(float)
        y = arm_sub["y_mpc_h"].to_numpy(float)
        ax.plot(x, y, marker="o", lw=1.8, ms=4.0, label=f"arm {int(arm_id)}")

        intercept = float(arm_sub["fit_intercept_arm_rad"].iloc[0]) if "fit_intercept_arm_rad" in arm_sub else np.nan
        if np.isfinite(beta) and np.isfinite(intercept):
            rmin = max(float(np.nanmin(arm_sub["rho_mpc_h"])), 1e-3)
            rmax = min(float(np.nanmax(arm_sub["rho_mpc_h"])), cfg.arm_radius_mpc_h)
            if rmax > rmin:
                rr = np.linspace(rmin, rmax, 120)
                rref = float(np.nanmedian(subseg["rho_mpc_h"]))
                phi_fit = intercept + beta * np.log(rr / rref)
                ax.plot(rr * np.cos(phi_fit), rr * np.sin(phi_fit), color="0.15", alpha=0.22, lw=2.2)
                if np.isfinite(beta_err) and beta_err > 0:
                    phi_lo = intercept + (beta - beta_err) * np.log(rr / rref)
                    phi_hi = intercept + (beta + beta_err) * np.log(rr / rref)
                    xpoly = np.r_[rr * np.cos(phi_lo), (rr[::-1] * np.cos(phi_hi[::-1]))]
                    ypoly = np.r_[rr * np.sin(phi_lo), (rr[::-1] * np.sin(phi_hi[::-1]))]
                    ax.fill(xpoly, ypoly, color="0.2", alpha=0.08, lw=0)

    circle = plt.Circle((0, 0), cfg.arm_radius_mpc_h, fill=False, color="0.45", lw=0.8, ls="--")
    ax.add_patch(circle)
    ax.axhline(0, color="0.86", lw=0.7)
    ax.axvline(0, color="0.86", lw=0.7)
    pad = max_query * 1.08
    ax.set_xlim(-pad, pad)
    ax.set_ylim(-pad, pad)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"$\Delta x$ [$h^{-1}\,{\rm Mpc}$]")
    ax.set_ylabel(r"$\Delta y$ [$h^{-1}\,{\rm Mpc}$]")
    S = node_row.get("S_tangential", np.nan)
    Serr = node_row.get("S_tangential_err", np.nan)
    bdeg = node_row.get("beta_common_deg", np.nan)
    bdegerr = node_row.get("beta_common_err_deg", np.nan)
    txt = (
        f"node {node_id}\n"
        f"RA={float(node_row.get('ra', np.nan)):.3f}°, Dec={float(node_row.get('dec', np.nan)):.3f}°\n"
        f"z={float(node_row['z_low']):.3f}\n"
        f"$S_\\phi$={S:.3f} $\\pm$ {Serr:.3f}\n"
        f"$\\beta$={bdeg:.2f} $\\pm$ {bdegerr:.2f} deg/ln r\n"
        f"arms={int(node_row.get('n_arms_valid', 0))}, segments={int(node_row.get('n_arm_segments_total', 0))}"
    )
    ax.text(0.03, 0.97, txt, transform=ax.transAxes, va="top", ha="left", fontsize=8.5,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.75", alpha=0.86))
    ax.legend(loc="lower right", fontsize=7.0, frameon=True, ncol=1)
    fig.tight_layout()

    outdir.mkdir(parents=True, exist_ok=True)
    out = outdir / f"spiral_node_{node_id:07d}_z{float(node_row['z_low']):.5f}.{fmt}"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out



def _assigned_raw_points_for_gallery(node_row: pd.Series, seg_df: pd.DataFrame, sd: SliceData, cfg: Config) -> Tuple[np.ndarray, np.ndarray]:
    """Return raw SCONCE points assigned to the plotted arms.

    The selection follows the already measured arm segments: for each segment we
    keep raw filament-map points in the corresponding radial bin and within the
    arm-opening angle of the measured segment centroid.  This avoids plotting
    unrelated nearby filaments while showing the points that supported the arm
    measurements.
    """
    try:
        ra = float(node_row["ra"])
        dec = float(node_row["dec"])
        node_x_slice, node_y_slice = project_slice_xy_mpc_h(
            np.array([ra]), np.array([dec]), ra0_deg=sd.ra0_deg, dec0_deg=sd.dec0_deg, chi_mpc_h=sd.chi_mpc_h
        )
        nx = float(node_x_slice[0])
        ny = float(node_y_slice[0])
        max_query = cfg.arm_radius_mpc_h + cfg.node_link_radius_mpc_h
        all_idx = np.array(sd.tree.query_ball_point([nx, ny], r=max_query), dtype=np.int64)
        if all_idx.size == 0:
            return np.array([], dtype=float), np.array([], dtype=float)
        dx = sd.x_mpc_h[all_idx] - nx
        dy = sd.y_mpc_h[all_idx] - ny
        rho = np.hypot(dx, dy)
        phi = np.arctan2(dy, dx)
        node_id = int(node_row["node_id"])
        subseg = seg_df[seg_df["node_id"].astype(int) == node_id].copy()
        if len(subseg) == 0:
            return np.array([], dtype=float), np.array([], dtype=float)
        keep = np.zeros(len(all_idx), dtype=bool)
        opening = float(cfg.arm_opening_deg) * DEG2RAD
        for _, seg in subseg.iterrows():
            k = int(seg.get("arm_segment_index", 0))
            r_lo = max(float(cfg.inner_exclusion_mpc_h), k * float(cfg.segment_length_mpc_h))
            r_hi = (k + 1) * float(cfg.segment_length_mpc_h)
            phi0 = float(seg.get("phi_rad", np.nan))
            if not np.isfinite(phi0):
                phi0 = math.atan2(float(seg.get("y_mpc_h", np.nan)), float(seg.get("x_mpc_h", np.nan)))
            if not np.isfinite(phi0):
                continue
            dphi = np.abs(angular_diff_rad(phi, phi0))
            keep |= (rho >= r_lo) & (rho < r_hi) & (dphi <= opening)
        return dx[keep], dy[keep]
    except Exception:
        return np.array([], dtype=float), np.array([], dtype=float)


def _plot_compact_node_axis(ax, node_row: pd.Series, seg_df: pd.DataFrame, cfg: Config, *, sd: Optional[SliceData] = None, show_text: bool = True, line_color: str = '0.22', centroid_color: str = 'k', raw_color: str = '0.58', link_color: str = '0.80') -> None:
    """Compact node diagnostic used by the multi-panel gallery.

    Gray small points are the raw SCONCE filament-map points assigned to the
    selected arms.  Dark gray thick lines are the measured 10 Mpc/h arm-segment
    paths, drawn behind the black segment-centroid measurements.  The gray filled
    circle marks the node-link radius used to identify the initial arms.
    """
    node_id = int(node_row["node_id"])
    subseg = seg_df[seg_df["node_id"].astype(int) == node_id].copy()
    ax.set_aspect("equal", adjustable="box")
    max_r = float(cfg.arm_radius_mpc_h)
    pad = max(max_r, float(cfg.node_link_radius_mpc_h)) * 1.12

    # Light gray seed/link region and radial bin circles.
    link = plt.Circle((0.0, 0.0), float(cfg.node_link_radius_mpc_h), color=link_color, alpha=0.18, lw=0, zorder=0)
    ax.add_patch(link)
    for k in range(1, int(cfg.n_arm_segments) + 1):
        r = k * float(cfg.segment_length_mpc_h)
        circ = plt.Circle((0.0, 0.0), r, fill=False, color="0.76", lw=0.50, ls="--", alpha=0.70, zorder=0)
        ax.add_patch(circ)

    # Raw assigned SCONCE points, smaller and behind the measured segment points.
    if sd is not None:
        xp, yp = _assigned_raw_points_for_gallery(node_row, seg_df, sd, cfg)
        if len(xp) > 0:
            ax.scatter(xp, yp, s=2.8, color=raw_color, alpha=0.30, linewidths=0, rasterized=True, zorder=1)

    # Node as black x inside the gray circle.
    ax.scatter([0.0], [0.0], marker="x", s=30, color="k", lw=0.9, zorder=6)
    if len(subseg) > 0:
        for _, arm_sub in subseg.groupby("arm_local_id"):
            arm_sub = arm_sub.sort_values("arm_segment_index")
            x = arm_sub["x_mpc_h"].to_numpy(float)
            y = arm_sub["y_mpc_h"].to_numpy(float)
            # Dark gray segment path behind measurement points.
            ax.plot(x, y, color=line_color, lw=2.5, alpha=0.12, zorder=2)
            ax.scatter(x, y, s=8.0, color=centroid_color, alpha=0.68, zorder=4)
    ax.axhline(0, color="0.88", lw=0.4, zorder=0)
    ax.axvline(0, color="0.88", lw=0.4, zorder=0)
    ax.set_xlim(-pad, pad)
    ax.set_ylim(-pad, pad)
    ax.tick_params(labelsize=8.0, length=2.4, pad=0.8, width=0.7)
    ax.set_xticks([-20, 0, 20])
    ax.set_yticks([-20, 0, 20])
    if show_text:
        txt = (
            rf"$\alpha$={float(node_row.get('ra', np.nan)):.1f}°, "
            rf"$\delta$={float(node_row.get('dec', np.nan)):.1f}°, "
            rf"$z$={float(node_row.get('z_low', np.nan)):.3f}"
        )
        ax.text(0.04, 0.965, txt, transform=ax.transAxes, va="top", ha="left", fontsize=6.8,
                bbox=dict(boxstyle="round,pad=0.10", fc="white", ec="none", alpha=0.62), zorder=7)


def _save_gallery_panel(nodes: pd.DataFrame, seg_df: pd.DataFrame, slices: Dict[float, SliceData], cfg: Config, out: Path, title: str, *, ncols: int, nrows: int) -> None:
    setup_matplotlib()
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 1.58, nrows * 1.58), sharex=True, sharey=True)
    axes_arr = np.asarray(axes).reshape(nrows, ncols)
    for ax in axes_arr.ravel():
        ax.set_visible(False)
    name = str(out.name).lower()
    if "sconce" in name:
        line_color, centroid_color, raw_color, link_color = "seagreen", "seagreen", "lightgreen", "lightgreen"
    elif "carron" in name or "carrón" in name:
        line_color, centroid_color, raw_color, link_color = "royalblue", "royalblue", "lightskyblue", "lightskyblue"
    elif "chen" in name:
        line_color, centroid_color, raw_color, link_color = "orangered", "orangered", "lightsalmon", "lightsalmon"
    else:
        line_color, centroid_color, raw_color, link_color = "0.22", "k", "0.58", "0.80"
    for i, (_, row) in enumerate(nodes.head(ncols * nrows).iterrows()):
        ax = axes_arr.ravel()[i]
        ax.set_visible(True)
        sd = slices.get(float(row.get("z_key", np.nan)))
        _plot_compact_node_axis(
            ax, row, seg_df, cfg, sd=sd, show_text=True,
            line_color=line_color, centroid_color=centroid_color,
            raw_color=raw_color, link_color=link_color,
        )
    fig.suptitle(title, y=0.994, fontsize=22, fontweight="bold")
    fig.supxlabel(r"$\Delta x$ [$h^{-1}$ Mpc]", y=0.006, fontsize=20)
    fig.supylabel(r"$\Delta y$ [$h^{-1}$ Mpc]", x=0.003, fontsize=20)
    # Manual spacing keeps axes frames touching horizontally and vertically.
    fig.subplots_adjust(left=0.052, right=0.998, bottom=0.035, top=0.970, wspace=-0.045, hspace=-0.045)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=260, bbox_inches="tight")
    plt.close(fig)



def _gallery_catalog_style(catalog_key: str) -> tuple[str, str, str, str]:
    """Return line/centroid/raw/link colours matching the native gallery style."""
    key = str(catalog_key).lower()
    if "sconce" in key:
        return "seagreen", "seagreen", "lightgreen", "lightgreen"
    if "carron" in key or "carr" in key:
        return "royalblue", "royalblue", "lightskyblue", "lightskyblue"
    if "chen" in key:
        return "orangered", "orangered", "lightsalmon", "lightsalmon"
    return "0.22", "k", "0.58", "0.80"


def _safe_gallery_id(x) -> str:
    try:
        if np.isfinite(float(x)):
            return str(int(round(float(x))))
    except Exception:
        pass
    return str(x).replace("/", "_").replace(" ", "_")


def _save_random_individual_fans_pngs(
    node_df: pd.DataFrame,
    seg_df: pd.DataFrame,
    slices: Dict[float, SliceData],
    cfg: Config,
    args: argparse.Namespace,
) -> None:
    """Save one PNG per randomly selected accepted FANS using the native gallery plotter.

    This uses the same in-memory slice/KDTree plotting logic as the multi-panel
    native galleries, so the individual PNGs show the reconstructed-ridge points
    used by the catalogue around each node, not only the segment centroids saved
    in the CSV products.
    """
    if not bool(getattr(args, "make_random_fans_gallery", False)):
        return
    if len(node_df) == 0 or len(seg_df) == 0:
        return
    outroot = Path(getattr(args, "random_fans_gallery_outdir", "random_fans_gallery")).expanduser().resolve()
    catalog_key = str(getattr(args, "random_fans_gallery_catalog_key", "catalog"))
    n_per = int(getattr(args, "random_fans_gallery_n_per_arm_count", 1000))
    arm_counts = [int(x) for x in getattr(args, "random_fans_gallery_arm_counts", [2, 3])]
    seed = int(getattr(args, "gallery_seed", None) if getattr(args, "gallery_seed", None) is not None else getattr(args, "seed", 2026))
    rng = np.random.default_rng(seed + 1701)

    ok = node_df[node_df.get("status", pd.Series("ok", index=node_df.index)).astype(str).eq("ok")].copy()
    if "node_id" not in ok.columns or "node_id" not in seg_df.columns:
        print(f"[gallery-random-native] {catalog_key}: missing node_id; skipped")
        return
    valid_nodes = set(pd.to_numeric(seg_df["node_id"], errors="coerce").dropna().astype(int).tolist())
    ok = ok[ok["node_id"].apply(lambda x: np.isfinite(float(x)) if str(x) != 'nan' else False)].copy()
    ok = ok[ok["node_id"].astype(float).round().astype(int).isin(valid_nodes)].copy()
    if len(ok) == 0:
        print(f"[gallery-random-native] {catalog_key}: no accepted nodes with segments; skipped")
        return
    if "n_arms_valid" not in ok.columns:
        arm_counts_by_node = seg_df.groupby("node_id")["arm_local_id"].nunique()
        ok["n_arms_valid"] = ok["node_id"].map(arm_counts_by_node)
    line_color, centroid_color, raw_color, link_color = _gallery_catalog_style(catalog_key)
    setup_matplotlib()
    for ac in arm_counts:
        pool = ok[pd.to_numeric(ok["n_arms_valid"], errors="coerce").fillna(-1).astype(int).eq(int(ac))].copy()
        if len(pool) == 0:
            print(f"[gallery-random-native] {catalog_key}: no accepted FANS with {ac} arms")
            continue
        nsel = min(n_per, len(pool))
        chosen_idx = rng.choice(pool.index.to_numpy(), size=nsel, replace=False)
        chosen = pool.loc[chosen_idx].copy()
        subdir = outroot / f"{catalog_key}_{int(ac)}arms"
        subdir.mkdir(parents=True, exist_ok=True)
        rows = []
        for jj, (_, row) in enumerate(chosen.iterrows()):
            node_id = row.get("node_id")
            fans_id = f"{catalog_key}_FANS_{_safe_gallery_id(node_id)}"
            fig, ax = plt.subplots(1, 1, figsize=(3.2, 3.2))
            sd = slices.get(float(row.get("z_key", np.nan)))
            _plot_compact_node_axis(
                ax, row, seg_df, cfg, sd=sd, show_text=True,
                line_color=line_color, centroid_color=centroid_color,
                raw_color=raw_color, link_color=link_color,
            )
            ax.set_title(fans_id, fontsize=8.5, pad=2.0)
            out_png = subdir / f"{fans_id}.png"
            fig.savefig(out_png, dpi=220, bbox_inches="tight")
            plt.close(fig)
            rows.append({"catalog_key": catalog_key, "fans_id": fans_id, "node_id": node_id, "n_arms": int(ac), "png": str(out_png)})
            if (jj + 1) % 100 == 0:
                print(f"[gallery-random-native] {catalog_key} {ac} arms: {jj+1}/{nsel}")
        pd.DataFrame(rows).to_csv(subdir / "selected_fans.csv", index=False)
        print(f"[gallery-random-native] wrote {nsel} PNGs to {subdir}")

def make_node_gallery(node_df: pd.DataFrame, seg_df: pd.DataFrame, slices: Dict[float, SliceData], cfg: Config, args: argparse.Namespace, outdir: Path) -> Optional[Path]:
    """Make two separate galleries: optimal and typical."""
    if len(node_df) == 0 or len(seg_df) == 0:
        return None
    ok = node_df[node_df["status"].eq("ok")].copy()
    ok = ok[ok["node_id"].isin(seg_df["node_id"].astype(int).unique())].copy()
    if len(ok) == 0:
        return None
    ncols = int(getattr(args, "gallery_ncols", 8))
    nrows_total = int(getattr(args, "gallery_nrows", 12))
    nrows = max(1, nrows_total)
    n_each = ncols * nrows
    rng = np.random.default_rng(args.gallery_seed if getattr(args, "gallery_seed", None) is not None else args.seed)

    Sabs = np.abs(ok.get("S_tangential", pd.Series(np.nan, index=ok.index)).to_numpy(float))
    nseg = np.maximum(ok.get("n_arm_segments_total", pd.Series(1, index=ok.index)).to_numpy(float), 1.0)
    crad = np.clip(ok.get("mean_t_radial", pd.Series(1, index=ok.index)).to_numpy(float), 0.0, 1.0)
    reject = ok.get("segment_radial_reject_sigma", pd.Series(np.nan, index=ok.index)).to_numpy(float)
    reject = np.where(np.isfinite(reject), np.maximum(reject, 0.0), 0.0)
    score = Sabs * np.sqrt(nseg) * crad * (1.0 + reject)
    score = np.where(np.isfinite(score), score, -np.inf)
    ok = ok.assign(gallery_score=score)

    arms_series = pd.to_numeric(ok.get("n_arms_valid", pd.Series(0, index=ok.index)), errors="coerce").fillna(-1).astype(int)
    npos_series = pd.to_numeric(ok.get("n_arms_positive_tphi", pd.Series(0, index=ok.index)), errors="coerce").fillna(0).astype(int)
    nneg_series = pd.to_numeric(ok.get("n_arms_negative_tphi", pd.Series(0, index=ok.index)), errors="coerce").fillna(0).astype(int)
    optimal_exact_arms = int(getattr(args, "gallery_optimal_arms", 0))
    optimal_min_arms = int(getattr(args, "gallery_optimal_min_arms", 3))
    if optimal_exact_arms > 0:
        opt_mask = arms_series.eq(optimal_exact_arms)
    else:
        opt_mask = arms_series.ge(optimal_min_arms)
    if not bool(getattr(args, "gallery_allow_mixed_arm_sign", False)):
        # Require all valid arms to have the same sign of mean_t_phi.
        opt_mask &= (npos_series.eq(arms_series) | nneg_series.eq(arms_series))
    opt_base = ok[opt_mask].copy()
    if len(opt_base) == 0:
        # Fallback keeps the script usable, but the CSV will reveal that the strict
        # optimal-selection criteria found no nodes.
        opt_base = ok.copy()
    opt_sorted = opt_base.sort_values(["gallery_score", "segment_radial_reject_sigma", "S_tangential_abs"], ascending=False).copy()
    pool_n = max(n_each, int(math.ceil(float(getattr(args, "gallery_optimal_pool_frac", 0.02)) * len(opt_sorted))))
    pool_n = min(pool_n, len(opt_sorted))
    opt_pool = opt_sorted.head(pool_n)
    opt_take = min(n_each, len(opt_pool))
    opt_idx = rng.choice(opt_pool.index.to_numpy(), size=opt_take, replace=False) if opt_take > 0 else np.array([], dtype=int)
    opt_nodes = ok.loc[opt_idx].sort_values(["gallery_score", "segment_radial_reject_sigma"], ascending=False).copy() if len(opt_idx) else ok.head(0).copy()

    remaining = ok.drop(index=opt_idx, errors="ignore")
    typ_take = min(n_each, len(remaining))
    typ_idx = rng.choice(remaining.index.to_numpy(), size=typ_take, replace=False) if typ_take > 0 else np.array([], dtype=int)
    typ_nodes = remaining.loc[typ_idx].copy() if len(typ_idx) else ok.head(0).copy()

    outdir.mkdir(parents=True, exist_ok=True)
    base = str(getattr(args, "gallery_filename", "node_spiral_gallery"))
    out_opt = outdir / f"{base}_optimal.{args.plot_format}"
    out_typ = outdir / f"{base}_typical.{args.plot_format}"
    _save_gallery_panel(opt_nodes, seg_df, slices, cfg, out_opt, "Optimal examples", ncols=ncols, nrows=nrows)
    _save_gallery_panel(typ_nodes, seg_df, slices, cfg, out_typ, "Typical examples", ncols=ncols, nrows=nrows)

    selection_tables = []
    sel_opt = opt_nodes.copy(); sel_opt.insert(0, "gallery_rank", np.arange(len(sel_opt))); sel_opt.insert(1, "gallery_class", "optimal")
    sel_typ = typ_nodes.copy(); sel_typ.insert(0, "gallery_rank", np.arange(len(sel_typ))); sel_typ.insert(1, "gallery_class", "typical")
    selection_tables.extend([sel_opt, sel_typ])

    # Additional typical galleries with a fixed number of valid arms.
    # These are useful morphology diagnostics and do not alter the science measurements.
    for narm in [2, 3, 4]:
        narm_pool = ok[pd.to_numeric(ok.get("n_arms_valid", pd.Series(-1, index=ok.index)), errors="coerce").fillna(-1).astype(int).eq(narm)].copy()
        if len(narm_pool) == 0:
            continue
        take = min(n_each, len(narm_pool))
        idx_narm = rng.choice(narm_pool.index.to_numpy(), size=take, replace=False) if take > 0 else np.array([], dtype=int)
        nodes_narm = narm_pool.loc[idx_narm].copy() if len(idx_narm) else narm_pool.head(0).copy()
        out_narm = outdir / f"{base}_typical_{narm}arms.{args.plot_format}"
        _save_gallery_panel(nodes_narm, seg_df, slices, cfg, out_narm, f"Typical examples: {narm} arms", ncols=ncols, nrows=nrows)
        sel_narm = nodes_narm.copy(); sel_narm.insert(0, "gallery_rank", np.arange(len(sel_narm))); sel_narm.insert(1, "gallery_class", f"typical_{narm}arms")
        selection_tables.append(sel_narm)

    pd.concat(selection_tables, ignore_index=True).to_csv((outdir / base).with_suffix('.csv'), index=False)
    _save_random_individual_fans_pngs(node_df, seg_df, slices, cfg, args)
    return out_opt




def _prepare_arm_pattern_table(node_df: pd.DataFrame) -> pd.DataFrame:
    ok = node_df[node_df["status"].eq("ok")].copy() if "status" in node_df.columns else node_df.copy()
    needed = ["n_arms_valid", "n_arms_positive_tphi", "n_arms_negative_tphi"]
    for c in needed:
        if c not in ok.columns:
            ok[c] = np.nan
    ok["n_arms_valid"] = pd.to_numeric(ok["n_arms_valid"], errors="coerce").fillna(0).astype(int)
    ok["n_arms_positive_tphi"] = pd.to_numeric(ok["n_arms_positive_tphi"], errors="coerce").fillna(0).astype(int)
    ok["n_arms_negative_tphi"] = pd.to_numeric(ok["n_arms_negative_tphi"], errors="coerce").fillna(0).astype(int)
    ok = ok[ok["n_arms_valid"] > 0].copy()
    if len(ok) == 0:
        return ok
    ok["n_arms_same_majority"] = np.maximum(ok["n_arms_positive_tphi"], ok["n_arms_negative_tphi"])
    ok["n_arms_same_minority"] = np.minimum(ok["n_arms_positive_tphi"], ok["n_arms_negative_tphi"])
    ok["all_arms_same_direction"] = ok["n_arms_same_majority"].eq(ok["n_arms_valid"])
    ok["arm_sign_pattern"] = ok["n_arms_positive_tphi"].astype(str) + " positive / " + ok["n_arms_negative_tphi"].astype(str) + " negative"
    if "gal_b_deg" not in ok.columns:
        ok = add_galactic_columns(ok)
    ok["gal_sample"] = np.where(ok["gal_b_deg"].to_numpy(float) >= 0.0, "north", "south")
    return ok


def _summarize_arm_patterns_for_subset(sub: pd.DataFrame, *, scope: str, sample: str, zbin_id: int, z_min: float, z_max: float) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    rows_pattern: List[Dict[str, object]] = []
    rows_narms: List[Dict[str, object]] = []
    if len(sub) == 0:
        return rows_pattern, rows_narms
    n_total = int(len(sub))
    n_all_same_total = int(np.count_nonzero(sub["all_arms_same_direction"].to_numpy(bool)))
    for n_arms, g_n in sub.groupby("n_arms_valid"):
        n_arms = int(n_arms)
        n_with_n = int(len(g_n))
        n_same = int(np.count_nonzero(g_n["all_arms_same_direction"].to_numpy(bool)))
        rows_narms.append({
            "scope": scope,
            "sample": sample,
            "zbin_id": int(zbin_id),
            "z_min": float(z_min),
            "z_max": float(z_max),
            "n_arms_valid": n_arms,
            "n_nodes": n_with_n,
            "n_nodes_total_scope": n_total,
            "fraction_of_scope": float(n_with_n / max(n_total, 1)),
            "n_all_arms_same_direction": n_same,
            "fraction_all_arms_same_direction_within_narms": float(n_same / max(n_with_n, 1)),
            "fraction_all_arms_same_direction_total_scope": float(n_all_same_total / max(n_total, 1)),
        })
        for (npos, nneg, nmaj, nmin, allsame), g_pat in g_n.groupby([
            "n_arms_positive_tphi",
            "n_arms_negative_tphi",
            "n_arms_same_majority",
            "n_arms_same_minority",
            "all_arms_same_direction",
        ]):
            count = int(len(g_pat))
            rows_pattern.append({
                "scope": scope,
                "sample": sample,
                "zbin_id": int(zbin_id),
                "z_min": float(z_min),
                "z_max": float(z_max),
                "n_arms_valid": n_arms,
                "n_arms_positive_tphi": int(npos),
                "n_arms_negative_tphi": int(nneg),
                "n_arms_same_majority": int(nmaj),
                "n_arms_same_minority": int(nmin),
                "all_arms_same_direction": bool(allsame),
                "n_nodes": count,
                "n_nodes_total_scope": n_total,
                "n_nodes_with_this_narms": n_with_n,
                "fraction_of_scope": float(count / max(n_total, 1)),
                "fraction_within_narms": float(count / max(n_with_n, 1)),
            })
    return rows_pattern, rows_narms


def build_arm_handedness_summary(node_df: pd.DataFrame, args: argparse.Namespace) -> Tuple[pd.DataFrame, pd.DataFrame]:
    tab = _prepare_arm_pattern_table(node_df)
    if len(tab) == 0:
        return pd.DataFrame(), pd.DataFrame()
    rows_pattern: List[Dict[str, object]] = []
    rows_narms: List[Dict[str, object]] = []

    # All selected redshifts.
    zvals = tab["z_mid"].to_numpy(float) if "z_mid" in tab.columns else tab["z_low"].to_numpy(float)
    zmin = float(np.nanmin(zvals)) if np.any(np.isfinite(zvals)) else np.nan
    zmax = float(np.nanmax(zvals)) if np.any(np.isfinite(zvals)) else np.nan
    for sample_name, mask in [
        ("all", np.ones(len(tab), dtype=bool)),
        ("north", tab["gal_sample"].eq("north").to_numpy(bool)),
        ("south", tab["gal_sample"].eq("south").to_numpy(bool)),
    ]:
        rp, rn = _summarize_arm_patterns_for_subset(tab.loc[mask], scope="allz", sample=sample_name, zbin_id=-1, z_min=zmin, z_max=zmax)
        rows_pattern.extend(rp); rows_narms.extend(rn)

    # Redshift bins, if requested.
    edges = parse_zbin_edges(args, tab)
    if edges is not None:
        for ib in range(len(edges) - 1):
            z0 = float(edges[ib]); z1 = float(edges[ib + 1])
            zmask = np.isfinite(zvals) & (zvals >= z0) & ((zvals <= z1) if ib == len(edges) - 2 else (zvals < z1))
            for sample_name, smask in [
                ("all", np.ones(len(tab), dtype=bool)),
                ("north", tab["gal_sample"].eq("north").to_numpy(bool)),
                ("south", tab["gal_sample"].eq("south").to_numpy(bool)),
            ]:
                rp, rn = _summarize_arm_patterns_for_subset(tab.loc[zmask & smask], scope="zbin", sample=sample_name, zbin_id=ib, z_min=z0, z_max=z1)
                rows_pattern.extend(rp); rows_narms.extend(rn)
    return pd.DataFrame(rows_pattern), pd.DataFrame(rows_narms)


def write_arm_handedness_products(node_df: pd.DataFrame, args: argparse.Namespace, outdir: Path) -> None:
    pattern_df, narms_df = build_arm_handedness_summary(node_df, args)
    pattern_path = outdir / "arm_handedness_patterns.csv"
    narms_path = outdir / "arm_handedness_by_narms.csv"
    pattern_df.to_csv(pattern_path, index=False)
    narms_df.to_csv(narms_path, index=False)
    payload = {
        "patterns_rows": int(len(pattern_df)),
        "narms_rows": int(len(narms_df)),
    }
    if len(narms_df):
        allz = narms_df[(narms_df["scope"].eq("allz")) & (narms_df["sample"].eq("all"))]
        payload["allz_by_narms"] = allz.replace({np.nan: None}).to_dict(orient="records")
    with open(outdir / "arm_handedness_summary.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print("[write]", pattern_path)
    print("[write]", narms_path)


def _require_healpy():
    try:
        import healpy as hp  # type: ignore
        return hp
    except Exception as exc:
        raise ImportError(
            "The --make-sky-maps option requires healpy. Install healpy in the runtime environment, "
            "or rerun without --make-sky-maps."
        ) from exc


def _resolve_sky_nsides(args: argparse.Namespace) -> Tuple[int, int]:
    """Return (nside_zbin, nside_global), respecting the deprecated --sky-nside alias."""
    if getattr(args, "sky_nside", None) is not None:
        ns = int(args.sky_nside)
        return ns, ns
    return int(getattr(args, "sky_nside_zbin", 4)), int(getattr(args, "sky_nside_global", 8))


def healpix_node_ipix(gal_l_deg: np.ndarray, gal_b_deg: np.ndarray, nside: int) -> np.ndarray:
    """HEALPix RING pixel index for Galactic longitude/latitude in degrees."""
    hp = _require_healpy()
    l = np.asarray(gal_l_deg, dtype=float) % 360.0
    b = np.asarray(gal_b_deg, dtype=float)
    theta = np.deg2rad(90.0 - np.clip(b, -90.0, 90.0))
    phi = np.deg2rad(l)
    return hp.ang2pix(int(nside), theta, phi, nest=False)


def build_healpix_map_summary(
    node_df: pd.DataFrame,
    cfg: Config,
    *,
    nside: int,
    min_count: int,
    z_edges: Optional[np.ndarray] = None,
    label: str = "allz",
) -> Tuple[pd.DataFrame, Dict[Tuple[int, str], Dict[str, np.ndarray]]]:
    """Build per-pixel summaries and full-sky HEALPix arrays."""
    hp = _require_healpy()
    ok = node_df[node_df["status"].eq("ok")].copy() if "status" in node_df.columns else node_df.copy()
    if len(ok) == 0:
        return pd.DataFrame(), {}
    if "gal_l_deg" not in ok.columns or "gal_b_deg" not in ok.columns:
        ok = add_galactic_columns(ok)
    zcol_name = "z_mid" if "z_mid" in ok.columns else "z_low"
    zvals = ok[zcol_name].to_numpy(float)
    ok = ok.assign(_w_node=node_weights(ok, cfg.node_weight))
    ok = ok.assign(_hp_ipix=healpix_node_ipix(ok["gal_l_deg"].to_numpy(float), ok["gal_b_deg"].to_numpy(float), nside))
    if z_edges is None:
        finite = zvals[np.isfinite(zvals)]
        z0 = float(np.nanmin(finite)) if len(finite) else np.nan
        z1 = float(np.nanmax(finite)) if len(finite) else np.nan
        bins = [(-1, z0, z1, np.isfinite(zvals))]
    else:
        bins = []
        for ib in range(len(z_edges) - 1):
            z0 = float(z_edges[ib])
            z1 = float(z_edges[ib + 1])
            mask = np.isfinite(zvals) & (zvals >= z0) & ((zvals <= z1) if ib == len(z_edges) - 2 else (zvals < z1))
            bins.append((ib, z0, z1, mask))
    npix = hp.nside2npix(int(nside))
    maps: Dict[Tuple[int, str], Dict[str, np.ndarray]] = {}
    rows: List[Dict[str, object]] = []
    value_specs = [
        ("H_S_weighted_mean", "S_tangential"),
        ("mean_abs_projected_twist", "S_tangential_abs"),
        ("mean_transverse_power", "S_tangential_sq"),
        ("mean_radial_coherence", "mean_t_radial"),
    ]
    theta_c, phi_c = hp.pix2ang(int(nside), np.arange(npix), nest=False)
    lon_c = np.rad2deg(phi_c) % 360.0
    lat_c = 90.0 - np.rad2deg(theta_c)
    for ib, z0, z1, mask in bins:
        sub = ok.loc[mask].copy()
        for value_name, _ in value_specs:
            maps[(int(ib), value_name)] = {
                "map": np.full(npix, hp.UNSEEN, dtype=float),
                "count": np.zeros(npix, dtype=float),
                "weight": np.zeros(npix, dtype=float),
            }
        if len(sub) == 0:
            continue
        for pix, g in sub.groupby("_hp_ipix"):
            pix = int(pix)
            n = int(len(g))
            if n < int(min_count):
                continue
            w = g["_w_node"].to_numpy(float)
            base = {
                "map_label": label,
                "nside": int(nside),
                "zbin_id": int(ib),
                "z_min": float(z0),
                "z_max": float(z1),
                "z_center": float(0.5 * (z0 + z1)) if np.isfinite(z0) and np.isfinite(z1) else np.nan,
                "hp_ipix": pix,
                "gal_l_center_deg": float(lon_c[pix]),
                "gal_b_center_deg": float(lat_c[pix]),
                "n_nodes_ok": n,
                "weight_sum": float(np.nansum(w)),
                "gal_hemi": "north" if float(lat_c[pix]) >= 0.0 else "south",
            }
            vals = {}
            for value_name, col in value_specs:
                val = weighted_global_mean(g[col].to_numpy(float), w) if col in g.columns else np.nan
                vals[value_name] = val
                maps[(int(ib), value_name)]["map"][pix] = val
                maps[(int(ib), value_name)]["count"][pix] = n
                maps[(int(ib), value_name)]["weight"][pix] = float(np.nansum(w))
            row = dict(base)
            row.update(vals)
            rows.append(row)
    return pd.DataFrame(rows), maps


def _plot_healpix_panel_maps(
    maps: Dict[Tuple[int, str], Dict[str, np.ndarray]],
    summary_df: pd.DataFrame,
    value_name: str,
    out: Path,
    *,
    title: str,
    cmap: str,
    symmetric: bool,
) -> None:
    hp = _require_healpy()
    keys = sorted([k for k in maps.keys() if k[1] == value_name], key=lambda x: x[0])
    if not keys:
        return
    vals: List[float] = []
    for k in keys:
        m = np.asarray(maps[k]["map"], dtype=float)
        vals.extend(m[np.isfinite(m) & (m != hp.UNSEEN)].tolist())
    vals = np.asarray(vals, dtype=float)
    if len(vals) == 0:
        return
    if symmetric:
        vmax = float(np.nanmax(np.abs(vals)))
        vmin = -vmax
    else:
        vmin = float(np.nanmin(vals))
        vmax = float(np.nanmax(vals))
    n = len(keys)
    setup_matplotlib()
    fig = plt.figure(figsize=(4.4 * n, 3.5))
    last = None
    for i, k in enumerate(keys, start=1):
        ib = int(k[0])
        m = np.asarray(maps[k]["map"], dtype=float)
        sub = summary_df[summary_df["zbin_id"].eq(ib)]
        if len(sub) > 0 and ib >= 0:
            z0 = float(sub["z_min"].iloc[0])
            z1 = float(sub["z_max"].iloc[0])
            label = f"{z0:.2f} ≤ z {'≤' if i == n else '<'} {z1:.2f}"
        else:
            label = "all selected z"
        hp.mollview(
            m,
            fig=fig.number,
            sub=(1, n, i),
            title=label,
            coord="G",
            cmap=cmap,
            min=vmin,
            max=vmax,
            cbar=False,
            notext=True,
            nest=False,
        )
        hp.graticule(dpar=30, dmer=60, verbose=False, alpha=0.35)
    fig.suptitle(title, fontsize=15, y=0.98)
    import matplotlib as mpl
    sm = mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(vmin=vmin, vmax=vmax), cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=fig.axes, orientation="horizontal", fraction=0.055, pad=0.05)
    cbar.set_label(value_name)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=240, bbox_inches="tight")
    plt.close(fig)


def write_healpix_maps(maps: Dict[Tuple[int, str], Dict[str, np.ndarray]], outdir: Path, *, prefix: str) -> None:
    hp = _require_healpy()
    outdir.mkdir(parents=True, exist_ok=True)
    wrote_counts: set = set()
    for (ib, value_name), payload in maps.items():
        hp.write_map(outdir / f"{prefix}_zbin{ib}_{value_name}.fits", np.asarray(payload["map"], dtype=float), overwrite=True, dtype=np.float64)
        if ib not in wrote_counts:
            hp.write_map(outdir / f"{prefix}_zbin{ib}_count.fits", np.asarray(payload["count"], dtype=float), overwrite=True, dtype=np.float64)
            wrote_counts.add(ib)


def write_sky_map_products(node_df: pd.DataFrame, cfg: Config, args: argparse.Namespace, outdir: Path) -> None:
    """Write HEALPix maps for redshift bins and for the full selected sample."""
    _require_healpy()
    edges = parse_zbin_edges(args, node_df)
    nside_zbin, nside_global = _resolve_sky_nsides(args)
    sky_dir = outdir / "sky_maps"
    sky_dir.mkdir(parents=True, exist_ok=True)
    if edges is not None:
        zdf, zmaps = build_healpix_map_summary(
            node_df,
            cfg,
            nside=nside_zbin,
            min_count=int(getattr(args, "sky_min_count", 3)),
            z_edges=edges,
            label="zbin",
        )
        zdf.to_csv(sky_dir / "sky_map_summary_zbins.csv", index=False)
        write_healpix_maps(zmaps, sky_dir, prefix=f"nside{nside_zbin}_zbin")
        _plot_healpix_panel_maps(zmaps, zdf, "H_S_weighted_mean", sky_dir / f"sky_map_H_S_zbins_nside{nside_zbin}.{args.plot_format}", title=f"Handedness, NSIDE={nside_zbin}", cmap="RdBu_r", symmetric=True)
        _plot_healpix_panel_maps(zmaps, zdf, "mean_abs_projected_twist", sky_dir / f"sky_map_abs_S_zbins_nside{nside_zbin}.{args.plot_format}", title=f"Spirality amplitude, NSIDE={nside_zbin}", cmap="Greys", symmetric=False)
        print("[write]", sky_dir / "sky_map_summary_zbins.csv")
    gdf, gmaps = build_healpix_map_summary(
        node_df,
        cfg,
        nside=nside_global,
        min_count=int(getattr(args, "sky_min_count", 3)),
        z_edges=None,
        label="allz",
    )
    gdf.to_csv(sky_dir / "sky_map_summary_allz.csv", index=False)
    write_healpix_maps(gmaps, sky_dir, prefix=f"nside{nside_global}_allz")
    _plot_healpix_panel_maps(gmaps, gdf, "H_S_weighted_mean", sky_dir / f"sky_map_H_S_allz_nside{nside_global}.{args.plot_format}", title=f"Handedness, all z, NSIDE={nside_global}", cmap="RdBu_r", symmetric=True)
    _plot_healpix_panel_maps(gmaps, gdf, "mean_abs_projected_twist", sky_dir / f"sky_map_abs_S_allz_nside{nside_global}.{args.plot_format}", title=f"Spirality amplitude, all z, NSIDE={nside_global}", cmap="Greys", symmetric=False)
    meta = {
        "nside_zbin": int(nside_zbin),
        "nside_global": int(nside_global),
        "sky_min_count": int(getattr(args, "sky_min_count", 3)),
        "zbin_pixels": int(len(zdf)) if edges is not None else 0,
        "allz_pixels": int(len(gdf)),
        "zbin_mean_nodes_per_filled_pixel": float(np.nanmean(zdf["n_nodes_ok"].to_numpy(float))) if edges is not None and len(zdf) else np.nan,
        "zbin_median_nodes_per_filled_pixel": float(np.nanmedian(zdf["n_nodes_ok"].to_numpy(float))) if edges is not None and len(zdf) else np.nan,
        "allz_mean_nodes_per_filled_pixel": float(np.nanmean(gdf["n_nodes_ok"].to_numpy(float))) if len(gdf) else np.nan,
        "allz_median_nodes_per_filled_pixel": float(np.nanmedian(gdf["n_nodes_ok"].to_numpy(float))) if len(gdf) else np.nan,
    }
    with open(sky_dir / "sky_map_summary.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print("[write]", sky_dir / "sky_map_summary_allz.csv")



def write_node_status_summaries(node_df: pd.DataFrame, args: argparse.Namespace, outdir: Path) -> None:
    """Write diagnostic tables explaining why candidate knots/nodes were kept or rejected."""
    if node_df is None or len(node_df) == 0 or "status" not in node_df.columns:
        return
    outdir.mkdir(parents=True, exist_ok=True)

    tab = node_df.copy()
    status_counts = (
        tab["status"].astype(str)
        .value_counts(dropna=False)
        .rename_axis("status")
        .reset_index(name="n_nodes")
    )
    total = float(len(tab))
    status_counts["fraction"] = status_counts["n_nodes"] / total if total > 0 else np.nan
    path = outdir / "node_status_summary.csv"
    status_counts.to_csv(path, index=False)
    print("[write]", path)

    edges = parse_zbin_edges(args, tab)
    if edges is None:
        return
    zcol = "z_mid" if "z_mid" in tab.columns and np.isfinite(pd.to_numeric(tab["z_mid"], errors="coerce")).any() else "z_low"
    z = pd.to_numeric(tab[zcol], errors="coerce").to_numpy(float)
    rows = []
    for ib in range(len(edges) - 1):
        z0 = float(edges[ib])
        z1 = float(edges[ib + 1])
        if ib == len(edges) - 2:
            mask = np.isfinite(z) & (z >= z0) & (z <= z1)
        else:
            mask = np.isfinite(z) & (z >= z0) & (z < z1)
        sub = tab.loc[mask].copy()
        ntot = float(len(sub))
        if ntot == 0:
            continue
        vc = sub["status"].astype(str).value_counts(dropna=False)
        for status, n in vc.items():
            rows.append({
                "zbin_id": ib,
                "z_min": z0,
                "z_max": z1,
                "z_center": 0.5 * (z0 + z1),
                "status": status,
                "n_nodes": int(n),
                "fraction_in_zbin": float(n / ntot),
                "n_nodes_total_zbin": int(ntot),
            })
    zpath = outdir / "node_status_summary_by_zbin.csv"
    pd.DataFrame(rows).to_csv(zpath, index=False)
    print("[write]", zpath)


def make_arm_level_table(arm_df: pd.DataFrame, node_df: pd.DataFrame) -> pd.DataFrame:
    """Convert per-arm rows into a node-like table for the same S_phi summaries.

    Each arm has a privileged origin (the node), so its mean_t_phi has a well-defined
    signed handedness.  This table lets us measure handedness, |S_phi|,
    S_phi^2 and C_radial using individual arms instead of whole node systems.
    """
    if arm_df is None or len(arm_df) == 0:
        return pd.DataFrame()
    from spirality.selection import attach_parent_metadata
    a = attach_parent_metadata(arm_df, node_df, accepted_only=True)
    # Attach node-level coordinates, redshift and Galactic hemisphere when absent.
    merge_cols = [c for c in [
        "node_id", "ra", "dec", "z_mid", "z_high", "gal_l_deg", "gal_b_deg", "gal_hemi", "mean_segment_density"
    ] if c in node_df.columns]
    if "node_id" in merge_cols:
        nmeta = node_df[merge_cols].drop_duplicates("node_id")
        a = a.merge(nmeta, on="node_id", how="left", suffixes=("", "_node"))
    if "ra" not in a.columns and "ra_node" in a.columns:
        a["ra"] = a["ra_node"]
    if "dec" not in a.columns and "dec_node" in a.columns:
        a["dec"] = a["dec_node"]
    # Robust redshift coordinate for arm-level summaries.  Some tables may carry
    # an all-NaN z_mid column after merges; in that case fall back to z_low.
    if "z_mid" not in a.columns:
        a["z_mid"] = a["z_low"] if "z_low" in a.columns else np.nan
    elif "z_low" in a.columns:
        zmid_tmp = pd.to_numeric(a["z_mid"], errors="coerce").to_numpy(float)
        zlow_tmp = pd.to_numeric(a["z_low"], errors="coerce").to_numpy(float)
        a["z_mid"] = np.where(np.isfinite(zmid_tmp), zmid_tmp, zlow_tmp)
    if "gal_l_deg" not in a.columns or "gal_b_deg" not in a.columns:
        a = add_galactic_columns(a)
    # Important: create the output with a real index before scalar assignments.
    # If we assign out["status"] = "ok" to an empty DataFrame first, pandas keeps
    # an empty column and later-added rows get NaN status. That was the reason
    # the arm-level z-bin summaries had n_nodes_ok = 0.
    out = pd.DataFrame(index=np.arange(len(a)))
    out["status"] = a["parent_status"].to_numpy()
    for col in ["parent_status", "parent_n_arms_valid", "fans_n_arms", "analysis_schema"]:
        out[col] = a[col].to_numpy()
    out["node_id"] = a["node_id"].to_numpy(int) if "node_id" in a.columns else np.arange(len(a), dtype=int)
    out["arm_local_id"] = a["arm_local_id"].to_numpy(int) if "arm_local_id" in a.columns else np.zeros(len(a), dtype=int)
    out["arm_global_id"] = np.arange(len(a), dtype=int)
    for col in ["ra", "dec", "z_low", "z_mid", "z_high", "gal_l_deg", "gal_b_deg", "gal_hemi"]:
        if col in a.columns:
            out[col] = a[col].to_numpy() if col != "gal_hemi" else a[col].astype(str).to_numpy()
    if "z_low" not in out.columns and "z_mid" in out.columns:
        out["z_low"] = out["z_mid"].to_numpy(float)
    out["S_tangential"] = a["mean_t_phi"].to_numpy(float)
    out["S_tangential_err"] = a["mean_t_phi_err"].to_numpy(float) if "mean_t_phi_err" in a.columns else np.nan
    out["S_tangential_abs"] = np.abs(out["S_tangential"].to_numpy(float))
    out["S_tangential_sq"] = out["S_tangential"].to_numpy(float) ** 2
    out["mean_t_radial"] = a["mean_t_radial"].to_numpy(float) if "mean_t_radial" in a.columns else np.nan
    out["n_arm_segments_total"] = a["n_arm_segments_valid"].to_numpy(float) if "n_arm_segments_valid" in a.columns else 1.0
    out["n_points_used_total"] = a["n_points_total"].to_numpy(float) if "n_points_total" in a.columns else out["n_arm_segments_total"].to_numpy(float)
    if "mean_arm_density" in a.columns:
        den = a["mean_arm_density"].to_numpy(float)
    elif "mean_segment_density" in a.columns:
        den = a["mean_segment_density"].to_numpy(float)
    else:
        den = np.ones(len(a), dtype=float)
    out["mean_segment_density"] = np.where(np.isfinite(den) & (den > 0), den, 1.0)
    out["n_arms_valid"] = a["parent_n_arms_valid"].to_numpy()  # Actual FANS multiplicity
    out["n_arms_positive_tphi"] = (out["S_tangential"].to_numpy(float) > 0).astype(int)
    out["n_arms_negative_tphi"] = (out["S_tangential"].to_numpy(float) < 0).astype(int)
    # Columns expected by the generic node-level summarizer but not meaningful for arms.
    if "beta_common_rad" not in out.columns:
        out["beta_common_rad"] = np.nan
    if "gal_hemi" not in out.columns and "gal_b_deg" in out.columns:
        out["gal_hemi"] = np.where(out["gal_b_deg"].to_numpy(float) >= 0.0, "north", "south")
    if "z_mid" not in out.columns and "z_low" in out.columns:
        out["z_mid"] = out["z_low"].to_numpy(float)
    elif "z_mid" in out.columns and "z_low" in out.columns:
        zmid_tmp = pd.to_numeric(out["z_mid"], errors="coerce").to_numpy(float)
        zlow_tmp = pd.to_numeric(out["z_low"], errors="coerce").to_numpy(float)
        out["z_mid"] = np.where(np.isfinite(zmid_tmp), zmid_tmp, zlow_tmp)
    if not np.isfinite(out["S_tangential"].to_numpy(float)).all():
        raise ValueError("Nonfinite winding in an accepted parent; do not silently alter its multiplicity")
    out = out.reset_index(drop=True)
    return out


def _cfg_with_node_weight(cfg: Config, mode: str) -> Config:
    d = asdict(cfg)
    d["node_weight"] = str(mode)
    return Config(**d)



def _handedness_bin_stats(tab: pd.DataFrame, cfg: Config, edges: np.ndarray, *, sample: str, B: int, rng: np.random.Generator) -> Tuple[pd.DataFrame, Dict[int, np.ndarray]]:
    """Return per-bin H_S and bootstrap arrays for a sample all/north/south."""
    work = tab.copy()
    if "status" in work.columns:
        work = work[work["status"].eq("ok")].copy()
    if len(work) == 0:
        return pd.DataFrame(), {}
    if "gal_b_deg" not in work.columns:
        work = add_galactic_columns(work)
    if sample == "north":
        work = work[pd.to_numeric(work["gal_b_deg"], errors="coerce") >= 0.0].copy()
    elif sample == "south":
        work = work[pd.to_numeric(work["gal_b_deg"], errors="coerce") < 0.0].copy()
    if len(work) == 0:
        return pd.DataFrame(), {}
    zcol = "z_mid" if "z_mid" in work.columns and np.isfinite(pd.to_numeric(work["z_mid"], errors="coerce")).any() else "z_low"
    z = pd.to_numeric(work[zcol], errors="coerce").to_numpy(float)
    rows=[]
    boots={}
    for ib in range(len(edges)-1):
        z0=float(edges[ib]); z1=float(edges[ib+1])
        mask = np.isfinite(z) & (z>=z0) & ((z<=z1) if ib==len(edges)-2 else (z<z1))
        sub = work.loc[mask].copy()
        if len(sub)==0 or "S_tangential" not in sub.columns:
            rows.append({"zbin_id":ib,"z_min":z0,"z_max":z1,"z_center":0.5*(z0+z1),"n_ok":0,"H_S":np.nan,"H_S_err":np.nan})
            continue
        vals = pd.to_numeric(sub["S_tangential"], errors="coerce").to_numpy(float)
        w = node_weights(sub, cfg.node_weight)
        obs, err, zscore, boot = bootstrap_global(vals, w, B=B, rng=rng)
        boots[ib]=boot
        rows.append({"zbin_id":ib,"z_min":z0,"z_max":z1,"z_center":0.5*(z0+z1),"n_ok":int(np.count_nonzero(np.isfinite(vals)&np.isfinite(w)&(w>0))),"H_S":obs,"H_S_err":err,"H_S_z":zscore})
    return pd.DataFrame(rows), boots


def _one_sided_gaussian_pvalues(z: float) -> Tuple[float, float, float]:
    """Return one-sided p-values for negative, positive and observed-direction alternatives.

    For a Gaussian test statistic Z:
      p_negative = P(Z_null <= Z_obs)
      p_positive = P(Z_null >= Z_obs)
      p_observed_direction = min(p_negative, p_positive)
    """
    if not np.isfinite(z):
        return np.nan, np.nan, np.nan
    p_negative = float(norm_dist.cdf(z))
    p_positive = float(norm_dist.sf(z))
    p_observed = float(min(p_negative, p_positive))
    return p_negative, p_positive, p_observed


def _weighted_linear_slope(x: np.ndarray, y: np.ndarray, yerr: np.ndarray) -> Tuple[float,float,float,float]:
    ok=np.isfinite(x)&np.isfinite(y)&np.isfinite(yerr)&(yerr>0)
    if np.count_nonzero(ok)<2:
        return np.nan,np.nan,np.nan,np.nan
    xx=x[ok]; yy=y[ok]; ww=1.0/yerr[ok]**2
    X=np.column_stack([np.ones(len(xx)), xx-np.average(xx, weights=ww)])
    sw=np.sqrt(ww)
    try:
        beta, *_=np.linalg.lstsq(X*sw[:,None], yy*sw, rcond=None)
        cov=np.linalg.pinv(X.T@(ww[:,None]*X))
        slope=float(beta[1]); err=float(np.sqrt(cov[1,1])) if cov.shape[0]>1 and cov[1,1]>=0 else np.nan
        z=float(slope/err) if np.isfinite(err) and err>0 else np.nan
        p=float(2.0*norm_dist.sf(abs(z))) if np.isfinite(z) else np.nan
        return slope,err,z,p
    except Exception:
        return np.nan,np.nan,np.nan,np.nan


def write_redshift_handedness_trend_summary(tab: pd.DataFrame, cfg: Config, args: argparse.Namespace, outdir: Path, *, label: str) -> None:
    """Write first-vs-last and linear-trend significance of H_S(z)."""
    edges=parse_zbin_edges(args, tab)
    if edges is None or len(edges)<3:
        return
    outdir.mkdir(parents=True, exist_ok=True)
    rows=[]
    B=int(getattr(args, "bootstrap_B", cfg.bootstrap_B) or cfg.bootstrap_B)
    for sample in ["all","north","south"]:
        rng=np.random.default_rng(cfg.seed + 81237 + (0 if label=="nodes" else 100000) + {"all":1,"north":2,"south":3}[sample])
        zdf, boots=_handedness_bin_stats(tab, cfg, edges, sample=sample, B=B, rng=rng)
        if len(zdf)==0:
            continue
        first=int(zdf["zbin_id"].min()); last=int(zdf["zbin_id"].max())
        h0=float(zdf.loc[zdf["zbin_id"].eq(first), "H_S"].iloc[0]) if np.any(zdf["zbin_id"].eq(first)) else np.nan
        h1=float(zdf.loc[zdf["zbin_id"].eq(last), "H_S"].iloc[0]) if np.any(zdf["zbin_id"].eq(last)) else np.nan
        e0=float(zdf.loc[zdf["zbin_id"].eq(first), "H_S_err"].iloc[0]) if np.any(zdf["zbin_id"].eq(first)) else np.nan
        e1=float(zdf.loc[zdf["zbin_id"].eq(last), "H_S_err"].iloc[0]) if np.any(zdf["zbin_id"].eq(last)) else np.nan
        delta=h1-h0
        err_delta=float(np.sqrt(e0*e0+e1*e1)) if np.isfinite(e0) and np.isfinite(e1) else np.nan
        z_delta=float(delta/err_delta) if np.isfinite(err_delta) and err_delta>0 else np.nan
        p_delta=float(2.0*norm_dist.sf(abs(z_delta))) if np.isfinite(z_delta) else np.nan
        p_delta_neg, p_delta_pos, p_delta_obs = _one_sided_gaussian_pvalues(z_delta)
        b0=boots.get(first, np.array([],dtype=float)); b1=boots.get(last, np.array([],dtype=float))
        if len(b0)>0 and len(b1)>0:
            n=min(len(b0),len(b1)); bd=b1[:n]-b0[:n]
            q16=float(np.nanpercentile(bd,16)); q84=float(np.nanpercentile(bd,84))
        else:
            q16=q84=np.nan
        slope, slope_err, slope_z, slope_p = _weighted_linear_slope(zdf["z_center"].to_numpy(float), zdf["H_S"].to_numpy(float), zdf["H_S_err"].to_numpy(float))
        p_slope_neg, p_slope_pos, p_slope_obs = _one_sided_gaussian_pvalues(slope_z)
        rows.append({
            "level":label,"sample":sample,"first_zbin":first,"last_zbin":last,
            "H_S_lowz":h0,"H_S_highz":h1,"delta_high_minus_low":delta,
            "delta_err_gaussian":err_delta,
            "delta_z_gaussian":z_delta,
            "delta_p_two_sided_gaussian":p_delta,
            "delta_p_one_sided_negative_gaussian":p_delta_neg,
            "delta_p_one_sided_positive_gaussian":p_delta_pos,
            "delta_p_one_sided_observed_direction_gaussian":p_delta_obs,
            "delta_bootstrap_q16":q16,"delta_bootstrap_q84":q84,
            "linear_slope_dH_dz":slope,
            "linear_slope_err":slope_err,
            "linear_slope_z":slope_z,
            "linear_slope_p_two_sided":slope_p,
            "linear_slope_p_one_sided_negative":p_slope_neg,
            "linear_slope_p_one_sided_positive":p_slope_pos,
            "linear_slope_p_one_sided_observed_direction":p_slope_obs,
            "n_lowz": int(zdf.loc[zdf["zbin_id"].eq(first), "n_ok"].iloc[0]) if np.any(zdf["zbin_id"].eq(first)) else 0,
            "n_highz": int(zdf.loc[zdf["zbin_id"].eq(last), "n_ok"].iloc[0]) if np.any(zdf["zbin_id"].eq(last)) else 0,
        })
        zdf.to_csv(outdir / f"redshift_handedness_trend_{label}_{sample}_zbins.csv", index=False)
    path=outdir / f"redshift_handedness_trend_{label}.csv"
    pd.DataFrame(rows).to_csv(path,index=False)
    print("[write]", path)

def write_arm_level_products(arm_df: pd.DataFrame, node_df: pd.DataFrame, cfg: Config, args: argparse.Namespace, outdir: Path) -> None:
    """Write arm-level analogues of the main four-panel summaries."""
    arm_tab = make_arm_level_table(arm_df, node_df)
    out_path = outdir / "arm_level_spirality.csv"
    arm_tab.to_csv(out_path, index=False)
    print("[write]", out_path)
    print(f"[arms] arm-level rows: {len(arm_tab):,}")
    if len(arm_tab) > 0 and "status" in arm_tab.columns:
        print("[arms] status counts:", arm_tab["status"].astype(str).value_counts(dropna=False).to_dict())
    if len(arm_tab) == 0:
        return
    edges = parse_zbin_edges(args, arm_tab)
    if edges is None:
        return
    fmt = args.plot_format.lower()
    plot_dir = outdir / "zbin_plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    for mode in ["uniform", "mean_density"]:
        cfgm = _cfg_with_node_weight(cfg, mode)
        zdf = build_zbin_summary_by_gal_hemi(arm_tab, cfgm, edges)
        csv = outdir / f"zbin_summary_arms_{mode}.csv"
        jsn = outdir / f"zbin_summary_arms_{mode}.json"
        zdf.to_csv(csv, index=False)
        # Explicit source CSVs for the paper-style arm plots, in addition to the
        # per-panel CSV generated by the plotting function.
        zdf.to_csv(plot_dir / f"zbin_main_arm_S_panels_by_gal_hemi_{mode}_source.csv", index=False)
        zdf.to_csv(plot_dir / f"zbin_main_arm_S_angle_panels_by_gal_hemi_{mode}_source.csv", index=False)
        with open(jsn, "w", encoding="utf-8") as f:
            json.dump({"rows": zdf.replace({np.nan: None}).to_dict(orient="records")}, f, indent=2, ensure_ascii=False)
        n_ok_total = int(zdf.loc[zdf["sample"].eq("all"), "n_nodes_ok"].fillna(0).sum()) if "n_nodes_ok" in zdf.columns else 0
        print(f"[arms] z-bin summary mode={mode}: rows={len(zdf):,}, summed n_arms_ok(all)={n_ok_total:,}")
        dr12_arm_zdf = _read_optional_catalog_zbin_summary(
            getattr(args, "dr12_arm_zbin_summary", None),
            catalog="DR12",
            label=str(getattr(args, "dr12_label", "DR12 Chen")),
        )
        _plot_main_S_panels_catalog_compare(
            zdf,
            plot_dir / f"zbin_main_arm_S_panels_catalogs_{mode}.{fmt}",
            angle_version=False,
            zdf_dr12=dr12_arm_zdf,
            dr17_label=str(getattr(args, "dr17_label", "DR17 SCONCE")),
            dr12_label=str(getattr(args, "dr12_label", "DR12 Chen")),
        )
        _plot_main_S_panels_catalog_compare(
            zdf,
            plot_dir / f"zbin_main_arm_S_angle_panels_catalogs_{mode}.{fmt}",
            angle_version=True,
            zdf_dr12=dr12_arm_zdf,
            dr17_label=str(getattr(args, "dr17_label", "DR17 SCONCE")),
            dr12_label=str(getattr(args, "dr12_label", "DR12 Chen")),
        )
        _plot_main_S_panels_by_gal_hemi(zdf, plot_dir / f"zbin_main_arm_S_panels_by_gal_hemi_{mode}.{fmt}", angle_version=False)
        _plot_main_S_panels_by_gal_hemi(zdf, plot_dir / f"zbin_main_arm_S_angle_panels_by_gal_hemi_{mode}.{fmt}", angle_version=True)
        print("[write]", csv)
        print("[write]", plot_dir / f"zbin_main_arm_S_panels_by_gal_hemi_{mode}_source.csv")
    write_redshift_handedness_trend_summary(arm_tab, cfg, args, outdir, label="arms")
    write_arm_direction_count_products(arm_tab, args, outdir)


def _direction_count_rows(tab: pd.DataFrame, *, scope: str, sample: str, zbin_id: int, z_min: float, z_max: float) -> List[Dict[str, object]]:
    if len(tab) == 0:
        return []
    s = tab["S_tangential"].to_numpy(float)
    den = tab["mean_segment_density"].to_numpy(float) if "mean_segment_density" in tab.columns else np.ones(len(tab), dtype=float)
    den = np.where(np.isfinite(den) & (den > 0), den, 1.0)
    pos = np.isfinite(s) & (s > 0)
    neg = np.isfinite(s) & (s < 0)
    zero = np.isfinite(s) & (s == 0)
    n = int(np.count_nonzero(np.isfinite(s)))
    wp = float(np.sum(den[pos])); wn = float(np.sum(den[neg])); wz = float(np.sum(den[zero]))
    wt = wp + wn + wz
    return [{
        "scope": scope, "sample": sample, "zbin_id": int(zbin_id), "z_min": float(z_min), "z_max": float(z_max),
        "n_arms_total": n,
        "n_arms_positive": int(np.count_nonzero(pos)),
        "n_arms_negative": int(np.count_nonzero(neg)),
        "n_arms_zero": int(np.count_nonzero(zero)),
        "fraction_positive": float(np.count_nonzero(pos) / max(n, 1)),
        "fraction_negative": float(np.count_nonzero(neg) / max(n, 1)),
        "density_weight_positive": wp,
        "density_weight_negative": wn,
        "density_weight_zero": wz,
        "density_weight_total": wt,
        "density_weight_fraction_positive": float(wp / wt) if wt > 0 else np.nan,
        "density_weight_fraction_negative": float(wn / wt) if wt > 0 else np.nan,
        "unweighted_sign_asymmetry": float((np.count_nonzero(pos) - np.count_nonzero(neg)) / max(n, 1)),
        "density_weighted_sign_asymmetry": float((wp - wn) / wt) if wt > 0 else np.nan,
    }]


def write_arm_direction_count_products(arm_tab: pd.DataFrame, args: argparse.Namespace, outdir: Path) -> None:
    if len(arm_tab) == 0:
        return
    tab = arm_tab.copy()
    if "gal_b_deg" not in tab.columns:
        tab = add_galactic_columns(tab)
    zvals = tab["z_mid"].to_numpy(float) if "z_mid" in tab.columns else tab["z_low"].to_numpy(float)
    edges = parse_zbin_edges(args, tab)
    rows: List[Dict[str, object]] = []
    zmin = float(np.nanmin(zvals)) if np.any(np.isfinite(zvals)) else np.nan
    zmax = float(np.nanmax(zvals)) if np.any(np.isfinite(zvals)) else np.nan
    specs = [
        ("all", np.ones(len(tab), dtype=bool)),
        ("north", tab["gal_b_deg"].to_numpy(float) >= 0),
        ("south", tab["gal_b_deg"].to_numpy(float) < 0),
    ]
    for sample, mask in specs:
        rows.extend(_direction_count_rows(tab.loc[mask], scope="allz", sample=sample, zbin_id=-1, z_min=zmin, z_max=zmax))
    if edges is not None:
        for ib in range(len(edges) - 1):
            z0 = float(edges[ib]); z1 = float(edges[ib + 1])
            zmask = np.isfinite(zvals) & (zvals >= z0) & ((zvals <= z1) if ib == len(edges) - 2 else (zvals < z1))
            for sample, smask in specs:
                rows.extend(_direction_count_rows(tab.loc[zmask & smask], scope="zbin", sample=sample, zbin_id=ib, z_min=z0, z_max=z1))
    df = pd.DataFrame(rows)
    path = outdir / "arm_direction_counts.csv"
    df.to_csv(path, index=False)
    print("[write]", path)


def read_sdss_density_catalog(path: str | Path, *, z_min: float, z_max: float, class_name: str = "GALAXY") -> pd.DataFrame:
    """Read SDSS DR17 spectroscopic catalogue for a simple galaxy-count density map."""
    path = Path(path)
    with fits.open(path, memmap=True) as hdul:
        data = hdul[1].data
        z = np.asarray(data["Z"], dtype=float)
        cls_raw = data["CLASS"]
        if cls_raw.dtype.kind in {"S", "O"}:
            cls = np.array([x.decode("utf-8", "ignore").strip() if isinstance(x, (bytes, bytearray)) else str(x).strip() for x in cls_raw])
        else:
            cls = np.asarray(cls_raw).astype(str)
        if "PLUG_RA" in data.names and "PLUG_DEC" in data.names:
            ra = np.asarray(data["PLUG_RA"], dtype=float)
            dec = np.asarray(data["PLUG_DEC"], dtype=float)
        else:
            cx = np.asarray(data["CX"], dtype=float); cy = np.asarray(data["CY"], dtype=float); cz = np.asarray(data["CZ"], dtype=float)
            ra = (np.degrees(np.arctan2(cy, cx)) + 360.0) % 360.0
            dec = np.degrees(np.arcsin(np.clip(cz / np.sqrt(cx*cx + cy*cy + cz*cz), -1, 1)))
    mask = np.isfinite(z) & (z >= float(z_min)) & (z <= float(z_max)) & np.isfinite(ra) & np.isfinite(dec) & (cls == str(class_name))
    out = pd.DataFrame({"ra": ra[mask], "dec": dec[mask], "z": z[mask]})
    gl, gb = radec_to_galactic_deg(out["ra"].to_numpy(float), out["dec"].to_numpy(float))
    out["gal_l_deg"] = gl
    out["gal_b_deg"] = gb
    return out


def build_galaxy_count_pixels(gal_df: pd.DataFrame, edges: Optional[np.ndarray], *, nside: int) -> pd.DataFrame:
    hp = _require_healpy()
    if len(gal_df) == 0:
        return pd.DataFrame()
    ipix = healpix_node_ipix(gal_df["gal_l_deg"].to_numpy(float), gal_df["gal_b_deg"].to_numpy(float), nside)
    z = gal_df["z"].to_numpy(float)
    rows = []
    if edges is None:
        bins = [(-1, float(np.nanmin(z)), float(np.nanmax(z)), np.isfinite(z))]
    else:
        bins = []
        for ib in range(len(edges) - 1):
            z0 = float(edges[ib]); z1 = float(edges[ib+1])
            bins.append((ib, z0, z1, np.isfinite(z) & (z >= z0) & ((z <= z1) if ib == len(edges)-2 else (z < z1))))
    for ib, z0, z1, mask in bins:
        pix, cnt = np.unique(ipix[mask], return_counts=True)
        for p, c in zip(pix, cnt):
            rows.append({"zbin_id": int(ib), "z_min": float(z0), "z_max": float(z1), "hp_ipix": int(p), "n_galaxy": int(c), "nside": int(nside)})
    return pd.DataFrame(rows)


def build_density_trend(metric_df: pd.DataFrame, galpix_df: pd.DataFrame, *, value_cols: Sequence[str], nbins_delta: int, min_galaxies: int, label: str) -> pd.DataFrame:
    if len(metric_df) == 0 or len(galpix_df) == 0:
        return pd.DataFrame()
    m = metric_df.merge(galpix_df[["zbin_id", "hp_ipix", "n_galaxy"]], on=["zbin_id", "hp_ipix"], how="inner")
    m = m[m["n_galaxy"].ge(int(min_galaxies))].copy()
    if len(m) == 0:
        return pd.DataFrame()
    rows = []
    for ib, sub in m.groupby("zbin_id"):
        mean_n = float(np.nanmean(sub["n_galaxy"].to_numpy(float)))
        if not np.isfinite(mean_n) or mean_n <= 0:
            continue
        sub = sub.copy()
        sub["delta_density"] = sub["n_galaxy"].to_numpy(float) / mean_n - 1.0
        d = sub["delta_density"].to_numpy(float)
        finite = np.isfinite(d)
        if np.count_nonzero(finite) < 3:
            continue
        # Quantile bins are more stable for an irregular survey footprint.
        q = np.linspace(0, 1, int(nbins_delta) + 1)
        edges = np.unique(np.nanquantile(d[finite], q))
        if len(edges) < 3:
            edges = np.linspace(float(np.nanmin(d[finite])), float(np.nanmax(d[finite])), min(int(nbins_delta), np.count_nonzero(finite)) + 1)
        for j in range(len(edges) - 1):
            lo = float(edges[j]); hi = float(edges[j+1])
            mm = np.isfinite(d) & (d >= lo) & ((d <= hi) if j == len(edges)-2 else (d < hi))
            g = sub.loc[mm]
            if len(g) == 0:
                continue
            w = g["n_nodes_ok"].to_numpy(float) if "n_nodes_ok" in g.columns else np.ones(len(g), dtype=float)
            row = {
                "catalogue": label,
                "zbin_id": int(ib),
                "z_min": float(g["z_min"].iloc[0]) if "z_min" in g.columns else np.nan,
                "z_max": float(g["z_max"].iloc[0]) if "z_max" in g.columns else np.nan,
                "delta_bin": int(j),
                "delta_min": lo,
                "delta_max": hi,
                "delta_mean": weighted_global_mean(g["delta_density"].to_numpy(float), w),
                "n_pixels": int(len(g)),
                "n_objects_total": int(np.nansum(w)),
                "n_galaxy_mean": float(np.nanmean(g["n_galaxy"].to_numpy(float))),
            }
            rng = np.random.default_rng(12345 + int(ib) * 1000 + int(j))
            for col in value_cols:
                if col in g.columns:
                    vals = g[col].to_numpy(float)
                    row[col] = weighted_global_mean(vals, w)
                    okv = np.isfinite(vals) & np.isfinite(w) & (w > 0)
                    vv = vals[okv]
                    ww = w[okv]
                    if len(vv) >= 2:
                        Berr = 400
                        boots = np.empty(Berr, dtype=float)
                        for bb in range(Berr):
                            ii = rng.integers(0, len(vv), size=len(vv))
                            boots[bb] = weighted_global_mean(vv[ii], ww[ii])
                        row[f"{col}_q16"] = float(np.nanpercentile(boots, 16.0))
                        row[f"{col}_q84"] = float(np.nanpercentile(boots, 84.0))
                        row[f"{col}_err"] = float(np.nanstd(boots, ddof=1))
                    else:
                        row[f"{col}_q16"] = np.nan
                        row[f"{col}_q84"] = np.nan
                        row[f"{col}_err"] = np.nan
            rows.append(row)
    return pd.DataFrame(rows)




def _density_trend_yerr(sub: pd.DataFrame, metric: str):
    if f"{metric}_q16" in sub.columns and f"{metric}_q84" in sub.columns:
        y = sub[metric].to_numpy(float)
        q16 = sub[f"{metric}_q16"].to_numpy(float)
        q84 = sub[f"{metric}_q84"].to_numpy(float)
        lo = y - q16
        hi = q84 - y
        out = np.vstack([lo, hi])
        if np.any(np.isfinite(out)):
            return out
    if f"{metric}_err" in sub.columns:
        e = sub[f"{metric}_err"].to_numpy(float)
        if np.any(np.isfinite(e)):
            return e
    return None

def plot_density_trends(trend_df: pd.DataFrame, out: Path, *, title: str) -> None:
    if len(trend_df) == 0:
        return
    setup_matplotlib()
    metrics = [
        ("H_S_weighted_mean", "Handedness"),
        ("mean_abs_projected_twist", "Spirality/Torsion"),
        ("mean_transverse_power", "Transverse power"),
        ("mean_radial_coherence", "Radial coherence"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(15.5, 3.4), sharex=False)
    colors = plt.cm.viridis(np.linspace(0.12, 0.88, max(1, trend_df["zbin_id"].nunique())))
    zids = sorted(trend_df["zbin_id"].unique())
    for ax, (col, lab) in zip(axes, metrics):
        for k, zid in enumerate(zids):
            sub = trend_df[trend_df["zbin_id"].eq(zid)].sort_values("delta_mean")
            if len(sub) == 0 or col not in sub.columns:
                continue
            if int(zid) == -1:
                color = "k"; ls = "--"; label = "all z"; lw = 2.0
            else:
                color = colors[k % len(colors)]; ls = "-"; lw = 1.35
                label = f"{float(sub['z_min'].iloc[0]):.2f}<z<{float(sub['z_max'].iloc[0]):.2f}"
            ax.errorbar(sub["delta_mean"], sub[col], yerr=_density_trend_yerr(sub, col), marker="o", ms=4.5, lw=lw, ls=ls, color=color, ecolor=color, elinewidth=0.8, capsize=2.0, label=label)
        ax.axvline(0, color="0.6", lw=0.8, ls=":")
        if col == "H_S_weighted_mean":
            ax.axhline(0, color="0.4", lw=0.8)
        ax.set_xlabel(r"galaxy overdensity $\delta_g$")
        ax.set_title(lab)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("metric value")
    axes[0].legend(fontsize=7, frameon=False, loc="best")
    fig.suptitle(title, y=1.02, fontsize=13)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=240, bbox_inches="tight")
    plt.close(fig)




def plot_density_trends_node_vs_arm(trend_nodes: pd.DataFrame, trend_arms: pd.DataFrame, out: Path, *, arm_label: str = "Individual arms") -> None:
    """Two-row density-dependence panel: complete node systems versus individual arms."""
    if len(trend_nodes) == 0 and len(trend_arms) == 0:
        return
    setup_matplotlib()
    metrics = [
        ("H_S_weighted_mean", "Handedness"),
        ("mean_abs_projected_twist", "Spirality/Torsion"),
        ("mean_transverse_power", "Transverse power"),
        ("mean_radial_coherence", "Radial coherence"),
    ]
    fig, axes = plt.subplots(2, 4, figsize=(15.8, 6.8), sharex="col")
    datasets = [(trend_nodes, "Complete node systems"), (trend_arms, arm_label)]
    for row, (tdf, row_title) in enumerate(datasets):
        zids = sorted(tdf["zbin_id"].unique()) if len(tdf) else []
        colors = plt.cm.viridis(np.linspace(0.12, 0.88, max(1, len([z for z in zids if int(z) != -1]))))
        for col_i, (metric, lab) in enumerate(metrics):
            ax = axes[row, col_i]
            if len(tdf) > 0 and metric in tdf.columns:
                cidx = 0
                for zid in zids:
                    sub = tdf[tdf["zbin_id"].eq(zid)].sort_values("delta_mean")
                    if len(sub) == 0:
                        continue
                    if int(zid) == -1:
                        color = "black"; ls = "--"; lw = 2.0; label = "all z"
                    else:
                        color = colors[cidx % len(colors)]; cidx += 1
                        ls = "-"; lw = 1.35
                        label = f"{float(sub['z_min'].iloc[0]):.2f}<z<{float(sub['z_max'].iloc[0]):.2f}"
                    ax.errorbar(sub["delta_mean"], sub[metric], yerr=_density_trend_yerr(sub, metric), marker="o", ms=4.0, lw=lw, ls=ls, color=color, ecolor=color, elinewidth=0.8, capsize=2.0, label=label)
            ax.axvline(0, color="0.6", lw=0.8, ls=":")
            if metric == "H_S_weighted_mean":
                ax.axhline(0, color="0.4", lw=0.8)
            if row == 1:
                ax.set_xlabel(r"galaxy overdensity $\delta_g$")
            if row == 0:
                ax.set_title(lab)
            if col_i == 0:
                ax.set_ylabel(row_title + "\nmetric value")
            ax.grid(alpha=0.25)
            if row == 0 and col_i == 0:
                ax.legend(fontsize=7, frameon=False, loc="best")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=250, bbox_inches="tight")
    plt.close(fig)


def write_density_correlation_products(node_df: pd.DataFrame, arm_df: pd.DataFrame, cfg: Config, args: argparse.Namespace, outdir: Path) -> None:
    if not getattr(args, "density_fits", None):
        return
    _require_healpy()
    edges = parse_zbin_edges(args, node_df)
    nside_zbin, nside_global = _resolve_sky_nsides(args)
    dens_dir = outdir / "density_correlations"
    dens_dir.mkdir(parents=True, exist_ok=True)
    zmin = float(getattr(args, "z_min", 0.0) if getattr(args, "z_min", None) is not None else np.nanmin(node_df["z_low"].to_numpy(float)))
    zmax = float(getattr(args, "z_max", 1.0) if getattr(args, "z_max", None) is not None else np.nanmax(node_df["z_low"].to_numpy(float)))
    print("[density] reading", args.density_fits)
    gal = read_sdss_density_catalog(args.density_fits, z_min=zmin, z_max=zmax, class_name=args.density_class)
    with open(dens_dir / "density_catalog_used_summary.json", "w", encoding="utf-8") as f:
        json.dump({
            "density_fits": str(args.density_fits),
            "class_used": str(args.density_class),
            "z_min": float(zmin),
            "z_max": float(zmax),
            "n_galaxies_used": int(len(gal)),
        }, f, indent=2, ensure_ascii=False)
    value_cols = ["H_S_weighted_mean", "mean_abs_projected_twist", "mean_transverse_power", "mean_radial_coherence"]

    # Node/complete-spiral metric pixels.
    node_zdf, _ = build_healpix_map_summary(node_df, cfg, nside=nside_zbin, min_count=int(args.sky_min_count), z_edges=edges, label="zbin") if edges is not None else (pd.DataFrame(), {})
    node_gdf, _ = build_healpix_map_summary(node_df, cfg, nside=nside_global, min_count=int(args.sky_min_count), z_edges=None, label="allz")
    gal_z = build_galaxy_count_pixels(gal, edges, nside=nside_zbin) if edges is not None else pd.DataFrame()
    gal_g = build_galaxy_count_pixels(gal, None, nside=nside_global)
    trend_nodes = pd.concat([
        build_density_trend(node_zdf, gal_z, value_cols=value_cols, nbins_delta=int(args.density_delta_bins), min_galaxies=int(args.density_min_galaxies_per_pixel), label="nodes_zbin"),
        build_density_trend(node_gdf, gal_g, value_cols=value_cols, nbins_delta=int(args.density_delta_bins), min_galaxies=int(args.density_min_galaxies_per_pixel), label="nodes_allz"),
    ], ignore_index=True)
    trend_nodes.to_csv(dens_dir / "density_trend_nodes.csv", index=False)
    plot_density_trends(trend_nodes, dens_dir / f"density_trend_nodes.{args.plot_format}", title="Complete node systems")

    # Arm-level metric pixels, unweighted and density-weighted.
    arm_tab = make_arm_level_table(arm_df, node_df)
    for mode in ["uniform", "mean_density"]:
        cfgm = _cfg_with_node_weight(cfg, mode)
        arm_zdf, _ = build_healpix_map_summary(arm_tab, cfgm, nside=nside_zbin, min_count=int(args.sky_min_count), z_edges=edges, label="zbin") if edges is not None else (pd.DataFrame(), {})
        arm_gdf, _ = build_healpix_map_summary(arm_tab, cfgm, nside=nside_global, min_count=int(args.sky_min_count), z_edges=None, label="allz")
        trend_arms = pd.concat([
            build_density_trend(arm_zdf, gal_z, value_cols=value_cols, nbins_delta=int(args.density_delta_bins), min_galaxies=int(args.density_min_galaxies_per_pixel), label=f"arms_{mode}_zbin"),
            build_density_trend(arm_gdf, gal_g, value_cols=value_cols, nbins_delta=int(args.density_delta_bins), min_galaxies=int(args.density_min_galaxies_per_pixel), label=f"arms_{mode}_allz"),
        ], ignore_index=True)
        trend_arms.to_csv(dens_dir / f"density_trend_arms_{mode}.csv", index=False)
        plot_density_trends(trend_arms, dens_dir / f"density_trend_arms_{mode}.{args.plot_format}", title=f"Individual arms ({mode} weights)")
        plot_density_trends_node_vs_arm(
            trend_nodes, trend_arms,
            dens_dir / f"density_trend_nodes_vs_arms_{mode}.{args.plot_format}",
            arm_label=f"Individual arms ({mode} weights)",
        )
    print("[write]", dens_dir)


# -----------------------------------------------------------------------------
# Main orchestration
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--sconce-fits", required=True, help="Path to nodes_scms.fits")
    p.add_argument("--outdir", required=True, help="Output directory")

    p.add_argument("--segment-length-mpc-h", type=float, default=10.0, help="Radial bin length used to extend each arm")
    p.add_argument("--n-arm-segments", type=int, default=3, help="Number of radial bins/segments per arm")
    p.add_argument("--node-link-radius-mpc-h", type=float, default=8.0, help="Radius around node used to seed arms")
    p.add_argument("--inner-exclusion-mpc-h", type=float, default=0.0, help="Ignore points closer than this to the node")
    p.add_argument("--arm-opening-deg", type=float, default=35.0, help="Half-opening angle of each arm sector while tracing")
    p.add_argument("--cluster-gap-deg", type=float, default=35.0, help="Circular gap used to split seed points into arms")
    p.add_argument("--seed-max-tangent-angle-deg", type=float, default=75.0, help="Max tangent-radial mismatch for seed points")
    p.add_argument("--min-seed-points", type=int, default=2)
    p.add_argument("--min-points-per-arm-segment", type=int, default=3)
    p.add_argument("--min-valid-arm-segments", type=int, default=2)
    p.add_argument("--min-arms-node", type=int, default=2)
    p.add_argument("--max-arms-node", type=int, default=12)

    p.add_argument("--edge-mode", default="convexhull", choices=["convexhull", "none"], help="How to reject nodes near the footprint edge")
    p.add_argument("--edge-margin-mpc-h", type=float, default=35.0, help="Minimum distance to convex-hull boundary")
    p.add_argument("--omega-m", type=float, default=PLANCK18_OM0, help="Matter density for the legacy zmid mode; fixed Planck18 mode ignores this option")
    p.add_argument("--h", type=float, default=PLANCK18_H, help="Distance conversion for legacy zmid/fits modes; fixed Planck18 mode ignores this option")
    p.add_argument("--distance-source", default="planck18", choices=["planck18", "zmid", "fits"], help="Default: recompute shell-boundary distances with Astropy Planck18; zmid/fits are explicit legacy modes")
    p.add_argument("--fits-distance-units", default="mpc", choices=["mpc_h", "mpc"], help="Units of comov_dist_low/high if --distance-source fits")

    p.add_argument("--sconce-gradient-mode", default="tangent", choices=["tangent", "perp"],
                   help="Use projected grad_Dir as filament tangent, or rotate it by 90 deg as a diagnostic")
    p.add_argument("--no-cluster-knots", action="store_true",
                   help="Use every knot-labeled point as an independent node. Not recommended for final science.")
    p.add_argument("--knot-cluster-radius-mpc-h", type=float, default=5.0,
                   help="Radius used to merge nearby SCONCE knot-labeled points into one physical node")
    p.add_argument("--knot-cluster-weight", default="density", choices=["uniform", "density", "inv_unc2"],
                   help="Weights for cluster node center")
    p.add_argument("--min-knot-points-per-node", type=int, default=1,
                   help="Minimum number of knot-labeled points in a clustered node")

    p.add_argument("--n-intersec", type=int, default=None, help="Random subsample of clustered nodes after z selection")
    p.add_argument("--z-min", type=float, default=None)
    p.add_argument("--z-max", type=float, default=None)
    p.add_argument("--z-selection", default="mid", choices=["mid", "low", "high"],
                   help="Which redshift coordinate to use for global z_min/z_max row selection")
    p.add_argument("--zbin-nbins", type=int, default=0,
                   help="If >0, compute redshift-bin summaries with this number of equal-width bins")
    p.add_argument("--zbin-min", type=float, default=None, help="Minimum redshift for z-bin summaries; defaults to --z-min or data minimum")
    p.add_argument("--zbin-max", type=float, default=None, help="Maximum redshift for z-bin summaries; defaults to --z-max or data maximum")
    p.add_argument("--zbin-edges", default=None,
                   help="Comma-separated explicit z-bin edges, overriding --zbin-nbins/min/max, e.g. 0.05,0.2875,0.525,0.7625,1.0")
    p.add_argument("--zbin-plot-format", default="pdf", choices=["pdf", "png", "jpg"],
                   help="File format for redshift-bin evolution plots")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--n-jobs", type=int, default=1, help="Thread workers for node-level computation")

    p.add_argument("--bootstrap-B", type=int, default=1000)
    p.add_argument("--signflip-B", type=int, default=2000)
    p.add_argument("--node-weight", default="uniform", choices=["uniform", "n_segments", "sqrt_points", "inv_var", "mean_density", "sqrt_mean_density"],
                   help="Node weights for global/z-bin summaries. Keep uniform as the fiducial choice; density weights are useful systematics checks.")
    p.add_argument("--point-weight", default="density",
                   choices=["uniform", "density", "sqrt_density", "log_density", "inv_unc2", "density_inv_unc2"],
                   help="Point weights inside each radial arm segment when computing centroids and tangent averages")
    p.add_argument("--point-unc-floor", type=float, default=1e-3,
                   help="Floor added in quadrature to SCONCE unc_meas for inv_unc2 point weighting")
    p.add_argument("--point-unc-max-weight", type=float, default=1e6,
                   help="Maximum inverse-uncertainty point weight")
    p.add_argument("--entropy-bins", type=int, default=31, help="Fixed bins in S_phi in [-1,1] for normalized binned entropy")
    p.add_argument("--arm-angle-entropy-bins", type=int, default=8, help="Angular bins around a node for arm-angle entropy")

    p.add_argument("--make-plots", action="store_true", help="Save per-node spiral plots")
    p.add_argument("--plot-format", default="pdf", choices=["pdf", "png", "jpg"])
    p.add_argument("--max-spiral-plots", type=int, default=200, help="Max number of OK nodes to plot; use -1 for all")
    p.add_argument("--make-gallery", action="store_true", help="Save one large gallery with optimal and typical nodes")
    p.add_argument("--gallery-seed", type=int, default=None, help="Random seed for gallery selection; defaults to --seed")
    p.add_argument("--gallery-ncols", type=int, default=8)
    p.add_argument("--gallery-nrows", type=int, default=12)
    p.add_argument("--gallery-optimal-pool-frac", type=float, default=0.02, help="Fraction of highest-score OK nodes used as the optimal pool")
    p.add_argument("--gallery-optimal-arms", type=int, default=0, help="If >0, require exactly this number of arms for optimal-gallery examples")
    p.add_argument("--gallery-optimal-min-arms", type=int, default=3, help="If --gallery-optimal-arms<=0, require at least this many arms for optimal-gallery examples")
    p.add_argument("--gallery-allow-mixed-arm-sign", action="store_true", help="Allow optimal-gallery examples with arms winding in mixed signs; default requires all arms to have the same sign")
    p.add_argument("--gallery-filename", default="node_spiral_gallery", help="Base name for the gallery plot")
    p.add_argument("--make-random-fans-gallery", action="store_true",
                   help="Save individual random FANS PNGs using the native gallery plotter.")
    p.add_argument("--random-fans-gallery-outdir", default="random_fans_gallery")
    p.add_argument("--random-fans-gallery-catalog-key", default="sconce")
    p.add_argument("--random-fans-gallery-n-per-arm-count", type=int, default=1000)
    p.add_argument("--random-fans-gallery-arm-counts", nargs="+", type=int, default=[2, 3])
    p.add_argument("--dr12-node-zbin-summary", default=None,
                   help="Optional Chen DR12 node-level zbin_summary.csv to overlay in DR17/DR12 paper panels")
    p.add_argument("--dr12-arm-zbin-summary", default=None,
                   help="Optional Chen DR12 arm-level zbin_summary.csv to overlay in arm-level DR17/DR12 paper panels")
    p.add_argument("--dr12-label", default="DR12 Chen",
                   help="Legend label for the optional Chen DR12 comparison")
    p.add_argument("--dr17-label", default="DR17 SCONCE",
                   help="Legend label for the SCONCE/DR17 catalogue")
    p.add_argument("--make-sky-maps", action="store_true", help="Write HEALPix sky maps of H_S and <|S_phi|>")
    p.add_argument("--sky-nside-zbin", type=int, default=4, help="HEALPix NSIDE for redshift-binned maps")
    p.add_argument("--sky-nside-global", type=int, default=8, help="HEALPix NSIDE for maps integrated over all selected redshifts")
    p.add_argument("--sky-nside", type=int, default=None, help="Deprecated alias: if set, overrides both sky-nside-zbin and sky-nside-global")
    p.add_argument("--sky-min-count", type=int, default=3, help="Minimum nodes per HEALPix pixel to plot a value")
    p.add_argument("--make-arm-level-products", action="store_true", default=True, help="Write arm-level z-bin panels and direction-count statistics")
    p.add_argument("--density-fits", default=None, help="Optional SDSS DR17 spectroscopic catalogue for density-correlation plots")
    p.add_argument("--density-class", default="GALAXY", help="CLASS value to use from --density-fits")
    p.add_argument("--density-delta-bins", type=int, default=8, help="Number of quantile bins in galaxy overdensity for density-trend plots")
    p.add_argument("--density-min-galaxies-per-pixel", type=int, default=1, help="Minimum galaxies per HEALPix pixel for density-trend plots")
    p.add_argument("--timing-interval", type=int, default=500, help="Progress/timing print interval in processed nodes")
    return p.parse_args()


def main() -> None:
    t_program_start = time.perf_counter()
    args = parse_args()
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    cfg = Config(
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
        cluster_knots=(not args.no_cluster_knots),
        knot_cluster_radius_mpc_h=args.knot_cluster_radius_mpc_h,
        knot_cluster_weight=args.knot_cluster_weight,
        min_knot_points_per_node=args.min_knot_points_per_node,
        seed=args.seed,
    )

    with open(outdir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump({"args": vars(args), "config": asdict(cfg)}, f, indent=2, ensure_ascii=False)

    t_read_start = time.perf_counter()
    print("[read] SCONCE FITS:", args.sconce_fits)
    fmaps, raw_knots, read_meta = read_sconce_fits(args.sconce_fits, z_min=args.z_min, z_max=args.z_max, z_selection=args.z_selection)
    t_read_end = time.perf_counter()

    print(f"[data] selected SCONCE filament points: {len(fmaps):,}")
    print(f"[data] raw knot-labeled points: {len(raw_knots):,}")
    with open(outdir / "input_read_summary.json", "w", encoding="utf-8") as f:
        json.dump(read_meta, f, indent=2, ensure_ascii=False)

    t_slice_start = time.perf_counter()
    slices = build_sconce_slices(fmaps, cfg, verbose=True)
    t_slice_end = time.perf_counter()
    if not slices:
        raise RuntimeError("No valid redshift slices were built from the SCONCE filament map.")

    t_nodes_start = time.perf_counter()
    print("[nodes] building nodes from SCONCE knot_label == 'Knot'")
    nodes = build_nodes_from_sconce_knots(raw_knots, slices, cfg)
    t_nodes_end = time.perf_counter()
    if len(nodes) == 0:
        raise RuntimeError("No nodes found from SCONCE knots after selection/clustering.")
    nodes_unsubsampled = int(len(nodes))
    nodes.to_csv(outdir / "sconce_nodes_used_before_sampling.csv", index=False)
    print(f"[nodes] clustered/usable nodes before sampling: {len(nodes):,}")

    if args.n_intersec is not None and int(args.n_intersec) > 0 and len(nodes) > int(args.n_intersec):
        rng = np.random.default_rng(args.seed)
        take = rng.choice(len(nodes), size=int(args.n_intersec), replace=False)
        nodes = nodes.iloc[np.sort(take)].reset_index(drop=True)
        print(f"[sample] using n_intersec={len(nodes)}")

    # Keep only nodes whose slice exists.
    if "z_key" not in nodes.columns:
        nodes["z_key"] = nodes["z_low"].map(zkey)
    has_slice = nodes["z_key"].isin(slices.keys())
    if not np.all(has_slice):
        print(f"[warn] dropping {np.count_nonzero(~has_slice)} nodes with no matching SCONCE slice")
    nodes = nodes.loc[has_slice].reset_index(drop=True)
    n_nodes_to_process = int(len(nodes))

    node_rows: List[Dict[str, object]] = []
    arm_rows: List[Dict[str, object]] = []
    seg_rows: List[Dict[str, object]] = []

    def run_node(i: int) -> Tuple[Dict[str, object], List[Dict[str, object]], List[Dict[str, object]]]:
        t0 = time.perf_counter()
        row = nodes.iloc[i]
        sd = slices[float(row["z_key"])]
        nr, ar, sr = process_one_node(row, sd, cfg)
        nr["node_wall_time_s"] = float(time.perf_counter() - t0)
        # Preserve clustering metadata if available.
        for col in ["n_knot_points_clustered", "cluster_radius_mpc_h", "cluster_rms_mpc_h", "mean_knot_density", "mean_unc_meas", "z_mid", "z_high", "gal_l_deg", "gal_b_deg", "gal_hemi"]:
            if col in row.index:
                nr[col] = row[col]
        return nr, ar, sr

    print(f"[run] processing {len(nodes):,} nodes with n_jobs={args.n_jobs}")
    t_process_start = time.perf_counter()
    if int(args.n_jobs) <= 1:
        for i in range(len(nodes)):
            nr, ar, sr = run_node(i)
            node_rows.append(nr)
            arm_rows.extend(ar)
            seg_rows.extend(sr)
            if args.timing_interval and (i + 1) % int(args.timing_interval) == 0:
                elapsed = time.perf_counter() - t_process_start
                rate = (i + 1) / elapsed if elapsed > 0 else float("nan")
                print(f"  processed {i+1:,}/{len(nodes):,} | wall={elapsed/60:.2f} min | rate={rate:.2f} nodes/s")
    else:
        with ThreadPoolExecutor(max_workers=int(args.n_jobs)) as ex:
            futs = {ex.submit(run_node, i): i for i in range(len(nodes))}
            done = 0
            for fut in as_completed(futs):
                nr, ar, sr = fut.result()
                node_rows.append(nr)
                arm_rows.extend(ar)
                seg_rows.extend(sr)
                done += 1
                if args.timing_interval and done % int(args.timing_interval) == 0:
                    elapsed = time.perf_counter() - t_process_start
                    rate = done / elapsed if elapsed > 0 else float("nan")
                    eta = (len(nodes) - done) / rate if rate > 0 else float("nan")
                    print(f"  processed {done:,}/{len(nodes):,} | wall={elapsed/60:.2f} min | rate={rate:.2f} nodes/s | ETA={eta/60:.2f} min")
    t_process_end = time.perf_counter()

    node_df = pd.DataFrame(node_rows).sort_values("node_id").reset_index(drop=True)
    arm_df = pd.DataFrame(arm_rows).sort_values(["node_id", "arm_local_id"]).reset_index(drop=True) if arm_rows else pd.DataFrame()
    seg_df = pd.DataFrame(seg_rows).sort_values(["node_id", "arm_local_id", "arm_segment_index"]).reset_index(drop=True) if seg_rows else pd.DataFrame()

    node_path = outdir / "node_spirality.csv"
    arm_path = outdir / "arm_spirality.csv"
    seg_path = outdir / "arm_segments.csv"
    from spirality.selection import write_construction_diagnostics
    write_construction_diagnostics(node_df, outdir)
    node_df.to_csv(node_path, index=False)
    arm_df.to_csv(arm_path, index=False)
    seg_df.to_csv(seg_path, index=False)
    print("[write]", node_path)
    print("[write]", arm_path)
    print("[write]", seg_path)
    write_node_status_summaries(node_df, args, outdir)

    summary = build_global_summary(node_df, cfg)
    with open(outdir / "global_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    pd.DataFrame([summary]).to_csv(outdir / "global_summary.csv", index=False)
    print("[write]", outdir / "global_summary.json")
    print(json.dumps({k: v for k, v in summary.items() if k != "config"}, indent=2, ensure_ascii=False))

    global_hemi_df = build_global_summary_by_gal_hemi(node_df, cfg)
    global_hemi_path = outdir / "global_summary_by_gal_hemi.csv"
    global_hemi_json = outdir / "global_summary_by_gal_hemi.json"
    global_hemi_df.to_csv(global_hemi_path, index=False)
    with open(global_hemi_json, "w", encoding="utf-8") as f:
        json.dump({"rows": global_hemi_df.replace({np.nan: None}).to_dict(orient="records")}, f, indent=2, ensure_ascii=False)
    print("[write]", global_hemi_path)

    write_zbin_products(node_df, cfg, args, outdir)
    write_redshift_handedness_trend_summary(node_df, cfg, args, outdir, label="nodes")
    if getattr(args, "make_arm_level_products", True):
        write_arm_level_products(arm_df, node_df, cfg, args, outdir)
    write_arm_handedness_products(node_df, args, outdir)
    if getattr(args, "make_sky_maps", False):
        print("[sky-maps] writing HEALPix maps to", outdir / "sky_maps")
        write_sky_map_products(node_df, cfg, args, outdir)
    if getattr(args, "density_fits", None):
        write_density_correlation_products(node_df, arm_df, cfg, args, outdir)

    n_ok = int(node_df.get("status", pd.Series([], dtype=str)).eq("ok").sum()) if len(node_df) else 0
    process_wall_s = float(t_process_end - t_process_start)
    read_wall_s = float(t_read_end - t_read_start)
    slice_wall_s = float(t_slice_end - t_slice_start)
    node_build_wall_s = float(t_nodes_end - t_nodes_start)
    rate_nodes_per_s = float(n_nodes_to_process / process_wall_s) if process_wall_s > 0 and n_nodes_to_process > 0 else float("nan")
    ok_rate_nodes_per_s = float(n_ok / process_wall_s) if process_wall_s > 0 and n_ok > 0 else float("nan")
    n_full_est = int(nodes_unsubsampled)
    full_process_est_s = float(n_full_est / rate_nodes_per_s) if rate_nodes_per_s > 0 else float("nan")
    full_total_no_plots_est_s = float(read_wall_s + slice_wall_s + node_build_wall_s + full_process_est_s) if np.isfinite(full_process_est_s) else float("nan")
    node_time = node_df["node_wall_time_s"].to_numpy(float) if "node_wall_time_s" in node_df.columns else np.array([], dtype=float)
    timing = {
        "n_selected_filament_points": int(len(fmaps)),
        "n_raw_knot_rows": int(len(raw_knots)),
        "n_clustered_nodes_before_sampling": int(nodes_unsubsampled),
        "n_nodes_processed_this_run": n_nodes_to_process,
        "n_nodes_ok_this_run": n_ok,
        "read_wall_s": read_wall_s,
        "build_slices_wall_s": slice_wall_s,
        "build_nodes_wall_s": node_build_wall_s,
        "node_processing_wall_s": process_wall_s,
        "node_processing_rate_nodes_per_s": rate_nodes_per_s,
        "ok_node_rate_ok_per_s": ok_rate_nodes_per_s,
        "mean_per_node_worker_time_s": float(np.nanmean(node_time)) if len(node_time) else float("nan"),
        "median_per_node_worker_time_s": float(np.nanmedian(node_time)) if len(node_time) else float("nan"),
        "p90_per_node_worker_time_s": float(np.nanpercentile(node_time, 90.0)) if len(node_time) else float("nan"),
        "estimated_full_process_wall_s_no_plots": full_process_est_s,
        "estimated_full_total_wall_s_no_plots": full_total_no_plots_est_s,
        "estimated_full_total_wall_min_no_plots": full_total_no_plots_est_s / 60.0 if np.isfinite(full_total_no_plots_est_s) else float("nan"),
        "n_jobs": int(args.n_jobs),
        "note": "Full-run estimate assumes node-processing time scales linearly with the number of selected clustered nodes and excludes per-node plotting.",
    }
    with open(outdir / "timing_summary.json", "w", encoding="utf-8") as f:
        json.dump(timing, f, indent=2, ensure_ascii=False)
    pd.DataFrame([timing]).to_csv(outdir / "timing_summary.csv", index=False)
    print("[timing]", json.dumps(timing, indent=2, ensure_ascii=False))

    if args.make_plots and len(seg_df) > 0:
        setup_matplotlib()
        ok_nodes = node_df[node_df["status"].eq("ok")].copy()
        if args.max_spiral_plots is not None and int(args.max_spiral_plots) >= 0:
            ok_nodes = ok_nodes.head(int(args.max_spiral_plots)).copy()
        plot_dir = outdir / "plots"
        print(f"[plots] writing {len(ok_nodes):,} node plots to {plot_dir}")
        for j, (_, nr) in enumerate(ok_nodes.iterrows()):
            sd = slices[float(nr["z_key"])]
            plot_one_node(nr, seg_df, sd, cfg, plot_dir, fmt=args.plot_format)
            if (j + 1) % 100 == 0:
                print(f"  plotted {j+1:,}/{len(ok_nodes):,}")

    if args.make_gallery and len(seg_df) > 0:
        gallery_dir = outdir / "gallery"
        print(f"[gallery] writing optimal/typical node gallery to {gallery_dir}")
        gout = make_node_gallery(node_df, seg_df, slices, cfg, args, gallery_dir)
        if gout is not None:
            print("[write]", gout)

    t_end = time.perf_counter()
    print(f"[done] total wall time = {(t_end - t_program_start)/60.0:.2f} min")


if __name__ == "__main__":
    main()
