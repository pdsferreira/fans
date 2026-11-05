#!/usr/bin/env python3
r"""
Measure projected filament-arm winding / handedness around Chen DR12 intersections.

This is an exploratory statistic for tomographic / sliced filament catalogues.
It is NOT a full 3D chirality estimator.  It measures projected winding of
filament arms around intersection nodes inside each redshift slice.

Inputs expected by default:
  - dr12_Fint.txt       : intersection catalogue with columns RA dec zlo
  - dr12_FMaps_full.txt : Chen filament map with columns
                         RA dec z_low density H UM v_ra v_dec

Main outputs:
  - node_spirality.csv  : one row per intersection/node, including cuts/status
  - arm_spirality.csv   : one row per detected arm around each node
  - arm_segments.csv    : one row per radial arm-bin/segment
  - global_summary.json : global handedness estimator and bootstrap/sign-flip errors
  - zbin_summary.csv   : optional redshift-bin evolution table
  - zbin_plots/        : optional redshift-bin evolution plots
  - plots/             : per-node projected spiral plots

Basic idea
----------
For each intersection, in its redshift slice:
  1. reject nodes too close to the footprint boundary using a convex-hull margin;
  2. identify near-node filament arms from raw map points around the node;
  3. trace each arm in N radial bins of fixed length, e.g. 2--3 bins of 10 Mpc/h;
  4. compute signed tangential component t_phi and a log-spiral slope beta;
  5. combine node-level handedness into a global mean H with bootstrap errors.

Sign convention
---------------
Local x is increasing RA*cos(dec), local y is increasing Dec.  The polar angle
phi = atan2(y, x) increases counter-clockwise on the sky tangent plane.  For each
map point, the headless filament direction is oriented outward from the node by
flipping the tangent when t dot r_hat < 0.  Then

    t_phi = e_phi dot t_out

is positive for counter-clockwise projected winding as radius increases.

Caveat
------
For Chen/Carron-like sliced catalogues this is a projected handedness / winding
statistic.  A cosmological parity claim needs mocks, mirror nulls, footprint tests,
RSD/projection tests, and preferably a 3D catalogue.
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
from scipy.spatial import ConvexHull, QhullError, cKDTree
from concurrent.futures import ThreadPoolExecutor, as_completed

DEG2RAD = np.pi / 180.0
RAD2DEG = 180.0 / np.pi
C_OVER_H0_MPC_H = 2997.92458


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
    omega_m: float = 0.3
    bootstrap_B: int = 1000
    signflip_B: int = 2000
    node_weight: str = "uniform"
    point_weight: str = "density"
    point_unc_floor: float = 1e-3
    point_unc_max_weight: float = 1e6
    chen_direction_convention: str = "physical"
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
    tree: cKDTree
    hull_equations: Optional[np.ndarray]
    uncertainty: Optional[np.ndarray] = None


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


def comoving_distance_mpc_h(z: float, omega_m: float = 0.3, ngrid: int = 2048) -> float:
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
        uncertainty = sub["UM"].to_numpy(float) if "UM" in sub.columns else np.full(len(sub), np.nan, dtype=float)
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
            uncertainty=uncertainty,
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


def point_weights_from_density(
    density: np.ndarray,
    mode: str,
    uncertainty: Optional[np.ndarray] = None,
    *,
    unc_floor: float = 1e-3,
    max_weight: float = 1e6,
) -> np.ndarray:
    """Return point weights for local segment/arm construction.

    Modes:
      uniform           : all catalogue points equal
      density           : use the catalogue density column
      sqrt_density      : sqrt(density)
      log_density       : log(1+density)
      inv_unc2          : 1/(uncertainty^2 + floor^2)
      density_inv_unc2  : density/(uncertainty^2 + floor^2)

    For Chen, uncertainty is the native UM column. It is only used inside the
    Chen pipeline and is not assumed to be cross-calibrated to SCONCE/Carrón.
    """
    den = np.asarray(density, dtype=float)
    den = np.where(np.isfinite(den) & (den > 0), den, 1.0)

    if uncertainty is None:
        unc = np.full_like(den, np.nan, dtype=float)
    else:
        unc = np.asarray(uncertainty, dtype=float)
        if unc.shape != den.shape:
            unc = np.full_like(den, np.nan, dtype=float)

    floor = float(unc_floor)
    floor = floor if np.isfinite(floor) and floor > 0 else 1e-3
    invu = np.where(np.isfinite(unc) & (unc > 0), 1.0 / (unc * unc + floor * floor), 1.0)
    maxw = float(max_weight)
    if np.isfinite(maxw) and maxw > 0:
        invu = np.minimum(invu, maxw)

    mode = str(mode).lower()
    if mode == "density":
        w = den
    elif mode == "uniform":
        w = np.ones_like(den, dtype=float)
    elif mode == "sqrt_density":
        w = np.sqrt(den)
    elif mode == "log_density":
        w = np.log1p(den)
    elif mode == "inv_unc2":
        w = invu
    elif mode == "density_inv_unc2":
        w = den * invu
    else:
        raise ValueError("point_weight must be one of: uniform, density, sqrt_density, log_density, inv_unc2, density_inv_unc2")
    return np.where(np.isfinite(w) & (w > 0), w, 1.0)


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
    unc_src = sd.uncertainty if getattr(sd, "uncertainty", None) is not None else np.full_like(sd.density, np.nan, dtype=float)
    unc_all = unc_src[all_idx]
    point_w_all = point_weights_from_density(den_all, cfg.point_weight, uncertainty=unc_all, unc_floor=cfg.point_unc_floor, max_weight=cfg.point_unc_max_weight)

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

    for arm_id, cloc in enumerate(clusters):
        seed_phi = circular_mean_rad(phi[cloc])
        segs = trace_arm_segments(sd, node_x=nx, node_y=ny, all_idx=all_idx, seed_phi=seed_phi, arm_local_id=arm_id, cfg=cfg)
        if len(segs) < int(cfg.min_valid_arm_segments):
            continue
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
        _, beta, _, beta_err, dof = fit_line_wls(np.log(rr / rref), ph, ww)
        tphi = np.array([r["t_phi"] for r in segs], dtype=float)
        trad = np.array([r["t_radial"] for r in segs], dtype=float)
        seg_den = np.array([r.get("mean_density", np.nan) for r in segs], dtype=float)
        mean_arm_density = float(np.average(seg_den, weights=np.maximum(ww, 1.0))) if np.any(np.isfinite(seg_den)) else np.nan
        mean_tphi, err_tphi, neff = effective_weighted_mean(tphi, ww)
        mean_trad, _, _ = effective_weighted_mean(trad, ww)
        arm_rows.append({
            "node_id": node_id,
            "arm_local_id": int(arm_id),
            "spiral_id": f"z{sd.z_key:.5f}_node{node_id:07d}_arm{arm_id:02d}",
            "ra_node": ra,
            "dec_node": dec,
            "z_low": z,
            "n_arm_segments_valid": int(len(segs)),
            "n_points_total": int(np.sum(ww)),
            "seed_phi_deg": float(seed_phi * RAD2DEG),
            "delta_phi_deg": float((ph[-1] - ph[0]) * RAD2DEG),
            "beta_logspiral_rad": float(beta),
            "beta_logspiral_err_rad": float(beta_err),
            "beta_logspiral_deg": float(beta * RAD2DEG) if np.isfinite(beta) else np.nan,
            "beta_logspiral_err_deg": float(beta_err * RAD2DEG) if np.isfinite(beta_err) else np.nan,
            "beta_dof": int(dof),
            "mean_t_phi": mean_tphi,
            "mean_t_phi_err": err_tphi,
            "mean_t_radial": mean_trad,
            "mean_arm_density": mean_arm_density,
            "neff_segments": neff,
        })
        seg_rows.extend(segs)

    if len(arm_rows) < int(cfg.min_arms_node):
        base.update({
            "status": "too_few_valid_arms",
            "n_arms_valid": int(len(arm_rows)),
            "n_near_points": int(all_idx.size),
            "n_seed_points": int(seed_local.size),
        })
        return base, arm_rows, seg_rows

    seg_df = pd.DataFrame(seg_rows)
    beta_node, beta_node_err, intercepts, beta_dof = fit_common_beta(seg_df)
    tphi_all = seg_df["t_phi"].to_numpy(float)
    trad_all = seg_df["t_radial"].to_numpy(float)
    wseg = seg_df["n_points"].to_numpy(float)
    S_node, S_node_err, neff = effective_weighted_mean(tphi_all, wseg)
    R_node, _, _ = effective_weighted_mean(trad_all, wseg)
    arm_beta = np.array([a["beta_logspiral_rad"] for a in arm_rows], dtype=float)
    arm_tphi = np.array([a["mean_t_phi"] for a in arm_rows], dtype=float)
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
        "mean_t_radial": float(R_node),
        "beta_common_rad": float(beta_node),
        "beta_common_err_rad": float(beta_node_err),
        "beta_common_deg": float(beta_node * RAD2DEG) if np.isfinite(beta_node) else np.nan,
        "beta_common_err_deg": float(beta_node_err * RAD2DEG) if np.isfinite(beta_node_err) else np.nan,
        "beta_common_dof": int(beta_dof),
        "beta_arm_mean_rad": float(np.nanmean(arm_beta)) if np.any(np.isfinite(arm_beta)) else np.nan,
        "S_arm_mean": float(np.nanmean(arm_tphi)) if np.any(np.isfinite(arm_tphi)) else np.nan,
        "mean_segment_density": mean_seg_density,
        "neff_segments": float(neff),
    })
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
    for col, label in [("S_tangential_abs", "mean_abs_projected_twist"), ("mean_t_radial", "mean_radial_coherence")]:
        if col in nd.columns:
            vals = nd[col].to_numpy(float)
            obs, err, q16, q84, _ = bootstrap_weighted_mean(vals, w, B=cfg.bootstrap_B, rng=rng)
            out[label] = obs
            out[f"{label}_bootstrap_err"] = err
            out[f"{label}_bootstrap_q16"] = q16
            out[f"{label}_bootstrap_q84"] = q84

    if "S_tangential" in nd.columns:
        vals = nd["S_tangential"].to_numpy(float) ** 2
        obs, err, q16, q84, _ = bootstrap_weighted_mean(vals, w, B=cfg.bootstrap_B, rng=rng)
        out["mean_transverse_power"] = obs
        out["mean_transverse_power_bootstrap_err"] = err
        out["mean_transverse_power_bootstrap_q16"] = q16
        out["mean_transverse_power_bootstrap_q84"] = q84
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
    zcol = node_df["z_low"].to_numpy(float) if "z_low" in node_df.columns else np.array([], dtype=float)
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
    z = node_df["z_low"].to_numpy(float)
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
            "z_mean_ok": float(np.nanmean(ok_sub["z_low"].to_numpy(float))) if len(ok_sub) else np.nan,
            "z_median_ok": float(np.nanmedian(ok_sub["z_low"].to_numpy(float))) if len(ok_sub) else np.nan,
        }
        # Avoid dumping a nested config into each CSV row.
        summ = {k: v for k, v in summ.items() if k != "config"}
        row.update(summ)
        rows.append(row)
    return pd.DataFrame(rows)


def _plot_zbin_one(zdf: pd.DataFrame, ycol: str, errcol: Optional[str], ylabel: str, out: Path, *, zero_line: bool = True) -> None:
    setup_matplotlib()
    x = zdf["z_center"].to_numpy(float)
    xerr = 0.5 * zdf["z_width"].to_numpy(float)
    y = zdf[ycol].to_numpy(float)
    yerr = zdf[errcol].to_numpy(float) if errcol and errcol in zdf.columns else None
    fig, ax = plt.subplots(figsize=(6.2, 4.1))
    ax.errorbar(x, y, xerr=xerr, yerr=yerr, fmt="o", capsize=3.0, lw=1.1, ms=4.5)
    if zero_line:
        ax.axhline(0.0, color="0.35", lw=0.8, ls="--")
    ax.set_xlabel(r"redshift $z$")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.22)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_zbin_nnodes(zdf: pd.DataFrame, out: Path) -> None:
    setup_matplotlib()
    x = zdf["z_center"].to_numpy(float)
    xerr = 0.5 * zdf["z_width"].to_numpy(float)
    y = zdf["n_nodes_ok"].to_numpy(float)
    fig, ax = plt.subplots(figsize=(6.2, 4.1))
    ax.errorbar(x, y, xerr=xerr, fmt="o", capsize=3.0, lw=1.1, ms=4.5)
    ax.set_xlabel(r"redshift $z$")
    ax.set_ylabel(r"valid nodes")
    ax.grid(alpha=0.22)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)


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
    for ax, (ycol, ecol, ylabel, zline) in zip(axes.ravel(), panels):
        y = zdf[ycol].to_numpy(float)
        yerr = zdf[ecol].to_numpy(float) if ecol in zdf.columns else None
        ax.errorbar(x, y, xerr=xerr, yerr=yerr, fmt="o", capsize=3.0, lw=1.05, ms=4.0)
        if zline:
            ax.axhline(0.0, color="0.35", lw=0.8, ls="--")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.22)
    axes[1, 0].set_xlabel(r"redshift $z$")
    axes[1, 1].set_xlabel(r"redshift $z$")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)


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
    if "mean_transverse_power" in zdf.columns:
        _plot_zbin_one(zdf, "mean_transverse_power", "mean_transverse_power_bootstrap_err", r"$\langle S_\phi^2\rangle$", plot_dir / f"zbin_transverse_power.{fmt}", zero_line=False)
    _plot_zbin_nnodes(zdf, plot_dir / f"zbin_n_nodes_ok.{fmt}")
    _plot_zbin_panels(zdf, plot_dir / f"zbin_evolution_panels.{fmt}")
    print("[write]", plot_dir)


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------

def setup_matplotlib() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
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
    ax.scatter(xloc, yloc, s=4, color="0.72", alpha=0.45, rasterized=True, label="Chen map points")
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
        f"node {node_id}  z={float(node_row['z_low']):.3f}\n"
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



# -----------------------------------------------------------------------------
# Native compact gallery, matched to the original SCONCE gallery style
# -----------------------------------------------------------------------------

def _assigned_raw_points_for_gallery(node_row: pd.Series, seg_df: pd.DataFrame, sd: SliceData, cfg: Config) -> Tuple[np.ndarray, np.ndarray]:
    """Return raw filament-map points assigned to the plotted arms.

    This reproduces the successful old SCONCE-gallery logic: use the in-memory
    KDTree/slice geometry, query points around the node, and keep only points
    falling in the measured radial bins and within the arm-opening angle of a
    measured segment.  This avoids plotting unrelated filaments in the panel.
    """
    try:
        ra = float(node_row['ra'])
        dec = float(node_row['dec'])
        node_x_slice, node_y_slice = project_slice_xy_mpc_h(
            np.array([ra]), np.array([dec]), ra0_deg=sd.ra0_deg, dec0_deg=sd.dec0_deg, chi_mpc_h=sd.chi_mpc_h
        )
        nx = float(node_x_slice[0]); ny = float(node_y_slice[0])
        max_query = cfg.arm_radius_mpc_h + cfg.node_link_radius_mpc_h
        all_idx = np.array(sd.tree.query_ball_point([nx, ny], r=max_query), dtype=np.int64)
        if all_idx.size == 0:
            return np.array([], dtype=float), np.array([], dtype=float)
        dx = sd.x_mpc_h[all_idx] - nx
        dy = sd.y_mpc_h[all_idx] - ny
        rho = np.hypot(dx, dy)
        phi = np.arctan2(dy, dx)
        node_id = int(node_row['node_id'])
        subseg = seg_df[seg_df['node_id'].astype(int) == node_id].copy()
        if len(subseg) == 0:
            return np.array([], dtype=float), np.array([], dtype=float)
        keep = np.zeros(len(all_idx), dtype=bool)
        opening = float(cfg.arm_opening_deg) * DEG2RAD
        for _, seg in subseg.iterrows():
            k = int(seg.get('arm_segment_index', 0))
            r_lo = max(float(cfg.inner_exclusion_mpc_h), k * float(cfg.segment_length_mpc_h))
            r_hi = (k + 1) * float(cfg.segment_length_mpc_h)
            phi0 = float(seg.get('phi_rad', np.nan))
            if not np.isfinite(phi0):
                phi0 = math.atan2(float(seg.get('y_mpc_h', np.nan)), float(seg.get('x_mpc_h', np.nan)))
            if not np.isfinite(phi0):
                continue
            dphi = np.abs(angular_diff_rad(phi, phi0))
            keep |= (rho >= r_lo) & (rho < r_hi) & (dphi <= opening)
        return dx[keep], dy[keep]
    except Exception:
        return np.array([], dtype=float), np.array([], dtype=float)


def _plot_compact_node_axis(ax, node_row: pd.Series, seg_df: pd.DataFrame, cfg: Config, *, sd: Optional[SliceData] = None, show_text: bool = True, line_color: str = '0.22', centroid_color: str = 'k', raw_color: str = '0.58', link_color: str = '0.80') -> None:
    """Compact node diagnostic used by the multi-panel native gallery."""
    node_id = int(node_row['node_id'])
    subseg = seg_df[seg_df['node_id'].astype(int) == node_id].copy()
    ax.set_aspect('equal', adjustable='box')
    max_r = float(cfg.arm_radius_mpc_h)
    pad = max(max_r, float(cfg.node_link_radius_mpc_h)) * 1.12

    link = plt.Circle((0.0, 0.0), float(cfg.node_link_radius_mpc_h), color=link_color, alpha=0.18, lw=0, zorder=0)
    ax.add_patch(link)
    for k in range(1, int(cfg.n_arm_segments) + 1):
        r = k * float(cfg.segment_length_mpc_h)
        circ = plt.Circle((0.0, 0.0), r, fill=False, color='0.76', lw=0.50, ls='--', alpha=0.70, zorder=0)
        ax.add_patch(circ)

    if sd is not None:
        xp, yp = _assigned_raw_points_for_gallery(node_row, seg_df, sd, cfg)
        if len(xp) > 0:
            ax.scatter(xp, yp, s=2.8, color=raw_color, alpha=0.30, linewidths=0, rasterized=True, zorder=1)

    ax.scatter([0.0], [0.0], marker='x', s=30, color='k', lw=0.9, zorder=6)
    if len(subseg) > 0:
        for _, arm_sub in subseg.groupby('arm_local_id'):
            arm_sub = arm_sub.sort_values('arm_segment_index')
            x = arm_sub['x_mpc_h'].to_numpy(float)
            y = arm_sub['y_mpc_h'].to_numpy(float)
            ax.plot(x, y, color=line_color, lw=2.5, alpha=0.12, zorder=2)
            ax.scatter(x, y, s=8.0, color=centroid_color, alpha=0.68, zorder=4)
    ax.axhline(0, color='0.88', lw=0.4, zorder=0)
    ax.axvline(0, color='0.88', lw=0.4, zorder=0)
    ax.set_xlim(-pad, pad)
    ax.set_ylim(-pad, pad)
    ax.tick_params(labelsize=8.0, length=2.4, pad=0.8, width=0.7)
    ax.set_xticks([-20, 0, 20])
    ax.set_yticks([-20, 0, 20])
    if show_text:
        txt = (
            rf'$\alpha$={float(node_row.get("ra", np.nan)):.1f}°, '
            rf'$\delta$={float(node_row.get("dec", np.nan)):.1f}°, '
            rf'$z$={float(node_row.get("z_low", np.nan)):.3f}'
        )
        ax.text(0.04, 0.965, txt, transform=ax.transAxes, va='top', ha='left', fontsize=6.8,
                bbox=dict(boxstyle='round,pad=0.10', fc='white', ec='none', alpha=0.62), zorder=7)


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
        sd = slices.get(float(row.get('z_key', np.nan)))
        _plot_compact_node_axis(
            ax, row, seg_df, cfg, sd=sd, show_text=True,
            line_color=line_color, centroid_color=centroid_color,
            raw_color=raw_color, link_color=link_color,
        )
    fig.suptitle(title, y=0.994, fontsize=22, fontweight='bold')
    fig.supxlabel(r'$\Delta x$ [$h^{-1}$ Mpc]', y=0.006, fontsize=20)
    fig.supylabel(r'$\Delta y$ [$h^{-1}$ Mpc]', x=0.003, fontsize=20)
    fig.subplots_adjust(left=0.052, right=0.998, bottom=0.035, top=0.970, wspace=-0.045, hspace=-0.045)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=260, bbox_inches='tight')
    plt.close(fig)



def _gallery_catalog_style(catalog_key: str) -> tuple[str, str, str, str]:
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
    return str(x).replace('/', '_').replace(' ', '_')


def _save_random_individual_fans_pngs(
    node_df: pd.DataFrame,
    seg_df: pd.DataFrame,
    slices: Dict[float, SliceData],
    cfg: Config,
    args: argparse.Namespace,
) -> None:
    """Save one PNG per randomly selected accepted FANS using the native gallery plotter."""
    if not bool(getattr(args, 'make_random_fans_gallery', False)):
        return
    if len(node_df) == 0 or len(seg_df) == 0:
        return
    outroot = Path(getattr(args, 'random_fans_gallery_outdir', 'random_fans_gallery')).expanduser().resolve()
    catalog_key = str(getattr(args, 'random_fans_gallery_catalog_key', 'chen'))
    n_per = int(getattr(args, 'random_fans_gallery_n_per_arm_count', 1000))
    arm_counts = [int(x) for x in getattr(args, 'random_fans_gallery_arm_counts', [2, 3])]
    seed = int(getattr(args, 'gallery_seed', None) if getattr(args, 'gallery_seed', None) is not None else getattr(args, 'seed', 2026))
    rng = np.random.default_rng(seed + 1701)

    ok = node_df[node_df.get('status', pd.Series('ok', index=node_df.index)).astype(str).eq('ok')].copy()
    if 'node_id' not in ok.columns or 'node_id' not in seg_df.columns:
        print(f'[gallery-random-native] {catalog_key}: missing node_id; skipped')
        return
    valid_nodes = set(pd.to_numeric(seg_df['node_id'], errors='coerce').dropna().astype(int).tolist())
    ok = ok[ok['node_id'].apply(lambda x: np.isfinite(float(x)) if str(x) != 'nan' else False)].copy()
    ok = ok[ok['node_id'].astype(float).round().astype(int).isin(valid_nodes)].copy()
    if len(ok) == 0:
        print(f'[gallery-random-native] {catalog_key}: no accepted nodes with segments; skipped')
        return
    if 'n_arms_valid' not in ok.columns:
        arm_counts_by_node = seg_df.groupby('node_id')['arm_local_id'].nunique()
        ok['n_arms_valid'] = ok['node_id'].map(arm_counts_by_node)
    line_color, centroid_color, raw_color, link_color = _gallery_catalog_style(catalog_key)
    setup_matplotlib()
    for ac in arm_counts:
        pool = ok[pd.to_numeric(ok['n_arms_valid'], errors='coerce').fillna(-1).astype(int).eq(int(ac))].copy()
        if len(pool) == 0:
            print(f'[gallery-random-native] {catalog_key}: no accepted FANS with {ac} arms')
            continue
        nsel = min(n_per, len(pool))
        chosen_idx = rng.choice(pool.index.to_numpy(), size=nsel, replace=False)
        chosen = pool.loc[chosen_idx].copy()
        subdir = outroot / f'{catalog_key}_{int(ac)}arms'
        subdir.mkdir(parents=True, exist_ok=True)
        rows = []
        for jj, (_, row) in enumerate(chosen.iterrows()):
            node_id = row.get('node_id')
            fans_id = f'{catalog_key}_FANS_{_safe_gallery_id(node_id)}'
            fig, ax = plt.subplots(1, 1, figsize=(3.2, 3.2))
            sd = slices.get(float(row.get('z_key', np.nan)))
            _plot_compact_node_axis(
                ax, row, seg_df, cfg, sd=sd, show_text=True,
                line_color=line_color, centroid_color=centroid_color,
                raw_color=raw_color, link_color=link_color,
            )
            ax.set_title(fans_id, fontsize=8.5, pad=2.0)
            out_png = subdir / f'{fans_id}.png'
            fig.savefig(out_png, dpi=220, bbox_inches='tight')
            plt.close(fig)
            rows.append({'catalog_key': catalog_key, 'fans_id': fans_id, 'node_id': node_id, 'n_arms': int(ac), 'png': str(out_png)})
            if (jj + 1) % 100 == 0:
                print(f'[gallery-random-native] {catalog_key} {ac} arms: {jj+1}/{nsel}')
        pd.DataFrame(rows).to_csv(subdir / 'selected_fans.csv', index=False)
        print(f'[gallery-random-native] wrote {nsel} PNGs to {subdir}')

def make_node_gallery(node_df: pd.DataFrame, seg_df: pd.DataFrame, slices: Dict[float, SliceData], cfg: Config, args: argparse.Namespace, outdir: Path) -> Optional[Path]:
    """Make two separate native galleries: optimal and typical."""
    if len(node_df) == 0 or len(seg_df) == 0:
        return None
    ok = node_df[node_df['status'].eq('ok')].copy()
    ok = ok[ok['node_id'].isin(seg_df['node_id'].astype(int).unique())].copy()
    if len(ok) == 0:
        return None
    ncols = int(getattr(args, 'gallery_ncols', 8))
    nrows = max(1, int(getattr(args, 'gallery_nrows', 12)))
    n_each = ncols * nrows
    rng = np.random.default_rng(args.gallery_seed if getattr(args, 'gallery_seed', None) is not None else args.seed)

    Sabs = np.abs(ok.get('S_tangential', pd.Series(np.nan, index=ok.index)).to_numpy(float))
    nseg = np.maximum(ok.get('n_arm_segments_total', pd.Series(1, index=ok.index)).to_numpy(float), 1.0)
    crad = np.clip(ok.get('mean_t_radial', pd.Series(1, index=ok.index)).to_numpy(float), 0.0, 1.0)
    score = Sabs * np.sqrt(nseg) * crad
    score = np.where(np.isfinite(score), score, -np.inf)
    ok = ok.assign(gallery_score=score)

    arms_series = pd.to_numeric(ok.get('n_arms_valid', pd.Series(0, index=ok.index)), errors='coerce').fillna(-1).astype(int)
    npos_series = pd.to_numeric(ok.get('n_arms_positive_tphi', pd.Series(0, index=ok.index)), errors='coerce').fillna(0).astype(int)
    nneg_series = pd.to_numeric(ok.get('n_arms_negative_tphi', pd.Series(0, index=ok.index)), errors='coerce').fillna(0).astype(int)
    optimal_exact_arms = int(getattr(args, 'gallery_optimal_arms', 0))
    optimal_min_arms = int(getattr(args, 'gallery_optimal_min_arms', 3))
    if optimal_exact_arms > 0:
        opt_mask = arms_series.eq(optimal_exact_arms)
    else:
        opt_mask = arms_series.ge(optimal_min_arms)
    if not bool(getattr(args, 'gallery_allow_mixed_arm_sign', False)):
        opt_mask &= (npos_series.eq(arms_series) | nneg_series.eq(arms_series))
    opt_base = ok[opt_mask].copy()
    if len(opt_base) == 0:
        opt_base = ok.copy()
    opt_sorted = opt_base.sort_values('gallery_score', ascending=False).copy()
    pool_n = max(n_each, int(math.ceil(float(getattr(args, 'gallery_optimal_pool_frac', 0.02)) * len(opt_sorted))))
    pool_n = min(pool_n, len(opt_sorted))
    opt_pool = opt_sorted.head(pool_n)
    opt_take = min(n_each, len(opt_pool))
    opt_idx = rng.choice(opt_pool.index.to_numpy(), size=opt_take, replace=False) if opt_take > 0 else np.array([], dtype=int)
    opt_nodes = ok.loc[opt_idx].sort_values('gallery_score', ascending=False).copy() if len(opt_idx) else ok.head(0).copy()

    remaining = ok.drop(index=opt_idx, errors='ignore')
    typ_take = min(n_each, len(remaining))
    typ_idx = rng.choice(remaining.index.to_numpy(), size=typ_take, replace=False) if typ_take > 0 else np.array([], dtype=int)
    typ_nodes = remaining.loc[typ_idx].copy() if len(typ_idx) else ok.head(0).copy()

    outdir.mkdir(parents=True, exist_ok=True)
    base = str(getattr(args, 'gallery_filename', 'node_spiral_gallery'))
    out_opt = outdir / f'{base}_optimal.{args.plot_format}'
    out_typ = outdir / f'{base}_typical.{args.plot_format}'
    _save_gallery_panel(opt_nodes, seg_df, slices, cfg, out_opt, 'Optimal examples', ncols=ncols, nrows=nrows)
    _save_gallery_panel(typ_nodes, seg_df, slices, cfg, out_typ, 'Typical examples', ncols=ncols, nrows=nrows)

    selection_tables = []
    sel_opt = opt_nodes.copy(); sel_opt.insert(0, 'gallery_rank', np.arange(len(sel_opt))); sel_opt.insert(1, 'gallery_class', 'optimal')
    sel_typ = typ_nodes.copy(); sel_typ.insert(0, 'gallery_rank', np.arange(len(sel_typ))); sel_typ.insert(1, 'gallery_class', 'typical')
    selection_tables.extend([sel_opt, sel_typ])

    # Additional typical galleries with a fixed number of valid arms.
    # These are useful morphology diagnostics and do not alter the science measurements.
    for narm in [2, 3, 4]:
        narm_pool = ok[pd.to_numeric(ok.get('n_arms_valid', pd.Series(-1, index=ok.index)), errors='coerce').fillna(-1).astype(int).eq(narm)].copy()
        if len(narm_pool) == 0:
            continue
        take = min(n_each, len(narm_pool))
        idx_narm = rng.choice(narm_pool.index.to_numpy(), size=take, replace=False) if take > 0 else np.array([], dtype=int)
        nodes_narm = narm_pool.loc[idx_narm].copy() if len(idx_narm) else narm_pool.head(0).copy()
        out_narm = outdir / f'{base}_typical_{narm}arms.{args.plot_format}'
        _save_gallery_panel(nodes_narm, seg_df, slices, cfg, out_narm, f'Typical examples: {narm} arms', ncols=ncols, nrows=nrows)
        sel_narm = nodes_narm.copy(); sel_narm.insert(0, 'gallery_rank', np.arange(len(sel_narm))); sel_narm.insert(1, 'gallery_class', f'typical_{narm}arms')
        selection_tables.append(sel_narm)

    pd.concat(selection_tables, ignore_index=True).to_csv((outdir / base).with_suffix('.csv'), index=False)
    _save_random_individual_fans_pngs(node_df, seg_df, slices, cfg, args)
    return out_opt

# -----------------------------------------------------------------------------
# Main orchestration
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--intersections", required=True, help="Path to dr12_Fint.txt")
    p.add_argument("--fmaps", required=True, help="Path to dr12_FMaps_full.txt")
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
    p.add_argument("--omega-m", type=float, default=0.3)

    p.add_argument("--n-intersec", type=int, default=None, help="Random subsample of intersections after z selection")
    p.add_argument("--z-min", type=float, default=None)
    p.add_argument("--z-max", type=float, default=None)
    p.add_argument("--zbin-nbins", type=int, default=0,
                   help="If >0, compute redshift-bin summaries with this number of equal-width bins")
    p.add_argument("--zbin-min", type=float, default=None, help="Minimum redshift for z-bin summaries; defaults to --z-min or data minimum")
    p.add_argument("--zbin-max", type=float, default=None, help="Maximum redshift for z-bin summaries; defaults to --z-max or data maximum")
    p.add_argument("--zbin-edges", default=None,
                   help="Comma-separated explicit z-bin edges, overriding --zbin-nbins/min/max, e.g. 0.05,0.2125,0.375,0.5375,0.7")
    p.add_argument("--zbin-plot-format", default="pdf", choices=["pdf", "png", "jpg"],
                   help="File format for redshift-bin evolution plots")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--n-jobs", type=int, default=1, help="Thread workers for node-level computation")

    p.add_argument("--bootstrap-B", type=int, default=1000)
    p.add_argument("--signflip-B", type=int, default=2000)
    p.add_argument("--node-weight", default="uniform",
                   choices=["uniform", "n_segments", "sqrt_points", "inv_var", "mean_density", "sqrt_mean_density"],
                   help="Node weights for global/z-bin summaries. mean_density uses the mean density of the traced arm segments.")
    p.add_argument("--point-weight", default="density",
                   choices=["uniform", "density", "sqrt_density", "log_density", "inv_unc2", "density_inv_unc2"],
                   help="Point weights used locally to compute each radial arm segment centroid and mean tangent.")
    p.add_argument("--point-unc-floor", type=float, default=1e-3,
                   help="Floor added in quadrature to Chen UM for inv_unc2 point weighting")
    p.add_argument("--point-unc-max-weight", type=float, default=1e6,
                   help="Maximum inverse-uncertainty point weight")
    p.add_argument("--chen-direction-convention", default="physical", choices=["physical", "raw"],
                   help="How to convert Chen v_ra/v_dec into a physical tangent direction. physical applies E=v_ra*cos(dec), N=v_dec; raw is only for diagnostics.")

    p.add_argument("--make-gallery", action="store_true", help="Save native optimal/typical node galleries")
    p.add_argument("--gallery-seed", type=int, default=None, help="Random seed for gallery selection; defaults to --seed")
    p.add_argument("--gallery-ncols", type=int, default=8)
    p.add_argument("--gallery-nrows", type=int, default=12)
    p.add_argument("--gallery-optimal-pool-frac", type=float, default=0.02)
    p.add_argument("--gallery-optimal-arms", type=int, default=0)
    p.add_argument("--gallery-optimal-min-arms", type=int, default=3)
    p.add_argument("--gallery-allow-mixed-arm-sign", action="store_true")
    p.add_argument("--gallery-filename", default="chen_node_spiral_gallery")
    p.add_argument("--make-random-fans-gallery", action="store_true",
                   help="Save individual random FANS PNGs using the native gallery plotter.")
    p.add_argument("--random-fans-gallery-outdir", default="random_fans_gallery")
    p.add_argument("--random-fans-gallery-catalog-key", default="chen")
    p.add_argument("--random-fans-gallery-n-per-arm-count", type=int, default=1000)
    p.add_argument("--random-fans-gallery-arm-counts", nargs="+", type=int, default=[2, 3])
    p.add_argument("--make-plots", action="store_true", help="Save per-node spiral plots")
    p.add_argument("--plot-format", default="pdf", choices=["pdf", "png", "jpg"])
    p.add_argument("--max-spiral-plots", type=int, default=200, help="Max number of OK nodes to plot; use -1 for all")
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
        chen_direction_convention=args.chen_direction_convention,
        seed=args.seed,
    )

    with open(outdir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump({"args": vars(args), "config": asdict(cfg)}, f, indent=2, ensure_ascii=False)

    t_read_start = time.perf_counter()
    print("[read] intersections:", args.intersections)
    nodes = read_intersections(args.intersections)
    print("[read] fmaps:", args.fmaps)
    fmaps = read_chen_fmaps(args.fmaps)
    t_read_end = time.perf_counter()
    n_nodes_before_z = int(len(nodes))

    if args.z_min is not None:
        nodes = nodes[nodes["z_low"] >= float(args.z_min)].copy()
    if args.z_max is not None:
        nodes = nodes[nodes["z_low"] <= float(args.z_max)].copy()
    nodes = nodes.reset_index(drop=True)
    n_nodes_after_z_before_sample = int(len(nodes))

    if args.n_intersec is not None and int(args.n_intersec) > 0 and len(nodes) > int(args.n_intersec):
        rng = np.random.default_rng(args.seed)
        take = rng.choice(len(nodes), size=int(args.n_intersec), replace=False)
        nodes = nodes.iloc[np.sort(take)].reset_index(drop=True)
        print(f"[sample] using n_intersec={len(nodes)}")

    print(f"[data] intersections after selection: {len(nodes):,}")
    print(f"[data] filament map points: {len(fmaps):,}")

    t_slice_start = time.perf_counter()
    slices = build_slices(fmaps, cfg, verbose=True)
    t_slice_end = time.perf_counter()
    if not slices:
        raise RuntimeError("No valid redshift slices were built from the filament map.")

    # Keep only intersections whose slice exists.
    nodes["z_key"] = nodes["z_low"].map(zkey)
    has_slice = nodes["z_key"].isin(slices.keys())
    unassessed_nodes = nodes.loc[~has_slice].copy()
    if not np.all(has_slice):
        print(f"[warn] {len(unassessed_nodes)} nodes marked unassessed: no matching FMaps slice")
    coverage = nodes[[c for c in ["node_id", "ra", "dec", "z_low", "z_key"] if c in nodes]].copy()
    coverage["has_ridge_slice"] = has_slice.to_numpy()
    coverage.to_csv(outdir / "input_node_coverage.csv", index=False)
    nodes = nodes.loc[has_slice].reset_index(drop=True)
    n_nodes_to_process = int(len(nodes))

    node_rows: List[Dict[str, object]] = []
    for _, missing in unassessed_nodes.iterrows():
        row = missing.to_dict()
        row.update(status="no_matched_slice", n_arms_valid=np.nan,
                   audit_evaluated=False, audit_trace_evaluated=False,
                   audit_m_candidates=np.nan, audit_m_valid=np.nan,
                   audit_reason="no_matched_slice")
        node_rows.append(row)
    arm_rows: List[Dict[str, object]] = []
    seg_rows: List[Dict[str, object]] = []

    def run_node(i: int) -> Tuple[Dict[str, object], List[Dict[str, object]], List[Dict[str, object]]]:
        t0 = time.perf_counter()
        row = nodes.iloc[i]
        sd = slices[float(row["z_key"])]
        nr, ar, sr = process_one_node(row, sd, cfg)
        nr["node_wall_time_s"] = float(time.perf_counter() - t0)
        return nr, ar, sr

    print(f"[run] processing {len(nodes):,} nodes with n_jobs={args.n_jobs}")
    t_process_start = time.perf_counter()
    if int(args.n_jobs) <= 1:
        iterator: Iterable[int] = range(len(nodes))
        for i in iterator:
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

    summary = build_global_summary(node_df, cfg)
    with open(outdir / "global_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    pd.DataFrame([summary]).to_csv(outdir / "global_summary.csv", index=False)
    print("[write]", outdir / "global_summary.json")
    print(json.dumps({k: v for k, v in summary.items() if k != "config"}, indent=2, ensure_ascii=False))

    write_zbin_products(node_df, cfg, args, outdir)

    t_before_plots = time.perf_counter()
    n_ok = int(node_df.get("status", pd.Series([], dtype=str)).eq("ok").sum()) if len(node_df) else 0
    process_wall_s = float(t_process_end - t_process_start)
    read_wall_s = float(t_read_end - t_read_start)
    slice_wall_s = float(t_slice_end - t_slice_start)
    rate_nodes_per_s = float(n_nodes_to_process / process_wall_s) if process_wall_s > 0 and n_nodes_to_process > 0 else float("nan")
    ok_rate_nodes_per_s = float(n_ok / process_wall_s) if process_wall_s > 0 and n_ok > 0 else float("nan")
    # For full-run estimate: reading and slice-building are approximately fixed; node processing scales with number of intersections.
    n_full_est = int(n_nodes_after_z_before_sample)
    full_process_est_s = float(n_full_est / rate_nodes_per_s) if rate_nodes_per_s > 0 else float("nan")
    full_total_no_plots_est_s = float(read_wall_s + slice_wall_s + full_process_est_s) if np.isfinite(full_process_est_s) else float("nan")
    node_time = node_df["node_wall_time_s"].to_numpy(float) if "node_wall_time_s" in node_df.columns else np.array([], dtype=float)
    timing = {
        "n_nodes_before_z": n_nodes_before_z,
        "n_nodes_after_z_before_sample": n_nodes_after_z_before_sample,
        "n_nodes_processed_this_run": n_nodes_to_process,
        "n_nodes_ok_this_run": n_ok,
        "read_wall_s": read_wall_s,
        "build_slices_wall_s": slice_wall_s,
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
        "note": "Full-run estimate assumes node-processing time scales linearly with the number of selected intersections and excludes per-node plotting.",
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
        print(f"[gallery] writing native optimal/typical gallery to {gallery_dir}")
        make_node_gallery(node_df, seg_df, slices, cfg, args, gallery_dir)

    t_end = time.perf_counter()
    print(f"[done] total wall time = {(t_end - t_program_start)/60.0:.2f} min")


if __name__ == "__main__":
    main()
