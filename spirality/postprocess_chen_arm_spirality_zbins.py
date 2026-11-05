#!/usr/bin/env python3
"""
Post-process Chen DR12 arm_spirality.csv into SCONCE-compatible arm-level
redshift-bin summaries.

The output columns match the subset needed by sconce_node_spirality_v20+ for
DR12/DR17 catalogue-comparison panels:
  H_S_weighted_mean, mean_abs_projected_twist, mean_radial_coherence,
  mean_transverse_power, and bootstrap errors/quantiles.

For Chen arm rows:
  S_phi is mean_t_phi
  C_radial is mean_t_radial
  density weight is mean_arm_density, if available.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    ok = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not np.any(ok):
        return np.nan
    return float(np.sum(weights[ok] * values[ok]) / np.sum(weights[ok]))


def weights_for(df: pd.DataFrame, mode: str) -> np.ndarray:
    mode = str(mode).lower()
    if mode == "uniform":
        return np.ones(len(df), dtype=float)
    if mode == "mean_density":
        den = df["mean_arm_density"].to_numpy(float) if "mean_arm_density" in df.columns else np.ones(len(df), dtype=float)
        return np.where(np.isfinite(den) & (den > 0), den, 0.0)
    if mode == "sqrt_mean_density":
        den = df["mean_arm_density"].to_numpy(float) if "mean_arm_density" in df.columns else np.ones(len(df), dtype=float)
        return np.where(np.isfinite(den) & (den > 0), np.sqrt(den), 0.0)
    if mode == "n_segments":
        return df["n_arm_segments_valid"].to_numpy(float) if "n_arm_segments_valid" in df.columns else np.ones(len(df), dtype=float)
    if mode == "sqrt_points":
        pts = df["n_points_total"].to_numpy(float) if "n_points_total" in df.columns else np.ones(len(df), dtype=float)
        return np.sqrt(np.maximum(pts, 0.0))
    raise ValueError(f"Unknown arm/node weight mode: {mode}")


def bootstrap_mean(values: np.ndarray, weights: np.ndarray, *, B: int, rng: np.random.Generator) -> Tuple[float, float, float, float, np.ndarray]:
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    ok = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    v = values[ok]
    w = weights[ok]
    n = len(v)
    if n == 0:
        return np.nan, np.nan, np.nan, np.nan, np.array([], dtype=float)
    obs = weighted_mean(v, w)
    if B <= 0 or n < 2:
        return obs, np.nan, np.nan, np.nan, np.array([], dtype=float)
    boots = np.empty(B, dtype=float)
    for b in range(B):
        ii = rng.integers(0, n, size=n)
        boots[b] = weighted_mean(v[ii], w[ii])
    return (
        obs,
        float(np.nanstd(boots, ddof=1)),
        float(np.nanpercentile(boots, 16.0)),
        float(np.nanpercentile(boots, 84.0)),
        boots,
    )


def signflip_null(values: np.ndarray, weights: np.ndarray, *, B: int, rng: np.random.Generator) -> Tuple[float, float]:
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    ok = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    v = values[ok]
    w = weights[ok]
    if len(v) == 0 or B <= 0:
        return np.nan, np.nan
    obs = weighted_mean(v, w)
    null = np.empty(B, dtype=float)
    for b in range(B):
        signs = rng.choice(np.array([-1.0, 1.0]), size=len(v), replace=True)
        null[b] = weighted_mean(signs * v, w)
    p = float((1.0 + np.count_nonzero(np.abs(null) >= abs(obs))) / (B + 1.0))
    z = float(obs / np.nanstd(null, ddof=1)) if np.nanstd(null, ddof=1) > 0 else np.nan
    return p, z


def summarize(df: pd.DataFrame, *, weight_mode: str, bootstrap_B: int, signflip_B: int, seed: int) -> Dict[str, float]:
    out: Dict[str, float] = {
        "n_nodes_total_rows": int(len(df)),  # kept for compatibility; this is n arms here
        "n_nodes_ok": int(len(df)),          # kept for compatibility; this is n arms here
        "n_arms_ok": int(len(df)),
    }
    if len(df) == 0:
        return out
    rng = np.random.default_rng(seed)
    w = weights_for(df, weight_mode)

    vals = df["mean_t_phi"].to_numpy(float) if "mean_t_phi" in df.columns else np.full(len(df), np.nan)
    obs, err, q16, q84, _ = bootstrap_mean(vals, w, B=bootstrap_B, rng=rng)
    p, zsf = signflip_null(vals, w, B=signflip_B, rng=rng)
    out.update({
        "H_S_weighted_mean": obs,
        "H_S_bootstrap_err": err,
        "H_S_bootstrap_z": obs / err if np.isfinite(err) and err > 0 else np.nan,
        "H_S_bootstrap_q16": q16,
        "H_S_bootstrap_q84": q84,
        "H_S_signflip_p_two_sided": p,
        "H_S_signflip_z": zsf,
    })

    for values, label in [
        (np.abs(vals), "mean_abs_projected_twist"),
        (df["mean_t_radial"].to_numpy(float) if "mean_t_radial" in df.columns else np.full(len(df), np.nan), "mean_radial_coherence"),
        (vals**2, "mean_transverse_power"),
    ]:
        obs, err, q16, q84, _ = bootstrap_mean(values, w, B=bootstrap_B, rng=rng)
        out[label] = obs
        out[f"{label}_bootstrap_err"] = err
        out[f"{label}_bootstrap_q16"] = q16
        out[f"{label}_bootstrap_q84"] = q84

    return out


def parse_edges(edges_str: str) -> np.ndarray:
    edges = np.array([float(x) for x in edges_str.replace(";", ",").split(",") if x.strip()], dtype=float)
    if len(edges) < 2 or not np.all(np.diff(edges) > 0):
        raise ValueError("Invalid --zbin-edges")
    return edges


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--arm-spirality", required=True, help="Chen arm_spirality.csv")
    p.add_argument("--outdir", required=True)
    p.add_argument("--zbin-edges", required=True, help="Comma-separated z-bin edges")
    p.add_argument("--weight-mode", default="uniform",
                   choices=["uniform", "mean_density", "sqrt_mean_density", "n_segments", "sqrt_points"])
    p.add_argument("--bootstrap-B", type=int, default=1000)
    p.add_argument("--signflip-B", type=int, default=2000)
    p.add_argument("--seed", type=int, default=2026)
    args = p.parse_args()

    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.arm_spirality)
    from spirality.selection import attach_parent_metadata
    node_path = Path(args.arm_spirality).parent / "node_spirality.csv"
    if not node_path.exists():
        raise FileNotFoundError("Parent node table required for Chen arm selection")
    df = attach_parent_metadata(df, pd.read_csv(node_path), accepted_only=True)
    if "z_low" not in df.columns:
        raise ValueError("arm_spirality.csv must contain z_low")
    edges = parse_edges(args.zbin_edges)

    rows = []
    for ib in range(len(edges)-1):
        zlo, zhi = float(edges[ib]), float(edges[ib+1])
        if ib == len(edges)-2:
            mask = (df["z_low"].to_numpy(float) >= zlo) & (df["z_low"].to_numpy(float) <= zhi)
        else:
            mask = (df["z_low"].to_numpy(float) >= zlo) & (df["z_low"].to_numpy(float) < zhi)
        sub = df.loc[mask].copy()
        summ = summarize(sub, weight_mode=args.weight_mode, bootstrap_B=args.bootstrap_B,
                         signflip_B=args.signflip_B, seed=args.seed + 700 + 91*ib)
        summ.update({
            "zbin_id": ib,
            "z_min": zlo,
            "z_max": zhi,
            "z_center": 0.5*(zlo+zhi),
            "z_width": zhi-zlo,
            "sample": "all",
            "weight_mode": args.weight_mode,
        })
        rows.append(summ)

    zdf = pd.DataFrame(rows)
    # Backwards-compatible generic output
    zdf.to_csv(outdir / "zbin_summary_arms.csv", index=False)
    with open(outdir / "zbin_summary_arms.json", "w") as f:
        json.dump(rows, f, indent=2)

    # SCONCE-compatible mode-specific output, needed by joint plot scripts.
    mode = str(args.weight_mode)
    zdf.to_csv(outdir / f"zbin_summary_arms_{mode}.csv", index=False)
    with open(outdir / f"zbin_summary_arms_{mode}.json", "w") as f:
        json.dump(rows, f, indent=2)

    print(f"Wrote {outdir / 'zbin_summary_arms.csv'}")
    print(f"Wrote {outdir / f'zbin_summary_arms_{mode}.csv'}")


if __name__ == "__main__":
    main()
