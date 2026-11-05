#!/usr/bin/env python3
"""Reconstruct FANS projected/extrinsic diagnostic plots from exported FITS files.

This is a consistency-check script for public FITS products created by
``export_fans_fits.py``.  It does not need the original catalogue ridge files or
pipeline intermediate CSVs.  It reconstructs arm-level handedness from the
SEGMENTS HDU by grouping segments with the same (FANS_ID, ARM_ID):

    h_arm = sum_k npoints_k * T_PHI_k / sum_k npoints_k.

The projected panel then recomputes H_proj, A_sign and S_proj in each redshift
bin.  With ``--extrinsic`` it also recomputes the dipole-selected extrinsic
statistics from the same reconstructed arm table.  The output is meant as a
sanity check, not as a replacement for the full bootstrap/null pipeline.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.io import fits

from spirality.extrinsic_spirality import (
    healpix_direction_grid,
    bin_extrinsic_stats, scan_axis_grid, best_axis_from_scan, unit_xyz_from_radec, arm_weights,
)

CAT_COLOR = {
    "sconce": "0.35",
    "carron": "royalblue",
    "chen": "orangered",
}
CAT_LABEL = {
    "sconce": "SCONCE",
    "carron": "Carrón",
    "chen": "Chen",
}


def _parse_edges(text: str) -> np.ndarray:
    vals = [float(x) for x in str(text).replace(";", ",").split(",") if x.strip()]
    if len(vals) < 2:
        raise ValueError("Need at least two z-bin edges")
    return np.asarray(vals, dtype=float)


def _decode_array(arr) -> np.ndarray:
    out = np.asarray(arr)
    if out.dtype.kind in {"S", "O"}:
        return np.asarray([x.decode("utf-8", "ignore") if isinstance(x, (bytes, bytearray)) else str(x) for x in out])
    return out


def _hdu_to_df(hdul: fits.HDUList, name: str) -> pd.DataFrame:
    if name not in hdul:
        return pd.DataFrame()
    data = hdul[name].data
    if data is None:
        return pd.DataFrame()
    cols = {}
    for c in data.names:
        cols[c.lower()] = _decode_array(data[c])
    return pd.DataFrame(cols)


def reconstruct_arm_table_from_fits(path: Path, *, accepted_only: bool = True) -> pd.DataFrame:
    with fits.open(path, memmap=True) as hdul:
        # The public FITS table intentionally does not repeat the catalogue key
        # as a column.  Infer it from the filename, e.g. carron_fans_all.fits.
        catalog = str(path.stem.split("_")[0]).strip().lower()
        nodes = _hdu_to_df(hdul, "NODES")
        seg = _hdu_to_df(hdul, "SEGMENTS")
    if len(nodes) == 0 or len(seg) == 0:
        return pd.DataFrame()
    nodes = nodes.copy()
    seg = seg.copy()
    if accepted_only and "quality" in nodes.columns:
        nodes = nodes.loc[nodes["quality"].astype(bool)].copy()
    keep_ids = set(nodes["fans_id"].astype(str)) if "fans_id" in nodes.columns else set()
    if accepted_only and keep_ids and "fans_id" in seg.columns:
        seg = seg.loc[seg["fans_id"].astype(str).isin(keep_ids)].copy()
    if len(seg) == 0:
        return pd.DataFrame()

    for c in ["t_phi", "npoints", "arm_id", "z"]:
        if c in seg.columns:
            seg[c] = pd.to_numeric(seg[c], errors="coerce")
    # Node coordinates are more stable for the extrinsic arm position than
    # segment centroid coordinates; merge them back from NODES.
    node_meta = nodes[[c for c in ["fans_id", "ra", "dec", "z", "quality"] if c in nodes.columns]].copy()
    node_meta = node_meta.rename(columns={"ra": "ra_node", "dec": "dec_node", "z": "z_node"})
    seg = seg.merge(node_meta, on="fans_id", how="left", suffixes=("", "_node"))

    rows = []
    group_cols = ["fans_id", "arm_id"]
    for (fans_id, arm_id), sub in seg.groupby(group_cols, dropna=True):
        h = pd.to_numeric(sub["t_phi"], errors="coerce").to_numpy(float)
        w = pd.to_numeric(sub.get("npoints", pd.Series(np.ones(len(sub)))), errors="coerce").to_numpy(float)
        ok = np.isfinite(h) & np.isfinite(w) & (w > 0)
        if not np.any(ok):
            continue
        ww = w[ok]
        hh = h[ok]
        rows.append({
            "catalog_key": catalog,
            "fans_id": str(fans_id),
            "arm_local_id": int(arm_id),
            "ra": float(pd.to_numeric(sub["ra_node"], errors="coerce").dropna().iloc[0]) if "ra_node" in sub.columns and pd.to_numeric(sub["ra_node"], errors="coerce").notna().any() else np.nan,
            "dec": float(pd.to_numeric(sub["dec_node"], errors="coerce").dropna().iloc[0]) if "dec_node" in sub.columns and pd.to_numeric(sub["dec_node"], errors="coerce").notna().any() else np.nan,
            "z_eff": float(pd.to_numeric(sub["z_node"], errors="coerce").dropna().iloc[0]) if "z_node" in sub.columns and pd.to_numeric(sub["z_node"], errors="coerce").notna().any() else float(np.nanmean(pd.to_numeric(sub.get("z", np.nan), errors="coerce"))),
            "S_tangential": float(np.sum(ww * hh) / np.sum(ww)),
            "S_tangential_abs": float(abs(np.sum(ww * hh) / np.sum(ww))),
            "n_arm_segments_total": int(np.count_nonzero(ok)),
            "n_points_used_total": int(np.nansum(ww)),
        })
    return pd.DataFrame(rows)


def summarize_projected(arms: pd.DataFrame, edges: np.ndarray) -> pd.DataFrame:
    if len(arms) == 0:
        return pd.DataFrame()
    h = pd.to_numeric(arms["S_tangential"], errors="coerce").to_numpy(float)
    z = pd.to_numeric(arms["z_eff"], errors="coerce").to_numpy(float)
    w = pd.to_numeric(arms.get("n_arm_segments_total", pd.Series(np.ones(len(arms)))), errors="coerce").to_numpy(float)
    w = np.where(np.isfinite(w) & (w > 0), w, 1.0)
    rows = []
    for ib in range(len(edges) - 1):
        z0, z1 = float(edges[ib]), float(edges[ib + 1])
        if ib == len(edges) - 2:
            m = np.isfinite(z) & (z >= z0) & (z <= z1)
        else:
            m = np.isfinite(z) & (z >= z0) & (z < z1)
        ok = m & np.isfinite(h) & np.isfinite(w) & (w > 0)
        if not np.any(ok):
            continue
        hh, ww = h[ok], w[ok]
        sgn = np.sign(hh)
        sgood = np.isfinite(sgn) & (sgn != 0)
        rows.append({
            "catalog_key": str(arms["catalog_key"].iloc[0]),
            "zbin_id": ib,
            "z_min": z0,
            "z_max": z1,
            "z_center": float(np.nanmean(z[ok])),
            "n_arms": int(np.count_nonzero(ok)),
            "H_proj": float(np.sum(ww * hh) / np.sum(ww)),
            "A_sign": float(np.mean(sgn[sgood])) if np.any(sgood) else np.nan,
            "S_proj": float(np.sum(ww * np.abs(hh)) / np.sum(ww)),
        })
    return pd.DataFrame(rows)


def plot_projected(datasets: List[pd.DataFrame], out: Path) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(8.5, 10.0), sharex=True)
    specs = [("H_proj", r"$H_{\rm proj}$"), ("A_sign", r"$A_{\rm sign}$"), ("S_proj", r"$S_{\rm proj}$")]
    for ax, (col, ylabel) in zip(axes, specs):
        for d in datasets:
            if len(d) == 0:
                continue
            key = str(d["catalog_key"].iloc[0])
            ax.plot(d["z_center"], d[col], marker="o", lw=2.0, color=CAT_COLOR.get(key, "0.2"), label=CAT_LABEL.get(key, key))
        if col != "S_proj":
            ax.axhline(0, color="0.3", lw=0.9)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
    axes[-1].set_xlabel(r"$z$")
    axes[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def plot_extrinsic(datasets: List[pd.DataFrame], out: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(8.5, 7.0), sharex=True)
    specs = [("H_ext", r"$H_{\rm ext}$"), ("sign_asym_ext_percent", r"$A_{\rm sign}^{\rm ext}$ [%]")]
    for ax, (col, ylabel) in zip(axes, specs):
        for d in datasets:
            if len(d) == 0 or col not in d.columns:
                continue
            key = str(d["catalog_key"].iloc[0])
            ax.plot(d["z_center"], d[col], marker="o", lw=2.0, color=CAT_COLOR.get(key, "0.2"), label=CAT_LABEL.get(key, key))
        ax.axhline(0, color="0.3", lw=0.9)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
    axes[-1].set_xlabel(r"$z$")
    axes[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fits", nargs="+", required=True, help="Exported FANS FITS files.")
    p.add_argument("--outdir", required=True)
    p.add_argument("--zbin-edges", default="0.1,0.225,0.35,0.475,0.6")
    p.add_argument("--plot-format", default="pdf", choices=["pdf", "png"])
    p.add_argument("--include-all", dest="include_all", action="store_true",
                   help="Do not restrict to QUALITY=True nodes.")
    p.add_argument("--extrinsic", action="store_true", help="Also reconstruct extrinsic z-bin panel from FITS.")
    p.add_argument("--nside-scan", type=int, default=32)
    args = p.parse_args()

    outdir = Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    edges = _parse_edges(args.zbin_edges)

    arm_tables = []
    projected_tables = []
    extrinsic_tables = []
    for f in args.fits:
        path = Path(f).expanduser().resolve()
        arms = reconstruct_arm_table_from_fits(path, accepted_only=not bool(args.include_all))
        if len(arms) == 0:
            print(f"[warn] no arms reconstructed from {path}")
            continue
        arm_tables.append(arms)
        proj = summarize_projected(arms, edges)
        projected_tables.append(proj)
        print(f"[read] {path}: reconstructed arms={len(arms):,}")
        if args.extrinsic:
            dirs = healpix_direction_grid(int(args.nside_scan))
            scan = scan_axis_grid(arm_xyz=unit_xyz_from_radec(arms.ra.to_numpy(float), arms.dec.to_numpy(float)),
                                  sphi=arms.S_tangential.to_numpy(float), directions=dirs,
                                  weights=arm_weights(arms, "n_segments"), chunk_size=128)
            best = best_axis_from_scan(scan, objective="dipole")
            axis = unit_xyz_from_radec(np.array([best["ra_deg"]]), np.array([best["dec_deg"]]))[0]
            ext = bin_extrinsic_stats(arms, directions=dirs, edges=edges, weight_mode="n_segments",
                                      chunk_size=128, fixed_axis=best, zbin_axis_mode="fixed_global")
            ext.insert(0, "catalog_key", str(arms["catalog_key"].iloc[0]))
            extrinsic_tables.append(ext)

    if arm_tables:
        pd.concat(arm_tables, ignore_index=True).to_csv(outdir / "fits_reconstructed_arm_level.csv", index=False)
    if projected_tables:
        projected = pd.concat(projected_tables, ignore_index=True)
        projected.to_csv(outdir / "fits_projected_zbins.csv", index=False)
        plot_projected(projected_tables, outdir / f"fits_projected_recheck.{args.plot_format}")
        print("[write]", outdir / "fits_projected_zbins.csv")
    if extrinsic_tables:
        extrinsic = pd.concat(extrinsic_tables, ignore_index=True)
        extrinsic.to_csv(outdir / "fits_extrinsic_zbins.csv", index=False)
        plot_extrinsic(extrinsic_tables, outdir / f"fits_extrinsic_recheck.{args.plot_format}")
        print("[write]", outdir / "fits_extrinsic_zbins.csv")


if __name__ == "__main__":
    main()
