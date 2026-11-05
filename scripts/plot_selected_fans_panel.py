#!/usr/bin/env python3
"""Plot a paper-style panel of selected FANS IDs from public FITS files.

This script works only with the compact public FITS products.  It plots each
selected FANS in the local tangent plane using the SEGMENTS extension:

  * node at (0, 0);
  * segment centroids at (X_MPC_H, Y_MPC_H);
  * one polyline per arm, sorted by SEG_INDEX;
  * short arrows showing the outward-oriented local tangent direction estimated
    from T_RADIAL, T_PHI and PHI_DEG.

Because the public FITS tables are intentionally compact, this panel does not
show all original ridge points.  Use the native gallery products from the full
pipeline if a raw-point gallery is required.
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from astropy.io import fits


def use_times_or_fallback() -> None:
    candidates = [
        "/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf",
        "/usr/share/fonts/truetype/noto/NotoSerif-Regular.ttf",
    ]
    for p in candidates:
        if os.path.exists(p):
            fm.fontManager.addfont(p)
    plt.rcParams["font.family"] = ["Times New Roman", "Liberation Serif", "Noto Serif", "DejaVu Serif", "serif"]


def read_fans_fits(path: Path, quality_only: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
    with fits.open(path) as hdul:
        ndata = hdul["NODES"].data
        sdata = hdul["SEGMENTS"].data
        nodes = pd.DataFrame({name: np.asarray(ndata[name]) for name in ndata.names})
        seg = pd.DataFrame({name: np.asarray(sdata[name]) for name in sdata.names})
    for df in [nodes, seg]:
        for col in df.columns:
            if df[col].dtype.kind == "S":
                df[col] = df[col].str.decode("utf-8").str.strip()
            elif df[col].dtype.kind == "O":
                df[col] = df[col].apply(lambda x: x.decode("utf-8").strip() if isinstance(x, (bytes, bytearray)) else x)
    if quality_only and "QUALITY" in nodes.columns:
        qn = pd.to_numeric(nodes["QUALITY"], errors="coerce").fillna(0).astype(int) > 0
        nodes = nodes.loc[qn].copy()
    if quality_only and "QUALITY" in seg.columns:
        qs = pd.to_numeric(seg["QUALITY"], errors="coerce").fillna(0).astype(int) > 0
        seg = seg.loc[qs].copy()
    nodes["SOURCE_FITS"] = str(path)
    seg["SOURCE_FITS"] = str(path)
    return nodes, seg


def read_all(fits_paths: List[Path], quality_only: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
    nodes_all = []
    seg_all = []
    for p in fits_paths:
        n, s = read_fans_fits(p, quality_only=quality_only)
        nodes_all.append(n)
        seg_all.append(s)
    return pd.concat(nodes_all, ignore_index=True), pd.concat(seg_all, ignore_index=True)


def _load_ids(args: argparse.Namespace) -> List[str]:
    ids: List[str] = []
    if args.fans_ids:
        for item in args.fans_ids:
            ids.extend([x.strip() for x in str(item).replace(",", " ").split() if x.strip()])
    if args.ids_file:
        for line in Path(args.ids_file).read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            ids.extend([x.strip() for x in line.replace(",", " ").split() if x.strip()])
    # preserve order, remove duplicates
    out = []
    seen = set()
    for x in ids:
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out


def tangent_xy(row: pd.Series, scale: float = 2.2) -> Tuple[float, float]:
    phi = math.radians(float(row.get("PHI_DEG", np.nan)))
    tr = float(row.get("T_RADIAL", np.nan))
    tp = float(row.get("T_PHI", np.nan))
    if not np.isfinite(phi) or not np.isfinite(tr) or not np.isfinite(tp):
        return np.nan, np.nan
    er = np.array([math.cos(phi), math.sin(phi)])
    ep = np.array([-math.sin(phi), math.cos(phi)])
    v = tr * er + tp * ep
    norm = np.linalg.norm(v)
    if norm > 0:
        v = v / norm
    return float(scale * v[0]), float(scale * v[1])


def plot_one(ax, node: pd.Series, seg: pd.DataFrame, *, arrow_scale: float = 2.2) -> None:
    fans_id = str(node["FANS_ID"])
    sub = seg[seg["FANS_ID"].astype(str) == fans_id].copy()
    sub["ARM_ID"] = pd.to_numeric(sub["ARM_ID"], errors="coerce")
    sub["SEG_INDEX"] = pd.to_numeric(sub["SEG_INDEX"], errors="coerce")
    sub["X_MPC_H"] = pd.to_numeric(sub["X_MPC_H"], errors="coerce")
    sub["Y_MPC_H"] = pd.to_numeric(sub["Y_MPC_H"], errors="coerce")

    ax.axhline(0, color="0.85", lw=0.8, zorder=0)
    ax.axvline(0, color="0.85", lw=0.8, zorder=0)
    ax.scatter([0], [0], s=32, color="black", zorder=5)

    if len(sub):
        arms = sorted([a for a in sub["ARM_ID"].dropna().unique()])
        for ai, arm in enumerate(arms):
            ss = sub[sub["ARM_ID"] == arm].sort_values("SEG_INDEX")
            x = ss["X_MPC_H"].to_numpy(float)
            y = ss["Y_MPC_H"].to_numpy(float)
            ax.plot(x, y, lw=2.4, color=f"C{ai % 10}", alpha=0.85, zorder=2)
            ax.scatter(x, y, s=18, color=f"C{ai % 10}", edgecolor="black", linewidth=0.3, zorder=3)
            for _, rr in ss.iterrows():
                dx, dy = tangent_xy(rr, scale=arrow_scale)
                if np.isfinite(dx) and np.isfinite(dy):
                    ax.arrow(
                        float(rr["X_MPC_H"]), float(rr["Y_MPC_H"]), dx, dy,
                        color=f"C{ai % 10}", lw=0.8,
                        head_width=0.7, head_length=0.8,
                        length_includes_head=True,
                        alpha=0.75,
                        zorder=4,
                    )

    # limits from available local coordinates
    vals = []
    if len(sub):
        vals.extend(sub["X_MPC_H"].dropna().to_numpy(float).tolist())
        vals.extend(sub["Y_MPC_H"].dropna().to_numpy(float).tolist())
    lim = max(32.0, np.nanmax(np.abs(vals)) * 1.18 if vals else 32.0)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal", adjustable="box")

    narm = int(node.get("NARMS", -1)) if np.isfinite(float(node.get("NARMS", np.nan))) else -1
    z = float(node.get("Z", np.nan))
    h = float(node.get("HANDEDNESS", np.nan))
    s = float(node.get("SPIRALITY", np.nan))
    short_id = fans_id.replace("_FANS_", "\n")
    ax.set_title(f"{short_id}\n$z={z:.3f}$, $N_{{arm}}={narm}$\n$H={h:+.3f}$, $S={s:.3f}$", fontsize=9)
    ax.tick_params(direction="in", top=True, right=True, labelsize=8, length=3)
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fits", nargs="+", required=True, help="Input public FANS FITS files, e.g. carron_fans_all.fits chen_fans_all.fits")
    p.add_argument("--fans-ids", nargs="*", default=None, help="FANS IDs to plot. May be comma- or space-separated.")
    p.add_argument("--ids-file", default=None, help="Optional text file with FANS IDs to plot.")
    p.add_argument("--out", required=True, help="Output PDF/PNG path.")
    p.add_argument("--ncols", type=int, default=4)
    p.add_argument("--quality-only", action="store_true", help="Use only QUALITY=1 rows from FITS.")
    p.add_argument("--arrow-scale", type=float, default=2.2)
    args = p.parse_args()

    use_times_or_fallback()
    plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "mathtext.fontset": "stix"})

    fits_paths = [Path(x).expanduser().resolve() for x in args.fits]
    nodes, seg = read_all(fits_paths, quality_only=bool(args.quality_only))
    ids = _load_ids(args)
    if not ids:
        raise SystemExit("No FANS IDs supplied. Use --fans-ids or --ids-file.")
    have = set(nodes["FANS_ID"].astype(str))
    missing = [x for x in ids if x not in have]
    if missing:
        print("[warn] missing IDs:", ", ".join(missing))
    ids = [x for x in ids if x in have]
    if not ids:
        examples = nodes["FANS_ID"].astype(str).head(20).tolist()
        raise SystemExit("No requested IDs found. Example available IDs:\n" + "\n".join(examples))

    n = len(ids)
    ncols = max(1, int(args.ncols))
    nrows = int(math.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.0 * ncols, 3.35 * nrows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, fid in zip(axes.ravel(), ids):
        ax.axis("on")
        node = nodes[nodes["FANS_ID"].astype(str) == fid].iloc[0]
        plot_one(ax, node, seg, arrow_scale=float(args.arrow_scale))
        if ax.is_first_col():
            ax.set_ylabel(r"$Y\,[h^{-1}{\rm Mpc}]$", fontsize=9)
        if ax.is_last_row():
            ax.set_xlabel(r"$X\,[h^{-1}{\rm Mpc}]$", fontsize=9)

    fig.tight_layout(pad=0.9)
    out = Path(args.out).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight", dpi=300)
    print(f"[write] {out}")


if __name__ == "__main__":
    main()
