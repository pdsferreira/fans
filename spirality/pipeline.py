#!/usr/bin/env python3
"""
Organized driver for the filament-node spirality analysis (v27).

This module orchestrates three independent catalogue adapters:

  * Zhang et al. / SCONCE DR17 native nodes and ridges;
  * Carrón et al. DR17 ridges measured around SCONCE node positions;
  * Chen et al. DR12 native nodes and ridges.

Fiducial conventions are fixed:

  * Zhang/SCONCE vector interpretation: tangent;
  * Carrón angle convention: E = cos(angle), N = sin(angle), offset = 0 deg;
  * Chen direction convention: physical, E = v_ra cos(dec), N = v_dec.

Development-only PA tests and combined SCONCE--Carrón overlay galleries are not
part of this public pipeline.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shlex
import shutil
import subprocess
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DEG = 180.0 / np.pi
LOG_LOCK = threading.Lock()


# Helper modules are ordinary package files; no embedded helper blobs are used.


# -----------------------------------------------------------------------------
# Command utilities
# -----------------------------------------------------------------------------

def resolve_script(path: Optional[str], default_name: str, *, helper_dir: Optional[Path] = None) -> Path:
    """Resolve an external helper path or a helper script shipped with the package."""
    if path is None or str(path).strip() == "":
        if helper_dir is not None:
            p = helper_dir / default_name
        else:
            p = Path(__file__).resolve().parent / default_name
    else:
        p = Path(path).expanduser()
    p = p.resolve()
    if not p.exists():
        raise FileNotFoundError(
            f"Required helper script not found: {p}\n"
            f"Pass the full path, or keep the package helper next to pipeline.py."
        )
    return p


def run_cmd(cmd: List[str], *, dry_run: bool, log_file: Path) -> None:
    line = " ".join(shlex.quote(str(x)) for x in cmd)
    print("\n" + line + "\n")
    with LOG_LOCK:
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(line + "\n\n")
    if not dry_run:
        subprocess.run(cmd, check=True)


def run_pipeline_task(name: str, cmd: List[str], post_cmds: List[List[str]], *, dry_run: bool, log_file: Path) -> dict:
    """Run one catalogue pipeline command plus optional dependent post-commands."""
    t0 = time.perf_counter()
    print(f"[pipeline:{name}] start")
    run_cmd(cmd, dry_run=dry_run, log_file=log_file)
    for post in post_cmds:
        run_cmd(post, dry_run=dry_run, log_file=log_file)
    t1 = time.perf_counter()
    row = {
        "pipeline": name,
        "wall_time_s": t1 - t0,
        "n_post_commands": len(post_cmds),
        "command": " ".join(shlex.quote(str(x)) for x in cmd),
    }
    print(f"[pipeline:{name}] done in {(t1 - t0)/60:.2f} min")
    return row


def write_pipeline_timing(rows: List[dict], out_csv: Path) -> None:
    if not rows:
        return
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    keys = ["pipeline", "wall_time_s", "n_post_commands", "command"]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in keys})
    print(f"[write] {out_csv}")


def run_pipeline_tasks(
    tasks: List[Tuple[str, List[str], List[List[str]]]],
    *,
    workers: int,
    dry_run: bool,
    log_file: Path,
    timing_csv: Path,
) -> None:
    """Run independent catalogue pipelines serially or in parallel."""
    if len(tasks) == 0:
        return
    rows: List[dict] = []
    if workers <= 1 or len(tasks) == 1 or dry_run:
        for name, cmd, post_cmds in tasks:
            rows.append(run_pipeline_task(name, cmd, post_cmds, dry_run=dry_run, log_file=log_file))
        write_pipeline_timing(rows, timing_csv)
        return

    print(f"[parallel] running {len(tasks)} pipeline tasks with {workers} workers")
    with ThreadPoolExecutor(max_workers=int(workers)) as ex:
        futs = {
            ex.submit(run_pipeline_task, name, cmd, post_cmds, dry_run=dry_run, log_file=log_file): name
            for name, cmd, post_cmds in tasks
        }
        for fut in as_completed(futs):
            name = futs[fut]
            try:
                rows.append(fut.result())
            except Exception as exc:
                print(f"[pipeline:{name}] failed: {exc!r}")
                raise
    rows.sort(key=lambda r: str(r.get("pipeline", "")))
    write_pipeline_timing(rows, timing_csv)


def prepare_dir(path: Path, *, overwrite: bool, dry_run: bool) -> None:
    if path.exists() and overwrite:
        print(f"[clean] removing {path}")
        if not dry_run:
            shutil.rmtree(path)
    if not dry_run:
        path.mkdir(parents=True, exist_ok=True)


def existing_pipeline_products_ok(path: Path) -> bool:
    """True when the heavy node/arm/segment measurements already exist."""
    return all((path / name).exists() for name in ["node_spirality.csv", "arm_spirality.csv", "arm_segments.csv"])


def prune_legacy_plot_files(root: Path) -> None:
    """Remove helper-generated one-catalogue plot files to avoid plot clutter.

    This keeps all CSV/FITS/JSON products.  The final products intended for
    inspection are in three_catalog_plots/ and triple_gallery/.
    """
    if not root.exists():
        return
    removed = 0
    # Do NOT prune "gallery": those are now the trusted native per-pipeline
    # diagnostic galleries inherited from the old working SCONCE logic.
    for subdir_name in ["zbin_plots", "sky_maps", "density_correlations"]:
        for d in root.rglob(subdir_name):
            if not d.is_dir():
                continue
            for ext in ("*.pdf", "*.png", "*.jpg", "*.jpeg"):
                for f in d.glob(ext):
                    try:
                        f.unlink()
                        removed += 1
                    except OSError:
                        pass
    if removed:
        print(f"[clean] removed {removed} legacy helper plot files under {root}")


# -----------------------------------------------------------------------------
# Joint plot utilities
# -----------------------------------------------------------------------------

STYLE = {
    "sconce": dict(label="Zhang et al. SDSS DR17", line="seagreen", fill="lightgreen", marker="o", ms=7.5, z=2),
    "carron": dict(label="Carrón et al. SDSS DR16", line="royalblue", fill="lightblue", marker="o", ms=7.8, z=4),
    "chen": dict(label="Chen et al. SDSS DR12", line="orangered", fill="lightsalmon", marker="o", ms=7.8, z=5),
}


def setup_matplotlib() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "axes.linewidth": 1.1,
        "axes.labelsize": 20,
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,
        "legend.fontsize": 15,
        "figure.dpi": 140,
        "savefig.dpi": 260,
    })



def _catalog_linestyle(key: str) -> str:
    key = str(key).lower()
    if key == "carron":
        return "--"
    if key == "chen":
        return ":"
    if key == "sconce":
        return "-."
    return "--"


def apply_bin_filter(df: pd.DataFrame, *, zmin: Optional[float], zmax: Optional[float], mode: str) -> pd.DataFrame:
    mode = str(mode).lower()
    d = df.copy()
    if "sample" in d.columns:
        d = d[d["sample"].astype(str).eq("all")].copy()
    if "z_center" not in d.columns and {"z_min", "z_max"}.issubset(d.columns):
        d["z_center"] = 0.5 * (d["z_min"].astype(float) + d["z_max"].astype(float))
    if mode == "none" or (zmin is None and zmax is None):
        return d.sort_values("z_center").reset_index(drop=True)

    if not {"z_min", "z_max", "z_center"}.issubset(d.columns):
        raise ValueError("Redshift filtering requires z_min, z_max, z_center columns.")

    zlo = d["z_min"].to_numpy(float)
    zhi = d["z_max"].to_numpy(float)
    zc = d["z_center"].to_numpy(float)
    mask = np.ones(len(d), dtype=bool)
    if mode == "overlap":
        if zmin is not None:
            mask &= zhi > float(zmin)
        if zmax is not None:
            mask &= zlo < float(zmax)
    elif mode == "center":
        if zmin is not None:
            mask &= zc >= float(zmin)
        if zmax is not None:
            mask &= zc <= float(zmax)
    elif mode == "contained":
        if zmin is not None:
            mask &= zlo >= float(zmin)
        if zmax is not None:
            mask &= zhi <= float(zmax)
    else:
        raise ValueError("--plot-bin-filter-mode must be none, overlap, center, or contained")
    return d.loc[mask].sort_values("z_center").reset_index(drop=True)


def read_summary_file(path: Path, *, key: str, label: str,
                      zmin: Optional[float], zmax: Optional[float], mode: str) -> Optional[pd.DataFrame]:
    if not path.exists():
        print(f"[warn] missing summary for {label}: {path}")
        return None
    d = pd.read_csv(path)
    d["__source_file"] = str(path)
    d = apply_bin_filter(d, zmin=zmin, zmax=zmax, mode=mode)
    d["catalog_key"] = key
    d["catalog_label"] = label
    print(f"[summary] {label:8s}: {len(d)} rows from {path}")
    if len(d) and {"z_min", "z_max", "z_center"}.issubset(d.columns):
        print(d[["z_min", "z_max", "z_center"]].to_string(index=False))
    metrics = ["H_S_weighted_mean", "mean_abs_projected_twist", "mean_radial_coherence", "mean_transverse_power", "sign_asymmetry_percent"]
    counts = {m: int(np.count_nonzero(np.isfinite(pd.to_numeric(d[m], errors="coerce"))))
              for m in metrics if m in d.columns}
    if counts:
        print(f"[summary metrics] {label}: {counts}")
    return d


def err_band(df: pd.DataFrame, ycol: str, ecol: Optional[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    y = df[ycol].to_numpy(float)
    q16 = f"{ycol}_bootstrap_q16"
    q84 = f"{ycol}_bootstrap_q84"
    if q16 in df.columns and q84 in df.columns:
        lo = df[q16].to_numpy(float)
        hi = df[q84].to_numpy(float)
        ok = np.isfinite(lo) & np.isfinite(hi)
        if np.count_nonzero(ok) >= max(1, len(df)//2):
            return y, lo, hi
    if ecol and ecol in df.columns:
        e = df[ecol].to_numpy(float)
        return y, y - e, y + e
    return y, np.full_like(y, np.nan), np.full_like(y, np.nan)


def asin_deg_clip(x: np.ndarray) -> np.ndarray:
    return np.arcsin(np.clip(np.asarray(x, dtype=float), -1.0, 1.0)) * DEG


def y_and_band(df: pd.DataFrame, ycol: str, ecol: Optional[str], *, angle: bool) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    y, lo, hi = err_band(df, ycol, ecol)
    if not angle:
        return y, lo, hi
    return asin_deg_clip(y), asin_deg_clip(lo), asin_deg_clip(hi)


def plot_one_catalog(ax, df: pd.DataFrame, *, ycol: str, ecol: str, angle: bool, key: str, label: str) -> Optional[pd.DataFrame]:
    st = STYLE[key]
    x = df["z_center"].to_numpy(float)
    y, lo, hi = y_and_band(df, ycol, ecol, angle=angle)
    good = np.isfinite(x) & np.isfinite(y)
    good_band = good & np.isfinite(lo) & np.isfinite(hi)
    if np.count_nonzero(good) == 0:
        print(f"[warn] no finite values for {label} column {ycol} from {df['__source_file'].iloc[0] if '__source_file' in df.columns and len(df) else 'unknown'}")

    if np.count_nonzero(good_band) >= 2:
        ax.fill_between(x[good_band], lo[good_band], hi[good_band],
                        color=st["fill"], alpha=0.34 if key != "sconce" else 0.42,
                        lw=0, zorder=st["z"])
        ax.plot(x[good_band], lo[good_band], color=st["line"], lw=1.05, alpha=0.78, zorder=st["z"]+0.1)
        ax.plot(x[good_band], hi[good_band], color=st["line"], lw=1.05, alpha=0.78, zorder=st["z"]+0.1)
    if np.count_nonzero(good) >= 2:
        order = np.argsort(x[good])
        ax.plot(x[good][order], y[good][order], ls="--", lw=1.35, color=st["line"], alpha=0.78, zorder=st["z"]+0.75)
    ax.plot(x[good], y[good], ls="none", marker=st["marker"], ms=st["ms"],
            color=st["line"], markerfacecolor=st["line"], markeredgecolor=st["line"],
            markeredgewidth=1.0, label=label, zorder=st["z"]+1.0)

    return pd.DataFrame({
        "catalog": label,
        "catalog_key": key,
        "z_min": df["z_min"].to_numpy(float) if "z_min" in df.columns else np.nan,
        "z_max": df["z_max"].to_numpy(float) if "z_max" in df.columns else np.nan,
        "z_center": x,
        "y": y,
        "y_low": lo,
        "y_high": hi,
        "raw_column": ycol,
        "angle_version": bool(angle),
        "source_file": df["__source_file"].iloc[0] if "__source_file" in df.columns and len(df) else "",
    })



def compute_sign_asymmetry_from_arm_table(arm_path: Path, edges: np.ndarray, *, key: str, label: str) -> pd.DataFrame:
    """Compute arm-level sign asymmetry in redshift bins from an arm-level table.

    This standalone diagnostic plot is kept for backward compatibility with the
    earlier pipeline outputs.  The public 2x2 panels already include Sign
    Asymmetry, but this one-panel plot is useful for quick checks.
    """
    if not arm_path.exists():
        print(f"[warn] missing arm table for sign asymmetry: {arm_path}")
        return pd.DataFrame()
    d = pd.read_csv(arm_path)
    d = normalise_measurement_columns(d)
    d["z_center"] = robust_z_center(d)

    z = pd.to_numeric(d["z_center"], errors="coerce").to_numpy(float)
    s = pd.to_numeric(d.get("S_tangential", np.nan), errors="coerce").to_numpy(float)

    rows = []
    for ib in range(len(edges) - 1):
        z0, z1 = float(edges[ib]), float(edges[ib + 1])
        if ib == len(edges) - 2:
            mask = np.isfinite(z) & np.isfinite(s) & (z >= z0) & (z <= z1)
        else:
            mask = np.isfinite(z) & np.isfinite(s) & (z >= z0) & (z < z1)
        subz = z[mask]
        subs = s[mask]
        n_plus = int(np.count_nonzero(subs > 0))
        n_minus = int(np.count_nonzero(subs < 0))
        n_tot = n_plus + n_minus
        asym = float((n_plus - n_minus) / n_tot) if n_tot > 0 else np.nan
        err = float(np.sqrt(max(0.0, 1.0 - asym * asym) / n_tot)) if n_tot > 0 and np.isfinite(asym) else np.nan
        rows.append({
            "catalog_key": key,
            "catalog_label": label,
            "z_min": z0,
            "z_max": z1,
            "z_center": float(np.nanmean(subz)) if len(subz) else 0.5 * (z0 + z1),
            "n_plus": n_plus,
            "n_minus": n_minus,
            "n_total": n_tot,
            "sign_asymmetry": asym,
            "sign_asymmetry_percent": 100.0 * asym if np.isfinite(asym) else np.nan,
            "sign_asymmetry_err": err,
            "sign_asymmetry_percent_err": 100.0 * err if np.isfinite(err) else np.nan,
            "source_file": str(arm_path),
        })
    return pd.DataFrame(rows)


def plot_sign_asymmetry(*, sconce_dir: Path, carron_dir: Path, chen_dir: Path,
                        outdir: Path, fmt: str, sconce_edges: str, carron_edges: str, chen_edges: str) -> None:
    """Write the legacy one-panel arm sign-asymmetry diagnostic plot."""
    setup_matplotlib()
    datasets = []
    specs = [
        ("sconce", "Zhang et al. SDSS DR17", sconce_dir / "arm_level_spirality.csv", parse_edges_string(sconce_edges)),
        ("carron", "Carrón et al. SDSS DR16", carron_dir / "arm_level_spirality.csv", parse_edges_string(carron_edges)),
        ("chen", "Chen et al. SDSS DR12", chen_dir / "arm_level_spirality.csv", parse_edges_string(chen_edges)),
    ]
    for key, label, path, edges in specs:
        ds = compute_sign_asymmetry_from_arm_table(path, edges, key=key, label=label)
        if len(ds):
            datasets.append(ds)

    if not datasets:
        print("[warn] no sign-asymmetry datasets available")
        return

    dcat = pd.concat(datasets, ignore_index=True)
    outdir.mkdir(parents=True, exist_ok=True)
    csv = outdir / "three_catalog_sign_asymmetry_arms.csv"
    dcat.to_csv(csv, index=False)

    fig, ax = plt.subplots(figsize=(9.0, 6.1))
    for key, label in [
        ("sconce", "Zhang et al. SDSS DR17"),
        ("carron", "Carrón et al. SDSS DR16"),
        ("chen", "Chen et al. SDSS DR12"),
    ]:
        d = dcat[dcat["catalog_key"].eq(key)].copy()
        if len(d) == 0:
            continue
        st = STYLE[key]
        x = d["z_center"].to_numpy(float)
        y = d["sign_asymmetry_percent"].to_numpy(float)
        e = d["sign_asymmetry_percent_err"].to_numpy(float)
        good = np.isfinite(x) & np.isfinite(y)
        good_err = good & np.isfinite(e)
        if np.count_nonzero(good_err) >= 2:
            ax.fill_between(x[good_err], y[good_err] - e[good_err], y[good_err] + e[good_err],
                            color=st["fill"], alpha=0.30, lw=0)
        if np.count_nonzero(good) >= 2:
            order = np.argsort(x[good])
            ax.plot(x[good][order], y[good][order], "--", lw=1.5, color=st["line"], alpha=0.85)
        ax.plot(x[good], y[good], ls="none", marker=st["marker"], ms=7.5,
                color=st["line"], label=label)

    ax.axhline(0.0, color="0.25", lw=1.2)
    ax.set_xlabel(r"redshift $z$")
    ax.set_ylabel(r"$(N_{\rm L}-N_{\rm R})/(N_{\rm L}+N_{\rm R})$ [%]")
    ax.grid(alpha=0.22)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=3,
              frameon=False, fontsize=11.8, handlelength=1.9, columnspacing=1.0)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.88])
    out = outdir / f"three_catalog_sign_asymmetry_arms.{fmt}"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print("[write]", out)
    print("[write]", csv)


def plot_one_catalog_rect(ax, df: pd.DataFrame, *, ycol: str, ecol: str, angle: bool, key: str, label: str) -> Optional[pd.DataFrame]:
    st = STYLE[key]
    x = df["z_center"].to_numpy(float)
    zlo = df["z_min"].to_numpy(float) if "z_min" in df.columns else x
    zhi = df["z_max"].to_numpy(float) if "z_max" in df.columns else x
    y, lo, hi = y_and_band(df, ycol, ecol, angle=angle)
    good = np.isfinite(x) & np.isfinite(y)
    good_band = good & np.isfinite(lo) & np.isfinite(hi) & np.isfinite(zlo) & np.isfinite(zhi)

    for xx0, xx1, yy0, yy1 in zip(zlo[good_band], zhi[good_band], lo[good_band], hi[good_band]):
        ax.fill_between([xx0, xx1], [yy0, yy0], [yy1, yy1],
                        color=st["fill"], alpha=0.30 if key != "sconce" else 0.38,
                        lw=0, zorder=st["z"])
        ax.plot([xx0, xx1], [yy0, yy0], color=st["line"], lw=1.0, alpha=0.75, zorder=st["z"]+0.1)
        ax.plot([xx0, xx1], [yy1, yy1], color=st["line"], lw=1.0, alpha=0.75, zorder=st["z"]+0.1)

    if np.count_nonzero(good) >= 2:
        order = np.argsort(x[good])
        ax.plot(x[good][order], y[good][order], ls=_catalog_linestyle(key), lw=1.4, color=st["line"], alpha=0.78, zorder=st["z"]+0.8)
    ax.plot(x[good], y[good], ls="none", marker=st["marker"], ms=st["ms"],
            color=st["line"], markerfacecolor=st["line"], markeredgecolor=st["line"],
            markeredgewidth=1.0, label=label, zorder=st["z"]+1.0)
    return pd.DataFrame({
        "catalog": label, "catalog_key": key,
        "z_min": zlo, "z_max": zhi, "z_center": x,
        "y": y, "y_low": lo, "y_high": hi, "raw_column": ycol,
        "angle_version": bool(angle),
        "source_file": df["__source_file"].iloc[0] if "__source_file" in df.columns and len(df) else "",
    })



GLOBAL_H_LABEL_CACHE: dict[tuple[str, str], str] = {}


def _short_catalog_label_public(label: str) -> str:
    if "Carr" in str(label):
        return "Carrón"
    if "Chen" in str(label):
        return "Chen"
    if "Zhang" in str(label) or "SCONCE" in str(label):
        return "Zhang"
    return str(label)


def _math_pm_compact(val: float, ep: float, em: float, *, nd: int) -> str:
    return rf"{{{val:.{nd}f}}}^{{+{ep:.{nd}f}}}_{{-{em:.{nd}f}}}"


def _infer_measurement_table_and_weight(summary_df: pd.DataFrame) -> tuple[Optional[Path], str]:
    if "__source_file" not in summary_df.columns or len(summary_df) == 0:
        return None, "uniform"
    src = Path(str(summary_df["__source_file"].iloc[0]))
    name = src.name.lower()
    parent = src.parent
    if "arms" in name:
        table = parent / "arm_level_spirality.csv"
        # Chen arm-level summaries are written in chen_dr12_native_arms, while
        # the actual arm-level table lives in chen_dr12_native.
        if not table.exists() and parent.name.endswith("_arms"):
            candidate = parent.parent / parent.name.replace("_arms", "") / "arm_level_spirality.csv"
            if candidate.exists():
                table = candidate
        if not table.exists():
            candidate = parent.parent / "chen_dr12_native" / "arm_level_spirality.csv"
            if candidate.exists():
                table = candidate
    else:
        table = parent / "node_spirality.csv"
    weight_mode = "n_segments" if "n_segments" in name else ("mean_density" if "mean_density" in name else "uniform")
    return table, weight_mode


def _global_h_bootstrap_label(summary_df: pd.DataFrame, *, B: int = 1000, seed: int = 2027) -> str:
    """Compact global H label for the non-extrinsic 2x2 panels.

    It recomputes the unbinned H_S from the node/arm table used to create the
    z-bin summary.  For arms, resampling is clustered by node_id when available.
    """
    key = str(summary_df["catalog_key"].iloc[0]) if "catalog_key" in summary_df.columns and len(summary_df) else "unknown"
    label = str(summary_df["catalog_label"].iloc[0]) if "catalog_label" in summary_df.columns and len(summary_df) else key
    tab_path, weight_mode = _infer_measurement_table_and_weight(summary_df)
    cache_key = (str(tab_path), weight_mode)
    if cache_key in GLOBAL_H_LABEL_CACHE:
        return GLOBAL_H_LABEL_CACHE[cache_key]

    short = _short_catalog_label_public(label)
    if tab_path is None or not tab_path.exists():
        out = f"{short}: $H_S$ unavailable"
        GLOBAL_H_LABEL_CACHE[cache_key] = out
        return out

    d = pd.read_csv(tab_path)
    if "status" in d.columns:
        d = d[d["status"].astype(str).eq("ok")].copy()
    if "S_tangential" not in d.columns:
        out = f"{short}: $H_S$ unavailable"
        GLOBAL_H_LABEL_CACHE[cache_key] = out
        return out
    s = pd.to_numeric(d["S_tangential"], errors="coerce").to_numpy(float)
    w = summary_weights(d, weight_mode)
    ok = np.isfinite(s) & np.isfinite(w) & (w > 0)
    d = d.loc[ok].copy()
    s = s[ok]
    w = w[ok]
    if len(s) < 2 or np.sum(w) <= 0:
        out = f"{short}: $H_S$ unavailable"
        GLOBAL_H_LABEL_CACHE[cache_key] = out
        return out

    obs = float(np.sum(w * s) / np.sum(w))
    rng = np.random.default_rng(seed + (17 if key == "carron" else 29 if key == "chen" else 41))

    boots = np.empty(int(B), dtype=float)
    if "node_id" in d.columns and d["node_id"].nunique() > 1:
        ids = pd.unique(d["node_id"])
        idarr = d["node_id"].to_numpy()
        groups = {cid: np.flatnonzero(idarr == cid) for cid in ids}
        for b in range(int(B)):
            sample_ids = rng.choice(ids, size=len(ids), replace=True)
            idx = np.concatenate([groups[cid] for cid in sample_ids])
            boots[b] = float(np.sum(w[idx] * s[idx]) / np.sum(w[idx]))
    else:
        n = len(s)
        for b in range(int(B)):
            idx = rng.integers(0, n, size=n)
            boots[b] = float(np.sum(w[idx] * s[idx]) / np.sum(w[idx]))

    q16, q84 = np.nanpercentile(boots, [16.0, 84.0])
    sigma = float(np.nanstd(boots, ddof=1))
    z = obs / sigma if np.isfinite(sigma) and sigma > 0 else np.nan
    em = abs(obs - float(q16))
    ep = abs(float(q84) - obs)
    out = rf"{short}: $H_S={_math_pm_compact(obs, ep, em, nd=3)}$"
    GLOBAL_H_LABEL_CACHE[cache_key] = out
    return out



def _weighted_neff(weights: np.ndarray) -> float:
    w = np.asarray(weights, dtype=float)
    ok = np.isfinite(w) & (w > 0)
    if not np.any(ok):
        return 0.0
    sw = float(np.sum(w[ok]))
    sw2 = float(np.sum(w[ok] ** 2))
    return sw * sw / sw2 if sw2 > 0 else 0.0


def _sign_noise_percent_from_row(df: pd.DataFrame) -> np.ndarray:
    if "sign_noise_percent" in df.columns:
        return pd.to_numeric(df["sign_noise_percent"], errors="coerce").to_numpy(float)
    if "n_sign_total" in df.columns:
        n = pd.to_numeric(df["n_sign_total"], errors="coerce").to_numpy(float)
    elif "n_total" in df.columns:
        n = pd.to_numeric(df["n_total"], errors="coerce").to_numpy(float)
    elif "n_arms" in df.columns:
        n = pd.to_numeric(df["n_arms"], errors="coerce").to_numpy(float)
    elif "n_nodes_ok" in df.columns:
        n = pd.to_numeric(df["n_nodes_ok"], errors="coerce").to_numpy(float)
    else:
        return np.full(len(df), np.nan)
    return np.where(np.isfinite(n) & (n > 0), 100.0 / np.sqrt(n), np.nan)


def _global_sign_asymmetry_label(summary_df: pd.DataFrame, *, B: int = 1000, seed: int = 4027) -> str:
    """Unbinned sign-asymmetry label with clustered-bootstrap error and sign noise."""
    key = str(summary_df["catalog_key"].iloc[0]) if "catalog_key" in summary_df.columns and len(summary_df) else "unknown"
    label = str(summary_df["catalog_label"].iloc[0]) if "catalog_label" in summary_df.columns and len(summary_df) else key
    tab_path, weight_mode = _infer_measurement_table_and_weight(summary_df)
    cache_key = (str(tab_path), weight_mode, "sign")
    if cache_key in GLOBAL_H_LABEL_CACHE:
        return GLOBAL_H_LABEL_CACHE[cache_key]
    short = _short_catalog_label_public(label)

    if tab_path is None or not tab_path.exists():
        out = f"{short}: sign asym. unavailable"
        GLOBAL_H_LABEL_CACHE[cache_key] = out
        return out

    d = normalise_measurement_columns(pd.read_csv(tab_path))
    if "status" in d.columns:
        st = d["status"].astype(str).str.lower()
        keep = st.eq("ok")
        d = d.loc[keep].copy()
    if "S_tangential" not in d.columns:
        out = f"{short}: sign asym. unavailable"
        GLOBAL_H_LABEL_CACHE[cache_key] = out
        return out

    s = pd.to_numeric(d["S_tangential"], errors="coerce").to_numpy(float)
    sign_ok = np.isfinite(s) & (s != 0)
    d = d.loc[sign_ok].copy().reset_index(drop=True)
    signs = np.where(s[sign_ok] > 0, 1.0, -1.0)
    n = len(signs)
    if n < 2:
        out = f"{short}: sign asym. unavailable"
        GLOBAL_H_LABEL_CACHE[cache_key] = out
        return out

    # Sign asymmetry is intentionally unweighted/count-based.
    obs = float(np.mean(signs))
    sign_noise = 100.0 / np.sqrt(float(n))

    rng = np.random.default_rng(seed + (17 if key == "carron" else 29 if key == "chen" else 41))
    boots = np.empty(int(B), dtype=float)
    if "node_id" in d.columns and d["node_id"].nunique() > 1:
        ids = pd.unique(d["node_id"])
        idarr = d["node_id"].to_numpy()
        groups = {cid: np.flatnonzero(idarr == cid) for cid in ids}
        for b in range(int(B)):
            sample_ids = rng.choice(ids, size=len(ids), replace=True)
            idx = np.concatenate([groups[cid] for cid in sample_ids])
            boots[b] = float(np.mean(signs[idx]))
    else:
        for b in range(int(B)):
            idx = rng.integers(0, n, size=n)
            boots[b] = float(np.mean(signs[idx]))

    q16, q84 = np.nanpercentile(boots, [16.0, 84.0])
    em = 100.0 * abs(obs - float(q16))
    ep = 100.0 * abs(float(q84) - obs)
    out = rf"{short}: $A={{{100.0*obs:.2f}}}^{{+{ep:.2f}}}_{{-{em:.2f}}}\%$"
    GLOBAL_H_LABEL_CACHE[cache_key] = out
    return out


def _global_h_noise_from_table(summary_df: pd.DataFrame) -> float:
    """Independent-sign shot noise for H_S: sqrt(sum w_i^2 S_i^2)/sum w_i."""
    tab_path, weight_mode = _infer_measurement_table_and_weight(summary_df)
    if tab_path is None or not tab_path.exists():
        return np.nan
    d = normalise_measurement_columns(pd.read_csv(tab_path))
    if "status" in d.columns:
        st = d["status"].astype(str).str.lower()
        keep = st.eq("ok")
        d = d.loc[keep].copy()
    if "S_tangential" not in d.columns:
        return np.nan
    s = pd.to_numeric(d["S_tangential"], errors="coerce").to_numpy(float)
    w = summary_weights(d, weight_mode)
    ok = np.isfinite(s) & np.isfinite(w) & (w > 0)
    if not np.any(ok):
        return np.nan
    return float(np.sqrt(np.sum((w[ok] ** 2) * (s[ok] ** 2))) / np.sum(w[ok]))


def _add_global_sign_label_to_axis(ax, datasets: List[pd.DataFrame]) -> None:
    lines = []
    for d in datasets:
        try:
            lines.append(_global_sign_asymmetry_label(d))
        except Exception as exc:
            label = str(d["catalog_label"].iloc[0]) if "catalog_label" in d.columns and len(d) else "catalog"
            lines.append(f"{_short_catalog_label_public(label)}: sign label failed")
            print(f"[warn] global sign label failed for {label}: {exc!r}")
    if not lines:
        return
    ax.text(0.015, 0.84, "\n".join(lines), transform=ax.transAxes,
            va="top", ha="left", fontsize=9.3,
            bbox=dict(boxstyle="round,pad=0.16", fc="white", ec="none", alpha=0.78),
            zorder=20)


def _plot_sign_noise_background_pchip(
    ax,
    df: pd.DataFrame,
    *,
    key: str,
    extend_band_to_bin_edges: bool,
) -> Optional[pd.DataFrame]:
    """Draw +/- sign-noise background curves on sign-asymmetry panels."""
    x = df["z_center"].to_numpy(float)
    zlo = df["z_min"].to_numpy(float) if "z_min" in df.columns else x
    zhi = df["z_max"].to_numpy(float) if "z_max" in df.columns else x
    noise = _sign_noise_percent_from_row(df)
    good = np.isfinite(x) & np.isfinite(noise)
    if np.count_nonzero(good) < 2:
        return None
    xline = x[good].copy()
    yline = noise[good].copy()
    if extend_band_to_bin_edges:
        zl = zlo[good].copy()
        zh = zhi[good].copy()
        order = np.argsort(xline)
        xline, yline, zl, zh = xline[order], yline[order], zl[order], zh[order]
        if np.isfinite(zl[0]) and zl[0] < xline[0]:
            xline = np.insert(xline, 0, zl[0])
            yline = np.insert(yline, 0, yline[0])
        if np.isfinite(zh[-1]) and zh[-1] > xline[-1]:
            xline = np.append(xline, zh[-1])
            yline = np.append(yline, yline[-1])
    xs, ys = _pchip_interp_curve(xline, yline)
    ls = _catalog_linestyle(key)
    # Show symmetric 1-sigma sign-noise levels.
    ax.plot(xs, ys, ls=ls, lw=1.4, color="black", alpha=0.55, zorder=-5)
    ax.plot(xs, -ys, ls=ls, lw=1.4, color="black", alpha=0.55, zorder=-5)
    return pd.DataFrame({
        "catalog_key": key,
        "z_min": zlo,
        "z_max": zhi,
        "z_center": x,
        "sign_noise_percent": noise,
    })


def _H_noise_from_row(df: pd.DataFrame, *, angle: bool) -> np.ndarray:
    if "H_S_sign_noise_angle_deg" in df.columns and angle:
        y = pd.to_numeric(df["H_S_sign_noise_angle_deg"], errors="coerce").to_numpy(float)
        if np.any(np.isfinite(y)):
            return y
    if "H_S_sign_noise" in df.columns:
        y = pd.to_numeric(df["H_S_sign_noise"], errors="coerce").to_numpy(float)
    else:
        return np.full(len(df), np.nan)
    return asin_deg_clip(y) if angle else y


def _plot_H_noise_background_pchip(
    ax,
    df: pd.DataFrame,
    *,
    key: str,
    angle: bool,
    extend_band_to_bin_edges: bool,
) -> Optional[pd.DataFrame]:
    """Draw +/- independent-sign H_S noise curves on handedness panels."""
    x = df["z_center"].to_numpy(float)
    zlo = df["z_min"].to_numpy(float) if "z_min" in df.columns else x
    zhi = df["z_max"].to_numpy(float) if "z_max" in df.columns else x
    noise = _H_noise_from_row(df, angle=angle)
    good = np.isfinite(x) & np.isfinite(noise)
    if np.count_nonzero(good) < 2:
        return None
    xline = x[good].copy()
    yline = noise[good].copy()
    if extend_band_to_bin_edges:
        zl = zlo[good].copy()
        zh = zhi[good].copy()
        order = np.argsort(xline)
        xline, yline, zl, zh = xline[order], yline[order], zl[order], zh[order]
        if np.isfinite(zl[0]) and zl[0] < xline[0]:
            xline = np.insert(xline, 0, zl[0])
            yline = np.insert(yline, 0, yline[0])
        if np.isfinite(zh[-1]) and zh[-1] > xline[-1]:
            xline = np.append(xline, zh[-1])
            yline = np.append(yline, yline[-1])
    xs, ys = _pchip_interp_curve(xline, yline)
    ls = _catalog_linestyle(key)
    ax.plot(xs, ys, ls=ls, lw=1.4, color="black", alpha=0.55, zorder=-5)
    ax.plot(xs, -ys, ls=ls, lw=1.4, color="black", alpha=0.55, zorder=-5)
    return pd.DataFrame({
        "catalog_key": key,
        "z_min": zlo,
        "z_max": zhi,
        "z_center": x,
        "H_S_sign_noise_plot": noise,
        "angle_version": bool(angle),
    })




# -----------------------------------------------------------------------------
# Block sign-flip nulls for projected/standard panels
# -----------------------------------------------------------------------------

GLOBAL_BLOCK_SIGNFLIP_NULL_CACHE: dict[tuple[str, str, str, int, int], pd.DataFrame] = {}
DEFAULT_STANDARD_SIGNFLIP_B = 1000
DEFAULT_STANDARD_SIGNFLIP_BLOCK_NSIDE = 16


def _catalog_seed_offset(key: str) -> int:
    key = str(key).lower()
    if key == "carron":
        return 101
    if key == "chen":
        return 211
    if key == "sconce":
        return 307
    return 409


def _plot_extend_series_to_edges(x: np.ndarray, y: np.ndarray, zlo: np.ndarray, zhi: np.ndarray):
    x = np.asarray(x, dtype=float).copy()
    y = np.asarray(y, dtype=float).copy()
    zlo = np.asarray(zlo, dtype=float).copy()
    zhi = np.asarray(zhi, dtype=float).copy()
    ok = np.isfinite(x) & np.isfinite(y)
    if not np.any(ok):
        return x[ok], y[ok]
    x, y, zlo, zhi = x[ok], y[ok], zlo[ok], zhi[ok]
    order = np.argsort(x)
    x, y, zlo, zhi = x[order], y[order], zlo[order], zhi[order]
    if np.isfinite(zlo[0]) and zlo[0] < x[0]:
        x = np.insert(x, 0, zlo[0])
        y = np.insert(y, 0, y[0])
    if np.isfinite(zhi[-1]) and zhi[-1] > x[-1]:
        x = np.append(x, zhi[-1])
        y = np.append(y, y[-1])
    return x, y


def _compute_block_signflip_null_zbins(
    summary_df: pd.DataFrame,
    *,
    statistic: str,
    B: int = DEFAULT_STANDARD_SIGNFLIP_B,
    block_nside: int = DEFAULT_STANDARD_SIGNFLIP_BLOCK_NSIDE,
    seed: int = 20260701,
) -> Optional[pd.DataFrame]:
    """Block sign-flip null for projected Handedness or Sign Asymmetry.

    This preserves the observed arm/node positions, weights, redshifts and
    amplitudes, while randomizing the handedness coherently per HEALPix block.
    It is intended for the gray null bands in the projected/standard panels.
    """
    if summary_df is None or len(summary_df) == 0:
        return None
    key = str(summary_df["catalog_key"].iloc[0]) if "catalog_key" in summary_df.columns else "catalog"
    tab_path, weight_mode = _infer_measurement_table_and_weight(summary_df)
    if tab_path is None or not tab_path.exists():
        return None
    if weight_mode == "n_segments":
        from spirality.revision_plot_data import find_revision_output, binned_null_table
        revised = find_revision_output(tab_path, key)
        if revised is not None:
            return binned_null_table(revised, summary_df, statistic=statistic, catalog=key)
    cache_key = (str(tab_path), str(weight_mode), str(statistic), int(B), int(block_nside))
    if cache_key in GLOBAL_BLOCK_SIGNFLIP_NULL_CACHE:
        return GLOBAL_BLOCK_SIGNFLIP_NULL_CACHE[cache_key].copy()

    d = normalise_measurement_columns(pd.read_csv(tab_path))
    if "status" in d.columns:
        st = d["status"].astype(str).str.lower()
        keep = st.eq("ok")
        d = d.loc[keep].copy()
    if "S_tangential" not in d.columns or len(d) == 0:
        return None

    h = pd.to_numeric(d["S_tangential"], errors="coerce").to_numpy(float)
    z = robust_z_center(d)
    w = summary_weights(d, weight_mode)
    blocks = assign_healpix_blocks(d, int(block_nside))
    ok = np.isfinite(h) & np.isfinite(z) & np.isfinite(w) & (w > 0)
    if str(statistic).lower().startswith("sign"):
        ok &= (h != 0.0)
    if not np.any(ok):
        return None
    h, z, w, blocks = h[ok], z[ok], w[ok], blocks[ok]

    rng = np.random.default_rng(int(seed) + _catalog_seed_offset(key))
    rows = []
    for _, sr in summary_df.iterrows():
        z0 = float(sr["z_min"])
        z1 = float(sr["z_max"])
        ib = int(sr["zbin_id"]) if "zbin_id" in summary_df.columns and np.isfinite(sr.get("zbin_id", np.nan)) else len(rows)
        # Shared boundaries belong to one bin only; the final upper edge is included.
        last_edge = float(pd.to_numeric(summary_df["z_max"], errors="coerce").max())
        m = (z >= z0) & ((z <= z1) if z1 == last_edge else (z < z1))
        if not np.any(m):
            vals = np.full(int(B), np.nan)
        else:
            hb = h[m]
            wb = w[m]
            pb = blocks[m]
            ub, inv = np.unique(pb, return_inverse=True)
            vals = np.empty(int(B), dtype=float)
            for b in range(int(B)):
                eps = rng.choice(np.array([-1.0, 1.0]), size=len(ub), replace=True)
                hh = hb * eps[inv]
                if str(statistic).lower().startswith("sign"):
                    sgn = np.sign(hh)
                    good = np.isfinite(sgn) & (sgn != 0.0)
                    vals[b] = 100.0 * float(np.mean(sgn[good])) if np.any(good) else np.nan
                else:
                    vals[b] = weighted_mean_np(hh, wb)
        q = np.nanpercentile(vals, [2.2750131948, 15.8655253931, 50.0, 84.1344746069, 97.7249868052]) if np.any(np.isfinite(vals)) else [np.nan]*5
        rows.append({
            "catalog_key": key,
            "zbin_id": ib,
            "z_min": z0,
            "z_max": z1,
            "z_center": float(sr["z_center"]),
            "statistic": str(statistic),
            "B": int(B),
            "block_nside": int(block_nside),
            "null_q025": float(q[0]),
            "null_q16": float(q[1]),
            "null_q50": float(q[2]),
            "null_q84": float(q[3]),
            "null_q975": float(q[4]),
        })
    out = pd.DataFrame(rows)
    GLOBAL_BLOCK_SIGNFLIP_NULL_CACHE[cache_key] = out.copy()
    return out


def _plot_block_signflip_null_background_pchip(
    ax,
    df: pd.DataFrame,
    *,
    key: str,
    statistic: str,
    angle: bool,
    extend_band_to_bin_edges: bool,
    cached_null: Optional[pd.DataFrame] = None,
) -> Optional[pd.DataFrame]:
    # Publication-only refresh supplies the already calibrated null table.
    # No sign flips are generated when cached_null is provided.
    null = cached_null.copy() if cached_null is not None else _compute_block_signflip_null_zbins(df, statistic=statistic)
    if null is None or len(null) == 0:
        return None
    x = null["z_center"].to_numpy(float)
    zlo = null["z_min"].to_numpy(float)
    zhi = null["z_max"].to_numpy(float)
    q025 = null["null_q025"].to_numpy(float)
    q16 = null["null_q16"].to_numpy(float)
    q50 = null["null_q50"].to_numpy(float)
    q84 = null["null_q84"].to_numpy(float)
    q975 = null["null_q975"].to_numpy(float)
    if angle and not str(statistic).lower().startswith("sign"):
        q025, q16, q50, q84, q975 = map(asin_deg_clip, [q025, q16, q50, q84, q975])
    good = np.isfinite(x) & np.isfinite(q16) & np.isfinite(q84)
    if np.count_nonzero(good) < 2:
        return null

    # Draw Gaussian-equivalent 1σ and 2σ block sign-flip null intervals.
    # The line style identifies the catalogue (Carrón dashed, Chen dotted);
    # darker/thicker 2σ curves match the extrinsic look-elsewhere style.
    x68_lo, y68_lo = _plot_extend_series_to_edges(x[good], q16[good], zlo[good], zhi[good]) if extend_band_to_bin_edges else (x[good], q16[good])
    x68_hi, y68_hi = _plot_extend_series_to_edges(x[good], q84[good], zlo[good], zhi[good]) if extend_band_to_bin_edges else (x[good], q84[good])
    x95_lo, y95_lo = _plot_extend_series_to_edges(x[good], q025[good], zlo[good], zhi[good]) if extend_band_to_bin_edges else (x[good], q025[good])
    x95_hi, y95_hi = _plot_extend_series_to_edges(x[good], q975[good], zlo[good], zhi[good]) if extend_band_to_bin_edges else (x[good], q975[good])

    ls = _catalog_linestyle(key)
    for xx, yy in [(x68_lo, y68_lo), (x68_hi, y68_hi)]:
        xs, ys = _pchip_interp_curve(xx, yy)
        if len(xs):
            ax.plot(xs, ys, ls=ls, color="0.25", lw=1.30, alpha=0.72, zorder=-6)
    for xx, yy in [(x95_lo, y95_lo), (x95_hi, y95_hi)]:
        xs, ys = _pchip_interp_curve(xx, yy)
        if len(xs):
            ax.plot(xs, ys, ls=ls, color="0.08", lw=1.85, alpha=0.84, zorder=-7)

    out = null.copy()
    out["angle_version"] = bool(angle)
    return out

def _add_global_h_label_to_axis(ax, datasets: List[pd.DataFrame]) -> None:
    lines = []
    for d in datasets:
        try:
            lines.append(_global_h_bootstrap_label(d))
        except Exception as exc:
            label = str(d["catalog_label"].iloc[0]) if "catalog_label" in d.columns and len(d) else "catalog"
            lines.append(f"{_short_catalog_label_public(label)}: $H_S$ label failed")
            print(f"[warn] global H label failed for {label}: {exc!r}")
    if not lines:
        return
    ax.text(0.015, 0.82, "\n".join(lines), transform=ax.transAxes,
            va="top", ha="left", fontsize=9.6,
            bbox=dict(boxstyle="round,pad=0.16", fc="white", ec="none", alpha=0.78),
            zorder=20)



def _add_top_legend(fig, axes):
    """Add a robust, compact legend above a panel figure.

    This helper was used by the pchip panels; keeping it as a standalone
    function lets both the rectangular and pchip 2x2 panels share the same
    external legend logic.
    """
    try:
        flat_axes = np.ravel(axes)
    except Exception:
        flat_axes = [axes]
    handles = []
    labels = []
    for ax in flat_axes:
        h, l = ax.get_legend_handles_labels()
        for hh, ll in zip(h, l):
            if ll and ll not in labels:
                handles.append(hh)
                labels.append(ll)
    if not handles:
        return None
    ncol = min(max(len(labels), 1), 4)
    leg = fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=ncol,
        frameon=False,
        fontsize=13.0,
        handlelength=1.8,
        columnspacing=1.2,
        borderaxespad=0.0,
    )
    return leg



def _set_exact_redshift_xlim(axes, datasets: List[pd.DataFrame]) -> None:
    vals_lo, vals_hi = [], []
    for d in datasets:
        if "z_min" in d.columns:
            vals_lo.extend(pd.to_numeric(d["z_min"], errors="coerce").dropna().tolist())
        if "z_max" in d.columns:
            vals_hi.extend(pd.to_numeric(d["z_max"], errors="coerce").dropna().tolist())
    if vals_lo and vals_hi:
        x0, x1 = float(np.nanmin(vals_lo)), float(np.nanmax(vals_hi))
        for ax in np.ravel(axes):
            ax.set_xlim(x0, x1)


def plot_joint_panel_rect(datasets: List[pd.DataFrame], out: Path, *, angle_version: bool) -> None:
    """Paper-facing projected panel: Handedness, Sign Asymmetry, Spirality."""
    setup_matplotlib()
    panels = [
        ("H_S_weighted_mean", "H_S_bootstrap_err",
         r"$\theta_{H_S}=\sin^{-1}H_S$ [deg]" if angle_version else r"$H_S=\langle S_\phi\rangle$",
         True, angle_version, "Handedness"),
        ("sign_asymmetry_percent", "sign_asymmetry_percent_err",
         r"$(N_{\rm L}-N_{\rm R})/(N_{\rm L}+N_{\rm R})$ [%]",
         True, False, "Left/right Sign Asymmetry"),
        ("mean_abs_projected_twist", "mean_abs_projected_twist_bootstrap_err",
         r"$\sin^{-1}\langle |S_\phi|\rangle$ [deg]" if angle_version else r"$\langle |S_\phi|\rangle$",
         False, angle_version, "Spirality"),
    ]
    fig, axes = plt.subplots(3, 1, figsize=(8.6, 11.2), sharex=True)
    source_rows = []
    for pi, (ax, (ycol, ecol, ylabel, zline, make_angle, panel_name)) in enumerate(zip(axes, panels)):
        for d in datasets:
            key = str(d["catalog_key"].iloc[0])
            label = str(d["catalog_label"].iloc[0])
            if ycol not in d.columns:
                print(f"[warn] {label} lacks {ycol}; skipping {panel_name}")
                continue
            r = plot_one_catalog_rect(ax, d, ycol=ycol, ecol=ecol, angle=make_angle, key=key, label=label)
            if r is not None:
                r.insert(0, "panel", panel_name)
                source_rows.append(r)
        if zline:
            ax.axhline(0.0, color="0.25", lw=1.2, ls="-", zorder=0)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.20)
        ax.tick_params(labelsize=17, length=5.0, width=1.0)
        ax.text(0.50, 0.955, panel_name, transform=ax.transAxes,
                ha="center", va="top", fontsize=19, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.16", fc="white", ec="none", alpha=0.72))
        _joint_panel_common_format(ax, panel_name, angle_version=angle_version, pi=pi, product_kind=_panel_product_kind_from_path(out))
        if panel_name == "Handedness":
            _add_global_h_label_to_axis(ax, datasets)
        if panel_name == "Sign Asymmetry":
            _add_global_sign_label_to_axis(ax, datasets)
    axes[-1].set_xlabel(r"redshift $z$")
    _set_exact_redshift_xlim(axes, datasets)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.92])
    leg = _add_top_legend(fig, axes)
    out.parent.mkdir(parents=True, exist_ok=True)
    save_kwargs = {"bbox_inches": "tight", "pad_inches": 0.05}
    if leg is not None:
        save_kwargs["bbox_extra_artists"] = [leg]
    fig.savefig(out, **save_kwargs)
    plt.close(fig)
    if source_rows:
        pd.concat(source_rows, ignore_index=True).to_csv(out.with_suffix(".csv"), index=False)
        print("[write]", out.with_suffix(".csv"))
    print("[write]", out)


def plot_joint_panel(datasets: List[pd.DataFrame], out: Path, *, angle_version: bool) -> None:
    """Backward-compatible wrapper for the rectangular 2x2 panel plot."""
    return plot_joint_panel_rect(datasets, out, angle_version=angle_version)


def _panel_product_kind_from_path(out: Path) -> str:
    name = str(out.name).lower()
    if "nodes" in name:
        return "nodes"
    if "arms" in name:
        return "arms"
    return "unknown"


def _apply_panel_axis_limits(ax, panel_name: str, *, angle_version: bool, product_kind: str) -> None:
    """Autoscale revised data, including error bars; never clip to v87 results."""
    ax.autoscale(enable=True, axis="y")
    ax.margins(y=0.10)




def _joint_panel_common_format(ax, panel_name: str, *, angle_version: bool, pi: int, product_kind: str) -> None:
    _apply_panel_axis_limits(ax, panel_name, angle_version=angle_version, product_kind=product_kind)
    if panel_name in {"Handedness", "Sign Asymmetry"}:
        ax.text(0.50, 0.84, "Left-handed", transform=ax.transAxes,
                ha="center", va="center", fontsize=17, color="0.35", alpha=0.82)
        ax.text(0.50, 0.16, "Right-handed", transform=ax.transAxes,
                ha="center", va="center", fontsize=17, color="0.35", alpha=0.82)



def _pchip_interp_curve(x: np.ndarray, y: np.ndarray, *, n: int = 256) -> tuple[np.ndarray, np.ndarray]:
    """Monotone PCHIP interpolation when scipy is available; linear fallback."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    x = x[ok]
    y = y[ok]
    if len(x) < 2:
        return x, y
    order = np.argsort(x)
    x = x[order]
    y = y[order]
    # Drop duplicate x values to keep interpolators happy.
    xu, idx = np.unique(x, return_index=True)
    yu = y[idx]
    if len(xu) < 2:
        return xu, yu
    xx = np.linspace(float(np.nanmin(xu)), float(np.nanmax(xu)), int(n))
    try:
        from scipy.interpolate import PchipInterpolator
        yy = PchipInterpolator(xu, yu, extrapolate=False)(xx)
    except Exception:
        yy = np.interp(xx, xu, yu)
    return xx, yy


def plot_one_catalog_pchip(
    ax,
    df: pd.DataFrame,
    *,
    ycol: str,
    ecol: str,
    angle: bool,
    key: str,
    label: str,
    extend_band_to_bin_edges: bool = False,
) -> Optional[pd.DataFrame]:
    """Plot one catalogue with PCHIP central curve and +/- error envelope."""
    st = STYLE[key]
    x = df["z_center"].to_numpy(float)
    zlo = df["z_min"].to_numpy(float) if "z_min" in df.columns else x
    zhi = df["z_max"].to_numpy(float) if "z_max" in df.columns else x
    y, lo, hi = y_and_band(df, ycol, ecol, angle=angle)
    good = np.isfinite(x) & np.isfinite(y)
    good_band = good & np.isfinite(lo) & np.isfinite(hi) & np.isfinite(zlo) & np.isfinite(zhi)

    # Error envelope.  When requested, extend the envelope horizontally to the
    # first/last bin edge, while drawing the central dashed curve only between
    # the first and last marker.
    if np.count_nonzero(good_band) >= 2:
        xb = x[good_band].copy()
        lob = lo[good_band].copy()
        hib = hi[good_band].copy()
        zlob = zlo[good_band].copy()
        zhib = zhi[good_band].copy()
        order = np.argsort(xb)
        xb, lob, hib, zlob, zhib = xb[order], lob[order], hib[order], zlob[order], zhib[order]
        if extend_band_to_bin_edges:
            if np.isfinite(zlob[0]) and zlob[0] < xb[0]:
                xb = np.insert(xb, 0, zlob[0])
                lob = np.insert(lob, 0, lob[0])
                hib = np.insert(hib, 0, hib[0])
            if np.isfinite(zhib[-1]) and zhib[-1] > xb[-1]:
                xb = np.append(xb, zhib[-1])
                lob = np.append(lob, lob[-1])
                hib = np.append(hib, hib[-1])
        xs_lo, ys_lo = _pchip_interp_curve(xb, lob)
        xs_hi, ys_hi = _pchip_interp_curve(xb, hib)
        if len(xs_lo) and len(xs_hi) and len(xs_lo) == len(xs_hi):
            ax.fill_between(xs_lo, ys_lo, ys_hi, color=st["fill"], alpha=0.34 if key != "sconce" else 0.42,
                            lw=0, zorder=st["z"])
            ax.plot(xs_lo, ys_lo, color=st["line"], lw=1.05, alpha=0.78, zorder=st["z"]+0.1)
            ax.plot(xs_hi, ys_hi, color=st["line"], lw=1.05, alpha=0.78, zorder=st["z"]+0.1)

        # Vertical error bars at marker positions and horizontal bin-width ticks.
        for xx, yy0, yy1, zz0, zz1, yy in zip(x[good_band], lo[good_band], hi[good_band], zlo[good_band], zhi[good_band], y[good_band]):
            ax.plot([xx, xx], [yy0, yy1], color=st["line"], lw=1.05, alpha=0.70, zorder=st["z"]+0.35)
            ax.plot([zz0, zz1], [yy, yy], color=st["line"], lw=1.0, alpha=0.58, zorder=st["z"]+0.55)

    # Central curve.
    if np.count_nonzero(good) >= 2:
        xline = x[good].copy()
        yline = y[good].copy()
        visible_min = float(np.nanmin(xline))
        visible_max = float(np.nanmax(xline))
        if extend_band_to_bin_edges:
            zl = zlo[good].copy()
            zh = zhi[good].copy()
            order = np.argsort(xline)
            xline, yline, zl, zh = xline[order], yline[order], zl[order], zh[order]
            if np.isfinite(zl[0]) and zl[0] < xline[0]:
                xline = np.insert(xline, 0, zl[0])
                yline = np.insert(yline, 0, yline[0])
            if np.isfinite(zh[-1]) and zh[-1] > xline[-1]:
                xline = np.append(xline, zh[-1])
                yline = np.append(yline, yline[-1])
        xs, ys = _pchip_interp_curve(xline, yline)
        if extend_band_to_bin_edges and len(xs):
            vis = (xs >= visible_min) & (xs <= visible_max)
            xs = xs[vis]
            ys = ys[vis]
        ax.plot(xs, ys, ls=_catalog_linestyle(key), lw=1.35, color=st["line"], alpha=0.82, zorder=st["z"]+0.75)

    ax.plot(x[good], y[good], ls="none", marker=st["marker"], ms=st["ms"],
            color=st["line"], markerfacecolor=st["line"], markeredgecolor=st["line"],
            markeredgewidth=1.0, label=label, zorder=st["z"]+1.0)

    return pd.DataFrame({
        "catalog": label, "catalog_key": key,
        "z_min": zlo, "z_max": zhi, "z_center": x,
        "y": y, "y_low": lo, "y_high": hi, "raw_column": ycol,
        "angle_version": bool(angle),
        "source_file": df["__source_file"].iloc[0] if "__source_file" in df.columns and len(df) else "",
    })



def plot_joint_panel_pchip(
    datasets: List[pd.DataFrame],
    out: Path,
    *,
    angle_version: bool,
    extend_band_to_bin_edges: bool = False,
) -> None:
    """Paper-facing projected panel: Handedness, Sign Asymmetry, Spirality."""
    setup_matplotlib()
    product_kind = _panel_product_kind_from_path(out)
    panels = [
        ("H_S_weighted_mean", "H_S_bootstrap_err",
         r"$\theta_{H_S}=\sin^{-1}H_S$ [deg]" if angle_version else r"$H_S=\langle S_\phi\rangle$",
         True, angle_version, "Handedness"),
        ("sign_asymmetry_percent", "sign_asymmetry_percent_err",
         r"$(N_{\rm L}-N_{\rm R})/(N_{\rm L}+N_{\rm R})$ [%]",
         True, False, "Left/right Sign Asymmetry"),
        ("mean_abs_projected_twist", "mean_abs_projected_twist_bootstrap_err",
         r"$\sin^{-1}\langle |S_\phi|\rangle$ [deg]" if angle_version else r"$\langle |S_\phi|\rangle$",
         False, angle_version, "Spirality"),
    ]
    fig, axes = plt.subplots(3, 1, figsize=(8.8, 11.4), sharex=True)
    source_rows = []
    for pi, (ax, (ycol, ecol, ylabel, zline, make_angle, panel_name)) in enumerate(zip(axes, panels)):
        if panel_name == "Handedness":
            for d in datasets:
                key = str(d["catalog_key"].iloc[0])
                rn = _plot_block_signflip_null_background_pchip(
                    ax, d, key=key, statistic="H", angle=make_angle,
                    extend_band_to_bin_edges=extend_band_to_bin_edges,
                )
                if rn is not None:
                    rn.insert(0, "panel", "Block Sign-flip Null: Handedness")
                    source_rows.append(rn)
        if panel_name == "Sign Asymmetry":
            for d in datasets:
                key = str(d["catalog_key"].iloc[0])
                rn = _plot_block_signflip_null_background_pchip(
                    ax, d, key=key, statistic="sign_asymmetry", angle=False,
                    extend_band_to_bin_edges=extend_band_to_bin_edges,
                )
                if rn is not None:
                    rn.insert(0, "panel", "Block Sign-flip Null: Sign Asymmetry")
                    source_rows.append(rn)
        for d in datasets:
            key = str(d["catalog_key"].iloc[0])
            label = str(d["catalog_label"].iloc[0])
            if ycol not in d.columns:
                print(f"[warn] {label} lacks {ycol}; skipping {panel_name}")
                continue
            r = plot_one_catalog_pchip(
                ax, d, ycol=ycol, ecol=ecol, angle=make_angle, key=key, label=label,
                extend_band_to_bin_edges=extend_band_to_bin_edges,
            )
            if r is not None:
                r.insert(0, "panel", panel_name)
                source_rows.append(r)
        if zline:
            ax.axhline(0.0, color="0.25", lw=1.2, ls="-", zorder=0)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.22)
        ax.tick_params(labelsize=17, length=5.0, width=1.0)
        ax.text(0.50, 0.955, panel_name, transform=ax.transAxes,
                ha="center", va="top", fontsize=19, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.16", fc="white", ec="none", alpha=0.72))
        _joint_panel_common_format(ax, panel_name, angle_version=angle_version, pi=pi, product_kind=product_kind)
        if panel_name == "Handedness":
            _add_global_h_label_to_axis(ax, datasets)
        if panel_name == "Sign Asymmetry":
            _add_global_sign_label_to_axis(ax, datasets)
    axes[-1].set_xlabel(r"redshift $z$")
    _set_exact_redshift_xlim(axes, datasets)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.92])
    leg = _add_top_legend(fig, axes)
    out.parent.mkdir(parents=True, exist_ok=True)
    save_kwargs = {"bbox_inches": "tight", "pad_inches": 0.05}
    if leg is not None:
        save_kwargs["bbox_extra_artists"] = [leg]
    fig.savefig(out, **save_kwargs)
    plt.close(fig)
    if source_rows:
        pd.concat(source_rows, ignore_index=True).to_csv(out.with_suffix(".used_values.csv"), index=False)
    print(f"[write] {out}")


def plot_arm_shape_panel_pchip(
    datasets: List[pd.DataFrame],
    out: Path,
    *,
    extend_band_to_bin_edges: bool = True,
) -> None:
    """Plot <|S_phi|>, arm straightness, and transverse ratio in one panel."""
    setup_matplotlib()
    panels = [
        ("mean_abs_projected_twist", "mean_abs_projected_twist_bootstrap_err",
         r"$\langle |S_\phi| \rangle$", "Spirality/Torsion"),
        ("mean_arm_straightness", "mean_arm_straightness_bootstrap_err",
         r"$\langle D_{\rm end}/L_{\rm path}\rangle$", "Straightness"),
        ("mean_arm_transverse_ratio", "mean_arm_transverse_ratio_bootstrap_err",
         r"$\langle R_\perp/D_{\rm end}\rangle$", "Transverse ratio"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(18.2, 5.6), sharex=False)
    source_rows = []

    for pi, (ax, (ycol, ecol, ylabel, panel_name)) in enumerate(zip(axes, panels)):
        for d in datasets:
            key = str(d["catalog_key"].iloc[0])
            label = str(d["catalog_label"].iloc[0])
            if ycol not in d.columns:
                print(f"[warn] {label} lacks {ycol}; skipping arm-shape panel {panel_name}")
                continue
            r = plot_one_catalog_pchip(
                ax, d, ycol=ycol, ecol=ecol, angle=False, key=key, label=label,
                extend_band_to_bin_edges=extend_band_to_bin_edges,
            )
            if r is not None:
                r.insert(0, "panel", panel_name)
                source_rows.append(r)
        ax.set_ylabel(ylabel)
        ax.set_xlabel(r"redshift $z$")
        ax.grid(alpha=0.22)
        ax.tick_params(labelsize=17, length=5.0, width=1.0)
        ax.text(0.50, 0.965, panel_name, transform=ax.transAxes,
                ha="center", va="top", fontsize=20, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.72))
        if panel_name == "Straightness":
            ax.set_ylim(0.0, 1.05)
        elif panel_name == "Transverse ratio":
            ax.set_ylim(bottom=0.0)

    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.88])
    leg = _add_top_legend(fig, axes)
    out.parent.mkdir(parents=True, exist_ok=True)
    save_kwargs = {"bbox_inches": "tight", "pad_inches": 0.05}
    if leg is not None:
        save_kwargs["bbox_extra_artists"] = [leg]
    fig.savefig(out, **save_kwargs)
    plt.close(fig)
    if source_rows:
        pd.concat(source_rows, ignore_index=True).to_csv(out.with_suffix(".used_values.csv"), index=False)
    print(f"[write] {out}")


def make_joint_plots(*, sconce_dir: Path, carron_dir: Path, chen_node_dir: Path, chen_arm_dir: Path,
                     outdir: Path, fmt: str, zmin: Optional[float], zmax: Optional[float], filter_mode: str) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    jobs = [
        ("nodes", "zbin_summary.csv", chen_node_dir, "three_catalog_nodes"),
        ("arms_n_segments", "zbin_summary_arms_n_segments.csv", chen_arm_dir, "three_catalog_arms_n_segments"),
    ]
    for _, fname, chen_dir, stem in jobs:
        datasets = []
        # Public 2x2 panels show only the two published filament catalogues.
        # SCONCE/Zhang is still computed and kept for internal consistency checks,
        # but it is not shown in the paper-facing panel plots.
        for key, ddir, label in [
            ("carron", carron_dir, "Carrón et al. SDSS DR16"),
            ("chen", chen_dir, "Chen et al. SDSS DR12"),
        ]:
            ds = read_summary_file(ddir / fname, key=key, label=label, zmin=zmin, zmax=zmax, mode=filter_mode)
            if ds is not None and len(ds):
                datasets.append(ds)
        if len(datasets) < 2:
            print(f"[skip] {stem}: fewer than two valid datasets")
            continue
        plot_joint_panel(datasets, outdir / f"{stem}.{fmt}", angle_version=False)
        plot_joint_panel(datasets, outdir / f"{stem}_angle.{fmt}", angle_version=True)
        plot_joint_panel_pchip(datasets, outdir / f"{stem}_pchiperr.{fmt}", angle_version=False)
        plot_joint_panel_pchip(datasets, outdir / f"{stem}_angle_pchiperr.{fmt}", angle_version=True)
        plot_joint_panel_pchip(datasets, outdir / f"{stem}_pchiperr_extband.{fmt}", angle_version=False, extend_band_to_bin_edges=True)
        plot_joint_panel_pchip(datasets, outdir / f"{stem}_angle_pchiperr_extband.{fmt}", angle_version=True, extend_band_to_bin_edges=True)
        if "arms" in stem:
            plot_arm_shape_panel_pchip(
                datasets,
                outdir / f"{stem}_shape_geometry_pchiperr_extband.{fmt}",
                extend_band_to_bin_edges=True,
            )



# -----------------------------------------------------------------------------
# Rigorous z-bin summaries: node bootstrap + HEALPix jackknife
# -----------------------------------------------------------------------------

def parse_edges_string(edge_string: str) -> np.ndarray:
    edges = np.array([float(x) for x in str(edge_string).replace(";", ",").split(",") if str(x).strip()], dtype=float)
    if len(edges) < 2 or not np.all(np.diff(edges) > 0):
        raise ValueError(f"Invalid z-bin edges: {edge_string}")
    return edges


def summary_weights(df: pd.DataFrame, mode: str) -> np.ndarray:
    """Weights used when averaging nodes/arms inside one pipeline.

    These weights are intentionally internal to a single pipeline.  Do not
    interpret e.g. Chen UM and Carrón unc as cross-calibrated absolute errors.
    """
    mode = str(mode).lower()
    n = len(df)
    if mode == "uniform":
        return np.ones(n, dtype=float)
    if mode == "mean_density":
        den = df["mean_segment_density"].to_numpy(float) if "mean_segment_density" in df.columns else np.ones(n, dtype=float)
        return np.where(np.isfinite(den) & (den > 0), den, 0.0)
    if mode == "sqrt_mean_density":
        den = df["mean_segment_density"].to_numpy(float) if "mean_segment_density" in df.columns else np.ones(n, dtype=float)
        return np.where(np.isfinite(den) & (den > 0), np.sqrt(den), 0.0)
    if mode == "n_segments":
        col = "n_arm_segments_total" if "n_arm_segments_total" in df.columns else "n_segments_valid"
        vals = df[col].to_numpy(float) if col in df.columns else np.ones(n, dtype=float)
        return np.where(np.isfinite(vals) & (vals > 0), vals, 0.0)
    if mode == "sqrt_points":
        col = "n_points_used_total" if "n_points_used_total" in df.columns else "n_points_total"
        vals = df[col].to_numpy(float) if col in df.columns else np.ones(n, dtype=float)
        return np.sqrt(np.maximum(np.where(np.isfinite(vals), vals, 0.0), 0.0))
    if mode == "inv_var":
        err = df["S_tangential_err"].to_numpy(float) if "S_tangential_err" in df.columns else np.full(n, np.nan)
        return np.where(np.isfinite(err) & (err > 0), 1.0 / np.maximum(err * err, 1e-12), 0.0)
    raise ValueError(f"Unknown summary weight mode: {mode}")


def weighted_mean_np(values: np.ndarray, weights: np.ndarray) -> float:
    v = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    ok = np.isfinite(v) & np.isfinite(w) & (w > 0)
    if not np.any(ok):
        return np.nan
    return float(np.sum(w[ok] * v[ok]) / np.sum(w[ok]))


def cluster_bootstrap_mean(
    df: pd.DataFrame,
    values: np.ndarray,
    weights: np.ndarray,
    *,
    cluster_col: str,
    B: int,
    rng: np.random.Generator,
) -> tuple[float, float, float, float]:
    """Bootstrap a weighted mean by resampling clusters.

    For arm-level summaries, cluster_col='node_id'.  This resamples whole nodes
    and keeps all arms of a resampled node together, avoiding the common
    underestimation that happens when arms from the same node are treated as
    independent.
    """
    obs = weighted_mean_np(values, weights)
    if B <= 0 or len(df) < 2 or cluster_col not in df.columns:
        return obs, np.nan, np.nan, np.nan

    ids = pd.unique(df[cluster_col])
    ids = np.asarray(ids)
    if len(ids) < 2:
        return obs, np.nan, np.nan, np.nan

    groups = {cid: np.flatnonzero(df[cluster_col].to_numpy() == cid) for cid in ids}
    boots = np.empty(int(B), dtype=float)
    for b in range(int(B)):
        sample_ids = rng.choice(ids, size=len(ids), replace=True)
        idx = np.concatenate([groups[cid] for cid in sample_ids])
        boots[b] = weighted_mean_np(values[idx], weights[idx])
    return obs, float(np.nanstd(boots, ddof=1)), float(np.nanpercentile(boots, 16.0)), float(np.nanpercentile(boots, 84.0))




def block_bootstrap_mean(
    df: pd.DataFrame,
    values: np.ndarray,
    weights: np.ndarray,
    *,
    nside: int,
    B: int,
    rng: np.random.Generator,
) -> tuple[float, float, float, float]:
    """Bootstrap a weighted mean by resampling HEALPix spatial blocks."""
    obs = weighted_mean_np(values, weights)
    if B <= 0 or len(df) < 2:
        return obs, np.nan, np.nan, np.nan
    blocks = assign_healpix_blocks(df, int(nside))
    ids = np.unique(blocks)
    if len(ids) < 2:
        return obs, np.nan, np.nan, np.nan
    groups = {bid: np.flatnonzero(blocks == bid) for bid in ids}
    boots = np.empty(int(B), dtype=float)
    for b in range(int(B)):
        sample_ids = rng.choice(ids, size=len(ids), replace=True)
        idx = np.concatenate([groups[bid] for bid in sample_ids])
        boots[b] = weighted_mean_np(values[idx], weights[idx])
    return obs, float(np.nanstd(boots, ddof=1)), float(np.nanpercentile(boots, 16.0)), float(np.nanpercentile(boots, 84.0))

def assign_healpix_blocks(df: pd.DataFrame, nside: int) -> np.ndarray:
    """Actual HEALPix RING blocks; never silently substitute a different grid."""
    from spirality.resampling import healpix_blocks
    return healpix_blocks(df["ra"].to_numpy(float), df["dec"].to_numpy(float), nside)


def jackknife_block_mean(df: pd.DataFrame, values: np.ndarray, weights: np.ndarray, *, nside: int) -> tuple[float, float]:
    """HEALPix leave-one-block-out jackknife error for a weighted mean."""
    obs = weighted_mean_np(values, weights)
    if len(df) < 2:
        return obs, np.nan
    pix = assign_healpix_blocks(df, int(nside))
    unique = np.unique(pix[np.isfinite(pix)])
    if len(unique) < 2:
        return obs, np.nan
    jk = []
    for p in unique:
        keep = pix != p
        if np.count_nonzero(keep) < 1:
            continue
        jk.append(weighted_mean_np(values[keep], weights[keep]))
    jk = np.asarray(jk, dtype=float)
    jk = jk[np.isfinite(jk)]
    if len(jk) < 2:
        return obs, np.nan
    mean_jk = float(np.nanmean(jk))
    err = float(np.sqrt((len(jk) - 1.0) / len(jk) * np.nansum((jk - mean_jk) ** 2)))
    return obs, err


def combine_errors(boot_err: float, jk_err: float, mode: str) -> float:
    mode = str(mode).lower()
    be = float(boot_err) if np.isfinite(boot_err) else np.nan
    je = float(jk_err) if np.isfinite(jk_err) else np.nan
    if mode == "bootstrap":
        return be
    if mode == "jackknife":
        return je
    if mode == "quadrature":
        vals = [x for x in [be, je] if np.isfinite(x)]
        return float(np.sqrt(np.sum(np.square(vals)))) if vals else np.nan
    # default: max
    vals = [x for x in [be, je] if np.isfinite(x)]
    return float(np.nanmax(vals)) if vals else np.nan


def summarize_one_bin_rigorous(
    df: pd.DataFrame,
    *,
    weight_mode: str,
    B: int,
    rng: np.random.Generator,
    jackknife_nside: int,
    error_mode: str,
    cluster_col: str,
    bootstrap_mode: str = "cluster",
    bootstrap_nside: int = 16,
) -> dict:
    out: dict = {
        "n_nodes_total_rows": int(len(df)),
        "n_nodes_ok": int(len(df)),
        "error_model": f"{bootstrap_mode}_bootstrap+healpix_jackknife" if str(error_mode).lower() != "bootstrap" else f"{bootstrap_mode}_bootstrap",
        "jackknife_nside": int(jackknife_nside),
        "bootstrap_mode": str(bootstrap_mode),
        "bootstrap_nside": int(bootstrap_nside),
        "summary_weight_mode": str(weight_mode),
    }
    if len(df) == 0:
        return out
    w = summary_weights(df, weight_mode)

    specs = [
        ("S_tangential", "H_S_weighted_mean", "H_S_bootstrap_err", True),
        ("S_tangential_abs", "mean_abs_projected_twist", "mean_abs_projected_twist_bootstrap_err", False),
        ("S_tangential_sq", "mean_transverse_power", "mean_transverse_power_bootstrap_err", False),
        ("mean_t_radial", "mean_radial_coherence", "mean_radial_coherence_bootstrap_err", False),
        ("arm_straightness", "mean_arm_straightness", "mean_arm_straightness_bootstrap_err", False),
        ("arm_transverse_ratio", "mean_arm_transverse_ratio", "mean_arm_transverse_ratio_bootstrap_err", False),
        ("arm_L_path_mpc_h", "mean_arm_L_path_mpc_h", "mean_arm_L_path_mpc_h_bootstrap_err", False),
        ("arm_D_end_mpc_h", "mean_arm_D_end_mpc_h", "mean_arm_D_end_mpc_h_bootstrap_err", False),
        ("arm_R_perp_rms_mpc_h", "mean_arm_R_perp_rms_mpc_h", "mean_arm_R_perp_rms_mpc_h_bootstrap_err", False),
    ]
    for col, mean_name, err_name, signed in specs:
        if col not in df.columns:
            continue
        vals = df[col].to_numpy(float)
        if str(bootstrap_mode).lower() in {"block", "healpix", "healpix_block", "spatial"}:
            obs, berr, bq16, bq84 = block_bootstrap_mean(df, vals, w, nside=bootstrap_nside, B=B, rng=rng)
        else:
            obs, berr, bq16, bq84 = cluster_bootstrap_mean(df, vals, w, cluster_col=cluster_col, B=B, rng=rng)
        _, jerr = jackknife_block_mean(df, vals, w, nside=jackknife_nside)
        terr = combine_errors(berr, jerr, error_mode)

        out[mean_name] = obs
        out[err_name] = terr
        if mean_name == "H_S_weighted_mean":
            vals_for_noise = np.asarray(vals, dtype=float)
            w_for_noise = np.asarray(w, dtype=float)
            ok_noise = np.isfinite(vals_for_noise) & np.isfinite(w_for_noise) & (w_for_noise > 0)
            if np.any(ok_noise) and np.sum(w_for_noise[ok_noise]) > 0:
                h_sign_noise = float(np.sqrt(np.sum((w_for_noise[ok_noise] ** 2) * (vals_for_noise[ok_noise] ** 2))) / np.sum(w_for_noise[ok_noise]))
            else:
                h_sign_noise = np.nan
            out["H_S_sign_noise"] = h_sign_noise
            out["H_S_sign_noise_angle_deg"] = float(np.rad2deg(np.arcsin(np.clip(h_sign_noise, -1, 1)))) if np.isfinite(h_sign_noise) else np.nan
        # Do not use mean_name.replace(...) blindly here: for parity-even
        # quantities mean_name does not contain "weighted_mean", and the old
        # code overwrote the measured value itself with NaN.  That was the
        # reason the Spirality/Torsion, Radial coherence and Transverse power
        # panels became blank while Handedness survived.
        if mean_name.endswith("_weighted_mean"):
            out[mean_name.replace("_weighted_mean", "_cluster_bootstrap_err")] = berr
        out[mean_name + "_cluster_bootstrap_err"] = berr
        out[mean_name + "_healpix_jackknife_err"] = jerr
        out[mean_name + "_total_err"] = terr
        # The plotting code uses q16/q84 if present, so store the conservative
        # total-error band there while retaining the raw bootstrap error above.
        base = mean_name.replace("_weighted_mean", "")
        if mean_name == "H_S_weighted_mean":
            out["H_S_bootstrap_q16"] = float(obs - terr) if np.isfinite(terr) else bq16
            out["H_S_bootstrap_q84"] = float(obs + terr) if np.isfinite(terr) else bq84
            out["H_S_bootstrap_z"] = float(obs / terr) if np.isfinite(terr) and terr > 0 else np.nan
            out["H_S_cluster_bootstrap_err"] = berr
            out["H_S_healpix_jackknife_err"] = jerr
            out["H_S_total_err"] = terr
        else:
            out[err_name.replace("_bootstrap_err", "_bootstrap_q16")] = float(obs - terr) if np.isfinite(terr) else bq16
            out[err_name.replace("_bootstrap_err", "_bootstrap_q84")] = float(obs + terr) if np.isfinite(terr) else bq84
            out[err_name.replace("_bootstrap_err", "_cluster_bootstrap_err")] = berr
            out[err_name.replace("_bootstrap_err", "_healpix_jackknife_err")] = jerr
            out[err_name.replace("_bootstrap_err", "_total_err")] = terr
    # Sign asymmetry: count left/right signs only; ignore curvature amplitude.
    if "S_tangential" in df.columns:
        sv = pd.to_numeric(df["S_tangential"], errors="coerce").to_numpy(float)
        ok_sign = np.isfinite(sv) & (sv != 0.0)
        n_left = int(np.count_nonzero(sv[ok_sign] > 0))
        n_right = int(np.count_nonzero(sv[ok_sign] < 0))
        n_sign = n_left + n_right
        if n_sign > 0:
            sign_vals = np.where(sv[ok_sign] > 0, 1.0, -1.0)
            sign_df = df.loc[ok_sign].copy().reset_index(drop=True)
            if cluster_col not in sign_df.columns:
                sign_df[cluster_col] = np.arange(len(sign_df), dtype=int)
            sign_w = np.ones(len(sign_df), dtype=float)
            if str(bootstrap_mode).lower() in {"block", "healpix", "healpix_block", "spatial"}:
                obs_s, berr_s, bq16_s, bq84_s = block_bootstrap_mean(
                    sign_df, sign_vals, sign_w, nside=bootstrap_nside, B=B, rng=rng
                )
            else:
                obs_s, berr_s, bq16_s, bq84_s = cluster_bootstrap_mean(
                    sign_df, sign_vals, sign_w, cluster_col=cluster_col, B=B, rng=rng
                )
            _, jerr_s = jackknife_block_mean(sign_df, sign_vals, sign_w, nside=jackknife_nside)
            terr_s = combine_errors(berr_s, jerr_s, error_mode)
            out["n_left_handed"] = n_left
            out["n_right_handed"] = n_right
            out["n_sign_total"] = n_sign
            out["sign_asymmetry"] = obs_s
            out["sign_asymmetry_err"] = terr_s
            out["sign_asymmetry_cluster_bootstrap_err"] = berr_s
            out["sign_asymmetry_healpix_jackknife_err"] = jerr_s
            out["sign_asymmetry_bootstrap_q16"] = float(obs_s - terr_s) if np.isfinite(terr_s) else bq16_s
            out["sign_asymmetry_bootstrap_q84"] = float(obs_s + terr_s) if np.isfinite(terr_s) else bq84_s
            out["sign_asymmetry_percent"] = 100.0 * obs_s
            out["sign_asymmetry_percent_err"] = 100.0 * terr_s if np.isfinite(terr_s) else np.nan
            out["sign_noise"] = 1.0 / np.sqrt(float(n_sign)) if n_sign > 0 else np.nan
            out["sign_noise_percent"] = 100.0 / np.sqrt(float(n_sign)) if n_sign > 0 else np.nan
            out["sign_asymmetry_percent_bootstrap_q16"] = 100.0 * out["sign_asymmetry_bootstrap_q16"]
            out["sign_asymmetry_percent_bootstrap_q84"] = 100.0 * out["sign_asymmetry_bootstrap_q84"]

    return out


def normalise_measurement_columns(d: pd.DataFrame) -> pd.DataFrame:
    """Make the measurement-column contract explicit and robust.

    Required plotting/summary columns are:
      S_tangential, S_tangential_abs, S_tangential_sq, mean_t_radial.

    Some intermediate tables use arm-level names such as mean_t_phi.  This
    normalisation avoids silent blank panels when the upstream helper writes a
    valid quantity under a slightly different name.
    """
    d = d.copy()
    if "S_tangential" not in d.columns:
        for alt in ["mean_t_phi", "t_phi"]:
            if alt in d.columns:
                d["S_tangential"] = pd.to_numeric(d[alt], errors="coerce")
                break
    if "mean_t_radial" not in d.columns:
        for alt in ["t_radial", "C_radial", "radial_coherence"]:
            if alt in d.columns:
                d["mean_t_radial"] = pd.to_numeric(d[alt], errors="coerce")
                break
    if "S_tangential" in d.columns:
        s = pd.to_numeric(d["S_tangential"], errors="coerce").to_numpy(float)
        if "S_tangential_abs" not in d.columns or not np.any(np.isfinite(pd.to_numeric(d.get("S_tangential_abs", np.nan), errors="coerce"))):
            d["S_tangential_abs"] = np.abs(s)
        if "S_tangential_sq" not in d.columns or not np.any(np.isfinite(pd.to_numeric(d.get("S_tangential_sq", np.nan), errors="coerce"))):
            d["S_tangential_sq"] = s ** 2
    return d


def robust_z_center(d: pd.DataFrame) -> np.ndarray:
    """Return a usable z coordinate, falling back if z_mid exists but is all NaN."""
    n = len(d)
    z = np.full(n, np.nan, dtype=float)
    if "z_center" in d.columns:
        z = pd.to_numeric(d["z_center"], errors="coerce").to_numpy(float)
    if not np.any(np.isfinite(z)) and "z_mid" in d.columns:
        z = pd.to_numeric(d["z_mid"], errors="coerce").to_numpy(float)
    if (not np.any(np.isfinite(z))) and {"z_low", "z_high"}.issubset(d.columns):
        zlo = pd.to_numeric(d["z_low"], errors="coerce").to_numpy(float)
        zhi = pd.to_numeric(d["z_high"], errors="coerce").to_numpy(float)
        z = 0.5 * (zlo + zhi)
    if (not np.any(np.isfinite(z))) and "z_low" in d.columns:
        z = pd.to_numeric(d["z_low"], errors="coerce").to_numpy(float)
    return z


def build_rigorous_zbin_summary(
    table: pd.DataFrame,
    *,
    edges: np.ndarray,
    weight_mode: str,
    B: int,
    seed: int,
    jackknife_nside: int,
    error_mode: str,
    cluster_col: str,
    bootstrap_mode: str = "cluster",
    bootstrap_nside: int = 16,
) -> pd.DataFrame:
    d = normalise_measurement_columns(table.copy())
    if "status" in d.columns:
        st = d["status"].astype(str).str.lower()
        # Reject all non-accepted rows, including the all-rejected case.
        keep = st.eq("ok")
        d = d.loc[keep].copy()
    d["z_center"] = robust_z_center(d)
    if cluster_col not in d.columns:
        d[cluster_col] = np.arange(len(d), dtype=int)

    rows = []
    z = d["z_center"].to_numpy(float)
    for ib in range(len(edges) - 1):
        z0, z1 = float(edges[ib]), float(edges[ib + 1])
        mask = np.isfinite(z) & (z >= z0) & (z <= z1 if ib == len(edges)-2 else z < z1)
        sub = d.loc[mask].copy().reset_index(drop=True)
        rng = np.random.default_rng(int(seed) + 1009 * (ib + 1))
        row = summarize_one_bin_rigorous(
            sub,
            weight_mode=weight_mode,
            B=B,
            rng=rng,
            jackknife_nside=jackknife_nside,
            error_mode=error_mode,
            cluster_col=cluster_col,
            bootstrap_mode=bootstrap_mode,
            bootstrap_nside=bootstrap_nside,
        )
        z_eff = float(np.nanmean(sub["z_center"].to_numpy(float))) if len(sub) and np.any(np.isfinite(sub["z_center"].to_numpy(float))) else 0.5 * (z0 + z1)
        row.update({
            "zbin_id": ib,
            "z_min": z0,
            "z_max": z1,
            "z_center": z_eff,
            "z_bin_center": 0.5 * (z0 + z1),
            "z_width": z1 - z0,
            "sample": "all",
            "weight_mode": weight_mode,
        })
        rows.append(row)
    return pd.DataFrame(rows)



def compute_arm_shape_metrics_from_segments(seg_df: pd.DataFrame) -> pd.DataFrame:
    """Compute global arm-shape diagnostics from per-segment centroids.

    The segment coordinates x_mpc_h,y_mpc_h are measured in the local plane
    relative to the node.  For each arm we build the polyline

        node -> segment 0 centroid -> ... -> last valid segment centroid

    and compute:
      - arm_L_path_mpc_h: polyline length;
      - arm_D_end_mpc_h: distance from node to last segment centroid;
      - arm_straightness: D_end / L_path;
      - arm_R_perp_rms_mpc_h: RMS perpendicular displacement of the segment
        centroids with respect to the node--endpoint chord;
      - arm_transverse_ratio: R_perp_rms / D_end.

    These are auxiliary geometric diagnostics.  They are not used to define the
    handedness estimator.
    """
    if seg_df is None or len(seg_df) == 0:
        return pd.DataFrame()
    required = {"node_id", "arm_local_id", "x_mpc_h", "y_mpc_h"}
    if not required.issubset(seg_df.columns):
        missing = sorted(required.difference(seg_df.columns))
        print(f"[warn] cannot compute arm-shape metrics; missing segment columns: {missing}")
        return pd.DataFrame()

    d = seg_df.copy()
    d["node_id"] = pd.to_numeric(d["node_id"], errors="coerce")
    d["arm_local_id"] = pd.to_numeric(d["arm_local_id"], errors="coerce")
    d["x_mpc_h"] = pd.to_numeric(d["x_mpc_h"], errors="coerce")
    d["y_mpc_h"] = pd.to_numeric(d["y_mpc_h"], errors="coerce")
    if "arm_segment_index" in d.columns:
        d["arm_segment_index"] = pd.to_numeric(d["arm_segment_index"], errors="coerce")
    else:
        d["arm_segment_index"] = np.arange(len(d), dtype=float)

    rows = []
    for (node_id, arm_id), sub in d.groupby(["node_id", "arm_local_id"], dropna=True):
        sub = sub.sort_values("arm_segment_index").copy()
        x = sub["x_mpc_h"].to_numpy(float)
        y = sub["y_mpc_h"].to_numpy(float)
        ok = np.isfinite(x) & np.isfinite(y)
        x, y = x[ok], y[ok]
        if len(x) < 2:
            continue

        pts = np.column_stack([x, y])
        pts_path = np.vstack([np.zeros((1, 2), dtype=float), pts])
        diffs = np.diff(pts_path, axis=0)
        step = np.sqrt(np.sum(diffs * diffs, axis=1))
        L_path = float(np.nansum(step))
        end = pts[-1]
        D_end = float(np.sqrt(np.sum(end * end)))

        if not np.isfinite(L_path) or L_path <= 0 or not np.isfinite(D_end) or D_end <= 0:
            straightness = np.nan
            R_perp_rms = np.nan
            transverse_ratio = np.nan
            R_perp_max = np.nan
        else:
            straightness = float(D_end / L_path)
            u = end / D_end
            # 2D cross product magnitude relative to the endpoint chord.
            perp = np.abs(pts[:, 0] * u[1] - pts[:, 1] * u[0])
            if "n_points" in sub.columns:
                ww = pd.to_numeric(sub.loc[ok, "n_points"], errors="coerce").to_numpy(float)
            elif "weight_sum" in sub.columns:
                ww = pd.to_numeric(sub.loc[ok, "weight_sum"], errors="coerce").to_numpy(float)
            else:
                ww = np.ones(len(perp), dtype=float)
            ww = np.where(np.isfinite(ww) & (ww > 0), ww, 1.0)
            R_perp_rms = float(np.sqrt(np.sum(ww * perp * perp) / np.sum(ww)))
            R_perp_max = float(np.nanmax(perp)) if len(perp) else np.nan
            transverse_ratio = float(R_perp_rms / D_end)

        rows.append({
            "node_id": int(node_id),
            "arm_local_id": int(arm_id),
            "arm_L_path_mpc_h": L_path,
            "arm_D_end_mpc_h": D_end,
            "arm_straightness": straightness,
            "arm_R_perp_rms_mpc_h": R_perp_rms,
            "arm_R_perp_max_mpc_h": R_perp_max,
            "arm_transverse_ratio": transverse_ratio,
            "arm_n_shape_segments": int(len(x)),
        })
    return pd.DataFrame(rows)


def attach_arm_shape_metrics(arm_tab: pd.DataFrame, seg_path: Path) -> pd.DataFrame:
    """Merge arm-shape diagnostics into an arm-level table when possible."""
    if arm_tab is None or len(arm_tab) == 0:
        return arm_tab
    if not seg_path.exists():
        return arm_tab
    if not {"node_id", "arm_local_id"}.issubset(arm_tab.columns):
        return arm_tab
    try:
        metrics = compute_arm_shape_metrics_from_segments(pd.read_csv(seg_path))
    except Exception as exc:
        print(f"[warn] failed to compute arm-shape metrics from {seg_path}: {exc!r}")
        return arm_tab
    if len(metrics) == 0:
        return arm_tab

    out = arm_tab.copy()
    # Drop stale geometry columns before merging so --reuse-existing-products
    # refreshes the definitions if this helper changes.
    geom_cols = [c for c in metrics.columns if c not in {"node_id", "arm_local_id"}]
    out = out.drop(columns=[c for c in geom_cols if c in out.columns], errors="ignore")
    out = out.merge(metrics, on=["node_id", "arm_local_id"], how="left")
    return out


def make_arm_level_table_generic(arm_df: pd.DataFrame, node_df: pd.DataFrame) -> pd.DataFrame:
    """Build an arm-level table from arm_spirality.csv plus node metadata."""
    if arm_df is None or len(arm_df) == 0:
        return pd.DataFrame()
    from spirality.selection import attach_parent_metadata
    a = attach_parent_metadata(arm_df, node_df, accepted_only=True)
    merge_cols = [c for c in [
        "node_id", "ra", "dec", "z_low", "z_mid", "z_high", "gal_l_deg", "gal_b_deg", "gal_hemi", "mean_segment_density"
    ] if c in node_df.columns]
    if "node_id" in merge_cols:
        a = a.merge(node_df[merge_cols].drop_duplicates("node_id"), on="node_id", how="left", suffixes=("", "_node"))

    out = pd.DataFrame(index=np.arange(len(a)))
    out["status"] = a["parent_status"].to_numpy()
    for col in ["parent_status", "parent_n_arms_valid", "fans_n_arms", "analysis_schema"]:
        out[col] = a[col].to_numpy()
    out["node_id"] = a["node_id"].to_numpy(int) if "node_id" in a.columns else np.arange(len(a), dtype=int)
    out["arm_local_id"] = a["arm_local_id"].to_numpy(int) if "arm_local_id" in a.columns else np.zeros(len(a), dtype=int)

    for col in ["ra", "dec", "z_low", "z_mid", "z_high", "gal_l_deg", "gal_b_deg", "gal_hemi"]:
        if col in a.columns:
            out[col] = a[col].to_numpy() if col != "gal_hemi" else a[col].astype(str).to_numpy()
        elif f"{col}_node" in a.columns:
            out[col] = a[f"{col}_node"].to_numpy() if col != "gal_hemi" else a[f"{col}_node"].astype(str).to_numpy()
    if "z_mid" not in out.columns:
        if "z_low" in out.columns and "z_high" in out.columns:
            out["z_mid"] = 0.5 * (out["z_low"].astype(float) + out["z_high"].astype(float))
        elif "z_low" in out.columns:
            out["z_mid"] = out["z_low"].astype(float)
    else:
        zmid_tmp = pd.to_numeric(out["z_mid"], errors="coerce").to_numpy(float)
        if (not np.any(np.isfinite(zmid_tmp))) and "z_low" in out.columns:
            out["z_mid"] = pd.to_numeric(out["z_low"], errors="coerce").to_numpy(float)
    if "z_low" not in out.columns and "z_mid" in out.columns:
        out["z_low"] = out["z_mid"].astype(float)

    src_phi = "mean_t_phi" if "mean_t_phi" in a.columns else "S_tangential"
    out["S_tangential"] = a[src_phi].to_numpy(float)
    err_col = "mean_t_phi_err" if "mean_t_phi_err" in a.columns else "S_tangential_err"
    out["S_tangential_err"] = a[err_col].to_numpy(float) if err_col in a.columns else np.nan
    out["S_tangential_abs"] = np.abs(out["S_tangential"].to_numpy(float))
    out["S_tangential_sq"] = out["S_tangential"].to_numpy(float) ** 2
    out["mean_t_radial"] = a["mean_t_radial"].to_numpy(float) if "mean_t_radial" in a.columns else np.nan
    out["n_arm_segments_total"] = a["n_arm_segments_valid"].to_numpy(float) if "n_arm_segments_valid" in a.columns else 1.0
    out["n_points_used_total"] = a["n_points_total"].to_numpy(float) if "n_points_total" in a.columns else out["n_arm_segments_total"].to_numpy(float)
    if "mean_arm_density" in a.columns:
        den = a["mean_arm_density"].to_numpy(float)
    elif "mean_segment_density" in a.columns:
        den = a["mean_segment_density"].to_numpy(float)
    elif "mean_segment_density_node" in a.columns:
        den = a["mean_segment_density_node"].to_numpy(float)
    else:
        den = np.ones(len(a), dtype=float)
    out["mean_segment_density"] = np.where(np.isfinite(den) & (den > 0), den, 1.0)
    return out


def rewrite_rigorous_summaries_for_pipeline(
    *,
    pipeline_dir: Path,
    arm_summary_dir: Optional[Path],
    zbin_edges: str,
    arm_summary_weights: list[str],
    catalog_key: str = "catalog",
    B: int,
    seed: int,
    jackknife_nside: int,
    error_mode: str,
    bootstrap_mode: str = "cluster",
    bootstrap_nside: int = 16,
) -> None:
    """Overwrite zbin summary CSVs with clustered bootstrap + HEALPix jackknife errors."""
    edges = parse_edges_string(zbin_edges)
    node_path = pipeline_dir / "node_spirality.csv"
    if node_path.exists():
        node_df = pd.read_csv(node_path)
        node_summary = build_rigorous_zbin_summary(
            node_df,
            edges=edges,
            weight_mode="uniform",
            B=B,
            seed=seed + 31,
            jackknife_nside=jackknife_nside,
            error_mode=error_mode,
            cluster_col="node_id",
            bootstrap_mode=bootstrap_mode,
            bootstrap_nside=bootstrap_nside,
        )
        node_summary.to_csv(pipeline_dir / "zbin_summary.csv", index=False)
        print(f"[summary] rewrote {pipeline_dir / 'zbin_summary.csv'}")

    arm_path = pipeline_dir / "arm_level_spirality.csv"
    arm_raw = pipeline_dir / "arm_spirality.csv"
    if not node_path.exists() or not arm_raw.exists():
        raise FileNotFoundError(f"Missing raw arm/parent tables in {pipeline_dir}")
    node_df = pd.read_csv(node_path)
    try:
        raw = pd.read_csv(arm_raw)
    except pd.errors.EmptyDataError:
        raw = pd.DataFrame(columns=["node_id", "arm_local_id"])
    arm_tab = make_arm_level_table_generic(raw, node_df)
    arm_tab.to_csv(arm_path, index=False)
    from spirality.selection import audit_catalog
    segpath = pipeline_dir / "arm_segments.csv"
    segments = pd.read_csv(segpath) if segpath.exists() and segpath.stat().st_size > 1 else None
    audit = audit_catalog(node_df, raw, arm_tab, segments) if len(arm_tab) else {"passed": True, "n_analysis_arms": 0}
    (pipeline_dir / "sample_integrity.json").write_text(json.dumps(audit, indent=2))

    if len(arm_tab) > 0:
        # Add global shape diagnostics from arm_segments.csv.  This is done
        # even when --reuse-existing-products is used, so older arm-level tables
        # receive the new columns without rerunning the heavy filament tracing.
        arm_tab = attach_arm_shape_metrics(arm_tab, pipeline_dir / "arm_segments.csv")
        if "node_id" in arm_tab.columns:
            node_num = pd.to_numeric(arm_tab["node_id"], errors="coerce").fillna(-1).astype(int)
            arm_tab["catalog_key"] = str(catalog_key)
            arm_tab["fans_id"] = [f"{str(catalog_key)}_FANS_{x:06d}" if x >= 0 else f"{str(catalog_key)}_FANS_unknown_{i:06d}" for i, x in enumerate(node_num)]
        arm_tab.to_csv(pipeline_dir / "arm_level_spirality.csv", index=False)
        out_dir = arm_summary_dir if arm_summary_dir is not None else pipeline_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        for wmode in arm_summary_weights:
            arm_summary = build_rigorous_zbin_summary(
                arm_tab,
                edges=edges,
                weight_mode=wmode,
                B=B,
                seed=seed + 97,
                jackknife_nside=jackknife_nside,
                error_mode=error_mode,
                cluster_col="node_id",
                bootstrap_mode=bootstrap_mode,
                bootstrap_nside=bootstrap_nside,
            )
            outp = out_dir / f"zbin_summary_arms_{wmode}.csv"
            arm_summary.to_csv(outp, index=False)
            metric_counts = {c: int(np.count_nonzero(np.isfinite(pd.to_numeric(arm_summary[c], errors="coerce"))))
                             for c in ["H_S_weighted_mean", "mean_abs_projected_twist", "mean_radial_coherence", "mean_transverse_power"]
                             if c in arm_summary.columns}
            print(f"[summary] rewrote {outp} finite_counts={metric_counts}")


# -----------------------------------------------------------------------------
# Gallery utilities
# -----------------------------------------------------------------------------

def read_required(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def angsep_rad(ra1, dec1, ra2, dec2):
    ra1 = np.deg2rad(float(ra1)); dec1 = np.deg2rad(float(dec1))
    ra2 = np.deg2rad(np.asarray(ra2, dtype=float)); dec2 = np.deg2rad(np.asarray(dec2, dtype=float))
    s = np.sin(dec1)*np.sin(dec2) + np.cos(dec1)*np.cos(dec2)*np.cos(ra1-ra2)
    return np.arccos(np.clip(s, -1.0, 1.0))


def match_nodes(reference: pd.DataFrame, target: pd.DataFrame, *,
                max_sep_mpc_h: float, max_dchi_mpc_h: float, max_dz: float,
                max_3d_mpc_h: float = np.inf,
                ref_id_col: str = "node_id", target_name: str = "target") -> pd.DataFrame:
    tt = target[target["status"].astype(str).eq("ok")].copy()
    tra = tt["ra"].to_numpy(float)
    tdec = tt["dec"].to_numpy(float)
    tz = tt["z_low"].to_numpy(float)
    tchi = tt["chi_mpc_h"].to_numpy(float) if "chi_mpc_h" in tt.columns else np.full(len(tt), np.nan)

    rows = []
    for _, r in reference.iterrows():
        rchi = float(r.get("chi_mpc_h", np.nan))
        sep_ang = angsep_rad(float(r["ra"]), float(r["dec"]), tra, tdec)
        sep_trans = sep_ang * rchi if np.isfinite(rchi) else np.full_like(sep_ang, np.nan)
        dchi = np.abs(tchi - rchi) if np.isfinite(rchi) else np.full_like(sep_ang, np.nan)
        dz = np.abs(tz - float(r["z_low"]))
        d3 = np.sqrt(np.nan_to_num(sep_trans, nan=1e9)**2 + np.nan_to_num(dchi, nan=1e9)**2)
        metric = np.where(dz <= max_dz, d3, np.inf)
        if len(tt) == 0 or not np.any(np.isfinite(metric)):
            j = -1
            accept = False
        else:
            j = int(np.nanargmin(metric))
            accept = bool(
                np.isfinite(metric[j]) and
                sep_trans[j] <= max_sep_mpc_h and
                dchi[j] <= max_dchi_mpc_h and
                d3[j] <= max_3d_mpc_h and
                dz[j] <= max_dz
            )
        rows.append({
            "ref_node_id": int(r[ref_id_col]),
            f"{target_name}_node_id": int(tt.iloc[j]["node_id"]) if j >= 0 else -1,
            f"{target_name}_accepted": accept,
            f"{target_name}_sep_trans_mpc_h": float(sep_trans[j]) if j >= 0 else np.nan,
            f"{target_name}_dchi_mpc_h": float(dchi[j]) if j >= 0 else np.nan,
            f"{target_name}_d3_mpc_h": float(d3[j]) if j >= 0 else np.nan,
            f"{target_name}_dz": float(dz[j]) if j >= 0 else np.nan,
            f"{target_name}_ra": float(tt.iloc[j]["ra"]) if j >= 0 else np.nan,
            f"{target_name}_dec": float(tt.iloc[j]["dec"]) if j >= 0 else np.nan,
            f"{target_name}_z_low": float(tt.iloc[j]["z_low"]) if j >= 0 else np.nan,
        })
    return pd.DataFrame(rows)



# Development-only SCONCE--Carrón comparison galleries and PA diagnostics removed in v25.


# -----------------------------------------------------------------------------
# Random per-FANS PNG gallery
# -----------------------------------------------------------------------------

def _safe_int_id(x) -> str:
    try:
        v = int(float(x))
        return f"{v:06d}" if v >= 0 else "unknown"
    except Exception:
        return "unknown"




def _parse_packed_float_array(val) -> np.ndarray:
    """Parse semicolon-separated packed coordinates stored in arm_segments.csv."""
    try:
        if val is None or (isinstance(val, float) and not np.isfinite(val)):
            return np.array([], dtype=float)
        text = str(val).strip()
        if not text or text.lower() == "nan":
            return np.array([], dtype=float)
        return np.array([float(x) for x in text.split(";") if str(x).strip() != ""], dtype=float)
    except Exception:
        return np.array([], dtype=float)


def _collect_raw_points_from_segments(seg_sub: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    if not {"raw_x_mpc_h", "raw_y_mpc_h"}.issubset(seg_sub.columns):
        return np.array([], dtype=float), np.array([], dtype=float)
    xs, ys = [], []
    for _, rr in seg_sub.iterrows():
        x = _parse_packed_float_array(rr.get("raw_x_mpc_h", ""))
        y = _parse_packed_float_array(rr.get("raw_y_mpc_h", ""))
        n = min(len(x), len(y))
        if n > 0:
            xs.append(x[:n]); ys.append(y[:n])
    if not xs:
        return np.array([], dtype=float), np.array([], dtype=float)
    return np.concatenate(xs), np.concatenate(ys)

def _plot_single_fans_from_segments(
    *,
    node_id,
    fans_id: str,
    arms_sub: pd.DataFrame,
    seg_sub: pd.DataFrame,
    out_png: Path,
    catalog_label: str,
) -> None:
    """Save a single-FANS PNG using the same visual language as the native galleries.

    The native multi-panel galleries use pale link/radial circles, a black node
    marker, faint thick arm paths and compact segment-centroid markers.  This
    function mirrors that style so the random per-FANS PNGs can be inspected side
    by side with the gallery panels.  When the arm_segments.csv file contains
    packed raw point coordinates, the PNG also shows the ridge points that
    entered each measured radial segment.
    """
    setup_matplotlib()
    fig, ax = plt.subplots(figsize=(4.2, 4.2))

    label_l = str(catalog_label).lower()
    if "carr" in label_l:
        line_color, centroid_color, link_color = "royalblue", "royalblue", "lightskyblue"
    elif "chen" in label_l:
        line_color, centroid_color, link_color = "orangered", "orangered", "lightsalmon"
    elif "zhang" in label_l or "sconce" in label_l:
        line_color, centroid_color, link_color = "seagreen", "seagreen", "lightgreen"
    else:
        line_color, centroid_color, link_color = "0.22", "k", "0.80"

    if len(seg_sub) == 0:
        plt.close(fig)
        return

    seg_sub = seg_sub.copy()
    if "arm_local_id" not in seg_sub.columns:
        seg_sub["arm_local_id"] = 0
    if "arm_segment_index" in seg_sub.columns:
        seg_sub["arm_segment_index"] = pd.to_numeric(seg_sub["arm_segment_index"], errors="coerce")

    xs_all = pd.to_numeric(seg_sub.get("x_mpc_h", pd.Series(dtype=float)), errors="coerce").to_numpy(float)
    ys_all = pd.to_numeric(seg_sub.get("y_mpc_h", pd.Series(dtype=float)), errors="coerce").to_numpy(float)
    rho_all = np.hypot(xs_all, ys_all)
    max_r_data = float(np.nanmax(rho_all)) if np.any(np.isfinite(rho_all)) else 30.0
    max_seg_index = int(np.nanmax(seg_sub["arm_segment_index"].to_numpy(float))) if "arm_segment_index" in seg_sub.columns and np.any(np.isfinite(seg_sub["arm_segment_index"].to_numpy(float))) else 2
    nseg_draw = max(3, max_seg_index + 1)
    # Fiducial values used in the paper; fall back to data extent if needed.
    seg_len = 10.0
    arm_radius = max(30.0, nseg_draw * seg_len, max_r_data)
    link_radius = 8.0
    pad = max(arm_radius, link_radius) * 1.12

    # Light seed/link region and radial bin circles, matching native gallery style.
    link = plt.Circle((0.0, 0.0), link_radius, color=link_color, alpha=0.18, lw=0, zorder=0)
    ax.add_patch(link)
    for k in range(1, nseg_draw + 1):
        r = k * seg_len
        circ = plt.Circle((0.0, 0.0), r, fill=False, color="0.76", lw=0.50, ls="--", alpha=0.70, zorder=0)
        ax.add_patch(circ)

    raw_x, raw_y = _collect_raw_points_from_segments(seg_sub)
    if len(raw_x) > 0:
        ax.scatter(raw_x, raw_y, s=2.8, color="0.58", alpha=0.30, linewidths=0, rasterized=True, zorder=1)

    ax.scatter([0.0], [0.0], marker="x", s=30, color="k", lw=0.9, zorder=6)

    for _, arm in seg_sub.groupby("arm_local_id"):
        arm = arm.copy()
        if "arm_segment_index" in arm.columns:
            arm = arm.sort_values("arm_segment_index")
        x = pd.to_numeric(arm.get("x_mpc_h", np.nan), errors="coerce").to_numpy(float)
        y = pd.to_numeric(arm.get("y_mpc_h", np.nan), errors="coerce").to_numpy(float)
        ok = np.isfinite(x) & np.isfinite(y)
        x, y = x[ok], y[ok]
        if len(x) == 0:
            continue
        ax.plot(x, y, color=line_color, lw=2.5, alpha=0.12, zorder=2)
        ax.scatter(x, y, s=8.0, color=centroid_color, alpha=0.68, zorder=4)

    ax.axhline(0, color="0.88", lw=0.4, zorder=0)
    ax.axvline(0, color="0.88", lw=0.4, zorder=0)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-pad, pad)
    ax.set_ylim(-pad, pad)
    ax.tick_params(labelsize=8.0, length=2.4, pad=0.8, width=0.7)
    ax.set_xticks([-20, 0, 20])
    ax.set_yticks([-20, 0, 20])
    ax.set_xlabel(r"$\Delta x$ [$h^{-1}$ Mpc]", fontsize=10)
    ax.set_ylabel(r"$\Delta y$ [$h^{-1}$ Mpc]", fontsize=10)

    n_arms = int(arms_sub["arm_local_id"].nunique()) if "arm_local_id" in arms_sub.columns else int(len(arms_sub))
    ra = pd.to_numeric(arms_sub.get("ra", pd.Series(dtype=float)), errors="coerce").median() if "ra" in arms_sub.columns else np.nan
    dec = pd.to_numeric(arms_sub.get("dec", pd.Series(dtype=float)), errors="coerce").median() if "dec" in arms_sub.columns else np.nan
    zval = pd.to_numeric(arms_sub.get("z_mid", pd.Series(dtype=float)), errors="coerce").median() if "z_mid" in arms_sub.columns else np.nan
    if not np.isfinite(zval) and "z_low" in arms_sub.columns:
        zval = pd.to_numeric(arms_sub.get("z_low", pd.Series(dtype=float)), errors="coerce").median()
    txt = f"{fans_id} | {n_arms} arms"
    if np.isfinite(ra) and np.isfinite(dec) and np.isfinite(zval):
        txt += f"\n$\\alpha$={ra:.1f}°, $\\delta$={dec:.1f}°, $z$={zval:.3f}"
    elif np.isfinite(zval):
        txt += f"\n$z$={zval:.3f}"
    ax.text(0.04, 0.965, txt, transform=ax.transAxes, va="top", ha="left", fontsize=7.2,
            bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.68), zorder=7)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight", dpi=220)
    plt.close(fig)

def make_random_fans_gallery(
    *,
    catalog_key: str,
    catalog_label: str,
    pipeline_dir: Path,
    outroot: Path,
    arm_counts: list[int],
    n_per_count: int,
    seed: int,
) -> None:
    arm_path = pipeline_dir / "arm_level_spirality.csv"
    seg_path = pipeline_dir / "arm_segments.csv"
    node_path = pipeline_dir / "node_spirality.csv"
    if not arm_path.exists() or not seg_path.exists():
        print(f"[gallery-random] missing arm/segment products for {catalog_key}; skipped")
        return
    arms = pd.read_csv(arm_path)
    seg = pd.read_csv(seg_path)
    nodes = pd.read_csv(node_path) if node_path.exists() else pd.DataFrame()
    if len(arms) == 0 or len(seg) == 0 or "node_id" not in arms.columns or "node_id" not in seg.columns:
        print(f"[gallery-random] missing node_id or empty products for {catalog_key}; skipped")
        return
    if "arm_local_id" not in arms.columns:
        arms["arm_local_id"] = np.arange(len(arms), dtype=int)
    arms["node_id_num"] = pd.to_numeric(arms["node_id"], errors="coerce")
    seg["node_id_num"] = pd.to_numeric(seg["node_id"], errors="coerce")
    rng = np.random.default_rng(int(seed))

    # Use the same object-level acceptance as the native galleries: only nodes
    # with status=='ok' and the requested number of valid arms.  The previous
    # implementation grouped the arm table alone, which could pick visually poor
    # or partially reconstructed systems that the native typical gallery would
    # not select as accepted FANS.
    if len(nodes) and "node_id" in nodes.columns:
        nodes = nodes.copy()
        nodes["node_id_num"] = pd.to_numeric(nodes["node_id"], errors="coerce")
        ok_nodes = nodes[nodes.get("status", pd.Series("ok", index=nodes.index)).astype(str).eq("ok")].copy()
        if "n_arms_valid" in ok_nodes.columns:
            node_arm_counts = pd.to_numeric(ok_nodes["n_arms_valid"], errors="coerce")
        else:
            grouped_counts = arms.groupby("node_id_num")["arm_local_id"].nunique()
            node_arm_counts = ok_nodes["node_id_num"].map(grouped_counts)
        ok_nodes["_n_arms_for_gallery"] = node_arm_counts
        available = set(pd.to_numeric(seg["node_id_num"], errors="coerce").dropna().astype(int).tolist())
        ok_nodes = ok_nodes[ok_nodes["node_id_num"].apply(lambda x: np.isfinite(x) and int(x) in available)].copy()
        counts_source = ok_nodes.set_index("node_id_num")["_n_arms_for_gallery"]
    else:
        print(f"[gallery-random] warning: {node_path} missing; falling back to arm-table grouping")
        counts_source = arms.groupby("node_id_num")["arm_local_id"].nunique().dropna()

    for ac in arm_counts:
        candidates = pd.to_numeric(counts_source, errors="coerce")
        candidates = candidates[candidates.eq(int(ac))].index.to_numpy(dtype=float)
        if len(candidates) == 0:
            print(f"[gallery-random] {catalog_key}: no accepted FANS with {ac} arms")
            continue
        nsel = min(int(n_per_count), len(candidates))
        chosen = rng.choice(candidates, size=nsel, replace=False)
        subdir = outroot / f"{catalog_key}_{int(ac)}arms"
        rows = []
        for jj, nid in enumerate(chosen):
            fans_id = f"{catalog_key}_FANS_{_safe_int_id(nid)}"
            out_png = subdir / f"{fans_id}.png"
            arms_sub = arms.loc[arms["node_id_num"].eq(nid)].copy()
            seg_sub = seg.loc[seg["node_id_num"].eq(nid)].copy()
            _plot_single_fans_from_segments(
                node_id=nid, fans_id=fans_id, arms_sub=arms_sub, seg_sub=seg_sub,
                out_png=out_png, catalog_label=catalog_label,
            )
            rows.append({"catalog_key": catalog_key, "fans_id": fans_id, "node_id": nid, "n_arms": int(ac), "png": str(out_png)})
            if (jj + 1) % 100 == 0:
                print(f"[gallery-random] {catalog_key} {ac} arms: {jj+1}/{nsel}")
        pd.DataFrame(rows).to_csv(subdir / "selected_fans.csv", index=False)
        print(f"[gallery-random] wrote {nsel} PNGs to {subdir}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--outroot", required=True)
    p.add_argument("--run-name", default="fiducial_three_catalogs")
    p.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--reuse-existing-products", action="store_true",
                   help="Skip heavy pipeline execution when node/arm/segment CSV products already exist; still rewrite summaries, plots and diagnostics.")
    p.add_argument("--pipelines", nargs="+", default=["sconce", "carron", "chen"],
                   choices=["sconce", "carron", "chen"],
                   help="Catalogue pipelines to run. Examples: --pipelines carron chen, or --pipelines sconce chen.")
    p.add_argument("--parallel-pipelines", action=argparse.BooleanOptionalAction, default=False,
                   help="Run independent catalogue pipelines concurrently. Summaries and plots are still generated after all pipelines finish.")
    p.add_argument("--pipeline-workers", type=int, default=None,
                   help="Number of concurrent pipeline processes. Default: number of selected pipelines.")
    p.add_argument("--jobs-per-pipeline", type=int, default=None,
                   help="Override --n-jobs passed to each pipeline helper. If omitted with --parallel-pipelines, use floor(--n-jobs / number_of_active_pipelines).")
    p.add_argument("--sconce-jobs", type=int, default=None,
                   help="Override n-jobs passed only to the Zhang/SCONCE helper.")
    p.add_argument("--carron-jobs", type=int, default=None,
                   help="Override n-jobs passed only to the Carrón helper.")
    p.add_argument("--chen-jobs", type=int, default=None,
                   help="Override n-jobs passed only to the Chen helper.")

    p.add_argument("--sconce-script", default=None)
    p.add_argument("--carron-proxy-script", default=None)
    p.add_argument("--chen-script", default=None)
    p.add_argument("--chen-arm-postprocess-script", default=None)

    p.add_argument("--sconce-fits", default="data/nodes_scms.fits")
    p.add_argument("--carron-files", nargs="+", default=[
        "data/Block1.csv",
        "data/Block2.csv",
        "data/Block3.csv",
    ])
    p.add_argument("--chen-intersections", default="data/dr12_Fint.txt")
    p.add_argument("--chen-fmaps", default="data/dr12_FMaps_full.txt")

    # Separate science ranges.
    p.add_argument("--sconce-z-min", type=float, default=0.1)
    p.add_argument("--sconce-z-max", type=float, default=0.6)
    p.add_argument("--sconce-zbin-edges", default="0.1,0.225,0.35,0.475,0.6")

    p.add_argument("--carron-z-min", type=float, default=0.1)
    p.add_argument("--carron-z-max", type=float, default=0.6)
    p.add_argument("--carron-zbin-edges", default="0.1,0.225,0.35,0.475,0.6")

    p.add_argument("--chen-z-min", type=float, default=0.1)
    p.add_argument("--chen-z-max", type=float, default=0.6)
    p.add_argument("--chen-zbin-edges", default="0.1,0.225,0.35,0.475,0.6")

    # Common geometry.
    p.add_argument("--n-arm-segments", type=int, default=3)
    p.add_argument("--segment-length-mpc-h", type=float, default=10.0)
    p.add_argument("--node-link-radius-mpc-h", type=float, default=8.0)
    p.add_argument("--edge-margin-mpc-h", type=float, default=70.0)
    p.add_argument("--knot-cluster-radius-mpc-h", type=float, default=5.0)
    p.add_argument("--arm-opening-deg", type=float, default=35.0)
    p.add_argument("--cluster-gap-deg", type=float, default=35.0)
    p.add_argument("--seed-max-tangent-angle-deg", type=float, default=75.0)
    p.add_argument("--min-seed-points", type=int, default=2)
    p.add_argument("--min-points-per-arm-segment", type=int, default=3)
    p.add_argument("--min-valid-arm-segments", type=int, default=2)
    p.add_argument("--min-arms-node", type=int, default=2)
    p.add_argument("--max-arms-node", type=int, default=12)
    p.add_argument("--analysis-product", default="custom",
                   choices=["custom", "fiducial", "uniform_geometry", "uncertainty_weighted", "hybrid"],
                   help="Convenience preset. fiducial=density geometry; uniform_geometry=uniform; uncertainty_weighted=inv_unc2; hybrid=density_inv_unc2.")
    p.add_argument("--point-weight", default="density", choices=["uniform", "density", "sqrt_density", "log_density", "inv_unc2", "density_inv_unc2"])
    p.add_argument("--arms-segment-uncertainty", default="off", choices=["off", "inv_unc2"],
                   help="If inv_unc2, local segment construction uses catalogue point uncertainty. With --point-weight density this becomes density_inv_unc2.")
    p.add_argument("--point-unc-floor", type=float, default=1e-3,
                   help="Uncertainty floor added in quadrature for inv_unc2 point weights, in the native uncertainty units of each catalogue.")
    p.add_argument("--point-unc-max-weight", type=float, default=1e6,
                   help="Maximum point weight allowed for inverse-uncertainty weighting.")
    p.add_argument("--node-weight", default="uniform", choices=["uniform", "mean_density", "sqrt_mean_density", "n_segments", "sqrt_points", "inv_var"])
    p.add_argument("--knot-cluster-weight", default="density", choices=["uniform", "density", "inv_unc2"])
    p.add_argument("--sconce-gradient-mode", default="tangent", choices=["tangent", "perp"],
                   help="SCONCE vector interpretation for the main SCONCE-native run.")
    p.add_argument("--carron-angle-offset-deg", type=float, default=0.0,
                   help="Carrón angle offset for the main Carrón-proxy run.")

    p.add_argument("--bootstrap-B", type=int, default=1000)
    p.add_argument("--signflip-B", type=int, default=2000)
    p.add_argument("--summary-error-model", default="max", choices=["bootstrap", "jackknife", "max", "quadrature"],
                   help="Error used in final z-bin plots after recomputing summaries. Default max(cluster bootstrap, HEALPix jackknife).")
    p.add_argument("--summary-jackknife-nside", type=int, default=4,
                   help="HEALPix NSIDE for leave-one-pixel-out jackknife errors in final z-bin summaries.")
    p.add_argument("--summary-bootstrap-mode", default="cluster", choices=["cluster", "block", "healpix_block", "spatial"],
                   help="Bootstrap mode for final z-bin summaries. Use 'block' with --summary-bootstrap-nside 16 for paper-facing spatial block bootstrap errors.")
    p.add_argument("--summary-bootstrap-nside", type=int, default=16,
                   help="HEALPix NSIDE for block bootstrap errors when --summary-bootstrap-mode is block/healpix_block/spatial.")
    p.add_argument("--arm-summary-weights", nargs="+", default=["n_segments"],
                   choices=["uniform", "mean_density", "sqrt_mean_density", "n_segments", "sqrt_points", "inv_var"],
                   help="Arm-level summary weight modes to save and plot.")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--n-jobs", type=int, default=32)
    p.add_argument("--plot-format", default="pdf", choices=["pdf", "png", "jpg"])

    # Carron proxy shell match.
    p.add_argument("--match-min-overlap", type=float, default=0.5)
    p.add_argument("--match-max-dchi-mpc", type=float, default=5.0)

    # Joint plot filter: default none, because each catalog has intentionally different range/bins.
    p.add_argument("--plot-bin-filter-mode", default="none", choices=["none", "overlap", "center", "contained"])
    p.add_argument("--plot-z-min", type=float, default=None)
    p.add_argument("--plot-z-max", type=float, default=None)

    # Gallery special range/hemisphere.
    p.add_argument("--make-gallery", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--make-gallery-convention-grid", action=argparse.BooleanOptionalAction, default=False,
                   help=argparse.SUPPRESS)
    p.add_argument("--gallery-z-max", type=float, default=0.6)
    p.add_argument("--gallery-gal-hemi", default="all", choices=["all", "north", "south"])
    p.add_argument("--gallery-n", type=int, default=96)
    p.add_argument("--gallery-ncols", type=int, default=8)
    p.add_argument("--gallery-nrows", type=int, default=12)
    p.add_argument("--gallery-selection", default="random", choices=["random", "strongest", "first"], help="Legacy triple-gallery selector; retained for compatibility.")
    p.add_argument("--gallery-modes", nargs="+", default=["optimal", "typical"], choices=["optimal", "typical", "random"], help="SCONCE/Carrón gallery modes to make.")
    p.add_argument("--gallery-seed", type=int, default=2026)
    p.add_argument("--make-pa-diagnostics", action=argparse.BooleanOptionalAction, default=False,
                   help="Make automatic SCONCE-vs-Carrón angular-offset diagnostics.")
    p.add_argument("--make-raw-vector-pa-diagnostics", action=argparse.BooleanOptionalAction, default=True,
                   help="Direct point-level SCONCE/Carrón vector PA convention diagnostic in matched shells.")
    p.add_argument("--pa-vector-match-radius-mpc-h", type=float, default=2.0,
                   help="Nearest-neighbour radius for raw vector PA diagnostic.")
    p.add_argument("--pa-vector-max-carron-points-per-shell", type=int, default=12000,
                   help="Maximum Carrón raw points sampled per matched shell for direct vector PA diagnostics.")
    p.add_argument("--pa-diagnostic-carron-offsets", nargs="+", type=float, default=[0.0, 90.0, 180.0],
                   help="Carrón angle offsets tested against the fiducial SCONCE run for PA consistency.")
    p.add_argument("--use-native-galleries", action=argparse.BooleanOptionalAction, default=True,
                   help="Use the original in-memory slice/KDTree gallery implemented inside each pipeline helper. This is the recommended diagnostic gallery.")
    p.add_argument("--make-single-pipeline-galleries", action=argparse.BooleanOptionalAction, default=False,
                   help="Deprecated plot-space gallery from the driver; kept off by default because the native helper galleries are more reliable.")
    p.add_argument("--make-combined-gallery", action=argparse.BooleanOptionalAction, default=False,
                   help="Make driver-level SCONCE/Carrón combined galleries. Off by default; individual galleries are the primary diagnostics.")
    p.add_argument("--make-best-transform-gallery", action=argparse.BooleanOptionalAction, default=False,
                   help="Make SCONCE/Carrón galleries applying the best local-plane transform to Carrón in plot space. Off by default.")
    p.add_argument("--gallery-carron-match-sep-mpc-h", type=float, default=4.0)
    p.add_argument("--gallery-carron-match-dchi-mpc-h", type=float, default=8.0)
    p.add_argument("--gallery-carron-match-dz", type=float, default=0.01)
    p.add_argument("--gallery-chen-match-sep-mpc-h", type=float, default=3.0)
    p.add_argument("--gallery-chen-match-dchi-mpc-h", type=float, default=3.0)
    p.add_argument("--gallery-chen-match-radius-3d-mpc-h", type=float, default=3.0)
    p.add_argument("--gallery-chen-match-dz", type=float, default=0.015)
    p.add_argument("--clean-legacy-plots", action=argparse.BooleanOptionalAction, default=True,
                   help="Remove per-catalogue diagnostic plot files created by helper scripts; keep CSV/FITS tables and the final joint plots/gallery.")
    p.add_argument("--make-random-fans-gallery", action=argparse.BooleanOptionalAction, default=False,
                   help="Make PNG galleries of random FANS grouped by number of valid arms for Carron and Chen.")
    p.add_argument("--random-fans-gallery-n-per-arm-count", type=int, default=1000,
                   help="Maximum number of random FANS plotted per catalogue and arm-count class.")
    p.add_argument("--random-fans-gallery-arm-counts", nargs="+", type=int, default=[2, 3],
                   help="Arm-count classes to sample for the random FANS gallery.")

    p.add_argument("--skip-sconce", action="store_true")
    p.add_argument("--skip-carron", action="store_true")
    p.add_argument("--skip-chen", action="store_true")
    p.add_argument("--skip-plots", action="store_true")
    return p.parse_args()



def resolve_effective_point_weight(args: argparse.Namespace) -> str:
    """Resolve analysis-product and uncertainty flags into a helper point-weight mode."""
    product = str(args.analysis_product).lower()
    if product == "fiducial":
        return "density"
    if product == "uniform_geometry":
        return "uniform"
    if product == "uncertainty_weighted":
        return "inv_unc2"
    if product == "hybrid":
        return "density_inv_unc2"

    # custom mode: combine --point-weight and --arms-segment-uncertainty.
    point = str(args.point_weight).lower()
    unc = str(args.arms_segment_uncertainty).lower()
    if unc == "off":
        return point
    if unc == "inv_unc2":
        if point == "density":
            return "density_inv_unc2"
        if point == "uniform":
            return "inv_unc2"
        if point in {"inv_unc2", "density_inv_unc2"}:
            return point
        print(f"[warn] --arms-segment-uncertainty inv_unc2 with --point-weight {point}; using inv_unc2.")
        return "inv_unc2"
    return point


def main() -> None:
    args = parse_args()
    t0 = time.perf_counter()

    selected_pipelines = set(str(x).lower() for x in args.pipelines)

    # Official convention fixed after the raw-vector SCONCE/Carrón PA check:
    #   SCONCE grad_Dir -> tangent
    #   Carrón angle -> E=cos(theta), N=sin(theta), i.e. offset 0 deg
    #   Chen -> physical convention E=v_ra cos(dec), N=v_dec
    # These are now forced for the science run to avoid accidental drift.
    if str(args.sconce_gradient_mode).lower() != "tangent":
        print(f"[convention] overriding --sconce-gradient-mode {args.sconce_gradient_mode!r} -> 'tangent'")
    if abs(float(args.carron_angle_offset_deg)) > 1e-12:
        print(f"[convention] overriding --carron-angle-offset-deg {args.carron_angle_offset_deg!r} -> 0")
    args.sconce_gradient_mode = "tangent"
    args.carron_angle_offset_deg = 0.0
    print("[convention] official: SCONCE=tangent; Carrón angle_offset=0; Chen=physical")

    # Development-only PA diagnostics and combined overlay galleries are not
    # part of the organized public pipeline.
    args.make_pa_diagnostics = False
    args.make_raw_vector_pa_diagnostics = False
    args.make_combined_gallery = False
    args.make_best_transform_gallery = False
    args.make_single_pipeline_galleries = False
    effective_point_weight = resolve_effective_point_weight(args)
    print(f"[config] analysis_product={args.analysis_product}; point_weight={args.point_weight}; "
          f"arms_segment_uncertainty={args.arms_segment_uncertainty}; effective_point_weight={effective_point_weight}")
    args.skip_sconce = bool(args.skip_sconce or ("sconce" not in selected_pipelines))
    args.skip_carron = bool(args.skip_carron or ("carron" not in selected_pipelines))
    args.skip_chen = bool(args.skip_chen or ("chen" not in selected_pipelines))

    outroot = Path(args.outroot).expanduser().resolve() / args.run_name
    outroot.mkdir(parents=True, exist_ok=True)
    helper_dir = Path(__file__).resolve().parent

    sconce_script = resolve_script(args.sconce_script, "sconce_node_spirality.py", helper_dir=helper_dir)
    carron_script = resolve_script(args.carron_proxy_script, "carron_proxy_spirality.py", helper_dir=helper_dir)
    chen_script = resolve_script(args.chen_script, "chen_node_spirality.py", helper_dir=helper_dir)
    chen_arm_post = resolve_script(args.chen_arm_postprocess_script, "postprocess_chen_arm_spirality_zbins.py", helper_dir=helper_dir)

    sconce_out = outroot / "sconce_dr17_native"
    carron_root = outroot / "carron_dr17_proxy"
    carron_out = carron_root / "carron_proxy_ridges"
    chen_out = outroot / "chen_dr12_native"
    chen_arm_out = outroot / "chen_dr12_native_arms"
    plots_out = outroot / "three_catalog_plots"
    gallery_out = outroot / "triple_gallery"
    outroot.mkdir(parents=True, exist_ok=True)
    log_file = outroot / "run_commands.log"
    if log_file.exists() and args.overwrite:
        log_file.unlink()

    config_payload = vars(args).copy()
    config_payload["effective_point_weight"] = effective_point_weight
    with open(outroot / "driver_config.json", "w", encoding="utf-8") as f:
        json.dump(config_payload, f, indent=2)

    # Clean selected run directories unless explicitly reusing existing products.
    if args.reuse_existing_products:
        print("[reuse] keeping existing pipeline products; heavy runs are skipped when CSV products exist")
        for d in [sconce_out, carron_root, chen_out, chen_arm_out]:
            if not args.dry_run:
                d.mkdir(parents=True, exist_ok=True)
    else:
        if not args.skip_sconce:
            prepare_dir(sconce_out, overwrite=args.overwrite, dry_run=args.dry_run)
        if not args.skip_carron:
            prepare_dir(carron_root, overwrite=args.overwrite, dry_run=args.dry_run)
        if not args.skip_chen:
            prepare_dir(chen_out, overwrite=args.overwrite, dry_run=args.dry_run)
            prepare_dir(chen_arm_out, overwrite=args.overwrite, dry_run=args.dry_run)

    active_pipeline_count = int((not args.skip_sconce) + (not args.skip_carron) + (not args.skip_chen))
    if args.jobs_per_pipeline is not None:
        pipeline_n_jobs = max(1, int(args.jobs_per_pipeline))
    elif args.parallel_pipelines and active_pipeline_count > 1:
        pipeline_n_jobs = max(1, int(args.n_jobs) // active_pipeline_count)
    else:
        pipeline_n_jobs = int(args.n_jobs)

    pipeline_workers = int(args.pipeline_workers) if args.pipeline_workers is not None else active_pipeline_count
    if not args.parallel_pipelines:
        pipeline_workers = 1
    pipeline_workers = max(1, min(pipeline_workers, max(1, active_pipeline_count)))

    jobs_by_pipeline = {
        "sconce": max(1, int(args.sconce_jobs)) if args.sconce_jobs is not None else pipeline_n_jobs,
        "carron": max(1, int(args.carron_jobs)) if args.carron_jobs is not None else pipeline_n_jobs,
        "chen": max(1, int(args.chen_jobs)) if args.chen_jobs is not None else pipeline_n_jobs,
    }

    print(f"[parallel] parallel_pipelines={args.parallel_pipelines}; "
          f"active_pipelines={active_pipeline_count}; workers={pipeline_workers}; "
          f"default_n_jobs_per_pipeline={pipeline_n_jobs}; "
          f"jobs_by_pipeline={jobs_by_pipeline}")

    pipeline_tasks: List[Tuple[str, List[str], List[List[str]]]] = []

    if not args.skip_sconce:
        if args.reuse_existing_products and existing_pipeline_products_ok(sconce_out):
            print(f"[reuse] SCONCE products found; skipping heavy SCONCE run: {sconce_out}")
        else:
            cmd = [
                sys.executable, str(sconce_script),
                "--sconce-fits", str(args.sconce_fits),
                "--outdir", str(sconce_out),
                "--z-min", str(args.sconce_z_min),
                "--z-max", str(args.sconce_z_max),
                "--z-selection", "mid",
                "--zbin-edges", str(args.sconce_zbin_edges),
                "--n-arm-segments", str(args.n_arm_segments),
                "--segment-length-mpc-h", str(args.segment_length_mpc_h),
                "--node-link-radius-mpc-h", str(args.node_link_radius_mpc_h),
                "--edge-margin-mpc-h", str(args.edge_margin_mpc_h),
                "--knot-cluster-radius-mpc-h", str(args.knot_cluster_radius_mpc_h),
                "--knot-cluster-weight", str(args.knot_cluster_weight),
                "--arm-opening-deg", str(args.arm_opening_deg),
                "--cluster-gap-deg", str(args.cluster_gap_deg),
                "--seed-max-tangent-angle-deg", str(args.seed_max_tangent_angle_deg),
                "--min-seed-points", str(args.min_seed_points),
                "--min-points-per-arm-segment", str(args.min_points_per_arm_segment),
                "--min-valid-arm-segments", str(args.min_valid_arm_segments),
                "--min-arms-node", str(args.min_arms_node),
                "--max-arms-node", str(args.max_arms_node),
                "--sconce-gradient-mode", str(args.sconce_gradient_mode),
                "--point-weight", str(effective_point_weight),
                "--point-unc-floor", str(args.point_unc_floor),
                "--point-unc-max-weight", str(args.point_unc_max_weight),
                "--node-weight", str(args.node_weight),
                "--bootstrap-B", str(args.bootstrap_B),
                "--signflip-B", str(args.signflip_B),
                "--seed", str(args.seed),
                "--n-jobs", str(jobs_by_pipeline["sconce"]),
                "--zbin-plot-format", str(args.plot_format),
                "--plot-format", str(args.plot_format),
            ]
            if args.make_gallery and args.use_native_galleries:
                cmd += [
                    "--make-gallery",
                    "--gallery-seed", str(args.gallery_seed),
                    "--gallery-ncols", str(args.gallery_ncols),
                    "--gallery-nrows", str(args.gallery_nrows),
                    "--gallery-filename", "sconce_node_spiral_gallery",
                ]
            pipeline_tasks.append(("sconce", cmd, []))

    if not args.skip_carron:
        if args.reuse_existing_products and existing_pipeline_products_ok(carron_out):
            print(f"[reuse] Carrón products found; skipping heavy Carrón run: {carron_out}")
        else:
            cmd = [
                sys.executable, str(carron_script),
                "--sconce-pipeline-script", str(sconce_script),
                "--sconce-fits", str(args.sconce_fits),
                "--carron-files", *map(str, args.carron_files),
                "--outdir", str(carron_root),
                "--z-min", str(args.carron_z_min),
                "--z-max", str(args.carron_z_max),
                "--zbin-edges", str(args.carron_zbin_edges),
                "--n-arm-segments", str(args.n_arm_segments),
                "--segment-length-mpc-h", str(args.segment_length_mpc_h),
                "--node-link-radius-mpc-h", str(args.node_link_radius_mpc_h),
                "--edge-margin-mpc-h", str(args.edge_margin_mpc_h),
                "--knot-cluster-radius-mpc-h", str(args.knot_cluster_radius_mpc_h),
                "--knot-cluster-weight", str(args.knot_cluster_weight),
                "--arm-opening-deg", str(args.arm_opening_deg),
                "--cluster-gap-deg", str(args.cluster_gap_deg),
                "--seed-max-tangent-angle-deg", str(args.seed_max_tangent_angle_deg),
                "--min-seed-points", str(args.min_seed_points),
                "--min-points-per-arm-segment", str(args.min_points_per_arm_segment),
                "--min-valid-arm-segments", str(args.min_valid_arm_segments),
                "--min-arms-node", str(args.min_arms_node),
                "--max-arms-node", str(args.max_arms_node),
                "--point-weight", str(effective_point_weight),
                "--point-unc-floor", str(args.point_unc_floor),
                "--point-unc-max-weight", str(args.point_unc_max_weight),
                "--node-weight", str(args.node_weight),
                "--match-min-overlap", str(args.match_min_overlap),
                "--match-max-dchi-mpc", str(args.match_max_dchi_mpc),
                "--carron-angle-offset-deg", str(args.carron_angle_offset_deg),
                "--bootstrap-B", str(args.bootstrap_B),
                "--signflip-B", str(args.signflip_B),
                "--seed", str(args.seed),
                "--n-jobs", str(jobs_by_pipeline["carron"]),
                "--plot-format", str(args.plot_format),
                "--skip-sconce-ridge-measurement",
            ]
            if args.make_gallery and args.use_native_galleries:
                cmd += [
                    "--make-gallery",
                    "--gallery-seed", str(args.gallery_seed),
                    "--gallery-ncols", str(args.gallery_ncols),
                    "--gallery-nrows", str(args.gallery_nrows),
                    "--gallery-filename", "carron_node_spiral_gallery",
                ]
                if args.make_random_fans_gallery:
                    cmd += [
                        "--make-random-fans-gallery",
                        "--random-fans-gallery-outdir", str(outroot / "random_fans_gallery"),
                        "--random-fans-gallery-catalog-key", "carron",
                        "--random-fans-gallery-n-per-arm-count", str(args.random_fans_gallery_n_per_arm_count),
                        "--random-fans-gallery-arm-counts", *map(str, args.random_fans_gallery_arm_counts),
                    ]
            pipeline_tasks.append(("carron", cmd, []))

    if not args.skip_chen:
        if args.reuse_existing_products and existing_pipeline_products_ok(chen_out):
            print(f"[reuse] Chen products found; skipping heavy Chen run and Chen arm postprocess: {chen_out}")
        else:
            cmd = [
                sys.executable, str(chen_script),
                "--intersections", str(args.chen_intersections),
                "--fmaps", str(args.chen_fmaps),
                "--outdir", str(chen_out),
                "--omega-m", "0.3153",
                "--chen-direction-convention", "physical",
                "--cluster-gap-deg", str(args.cluster_gap_deg),
                "--seed-max-tangent-angle-deg", str(args.seed_max_tangent_angle_deg),
                "--min-seed-points", str(args.min_seed_points),
                "--min-points-per-arm-segment", str(args.min_points_per_arm_segment),
                "--min-valid-arm-segments", str(args.min_valid_arm_segments),
                "--min-arms-node", str(args.min_arms_node),
                "--max-arms-node", str(args.max_arms_node),
                "--zbin-plot-format", str(args.plot_format),
                "--plot-format", str(args.plot_format),
                "--node-weight", str(args.node_weight),
                "--point-weight", str(effective_point_weight),
                "--point-unc-floor", str(args.point_unc_floor),
                "--point-unc-max-weight", str(args.point_unc_max_weight),
                "--z-min", str(args.chen_z_min),
                "--z-max", str(args.chen_z_max),
                "--zbin-edges", str(args.chen_zbin_edges),
                "--n-arm-segments", str(args.n_arm_segments),
                "--segment-length-mpc-h", str(args.segment_length_mpc_h),
                "--node-link-radius-mpc-h", str(args.node_link_radius_mpc_h),
                "--edge-margin-mpc-h", str(args.edge_margin_mpc_h),
                "--arm-opening-deg", str(args.arm_opening_deg),
                "--bootstrap-B", str(args.bootstrap_B),
                "--signflip-B", str(args.signflip_B),
                "--seed", str(args.seed),
                "--n-jobs", str(jobs_by_pipeline["chen"]),
            ]
            if args.make_gallery and args.use_native_galleries:
                cmd += [
                    "--make-gallery",
                    "--gallery-seed", str(args.gallery_seed),
                    "--gallery-ncols", str(args.gallery_ncols),
                    "--gallery-nrows", str(args.gallery_nrows),
                    "--gallery-filename", "chen_node_spiral_gallery",
                ]
                if args.make_random_fans_gallery:
                    cmd += [
                        "--make-random-fans-gallery",
                        "--random-fans-gallery-outdir", str(outroot / "random_fans_gallery"),
                        "--random-fans-gallery-catalog-key", "chen",
                        "--random-fans-gallery-n-per-arm-count", str(args.random_fans_gallery_n_per_arm_count),
                        "--random-fans-gallery-arm-counts", *map(str, args.random_fans_gallery_arm_counts),
                    ]

            post_cmds: List[List[str]] = []
            for arm_weight_mode in ["uniform", "mean_density"]:
                post_cmds.append([
                    sys.executable, str(chen_arm_post),
                    "--arm-spirality", str(chen_out / "arm_spirality.csv"),
                    "--outdir", str(chen_arm_out),
                    "--zbin-edges", str(args.chen_zbin_edges),
                    "--weight-mode", arm_weight_mode,
                    "--bootstrap-B", str(args.bootstrap_B),
                    "--signflip-B", str(args.signflip_B),
                    "--seed", str(args.seed),
                ])
            pipeline_tasks.append(("chen", cmd, post_cmds))

    run_pipeline_tasks(
        pipeline_tasks,
        workers=pipeline_workers,
        dry_run=args.dry_run,
        log_file=log_file,
        timing_csv=outroot / "pipeline_wall_times.csv",
    )

    if args.dry_run:
        print("[dry-run] not making plots/gallery")
        return

    if not args.dry_run:
        if not args.skip_sconce and sconce_out.exists():
            rewrite_rigorous_summaries_for_pipeline(
                pipeline_dir=sconce_out,
                arm_summary_dir=sconce_out,
                zbin_edges=args.sconce_zbin_edges,
                arm_summary_weights=list(args.arm_summary_weights),
                catalog_key="sconce",
                B=args.bootstrap_B,
                seed=args.seed,
                jackknife_nside=args.summary_jackknife_nside,
                error_mode=args.summary_error_model,
                bootstrap_mode=args.summary_bootstrap_mode,
                bootstrap_nside=args.summary_bootstrap_nside,
            )
        if not args.skip_carron and carron_out.exists():
            rewrite_rigorous_summaries_for_pipeline(
                pipeline_dir=carron_out,
                arm_summary_dir=carron_out,
                zbin_edges=args.carron_zbin_edges,
                arm_summary_weights=list(args.arm_summary_weights),
                catalog_key="carron",
                B=args.bootstrap_B,
                seed=args.seed + 11,
                jackknife_nside=args.summary_jackknife_nside,
                error_mode=args.summary_error_model,
                bootstrap_mode=args.summary_bootstrap_mode,
                bootstrap_nside=args.summary_bootstrap_nside,
            )
        if not args.skip_chen and chen_out.exists():
            rewrite_rigorous_summaries_for_pipeline(
                pipeline_dir=chen_out,
                arm_summary_dir=chen_arm_out,
                zbin_edges=args.chen_zbin_edges,
                arm_summary_weights=list(args.arm_summary_weights),
                catalog_key="chen",
                B=args.bootstrap_B,
                seed=args.seed + 23,
                jackknife_nside=args.summary_jackknife_nside,
                error_mode=args.summary_error_model,
                bootstrap_mode=args.summary_bootstrap_mode,
                bootstrap_nside=args.summary_bootstrap_nside,
            )

    if not args.skip_plots:
        make_joint_plots(
            sconce_dir=sconce_out,
            carron_dir=carron_out,
            chen_node_dir=chen_out,
            chen_arm_dir=chen_arm_out,
            outdir=plots_out,
            fmt=args.plot_format,
            zmin=args.plot_z_min,
            zmax=args.plot_z_max,
            filter_mode=args.plot_bin_filter_mode,
        )
        plot_sign_asymmetry(
            sconce_dir=sconce_out,
            carron_dir=carron_out,
            chen_dir=chen_out,
            outdir=plots_out,
            fmt=args.plot_format,
            sconce_edges=args.sconce_zbin_edges,
            carron_edges=args.carron_zbin_edges,
            chen_edges=args.chen_zbin_edges,
        )

    if bool(args.make_random_fans_gallery) and not args.dry_run and bool(args.reuse_existing_products):
        # Fallback for reuse-only runs, where the native in-memory slice/KDTree
        # gallery cannot be reconstructed.  Full runs generate the preferred
        # native-style individual PNGs inside the catalogue helper scripts.
        gallery_root = outroot / "random_fans_gallery_from_csv"
        print("[gallery-random] reuse-existing-products=True: using CSV fallback; native-style PNGs require a full catalogue run")
        make_random_fans_gallery(
            catalog_key="carron", catalog_label="Carrón et al. SDSS DR16",
            pipeline_dir=carron_out, outroot=gallery_root,
            arm_counts=list(args.random_fans_gallery_arm_counts),
            n_per_count=int(args.random_fans_gallery_n_per_arm_count),
            seed=int(args.gallery_seed) + 500,
        )
        make_random_fans_gallery(
            catalog_key="chen", catalog_label="Chen et al. SDSS DR12",
            pipeline_dir=chen_out, outroot=gallery_root,
            arm_counts=list(args.random_fans_gallery_arm_counts),
            n_per_count=int(args.random_fans_gallery_n_per_arm_count),
            seed=int(args.gallery_seed) + 700,
        )

    if args.clean_legacy_plots and not args.dry_run:
        for d in [sconce_out, carron_root, chen_out, chen_arm_out]:
            prune_legacy_plot_files(d)

    dt = time.perf_counter() - t0
    print("\n[done]")
    print(f"  output root: {outroot}")
    print(f"  SCONCE native: {sconce_out}")
    print(f"  Carrón proxy:  {carron_out}")
    print(f"  Chen native:   {chen_out}")
    print(f"  Chen arms:     {chen_arm_out}  [zbin_summary_arms_n_segments.csv]")
    print(f"  plots:         {plots_out}")
    print("  native galleries: generated inside each catalogue output directory when --make-gallery is enabled")
    print(f"  command log:   {log_file}")
    print(f"  wall time:     {dt/60:.2f} min")


if __name__ == "__main__":
    main()
