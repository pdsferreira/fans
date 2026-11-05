#!/usr/bin/env python3
"""Summarize FANS object and arm-count statistics for a completed run.

The main pipeline writes detailed per-catalogue products such as
``node_spirality.csv``, ``node_status_summary.csv`` and arm-level tables.  This
small post-processing script collects the quantities most often needed in the
paper text/caption into two consolidated CSV files:

  * ``fans_count_summary.csv``: number of candidate/accepted FANS, numbers with
    2, 3, 4, and >=5 valid arms, total number of valid arms, and arm-count
    moments, for each catalogue, before and after the quality/status cut.
  * ``fans_status_summary_long.csv``: status counts before the quality/status
    cut, globally and by redshift bin.

The two stages here are processed_nodes_post_arm_selection and accepted_fans.
These are NOT a pre-arm-selection connectivity distribution. See
referee_multiplicity/<catalog>/multiplicity_selection_stages.csv for
separately evaluated candidate groups, trace survivors and unassessed nodes.
Unknown multiplicity is never counted as zero.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

CATALOG_SPECS = {
    "sconce": {
        "label": "SCONCE DR17 native",
        "node_candidates": [Path("sconce_dr17_native/node_spirality.csv")],
    },
    "carron": {
        "label": "Carrón SDSS DR16 proxy",
        "node_candidates": [
            Path("carron_dr17_proxy/carron_proxy_ridges/node_spirality.csv"),
            Path("carron_dr16_proxy/carron_proxy_ridges/node_spirality.csv"),
        ],
    },
    "chen": {
        "label": "Chen SDSS DR12 native",
        "node_candidates": [Path("chen_dr12_native/node_spirality.csv"), Path("chen_dr12_native_arms/node_spirality.csv")],
    },
}


def _parse_edges(text: str | None) -> Optional[np.ndarray]:
    if text is None or str(text).strip().lower() in {"", "none", "null"}:
        return None
    vals = [float(x) for x in str(text).replace(";", ",").split(",") if x.strip()]
    if len(vals) < 2:
        return None
    return np.asarray(vals, dtype=float)


def _choose_z_column(df: pd.DataFrame) -> Optional[str]:
    for c in ["z_mid", "z", "z_low", "z_center"]:
        if c in df.columns:
            vals = pd.to_numeric(df[c], errors="coerce")
            if np.isfinite(vals.to_numpy(float)).any():
                return c
    return None


def _arm_counts(df: pd.DataFrame) -> pd.Series:
    if "n_arms_valid" in df.columns:
        return pd.to_numeric(df["n_arms_valid"], errors="coerce")
    if "n_arms" in df.columns:
        return pd.to_numeric(df["n_arms"], errors="coerce")
    return pd.Series(np.nan, index=df.index, dtype=float)


def _status_ok_mask(df: pd.DataFrame) -> np.ndarray:
    if "status" not in df.columns:
        raise ValueError("Node status is required; refusing an ambiguous selection")
    return df["status"].astype(str).eq("ok").to_numpy(bool)


def _summarize_subset(
    sub: pd.DataFrame,
    *,
    catalog_key: str,
    catalog_label: str,
    stage: str,
    scope: str,
    zbin_id: int,
    z_min: float,
    z_max: float,
) -> Dict[str, object]:
    n = int(len(sub))
    narms = _arm_counts(sub)
    known = np.isfinite(narms.to_numpy(float))
    narms_int = narms.where(known).round()

    def count_eq(k: int) -> int:
        return int(np.count_nonzero(known & narms_int.eq(k).to_numpy(bool)))

    n0 = count_eq(0)
    n1 = count_eq(1)
    n2 = count_eq(2)
    n3 = count_eq(3)
    n4 = count_eq(4)
    n5p = int(np.count_nonzero(known & narms_int.ge(5).to_numpy(bool)))
    n_unknown = int(np.count_nonzero(~known))

    vals = narms.to_numpy(float)
    vals = vals[np.isfinite(vals)]
    total_valid_arms = int(np.nansum(vals)) if len(vals) else 0

    return {
        "catalog_key": catalog_key,
        "catalog_label": catalog_label,
        "stage": stage,
        "scope": scope,
        "zbin_id": int(zbin_id),
        "z_min": float(z_min) if np.isfinite(z_min) else np.nan,
        "z_max": float(z_max) if np.isfinite(z_max) else np.nan,
        "n_objects_total": n,
        "n_objects_with_known_narms": int(np.count_nonzero(known)),
        "n_objects_unknown_narms": n_unknown,
        "n_objects_0arms": n0,
        "n_objects_1arm": n1,
        "n_objects_2arms": n2,
        "n_objects_3arms": n3,
        "n_objects_4arms": n4,
        "n_objects_ge5arms": n5p,
        "n_objects_ge2arms": int(np.count_nonzero(known & narms_int.ge(2).to_numpy(bool))),
        "total_valid_arms": total_valid_arms,
        "mean_n_arms_valid": float(np.nanmean(vals)) if len(vals) else np.nan,
        "median_n_arms_valid": float(np.nanmedian(vals)) if len(vals) else np.nan,
        "min_n_arms_valid": float(np.nanmin(vals)) if len(vals) else np.nan,
        "max_n_arms_valid": float(np.nanmax(vals)) if len(vals) else np.nan,
    }


def _status_rows(
    sub: pd.DataFrame,
    *,
    catalog_key: str,
    catalog_label: str,
    scope: str,
    zbin_id: int,
    z_min: float,
    z_max: float,
) -> List[Dict[str, object]]:
    if "status" not in sub.columns:
        return []
    total = int(len(sub))
    rows: List[Dict[str, object]] = []
    for status, n in sub["status"].astype(str).value_counts(dropna=False).items():
        rows.append({
            "catalog_key": catalog_key,
            "catalog_label": catalog_label,
            "scope": scope,
            "zbin_id": int(zbin_id),
            "z_min": float(z_min) if np.isfinite(z_min) else np.nan,
            "z_max": float(z_max) if np.isfinite(z_max) else np.nan,
            "status": status,
            "n_objects": int(n),
            "n_objects_total_scope": total,
            "fraction_of_scope": float(int(n) / max(total, 1)),
        })
    return rows


def summarize_catalog(
    run_root: Path,
    catalog_key: str,
    *,
    z_edges: Optional[np.ndarray],
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    spec = CATALOG_SPECS[catalog_key]
    node_path = None
    for rel in spec["node_candidates"]:
        candidate = run_root / rel
        if candidate.exists():
            node_path = candidate
            break
    if node_path is None:
        print(f"[warn] missing {catalog_key}: none of {spec['node_candidates']} under {run_root}")
        return [], []

    df = pd.read_csv(node_path)
    label = str(spec["label"])
    print(f"[read] {catalog_key}: {node_path} ({len(df):,} rows)")

    rows_counts: List[Dict[str, object]] = []
    rows_status: List[Dict[str, object]] = []

    ok_mask = _status_ok_mask(df)
    stages = [
        ("processed_nodes_post_arm_selection", np.ones(len(df), dtype=bool)),
        ("accepted_fans", ok_mask),
    ]

    zcol = _choose_z_column(df)
    zvals = pd.to_numeric(df[zcol], errors="coerce").to_numpy(float) if zcol is not None else np.full(len(df), np.nan)
    zmin_all = float(np.nanmin(zvals)) if np.any(np.isfinite(zvals)) else np.nan
    zmax_all = float(np.nanmax(zvals)) if np.any(np.isfinite(zvals)) else np.nan

    for stage, smask in stages:
        sub = df.loc[smask].copy()
        rows_counts.append(_summarize_subset(
            sub,
            catalog_key=catalog_key,
            catalog_label=label,
            stage=stage,
            scope="allz",
            zbin_id=-1,
            z_min=zmin_all,
            z_max=zmax_all,
        ))

    rows_status.extend(_status_rows(
        df,
        catalog_key=catalog_key,
        catalog_label=label,
        scope="allz",
        zbin_id=-1,
        z_min=zmin_all,
        z_max=zmax_all,
    ))

    if z_edges is not None and len(z_edges) >= 2 and zcol is not None:
        for ib in range(len(z_edges) - 1):
            z0 = float(z_edges[ib])
            z1 = float(z_edges[ib + 1])
            if ib == len(z_edges) - 2:
                zmask = np.isfinite(zvals) & (zvals >= z0) & (zvals <= z1)
            else:
                zmask = np.isfinite(zvals) & (zvals >= z0) & (zvals < z1)

            for stage, smask in stages:
                sub = df.loc[zmask & smask].copy()
                rows_counts.append(_summarize_subset(
                    sub,
                    catalog_key=catalog_key,
                    catalog_label=label,
                    stage=stage,
                    scope="zbin",
                    zbin_id=ib,
                    z_min=z0,
                    z_max=z1,
                ))

            rows_status.extend(_status_rows(
                df.loc[zmask].copy(),
                catalog_key=catalog_key,
                catalog_label=label,
                scope="zbin",
                zbin_id=ib,
                z_min=z0,
                z_max=z1,
            ))

    return rows_counts, rows_status


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-root", required=True, help="Completed pipeline run directory, e.g. .../v66_fiducial_L10_N3_R10_A35")
    p.add_argument("--outdir", default=None, help="Output directory. Default: <run-root>/paper_tables")
    p.add_argument("--catalogs", nargs="+", default=["carron", "chen"], choices=list(CATALOG_SPECS))
    p.add_argument("--zbin-edges", default="0.1,0.225,0.35,0.475,0.6")
    args = p.parse_args()

    run_root = Path(args.run_root).expanduser().resolve()
    outdir = Path(args.outdir).expanduser().resolve() if args.outdir else (run_root / "paper_tables")
    outdir.mkdir(parents=True, exist_ok=True)
    z_edges = _parse_edges(args.zbin_edges)

    all_counts: List[Dict[str, object]] = []
    all_status: List[Dict[str, object]] = []
    for cat in args.catalogs:
        counts, status = summarize_catalog(run_root, cat, z_edges=z_edges)
        all_counts.extend(counts)
        all_status.extend(status)

    counts_df = pd.DataFrame(all_counts)
    status_df = pd.DataFrame(all_status)

    counts_path = outdir / "fans_count_summary.csv"
    status_path = outdir / "fans_status_summary_long.csv"
    counts_df.to_csv(counts_path, index=False)
    status_df.to_csv(status_path, index=False)

    print("[write]", counts_path)
    print("[write]", status_path)

    # A compact all-z table that is convenient for copy/paste into notes.
    compact = counts_df[(counts_df["scope"] == "allz")].copy()
    compact_path = outdir / "fans_count_summary_allz_compact.csv"
    compact.to_csv(compact_path, index=False)
    print("[write]", compact_path)


if __name__ == "__main__":
    main()
