#!/usr/bin/env python3
"""
Compare reference and candidate spirality outputs.

Temporary validation tool for refactoring. It is intentionally kept outside the
spirality package.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd


FILES_BY_PRODUCT = {
    "sconce_dr17_native": [
        "node_spirality.csv",
        "arm_level_spirality.csv",
        "arm_segments.csv",
        "zbin_summary.csv",
        "zbin_summary_arms_uniform.csv",
        "zbin_summary_arms_mean_density.csv",
    ],
    "carron_dr17_proxy/carron_proxy_ridges": [
        "node_spirality.csv",
        "arm_level_spirality.csv",
        "arm_segments.csv",
        "zbin_summary.csv",
        "zbin_summary_arms_uniform.csv",
        "zbin_summary_arms_mean_density.csv",
    ],
    "chen_dr12_native": [
        "node_spirality.csv",
        "arm_level_spirality.csv",
        "arm_segments.csv",
        "zbin_summary.csv",
    ],
    "chen_dr12_native_arms": [
        "zbin_summary_arms_uniform.csv",
        "zbin_summary_arms_mean_density.csv",
    ],
}


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def sort_for_compare(df: pd.DataFrame) -> pd.DataFrame:
    keys = [c for c in [
        "zbin_id", "z_min", "z_max", "node_id", "arm_local_id",
        "arm_segment_index", "z_center", "ra", "dec"
    ] if c in df.columns]
    if keys:
        return df.sort_values(keys, kind="mergesort").reset_index(drop=True)
    return df.reset_index(drop=True)


def compare_csv(ref: Path, cand: Path, *, rtol: float, atol: float) -> dict:
    a = sort_for_compare(read_csv(ref))
    b = sort_for_compare(read_csv(cand))
    out = {
        "ok": True,
        "reference": str(ref),
        "candidate": str(cand),
        "n_ref": len(a),
        "n_candidate": len(b),
        "missing_columns": "",
        "extra_columns": "",
        "max_abs_diff": 0.0,
        "max_rel_diff": 0.0,
        "n_numeric_fail": 0,
        "n_text_fail": 0,
        "error": "",
    }
    if len(a) != len(b):
        out["ok"] = False
        return out

    missing = [c for c in a.columns if c not in b.columns]
    extra = [c for c in b.columns if c not in a.columns]
    out["missing_columns"] = ",".join(missing)
    out["extra_columns"] = ",".join(extra)
    if missing:
        out["ok"] = False
        return out

    b = b[a.columns].copy()

    for c in a.columns:
        av = a[c]
        bv = b[c]
        if pd.api.types.is_numeric_dtype(av) or pd.api.types.is_numeric_dtype(bv):
            ax = pd.to_numeric(av, errors="coerce").to_numpy(float)
            bx = pd.to_numeric(bv, errors="coerce").to_numpy(float)
            ok = np.isclose(ax, bx, rtol=rtol, atol=atol, equal_nan=True)
            out["n_numeric_fail"] += int(np.count_nonzero(~ok))
            both_nan = ~np.isfinite(ax) & ~np.isfinite(bx)
            diff = np.abs(ax - bx)
            diff[both_nan] = 0.0
            finite = np.isfinite(diff)
            if np.any(finite):
                out["max_abs_diff"] = max(out["max_abs_diff"], float(np.nanmax(diff[finite])))
                denom = np.maximum(np.abs(ax), atol)
                rel = diff / denom
                out["max_rel_diff"] = max(out["max_rel_diff"], float(np.nanmax(rel[finite])))
        else:
            ax = av.fillna("<NA>").astype(str).to_numpy()
            bx = bv.fillna("<NA>").astype(str).to_numpy()
            out["n_text_fail"] += int(np.count_nonzero(ax != bx))

    if out["n_numeric_fail"] or out["n_text_fail"] or extra:
        out["ok"] = False
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference-root", required=True)
    p.add_argument("--candidate-root", required=True)
    p.add_argument("--rtol", type=float, default=1e-9)
    p.add_argument("--atol", type=float, default=1e-12)
    p.add_argument("--out", default="comparison_report.csv")
    args = p.parse_args()

    ref_root = Path(args.reference_root).expanduser().resolve()
    cand_root = Path(args.candidate_root).expanduser().resolve()

    rows = []
    for subdir, files in FILES_BY_PRODUCT.items():
        for fname in files:
            ref = ref_root / subdir / fname
            cand = cand_root / subdir / fname
            try:
                row = compare_csv(ref, cand, rtol=args.rtol, atol=args.atol)
            except Exception as exc:
                row = {
                    "ok": False,
                    "reference": str(ref),
                    "candidate": str(cand),
                    "n_ref": np.nan,
                    "n_candidate": np.nan,
                    "missing_columns": "",
                    "extra_columns": "",
                    "max_abs_diff": np.nan,
                    "max_rel_diff": np.nan,
                    "n_numeric_fail": np.nan,
                    "n_text_fail": np.nan,
                    "error": repr(exc),
                }
            row["product_dir"] = subdir
            row["filename"] = fname
            rows.append(row)

    report = pd.DataFrame(rows)
    out = Path(args.out).expanduser().resolve()
    report.to_csv(out, index=False)
    print(f"[write] {out}")
    n_ok = int(report["ok"].fillna(False).sum())
    print(f"[summary] {n_ok}/{len(report)} files passed")
    bad = report[~report["ok"].fillna(False)]
    if len(bad):
        cols = ["product_dir", "filename", "n_ref", "n_candidate", "max_abs_diff", "max_rel_diff", "n_numeric_fail", "n_text_fail", "error"]
        print(bad[cols].to_string(index=False))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
