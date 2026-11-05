#!/usr/bin/env python3
"""Consolidate per-run public FANS FITS files into one multi-extension FITS.

The normal public export writes one file per catalogue per parameter choice, e.g.

    v67_grid_z010_z060_L10_N3_R10_A35/fans_fits/carron_fans_all.fits
    v67_grid_z010_z060_L10_N3_R10_A35/fans_fits/chen_fans_all.fits

This script combines those files into a single FITS file.  For each catalogue and
parameter choice it writes two extensions with names such as

    CARRON_L10_N3_R10_A35_NODES
    CARRON_L10_N3_R10_A35_SEGMENTS

The PRIMARY header contains a compact README/definition block.  Each extension
also stores CATALOG, LSEG, NSEG, RLINK and AOPEN header keywords so the extension
name does not have to be parsed downstream.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
from astropy.io import fits

RUN_RE = re.compile(r".*_L(?P<L>[^_]+)_N(?P<N>[^_]+)_R(?P<R>[^_]+)_A(?P<A>[^_]+).*")


def _parse_float_token(text: object) -> float:
    try:
        return float(str(text).replace("p", "."))
    except Exception:
        return np.nan


def _token_from_float(x: float) -> str:
    if not np.isfinite(x):
        return "NA"
    # Keep integers compact: 10 -> 10; decimals: 7.5 -> 7P5.
    if abs(x - round(x)) < 1e-8:
        return str(int(round(x)))
    return (f"{x:g}").replace(".", "P")


def _infer_params(run_root: Path) -> dict:
    m = RUN_RE.match(run_root.name)
    if not m:
        return {
            "lseg": np.nan,
            "nseg": np.nan,
            "rlink": np.nan,
            "aopen": np.nan,
            "tag": "UNKNOWN",
        }
    lseg = _parse_float_token(m.group("L"))
    nseg = int(round(_parse_float_token(m.group("N")))) if np.isfinite(_parse_float_token(m.group("N"))) else -1
    rlink = _parse_float_token(m.group("R"))
    aopen = _parse_float_token(m.group("A"))
    tag = f"L{_token_from_float(lseg)}_N{nseg}_R{_token_from_float(rlink)}_A{_token_from_float(aopen)}"
    return {"lseg": lseg, "nseg": nseg, "rlink": rlink, "aopen": aopen, "tag": tag}


def _find_run_dirs(grid_root: Path) -> List[Path]:
    return sorted([d for d in grid_root.iterdir() if d.is_dir() and (d / "fans_fits").exists()]) if grid_root.exists() else []


def _find_catalog_fits(run_root: Path, catalog: str, accepted_only: bool = False) -> Optional[Path]:
    suffix = "quality" if accepted_only else "all"
    candidates = [
        run_root / "fans_fits" / f"{catalog}_fans_{suffix}.fits",
        run_root / "fans_fits" / f"{catalog}_fans_all.fits",
        run_root / "fans_fits" / f"{catalog}_fans_quality.fits",
        run_root / "fans_fits" / f"{catalog}_fans_accepted.fits",
    ]
    for p in candidates:
        if p.exists():
            return p
    found = sorted((run_root / "fans_fits").glob(f"{catalog}_fans*.fits")) if (run_root / "fans_fits").exists() else []
    return found[0] if found else None


def _safe_extname(catalog: str, params_tag: str, table: str) -> str:
    ext = f"{catalog.upper()}_{params_tag}_{table.upper()}"
    # FITS allows long string values, but keep this comfortably short.
    return ext[:68]


def _add_table_metadata(hdu: fits.BinTableHDU, *, catalog: str, run_root: Path, params: dict, source: Path, table: str) -> fits.BinTableHDU:
    hdu.header["CATALOG"] = catalog
    hdu.header["RUNNAME"] = run_root.name[:68]
    hdu.header["TABLE"] = table
    if np.isfinite(params.get("lseg", np.nan)):
        hdu.header["LSEG"] = float(params["lseg"])
    if int(params.get("nseg", -1)) >= 0:
        hdu.header["NSEG"] = int(params["nseg"])
    if np.isfinite(params.get("rlink", np.nan)):
        hdu.header["RLINK"] = float(params["rlink"])
    if np.isfinite(params.get("aopen", np.nan)):
        hdu.header["AOPEN"] = float(params["aopen"])
    hdu.header["SOURCE"] = str(source)[-68:]
    hdu.header["QUALDEF"] = "QUALITY=1 accepted final sample; QUALITY=0 rejected candidate"
    return hdu


def _primary_hdu(grid_root: Optional[Path], run_roots: List[Path], catalogs: Iterable[str]) -> fits.PrimaryHDU:
    hdr = fits.Header()
    hdr["ORIGIN"] = "FANS spirality pipeline"
    hdr["PRODUCT"] = "Consolidated public FANS FITS"
    if grid_root is not None:
        hdr["GRIDROOT"] = str(grid_root)[-68:]
    hdr["NRUNS"] = len(run_roots)
    hdr["CATALOGS"] = ",".join(catalogs)[:68]
    hdr["COMMENT"] = "README: each catalogue/parameter choice has NODES and SEGMENTS extensions."
    hdr["COMMENT"] = "Extension names are <CAT>_L<LSEG>_N<NSEG>_R<RLINK>_A<AOPEN>_<TABLE>."
    hdr["COMMENT"] = "NODES: FANS_ID, RA, DEC, Z, NARMS, NSEGS, NPOINTS, HANDEDNESS, SPIRALITY, QUALITY."
    hdr["COMMENT"] = "SEGMENTS: SEG_ID, FANS_ID, ARM_ID, SEG_INDEX, RA, DEC, Z, NPOINTS, X/Y, T_PHI, T_RADIAL, WEIGHT, QUALITY."
    hdr["COMMENT"] = "QUALITY is unsigned byte: 1=accepted final sample; 0=rejected candidate."
    hdr["COMMENT"] = "HANDEDNESS is signed projected winding; SPIRALITY=abs(HANDEDNESS)."
    hdr["COMMENT"] = "Segment RA/DEC are approximate centroids; X_MPC_H/Y_MPC_H are native tangent-plane offsets."
    return fits.PrimaryHDU(header=hdr)


def consolidate(run_roots: List[Path], catalogs: List[str], output: Path, accepted_only: bool = False, overwrite: bool = False, grid_root: Optional[Path] = None) -> Path:
    hdus: List[fits.hdu.base.ExtensionHDU] = [_primary_hdu(grid_root, run_roots, catalogs)]  # type: ignore[type-arg]
    seen = set()
    n_added = 0

    for run_root in run_roots:
        params = _infer_params(run_root)
        for catalog in catalogs:
            src = _find_catalog_fits(run_root, catalog, accepted_only=accepted_only)
            if src is None:
                print(f"[skip] missing FITS: run={run_root.name} catalog={catalog}")
                continue
            with fits.open(src) as hdul:
                for table in ["NODES", "SEGMENTS"]:
                    if table not in hdul:
                        print(f"[skip] {src}: missing {table}")
                        continue
                    hdu = hdul[table].copy()
                    extname = _safe_extname(catalog, params["tag"], table)
                    if extname in seen:
                        # Extremely unlikely; add a short suffix if needed.
                        ii = 2
                        base = extname[:64]
                        while f"{base}_{ii}" in seen:
                            ii += 1
                        extname = f"{base}_{ii}"
                    seen.add(extname)
                    hdu.name = extname
                    _add_table_metadata(hdu, catalog=catalog, run_root=run_root, params=params, source=src, table=table)
                    hdus.append(hdu)
                    n_added += 1
                    print(f"[add] {extname:45s} rows={len(hdu.data) if hdu.data is not None else 0:,}")

    output.parent.mkdir(parents=True, exist_ok=True)
    fits.HDUList(hdus).writeto(output, overwrite=overwrite, checksum=True)
    print(f"[write] {output}  extensions={n_added}")
    return output


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--grid-root", help="Root containing many completed parameter-run directories.")
    g.add_argument("--run-root", help="Single completed parameter-run directory.")
    p.add_argument("--catalogs", nargs="+", default=["carron", "chen"], help="Catalogue keys to include, e.g. carron chen sconce.")
    p.add_argument("--output", required=True, help="Output consolidated FITS file.")
    p.add_argument("--accepted-only", action="store_true", help="Prefer *_fans_quality.fits / accepted-only FITS where available.")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    if args.grid_root:
        grid_root = Path(args.grid_root).expanduser().resolve()
        run_roots = _find_run_dirs(grid_root)
        if not run_roots:
            raise SystemExit(f"No run directories with fans_fits/ found under {grid_root}")
    else:
        grid_root = None
        run_roots = [Path(args.run_root).expanduser().resolve()]

    consolidate(run_roots, list(args.catalogs), Path(args.output).expanduser().resolve(), accepted_only=bool(args.accepted_only), overwrite=bool(args.overwrite), grid_root=grid_root)


if __name__ == "__main__":
    main()
