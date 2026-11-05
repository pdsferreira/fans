#!/usr/bin/env python3
"""Export identified FANS to compact binary FITS tables.

For each catalogue in a completed FANS run this script writes one FITS file with
three HDUs:

  * PRIMARY: run/catalogue metadata;
  * NODES: one row per node/FANS candidate;
  * SEGMENTS: one row per measured arm segment.

The FITS files are meant for public release and consistency checks.  To keep the
public data model compact, repeated catalogue strings, internal node IDs, raw
status labels, redshift-slice bounds, radial-coherence diagnostics, and
object-level uncertainty placeholders are not written as table columns.  The
integer ``QUALITY`` column records whether each FANS object passes the final
``status == 'ok'`` cut: ``1`` means accepted and ``0`` means rejected.  By default all reconstructed nodes/segments are written;
pass ``--accepted-only`` to write only ``QUALITY=1`` systems. Use ``--write-quality-copy`` to write both the full and quality-only public files in one command.

Segment RA/Dec values are reconstructed from the node position plus the local
projected segment centroid offsets (``x_mpc_h``, ``y_mpc_h``), using Planck18 to
convert the node redshift to a comoving angular-diameter distance in h^-1 Mpc.
The original local offsets are also kept, which is the lossless representation
used by the pipeline.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
try:
    from astropy.io import fits
    from astropy.cosmology import Planck18
except ImportError:
    fits = None
    Planck18 = None

CATALOG_SPECS: Dict[str, Dict[str, object]] = {
    "sconce": {
        "label": "SCONCE DR17 native",
        "dirs": [Path("sconce_dr17_native")],
    },
    "carron": {
        "label": "Carron SDSS proxy",
        "dirs": [
            Path("carron_dr17_proxy/carron_proxy_ridges"),
            Path("carron_dr16_proxy/carron_proxy_ridges"),
        ],
    },
    "chen": {
        "label": "Chen SDSS DR12 native",
        "dirs": [Path("chen_dr12_native"), Path("chen_dr12_native_arms")],
    },
}

RUN_RE = re.compile(r".*_L(?P<L>[^_]+)_N(?P<N>[^_]+)_R(?P<R>[^_]+)_A(?P<A>[^_]+).*")


def _parse_float_token(text: object) -> float:
    try:
        return float(str(text).replace("p", "."))
    except Exception:
        return np.nan


def _infer_run_parameters(run_root: Path) -> Dict[str, object]:
    m = RUN_RE.match(run_root.name)
    if not m:
        return {}
    return {
        "segment_length_mpc_h": _parse_float_token(m.group("L")),
        "n_arm_segments": int(_parse_float_token(m.group("N"))) if np.isfinite(_parse_float_token(m.group("N"))) else np.nan,
        "node_link_radius_mpc_h": _parse_float_token(m.group("R")),
        "arm_opening_deg": _parse_float_token(m.group("A")),
    }


def _find_catalog_dir(run_root: Path, catalog: str) -> Optional[Path]:
    for rel in CATALOG_SPECS[catalog]["dirs"]:  # type: ignore[index]
        d = run_root / rel
        if (d / "node_spirality.csv").exists() or (d / "arm_segments.csv").exists():
            return d
    return None


def _read_optional_csv(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def _choose_z(df: pd.DataFrame) -> pd.Series:
    for c in ["z_mid", "z", "z_center", "z_low"]:
        if c in df.columns:
            vals = pd.to_numeric(df[c], errors="coerce")
            if np.isfinite(vals.to_numpy(float)).any():
                return vals
    if {"z_low", "z_high"}.issubset(df.columns):
        return 0.5 * (pd.to_numeric(df["z_low"], errors="coerce") + pd.to_numeric(df["z_high"], errors="coerce"))
    return pd.Series(np.nan, index=df.index, dtype=float)


def _node_status_ok(df: pd.DataFrame) -> np.ndarray:
    if "status" not in df.columns:
        raise ValueError("Node status is required for FITS quality selection")
    return df["status"].astype(str).str.lower().eq("ok").to_numpy(bool)


def _node_id_series(df: pd.DataFrame) -> pd.Series:
    if "node_id" in df.columns:
        return pd.to_numeric(df["node_id"], errors="coerce")
    return pd.Series(np.arange(len(df), dtype=int), index=df.index)


def _fans_id(catalog: str, node_id: object) -> str:
    try:
        nid = int(float(node_id))
        return f"{catalog}_FANS_{nid:06d}"
    except Exception:
        return f"{catalog}_FANS_unknown"


def _to_float_array(df: pd.DataFrame, col: str, default: float = np.nan) -> np.ndarray:
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce").to_numpy(float)
    return np.full(len(df), float(default), dtype=float)


def _to_int_array(df: pd.DataFrame, col: str, default: int = -1) -> np.ndarray:
    vals = _to_float_array(df, col, np.nan)
    out = np.full(len(vals), int(default), dtype=np.int64)
    ok = np.isfinite(vals)
    out[ok] = np.rint(vals[ok]).astype(np.int64)
    return out


def _string_array(values: Iterable[object], width: Optional[int] = None) -> np.ndarray:
    vals = ["" if v is None or (isinstance(v, float) and not np.isfinite(v)) else str(v) for v in values]
    if width is None:
        width = max(1, max((len(v) for v in vals), default=1))
    width = max(1, int(width))
    return np.asarray([v[:width] for v in vals], dtype=f"S{width}")


def _radec_from_node_offsets(
    node_ra_deg: np.ndarray,
    node_dec_deg: np.ndarray,
    z: np.ndarray,
    x_mpc_h: np.ndarray,
    y_mpc_h: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert local east/north h^-1 Mpc offsets to approximate RA/Dec.

    This is a small-angle reconstruction for convenience.  The exact pipeline
    geometry remains the local tangent-plane offsets stored in the FITS table.
    """
    ra = np.asarray(node_ra_deg, dtype=float).copy()
    dec = np.asarray(node_dec_deg, dtype=float).copy()
    z = np.asarray(z, dtype=float)
    x = np.asarray(x_mpc_h, dtype=float)
    y = np.asarray(y_mpc_h, dtype=float)
    out_ra = np.full(len(ra), np.nan, dtype=float)
    out_dec = np.full(len(ra), np.nan, dtype=float)
    ok = np.isfinite(ra) & np.isfinite(dec) & np.isfinite(z) & np.isfinite(x) & np.isfinite(y) & (z > 0)
    if not np.any(ok):
        return out_ra, out_dec
    # Planck18.comoving_distance is in Mpc. Multiplying by h gives the
    # numerical h^-1 Mpc coordinate convention used by the pipeline distances.
    if Planck18 is None:
        raise ImportError("Segment sky coordinates require astropy")
    dm_mpc_h = Planck18.comoving_distance(z[ok]).value * Planck18.h
    good = np.isfinite(dm_mpc_h) & (dm_mpc_h > 0)
    if not np.any(good):
        return out_ra, out_dec
    idx = np.flatnonzero(ok)[good]
    dm = dm_mpc_h[good]
    dec_rad = np.deg2rad(dec[idx])
    cos_dec = np.cos(dec_rad)
    cos_dec = np.where(np.abs(cos_dec) > 1e-8, cos_dec, np.nan)
    dra_deg = np.rad2deg(x[idx] / (dm * cos_dec))
    ddec_deg = np.rad2deg(y[idx] / dm)
    out_ra[idx] = (ra[idx] + dra_deg) % 360.0
    out_dec[idx] = np.clip(dec[idx] + ddec_deg, -90.0, 90.0)
    return out_ra, out_dec


def _fits_columns_from_dataframe(df: pd.DataFrame, schema: List[Tuple[str, str, str]]) -> List[fits.Column]:
    """Create FITS columns from a simple schema: (out_name, source_col, format)."""
    cols: List[fits.Column] = []
    n = len(df)
    for out_name, source, fmt in schema:
        if fmt.endswith("A") or (len(fmt) > 1 and fmt[-1] == "A"):
            width = int(fmt[:-1]) if fmt[:-1] else None
            arr = _string_array(df[source] if source in df.columns else [""] * n, width=width)
        elif fmt in {"K", "J", "I", "B", "L"}:
            if fmt == "L":
                # FITS logical columns are stored as ASCII T/F.  Some low-level
                # readers display these as byte values 84/70.  We keep support
                # for L internally, but public QUALITY flags use unsigned byte
                # format B below so users see 1/0 consistently.
                arr = np.asarray(df[source].to_numpy(bool) if source in df.columns else np.zeros(n, dtype=bool), dtype=bool)
            elif fmt == "B":
                if source in df.columns:
                    arr = np.asarray(df[source].to_numpy(bool), dtype=np.uint8)
                else:
                    arr = np.zeros(n, dtype=np.uint8)
            else:
                arr = _to_int_array(df, source)
        else:
            arr = _to_float_array(df, source).astype(np.float64 if fmt == "D" else np.float32)
        cols.append(fits.Column(name=out_name, array=arr, format=fmt))
    return cols


def _prepare_nodes(catalog: str, nodes: pd.DataFrame, accepted_only: bool) -> pd.DataFrame:
    from spirality.selection import validate_nodes
    validate_nodes(nodes)
    n = nodes.copy()
    if len(n) == 0:
        return n
    n["node_id_out"] = _node_id_series(n)
    n["fans_id"] = [_fans_id(catalog, x) for x in n["node_id_out"]]
    n["z"] = _choose_z(n)
    n["is_quality_ok"] = _node_status_ok(n)
    if accepted_only:
        n = n.loc[n["is_quality_ok"]].copy()
    n["narms"] = _to_int_array(n, "n_arms_valid")
    n["nsegs"] = _to_int_array(n, "n_arm_segments_total")
    if "S_tangential" in n.columns:
        n["handedness"] = pd.to_numeric(n["S_tangential"], errors="coerce")
        n["spirality"] = np.abs(n["handedness"].to_numpy(float))
    else:
        n["handedness"] = np.nan
        n["spirality"] = np.nan
    if "status" not in n.columns:
        n["status"] = "ok"
    return n.reset_index(drop=True)


def _prepare_segments(catalog: str, nodes: pd.DataFrame, seg: pd.DataFrame, accepted_only: bool) -> pd.DataFrame:
    s = seg.copy()
    if len(s) == 0:
        return s
    node_cols = ["node_id_out", "fans_id", "ra", "dec", "z", "status", "is_quality_ok"]
    if not set(node_cols).issubset(nodes.columns) or "node_id" not in s:
        raise ValueError("Segments require a complete parent-node table for FITS export")
    nn = nodes[node_cols].copy()
    s["node_id_out"] = _node_id_series(s)
    s = s.merge(nn, on="node_id_out", how="left", suffixes=("", "_node"), validate="many_to_one", indicator=True)
    if (s["_merge"] != "both").any():
        raise ValueError("Orphan segments in FITS export; supply all parent nodes before filtering")
    s = s.drop(columns="_merge")
    if accepted_only:
        s = s.loc[s["is_quality_ok"].fillna(False).eq(True)].copy()
    if len(s) == 0:
        return s.reset_index(drop=True)

    s["segment_index"] = _to_int_array(s, "arm_segment_index")
    s["arm_id"] = _to_int_array(s, "arm_local_id")
    s["npoints"] = _to_int_array(s, "n_points")
    s["segment_id"] = [
        f"{catalog}_FANS_{int(nid):06d}_arm{int(aid):02d}_seg{int(k):02d}"
        if nid >= 0 and aid >= 0 and k >= 0 else f"{catalog}_segment_{i:08d}"
        for i, (nid, aid, k) in enumerate(zip(_to_int_array(s, "node_id_out"), _to_int_array(s, "arm_id"), _to_int_array(s, "segment_index")))
    ]

    node_ra = _to_float_array(s, "ra")
    node_dec = _to_float_array(s, "dec")
    z = _to_float_array(s, "z")
    x = _to_float_array(s, "x_mpc_h")
    y = _to_float_array(s, "y_mpc_h")
    seg_ra, seg_dec = _radec_from_node_offsets(node_ra, node_dec, z, x, y)
    s["ra_segment"] = seg_ra
    s["dec_segment"] = seg_dec
    return s.reset_index(drop=True)


def _make_nodes_hdu(nodes: pd.DataFrame) -> fits.BinTableHDU:
    schema = [
        ("FANS_ID", "fans_id", "48A"),
        ("RA", "ra", "D"),
        ("DEC", "dec", "D"),
        ("Z", "z", "D"),
        ("NARMS", "narms", "K"),
        ("NSEGS", "nsegs", "K"),
        ("NPOINTS", "n_points_used_total", "K"),
        ("HANDEDNESS", "handedness", "D"),
        ("SPIRALITY", "spirality", "D"),
        ("QUALITY", "is_quality_ok", "B"),
    ]
    hdu = fits.BinTableHDU.from_columns(_fits_columns_from_dataframe(nodes, schema), name="NODES")
    hdu.header["EXTDESC"] = "One row per FANS node/candidate"
    hdu.header["HUNIT"] = "Dimensionless NODE-level winding, not the paper arm-population mean"
    hdu.header["COMMENT"] = "NODES.SPIRALITY is abs(NODE HANDEDNESS), NOT S_proj."
    hdu.header["COMMENT"] = "Reconstruct arm h from SEGMENTS using NPOINTS, then W=n_valid_segments."
    hdu.header["QUALDEF"] = "QUALITY=1 accepted final sample; QUALITY=0 rejected candidate"
    return hdu


def _make_segments_hdu(seg: pd.DataFrame) -> fits.BinTableHDU:
    schema = [
        ("SEG_ID", "segment_id", "64A"),
        ("FANS_ID", "fans_id", "48A"),
        ("ARM_ID", "arm_id", "K"),
        ("SEG_INDEX", "segment_index", "K"),
        ("RA", "ra_segment", "D"),
        ("DEC", "dec_segment", "D"),
        ("Z", "z", "D"),
        ("NPOINTS", "npoints", "K"),
        ("X_MPC_H", "x_mpc_h", "D"),
        ("Y_MPC_H", "y_mpc_h", "D"),
        ("RHO_MPC_H", "rho_mpc_h", "D"),
        ("PHI_DEG", "phi_deg", "D"),
        ("T_PHI", "t_phi", "D"),
        ("T_RADIAL", "t_radial", "D"),
        ("WEIGHT", "weight_sum", "D"),
        ("QUALITY", "is_quality_ok", "B"),
    ]
    hdu = fits.BinTableHDU.from_columns(_fits_columns_from_dataframe(seg, schema), name="SEGMENTS")
    hdu.header["EXTDESC"] = "One row per measured arm segment"
    hdu.header["COORD"] = "RA/DEC are approximate segment centroids; X/Y are pipeline tangent-plane offsets"
    hdu.header["QUALDEF"] = "QUALITY=1 if parent FANS is accepted; 0 otherwise"
    return hdu


def export_catalog(run_root: Path, catalog: str, outdir: Path, accepted_only: bool, overwrite: bool) -> Optional[Path]:
    cdir = _find_catalog_dir(run_root, catalog)
    if cdir is None:
        print(f"[skip] {catalog}: no catalogue products found under {run_root}")
        return None
    nodes_raw = _read_optional_csv(cdir / "node_spirality.csv")
    seg_raw = _read_optional_csv(cdir / "arm_segments.csv")
    if len(nodes_raw) == 0:
        print(f"[skip] {catalog}: missing/empty node_spirality.csv in {cdir}")
        return None

    if fits is None:
        raise ImportError("FITS export requires astropy; install requirements.txt")
    all_nodes = _prepare_nodes(catalog, nodes_raw, accepted_only=False)
    seg = _prepare_segments(catalog, all_nodes, seg_raw, accepted_only=accepted_only)
    nodes = all_nodes.loc[all_nodes.is_quality_ok].copy() if accepted_only else all_nodes
    if accepted_only and not set(seg.get("fans_id", [])).issubset(set(nodes.fans_id)):
        raise ValueError("FITS quality export contains segments outside accepted FANS")

    outdir.mkdir(parents=True, exist_ok=True)
    suffix = "quality" if accepted_only else "all"
    outpath = outdir / f"{catalog}_fans_{suffix}.fits"

    hdr = fits.Header()
    hdr["ORIGIN"] = "FANS spirality pipeline"
    hdr["SCHEMA"] = "88.1"
    hdr["RUNROOT"] = str(run_root)
    # The catalogue key is encoded in the output filename, e.g. carron_fans_all.fits.
    hdr["ACCEPTED"] = bool(accepted_only)
    params = _infer_run_parameters(run_root)
    for key, val in params.items():
        if np.isscalar(val) and not isinstance(val, str):
            hdr[key[:8].upper()] = val
    hdr["COMMENT"] = "NODES contains object-level FANS quantities."
    hdr["COMMENT"] = "SEGMENTS contains segment centroids and local tangent measurements."
    hdr["COMMENT"] = "QUALITY is stored as unsigned byte: 1=accepted, 0=rejected."
    primary = fits.PrimaryHDU(header=hdr)

    hdul = fits.HDUList([primary, _make_nodes_hdu(nodes), _make_segments_hdu(seg)])
    hdul.writeto(outpath, overwrite=overwrite, checksum=True)
    print(f"[write] {outpath}  nodes={len(nodes):,} segments={len(seg):,}")
    return outpath


def _find_run_dirs(root: Path, catalogs: Iterable[str]) -> List[Path]:
    runs: List[Path] = []
    for d in sorted(root.iterdir() if root.exists() else []):
        if not d.is_dir():
            continue
        if any(_find_catalog_dir(d, c) is not None for c in catalogs):
            runs.append(d)
    return runs


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-root", default=None, help="One completed run directory.")
    p.add_argument("--grid-root", default=None, help="Directory containing many completed run directories, e.g. the 27-variation grid root.")
    p.add_argument("--outdir", default=None, help="Output directory for a single run. Default: <run-root>/fans_fits")
    p.add_argument("--out-root", default=None, help="Output root for --grid-root. Default: write inside each run under fans_fits/")
    p.add_argument("--catalogs", nargs="+", default=["sconce", "carron", "chen"], choices=list(CATALOG_SPECS))
    p.add_argument("--accepted-only", action="store_true", help="Write only QUALITY=1 FANS and their segments. Output suffix is _fans_quality.fits.")
    p.add_argument("--write-quality-copy", action="store_true", help="When writing the full catalogue, also write a quality-only *_fans_quality.fits file.")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing FITS files.")
    args = p.parse_args()

    if not args.run_root and not args.grid_root:
        p.error("Provide --run-root or --grid-root")

    if args.run_root:
        run_root = Path(args.run_root).expanduser().resolve()
        outdir = Path(args.outdir).expanduser().resolve() if args.outdir else run_root / "fans_fits"
        for cat in args.catalogs:
            export_catalog(run_root, cat, outdir, accepted_only=bool(args.accepted_only), overwrite=bool(args.overwrite))
            if bool(args.write_quality_copy) and not bool(args.accepted_only):
                export_catalog(run_root, cat, outdir, accepted_only=True, overwrite=bool(args.overwrite))

    if args.grid_root:
        grid_root = Path(args.grid_root).expanduser().resolve()
        runs = _find_run_dirs(grid_root, args.catalogs)
        if not runs:
            print(f"[warn] no completed run directories found under {grid_root}")
        for run_root in runs:
            if args.out_root:
                outdir = Path(args.out_root).expanduser().resolve() / run_root.name / "fans_fits"
            else:
                outdir = run_root / "fans_fits"
            for cat in args.catalogs:
                export_catalog(run_root, cat, outdir, accepted_only=bool(args.accepted_only), overwrite=bool(args.overwrite))
                if bool(args.write_quality_copy) and not bool(args.accepted_only):
                    export_catalog(run_root, cat, outdir, accepted_only=True, overwrite=bool(args.overwrite))


if __name__ == "__main__":
    main()
