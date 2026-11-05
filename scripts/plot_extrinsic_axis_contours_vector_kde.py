#!/usr/bin/env python3
"""Vector KDE contour plots for extrinsic-axis bootstrap directions.

Reads existing extrinsic bootstrap CSV files and makes smooth vector contours
using healpy.projplot/projscatter.

Recommended for axial directions:
  --axis-mode nearest-best

Modes:
  nearest-best : choose each bootstrap axis or its antipode so it is closest to
                 the best-fit axis; this is the recommended final mode.
  antipode     : hard diagnostic convention, RA<180 -> antipode.
  unwrap360   : no antipode; RA<=180 is shifted to RA+360 for the KDE plane.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

try:
    import healpy as hp
except Exception as exc:
    raise SystemExit(f"healpy is required: {exc!r}")

DEG = 180.0 / np.pi

STYLE = {
    "carron": dict(label="Carrón et al. SDSS DR16", line="royalblue", marker="o"),
    "chen": dict(label="Chen et al. SDSS DR12", line="orangered", marker="o"),
}


def setup_matplotlib() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "axes.linewidth": 1.1,
        "axes.labelsize": 15,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "figure.dpi": 140,
        "savefig.dpi": 260,
    })


def unit_xyz_from_radec(ra_deg, dec_deg):
    ra = np.deg2rad(np.asarray(ra_deg, dtype=float))
    dec = np.deg2rad(np.asarray(dec_deg, dtype=float))
    c = np.cos(dec)
    return np.column_stack([c*np.cos(ra), c*np.sin(ra), np.sin(dec)])


def canonicalize(ra, dec, mode: str, *, ref_ra=None, ref_dec=None):
    ra = np.asarray(ra, dtype=float).copy() % 360.0
    dec = np.asarray(dec, dtype=float).copy()
    if mode == "nearest-best":
        if ref_ra is None or ref_dec is None:
            raise ValueError("axis-mode nearest-best requires ref_ra/ref_dec")
        v = unit_xyz_from_radec(ra, dec)
        ref = unit_xyz_from_radec([float(ref_ra)], [float(ref_dec)])[0]
        m = (v @ ref) < 0.0
        ra[m] = (ra[m] + 180.0) % 360.0
        dec[m] = -dec[m]
    elif mode == "antipode":
        m = ra < 180.0
        ra[m] = (ra[m] + 180.0) % 360.0
        dec[m] = -dec[m]
    elif mode == "unwrap360":
        m = ra <= 180.0
        ra[m] = ra[m] + 360.0
    else:
        raise ValueError(mode)
    return ra, dec


def wrap_for_plot(ra):
    return np.asarray(ra, dtype=float) % 360.0


def split_on_wrap(ra, dec):
    ra_plot = wrap_for_plot(ra)
    dec = np.asarray(dec, dtype=float)
    jumps = np.where(np.abs(np.diff(ra_plot)) > 180.0)[0]
    parts = []
    start = 0
    for j in jumps:
        if j + 1 - start >= 2:
            parts.append((ra_plot[start:j+1], dec[start:j+1]))
        start = j + 1
    if len(ra_plot) - start >= 2:
        parts.append((ra_plot[start:], dec[start:]))
    return parts


def local_xy(ra, dec, ra0, dec0):
    x = (np.asarray(ra) - ra0) * np.cos(np.deg2rad(dec0))
    y = np.asarray(dec) - dec0
    return x, y


def lonlat_from_xy(x, y, ra0, dec0):
    ra = ra0 + np.asarray(x) / max(1e-8, np.cos(np.deg2rad(dec0)))
    dec = dec0 + np.asarray(y)
    return ra, np.clip(dec, -90.0, 90.0)


def kde_hpd_thresholds(kde, ra0, dec0, mode: str, nside_area: int, *, ref_ra=None, ref_dec=None, levels=(0.6827, 0.9545)):
    npix = hp.nside2npix(int(nside_area))
    pix = np.arange(npix)
    theta, phi = hp.pix2ang(int(nside_area), pix, nest=False)
    ra = np.rad2deg(phi) % 360.0
    dec = 90.0 - np.rad2deg(theta)

    if mode == "nearest-best":
        # The canonical domain is the half-sky closer to the reference axis.
        v = unit_xyz_from_radec(ra, dec)
        ref = unit_xyz_from_radec([float(ref_ra)], [float(ref_dec)])[0]
        mask = (v @ ref) >= 0.0
        ra_c, dec_c = canonicalize(ra[mask], dec[mask], mode, ref_ra=ref_ra, ref_dec=ref_dec)
    elif mode == "antipode":
        ra_c_all, dec_c_all = canonicalize(ra, dec, mode)
        mask = ra_c_all >= 180.0
        ra_c, dec_c = ra_c_all[mask], dec_c_all[mask]
    elif mode == "unwrap360":
        mask = dec >= 0.0
        ra_c, dec_c = canonicalize(ra[mask], dec[mask], mode)
    else:
        raise ValueError(mode)

    x, y = local_xy(ra_c, dec_c, ra0, dec0)
    dens = kde(np.vstack([x, y]))
    dens = np.where(np.isfinite(dens) & (dens > 0), dens, 0.0)
    order = np.argsort(dens)[::-1]
    ds = dens[order]
    cs = np.cumsum(ds)
    cs /= cs[-1] if cs[-1] > 0 else 1.0

    rows, thresholds = [], {}
    for lev in levels:
        idx = int(np.searchsorted(cs, float(lev), side="left"))
        idx = min(idx, len(ds) - 1)
        thr = float(ds[idx])
        thresholds[float(lev)] = thr
        nreg = int(np.count_nonzero(dens >= thr))
        nmask = int(len(dens))
        pix_area_deg2 = float(hp.nside2pixarea(int(nside_area), degrees=True))
        area_deg2 = nreg * pix_area_deg2
        rows.append({
            "level": float(lev),
            "threshold": thr,
            "n_pix_area_grid": nreg,
            "n_pix_domain": nmask,
            "area_deg2": area_deg2,
            "f_sky_axial_domain": nreg / nmask if nmask > 0 else np.nan,
            "f_sky_full_if_antipodal": (2.0 * area_deg2) / (4.0 * np.pi * DEG * DEG),
            "nside_area": int(nside_area),
        })
    return thresholds, pd.DataFrame(rows)


def make_grid(x, y, pad_deg=8.0, grid_size=260):
    xmin, xmax = np.nanpercentile(x, [0.5, 99.5])
    ymin, ymax = np.nanpercentile(y, [0.5, 99.5])
    xmin -= pad_deg; xmax += pad_deg
    ymin -= pad_deg; ymax += pad_deg
    xx = np.linspace(xmin, xmax, int(grid_size))
    yy = np.linspace(ymin, ymax, int(grid_size))
    X, Y = np.meshgrid(xx, yy)
    return X, Y


def contour_segments(X, Y, Z, level):
    """Return contour line segments, robust to Matplotlib API changes."""
    fig_tmp, ax_tmp = plt.subplots()
    cs = ax_tmp.contour(X, Y, Z, levels=[level])
    segs = []
    if hasattr(cs, "allsegs"):
        for seg in cs.allsegs[0]:
            if len(seg) >= 2:
                segs.append(seg.copy())
    elif hasattr(cs, "collections"):
        for coll in cs.collections:
            for path in coll.get_paths():
                vv = path.vertices
                if len(vv) >= 2:
                    segs.append(vv.copy())
    plt.close(fig_tmp)
    return segs


def plot_one_catalog(key, label, boot, best, args):
    st = STYLE[key]
    ref_ra = float(best.get("ra_deg", best.get("ra_deg_display", np.nan)))
    ref_dec = float(best.get("dec_deg", best.get("dec_deg_display", np.nan)))

    ra_c, dec_c = canonicalize(
        boot["ra_deg"].to_numpy(float),
        boot["dec_deg"].to_numpy(float),
        args.axis_mode,
        ref_ra=ref_ra,
        ref_dec=ref_dec,
    )

    # Center on the bootstrap median in the same canonical chart.
    ra_med0 = float(np.nanmedian(ra_c))
    ra_un = ra_med0 + ((ra_c - ra_med0 + 180.0) % 360.0 - 180.0)
    ra_med = float(np.nanmedian(ra_un))
    dec_med = float(np.nanmedian(dec_c))

    x, y = local_xy(ra_un, dec_c, ra_med, dec_med)
    kde = gaussian_kde(np.vstack([x, y]), bw_method=args.kde_bw_method)
    kde.set_bandwidth(kde.factor * float(args.kde_bw_scale))

    X, Y = make_grid(x, y, pad_deg=args.grid_pad_deg, grid_size=args.grid_size)
    Z = kde(np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape)

    thresholds, summary = kde_hpd_thresholds(
        kde, ra_med, dec_med, args.axis_mode,
        nside_area=args.area_nside,
        ref_ra=ref_ra,
        ref_dec=ref_dec,
        levels=(0.6827, 0.9545),
    )
    summary.insert(0, "catalog_label", label)
    summary.insert(0, "catalog_key", key)
    summary["axis_mode"] = args.axis_mode
    summary["kde_bw_method"] = str(args.kde_bw_method)
    summary["kde_bw_scale"] = float(args.kde_bw_scale)

    try:
        hp.projscatter(wrap_for_plot(ra_un), dec_c, lonlat=True, s=6, color=st["line"], alpha=0.23, marker=".")
    except Exception:
        hp.projplot(wrap_for_plot(ra_un), dec_c, ".", lonlat=True, color=st["line"], alpha=0.23, markersize=2.3)

    for lev, ls, lw in [(0.9545, "--", 1.8), (0.6827, "-", 2.3)]:
        for seg in contour_segments(X, Y, Z, thresholds[lev]):
            ra_line, dec_line = lonlat_from_xy(seg[:, 0], seg[:, 1], ra_med, dec_med)
            for rp, dp in split_on_wrap(ra_line, dec_line):
                hp.projplot(rp, dp, lonlat=True, color=st["line"], linestyle=ls, linewidth=lw)

    best_ra0 = float(best.get("ra_deg", np.nan))
    best_dec0 = float(best.get("dec_deg", np.nan))
    best_ra_c, best_dec_c = canonicalize(np.array([best_ra0]), np.array([best_dec0]), args.axis_mode, ref_ra=ref_ra, ref_dec=ref_dec)
    hp.projplot(wrap_for_plot(best_ra_c), best_dec_c, "*", lonlat=True, color=st["line"], markersize=13, markeredgecolor="k", markeredgewidth=0.55)
    hp.projplot(wrap_for_plot(np.array([ra_med])), np.array([dec_med]), "D", lonlat=True, color=st["line"], markersize=7, markeredgecolor="k", markeredgewidth=0.55)
    return summary


def plot_map(items, args, out, *, combined=True):
    setup_matplotlib()
    npix = hp.nside2npix(32)
    neutral = np.zeros(npix)
    plt.figure(figsize=(10.6, 6.3))
    hp.mollview(neutral, coord="C", title=("Extrinsic-axis bootstrap KDE contours" if combined else ""), cmap="Greys", min=0, max=1, cbar=False, hold=True)
    hp.graticule(dpar=args.graticule_step_deg, dmer=args.graticule_step_deg, alpha=0.32)

    summaries = []
    for key, label, boot, best in items:
        summaries.append(plot_one_catalog(key, label, boot, best, args))
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, bbox_inches="tight", dpi=260)
    plt.close()
    return pd.concat(summaries, ignore_index=True) if summaries else pd.DataFrame()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--extrinsic-dir", required=True)
    p.add_argument("--catalogs", nargs="+", default=["carron", "chen"], choices=["carron", "chen"])
    p.add_argument("--axis-mode", default="nearest-best", choices=["nearest-best", "antipode", "unwrap360"])
    p.add_argument("--kde-bw-method", default="scott", help="scipy gaussian_kde bw_method")
    p.add_argument("--kde-bw-scale", type=float, default=1.0)
    p.add_argument("--grid-size", type=int, default=280)
    p.add_argument("--grid-pad-deg", type=float, default=10.0)
    p.add_argument("--area-nside", type=int, default=128)
    p.add_argument("--graticule-step-deg", type=float, default=30.0)
    p.add_argument("--output-prefix", default="vector_kde_contours")
    p.add_argument("--plot-format", default="pdf")
    args = p.parse_args()

    edir = Path(args.extrinsic_dir).expanduser().resolve()
    items = []
    for key in args.catalogs:
        label = STYLE[key]["label"]
        boot_path = edir / f"{key}_extrinsic_bootstrap_bestfits.csv"
        best_path = edir / f"{key}_extrinsic_bestfit.csv"
        if not boot_path.exists() or not best_path.exists():
            print(f"[skip] missing {key} files")
            continue
        boot = pd.read_csv(boot_path)
        best = pd.read_csv(best_path).iloc[0].to_dict()
        items.append((key, label, boot, best))

    allsum = []
    for item in items:
        key, label, boot, best = item
        out = edir / f"{args.output_prefix}_{key}.{args.plot_format}"
        summ = plot_map([item], args, out, combined=False)
        summ.to_csv(edir / f"{args.output_prefix}_{key}_summary.csv", index=False)
        print(f"[write] {out}")
        print(f"[write] {edir / f'{args.output_prefix}_{key}_summary.csv'}")
        allsum.append(summ)

    if items:
        out = edir / f"{args.output_prefix}_combined.{args.plot_format}"
        summ = plot_map(items, args, out, combined=True)
        summ.to_csv(edir / f"{args.output_prefix}_combined_summary.csv", index=False)
        print(f"[write] {out}")
        print(f"[write] {edir / f'{args.output_prefix}_combined_summary.csv'}")
        allsum.append(summ)
    if allsum:
        pd.concat(allsum, ignore_index=True).to_csv(edir / f"{args.output_prefix}_all_summary.csv", index=False)
        print(f"[write] {edir / f'{args.output_prefix}_all_summary.csv'}")


if __name__ == "__main__":
    main()
