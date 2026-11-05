"""Spatial resampling primitives with explicit group multiplicities.

HEALPix is required for sky partitioning. Tests can supply known integer block
labels to the statistical functions; there is no substitute angular grid.
"""
from __future__ import annotations
import numpy as np


def healpix_blocks(ra, dec, nside: int) -> np.ndarray:
    try:
        import healpy as hp
    except ImportError as exc:
        raise ImportError("Spatial HEALPix inference requires healpy. Install requirements.txt; no fallback grid is used.") from exc
    if not hp.isnsideok(int(nside), nest=False):
        raise ValueError(f"Invalid HEALPix NSIDE {nside}")
    ra, dec = np.asarray(ra, float), np.asarray(dec, float)
    if ra.shape != dec.shape or not np.all(np.isfinite(ra)) or not np.all(np.isfinite(dec)) or np.any(np.abs(dec) > 90):
        raise ValueError("Finite, matching RA/Dec arrays with -90 <= Dec <= 90 are required")
    return hp.ang2pix(int(nside), np.mod(ra, 360), dec, lonlat=True, nest=False)


def block_draw_counts(block_ids, B: int, seed: int):
    """Multinomial pairs/block bootstrap, sampling occupied blocks uniformly."""
    ids = np.asarray(block_ids)
    unique, inv = np.unique(ids, return_inverse=True)
    ng = len(unique)
    if ng == 0:
        raise ValueError("No occupied blocks")
    if B < 0:
        raise ValueError("B must be nonnegative")
    counts = np.random.default_rng(int(seed)).multinomial(ng, np.full(ng, 1.0 / ng), size=int(B))
    return unique, inv, counts


def safe_ratio(num, den):
    num, den = np.asarray(num, float), np.asarray(den, float)
    return np.divide(num, den, out=np.full(np.broadcast_shapes(num.shape, den.shape), np.nan), where=den > 0)


def bootstrap_summary(samples, *, min_valid_fraction: float = .9) -> dict:
    s = np.asarray(samples, float)
    valid = s[np.isfinite(s)]
    out = {"bootstrap_n_valid": int(len(valid)), "bootstrap_valid_fraction": len(valid) / len(s) if len(s) else np.nan,
           "err": np.nan, "q16": np.nan, "q84": np.nan, "q025": np.nan, "q975": np.nan}
    if len(valid) >= 2 and len(valid) >= min_valid_fraction * len(s):
        qs = np.percentile(valid, [15.8655253931, 84.1344746069, 2.5, 97.5])
        out.update(err=float(np.std(valid, ddof=1)), q16=float(qs[0]), q84=float(qs[1]), q025=float(qs[2]), q975=float(qs[3]))
    return out


def covariance_complete(samples):
    """PSD covariance using common complete draws, NOT pairwise deletion."""
    a = np.asarray(samples, float)
    usable = np.any(np.isfinite(a), axis=0) if len(a) else np.zeros(a.shape[1], bool)
    cov = np.full((a.shape[1], a.shape[1]), np.nan)
    if not usable.any():
        return cov, 0
    complete = np.all(np.isfinite(a[:, usable]), axis=1)
    n = int(complete.sum())
    if n > 1:
        c = np.atleast_2d(np.cov(a[complete][:, usable], rowvar=False, ddof=1))
        ii = np.flatnonzero(usable)
        cov[np.ix_(ii, ii)] = c
    return cov, n
