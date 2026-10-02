"""
Observation-band preprocessing for the scHiC latent-diffusion EM.

Build selectable preprocessed per-cell band observations x_aux_n in log-space.
For the exposure-aware DiffusionEM C2 path these arrays are auxiliary inputs for
conditioning/proposals; the EM data likelihood is defined on raw counts plus
exposure, not on x_aux.  Other legacy methods may still use this output as a
reconstruction target.

Supported methods (config: training.preprocess.method):
  - "rwr" / "schicluster" : scHiCluster random-walk-with-restart imputation
                            (smooth, dense; the original default).  Requires the
                            `schicluster` package.
  - "bandnorm"            : BandNorm-style per-cell per-band depth normalization
                            (Zheng, Kele艧 2022) 鈫?log1p.  Cheap, no imputation.
  - "bandnorm_smooth"     : bandnorm + Gaussian smoothing along the genomic
                            (i) axis (a light, RWR-free denoiser).
  - "raw_log1p"           : log1p of the raw band counts (rawest baseline).

All methods return (N, M, D) float32 in auxiliary log-space (already
log1p'd / non-negative), with rows aligned to `cell_paths`.
"""

from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Raw band loader (shared by the imputation-free methods)
# ---------------------------------------------------------------------------

def load_raw_band(
    scool_path: str,
    cell_paths: List[str],
    chrom: str,
    M: int,
    D: int,
    progress_prefix: str = "",
) -> np.ndarray:
    """Raw upper-band contact counts 鈫?(N, M, D) float32, aligned to cell_paths."""
    import cooler
    from .schic import safe_fetch_cooler

    N = len(cell_paths)
    T = np.zeros((N, M, D), dtype=np.float32)
    for n, cp in enumerate(cell_paths):
        c = cooler.Cooler(f"{scool_path}::{cp}")
        try:
            coo = safe_fetch_cooler(c, chrom).tocoo()
        except Exception:
            continue
        i = coo.row.astype(np.int64)
        j = coo.col.astype(np.int64)
        d = j - i
        keep = (d >= 1) & (d <= D) & (i < M)
        T[n, i[keep], d[keep] - 1] = coo.data[keep].astype(np.float32)
        if progress_prefix and (n + 1) % 256 == 0:
            log.info("%s%s: loaded raw %d/%d", progress_prefix, chrom, n + 1, N)
    return T


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------

def apply_bandnorm(raw: np.ndarray) -> np.ndarray:
    """BandNorm: equalize per-cell sequencing depth within each genomic band.

    For band d, cell c with band-total S_d^(c) = 危_i raw[c,i,d], rescale to the
    cross-cell mean band-total:  x[c,i,d] = raw[c,i,d] 路 mean_c'(S_d) / S_d^(c).
    Cells with zero band-total are left at zero.
    """
    N, M, D = raw.shape
    band_tot = raw.sum(axis=1)                                # (N, D)
    nz = band_tot > 0
    mean_band = np.zeros(D, dtype=np.float64)
    for d in range(D):
        col = band_tot[nz[:, d], d]
        mean_band[d] = col.mean() if col.size else 0.0
    scale = np.zeros_like(band_tot, dtype=np.float64)
    scale[nz] = mean_band[None, :].repeat(N, 0)[nz] / band_tot[nz]
    return (raw * scale[:, None, :]).astype(np.float32)


def gaussian_smooth_i(x: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian smoothing along the genomic (i) axis, per band.  x: (N, M, D)."""
    if sigma <= 0:
        return x
    try:
        from scipy.ndimage import gaussian_filter1d
        return gaussian_filter1d(x, sigma=sigma, axis=1, mode="nearest").astype(np.float32)
    except ImportError:
        radius = max(1, int(3 * sigma))
        ker = np.exp(-0.5 * (np.arange(-radius, radius + 1) / sigma) ** 2)
        ker /= ker.sum()
        out = np.empty_like(x)
        for d in range(x.shape[2]):
            out[:, :, d] = np.apply_along_axis(
                lambda r: np.convolve(r, ker, mode="same"), 1, x[:, :, d]
            )
        return out.astype(np.float32)


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

def compute_observation_band(
    method: str,
    scool_path: str,
    cell_paths: List[str],
    chrom: str,
    resolution: int,
    M: int,
    D_band: int,
    cfg: Optional[dict] = None,
    progress_prefix: str = "",
) -> np.ndarray:
    """Return the (N, M, D) preprocessed observation band (log-space, 鈮?)."""
    cfg = cfg or {}
    method = (method or "rwr").lower()

    if method in ("rwr", "schicluster"):
        from ..utils.schicluster_impute import compute_imputed_target_band
        raw = compute_imputed_target_band(
            scool_path=scool_path,
            cell_paths=cell_paths,
            chrom=chrom,
            resolution=resolution,
            M=M,
            D_band=D_band,
            n_jobs=int(cfg.get("n_jobs", 1)),
            batch_size=int(cfg.get("batch_size", 64)),
            pad=int(cfg.get("pad", 1)),
            std=float(cfg.get("std", 1.0)),
            rp=float(cfg.get("rp", 0.5)),
            tol=float(cfg.get("tol", 0.01)),
            logscale=bool(cfg.get("logscale", False)),
            output_dist=int(cfg.get("output_dist", 500_000_000)),
            min_cutoff=float(cfg.get("min_cutoff", 1e-6)),
            progress_prefix=progress_prefix,
        )
        return np.log1p(raw).astype(np.float32)

    raw = load_raw_band(scool_path, cell_paths, chrom, M, D_band, progress_prefix)

    if method == "raw_log1p":
        return np.log1p(raw).astype(np.float32)

    if method in ("bandnorm", "bandnorm_smooth"):
        x = np.log1p(apply_bandnorm(raw))
        if method == "bandnorm_smooth":
            x = gaussian_smooth_i(x, float(cfg.get("smooth_sigma", 1.0)))
        return x.astype(np.float32)

    raise ValueError(
        f"Unknown preprocess method '{method}'. Choose from: "
        "rwr, bandnorm, bandnorm_smooth, raw_log1p."
    )
