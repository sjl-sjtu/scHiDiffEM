"""
EM C2: improved latent diffusion on Z (geometric Z + EM preserved).

This is a full rewrite of the C1 method incorporating the portable tricks from
ChromoGen (and standard diffusion best practice), while keeping the project's
geometric-latent + EM + Gaussian/ZINB-likelihood framework intact:

  1. Z normalization  --the diffusion (DSM + prior score) operates on a
     globally standardized Z_tilde = Z / z_scale (z_scale = global std of the current
     Z_hats, refreshed each EM iter).  The DDPM forward process assumes unit-
     variance data; the C1 code diffused on an arbitrarily-scaled, drifting Z.
     This is the single most important fix.

  2. Cosine noise schedule (+ Min-SNR-gamma DSM loss weighting).

  3. Periodic GPA --the score net is not rotation-equivariant, so the implicit
     gauge can drift across EM iters.  C1 aligned Z once; C2 re-aligns every
     `gpa_every` iters to keep the gauge stable for the diffusion prior.

  4. Auxiliary conditioning remains available for CFG.  The score net can be
     conditional on an auxiliary preprocessed observation x_aux (bandnorm/RWR/etc),
     trained with condition dropout so it learns p(Z) and p(Z|x_aux).  x_aux is
     not the EM reconstruction target.

E-step:  exposure-aware raw-count NB data consistency alternated with an
         annealed diffusion denoising proximal update.
M-step:  (1) B(d) gradient descent under the same raw likelihood; (2)
         Min-SNR-weighted DSM on the conditional score net.
Inference: MAP or DDIM sampling with x_aux as condition/proposal and raw-count
           likelihood guidance when available.
"""

from __future__ import annotations

import logging
import math
import os
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from ..data.schic import ScHiCDataset, resolve_resolution_rough
from ..models.geometric import BandBiasNetwork
from ..models.schedules import make_alpha_bars, min_snr_weight
from ..utils.ema import EMA
from .schic_em import forward_diffuse, band_valid_mask


def _resolve_binsize(
    scool_path: str,
    cell_path: str,
    chrom: str,
    fallback_resolution: Optional[int] = None,
) -> int:
    """Rough resolution (resolve_resolution_rough): this is only used to convert
    RWR's output_dist (500,000,000 bp) into a bin cutoff, and that cutoff is
    already far larger than any real chromosome, so it never binds -- exact bp
    accuracy is not needed here. `cell_path` is unused (kept for call-site
    compatibility) -- the shared resolver inspects the whole scool."""
    return resolve_resolution_rough(scool_path, [chrom], fallback_resolution)


def batched_procrustes_align(
    Z: torch.Tensor,
    max_iter: int = 10,
    tol: float = 1e-5,
) -> torch.Tensor:
    """Generalized Procrustes Analysis on Z: (N, M, K) -> aligned (N, M, K).

    Cost: per iter, N batched (K, K) SVDs (~mus each).  Total ~1s for N=5000.
    """
    Zal = Z.clone()
    ref = Zal.mean(dim=0)                                   # (M, K)
    for it in range(max_iter):
        A = torch.einsum('nmk,mj->nkj', Zal, ref)          # (N, K, K)
        U, _, Vh = torch.linalg.svd(A, full_matrices=False)
        R = U @ Vh                                          # (N, K, K)
        Zal = torch.bmm(Zal, R)                             # (N, M, K)
        new_ref = Zal.mean(dim=0)
        diff = (new_ref - ref).norm() / (ref.norm() + 1e-8)
        ref = new_ref
        if diff.item() < tol:
            logger.info("  GPA converged at iter %d (Delta=%.2e)", it + 1, diff.item())
            break
    return Zal

logger = logging.getLogger(__name__)


class ScHiCEMC2:
    def __init__(
        self,
        score_net: nn.Module,
        band_bias: BandBiasNetwork,
        config: dict,
        device: torch.device,
    ) -> None:
        self.score_net = score_net.to(device)
        self.band_bias = band_bias.to(device)
        self.config = config
        self.device = device

        diff_cfg = config["diffusion"]
        self.T = int(diff_cfg["num_timesteps"])
        self.t_min = int(diff_cfg["t_min"])
        self.schedule = diff_cfg.get("schedule", "cosine")

        betas, alpha_bars = make_alpha_bars(
            self.T,
            schedule=self.schedule,
            beta_start=float(diff_cfg.get("beta_start", 1e-4)),
            beta_end=float(diff_cfg.get("beta_end", 0.02)),
            cosine_s=float(diff_cfg.get("cosine_s", 0.008)),
            device=device,
        )
        self.betas = betas
        self.alpha_bars = alpha_bars

        tr = config["training"]
        self.K_E = int(tr["K_E"])
        self.lr_e = float(tr["lr_e"])
        self.lr_m = float(tr["lr_m"])
        self.lr_m_dsm = float(tr["lr_m_dsm"])
        self.batch_size_e = int(tr["batch_size_e"])
        self.batch_size_m = int(tr["batch_size_m"])
        self.batch_size_dsm = int(tr["batch_size_dsm"])
        self.m_epochs = int(tr["m_epochs"])
        self.dsm_epochs = int(tr["dsm_epochs"])
        self.em_iterations = int(tr["em_iterations"])
        self.warmup_iters = int(tr.get("warmup_iters", 0))
        self.prior_weight = float(tr["prior_weight"])
        self.prior_weight_start = float(
            tr.get("prior_weight_start", self.prior_weight)
        )
        self.prior_ramp_iters = int(tr.get("prior_ramp_iters", 0))
        self.map_prior_mode = str(tr.get("map_prior_mode", "score")).lower()
        if self.map_prior_mode not in {"score", "denoise"}:
            raise ValueError("training.map_prior_mode must be 'score' or 'denoise'")
        self.map_denoise_scaling = str(
            tr.get("map_denoise_scaling", "data_relative")
        ).lower()
        if self.map_denoise_scaling not in {"data_relative", "fixed"}:
            raise ValueError(
                "training.map_denoise_scaling must be 'data_relative' or 'fixed'"
            )
        self.map_denoise_max_rel = float(tr.get("map_denoise_max_rel", 0.05))
        self.map_t_start = int(tr.get("map_t_start", max(self.t_min, self.T // 4)))
        self.map_t_start = min(max(self.map_t_start, self.t_min), self.T - 1)
        self.map_denoise_samples = max(1, int(tr.get("map_denoise_samples", 1)))
        self.checkpoint_every = int(tr["checkpoint_every"])
        self.ema_decay = float(tr["ema_decay"])

        # Min-SNR-gamma DSM weighting.
        self.min_snr_gamma = float(tr.get("min_snr_gamma", 5.0))

        # Condition dropout (CFG) at DSM time.
        self.cond_drop_prob = float(config["model"].get("cond_drop_prob", 0.15))

        # CFG inference.
        self.cfg_scale = float(tr.get("cfg_scale", 2.0))
        self.cfg_num_steps = int(tr.get("cfg_num_steps", 100))
        self.cfg_sigma_type = tr.get("cfg_sigma_type", "ddim")

        # UNIFIED method (train E-step + inference sampler must be consistent):
        #   map --MAP E-step + MAP inference        (unconditional net)
        #   dps --posterior-sampling E-step + DPS inference  (unconditional net)
        #   cfg --posterior-sampling E-step + CFG inference  (conditional net)
        # `conditional` says whether the score net takes x_tilde as input.
        self.method = tr.get("method", "map")
        self.conditional = (self.method == "cfg")
        self.sample_steps = int(tr.get("sample_steps", self.cfg_num_steps))
        self.dps_guidance_scale = float(tr.get("dps_guidance_scale", 1.0))
        # Static thresholding: clamp the predicted x_hat (normalized) each reverse
        # step so the high-t Tweedie blow-up (/sqrtalpha_bar_t, alpha_barT~=e-8) can't diverge ->NaN.
        self.sample_clip = float(tr.get("sample_clip", 2.5))

        # GPA gauge alignment.
        self.gpa_init = bool(tr.get("gpa_init", True))
        self.gpa_every = int(tr.get("gpa_every", 5))      # 0 ->one-time only
        self.gpa_iter = int(tr.get("gpa_iter", 10))
        self.gpa_tol = float(tr.get("gpa_tol", 1e-5))

        # Observation preprocessing.  `training.preprocess.method` selects the
        # method; RWR-specific kwargs still come from `training.rwr_target` and
        # are merged in (so existing configs keep working with method=rwr).
        self.rwr_cfg = tr.get("rwr_target", {}) or {}
        pp = tr.get("preprocess", {}) or {}
        self.preprocess_method = pp.get("method", "rwr")
        self.preprocess_cfg = {**self.rwr_cfg, **pp}


        mdl = config["model"]
        self.K = int(mdl["K"])

        self.exposure_min = float(tr.get("exposure_min", 0.05))
        self.exposure_max = float(tr.get("exposure_max", 20.0))
        self.nb_theta_min = float(tr.get("nb_theta_min", 0.1))
        self.nb_theta_max = float(tr.get("nb_theta_max", 100.0))
        theta0 = float(tr.get("nb_init_theta", 5.0))
        self.log_theta = nn.Parameter(
            torch.full(
                (int(config["model"].get("band_width", 50)),),
                math.log(theta0),
                device=device,
            )
        )

        seed_cfg = tr.get("clean_prior_init", {}) or {}
        self.seed_enabled = bool(seed_cfg.get("enabled", True))
        raw_n_clusters = seed_cfg.get("n_clusters", "auto")
        self.seed_n_clusters = (
            None if str(raw_n_clusters).lower() == "auto" else int(raw_n_clusters)
        )
        self.seed_cells_per_cluster = int(seed_cfg.get("cells_per_cluster", 200))
        self.seed_min_clusters = int(seed_cfg.get("min_clusters", 16))
        self.seed_max_clusters = int(seed_cfg.get("max_clusters", 32))
        self.seed_group_size = int(seed_cfg.get("group_size", 16))
        self.seed_max = int(seed_cfg.get("max_seeds", 320))
        self.seed_feature_bins = int(seed_cfg.get("feature_bins", 64))
        self.seed_max_strata = int(seed_cfg.get("max_strata", 15))
        self.seed_factor_epochs = int(seed_cfg.get("factor_epochs", 10))
        self.seed_factor_lr = float(seed_cfg.get("factor_lr", 1e-2))
        self.seed_pretrain_epochs = int(seed_cfg.get("pretrain_epochs", 20))
        self.seed_replay_fraction = float(seed_cfg.get("replay_fraction", 0.2))
        self.seed_only_dsm = bool(seed_cfg.get("seed_only_dsm", False))
        self.seed_noise = float(seed_cfg.get("seed_noise", 0.02))
        self.seed_random_state = int(seed_cfg.get("random_seed", 0))
        recovery_cfg = tr.get("population_recovery", {}) or {}
        self.recovery_enabled = bool(recovery_cfg.get("enabled", False))
        self.recovery_corruptions_per_seed = max(
            1, int(recovery_cfg.get("corruptions_per_seed", 1)))
        self.recovery_factor_epochs = max(
            1, int(recovery_cfg.get("factor_epochs", 20)))
        self.recovery_factor_lr = float(
            recovery_cfg.get("factor_lr", self.seed_factor_lr))
        self.recovery_loss_weight = float(
            recovery_cfg.get("loss_weight", 1.0))
        self.recovery_t_min = max(
            0, int(recovery_cfg.get("t_min", self.t_min)))
        self.recovery_t_max = min(
            self.T - 1,
            int(recovery_cfg.get("t_max", self.map_t_start)),
        )
        if self.recovery_t_max < self.recovery_t_min:
            raise ValueError(
                "training.population_recovery.t_max must be >= t_min")
        self.recovery_batch_size = max(
            1, int(recovery_cfg.get("batch_size", self.batch_size_dsm)))
        if self.recovery_enabled and not self.seed_enabled:
            raise ValueError(
                "population recovery requires clean_prior_init.enabled=true")
        pop_cfg = tr.get("population_conditioning", {}) or {}
        self.population_conditioned = bool(pop_cfg.get("enabled", False))
        self.population_temperature = float(pop_cfg.get("temperature", 1.0))
        if self.population_conditioned and self.method != "map":
            raise ValueError(
                "training.population_conditioning currently requires method=map"
            )
        if self.population_conditioned and not self.seed_enabled:
            raise ValueError(
                "population conditioning requires clean_prior_init.enabled=true"
            )
        if self.recovery_enabled and not self.population_conditioned:
            raise ValueError(
                "population recovery requires population_conditioning.enabled=true"
            )

        # State.
        self.z_scale: float = 1.0                             # global std of Z
        self.Z_hats: Optional[torch.Tensor] = None           # (N, M, K) real scale
        self._x_tilde: Optional[torch.Tensor] = None         # (N, M, D) auxiliary preprocessed input
        self._y_raw_band: Optional[torch.Tensor] = None      # (N, M, D) observed raw counts
        self._exposure: Optional[torch.Tensor] = None        # (N, D) cell/band depth exposure
        self._cond: Optional[torch.Tensor] = None            # normalized auxiliary input for net
        self._seed_Z: Optional[torch.Tensor] = None
        self._seed_cluster: Optional[torch.Tensor] = None
        self._cell_cluster: Optional[torch.Tensor] = None
        self._seed_population: Optional[torch.Tensor] = None
        self._recovery_corrupt_Z: Optional[torch.Tensor] = None
        self._recovery_seed_index: Optional[torch.Tensor] = None
        self._cell_population: Optional[torch.Tensor] = None
        self._population_distance_scale: float = 1.0
        self._seed_feature_mean: Optional[torch.Tensor] = None
        self._seed_feature_std: Optional[torch.Tensor] = None
        self._cluster_feature_centers: Optional[torch.Tensor] = None
        self._cond_mean: float = 0.0
        self._cond_std: float = 1.0
        self.ema: Optional[EMA] = None
        self._elbo_history: List[float] = []
        self._cell_names: List[str] = []
        self._map_update_stats = {
            "data": 0.0, "prior": 0.0, "latent": 0.0, "count": 0
        }
        self._last_recovery_loss: float = 0.0

    # ------------------------------------------------------------------
    # Helpers (band reconstruction / B(d)) --identical math to C1
    # ------------------------------------------------------------------

    def _resolve_band_width(self, M: int) -> int:
        D = self.config.get("model", {}).get("band_width") or 100
        return max(1, min(int(D), M - 1))


    def reconstruct_band(
        self,
        Z: torch.Tensor,
        b_dist: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """F_band[b, i, d-1] = <Z_i, Z_{i+d}> + B(d).  Z: (B, M, K)."""
        B_sz, M, _ = Z.shape
        D = self.D_band
        if b_dist is None:
            b_dist = self._b_dist

        F_band = torch.zeros(B_sz, M, D, device=Z.device, dtype=Z.dtype)
        for d in range(1, D + 1):
            if d >= M:
                break
            inner = (Z[:, : M - d, :] * Z[:, d:, :]).sum(-1)
            F_band[:, : M - d, d - 1] = inner

        return F_band + b_dist.view(1, 1, D)

    def _refresh_b_dist(self) -> None:
        with torch.no_grad():
            d_idx = torch.arange(
                1, self.D_band + 1, device=self.device, dtype=torch.float32
            )
            self._b_dist = self.band_bias.forward_distances(torch.log1p(d_idx)).detach()

    def _compute_b_dist_with_grad(self) -> torch.Tensor:
        d_idx = torch.arange(
            1, self.D_band + 1, device=self.device, dtype=torch.float32
        )
        return self.band_bias.forward_distances(torch.log1p(d_idx))

    @property
    def nb_theta(self) -> torch.Tensor:
        return self.log_theta[: self.D_band].exp().clamp(
            min=self.nb_theta_min, max=self.nb_theta_max
        )

    def _raw_nb_nll(
        self,
        F_band: torch.Tensor,
        y_raw: torch.Tensor,
        exposure: torch.Tensor,
        w: torch.Tensor,
        normalize_by: int,
    ) -> torch.Tensor:
        """Exposure-aware NB2 NLL on raw counts.

        The decoder emits a common-depth clean rate. Cell-level exposure absorbs
        library-size bias. Per-distance dispersion handles biological and
        sampling over-dispersion. All zeros remain supervised; unlike Poisson,
        their gradient with respect to log-mean saturates instead of growing
        without bound.
        """
        clean_rate = F.softplus(F_band)
        mu = clean_rate * exposure.view(exposure.shape[0], 1, self.D_band)
        theta = self.nb_theta.view(1, 1, self.D_band)
        log_theta_mu = torch.log(theta + mu + 1e-8)
        elem = (
            torch.lgamma(theta)
            + torch.lgamma(y_raw + 1.0)
            - torch.lgamma(y_raw + theta)
            + theta * (log_theta_mu - torch.log(theta))
            + y_raw * (log_theta_mu - torch.log(mu + 1e-8))
        )
        return (elem * w).sum() / max(int(normalize_by), 1)


    @staticmethod
    def _compute_exposure_from_raw(
        raw: np.ndarray,
        max_exposure: float = 20.0,
        min_exposure: float = 0.05,
    ) -> np.ndarray:
        """Strictly positive cell library-size exposure, repeated by distance.

        A per-cell/per-distance factor can be exactly zero when one stratum was
        not observed, which incorrectly makes every count at that distance
        impossible and removes its zero supervision. Sequencing depth is a
        cell-level nuisance, so estimate it from the whole modeled band.
        """
        cell_tot = raw.sum(axis=(1, 2), dtype=np.float64)
        positive = cell_tot > 0
        reference = cell_tot[positive].mean() if positive.any() else 1.0
        depth = cell_tot / max(reference, 1e-8)
        upper = max_exposure if max_exposure > 0 else np.inf
        depth = np.clip(depth, max(min_exposure, 1e-6), upper)
        exposure = np.repeat(depth[:, None], raw.shape[2], axis=1)
        return exposure.astype(np.float32)
    # ------------------------------------------------------------------
    # Normalization
    # ------------------------------------------------------------------

    def _refresh_z_scale(self) -> None:
        """Global std of the current Z_hats --defines Z_tilde = Z / z_scale on which
        the diffusion operates so that Var(Z_tilde) ~=1 (DDPM assumption)."""
        with torch.no_grad():
            # Keep restoration inputs on the normalization used to train the
            # fixed clean pseudo-bulk prior.
            source = (
                self._seed_Z
                if self.seed_only_dsm and self._seed_Z is not None
                else self.Z_hats
            )
            s = source.std()
            if torch.isfinite(s):
                self.z_scale = float(s.clamp(min=1e-3).item())
            else:                          # keep previous z_scale rather than go NaN
                logger.warning("  z_scale: non-finite std --keeping %.4f", self.z_scale)

    def _prior_score(
        self,
        Z: torch.Tensor,
        population_cond: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Unconditional diffusion prior score grad_Z log p(Z) at t_min, real scale.

        Works in normalized space then converts back:
            Z_tilde = Z / z_scale,  score_Z_tilde = -eps_theta(Z_tilde,t_min)/sqrt(1-alpha_bar),
            grad_Z log p(Z) = score_Z_tilde / z_scale.
        """
        zs = self.z_scale
        with torch.no_grad():
            Zt = Z / zs
            t = torch.full((Z.shape[0],), self.t_min, device=self.device, dtype=torch.long)
            eps_pred = self._call_net(Zt, t, population_cond)
        score_norm = -eps_pred / (1 - self.alpha_bars[self.t_min]).sqrt()
        return score_norm / zs

    def _diffusion_denoise(
        self,
        Z: torch.Tensor,
        t_idx: int,
        population_cond: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Estimate clean Z from correctly noised inputs at one diffusion level."""
        zs = self.z_scale
        ab = self.alpha_bars[t_idx]
        clean = Z / zs
        estimates = []
        with torch.no_grad():
            t = torch.full(
                (Z.shape[0],), t_idx, device=self.device, dtype=torch.long
            )
            for _ in range(self.map_denoise_samples):
                eps = torch.randn_like(clean)
                noisy = ab.sqrt() * clean + (1 - ab).sqrt() * eps
                eps_pred = self._call_net(noisy, t, population_cond)
                estimate = (
                    noisy - (1 - ab).sqrt() * eps_pred
                ) / ab.sqrt().clamp(min=1e-6)
                estimates.append(estimate.clamp(-self.sample_clip, self.sample_clip))
        return torch.stack(estimates).mean(dim=0) * zs

    def _call_net(self, Zt: torch.Tensor, t: torch.Tensor, cond) -> torch.Tensor:
        """Uniform score-net call: conditional net takes `cond`; unconditional
        net (map/dps variants) ignores it."""
        if self.population_conditioned:
            return self.score_net(Zt, t, population_cond=cond)
        if self.conditional:
            return self.score_net(Zt, t, cond=cond)
        return self.score_net(Zt, t)

    # ------------------------------------------------------------------
    # Auxiliary preprocessing + raw-count observation tensors
    # ------------------------------------------------------------------

    def _setup_imputed_target(self, dataset: ScHiCDataset, M: int) -> None:
        from ..data.preprocess import (
            apply_bandnorm, compute_observation_band, gaussian_smooth_i, load_raw_band,
        )

        cfg = self.preprocess_cfg
        method = self.preprocess_method
        chrom = dataset.chrom
        scool_path = dataset.scool_path
        cell_paths = list(dataset.cell_paths)
        cell_names = list(dataset.cell_names)
        N = len(cell_names)

        resolution = _resolve_binsize(
            scool_path, cell_paths[0], chrom,
            self.config.get("data", {}).get("resolution"),
        )
        cache_dir = cfg.get("cache_dir")
        # Cache key includes the method so switching preprocessing invalidates it.
        cache_path = (
            os.path.join(cache_dir, f"{chrom}_{method}_xtilde.npz")
            if cache_dir else None
        )

        band = None
        raw_from_aux = None
        if cache_path and os.path.isfile(cache_path):
            with np.load(cache_path, allow_pickle=True) as npz:
                cached = npz["band"]
                cached_names = npz["cell_names"].astype(str).tolist() \
                    if "cell_names" in npz.files else None
            if (cached.shape == (N, M, self.D_band)
                    and (cached_names is None or cached_names == cell_names)):
                band = cached.astype(np.float32)
                logger.info("  %s: loaded cached x_tilde (%s) from %s",
                            chrom, method, cache_path)
            else:
                logger.info("  %s: cache stale at %s; recomputing.", chrom, cache_path)

        if band is None:
            logger.info("  %s: computing x_tilde via preprocess method '%s' ...",
                        chrom, method)
            if method in ("bandnorm", "bandnorm_smooth", "raw_log1p"):
                logger.info("  %s: loading raw band once for auxiliary input and NB likelihood ...",
                            chrom)
                raw_from_aux = load_raw_band(
                    scool_path, cell_paths, chrom, M, self.D_band, progress_prefix="    "
                )
                if method == "raw_log1p":
                    band = np.log1p(raw_from_aux).astype(np.float32)
                else:
                    band = np.log1p(apply_bandnorm(raw_from_aux))
                    if method == "bandnorm_smooth":
                        band = gaussian_smooth_i(
                            band, float(cfg.get("smooth_sigma", 1.0))
                        )
                    band = band.astype(np.float32)
            else:
                band = compute_observation_band(
                    method=method,
                    scool_path=scool_path,
                    cell_paths=cell_paths,
                    chrom=chrom,
                    resolution=resolution,
                    M=M,
                    D_band=self.D_band,
                    cfg=cfg,
                    progress_prefix="    ",
                )
            if cache_path:
                os.makedirs(cache_dir, exist_ok=True)
                np.savez_compressed(
                    cache_path, band=band, cell_names=np.array(cell_names, dtype=str),
                )
                logger.info("  %s: cached x_tilde ->%s", chrom, cache_path)

        raw_cache_path = (
            os.path.join(cache_dir, f"{chrom}_raw_band_D{self.D_band}_cell_depth_nb_v1.npz")
            if cache_dir else None
        )
        raw = raw_from_aux
        exposure = (
            self._compute_exposure_from_raw(raw, self.exposure_max, self.exposure_min)
            if raw is not None else None
        )
        if raw_cache_path and os.path.isfile(raw_cache_path):
            with np.load(raw_cache_path, allow_pickle=True) as npz:
                cached_raw = npz["raw"]
                cached_exp = npz["exposure"]
                cached_names = npz["cell_names"].astype(str).tolist() \
                    if "cell_names" in npz.files else None
            if (cached_raw.shape == (N, M, self.D_band)
                    and cached_exp.shape == (N, self.D_band)
                    and (cached_names is None or cached_names == cell_names)):
                raw = cached_raw.astype(np.float32)
                exposure = cached_exp.astype(np.float32)
                logger.info("  %s: loaded cached raw band/exposure from %s",
                            chrom, raw_cache_path)
            else:
                logger.info("  %s: raw cache stale at %s; recomputing.",
                            chrom, raw_cache_path)
        if raw is None:
            logger.info("  %s: loading raw observed band for EM likelihood ...", chrom)
            raw = load_raw_band(scool_path, cell_paths, chrom, M, self.D_band)
            exposure = self._compute_exposure_from_raw(raw, self.exposure_max, self.exposure_min)
            if raw_cache_path:
                os.makedirs(cache_dir, exist_ok=True)
                np.savez_compressed(
                    raw_cache_path,
                    raw=raw,
                    exposure=exposure,
                    cell_names=np.array(cell_names, dtype=str),
                )
                logger.info("  %s: cached raw band/exposure ->%s",
                            chrom, raw_cache_path)

        self._y_raw_band = torch.from_numpy(raw).to(self.device)
        self._exposure = torch.from_numpy(exposure).to(self.device)
        self._cond_mean = float(band.mean())
        self._cond_std = float(band.std() + 1e-6)
        if self.conditional:
            self._x_tilde = torch.from_numpy(band).to(self.device)
            self._cond = ((self._x_tilde - self._cond_mean) / self._cond_std)
        else:
            self._x_tilde = None
            self._cond = None
        self._setup_clean_prior_seeds(band, M)
        pos = band[band > 0]
        exp_pos = exposure[exposure > 0]
        logger.info(
            "  %s: x_aux shape=%s, nnz=%d, mean(pos)=%.3g, max=%.3g | raw nnz=%d, exposure mean(pos)=%.3g max=%.3g | cond mu=%.3g sigma=%.3g",
            chrom, tuple(band.shape), pos.size,
            float(pos.mean()) if pos.size else 0.0, float(band.max()),
            int((raw > 0).sum()),
            float(exp_pos.mean()) if exp_pos.size else 0.0,
            float(exposure.max()) if exposure.size else 0.0,
            self._cond_mean, self._cond_std,
        )


    def _compressed_aux_features(
        self,
        band: np.ndarray,
        fit_stats: bool = False,
    ) -> np.ndarray:
        """Compact auxiliary features for seed clustering and reassignment."""
        _, M, D = band.shape
        S = min(max(1, self.seed_max_strata), D)
        B = min(max(1, self.seed_feature_bins), M)
        edges = np.linspace(0, M, B + 1).astype(np.int64)
        x = band[:, :, :S].astype(np.float32, copy=False)
        feats = []
        for b in range(B):
            lo, hi = int(edges[b]), int(edges[b + 1])
            if hi > lo:
                feats.append(x[:, lo:hi, :].mean(axis=1))
        feats.append(x.mean(axis=1))
        feat = np.concatenate(feats, axis=1).astype(np.float32)
        if fit_stats:
            mean = feat.mean(axis=0, keepdims=True)
            std = np.maximum(feat.std(axis=0, keepdims=True), 1e-6)
            self._seed_feature_mean = torch.from_numpy(mean.astype(np.float32))
            self._seed_feature_std = torch.from_numpy(std.astype(np.float32))
        elif self._seed_feature_mean is not None and self._seed_feature_std is not None:
            mean = self._seed_feature_mean.cpu().numpy()
            std = self._seed_feature_std.cpu().numpy()
            if mean.shape[1] != feat.shape[1]:
                raise ValueError(
                    f"seed feature width mismatch: checkpoint={mean.shape[1]}, input={feat.shape[1]}"
                )
        else:
            raise RuntimeError("seed feature normalization statistics are unavailable")
        return ((feat - mean) / std).astype(np.float32)

    def _population_from_features(
        self,
        feat: np.ndarray,
        fit_scale: bool = False,
    ) -> np.ndarray:
        if self._cluster_feature_centers is None:
            raise RuntimeError("population cluster centers are unavailable")
        centers = self._cluster_feature_centers.cpu().numpy()
        dist = ((feat[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        delta = dist - dist.min(axis=1, keepdims=True)
        if fit_scale:
            if centers.shape[0] > 1:
                second = np.partition(delta, 1, axis=1)[:, 1]
                positive = second[second > 1e-8]
                self._population_distance_scale = (
                    float(np.median(positive)) if positive.size else 1.0
                )
            else:
                self._population_distance_scale = 1.0
        scale = max(
            self._population_distance_scale
            * max(self.population_temperature, 1e-4),
            1e-6,
        )
        logits = -delta / scale
        logits -= logits.max(axis=1, keepdims=True)
        prob = np.exp(logits)
        prob /= np.maximum(prob.sum(axis=1, keepdims=True), 1e-12)
        padded = np.zeros((feat.shape[0], self.seed_max_clusters), dtype=np.float32)
        padded[:, :centers.shape[0]] = prob.astype(np.float32)
        return padded

    def population_condition_from_aux(self, band: np.ndarray) -> torch.Tensor:
        feat = self._compressed_aux_features(band, fit_stats=False)
        prob = self._population_from_features(feat, fit_scale=False)
        return torch.from_numpy(prob).to(self.device)

    def _setup_clean_prior_seeds(self, band: np.ndarray, M: int) -> None:
        """Build and factorize unsupervised mini-pseudo-bulk clean-rate seeds.

        Clustering uses only auxiliary preprocessed features. Seed targets are
        depth-normalized sums of raw counts, so the score prior never learns to
        reproduce bandnorm/RWR values.
        """
        if not self.seed_enabled or self._seed_Z is not None:
            return
        from sklearn.cluster import MiniBatchKMeans

        N = band.shape[0]
        if self.seed_n_clusters is None:
            target = math.ceil(N / max(1, self.seed_cells_per_cluster))
            requested = max(
                self.seed_min_clusters, min(self.seed_max_clusters, target)
            )
            max_groups = max(1, N // max(2, self.seed_group_size // 2))
            n_clusters = min(N, max_groups, requested)
        else:
            n_clusters = min(max(1, self.seed_n_clusters), N)
        if self.population_conditioned and n_clusters > self.seed_max_clusters:
            raise ValueError(
                f"population conditioning supports at most "
                f"{self.seed_max_clusters} clusters, got {n_clusters}"
            )
        feat = self._compressed_aux_features(band, fit_stats=True)
        kmeans = MiniBatchKMeans(
            n_clusters=n_clusters,
            batch_size=min(1024, max(32, N)),
            n_init=10,
            random_state=self.seed_random_state,
        )
        labels = kmeans.fit_predict(feat)
        self._cluster_feature_centers = torch.from_numpy(
            kmeans.cluster_centers_.astype(np.float32)
        )
        cell_population = self._population_from_features(feat, fit_scale=True)
        rng = np.random.default_rng(self.seed_random_state)
        groups = []
        group_labels = []
        min_group = max(2, self.seed_group_size // 2)
        for c in range(n_clusters):
            members = np.flatnonzero(labels == c)
            rng.shuffle(members)
            if members.size < 2:
                continue
            for start in range(0, members.size, self.seed_group_size):
                group = members[start:start + self.seed_group_size]
                if group.size >= min_group or (start == 0 and members.size >= 2):
                    groups.append(group)
                    group_labels.append(c)
        if not groups:
            raise RuntimeError("clean_prior_init could not form any pseudo-bulk group")
        if len(groups) > self.seed_max > 0:
            keep = np.sort(rng.choice(len(groups), self.seed_max, replace=False))
            groups = [groups[i] for i in keep]
            group_labels = [group_labels[i] for i in keep]

        raw = self._y_raw_band
        exposure = self._exposure[:, 0]
        targets = []
        seed_population = []
        for group in groups:
            idx = torch.as_tensor(group, device=self.device, dtype=torch.long)
            denom = exposure[idx].sum().clamp(min=1e-6)
            targets.append(raw[idx].sum(dim=0) / denom)
            seed_population.append(cell_population[group].mean(axis=0))
        target = torch.stack(targets).detach()

        scale = 1.0 / math.sqrt(self.K)
        seed_Z = (scale * torch.randn(
            len(groups), M, self.K, device=self.device
        )).requires_grad_(True)
        seed_params = [seed_Z] + list(self.band_bias.parameters())
        opt = optim.Adam(seed_params, lr=self.seed_factor_lr)
        w = (self._dist_w.view(1, 1, self.D_band)
             * self._valid_mask.view(1, M, self.D_band))
        for epoch in range(self.seed_factor_epochs):
            opt.zero_grad()
            b_dist_g = self._compute_b_dist_with_grad()
            rate = F.softplus(self.reconstruct_band(seed_Z, b_dist=b_dist_g))
            loss = ((rate - target * torch.log(rate + 1e-8)) * w).sum() / len(groups)
            loss.backward()
            nn.utils.clip_grad_norm_(seed_params, max_norm=5.0)
            opt.step()
        self._refresh_b_dist()
        with torch.no_grad():
            fitted = F.softplus(self.reconstruct_band(seed_Z))
            valid = w.expand_as(target) > 0
            xf = fitted[valid].float()
            yf = target[valid].float()
            xc = xf - xf.mean()
            yc = yf - yf.mean()
            seed_corr = float(
                ((xc * yc).mean() /
                 (xc.std(unbiased=False) * yc.std(unbiased=False) + 1e-8)).item()
            )
            seed_mae = float((xf - yf).abs().mean().item())
            self._seed_Z = batched_procrustes_align(
                seed_Z.detach(), max_iter=self.gpa_iter, tol=self.gpa_tol
            )
        self._seed_cluster = torch.as_tensor(
            group_labels, device=self.device, dtype=torch.long
        )
        self._cell_cluster = torch.as_tensor(labels, device=self.device, dtype=torch.long)
        self._cell_population = torch.from_numpy(cell_population).to(self.device)
        self._seed_population = torch.from_numpy(
            np.stack(seed_population).astype(np.float32)
        ).to(self.device)
        if self.recovery_enabled:
            self._setup_population_recovery_pairs(target)
        pop_entropy = -(
            cell_population * np.log(np.maximum(cell_population, 1e-12))
        ).sum(axis=1)
        pop_confidence = cell_population.max(axis=1)
        logger.info(
            "  clean prior seeds: %d pseudo-bulks from %d clusters (%s), group size <= %d, target zeros=%.1f%%, factor corr=%.4f MAE=%.4g",
            len(groups), n_clusters,
            "auto" if self.seed_n_clusters is None else "fixed",
            self.seed_group_size,
            100.0 * float((target <= 0).float().mean().item()),
            seed_corr,
            seed_mae,
        )
        logger.info(
            "  population condition: enabled=%s dim=%d active=%d distance_scale=%.4g "
            "max-prob mean=%.4f entropy mean=%.4f",
            self.population_conditioned, self.seed_max_clusters, n_clusters,
            self._population_distance_scale,
            float(pop_confidence.mean()), float(pop_entropy.mean()),
        )

    def _setup_population_recovery_pairs(
        self,
        clean_rate: torch.Tensor,
    ) -> None:
        """Create paired corrupted/clean seed latents using realistic depth.

        Each clean pseudo-bulk rate is sampled at an observed single-cell
        exposure. A data-only NB fit turns those thinned counts into the
        corrupted latent seen by the recovery denoiser.
        """
        if self._seed_Z is None or self._recovery_corrupt_Z is not None:
            return
        n_seed = self._seed_Z.shape[0]
        seed_index = torch.arange(
            n_seed, device=self.device, dtype=torch.long
        ).repeat_interleave(self.recovery_corruptions_per_seed)
        rng = np.random.default_rng(self.seed_random_state + 1701)
        observed_depth = self._exposure[:, 0].detach().cpu().numpy()
        depth = torch.from_numpy(
            rng.choice(observed_depth, size=len(seed_index), replace=True)
            .astype(np.float32)
        ).to(self.device)
        generator = torch.Generator(device=self.device)
        generator.manual_seed(self.seed_random_state + 1701)
        mu = clean_rate[seed_index] * depth.view(-1, 1, 1)
        synthetic = torch.poisson(mu.clamp(min=0), generator=generator)
        exposure = depth[:, None].expand(-1, self.D_band)
        w = (
            self._dist_w.view(1, 1, self.D_band)
            * self._valid_mask.view(1, clean_rate.shape[1], self.D_band)
        )

        corrupt_parts = []
        losses = []
        for start in range(0, len(seed_index), self.recovery_batch_size):
            sl = slice(start, min(start + self.recovery_batch_size, len(seed_index)))
            clean_z = self._seed_Z[seed_index[sl]].detach()
            z_corrupt = (
                clean_z + self.seed_noise * torch.randn_like(clean_z)
            ).requires_grad_(True)
            opt = optim.Adam([z_corrupt], lr=self.recovery_factor_lr)
            last = 0.0
            for _ in range(self.recovery_factor_epochs):
                opt.zero_grad()
                logits = self.reconstruct_band(z_corrupt)
                loss = self._raw_nb_nll(
                    logits, synthetic[sl], exposure[sl], w,
                    z_corrupt.shape[0],
                )
                loss.backward()
                nn.utils.clip_grad_norm_([z_corrupt], max_norm=5.0)
                opt.step()
                last = float(loss.item())
            with torch.no_grad():
                # Keep every pair in the clean seed's factor gauge.
                cross = torch.einsum("bmk,bmj->bkj", z_corrupt, clean_z)
                u, _, vh = torch.linalg.svd(cross, full_matrices=False)
                aligned = torch.bmm(z_corrupt, u @ vh)
                corrupt_parts.append(aligned.detach())
                losses.append(last)
        self._recovery_corrupt_Z = torch.cat(corrupt_parts, dim=0)
        self._recovery_seed_index = seed_index
        with torch.no_grad():
            clean = self._seed_Z[seed_index]
            displacement = (
                (self._recovery_corrupt_Z - clean).flatten(1).norm(dim=1)
                / clean.flatten(1).norm(dim=1).clamp(min=1e-8)
            )
        logger.info(
            "  population recovery pairs: %d | synthetic zeros=%.1f%% | "
            "data-fit NLL=%.4f | latent displacement=%.4f",
            len(seed_index),
            100.0 * float((synthetic <= 0).float().mean().item()),
            float(np.mean(losses)) if losses else float("nan"),
            float(displacement.mean().item()),
        )

    # ------------------------------------------------------------------
    # Z initialization
    # ------------------------------------------------------------------

    def _cluster_seed_centers(self) -> torch.Tensor:
        if self._seed_Z is None or self._seed_cluster is None:
            raise RuntimeError("pseudo-bulk seeds are unavailable")
        global_seed = self._seed_Z.mean(dim=0)
        if self._cluster_feature_centers is not None:
            n_clusters = self._cluster_feature_centers.shape[0]
        elif self._cell_cluster is not None:
            n_clusters = int(self._cell_cluster.max().item()) + 1
        else:
            n_clusters = int(self._seed_cluster.max().item()) + 1
        centers = []
        for c in range(n_clusters):
            hit = self._seed_cluster == c
            centers.append(self._seed_Z[hit].mean(dim=0) if hit.any() else global_seed)
        return torch.stack(centers)

    def initialize_Z_from_aux(
        self,
        band: np.ndarray,
        return_population: bool = False,
    ):
        """Reproduce training-time cluster-seed initialization for new observations."""
        if self._cluster_feature_centers is None:
            raise RuntimeError("checkpoint has no seed feature cluster centers")
        population = self.population_condition_from_aux(band)
        labels = population[:, :self._cluster_feature_centers.shape[0]].argmax(dim=1)
        Z = self._cluster_seed_centers()[labels].clone()
        if self.seed_noise > 0:
            Z.add_(self.seed_noise * torch.randn_like(Z))
        return (Z, population) if return_population else Z

    def _init_Z_hats(self, N: int, M: int) -> None:
        if self._seed_Z is None or self._cell_cluster is None:
            scale = 1.0 / math.sqrt(self.K)
            self.Z_hats = scale * torch.randn(N, M, self.K, device=self.device)
            return
        self.Z_hats = self._cluster_seed_centers()[self._cell_cluster].clone()
        if self.seed_noise > 0:
            self.Z_hats.add_(self.seed_noise * torch.randn_like(self.Z_hats))

    # ------------------------------------------------------------------
    # E-step (MAP data consistency + diffusion proximal prior)
    # ------------------------------------------------------------------

    def e_step_map_batch(
        self,
        Z_init: torch.Tensor,
        y_raw_batch: torch.Tensor,
        exposure_batch: torch.Tensor,
        M: int,
        lam: float,
        population_cond: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        Z = Z_init.detach().clone().requires_grad_(True)
        opt = optim.Adam([Z], lr=self.lr_e)
        w = self._dist_w.view(1, 1, self.D_band) * self._valid_mask.view(1, M, self.D_band)
        B = Z.shape[0]

        for step in range(self.K_E):
            Z_before = Z.detach().clone()
            opt.zero_grad()
            F_band = self.reconstruct_band(Z)
            total = self._raw_nb_nll(F_band, y_raw_batch, exposure_batch, w, B)
            if lam > 0 and self.map_prior_mode == "score":
                score = self._prior_score(Z.detach(), population_cond)
                prior_surrogate = -(score * Z).sum() / B
                total = total + lam * prior_surrogate
            total.backward()
            opt.step()
            data_delta = Z.detach() - Z_before

            if lam > 0 and self.map_prior_mode == "denoise":
                frac = 1.0 if self.K_E <= 1 else step / (self.K_E - 1)
                t_idx = int(round(
                    self.map_t_start + frac * (self.t_min - self.map_t_start)
                ))
                with torch.no_grad():
                    denoised = self._diffusion_denoise(
                        Z.detach(), t_idx, population_cond,
                    )
                    prior_delta = denoised - Z.detach()
                    data_norm = data_delta.flatten(1).norm(dim=1)
                    latent_norm = Z.detach().flatten(1).norm(dim=1)
                    prior_norm = prior_delta.flatten(1).norm(dim=1)
                    if self.map_denoise_scaling == "data_relative":
                        # Legacy behavior retained for old checkpoints.
                        scale = (
                            lam * data_norm / prior_norm.clamp(min=1e-8)
                        ).clamp(max=1.0)
                        applied = scale.view(-1, 1, 1) * prior_delta
                    else:
                        # Fixed PnP proximal relaxation. The clean-prior update
                        # remains active even when the NB data step is small.
                        applied = lam * prior_delta
                        applied_norm = applied.flatten(1).norm(dim=1)
                        if self.map_denoise_max_rel > 0:
                            typical_norm = self.z_scale * math.sqrt(M * self.K)
                            reference_norm = torch.maximum(
                                latent_norm,
                                torch.full_like(latent_norm, 0.25 * typical_norm),
                            )
                            max_norm = self.map_denoise_max_rel * reference_norm
                            clip = (
                                max_norm / applied_norm.clamp(min=1e-8)
                            ).clamp(max=1.0)
                            applied = clip.view(-1, 1, 1) * applied
                    Z.add_(applied)
                    self._map_update_stats["data"] += float(data_norm.mean().item())
                    self._map_update_stats["prior"] += float(
                        applied.flatten(1).norm(dim=1).mean().item()
                    )
                    self._map_update_stats["latent"] += float(
                        latent_norm.mean().item()
                    )
                    self._map_update_stats["count"] += 1
        return Z.detach()

    # ------------------------------------------------------------------
    # M-step part 1: B(d) and NB dispersion
    # ------------------------------------------------------------------

    def m_step_observation(self, M: int) -> float:
        N = self.Z_hats.shape[0]
        w = self._dist_w.view(1, 1, self.D_band) * self._valid_mask.view(1, M, self.D_band)
        last_loss = 0.0
        for _ in range(self.m_epochs):
            perm = torch.randperm(N).tolist()
            epoch_loss = 0.0
            for start in range(0, N, self.batch_size_m):
                idx = perm[start: start + self.batch_size_m]
                Z_batch = self.Z_hats[idx].detach()
                yb = self._y_raw_band[idx]
                eb = self._exposure[idx]

                self.m_opt.zero_grad()
                b_dist_g = self._compute_b_dist_with_grad()
                F_band = self.reconstruct_band(Z_batch, b_dist=b_dist_g)
                total_nll = self._raw_nb_nll(F_band, yb, eb, w, len(idx))
                total_nll.backward()
                m_params = list(self.band_bias.parameters()) + [self.log_theta]
                nn.utils.clip_grad_norm_(m_params, max_norm=1.0)
                self.m_opt.step()
                epoch_loss += total_nll.item()
            last_loss = epoch_loss / N
        self._refresh_b_dist()
        return last_loss

    # ------------------------------------------------------------------
    # M-step part 2: conditional DSM (Min-SNR weighted, CFG dropout)
    # ------------------------------------------------------------------

    def _dsm_batch(self, Z0: torch.Tensor, cond=None) -> float:
        t = torch.randint(0, self.T, (Z0.shape[0],), device=self.device)
        Z_t, eps = forward_diffuse(Z0, t, self.alpha_bars)
        if self.population_conditioned:
            if cond is None:
                raise RuntimeError("population-conditioned DSM requires conditions")
            eps_pred = self._call_net(Z_t, t, cond)
        elif self.conditional and cond is not None:
            eps_pred = self.score_net(
                Z_t, t, cond=cond, cond_drop_prob=self.cond_drop_prob
            )
        else:
            eps_pred = self._call_net(Z_t, t, None)
        wt = min_snr_weight(self.alpha_bars, t, self.min_snr_gamma).view(-1, 1, 1)
        loss = (wt * (eps_pred - eps) ** 2).mean()
        self.dsm_opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.score_net.parameters(), max_norm=1.0)
        self.dsm_opt.step()
        self.ema.update()
        return float(loss.item())

    def _population_recovery_batch(self, pair_idx: torch.Tensor) -> float:
        clean_idx = self._recovery_seed_index[pair_idx]
        clean = self._seed_Z[clean_idx].detach() / self.z_scale
        corrupt = self._recovery_corrupt_Z[pair_idx].detach() / self.z_scale
        cond = (
            self._seed_population[clean_idx]
            if self.population_conditioned else None
        )
        t = torch.randint(
            self.recovery_t_min, self.recovery_t_max + 1,
            (len(pair_idx),), device=self.device,
        )
        ab = self.alpha_bars[t].view(-1, 1, 1)
        eps = torch.randn_like(corrupt)
        z_t = ab.sqrt() * corrupt + (1 - ab).sqrt() * eps
        eps_pred = self._call_net(z_t, t, cond)
        clean_pred = (
            z_t - (1 - ab).sqrt() * eps_pred
        ) / ab.sqrt().clamp(min=1e-6)
        loss = self.recovery_loss_weight * F.smooth_l1_loss(
            clean_pred, clean, beta=0.1,
        )
        self.dsm_opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.score_net.parameters(), max_norm=1.0)
        self.dsm_opt.step()
        self.ema.update()
        return float(loss.item())

    def _population_recovery_epoch(self) -> float:
        if (not self.recovery_enabled
                or self._recovery_corrupt_Z is None
                or self._recovery_seed_index is None):
            return 0.0
        perm = torch.randperm(
            len(self._recovery_seed_index), device=self.device)
        total = 0.0
        steps = 0
        for start in range(0, len(perm), self.recovery_batch_size):
            idx = perm[start:start + self.recovery_batch_size]
            total += self._population_recovery_batch(idx)
            steps += 1
        return total / max(steps, 1)

    def _pretrain_clean_prior(self) -> float:
        if self._seed_Z is None or self.seed_pretrain_epochs <= 0:
            return 0.0
        seed_scale = float(self._seed_Z.std().clamp(min=1e-3).item())
        self.z_scale = seed_scale
        n = self._seed_Z.shape[0]
        total = 0.0
        steps = 0
        self.score_net.train()
        for _ in range(self.seed_pretrain_epochs):
            perm = torch.randperm(n, device=self.device)
            for s in range(0, n, self.batch_size_dsm):
                idx = perm[s:s + self.batch_size_dsm]
                cond = (
                    self._seed_population[idx]
                    if self.population_conditioned else None
                )
                total += self._dsm_batch(
                    self._seed_Z[idx] / seed_scale, cond=cond,
                )
                steps += 1
            self._last_recovery_loss = self._population_recovery_epoch()
        self.score_net.eval()
        return total / max(steps, 1)

    def m_step_dsm(self) -> float:
        source = (
            self._seed_Z
            if self.seed_only_dsm and self._seed_Z is not None
            else self.Z_hats
        )
        N = source.shape[0]
        zs = self.z_scale
        self.score_net.train()
        total = 0.0
        steps = 0
        recovery_total = 0.0
        recovery_steps = 0
        for _ in range(self.dsm_epochs):
            perm = torch.randperm(N, device=self.device)
            for s in range(0, N, self.batch_size_dsm):
                idx = perm[s:s + self.batch_size_dsm]
                Z0 = source[idx].detach() / zs
                if self.population_conditioned:
                    population = (
                        self._seed_population
                        if source is self._seed_Z else self._cell_population
                    )
                    if population is None:
                        raise RuntimeError("population conditions are unavailable")
                    cond = population[idx]
                else:
                    cond = (
                        self._cond[idx]
                        if self.conditional and source is self.Z_hats
                        else None
                    )
                if (not self.seed_only_dsm and not self.conditional
                        and self._seed_Z is not None
                        and self.seed_replay_fraction > 0):
                    replay_n = max(1, round(Z0.shape[0] * self.seed_replay_fraction))
                    seed_idx = torch.randint(
                        self._seed_Z.shape[0], (replay_n,), device=self.device
                    )
                    Z0 = torch.cat((Z0, self._seed_Z[seed_idx] / zs), dim=0)
                total += self._dsm_batch(Z0, cond=cond)
                steps += 1
            if self.recovery_enabled:
                recovery_total += self._population_recovery_epoch()
                recovery_steps += 1
        self._last_recovery_loss = (
            recovery_total / max(recovery_steps, 1)
            if recovery_steps else 0.0
        )
        self.score_net.eval()
        return total / max(steps, 1)
    # ------------------------------------------------------------------
    # ELBO proxy
    # ------------------------------------------------------------------

    def compute_elbo(self, M: int, dsm_loss: float) -> float:
        N = self.Z_hats.shape[0]
        w = self._dist_w.view(1, 1, self.D_band) * self._valid_mask.view(1, M, self.D_band)
        nll_data = 0.0
        with torch.no_grad():
            for s in range(0, N, self.batch_size_m):
                idx = list(range(s, min(s + self.batch_size_m, N)))
                Z = self.Z_hats[idx]
                yb = self._y_raw_band[idx]
                eb = self._exposure[idx]
                F_band = self.reconstruct_band(Z)
                nll = self._raw_nb_nll(F_band, yb, eb, w, len(idx)) * len(idx)
                nll_data += nll.item()
        return -nll_data / N - dsm_loss

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def em_train(
        self,
        dataset: ScHiCDataset,
        checkpoint_dir: Optional[str] = None,
        wandb_run=None,
    ) -> None:
        if checkpoint_dir:
            os.makedirs(checkpoint_dir, exist_ok=True)
        N = len(dataset)
        M = dataset.M
        self._cell_names = list(dataset.cell_names)
        wandb_prefix = f"{dataset.chrom}/" if dataset.chrom else ""

        self.D_band = self._resolve_band_width(M)
        self.config.setdefault("model", {})["band_width"] = self.D_band
        d_idx = torch.arange(1, self.D_band + 1, device=self.device, dtype=torch.float32)
        self._valid_mask = band_valid_mask(M, self.D_band, self.device)
        short_range_boost = 1.0 + 3.0 * torch.exp(-d_idx / 8.0)
        self._dist_w = (1.0 / (M - d_idx).clamp(min=1.0)) * short_range_boost
        self._refresh_b_dist()

        logger.info("Setting up auxiliary input plus raw-count EM observations ...")
        self._setup_imputed_target(dataset, M)
        if self.recovery_enabled and (
            self._recovery_corrupt_Z is None
            or self._recovery_seed_index is None
        ):
            raise RuntimeError(
                "Population recovery is enabled but recovery pairs are absent. "
                "An older checkpoint cannot resume this training objective; "
                "start a fresh run.")

        m_params = list(self.band_bias.parameters()) + [self.log_theta]
        self.m_opt = optim.Adam(m_params, lr=self.lr_m)
        self.dsm_opt = optim.Adam(self.score_net.parameters(), lr=self.lr_m_dsm)
        if self.ema is None:
            self.ema = EMA(self.score_net, decay=self.ema_decay)

        if self.Z_hats is None:
            if self._seed_Z is not None:
                logger.info("Pretraining diffusion prior on clean-ish pseudo-bulk seeds ...")
                seed_dsm = self._pretrain_clean_prior()
                logger.info(
                    "  seed DSM loss = %.6f | population recovery loss = %.6f",
                    seed_dsm, self._last_recovery_loss,
                )
                logger.info("Initializing Z_hats from pseudo-bulk cluster seeds ...")
            else:
                logger.warning("No clean-prior seeds; falling back to Gaussian Z initialization.")
            self._init_Z_hats(N, M)
        self._refresh_z_scale()

        gpa_done = not self.gpa_init
        for em_iter in range(self.em_iterations):
            # E-step method: MAP during warmup and always for method='map';
            # otherwise the posterior-sampling E-step matching self.method (dps/cfg),
            # so training and inference stay consistent.
            use_map = (self.method == "map") or (em_iter < self.warmup_iters)

            lam = 0.0
            prior_ready = self._seed_Z is not None or em_iter >= self.warmup_iters
            if use_map and prior_ready:
                ramp = (min(em_iter / self.prior_ramp_iters, 1.0)
                        if self.prior_ramp_iters > 0 else 1.0)
                prior_ratio = self.prior_weight_start + ramp * (
                    self.prior_weight - self.prior_weight_start
                )
                if self.map_prior_mode == "denoise":
                    lam = prior_ratio
                    logger.info(
                        "\n=== EM iter %d/%d | E=map-denoise | "
                        "scaling=%s strength=%.3g max_rel=%.3g | t=%d->%d ===",
                        em_iter, self.em_iterations - 1,
                        self.map_denoise_scaling, lam, self.map_denoise_max_rel,
                        self.map_t_start, self.t_min,
                    )
                else:
                    cal_idx = list(range(min(self.batch_size_e, N)))
                    Z_cal = self.Z_hats[cal_idx].detach().clone().requires_grad_(True)
                    yb = self._y_raw_band[cal_idx]
                    eb = self._exposure[cal_idx]
                    w = (self._dist_w.view(1, 1, self.D_band)
                         * self._valid_mask.view(1, M, self.D_band))
                    F_band = self.reconstruct_band(Z_cal)
                    data_loss = self._raw_nb_nll(F_band, yb, eb, w, len(cal_idx))
                    data_g = torch.autograd.grad(data_loss, Z_cal)[0]
                    pop_cal = (
                        self._cell_population[cal_idx]
                        if self.population_conditioned else None
                    )
                    score = self._prior_score(Z_cal.detach(), pop_cal)
                    zn = data_g.flatten(1).norm(dim=1).mean().item()
                    pn = score.flatten(1).norm(dim=1).mean().item()
                    lam = prior_ratio * (zn / pn if pn > 1e-8 else 1.0)
                    logger.info("\n=== EM iter %d/%d | E=map-score | ||grad_data||=%.3g ||grad_prior||=%.3g -> lambda=%.3g (ratio=%.3g) ===",
                                em_iter, self.em_iterations - 1, zn, pn, lam, prior_ratio)
            else:
                logger.info("\n=== EM iter %d/%d | E=%s%s ===", em_iter,
                            self.em_iterations - 1, "map" if use_map else self.method,
                            " (WARMUP)" if em_iter < self.warmup_iters else "")

            # GPA gauge alignment: once at end of warmup, then every gpa_every.
            do_gpa = False
            if not gpa_done and em_iter >= self.warmup_iters:
                do_gpa = True
                gpa_done = True
            elif (gpa_done and self.gpa_every > 0 and em_iter > self.warmup_iters
                  and (em_iter - self.warmup_iters) % self.gpa_every == 0):
                do_gpa = True
            if do_gpa:
                logger.info("Running GPA on Z_hats (max_iter=%d) ...", self.gpa_iter)
                with torch.no_grad():
                    self.Z_hats = batched_procrustes_align(
                        self.Z_hats, max_iter=self.gpa_iter, tol=self.gpa_tol,
                    )

            # ---- E-step (MAP or posterior sampling, per self.method) ----
            logger.info("E-step (%s): updating Z_n ...", "map" if use_map else self.method)
            self._map_update_stats = {
                "data": 0.0, "prior": 0.0, "latent": 0.0, "count": 0
            }
            for start in range(0, N, self.batch_size_e):
                idx = list(range(start, min(start + self.batch_size_e, N)))
                if use_map:
                    self.Z_hats[idx] = self.e_step_map_batch(
                        self.Z_hats[idx], self._y_raw_band[idx],
                        self._exposure[idx], M, lam,
                        self._cell_population[idx]
                        if self.population_conditioned else None,
                    )
                else:                                  # dps/cfg posterior sample
                    cond = self._cond[idx] if self.conditional else None
                    Zs = self._sample(self._y_raw_band[idx], self._exposure[idx], cond, M, self.sample_steps)
                    bad = ~torch.isfinite(Zs).flatten(1).all(dim=1)
                    if bad.any():                      # keep previous Z for diverged cells
                        Zs[bad] = self.Z_hats[idx][bad]
                        logger.warning("  %d/%d sampled Z non-finite --kept previous.",
                                       int(bad.sum()), len(idx))
                    self.Z_hats[idx] = Zs
                if start % max(1, N // 5) == 0:
                    logger.info("  cell %d/%d", start, N)
            if use_map and self.map_prior_mode == "denoise":
                ns = self._map_update_stats["count"]
                if ns:
                    data_update = self._map_update_stats["data"] / ns
                    prior_update = self._map_update_stats["prior"] / ns
                    latent_norm = self._map_update_stats["latent"] / ns
                    logger.info(
                        "  MAP update norms: data=%.4g diffusion=%.4g "
                        "diffusion/data=%.3f diffusion/latent=%.4f",
                        data_update, prior_update,
                        prior_update / max(data_update, 1e-12),
                        prior_update / max(latent_norm, 1e-12),
                    )

            # Refresh z_scale ONLY from MAP estimates (data-driven, stable).  During
            # a sampling E-step, Z_hats = x_hat*z_scale, so refreshing from them creates
            # a positive feedback (z_scale ->std(x_hat*z_scale)) that blows up ->freeze it.
            if use_map:
                self._refresh_z_scale()
            logger.info("  z_scale = %.4f", self.z_scale)

            # ---- M-step 1: B(d) + NB dispersion ----
            logger.info("M-step 1: B(d) + NB dispersion ...")
            avg_nll = self.m_step_observation(M)
            logger.info("  avg NLL = %.4f | theta=[%.3g, %.3g]", avg_nll,
                        float(self.nb_theta.min().detach()),
                        float(self.nb_theta.max().detach()))

            # ---- M-step 2: DSM (conditional for cfg, else unconditional) ----
            logger.info("M-step 2: DSM (%s, Min-SNR gamma=%.1f) ...",
                        ("population-cond" if self.population_conditioned
                         else "cond" if self.conditional else "uncond"),
                        self.min_snr_gamma)
            dsm_loss = self.m_step_dsm()
            logger.info(
                "  DSM loss = %.6f | population recovery loss = %.6f",
                dsm_loss, self._last_recovery_loss,
            )

            elbo = self.compute_elbo(M, dsm_loss)
            self._elbo_history.append(elbo)
            logger.info("  ELBO ~=%.4f", elbo)

            if wandb_run is not None:
                wandb_run.log({
                    "em_iter": em_iter,
                    f"{wandb_prefix}neg_elbo": -elbo,
                    f"{wandb_prefix}avg_nll": avg_nll,
                    f"{wandb_prefix}dsm_loss": dsm_loss,
                    f"{wandb_prefix}population_recovery_loss":
                        self._last_recovery_loss,
                    f"{wandb_prefix}nb_theta_mean": float(self.nb_theta.mean().item()),
                    f"{wandb_prefix}z_scale": self.z_scale,
                    f"{wandb_prefix}lambda": lam,
                })

            if checkpoint_dir and (em_iter % self.checkpoint_every == 0 or em_iter == 0):
                self.save_checkpoint(checkpoint_dir, em_iter, lam)

        if checkpoint_dir:
            self.save_checkpoint(checkpoint_dir, "final", lam)
        logger.info("EM training complete.")

    # ------------------------------------------------------------------
    # Inference: posterior sampling (DPS or CFG, per self.method)
    # ------------------------------------------------------------------

    def sample_posterior(
        self,
        x_tilde: torch.Tensor,        # (M, D_band) auxiliary preprocessed band
        y_raw: Optional[torch.Tensor] = None,
        exposure: Optional[torch.Tensor] = None,
        num_samples: int = 1,
        num_steps: Optional[int] = None,
        use_ema: bool = True,
        **_,                          # tolerate legacy kwargs (e.g. cfg_scale)
    ) -> torch.Tensor:
        """Posterior sample Z using auxiliary conditioning plus raw likelihood.

        For CFG checkpoints, x_tilde is only the conditioning/proposal signal.
        DPS guidance uses y_raw/exposure when supplied; if omitted, it falls back
        to expm1(x_tilde) with unit exposure for legacy callers.
        """
        num_steps = self.sample_steps if num_steps is None else num_steps
        if use_ema and self.ema is not None:
            self.ema.apply_shadow()
        try:
            M = x_tilde.shape[0]
            if y_raw is None:
                y_raw = torch.expm1(x_tilde)
            if exposure is None:
                exposure = torch.ones(self.D_band, device=self.device, dtype=torch.float32)
            yb = y_raw.to(self.device).unsqueeze(0).expand(num_samples, -1, -1)
            eb = exposure.to(self.device).unsqueeze(0).expand(num_samples, -1)
            cond = None
            if self.conditional:      # cfg: standardize auxiliary input as the network condition
                cond = ((x_tilde.to(self.device) - self._cond_mean) / self._cond_std)
                cond = cond.unsqueeze(0).expand(num_samples, -1, -1)
            Z = self._sample(yb, eb, cond, M, num_steps)
        finally:
            if use_ema and self.ema is not None:
                self.ema.restore()
        return Z.cpu()

    def _sample(
        self,
        y_raw: torch.Tensor,
        exposure: torch.Tensor,
        cond,
        M: int,
        num_steps: int,
    ) -> torch.Tensor:
        """Unified DDIM reverse sampler in NORMALIZED Z_tilde space, returning real-
        scale Z (xz_scale).  cond is not None ->CFG; cond is None ->DPS with
        raw-count exposure-aware likelihood guidance.  Used by both E-step and
        inference."""
        B = cond.shape[0] if cond is not None else y_raw.shape[0]
        zs = self.z_scale
        w_band = (self._dist_w.view(1, 1, self.D_band)
                  * self._valid_mask.view(1, M, self.D_band))
        step_idx = torch.linspace(self.T - 1, 0, num_steps + 1).long().clamp(0, self.T - 1)
        Z_t = torch.randn(B, M, self.K, device=self.device)
        self.score_net.eval()
        for i in range(len(step_idx) - 1):
            t_cur = int(step_idx[i].item())
            t_prev = int(step_idx[i + 1].item())
            ab_t = self.alpha_bars[t_cur]
            ab_prev = (self.alpha_bars[t_prev] if t_prev >= 0
                       else torch.tensor(1.0, device=self.device))
            t_vec = torch.full((B,), t_cur, device=self.device, dtype=torch.long)

            if cond is not None:                               # ---- CFG ----
                with torch.no_grad():
                    eps_c = self._call_net(Z_t, t_vec, cond)
                    if abs(self.cfg_scale - 1.0) > 1e-6:
                        eps_u = self._call_net(Z_t, t_vec, None)
                        eps_used = eps_u + self.cfg_scale * (eps_c - eps_u)
                    else:
                        eps_used = eps_c
                Z0 = (Z_t - (1 - ab_t).sqrt() * eps_used) / ab_t.sqrt()
            else:                                              # ---- DPS ----
                Zt_g = Z_t.detach().requires_grad_(True)
                eps_pred = self._call_net(Zt_g, t_vec, None)
                Z0n = (Zt_g - (1 - ab_t).sqrt() * eps_pred) / ab_t.sqrt()
                F0 = self.reconstruct_band(Z0n * zs)
                # Normalized raw-count likelihood gradient to prevent explosion.
                n_eff = w_band.sum().clamp(min=1.0)
                nll = self._raw_nb_nll(F0, y_raw, exposure, w_band, 1)
                log_lik = -nll / n_eff
                grad = torch.autograd.grad(log_lik, Zt_g)[0]
                w_t = self.dps_guidance_scale * (ab_t.item() ** 0.5)
                eps_used = eps_pred.detach() - (1 - ab_t).sqrt() * w_t * grad.detach()
                Z0 = Z0n.detach()

            Z0 = Z0.clamp(-self.sample_clip, self.sample_clip)   # static thresholding

            if self.cfg_sigma_type == "ddim":
                sigma_t = 0.0
            else:
                beta_t = self.betas[t_cur]
                sigma_t = float(((1 - ab_prev) * beta_t / (1 - ab_t)).sqrt().item())
            dir_coeff = (1 - ab_prev - sigma_t ** 2).clamp(min=0).sqrt()
            Z_t = (ab_prev.sqrt() * Z0 + dir_coeff * eps_used).detach()
            if sigma_t > 0:
                Z_t = Z_t + sigma_t * torch.randn_like(Z_t)
        return (Z_t * zs).detach()

    # ------------------------------------------------------------------
    # Checkpoint I/O
    # ------------------------------------------------------------------

    def save_checkpoint(self, checkpoint_dir: str, label, lam: float) -> None:
        path = os.path.join(checkpoint_dir, f"schic_em_c2_{label}.pt")
        payload = {
            "score_net": self.score_net.state_dict(),
            "band_bias": self.band_bias.state_dict(),
            "log_theta": self.log_theta.detach().cpu(),
            "seed_Z": self._seed_Z.detach().cpu().half() if self._seed_Z is not None else None,
            "seed_cluster": self._seed_cluster.detach().cpu() if self._seed_cluster is not None else None,
            "cell_cluster": self._cell_cluster.detach().cpu() if self._cell_cluster is not None else None,
            "seed_population": (
                self._seed_population.detach().cpu().half()
                if self._seed_population is not None else None
            ),
            "recovery_corrupt_Z": (
                self._recovery_corrupt_Z.detach().cpu().half()
                if self._recovery_corrupt_Z is not None else None
            ),
            "recovery_seed_index": (
                self._recovery_seed_index.detach().cpu()
                if self._recovery_seed_index is not None else None
            ),
            "cell_population": (
                self._cell_population.detach().cpu().half()
                if self._cell_population is not None else None
            ),
            "population_distance_scale": self._population_distance_scale,
            "seed_feature_mean": self._seed_feature_mean,
            "seed_feature_std": self._seed_feature_std,
            "cluster_feature_centers": self._cluster_feature_centers,
            "Z_hats": self.Z_hats.detach().cpu().half(),      # real scale
            "z_scale": self.z_scale,
            "cond_mean": self._cond_mean,
            "cond_std": self._cond_std,
            "cell_names": self._cell_names,
            "elbo_history": self._elbo_history,
            "lambda": lam,
            "ema": self.ema.state_dict() if self.ema is not None else None,
            "config": self.config,
        }
        tmp_path = path + ".tmp"
        try:
            torch.save(payload, tmp_path)
            os.replace(tmp_path, path)
            logger.info("Checkpoint saved: %s (%.1f MB)",
                        path, os.path.getsize(path) / 1024 ** 2)
        except (OSError, RuntimeError) as e:
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass
            raise RuntimeError(f"Checkpoint save failed at {path}: {e}") from e

    def load_checkpoint(self, path: str) -> dict:
        ckpt = torch.load(path, map_location=self.device)
        self.score_net.load_state_dict(ckpt["score_net"])
        self.band_bias.load_state_dict(ckpt["band_bias"])
        with torch.no_grad():
            if "log_theta" in ckpt:
                saved_theta = ckpt["log_theta"].to(self.device)
                n = min(saved_theta.numel(), self.log_theta.numel())
                self.log_theta[:n].copy_(saved_theta[:n])
        self.Z_hats = ckpt["Z_hats"].float().to(self.device)
        if ckpt.get("seed_Z") is not None:
            self._seed_Z = ckpt["seed_Z"].float().to(self.device)
        if ckpt.get("seed_cluster") is not None:
            self._seed_cluster = ckpt["seed_cluster"].long().to(self.device)
        if ckpt.get("cell_cluster") is not None:
            self._cell_cluster = ckpt["cell_cluster"].long().to(self.device)
        if ckpt.get("seed_population") is not None:
            self._seed_population = ckpt["seed_population"].float().to(self.device)
        if ckpt.get("recovery_corrupt_Z") is not None:
            self._recovery_corrupt_Z = (
                ckpt["recovery_corrupt_Z"].float().to(self.device)
            )
        if ckpt.get("recovery_seed_index") is not None:
            self._recovery_seed_index = (
                ckpt["recovery_seed_index"].long().to(self.device)
            )
        if ckpt.get("cell_population") is not None:
            self._cell_population = ckpt["cell_population"].float().to(self.device)
        self._population_distance_scale = float(
            ckpt.get("population_distance_scale", 1.0)
        )
        if ckpt.get("seed_feature_mean") is not None:
            self._seed_feature_mean = ckpt["seed_feature_mean"].float().cpu()
        if ckpt.get("seed_feature_std") is not None:
            self._seed_feature_std = ckpt["seed_feature_std"].float().cpu()
        if ckpt.get("cluster_feature_centers") is not None:
            self._cluster_feature_centers = ckpt["cluster_feature_centers"].float().cpu()
        self.z_scale = float(ckpt.get("z_scale", 1.0))
        self._cond_mean = float(ckpt.get("cond_mean", 0.0))
        self._cond_std = float(ckpt.get("cond_std", 1.0))
        self._cell_names = ckpt.get("cell_names", [])
        self._elbo_history = ckpt.get("elbo_history", [])
        if ckpt.get("ema") is not None:
            if self.ema is None:
                self.ema = EMA(self.score_net, decay=self.ema_decay)
            self.ema.load_state_dict(ckpt["ema"])
        logger.info("Checkpoint loaded: %s", path)
        return ckpt
