"""
Diffusion noise schedules and loss weighting for the scHiC latent diffusion.

Borrows two well-established tricks (cf. ChromoGen, Nichol & Dhariwal 2021,
Hang et al. 2023 "Min-SNR"):

  - cosine beta schedule: preserves signal far longer in the high-t regime
    than the linear schedule, which matters a lot for low-dimensional /
    structured latents like Z.
  - Min-SNR-gamma loss weighting: caps the effective SNR of the eps-prediction
    DSM loss, balancing the gradient contribution across timesteps and
    speeding convergence.
"""

from __future__ import annotations

import math
from typing import Tuple

import torch


def make_alpha_bars(
    T: int,
    schedule: str = "cosine",
    beta_start: float = 1e-4,
    beta_end: float = 0.02,
    cosine_s: float = 0.008,
    device: torch.device = torch.device("cpu"),
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (betas, alpha_bars), both shape (T,).

    schedule = "linear"  → original DDPM linear beta schedule.
    schedule = "cosine"  → Nichol & Dhariwal cosine ᾱ schedule (default).
    """
    if schedule == "linear":
        betas = torch.linspace(beta_start, beta_end, T, device=device)
        alpha_bars = torch.cumprod(1.0 - betas, dim=0)
        return betas, alpha_bars

    if schedule == "cosine":
        steps = torch.arange(T + 1, device=device, dtype=torch.float64)
        f = torch.cos(((steps / T + cosine_s) / (1 + cosine_s)) * math.pi / 2) ** 2
        ab_full = (f / f[0]).clamp(1e-8, 1.0)            # ᾱ_0..ᾱ_T, ᾱ_0 = 1
        alpha_bars = ab_full[1:]                          # ᾱ_1..ᾱ_T  → (T,)
        # β_t = 1 - ᾱ_t / ᾱ_{t-1}, with β_1 = 1 - ᾱ_1.
        betas = (1.0 - ab_full[1:] / ab_full[:-1]).clamp(1e-8, 0.999)
        return betas.float().to(device), alpha_bars.float().to(device)

    raise ValueError(f"Unknown schedule '{schedule}'. Choose 'cosine' or 'linear'.")


def min_snr_weight(
    alpha_bars: torch.Tensor,
    t_idx: torch.Tensor,
    gamma: float = 5.0,
) -> torch.Tensor:
    """Min-SNR-gamma weight for an eps-prediction loss.

    SNR(t) = ᾱ_t / (1 - ᾱ_t).  For eps-prediction the per-sample weight is
    min(SNR, gamma) / SNR = min(1, gamma / SNR).  Returns (B,).
    """
    ab = alpha_bars[t_idx]
    snr = ab / (1.0 - ab).clamp(min=1e-8)
    return torch.clamp(gamma / snr.clamp(min=1e-8), max=1.0)
