"""
Geometric components for scHiC latent diffusion:
  - BandBiasNetwork  : g_phi(|i-j|) → distance-decay bias matrix B (M x M)
  - ZINBLikelihood   : ZINB observation model with gradients
  - GeometricModel   : combines Z, B → F = ZZ^T + B, wraps ZINB

Key design choices vs. standard RNA-seq ZINB (scVI):
  1. mu normalization: mu_ij = s × softmax_upper(F)_ij so sum(mu) = s (library
     size).  Without this, exp(Z@Z^T) is O(exp(K)) and mu explodes.
  2. dropout direction: pi_ij increases with genomic distance.  Long-range
     entries are predominantly structural zeros; short-range has genuine signal.
  3. all log-probabilities computed in log-space (logaddexp) to avoid overflow
     when theta or mu are extreme.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Band-bias network  g_phi : log(1+|i-j|) → scalar
# ---------------------------------------------------------------------------

class BandBiasNetwork(nn.Module):
    """
    Neural network that maps log-distance to a scalar bias.
    B[i,j] = g_phi(log(1 + |i-j|))
    """

    def __init__(self, hidden_dim: int = 64, num_layers: int = 3) -> None:
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(1, hidden_dim), nn.SiLU()]
        for _ in range(num_layers - 2):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.SiLU()]
        layers.append(nn.Linear(hidden_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, M: int, device: torch.device) -> torch.Tensor:
        """Returns (M, M) symmetric bias matrix. Only M MLP evals (one per unique distance)."""
        d_vals = torch.arange(M, device=device, dtype=torch.float32)
        b_vals = self.net(torch.log1p(d_vals).unsqueeze(-1)).squeeze(-1)  # (M,)
        idx = torch.arange(M, device=device)
        dist_mat = (idx.unsqueeze(0) - idx.unsqueeze(1)).abs()            # (M, M) int
        return b_vals[dist_mat]

    def forward_distances(self, log_dists: torch.Tensor) -> torch.Tensor:
        return self.net(log_dists.unsqueeze(-1)).squeeze(-1)


# ---------------------------------------------------------------------------
# ZINB likelihood
# ---------------------------------------------------------------------------

class ZINBLikelihood(nn.Module):
    """
    Zero-inflated negative binomial for Hi-C contact counts.

    Parameterization (scVI-3D style):
      mu_ij  = s × softmax_over_upper_triangle(F)_ij
               → guarantees sum(mu) = s (library size)
      pi_ij  = sigmoid(alpha_0 + alpha_1 × log(1+|i-j|))
               → increases with distance (more structural zeros far-range)
      theta  : NB inverse-dispersion, clamped to (1e-4, 1e4)

    All log-probabilities computed via logaddexp for numerical stability.
    """

    def __init__(
        self,
        init_theta: float = 1.0,
        init_alpha0: float = -2.0,
        init_alpha1: float = 1.0,
    ) -> None:
        super().__init__()
        self.log_theta = nn.Parameter(torch.tensor(math.log(init_theta)))
        self.alpha_0 = nn.Parameter(torch.tensor(float(init_alpha0)))
        self.alpha_1 = nn.Parameter(torch.tensor(float(init_alpha1)))

    @property
    def theta(self) -> torch.Tensor:
        return self.log_theta.exp().clamp(min=1e-4, max=1e4)

    def dropout_prob(self, M: int, device: torch.device) -> torch.Tensor:
        """pi (M, M): increases with genomic distance."""
        idx = torch.arange(M, device=device, dtype=torch.float32)
        dist = (idx.unsqueeze(0) - idx.unsqueeze(1)).abs()
        log_dist = torch.log1p(dist)
        return torch.sigmoid(self.alpha_0 + self.alpha_1 * log_dist)

    def _get_pi(self, M: int, device: torch.device) -> torch.Tensor:
        """Return cached pi if available, otherwise compute fresh."""
        cache = getattr(self, "_pi_cache", None)
        if cache is not None and cache.shape[0] == M:
            return cache
        return self.dropout_prob(M, device)

    # ------------------------------------------------------------------
    # Internal helpers (log-space for stability)
    # ------------------------------------------------------------------

    def _log_nb_zero(self, log_mu: torch.Tensor) -> torch.Tensor:
        """
        log P(NB = 0 | mu, theta) = theta × log(theta / (mu + theta))
                                   = theta × (log_theta - log(exp(log_mu) + theta))

        Works in log-mu space to avoid exp overflow.
        """
        theta = self.theta
        log_theta = theta.log()
        log_mu_plus_theta = torch.logaddexp(log_mu, log_theta.expand_as(log_mu))
        return theta * (log_theta - log_mu_plus_theta)

    def _log_nb_positive(
        self,
        Y: torch.Tensor,       # positive counts (float)
        log_mu: torch.Tensor,  # log mean
    ) -> torch.Tensor:
        """
        log P(NB = Y | mu, theta) for Y > 0.
        Uses log_mu to avoid exp overflow in intermediate steps.
        """
        theta = self.theta
        log_theta = theta.log()
        log_mu_plus_theta = torch.logaddexp(log_mu, log_theta.expand_as(log_mu))
        return (
            torch.lgamma(Y + theta)
            - torch.lgamma(theta)
            - torch.lgamma(Y + 1)
            + theta * (log_theta - log_mu_plus_theta)
            + Y    * (log_mu   - log_mu_plus_theta)
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def log_prob(
        self,
        Y: torch.Tensor,       # (M, M) observed counts (symmetric)
        F_hat: torch.Tensor,   # (M, M) predicted log-signal
    ) -> torch.Tensor:
        """
        ZINB log-likelihood summed over the upper triangle.

        mu_ij = softplus(F_ij) — ensures mu > 0 without exp overflow.
        No global normalization; the band-bias network B(d) absorbs scale.
        Diagonal excluded (self-ligation artifacts; F[i,i]=||Z_i||² always large).
        """
        M = Y.shape[0]
        device = Y.device

        # Exclude diagonal: self-ligation artifacts + F[i,i]=||Z_i||² dominates.
        triu_mask = torch.ones(M, M, device=device, dtype=torch.bool).triu(diagonal=1)
        F_upper = F_hat[triu_mask]                        # (n_offdiag,)
        Y_upper = Y[triu_mask]
        pi_upper = self._get_pi(M, device)[triu_mask]

        # mu = softplus(F): linear for large F (no exp explosion), smooth near 0.
        log_mu_upper = torch.nn.functional.softplus(F_upper).clamp(min=1e-8).log()

        log_nb0 = self._log_nb_zero(log_mu_upper)         # (n_tri,)

        # log pi and log(1-pi)
        log_pi    = torch.log(pi_upper.clamp(min=1e-8))
        log_1mpi  = torch.log((1.0 - pi_upper).clamp(min=1e-8))

        zero_mask = Y_upper == 0
        pos_mask  = ~zero_mask

        # Zero entries: log(pi + (1-pi)*NB(0))
        # = logaddexp(log_pi, log_1mpi + log_nb0)
        log_lik_zero = torch.logaddexp(
            log_pi[zero_mask],
            log_1mpi[zero_mask] + log_nb0[zero_mask],
        )

        # Positive entries: log((1-pi) * NB(y))
        log_lik_pos = (
            log_1mpi[pos_mask]
            + self._log_nb_positive(Y_upper[pos_mask].float(), log_mu_upper[pos_mask])
        )

        return log_lik_zero.sum() + log_lik_pos.sum()

    def gradient_R(
        self,
        Y: torch.Tensor,       # (M, M) observed counts
        F_hat: torch.Tensor,   # (M, M) predicted log-signal
    ) -> torch.Tensor:
        """
        Approximate score R_sym (M, M): d log p / d F_ij for DPS guidance.

        Uses softplus(F) for mu, matching log_prob exactly.
        d softplus(F) / d F = sigmoid(F), so chain rule: dL/dF = dL/dmu * sigmoid(F).
        """
        M = Y.shape[0]
        device = Y.device
        theta = self.theta.detach()
        pi = self._get_pi(M, device).detach()

        triu_mask = torch.ones(M, M, device=device, dtype=torch.bool).triu(diagonal=1)
        F_upper = F_hat[triu_mask].detach()
        mu_u = torch.nn.functional.softplus(F_upper).clamp(min=1e-8)
        sig_u = torch.sigmoid(F_upper)   # d softplus(F) / d F

        # Reconstruct full symmetric mu and sig (off-diagonal only; diagonal=0)
        mu = torch.zeros(M, M, device=device)
        mu[triu_mask] = mu_u
        mu = mu + mu.t()

        sig = torch.zeros(M, M, device=device)
        sig[triu_mask] = sig_u
        sig = sig + sig.t()

        # P(NB=0): (theta/(mu+theta))^theta
        log_q = theta * (theta.log() - (mu + theta).clamp(min=1e-8).log())
        q = log_q.exp()

        R = torch.zeros(M, M, device=device)
        zero_mask = (Y == 0) & triu_mask
        pos_mask  = (Y  > 0) & triu_mask

        # Chain rule: dL/dF = dL/dmu * sigmoid(F)
        denom_z = (pi + (1 - pi) * q).clamp(min=1e-8)
        R[zero_mask] = (-(1 - pi) * theta * q * sig / ((mu + theta) * denom_z))[zero_mask]
        R[pos_mask]  = (theta * (Y - mu) * sig / (mu * (mu + theta)))[pos_mask]

        return 0.5 * (R + R.t())


# ---------------------------------------------------------------------------
# Geometric model: F = ZZ^T + B
# ---------------------------------------------------------------------------

class GeometricModel(nn.Module):
    """
    Combines BandBiasNetwork and ZINBLikelihood.

    F_n = Z_n Z_n^T + B(d)
    log p(Y_n | Z_n) via ZINB with softplus(F) mean parameterization.
    Gradient d log p / d Z_n = 2 * R @ Z  (analytical, no autograd needed).

    Cache workflow (speeds up E-step dramatically):
        geo_model.precompute(M, device)   # once before E-step loop
        ... run E-step for all cells ...
        geo_model.clear_cache()           # before M-step (params will change)
    """

    def __init__(
        self,
        band_bias: BandBiasNetwork,
        zinb: ZINBLikelihood,
    ) -> None:
        super().__init__()
        self.band_bias = band_bias
        self.zinb = zinb
        self._B_cache: torch.Tensor | None = None

    def precompute(self, M: int, device: torch.device) -> None:
        """Cache B and pi (constant within one EM sweep). Always call before E-step."""
        with torch.no_grad():
            self._B_cache = self.band_bias(M, device).detach()
            self.zinb._pi_cache = self.zinb.dropout_prob(M, device).detach()

    def clear_cache(self) -> None:
        """Invalidate cache after M-step updates band_bias or zinb parameters."""
        self._B_cache = None
        self.zinb._pi_cache = None  # type: ignore[assignment]

    def reconstruct(self, Z_n: torch.Tensor) -> torch.Tensor:
        """F = Z Z^T + B.  Z_n : (M, K) → (M, M)."""
        M = Z_n.shape[0]
        B = self._B_cache if self._B_cache is not None else self.band_bias(M, Z_n.device)
        return Z_n @ Z_n.t() + B

    def total_log_lik(self, Y: torch.Tensor, Z_n: torch.Tensor) -> torch.Tensor:
        """ZINB log-likelihood for one cell (scalar)."""
        F = self.reconstruct(Z_n)
        return self.zinb.log_prob(Y, F)

    def zinb_grad_Z(self, Y: torch.Tensor, Z_n: torch.Tensor) -> torch.Tensor:
        """
        Analytical gradient of log p(Y | Z) w.r.t. Z_n.

        With softplus(F) parameterization there is no cross-entry coupling,
        so the exact gradient is:  dL/dZ = 2 * R @ Z
        where R_ij = d log p / d F_ij  (computed in gradient_R).

        Returns: (M, K), detached.
        """
        with torch.no_grad():
            Z = Z_n.detach()
            F = self.reconstruct(Z)
            R = self.zinb.gradient_R(Y, F)   # (M, M) symmetric, zero diagonal
        return 2.0 * (R @ Z)


def create_geometric_model(config: dict) -> GeometricModel:
    bb_cfg   = config["model"].get("band_bias", {})
    zinb_cfg = config.get("zinb", {})
    band_bias = BandBiasNetwork(
        hidden_dim=bb_cfg.get("hidden_dim", 64),
        num_layers=bb_cfg.get("num_layers", 3),
    )
    zinb = ZINBLikelihood(
        init_theta=zinb_cfg.get("init_theta", 1.0),
        init_alpha0=zinb_cfg.get("init_alpha0", -2.0),
        init_alpha1=zinb_cfg.get("init_alpha1", 1.0),
    )
    return GeometricModel(band_bias, zinb)
