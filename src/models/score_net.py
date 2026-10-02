"""
Score networks for the scHiC latent diffusion model.

All variants share the interface:
    forward(Z_t: Tensor[B, M, K], t: Tensor[B]) -> Tensor[B, M, K]

where Z_t is the noised latent factor at diffusion timestep t,
and the output is the predicted noise epsilon_theta(Z_t, t).

Three architectures available (selectable via `create_score_net`):
  - transformer  : rows of Z treated as sequence tokens over M bins
  - mlp          : flatten Z, deep MLP with FiLM time conditioning
  - resnet1d     : 1-D ResNet treating M as sequence length
"""

from __future__ import annotations

import math
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Shared: sinusoidal time embedding
# ---------------------------------------------------------------------------

class SinusoidalTimeEmb(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """t: (B,) integer or float timesteps → (B, dim)."""
        device = t.device
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=device) / (half - 1)
        )
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)   # (B, half)
        return torch.cat([args.sin(), args.cos()], dim=-1)    # (B, dim)


class TimeEmbedding(nn.Module):
    """Sinusoidal → MLP projection."""

    def __init__(self, sin_dim: int, out_dim: int) -> None:
        super().__init__()
        self.sin_emb = SinusoidalTimeEmb(sin_dim)
        self.proj = nn.Sequential(
            nn.Linear(sin_dim, out_dim),
            nn.SiLU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.proj(self.sin_emb(t))   # (B, out_dim)


# ---------------------------------------------------------------------------
# 1.  Transformer score network
# ---------------------------------------------------------------------------

class TransformerScoreNet(nn.Module):
    """
    Treat each of the M rows of Z as a token (dim K).
    Self-attention over M bins captures inter-bin dependencies.
    """

    def __init__(
        self,
        M: int,
        K: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.M = M
        self.K = K
        self.d_model = d_model

        # Time embedding → d_model
        self.time_emb = TimeEmbedding(d_model, d_model)

        # Input projection K → d_model
        self.input_proj = nn.Linear(K, d_model)

        # Learned positional encoding over M bins
        self.pos_emb = nn.Embedding(M, d_model)

        # Transformer encoder
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,   # (B, seq, d_model)
            norm_first=True,    # pre-norm for stability
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        # Output projection d_model → K
        self.output_proj = nn.Linear(d_model, K)

    def forward(self, Z_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Z_t : (B, M, K)
        t   : (B,)
        Returns: (B, M, K) predicted noise
        """
        B, M, K = Z_t.shape
        device = Z_t.device

        # Time embedding → (B, d_model) → broadcast over M tokens
        t_emb = self.time_emb(t)                          # (B, d_model)
        t_emb = t_emb.unsqueeze(1).expand(-1, M, -1)      # (B, M, d_model)

        # Input embedding
        x = self.input_proj(Z_t)                          # (B, M, d_model)

        # Positional encoding
        pos = torch.arange(M, device=device)
        x = x + self.pos_emb(pos).unsqueeze(0)            # (B, M, d_model)

        # Add time conditioning
        x = x + t_emb

        # Transformer
        x = self.encoder(x)                               # (B, M, d_model)

        # Output projection
        return self.output_proj(x)                        # (B, M, K)


# ---------------------------------------------------------------------------
# 2.  MLP score network with FiLM conditioning
# ---------------------------------------------------------------------------

class FiLMLayer(nn.Module):
    """Feature-wise Linear Modulation: scale+shift from time embedding."""

    def __init__(self, hidden_dim: int, time_dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(time_dim, 2 * hidden_dim)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        """x: (B, hidden), t_emb: (B, time_dim)."""
        gamma, beta = self.proj(t_emb).chunk(2, dim=-1)
        return (1 + gamma) * x + beta


class MLPScoreNet(nn.Module):
    """
    Flatten Z (M×K) → deep MLP with FiLM time conditioning → reshape.
    """

    def __init__(
        self,
        M: int,
        K: int,
        hidden_dim: int = 512,
        num_layers: int = 6,
    ) -> None:
        super().__init__()
        self.M = M
        self.K = K
        flat_dim = M * K

        self.time_emb = TimeEmbedding(hidden_dim, hidden_dim)

        self.input_proj = nn.Linear(flat_dim, hidden_dim)

        self.layers = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)
        ])
        self.film_layers = nn.ModuleList([
            FiLMLayer(hidden_dim, hidden_dim) for _ in range(num_layers)
        ])
        self.norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])

        self.output_proj = nn.Linear(hidden_dim, flat_dim)

    def forward(self, Z_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        B, M, K = Z_t.shape
        x = Z_t.reshape(B, M * K)                  # (B, M*K)
        x = self.input_proj(x)                     # (B, hidden)
        t_emb = self.time_emb(t)                   # (B, hidden)

        for lin, film, norm in zip(self.layers, self.film_layers, self.norms):
            residual = x
            x = norm(x)
            x = F.silu(lin(x))
            x = film(x, t_emb)
            x = x + residual

        x = self.output_proj(x)                    # (B, M*K)
        return x.reshape(B, M, K)


# ---------------------------------------------------------------------------
# 3.  1-D ResNet score network
# ---------------------------------------------------------------------------

class ResBlock1D(nn.Module):
    """1-D residual block with GroupNorm and FiLM time conditioning."""

    def __init__(self, in_ch: int, out_ch: int, time_dim: int) -> None:
        super().__init__()
        groups = min(8, in_ch)
        self.norm1 = nn.GroupNorm(groups, in_ch)
        self.conv1 = nn.Conv1d(in_ch, out_ch, 3, padding=1)
        self.film  = nn.Linear(time_dim, 2 * out_ch)
        groups2 = min(8, out_ch)
        self.norm2 = nn.GroupNorm(groups2, out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, 3, padding=1)
        self.skip  = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = F.silu(self.norm1(x))
        h = self.conv1(h)
        # FiLM conditioning
        gamma, beta = self.film(t_emb).chunk(2, dim=-1)  # (B, out_ch) each
        h = h * (1 + gamma.unsqueeze(-1)) + beta.unsqueeze(-1)
        h = F.silu(self.norm2(h))
        h = self.conv2(h)
        return h + self.skip(x)


class SelfAttention1D(nn.Module):
    """Multi-head self-attention over the sequence (M) dimension."""

    def __init__(self, channels: int, nhead: int = 4) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(min(8, channels), channels)
        self.attn = nn.MultiheadAttention(channels, nhead, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, M)
        h = self.norm(x).permute(0, 2, 1)        # (B, M, C)
        h, _ = self.attn(h, h, h)
        return x + h.permute(0, 2, 1)            # (B, C, M)


class ResNet1DScoreNet(nn.Module):
    """
    Treat Z (M, K) as a 1-D signal of length M with K channels.
    Simple UNet-style encoder-decoder:
      - num_res_blocks encoder stages, each: ResBlock1D → downsample (stride-2 conv)
      - bottleneck: ResBlock1D → SelfAttention1D → ResBlock1D
      - num_res_blocks decoder stages, each: upsample (interp+conv) → ResBlock1D with skip

    Channel schedule: [base, 2*base, 4*base, ...], capped at 8*base.
    """

    def __init__(
        self,
        M: int,
        K: int,
        base_channels: int = 64,
        num_res_blocks: int = 3,
        population_cond_dim: int = 0,
    ) -> None:
        super().__init__()
        self.M = M
        self.K = K
        self.population_cond_dim = int(population_cond_dim)
        time_dim = base_channels * 4

        self.time_emb = TimeEmbedding(base_channels, time_dim)
        self.population_emb = (
            nn.Sequential(
                nn.Linear(self.population_cond_dim, time_dim),
                nn.SiLU(),
                nn.Linear(time_dim, time_dim),
            )
            if self.population_cond_dim > 0 else None
        )
        self.input_conv = nn.Conv1d(K, base_channels, 3, padding=1)

        # Build channel schedule
        ch_list: List[int] = [
            min(base_channels * (2 ** i), base_channels * 8)
            for i in range(num_res_blocks + 1)
        ]
        # ch_list[0] = base, ch_list[-1] = bottleneck channels

        # Encoder: ResBlock1D + stride-2 downsample (except after last stage)
        self.enc_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        in_ch = base_channels
        for i in range(num_res_blocks):
            out_ch = ch_list[i + 1]
            self.enc_blocks.append(ResBlock1D(in_ch, out_ch, time_dim))
            if i < num_res_blocks - 1:
                self.downsamples.append(nn.Conv1d(out_ch, out_ch, 3, stride=2, padding=1))
            else:
                self.downsamples.append(nn.Identity())
            in_ch = out_ch

        # Bottleneck
        bot_ch = ch_list[-1]
        self.mid1 = ResBlock1D(bot_ch, bot_ch, time_dim)
        nhead = max(1, bot_ch // 64)
        self.mid_attn = SelfAttention1D(bot_ch, nhead=nhead)
        self.mid2 = ResBlock1D(bot_ch, bot_ch, time_dim)

        # Decoder: upsample + skip concat → ResBlock1D
        self.upsamples = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()
        in_ch = bot_ch
        for i in reversed(range(num_res_blocks)):
            skip_ch = ch_list[i + 1]
            out_ch  = ch_list[i] if i > 0 else base_channels
            # Upsample: interpolate then conv (no rounding issues)
            self.upsamples.append(
                nn.Conv1d(in_ch, skip_ch, 3, padding=1)  # channel adjustment before concat
            )
            self.dec_blocks.append(ResBlock1D(skip_ch * 2, out_ch, time_dim))
            in_ch = out_ch

        groups = min(8, base_channels)
        self.output_norm = nn.GroupNorm(groups, base_channels)
        self.output_conv = nn.Conv1d(base_channels, K, 3, padding=1)

    def forward(
        self,
        Z_t: torch.Tensor,
        t: torch.Tensor,
        population_cond: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Z_t: (B, M, K) → (B, M, K)."""
        x = Z_t.permute(0, 2, 1).contiguous()   # (B, K, M)
        t_emb = self.time_emb(t)                 # (B, time_dim)
        if self.population_emb is not None:
            if population_cond is None:
                population_cond = torch.zeros(
                    Z_t.shape[0], self.population_cond_dim,
                    device=Z_t.device, dtype=Z_t.dtype,
                )
            if population_cond.shape != (Z_t.shape[0], self.population_cond_dim):
                raise ValueError(
                    "population_cond must have shape "
                    f"({Z_t.shape[0]}, {self.population_cond_dim}), got "
                    f"{tuple(population_cond.shape)}"
                )
            t_emb = t_emb + self.population_emb(
                population_cond.to(device=Z_t.device, dtype=Z_t.dtype)
            )

        x = self.input_conv(x)                   # (B, base, M)

        # Encoder — save pre-downsample features as skips
        skips: List[torch.Tensor] = []
        for enc, ds in zip(self.enc_blocks, self.downsamples):
            x = enc(x, t_emb)
            skips.append(x)       # skip before downsampling
            x = ds(x)

        # Bottleneck
        x = self.mid1(x, t_emb)
        x = self.mid_attn(x)
        x = self.mid2(x, t_emb)

        # Decoder — match skip lengths via interpolation
        for up_conv, dec, skip in zip(self.upsamples, self.dec_blocks, reversed(skips)):
            # Upsample spatially to match skip length
            x = F.interpolate(x, size=skip.shape[-1], mode="linear", align_corners=False)
            x = up_conv(x)                        # adjust channels to skip_ch
            x = torch.cat([x, skip], dim=1)       # (B, 2*skip_ch, L)
            x = dec(x, t_emb)

        x = self.output_conv(F.silu(self.output_norm(x)))  # (B, K, M)
        return x.permute(0, 2, 1)                          # (B, M, K)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_score_net(
    arch: str,
    M: int,
    K: int,
    config: dict,
) -> nn.Module:
    """
    Create a score network.

    arch : one of 'transformer', 'mlp', 'resnet1d'
    M    : number of genomic bins
    K    : latent rank
    config: full config dict (reads config['model'][arch] sub-dict)
    """
    model_cfg = config.get("model", {})

    if arch == "transformer":
        cfg = model_cfg.get("transformer", {})
        return TransformerScoreNet(
            M=M,
            K=K,
            d_model=cfg.get("d_model", 128),
            nhead=cfg.get("nhead", 4),
            num_layers=cfg.get("num_layers", 4),
            dim_feedforward=cfg.get("dim_feedforward", 512),
            dropout=cfg.get("dropout", 0.1),
        )
    elif arch == "mlp":
        cfg = model_cfg.get("mlp", {})
        return MLPScoreNet(
            M=M,
            K=K,
            hidden_dim=cfg.get("hidden_dim", 512),
            num_layers=cfg.get("num_layers", 6),
        )
    elif arch == "resnet1d":
        cfg = model_cfg.get("resnet1d", {})
        pop_cfg = config.get("training", {}).get("population_conditioning", {}) or {}
        seed_cfg = config.get("training", {}).get("clean_prior_init", {}) or {}
        population_cond_dim = (
            int(seed_cfg.get("max_clusters", 32))
            if pop_cfg.get("enabled", False) else 0
        )
        return ResNet1DScoreNet(
            M=M,
            K=K,
            base_channels=cfg.get("base_channels", 64),
            num_res_blocks=cfg.get("num_res_blocks", 3),
            population_cond_dim=population_cond_dim,
        )
    else:
        raise ValueError(f"Unknown score_arch '{arch}'. Choose from: transformer, mlp, resnet1d")
