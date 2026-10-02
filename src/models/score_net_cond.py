"""
Conditional score networks with classifier-free guidance (CFG) support.

Borrowed from ChromoGen, which uses CFG (jointly trained conditional +
unconditional model with condition dropout) instead of likelihood-gradient
guidance (DPS).  CFG is more robust and removes the A1-A3 / Jacobian
approximations of DPS.

The conditioning signal here is the cell's own RWR-imputed band observation
x̃ ∈ R^{M × D_cond}.  At training time it is dropped with probability
`cond_drop_prob` (replaced by a learned null embedding), so the same network
learns both p(Z) (unconditional) and p(Z | x̃) (conditional).  The
unconditional path is used as the EM prior; the conditional path drives
CFG imputation at inference.

Interface:
    forward(Z_t: (B, M, K), t: (B,), cond: (B, M, D_cond) | None) -> (B, M, K)
    cond = None  → unconditional (uses the learned null embedding).
"""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .score_net import ResBlock1D, SelfAttention1D, TimeEmbedding


class CondResNet1DScoreNet(nn.Module):
    """1-D UNet score net (M = sequence, K = channels) with an additive
    conditioning branch encoding x̃ and CFG-style condition dropout.

    Mirrors ResNet1DScoreNet's UNet topology; the only additions are
    `cond_encoder`, the learned `null_cond` embedding, and the dropout logic.
    """

    def __init__(
        self,
        M: int,
        K: int,
        D_cond: int,
        base_channels: int = 64,
        num_res_blocks: int = 3,
        cond_drop_prob: float = 0.15,
    ) -> None:
        super().__init__()
        self.M = M
        self.K = K
        self.D_cond = D_cond
        self.cond_drop_prob = cond_drop_prob
        time_dim = base_channels * 4

        self.time_emb = TimeEmbedding(base_channels, time_dim)
        self.input_conv = nn.Conv1d(K, base_channels, 3, padding=1)

        # Conditioning branch: x̃ (B, D_cond, M) → (B, base_channels, M),
        # added to the input feature map.  A learned null token represents the
        # "no condition" case (used both for dropout and unconditional eval).
        self.cond_encoder = nn.Sequential(
            nn.Conv1d(D_cond, base_channels, 3, padding=1),
            nn.SiLU(),
            nn.Conv1d(base_channels, base_channels, 3, padding=1),
        )
        self.null_cond = nn.Parameter(torch.zeros(1, D_cond, 1))

        ch_list: List[int] = [
            min(base_channels * (2 ** i), base_channels * 8)
            for i in range(num_res_blocks + 1)
        ]

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

        bot_ch = ch_list[-1]
        self.mid1 = ResBlock1D(bot_ch, bot_ch, time_dim)
        nhead = max(1, bot_ch // 64)
        self.mid_attn = SelfAttention1D(bot_ch, nhead=nhead)
        self.mid2 = ResBlock1D(bot_ch, bot_ch, time_dim)

        self.upsamples = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()
        in_ch = bot_ch
        for i in reversed(range(num_res_blocks)):
            skip_ch = ch_list[i + 1]
            out_ch = ch_list[i] if i > 0 else base_channels
            self.upsamples.append(nn.Conv1d(in_ch, skip_ch, 3, padding=1))
            self.dec_blocks.append(ResBlock1D(skip_ch * 2, out_ch, time_dim))
            in_ch = out_ch

        groups = min(8, base_channels)
        self.output_norm = nn.GroupNorm(groups, base_channels)
        self.output_conv = nn.Conv1d(base_channels, K, 3, padding=1)

    def _cond_feature(
        self,
        cond: Optional[torch.Tensor],
        B: int,
        M: int,
        device: torch.device,
        cond_drop_prob: Optional[float],
    ) -> torch.Tensor:
        """Build the (B, D_cond, M) conditioning tensor, applying the learned
        null embedding for missing / dropped conditions, then encode it."""
        null = self.null_cond.expand(B, self.D_cond, M)
        if cond is None:
            cond_t = null
        else:
            cond_t = cond.permute(0, 2, 1)                      # (B, D_cond, M)
            p = self.cond_drop_prob if cond_drop_prob is None else cond_drop_prob
            if self.training and p > 0:
                drop = (torch.rand(B, 1, 1, device=device) < p)
                cond_t = torch.where(drop, null, cond_t)
        return self.cond_encoder(cond_t)                        # (B, base, M)

    def forward(
        self,
        Z_t: torch.Tensor,
        t: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
        cond_drop_prob: Optional[float] = None,
    ) -> torch.Tensor:
        B, M, _ = Z_t.shape
        x = Z_t.permute(0, 2, 1).contiguous()                   # (B, K, M)
        t_emb = self.time_emb(t)

        x = self.input_conv(x)
        x = x + self._cond_feature(cond, B, M, Z_t.device, cond_drop_prob)

        skips: List[torch.Tensor] = []
        for enc, ds in zip(self.enc_blocks, self.downsamples):
            x = enc(x, t_emb)
            skips.append(x)
            x = ds(x)

        x = self.mid1(x, t_emb)
        x = self.mid_attn(x)
        x = self.mid2(x, t_emb)

        for up_conv, dec, skip in zip(self.upsamples, self.dec_blocks, reversed(skips)):
            x = F.interpolate(x, size=skip.shape[-1], mode="linear", align_corners=False)
            x = up_conv(x)
            x = torch.cat([x, skip], dim=1)
            x = dec(x, t_emb)

        x = self.output_conv(F.silu(self.output_norm(x)))       # (B, K, M)
        return x.permute(0, 2, 1)


def create_cond_score_net(
    M: int,
    K: int,
    D_cond: int,
    config: dict,
) -> CondResNet1DScoreNet:
    """Factory for the conditional score net (c2 uses resnet1d only — the
    architecture that scales with M)."""
    model_cfg = config.get("model", {})
    cfg = model_cfg.get("resnet1d", {})
    return CondResNet1DScoreNet(
        M=M,
        K=K,
        D_cond=D_cond,
        base_channels=cfg.get("base_channels", 64),
        num_res_blocks=cfg.get("num_res_blocks", 3),
        cond_drop_prob=float(model_cfg.get("cond_drop_prob", 0.15)),
    )
