#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn

from spk_enc_model import ConvNeXtBlock1d, AttentiveStatsPooling


class MelDualEncoder(nn.Module):
    """
    Shared backbone, two heads:
      - z_spk:   [B,d_spk]   L2-normalized
      - z_style: [B,d_style] L2-normalized, optionally projected off speaker direction
    """

    def __init__(
        self,
        *,
        n_mels: int = 100,
        d_spk: int = 256,
        d_style: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 4,
        attn_head_dim: int = 128,
        style_project_off_spk: bool = True,
    ):
        super().__init__()
        n_mels = int(n_mels)
        d_spk = int(d_spk)
        d_style = int(d_style)
        hidden_dim = int(hidden_dim)
        num_layers = int(max(1, num_layers))
        attn_head_dim = int(max(8, attn_head_dim))

        self.d_spk = int(d_spk)
        self.d_style = int(d_style)
        self.style_project_off_spk = bool(style_project_off_spk)

        if int(d_spk) == int(d_style):
            self.spk_to_style: nn.Module = nn.Identity()
        else:
            self.spk_to_style = nn.Linear(int(d_spk), int(d_style), bias=False)

        self.pre_conv = nn.Sequential(
            nn.Conv1d(n_mels, hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.GELU(),
        )
        self.blocks = nn.ModuleList([ConvNeXtBlock1d(hidden_dim) for _ in range(num_layers)])
        self.pooling = AttentiveStatsPooling(hidden_dim * num_layers, head_dim=attn_head_dim)

        pooled_dim = 2 * (hidden_dim * num_layers)
        hid_spk = int(max(256, d_spk * 2))
        hid_style = int(max(256, d_style * 2))
        self.mlp_spk = nn.Sequential(
            nn.LayerNorm(pooled_dim),
            nn.Linear(pooled_dim, hid_spk),
            nn.GELU(),
            nn.Linear(hid_spk, d_spk),
        )
        self.mlp_style = nn.Sequential(
            nn.LayerNorm(pooled_dim),
            nn.Linear(pooled_dim, hid_style),
            nn.GELU(),
            nn.Linear(hid_style, d_style),
        )

    def spk_dir_in_style_space(self, z_spk_bd: torch.Tensor) -> torch.Tensor:
        s = self.spk_to_style(z_spk_bd.to(dtype=torch.float32))
        return s / s.norm(dim=-1, keepdim=True).clamp_min(1e-12)

    def forward(
        self,
        mel_bct: torch.Tensor,
        *,
        mask_bt: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.pre_conv(mel_bct)
        outs = []
        for blk in self.blocks:
            x = blk(x)
            outs.append(x)
        x_agg = torch.cat(outs, dim=1)
        feat = self.pooling(x_agg, mask_bt=mask_bt)

        z_spk = self.mlp_spk(feat)
        z_spk = z_spk / z_spk.norm(dim=-1, keepdim=True).clamp_min(1e-12)

        t = self.mlp_style(feat)
        if bool(self.style_project_off_spk):
            s_hat = self.spk_dir_in_style_space(z_spk.detach())
            proj = (t * s_hat).sum(dim=-1, keepdim=True) * s_hat
            t = t - proj
        z_style = t / t.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        return z_spk, z_style
