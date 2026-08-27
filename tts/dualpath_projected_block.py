#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Projected/wide DualPath block for controlled bottleneck ablations.

This keeps the public forward API of ``DualPathTransformerBlock`` but replaces
the hard ``x.chunk(2)`` split with learned branch projections:

    x[D] -> attn_in[D->A] -> self-attn(A)
    x[D] -> conv_in[D->C] -> multi-kernel depthwise conv(C)
    concat[A+C] -> merge[A+C->D]

With ``attn_dim=conv_dim=D//2`` and split-identity initialization it starts close to
the original block, while allowing both branches to learn from the full hidden
state. With ``attn_dim=D, conv_dim=D`` it becomes the full-width 512/512 variant.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from backbone_tts import (
    AdaLN,
    DualAdaptiveLayerNorm,
    SelfAttention,
    SwiGLUFFN,
    TimeAdaptiveLayerNorm,
)


class DualPathProjectedBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        kernel_sizes=(3, 9, 15),
        dilations=(1, 2, 4),
        use_sdpa: bool = False,
        use_adaln: bool = False,
        use_dual_adaln: bool = False,
        use_time_adaln: bool = False,
        cond_dim: int = 0,
        conv_act: str = "gelu",
        branch_dim: Optional[int] = None,
        attn_dim: Optional[int] = None,
        conv_dim: Optional[int] = None,
        init_split_identity: bool = True,
    ):
        super().__init__()
        dim = int(dim)
        if branch_dim is not None:
            attn_dim = int(branch_dim) if attn_dim is None else int(attn_dim)
            conv_dim = int(branch_dim) if conv_dim is None else int(conv_dim)
        attn_dim = int(attn_dim or (dim // 2))
        conv_dim = int(conv_dim or (dim // 2))
        if attn_dim <= 0 or conv_dim <= 0:
            raise ValueError("attn_dim and conv_dim must be positive")
        if attn_dim % int(num_heads) != 0:
            raise ValueError(f"attn_dim={attn_dim} must be divisible by num_heads={num_heads}")

        self.dim = dim
        self.attn_dim = attn_dim
        self.conv_dim = conv_dim
        self.branch_dim = attn_dim  # legacy/introspection alias
        self.use_adaln = bool(use_adaln or use_time_adaln)
        self.use_dual_adaln = bool(use_dual_adaln)
        self.use_time_adaln = bool(use_time_adaln)
        if self.use_dual_adaln and self.use_time_adaln:
            raise ValueError("DualPathProjectedBlock: use_dual_adaln and use_time_adaln are mutually exclusive.")

        K = len(kernel_sizes)

        self.norm1 = nn.LayerNorm(dim, elementwise_affine=not self.use_adaln)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=not self.use_adaln)
        self.norm3 = nn.LayerNorm(dim, elementwise_affine=not self.use_adaln)
        if self.use_adaln:
            if self.use_dual_adaln:
                self.ada1 = DualAdaptiveLayerNorm(dim)
                self.ada2 = DualAdaptiveLayerNorm(dim)
                self.ada3 = DualAdaptiveLayerNorm(dim)
            elif self.use_time_adaln:
                self.ada1 = TimeAdaptiveLayerNorm(dim)
                self.ada2 = TimeAdaptiveLayerNorm(dim)
                self.ada3 = TimeAdaptiveLayerNorm(dim)
            else:
                cdim = int(cond_dim) if int(cond_dim) > 0 else dim
                self.ada1 = AdaLN(dim, cdim)
                self.ada2 = AdaLN(dim, cdim)
                self.ada3 = AdaLN(dim, cdim)

        self.ffn_pre = SwiGLUFFN(dim, expansion=4, dropout=0.1)
        self.ffn_post = SwiGLUFFN(dim, expansion=4, dropout=0.1)

        self.attn_in = nn.Linear(dim, attn_dim)
        self.conv_in = nn.Linear(dim, conv_dim)
        self.attn = SelfAttention(attn_dim, int(num_heads), use_sdpa=use_sdpa)

        self.pre_conv_norm = nn.LayerNorm(conv_dim)
        self.convs = nn.ModuleList(
            [
                nn.Conv1d(
                    conv_dim,
                    conv_dim,
                    kernel_size=int(k),
                    padding=(int(k) // 2) * int(d),
                    dilation=int(d),
                    groups=conv_dim,
                )
                for k, d in zip(kernel_sizes, dilations)
            ]
        )
        self.merge_conv = nn.Conv1d(conv_dim, conv_dim, kernel_size=1)
        self.post_conv_norm = nn.LayerNorm(conv_dim)

        self.gate1 = nn.Linear(conv_dim, 2 * K)
        self.gate2 = nn.Linear(attn_dim + conv_dim, 4)
        self.branch_merge = nn.Linear(attn_dim + conv_dim, dim)
        self.conv_act = str(conv_act).lower().strip()

        if bool(init_split_identity):
            self._init_split_identity()

    def _init_split_identity(self) -> None:
        with torch.no_grad():
            self.attn_in.weight.zero_()
            self.conv_in.weight.zero_()
            self.attn_in.bias.zero_()
            self.conv_in.bias.zero_()
            half = self.dim // 2
            n_attn = min(self.attn_dim, half)
            n_conv = min(self.conv_dim, self.dim - half)
            self.attn_in.weight[:n_attn, :n_attn] = torch.eye(n_attn, dtype=self.attn_in.weight.dtype)
            self.conv_in.weight[:n_conv, half:half + n_conv] = torch.eye(n_conv, dtype=self.conv_in.weight.dtype)

            self.branch_merge.weight.zero_()
            self.branch_merge.bias.zero_()
            self.branch_merge.weight[:n_attn, :n_attn] = torch.eye(n_attn, dtype=self.branch_merge.weight.dtype)
            self.branch_merge.weight[half:half + n_conv, self.attn_dim:self.attn_dim + n_conv] = torch.eye(
                n_conv,
                dtype=self.branch_merge.weight.dtype,
            )

    def _adaln(
        self,
        layer_name: str,
        x: torch.Tensor,
        cond: torch.Tensor | None,
        base_spk: torch.Tensor | None,
        style_vec: torch.Tensor | None,
        base_t: torch.Tensor | None,
    ) -> torch.Tensor:
        if not self.use_adaln:
            return x
        layer = getattr(self, layer_name)
        if self.use_dual_adaln:
            return layer(x, base_spk, style_vec, base_t)
        if self.use_time_adaln:
            return layer(x, base_t)
        return layer(x, cond)

    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor | None = None,
        t_emb: torch.Tensor | None = None,
        spk_emb: torch.Tensor | None = None,
        style_vec: torch.Tensor | None = None,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, _L, _D = x.shape
        pad_mask = None
        valid_f = None
        if key_padding_mask is not None:
            pad_mask = key_padding_mask.to(dtype=torch.bool, device=x.device)
            valid_f = (~pad_mask).to(dtype=x.dtype).unsqueeze(-1)
            x = x * valid_f

        base_spk = None
        base_t = None
        if self.use_dual_adaln:
            base_spk = spk_emb
            base_t = t_emb if t_emb is not None else cond
        elif self.use_time_adaln:
            base_t = t_emb if t_emb is not None else cond

        res = x
        h1 = self.norm1(x)
        h1 = self._adaln("ada1", h1, cond, base_spk, style_vec, base_t)
        x = x + 0.5 * self.ffn_pre(h1)
        if valid_f is not None:
            x = x * valid_f

        h2 = self.norm2(x)
        h2 = self._adaln("ada2", h2, cond, base_spk, style_vec, base_t)

        x_attn = self.attn_in(h2)
        x_conv = self.conv_in(h2)

        a = self.attn(x_attn, key_padding_mask=pad_mask)
        if valid_f is not None:
            a = a * valid_f

        x2n = self.pre_conv_norm(x_conv)
        x2n = torch.sin(x2n) if self.conv_act == "sin" else F.gelu(x2n)
        if valid_f is not None:
            x2n = x2n * valid_f
        x2_t = x2n.transpose(1, 2)
        conv_outs = [conv(x2_t) for conv in self.convs]
        c_stack = torch.stack(conv_outs, dim=-1)

        if valid_f is not None:
            denom = valid_f.sum(dim=1).clamp_min(1.0)
            gate_pool = (x2n * valid_f).sum(dim=1) / denom
        else:
            gate_pool = x2n.mean(dim=1)
        gate_out = self.gate1(gate_pool)
        K = len(self.convs)
        ga, gb = gate_out.chunk(2, dim=-1)
        w1 = (ga * torch.sigmoid(gb)).softmax(dim=-1).view(B, 1, 1, K)
        c_weighted = (c_stack * w1).sum(dim=-1)
        c = self.merge_conv(c_weighted).transpose(1, 2)
        c = self.post_conv_norm(c)
        if valid_f is not None:
            c = c * valid_f

        combined = torch.cat([a, c], dim=-1)
        g2a, g2b = self.gate2(combined).chunk(2, dim=-1)
        w2 = (g2a * torch.sigmoid(g2b)).softmax(dim=-1)
        merged = torch.cat([w2[..., 0:1] * a, w2[..., 1:2] * c], dim=-1)
        merged = self.branch_merge(merged)

        x = res + merged
        if valid_f is not None:
            x = x * valid_f

        h3 = self.norm3(x)
        h3 = self._adaln("ada3", h3, cond, base_spk, style_vec, base_t)
        x = x + 0.5 * self.ffn_post(h3)
        if valid_f is not None:
            x = x * valid_f
        return x
