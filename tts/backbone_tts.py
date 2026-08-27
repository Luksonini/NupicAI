#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
backbone_tts.py (SalmonTTS2 production)

Minimalny backbone używany przez SalmonTTS2:
- tekstowy encoder: embedding + pozycje + DualPathTransformerBlock stack
- FlowDurationPredictor: rectified-flow na log-durations

Cel: produkcyjny, mały plik (bez starych DurationPredictorV2/V3, dekoderów czasu, MoE, itp.).
Pełne/legacy warianty zostają w `legacy/backbone_tts.py`.
"""

from __future__ import annotations

import math
import random
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from benchmark_config import BenchmarkConfig


class SelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 4, use_sdpa: bool = False):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = int(dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.dim // self.num_heads
        self.qkv = nn.Linear(self.dim, self.dim * 3, bias=True)
        self.out = nn.Linear(self.dim, self.dim, bias=True)
        self.use_sdpa = bool(use_sdpa)

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        B, L, D = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        attn_mask = None
        if key_padding_mask is not None:
            attn_mask = torch.where(
                key_padding_mask[:, None, None, :].to(dtype=torch.bool, device=x.device),
                torch.tensor(float("-inf"), device=x.device, dtype=q.dtype),
                torch.zeros(1, device=x.device, dtype=q.dtype),
            )

        if self.use_sdpa and hasattr(F, "scaled_dot_product_attention") and x.is_cuda:
            # SDPA wybierze najlepszy backend (Flash / mem-efficient) dla 3090 (Ampere),
            # ale jest opcjonalne i działa też bez CUDA.
            try:
                # New API (torch>=2.1): avoids torch.backends.cuda.sdp_kernel deprecation warnings.
                from torch.nn.attention import SDPBackend, sdpa_kernel  # type: ignore
                with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]):
                    out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=0.0, is_causal=False)
            except Exception:
                with torch.backends.cuda.sdp_kernel(enable_math=False, enable_flash=True, enable_mem_efficient=True):
                    out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=0.0, is_causal=False)
        else:
            attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            if attn_mask is not None:
                attn = attn + attn_mask
            attn = attn.softmax(dim=-1)
            out = torch.matmul(attn, v)

        out = out.transpose(1, 2).contiguous().view(B, L, D)
        return self.out(out)


class AdaLN(nn.Module):
    def __init__(self, dim: int, cond_dim: int):
        super().__init__()
        self.proj = nn.Linear(int(cond_dim), int(dim) * 2)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor | None) -> torch.Tensor:
        if cond is None:
            return x
        h = self.proj(cond.to(x.dtype)).unsqueeze(1)
        gamma, beta = h.chunk(2, dim=-1)
        return x * (1.0 + gamma) + beta


class TimeAdaptiveLayerNorm(nn.Module):
    """Time-only AdaLN used by speakerless decoder variants."""

    def __init__(self, dim: int, init_t_gain: float = 1.00):
        super().__init__()
        dim = int(dim)
        self.ln = nn.LayerNorm(dim)
        self.t_scale = nn.Linear(dim, dim)
        self.t_shift = nn.Linear(dim, dim)
        self.g_t = nn.Parameter(torch.tensor(float(init_t_gain)))

    def forward(self, x_btd: torch.Tensor, t_vec_bd: torch.Tensor | None) -> torch.Tensor:
        x = self.ln(x_btd)
        if t_vec_bd is None:
            return x
        t_scale = self.t_scale(t_vec_bd)[:, None, :]
        t_shift = self.t_shift(t_vec_bd)[:, None, :]
        scale = 1.0 + self.g_t * t_scale
        shift = self.g_t * t_shift
        return x * scale + shift


class DualAdaptiveLayerNorm(nn.Module):
    """
    3-way conditioning (speaker/style/time) fused additively:
      x_norm * (1 + spk + style + t) + (shift_spk + shift_style + shift_t)

    This keeps the "time/flow-step" conditioning separate from speaker/style.
    """

    def __init__(
        self,
        dim: int,
        init_spk_gain: float = 0.30,
        init_style_gain: float = 1.00,
        init_t_gain: float = 1.00,
    ):
        super().__init__()
        dim = int(dim)
        self.ln = nn.LayerNorm(dim)
        self.spk_scale = nn.Linear(dim, dim)
        self.spk_shift = nn.Linear(dim, dim)
        self.style_scale = nn.Linear(dim, dim)
        self.style_shift = nn.Linear(dim, dim)
        self.t_scale = nn.Linear(dim, dim)
        self.t_shift = nn.Linear(dim, dim)
        self.g_spk = nn.Parameter(torch.tensor(float(init_spk_gain)))
        self.g_style = nn.Parameter(torch.tensor(float(init_style_gain)))
        self.g_t = nn.Parameter(torch.tensor(float(init_t_gain)))

    def forward(
        self,
        x_btd: torch.Tensor,
        spk_vec_bd: torch.Tensor | None,
        style_vec_bd: torch.Tensor | None,
        t_vec_bd: torch.Tensor | None,
    ) -> torch.Tensor:
        x = self.ln(x_btd)

        scale = x.new_ones((x.size(0), 1, x.size(-1)))
        shift = x.new_zeros((x.size(0), 1, x.size(-1)))

        if spk_vec_bd is not None:
            spk_scale = self.spk_scale(spk_vec_bd)[:, None, :]
            spk_shift = self.spk_shift(spk_vec_bd)[:, None, :]
            scale = scale + self.g_spk * spk_scale
            shift = shift + self.g_spk * spk_shift

        if style_vec_bd is not None:
            sty_scale = self.style_scale(style_vec_bd)[:, None, :]
            sty_shift = self.style_shift(style_vec_bd)[:, None, :]
            scale = scale + self.g_style * sty_scale
            shift = shift + self.g_style * sty_shift

        if t_vec_bd is not None:
            t_scale = self.t_scale(t_vec_bd)[:, None, :]
            t_shift = self.t_shift(t_vec_bd)[:, None, :]
            scale = scale + self.g_t * t_scale
            shift = shift + self.g_t * t_shift

        return x * scale + shift


class SwiGLUFFN(nn.Module):
    def __init__(self, dim: int, expansion: int = 4, dropout: float = 0.1):
        super().__init__()
        dim = int(dim)
        inner = dim * int(expansion)
        self.fc1 = nn.Linear(dim, inner)
        self.fc2 = nn.Linear(dim, inner)
        self.fc_out = nn.Linear(inner, dim)
        self.dropout = nn.Dropout(float(dropout))
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h1 = self.act(self.fc1(x))
        h2 = self.fc2(x)
        h = h1 * h2
        h = self.dropout(h)
        return self.fc_out(h)


class DualPathTransformerBlock(nn.Module):
    """
    Blok encodera tekstu:
    - pre-FFN
    - split kanałów: half-attn + half-conv (multi-kernel depthwise)
    - gating attn vs conv
    - post-FFN
    """

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
        branch_mode: str = "split",
        attn_dim: Optional[int] = None,
        conv_dim: Optional[int] = None,
        init_split_identity: bool = True,
    ):
        super().__init__()
        dim = int(dim)
        branch_mode = str(branch_mode or "split").lower().strip()
        if branch_mode not in {"split", "projected"}:
            raise ValueError(f"DualPathTransformerBlock: unknown branch_mode={branch_mode!r}")
        if branch_mode == "split":
            assert dim % 2 == 0, "dim powinno być parzyste (dzielone na dwie gałęzie)."
        self.dim = dim
        self.branch_mode = branch_mode
        self.use_adaln = bool(use_adaln or use_time_adaln)
        self.use_dual_adaln = bool(use_dual_adaln)
        self.use_time_adaln = bool(use_time_adaln)
        if self.use_dual_adaln and self.use_time_adaln:
            raise ValueError("DualPathTransformerBlock: use_dual_adaln and use_time_adaln are mutually exclusive.")
        half = dim // 2
        if self.branch_mode == "projected":
            attn_dim = int(attn_dim or half)
            conv_dim = int(conv_dim or half)
            if attn_dim <= 0 or conv_dim <= 0:
                raise ValueError("DualPathTransformerBlock: attn_dim and conv_dim must be positive")
            if attn_dim % int(num_heads) != 0:
                raise ValueError(
                    f"DualPathTransformerBlock: attn_dim={attn_dim} must divide num_heads={int(num_heads)}"
                )
            self.attn_dim = int(attn_dim)
            self.conv_dim = int(conv_dim)
        else:
            self.attn_dim = half
            self.conv_dim = half
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

        if self.branch_mode == "projected":
            self.attn_in = nn.Linear(dim, self.attn_dim)
            self.conv_in = nn.Linear(dim, self.conv_dim)
            self.branch_merge = nn.Linear(self.attn_dim + self.conv_dim, dim)

        self.attn = SelfAttention(self.attn_dim, int(num_heads), use_sdpa=use_sdpa)

        self.pre_conv_norm = nn.LayerNorm(self.conv_dim)
        self.convs = nn.ModuleList(
            [
                nn.Conv1d(
                    self.conv_dim,
                    self.conv_dim,
                    kernel_size=int(k),
                    padding=(int(k) // 2) * int(d),
                    dilation=int(d),
                    groups=self.conv_dim,
                )
                for k, d in zip(kernel_sizes, dilations)
            ]
        )
        self.merge_conv = nn.Conv1d(self.conv_dim, self.conv_dim, kernel_size=1)
        self.post_conv_norm = nn.LayerNorm(self.conv_dim)

        self.gate1 = nn.Linear(self.conv_dim, 2 * K)
        self.gate2 = nn.Linear(self.attn_dim + self.conv_dim, 4)
        self.conv_act = str(conv_act).lower().strip()
        if self.branch_mode == "projected" and bool(init_split_identity):
            self._init_split_identity()

    def _init_split_identity(self) -> None:
        if self.branch_mode != "projected":
            return
        with torch.no_grad():
            self.attn_in.weight.zero_()
            self.attn_in.bias.zero_()
            self.conv_in.weight.zero_()
            self.conv_in.bias.zero_()
            self.branch_merge.weight.zero_()
            self.branch_merge.bias.zero_()

            half = self.dim // 2
            n_attn = min(self.attn_dim, half)
            n_conv = min(self.conv_dim, self.dim - half)
            self.attn_in.weight[:n_attn, :n_attn] = torch.eye(n_attn, dtype=self.attn_in.weight.dtype)
            self.conv_in.weight[:n_conv, half:half + n_conv] = torch.eye(n_conv, dtype=self.conv_in.weight.dtype)
            self.branch_merge.weight[:n_attn, :n_attn] = torch.eye(n_attn, dtype=self.branch_merge.weight.dtype)
            self.branch_merge.weight[half:half + n_conv, self.attn_dim:self.attn_dim + n_conv] = torch.eye(
                n_conv,
                dtype=self.branch_merge.weight.dtype,
            )

    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor | None = None,
        t_emb: torch.Tensor | None = None,
        spk_emb: torch.Tensor | None = None,
        style_vec: torch.Tensor | None = None,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, L, D = x.shape
        pad_mask = None
        valid_f = None
        if key_padding_mask is not None:
            pad_mask = key_padding_mask.to(dtype=torch.bool, device=x.device)
            valid_f = (~pad_mask).to(dtype=x.dtype).unsqueeze(-1)
            x = x * valid_f
        res = x
        base_spk = None
        base_t = None
        if self.use_dual_adaln:
            base_spk = spk_emb
            base_t = t_emb
            # Back-compat: many callers pass only `cond` (historically: time embedding).
            if base_t is None and cond is not None:
                base_t = cond
        elif self.use_time_adaln:
            base_t = t_emb
            if base_t is None and cond is not None:
                base_t = cond

        h1 = self.norm1(x)
        if self.use_adaln:
            if self.use_dual_adaln:
                h1 = self.ada1(h1, base_spk, style_vec, base_t)
            elif self.use_time_adaln:
                h1 = self.ada1(h1, base_t)
            else:
                h1 = self.ada1(h1, cond)
        x = x + 0.5 * self.ffn_pre(h1)
        if valid_f is not None:
            x = x * valid_f

        x = self.norm2(x)
        if self.use_adaln:
            if self.use_dual_adaln:
                x = self.ada2(x, base_spk, style_vec, base_t)
            elif self.use_time_adaln:
                x = self.ada2(x, base_t)
            else:
                x = self.ada2(x, cond)
        if self.branch_mode == "projected":
            x1 = self.attn_in(x)
            x2 = self.conv_in(x)
        else:
            x1, x2 = x.chunk(2, dim=-1)

        a = self.attn(x1, key_padding_mask=pad_mask)
        if valid_f is not None:
            a = a * valid_f

        x2n = self.pre_conv_norm(x2)
        if self.conv_act == "sin":
            x2n = torch.sin(x2n)
        else:
            x2n = F.gelu(x2n)
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
        c_proj = self.merge_conv(c_weighted)
        c = c_proj.transpose(1, 2)
        c = self.post_conv_norm(c)
        if valid_f is not None:
            c = c * valid_f

        combined = torch.cat([a, c], dim=-1)
        g2a, g2b = self.gate2(combined).chunk(2, dim=-1)
        w2 = (g2a * torch.sigmoid(g2b)).softmax(dim=-1)
        a_w = w2[..., 0:1] * a
        c_w = w2[..., 1:2] * c
        merged = torch.cat([a_w, c_w], dim=-1)
        if self.branch_mode == "projected":
            merged = self.branch_merge(merged)

        x = res + merged
        if valid_f is not None:
            x = x * valid_f
        h3 = self.norm3(x)
        if self.use_adaln:
            if self.use_dual_adaln:
                h3 = self.ada3(h3, base_spk, style_vec, base_t)
            elif self.use_time_adaln:
                h3 = self.ada3(h3, base_t)
            else:
                h3 = self.ada3(h3, cond)
        x = x + 0.5 * self.ffn_post(h3)
        if valid_f is not None:
            x = x * valid_f
        return x


class BackboneTTS(nn.Module):
    """
    Minimalny tekstowy backbone dla SalmonTTS2:
    - embed(ids) -> [B,L,D]
    - pos -> uczone pozycje
    - enc -> ModuleList[DualPathTransformerBlock]

    Uwaga: SalmonTTS2 ma osobne moduły duration/prior/flow, więc tutaj nie trzymamy starych predictorów.
    Atrybut `dur` zostaje jako placeholder (nadpisywany w SalmonTTS2).
    """

    def __init__(
        self,
        cfg: BenchmarkConfig,
        n_layers: int = 6,
        n_heads: int = 6,
        use_sdpa: bool = False,
        dualpath_branch_mode: str = "split",
        dualpath_attn_dim: Optional[int] = None,
        dualpath_conv_dim: Optional[int] = None,
        dualpath_init_split_identity: bool = True,
    ):
        super().__init__()
        D = int(cfg.hidden_dim)
        self.embed = nn.Embedding(int(cfg.vocab_size), D)
        self.pos = nn.Parameter(torch.randn(1, int(cfg.max_pos), D) * 0.01)
        self.enc = nn.ModuleList(
            [
                DualPathTransformerBlock(
                    D,
                    int(n_heads),
                    use_sdpa=bool(use_sdpa),
                    branch_mode=str(dualpath_branch_mode or "split"),
                    attn_dim=dualpath_attn_dim,
                    conv_dim=dualpath_conv_dim,
                    init_split_identity=bool(dualpath_init_split_identity),
                )
                for _ in range(int(n_layers))
            ]
        )
        self.dur = nn.Identity()

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        x = self.embed(ids)
        x = x + self.pos[:, : x.size(1)]
        for blk in self.enc:
            x = blk(x)
        return x


class DurationPriorHead(nn.Module):
    """
    Token-wise prior dla log-duration:
      x_tok -> (mu, logs), gdzie x0 = mu + sigma*eps służy jako start dla duration flow.
    """
    def __init__(self, dim: int, p_drop: float = 0.1):
        super().__init__()
        dim = int(dim)
        self.in_norm = nn.LayerNorm(dim)
        self.drop = nn.Dropout(float(p_drop))
        self.b1 = nn.Sequential(nn.Conv1d(dim, dim, 3, padding=1, groups=dim), nn.SiLU(), nn.Conv1d(dim, dim, 1))
        self.b2 = nn.Sequential(nn.Conv1d(dim, dim, 3, padding=2, dilation=2, groups=dim), nn.SiLU(), nn.Conv1d(dim, dim, 1))
        self.b3 = nn.Sequential(nn.Conv1d(dim, dim, 3, padding=4, dilation=4, groups=dim), nn.SiLU(), nn.Conv1d(dim, dim, 1))
        self.out_ln = nn.LayerNorm(dim)
        self.head_mu = nn.Conv1d(dim, 1, 1)
        self.head_logs = nn.Conv1d(dim, 1, 1)
        with torch.no_grad():
            self.head_logs.bias.fill_(-2.0)

    def forward(self, x_bld: torch.Tensor, x_mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        # x_bld: [B,L,D], x_mask: [B,1,L]
        x = x_bld
        if x_mask is not None:
            x = x * x_mask.transpose(1, 2)
        h = self.drop(self.in_norm(x))
        h = h.transpose(1, 2).contiguous()  # [B,D,L]
        if x_mask is not None:
            h = h * x_mask
        h = h + self.b1(h)
        h = h + self.b2(h)
        h = h + self.b3(h)
        h = h.transpose(1, 2).contiguous()
        h = self.out_ln(h).transpose(1, 2).contiguous()  # [B,D,L]
        if x_mask is not None:
            h = h * x_mask
        mu = self.head_mu(h).squeeze(1)    # [B,L]
        logs = self.head_logs(h).squeeze(1)  # [B,L]
        return mu, logs


class FlowDurationPredictorWithPrior(nn.Module):
    """
    Flow-based duration predictor (rectified flow on log-durations).

    API zgodne z użyciem w SalmonTTS2.py:
      - flow_loss(x_tok, x_mask, logdur_target, spk_emb=..., cond=...)
      - sample(x_tok, x_mask, spk_emb=..., cond=..., steps=..., noise_scale=..., use_heun=...)
      - flow_loss_with_prior_x0(...): flow loss z x0 z priora
      - sample_with_prior_x0(...): sampling x0 z priora
    """

    def __init__(
        self,
        dim: int = 256,
        spk_dim: int | None = None,
        hidden: int | None = None,
        cond_dim: int = 0,
        p_drop: float = 0.1,
        style_dim: int = 0,
        *,
        use_prior: bool = False,
        prior_p_drop: float = 0.1,
    ):
        super().__init__()
        dim = int(dim)
        spk_dim = dim if spk_dim is None else int(spk_dim)
        hidden = dim if hidden is None else int(hidden)
        self.cond_dim = int(cond_dim)
        self.style_dim = int(style_dim)
        self.in_norm = nn.LayerNorm(dim)
        self.drop = nn.Dropout(float(p_drop))
        self.in_proj = nn.Linear(dim + 2 + self.cond_dim, hidden)
        self.b1 = nn.Sequential(nn.Conv1d(hidden, hidden, 3, padding=1, groups=hidden), nn.SiLU(), nn.Conv1d(hidden, hidden, 1))
        self.b2 = nn.Sequential(nn.Conv1d(hidden, hidden, 3, padding=2, dilation=2, groups=hidden), nn.SiLU(), nn.Conv1d(hidden, hidden, 1))
        self.b3 = nn.Sequential(nn.Conv1d(hidden, hidden, 3, padding=4, dilation=4, groups=hidden), nn.SiLU(), nn.Conv1d(hidden, hidden, 1))
        self.out_ln = nn.LayerNorm(hidden)
        self.head = nn.Conv1d(hidden, 1, 1)
        self.spk_proj = nn.Identity() if spk_dim == dim else nn.Linear(spk_dim, dim)
        self.prior = DurationPriorHead(dim=dim, p_drop=float(prior_p_drop)) if bool(use_prior) else None
        if self.style_dim > 0:
            self.style_proj = nn.Linear(self.style_dim, hidden)
            self.style_ada_in = AdaLN(hidden, hidden)
            self.style_ada1 = AdaLN(hidden, hidden)
            self.style_ada2 = AdaLN(hidden, hidden)
            self.style_ada3 = AdaLN(hidden, hidden)
        else:
            self.style_proj = None
            self.style_ada_in = None
            self.style_ada1 = None
            self.style_ada2 = None
            self.style_ada3 = None

    def _forward_v(
        self,
        x: torch.Tensor,
        xt: torch.Tensor,
        t: torch.Tensor,
        spk_emb: torch.Tensor | None = None,
        style_vec: torch.Tensor | None = None,
        x_mask: torch.Tensor | None = None,
        cond: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if x_mask is not None:
            # [B,1,L] -> [B,L]
            m = x_mask.squeeze(1)
        else:
            m = None

        h = self.in_norm(x)
        if spk_emb is None:
            spk = torch.zeros((x.size(0), x.size(-1)), device=x.device, dtype=x.dtype)
        else:
            spk = self.spk_proj(spk_emb).to(x.dtype)

        xt_in = xt.unsqueeze(-1)
        # t is typically [B,1]; expand needs a 3D tensor.
        t_in = t.reshape(x.size(0), 1, 1).expand(x.size(0), x.size(1), 1)
        spk_in = spk.unsqueeze(1).expand(x.size(0), x.size(1), spk.size(-1))

        feats = [h, xt_in, t_in]
        if self.cond_dim > 0 and cond is not None:
            feats.append(cond.unsqueeze(1).expand(x.size(0), x.size(1), cond.size(-1)))
        inp = torch.cat(feats, dim=-1)

        inp = self.drop(inp)
        y = self.in_proj(inp) + spk_in  # speaker conditioning add-to-token
        style_cond = None
        if (self.style_proj is not None) and (style_vec is not None):
            style_cond = self.style_proj(style_vec).to(y.dtype)
            y = self.style_ada_in(y, style_cond)
        y = y.transpose(1, 2)
        y = y + self.b1(y)
        y = y.transpose(1, 2)
        if self.style_ada1 is not None:
            y = self.style_ada1(y, style_cond)
        y = y.transpose(1, 2)
        y = y + self.b2(y)
        y = y.transpose(1, 2)
        if self.style_ada2 is not None:
            y = self.style_ada2(y, style_cond)
        y = y.transpose(1, 2)
        y = y + self.b3(y)
        y = y.transpose(1, 2)
        if self.style_ada3 is not None:
            y = self.style_ada3(y, style_cond)
        y = self.out_ln(y).transpose(1, 2)
        if m is not None:
            y = y * m.unsqueeze(1)
        return self.head(y).squeeze(1)

    def flow_loss(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        logdur_target: torch.Tensor,
        spk_emb: torch.Tensor | None = None,
        style_vec: torch.Tensor | None = None,
        cond: torch.Tensor | None = None,
        *,
        v_loss: str = "mse",
    ) -> torch.Tensor:
        z = torch.randn_like(logdur_target)
        t = torch.rand((x.size(0), 1), device=x.device, dtype=logdur_target.dtype)
        xt = (1.0 - t) * z + t * logdur_target
        v = self._forward_v(x, xt, t, spk_emb=spk_emb, style_vec=style_vec, x_mask=x_mask, cond=cond)
        target_v = logdur_target - z
        mask = x_mask.squeeze(1)
        denom = mask.sum().clamp_min(1.0)
        kind = str(v_loss).lower().strip()
        if kind == "mse":
            return ((v - target_v) ** 2 * mask).sum() / denom
        if kind == "l1":
            return (torch.abs(v - target_v) * mask).sum() / denom
        if kind == "huber":
            return F.smooth_l1_loss(v * mask, target_v * mask, reduction="sum", beta=1.0) / denom
        raise ValueError(f"Unknown v_loss: {v_loss!r} (expected: mse|l1|huber)")

    @torch.no_grad()
    def sample(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        spk_emb: torch.Tensor | None = None,
        style_vec: torch.Tensor | None = None,
        cond: torch.Tensor | None = None,
        steps: int = 8,
        noise_scale: float = 0.667,
        use_heun: bool = True,
        *,
        clip_abs_min: float | None = None,
        clip_abs_max: float | None = None,
    ) -> torch.Tensor:
        x_cur = torch.randn(x.size(0), x.size(1), device=x.device, dtype=x.dtype) * float(noise_scale)
        dt = 1.0 / float(max(1, int(steps)))
        for i in range(int(steps)):
            t_val = float(i) * dt
            t = torch.full((x.size(0), 1), t_val, device=x.device, dtype=x.dtype)
            v_start = self._forward_v(x, x_cur, t, spk_emb=spk_emb, style_vec=style_vec, x_mask=x_mask, cond=cond)
            if use_heun and i < int(steps) - 1:
                x_next = x_cur + v_start * dt
                t_next = torch.full((x.size(0), 1), t_val + dt, device=x.device, dtype=x.dtype)
                v_end = self._forward_v(x, x_next, t_next, spk_emb=spk_emb, style_vec=style_vec, x_mask=x_mask, cond=cond)
                x_cur = x_cur + 0.5 * (v_start + v_end) * dt
            else:
                x_cur = x_cur + v_start * dt
            if (clip_abs_min is not None) or (clip_abs_max is not None):
                x_cur = x_cur.clamp(
                    min=float(clip_abs_min) if clip_abs_min is not None else None,
                    max=float(clip_abs_max) if clip_abs_max is not None else None,
                )
        return x_cur

    @staticmethod
    def _gaussian_nll_masked(mu: torch.Tensor, logs: torch.Tensor, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        m = mask.float()
        denom = m.sum().clamp_min(1.0)
        logs = logs.clamp(min=-7.0, max=2.0)
        sigma2 = torch.exp(2.0 * logs).clamp_min(1e-8)
        nll = ((x - mu) ** 2) / (2.0 * sigma2) + logs
        return (nll * m).sum() / denom

    @staticmethod
    def _kl_diag_gauss_to_std_normal_masked(mu: torch.Tensor, logs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        m = mask.float()
        denom = m.sum().clamp_min(1.0)
        kl = 0.5 * (torch.exp(2.0 * logs) + mu * mu - 1.0 - 2.0 * logs)
        return (kl * m).sum() / denom

    def flow_loss_with_prior_x0(
        self,
        *,
        x_tok_bld: torch.Tensor,
        x_mask_b1l: torch.Tensor,
        target_log_bl: torch.Tensor,
        spk_emb_bd: torch.Tensor,
        style_vec_bd: torch.Tensor | None = None,
        cond: torch.Tensor | None,
        prior_loss_mode: str = "nll",
        prior_w: float = 0.10,
        prior_kl_w: float = 0.0,
        prior_logs_min: float = -5.0,
        prior_logs_max: float = 2.0,
        prior_sigma_min: float = 0.1,
        x0_noise_scale: float = 1.0,
        sigma0_train_min: float = 0.0,
        sigma0_train_max: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.prior is None:
            raise RuntimeError("FlowDurationPredictorWithPrior: prior head not initialized (use_prior=False).")

        mask = x_mask_b1l.squeeze(1).float()
        mu, logs = self.prior(x_tok_bld, x_mask=x_mask_b1l)
        logs = logs.clamp(min=float(prior_logs_min), max=float(prior_logs_max))

        if str(prior_loss_mode).lower().strip() == "nll":
            prior_aux = self._gaussian_nll_masked(mu, logs, target_log_bl, mask)
        else:
            denom = mask.sum().clamp_min(1.0)
            prior_aux = (((mu - target_log_bl) ** 2) * mask).sum() / denom

        if float(prior_kl_w) > 0.0:
            prior_aux = prior_aux + float(prior_kl_w) * self._kl_diag_gauss_to_std_normal_masked(mu, logs, mask)

        sigma = torch.exp(logs).clamp_min(1e-8)
        if float(prior_sigma_min) > 0.0:
            sigma = sigma.clamp_min(float(prior_sigma_min))

        sigma0 = 0.0
        if float(sigma0_train_max) > float(sigma0_train_min) and float(sigma0_train_max) > 0.0:
            sigma0 = float(sigma0_train_min + (sigma0_train_max - sigma0_train_min) * random.random())
        elif float(sigma0_train_min) > 0.0:
            sigma0 = float(sigma0_train_min)
        if sigma0 > 0.0:
            sigma_eff = torch.sqrt(sigma * sigma + (sigma0 * sigma0))
        else:
            sigma_eff = sigma

        x0 = mu + torch.randn_like(mu) * sigma_eff * float(x0_noise_scale)
        x1 = target_log_bl
        t = torch.rand((x_tok_bld.size(0), 1), device=x_tok_bld.device, dtype=target_log_bl.dtype)
        xt = (1.0 - t) * x0 + t * x1
        target_v = x1 - x0
        v = self._forward_v(
            x_tok_bld,
            xt,
            t,
            spk_emb=spk_emb_bd,
            style_vec=style_vec_bd,
            x_mask=x_mask_b1l,
            cond=cond,
        )

        denom = mask.sum().clamp_min(1.0)
        flow_loss = (((v - target_v) ** 2) * mask).sum() / denom
        total_loss = flow_loss + float(prior_w) * prior_aux
        return total_loss, flow_loss, prior_aux

    @torch.no_grad()
    def sample_with_prior_x0(
        self,
        *,
        x_tok_bld: torch.Tensor,
        x_mask_b1l: torch.Tensor,
        spk_emb_bd: torch.Tensor,
        style_vec_bd: torch.Tensor | None = None,
        cond: torch.Tensor | None,
        steps: int,
        noise_scale: float,
        use_heun: bool,
        prior_logs_min: float,
        prior_logs_max: float,
        prior_sigma_min: float,
        sigma0_demo: float,
        clip_sigma: float = 0.0,
        clip_abs_min: float | None = None,
        clip_abs_max: float | None = None,
    ) -> torch.Tensor:
        if self.prior is None:
            raise RuntimeError("FlowDurationPredictorWithPrior: prior head not initialized (use_prior=False).")

        mu, logs = self.prior(x_tok_bld, x_mask=x_mask_b1l)
        logs = logs.clamp(min=float(prior_logs_min), max=float(prior_logs_max))
        sigma = torch.exp(logs).clamp_min(1e-8)
        if float(prior_sigma_min) > 0.0:
            sigma = sigma.clamp_min(float(prior_sigma_min))
        if float(sigma0_demo) > 0.0:
            sigma = torch.sqrt(sigma * sigma + (float(sigma0_demo) ** 2))

        x_cur = mu + torch.randn_like(mu) * sigma * float(noise_scale)
        dt = 1.0 / float(max(1, int(steps)))
        for i in range(int(steps)):
            t_val = float(i) * dt
            t = torch.full((x_tok_bld.size(0), 1), t_val, device=x_tok_bld.device, dtype=x_tok_bld.dtype)
            v_start = self._forward_v(
                x_tok_bld,
                x_cur,
                t,
                spk_emb=spk_emb_bd,
                style_vec=style_vec_bd,
                x_mask=x_mask_b1l,
                cond=cond,
            )
            if use_heun and i < int(steps) - 1:
                x_next = x_cur + v_start * dt
                t_next = torch.full((x_tok_bld.size(0), 1), t_val + dt, device=x_tok_bld.device, dtype=x_tok_bld.dtype)
                v_end = self._forward_v(
                    x_tok_bld,
                    x_next,
                    t_next,
                    spk_emb=spk_emb_bd,
                    style_vec=style_vec_bd,
                    x_mask=x_mask_b1l,
                    cond=cond,
                )
                x_cur = x_cur + 0.5 * (v_start + v_end) * dt
            else:
                x_cur = x_cur + v_start * dt
            if float(clip_sigma) > 0.0:
                lim = float(clip_sigma) * sigma
                x_cur = mu + (x_cur - mu).clamp(min=-lim, max=lim)
            if (clip_abs_min is not None) or (clip_abs_max is not None):
                x_cur = x_cur.clamp(
                    min=float(clip_abs_min) if clip_abs_min is not None else None,
                    max=float(clip_abs_max) if clip_abs_max is not None else None,
                )
        return x_cur * x_mask_b1l.squeeze(1)


# Backward compatibility: keep old name.
FlowDurationPredictor = FlowDurationPredictorWithPrior
