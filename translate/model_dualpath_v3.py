"""
WegorzTranslatorDualPathV3 — DualPath encoder + GQA decoder z RMSNorm.

Zmiany względem v1 (model_dualpath.py):
  - RMSNorm zamiast LayerNorm (encoder + decoder) — ~10% szybszy, stabilniejszy
  - Grouped-Query Attention (GQA) w dekoderze — n_kv_heads < n_heads
      small (4Q): 2 KV heads  → 2× mniej KV cache
      base  (8Q): 2 KV heads  → 4× mniej KV cache
      large (12Q): 3 KV heads → 4× mniej KV cache
  - Encoder (DualPathTransformerBlock) bez zmian — ma własne normy wewnętrznie

V3 dodaje szybką ścieżkę inference:
  - batch greedy decode dla beam_size=1
  - self-attention KV cache
  - cross-attention KV cache z encodera
  - prealokowany bufor tokenów

Nazwy parametrów są zgodne z V2, więc checkpointy V2 można wczytać 1:1.
"""
from __future__ import annotations

import sys
import math
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model_shared import SwiGLU, alibi_bias, alibi_cross_bias


# ── Local DualPath encoder block ─────────────────────────────────────────────

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

    def forward(self, x: Tensor, key_padding_mask: Tensor | None = None) -> Tensor:
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
            try:
                from torch.nn.attention import SDPBackend, sdpa_kernel  # type: ignore
                with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]):
                    out = F.scaled_dot_product_attention(
                        q, k, v, attn_mask=attn_mask, dropout_p=0.0, is_causal=False
                    )
            except Exception:
                with torch.backends.cuda.sdp_kernel(enable_math=False, enable_flash=True, enable_mem_efficient=True):
                    out = F.scaled_dot_product_attention(
                        q, k, v, attn_mask=attn_mask, dropout_p=0.0, is_causal=False
                    )
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

    def forward(self, x: Tensor, cond: Tensor | None) -> Tensor:
        if cond is None:
            return x
        h = self.proj(cond.to(x.dtype)).unsqueeze(1)
        gamma, beta = h.chunk(2, dim=-1)
        return x * (1.0 + gamma) + beta


class TimeAdaptiveLayerNorm(nn.Module):
    def __init__(self, dim: int, init_t_gain: float = 1.00):
        super().__init__()
        dim = int(dim)
        self.ln = nn.LayerNorm(dim)
        self.t_scale = nn.Linear(dim, dim)
        self.t_shift = nn.Linear(dim, dim)
        self.g_t = nn.Parameter(torch.tensor(float(init_t_gain)))

    def forward(self, x_btd: Tensor, t_vec_bd: Tensor | None) -> Tensor:
        x = self.ln(x_btd)
        if t_vec_bd is None:
            return x
        t_scale = self.t_scale(t_vec_bd)[:, None, :]
        t_shift = self.t_shift(t_vec_bd)[:, None, :]
        scale = 1.0 + self.g_t * t_scale
        shift = self.g_t * t_shift
        return x * scale + shift


class DualAdaptiveLayerNorm(nn.Module):
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
        x_btd: Tensor,
        spk_vec_bd: Tensor | None,
        style_vec_bd: Tensor | None,
        t_vec_bd: Tensor | None,
    ) -> Tensor:
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

    def forward(self, x: Tensor) -> Tensor:
        h1 = self.act(self.fc1(x))
        h2 = self.fc2(x)
        h = h1 * h2
        h = self.dropout(h)
        return self.fc_out(h)


class DualPathTransformerBlock(nn.Module):
    """
    Local copy of the SalmonTTS DualPath text encoder block.

    Kept here so the translator does not depend on the TTS backbone module.
    Parameter names match the original block, so existing V2/V3 checkpoints
    remain compatible.
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
    ):
        super().__init__()
        dim = int(dim)
        assert dim % 2 == 0, "dim powinno być parzyste (dzielone na dwie gałęzie)."
        self.dim = dim
        self.use_adaln = bool(use_adaln or use_time_adaln)
        self.use_dual_adaln = bool(use_dual_adaln)
        self.use_time_adaln = bool(use_time_adaln)
        if self.use_dual_adaln and self.use_time_adaln:
            raise ValueError("DualPathTransformerBlock: use_dual_adaln and use_time_adaln are mutually exclusive.")
        half = dim // 2
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
        self.attn = SelfAttention(half, int(num_heads), use_sdpa=use_sdpa)

        self.pre_conv_norm = nn.LayerNorm(half)
        self.convs = nn.ModuleList(
            [
                nn.Conv1d(
                    half,
                    half,
                    kernel_size=int(k),
                    padding=(int(k) // 2) * int(d),
                    dilation=int(d),
                    groups=half,
                )
                for k, d in zip(kernel_sizes, dilations)
            ]
        )
        self.merge_conv = nn.Conv1d(half, half, kernel_size=1)
        self.post_conv_norm = nn.LayerNorm(half)

        self.gate1 = nn.Linear(half, 2 * K)
        self.gate2 = nn.Linear(dim, 4)
        self.conv_act = str(conv_act).lower().strip()

    def forward(
        self,
        x: Tensor,
        cond: Tensor | None = None,
        t_emb: Tensor | None = None,
        spk_emb: Tensor | None = None,
        style_vec: Tensor | None = None,
        key_padding_mask: Tensor | None = None,
    ) -> Tensor:
        B, _, _ = x.shape
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


# ── RMSNorm ───────────────────────────────────────────────────────────────────

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization — używa sfuzowanego F.rms_norm (PyTorch ≥ 2.4)."""

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        return F.rms_norm(x, self.weight.shape, self.weight, self.eps)


# ── GQA attention ─────────────────────────────────────────────────────────────

def _repeat_kv(x: Tensor, n_rep: int) -> Tensor:
    """[B, n_kv, T, d] → [B, n_kv*n_rep, T, d]"""
    if n_rep == 1:
        return x
    B, n_kv, T, d = x.shape
    return (x[:, :, None, :, :]
              .expand(B, n_kv, n_rep, T, d)
              .reshape(B, n_kv * n_rep, T, d))


class GQAttention(nn.Module):
    """Multi-Query / Grouped-Query Attention.

    n_kv_heads == n_heads  → standard MHA
    n_kv_heads == 1        → Multi-Query Attention
    else                   → Grouped-Query Attention
    """

    def __init__(self, dim: int, n_heads: int, n_kv_heads: int,
                 dropout: float = 0.1):
        super().__init__()
        assert dim % n_heads == 0, "dim musi być podzielne przez n_heads"
        assert n_heads % n_kv_heads == 0, "n_heads musi być podzielne przez n_kv_heads"
        self.n_heads    = n_heads
        self.n_kv_heads = n_kv_heads
        self.n_rep      = n_heads // n_kv_heads
        self.head_dim   = dim // n_heads
        self.scale      = self.head_dim ** -0.5

        self.q   = nn.Linear(dim, n_heads    * self.head_dim, bias=False)
        self.k   = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.v   = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(
        self,
        q: Tensor, k: Tensor, v: Tensor,
        attn_mask: Optional[Tensor] = None,
        alibi: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        B, T_q, _ = q.shape
        T_k = k.shape[1]

        Q = self.q(q).view(B, T_q, self.n_heads,    self.head_dim).transpose(1, 2)
        K = self.k(k).view(B, T_k, self.n_kv_heads, self.head_dim).transpose(1, 2)
        V = self.v(v).view(B, T_k, self.n_kv_heads, self.head_dim).transpose(1, 2)

        K = _repeat_kv(K, self.n_rep)   # [B, n_heads, T_k, head_dim]
        V = _repeat_kv(V, self.n_rep)

        # Złącz wszystkie maski w jeden addytywny tensor dla SDPA
        combined: Optional[Tensor] = alibi
        if attn_mask is not None:
            combined = attn_mask if combined is None else combined + attn_mask
        if key_padding_mask is not None:
            # UWAGA: bool*(-inf) = NaN gdy maska jest False — używamy where zamiast mnożenia
            kpm = torch.where(key_padding_mask[:, None, None, :],
                              torch.tensor(float("-inf"), device=q.device, dtype=q.dtype),
                              torch.zeros(1, device=q.device, dtype=q.dtype))
            combined = kpm if combined is None else combined + kpm

        dropout_p = self.drop.p if self.training else 0.0
        out = F.scaled_dot_product_attention(Q, K, V,
                                             attn_mask=combined,
                                             dropout_p=dropout_p)
        out = out.transpose(1, 2).reshape(B, T_q, -1)
        return self.out(out)

    def project_kv(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        B, T, _ = x.shape
        K = self.k(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        V = self.v(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        return K, V

    def attend_kv(
        self,
        q: Tensor,
        K: Tensor,
        V: Tensor,
        alibi: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        B, T_q, _ = q.shape
        Q = self.q(q).view(B, T_q, self.n_heads, self.head_dim).transpose(1, 2)
        K_rep = _repeat_kv(K, self.n_rep)
        V_rep = _repeat_kv(V, self.n_rep)

        combined: Optional[Tensor] = alibi
        if key_padding_mask is not None:
            kpm = torch.where(key_padding_mask[:, None, None, :],
                              torch.tensor(float("-inf"), device=q.device, dtype=q.dtype),
                              torch.zeros(1, device=q.device, dtype=q.dtype))
            combined = kpm if combined is None else combined + kpm

        out = F.scaled_dot_product_attention(Q, K_rep, V_rep,
                                             attn_mask=combined,
                                             dropout_p=0.0)
        out = out.transpose(1, 2).reshape(B, T_q, -1)
        return self.out(out)


# ── Decoder V2 ────────────────────────────────────────────────────────────────

class DecoderLayerV2(nn.Module):
    def __init__(self, dim: int, n_heads: int, n_kv_heads: int,
                 ffn_dim: int, dropout: float = 0.1):
        super().__init__()
        self.norm1      = RMSNorm(dim)
        self.self_attn  = GQAttention(dim, n_heads, n_kv_heads, dropout)
        self.norm2      = RMSNorm(dim)
        self.cross_attn = GQAttention(dim, n_heads, n_kv_heads, dropout)
        self.norm3      = RMSNorm(dim)
        self.ffn        = SwiGLU(dim, ffn_dim, dropout)
        self.drop       = nn.Dropout(dropout)

    def forward(
        self, x: Tensor, enc_out: Tensor,
        self_attn_mask: Optional[Tensor] = None,
        enc_key_padding_mask: Optional[Tensor] = None,
        self_alibi: Optional[Tensor] = None,
        cross_alibi: Optional[Tensor] = None,
    ) -> Tensor:
        h = self.norm1(x)
        x = x + self.drop(self.self_attn(h, h, h,
                                          attn_mask=self_attn_mask,
                                          alibi=self_alibi))
        h = self.norm2(x)
        x = x + self.drop(self.cross_attn(h, enc_out, enc_out,
                                           alibi=cross_alibi,
                                           key_padding_mask=enc_key_padding_mask))
        x = x + self.drop(self.ffn(self.norm3(x)))
        return x

    def init_cross_cache(self, enc_out: Tensor) -> Tuple[Tensor, Tensor]:
        return self.cross_attn.project_kv(enc_out)

    def step(
        self,
        x: Tensor,
        self_cache: Optional[Tuple[Tensor, Tensor]],
        cross_cache: Tuple[Tensor, Tensor],
        self_alibi: Tensor,
        cross_alibi: Tensor,
        enc_key_padding_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        h = self.norm1(x)
        K_new, V_new = self.self_attn.project_kv(h)
        if self_cache is None:
            K_self, V_self = K_new, V_new
        else:
            K_self = torch.cat([self_cache[0], K_new], dim=2)
            V_self = torch.cat([self_cache[1], V_new], dim=2)
        x = x + self.self_attn.attend_kv(h, K_self, V_self, alibi=self_alibi)

        h = self.norm2(x)
        x = x + self.cross_attn.attend_kv(
            h, cross_cache[0], cross_cache[1],
            alibi=cross_alibi,
            key_padding_mask=enc_key_padding_mask,
        )
        x = x + self.ffn(self.norm3(x))
        return x, (K_self, V_self)


class DecoderV2(nn.Module):
    def __init__(self, vocab_size: int, dim: int, n_layers: int,
                 n_heads: int, n_kv_heads: int, ffn_dim: int,
                 dropout: float = 0.1):
        super().__init__()
        self.embed  = nn.Embedding(vocab_size, dim, padding_idx=0)
        self.drop   = nn.Dropout(dropout)
        self.layers = nn.ModuleList([
            DecoderLayerV2(dim, n_heads, n_kv_heads, ffn_dim, dropout)
            for _ in range(n_layers)
        ])
        self.norm    = RMSNorm(dim)
        self.n_heads = n_heads

    def forward(
        self, tgt: Tensor, enc_out: Tensor,
        enc_key_padding_mask: Optional[Tensor] = None,
        return_exit_layer: Optional[int] = None,
    ) -> Tensor:
        x   = self.drop(self.embed(tgt))
        T_q = tgt.shape[1]
        T_k = enc_out.shape[1]

        causal = torch.full((T_q, T_q), float("-inf"), device=tgt.device, dtype=x.dtype)
        causal = torch.triu(causal, diagonal=1)

        self_alibi  = alibi_bias(T_q, self.n_heads, tgt.device, x.dtype)
        cross_alibi = alibi_cross_bias(T_q, T_k, self.n_heads, tgt.device, x.dtype)

        exit_x = None
        for idx, layer in enumerate(self.layers, 1):
            x = layer(x, enc_out,
                      self_attn_mask=causal,
                      enc_key_padding_mask=enc_key_padding_mask,
                      self_alibi=self_alibi,
                      cross_alibi=cross_alibi)
            if return_exit_layer is not None and idx == return_exit_layer:
                exit_x = self.norm(x)
        x = self.norm(x)
        if return_exit_layer is not None:
            return x, (exit_x if exit_x is not None else x)
        return x

    def init_cross_cache(self, enc_out: Tensor) -> list[Tuple[Tensor, Tensor]]:
        return [layer.init_cross_cache(enc_out) for layer in self.layers]

    def _self_step_alibi(self, step: int, device, dtype) -> Tensor:
        slopes = torch.tensor(
            [2.0 ** (-8.0 * i / self.n_heads) for i in range(1, self.n_heads + 1)],
            device=device, dtype=torch.float32,
        )
        keys = torch.arange(step + 1, device=device, dtype=torch.float32)
        dist = (float(step) - keys).abs()
        return -(slopes[:, None, None] * dist[None, None, :]).unsqueeze(0).to(dtype)

    def _cross_step_alibi(self, step: int, T_k: int, device, dtype) -> Tensor:
        slopes = torch.tensor(
            [2.0 ** (-8.0 * i / self.n_heads) for i in range(1, self.n_heads + 1)],
            device=device, dtype=torch.float32,
        )
        q_scaled = 0.0 if step == 0 else float(T_k - 1)
        keys = torch.arange(T_k, device=device, dtype=torch.float32)
        dist = (q_scaled - keys).abs()
        return -(slopes[:, None, None] * dist[None, None, :]).unsqueeze(0).to(dtype)

    def decode_step(
        self,
        token: Tensor,
        step: int,
        self_caches: list[Optional[Tuple[Tensor, Tensor]]],
        cross_caches: list[Tuple[Tensor, Tensor]],
        enc_key_padding_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, list[Tuple[Tensor, Tensor]]]:
        x = self.embed(token[:, None])
        T_k = cross_caches[0][0].shape[2]
        self_alibi = self._self_step_alibi(step, token.device, x.dtype)
        cross_alibi = self._cross_step_alibi(step, T_k, token.device, x.dtype)
        new_caches: list[Tuple[Tensor, Tensor]] = []
        for idx, layer in enumerate(self.layers):
            x, cache = layer.step(
                x,
                self_caches[idx],
                cross_caches[idx],
                self_alibi,
                cross_alibi,
                enc_key_padding_mask=enc_key_padding_mask,
            )
            new_caches.append(cache)
        return self.norm(x), new_caches


# ── DualPath Encoder z RMSNorm ────────────────────────────────────────────────

class DualPathEncoderV2(nn.Module):
    """DualPathTransformerBlock stack z RMSNorm na wyjściu (zamiast LayerNorm)."""

    def __init__(self, vocab_size: int, dim: int, n_layers: int, n_heads: int,
                 dropout: float = 0.1,
                 kernel_sizes: tuple = (3, 9, 15),
                 dilations: tuple    = (1, 2, 4)):
        super().__init__()
        assert dim % 2 == 0
        self.embed  = nn.Embedding(vocab_size, dim, padding_idx=0)
        self.drop   = nn.Dropout(dropout)
        self.layers = nn.ModuleList([
            DualPathTransformerBlock(dim=dim, num_heads=n_heads,
                                     kernel_sizes=kernel_sizes,
                                     dilations=dilations,
                                     use_sdpa=True, use_adaln=False)
            for _ in range(n_layers)
        ])
        self.norm = RMSNorm(dim)

    def forward(self, src: Tensor,
                src_padding_mask: Optional[Tensor] = None) -> Tensor:
        x = self.drop(self.embed(src))
        if src_padding_mask is not None:
            keep = (~src_padding_mask).to(dtype=x.dtype, device=x.device).unsqueeze(-1)
            x = x * keep
        for layer in self.layers:
            x = layer(x, key_padding_mask=src_padding_mask)
        x = self.norm(x)
        if src_padding_mask is not None:
            x = x * keep
        return x


# ── Causal DualPath decoder experiment ────────────────────────────────────────

class CausalDualPathSelfBlock(nn.Module):
    """Causal decoder-side variant of DualPath block.

    Differences from encoder block:
      - self-attention receives a causal mask / KV cache
      - conv branch uses left-only padding
      - kernel gate is token-wise, not global pooled over the full sequence
    """

    def __init__(
        self,
        dim: int,
        n_heads: int,
        n_kv_heads: int,
        kernel_sizes=(3, 9, 15),
        dilations=(1, 2, 4),
        dropout: float = 0.1,
        conv_act: str = "gelu",
    ):
        super().__init__()
        assert dim % 2 == 0
        half = dim // 2
        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)
        self.norm3 = RMSNorm(dim)
        self.ffn_pre = SwiGLU(dim, dim * 4, dropout)
        self.ffn_post = SwiGLU(dim, dim * 4, dropout)
        self.attn = GQAttention(half, n_heads, n_kv_heads, dropout)
        self.pre_conv_norm = nn.LayerNorm(half)
        self.convs = nn.ModuleList([
            nn.Conv1d(
                half, half,
                kernel_size=int(k),
                padding=0,
                dilation=int(d),
                groups=half,
            )
            for k, d in zip(kernel_sizes, dilations)
        ])
        self.left_pads = [(int(k) - 1) * int(d) for k, d in zip(kernel_sizes, dilations)]
        self.merge_conv = nn.Conv1d(half, half, kernel_size=1)
        self.post_conv_norm = nn.LayerNorm(half)
        self.gate1 = nn.Linear(half, 2 * len(kernel_sizes))
        self.gate2 = nn.Linear(dim, 4)
        self.drop = nn.Dropout(dropout)
        self.conv_act = str(conv_act).lower().strip()

    def _conv_full(self, x2n: Tensor) -> Tensor:
        x2_t = x2n.transpose(1, 2)
        conv_outs = [
            conv(F.pad(x2_t, (left_pad, 0)))
            for conv, left_pad in zip(self.convs, self.left_pads)
        ]
        c_stack = torch.stack(conv_outs, dim=-1)  # [B, half, T, K]
        gate_out = self.gate1(x2n)                # [B, T, 2K]
        K = len(self.convs)
        ga, gb = gate_out.chunk(2, dim=-1)
        w1 = (ga * torch.sigmoid(gb)).softmax(dim=-1).unsqueeze(1)
        c_weighted = (c_stack * w1).sum(dim=-1)
        c = self.merge_conv(c_weighted).transpose(1, 2)
        return self.post_conv_norm(c)

    def forward(
        self,
        x: Tensor,
        self_attn_mask: Optional[Tensor] = None,
        self_alibi: Optional[Tensor] = None,
    ) -> Tensor:
        res = x
        x = x + 0.5 * self.drop(self.ffn_pre(self.norm1(x)))
        h = self.norm2(x)
        x1, x2 = h.chunk(2, dim=-1)

        a = self.attn(x1, x1, x1, attn_mask=self_attn_mask, alibi=self_alibi)

        x2n = self.pre_conv_norm(x2)
        x2n = torch.sin(x2n) if self.conv_act == "sin" else F.gelu(x2n)
        c = self._conv_full(x2n)

        combined = torch.cat([a, c], dim=-1)
        g2a, g2b = self.gate2(combined).chunk(2, dim=-1)
        w2 = (g2a * torch.sigmoid(g2b)).softmax(dim=-1)
        merged = torch.cat([w2[..., 0:1] * a, w2[..., 1:2] * c], dim=-1)

        x = res + self.drop(merged)
        x = x + 0.5 * self.drop(self.ffn_post(self.norm3(x)))
        return x

    def step(
        self,
        x: Tensor,
        cache: Optional[tuple[Tensor, Tensor, Tensor]],
        self_alibi: Tensor,
    ) -> tuple[Tensor, tuple[Tensor, Tensor, Tensor]]:
        res = x
        x = x + 0.5 * self.ffn_pre(self.norm1(x))
        h = self.norm2(x)
        x1, x2 = h.chunk(2, dim=-1)

        K_new, V_new = self.attn.project_kv(x1)
        if cache is None:
            K_self, V_self = K_new, V_new
            x2_hist = None
        else:
            K_self = torch.cat([cache[0], K_new], dim=2)
            V_self = torch.cat([cache[1], V_new], dim=2)
            x2_hist = cache[2]
        a = self.attn.attend_kv(x1, K_self, V_self, alibi=self_alibi)

        x2n = self.pre_conv_norm(x2)
        x2n = torch.sin(x2n) if self.conv_act == "sin" else F.gelu(x2n)
        x2_all = x2n if x2_hist is None else torch.cat([x2_hist, x2n], dim=1)
        c = self._conv_full(x2_all)[:, -1:, :]

        combined = torch.cat([a, c], dim=-1)
        g2a, g2b = self.gate2(combined).chunk(2, dim=-1)
        w2 = (g2a * torch.sigmoid(g2b)).softmax(dim=-1)
        merged = torch.cat([w2[..., 0:1] * a, w2[..., 1:2] * c], dim=-1)

        x = res + merged
        x = x + 0.5 * self.ffn_post(self.norm3(x))
        return x, (K_self, V_self, x2_all)


class CausalDualPathDecoderLayer(nn.Module):
    def __init__(
        self,
        dim: int,
        n_heads: int,
        n_kv_heads: int,
        ffn_dim: int,
        dropout: float = 0.1,
        kernel_sizes=(3, 9, 15),
        dilations=(1, 2, 4),
    ):
        super().__init__()
        self.self_block = CausalDualPathSelfBlock(
            dim, n_heads, n_kv_heads,
            kernel_sizes=kernel_sizes,
            dilations=dilations,
            dropout=dropout,
        )
        self.norm_cross = RMSNorm(dim)
        self.cross_attn = GQAttention(dim, n_heads, n_kv_heads, dropout)
        self.norm_ffn = RMSNorm(dim)
        self.ffn = SwiGLU(dim, ffn_dim, dropout)
        self.drop = nn.Dropout(dropout)

    def forward(
        self,
        x: Tensor,
        enc_out: Tensor,
        self_attn_mask: Optional[Tensor] = None,
        enc_key_padding_mask: Optional[Tensor] = None,
        self_alibi: Optional[Tensor] = None,
        cross_alibi: Optional[Tensor] = None,
    ) -> Tensor:
        x = self.self_block(x, self_attn_mask=self_attn_mask, self_alibi=self_alibi)
        h = self.norm_cross(x)
        x = x + self.drop(self.cross_attn(
            h, enc_out, enc_out,
            alibi=cross_alibi,
            key_padding_mask=enc_key_padding_mask,
        ))
        x = x + self.drop(self.ffn(self.norm_ffn(x)))
        return x

    def init_cross_cache(self, enc_out: Tensor) -> Tuple[Tensor, Tensor]:
        return self.cross_attn.project_kv(enc_out)

    def step(
        self,
        x: Tensor,
        self_cache: Optional[tuple[Tensor, Tensor, Tensor]],
        cross_cache: Tuple[Tensor, Tensor],
        self_alibi: Tensor,
        cross_alibi: Tensor,
        enc_key_padding_mask: Optional[Tensor] = None,
    ) -> tuple[Tensor, tuple[Tensor, Tensor, Tensor]]:
        x, cache = self.self_block.step(x, self_cache, self_alibi)
        h = self.norm_cross(x)
        x = x + self.cross_attn.attend_kv(
            h, cross_cache[0], cross_cache[1],
            alibi=cross_alibi,
            key_padding_mask=enc_key_padding_mask,
        )
        x = x + self.ffn(self.norm_ffn(x))
        return x, cache


class CausalDualPathDecoderV3(DecoderV2):
    def __init__(
        self,
        vocab_size: int,
        dim: int,
        n_layers: int,
        n_heads: int,
        n_kv_heads: int,
        ffn_dim: int,
        dropout: float = 0.1,
        kernel_sizes=(3, 9, 15),
        dilations=(1, 2, 4),
    ):
        nn.Module.__init__(self)
        self.embed = nn.Embedding(vocab_size, dim, padding_idx=0)
        self.drop = nn.Dropout(dropout)
        self.layers = nn.ModuleList([
            CausalDualPathDecoderLayer(
                dim, n_heads, n_kv_heads, ffn_dim, dropout,
                kernel_sizes=kernel_sizes,
                dilations=dilations,
            )
            for _ in range(n_layers)
        ])
        self.norm = RMSNorm(dim)
        self.n_heads = n_heads


# ── Full model V3 ─────────────────────────────────────────────────────────────

class WegorzTranslatorDualPathV3(nn.Module):
    """DualPath encoder + GQA decoder z RMSNorm.

    n_kv_heads < n_heads → mniejszy KV cache przy inference.
    """

    def __init__(
        self,
        vocab_size: int,
        dim: int          = 512,
        n_enc_layers: int = 6,
        n_dec_layers: int = 6,
        n_heads: int      = 8,
        n_kv_heads: int   = 2,
        ffn_dim: int      = 2048,
        dropout: float    = 0.1,
        pad_id: int       = 0,
        bos_id: int       = 1,
        eos_id: int       = 2,
        kernel_sizes: tuple = (3, 9, 15),
        dilations: tuple    = (1, 2, 4),
    ):
        super().__init__()
        self.pad_id = pad_id
        self.bos_id = bos_id
        self.eos_id = eos_id

        self.encoder = DualPathEncoderV2(
            vocab_size, dim, n_enc_layers, n_heads, dropout,
            kernel_sizes=kernel_sizes, dilations=dilations,
        )
        self.decoder = DecoderV2(
            vocab_size, dim, n_dec_layers, n_heads, n_kv_heads, ffn_dim, dropout
        )
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)
        self.decoder.embed.weight = self.lm_head.weight   # weight tying

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)
                if m.padding_idx is not None:
                    m.weight.data[m.padding_idx].zero_()

    def encode(self, src: Tensor) -> Tuple[Tensor, Tensor]:
        pad_mask = src.eq(self.pad_id)
        enc_out  = self.encoder(src, src_padding_mask=pad_mask)
        return enc_out, pad_mask

    def forward(self, src: Tensor, tgt: Tensor) -> Tensor:
        enc_out, enc_pad = self.encode(src)
        dec_out = self.decoder(tgt, enc_out, enc_key_padding_mask=enc_pad)
        return self.lm_head(dec_out)

    @torch.no_grad()
    def generate(
        self,
        src: Tensor,
        max_new_tokens: int  = 200,
        beam_size: int       = 4,
        length_penalty: float = 0.6,
        temperature: float   = 1.0,
    ) -> Tensor:
        if beam_size == 1:
            return self.generate_greedy_batch(
                src,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )

        device   = src.device
        enc_out, enc_pad = self.encode(src)

        beams     = [(0.0, [self.bos_id])]
        completed = []

        for _ in range(max_new_tokens):
            candidates = []
            for score, tokens in beams:
                if tokens[-1] == self.eos_id:
                    completed.append((score, tokens))
                    continue
                tgt     = torch.tensor([tokens], device=device)
                dec_out = self.decoder(tgt, enc_out.expand(1, -1, -1),
                                       enc_key_padding_mask=enc_pad)
                logits   = self.lm_head(dec_out[:, -1, :]) / temperature
                log_probs = F.log_softmax(logits, dim=-1)[0]
                topk = torch.topk(log_probs, beam_size)
                for log_p, idx in zip(topk.values.tolist(), topk.indices.tolist()):
                    candidates.append((score + log_p, tokens + [idx]))

            if not candidates:
                break
            candidates.sort(key=lambda x: x[0] / (len(x[1]) ** length_penalty),
                            reverse=True)
            beams = candidates[:beam_size]
            if all(t[-1] == self.eos_id for _, t in beams):
                completed.extend(beams)
                break

        pool = completed if completed else beams
        best = max(pool, key=lambda x: x[0] / (len(x[1]) ** length_penalty))
        return torch.tensor([best[1]], device=device)

    @torch.no_grad()
    def generate_greedy_batch(
        self,
        src: Tensor,
        max_new_tokens: int = 200,
        temperature: float = 1.0,
    ) -> Tensor:
        return self._generate_greedy_batch_same_len(
            src,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )

    @torch.no_grad()
    def _generate_greedy_batch_same_len(
        self,
        src: Tensor,
        max_new_tokens: int = 200,
        temperature: float = 1.0,
    ) -> Tensor:
        self.eval()
        B = src.shape[0]
        device = src.device
        enc_out, enc_pad = self.encode(src)
        cross_caches = self.decoder.init_cross_cache(enc_out)
        self_caches: list[Optional[Tuple[Tensor, Tensor]]] = [None] * len(self.decoder.layers)

        tokens = torch.full((B, max_new_tokens + 1), self.pad_id,
                            dtype=torch.long, device=device)
        tokens[:, 0] = self.bos_id
        cur = tokens[:, 0]
        finished = torch.zeros(B, dtype=torch.bool, device=device)

        for step in range(max_new_tokens):
            dec_out, self_caches = self.decoder.decode_step(
                cur, step, self_caches, cross_caches,
                enc_key_padding_mask=enc_pad,
            )
            logits = self.lm_head(dec_out[:, -1, :]) / temperature
            nxt = torch.argmax(logits, dim=-1)
            nxt = torch.where(finished, torch.full_like(nxt, self.pad_id), nxt)
            tokens[:, step + 1] = nxt
            finished |= nxt.eq(self.eos_id)
            cur = nxt
            if bool(finished.all()):
                return tokens[:, :step + 2]
        return tokens

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


class WegorzTranslatorDualPathV3CausalDec(WegorzTranslatorDualPathV3):
    """DualPathV3 z kauzalnym DualPath self-blockiem w decoderze."""

    def __init__(
        self,
        vocab_size: int,
        dim: int          = 512,
        n_enc_layers: int = 6,
        n_dec_layers: int = 6,
        n_heads: int      = 8,
        n_kv_heads: int   = 2,
        ffn_dim: int      = 2048,
        dropout: float    = 0.1,
        pad_id: int       = 0,
        bos_id: int       = 1,
        eos_id: int       = 2,
        kernel_sizes: tuple = (3, 9, 15),
        dilations: tuple    = (1, 2, 4),
    ):
        super().__init__(
            vocab_size=vocab_size,
            dim=dim,
            n_enc_layers=n_enc_layers,
            n_dec_layers=n_dec_layers,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            ffn_dim=ffn_dim,
            dropout=dropout,
            pad_id=pad_id,
            bos_id=bos_id,
            eos_id=eos_id,
            kernel_sizes=kernel_sizes,
            dilations=dilations,
        )
        self.decoder = CausalDualPathDecoderV3(
            vocab_size, dim, n_dec_layers, n_heads, n_kv_heads, ffn_dim, dropout,
            kernel_sizes=kernel_sizes,
            dilations=dilations,
        )
        self.decoder.embed.weight = self.lm_head.weight
        self._init_weights()


class WegorzTranslatorDualPathV3EarlyExit(WegorzTranslatorDualPathV3):
    """DualPathV3 z early-exit logits po plytkiej warstwie AR decodera."""

    def __init__(self, *args, exit_layer: int = 2, **kwargs):
        super().__init__(*args, **kwargs)
        self.exit_layer = int(exit_layer)

    def forward(self, src: Tensor, tgt: Tensor,
                return_aux: bool = False):
        enc_out, enc_pad = self.encode(src)
        if not return_aux:
            dec_out = self.decoder(tgt, enc_out, enc_key_padding_mask=enc_pad)
            return self.lm_head(dec_out)
        dec_out, exit_out = self.decoder(
            tgt, enc_out,
            enc_key_padding_mask=enc_pad,
            return_exit_layer=self.exit_layer,
        )
        return self.lm_head(dec_out), self.lm_head(exit_out)


# ── Configs ───────────────────────────────────────────────────────────────────
# n_kv_heads: small=2 (2× redukcja), base=2 (4×), large=3 (4×)

CONFIGS_DUALPATH_V3 = {
    "small": dict(dim=256, n_enc_layers=4, n_dec_layers=4, n_heads=4,  n_kv_heads=2, ffn_dim=1024),
    "base":  dict(dim=512, n_enc_layers=6, n_dec_layers=6, n_heads=8,  n_kv_heads=2, ffn_dim=2048),
    "large": dict(dim=768, n_enc_layers=8, n_dec_layers=8, n_heads=12, n_kv_heads=3, ffn_dim=3072),
    # Deep-encoder / shallow-decoder architecture probes.
    # Paper baseline notation E-D: base_e12d1 = 12 encoder layers, 1 decoder layer.
    "base_e6d1":  dict(dim=512, n_enc_layers=6,  n_dec_layers=1, n_heads=8, n_kv_heads=2, ffn_dim=2048),
    "base_e6d3":  dict(dim=512, n_enc_layers=6,  n_dec_layers=3, n_heads=8, n_kv_heads=2, ffn_dim=2048),
    "base_e6d6":  dict(dim=512, n_enc_layers=6,  n_dec_layers=6, n_heads=8, n_kv_heads=2, ffn_dim=2048),
    "base_e12d1": dict(dim=512, n_enc_layers=12, n_dec_layers=1, n_heads=8, n_kv_heads=2, ffn_dim=2048),
    "base_e12d3": dict(dim=512, n_enc_layers=12, n_dec_layers=3, n_heads=8, n_kv_heads=2, ffn_dim=2048),
    "base_e12d6": dict(dim=512, n_enc_layers=12, n_dec_layers=6, n_heads=8, n_kv_heads=2, ffn_dim=2048),
}

CONFIGS_DUALPATH_V2 = CONFIGS_DUALPATH_V3
WegorzTranslatorDualPathV2 = WegorzTranslatorDualPathV3


if __name__ == "__main__":
    import time
    V = 8000
    for name, cfg in CONFIGS_DUALPATH_V2.items():
        m = WegorzTranslatorDualPathV3(vocab_size=V, **cfg)
        src = torch.randint(3, V, (2, 30))
        tgt = torch.randint(3, V, (2, 20))
        t0 = time.perf_counter()
        for _ in range(10):
            _ = m(src, tgt)
        ms = (time.perf_counter() - t0) / 10 * 1000
        kv_ratio = cfg['n_kv_heads'] / cfg['n_heads']
        print(f"  {name:6s}  {m.n_params()/1e6:.1f}M params  "
              f"KV={cfg['n_kv_heads']}/{cfg['n_heads']} ({kv_ratio:.0%})  "
              f"{ms:.1f}ms/batch")



'''
Casual
============================================================
Epoka 1/5
  step    200  loss 7.5942  lr 1.00e-04  44s
  step    400  loss 7.0510  lr 2.00e-04  89s
  step    600  loss 6.7152  lr 3.00e-04  134s
  step    800  loss 6.4463  lr 4.00e-04  179s
  step   1000  loss 6.2092  lr 5.00e-04  224s
  step   1200  loss 5.9890  lr 5.00e-04  273s
  step   1400  loss 5.7826  lr 4.99e-04  323s
  step   1600  loss 5.5952  lr 4.98e-04  372s
  step   1800  loss 5.4240  lr 4.96e-04  418s
  step   2000  loss 5.2699  lr 4.94e-04  464s
  step   2200  loss 5.1317  lr 4.92e-04  511s
  step   2400  loss 5.0050  lr 4.89e-04  558s
  step   2600  loss 4.8907  lr 4.85e-04  605s
  step   2800  loss 4.7869  lr 4.82e-04  652s
  step   3000  loss 4.6922  lr 4.77e-04  699s
  Train loss: 4.6361  Val loss: 3.2380
  EN: The scientist refused to comment on the controversial new study.
  PL: Uczestnik odmówił się do komentarza na temat kontrowersji.

  EN: She was beating around the bush instead of getting straight to the point.
  PL: Była to, że wokół krzewów, zamiast udowodnić na punkt.

  EN: The algorithm computes the shortest path using a modified Dijkstra's approach.
  PL: Numer algorytmu wzbudza największą ścieżkę za pomocą manipulacji Dijkstraa.

  EN: After weeks of negotiations, both sides finally reached a compromise.
  PL: Po tygodniach negocjacji, obie strony w końcu dotarły do kompromisu.

  EN: He realized too late that he had missed the point of the argument.
  PL: Zastoszył się zbyt późno, że zabrał punkt.

  EN: The new policy aims to reduce energy consumption without slowing economic growth.
  PL: Nowy polityka ma na celu zmniejszenie zużycia energii bez wolnego kontroli gospodarczego.

  *** Nowy najlepszy model: val=3.2380 ***
  
>   --model dualpathv3 
egorz_translator$ /home/rizos/Miniforge3/envs/uvtts2/bin/python train.py \
>   --model dualpathv3 \
>   --config base_e6d6 \
>   --train data/coverage_balanced_5m_norm.jsonl \
>   --valid data/valid_mini_norm.jsonl \
>   --tokenizer tokenizer_norm_8k/wegorz.model \
>   --batch 32 \
>   --epochs 5 \
>   --lr 5e-4 \
>   --warmup 1000 \
>   --max-items 100000 \
>   --sample-items \
>   --cap-encode \
>   --out runs/archtest_v3_base_e6d6_8k_cov5m_100k_ep5
Device: cuda
Vocab: 8000
  [Dataset] Odfiltrowano 7 par (puste/krótkie/EN==PL)
Train: 100,000  Valid: 1,993
Model: dualpathv3  Config: base_e6d6  Parametry: 74.8M

============================================================
Epoka 1/5
  step    200  loss 7.6559  lr 1.00e-04  32s
  step    400  loss 7.1161  lr 2.00e-04  64s
  step    600  loss 6.7703  lr 3.00e-04  97s
  step    800  loss 6.4963  lr 4.00e-04  130s
  step   1000  loss 6.2570  lr 5.00e-04  163s
  step   1200  loss 6.0385  lr 5.00e-04  197s
  step   1400  loss 5.8374  lr 4.99e-04  232s
  step   1600  loss 5.6506  lr 4.98e-04  265s
  step   1800  loss 5.4817  lr 4.96e-04  298s
  step   2000  loss 5.3310  lr 4.94e-04  331s
  step   2200  loss 5.1947  lr 4.92e-04  364s
  step   2400  loss 5.0707  lr 4.89e-04  397s
  step   2600  loss 4.9580  lr 4.85e-04  430s
  step   2800  loss 4.8547  lr 4.82e-04  463s
  step   3000  loss 4.7615  lr 4.77e-04  496s
  Train loss: 4.7066  Val loss: 3.3058
  EN: The scientist refused to comment on the controversial new study.
  PL: Naukowcy odnotowali się do komentowania nowych badania.

  EN: She was beating around the bush instead of getting straight to the point.
  PL: Była po brzegu zamiast doceny do punktu.

  EN: The algorithm computes the shortest path using a modified Dijkstra's approach.
  PL: Komórka algorytmuje najkrótszą drogę przy użyciu modyfikowanej podejścia Dijkstra.

  EN: After weeks of negotiations, both sides finally reached a compromise.
  PL: Po tygodniach negocjacji, zarówno stron, jak i przykładów wreszcie dotarła do kompromisu.

  EN: He realized too late that he had missed the point of the argument.
  PL: Okazał się zbyt późno, że odnotował punktu.

  EN: The new policy aims to reduce energy consumption without slowing economic growth.
  PL: Nowa polityka ma na celu zmniejszenie zużycia zużycia energii bez prędkości gospodarczego.

  *** Nowy najlepszy model: val=3.3058 ***


KeyboardInterrupt
(uvtts2) rizos@rizos-Z390-GAMING-SLI:~/Downloads/SalmonTTS2/wegorz_translator$ /home/rizos/Miniforge3/envs/uvtts2/bin/python compare_wegorz_variants_bleu.py \
>   --valid data/valid_mini_norm.jsonl \
>   --n 300 \
>   --beam 1 \
>   --batch-size 16 \
>   --variant base_e6d6:dualpathv3:runs/archtest_v3_base_e6d6_8k_cov5m_100k_ep5/best.pt:tokenizer_norm_8k/wegorz.model \
>   --variant causaldec_e6d6:dualpathv3-causaldec:runs/archtest_v3_causaldec_e6d6_8k_cov5m_100k_ep5/best.pt:tokenizer_norm_8k/wegorz.model \
>   --out-jsonl runs/translator_bleu_compare_base_vs_causaldec_8k_cov5m_100k_300.jsonl
Próbek: 300  beam=1  batch_size=16  flow_steps=32  device=cuda
Valid: data/valid_mini_norm.jsonl

== base_e6d6 (dualpathv3) ==
ckpt: runs/archtest_v3_base_e6d6_8k_cov5m_100k_ep5/best.pt
tokenizer: tokenizer_norm_8k/wegorz.model
BLEU: 9.31  load=1.4s  gen=6.5s  22 ms/zdanie
EN : <CAP>there are only about five full-time neighbours scattered throughout the neighbourhood, so don't expect to see a lot
REF: Jest tylko około pięciu sąsiadów pełnoetatowych rozproszonych po okolicy, więc nie oczekuj, że zobaczy się wiele osób.

Ranking BLEU:
  causaldec_e6d6           BLEU=10.39  gen=14.7s  49 ms/zdanie
  base_e6d6                BLEU=9.31  gen=6.5s  22 ms/zdanie

Zapisano predykcje: runs/translator_bleu_compare_base_vs_causaldec_8k_cov5m_100k_300.jsonl



cd /home/rizos/Downloads/SalmonTTS2/wegorz_translator

/home/rizos/Miniforge3/envs/uvtts2/bin/python train.py \
  --model dualpathv3-2dec \
  --config base_e6d6 \
  --train data/coverage_balanced_5m_norm_bidir_full.jsonl \
  --valid data/valid_mini_norm.jsonl \
  --tokenizer tokenizer_norm_8k/wegorz.model \
  --batch 60 \
  --epochs 5 \
  --lr 5e-4 \
  --warmup 4000 \
  --cap-encode \
  --save-steps 5000 \
  --out runs/wegorz_dualpathv3_2dec_e6d6_8k_cov5m_bidir_full_ep5

  
cd /home/rizos/Downloads/SalmonTTS2/wegorz_translator

/home/rizos/Miniforge3/envs/uvtts2/bin/python train.py \
  --model dualpathv3-2dec \
  --config base_e6d6 \
  --train data/coverage_balanced_5m_norm_bidir_full.jsonl \
  --valid data/valid_mini_norm.jsonl \
  --tokenizer tokenizer_norm_8k/wegorz.model \
  --batch 60 \
  --epochs 10 \
  --lr 5e-4 \
  --warmup 4000 \
  --cap-encode \
  --save-steps 5000 \
  --resume /home/rizos/Downloads/SalmonTTS2/wegorz_translator/runs/wegorz_dualpathv3_2dec_e6d6_8k_cov5m_bidir_full_ep5/step_0230000.pt \
  --out runs/wegorz_dualpathv3_2dec_e6d6_8k_cov5m_bidir_full_ep5

  


step    200  loss 2.7106  lr 2.00e-05  38s
  step    400  loss 2.6808  lr 4.00e-05  76s
  step    600  loss 2.6531  lr 6.00e-05  113s
  step    800  loss 2.6287  lr 8.00e-05  149s
  step   1000  loss 2.6059  lr 1.00e-04  185s
  step   1200  loss 2.5889  lr 1.00e-04  222s
  step   1400  loss 2.5726  lr 9.98e-05  260s
  step   1600  loss 2.5599  lr 9.96e-05  298s
  step   1800  loss 2.5482  lr 9.93e-05  335s
  step   2000  loss 2.5364  lr 9.89e-05  373s
  step   2200  loss 2.5267  lr 9.83e-05  411s
  step   2400  loss 2.5178  lr 9.78e-05  449s
  step   2600  loss 2.5090  lr 9.71e-05  487s
  step   2800  loss 2.5019  lr 9.63e-05  525s
  step   3000  loss 2.4952  lr 9.55e-05  563s
  step   3200  loss 2.4877  lr 9.45e-05  602s
  step   3400  loss 2.4819  lr 9.35e-05  640s
  step   3600  loss 2.4765  lr 9.24e-05  679s
  step   3800  loss 2.4711  lr 9.12e-05  717s
  step   4000  loss 2.4659  lr 9.00e-05  756s
  step   4200  loss 2.4612  lr 8.86e-05  796s
  step   4400  loss 2.4568  lr 8.72e-05  834s
  step   4600  loss 2.4524  lr 8.58e-05  873s
  step   4800  loss 2.4480  lr 8.42e-05  912s
  step   5000  loss 2.4440  lr 8.27e-05  951s
  step   5200  loss 2.4403  lr 8.10e-05  991s
  step   5400  loss 2.4367  lr 7.93e-05  1030s
  step   5600  loss 2.4334  lr 7.75e-05  1069s
  step   5800  loss 2.4298  lr 7.57e-05  1108s
  step   6000  loss 2.4269  lr 7.38e-05  1147s
  step   6200  loss 2.4238  lr 7.19e-05  1186s
  step   6400  loss 2.4207  lr 7.00e-05  1224s
  step   6600  loss 2.4179  lr 6.80e-05  1263s
  step   6800  loss 2.4154  lr 6.60e-05  1302s
  step   7000  loss 2.4125  lr 6.39e-05  1340s
  step   7200  loss 2.4100  lr 6.18e-05  1379s
  step   7400  loss 2.4077  lr 5.97e-05  1417s
  step   7600  loss 2.4052  lr 5.76e-05  1456s
  step   7800  loss 2.4031  lr 5.55e-05  1495s
  step   8000  loss 2.4009  lr 5.34e-05  1533s
  step   8200  loss 2.3990  lr 5.12e-05  1571s
  step   8400  loss 2.3970  lr 4.91e-05  1610s
  step   8600  loss 2.3949  lr 4.69e-05  1649s
  step   8800  loss 2.3928  lr 4.48e-05  1687s
  step   9000  loss 2.3910  lr 4.26e-05  1725s
  step   9200  loss 2.3894  lr 4.05e-05  1764s
  step   9400  loss 2.3879  lr 3.84e-05  1802s
  step   9600  loss 2.3860  lr 3.63e-05  1841s
  step   9800  loss 2.3843  lr 3.43e-05  1879s
  step  10000  loss 2.3830  lr 3.23e-05  1917s
  step  10200  loss 2.3811  lr 3.03e-05  1957s
  step  10400  loss 2.3798  lr 2.83e-05  1995s
  step  10600  loss 2.3782  lr 2.64e-05  2034s
  step  10800  loss 2.3767  lr 2.45e-05  2071s
  step  11000  loss 2.3753  lr 2.27e-05  2110s
  step  11200  loss 2.3739  lr 2.09e-05  2148s
  step  11400  loss 2.3726  lr 1.92e-05  2186s
  step  11600  loss 2.3712  lr 1.76e-05  2225s
  step  11800  loss 2.3699  lr 1.59e-05  2263s
  step  12000  loss 2.3686  lr 1.44e-05  2302s
  step  12200  loss 2.3673  lr 1.29e-05  2340s
  step  12400  loss 2.3663  lr 1.15e-05  2378s
  step  12600  loss 2.3653  lr 1.02e-05  2417s
  step  12800  loss 2.3641  lr 8.93e-06  2455s
  step  13000  loss 2.3629  lr 7.74e-06  2493s
  step  13200  loss 2.3617  lr 6.63e-06  2531s
  step  13400  loss 2.3606  lr 5.60e-06  2569s
  step  13600  loss 2.3596  lr 4.66e-06  2605s
  step  13800  loss 2.3585  lr 3.79e-06  2643s
  step  14000  loss 2.3576  lr 3.02e-06  2680s
  step  14200  loss 2.3567  lr 2.32e-06  2718s
  step  14400  loss 2.3560  lr 1.72e-06  2755s
  step  14600  loss 2.3550  lr 1.21e-06  2793s
  step  14800  loss 2.3540  lr 7.83e-07  2831s
  step  15000  loss 2.3533  lr 4.50e-07  2869s
  step  15200  loss 2.3525  lr 2.08e-07  2907s
  step  15400  loss 2.3517  lr 5.84e-08  2945s
  step  15600  loss 2.3508  lr 7.21e-10  2983s


  cd /home/rizos/Downloads/SalmonTTS2/wegorz_translator

/home/rizos/Miniforge3/envs/uvtts2/bin/python train.py \
  --model dualpathv3 \
  --config base \
  --train data/coverage_balanced_5m_norm.jsonl \
  --valid data/valid_mini_norm.jsonl \
  --tokenizer tokenizer_norm_full/wegorz.model \
  --batch 64 \
  --epochs 10 \
  --lr 5e-5 \
  --warmup 1000 \
  --cap-encode \
  --resume runs/test_v3_base_32k_cov5m_500k_ep1_from1915k/best.pt \
  --reset-optimizer \
  --save-steps 5000 \
  --out runs/wegorz_dualpath_V3_32k_cov5m_from_base500k_ep4

'''