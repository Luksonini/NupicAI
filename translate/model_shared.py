"""
Dzielone komponenty architektoniczne dla wszystkich modeli NAR/AR.

Zawiera:
  SwiGLU            — FFN z SiLU gating (LLaMA-style)
  alibi_bias        — ALiBi bias dla self-attention
  alibi_cross_bias  — ALiBi bias dla cross-attention
  NATDecoderLayer   — blok dekodera NAR (self-attn + cross-attn)
  LengthPredictor   — predyktor długości wyjścia z mean-pool encodera
  _sinusoidal_pos_enc — standardowe encodowanie pozycji sinusoidalne
  _sinusoidal_t     — embeddingi czasu dla flow/diffusion
  _sample_t         — próbkowanie czasu t ∈ (0, 1) dla flow matching
  _dag_soft_logit_forward — miękki DAG forward DP + soft alignment
  _viterbi_align    — Viterbi z backtraceiem pozycji źródłowych
"""
from __future__ import annotations

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ── SwiGLU FFN ────────────────────────────────────────────────────────────────

class SwiGLU(nn.Module):
    """SwiGLU feed-forward block (Noam Shazeer 2020 / LLaMA-style)."""

    def __init__(self, dim: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__()
        self.w1   = nn.Linear(dim, ffn_dim, bias=False)
        self.w2   = nn.Linear(ffn_dim, dim, bias=False)
        self.gate = nn.Linear(dim, ffn_dim, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        return self.drop(self.w2(F.silu(self.w1(x)) * self.gate(x)))


# ── ALiBi positional biases ───────────────────────────────────────────────────

def alibi_bias(T: int, n_heads: int, device, dtype) -> Tensor:
    """ALiBi bias dla self-attention: [1, n_heads, T, T]."""
    m = torch.tensor(
        [2.0 ** (-8.0 * i / n_heads) for i in range(1, n_heads + 1)],
        device=device, dtype=torch.float32,
    )
    rng  = torch.arange(T, device=device, dtype=torch.float32)
    dist = (rng.unsqueeze(0) - rng.unsqueeze(1)).abs()              # [T, T]
    bias = -(m[:, None, None] * dist[None])                          # [n_heads, T, T]
    return bias.unsqueeze(0).to(dtype)                               # [1, n_heads, T, T]


def alibi_cross_bias(T_q: int, T_k: int, n_heads: int, device, dtype) -> Tensor:
    """ALiBi bias dla cross-attention: [1, n_heads, T_q, T_k]."""
    m = torch.tensor(
        [2.0 ** (-8.0 * i / n_heads) for i in range(1, n_heads + 1)],
        device=device, dtype=torch.float32,
    )
    q_pos = torch.arange(T_q, device=device, dtype=torch.float32)
    k_pos = torch.arange(T_k, device=device, dtype=torch.float32)
    q_scaled = q_pos * ((T_k - 1) / max(T_q - 1, 1))               # skaluj do zakresu k
    dist = (q_scaled.unsqueeze(1) - k_pos.unsqueeze(0)).abs()       # [T_q, T_k]
    bias = -(m[:, None, None] * dist[None])                          # [n_heads, T_q, T_k]
    return bias.unsqueeze(0).to(dtype)                               # [1, n_heads, T_q, T_k]


# ── Pozycjonowanie i czas ─────────────────────────────────────────────────────

def _sinusoidal_pos_enc(T: int, dim: int, device, dtype) -> Tensor:
    """Sinusoidalne encodowanie pozycji: [1, T, dim]."""
    pos   = torch.arange(T, device=device, dtype=torch.float32).unsqueeze(1)
    i     = torch.arange(0, dim, 2, device=device, dtype=torch.float32)
    denom = 10000.0 ** (i / dim)
    pe    = torch.zeros(1, T, dim, device=device, dtype=torch.float32)
    pe[0, :, 0::2] = torch.sin(pos / denom)
    pe[0, :, 1::2] = torch.cos(pos / (denom[:dim // 2] if dim % 2 else denom))
    return pe.to(dtype)


def _sinusoidal_t(t: Tensor, dim: int) -> Tensor:
    """Sinusoidalne embeddingi czasu: t [B] → [B, dim]."""
    half  = dim // 2
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(half, device=t.device, dtype=t.dtype) / half
    )
    args  = t[:, None] * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


def _sample_t(B: int, device, dtype) -> Tensor:
    """Próbkuj B wartości czasu z Uniform(0, 1)."""
    return torch.rand(B, device=device, dtype=dtype)


# ── NAT Decoder Layer ─────────────────────────────────────────────────────────

class NATDecoderLayer(nn.Module):
    """
    Warstwa dekodera dla NAR — non-causal self-attn + cross-attn + SwiGLU FFN.
    Brak maski kauzalnej — cała sekwencja widzi siebie nawzajem.
    """

    def __init__(self, dim: int, n_heads: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__()
        self.norm1      = nn.LayerNorm(dim)
        self.self_attn  = nn.MultiheadAttention(
            dim, n_heads, dropout=dropout, batch_first=True
        )
        self.norm2      = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(
            dim, n_heads, dropout=dropout, batch_first=True
        )
        self.norm3 = nn.LayerNorm(dim)
        self.ffn   = SwiGLU(dim, ffn_dim, dropout)
        self.drop  = nn.Dropout(dropout)

    def forward(
        self, x: Tensor, memory: Tensor,
        enc_key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        h = self.norm1(x)
        h, _ = self.self_attn(h, h, h, need_weights=False)
        x = x + self.drop(h)

        h = self.norm2(x)
        h, _ = self.cross_attn(
            h, memory, memory,
            key_padding_mask=enc_key_padding_mask,
            need_weights=False,
        )
        x = x + self.drop(h)

        x = x + self.drop(self.ffn(self.norm3(x)))
        return x


# ── Length Predictor ──────────────────────────────────────────────────────────

class LengthPredictor(nn.Module):
    """
    Predyktor długości wyjścia.
    Przewiduje offset = T_tgt - T_src z mean-pool encodera.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.proj = nn.Linear(dim, 1)

    def _mean_pool(self, enc_src: Tensor, enc_pad: Tensor) -> Tensor:
        mask   = (~enc_pad).float()                                  # [B, N]
        counts = mask.sum(1, keepdim=True).clamp_min(1.0)           # [B, 1]
        pooled = (enc_src * mask.unsqueeze(-1)).sum(1) / counts      # [B, D]
        return self.proj(pooled).squeeze(-1)                         # [B]

    def forward(self, enc_src: Tensor, enc_pad: Tensor) -> Tensor:
        """Zwraca przewidywany offset [B] (tgt_len - src_len)."""
        return self._mean_pool(enc_src, enc_pad)

    def predict_length(
        self, enc_src: Tensor, src_lens: Tensor,
        enc_pad: Tensor, min_len: int = 1, max_len: int = 512,
    ) -> Tensor:
        """Zwraca przewidywaną długość celu [B] (całkowita, nie offset)."""
        offset = self._mean_pool(enc_src, enc_pad).round().long()
        return (src_lens.long() + offset).clamp(min_len, max_len)


# ── DAG Soft Logit Forward ────────────────────────────────────────────────────

def _dag_soft_logit_forward(
    logits: Tensor, log_q: Tensor, tgt: Tensor, pad_id: int = 0,
) -> Tuple[Tensor, Tensor]:
    """
    Miękki DAG forward DP.

    Dla każdej pary (source_pos, target_pos) oblicza miękkie prawdopodobieństwo
    alignmentu r[b, i, j] = softmax_i(alpha[b,i,j] + log_q_tgt[b,i,j]).
    Następnie: x_0_logits[b, j] = Σ_i r[b,i,j] * logits[b, i]  (ważona suma logitów).

    logits : [B, N, V] — surowe logity
    log_q  : [B, N, V] — log_softmax(logits)
    tgt    : [B, M]    — tokeny docelowe

    Zwraca: (L_dag, x_0_logits [B, M, V])
    """
    B, N, V = log_q.shape
    M       = tgt.shape[1]
    device, dtype = log_q.device, log_q.dtype

    tgt_lens  = (tgt != pad_id).sum(1).long()
    tgt_clamp = tgt.clamp(0, V - 1)

    log_q_tgt = log_q.gather(
        2, tgt_clamp.unsqueeze(1).expand(B, N, M)
    )
    pad_mask  = (tgt == pad_id).unsqueeze(1).expand(B, N, M)
    log_q_tgt = log_q_tgt.masked_fill(pad_mask, float('-inf'))

    # Forward DP + zbieramy alpha[b, i, j] = dp[b, j] PRZED krokiem i
    alpha = torch.full((B, N, M), float('-inf'), device=device, dtype=dtype)
    dp    = torch.full((B, M + 1), float('-inf'), device=device, dtype=dtype)
    dp[:, 0] = 0.0

    for i in range(N):
        alpha[:, i, :] = dp[:, :M]                                  # stan przed krokiem i
        use    = dp[:, :M] + log_q_tgt[:, i, :]
        dp_new = dp.clone()
        dp_new[:, 1:] = torch.logaddexp(dp[:, 1:], use)
        dp = dp_new

    # DAG loss
    loss_vals = -dp[torch.arange(B, device=device), tgt_lens]
    valid     = (tgt_lens > 0).float()
    L_dag     = (loss_vals * valid).sum() / valid.sum().clamp_min(1)

    # Miękkie przypisanie: r[b, i, j] = softmax_i(alpha + log_q_tgt)
    log_r = alpha + log_q_tgt
    r     = torch.softmax(log_r, dim=1)
    r     = torch.nan_to_num(r, nan=0.0)                             # all-inf pad → 0

    # x_0_logits[b, j, v] = Σ_i r[b,i,j] * logits[b,i,v]
    x_0_logits = torch.bmm(r.permute(0, 2, 1), logits)              # [B, M, V]

    return L_dag, x_0_logits


# ── Viterbi alignment ─────────────────────────────────────────────────────────

def _viterbi_align(log_q: Tensor, T_tgt: int) -> List[int]:
    """
    Viterbi na DAG — zwraca listę T_tgt pozycji źródłowych (indeksów).

    log_q : [N, V] — log prawdopodobieństwa dla jednego przykładu
    """
    N, _V = log_q.shape
    M     = T_tgt
    if M == 0 or N == 0:
        return []

    best_lq, _ = log_q.max(dim=1)                                   # [N]
    vit = torch.full((M + 1,), float('-inf'), device=log_q.device, dtype=log_q.dtype)
    vit[0] = 0.0
    bp  = torch.zeros(N, M + 1, dtype=torch.bool, device=log_q.device)

    for i in range(N):
        use_scores = vit[:M] + best_lq[i]
        use_better = use_scores > vit[1:]
        bp[i, 1:]  = use_better
        vit_new    = vit.clone()
        vit_new[1:] = torch.where(use_better, use_scores, vit[1:])
        vit = vit_new

    positions: List[int] = []
    j = M
    for i in range(N - 1, -1, -1):
        if j > 0 and bp[i, j]:
            positions.append(i)
            j -= 1
        if j == 0:
            break
    positions.reverse()
    return positions
