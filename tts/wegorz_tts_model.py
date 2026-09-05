#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Wegorz projected-DualPath + Gaussian flow attention with learned per-speaker voice tables.

This variant intentionally removes zero-shot/reference-encoder conditioning from the
main path: speaker/style conditioning comes from trainable nn.Embedding tables keyed
by dataset speaker_id. WavLM/verifier losses remain optional for final finetuning.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
import time
import zlib
from collections import Counter, defaultdict
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Tuple, List, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.functional as TAF_AUDIO
try:
    from online_ctc import build_asr_student_v2 as _online_ctc_build_asr_student_v2
except ImportError:
    _online_ctc_build_asr_student_v2 = None  # type: ignore[assignment]

# Orange WavLM-TBR is optional and loaded only when verifier loss is enabled.

# ---- Compatibility shim for Speakder dualhead ckpt loading (torch>=2.6) ----
# Speakder trainer checkpoints may pickle small dataclass configs under __main__ (e.g. SupConConfig).
# When we load them from this script (also __main__), the symbol must exist for safe/unsafe loads.
@dataclass(frozen=True)
class SupConConfig:
    tau: float = 0.07

# ---------------- PATH SETUP ----------------
_THIS_DIR = Path(__file__).resolve().parent
_SRC_DIR = _THIS_DIR.parent
_PO_ROOT = next((p for p in (_THIS_DIR, *_THIS_DIR.parents) if (p / "utils.py").is_file()), _THIS_DIR.parents[1])
for _p in (str(_THIS_DIR), str(_SRC_DIR)):
    while _p in sys.path:
        sys.path.remove(_p)
    sys.path.insert(0, _p)
if str(_PO_ROOT) in sys.path:
    sys.path.remove(str(_PO_ROOT))
sys.path.append(str(_PO_ROOT))

if _online_ctc_build_asr_student_v2 is None:
    try:
        from online_ctc import build_asr_student_v2 as _online_ctc_build_asr_student_v2
    except ImportError:
        pass

# ---------------- PROJECT IMPORTS ----------------
from utils_bilingual_bridge import (
    AlignedTTSDataset,
    N_MELS,
    PAD_ID,
    SYMBOL2ID,
    ID2SYMBOL,
    PROSODY_IDS,
    ROLE_IDS,
    collate_fn,
    PLTokenizer,
)
from tts_helpers import GenericTTS, SpeakerAdapter
from backbone_tts import TimeAdaptiveLayerNorm
from dualpath_projected_block import DualPathProjectedBlock
from boundary_jitter_dataset import BoundaryJitterDataset

EMOTION_GROUP_TO_ID = {
    "neutral": 0,
    "happy": 1,
    "calm": 2,
    "sad": 3,
    "angry": 4,
    "fearful": 5,
    "surprised": 6,
    "cute": 7,
    "embarrassed": 8,
}
EMOTION_ID_TO_GROUP = {v: k for k, v in EMOTION_GROUP_TO_ID.items()}


def _emotion_id_from_item(item: Dict[str, Any]) -> int:
    group = str(item.get("emotion_group", item.get("emotion", "neutral")) or "neutral").strip().lower()
    return int(EMOTION_GROUP_TO_ID.get(group, 0))
from tts_helpers import (
    load_checkpoint,
    maybe_load_vocos,
    TEST_SENTENCES_PL,
    LONG_DEMO_CHUNKS_PL,
    NEWS_DEMO_CHUNKS_PL,
    EXTRA_DEMO_SENTENCES_PL,
    EXTRA_LONG_DEMO_CHUNKS_PL,
    EXTRA_EXTREME_LONG_DEMO_CHUNKS_PL,
)
from inference_helpers import (
    collect_reference_audio_paths,
    decode_and_save_mel,
    decode_chunks_and_save,
    load_ref_mel_pt as _demo_load_ref_mel_pt,
    save_wav,
    wav_to_vocos_mel,
)

STYLE_ENCODER128_DIR = _THIS_DIR / "style_encoder_128"
if STYLE_ENCODER128_DIR.is_dir() and str(STYLE_ENCODER128_DIR) not in sys.path:
    sys.path.insert(0, str(STYLE_ENCODER128_DIR))
try:
    from train_style_encoder_128 import WęgorzStyleEncoder128 as _WegorzStyleEncoder128  # type: ignore
except Exception:
    _WegorzStyleEncoder128 = None  # type: ignore[assignment]

_SECS_PER_FRAME = 0.010666666666666666
# Simpler + safer than estimating from dur tokens:
# always clamp+trim a small, fixed prefix window when we have a previous chunk.
_SECS_PER_FRAME_DEFAULT = 256.0 / 24000.0  # hop=256 @ 24kHz => ~10.6667ms
_SHORT_CONTINUITY_MS_DEFAULT = 0.0
_ACOUSTIC_PROMPT_MS_DEFAULT = 1000.0
_PREFIX_FIXED_MS_DEFAULT = _SHORT_CONTINUITY_MS_DEFAULT  # legacy alias; kept for old helpers/checkpoints


class FrozenStyleEncoder128ForTTS(nn.Module):
    """Frozen style_encoder128 plus a trainable projection into the old TTS spk_256 space."""

    def __init__(self, encoder: nn.Module, *, spk_out_dim: int = 256, train_encoder: bool = False):
        super().__init__()
        self.encoder = encoder
        self.spk_proj = nn.Linear(128, int(spk_out_dim))
        nn.init.xavier_uniform_(self.spk_proj.weight, gain=0.25)
        nn.init.zeros_(self.spk_proj.bias)
        for p in self.encoder.parameters():
            p.requires_grad_(bool(train_encoder))
        if bool(train_encoder):
            self.encoder.train()
        else:
            self.encoder.eval()

    def forward(self, mel_bct: torch.Tensor, *, mask_bt: Optional[torch.Tensor] = None) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.set_grad_enabled(any(bool(p.requires_grad) for p in self.encoder.parameters())):
            z_spk128, z_style128 = self.encoder(mel_bct, mask_bt=mask_bt)  # type: ignore[misc,call-arg]
        z_spk = self.spk_proj(z_spk128.float())
        z_spk = F.normalize(z_spk, dim=-1)
        return z_spk.to(dtype=mel_bct.dtype), z_style128.to(dtype=mel_bct.dtype)


class ARDurationTransformer(nn.Module):
    """
    Autoregressive duration predictor — LSTM over text encoder states + previous duration.

    Predicts log(duration + 1) with teacher forcing during training,
    greedy autoregressive decoding during inference. LSTM avoids the
    causal-mask NaN issues of transformer-based variants and converges
    much faster from random init.
    """

    def __init__(
        self,
        dim: int,
        *,
        layers: int = 2,
        heads: int = 4,       # unused, kept for CLI compat
        ffn_mult: int = 4,    # unused, kept for CLI compat
        dropout: float = 0.1,
        max_delta: float = 6.0,
        style_dim: int = 128,
    ):
        super().__init__()
        dim = int(dim)
        hidden = dim
        layers = max(1, int(layers))
        self.dim = dim
        self.layers = int(layers)
        self.hidden = int(hidden)
        self.max_delta = float(max_delta)
        # project text encoder state + scalar prev-logdur → LSTM input
        self.in_proj  = nn.Linear(dim + 1, hidden)
        self.in_norm  = nn.LayerNorm(hidden)
        self.lstm     = nn.LSTM(hidden, hidden, num_layers=self.layers,
                                batch_first=True, dropout=float(dropout) if layers > 1 else 0.0)
        self.out_norm = nn.LayerNorm(hidden)
        self.head     = nn.Linear(hidden, 1)
        self.style_proj = nn.Linear(int(style_dim), hidden) if int(style_dim) > 0 else None
        self.style_gate = nn.Parameter(torch.tensor(0.01))
        if self.style_proj is not None:
            nn.init.xavier_uniform_(self.style_proj.weight, gain=0.01)
            nn.init.zeros_(self.style_proj.bias)
        self.style_to_h0 = nn.Linear(int(style_dim), self.layers * hidden) if int(style_dim) > 0 else None
        self.style_to_c0 = nn.Linear(int(style_dim), self.layers * hidden) if int(style_dim) > 0 else None
        if self.style_to_h0 is not None:
            nn.init.xavier_uniform_(self.style_to_h0.weight, gain=0.02)
            nn.init.zeros_(self.style_to_h0.bias)
        if self.style_to_c0 is not None:
            nn.init.xavier_uniform_(self.style_to_c0.weight, gain=0.02)
            nn.init.zeros_(self.style_to_c0.bias)

    def _add_style(self, inp: torch.Tensor, style_vec: Optional[torch.Tensor]) -> torch.Tensor:
        if self.style_proj is None or style_vec is None:
            return inp
        s = self.style_proj(style_vec.to(device=inp.device, dtype=inp.dtype))[:, None, :]
        return inp + self.style_gate.to(dtype=inp.dtype) * s

    def _style_initial_hc(
        self,
        style_vec: Optional[torch.Tensor],
        *,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
        initial_hc: "tuple | None" = None,
    ) -> "tuple | None":
        if initial_hc is not None:
            return initial_hc
        if style_vec is None or self.style_to_h0 is None or self.style_to_c0 is None:
            return None
        style = style_vec.to(device=device, dtype=dtype)
        h0 = torch.tanh(self.style_to_h0(style)).view(batch_size, self.layers, self.hidden).permute(1, 0, 2).contiguous()
        c0 = torch.tanh(self.style_to_c0(style)).view(batch_size, self.layers, self.hidden).permute(1, 0, 2).contiguous()
        return h0, c0

    # ------------------------------------------------------------------
    def _forward_teacher(
        self,
        x_tok: torch.Tensor,
        target_log: torch.Tensor,
        x_mask: torch.Tensor,
        style_vec: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Teacher-forced forward pass. Returns logdur [B, L]."""
        B, L, _ = x_tok.shape
        prev = torch.cat([target_log.new_zeros((B, 1)), target_log[:, :-1]], dim=1)  # [B, L]
        inp  = self.in_norm(self.in_proj(
            torch.cat([x_tok.float(), prev.unsqueeze(-1).float()], dim=-1)
        ))                                                                             # [B, L, hidden]
        inp = self._add_style(inp, style_vec)
        h_c0 = self._style_initial_hc(style_vec, batch_size=B, device=inp.device, dtype=inp.dtype)
        y, _ = self.lstm(inp, h_c0)                                                   # [B, L, hidden]
        out  = self.head(self.out_norm(y)).squeeze(-1)                                # [B, L]
        return out.clamp(min=-self.max_delta, max=self.max_delta)

    # ------------------------------------------------------------------
    def loss(
        self, x_tok: torch.Tensor, x_mask: torch.Tensor, target_log: torch.Tensor,
        *, kind: str = "huber", style_vec: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        pred  = self._forward_teacher(x_tok, target_log, x_mask, style_vec=style_vec)
        mask  = x_mask.squeeze(1).float()
        denom = mask.sum().clamp_min(1.0)
        if kind == "l1":
            return (torch.abs(pred - target_log.float()) * mask).sum() / denom
        if kind == "mse":
            return (((pred - target_log.float()) ** 2) * mask).sum() / denom
        return F.smooth_l1_loss(
            pred * mask, target_log.float() * mask, reduction="sum", beta=0.5
        ) / denom

    # ------------------------------------------------------------------
    @torch.no_grad()
    def predict_logdur(
        self,
        x_tok: torch.Tensor,
        x_mask: torch.Tensor,
        initial_hc: "tuple | None" = None,
        style_vec: Optional[torch.Tensor] = None,
    ) -> "tuple[torch.Tensor, tuple | None]":
        """Greedy AR inference — one step at a time through the LSTM.

        Returns (logdur [B,L], final_hc) so callers in stateful mode can
        pass final_hc back as initial_hc for the next chunk, preserving
        the rhythm context across sentence boundaries.
        """
        B, L, _ = x_tok.shape
        pred  = x_tok.new_zeros((B, L), dtype=torch.float32)
        valid = x_mask.squeeze(1).to(dtype=torch.bool, device=x_tok.device)
        h_c   = self._style_initial_hc(style_vec, batch_size=B, device=x_tok.device, dtype=x_tok.dtype, initial_hc=initial_hc)
        for i in range(L):
            prev_val = pred[:, i - 1] if i > 0 else pred.new_zeros(B)
            inp_i    = self.in_norm(self.in_proj(
                torch.cat([x_tok[:, i].float(), prev_val.unsqueeze(-1)], dim=-1)
            )).unsqueeze(1)                                          # [B, 1, hidden]
            inp_i = self._add_style(inp_i, style_vec)
            y_i, h_c = self.lstm(inp_i, h_c)                        # [B, 1, hidden]
            val = self.head(self.out_norm(y_i.squeeze(1))).squeeze(-1)  # [B]
            val = val.clamp(min=-self.max_delta, max=self.max_delta)
            pred[:, i] = torch.where(valid[:, i], val, pred.new_zeros(B))
        return pred, h_c


def _masked_duration_loss(pred: torch.Tensor, target_log: torch.Tensor, x_mask: torch.Tensor, *, kind: str) -> torch.Tensor:
    mask = x_mask.squeeze(1).float()
    denom = mask.sum().clamp_min(1.0)
    if kind == "l1":
        return (torch.abs(pred - target_log.float()) * mask).sum() / denom
    if kind == "mse":
        return (((pred - target_log.float()) ** 2) * mask).sum() / denom
    return F.smooth_l1_loss(pred * mask, target_log.float() * mask, reduction="sum", beta=0.5) / denom


class MiniDualPathDurationBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        *,
        attn_dim: int = 128,
        conv_dim: int = 128,
        heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.attn_in = nn.Linear(int(dim), int(attn_dim))
        self.conv_in = nn.Linear(int(dim), int(conv_dim))
        self.attn_norm = nn.LayerNorm(int(attn_dim))
        self.conv_norm = nn.LayerNorm(int(conv_dim))
        self.attn = nn.MultiheadAttention(int(attn_dim), int(heads), dropout=float(dropout), batch_first=True)
        self.convs = nn.ModuleList(
            [nn.Conv1d(int(conv_dim), int(conv_dim), k, padding=k // 2, groups=1) for k in (3, 9, 15)]
        )
        self.conv_gate = nn.Linear(int(conv_dim), 3)
        self.merge = nn.Linear(int(attn_dim) + int(conv_dim), int(dim))
        self.out_norm = nn.LayerNorm(int(dim))
        self.drop = nn.Dropout(float(dropout))

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor) -> torch.Tensor:
        residual = x
        valid = x_mask.squeeze(1).bool()
        key_padding_mask = ~valid

        a = self.attn_norm(self.attn_in(x))
        a_upd, _ = self.attn(a, a, a, key_padding_mask=key_padding_mask, need_weights=False)

        c = self.conv_norm(self.conv_in(x))
        c_bct = c.transpose(1, 2).contiguous()
        conv_outs = torch.stack([F.silu(conv(c_bct)).transpose(1, 2) for conv in self.convs], dim=-2)
        gate = torch.softmax(self.conv_gate(c), dim=-1).unsqueeze(-1)
        c_upd = (conv_outs * gate).sum(dim=-2)

        y = self.merge(torch.cat([a_upd, c_upd], dim=-1))
        y = self.drop(y)
        y = torch.where(valid.unsqueeze(-1), y, torch.zeros_like(y))
        return self.out_norm(residual + y)


class MiniDualPathDurationPredictor(nn.Module):
    """Non-AR duration predictor: projected DualPath stack -> direct log-duration."""

    def __init__(
        self,
        dim: int,
        *,
        layers: int = 3,
        attn_dim: int = 128,
        conv_dim: int = 128,
        heads: int = 4,
        dropout: float = 0.1,
        max_delta: float = 6.0,
        style_dim: int = 128,
    ):
        super().__init__()
        self.max_delta = float(max_delta)
        self.style_proj = nn.Linear(int(style_dim), int(dim)) if int(style_dim) > 0 else None
        self.style_gate = nn.Parameter(torch.tensor(0.01))
        if self.style_proj is not None:
            nn.init.xavier_uniform_(self.style_proj.weight, gain=0.01)
            nn.init.zeros_(self.style_proj.bias)
        self.blocks = nn.ModuleList(
            [
                MiniDualPathDurationBlock(
                    int(dim),
                    attn_dim=int(attn_dim),
                    conv_dim=int(conv_dim),
                    heads=int(heads),
                    dropout=float(dropout),
                )
                for _ in range(int(layers))
            ]
        )
        self.out_norm = nn.LayerNorm(int(dim))
        self.head = nn.Linear(int(dim), 1)

    def _encode(self, x_tok: torch.Tensor, x_mask: torch.Tensor, style_vec: Optional[torch.Tensor]) -> torch.Tensor:
        x = x_tok.float()
        if self.style_proj is not None and style_vec is not None:
            s = self.style_proj(style_vec.to(device=x.device, dtype=x.dtype))[:, None, :]
            x = x + self.style_gate.to(dtype=x.dtype) * s
        x = x * x_mask.transpose(1, 2).float()
        for blk in self.blocks:
            x = blk(x, x_mask)
        return self.out_norm(x)

    def loss(
        self,
        x_tok: torch.Tensor,
        x_mask: torch.Tensor,
        target_log: torch.Tensor,
        *,
        kind: str = "huber",
        style_vec: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        pred, _ = self.predict_logdur(x_tok, x_mask, style_vec=style_vec)
        return _masked_duration_loss(pred, target_log, x_mask, kind=kind)

    def predict_logdur(
        self,
        x_tok: torch.Tensor,
        x_mask: torch.Tensor,
        initial_hc: "tuple | None" = None,
        style_vec: Optional[torch.Tensor] = None,
    ) -> "tuple[torch.Tensor, None]":
        del initial_hc
        x = self._encode(x_tok, x_mask, style_vec)
        pred = self.head(x).squeeze(-1).clamp(min=-self.max_delta, max=self.max_delta)
        pred = pred * x_mask.squeeze(1).float()
        return pred, None


class MiniDualPathBinsMaskGITDurationPredictor(nn.Module):
    """Total-duration-aware categorical MiniDualPath with parallel MaskGIT decoding."""

    is_tda_maskgit = True

    def __init__(
        self,
        dim: int,
        *,
        layers: int = 3,
        attn_dim: int = 128,
        conv_dim: int = 128,
        heads: int = 4,
        dropout: float = 0.1,
        style_dim: int = 128,
        max_dur_bins: int = 100,
        iterations: int = 8,
        mask_ratio_min: float = 0.30,
        mask_ratio_max: float = 1.0,
        all_mask_prob: float = 0.20,
        budget_loss_weight: float = 0.20,
        total_head_weight: float = 0.05,
        budget_mode: str = "predicted",
        mask_schedule: str = "cosine",
        min_token_frames: int = 1,
        min_pause_frames: int = 1,
        rhythm_dim: int = 6,
        rhythm_gate_init: float = 0.02,
    ):
        super().__init__()
        self.max_bin = int(max_dur_bins)
        self.mask_class = self.max_bin + 1
        self.iterations = int(iterations)
        self.mask_ratio_min = float(mask_ratio_min)
        self.mask_ratio_max = float(mask_ratio_max)
        self.all_mask_prob = float(all_mask_prob)
        self.budget_loss_weight = float(budget_loss_weight)
        self.total_head_weight = float(total_head_weight)
        self.budget_mode = str(budget_mode).lower().strip()
        self.mask_schedule = str(mask_schedule).lower().strip()
        self.min_token_frames = max(0, int(min_token_frames))
        self.min_pause_frames = max(0, int(min_pause_frames))

        self.style_proj = nn.Linear(int(style_dim), int(dim)) if int(style_dim) > 0 else None
        self.style_gate = nn.Parameter(torch.tensor(0.01))
        if self.style_proj is not None:
            nn.init.xavier_uniform_(self.style_proj.weight, gain=0.01)
            nn.init.zeros_(self.style_proj.bias)
        self.rhythm_norm = nn.LayerNorm(int(rhythm_dim))
        self.rhythm_mlp = nn.Sequential(
            nn.Linear(int(rhythm_dim), 64),
            nn.SiLU(),
            nn.Linear(64, int(dim)),
        )
        self.rhythm_gate = nn.Parameter(torch.tensor(float(rhythm_gate_init)))
        self.duration_embed = nn.Embedding(self.max_bin + 2, int(dim))
        self.budget_mlp = nn.Sequential(nn.Linear(2, int(dim)), nn.SiLU(), nn.Linear(int(dim), int(dim)))
        self.blocks = nn.ModuleList([
            MiniDualPathDurationBlock(
                int(dim), attn_dim=int(attn_dim), conv_dim=int(conv_dim),
                heads=int(heads), dropout=float(dropout),
            )
            for _ in range(int(layers))
        ])
        self.out_norm = nn.LayerNorm(int(dim))
        self.head = nn.Linear(int(dim), self.max_bin + 1)
        half = max(32, int(dim) // 2)
        self.total_head = nn.Sequential(nn.Linear(int(dim), half), nn.SiLU(), nn.Linear(half, 1))
        self.last_infer_stats: Dict[str, float] = {}

    def _base(
        self,
        x_tok: torch.Tensor,
        x_mask: torch.Tensor,
        style_vec: Optional[torch.Tensor],
        rhythm_state: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = x_tok.float() * x_mask.transpose(1, 2).float()
        if self.style_proj is not None and style_vec is not None:
            style = self.style_proj(style_vec.to(device=x.device, dtype=x.dtype))[:, None]
            x = x + self.style_gate.to(dtype=x.dtype) * style
        if rhythm_state is not None:
            rhythm = rhythm_state.to(device=x.device, dtype=x.dtype)
            available = rhythm[:, -1:].clamp(0.0, 1.0)
            rhythm = (self.rhythm_mlp(self.rhythm_norm(rhythm)) * available)[:, None]
            x = x + self.rhythm_gate.to(dtype=x.dtype) * rhythm
        return x

    def _predict_total(self, base: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        per_token = F.softplus(self.total_head(base).squeeze(-1)) * valid.float()
        total = per_token.sum(1).clamp_min(1.0)
        minimum = valid.sum(1).float()
        maximum = minimum * float(self.max_bin)
        return torch.maximum(total, minimum).minimum(maximum.clamp_min(minimum))

    def _logits(
        self,
        base: torch.Tensor,
        x_mask: torch.Tensor,
        observed: torch.Tensor,
        total: torch.Tensor,
    ) -> torch.Tensor:
        valid = x_mask.squeeze(1).bool()
        known = valid & (observed != self.mask_class)
        known_sum = torch.where(known, observed.float(), torch.zeros_like(observed, dtype=torch.float32)).sum(1)
        remaining = (total.float() - known_sum).clamp_min(0.0)
        budget = torch.stack([torch.log1p(total.float()), torch.log1p(remaining)], dim=-1)
        h = base + self.duration_embed(observed.clamp(0, self.mask_class))
        h = h + self.budget_mlp(budget)[:, None]
        for block in self.blocks:
            h = block(h, x_mask)
        return self.head(self.out_norm(h))

    @staticmethod
    def _largest_remainder(values: torch.Tensor, total: int, floors: torch.Tensor) -> torch.Tensor:
        values = values.float().clamp_min(0.0)
        floors = floors.long().clamp_min(0)
        total = max(int(total), int(floors.sum().item()))
        available = int(total - int(floors.sum().item()))
        if available <= 0:
            return floors
        weights = (values - floors.float()).clamp_min(0.0)
        if float(weights.sum().item()) <= 1e-8:
            weights = torch.ones_like(values)
        scaled_extra = weights * (float(available) / weights.sum().clamp_min(1e-8))
        extra = torch.floor(scaled_extra).long()
        remainder = int(available - int(extra.sum().item()))
        if remainder > 0:
            frac = scaled_extra - extra.float()
            chosen = torch.topk(frac, k=min(remainder, int(frac.numel()))).indices
            extra[chosen] += 1
        return floors + extra

    def _normalize_pending(
        self,
        values: torch.Tensor,
        token_ids: Optional[torch.Tensor],
        pending: torch.Tensor,
        remaining_total: int,
    ) -> torch.Tensor:
        indices = torch.nonzero(pending, as_tuple=False).flatten()
        if indices.numel() == 0:
            return values.new_zeros(values.shape, dtype=torch.long)
        floors = torch.full(
            (int(indices.numel()),), self.min_token_frames,
            device=values.device, dtype=torch.long,
        )
        if token_ids is not None:
            is_pause = token_ids[indices] == int(SYMBOL2ID.get("<sp>", -999999))
            floors = torch.where(is_pause, torch.full_like(floors, self.min_pause_frames), floors)
        normalized = self._largest_remainder(values[indices], int(remaining_total), floors)
        out = values.new_zeros(values.shape, dtype=torch.long)
        out[indices] = normalized
        return out

    def _maskgit_take_count(self, pending_count: int, step: int) -> int:
        if step >= self.iterations - 1:
            return int(pending_count)
        if self.mask_schedule != "cosine":
            return max(1, int(math.ceil(float(pending_count) / float(max(1, self.iterations - step)))))
        previous_fraction = 1.0 - math.cos(0.5 * math.pi * float(step) / float(self.iterations))
        next_fraction = 1.0 - math.cos(0.5 * math.pi * float(step + 1) / float(self.iterations))
        conditional_fraction = (next_fraction - previous_fraction) / max(1e-8, 1.0 - previous_fraction)
        return max(1, min(int(pending_count), int(math.ceil(conditional_fraction * pending_count))))

    @staticmethod
    def _fit_budget(
        values: torch.Tensor,
        confidence: torch.Tensor,
        token_ids: Optional[torch.Tensor],
        valid: torch.Tensor,
        total: int,
    ) -> torch.Tensor:
        out = values.round().long().clamp_min(0)
        diff = int(total - int(out[valid].sum()))
        pause_idx = (
            torch.nonzero(valid & (token_ids == int(SYMBOL2ID.get("<sp>", -999999))), as_tuple=False)
            .flatten().tolist()
            if token_ids is not None else []
        )
        fallback = torch.nonzero(valid, as_tuple=False).flatten().tolist()
        if not fallback:
            return out.float()
        if diff > 0:
            targets = pause_idx or sorted(fallback, key=lambda i: float(confidence[i]))
            for step in range(diff):
                out[targets[step % len(targets)]] += 1
        elif diff < 0:
            need = -diff
            order = pause_idx + [i for i in sorted(fallback, key=lambda i: float(confidence[i])) if i not in pause_idx]
            for index in order:
                floor = 0 if index in pause_idx else 1
                take = min(need, max(0, int(out[index]) - floor))
                out[index] -= take
                need -= take
                if need <= 0:
                    break
        return out.float()

    @torch.no_grad()
    def predict_durations(
        self,
        x_tok: torch.Tensor,
        x_mask: torch.Tensor,
        *,
        token_ids: Optional[torch.Tensor] = None,
        style_vec: Optional[torch.Tensor] = None,
        total_hint: Optional[torch.Tensor] = None,
        rhythm_state: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        valid = x_mask.squeeze(1).bool()
        base = self._base(x_tok, x_mask, style_vec, rhythm_state)
        if self.budget_mode == "external" and total_hint is None:
            raise ValueError("duration budget mode 'external' requires total_hint")
        total_float = self._predict_total(base, valid) if total_hint is None else total_hint.float()
        total = total_float.round().long().clamp_min(1)
        minimum_total = valid.sum(1).long() * self.min_token_frames
        if token_ids is not None and self.min_pause_frames != self.min_token_frames:
            pause_count = (valid & (token_ids == int(SYMBOL2ID.get("<sp>", -999999)))).sum(1).long()
            minimum_total += pause_count * int(self.min_pause_frames - self.min_token_frames)
        total = torch.maximum(total, minimum_total.clamp_min(1))
        observed = torch.full(valid.shape, self.mask_class, device=x_tok.device, dtype=torch.long)
        observed[~valid] = 0
        confidence = torch.zeros(valid.shape, device=x_tok.device)
        sampled = torch.zeros(valid.shape, device=x_tok.device, dtype=torch.long)
        for step in range(self.iterations):
            probs = torch.softmax(self._logits(base, x_mask, observed, total).float(), dim=-1)
            conf, pred = probs.max(dim=-1)
            for b in range(int(valid.size(0))):
                pending = torch.nonzero(valid[b] & (observed[b] == self.mask_class), as_tuple=False).flatten()
                if pending.numel() == 0:
                    continue
                sampled[b, pending] = pred[b, pending]
                candidates = pred[b].long()
                if self.budget_mode in {"predicted", "external"}:
                    known = valid[b] & (observed[b] != self.mask_class)
                    remaining_total = max(0, int(total[b]) - int(observed[b, known].sum()))
                    candidates = self._normalize_pending(
                        candidates.float(), token_ids[b] if token_ids is not None else None,
                        valid[b] & (observed[b] == self.mask_class), remaining_total,
                    )
                take = self._maskgit_take_count(int(pending.numel()), step)
                chosen = pending[torch.topk(conf[b, pending], k=min(take, int(pending.numel()))).indices]
                observed[b, chosen] = candidates[chosen]
                confidence[b, chosen] = conf[b, chosen]
        decoded = observed.clamp_min(0).float() * valid.float()
        raw = sampled.clamp(0, self.max_bin).float() * valid.float()
        if self.budget_mode == "legacy":
            decoded = raw.clone()
            for b in range(int(valid.size(0))):
                decoded[b] = self._fit_budget(
                    decoded[b], confidence[b], token_ids[b] if token_ids is not None else None,
                    valid[b], int(total[b]),
                )
        elif self.budget_mode == "natural":
            decoded = raw
        self.last_infer_stats = {
            "pred_total": float(total_float.mean()),
            "raw_ratio": float((raw.sum(1) / total.float().clamp_min(1.0)).mean()),
            "budget_fix": float((decoded - raw).abs().sum(1).mean()),
            "final_ratio": float((decoded.sum(1) / total.float().clamp_min(1.0)).mean()),
        }
        return decoded * valid.float()

    def predict_logdur(
        self,
        x_tok: torch.Tensor,
        x_mask: torch.Tensor,
        initial_hc: "tuple | None" = None,
        style_vec: Optional[torch.Tensor] = None,
        rhythm_state: Optional[torch.Tensor] = None,
        token_ids: Optional[torch.Tensor] = None,
    ) -> "tuple[torch.Tensor, None]":
        del initial_hc
        durations = self.predict_durations(
            x_tok, x_mask, token_ids=token_ids, style_vec=style_vec, rhythm_state=rhythm_state,
        )
        return torch.log1p(durations), None


MiniDualPathBinsGPTDurationPredictor = MiniDualPathBinsMaskGITDurationPredictor


def _ensure_duration_style_adapter_live(module: nn.Module, *, gate_value: float = 0.01, gain: float = 0.01) -> bool:
    proj = getattr(module, "style_proj", None)
    gate = getattr(module, "style_gate", None)
    if proj is None or gate is None or not hasattr(proj, "weight") or not torch.is_tensor(gate):
        return False
    with torch.no_grad():
        zero = torch.zeros((), device=proj.weight.device)
        weight_zero = bool(torch.isclose(proj.weight.detach().abs().max(), zero).item())
        bias = getattr(proj, "bias", None)
        bias_zero = True if bias is None else bool(torch.isclose(bias.detach().abs().max(), torch.zeros((), device=bias.device)).item())
        gate_zero = bool(torch.isclose(gate.detach().abs().max(), torch.zeros((), device=gate.device)).item())
        if not (weight_zero and bias_zero and gate_zero):
            return False
        nn.init.xavier_uniform_(proj.weight, gain=float(gain))
        if bias is not None:
            nn.init.zeros_(bias)
        gate.fill_(float(gate_value))
    return True


def _sync_missing_style_pooling(module: Optional[nn.Module]) -> bool:
    if module is None or not hasattr(module, "pooling") or not hasattr(module, "pooling_style"):
        return False
    try:
        getattr(module, "pooling_style").load_state_dict(getattr(module, "pooling").state_dict(), strict=True)
        return True
    except Exception:
        return False


def _prefix_frames_from_ms(prefix_ms: float, secs_per_frame: float = _SECS_PER_FRAME_DEFAULT) -> int:
    try:
        v = float(prefix_ms)
    except Exception:
        v = float(_PREFIX_FIXED_MS_DEFAULT)
    if v <= 0.0:
        return 0
    spf = float(secs_per_frame) if float(secs_per_frame) > 0 else float(_SECS_PER_FRAME_DEFAULT)
    return max(1, int(round((v / 1000.0) / spf)))


@torch.no_grad()
def sample_mel_flow_with_prefix(
    mel_flow: nn.Module,
    x0_bct: torch.Tensor,
    text_seq_b: Optional[torch.Tensor],
    speaker_ids: torch.Tensor,
    *,
    steps: int,
    spk_vec_override: Optional[torch.Tensor],
    prefix_tail_bct: torch.Tensor,
    prefix_k: int,
) -> torch.Tensor:
    """
    Same Euler sampler as `sample_mel_flow`, but hard-clamps the first K frames
    to `prefix_tail_bct` after each step (and initializes x[:, :, :K] too).
    """
    k = int(prefix_k)
    if k <= 0:
        return sample_mel_flow(
            mel_flow,
            x0_bct,
            text_seq_b,
            speaker_ids,
            steps=int(steps),
            spk_vec_override=spk_vec_override,
        )
    k = int(min(k, int(x0_bct.size(-1)), int(prefix_tail_bct.size(-1))))
    if k <= 0:
        return sample_mel_flow(
            mel_flow,
            x0_bct,
            text_seq_b,
            speaker_ids,
            steps=int(steps),
            spk_vec_override=spk_vec_override,
        )

    x = x0_bct
    # initialize prefix
    x[:, :, :k] = prefix_tail_bct[:, :, :k].to(dtype=x.dtype, device=x.device)
    dt = 1.0 / float(max(1, int(steps)))
    for kk in range(int(steps)):
        t = torch.full((x.size(0),), float(kk) / float(max(1, int(steps))), device=x.device, dtype=x.dtype)
        v = mel_flow(
            x,
            t,
            speaker_ids,
            text_seq_b,
            spk_vec_override=spk_vec_override,
        )
        x = x + v * dt
        x[:, :, :k] = prefix_tail_bct[:, :, :k].to(dtype=x.dtype, device=x.device)
    return x

try:
    from backbone_tts import FlowDurationPredictor  # type: ignore
except Exception as exc:
    raise RuntimeError("Nie mogę zaimportować FlowDurationPredictor z backbone_tts.py") from exc


# ---------------- DEFAULTS ----------------
CONFIG = {
    # duration flow
    "dur_flow_steps": 10,
    "dur_flow_noise_scale": 1,

    # gauss upsampling (prior)
    "gauss_sigma": 1.0,
    "gauss_topk": 2,
    # How to treat pause tokens in gauss upsampling:
    #   - "mix":     allow mixing across pause boundaries (legacy)
    #   - "isolate": never mix pause and non-pause within a frame (reduces "silence leakage")
    "gauss_sp_mode": "mix",

    # mel flow demo steps
    "mel_flow_steps_demo": 8,

    # conditioning
    "dur_flow_cond_mode": "none",       # "none"|"rate"|"len"|"both"
    "dur_flow_cond_source_train": "gt", # "gt" or "pred"
    "dec_dur_source_train": "pred",      # prior w treningu: predicted duracje (jak StyleTTS2)
    "dec_dur_source_demo": "pred",       # w demo: predicted duracje
    "text_cross_attn": True,
    # prefix tokens in encoder: [speaker, gender, global_mem]
    "special_len": 3,
    # context bridge
    "context_attn_heads": 4,
    "context_attn_drop": 0.0,
    "context_mem_dim": 0,  # 0 => use text_dim
}

DEFAULT_BILINGUAL_ASR_CKPT = (
    "/home/rizos/Downloads/SalmonTTS2/test_paraqueet/runs/"
    "asr_nano_4layers_split80_160_max60min/ckpt_best.pt"
)

# Token ids (from vocab used by PLTokenizer in this repo).
SP_ID = int(SYMBOL2ID.get("<sp>", 2))
BOS_ID = int(SYMBOL2ID.get("<BOS>", 4))
EOS_ID = int(SYMBOL2ID.get("<EOS>", 5))
PAUSE_TOKEN_IDS = tuple(sorted({int(SP_ID), int(BOS_ID), int(EOS_ID)}))


def _pause_mask_from_ids(ids: torch.Tensor) -> torch.Tensor:
    mask = torch.zeros_like(ids, dtype=torch.bool)
    for tid in PAUSE_TOKEN_IDS:
        mask |= (ids == int(tid))
    return mask


def _looks_sentence_final(s: str) -> bool:
    return bool(re.search(r'[.!?…]["”’)\]]*\s*$', str(s or "").strip()))


def _ensure_boundary_tokens(s: str, *, continuation_out: bool = False) -> str:
    s = str(s).strip()
    if not s:
        return s
    s = re.sub(r"^(?:\s*<(?:pad|sp|BOS|EOS)>\s*)+", "", s, flags=re.IGNORECASE)
    s = re.sub(r"(?:\s*<(?:pad|sp|BOS|EOS)>\s*)+$", "", s, flags=re.IGNORECASE)
    # Training manifests do not contain explicit <BOS>/<EOS>/<pad> boundary tokens.
    # Keep text-demo/infer tokenization matched to training and rely on the
    # sentence text/punctuation plus stateful duration/mel continuity instead.
    return s


def _ensure_lang_prefix(s: str, default_lang: str = "<pl>") -> str:
    s = str(s or "").strip()
    if not s:
        return s
    if re.match(r"^\s*<(?:pl|en)>\s*", s, flags=re.IGNORECASE):
        return s
    return f"{default_lang} {s}".strip()


def _partial_load(module: nn.Module, state_dict: Dict[str, torch.Tensor], *, ignore_prefixes: tuple[str, ...] = ()) -> List[str]:
    own = module.state_dict()
    filtered: Dict[str, torch.Tensor] = {}
    skipped: List[str] = []
    for key, value in state_dict.items():
        if any(str(key).startswith(pref) for pref in ignore_prefixes):
            skipped.append(str(key))
            continue
        if key in own and tuple(own[key].shape) == tuple(value.shape):
            filtered[key] = value
        else:
            skipped.append(str(key))
    module.load_state_dict(filtered, strict=False)
    return skipped


def _load_embedding_partial(module: nn.Embedding, state_dict: Dict[str, torch.Tensor]) -> bool:
    incoming = state_dict.get("weight", None)
    if not torch.is_tensor(incoming):
        return False
    with torch.no_grad():
        cur = module.weight
        if int(incoming.ndim) != 2 or int(cur.ndim) != 2 or int(incoming.size(1)) != int(cur.size(1)):
            return False
        n = min(int(incoming.size(0)), int(cur.size(0)))
        cur[:n].copy_(incoming[:n].to(device=cur.device, dtype=cur.dtype))
    return tuple(incoming.shape) == tuple(module.weight.shape)


def _grad_group_stats(module: nn.Module) -> Dict[str, Dict[str, float]]:
    groups = {
        "ffn_pre": ("ffn_pre.",),
        "ffn_post": ("ffn_post.",),
        "attn": ("attn.",),
        "convs": ("convs.", "merge_conv."),
        "gate1": ("gate1.",),
        "gate2": ("gate2.",),
        "branch_proj": ("attn_in.", "conv_in.", "branch_merge."),
        "adaln_norm": ("norm", "ada", "dual_adaln", "timbre_adaln"),
    }
    out: Dict[str, Dict[str, float]] = {}
    for group, prefixes in groups.items():
        sq = 0.0
        abs_sum = 0.0
        max_abs = 0.0
        n = 0
        tensors = 0
        for name, p in module.named_parameters():
            if not any(str(name).startswith(pref) or (pref in str(name)) for pref in prefixes):
                continue
            if p.grad is None:
                continue
            g = p.grad.detach().float()
            if g.numel() <= 0:
                continue
            tensors += 1
            n += int(g.numel())
            sq += float(g.pow(2).sum().cpu())
            abs_sum += float(g.abs().sum().cpu())
            max_abs = max(max_abs, float(g.abs().max().cpu()))
        out[group] = {
            "tensors": float(tensors),
            "params": float(n),
            "rms": math.sqrt(sq / max(1, n)),
            "mean_abs": abs_sum / max(1, n),
            "max_abs": max_abs,
        }
    return out


def _print_grad_diagnostics(*, ep: int, batch_i: int, total_loss: torch.Tensor, prior_mu: nn.Module, mel_flow: nn.Module) -> None:
    print(f"🧪 grad-diagnose: ep={int(ep):04d} batch={int(batch_i)} loss={float(total_loss.detach().cpu()):.6f}", flush=True)
    for module_name, module in (("prior_mu", prior_mu), ("mel_flow", mel_flow)):
        stats = _grad_group_stats(module)
        print(f"🧪 grad-diagnose {module_name}:", flush=True)
        for group in ("ffn_pre", "ffn_post", "attn", "convs", "gate1", "gate2", "branch_proj", "adaln_norm"):
            s = stats.get(group, {})
            print(
                "  "
                f"{group:12s} tensors={int(s.get('tensors', 0.0)):3d} "
                f"params={int(s.get('params', 0.0)):9d} "
                f"rms={float(s.get('rms', 0.0)):.6e} "
                f"mean_abs={float(s.get('mean_abs', 0.0)):.6e} "
                f"max_abs={float(s.get('max_abs', 0.0)):.6e}",
                flush=True,
            )


def _build_asr_student_v2(asr_ckpt: str, device: torch.device) -> Tuple[nn.Module, object]:
    if _online_ctc_build_asr_student_v2 is None:
        raise RuntimeError(
            "Nie udało się zaimportować lokalnego online CTC. "
            "Sprawdź katalog test_paraqueet/wegorz_dubbingTTS/src/online_ctc."
        )
    return _online_ctc_build_asr_student_v2(  # type: ignore[misc]
        asr_ckpt,
        device,
        n_mels_default=int(N_MELS),
        vocab_size=int(len(SYMBOL2ID)),
        new_token2id={str(k): int(v) for k, v in dict(SYMBOL2ID).items()},
    )


@torch.no_grad()
def _reconstruct_online_full_durs(
    aligner_mod,
    log_probs_btv: torch.Tensor,
    tok_pad: torch.Tensor,
    tok_lens: torch.Tensor,
    out_lens: torch.Tensor,
    *,
    target_frame_lens: torch.Tensor,
) -> torch.Tensor:
    B, N = int(tok_pad.size(0)), int(tok_pad.size(1))
    out = log_probs_btv.new_zeros((B, N), dtype=torch.float32)
    sp_id = int(SYMBOL2ID.get("<sp>", -999999))
    bos_id = int(SYMBOL2ID.get("<BOS>", -999999))
    eos_id = int(SYMBOL2ID.get("<EOS>", -999999))
    viterbi_impl = "torchaudio" if getattr(aligner_mod, "TAF", None) is not None else "eager"
    fix_sum_fn = getattr(aligner_mod, "_fix_dur_sum", None)
    recon_fn = getattr(aligner_mod, "reconstruct_full_durs_from_blanks", None)
    if recon_fn is None:
        raise RuntimeError("CTC_aligner_v3 is missing reconstruct_full_durs_from_blanks")
    for b in range(B):
        Tb = int(out_lens[b].item())
        Nb = int(tok_lens[b].item())
        Tf = int(target_frame_lens[b].item())
        if Tb <= 0 or Nb <= 0 or Tf <= 0:
            continue
        ids_full = [int(x) for x in tok_pad[b, :Nb].detach().cpu().tolist()]
        punct_mask = [int(t) in PROSODY_IDS for t in ids_full]
        sp_mask = [int(t) == sp_id for t in ids_full]
        no_dur_mask = [int(t) in ROLE_IDS for t in ids_full]
        keep_labels = [
            tid
            for tid, is_p, is_sp, is_nd in zip(ids_full, punct_mask, sp_mask, no_dur_mask)
            if (not is_p) and (not is_sp) and (not is_nd)
        ]
        if len(keep_labels) <= 0:
            continue
        target = torch.tensor(keep_labels, dtype=torch.long, device=log_probs_btv.device)
        dur_tok, dur_blank = aligner_mod.align_to_token_blank_durs_fast(
            log_probs_btv[b, :Tb].detach(),
            target,
            blank_bias=0.0,
            viterbi_impl=viterbi_impl,
        )
        durs = recon_fn(
            ids_full,
            punct_mask,
            sp_mask,
            dur_tok,
            dur_blank,
            no_dur_mask,
            blank_gap_mode="proportional_centered",
            bos_id=(None if bos_id < 0 else bos_id),
            eos_id=(None if eos_id < 0 else eos_id),
        )
        if Tf != Tb:
            scale = float(Tf) / float(max(1, Tb))
            durs = [int(round(float(x) * scale)) for x in durs]
        if callable(fix_sum_fn):
            durs = fix_sum_fn(ids_full, punct_mask, sp_mask, no_dur_mask, durs, Tf)
        out[b, :Nb] = torch.tensor(durs[:Nb], dtype=torch.float32, device=out.device)
    return out


def _build_ctc_targets(tok_pad: torch.Tensor, tok_lens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    parts: List[torch.Tensor] = []
    lens: List[int] = []
    for b in range(int(tok_pad.size(0))):
        Nb = int(tok_lens[b].item())
        ids_full = [int(x) for x in tok_pad[b, :Nb].detach().cpu().tolist()]
        keep_ids = [tid for tid in ids_full if (tid not in PROSODY_IDS) and (tid not in ROLE_IDS) and (tid != int(SYMBOL2ID.get("<sp>", -999999)))]
        lens.append(int(len(keep_ids)))
        if keep_ids:
            parts.append(torch.tensor(keep_ids, dtype=torch.long, device=tok_pad.device))
    flat = torch.cat(parts, dim=0) if parts else tok_pad.new_zeros((0,), dtype=torch.long)
    return flat, torch.tensor(lens, dtype=torch.long, device=tok_pad.device)


def _duration_budget_loss(
    dur_values: torch.Tensor,
    dur_allowed: torch.Tensor,
    target_frames: torch.Tensor,
    *,
    eps: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Global speaking-time budget loss.

    This intentionally operates on total duration only. It teaches the duration
    path to fit a frame budget without dictating where every pause must land.
    """
    if dur_values.numel() <= 0:
        z = dur_values.new_zeros(())
        return z, z
    allowed = dur_allowed.to(device=dur_values.device, dtype=torch.bool)
    pred_total = torch.where(allowed, dur_values.float().clamp_min(0.0), torch.zeros_like(dur_values.float())).sum(dim=1)
    tgt_total = target_frames.to(device=dur_values.device, dtype=torch.float32).view(-1).clamp_min(1.0)
    if int(tgt_total.numel()) != int(pred_total.numel()):
        tgt_total = tgt_total[: int(pred_total.numel())]
    loss = (torch.log(pred_total.clamp_min(0.0) + float(eps)) - torch.log(tgt_total + float(eps))).abs().mean()
    ratio = (pred_total / tgt_total.clamp_min(1.0)).mean()
    return loss, ratio


def _token_duration_floor_loss(
    dur_values: torch.Tensor,
    ids_full: torch.Tensor,
    dur_allowed: torch.Tensor,
    *,
    min_text_frames: float,
) -> torch.Tensor:
    """Soft guard against solving the budget by collapsing real text tokens."""
    min_f = float(min_text_frames)
    if min_f <= 0.0:
        return dur_values.new_zeros(())
    pause_mask = _pause_mask_from_ids(ids_full).to(device=dur_values.device, dtype=torch.bool)
    text_mask = dur_allowed.to(device=dur_values.device, dtype=torch.bool) & (~pause_mask)
    if not bool(text_mask.any().item()):
        return dur_values.new_zeros(())
    return F.relu(float(min_f) - dur_values.float())[text_mask].mean()


def _gender_id_from_name(name: Optional[str]) -> int:
    s = str(name or "").strip()
    if s.endswith("_F"):
        return 1
    if s.endswith("_M"):
        return 2
    return 0


# ---------------- SMALL HELPERS ----------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _ensure_mel_bct(mel: torch.Tensor) -> torch.Tensor:
    """Akceptuje [C,T], [T,C], [B,T,C] lub [B,C,T], zwraca [B,C,T]."""
    if mel.dim() == 2:
        if mel.size(0) == N_MELS:
            return mel.unsqueeze(0).contiguous()
        if mel.size(1) == N_MELS:
            return mel.transpose(0, 1).unsqueeze(0).contiguous()
        raise ValueError(f"Bad mel shape: {mel.shape}")
    if mel.dim() != 3:
        raise ValueError(f"Bad mel shape: {mel.shape}")
    if mel.size(-1) == N_MELS:
        return mel.transpose(1, 2).contiguous()
    return mel.contiguous()


def _crop_or_pad_bct(x_bct: torch.Tensor, T: int) -> torch.Tensor:
    B, C, t = x_bct.shape
    if t == T:
        return x_bct
    if t > T:
        return x_bct[:, :, :T].contiguous()
    return F.pad(x_bct, (0, T - t), value=0.0)


def _make_tmask_from_Tlen(T_len: torch.Tensor, T: int) -> torch.Tensor:
    """mask [B,1,T]"""
    idx = torch.arange(int(T), device=T_len.device)[None, :]
    return (idx < T_len.long().clamp_min(0)[:, None]).float().unsqueeze(1)


def _masked_l1_bct(pred_bct: torch.Tensor, tgt_bct: torch.Tensor, tmask_b1t: torch.Tensor) -> torch.Tensor:
    """L1 z maską po czasie: normalizacja po liczbie aktywnych ramek i mel-binach."""
    loss = (pred_bct - tgt_bct).abs() * tmask_b1t
    denom = (tmask_b1t.sum() * pred_bct.size(1)).clamp_min(1.0)
    return loss.sum() / denom


class FrozenOrangeTBRSpeakerVerifier(nn.Module):
    """Frozen Orange WavLM-TBR timbre verifier.

    This is not a generic WavLM hidden-state mean pooler. It uses the
    repository-provided EmbeddingsModel head, trained to compare timbral traits.
    Parameters are frozen, while gradients can still flow to the input waveform.
    """

    def __init__(self, source: str, device: torch.device, sample_rate: int = 16000):
        super().__init__()
        self.sample_rate = int(sample_rate)
        source_path = Path(str(source)).expanduser()
        if not source_path.exists():
            raise RuntimeError(
                f"Orange TBR verifier source does not exist: {source_path}. "
                "Download Orange/Speaker-wavLM-tbr into external/Orange_Speaker-wavLM-tbr first."
            )
        if str(source_path) not in sys.path:
            sys.path.insert(0, str(source_path))
        try:
            from spk_embeddings import EmbeddingsModel  # type: ignore
        except Exception as exc:
            raise RuntimeError(
                f"Cannot import spk_embeddings.py from Orange TBR verifier source: {source_path}"
            ) from exc
        self.model = EmbeddingsModel.from_pretrained(str(source_path)).to(device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    def encode(self, wav_bt: torch.Tensor, sample_rate: int) -> torch.Tensor:
        wav_bt = torch.as_tensor(wav_bt, dtype=torch.float32)
        if wav_bt.dim() == 1:
            wav_bt = wav_bt.unsqueeze(0)
        wav_bt = torch.nan_to_num(wav_bt, nan=0.0, posinf=0.0, neginf=0.0)
        wav_bt = wav_bt - wav_bt.mean(dim=-1, keepdim=True)
        peak = wav_bt.abs().amax(dim=-1, keepdim=True).clamp_min(1e-4)
        wav_bt = (wav_bt / peak.clamp_min(0.05)).clamp(-1.0, 1.0)
        if int(sample_rate) != self.sample_rate:
            wav_bt = TAF_AUDIO.resample(wav_bt, int(sample_rate), self.sample_rate)
        # Keep the same cap used by the model card helper to avoid quadratic WavLM memory spikes.
        wav_bt = wav_bt[:, :320000].contiguous()
        emb = self.model(wav_bt)
        return F.normalize(emb.float(), dim=-1)


def _decode_mel_for_verifier(
    vocos: Optional[nn.Module],
    mel_bct: torch.Tensor,
    *,
    max_sec: float,
) -> Optional[torch.Tensor]:
    if vocos is None or not torch.is_tensor(mel_bct) or mel_bct.numel() == 0:
        return None
    max_frames = int(max(1, round(float(max_sec) / _SECS_PER_FRAME_DEFAULT)))
    mel_bct = mel_bct[:, :, : min(int(mel_bct.size(-1)), max_frames)].contiguous().float()
    if mel_bct.numel() == 0 or (not bool(torch.isfinite(mel_bct).all().item())):
        return None
    wav = vocos.decode(mel_bct)
    if wav.dim() == 3:
        wav = wav.squeeze(1)
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)
    return wav.float()


def _frame_rms(wav_bt: torch.Tensor, *, frame: int = 1024, hop: int = 256) -> torch.Tensor:
    if wav_bt.dim() == 1:
        wav_bt = wav_bt.unsqueeze(0)
    if int(wav_bt.size(-1)) < int(frame):
        wav_bt = F.pad(wav_bt, (0, int(frame) - int(wav_bt.size(-1))))
    frames = wav_bt.unfold(dimension=-1, size=int(frame), step=int(hop))
    return (frames.pow(2).mean(dim=-1) + 1e-7).sqrt()


def _energy_consistency_loss(wav_hat_bt: torch.Tensor, wav_ref_bt: torch.Tensor) -> torch.Tensor:
    eh = _frame_rms(wav_hat_bt)
    with torch.no_grad():
        er = _frame_rms(wav_ref_bt.detach())
    n = min(int(eh.size(-1)), int(er.size(-1)))
    if n <= 0:
        return wav_hat_bt.new_zeros(())
    return F.l1_loss(torch.log(eh[:, :n] + 1e-5), torch.log(er[:, :n] + 1e-5))


def _autocorr_pitch_distribution(
    wav_bt: torch.Tensor,
    *,
    sample_rate: int = 24000,
    frame: int = 1024,
    hop: int = 256,
    fmin: float = 80.0,
    fmax: float = 500.0,
    temperature: float = 12.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if wav_bt.dim() == 1:
        wav_bt = wav_bt.unsqueeze(0)
    if int(wav_bt.size(-1)) < int(frame):
        wav_bt = F.pad(wav_bt, (0, int(frame) - int(wav_bt.size(-1))))
    frames = wav_bt.unfold(dimension=-1, size=int(frame), step=int(hop))
    frames = frames - frames.mean(dim=-1, keepdim=True)
    rms = (frames.pow(2).mean(dim=-1) + 1e-7).sqrt()
    min_lag = max(1, int(round(float(sample_rate) / float(fmax))))
    max_lag = min(int(frame) - 2, int(round(float(sample_rate) / float(fmin))))
    lags = torch.arange(min_lag, max_lag + 1, device=wav_bt.device, dtype=torch.long)
    corr_parts: List[torch.Tensor] = []
    for lag_i in lags.tolist():
        a = frames[..., :-int(lag_i)]
        b = frames[..., int(lag_i):]
        num = (a * b).sum(dim=-1)
        den = (a.pow(2).sum(dim=-1).clamp_min(1e-7) * b.pow(2).sum(dim=-1).clamp_min(1e-7)).sqrt()
        corr_parts.append(num / den)
    corr = torch.stack(corr_parts, dim=-1)
    dist = F.softmax(corr * float(temperature), dim=-1)
    freqs = float(sample_rate) / lags.to(dtype=torch.float32)
    return dist, freqs, rms


def _pitch_consistency_loss(wav_hat_bt: torch.Tensor, wav_ref_bt: torch.Tensor, *, sample_rate: int = 24000) -> torch.Tensor:
    dist_h, freqs, rms_h = _autocorr_pitch_distribution(wav_hat_bt, sample_rate=int(sample_rate))
    with torch.no_grad():
        dist_r, _freqs_r, rms_r = _autocorr_pitch_distribution(wav_ref_bt.detach(), sample_rate=int(sample_rate))
    nf = min(int(dist_h.size(1)), int(dist_r.size(1)))
    if nf <= 0:
        return wav_hat_bt.new_zeros(())
    dist_h = dist_h[:, :nf, :]
    dist_r = dist_r[:, :nf, :]
    rms_r = rms_r[:, :nf]
    rms_h = rms_h[:, :nf]
    voiced = (rms_r > rms_r.detach().quantile(0.35, dim=1, keepdim=True)).to(dist_h.dtype)
    denom = voiced.sum().clamp_min(1.0)
    log_f = torch.log(freqs.clamp_min(1.0)).to(device=dist_h.device, dtype=dist_h.dtype)
    exp_h = (dist_h * log_f[None, None, :]).sum(dim=-1)
    exp_r = (dist_r * log_f[None, None, :]).sum(dim=-1)
    pitch_mean = ((exp_h - exp_r).abs() * voiced).sum() / denom
    kl = (dist_r * (torch.log(dist_r.clamp_min(1e-7)) - torch.log(dist_h.clamp_min(1e-7)))).sum(dim=-1)
    pitch_shape = (kl * voiced).sum() / denom
    # Small voiced-energy mask term prevents the model from hiding pitch by dropping energy.
    voiced_energy = F.l1_loss(torch.log(rms_h[:, :nf] + 1e-5) * voiced, torch.log(rms_r[:, :nf] + 1e-5) * voiced)
    return pitch_mean + 0.10 * pitch_shape + 0.05 * voiced_energy


def _gaussian_nll_bct(mu_bct: torch.Tensor, logs_bct: torch.Tensor, x_bct: torch.Tensor, tmask_b1t: torch.Tensor) -> torch.Tensor:
    """
    NLL dla Gaussa per-frame:
      nll = (x-mu)^2/(2*sigma^2) + log(sigma)
    gdzie logs_bct = log(sigma).
    """
    logs = logs_bct.clamp(min=-7.0, max=2.0)
    sigma2 = torch.exp(2.0 * logs).clamp_min(1e-6)
    nll = ((x_bct - mu_bct) ** 2) / (2.0 * sigma2) + logs
    nll = nll * tmask_b1t
    denom = (tmask_b1t.sum().clamp_min(1.0) * x_bct.size(1))
    return nll.sum() / denom


def _pause_center_loss_weights_from_durations(
    ids_full: torch.Tensor,
    dur_values: torch.Tensor,
    T: int,
    *,
    edge_frames: int,
    pause_mid_weight: float,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Frame weights [B,1,T] that down-weight the middle of pause spans.

    Non-pause frames keep weight 1.0. Pause-token frames keep full weight on the
    first/last `edge_frames` frames and use `pause_mid_weight` in the middle.
    With edge_frames<=0 and pause_mid_weight>=1.0 this becomes all ones.
    """
    B = int(ids_full.size(0))
    T = int(max(0, T))
    edge_frames = int(max(0, edge_frames))
    pause_mid_weight = float(max(0.0, min(1.0, pause_mid_weight)))
    out = torch.ones((B, 1, T), device=dur_values.device, dtype=dtype)
    if T <= 0 or ((edge_frames <= 0) and pause_mid_weight >= 0.9999):
        return out

    pause_tok = _pause_mask_from_ids(ids_full)
    dur_int = torch.round(dur_values.float().clamp_min(0.0)).long()
    for b in range(B):
        parts: List[torch.Tensor] = []
        for j in range(int(ids_full.size(1))):
            d = int(dur_int[b, j].item())
            if d <= 0:
                continue
            if bool(pause_tok[b, j].item()):
                w = torch.full((d,), pause_mid_weight, device=dur_values.device, dtype=dtype)
                k = int(min(edge_frames, d))
                if k > 0:
                    w[:k] = 1.0
                    w[-k:] = 1.0
            else:
                w = torch.ones((d,), device=dur_values.device, dtype=dtype)
            parts.append(w)
        if parts:
            wb = torch.cat(parts, dim=0)
        else:
            wb = torch.empty((0,), device=dur_values.device, dtype=dtype)
        if int(wb.numel()) < T:
            fill = float(wb[-1].item()) if int(wb.numel()) > 0 else 1.0
            wb = F.pad(wb, (0, T - int(wb.numel())), value=fill)
        elif int(wb.numel()) > T:
            wb = wb[:T]
        out[b, 0, :] = wb
    return out


def _pause_middle_frame_mask_from_durations(
    ids_full: torch.Tensor,
    dur_values: torch.Tensor,
    T: int,
    *,
    edge_frames: int,
) -> torch.Tensor:
    """Boolean [B,1,T] mask for the middle of pause spans."""
    weights = _pause_center_loss_weights_from_durations(
        ids_full=ids_full,
        dur_values=dur_values,
        T=int(T),
        edge_frames=int(edge_frames),
        pause_mid_weight=0.0,
        dtype=torch.float32,
    )
    return weights <= 0.5


def _apply_digital_silence_to_wav(
    wav_1t: torch.Tensor,
    frame_mask_b1t: Optional[torch.Tensor],
    *,
    hop_length: int = 256,
) -> torch.Tensor:
    """Zero waveform samples covered by True mel-frame mask."""
    if (not torch.is_tensor(wav_1t)) or frame_mask_b1t is None or (not torch.is_tensor(frame_mask_b1t)):
        return wav_1t
    if wav_1t.ndim != 2 or int(wav_1t.size(0)) < 1 or int(wav_1t.size(-1)) <= 0:
        return wav_1t
    mask = frame_mask_b1t
    if mask.ndim == 3:
        mask = mask[0, 0]
    elif mask.ndim == 2:
        mask = mask[0]
    else:
        mask = mask.reshape(-1)
    mask = mask.detach().to(device="cpu", dtype=torch.bool)
    if int(mask.numel()) <= 0 or not bool(mask.any().item()):
        return wav_1t
    out = wav_1t.clone()
    T_wav = int(out.size(-1))
    hop = int(max(1, hop_length))
    for fi in torch.nonzero(mask, as_tuple=False).flatten().tolist():
        s = int(fi) * hop
        e = min(T_wav, (int(fi) + 1) * hop)
        if e > s:
            out[:, s:e] = 0.0
    return out


# ---------------- MASKS / CONDITIONING ----------------


class SpeakerVecAdapter(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, p_drop: float = 0.0):
        super().__init__()
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        hid_dim = int(max(int(out_dim), int(in_dim)))
        self.backbone = nn.Sequential(
            nn.Linear(int(in_dim), hid_dim),
            nn.GELU(),
            nn.Linear(hid_dim, hid_dim),
            nn.GELU(),
            nn.LayerNorm(hid_dim),
        )
        self.token_head = nn.Sequential(
            nn.Linear(hid_dim, int(out_dim)),
            nn.LayerNorm(int(out_dim)),
        )
        self.teacher_head = nn.Sequential(
            nn.Linear(hid_dim, int(in_dim)),
            nn.LayerNorm(int(in_dim)),
        )
        self.drop = nn.Dropout(float(p_drop))

    def forward(self, x_bd: torch.Tensor) -> torch.Tensor:
        x_bd = torch.as_tensor(x_bd)
        if x_bd.dim() != 2 or int(x_bd.size(-1)) != int(self.in_dim):
            raise ValueError(f"Speaker vector must be [B,{self.in_dim}], got {tuple(x_bd.shape)}")
        x_bd = self.drop(x_bd)
        h = self.backbone(x_bd)
        return self.token_head(h)

    def encode_teacher_space(self, x_bd: torch.Tensor) -> torch.Tensor:
        x_bd = torch.as_tensor(x_bd)
        if x_bd.dim() != 2 or int(x_bd.size(-1)) != int(self.in_dim):
            raise ValueError(f"Speaker vector must be [B,{self.in_dim}], got {tuple(x_bd.shape)}")
        h = self.backbone(self.drop(x_bd))
        z = self.teacher_head(h)
        return z / z.norm(dim=-1, keepdim=True).clamp_min(1e-12)
def _build_text_masks(tok_pad: torch.Tensor, special_len: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Buduje maski na sekwencji z prefiksem (speaker/bridge).

    Zwraca:
      ids_full:   [B, L+special_len] (PAD w prefiksie)
      tok_valid:  bool (nie-PAD)
      text_mask:  bool (tylko tokeny 'merytoryczne' do poolingu/rate: litery+digrafy; bez pauz, bez prefiksu)
    """
    ids_full = F.pad(tok_pad, (int(special_len), 0), value=PAD_ID)
    tok_valid = (ids_full != PAD_ID)

    # prefix mask
    prefix_mask = torch.zeros_like(ids_full, dtype=torch.bool)
    if int(special_len) > 0:
        prefix_mask[:, :int(special_len)] = True

    pause_mask = _pause_mask_from_ids(ids_full)

    # mask na tokeny 'tekstowe' (litery+digrafy), bez pauz
    text_ids_mask = torch.zeros_like(ids_full, dtype=torch.bool)
    for tid in _DUR_TEXT_IDS:
        text_ids_mask |= (ids_full == int(tid))

    text_mask = tok_valid & (~prefix_mask) & (~pause_mask) & text_ids_mask
    return ids_full, tok_valid, text_mask

# ---------------- DURATION TOKEN POLICY ----------------
# Chcemy, aby RAMKI (duracje) dostawały tylko:
#  - tokeny "tekstowe" (litery + digrafy typu <rz>, <cz>, <dź> ...)
#  - tokeny pauzy (<sp>, <BOS>, <EOS>)
# Cała reszta (interpunkcja, role, prosody, specjalne) ma mieć dur=0 i NIE być trenowana w dur-loss.

_ALLOWED_LETTERS = set("abcdefghijklmnopqrstuvwxyz" + "ąćęłńóśźż")

def _is_text_symbol(sym: str) -> bool:
    """Czy token ma dostawać ramki/duration?

    Zgodnie z Twoją polityką:
      - ramki tylko dla: liter (lowercase) + digrafów (<sz>, <cz>, <dz>, <dź>, <dż>, <ch>, <rz>)
      - tokeny pauzy (<sp>, <BOS>, <EOS>) dostają ramki
      - reszta (interpunkcja, <CAP>/<nar>/<akt>/<reserved*>, prefix z bridge/spk) ma dur=0
    """
    if not sym:
        return False

    # pojedyncza litera (PL), zakładamy lowercase vocab
    if len(sym) == 1 and sym.isalpha():
        return True

    # digrafy / znaki wieloliterowe w nawiasach
    if sym.startswith("<") and sym.endswith(">") and len(sym) >= 3:
        inner = sym[1:-1]
        # whitelist digrafów (zgodna z centralnym tokenizerem)
        DIGRAPHS = {"sz", "cz", "dz", "dź", "dż", "ch", "rz"}
        return inner.lower() in DIGRAPHS
    return False

# Zbuduj listy ID raz (na starcie), na podstawie słownika.
_DUR_TEXT_IDS: List[int] = []
for _tid, _sym in ID2SYMBOL.items():
    try:
        if _is_text_symbol(str(_sym)):
            _DUR_TEXT_IDS.append(int(_tid))
    except Exception:
        pass

_DUR_ALLOWED_IDS = set(_DUR_TEXT_IDS)
for _tid in PAUSE_TOKEN_IDS:
    _DUR_ALLOWED_IDS.add(int(_tid))

def _build_dur_allowed_mask(ids_full: torch.Tensor, special_len: int) -> torch.Tensor:
    """bool [B,L] gdzie token MA mieć duracje/ramki (tekst + pauzy), bez prefixu special_len."""
    if not _DUR_ALLOWED_IDS:
        m = torch.zeros_like(ids_full, dtype=torch.bool)
    else:
        m = torch.zeros_like(ids_full, dtype=torch.bool)
        for tid in _DUR_ALLOWED_IDS:
            m |= (ids_full == int(tid))
    if int(special_len) > 0:
        m[:, :int(special_len)] = False
    return m


def _masked_mean_text(x_tok: torch.Tensor, mask_bool: torch.Tensor) -> torch.Tensor:
    w = mask_bool.float().unsqueeze(-1)
    denom = w.sum(dim=1).clamp_min(1.0)
    return (x_tok * w).sum(dim=1) / denom




def _book_ids_to_tensor(book_ids: object) -> torch.Tensor:
    """
    `collate_fn` zwraca `book_ids` jako listę wartości (często str/int).
    ContextBridgeCache wymaga stabilnych intów, więc:
      - jeśli da się zrzutować do int -> użyj int,
      - inaczej -> deterministyczny hash (crc32).
    """
    if isinstance(book_ids, torch.Tensor):
        return book_ids.detach().cpu().long()
    if not isinstance(book_ids, (list, tuple)):
        book_ids = [book_ids]
    out: List[int] = []
    for b in book_ids:
        if isinstance(b, (int,)):
            out.append(int(b))
            continue
        s = str(b)
        try:
            out.append(int(s))
        except Exception:
            out.append(int(zlib.crc32(s.encode("utf-8"))) & 0x7FFFFFFF)
    return torch.tensor(out, dtype=torch.long)


@dataclass
class ContextBridgeState:
    mem: torch.Tensor  # [B, D_mem]


class ContextBridgeCache:
    """
    Cache per (speaker_id, book_id):
      - prev_h: cached token embeddings [1, L, D]
      - prev_mask: cached token mask [1, L]
      - mem: global memory vector [1, D_mem]
    """
    def __init__(self):
        self._mem: Dict[Tuple[int, int], torch.Tensor] = {}
        self._prev_h: Dict[Tuple[int, int], torch.Tensor] = {}
        self._prev_mask: Dict[Tuple[int, int], torch.Tensor] = {}
        self._last_chunk: Dict[Tuple[int, int], int] = {}

    def _reset_needed(self, key: Tuple[int, int], chunk_i: int) -> bool:
        if chunk_i == 0:
            return True
        last = self._last_chunk.get(key, None)
        if last is None:
            return True
        return (last + 1) != chunk_i

    def get_batch_state(
        self,
        *,
        speaker_ids: torch.Tensor,  # CPU tensor [B]
        book_ids: torch.Tensor,     # CPU tensor [B]
        chunk_idx: torch.Tensor,    # CPU tensor [B]
        mem_dim: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[ContextBridgeState, Optional[torch.Tensor], Optional[torch.Tensor], List[Tuple[int, int]], List[int]]:
        keys: List[Tuple[int, int]] = []
        chunks: List[int] = []
        mem_list: List[torch.Tensor] = []
        prev_h_list: List[Optional[torch.Tensor]] = []
        prev_mask_list: List[Optional[torch.Tensor]] = []

        sids = speaker_ids.detach().cpu().tolist()
        bids = book_ids.detach().cpu().tolist()
        cidx = chunk_idx.detach().cpu().tolist()

        for sid, bid, ci in zip(sids, bids, cidx):
            key = (int(sid), int(bid))
            ci = int(ci)
            keys.append(key)
            chunks.append(ci)

            if self._reset_needed(key, ci) or (key not in self._mem):
                mem = torch.zeros((1, mem_dim), device=device, dtype=dtype)
                prev_h = None
                prev_mask = None
            else:
                mem_cpu = self._mem[key]
                mem = mem_cpu.to(device=device, dtype=dtype)
                prev_h = self._prev_h.get(key, None)
                prev_mask = self._prev_mask.get(key, None)
                if prev_h is not None:
                    prev_h = prev_h.to(device=device, dtype=dtype)
                if prev_mask is not None:
                    prev_mask = prev_mask.to(device=device, dtype=torch.bool)

            mem_list.append(mem)
            prev_h_list.append(prev_h)
            prev_mask_list.append(prev_mask)

        mem_b = torch.cat(mem_list, dim=0)  # [B, D_mem]

        # Pad prev_h to max length if any exists
        if any(h is not None for h in prev_h_list):
            max_len = max(int(h.size(1)) for h in prev_h_list if h is not None)
            d = int(prev_h_list[0].size(-1)) if prev_h_list[0] is not None else int(
                next(h for h in prev_h_list if h is not None).size(-1)
            )
            prev_h_b = mem_b.new_zeros((len(prev_h_list), max_len, d))
            prev_mask_b = torch.zeros((len(prev_h_list), max_len), device=device, dtype=torch.bool)
            for i, (h, m) in enumerate(zip(prev_h_list, prev_mask_list)):
                if h is None:
                    continue
                L = int(h.size(1))
                prev_h_b[i, :L, :] = h
                if m is not None:
                    prev_mask_b[i, :L] = m
                else:
                    prev_mask_b[i, :L] = True
        else:
            prev_h_b = None
            prev_mask_b = None

        return ContextBridgeState(mem=mem_b), prev_h_b, prev_mask_b, keys, chunks

    def set_batch_state(
        self,
        *,
        keys: List[Tuple[int, int]],
        chunks: List[int],
        mem_after: torch.Tensor,          # [B, D_mem]
        h_curr: torch.Tensor,             # [B, L, D]
        mask_curr: Optional[torch.Tensor] = None,  # [B, L] bool
    ) -> None:
        for i, (key, ci) in enumerate(zip(keys, chunks)):
            self._mem[key] = mem_after[i:i+1].detach().to("cpu", dtype=torch.float32)
            self._prev_h[key] = h_curr[i:i+1].detach().to("cpu", dtype=torch.float32)
            if mask_curr is not None:
                self._prev_mask[key] = mask_curr[i:i+1].detach().to("cpu", dtype=torch.bool)
            else:
                self._prev_mask[key] = torch.ones((1, h_curr.size(1)), dtype=torch.bool)
            self._last_chunk[key] = int(ci)

class DurationRhythmCache:
    """Detached duration summary for the previous consecutive chunk."""

    feature_dim = 6

    def __init__(self):
        self._state: Dict[Tuple[int, int], torch.Tensor] = {}
        self._last_chunk: Dict[Tuple[int, int], int] = {}

    def clear(self) -> None:
        self._state.clear()
        self._last_chunk.clear()

    def get_batch(self, *, speaker_ids, book_ids, chunk_idx, device, dtype=torch.float32):
        rows = []
        for sid, bid, ci in zip(
            speaker_ids.detach().cpu().tolist(), book_ids.detach().cpu().tolist(),
            chunk_idx.detach().cpu().tolist(),
        ):
            key = (int(sid), int(bid))
            previous = self._state.get(key)
            consecutive = int(ci) > 0 and self._last_chunk.get(key) == int(ci) - 1
            rows.append(previous.clone() if previous is not None and consecutive else torch.zeros(self.feature_dim))
        return torch.stack(rows).to(device=device, dtype=dtype)

    @staticmethod
    def summarize(durations: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
        durations = durations.detach().float().clamp_min(0.0)
        token_ids = token_ids.detach().long()
        valid = (token_ids != int(PAD_ID)) & (durations > 0.0)
        pause = valid & (token_ids == int(SYMBOL2ID.get("<sp>", -999999)))
        speech = valid & ~pause
        valid_f, pause_f, speech_f = valid.float(), pause.float(), speech.float()
        total = (durations * valid_f).sum(1)
        pause_total = (durations * pause_f).sum(1)
        speech_mean = (durations * speech_f).sum(1) / speech_f.sum(1).clamp_min(1.0)
        pause_mean = pause_total / pause_f.sum(1).clamp_min(1.0)
        frames_per_token = total / valid_f.sum(1).clamp_min(1.0)
        pause_fraction = pause_total / total.clamp_min(1.0)
        final_pause = torch.zeros_like(total)
        for b in range(int(durations.size(0))):
            indices = torch.nonzero(pause[b], as_tuple=False).flatten()
            if indices.numel() > 0:
                final_pause[b] = durations[b, indices[-1]]
        return torch.stack([
            torch.log1p(speech_mean), torch.log1p(pause_mean), pause_fraction,
            torch.log1p(frames_per_token), torch.log1p(final_pause), torch.ones_like(total),
        ], dim=-1)

    def set_batch(self, *, speaker_ids, book_ids, chunk_idx, durations, token_ids) -> None:
        summaries = self.summarize(durations, token_ids).cpu()
        for i, (sid, bid, ci) in enumerate(zip(
            speaker_ids.detach().cpu().tolist(), book_ids.detach().cpu().tolist(),
            chunk_idx.detach().cpu().tolist(),
        )):
            key = (int(sid), int(bid))
            self._state[key] = summaries[i].clone()
            self._last_chunk[key] = int(ci)


class AcousticMemoryCache:
    """Detached mel tail for the previous consecutive chunk."""

    def __init__(self, max_frames: int):
        self.max_frames = max(1, int(max_frames))
        self._tails: Dict[Tuple[int, int], torch.Tensor] = {}
        self._last_chunk: Dict[Tuple[int, int], int] = {}

    def clear(self) -> None:
        self._tails.clear()
        self._last_chunk.clear()

    def get_batch(self, *, speaker_ids, book_ids, chunk_idx, n_mels, device, dtype):
        sids = speaker_ids.detach().cpu().tolist()
        bids = book_ids.detach().cpu().tolist()
        chunks = chunk_idx.detach().cpu().tolist()
        batch = torch.zeros(len(sids), int(n_mels), self.max_frames)
        lengths = torch.zeros(len(sids), dtype=torch.long)
        available = torch.zeros(len(sids))
        for i, (sid, bid, ci) in enumerate(zip(sids, bids, chunks)):
            key = (int(sid), int(bid))
            tail = self._tails.get(key)
            if tail is None or not (int(ci) > 0 and self._last_chunk.get(key) == int(ci) - 1):
                continue
            length = min(self.max_frames, int(tail.size(-1)))
            batch[i, :, -length:] = tail[0, :, -length:].float()
            lengths[i] = length
            available[i] = 1.0
        return batch.to(device=device, dtype=dtype), lengths.to(device=device), available.to(device=device, dtype=dtype)

    def set_batch(self, *, speaker_ids, book_ids, chunk_idx, mel_bct, frame_lengths) -> None:
        mel_cpu = mel_bct.detach().cpu().float()
        lengths = frame_lengths.detach().cpu().long().tolist()
        for i, (sid, bid, ci, length) in enumerate(zip(
            speaker_ids.detach().cpu().tolist(), book_ids.detach().cpu().tolist(),
            chunk_idx.detach().cpu().tolist(), lengths,
        )):
            key = (int(sid), int(bid))
            length = max(1, min(int(length), int(mel_cpu.size(-1))))
            self._tails[key] = mel_cpu[i:i + 1, :, max(0, length - self.max_frames):length].contiguous()
            self._last_chunk[key] = int(ci)


class PreviousAcousticMemoryEncoder(nn.Module):
    """Compress a previous mel tail into the checkpoint's attribute-prefix token."""

    def __init__(self, *, n_mels: int, hidden_dim: int, gate_init: float = 0.01, gate_max: float = 0.05):
        super().__init__()
        self.gate_max = float(max(1e-4, gate_max))
        ratio = min(1.0 - 1e-5, max(1e-5, float(gate_init) / self.gate_max))
        self.raw_gate = nn.Parameter(torch.tensor(math.log(ratio / (1.0 - ratio))))
        self.convs = nn.ModuleList([
            nn.Conv1d(int(n_mels), 128, 5, stride=2, padding=2),
            nn.Conv1d(128, 128, 5, stride=2, padding=2, groups=128),
            nn.Conv1d(128, 128, 5, stride=2, padding=2, groups=128),
        ])
        self.pointwise = nn.ModuleList([nn.Identity(), nn.Conv1d(128, 128, 1), nn.Conv1d(128, 128, 1)])
        self.norms = nn.ModuleList([nn.GroupNorm(8, 128) for _ in range(3)])
        self.attn_score = nn.Conv1d(128, 1, 1)
        self.out = nn.Linear(256, int(hidden_dim))
        self.out_norm = nn.LayerNorm(int(hidden_dim))

    def forward(self, mel_bct, lengths, available, *, dropout_prob=0.0, noise_std=0.0):
        x = mel_bct.float()
        reduced_lengths = lengths.long()
        for conv, pointwise, norm in zip(self.convs, self.pointwise, self.norms):
            x = F.silu(norm(pointwise(conv(x))))
            reduced_lengths = torch.div(reduced_lengths + 1, 2, rounding_mode="floor")
        time = int(x.size(-1))
        positions = torch.arange(time, device=x.device)[None]
        valid = positions >= (time - reduced_lengths.clamp(0, time))[:, None]
        weights = torch.softmax(self.attn_score(x).squeeze(1).masked_fill(~valid, -1e4), dim=-1) * valid.float()
        weights = weights / weights.sum(1, keepdim=True).clamp_min(1e-8)
        mean = torch.einsum("bt,bct->bc", weights, x)
        variance = torch.einsum("bt,bct->bc", weights, (x - mean[:, :, None]).square())
        token = self.out_norm(self.out(torch.cat([mean, variance.clamp_min(1e-8).sqrt()], dim=-1)))
        token = torch.nan_to_num(3.0 * torch.tanh(token / 3.0), nan=0.0, posinf=3.0, neginf=-3.0)
        return token * available.float()[:, None]

    @staticmethod
    def replace_prefix_token(base: torch.Tensor, token: torch.Tensor) -> torch.Tensor:
        token = token.to(device=base.device, dtype=base.dtype)
        active = token.detach().abs().amax(dim=-1, keepdim=True) > 0
        return torch.where(active, token, base)


class ContextBridge(nn.Module):
    """
    Context bridge with:
      - local context injection via cross-attention (Q=H_raw, K/V=H_prev)
      - global memory update via attentive pooling + GRU
    """
    def __init__(
        self,
        text_dim: int,
        mem_dim: Optional[int] = None,
        attn_heads: int = 4,
        attn_drop: float = 0.0,
    ):
        super().__init__()
        self.text_dim = int(text_dim)
        self.mem_dim = int(text_dim if mem_dim is None else mem_dim)
        self.context_attn = nn.MultiheadAttention(
            embed_dim=self.text_dim,
            num_heads=int(max(1, attn_heads)),
            dropout=float(attn_drop),
            batch_first=True,
        )
        self.pool_attn = nn.MultiheadAttention(
            embed_dim=self.text_dim,
            num_heads=int(max(1, attn_heads)),
            dropout=float(attn_drop),
            batch_first=True,
        )
        self.state_to_query = nn.Linear(self.mem_dim, self.text_dim)
        self.gru = nn.GRUCell(input_size=self.text_dim, hidden_size=self.mem_dim)
        self.mem_to_text = nn.Linear(self.mem_dim, self.text_dim)

    def inject_context(
        self,
        h_raw: torch.Tensor,              # [B,L,D]
        prev_h: Optional[torch.Tensor],   # [B,Lp,D]
        prev_mask: Optional[torch.Tensor] # [B,Lp]
    ) -> torch.Tensor:
        if prev_h is None:
            return h_raw
        key_padding_mask = None
        if prev_mask is not None:
            key_padding_mask = ~prev_mask.to(dtype=torch.bool, device=prev_h.device)
        attn_out, _ = self.context_attn(h_raw, prev_h, prev_h, key_padding_mask=key_padding_mask, need_weights=False)
        return h_raw + attn_out

    def update_memory(
        self,
        mem_before: torch.Tensor,      # [B,D_mem]
        h_curr: torch.Tensor,          # [B,L,D_text]
        mask_curr: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        q = self.state_to_query(mem_before).unsqueeze(1)  # [B,1,D_text]
        key_padding_mask = None
        if mask_curr is not None:
            key_padding_mask = ~mask_curr.to(dtype=torch.bool, device=h_curr.device)
        z_curr, _ = self.pool_attn(q, h_curr, h_curr, key_padding_mask=key_padding_mask, need_weights=False)
        z_curr = z_curr.squeeze(1)  # [B,D_text]
        mem_after = self.gru(z_curr, mem_before)
        return mem_after

def encode_text_features_stateful(
    *,
    model: nn.Module,
    spk_embed: nn.Module,
    gender_embed: nn.Module,
    emotion_token_embed: Optional[nn.Module] = None,
    tok_pad: torch.Tensor,      # [B,L]
    speaker_ids: torch.Tensor,  # [B] (na device)
    gender_ids: torch.Tensor,   # [B] (na device)
    emotion_ids: Optional[torch.Tensor] = None,  # [B] (na device)
    book_ids: torch.Tensor,     # [B] (CPU ok)
    chunk_idx: torch.Tensor,    # [B] (CPU ok)
    device: torch.device,
    bridge: ContextBridge,
    bridge_cache: ContextBridgeCache,
    spk_vec_override: Optional[torch.Tensor] = None,  # [B,D_text]
    require_spk_override: bool = True,
    use_emotion_token: bool = False,
    attribute_token_override: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, int, torch.Tensor]:
    """
    Stateful wariant encode_text_features (prefix tokens: speaker + attribute/emotion + mem):
      - bierze stan i poprzedni chunk z cache per (speaker_id, book_id)
      - resetuje, gdy chunk_idx==0 lub nie ma ciągłości
      - wstrzykuje lokalny kontekst przez cross-attn z poprzednim chunkiem
      - aktualizuje globalną pamięć przez attention + GRU
    """
    B = tok_pad.size(0)
    special_len = int(CONFIG.get("special_len", 0))

    # ids_full (prefix tokens at front)
    ids_full = F.pad(tok_pad, (int(special_len), 0), value=PAD_ID)

    # embedding bazowy
    x = model.embed(ids_full)  # [B, Ls, D_text]
    if spk_vec_override is not None:
        spk_tok = spk_vec_override.to(dtype=x.dtype, device=x.device)
    else:
        if int(special_len) >= 1 and bool(require_spk_override):
            raise RuntimeError(
                "Speaker override is required (require_spk_override=True) but spk_vec_override=None. "
                "This run is configured to never fall back to spk_embed(speaker_id)."
            )
        spk_tok = spk_embed(speaker_ids).to(x.dtype)  # [B, D_text]

    # pobierz stan batcha z cache (CPU->GPU)
    mem_dim = int(CONFIG.get("context_mem_dim", 0))
    if mem_dim <= 0:
        mem_dim = int(x.size(-1))
    state_before, prev_h, prev_mask, keys, chunks = bridge_cache.get_batch_state(
        speaker_ids=speaker_ids.detach().cpu(),
        book_ids=book_ids,
        chunk_idx=chunk_idx,
        mem_dim=mem_dim,
        device=device,
        dtype=x.dtype,
    )
    # insert prefix tokens
    if int(special_len) >= 1:
        x[:, 0, :] = spk_tok
    if int(special_len) >= 2:
        if bool(use_emotion_token) and emotion_token_embed is not None and emotion_ids is not None:
            attr_tok = emotion_token_embed(emotion_ids.clamp(0, len(EMOTION_GROUP_TO_ID) - 1)).to(dtype=x.dtype, device=x.device)
        else:
            attr_tok = gender_embed(gender_ids).to(dtype=x.dtype, device=x.device)
        x[:, 1, :] = attr_tok
    if int(special_len) >= 3:
        mem_tok = bridge.mem_to_text(state_before.mem).to(dtype=x.dtype, device=x.device)  # [B,D]
        x[:, 2, :] = mem_tok
    if attribute_token_override is not None:
        if int(special_len) < 2:
            raise RuntimeError("Acoustic memory requires attribute prefix slot [1].")
        x[:, 1, :] = attribute_token_override.to(dtype=x.dtype, device=x.device)

    # pozycje + encoder
    pos = getattr(model, "pos", None)
    if pos is not None:
        # Some runs use a learned positional table with a fixed max length (often 2000).
        # If a rare sample exceeds that, fall back to a sinusoidal continuation for the tail
        # to avoid crashing mid-training.
        pos_bt = pos.to(x.dtype)
        if int(pos_bt.size(1)) >= int(x.size(1)):
            x = x + pos_bt[:, : x.size(1), :]
        else:
            head = pos_bt
            tail_len = int(x.size(1) - pos_bt.size(1))
            d = int(x.size(-1))
            # [1, tail_len, d]
            t = torch.arange(int(pos_bt.size(1)), int(x.size(1)), device=x.device, dtype=torch.float32).unsqueeze(0)
            half = d // 2
            freqs = torch.exp(-math.log(10000.0) * torch.arange(0, half, device=x.device, dtype=torch.float32) / max(1, half))
            ang = t.transpose(0, 1) * freqs.unsqueeze(0)  # [tail_len, half]
            sin = torch.sin(ang)
            cos = torch.cos(ang)
            pe = torch.cat([sin, cos], dim=1)  # [tail_len, 2*half]
            if pe.size(1) < d:
                pe = F.pad(pe, (0, d - pe.size(1)))
            pe = pe[:tail_len, :d].unsqueeze(0).to(dtype=x.dtype)  # [1, tail_len, d]
            pos_full = torch.cat([head, pe], dim=1)  # [1, L, d]
            x = x + pos_full

    for blk in model.enc:
        x = blk(x)

    # ---- local context injection (cross-attn with prev chunk) ----
    if int(special_len) > 0:
        h_text = x[:, int(special_len) :, :]
    else:
        h_text = x
    h_text = bridge.inject_context(h_text, prev_h, prev_mask)
    if int(special_len) > 0:
        x = torch.cat([x[:, : int(special_len), :], h_text], dim=1)
    else:
        x = h_text
    h_curr = h_text

    # ---- global memory update ----
    token_mask = (ids_full != PAD_ID)
    # use only real text tokens for memory update (no prefix tokens)
    text_mask = token_mask[:, int(special_len) :]
    mem_after = bridge.update_memory(state_before.mem, h_curr, text_mask)

    # cache H_curr (without mem token) + mem
    bridge_cache.set_batch_state(keys=keys, chunks=chunks, mem_after=mem_after, h_curr=h_curr, mask_curr=text_mask)

    return x, ids_full, special_len, mem_after

# ---------------- ENCODER HELPER: prefix (speaker + bridge) ----------------
def encode_text_features(
    *,
    model: nn.Module,
    spk_embed: nn.Module,
    gender_embed: nn.Module,
    emotion_token_embed: Optional[nn.Module] = None,
    tok_pad: torch.Tensor,      # [B,L]
    speaker_ids: torch.Tensor,  # [B]
    gender_ids: torch.Tensor,   # [B]
    emotion_ids: Optional[torch.Tensor] = None,
    device: torch.device,
    spk_vec_override: Optional[torch.Tensor] = None,  # [B,D]
    require_spk_override: bool = True,
    use_emotion_token: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """
    Koduje tekst w przestrzeń ukrytą enkodera z prefiksami (speaker + gender + mem).
    """
    special_len = int(CONFIG.get("special_len", 1))
    ids_full = F.pad(tok_pad, (int(special_len), 0), value=PAD_ID)
    x = model.embed(ids_full)  # [B,Ls,D]
    if spk_vec_override is not None:
        spk_tok = spk_vec_override.to(dtype=x.dtype, device=x.device)
    else:
        if int(special_len) >= 1 and bool(require_spk_override):
            raise RuntimeError(
                "Speaker override is required (require_spk_override=True) but spk_vec_override=None. "
                "This run is configured to never fall back to spk_embed(speaker_id)."
            )
        spk_tok = spk_embed(speaker_ids).to(x.dtype)  # [B,D]
    if int(special_len) >= 1:
        x[:, 0, :] = spk_tok
    if int(special_len) >= 2:
        if bool(use_emotion_token) and emotion_token_embed is not None and emotion_ids is not None:
            x[:, 1, :] = emotion_token_embed(emotion_ids.clamp(0, len(EMOTION_GROUP_TO_ID) - 1)).to(dtype=x.dtype, device=x.device)
        else:
            x[:, 1, :] = gender_embed(gender_ids).to(dtype=x.dtype, device=x.device)
    if int(special_len) >= 3:
        x[:, 2, :] = x.new_zeros((x.size(0), x.size(-1)))
    # 5) pozycje + encoder
    pos = getattr(model, "pos", None)
    if pos is not None:
        pos_bt = pos.to(x.dtype)
        if int(pos_bt.size(1)) >= int(x.size(1)):
            x = x + pos_bt[:, : x.size(1), :]
        else:
            head = pos_bt
            tail_len = int(x.size(1) - pos_bt.size(1))
            d = int(x.size(-1))
            t = torch.arange(int(pos_bt.size(1)), int(x.size(1)), device=x.device, dtype=torch.float32).unsqueeze(0)
            half = d // 2
            freqs = torch.exp(-math.log(10000.0) * torch.arange(0, half, device=x.device, dtype=torch.float32) / max(1, half))
            ang = t.transpose(0, 1) * freqs.unsqueeze(0)
            sin = torch.sin(ang)
            cos = torch.cos(ang)
            pe = torch.cat([sin, cos], dim=1)
            if pe.size(1) < d:
                pe = F.pad(pe, (0, d - pe.size(1)))
            pe = pe[:tail_len, :d].unsqueeze(0).to(dtype=x.dtype)
            pos_full = torch.cat([head, pe], dim=1)
            x = x + pos_full

    for blk in model.enc:
        x = blk(x)

    return x, ids_full, special_len

def _build_flow_cond(
    mode: str,
    source: str,
    *,
    L_gt_full_used: Optional[torch.Tensor],
    T_len: torch.Tensor,
    total_len_pred: Optional[torch.Tensor],
    rate_pred: Optional[torch.Tensor],
    tok_pad: torch.Tensor,
    special_len: int,
) -> Optional[torch.Tensor]:
    """
    cond dla FlowDurationPredictor: [B,cond_dim] albo None
    """
    if mode == "none":
        return None

    conds: List[torch.Tensor] = []

    if source == "pred":
        if mode in ("len", "both"):
            if total_len_pred is None:
                return None
            conds.append(torch.log1p(total_len_pred))
        if mode in ("rate", "both"):
            if rate_pred is None:
                return None
            conds.append(torch.log1p(rate_pred))
        return torch.cat(conds, dim=1) if conds else None

    # GT
    if L_gt_full_used is None:
        return None

    _, _, text_mask = _build_text_masks(tok_pad, special_len)
    text_mask_f = text_mask.float()

    rate_gt = (L_gt_full_used * text_mask_f).sum(dim=1, keepdim=True) / text_mask_f.sum(dim=1, keepdim=True).clamp_min(1.0)

    if mode in ("len", "both"):
        conds.append(torch.log1p(T_len.float().unsqueeze(1)))
    if mode in ("rate", "both"):
        conds.append(torch.log1p(rate_gt))

    return torch.cat(conds, dim=1) if conds else None


# ---------------- FLOW MATCHING (MEL) ----------------
class FlowMatchHelper:
    """
    Rectified Flow:
      xt = (1-t)*x0 + t*x1
      target_v = x1 - x0
    """
    def get_flow_tuple(self, x0_bct: torch.Tensor, x1_bct: torch.Tensor, t_b: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        t = t_b[:, None, None]
        xt = (1.0 - t) * x0_bct + t * x1_bct
        target_v = x1_bct - x0_bct
        return xt, target_v


# ---------------- MODULES: PRIOR MU + MEL FLOW ----------------
class GatedStyleAdaLN(nn.Module):
    """
    Checkpoint-safe global style conditioning.

    This is AdaLN/FiLM-like conditioning initialized as an exact no-op:
      x -> x + gate * (scale(style) * LN(x) + shift(style))

    Because the projection and gate start at zero, resuming an older checkpoint
    keeps the original behavior until the adapter learns useful conditioning.
    """

    def __init__(self, dim: int, style_dim: int):
        super().__init__()
        self.dim = int(dim)
        self.style_dim = int(style_dim)
        self.ln = nn.LayerNorm(self.dim)
        self.to_scale_shift = nn.Linear(self.style_dim, self.dim * 2)
        self.gate = nn.Parameter(torch.zeros(()))
        nn.init.zeros_(self.to_scale_shift.weight)
        nn.init.zeros_(self.to_scale_shift.bias)

    def forward(self, x_btd: torch.Tensor, style_bd: Optional[torch.Tensor]) -> torch.Tensor:
        if style_bd is None:
            return x_btd
        ss = self.to_scale_shift(style_bd.to(device=x_btd.device, dtype=x_btd.dtype))
        scale, shift = ss.chunk(2, dim=-1)
        x_norm = self.ln(x_btd)
        g = self.gate.to(dtype=x_btd.dtype)
        delta = x_norm * scale[:, None, :] + shift[:, None, :]
        return x_btd + g * delta


def _sinusoidal_t(t: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(0, half, device=t.device, dtype=t.dtype) / max(1, half - 1))
    args = t[:, None] * freqs[None, :]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


def _sample_flow_t(
    batch_size: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
    mode: str = "logit_normal",
    logit_mu: float = 0.0,
    logit_sigma: float = 1.0,
    clamp_eps: float = 1e-4,
) -> torch.Tensor:
    mode = str(mode).lower().strip()
    if mode == "uniform":
        t_b = torch.rand((int(batch_size),), device=device, dtype=dtype)
    elif mode == "logit_normal":
        z = torch.randn((int(batch_size),), device=device, dtype=dtype)
        z = z * float(logit_sigma) + float(logit_mu)
        t_b = torch.sigmoid(z)
    else:
        raise RuntimeError(f"Unsupported t-sample mode: {mode}")
    eps = float(max(0.0, min(0.49, float(clamp_eps))))
    if eps > 0.0:
        t_b = t_b.clamp(min=eps, max=1.0 - eps)
    return t_b


class ProbabilisticPriorDecoder(nn.Module):
    """
    Probabilistyczny prior: uczy mu + log(sigma) per ramkę i zwraca także próbkę t0.

    forward() zwraca:
      t0_btc, mu_btc, logs_btc, T_vec
    gdzie *_btc są [B,T,C] (C=n_mels).
    """
    def __init__(
        self,
        dim: int,
        n_mels: int,
        n_layers: int = 3,
        n_heads: int = 8,
        logs_min: float = -7.0,
        logs_max: float = 2.0,
        dualpath_branch_dim: Optional[int] = None,
        dualpath_attn_dim: Optional[int] = None,
        dualpath_conv_dim: Optional[int] = None,
        dualpath_init_split_identity: bool = True,
    ):
        super().__init__()
        self.dim = dim
        self.n_mels = n_mels
        self.logs_min = float(logs_min)
        self.logs_max = float(logs_max)

        self.inp = nn.Linear(dim, dim)
        self.blocks = nn.ModuleList([
            DualPathProjectedBlock(
                dim,
                num_heads=n_heads,
                use_sdpa=True,
                use_adaln=True,
                use_dual_adaln=True,
                cond_dim=dim,
                branch_dim=dualpath_branch_dim,
                attn_dim=dualpath_attn_dim,
                conv_dim=dualpath_conv_dim,
                init_split_identity=dualpath_init_split_identity,
            )
            for _ in range(n_layers)
        ])
        self.timbre_adaln = nn.ModuleList([
            GatedStyleAdaLN(dim=int(dim), style_dim=int(dim))
            for _ in range(n_layers)
        ])
        self.to_stats = nn.Linear(dim, 2 * n_mels)

    def _upsample_gauss(
        self,
        h_tok: torch.Tensor,
        dur: torch.Tensor,
        T_hint: Optional[torch.Tensor],
        *,
        sp_mask_tok: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, L, D = h_tok.shape
        dur = torch.nan_to_num(dur, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
        valid_tok = (dur > 0.0)

        if T_hint is None:
            T_vec = torch.round(dur.sum(dim=1)).long().clamp_min(1)
        else:
            T_vec = T_hint.long().clamp_min(1)

        Tmax = int(T_vec.max().item())

        csum = torch.cumsum(dur, dim=1)
        centers = csum - 0.5 * dur

        t_idx = torch.arange(Tmax, device=h_tok.device, dtype=h_tok.dtype)[None, :, None]  # [1,T,1]
        dist = torch.abs(t_idx - centers[:, None, :])  # [B,T,L]
        dist = torch.where(valid_tok[:, None, :], dist, dist.new_full(dist.shape, 1e9))

        sigma = float(CONFIG["gauss_sigma"])
        weights = torch.exp(-0.5 * (dist / max(1e-6, sigma)) ** 2)  # [B,T,L]

        gauss_sp_mode = str(CONFIG.get("gauss_sp_mode", "mix")).lower().strip()
        if gauss_sp_mode == "isolate" and sp_mask_tok is not None:
            sp_mask_tok = sp_mask_tok.to(device=h_tok.device, dtype=torch.bool)
            idx_near = torch.argmin(dist, dim=2)  # [B,T]
            frame_is_sp = torch.gather(sp_mask_tok, 1, idx_near)  # [B,T]
            same_group = (sp_mask_tok[:, None, :] == frame_is_sp[:, :, None])  # [B,T,L]
            weights = torch.where(same_group, weights, weights.new_zeros(()))

        topk = int(CONFIG["gauss_topk"])
        if 0 < topk < L:
            w_top, idx_top = torch.topk(weights, k=topk, dim=2)  # [B,T,K]
            h_exp = h_tok[:, None, :, :].expand(B, Tmax, L, D)
            h_g = torch.gather(h_exp, 2, idx_top[:, :, :, None].expand(B, Tmax, topk, D))  # [B,T,K,D]
            w_norm = w_top / w_top.sum(dim=2, keepdim=True).clamp_min(1e-8)
            h_frame = (h_g * w_norm[:, :, :, None]).sum(dim=2)  # [B,T,D]
            w_norm_full = weights.new_zeros(B, Tmax, L)
            w_norm_full.scatter_(2, idx_top, w_norm)
        else:
            w_norm = weights / weights.sum(dim=2, keepdim=True).clamp_min(1e-8)
            h_frame = torch.einsum("btl,bld->btd", w_norm, h_tok)
            w_norm_full = w_norm

        # mask po T
        tmask_bt1 = _make_tmask_from_Tlen(T_vec, Tmax).transpose(1, 2)  # [B,T,1]
        self.last_gauss_align_btl = (w_norm_full * tmask_bt1).detach()
        return h_frame * tmask_bt1, T_vec

    def forward(
        self,
        h_tok: torch.Tensor,
        dur_values: torch.Tensor,
        *,
        cond: Optional[torch.Tensor] = None,
        T_hint: Optional[torch.Tensor] = None,
        noise_scale: float = 1,
        sp_mask_tok: Optional[torch.Tensor] = None,
        timbre_vec: Optional[torch.Tensor] = None,
        return_hidden: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.inp(h_tok)
        for i, blk in enumerate(self.blocks):
            if i < len(self.timbre_adaln):
                x = self.timbre_adaln[i](x, timbre_vec)
            x = blk(x, cond)

        h_frame, T_vec = self._upsample_gauss(x, dur_values, T_hint, sp_mask_tok=sp_mask_tok)
        stats = self.to_stats(h_frame)  # [B,T,2C]
        mu, logs = stats.chunk(2, dim=-1)
        logs = logs.clamp(min=self.logs_min, max=self.logs_max)
        eps = torch.randn_like(mu)
        t0 = mu + eps * torch.exp(logs) * float(noise_scale)
        if return_hidden:
            return t0, mu, logs, T_vec, h_frame
        return t0, mu, logs, T_vec


class PriorProsodyHeads(nn.Module):
    """Auxiliary prior supervision heads inspired by StyleTTS2 F0/N predictors."""

    def __init__(self, dim: int):
        super().__init__()
        self.energy = nn.Sequential(
            nn.LayerNorm(int(dim)),
            nn.Linear(int(dim), int(dim)),
            nn.SiLU(),
            nn.Linear(int(dim), 1),
        )
        self.f0 = nn.Sequential(
            nn.LayerNorm(int(dim)),
            nn.Linear(int(dim), int(dim)),
            nn.SiLU(),
            nn.Linear(int(dim), 1),
        )

    def forward(self, h_frame_btd: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        energy = self.energy(h_frame_btd).squeeze(-1)
        f0 = self.f0(h_frame_btd).squeeze(-1)
        return f0, energy


def _styletts_log_norm(mel_bct: torch.Tensor, mean: float = -4.0, std: float = 4.0) -> torch.Tensor:
    """StyleTTS2-style normalized log-mel energy/norm target, returns [B,T]."""
    x = torch.nan_to_num(mel_bct.float(), nan=0.0, posinf=0.0, neginf=0.0)
    return torch.log(torch.exp(x * float(std) + float(mean)).norm(dim=1).clamp_min(1e-8))


def _masked_smooth_l1_bt(pred_bt: torch.Tensor, target_bt: torch.Tensor, mask_b1t: torch.Tensor) -> torch.Tensor:
    T = min(int(pred_bt.size(-1)), int(target_bt.size(-1)), int(mask_b1t.size(-1)))
    pred = pred_bt[:, :T].float()
    target = target_bt[:, :T].float()
    mask = mask_b1t[:, 0, :T].float()
    loss = F.smooth_l1_loss(pred, target, reduction="none")
    return (loss * mask).sum() / mask.sum().clamp_min(1.0)


def _crop_or_pad_bt(x_bt: torch.Tensor, T: int) -> torch.Tensor:
    T = int(max(1, T))
    if int(x_bt.size(-1)) > T:
        return x_bt[:, :T].contiguous()
    if int(x_bt.size(-1)) < T:
        return F.pad(x_bt, (0, T - int(x_bt.size(-1))), value=0.0)
    return x_bt


def _load_cached_f0_targets(
    paths: Any,
    *,
    device: torch.device,
    T: int,
    log_hz: bool = False,
) -> Optional[torch.Tensor]:
    if not isinstance(paths, (list, tuple)) or len(paths) <= 0:
        return None
    rows: List[torch.Tensor] = []
    for p in paths:
        if not p:
            return None
        pp = Path(str(p)).expanduser()
        if not pp.is_file():
            return None
        try:
            data = torch.load(str(pp), map_location="cpu")
        except Exception:
            return None
        if isinstance(data, dict):
            x = data.get("f0", data.get("f0_hz", data.get("log_f0", None)))
        else:
            x = data
        if x is None:
            return None
        t = torch.as_tensor(x, dtype=torch.float32).view(-1)
        if bool(log_hz):
            # FCPE cache is expected in Hz; convert only voiced frames and leave unvoiced at zero.
            voiced = t > 0.0
            tt = torch.zeros_like(t)
            tt[voiced] = torch.log(t[voiced].clamp_min(1.0))
            t = tt
        rows.append(_crop_or_pad_bt(t.view(1, -1), int(T)).view(-1))
    if not rows:
        return None
    return torch.stack(rows, dim=0).to(device=device, dtype=torch.float32)


def _cosine_loss(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a = F.normalize(a, dim=-1)
    b = F.normalize(b, dim=-1)
    return (1.0 - (a * b).sum(dim=-1)).mean()


def _select_speaker_vec_256(
    *,
    args,
    mel_flow: nn.Module,
    mel_bct: torch.Tensor,
    T_len: torch.Tensor,
    speaker_emb_b: torch.Tensor,
    speaker_encoder: Optional[nn.Module] = None,
) -> torch.Tensor:
    src = str(getattr(args, "speaker_vector_source", "dataset_centroid")).lower().strip()
    if src == "speaker_encoder":
        if speaker_encoder is None:
            raise RuntimeError("speaker_vector_source=speaker_encoder requires trainable speaker_encoder.")
        T = int(mel_bct.size(-1))
        mask_bt = _make_tmask_from_Tlen(T_len.clamp_max(T), T).squeeze(1).to(dtype=torch.bool, device=mel_bct.device)
        z_ref_spk, _z_ref_style = speaker_encoder(mel_bct.float(), mask_bt=mask_bt)  # type: ignore[misc,call-arg]
        spk_256 = z_ref_spk.float()
    elif src == "gt_dualhead":
        if not bool(getattr(mel_flow, "spk_style_ref_ready", False)):
            raise RuntimeError("speaker_vector_source=gt_dualhead requires loaded frozen dual-head encoder.")
        T = int(mel_bct.size(-1))
        mask_bt = _make_tmask_from_Tlen(T_len.clamp_max(T), T).squeeze(1).to(dtype=torch.bool, device=mel_bct.device)
        with torch.no_grad():
            z_ref_spk, _z_ref_style = mel_flow.encode_ref_dual(  # type: ignore[attr-defined]
                mel_bct.float(),
                mask_bt=mask_bt,
            )
        spk_256 = z_ref_spk.detach().float()
    else:
        spk_256 = speaker_emb_b.to(dtype=torch.float32)
    return spk_256 / spk_256.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def _select_style_vec_128(
    *,
    args,
    mel_flow: nn.Module,
    mel_bct: torch.Tensor,
    T_len: torch.Tensor,
    speaker_encoder: Optional[nn.Module] = None,
) -> Optional[torch.Tensor]:
    """Return the dualhead style/prosody head for duration adapters when available."""
    src = str(getattr(args, "speaker_vector_source", "dataset_centroid")).lower().strip()
    if src == "speaker_encoder":
        if speaker_encoder is None:
            return None
        T = int(mel_bct.size(-1))
        mask_bt = _make_tmask_from_Tlen(T_len.clamp_max(T), T).squeeze(1).to(dtype=torch.bool, device=mel_bct.device)
        _z_ref_spk, z_ref_style = speaker_encoder(mel_bct.float(), mask_bt=mask_bt)  # type: ignore[misc,call-arg]
        return z_ref_style.float()
    if src == "gt_dualhead" and bool(getattr(mel_flow, "spk_style_ref_ready", False)):
        T = int(mel_bct.size(-1))
        mask_bt = _make_tmask_from_Tlen(T_len.clamp_max(T), T).squeeze(1).to(dtype=torch.bool, device=mel_bct.device)
        with torch.no_grad():
            _z_ref_spk, z_ref_style = mel_flow.encode_ref_dual(  # type: ignore[attr-defined]
                mel_bct.float(),
                mask_bt=mask_bt,
            )
        return z_ref_style.detach().float()
    return None


def _apply_emotion_style_conditioning(
    style_128: Optional[torch.Tensor],
    emotion_ids: torch.Tensor,
    *,
    enabled: bool,
    emotion_embed: nn.Module,
    emotion_to_style: nn.Module,
    emotion_style_gate: torch.Tensor,
) -> Optional[torch.Tensor]:
    if not bool(enabled):
        return style_128
    emo = emotion_embed(emotion_ids.clamp(0, len(EMOTION_GROUP_TO_ID) - 1))
    emo_style = emotion_to_style(emo.to(dtype=torch.float32))
    g = emotion_style_gate.to(device=emo_style.device, dtype=emo_style.dtype)
    if style_128 is None:
        base = torch.zeros_like(emo_style)
    else:
        base = style_128.to(device=emo_style.device, dtype=emo_style.dtype)
    return base + g * emo_style


def _kl_diag_gauss_to_std_normal(mu: torch.Tensor, logs: torch.Tensor) -> torch.Tensor:
    # KL(N(mu, sigma) || N(0,1)) for diagonal Gauss
    # = 0.5 * mean( sigma^2 + mu^2 - 1 - log(sigma^2) )
    return 0.5 * (torch.exp(2.0 * logs) + mu * mu - 1.0 - 2.0 * logs).mean()


class LayerNorm1d(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(int(dim)))
        self.bias = nn.Parameter(torch.zeros(int(dim)))
        self.eps = float(eps)

    def forward(self, x_bct: torch.Tensor) -> torch.Tensor:
        x = x_bct.transpose(1, 2)
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        x = (x - mean) / torch.sqrt(var + self.eps)
        x = x * self.weight + self.bias
        return x.transpose(1, 2).contiguous()


class ConvNeXtBlock1d(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        dim = int(dim)
        self.dwconv = nn.Conv1d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = LayerNorm1d(dim)
        self.pwconv1 = nn.Conv1d(dim, 4 * dim, kernel_size=1)
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv1d(4 * dim, dim, kernel_size=1)

    def forward(self, x_bct: torch.Tensor) -> torch.Tensor:
        residual = x_bct
        x = self.dwconv(x_bct)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        return residual + x


class AttentivePooling(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        dim = int(dim)
        self.query = nn.Parameter(torch.randn(1, dim, 1))
        self.key = nn.Conv1d(dim, dim, kernel_size=1)
        self.val = nn.Conv1d(dim, dim, kernel_size=1)
        self.scale = dim ** -0.5

    def forward(self, x_bct: torch.Tensor, mask_bt: Optional[torch.Tensor] = None) -> torch.Tensor:
        q = self.query
        k = self.key(x_bct)
        v = self.val(x_bct)
        attn = (q * k).sum(dim=1, keepdim=True) * self.scale
        if mask_bt is not None:
            m = mask_bt.to(dtype=torch.bool, device=x_bct.device).unsqueeze(1)
            attn = attn.masked_fill(~m, -1e9)
            # all-masked fallback
            valid = m.any(dim=-1, keepdim=True)
            attn = torch.where(valid, attn, torch.zeros_like(attn))
        w = F.softmax(attn, dim=-1)
        return (v * w).sum(dim=-1)


class AttentiveStatsPooling(nn.Module):
    """
    ECAPA-like attentive statistics pooling.
    Returns concatenated weighted mean and weighted std: [B, 2*C].
    """
    def __init__(self, in_dim: int, head_dim: int = 128):
        super().__init__()
        in_dim = int(in_dim)
        head_dim = int(head_dim)
        self.attn = nn.Sequential(
            nn.Conv1d(in_dim, head_dim, kernel_size=1),
            nn.Tanh(),
            nn.Conv1d(head_dim, in_dim, kernel_size=1),
        )

    def forward(self, x_bct: torch.Tensor, mask_bt: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x_bct: [B, C, T]
        logits = self.attn(x_bct)  # [B, C, T]
        if mask_bt is not None:
            m = mask_bt.to(dtype=torch.bool, device=x_bct.device).unsqueeze(1)  # [B,1,T]
            logits = logits.masked_fill(~m, -1e9)
            valid = m.any(dim=-1, keepdim=True)
            logits = torch.where(valid, logits, torch.zeros_like(logits))
        w = F.softmax(logits, dim=-1)  # [B, C, T]
        mu = (x_bct * w).sum(dim=-1)  # [B, C]
        x2 = ((x_bct * x_bct) * w).sum(dim=-1)  # [B, C]
        var = (x2 - mu * mu).clamp_min(1e-9)
        std = torch.sqrt(var)
        return torch.cat([mu, std], dim=-1)


class MelSpeakerEncoder(nn.Module):
    """
    Lightweight mel-audio speaker encoder:
      mel[B,C,T] -> z_spk[B,d_spk] (L2-normalized)

    Uses ConvNeXt1d blocks + attentive stats pooling (ECAPA-like).
    """

    def __init__(
        self,
        *,
        n_mels: int,
        d_spk: int = 256,
        hidden_dim: int = 256,
        num_layers: int = 4,
        attn_head_dim: int = 128,
    ):
        super().__init__()
        n_mels = int(n_mels)
        self.d_spk = int(d_spk)
        hidden_dim = int(hidden_dim)
        num_layers = int(max(1, num_layers))
        attn_head_dim = int(max(8, attn_head_dim))

        self.pre_conv = nn.Sequential(
            nn.Conv1d(n_mels, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.blocks = nn.ModuleList([ConvNeXtBlock1d(hidden_dim) for _ in range(num_layers)])
        self.pooling = AttentiveStatsPooling(hidden_dim * num_layers, head_dim=attn_head_dim)

        pooled_dim = 2 * (hidden_dim * num_layers)
        hid = int(max(256, self.d_spk * 2))
        self.mlp = nn.Sequential(
            nn.LayerNorm(pooled_dim),
            nn.Linear(pooled_dim, hid),
            nn.GELU(),
            nn.Linear(hid, self.d_spk),
        )

    def forward(self, mel_bct: torch.Tensor, *, mask_bt: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.pre_conv(mel_bct)
        outs = []
        for blk in self.blocks:
            x = blk(x)
            outs.append(x)
        x_agg = torch.cat(outs, dim=1)
        feat = self.pooling(x_agg, mask_bt=mask_bt)
        z = self.mlp(feat)
        z = z / z.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        return z


class SpeakerTimeAdaptiveLayerNorm(nn.Module):
    """AdaLN used in the active runtime: speaker + flow-time conditioning only."""

    def __init__(self, dim: int, init_spk_gain: float = 0.30, init_t_gain: float = 1.00):
        super().__init__()
        dim = int(dim)
        self.ln = nn.LayerNorm(dim)
        self.spk_scale = nn.Linear(dim, dim)
        self.spk_shift = nn.Linear(dim, dim)
        self.t_scale = nn.Linear(dim, dim)
        self.t_shift = nn.Linear(dim, dim)
        self.g_spk = nn.Parameter(torch.tensor(float(init_spk_gain)))
        self.g_t = nn.Parameter(torch.tensor(float(init_t_gain)))

    def forward(
        self,
        x_btd: torch.Tensor,
        spk_vec_bd: torch.Tensor,
        t_vec_bd: Optional[torch.Tensor],
    ) -> torch.Tensor:
        x = self.ln(x_btd)
        spk_scale = self.spk_scale(spk_vec_bd)[:, None, :]
        spk_shift = self.spk_shift(spk_vec_bd)[:, None, :]
        scale = 1.0 + self.g_spk * spk_scale
        shift = self.g_spk * spk_shift
        if t_vec_bd is not None:
            t_scale = self.t_scale(t_vec_bd)[:, None, :]
            t_shift = self.t_shift(t_vec_bd)[:, None, :]
            scale = scale + self.g_t * t_scale
            shift = shift + self.g_t * t_shift
        return x * scale + shift


class MelFlowDecoder(nn.Module):
    """
    Mel-domain rectified flow: v(x_t, t, cond) ~ (x1 - x0)
    Wejścia:
      x_bct: [B,C,T]
      t_b: [B]
      speaker_ids: [B]
      text_seq_b: [B,L,D] lub None
    Wyjście:
      v_bct: [B,C,T]
    """
    def __init__(
        self,
        dim: int,
        text_dim: int,
        n_mels: int,
        heads: int = 8,
        layers: int = 6,
        num_speakers: int = 256,
        conv_act: str = "gelu",
        use_timbre_adaln: bool = False,
        dualpath_branch_dim: Optional[int] = None,
        dualpath_attn_dim: Optional[int] = None,
        dualpath_conv_dim: Optional[int] = None,
        dualpath_init_split_identity: bool = True,
    ):
        super().__init__()
        self.dim = dim
        self.n_mels = n_mels
        self.spk_style_ref: Optional[nn.Module] = None  # frozen dualhead speaker reference encoder
        self.spk_style_ref_ready: bool = False

        self.mel_in = nn.Conv1d(n_mels, dim, kernel_size=1)
        self.mel_out = nn.Conv1d(dim, n_mels, kernel_size=1)

        self.spk = nn.Embedding(int(num_speakers), dim)
        self.spk_mlp = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.t_mlp = nn.Sequential(nn.Linear(128, dim), nn.SiLU(), nn.Linear(dim, dim))

        self.cross_ln = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, batch_first=True)
        self.text_proj = nn.Linear(text_dim, dim) if text_dim != dim else nn.Identity()

        self.blocks = nn.ModuleList([
            DualPathProjectedBlock(
                dim,
                num_heads=heads,
                use_sdpa=True,
                use_adaln=True,
                use_dual_adaln=True,
                cond_dim=dim,
                conv_act=str(conv_act),
                branch_dim=dualpath_branch_dim,
                attn_dim=dualpath_attn_dim,
                conv_dim=dualpath_conv_dim,
                init_split_identity=dualpath_init_split_identity,
            )
            for _ in range(layers)
        ])
        self.dual_adaln_in = SpeakerTimeAdaptiveLayerNorm(
            dim=int(dim),
            init_spk_gain=0.30,
        )

    @torch.no_grad()
    def encode_ref_dual(
        self,
        mel_bct: torch.Tensor,
        *,
        mask_bt: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode reference mel into (z_spk_256, z_style_128) using a frozen dualhead speaker encoder.
        """
        if self.spk_style_ref is None or (not bool(getattr(self, "spk_style_ref_ready", False))):
            raise RuntimeError("spk_style_ref encoder is not ready (missing --spk-style-ckpt and no resume weights).")
        z_spk, z_style = self.spk_style_ref(mel_bct, mask_bt=mask_bt)  # type: ignore[misc,call-arg]
        return z_spk, z_style

    def forward(
        self,
        x_bct: torch.Tensor,
        t_b: torch.Tensor,
        speaker_ids: torch.Tensor,
        text_seq_b: Optional[torch.Tensor],
        spk_vec_override: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = self.mel_in(x_bct).transpose(1, 2).contiguous()  # [B,T,D]

        t_emb = self.t_mlp(_sinusoidal_t(t_b.to(x.dtype), 128))
        if spk_vec_override is None:
            raw_spk = self.spk(speaker_ids)
        else:
            raw_spk = spk_vec_override
            if raw_spk.size(-1) != self.dim:
                raise RuntimeError(
                    f"spk_vec_override dim={int(raw_spk.size(-1))} incompatible with mel_flow dim={int(self.dim)}"
                )
        spk_emb = self.spk_mlp(raw_spk.to(x.dtype))
        x = self.dual_adaln_in(x, spk_emb, t_emb)

        if text_seq_b is not None:
            txt = self.text_proj(text_seq_b.to(x.dtype))
            x = x + self.cross_attn(self.cross_ln(x), txt, txt, need_weights=False)[0]

        for blk in self.blocks:
            x = blk(x, None, t_emb=t_emb, spk_emb=spk_emb, style_vec=None)

        return self.mel_out(x.transpose(1, 2).contiguous())  # [B,C,T]


class GaussianTokenAttentionDualPathBlock(DualPathProjectedBlock):
    """DualPath block variant that replaces frame-level self-attn with Gaussian token-slot self-attn.

    Frame states are pooled to token slots with the prior upsampling map, self-attention
    runs over the short token sequence, then token states are scattered back to frames.
    The convolution branch and branch gates stay identical to DualPathProjectedBlock.
    """

    def __init__(self, *args, token_group_size: int = 1, **kwargs):
        super().__init__(*args, **kwargs)
        self.token_group_size = max(1, int(token_group_size))

    def _group_token_slots(
        self,
        token_slots: torch.Tensor,
        token_pad: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        g = int(getattr(self, "token_group_size", 1))
        if g <= 1:
            B = int(token_slots.size(0))
            L = int(token_slots.size(1))
            group_to_token = torch.eye(L, device=token_slots.device, dtype=token_slots.dtype)[None, :, :].expand(B, L, L)
            return token_slots, token_pad, group_to_token

        B, L, A = token_slots.shape
        G = (L + g - 1) // g
        pad_l = G * g - L
        if pad_l > 0:
            token_slots_p = F.pad(token_slots, (0, 0, 0, pad_l))
            token_pad_p = F.pad(token_pad, (0, pad_l), value=True)
        else:
            token_slots_p = token_slots
            token_pad_p = token_pad

        slots_bgga = token_slots_p.view(B, G, g, A)
        pad_bgg = token_pad_p.view(B, G, g)
        valid = (~pad_bgg).to(dtype=token_slots.dtype)
        count = valid.sum(dim=2)
        group_slots = (slots_bgga * valid[:, :, :, None]).sum(dim=2) / count.clamp_min(1.0)[:, :, None]
        group_pad = count <= 0.0

        group_to_token = token_slots.new_zeros(B, G, L)
        for gi in range(G):
            start = gi * g
            end = min(L, start + g)
            if end > start:
                group_to_token[:, gi, start:end] = 1.0
        return group_slots, group_pad, group_to_token

    def _token_slot_attention(
        self,
        x_attn: torch.Tensor,
        *,
        gauss_align_btl: Optional[torch.Tensor],
        frame_pad_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if gauss_align_btl is None:
            return self.attn(x_attn, key_padding_mask=frame_pad_mask)

        B, T, A = x_attn.shape
        align = gauss_align_btl.to(device=x_attn.device, dtype=x_attn.dtype)
        Tm = min(int(T), int(align.size(1)))
        if Tm <= 0 or int(align.size(2)) <= 0:
            return self.attn(x_attn, key_padding_mask=frame_pad_mask)

        align = align[:, :Tm, :]
        x_src = x_attn[:, :Tm, :]
        token_mass = align.sum(dim=1)  # [B,L]
        token_pad = token_mass <= 1e-6
        token_slots = torch.einsum("btl,bta->bla", align, x_src) / token_mass.clamp_min(1e-6)[:, :, None]
        token_slots = torch.where(token_pad[:, :, None], token_slots.new_zeros(()), token_slots)
        group_slots, group_pad, group_to_token = self._group_token_slots(token_slots, token_pad)
        group_mixed = self.attn(group_slots, key_padding_mask=group_pad)
        token_denom = group_to_token.sum(dim=1).clamp_min(1.0)
        token_mixed = torch.einsum("bgl,bga->bla", group_to_token, group_mixed) / token_denom[:, :, None]
        token_mixed = torch.where(token_pad[:, :, None], token_mixed.new_zeros(()), token_mixed)
        a_part = torch.einsum("btl,bla->bta", align, token_mixed)

        if Tm == T:
            return a_part
        out = x_attn.new_zeros(B, T, A)
        out[:, :Tm, :] = a_part
        return out

    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor | None = None,
        t_emb: torch.Tensor | None = None,
        spk_emb: torch.Tensor | None = None,
        style_vec: torch.Tensor | None = None,
        key_padding_mask: torch.Tensor | None = None,
        gauss_align_btl: Optional[torch.Tensor] = None,
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

        a = self._token_slot_attention(x_attn, gauss_align_btl=gauss_align_btl, frame_pad_mask=pad_mask)
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


class SpeakerlessMelFlowDecoder(MelFlowDecoder):
    """
    Variant for prefix-only speaker conditioning:
      - speaker identity enters only through the text/prefix path
      - mel-flow decoder itself is time-conditioned only
    """

    class DecoderCrossBlock(nn.Module):
        """One decoder block with text cross-attention before the time-only main block."""

        def __init__(
            self,
            dim: int,
            heads: int,
            conv_act: str = "gelu",
            dualpath_branch_dim: Optional[int] = None,
            dualpath_attn_dim: Optional[int] = None,
            dualpath_conv_dim: Optional[int] = None,
            dualpath_init_split_identity: bool = True,
            use_gauss_token_attn: bool = False,
            use_gauss_cross_attn: bool = False,
            gauss_token_group_size: int = 1,
        ):
            super().__init__()
            self.cross_ln = nn.LayerNorm(int(dim))
            self.cross_attn = nn.MultiheadAttention(embed_dim=int(dim), num_heads=int(heads), batch_first=True)
            block_cls = GaussianTokenAttentionDualPathBlock if bool(use_gauss_token_attn) else DualPathProjectedBlock
            block_kwargs = {}
            if bool(use_gauss_token_attn):
                block_kwargs["token_group_size"] = int(gauss_token_group_size)
            self.use_gauss_token_attn = bool(use_gauss_token_attn)
            self.use_gauss_cross_attn = bool(use_gauss_cross_attn)
            self.main = block_cls(
                int(dim),
                num_heads=int(heads),
                use_sdpa=True,
                use_adaln=False,
                use_dual_adaln=False,
                use_time_adaln=True,
                cond_dim=int(dim),
                conv_act=str(conv_act),
                branch_dim=dualpath_branch_dim,
                attn_dim=dualpath_attn_dim,
                conv_dim=dualpath_conv_dim,
                init_split_identity=dualpath_init_split_identity,
                **block_kwargs,
            )

        def _gauss_cross_attn(
            self,
            x_btd: torch.Tensor,
            text_seq_b: torch.Tensor,
            gauss_align_btl: Optional[torch.Tensor],
        ) -> torch.Tensor:
            if gauss_align_btl is None:
                return x_btd + self.cross_attn(self.cross_ln(x_btd), text_seq_b, text_seq_b, need_weights=False)[0]

            B, T, D = x_btd.shape
            align = gauss_align_btl.to(device=x_btd.device, dtype=x_btd.dtype)
            Tm = min(int(T), int(align.size(1)))
            if Tm <= 0 or int(align.size(2)) <= 0:
                return x_btd + self.cross_attn(self.cross_ln(x_btd), text_seq_b, text_seq_b, need_weights=False)[0]

            align = align[:, :Tm, :]
            x_src = x_btd[:, :Tm, :]
            token_mass = align.sum(dim=1)  # [B,L]
            token_pad = token_mass <= 1e-6
            token_slots = torch.einsum("btl,btd->bld", align, x_src) / token_mass.clamp_min(1e-6)[:, :, None]
            token_slots = torch.where(token_pad[:, :, None], token_slots.new_zeros(()), token_slots)

            token_update = self.cross_attn(
                self.cross_ln(token_slots),
                text_seq_b,
                text_seq_b,
                key_padding_mask=None,
                need_weights=False,
            )[0]
            token_update = torch.where(token_pad[:, :, None], token_update.new_zeros(()), token_update)
            frame_update = torch.einsum("btl,bld->btd", align, token_update)

            out = x_btd.clone()
            out[:, :Tm, :] = out[:, :Tm, :] + frame_update
            return out

        def forward(
            self,
            x_btd: torch.Tensor,
            text_seq_b: Optional[torch.Tensor],
            t_emb: Optional[torch.Tensor],
            gauss_align_btl: Optional[torch.Tensor] = None,
        ) -> torch.Tensor:
            if text_seq_b is not None:
                if self.use_gauss_cross_attn:
                    x_btd = self._gauss_cross_attn(x_btd, text_seq_b, gauss_align_btl)
                else:
                    x_btd = x_btd + self.cross_attn(self.cross_ln(x_btd), text_seq_b, text_seq_b, need_weights=False)[0]
            if self.use_gauss_token_attn:
                return self.main(x_btd, cond=t_emb, t_emb=None, spk_emb=None, style_vec=None, gauss_align_btl=gauss_align_btl)
            return self.main(x_btd, cond=t_emb, t_emb=None, spk_emb=None, style_vec=None)

    def __init__(
        self,
        dim: int,
        text_dim: int,
        n_mels: int,
        heads: int = 8,
        layers: int = 6,
        num_speakers: int = 256,
        conv_act: str = "gelu",
        use_timbre_adaln: bool = False,
        dualpath_branch_dim: Optional[int] = None,
        dualpath_attn_dim: Optional[int] = None,
        dualpath_conv_dim: Optional[int] = None,
        dualpath_init_split_identity: bool = True,
        flow_gauss_token_attn: bool = False,
        flow_gauss_cross_attn: bool = False,
        flow_gauss_token_group_size: int = 1,
    ):
        super().__init__(
            dim=dim,
            text_dim=text_dim,
            n_mels=n_mels,
            heads=heads,
            layers=layers,
            num_speakers=num_speakers,
            conv_act=conv_act,
            dualpath_branch_dim=dualpath_branch_dim,
            dualpath_attn_dim=dualpath_attn_dim,
            dualpath_conv_dim=dualpath_conv_dim,
            dualpath_init_split_identity=dualpath_init_split_identity,
        )
        # Prefix-only variant does not use direct speaker conditioning inside mel-flow.
        del self.spk
        del self.spk_mlp
        self.blocks = nn.ModuleList([
            self.DecoderCrossBlock(
                dim=int(dim),
                heads=int(heads),
                conv_act=str(conv_act),
                dualpath_branch_dim=dualpath_branch_dim,
                dualpath_attn_dim=dualpath_attn_dim,
                dualpath_conv_dim=dualpath_conv_dim,
                dualpath_init_split_identity=dualpath_init_split_identity,
                use_gauss_token_attn=bool(flow_gauss_token_attn),
                use_gauss_cross_attn=bool(flow_gauss_cross_attn),
                gauss_token_group_size=int(flow_gauss_token_group_size),
            )
            for _ in range(layers)
        ])
        self.flow_gauss_token_attn = bool(flow_gauss_token_attn)
        self.flow_gauss_cross_attn = bool(flow_gauss_cross_attn)
        self.flow_gauss_token_group_size = int(flow_gauss_token_group_size)
        self.timbre_adaln = nn.ModuleList([
            GatedStyleAdaLN(dim=int(dim), style_dim=int(dim))
            for _ in range(layers)
        ])
        self.use_timbre_adaln = bool(use_timbre_adaln)
        self.dual_adaln_in = TimeAdaptiveLayerNorm(dim=int(dim))

    def forward(
        self,
        x_bct: torch.Tensor,
        t_b: torch.Tensor,
        speaker_ids: torch.Tensor,
        text_seq_b: Optional[torch.Tensor],
        spk_vec_override: Optional[torch.Tensor] = None,
        gauss_align_btl: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = self.mel_in(x_bct).transpose(1, 2).contiguous()  # [B,T,D]

        t_emb = self.t_mlp(_sinusoidal_t(t_b.to(x.dtype), 128))
        x = self.dual_adaln_in(x, t_emb)
        txt = self.text_proj(text_seq_b.to(x.dtype)) if text_seq_b is not None else None
        timbre_vec = None
        if spk_vec_override is not None and int(spk_vec_override.size(-1)) == int(self.dim):
            timbre_vec = spk_vec_override.to(device=x.device, dtype=x.dtype)
        for i, blk in enumerate(self.blocks):
            if self.use_timbre_adaln and timbre_vec is not None and i < len(self.timbre_adaln):
                x = self.timbre_adaln[i](x, timbre_vec)
            use_gauss_map = bool(self.flow_gauss_token_attn or self.flow_gauss_cross_attn)
            x = blk(x, txt, t_emb, gauss_align_btl=gauss_align_btl if use_gauss_map else None)

        return self.mel_out(x.transpose(1, 2).contiguous())  # [B,C,T]


# ---------------- DURATION HELPERS ----------------


def _predict_dur(
    model: nn.Module,
    x_tok: torch.Tensor,
    tok_pad: torch.Tensor,
    special_len: int,
    spk_embed: nn.Module,
    speaker_ids: torch.Tensor,
    flow_cond: Optional[torch.Tensor],
    spk_vec_override: Optional[torch.Tensor] = None,
    require_spk_override: bool = True,
    style_vec: Optional[torch.Tensor] = None,
    dur_x0_mode: str = "prior",
    dur_x0_noise_scale: float = 1.0,
    dur_prior_logs_min: float = -5.0,
    dur_prior_logs_max: float = 2.0,
    dur_prior_sigma_min: float = 0.1,
    dur_sigma0_demo: float = 0.0,
    *,
    steps_override: Optional[int] = None,
    noise_scale_override: Optional[float] = None,
    dur_flow_clip_sigma: float = 0.0,
    dur_flow_clip_abs_min: Optional[float] = None,
    dur_flow_clip_abs_max: Optional[float] = None,
    dur_flow_fix_total: bool = False,
    dur_flow_fix_total_mode: str = "prior_mu",
    initial_hc: "tuple | None" = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    ids_full = F.pad(tok_pad, (int(special_len), 0), value=PAD_ID)
    dur_allowed = _build_dur_allowed_mask(ids_full, special_len)
    x_mask = dur_allowed.float().unsqueeze(1)  # [B,1,L] (tylko tekst + pauzy)
    if hasattr(model.dur, "predict_logdur"):
        dur_logits, _final_hc = model.dur.predict_logdur(  # type: ignore[attr-defined]
            x_tok.float(),
            x_mask.float(),
            initial_hc=initial_hc,
            style_vec=style_vec,
        )
        _predict_dur._last_hc = _final_hc  # przekaż h_c wywołującemu przez atrybut funkcji
        dur_pred = (torch.exp(dur_logits) - 1.0).clamp_min(0.0)
        dur_pred = torch.where(dur_allowed, dur_pred, torch.zeros_like(dur_pred))
        dur_pred = torch.where(ids_full == PAD_ID, torch.zeros_like(dur_pred), dur_pred)
        pause_mask = _pause_mask_from_ids(ids_full)
        text_mask = dur_allowed & (~pause_mask)
        dur_pred = torch.where(text_mask, dur_pred.clamp_min(1.0), dur_pred)
        dur_pred = torch.where(pause_mask, dur_pred.clamp_min(1.0), dur_pred)
        return dur_pred, ids_full, dur_logits
    if spk_vec_override is None:
        if bool(require_spk_override):
            raise RuntimeError(
                "Speaker override is required (require_spk_override=True) but spk_vec_override=None in _predict_dur. "
                "This run is configured to never fall back to spk_embed(speaker_id)."
            )
        spk_vec = spk_embed(speaker_ids)
    else:
        spk_vec = spk_vec_override

    if (
        str(dur_x0_mode).lower().strip() == "prior"
        and hasattr(model.dur, "sample_with_prior_x0")
    ):
        dur_logits = model.dur.sample_with_prior_x0(  # type: ignore[attr-defined]
            x_tok_bld=x_tok,
            x_mask_b1l=x_mask,
            spk_emb_bd=spk_vec,
            style_vec_bd=style_vec,
            cond=flow_cond,
            steps=int(CONFIG["dur_flow_steps"] if steps_override is None else int(steps_override)),
            noise_scale=float(CONFIG["dur_flow_noise_scale"] if noise_scale_override is None else float(noise_scale_override)),
            use_heun=True,
            prior_logs_min=float(dur_prior_logs_min),
            prior_logs_max=float(dur_prior_logs_max),
            prior_sigma_min=float(dur_prior_sigma_min),
            sigma0_demo=float(dur_sigma0_demo),
            clip_sigma=float(dur_flow_clip_sigma),
            clip_abs_min=(None if dur_flow_clip_abs_min is None else float(dur_flow_clip_abs_min)),
            clip_abs_max=(None if dur_flow_clip_abs_max is None else float(dur_flow_clip_abs_max)),
        )
    else:
        dur_logits = model.dur.sample(
            x_tok,
            x_mask,
            spk_emb=spk_vec,
            style_vec=style_vec,
            cond=flow_cond,
            steps=int(CONFIG["dur_flow_steps"] if steps_override is None else int(steps_override)),
            noise_scale=float(CONFIG["dur_flow_noise_scale"] if noise_scale_override is None else float(noise_scale_override)),
            use_heun=True,
            clip_abs_min=(None if dur_flow_clip_abs_min is None else float(dur_flow_clip_abs_min)),
            clip_abs_max=(None if dur_flow_clip_abs_max is None else float(dur_flow_clip_abs_max)),
        )
    dur_pred = (torch.exp(dur_logits) - 1.0).clamp_min(0.0)
    # dur=0 dla PAD oraz dla tokenów nie-tekstowych (interpunkcja/role/prosody/specjalne)
    dur_pred = torch.where(dur_allowed, dur_pred, torch.zeros_like(dur_pred))
    dur_pred = torch.where(ids_full == PAD_ID, torch.zeros_like(dur_pred), dur_pred)

    # wymuś minimum 1 ramkę na tokenach pauzy, żeby cisza nie znikała przy predykcji duracji
    pause_mask = _pause_mask_from_ids(ids_full)
    dur_pred = torch.where(pause_mask, dur_pred.clamp_min(1.0), dur_pred)
    return dur_pred, ids_full, dur_logits


@torch.no_grad()
def _predict_dur_prior_direct(
    model: nn.Module,
    x_tok: torch.Tensor,
    tok_pad: torch.Tensor,
    special_len: int,
    *,
    source: str = "prior_mu",
    noise_scale: float = 0.0,
    dur_prior_logs_min: float = -5.0,
    dur_prior_logs_max: float = 2.0,
    dur_prior_sigma_min: float = 0.1,
    initial_hc: "tuple | None" = None,
    style_vec: Optional[torch.Tensor] = None,
    rhythm_state: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    ids_full = F.pad(tok_pad, (int(special_len), 0), value=PAD_ID)
    dur_allowed = _build_dur_allowed_mask(ids_full, special_len)
    x_mask = dur_allowed.float().unsqueeze(1)
    if hasattr(model.dur, "predict_durations"):
        dur_pred = model.dur.predict_durations(  # type: ignore[attr-defined]
            x_tok.float(), x_mask.float(), token_ids=ids_full,
            style_vec=style_vec, rhythm_state=rhythm_state,
        )
        dur_logits = torch.log1p(dur_pred.clamp_min(0.0))
        _predict_dur_prior_direct._last_hc = None
    elif hasattr(model.dur, "predict_logdur"):
        dur_logits, _final_hc = model.dur.predict_logdur(  # type: ignore[attr-defined]
            x_tok.float(),
            x_mask.float(),
            initial_hc=initial_hc,
            style_vec=style_vec,
        )
        _predict_dur_prior_direct._last_hc = _final_hc  # przekaż h_c wywołującemu przez atrybut funkcji
    else:
        if not hasattr(model.dur, "prior") or model.dur.prior is None:  # type: ignore[attr-defined]
            raise RuntimeError("deterministic duration requires FlowDurationPredictor prior head or ARDurationTransformer.")
        mu_d, logs_d = model.dur.prior(x_tok.float(), x_mask=x_mask)  # type: ignore[attr-defined]
        logs_d = logs_d.clamp(min=float(dur_prior_logs_min), max=float(dur_prior_logs_max))
        dur_logits = mu_d
        if str(source).lower().strip() == "prior_sample":
            sigma_d = torch.exp(logs_d).clamp_min(1e-8)
            sigma_min = float(dur_prior_sigma_min)
            if sigma_min > 0.0:
                sigma_d = sigma_d.clamp_min(sigma_min)
            dur_logits = mu_d + torch.randn_like(mu_d) * sigma_d * float(noise_scale)
    dur_logits = dur_logits * x_mask.squeeze(1)
    dur_pred = (torch.exp(dur_logits) - 1.0).clamp_min(0.0)
    dur_pred = torch.where(dur_allowed, dur_pred, torch.zeros_like(dur_pred))
    dur_pred = torch.where(ids_full == PAD_ID, torch.zeros_like(dur_pred), dur_pred)
    pause_mask = _pause_mask_from_ids(ids_full)
    text_mask = dur_allowed & (~pause_mask)
    dur_pred = torch.where(text_mask, dur_pred.clamp_min(1.0), dur_pred)
    dur_pred = torch.where(pause_mask, dur_pred.clamp_min(1.0), dur_pred)
    return dur_pred, ids_full, dur_logits


def _zero_spk_cond_like(x_bd: torch.Tensor) -> torch.Tensor:
    return torch.zeros_like(x_bd, memory_format=torch.contiguous_format)


def _select_dec_durations(L_gt_full: torch.Tensor, dur_pred: Optional[torch.Tensor], dec_src: str) -> torch.Tensor:
    if dec_src == "gt":
        return L_gt_full
    if dec_src == "pred" and dur_pred is not None:
        return dur_pred
    return L_gt_full


@torch.no_grad()
def sample_mel_flow(
    mel_flow: nn.Module,
    x0_bct: torch.Tensor,
    text_seq_b: Optional[torch.Tensor],
    speaker_ids: torch.Tensor,
    steps: int = 8,
    spk_vec_override: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Euler sampler: x_{k+1} = x_k + v(x_k,t_k)*dt"""
    x = x0_bct
    dt = 1.0 / float(max(1, steps))
    for k in range(int(steps)):
        t = torch.full((x.size(0),), float(k) / float(max(1, steps)), device=x.device, dtype=x.dtype)
        v = mel_flow(
            x,
            t,
            speaker_ids,
            text_seq_b,
            spk_vec_override=spk_vec_override,
        )
        x = x + v * dt
    return x


@torch.no_grad()
def sample_mel_flow_twopass(
    mel_flow: nn.Module,
    x0_bct: torch.Tensor,
    text_seq_b: Optional[torch.Tensor],
    speaker_ids: torch.Tensor,
    *,
    steps_first: int,
    steps_second: int,
    t_noise: float,
    spk_vec_override: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    mel_first = sample_mel_flow(
        mel_flow,
        x0_bct,
        text_seq_b,
        speaker_ids,
        steps=int(steps_first),
        spk_vec_override=spk_vec_override,
    )
    if int(steps_second) <= 0 or float(t_noise) <= 0.0:
        return mel_first
    mix = float(max(0.0, min(1.0, float(t_noise))))
    x_start = (1.0 - mix) * mel_first + mix * x0_bct
    return sample_mel_flow(
        mel_flow,
        x_start,
        text_seq_b,
        speaker_ids,
        steps=int(steps_second),
        spk_vec_override=spk_vec_override,
    )


@torch.no_grad()
def sample_mel_flow_with_prefix_twopass(
    mel_flow: nn.Module,
    x0_bct: torch.Tensor,
    text_seq_b: Optional[torch.Tensor],
    speaker_ids: torch.Tensor,
    *,
    steps_first: int,
    steps_second: int,
    t_noise: float,
    spk_vec_override: Optional[torch.Tensor],
    prefix_tail_bct: torch.Tensor,
    prefix_k: int,
) -> torch.Tensor:
    mel_first = sample_mel_flow_with_prefix(
        mel_flow,
        x0_bct,
        text_seq_b,
        speaker_ids,
        steps=int(steps_first),
        spk_vec_override=spk_vec_override,
        prefix_tail_bct=prefix_tail_bct,
        prefix_k=int(prefix_k),
    )
    if int(steps_second) <= 0 or float(t_noise) <= 0.0:
        return mel_first
    mix = float(max(0.0, min(1.0, float(t_noise))))
    x_start = (1.0 - mix) * mel_first + mix * x0_bct
    return sample_mel_flow_with_prefix(
        mel_flow,
        x_start,
        text_seq_b,
        speaker_ids,
        steps=int(steps_second),
        spk_vec_override=spk_vec_override,
        prefix_tail_bct=prefix_tail_bct,
        prefix_k=int(prefix_k),
    )


# ---------------- DATASET WRAPPER ----------------
class AlignedTTSDatasetWithAudio(AlignedTTSDataset):
    def __init__(
        self,
        json_path: str,
        max_items: Optional[int] = None,
        dur_source: str = "ctc",
        dur_field: str = "dur_tok_frames",
        prior_f0_cache_field: str = "fcpe_f0_path",
    ):
        super().__init__(json_path, max_items=max_items, dur_source=dur_source, dur_field=dur_field)
        self.prior_f0_cache_field = str(prior_f0_cache_field or "fcpe_f0_path")

    def __getitem__(self, idx: int) -> Dict:
        # `AlignedTTSDataset.__getitem__` returns tensors only; keep manifest metadata from `self.items`.
        item = self.items[idx]
        out = super().__getitem__(idx)
        audio_path = item.get("audio_path", item.get("path", item.get("audio", None)))
        out["audio_path"] = audio_path
        out["utt_id"] = item.get("utt_id", item.get("id", ""))
        out["speaker_name"] = item.get("speaker_name", item.get("speaker", None))
        out["speaker_embeds"] = item.get("speaker_embeds", None)
        out["fcpe_f0_path"] = item.get(
            self.prior_f0_cache_field,
            item.get("fcpe_f0_path", item.get("f0_fcpe_path", item.get("f0_path", None))),
        )
        out["gender_id"] = int(item.get("gender_id", _gender_id_from_name(item.get("speaker_name", item.get("author", None)))))
        out["emotion_id"] = _emotion_id_from_item(item)
        out["emotion_group"] = str(item.get("emotion_group", item.get("emotion", "neutral")) or "neutral")
        return out


class AlignedTTSDatasetWithAudioAndSpeakerEmb(AlignedTTSDatasetWithAudio):
    def __init__(
        self,
        json_path: str,
        *,
        max_items: Optional[int] = None,
        dur_source: str = "ctc",
        dur_field: str = "dur_tok_frames",
        speaker_emb_dim: int = 192,
        require_speaker_embeds: bool = True,
        l2norm_speaker_embeds: bool = True,
        prior_f0_cache_field: str = "fcpe_f0_path",
    ):
        super().__init__(
            json_path,
            max_items=max_items,
            dur_source=dur_source,
            dur_field=dur_field,
            prior_f0_cache_field=prior_f0_cache_field,
        )
        self.speaker_emb_dim = int(speaker_emb_dim)
        self.require_speaker_embeds = bool(require_speaker_embeds)
        self.l2norm_speaker_embeds = bool(l2norm_speaker_embeds)
        self._spk_cache: Dict[str, torch.Tensor] = {}

    def _load_spk_emb(self, p: str) -> torch.Tensor:
        if p in self._spk_cache:
            return self._spk_cache[p]
        data = torch.load(p, map_location="cpu")
        if isinstance(data, dict):
            emb = data.get("emb", data.get("speaker_emb", data.get("spk_emb", None)))
        else:
            emb = data
        if emb is None:
            raise RuntimeError(f"speaker_embeds missing 'emb': {p}")
        t = torch.as_tensor(emb, dtype=torch.float32).view(-1)
        if int(t.numel()) != int(self.speaker_emb_dim):
            raise RuntimeError(
                f"speaker_embeds dim mismatch: got {int(t.numel())}, expected {self.speaker_emb_dim}: {p}"
            )
        if self.l2norm_speaker_embeds:
            t = t / t.norm().clamp_min(1e-12)
        self._spk_cache[p] = t
        return t

    def __getitem__(self, idx: int) -> Dict:
        out = super().__getitem__(idx)
        p = out.get("speaker_embeds", None)
        if not p:
            if self.require_speaker_embeds:
                raise RuntimeError(
                    f"Missing speaker_embeds in dataset item idx={idx} utt_id={out.get('utt_id','')}"
                )
            out["speaker_emb_mean"] = None
            out["speaker_emb_chunk"] = out.get("spk_emb_chunk", None)
            if self.l2norm_speaker_embeds and (out["speaker_emb_chunk"] is not None):
                e = torch.as_tensor(out["speaker_emb_chunk"], dtype=torch.float32).view(-1)
                out["speaker_emb_chunk"] = e / e.norm().clamp_min(1e-12)
            return out
        out["speaker_emb_mean"] = self._load_spk_emb(str(p))
        out["speaker_emb_chunk"] = out.get("spk_emb_chunk", None)
        if self.l2norm_speaker_embeds and (out["speaker_emb_chunk"] is not None):
            e = torch.as_tensor(out["speaker_emb_chunk"], dtype=torch.float32).view(-1)
            out["speaker_emb_chunk"] = e / e.norm().clamp_min(1e-12)
        return out


def collate_fn_with_audio_and_speaker_emb(batch_items: List[Dict]):
    """
    Keep the base batch tensors + pass-through audio paths for demos/debug
    + per-item speaker centroid embedding loaded from `speaker_embeds.pt`.
    """
    # Some manifests store audio/mel payload paths under different keys.
    # Prefer an explicit audio path, but fall back to the packed chunk `.pt` path.
    audio_paths = []
    prior_f0_paths = []
    for it in batch_items:
        p = it.get("audio_path", None)
        if not p:
            p = it.get("wav_path", None)
        if not p:
            p = it.get("audio_pt", None)
        if not p:
            p = it.get("audio", None)
        audio_paths.append(p)
        prior_f0_paths.append(it.get("fcpe_f0_path", it.get("f0_fcpe_path", it.get("f0_path", None))))
    spk_embs = []
    gender_ids = []
    emotion_ids = []
    for it in batch_items:
        e = it.get("speaker_emb_mean", None)
        if e is None:
            e = it.get("speaker_emb_chunk", None)
        if e is None:
            e = torch.zeros(256, dtype=torch.float32)
        spk_embs.append(torch.as_tensor(e, dtype=torch.float32).view(-1))
        gender_ids.append(int(it.get("gender_id", 0)))
        emotion_ids.append(int(it.get("emotion_id", 0)))
    spk_emb_b = torch.stack(spk_embs, dim=0)  # [B, d_spk]
    gender_ids_t = torch.tensor(gender_ids, dtype=torch.long)
    emotion_ids_t = torch.tensor(emotion_ids, dtype=torch.long)
    return (*collate_fn(batch_items), audio_paths, spk_emb_b, gender_ids_t, emotion_ids_t, prior_f0_paths)


class StatefulChunkBatchSampler(torch.utils.data.Sampler[List[int]]):
    """
    Batch sampler dla stateful bridge:
      - w batchu jest max 1 przykład na (speaker_id, book_id),
      - dla każdego (speaker_id, book_id) przykłady idą rosnąco po chunk_idx.

    Dzięki temu `ContextBridgeCache` ma sens (kolejne chunki tej samej książki trafiają w kolejnych krokach,
    a nie razem w jednym batchu).
    """
    def __init__(
        self,
        *,
        items: List[Dict],
        batch_size: int,
        drop_last: bool,
        seed: int = 1234,
        shuffle_keys: bool = True,
    ):
        self.batch_size = int(max(1, batch_size))
        self.drop_last = bool(drop_last)
        self.seed = int(seed)
        self.shuffle_keys = bool(shuffle_keys)
        self.epoch = 0

        groups: Dict[Tuple[int, str], List[Tuple[int, int]]] = defaultdict(list)
        for idx, it in enumerate(items):
            try:
                sid = int(it.get("speaker_id", 0))
            except Exception:
                sid = 0
            bid = str(it.get("book_id", ""))
            try:
                ci = int(it.get("chunk_idx", it.get("chunk_index", idx)))
            except Exception:
                ci = int(idx)
            groups[(sid, bid)].append((ci, idx))

        self._group_indices: Dict[Tuple[int, str], List[int]] = {}
        for k, pairs in groups.items():
            pairs.sort(key=lambda x: int(x[0]))
            self._group_indices[k] = [int(i) for _, i in pairs]

        self._keys: List[Tuple[int, str]] = list(self._group_indices.keys())

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self):
        rng = random.Random(self.seed + int(self.epoch))
        keys = list(self._keys)
        if self.shuffle_keys:
            rng.shuffle(keys)

        ptr: Dict[Tuple[int, str], int] = {k: 0 for k in keys}
        remaining = {k for k in keys if len(self._group_indices.get(k, [])) > 0}
        start = 0

        while remaining:
            batch: List[int] = []
            visited = 0
            # One pass over keys (at most once per key) => no duplicate (speaker_id, book_id) in batch.
            while len(batch) < self.batch_size and remaining and visited < len(keys):
                k = keys[(start + visited) % len(keys)]
                visited += 1
                if k not in remaining:
                    continue
                gi = self._group_indices[k]
                p = ptr[k]
                if p >= len(gi):
                    remaining.discard(k)
                    continue
                batch.append(int(gi[p]))
                ptr[k] = p + 1
                if ptr[k] >= len(gi):
                    remaining.discard(k)

            start = (start + visited) % max(1, len(keys))
            if not batch:
                break
            if len(batch) < self.batch_size and self.drop_last:
                break
            yield batch

    def __len__(self) -> int:
        total = sum(len(v) for v in self._group_indices.values())
        if self.drop_last:
            return total // int(self.batch_size)
        return (total + int(self.batch_size) - 1) // int(self.batch_size)


class FixedSlotChunkBatchSampler(torch.utils.data.Sampler[List[int]]):
    """
    Stateful batch sampler with *fixed slots*.

    Properties:
      - At most 1 example per (speaker_id, book_id) in a batch (same as StatefulChunkBatchSampler).
      - Keeps exactly `batch_size` active streams ("slots") when possible.
      - Each slot sticks to one (speaker_id, book_id) and yields consecutive chunk_idx.
      - When a stream is exhausted, the slot is immediately filled with a new (speaker_id, book_id).

    This tends to be faster than round-robin for IO-bound datasets because the working set of active
    streams stays limited to `batch_size` keys.
    """

    def __init__(
        self,
        *,
        items: List[Dict],
        batch_size: int,
        drop_last: bool,
        seed: int = 1234,
        shuffle_keys: bool = True,
    ):
        self.batch_size = int(max(1, batch_size))
        self.drop_last = bool(drop_last)
        self.seed = int(seed)
        self.shuffle_keys = bool(shuffle_keys)
        self.epoch = 0

        groups: Dict[Tuple[int, str], List[Tuple[int, int]]] = defaultdict(list)
        for idx, it in enumerate(items):
            try:
                sid = int(it.get("speaker_id", 0))
            except Exception:
                sid = 0
            bid = str(it.get("book_id", ""))
            try:
                ci = int(it.get("chunk_idx", it.get("chunk_index", idx)))
            except Exception:
                ci = int(idx)
            groups[(sid, bid)].append((ci, idx))

        self._group_indices: Dict[Tuple[int, str], List[int]] = {}
        for k, pairs in groups.items():
            pairs.sort(key=lambda x: int(x[0]))
            self._group_indices[k] = [int(i) for _, i in pairs]

        self._keys: List[Tuple[int, str]] = list(self._group_indices.keys())

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self):
        rng = random.Random(self.seed + int(self.epoch))
        keys = list(self._keys)
        if self.shuffle_keys:
            rng.shuffle(keys)

        # Position pointers within each stream.
        ptr: Dict[Tuple[int, str], int] = {k: 0 for k in keys}

        # Key queue used to fill new slots.
        queue_i = 0

        def _pop_next_key(active: set) -> Optional[Tuple[int, str]]:
            nonlocal queue_i
            while queue_i < len(keys):
                k = keys[queue_i]
                queue_i += 1
                if k in active:
                    continue
                if ptr.get(k, 0) >= len(self._group_indices.get(k, [])):
                    continue
                return k
            return None

        # Initialize active slots.
        active_keys: List[Optional[Tuple[int, str]]] = []
        active_set: set = set()
        for _ in range(self.batch_size):
            k = _pop_next_key(active_set)
            if k is None:
                break
            active_keys.append(k)
            active_set.add(k)

        # Main loop: one batch = one item from each active slot.
        while True:
            # Refill any empty slots (e.g. after a stream ended in the previous iteration).
            for si in range(len(active_keys), self.batch_size):
                k = _pop_next_key(active_set)
                if k is None:
                    break
                active_keys.append(k)
                active_set.add(k)

            if not active_keys:
                break

            batch: List[int] = []
            # Iterate over slots; if a stream ends, mark slot empty (removed after batch is formed).
            ended_slots: List[int] = []
            for si, k in enumerate(active_keys):
                if k is None:
                    ended_slots.append(si)
                    continue
                gi = self._group_indices.get(k, [])
                p = ptr.get(k, 0)
                if p >= len(gi):
                    ended_slots.append(si)
                    active_set.discard(k)
                    continue
                batch.append(int(gi[p]))
                ptr[k] = p + 1
                if ptr[k] >= len(gi):
                    ended_slots.append(si)
                    active_set.discard(k)

            # Remove ended slots (from back to front to keep indices stable).
            for si in reversed(ended_slots):
                try:
                    del active_keys[si]
                except Exception:
                    pass

            if not batch:
                break
            if len(batch) < self.batch_size and self.drop_last:
                break
            yield batch

    def __len__(self) -> int:
        total = sum(len(v) for v in self._group_indices.values())
        if self.drop_last:
            return total // int(self.batch_size)
        return (total + int(self.batch_size) - 1) // int(self.batch_size)


class VirtualLaneDataset(torch.utils.data.Dataset):
    """
    Wraps an existing dataset and rewrites (book_id, chunk_idx) into multiple independent
    "virtual lanes" per (speaker_id, book_id), so you can keep a larger effective batch size
    even when `--max-items` contains only one real book.

    Each lane gets a contiguous slice of chunks and chunk_idx is re-numbered starting at 0,
    so `ContextBridgeCache` continuity rules still apply within a lane.
    """
    def __init__(self, base_ds: torch.utils.data.Dataset, virtual_meta: List[Dict]):
        super().__init__()
        self.base_ds = base_ds
        self.virtual_meta = list(virtual_meta)

    def __len__(self) -> int:
        return len(self.virtual_meta)

    @property
    def items(self) -> List[Dict]:
        base_items = list(getattr(self.base_ds, "items", []))
        merged: List[Dict] = []
        for idx, meta in enumerate(self.virtual_meta):
            item = dict(base_items[idx]) if idx < len(base_items) and isinstance(base_items[idx], dict) else {}
            item["book_id"] = meta.get("book_id", item.get("book_id", ""))
            item["chunk_idx"] = int(meta.get("chunk_idx", item.get("chunk_idx", 0)))
            item["chunk_index"] = int(meta.get("chunk_idx", item.get("chunk_index", item.get("chunk_idx", 0))))
            item["speaker_id"] = int(meta.get("speaker_id", item.get("speaker_id", item.get("speaker", 0))))
            merged.append(item)
        return merged

    def __getitem__(self, idx: int):
        out = dict(self.base_ds[idx])
        meta = self.virtual_meta[int(idx)]
        out["book_id"] = meta.get("book_id", out.get("book_id", ""))
        out["chunk_idx"] = int(meta.get("chunk_idx", out.get("chunk_idx", 0)))
        return out


def _build_virtual_lane_meta(*, items: List[Dict], lanes_per_key: int) -> List[Dict]:
    """
    Returns per-index metadata with rewritten:
      - book_id: f"{orig}::lane{lane}"
      - chunk_idx: 0.. within lane
    Keeps speaker_id as-is.
    """
    lanes_per_key = int(max(1, lanes_per_key))
    groups: Dict[Tuple[int, str], List[Tuple[int, int]]] = defaultdict(list)
    for idx, it in enumerate(items):
        try:
            sid = int(it.get("speaker_id", 0))
        except Exception:
            sid = 0
        bid = str(it.get("book_id", ""))
        try:
            ci = int(it.get("chunk_idx", it.get("chunk_index", idx)))
        except Exception:
            ci = int(idx)
        groups[(sid, bid)].append((ci, idx))

    meta = [{"book_id": it.get("book_id", ""), "chunk_idx": int(it.get("chunk_idx", 0)), "speaker_id": int(it.get("speaker_id", 0))} for it in items]

    for (sid, bid), pairs in groups.items():
        pairs.sort(key=lambda x: int(x[0]))
        idxs = [int(i) for _, i in pairs]
        n = len(idxs)
        lanes = int(min(lanes_per_key, n))
        if lanes <= 1:
            continue

        base = n // lanes
        rem = n - base * lanes
        sizes = [base + (1 if l < rem else 0) for l in range(lanes)]

        start = 0
        for lane, sz in enumerate(sizes):
            if sz <= 0:
                continue
            seg = idxs[start : start + sz]
            start += sz
            vbook = f"{bid}::lane{lane}"
            for j, orig_idx in enumerate(seg):
                meta[orig_idx] = {"speaker_id": int(sid), "book_id": vbook, "chunk_idx": int(j)}

    return meta


# ---------------- MAIN ----------------
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--ckpt",
        type=str,
        default="",
        help="(optional) checkpoint bazowego modelu TTS; gdy pusty -> trening od zera (base trainable).",
    )
    ap.add_argument(
        "--dataset-json",
        type=str,
        default="/home/rizos/Downloads/SalmonTTS2/test_paraqueet/wegorz_ablations/run_wegorz_bilingual_onlinectc_1k_continue/manifest_text_lang_prefixed_pl_orth_en_ipa__stateful_mixed_full.json",
    )
    ap.add_argument(
        "--asr-ckpt",
        type=str,
        default=DEFAULT_BILINGUAL_ASR_CKPT,
        help="Frozen WegorzASRNanoV2 checkpoint used as the online CTC duration/alignment model.",
    )
    ap.add_argument(
        "--online-ctc",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use frozen ASR V2 + aligner V3 online to generate duration targets on x2 and rescale to mel frames.",
    )
    ap.add_argument(
        "--online-ctc-trainable",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="If enabled, fine-tune ASR V2 with auxiliary CTC loss. Default keeps ASR frozen to preserve old acoustic path.",
    )
    ap.add_argument("--w-online-ctc", type=float, default=0.03, help="Weight of the auxiliary online CTC loss.")
    ap.add_argument(
        "--out-dir",
        type=str,
        default="/home/rizos/Downloads/SalmonTTS2/test_paraqueet/wegorz_dubbingTTS/src/experiments/runs_priorprosody",
        help="(optional) katalog wyjściowy eksperymentu; domyślnie zapisuje pod src/experiments.",
    )

    # Base model config (używane tylko jeśli --ckpt jest puste)
    ap.add_argument("--hidden-dim", type=int, default=512)
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--num-speakers", type=int, default=256)
    ap.add_argument("--no-sdpa", action="store_true", help="disable SDPA/flash attention")
    ap.add_argument("--spk-dim", type=int, default=256, help="dimension of learned speaker embedding z_spk / centroid bank")
    ap.add_argument("--spk-emb-drop", type=float, default=0.1, help="dropout on speaker embedding adapter (train-time)")
    ap.add_argument("--gender-emb-dim", type=int, default=512, help="hidden dim of gender embedding token (projected to model hidden size if needed)")
    ap.add_argument(
        "--gender-dropout-prob",
        type=float,
        default=0.25,
        help="Train-time probability of replacing gender_id with 0/unknown, to avoid over-reliance on gender tokens.",
    )
    ap.add_argument(
        "--disable-gender-token",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Production cleanup: keep the checkpoint-compatible gender token slot, but always feed unknown/0. "
            "This removes gender as a controllable conditioning signal without changing tensor shapes."
        ),
    )
    ap.add_argument(
        "--speaker-verifier",
        type=str,
        default="none",
        choices=["none", "wavlm_tbr"],
        help="Optional frozen Orange/Speaker-wavLM-tbr timbre verifier loss.",
    )
    ap.add_argument(
        "--speaker-verifier-source",
        type=str,
        default="/home/rizos/Downloads/SalmonTTS2/test_paraqueet/wegorz_dubbingTTS/external/Orange_Speaker-wavLM-tbr",
        help="Local Orange/Speaker-wavLM-tbr directory.",
    )
    ap.add_argument(
        "--speaker-verifier-savedir",
        type=str,
        default="",
        help="Deprecated no-op kept for old commands.",
    )
    ap.add_argument("--w-speaker-verifier", type=float, default=0.0, help="Weight for frozen external speaker verifier loss.")
    ap.add_argument("--speaker-verifier-every", type=int, default=8, help="Run external verifier loss every N batches.")
    ap.add_argument("--speaker-verifier-max-sec", type=float, default=3.0, help="Max seconds decoded for verifier loss.")
    ap.add_argument("--w-pitch", type=float, default=0.0, help="Weight for differentiable waveform pitch/F0 consistency loss.")
    ap.add_argument("--w-energy", type=float, default=0.0, help="Weight for differentiable waveform log-RMS energy consistency loss.")
    ap.add_argument("--prosody-loss-every", type=int, default=4, help="Run pitch/energy losses every N batches.")
    ap.add_argument("--prosody-loss-max-sec", type=float, default=2.0, help="Max seconds decoded for pitch/energy losses.")
    ap.add_argument("--w-prior-energy", type=float, default=0.0, help="Auxiliary StyleTTS2-like N/energy loss on prior hidden frames.")
    ap.add_argument("--w-prior-f0", type=float, default=0.0, help="Auxiliary FCPE F0 loss on prior hidden frames.")
    ap.add_argument(
        "--prior-f0-cache-field",
        type=str,
        default="fcpe_f0_path",
        help="Manifest field name used by preprocessing for cached FCPE F0 paths.",
    )
    ap.add_argument(
        "--emotion-conditioning",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use EARS emotion_group ids as a gated conditioning signal for duration/prosody.",
    )
    ap.add_argument("--emotion-emb-dim", type=int, default=128)
    ap.add_argument(
        "--demo-emotions",
        default="neutral,happy",
        help="Comma-separated emotion groups for text demos when --emotion-conditioning is enabled.",
    )
    ap.add_argument(
        "--prefix-fill-train",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Training ablation for dubbing: prepend previous GT mel tail from the same speaker/book as an acoustic "
            "prefix, mask that prefix out from acoustic losses, and train only the current text target."
        ),
    )
    ap.add_argument(
        "--prefix-fill-ms",
        type=float,
        default=_ACOUSTIC_PROMPT_MS_DEFAULT,
        help="Legacy alias for --acoustic-prompt-ms. Length of GT acoustic prompt prepended by --prefix-fill-train.",
    )
    ap.add_argument(
        "--acoustic-prompt-ms",
        type=float,
        default=_ACOUSTIC_PROMPT_MS_DEFAULT,
        help="Unified acoustic prompt length. LSTM variant default is 1000 ms to reduce flow destabilization from long generated prefixes.",
    )
    ap.add_argument(
        "--short-continuity-ms",
        type=float,
        default=_SHORT_CONTINUITY_MS_DEFAULT,
        help="Legacy generated-tail continuity window between chunks. Clean default is 0; use the 3s acoustic prompt instead.",
    )
    ap.add_argument(
        "--pause-mid-loss-weight",
        type=float,
        default=1.0,
        help="Weight for the middle frames of pause tokens (<sp>/<BOS>/<EOS>) in acoustic losses. 1.0 disables shaping.",
    )
    ap.add_argument(
        "--pause-mask-middle-train",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Shortcut: set pause middle acoustic-loss weight to 0.0 while keeping --pause-edge-frames at full weight.",
    )
    ap.add_argument(
        "--pause-force-digital-silence-demo",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="After Vocos decode, zero waveform samples that correspond to the middle of generated pause spans in demos.",
    )
    ap.add_argument(
        "--pause-force-digital-silence-infer",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="After Vocos decode, zero waveform samples that correspond to the middle of generated pause spans in infer-only.",
    )
    ap.add_argument(
        "--pause-edge-frames",
        type=int,
        default=10,
        help="How many edge frames on each side of a pause keep full acoustic loss weight. Used with --pause-mid-loss-weight.",
    )
    ap.add_argument("--mel-twopass", action=argparse.BooleanOptionalAction, default=True, help="enable simple global two-pass mel refine in demos/infer")
    ap.add_argument("--mel-twopass-steps-first", type=int, default=8, help="mel flow steps in pass 1")
    ap.add_argument("--mel-twopass-steps-second", type=int, default=3, help="mel flow steps in pass 2")
    ap.add_argument("--mel-twopass-t-noise", type=float, default=0.12, help="global re-edit noise mix for pass 2; x_start=(1-a)*pass1 + a*x0")
    ap.add_argument(
        "--require-spk-override",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If enabled, never fall back to spk_embed(speaker_id). Require passing spk_vec_override everywhere.",
    )
    ap.add_argument(
        "--spk-emb-dim",
        dest="spk_emb_dim",
        type=int,
        default=256,
        help="Dimension of external speaker embedding .pt files referenced by dataset ('speaker_embeds'). "
             "For Speakder_enkoder spk_enc this is typically 256.",
    )
    ap.add_argument(
        "--spk-emb-l2norm",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="L2-normalize external speaker embeddings loaded from .pt files (dataset speaker_embeds / infer overrides).",
    )

    ap.add_argument("--dur-gt-source", type=str, default="ctc", choices=["words", "ctc", "mas"])
    ap.add_argument(
        "--dur-field",
        "--dur-ctc-field",
        type=str,
        dest="dur_field",
        default="dur_tok_frames",
        help="Manifest field used when dur source is ctc/auto, e.g. dur_tok_frames_ctc_ft.",
    )
    ap.add_argument(
        "--boundary-jitter",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Train-only on-the-fly word boundary jitter between consecutive stateful chunks.",
    )
    ap.add_argument(
        "--boundary-jitter-prob",
        type=float,
        default=0.25,
        help="Probability of jittering a valid boundary between consecutive chunks.",
    )
    ap.add_argument(
        "--boundary-jitter-max-words",
        type=int,
        default=2,
        help="Maximum number of words moved across a jittered boundary.",
    )
    ap.add_argument(
        "--boundary-jitter-min-frames",
        type=int,
        default=24,
        help="Skip a jittered sample if the resulting mel would be shorter than this.",
    )
    ap.add_argument(
        "--boundary-jitter-max-frames",
        type=int,
        default=0,
        help="Skip a jittered sample if the resulting mel would exceed this frame count; 0 disables.",
    )
    ap.add_argument(
        "--boundary-jitter-seed",
        type=int,
        default=1234,
        help="Seed for deterministic boundary jitter decisions.",
    )
    ap.add_argument(
        "--boundary-jitter-epoch-vary",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Change deterministic boundary jitter choices each epoch.",
    )
    ap.add_argument(
        "--duration-model",
        type=str,
        default="lstm_ar",
        choices=["lstm_ar", "mini_dualpath"],
        help="Duration predictor: current causal/stateful LSTM AR or non-AR MiniDualPath direct logdur.",
    )
    ap.add_argument("--ar-dur-layers", type=int, default=4, help="Number of LSTM layers for duration predictor.")
    ap.add_argument("--ar-dur-heads", type=int, default=4, help="Legacy no-op kept for checkpoint/command compatibility.")
    ap.add_argument("--ar-dur-ffn-mult", type=int, default=4, help="Legacy no-op kept for checkpoint/command compatibility.")
    ap.add_argument("--ar-dur-dropout", type=float, default=0.1, help="Dropout for LSTM duration predictor.")
    ap.add_argument("--ar-dur-loss", type=str, default="huber", choices=["huber", "l1", "mse"], help="Loss for AR duration teacher-forcing.")
    ap.add_argument("--mini-dur-layers", type=int, default=3, help="Number of MiniDualPath duration blocks.")
    ap.add_argument("--mini-dur-attn-dim", type=int, default=128, help="MiniDualPath duration attention branch dim.")
    ap.add_argument("--mini-dur-conv-dim", type=int, default=128, help="MiniDualPath duration convolution branch dim.")
    ap.add_argument("--mini-dur-heads", type=int, default=4, help="MiniDualPath duration attention heads.")
    ap.add_argument("--mini-dur-dropout", type=float, default=0.1, help="MiniDualPath duration dropout.")
    ap.add_argument(
        "--duration-lr-mult",
        type=float,
        default=1.0,
        help="LR multiplier for duration predictor params. Useful when replacing duration model from random init.",
    )
    ap.add_argument(
        "--duration-state-carry",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Carry LSTM duration h/c across infer/text-demo chunks. Disable to reset duration state per chunk.",
    )
    ap.add_argument("--max-items", type=int, default=0)

    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--batch-size", type=int, default=10)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--prefetch-factor", type=int, default=4)
    ap.add_argument("--persistent-workers", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True, help="use AMP (autocast+GradScaler) on CUDA")
    ap.add_argument("--val-tail-per-book", type=int, default=4, help="ile końcowych chunków per (speaker,book) trafi do walidacji")
    ap.add_argument(
        "--mel-flow-fp32",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run mel_flow forward in fp32 even when --amp is enabled (prevents fp16 overflow/NaNs; slightly slower).",
    )
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--grad-clip-norm", type=float, default=1.0, help="If >0, clip global grad norm before optimizer step (helps avoid NaNs).")
    ap.add_argument("--demo-dur-steps", type=int, default=-1, help="Override duration sampler steps for text demos (<=0 => use CONFIG).")
    ap.add_argument("--demo-dur-noise-scale", type=float, default=-1.0, help="Override duration sampler noise_scale for text demos (<0 => use CONFIG).")
    ap.add_argument("--t-sample-mode", type=str, default="logit_normal", choices=["uniform", "logit_normal"], help="sampling timestepow dla mel-flow training")
    ap.add_argument("--t-logit-mu", type=float, default=0.0, help="mu dla logit-normal timestep sampling")
    ap.add_argument("--t-logit-sigma", type=float, default=1.0, help="sigma dla logit-normal timestep sampling")
    ap.add_argument("--dur-flow-clip-sigma", type=float, default=0.0, help="(infer/demo) clamp dur-flow state to mu ± clip*sigma each step; 0 disables.")
    ap.add_argument("--dur-flow-clip-abs-min", type=float, default=float("nan"), help="(infer/demo) abs clamp min for dur-flow logdur (NaN disables).")
    ap.add_argument("--dur-flow-clip-abs-max", type=float, default=float("nan"), help="(infer/demo) abs clamp max for dur-flow logdur (NaN disables).")
    ap.add_argument(
        "--dur-flow-fix-total",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="(infer/demo) DEPRECATED/NO-OP. Total duration rescaling was removed because it harmed endings (truncation/acceleration).",
    )
    ap.add_argument(
        "--dur-flow-fix-total-mode",
        type=str,
        default="prior_mu",
        choices=["prior_mu"],
        help="DEPRECATED/NO-OP (kept for backward compatibility).",
    )

    # Stateful batching / memory bridge controls.
    ap.add_argument(
        "--stateful-batching",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="use stateful batch sampler to keep (speaker,book) continuity across chunks",
    )
    ap.add_argument(
        "--stateful-shuffle-keys",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="shuffle (speaker,book) streams each epoch; disabling improves IO locality but reduces mixing",
    )
    ap.add_argument(
        "--stateful-mode",
        type=str,
        default="fixed",
        choices=["round_robin", "fixed"],
        help="stateful batching policy: round_robin (legacy) vs fixed (fixed slots per stream)",
    )
    ap.add_argument(
        "--virtual-lanes",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="when max-items is small, split one (speaker,book) into multiple lanes to keep batch_size>1 (ablation helper)",
    )
    ap.add_argument(
        "--virtual-lanes-threshold",
        type=int,
        default=0,
        help="enable virtual lanes only when --max-items <= threshold",
    )

    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--device", type=str, default="cuda")

    # --- Quick inference (no training) ---
    ap.add_argument("--infer-only", action="store_true", help="Run single-shot inference (no training loop) and exit.")
    ap.add_argument("--infer-text", type=str, default="", help="Text to synthesize (multi-line allowed; joined with short silences).")
    ap.add_argument(
        "--infer-tag",
        type=str,
        default="",
        help="Optional prefix for infer output filenames (infer__<tag>__A.wav etc). Useful for A/B comparisons in one folder.",
    )
    ap.add_argument("--infer-speaker-a", type=int, default=0, help="speaker_id for voice A (centroid from dataset-json).")
    ap.add_argument("--infer-speaker-b", type=int, default=-1, help="speaker_id for voice B (optional, for mixing).")
    ap.add_argument(
        "--infer-speaker-emb-pt",
        type=str,
        default="",
        help="Optional path to a .pt with speaker embedding for voice A in autospeaker space "
             "(dimension must match --spk-dim; overrides dataset centroid).",
    )
    ap.add_argument(
        "--infer-speaker-emb-pt-b",
        type=str,
        default="",
        help="Optional path to a .pt with speaker embedding for voice B in autospeaker space "
             "(dimension must match --spk-dim; overrides dataset centroid).",
    )
    ap.add_argument(
        "--infer-save-speaker-emb-pt",
        type=str,
        default="",
        help="Optional path where the extracted/loaded voice A speaker vector should be saved as .pt.",
    )
    ap.add_argument("--infer-mix-alpha", type=float, default=0.5, help="alpha for mixing: (1-a)*A + a*B")
    ap.add_argument("--infer-mix", action=argparse.BooleanOptionalAction, default=True, help="If true and speaker-b>=0, also render MIX.")
    ap.add_argument("--infer-seed", type=int, default=1234, help="Random seed for inference.")
    ap.add_argument("--infer-mel-steps", type=int, default=8, help="Euler steps for mel flow in inference.")
    ap.add_argument("--infer-dur-steps", type=int, default=10, help="Steps for duration sampler in inference.")
    ap.add_argument("--infer-dur-noise-scale", type=float, default=0.1, help="Noise scale for duration sampler in inference.")
    ap.add_argument(
        "--infer-dur-source",
        type=str,
        default="flow",
        choices=["flow", "prior_mu", "prior_sample"],
        help="How to get durations in inference: flow sampler vs dur-prior mean vs dur-prior sample (no flow steps).",
    )
    ap.add_argument(
        "--deterministic-dur",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Shortcut: use dur-prior mean for infer/text-demo durations and disable duration x0 noise.",
    )
    ap.add_argument("--infer-speed", type=float, default=1.0, help="Speed ratio (>1 faster => shorter durations).")
    ap.add_argument("--infer-out-dir", type=str, default="/tmp/wegorz_learnedvoice_infer", help="Output directory for inference wavs.")
    ap.add_argument("--infer-join-sil-ms", type=float, default=0.0, help="Silence inserted between chunks (ms). 0 disables.")
    ap.add_argument(
        "--infer-decode-joined",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Deprecated/ignored. Inference now always decodes chunk-by-chunk with Vocos for long-form safety.",
    )
    ap.add_argument(
        "--infer-mel-join-overlap-ms",
        type=float,
        default=0.0,
        help="Deprecated/ignored. Chunkwise Vocos decode no longer uses mel-overlap joining.",
    )

    ap.add_argument(
        "--resume",
        type=str,
        default="",
        help="checkpoint tego skryptu do wznowienia treningu (domyślnie pusty = start bez resume)",
    )
    ap.add_argument("--demo-every", type=int, default=1)
    ap.add_argument("--demo-count", type=int, default=10, help="ile przykładów zapisać w demówkach (z batcha)")
    ap.add_argument(
        "--demo-dur-source",
        type=str,
        default="flow",
        choices=["flow", "prior_mu", "prior_sample"],
        help="Duration source for generated text demos and speaker audit.",
    )
    ap.add_argument(
        "--demo-batches",
        type=int,
        default=10,
        help="ile batchy demo policzyć do uśrednienia metryk (WAVy zapisuje tylko dla pierwszego batcha)",
    )
    ap.add_argument(
        "--demo-long",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="zapisz też demówki z tekstu (TEST_SENTENCES_PL + LONG_DEMO_CHUNKS_PL) w demos/epXXXX/text_demo.",
    )
    ap.add_argument(
        "--demo-speaker-audit",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="render one fixed sentence in random speakers + GT reference clip for quick speaker-quality audit",
    )
    ap.add_argument("--demo-speaker-audit-count", type=int, default=10, help="number of random speakers for audit demo")
    ap.add_argument(
        "--demo-speaker-audit-text",
        type=str,
        default="To jest test jakości głosu. Sprawdzam barwę, czystość i stabilność intonacji w tym samym zdaniu.",
        help="single sentence used in speaker-audit demo",
    )
    ap.add_argument("--demo-long-chunks", type=int, default=7, help="ile kolejnych chunków z LONG_DEMO_CHUNKS_PL wygenerować (0=off)")
    ap.add_argument(
        "--demo-extra-long-chunks",
        type=int,
        default=7,
        help="dodatkowy dialog/story w text_demo (0=off)",
    )
    ap.add_argument(
        "--demo-extreme-long-chunks",
        type=int,
        default=14,
        help="dodatkowy bardzo długi dialog/story w text_demo (0=off)",
    )
    ap.add_argument(
        "--demo-long-speaker-id",
        type=int,
        default=-1,
        help="speaker_id dla text_demo (-1 = weź z pierwszego demo batcha; zalecane gdy dataset ma nieciągłe speaker_id).",
    )
    ap.add_argument(
        "--demo-long-speaker-lang",
        type=str,
        default="en",
        choices=["auto", "en", "pl"],
        help="Prefer speaker language for text_demo when --demo-long-speaker-id < 0. Default en helps judge PL spoken by an English voice.",
    )
    ap.add_argument(
        "--demo-ref-wav",
        type=str,
        default="",
        help="Optional external WAV/MP3/FLAC file used as text_demo voice reference instead of a dataset sample.",
    )
    ap.add_argument(
        "--demo-ref-wav-glob",
        type=str,
        default="",
        help="Ignored in learned_voice mode; text_demo uses a dataset speaker_id and learned tables.",
    )
    ap.add_argument(
        "--demo-ref-max-sec",
        type=float,
        default=10.0,
        help="Max seconds read from each --demo-ref-wav/--demo-ref-wav-glob file.",
    )
    ap.add_argument("--text-demo-dur-scale", type=float, default=1.0, help="skala dla dur_pred w text_demo (1.0 = bez zmian)")
    ap.add_argument("--dur-pred-sp-min-frames", type=float, default=1.0, help="minimalna duracja dla tokenów pauzy (<sp>/<BOS>/<EOS>) w text_demo (w ramkach)")
    ap.add_argument(
        "--vocab",
        type=str,
        default="/home/rizos/Downloads/SalmonTTS2/test_paraqueet/interlanguage_bridge/data_pl_orth_en_ipa_bridge/vocab_pl_orth_en_ipa_bridge.json",
        help="vocab json path for mixed PL orthography + EN IPA tokenizer",
    )
    ap.add_argument(
        "--extra-text-demos",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="dodaj dodatkowe zdania/dialogi do text_demo",
    )

    # -------- Style conditioning (shared latent + 3 heads: mel/dur/prior) --------
    # -------- Non-finite diagnostics --------
    ap.add_argument(
        "--stop-on-nonfinite",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If enabled, stop training on the first non-finite (NaN/Inf) event and write a detailed diagnostic dump.",
    )
    ap.add_argument(
        "--nonfinite-max-skips",
        type=int,
        default=0,
        help="How many non-finite steps to skip before aborting (0=abort immediately).",
    )
    ap.add_argument(
        "--nonfinite-dump-tensors",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If enabled, also save selected tensors (.pt) alongside the JSON report for debugging.",
    )

    # --- Mel Prior (zaszyte na stałe jako gauss_nll) ---
    ap.add_argument("--prior-layers", type=int, default=3)
    ap.add_argument("--prior-heads", type=int, default=8, help="Liczba głów attention w priorze (musi dzielić hidden_dim)")
    ap.add_argument("--prior-noise-scale", type=float, default=1, help="Skala szumu dla t0 mela")
    ap.add_argument("--prior-logs-min", type=float, default=-7.0, help="Clamp log(sigma) min")
    ap.add_argument("--prior-logs-max", type=float, default=2.0, help="Clamp log(sigma) max")
    ap.add_argument(
        "--dualpath-branch-dim",
        type=int,
        default=256,
        help=(
            "Symmetric projected DualPath branch width. Projection path default is 256/256."
        ),
    )
    ap.add_argument(
        "--dualpath-attn-dim",
        type=int,
        default=0,
        help="Attention branch width. 0 = use --dualpath-branch-dim.",
    )
    ap.add_argument(
        "--dualpath-conv-dim",
        type=int,
        default=0,
        help="Convolution branch width. 0 = use --dualpath-branch-dim.",
    )
    ap.add_argument(
        "--dualpath-init-split-identity",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Initialize branch projections close to the old hard channel split.",
    )
    ap.add_argument(
        "--text-dualpath-projection",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use projected DualPath branches in the text encoder too. Default on for this projection experiment.",
    )
    ap.add_argument(
        "--text-dualpath-attn-dim",
        type=int,
        default=256,
        help="Text encoder projected attention branch width.",
    )
    ap.add_argument(
        "--text-dualpath-conv-dim",
        type=int,
        default=256,
        help="Text encoder projected convolution branch width.",
    )
    ap.add_argument(
        "--grad-diagnose-once",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Run one train batch, print gradient norms for DualPath groups, and exit before optimizer step/checkpoint.",
    )
    # --- Wagi lossów (kluczowe filary) ---
    ap.add_argument("--w-flow", type=float, default=1.0, help="Waga melspectrogram flow")
    ap.add_argument("--w-prior", type=float, default=1.0, help="Waga NLL mela")
    ap.add_argument("--w-dur", type=float, default=1.0, help="Waga duration flow")
    ap.add_argument(
        "--w-total-dur",
        type=float,
        default=0.05,
        help="Supervised duration budget loss weight: log(sum predicted/teacher durations) vs log(GT mel frames).",
    )
    ap.add_argument(
        "--w-token-dur-floor",
        type=float,
        default=0.02,
        help="Optional supervised anti-collapse loss for non-pause token durations.",
    )
    ap.add_argument(
        "--token-dur-floor-frames",
        type=float,
        default=1.5,
        help="Minimum soft duration for non-pause text tokens when --w-token-dur-floor > 0.",
    )
    ap.add_argument(
        "--xling-ctc-train",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Reserve flag for cross-lingual no-mel-GT CTC batches in this variant.",
    )
    ap.add_argument("--xling-prob", type=float, default=0.05, help="Target probability of cross-lingual CTC-only batches.")
    ap.add_argument(
        "--xling-directions",
        type=str,
        default="both",
        choices=["both", "en_ref_pl_text", "pl_ref_en_text"],
        help="Cross-lingual pairing direction for future no-mel-GT batches.",
    )
    ap.add_argument("--xling-w-ctc", type=float, default=1.0)
    ap.add_argument("--xling-w-spk", type=float, default=0.15)
    ap.add_argument("--xling-w-total-dur", type=float, default=0.10)
    ap.add_argument("--xling-w-token-floor", type=float, default=0.05)
    ap.add_argument("--xling-w-pause-budget", type=float, default=0.02)
    # --- Duration Prior & Flow (start z priora) ---
    ap.add_argument("--dur-x0", type=str, default="prior", choices=["none", "prior"])
    ap.add_argument("--dur-x0-noise-scale", type=float, default=1.0, help="Skala eps w x0 duration prior")
    ap.add_argument("--dur-prior-w", type=float, default=0.10, help="Auxiliary loss dla dur-priora")
    ap.add_argument("--dur-prior-logs-min", type=float, default=-5.0, help="Clamp log(sigma) min w dur-priorze")
    ap.add_argument("--dur-prior-logs-max", type=float, default=2.0, help="Clamp log(sigma) max w dur-priorze")
    ap.add_argument("--dur-prior-sigma-min", type=float, default=0.1, help="Podłoga sigma=exp(logs) dla x0 duration")

    # Ablacje / architektura
    ap.add_argument("--flow-layers", type=int, default=6)
    ap.add_argument("--flow-heads", type=int, default=8, help="liczba głów attention w mel flow (musi dzielić hidden_dim)")
    ap.add_argument(
        "--flow-timbre-adaln",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Enable speaker/style AdaLN inside mel flow. Default is off because flow is sensitive; "
            "prior still uses timbre AdaLN and duration still uses style bias."
        ),
    )
    ap.add_argument(
        "--flow-gauss-token-attn",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Replace frame-level self-attention inside flow DualPath blocks with token-slot self-attention: "
            "frames are pooled by the prior Gaussian map, attention runs over token slots, then context is scattered back to frames."
        ),
    )
    ap.add_argument(
        "--flow-gauss-cross-attn",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Run flow text cross-attention on Gaussian token slots instead of frame queries, "
            "then scatter the text update back to frames. This file defaults to enabled."
        ),
    )
    ap.add_argument(
        "--gauss-sp-mode",
        type=str,
        default=str(CONFIG.get("gauss_sp_mode", "mix")),
        choices=["mix", "isolate"],
        help="How to treat pause tokens in prior upsampling: mix (legacy) vs isolate (no mixing across silence).",
    )

    args = ap.parse_args()
    # This learned-voice file is the simple gauss-cross baseline. Grouped token slots were
    # a separate ablation and are intentionally not configurable here.
    args.flow_gauss_token_group_size = 1
    args.speaker_vector_source = "learned_voice"
    args.duration_model_raw = str(getattr(args, "duration_model", "lstm_ar"))
    args.prefix_fill_ms = float(getattr(args, "acoustic_prompt_ms", getattr(args, "prefix_fill_ms", _ACOUSTIC_PROMPT_MS_DEFAULT)))
    args.short_continuity_ms = float(getattr(args, "short_continuity_ms", _SHORT_CONTINUITY_MS_DEFAULT))
    if bool(getattr(args, "disable_gender_token", True)):
        args.gender_dropout_prob = 1.0
    if bool(getattr(args, "deterministic_dur", False)):
        args.infer_dur_source = "prior_mu"
        args.demo_dur_source = "prior_mu"
        args.infer_dur_noise_scale = 0.0
        args.demo_dur_noise_scale = 0.0
        args.dur_x0_noise_scale = 0.0
    if bool(getattr(args, "pause_mask_middle_train", False)):
        args.pause_mid_loss_weight = 0.0

    device = torch.device(args.device)
    set_seed(int(args.seed))
    CONFIG["gauss_sp_mode"] = str(getattr(args, "gauss_sp_mode", CONFIG.get("gauss_sp_mode", "mix")))
    use_amp = bool(getattr(args, "amp", False)) and (device.type == "cuda")
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)  # type: ignore[attr-defined]
    except Exception:
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)  # type: ignore[attr-defined]

    def _autocast(*, enabled: bool):
        if not enabled:
            return nullcontext()
        try:
            return torch.amp.autocast(device_type=device.type, enabled=True)  # type: ignore[attr-defined]
        except Exception:
            return torch.cuda.amp.autocast(enabled=True)  # type: ignore[attr-defined]

    def _make_run_dir(base_dir: Path, out_dir_arg: str) -> Path:
        if str(out_dir_arg).strip():
            out = Path(str(out_dir_arg)).expanduser()
            out.mkdir(parents=True, exist_ok=True)
            return out
        ts = time.strftime("%Y%m%d_%H%M%S")
        out = base_dir / "_runs" / f"salmon_{ts}"
        out.mkdir(parents=True, exist_ok=True)
        return out

    out_dir = _make_run_dir(Path(__file__).resolve().parent, str(getattr(args, "out_dir", "")))
    (out_dir / "chkpts").mkdir(parents=True, exist_ok=True)
    (out_dir / "demos").mkdir(parents=True, exist_ok=True)
    print(f"📦 Output: {out_dir}")
    print(
        "⏱️ deterministic-dur: "
        f"{bool(getattr(args, 'deterministic_dur', False))} "
        f"(infer={getattr(args, 'infer_dur_source', 'flow')}, demo={getattr(args, 'demo_dur_source', 'flow')}, "
        f"dur_x0_noise={float(getattr(args, 'dur_x0_noise_scale', 1.0)):.3f})"
    )
    print(
        "⏱️ duration-budget: "
        f"w_total={float(getattr(args, 'w_total_dur', 0.0)):.4f} "
        f"w_token_floor={float(getattr(args, 'w_token_dur_floor', 0.0)):.4f} "
        f"floor_frames={float(getattr(args, 'token_dur_floor_frames', 0.0)):.2f}"
    )
    print(
        "⏱️ duration-model: "
        f"{str(getattr(args, 'duration_model', 'ar_transformer'))} "
        f"(ar_layers={int(getattr(args, 'ar_dur_layers', 4))} "
        f"mini_layers={int(getattr(args, 'mini_dur_layers', 3))} "
        f"mini_attn={int(getattr(args, 'mini_dur_attn_dim', 128))} "
        f"mini_conv={int(getattr(args, 'mini_dur_conv_dim', 128))} "
        f"loss={str(getattr(args, 'ar_dur_loss', 'huber'))})"
    )
    if str(getattr(args, "duration_model", "")).lower().strip() == "lstm_ar":
        if bool(getattr(args, "duration_state_carry", True)):
            print("⏱️ duration-continuation: LSTM hidden state is carried across infer/text-demo chunks")
        else:
            print("⏱️ duration-continuation: disabled (LSTM duration h/c reset per chunk)")
    _short_continuity_ms = float(getattr(args, "short_continuity_ms", _SHORT_CONTINUITY_MS_DEFAULT))
    _prefix_frames = _prefix_frames_from_ms(_short_continuity_ms)
    _prefix_fill_frames = _prefix_frames_from_ms(float(getattr(args, "prefix_fill_ms", 0.0)))
    _prefix_fill_enabled = bool(getattr(args, "prefix_fill_train", False)) and int(_prefix_fill_frames) > 0
    if _prefix_frames > 0:
        print(
            f"🧩 prefix-continuity: enabled (train+infer) prefix_ms={_short_continuity_ms:.1f} "
            f"frames={_prefix_frames} (applied only when chunk_idx is consecutive for same speaker_id+book_id)"
        )
    else:
        print("🧩 prefix-continuity: disabled (prefix_ms=0.0)")
    if _prefix_fill_enabled:
        print(
            f"🧩 acoustic-prompt-train: enabled prompt_ms={float(getattr(args, 'prefix_fill_ms', 0.0)):.1f} "
            f"frames={_prefix_fill_frames} source=prev_gt_tail"
        )
    else:
        print("🧩 acoustic-prompt-train: disabled")
    print(
        "🚻 gender-token: "
        f"{'disabled/unknown-only' if bool(getattr(args, 'disable_gender_token', True)) else 'enabled'}"
    )
    print(
        "🎯 speaker-conditioning: prefix-token-only "
        f"(speaker_loss_w={float(getattr(args, 'speaker_loss_w', 0.0)):.4f}, "
        f"speaker_teacher_w={float(getattr(args, 'speaker_teacher_w', 0.0)):.4f}, "
        "speaker_vector_source=learned_voice, "
        "decoder/prior/dur speaker projections disabled)"
    )
    print(
        "🧬 style-adapters: enabled "
        f"(z_spk -> speaker prefix + gated AdaLN in prior; "
        f"flow AdaLN={'on' if bool(getattr(args, 'flow_timbre_adaln', False)) else 'off'}; "
        "z_style -> duration bias + LSTM h0/c0)"
    )
    print(
        "🧬 style-vector: z_style=raw_mlp_no_l2 "
        "(z_spk remains L2-normalized)"
    )
    print("🧬 ref-encoder pooling: separate attentive pooling for z_spk and z_style")
    pause_mid_w = float(getattr(args, "pause_mid_loss_weight", 1.0))
    pause_edge_k = int(getattr(args, "pause_edge_frames", 0))
    if pause_mid_w < 0.9999 or pause_edge_k > 0:
        print(
            f"🔇 pause-loss-shaping: enabled "
            f"(mid_weight={pause_mid_w:.3f}, edge_frames={pause_edge_k}, "
            f"mask_middle={bool(getattr(args, 'pause_mask_middle_train', False))})"
        )
    else:
        print("🔇 pause-loss-shaping: disabled")
    if bool(getattr(args, "pause_force_digital_silence_demo", False)) or bool(getattr(args, "pause_force_digital_silence_infer", False)):
        print(
            "🔕 digital-pause: "
            f"demo={bool(getattr(args, 'pause_force_digital_silence_demo', False))} "
            f"infer={bool(getattr(args, 'pause_force_digital_silence_infer', False))} "
            f"edge_frames={pause_edge_k}"
        )
    print(
        f"🔁 mel two-pass: enabled={bool(getattr(args,'mel_twopass',False))} "
        f"steps_first={int(getattr(args,'mel_twopass_steps_first',8))} "
        f"steps_second={int(getattr(args,'mel_twopass_steps_second',3))} "
        f"t_noise={float(getattr(args,'mel_twopass_t_noise',0.12)):.3f}"
    )
    print(
        f"⏱️ flow t-sampling: mode={str(getattr(args,'t_sample_mode','logit_normal'))} "
        f"mu={float(getattr(args,'t_logit_mu',0.0)):.3f} "
        f"sigma={float(getattr(args,'t_logit_sigma',1.0)):.3f}"
    )
    if bool(getattr(args, "online_ctc", True)):
        print(
            "🧭 online CTC durations: enabled "
            f"(impl=WegorzASRNanoV2 align=x2->mel "
            f"trainable={bool(getattr(args, 'online_ctc_trainable', False))} "
            f"w={float(getattr(args, 'w_online_ctc', 0.10)):.3f} "
            f"asr_ckpt={str(getattr(args, 'asr_ckpt', ''))})"
        )
    else:
        print(f"🧭 online CTC durations: disabled (dur_gt_source={str(getattr(args, 'dur_gt_source', 'ctc'))})")

    # -------- Dataset / Loader --------
    dataset_dur_source = "auto" if bool(getattr(args, "online_ctc", True)) else str(args.dur_gt_source)
    dataset_dur_field = str(getattr(args, "dur_field", "dur_tok_frames") or "dur_tok_frames")
    print(
        f"🗂️ dataset dur source: {dataset_dur_source} "
        f"field={dataset_dur_field} "
        f"(online_ctc={'on' if bool(getattr(args, 'online_ctc', True)) else 'off'})"
    )
    ds = AlignedTTSDatasetWithAudioAndSpeakerEmb(
        args.dataset_json,
        max_items=None,
        dur_source=dataset_dur_source,
        dur_field=dataset_dur_field,
        speaker_emb_dim=int(getattr(args, "spk_emb_dim", int(getattr(args, "spk_dim", 256)))),
        require_speaker_embeds=False,
        l2norm_speaker_embeds=bool(getattr(args, "spk_emb_l2norm", True)),
    )
    # Build a stable dense speaker map from the full manifest before --max-items slicing.
    # This keeps learned-voice table indices compatible between smoke runs and full runs.
    full_raw_speaker_ids: List[int] = []
    for _it in list(ds.items):
        if not isinstance(_it, dict):
            continue
        try:
            full_raw_speaker_ids.append(int(_it.get("speaker_id", _it.get("speaker", 0))))
        except Exception:
            full_raw_speaker_ids.append(0)
    sid_to_dense = {sid: i for i, sid in enumerate(sorted(set(full_raw_speaker_ids)))}
    if int(args.max_items) > 0:
        ds.items = ds.items[: int(args.max_items)]

    # Map speaker_id -> speaker_embeds.pt (fixed centroids).
    speaker_embeds_pt_by_id: Dict[int, str] = {}
    speaker_chunk_emb_by_id: Dict[int, torch.Tensor] = {}
    speaker_gender_by_id: Dict[int, int] = {}
    for it in ds.items:
        if not isinstance(it, dict):
            continue
        sid = it.get("speaker_id", it.get("speaker", None))
        p = it.get("speaker_embeds", None)
        if sid is None or not p:
            continue
        try:
            sid_i = int(sid)
        except Exception:
            continue
        if sid_i not in speaker_embeds_pt_by_id:
            speaker_embeds_pt_by_id[sid_i] = str(p)
        if sid_i not in speaker_chunk_emb_by_id:
            try:
                chunk_emb = it.get("spk_emb_chunk", None)
                if chunk_emb is not None:
                    e = torch.as_tensor(chunk_emb, dtype=torch.float32).view(-1)
                    if bool(getattr(args, "spk_emb_l2norm", True)):
                        e = e / e.norm().clamp_min(1e-12)
                    speaker_chunk_emb_by_id[sid_i] = e
            except Exception:
                pass
        if sid_i not in speaker_gender_by_id:
            speaker_gender_by_id[sid_i] = int(
                it.get("gender_id", _gender_id_from_name(it.get("speaker_name", it.get("author", None))))
            )

    def _load_dataset_speaker_centroid(sid: int) -> torch.Tensor:
        sid = int(sid)
        p = speaker_embeds_pt_by_id.get(sid, "")
        t = None
        if p:
            data = torch.load(str(Path(p).expanduser()), map_location="cpu")
            if isinstance(data, dict):
                emb = data.get("emb", data.get("speaker_emb", data.get("spk_emb", None)))
            else:
                emb = data
            if emb is not None:
                t = torch.as_tensor(emb, dtype=torch.float32).view(-1)
        if t is None:
            t = speaker_chunk_emb_by_id.get(sid, None)
        if t is None:
            t = torch.zeros(int(getattr(args, "spk_dim", 256)), dtype=torch.float32)
        if bool(getattr(args, "spk_emb_l2norm", True)):
            t = t / t.norm().clamp_min(1e-12)
        return t

    def _gender_ids_from_speaker_ids(speaker_ids_in: torch.Tensor, *, device: torch.device) -> torch.Tensor:
        vals: List[int] = []
        try:
            sid_list = speaker_ids_in.detach().cpu().tolist()
        except Exception:
            sid_list = [int(x) for x in speaker_ids_in]
        for sid in sid_list:
            vals.append(int(speaker_gender_by_id.get(int(sid), 0)))
        return torch.tensor(vals, dtype=torch.long, device=device)

    def _split_items_last_part_tail_per_speaker_book(items: List[Dict[str, Any]], *, val_tail_per_book: int):
        # If the dataset was created with split_books_into_parts.py, `book_id` becomes
        # book_id__pXXXXX and `chunk_idx` is reset within each part.
        #
        # For validation/demo metrics we want:
        #   - per speaker,
        #   - per original book,
        #   - take the last N chunks from the *last part only*,
        # so we don't waste data by taking N from every part.
        val_tail_per_book = int(max(1, val_tail_per_book))

        def _sid(it: Dict[str, Any]) -> int:
            try:
                return int(it.get("speaker_id", it.get("speaker", 0)))
            except Exception:
                return 0

        def _base_book(it: Dict[str, Any]) -> str:
            b = it.get("book_id_orig", None)
            if b:
                return str(b)
            bid = str(it.get("book_id", ""))
            # strip "__p00012" suffix if present
            if "__p" in bid:
                left, right = bid.rsplit("__p", 1)
                if right.isdigit() and len(right) >= 1:
                    return str(left)
            return bid

        def _part_idx(it: Dict[str, Any]) -> int:
            try:
                if "book_part_idx" in it:
                    return int(it.get("book_part_idx", 0))
            except Exception:
                pass
            bid = str(it.get("book_id", ""))
            if "__p" in bid:
                try:
                    _left, right = bid.rsplit("__p", 1)
                    if right.isdigit():
                        return int(right)
                except Exception:
                    return 0
            return 0

        def _chunk_order(it: Dict[str, Any], fallback_i: int) -> int:
            # Prefer within-part ordering when present.
            for k in ("chunk_idx", "chunk_index", "chunk_i"):
                if k in it:
                    try:
                        return int(it.get(k, fallback_i))
                    except Exception:
                        break
            # Else original chunk order.
            for k in ("chunk_idx_orig",):
                if k in it:
                    try:
                        return int(it.get(k, fallback_i))
                    except Exception:
                        break
            return int(fallback_i)

        # Track last part per (speaker, base_book) and collect indices per (speaker, base_book, part_idx).
        last_part: Dict[Tuple[int, str], int] = {}
        by_part: Dict[Tuple[int, str, int], List[Tuple[int, int]]] = defaultdict(list)  # (order, idx)
        for i, it in enumerate(items):
            if not isinstance(it, dict):
                continue
            sid = _sid(it)
            bb = _base_book(it)
            pi = _part_idx(it)
            last_part[(sid, bb)] = max(int(last_part.get((sid, bb), -1)), int(pi))
            by_part[(sid, bb, pi)].append((_chunk_order(it, i), int(i)))

        val_idx: set[int] = set()
        for (sid, bb), pi in last_part.items():
            pairs = by_part.get((sid, bb, int(pi)), [])
            if not pairs:
                continue
            pairs.sort(key=lambda x: int(x[0]))
            take = min(int(val_tail_per_book), int(len(pairs)))
            for _ord, idx in pairs[-take:]:
                val_idx.add(int(idx))

        train_items: List[Dict[str, Any]] = []
        val_items: List[Dict[str, Any]] = []
        for i, it in enumerate(items):
            if not isinstance(it, dict):
                continue
            if int(i) in val_idx:
                val_items.append(it)
            else:
                train_items.append(it)

        speakers = {k[0] for k in last_part.keys()}
        books = {(k[0], k[1]) for k in last_part.keys()}
        return train_items, val_items, len(speakers), len(books), int(val_tail_per_book)

    ds_train = AlignedTTSDatasetWithAudioAndSpeakerEmb(
        args.dataset_json,
        max_items=None,
        dur_source=dataset_dur_source,
        dur_field=dataset_dur_field,
        speaker_emb_dim=int(getattr(args, "spk_emb_dim", int(getattr(args, "spk_dim", 256)))),
        require_speaker_embeds=False,
        l2norm_speaker_embeds=bool(getattr(args, "spk_emb_l2norm", True)),
        prior_f0_cache_field=str(getattr(args, "prior_f0_cache_field", "fcpe_f0_path")),
    )
    ds_val = AlignedTTSDatasetWithAudioAndSpeakerEmb(
        args.dataset_json,
        max_items=None,
        dur_source=dataset_dur_source,
        dur_field=dataset_dur_field,
        speaker_emb_dim=int(getattr(args, "spk_emb_dim", int(getattr(args, "spk_dim", 256)))),
        require_speaker_embeds=False,
        l2norm_speaker_embeds=bool(getattr(args, "spk_emb_l2norm", True)),
        prior_f0_cache_field=str(getattr(args, "prior_f0_cache_field", "fcpe_f0_path")),
    )
    # Default: last 4 chunks per (speaker, original book) from the last part only.
    train_items, val_items, n_speakers_seen, n_books_seen, val_tail = _split_items_last_part_tail_per_speaker_book(
        list(ds.items),
        val_tail_per_book=int(getattr(args, "val_tail_per_book", 4)),
    )
    # Learned voice tables must be indexed densely. Some manifests keep sparse/global
    # speaker IDs (e.g. >1M), which would otherwise allocate huge mostly-empty tables.
    for _it in list(train_items) + list(val_items):
        if not isinstance(_it, dict):
            continue
        try:
            raw_sid = int(_it.get("speaker_id", _it.get("speaker", 0)))
        except Exception:
            raw_sid = 0
        dense_sid = int(sid_to_dense.get(raw_sid, 0))
        _it.setdefault("speaker_id_raw", raw_sid)
        _it["speaker_id"] = dense_sid
        _it["speaker"] = dense_sid
    ds_train.items = train_items
    ds_val.items = val_items
    print(
        f"🔪 train/val split: last {int(val_tail)} chunks per (speaker,book) from last part -> "
        f"train_items={len(ds_train.items)} val_items={len(ds_val.items)} speakers={int(n_speakers_seen)} books={int(n_books_seen)}"
    )
    if sid_to_dense:
        print(
            "🧬 learned_voice speaker_id remap: "
            f"raw_speakers={len(sid_to_dense)} dense_range=0..{len(sid_to_dense)-1}"
        )
    if bool(getattr(args, "boundary_jitter", False)):
        ds_train = BoundaryJitterDataset(
            ds_train,
            prob=float(getattr(args, "boundary_jitter_prob", 0.25)),
            max_words=int(getattr(args, "boundary_jitter_max_words", 2)),
            seed=int(getattr(args, "boundary_jitter_seed", int(getattr(args, "seed", 1234)))),
            epoch_vary=bool(getattr(args, "boundary_jitter_epoch_vary", True)),
            min_frames=int(getattr(args, "boundary_jitter_min_frames", 24)),
            max_frames=int(getattr(args, "boundary_jitter_max_frames", 0)),
        )
        print(
            "🧩 boundary-jitter: enabled "
            f"prob={float(getattr(args, 'boundary_jitter_prob', 0.25)):.3f} "
            f"max_words={int(getattr(args, 'boundary_jitter_max_words', 2))} "
            f"min_frames={int(getattr(args, 'boundary_jitter_min_frames', 24))} "
            f"max_frames={int(getattr(args, 'boundary_jitter_max_frames', 0))} "
            f"epoch_vary={bool(getattr(args, 'boundary_jitter_epoch_vary', True))}"
        )

    # Uwaga: bridge_cache jest stanowy per (speaker_id, book_id), więc losowe tasowanie oraz duplikaty klucza w batchu
    # psują przekazywanie stanu między chunkami. StatefulChunkBatchSampler gwarantuje unikalny (speaker_id,book_id) w batchu.
    stateful_sampler = None
    if bool(getattr(args, "stateful_batching", True)):
        # Optional ablation helper: if `--max-items` contains too few unique (speaker,book),
        # split a single book into multiple "virtual lanes" so batch_size can stay >1.
        virtual_meta = list(ds_train.items)
        do_virtual = bool(getattr(args, "virtual_lanes", True)) and (int(getattr(args, "max_items", 0)) <= int(getattr(args, "virtual_lanes_threshold", 2000)))
        if do_virtual:
            keys = set()
            for it in ds_train.items:
                try:
                    sid = int(it.get("speaker_id", 0))
                except Exception:
                    sid = 0
                bid = str(it.get("book_id", ""))
                keys.add((sid, bid))
            if len(keys) < int(args.batch_size) and int(args.batch_size) > 1 and len(ds_train.items) > 1:
                virtual_meta = _build_virtual_lane_meta(items=list(ds_train.items), lanes_per_key=int(args.batch_size))
                ds_train = VirtualLaneDataset(ds_train, virtual_meta=virtual_meta)  # type: ignore[assignment]
                print(f"🧵 virtual-lanes: keys {len(keys)} -> {len(set((m.get('speaker_id',0), m.get('book_id','')) for m in virtual_meta))} (batch_size={int(args.batch_size)})")

        stateful_mode = str(getattr(args, "stateful_mode", "round_robin")).lower().strip()
        SamplerCls = FixedSlotChunkBatchSampler if stateful_mode == "fixed" else StatefulChunkBatchSampler
        stateful_sampler = SamplerCls(
            items=list(virtual_meta),
            batch_size=int(args.batch_size),
            # drop_last=True może dać 0 batchy, gdy jest mało unikalnych (speaker_id, book_id) (np. jedna książka).
            drop_last=False,
            seed=int(getattr(args, "seed", 1234)),
            shuffle_keys=bool(getattr(args, "stateful_shuffle_keys", False)),
        )
        dl_kwargs = {}
        if int(getattr(args, "workers", 0)) > 0:
            dl_kwargs["persistent_workers"] = bool(getattr(args, "persistent_workers", True))
            dl_kwargs["prefetch_factor"] = int(getattr(args, "prefetch_factor", 4))
        dl_train = torch.utils.data.DataLoader(
            ds_train,
            batch_sampler=stateful_sampler,
            collate_fn=collate_fn_with_audio_and_speaker_emb,
            num_workers=int(getattr(args, "workers", 0)),
            pin_memory=(device.type == "cuda"),
            **dl_kwargs,
        )
    else:
        dl_kwargs = {}
        if int(getattr(args, "workers", 0)) > 0:
            dl_kwargs["persistent_workers"] = bool(getattr(args, "persistent_workers", True))
            dl_kwargs["prefetch_factor"] = int(getattr(args, "prefetch_factor", 4))
        dl_train = torch.utils.data.DataLoader(
            ds_train,
            batch_size=int(args.batch_size),
            shuffle=False,
            drop_last=True,
            collate_fn=collate_fn_with_audio_and_speaker_emb,
            num_workers=int(getattr(args, "workers", 0)),
            pin_memory=(device.type == "cuda"),
            **dl_kwargs,
        )
    dl_demo_kwargs = {}
    if int(getattr(args, "workers", 0)) > 0:
        dl_demo_kwargs["persistent_workers"] = bool(getattr(args, "persistent_workers", True))
        dl_demo_kwargs["prefetch_factor"] = int(getattr(args, "prefetch_factor", 4))
    # Use the compact val subset for demos/health metrics (speaker-balanced).
    ds_demo = ds_val if len(ds_val) > 0 else ds_train
    dl_demo = torch.utils.data.DataLoader(
        ds_demo,
        batch_size=max(1, int(getattr(args, "demo_count", 1))),
        shuffle=False,
        drop_last=False,
        collate_fn=collate_fn_with_audio_and_speaker_emb,
        num_workers=int(getattr(args, "workers", 0)),
        pin_memory=(device.type == "cuda"),
        **dl_demo_kwargs,
    )
    speaker_to_demo_indices: Dict[int, List[int]] = defaultdict(list)
    for ii, it in enumerate(ds_demo.items):
        if not isinstance(it, dict):
            continue
        sid = it.get("speaker_id", it.get("speaker", None))
        if sid is None:
            continue
        try:
            speaker_to_demo_indices[int(sid)].append(int(ii))
        except Exception:
            continue
    speaker_to_train_indices: Dict[int, List[int]] = defaultdict(list)
    for ii, it in enumerate(ds_train.items):
        if not isinstance(it, dict):
            continue
        sid = it.get("speaker_id", it.get("speaker", None))
        if sid is None:
            continue
        try:
            speaker_to_train_indices[int(sid)].append(int(ii))
        except Exception:
            continue
    speaker_book_chunk_to_train_idx: Dict[Tuple[int, str, int], int] = {}
    speaker_book_to_train_indices: Dict[Tuple[int, str], List[Tuple[int, int]]] = defaultdict(list)
    for ii, it in enumerate(ds_train.items):
        if not isinstance(it, dict):
            continue
        try:
            sid_i = int(it.get("speaker_id", it.get("speaker", 0)))
            bid_s = str(it.get("book_id", it.get("book", "")))
            cidx_i = int(it.get("chunk_idx", it.get("chunk_index", 0)))
        except Exception:
            continue
        speaker_book_chunk_to_train_idx[(sid_i, bid_s, cidx_i)] = int(ii)
        speaker_book_to_train_indices[(sid_i, bid_s)].append((cidx_i, int(ii)))
    for key in list(speaker_book_to_train_indices.keys()):
        speaker_book_to_train_indices[key].sort(key=lambda x: int(x[0]))

    _ref_source_counts: Counter[str] = Counter()

    def _sample_dualhead_ref_batch(
        *,
        ds_obj,
        target_mel_bct: torch.Tensor,
        target_T_len: torch.Tensor,
        speaker_ids_bt: torch.Tensor,
        book_ids_obj,
        chunk_idx_bt,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        mode = str(getattr(args, "dualhead_ref_source", "same_chunk")).lower().strip()
        if mode == "same_chunk":
            _ref_source_counts["same_chunk"] += int(target_mel_bct.size(0))
            return target_mel_bct, target_T_len

        try:
            sid_list = [int(x) for x in speaker_ids_bt.detach().cpu().tolist()]
        except Exception:
            sid_list = [int(x) for x in speaker_ids_bt]
        try:
            chunk_list = [int(x) for x in chunk_idx_bt.detach().cpu().tolist()] if torch.is_tensor(chunk_idx_bt) else [int(x) for x in chunk_idx_bt]
        except Exception:
            chunk_list = [0 for _ in sid_list]
        try:
            book_list = [str(x) for x in book_ids_obj]
        except Exception:
            book_list = ["" for _ in sid_list]

        ref_mels: List[torch.Tensor] = []
        ref_lens: List[int] = []
        for bi, sid_i in enumerate(sid_list):
            bid_s = str(book_list[bi]) if bi < len(book_list) else ""
            cidx_i = int(chunk_list[bi]) if bi < len(chunk_list) else 0
            chosen_idx: Optional[int] = None
            current_idx = speaker_book_chunk_to_train_idx.get((sid_i, bid_s, cidx_i))
            tag = "same_chunk"

            if mode == "neighbor_same_book":
                prev_idx = speaker_book_chunk_to_train_idx.get((sid_i, bid_s, cidx_i - 1))
                next_idx = speaker_book_chunk_to_train_idx.get((sid_i, bid_s, cidx_i + 1))
                if prev_idx is not None:
                    chosen_idx = int(prev_idx)
                    tag = "neighbor_prev"
                elif next_idx is not None:
                    chosen_idx = int(next_idx)
                    tag = "neighbor_next"

            if chosen_idx is None and mode in ("neighbor_same_book", "random_same_speaker"):
                pool = list(speaker_to_train_indices.get(int(sid_i), []))
                if current_idx is not None and len(pool) > 1:
                    pool = [int(x) for x in pool if int(x) != int(current_idx)]
                if pool:
                    chosen_idx = int(random.choice(pool))
                    tag = "random_same_speaker"

            if chosen_idx is None:
                ref_mels.append(target_mel_bct[bi : bi + 1].detach().to(dtype=torch.float32))
                t_i = int(target_T_len[bi].detach().cpu().item()) if torch.is_tensor(target_T_len) else int(target_mel_bct.size(-1))
                ref_lens.append(max(1, t_i))
                _ref_source_counts["same_chunk_fallback"] += 1
                continue

            try:
                ref_item = ds_obj[int(chosen_idx)]
                mel_ref = _ensure_mel_bct(ref_item["mel"]).to(dtype=torch.float32)
                t_ref = int(ref_item.get("T_mel", int(mel_ref.size(-1))))
                mel_ref = mel_ref[:, :, : max(1, t_ref)].contiguous()
                ref_mels.append(mel_ref)
                ref_lens.append(max(1, t_ref))
                _ref_source_counts[tag] += 1
            except Exception:
                ref_mels.append(target_mel_bct[bi : bi + 1].detach().to(dtype=torch.float32))
                t_i = int(target_T_len[bi].detach().cpu().item()) if torch.is_tensor(target_T_len) else int(target_mel_bct.size(-1))
                ref_lens.append(max(1, t_i))
                _ref_source_counts["same_chunk_error_fallback"] += 1

        Tm = int(max(ref_lens)) if ref_lens else int(target_mel_bct.size(-1))
        ref_batch = torch.cat(
            [
                _crop_or_pad_bct(m, Tm).to(device=target_mel_bct.device, dtype=torch.float32)
                for m in ref_mels
            ],
            dim=0,
        )
        ref_len_t = torch.tensor(ref_lens, dtype=torch.long, device=target_mel_bct.device)
        return ref_batch, ref_len_t

    # -------- Base model (z ckpt albo od zera) --------
    from utils import BenchmarkConfig

    train_base = (str(getattr(args, "ckpt", "")).strip() == "")
    m_args = {}
    if train_base:
        hidden_dim = int(getattr(args, "hidden_dim", 512))
        num_layers = int(getattr(args, "layers", 8))
        num_speakers = int(getattr(args, "num_speakers", 256))
        n_heads = int(getattr(args, "heads", 8))
        use_sdpa = not bool(getattr(args, "no_sdpa", False))
        ckpt_path = None
    else:
        ckpt_path = Path(str(args.ckpt))
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        m_args = ckpt.get("args", {}) or {}
        hidden_dim = int(m_args.get("hidden_dim", 512))
        num_layers = int(m_args.get("layers", 8))
        num_speakers = int(m_args.get("num_speakers", 256))
        # --- Text encoder config (zgodne z WegorzTTS2/3) ---
        n_heads = int(m_args.get("heads", m_args.get("n_heads", 6)))
        use_sdpa = bool(m_args.get("use_sdpa", False))

    max_dataset_speaker_id = 0
    for _it in list(ds_train.items) + list(ds_val.items):
        if not isinstance(_it, dict):
            continue
        try:
            max_dataset_speaker_id = max(max_dataset_speaker_id, int(_it.get("speaker_id", _it.get("speaker", 0))))
        except Exception:
            continue
    needed_num_speakers = max(int(max_dataset_speaker_id) + 1, int(len(sid_to_dense)))
    if needed_num_speakers > int(num_speakers):
        print(
            "🧬 learned_voice num_speakers expanded from checkpoint/default "
            f"{int(num_speakers)} -> {int(needed_num_speakers)} "
            f"(max dataset speaker_id={int(max_dataset_speaker_id)})"
        )
        num_speakers = int(needed_num_speakers)

    text_dualpath_projection = bool(getattr(args, "text_dualpath_projection", True))
    text_dualpath_attn_dim = int(getattr(args, "text_dualpath_attn_dim", hidden_dim // 2))
    text_dualpath_conv_dim = int(getattr(args, "text_dualpath_conv_dim", hidden_dim // 2))
    if text_dualpath_projection and int(text_dualpath_attn_dim) % int(n_heads) != 0:
        raise ValueError(
            f"--text-dualpath-attn-dim must divide text encoder heads "
            f"(got text_dualpath_attn_dim={text_dualpath_attn_dim}, heads={n_heads})"
        )

    model = GenericTTS(
        BenchmarkConfig(vocab_size=len(SYMBOL2ID), hidden_dim=hidden_dim, num_layers=num_layers, n_mels=N_MELS),
        n_layers=num_layers,
        n_heads=n_heads,
        use_sdpa=use_sdpa,
        dualpath_branch_mode="projected" if text_dualpath_projection else "split",
        dualpath_attn_dim=int(text_dualpath_attn_dim) if text_dualpath_projection else None,
        dualpath_conv_dim=int(text_dualpath_conv_dim) if text_dualpath_projection else None,
        dualpath_init_split_identity=bool(getattr(args, "dualpath_init_split_identity", True)),
    ).to(device)
    if text_dualpath_projection:
        print(
            "🧪 text-encoder projection: "
            f"attn_dim={int(text_dualpath_attn_dim)} conv_dim={int(text_dualpath_conv_dim)} "
            f"init_split_identity={bool(getattr(args, 'dualpath_init_split_identity', True))}"
        )

    spk_embed = SpeakerAdapter(num_speakers=num_speakers, hidden_dim=hidden_dim).to(device)
    # Learned-voice mode uses learned_spk_table -> spk_adapter as the only speaker
    # prefix source. Keep the legacy SpeakerAdapter present for API/checkpoint shape
    # compatibility, but do not train or rely on its internal table.
    spk_embed.eval()
    for _p in spk_embed.parameters():
        _p.requires_grad_(False)
    spk_adapter = SpeakerVecAdapter(
        int(getattr(args, "spk_dim", 256)),
        int(hidden_dim),
        p_drop=float(getattr(args, "spk_emb_drop", 0.1)),
    ).to(device)
    learned_spk_table = nn.Embedding(int(num_speakers), int(getattr(args, "spk_dim", 256))).to(device)
    learned_style_table = nn.Embedding(int(num_speakers), 128).to(device)
    nn.init.normal_(learned_spk_table.weight, mean=0.0, std=0.02)
    nn.init.normal_(learned_style_table.weight, mean=0.0, std=0.02)
    if str(getattr(args, "speaker_vector_source", "")).lower().strip() == "learned_voice":
        print(
            "🧬 learned_voice tables: enabled "
            f"spk=[{int(num_speakers)},{int(getattr(args, 'spk_dim', 256))}] "
            f"style=[{int(num_speakers)},128]"
        )
    gender_dim = int(getattr(args, "gender_emb_dim", hidden_dim))
    if int(gender_dim) == int(hidden_dim):
        gender_embed = nn.Embedding(3, int(hidden_dim)).to(device)
    else:
        gender_embed = nn.Sequential(
            nn.Embedding(3, int(gender_dim)),
            nn.Linear(int(gender_dim), int(hidden_dim)),
        ).to(device)
    emotion_conditioning_enabled = bool(getattr(args, "emotion_conditioning", False))
    emotion_embed = nn.Embedding(len(EMOTION_GROUP_TO_ID), int(getattr(args, "emotion_emb_dim", 128))).to(device)
    emotion_token_embed = nn.Embedding(len(EMOTION_GROUP_TO_ID), int(hidden_dim)).to(device)
    emotion_to_style = nn.Linear(int(getattr(args, "emotion_emb_dim", 128)), 128).to(device)
    emotion_style_gate = nn.Parameter(torch.zeros((), device=device))
    nn.init.normal_(emotion_to_style.weight, mean=0.0, std=0.02)
    nn.init.zeros_(emotion_to_style.bias)
    with torch.no_grad():
        try:
            neutral_tok = gender_embed(torch.zeros(1, dtype=torch.long, device=device)).detach().view(-1)
            emotion_token_embed.weight.copy_(neutral_tok[None, :].repeat(len(EMOTION_GROUP_TO_ID), 1))
            if len(EMOTION_GROUP_TO_ID) > 1:
                emotion_token_embed.weight[1:].add_(0.01 * torch.randn_like(emotion_token_embed.weight[1:]))
        except Exception:
            nn.init.normal_(emotion_token_embed.weight, mean=0.0, std=0.02)
    mem_dim = int(CONFIG.get("context_mem_dim", 0))
    if mem_dim <= 0:
        mem_dim = int(hidden_dim)
    bridge = ContextBridge(
        text_dim=int(hidden_dim),
        mem_dim=int(mem_dim),
        attn_heads=int(CONFIG.get("context_attn_heads", 4)),
        attn_drop=float(CONFIG.get("context_attn_drop", 0.0)),
    ).to(device)
    bridge_cache = ContextBridgeCache()

    tempo_predictor = nn.Identity().to(device)

    if not train_base and ckpt_path is not None:
        _ = load_checkpoint(
            path=str(ckpt_path),
            device=device,
            model=model,
            bridge=None,
            spk_embed=spk_embed,
            tempo_predictor=tempo_predictor,
            style_head=None,
            style_to_token=None,
            opt=None,
            scaler=None,
        )

        # freeze base
        model.eval()
        spk_embed.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        for p in spk_embed.parameters():
            p.requires_grad_(False)

    # -------- Trainable modules (skalpel mode) --------
    prior_layers = max(1, int(args.prior_layers))
    prior_heads = max(1, int(args.prior_heads))
    dualpath_branch_dim = int(getattr(args, "dualpath_branch_dim", hidden_dim // 2))
    dualpath_attn_dim = int(getattr(args, "dualpath_attn_dim", 0) or dualpath_branch_dim)
    dualpath_conv_dim = int(getattr(args, "dualpath_conv_dim", 0) or dualpath_branch_dim)
    if int(hidden_dim) % int(prior_heads) != 0:
        raise SystemExit(f"--prior-heads must divide --hidden-dim (got hidden_dim={hidden_dim}, prior_heads={prior_heads})")
    if int(dualpath_attn_dim) % int(prior_heads) != 0:
        raise SystemExit(
            f"--dualpath-attn-dim must divide --prior-heads/--flow-heads usage "
            f"(got dualpath_attn_dim={dualpath_attn_dim}, prior_heads={prior_heads})"
        )
    prior_mu = ProbabilisticPriorDecoder(
        dim=hidden_dim,
        n_mels=N_MELS,
        n_layers=prior_layers,
        n_heads=prior_heads,
        logs_min=float(args.prior_logs_min),
        logs_max=float(args.prior_logs_max),
        dualpath_branch_dim=None,
        dualpath_attn_dim=int(dualpath_attn_dim),
        dualpath_conv_dim=int(dualpath_conv_dim),
        dualpath_init_split_identity=bool(getattr(args, "dualpath_init_split_identity", True)),
    ).to(device)
    prior_prosody = PriorProsodyHeads(dim=int(hidden_dim)).to(device)
    use_dur_prior = str(args.dur_x0).lower().strip() == "prior"
    duration_model_name = str(getattr(args, "duration_model", "lstm_ar")).lower().strip()
    if duration_model_name == "mini_dualpath":
        dur_predictor = MiniDualPathDurationPredictor(
            dim=hidden_dim,
            layers=int(getattr(args, "mini_dur_layers", 3)),
            attn_dim=int(getattr(args, "mini_dur_attn_dim", 128)),
            conv_dim=int(getattr(args, "mini_dur_conv_dim", 128)),
            heads=int(getattr(args, "mini_dur_heads", 4)),
            dropout=float(getattr(args, "mini_dur_dropout", 0.1)),
        ).to(device)
    else:
        dur_predictor = ARDurationTransformer(
            dim=hidden_dim,
            layers=int(getattr(args, "ar_dur_layers", 4)),
            heads=int(getattr(args, "ar_dur_heads", 4)),
            ffn_mult=int(getattr(args, "ar_dur_ffn_mult", 4)),
            dropout=float(getattr(args, "ar_dur_dropout", 0.1)),
        ).to(device)
    flow_layers = max(1, int(args.flow_layers))
    flow_heads = max(1, int(args.flow_heads))
    if int(hidden_dim) % int(flow_heads) != 0:
        raise SystemExit(f"--flow-heads must divide --hidden-dim (got hidden_dim={hidden_dim}, flow_heads={flow_heads})")
    if int(dualpath_attn_dim) % int(flow_heads) != 0:
        raise SystemExit(
            f"--dualpath-attn-dim must divide --flow-heads "
            f"(got dualpath_attn_dim={dualpath_attn_dim}, flow_heads={flow_heads})"
        )
    mel_flow = SpeakerlessMelFlowDecoder(
        dim=hidden_dim,
        text_dim=hidden_dim,
        n_mels=N_MELS,
        heads=flow_heads,
        layers=flow_layers,
        num_speakers=num_speakers,
        use_timbre_adaln=bool(getattr(args, "flow_timbre_adaln", False)),
        dualpath_branch_dim=None,
        dualpath_attn_dim=int(dualpath_attn_dim),
        dualpath_conv_dim=int(dualpath_conv_dim),
        dualpath_init_split_identity=bool(getattr(args, "dualpath_init_split_identity", True)),
        flow_gauss_token_attn=bool(getattr(args, "flow_gauss_token_attn", False)),
        flow_gauss_cross_attn=bool(getattr(args, "flow_gauss_cross_attn", False)),
        flow_gauss_token_group_size=1,
    ).to(device)
    print(
        "🧪 dualpath-projection: "
        f"attn_dim={int(dualpath_attn_dim)} conv_dim={int(dualpath_conv_dim)} "
        f"init_split_identity={bool(getattr(args, 'dualpath_init_split_identity', True))}"
    )
    if bool(getattr(args, "flow_gauss_token_attn", False)):
        print(
            "🧪 flow-gauss-token-attn: enabled "
            "(DualPath flow attention branch runs on Gaussian token slots, group_size=1)"
        )
    else:
        print("🧪 flow-gauss-token-attn: disabled")
    if bool(getattr(args, "flow_gauss_cross_attn", False)):
        print("🧪 flow-gauss-cross-attn: enabled (flow text cross-attn runs on Gaussian token slots)")
    else:
        print("🧪 flow-gauss-cross-attn: disabled")
    print("🧬 learned-voice: speaker/style nn.Embedding tables -> speaker prefix + duration style conditioning")

    # -------- Speakder dualhead-derived speaker encoder --------
    speaker_loss_enabled = float(getattr(args, "speaker_loss_w", 0.0)) > 0.0
    speaker_teacher_enabled = float(getattr(args, "speaker_teacher_w", 0.0)) > 0.0
    if speaker_loss_enabled or speaker_teacher_enabled:
        raise RuntimeError(
            "learned_voice clean variant does not support legacy dualhead speaker_loss/speaker_teacher. "
            "Use optional --speaker-verifier wavlm_tbr for final speaker/timbre finetuning."
        )
    speaker_encoder: Optional[nn.Module] = None
    use_style_encoder128 = False
    needs_frozen_dualhead = False
    speaker_encoder_enabled = False
    if needs_frozen_dualhead or speaker_encoder_enabled:
        spkenc_dir = (_PO_ROOT / "Speakder_enkoder")
        if spkenc_dir.is_dir() and str(spkenc_dir) not in sys.path:
            sys.path.append(str(spkenc_dir))
        MelDualEncoder = None
        if needs_frozen_dualhead or (speaker_encoder_enabled and not use_style_encoder128):
            try:
                from spk_style_dualhead_model import MelDualEncoder  # type: ignore
            except Exception as exc:
                raise RuntimeError(f"Cannot import Speakder dualhead model (spk_style_dualhead_model.py): {exc}")

        ref_dev = str(getattr(args, "spk_style_device", "auto")).lower().strip()
        if ref_dev == "auto":
            ref_device = device
        else:
            ref_device = torch.device(ref_dev)

        if needs_frozen_dualhead:
            assert MelDualEncoder is not None
            mel_flow.spk_style_ref = MelDualEncoder(
                n_mels=int(N_MELS),
                d_spk=int(getattr(args, "spk_dim", 256)),
                d_style=128,
                hidden_dim=256,
                num_layers=4,
                attn_head_dim=128,
                style_project_off_spk=True,
            ).to(ref_device).eval()
            for p in mel_flow.spk_style_ref.parameters():  # type: ignore[union-attr]
                p.requires_grad_(False)
            mel_flow.spk_style_ref_ready = False  # set True when loaded from ckpt/resume
        if speaker_encoder_enabled:
            if use_style_encoder128:
                if _WegorzStyleEncoder128 is None:
                    raise RuntimeError(f"Cannot import WęgorzStyleEncoder128 from {STYLE_ENCODER128_DIR}")
                ckpt_p = Path(str(getattr(args, "style_encoder128_ckpt", ""))).expanduser()
                payload128 = torch.load(str(ckpt_p), map_location="cpu", weights_only=False)
                enc_args = dict(payload128.get("args") or {})
                enc128 = _WegorzStyleEncoder128(
                    n_mels=int(enc_args.get("n_mels", N_MELS)),
                    spk_dim=int(enc_args.get("spk_dim", 128)),
                    style_dim=int(enc_args.get("style_dim", 128)),
                    hidden_dim=int(enc_args.get("hidden_dim", 256)),
                    layers=int(enc_args.get("layers", 4)),
                    attn_head_dim=int(enc_args.get("attn_head_dim", 128)),
                    shared_backbone=bool(enc_args.get("shared_backbone", False)),
                )
                enc128.load_state_dict(payload128.get("model", payload128), strict=True)
                speaker_encoder = FrozenStyleEncoder128ForTTS(
                    enc128,
                    spk_out_dim=int(getattr(args, "spk_dim", 256)),
                    train_encoder=bool(getattr(args, "style_encoder128_trainable", False)),
                ).to(ref_device)
                print(
                    "✅ Loaded style_encoder128 for TTS reference encoding: "
                    f"{ckpt_p} (encoder_trainable={bool(getattr(args, 'style_encoder128_trainable', False))}, "
                    "spk_proj_trainable=True)"
                )
            else:
                assert MelDualEncoder is not None
                speaker_encoder = MelDualEncoder(
                    n_mels=int(N_MELS),
                    d_spk=int(getattr(args, "spk_dim", 256)),
                    d_style=128,
                    hidden_dim=256,
                    num_layers=4,
                    attn_head_dim=128,
                    style_project_off_spk=True,
                ).to(ref_device)
                if bool(getattr(args, "speaker_encoder_trainable", True)):
                    speaker_encoder.train()
                    for p in speaker_encoder.parameters():
                        p.requires_grad_(True)
                else:
                    speaker_encoder.eval()
                    for p in speaker_encoder.parameters():
                        p.requires_grad_(False)

        spk_style_ckpt = str(getattr(args, "spk_style_ckpt", "")).strip()
        if spk_style_ckpt:
            ckpt_p = Path(spk_style_ckpt).expanduser()
            try:
                # torch>=2.6 defaults to weights_only=True and requires allowlisting custom globals.
                # Speakder ckpts may reference spk_enc_model.ArcFaceConfig and __main__.SupConConfig.
                try:
                    import torch.serialization as _ts  # type: ignore
                except Exception:  # pragma: no cover
                    _ts = None

                payload = None
                if _ts is not None and hasattr(_ts, "safe_globals"):
                    try:
                        import spk_enc_model as _spkmod  # type: ignore
                        allow = [_spkmod.ArcFaceConfig, SupConConfig]
                    except Exception:
                        allow = [SupConConfig]
                    with _ts.safe_globals(allow):  # type: ignore[attr-defined]
                        payload = torch.load(str(ckpt_p), map_location="cpu", weights_only=True)  # type: ignore[call-arg]
                else:
                    payload = torch.load(str(ckpt_p), map_location="cpu")
            except Exception as exc:
                msg = str(exc)
                if ("Weights only load failed" in msg or "weights_only" in msg) and bool(getattr(args, "spk_style_unsafe", False)):
                    # Unsafe fallback: full unpickle (requires SupConConfig symbol in __main__).
                    payload = torch.load(str(ckpt_p), map_location="cpu", weights_only=False)  # type: ignore[call-arg]
                else:
                    raise
            if not isinstance(payload, dict) or ("model" not in payload):
                raise RuntimeError(f"Bad --spk-style-ckpt payload (missing 'model'): {ckpt_p}")
            if needs_frozen_dualhead:
                inc = mel_flow.spk_style_ref.load_state_dict(payload["model"], strict=False)  # type: ignore[union-attr]
                if any(str(k).startswith("pooling_style.") for k in getattr(inc, "missing_keys", [])):
                    _sync_missing_style_pooling(mel_flow.spk_style_ref)  # type: ignore[arg-type,union-attr]
                    print("ℹ️ initialized frozen spk_style pooling_style from speaker pooling.")
                mel_flow.spk_style_ref_ready = True
                print(f"✅ Loaded frozen spk_style dualhead: {ckpt_p}")
            if speaker_encoder is not None:
                if use_style_encoder128:
                    print("ℹ️ skipping dualhead load into style_encoder128 speaker_encoder.")
                else:
                    inc = speaker_encoder.load_state_dict(payload["model"], strict=False)
                    if any(str(k).startswith("pooling_style.") for k in getattr(inc, "missing_keys", [])):
                        _sync_missing_style_pooling(speaker_encoder)
                        print("ℹ️ initialized speaker_encoder pooling_style from speaker pooling.")
                    mode = "trainable" if bool(getattr(args, "speaker_encoder_trainable", True)) else "frozen"
                    print(f"✅ Loaded {mode} speaker_encoder copy from dualhead: {ckpt_p}")
        else:
            if (not bool(getattr(args, "infer_only", False))) and (not bool(getattr(args, "resume", ""))) and (speaker_loss_enabled or speaker_teacher_enabled):
                raise RuntimeError("--spk-style-ckpt is empty and no --resume was provided, but this run needs a frozen dualhead encoder.")

    # attach for convenience
    model.dec_mu = prior_mu
    model.dur = dur_predictor

    # duration inference: legacy flow sampler (dur_x0=prior when available)
    emotion_params: List[nn.Parameter] = []
    if bool(emotion_conditioning_enabled):
        emotion_params = (
            list(emotion_embed.parameters())
            + list(emotion_token_embed.parameters())
            + list(emotion_to_style.parameters())
            + [emotion_style_gate]
        )

    train_params = (
        list(prior_mu.parameters())
        + list(prior_prosody.parameters())
        + list(dur_predictor.parameters())
        + list(mel_flow.parameters())
        + list(spk_adapter.parameters())
        + list(learned_spk_table.parameters())
        + list(learned_style_table.parameters())
        + list(gender_embed.parameters())
        + emotion_params
    )
    speaker_encoder_params: List[nn.Parameter] = []
    if speaker_encoder is not None:
        # In the style_encoder128 experiment the encoder itself is frozen by default,
        # but the spk128->spk256 projection remains trainable and must enter the optimizer.
        speaker_encoder_params = [p for p in speaker_encoder.parameters() if bool(getattr(p, "requires_grad", False))]
        train_params += speaker_encoder_params
    if train_base:
        train_params += list(model.parameters()) + list(bridge.parameters())

    # Uniknij duplikatów (np. gdy model.dec_mu=model.prior_mu albo model.dur=dur_predictor)
    seen = set()
    unique_params = []
    for p in train_params:
        if not bool(getattr(p, "requires_grad", False)):
            continue
        pid = id(p)
        if pid in seen:
            continue
        seen.add(pid)
        unique_params.append(p)

    speaker_encoder_param_ids = {id(p) for p in speaker_encoder_params}
    duration_param_ids = {id(p) for p in dur_predictor.parameters() if bool(getattr(p, "requires_grad", False))}
    main_params = [
        p for p in unique_params
        if id(p) not in speaker_encoder_param_ids and id(p) not in duration_param_ids
    ]
    spkenc_params = [p for p in unique_params if id(p) in speaker_encoder_param_ids]
    duration_params = [p for p in unique_params if id(p) in duration_param_ids]
    param_groups = []
    if main_params:
        param_groups.append({"params": main_params, "lr": float(args.lr)})
    if duration_params:
        param_groups.append({"params": duration_params, "lr": float(args.lr) * float(getattr(args, "duration_lr_mult", 1.0))})
    if spkenc_params:
        param_groups.append({"params": spkenc_params, "lr": float(args.lr) * float(getattr(args, "speaker_encoder_lr_mult", 1.0))})
    if param_groups:
        opt = torch.optim.AdamW(param_groups, lr=float(args.lr))
        if duration_params:
            print(
                "⏱️ duration optimizer params "
                f"params={sum(p.numel() for p in duration_params):,} "
                f"lr={float(args.lr) * float(getattr(args, 'duration_lr_mult', 1.0)):.3g}"
            )
    else:
        opt = torch.optim.AdamW(unique_params, lr=float(args.lr))
    if spkenc_params:
        print(
            "🧬 speaker_encoder adapter/trainable params "
            f"params={sum(p.numel() for p in spkenc_params):,} "
            f"lr={float(args.lr) * float(getattr(args, 'speaker_encoder_lr_mult', 1.0)):.3g}"
        )
    elif not spkenc_params:
        if speaker_encoder is not None:
            print("🧬 speaker_encoder: fully frozen (no optimizer params)")

    # -------- Resume --------
    start_epoch = 1
    resume_payload = None
    resume_rpath = None
    resume_skip_optim = False
    if args.resume:
        rpath = Path(str(args.resume)).expanduser()
        resume_rpath = rpath
        if not rpath.exists():
            msg = f"⚠️ --resume path not found, skipping checkpoint load: {rpath}"
            print(msg)
            if bool(getattr(args, "infer_only", False)):
                raise RuntimeError(msg + " (infer-only would run with random weights)")
        else:
            payload = torch.load(rpath, map_location="cpu")
            resume_payload = payload
            if bool(payload.get("train_base", False)):
                skipped_base = _partial_load(model, dict(payload.get("base_model", {})))
                if len(skipped_base) > 0:
                    print(f"ℹ️ base_model partial load: skipped={len(skipped_base)}")
                try:
                    bridge.load_state_dict(payload.get("bridge", {}), strict=False)
                except Exception as exc:
                    print(f"⚠️ Nie udało się wczytać bridge: {exc}")
                skipped_spk_embed = _partial_load(spk_embed, dict(payload.get("spk_embed", {})))
                if skipped_spk_embed:
                    print("ℹ️ legacy spk_embed partial load skipped mismatched table rows (learned_voice uses learned_spk_table).")
            if "spk_adapter" in payload:
                try:
                    spk_adapter.load_state_dict(payload.get("spk_adapter", {}), strict=False)
                except Exception as exc:
                    print(f"⚠️ Nie udało się wczytać spk_adapter: {exc}")
                    # Speaker adapter input dim differs between ECAPA (192) and autospeaker (d_spk, e.g. 256).
                    # If we can't load it, we must NOT load optimizer state either (it would contain mismatched tensors).
                    resume_skip_optim = True
            if "gender_embed" in payload:
                try:
                    gender_embed.load_state_dict(payload.get("gender_embed", {}), strict=False)
                except Exception as exc:
                    print(f"⚠️ Nie udało się wczytać gender_embed: {exc}")
                    resume_skip_optim = True
            if "emotion_embed" in payload:
                try:
                    emotion_embed.load_state_dict(payload.get("emotion_embed", {}), strict=False)
                    emotion_token_embed.load_state_dict(payload.get("emotion_token_embed", {}), strict=False)
                    emotion_to_style.load_state_dict(payload.get("emotion_to_style", {}), strict=False)
                    gate_val = payload.get("emotion_style_gate", None)
                    if torch.is_tensor(gate_val):
                        emotion_style_gate.data.copy_(gate_val.to(device=device, dtype=emotion_style_gate.dtype).view(()))
                    print("✅ Loaded emotion conditioning modules from resume checkpoint.")
                    if "emotion_token_embed" not in payload:
                        resume_skip_optim = True
                except Exception as exc:
                    print(f"⚠️ Nie udało się wczytać emotion conditioning: {exc}")
                    resume_skip_optim = True
            elif bool(emotion_conditioning_enabled):
                print("ℹ️ emotion conditioning modules are new in this run; optimizer state will be skipped.")
                resume_skip_optim = True
            skipped_prior = _partial_load(prior_mu, dict(payload.get("prior_mu", {})))
            if skipped_prior:
                print(f"ℹ️ wideblock prior partial load: skipped={len(skipped_prior)}")
                resume_skip_optim = True
            if "prior_prosody" in payload:
                try:
                    prior_prosody.load_state_dict(payload.get("prior_prosody", {}), strict=False)
                    print("✅ Loaded prior_prosody auxiliary heads from resume checkpoint.")
                except Exception as exc:
                    print(f"⚠️ Nie udało się wczytać prior_prosody: {exc}")
                    resume_skip_optim = True
            else:
                if float(getattr(args, "w_prior_energy", 0.0)) > 0.0 or float(getattr(args, "w_prior_f0", 0.0)) > 0.0:
                    print("ℹ️ prior_prosody heads are new in this run; optimizer state will be skipped.")
                    resume_skip_optim = True
            if "learned_spk_table" in payload:
                exact = _load_embedding_partial(learned_spk_table, dict(payload.get("learned_spk_table", {})))
                print(
                    "✅ Loaded learned_spk_table from resume checkpoint."
                    if exact
                    else "ℹ️ Partially loaded learned_spk_table from resume checkpoint; new speaker rows stay initialized."
                )
                if not exact:
                    resume_skip_optim = True
            elif str(getattr(args, "speaker_vector_source", "")).lower().strip() == "learned_voice":
                print("ℹ️ learned_spk_table is new in this run; optimizer state will be skipped.")
                resume_skip_optim = True
            if "learned_style_table" in payload:
                exact = _load_embedding_partial(learned_style_table, dict(payload.get("learned_style_table", {})))
                print(
                    "✅ Loaded learned_style_table from resume checkpoint."
                    if exact
                    else "ℹ️ Partially loaded learned_style_table from resume checkpoint; new speaker rows stay initialized."
                )
                if not exact:
                    resume_skip_optim = True
            elif str(getattr(args, "speaker_vector_source", "")).lower().strip() == "learned_voice":
                print("ℹ️ learned_style_table is new in this run; optimizer state will be skipped.")
                resume_skip_optim = True
            if speaker_encoder is not None and "speaker_encoder" in payload:
                if use_style_encoder128:
                    print("ℹ️ skipping old TTS speaker_encoder weights; using frozen style_encoder128 + trainable spk_proj.")
                    resume_skip_optim = True
                else:
                    try:
                        inc = speaker_encoder.load_state_dict(payload.get("speaker_encoder", {}), strict=False)
                        if any(str(k).startswith("pooling_style.") for k in getattr(inc, "missing_keys", [])):
                            _sync_missing_style_pooling(speaker_encoder)
                            print("ℹ️ initialized resumed speaker_encoder pooling_style from speaker pooling.")
                            resume_skip_optim = True
                        print("✅ Loaded trainable speaker_encoder from resume checkpoint.")
                    except Exception as exc:
                        print(f"⚠️ Nie udało się wczytać speaker_encoder: {exc}")
                        resume_skip_optim = True
            elif speaker_encoder is not None:
                if use_style_encoder128:
                    print("ℹ️ speaker_encoder initialized from --style-encoder128-ckpt; optimizer state will be skipped.")
                else:
                    print("ℹ️ speaker_encoder initialized from --spk-style-ckpt; optimizer state will be skipped.")
                resume_skip_optim = True
            if "style_gauss" in payload:
                print("ℹ️ Ignoring style_gauss weights from resume checkpoint in novoiceprior_spkidfix variant.")
            if "voice_prior" in payload:
                print("ℹ️ Ignoring voice_prior weights from resume checkpoint in novoiceprior variant.")
            skipped_mel_flow = _partial_load(mel_flow, dict(payload.get("mel_flow", {})))
            if skipped_mel_flow:
                print(f"ℹ️ wideblock mel_flow partial load: skipped={len(skipped_mel_flow)}")
                resume_skip_optim = True
            try:
                mf_sd = payload.get("mel_flow", {})
                if (
                    isinstance(mf_sd, dict)
                    and any(str(k).startswith("spk_style_ref.") for k in mf_sd.keys())
                    and getattr(mel_flow, "spk_style_ref", None) is not None
                ):
                    if hasattr(mel_flow, "spk_style_ref_ready"):
                        mel_flow.spk_style_ref_ready = True  # type: ignore[attr-defined]
                    has_style_pooling = any(str(k).startswith("spk_style_ref.pooling_style.") for k in mf_sd.keys())
                    if (not has_style_pooling) and _sync_missing_style_pooling(getattr(mel_flow, "spk_style_ref", None)):
                        print("ℹ️ initialized resumed spk_style_ref pooling_style from speaker pooling.")
            except Exception:
                pass
            if "dur_predictor" in payload:
                try:
                    if hasattr(dur_predictor, "predict_logdur"):
                        skipped_dur = _partial_load(dur_predictor, dict(payload.get("dur_predictor", {})))
                        if skipped_dur:
                            print(
                                f"ℹ️ {duration_model_name} duration partial load: "
                                f"skipped={len(skipped_dur)} incompatible tensors from resume checkpoint"
                            )
                        else:
                            print(f"✅ Loaded {duration_model_name} dur_predictor from resume checkpoint.")
                        dur_sd = payload.get("dur_predictor", {})
                        if duration_model_name == "lstm_ar" and isinstance(dur_sd, dict) and (
                            "style_to_h0.weight" not in dur_sd or "style_to_c0.weight" not in dur_sd
                        ):
                            print("ℹ️ duration style h0/c0 initializers are new; optimizer state will be skipped.")
                            resume_skip_optim = True
                        if duration_model_name != "lstm_ar":
                            print("ℹ️ duration model changed from checkpoint; optimizer state will be skipped.")
                        resume_skip_optim = True
                    else:
                        dur_predictor.load_state_dict(payload.get("dur_predictor", {}), strict=False)
                        print("✅ Loaded BiLSTM dur_predictor from resume checkpoint.")
                except Exception as exc:
                    print(f"⚠️ Nie udało się wczytać dur_predictor: {exc}")
                    resume_skip_optim = True
            if _ensure_duration_style_adapter_live(dur_predictor):
                print("🧬 duration style adapter: revived dead zero init (gate=0.01, xavier gain=0.01)")
                resume_skip_optim = True
            if "context_bridge" in payload:
                try:
                    bridge.load_state_dict(payload.get("context_bridge", {}), strict=False)
                except Exception as exc:
                    print(f"⚠️ Nie udało się wczytać context_bridge: {exc}")
            start_epoch = int(payload.get("epoch", 0)) + 1
            print(f"✅ Resumed from: {rpath} (start_epoch={start_epoch})")

    # If speaker-related losses are enabled, we must have a ready frozen dualhead encoder.
    if speaker_loss_enabled or speaker_teacher_enabled:
        ready = bool(getattr(mel_flow, "spk_style_ref_ready", False)) and (getattr(mel_flow, "spk_style_ref", None) is not None)
        if not ready:
            spk_style_ckpt = str(getattr(args, "spk_style_ckpt", "")).strip()
            hint = (
                "Re-run with --spk-style-ckpt /path/to/Speakder_enkoder/_runs/spk_style_dualhead_v2/{best.pt,last.pt} "
                "(and --spk-style-unsafe if needed), OR disable speaker/style losses."
            )
            if spk_style_ckpt:
                hint = f"--spk-style-ckpt was set to '{spk_style_ckpt}' but the encoder is still not ready. " + hint
            raise RuntimeError(
                "Frozen dualhead encoder is required but not ready (no weights loaded from --spk-style-ckpt "
                "and the --resume checkpoint does not contain spk_style_ref.* weights). " + hint
            )

    def _sample_t0_from_prior_stats(mu_btc: torch.Tensor, logs_btc: torch.Tensor, noise_scale: float) -> torch.Tensor:
        return mu_btc + torch.randn_like(mu_btc) * torch.exp(logs_btc) * float(noise_scale)

    flow_helper = FlowMatchHelper()
    vocos = maybe_load_vocos(device)
    speaker_verifier: Optional[nn.Module] = None
    speaker_verifier_name = str(getattr(args, "speaker_verifier", "none")).lower().strip()
    if speaker_verifier_name == "wavlm_tbr":
        if vocos is None:
            raise RuntimeError("--speaker-verifier wavlm_tbr requires Vocos so generated mel can be decoded to waveform.")
        source = str(getattr(args, "speaker_verifier_source", "")).strip()
        if not source:
            source = "/home/rizos/Downloads/SalmonTTS2/test_paraqueet/wegorz_dubbingTTS/external/Orange_Speaker-wavLM-tbr"
        speaker_verifier = FrozenOrangeTBRSpeakerVerifier(
            source=source,
            device=device,
        ).to(device)
        print(
            "🧑‍⚖️ speaker-verifier: wavlm_tbr "
            f"source={source} "
            f"w={float(getattr(args, 'w_speaker_verifier', 0.0)):.4f} "
            f"every={int(getattr(args, 'speaker_verifier_every', 1))} "
            f"max_sec={float(getattr(args, 'speaker_verifier_max_sec', 3.0)):.2f}"
        )
    if float(getattr(args, "w_pitch", 0.0)) > 0.0 or float(getattr(args, "w_energy", 0.0)) > 0.0:
        if vocos is None:
            raise RuntimeError("pitch/energy losses require Vocos so generated mel can be decoded to waveform.")
        print(
            "🎚️ prosody-loss: "
            f"pitch_w={float(getattr(args, 'w_pitch', 0.0)):.4f} "
            f"energy_w={float(getattr(args, 'w_energy', 0.0)):.4f} "
            f"every={int(getattr(args, 'prosody_loss_every', 1))} "
            f"max_sec={float(getattr(args, 'prosody_loss_max_sec', 2.0)):.2f}"
        )
    if float(getattr(args, "w_prior_energy", 0.0)) > 0.0 or float(getattr(args, "w_prior_f0", 0.0)) > 0.0:
        print(
            "🎚️ prior-prosody auxiliary: "
            f"energy_w={float(getattr(args, 'w_prior_energy', 0.0)):.4f} "
            f"f0_w={float(getattr(args, 'w_prior_f0', 0.0)):.4f} "
            f"f0_source=fcpe_cache "
            f"f0_cache_field={str(getattr(args, 'prior_f0_cache_field', 'fcpe_f0_path'))}"
        )
    asr_model = None
    aligner_mod = None
    if bool(getattr(args, "online_ctc", True)):
        asr_ckpt = str(getattr(args, "asr_ckpt", "")).strip()
        if not asr_ckpt:
            raise RuntimeError("--online-ctc requires --asr-ckpt")
        asr_model, aligner_mod = _build_asr_student_v2(asr_ckpt, device)
        if resume_rpath is not None:
            sibling_online_ctc = resume_rpath.parent / "online_ctc_last.pt"
            legacy_sibling_online_asr = resume_rpath.parent / "online_asr_last.pt"
            loaded_online_ctc = False
            sibling_online_ctc_payload_path = None
            if sibling_online_ctc.exists():
                sibling_online_ctc_payload_path = sibling_online_ctc
            elif legacy_sibling_online_asr.exists():
                sibling_online_ctc_payload_path = legacy_sibling_online_asr
            if sibling_online_ctc_payload_path is not None:
                try:
                    online_ctc_payload = torch.load(sibling_online_ctc_payload_path, map_location="cpu")
                    skipped_online_ctc = _partial_load(asr_model, dict(online_ctc_payload.get("state_dict", {})))
                    print(
                        f"✅ Loaded resumed online CTC: {sibling_online_ctc_payload_path} "
                        f"(skipped={len(skipped_online_ctc)})"
                    )
                    loaded_online_ctc = True
                except Exception as exc:
                    print(f"⚠️ Nie udało się wczytać online CTC z {sibling_online_ctc_payload_path}: {exc}")
            if (not loaded_online_ctc) and isinstance(resume_payload, dict) and ("online_ctc" in resume_payload):
                try:
                    skipped_online_ctc = _partial_load(asr_model, dict(resume_payload.get("online_ctc", {})))
                    print(f"✅ Loaded resumed online CTC from TTS checkpoint (skipped={len(skipped_online_ctc)})")
                except Exception as exc:
                    print(f"⚠️ Nie udało się wczytać online CTC z checkpointu TTS: {exc}")
            if (not loaded_online_ctc) and isinstance(resume_payload, dict) and ("online_asr" in resume_payload):
                try:
                    skipped_online_ctc = _partial_load(asr_model, dict(resume_payload.get("online_asr", {})))
                    print(f"✅ Loaded resumed online CTC from legacy TTS key online_asr (skipped={len(skipped_online_ctc)})")
                except Exception as exc:
                    print(f"⚠️ Nie udało się wczytać legacy online_asr z checkpointu TTS: {exc}")
        if bool(getattr(args, "online_ctc_trainable", False)):
            asr_model.train()
            for p in asr_model.parameters():
                p.requires_grad_(True)
            asr_trainable = [p for p in asr_model.parameters() if bool(getattr(p, "requires_grad", False))]
            if asr_trainable:
                opt.add_param_group({"params": asr_trainable, "lr": float(args.lr)})
                print(f"🧪 online CTC trainable: enabled (params={sum(int(p.numel()) for p in asr_trainable)})")
    if (not resume_skip_optim) and isinstance(resume_payload, dict) and ("optim" in resume_payload):
        try:
            opt.load_state_dict(resume_payload["optim"])
        except Exception as exc:
            print(f"⚠️ Nie udało się wczytać optim: {exc}")
            resume_skip_optim = True
    if use_amp and (not resume_skip_optim) and isinstance(resume_payload, dict) and ("scaler" in resume_payload):
        try:
            scaler.load_state_dict(resume_payload["scaler"])
        except Exception as exc:
            print(f"⚠️ Nie udało się wczytać scaler: {exc}")

    # -------- Non-finite diagnostics helpers --------
    nonfinite_dir = out_dir / "nonfinite_debug"
    nonfinite_dir.mkdir(parents=True, exist_ok=True)
    nonfinite_skips = 0

    def _tstats(name: str, t: Optional[torch.Tensor]) -> Dict[str, Any]:
        if t is None or (not torch.is_tensor(t)):
            return {"name": name, "present": False}
        try:
            td = t.detach()
            # work on float32 for stable stats
            if td.dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
                td = td.to(dtype=torch.float32)
            else:
                td = td.to(dtype=torch.float32)
            flat = td.reshape(-1)
            # sample up to 200k values (avoid huge CPU copies)
            n = int(flat.numel())
            if n > 200_000:
                step = max(1, n // 200_000)
                flat_s = flat[::step]
            else:
                flat_s = flat
            finite = torch.isfinite(flat_s)
            finite_frac = float(finite.to(dtype=torch.float32).mean().cpu().item()) if flat_s.numel() > 0 else 0.0
            nan_ct = int(torch.isnan(flat_s).sum().cpu().item())
            inf_ct = int(torch.isinf(flat_s).sum().cpu().item())
            if bool(finite.any().cpu().item()):
                fv = flat_s[finite]
                mn = float(fv.min().cpu().item())
                mx = float(fv.max().cpu().item())
                mean = float(fv.mean().cpu().item())
                std = float(fv.std(unbiased=False).cpu().item()) if int(fv.numel()) > 1 else 0.0
            else:
                mn = mx = mean = std = float("nan")
            return {
                "name": name,
                "present": True,
                "shape": list(t.shape),
                "dtype": str(t.dtype),
                "device": str(t.device),
                "finite_frac_sampled": finite_frac,
                "nan_count_sampled": nan_ct,
                "inf_count_sampled": inf_ct,
                "min_finite_sampled": mn,
                "max_finite_sampled": mx,
                "mean_finite_sampled": mean,
                "std_finite_sampled": std,
            }
        except Exception as exc:
            return {"name": name, "present": True, "error": f"{type(exc).__name__}: {exc}"}

    def _param_nonfinite_report(mod: nn.Module, name: str) -> Dict[str, Any]:
        total = 0
        bad = 0
        try:
            for p in mod.parameters():
                total += int(p.numel())
                if not bool(torch.isfinite(p.detach()).all().cpu().item()):
                    bad += int(p.numel())
            return {"module": name, "params_total": total, "params_with_any_nonfinite": bad}
        except Exception as exc:
            return {"module": name, "error": f"{type(exc).__name__}: {exc}"}

    def _dump_and_maybe_abort(
        *,
        reason: str,
        ep: int,
        batch_i: int,
        speaker_ids: Optional[torch.Tensor] = None,
        audio_paths: Any = None,
        extra_scalars: Optional[Dict[str, Any]] = None,
        tensors: Optional[Dict[str, Optional[torch.Tensor]]] = None,
    ) -> None:
        nonlocal nonfinite_skips
        nonfinite_skips += 1
        sid0 = None
        try:
            sid0 = int(speaker_ids[0].detach().cpu().item()) if torch.is_tensor(speaker_ids) and speaker_ids.numel() > 0 else None
        except Exception:
            sid0 = None
        ap0 = None
        try:
            ap0 = (audio_paths[0] if isinstance(audio_paths, list) and len(audio_paths) > 0 else None)
        except Exception:
            ap0 = None

        rep: Dict[str, Any] = {
            "reason": str(reason),
            "epoch": int(ep),
            "batch_i": int(batch_i),
            "speaker_id0": sid0,
            "audio0": ap0,
            "nonfinite_skips": int(nonfinite_skips),
            "stop_on_nonfinite": bool(getattr(args, "stop_on_nonfinite", True)),
            "nonfinite_max_skips": int(getattr(args, "nonfinite_max_skips", 0)),
            "extra": dict(extra_scalars or {}),
            "tensors": [],
            "modules": [],
        }
        try:
            if use_amp:
                rep["amp_enabled"] = True
                try:
                    rep["amp_scale"] = float(scaler.get_scale())  # type: ignore[attr-defined]
                except Exception:
                    rep["amp_scale"] = None
            else:
                rep["amp_enabled"] = False
        except Exception:
            pass
        try:
            if tensors:
                rep["tensors"] = [_tstats(k, v) for k, v in tensors.items()]
        except Exception:
            pass
        try:
            rep["modules"] = [
                _param_nonfinite_report(mel_flow, "mel_flow"),
                _param_nonfinite_report(prior_mu, "prior_mu"),
                _param_nonfinite_report(dur_predictor, "dur_predictor"),
                _param_nonfinite_report(spk_adapter, "spk_adapter"),
            ]
        except Exception:
            pass

        dump_base = f"nonfinite_ep{int(ep):04d}_batch{int(batch_i):05d}"
        dump_json = nonfinite_dir / f"{dump_base}.json"
        try:
            dump_json.write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass

        if bool(getattr(args, "nonfinite_dump_tensors", True)) and tensors:
            # save a small subset (first element) to keep files manageable
            for k, v in tensors.items():
                if v is None or (not torch.is_tensor(v)):
                    continue
                try:
                    vv = v.detach()
                    if vv.dim() >= 1:
                        vv = vv[:1].contiguous()
                    torch.save(vv.cpu(), str(nonfinite_dir / f"{dump_base}__{k}.pt"))
                except Exception:
                    continue

        print(f"🧯 Non-finite detected: {reason} | dump={dump_json}")

        max_skips = int(getattr(args, "nonfinite_max_skips", 0))
        do_stop = bool(getattr(args, "stop_on_nonfinite", True)) and (int(nonfinite_skips) > int(max_skips))
        if do_stop:
            raise RuntimeError(f"Non-finite encountered ({reason}). Dump saved to: {dump_json}")

    def _vocos_decode_or_none(mel_1ct: torch.Tensor, *, tag: str) -> Optional[torch.Tensor]:
        if vocos is None:
            return None
        try:
            if not torch.is_tensor(mel_1ct):
                print(f"⚠️ vocos decode skipped ({tag}): mel is not a tensor")
                return None
            if mel_1ct.ndim != 3:
                print(f"⚠️ vocos decode skipped ({tag}): bad mel shape {tuple(mel_1ct.shape)}")
                return None
            if mel_1ct.numel() == 0 or int(mel_1ct.size(-1)) == 0:
                print(f"⚠️ vocos decode skipped ({tag}): empty mel shape {tuple(mel_1ct.shape)}")
                return None
            mel_1ct = mel_1ct.to(dtype=torch.float32)
            if not torch.isfinite(mel_1ct).all():
                print(f"⚠️ vocos decode skipped ({tag}): non-finite mel")
                return None
            with torch.no_grad():
                return vocos.decode(mel_1ct)
        except Exception as exc:
            print(f"⚠️ vocos decode failed ({tag}): {exc}")
            return None

    # -------- Speaker conditioning --------
    # We use fixed per-speaker centroids loaded from dataset `speaker_embeds.pt`.
    # No EMA updates, no online spkbank, no internal spk_enc training.
    spk_dim = int(getattr(args, "spk_dim", 256))

    try:
        import torchaudio  # noqa
        _HAS_TORCHAUDIO = True
    except Exception:
        _HAS_TORCHAUDIO = False

    def _load_ref_mel_pt(path: str) -> Tuple[torch.Tensor, torch.Tensor]:
        return _demo_load_ref_mel_pt(path, ensure_mel_bct=_ensure_mel_bct, device=device)

    def _ref_wav_to_mel(path: str, *, max_sec: float) -> Tuple[torch.Tensor, torch.Tensor]:
        return wav_to_vocos_mel(
            path,
            vocos=vocos,
            ensure_mel_bct=_ensure_mel_bct,
            n_mels=int(N_MELS),
            device=device,
            max_sec=float(max_sec),
        )

    # -------- Inference-only mode (no training loop) --------
    if bool(getattr(args, "infer_only", False)):
        text_in = str(getattr(args, "infer_text", "")).strip()
        if not text_in:
            raise SystemExit("--infer-only requires --infer-text")

        set_seed(int(getattr(args, "infer_seed", 1234)))

        out_inf = Path(str(getattr(args, "infer_out_dir", "/tmp/wegorz_learnedvoice_infer"))).expanduser()
        out_inf.mkdir(parents=True, exist_ok=True)
        infer_tag_prefix = str(getattr(args, "infer_tag", "")).strip()

        # Load speaker vectors (spk_dim) from .pt overrides or dataset centroids.
        spk_d = int(getattr(args, "spk_dim", 256))

        def _load_emb_spkdim_pt(path: str) -> torch.Tensor:
            pp = Path(str(path)).expanduser()
            if not pp.is_absolute():
                try:
                    base = Path(str(getattr(args, "dataset_json", ""))).expanduser().resolve().parent
                    cand = (base / pp).resolve()
                    if cand.exists():
                        pp = cand
                except Exception:
                    pass
            data = torch.load(str(pp), map_location="cpu")
            if isinstance(data, dict):
                emb = data.get("emb", data.get("speaker_emb", data.get("spk_emb", None)))
            else:
                emb = data
            if emb is None:
                raise RuntimeError(f"speaker emb .pt missing emb: {path}")
            t = torch.as_tensor(emb, dtype=torch.float32).view(-1)
            if int(t.numel()) != int(spk_d):
                raise RuntimeError(f"speaker emb dim mismatch: got {int(t.numel())}, expected {spk_d}: {path}")
            if bool(getattr(args, "spk_emb_l2norm", True)):
                t = t / t.norm().clamp_min(1e-12)
            return t

        def _load_wav_mono_24k(path: str, *, max_sec: float, start_sec: float = 0.0) -> torch.Tensor:
            p = Path(str(path)).expanduser()
            if not p.exists():
                raise RuntimeError(f"Reference WAV not found: {p}")
            try:
                import torchaudio  # type: ignore
            except Exception:
                torchaudio = None
            try:
                import soundfile as sf  # type: ignore
            except Exception:
                sf = None
            wav = None
            sr = None
            if torchaudio is not None:
                try:
                    wav, sr = torchaudio.load(str(p))
                    if wav.dim() == 2 and int(wav.size(0)) > 1:
                        wav = wav[:1]
                    wav = wav.to(torch.float32)
                except Exception as exc:
                    msg = str(exc).splitlines()[0] if str(exc).strip() else repr(exc)
                    print(f"⚠️ torchaudio.load failed for ref wav, falling back to soundfile: {msg}")
                    wav = None
                    sr = None
            if wav is None and sf is not None:
                import numpy as np

                arr, sr = sf.read(str(p), dtype="float32", always_2d=False)
                if arr.ndim == 2:
                    arr = arr[:, 0]
                wav = torch.from_numpy(np.asarray(arr)).view(1, -1).to(torch.float32)
            if wav is None or sr is None:
                raise RuntimeError("Cannot load WAV: install torchaudio or soundfile.")
            if int(sr) != 24000:
                if torchaudio is None:
                    raise RuntimeError(f"Need torchaudio to resample ({sr} -> 24000) for ref wav: {p}")
                wav = torchaudio.functional.resample(wav, int(sr), 24000)
            start_samp = max(0, int(float(start_sec) * 24000.0))
            if start_samp > 0 and start_samp < int(wav.size(-1)):
                wav = wav[..., start_samp:]
            if float(max_sec) > 0.0:
                max_samp = int(float(max_sec) * 24000.0)
                if wav.size(-1) > max_samp:
                    wav = wav[..., :max_samp]
            return wav.contiguous()

        # Build mapping speaker_id -> speaker_embeds.pt from dataset json (used as "known" centroids).
        spk_pt_by_id: Dict[int, str] = {}
        try:
            ds_items = json.loads(Path(str(getattr(args, "dataset_json", ""))).expanduser().read_text(encoding="utf-8"))
            if isinstance(ds_items, list):
                for it in ds_items:
                    if not isinstance(it, dict):
                        continue
                    sid = it.get("speaker_id", it.get("speaker", None))
                    p = it.get("speaker_embeds", None)
                    if sid is None or not p:
                        continue
                    sid_i = int(sid)
                    if sid_i not in spk_pt_by_id:
                        spk_pt_by_id[sid_i] = str(p)
        except Exception:
            spk_pt_by_id = {}

        def _centroid_from_dataset(sid: int) -> torch.Tensor:
            sid = int(sid)
            p = spk_pt_by_id.get(sid, "")
            if not p:
                # Common case: datasets have non-zero, non-contiguous speaker ids; infer defaults to 0.
                if spk_pt_by_id:
                    fallback_sid = int(sorted(spk_pt_by_id.keys())[0])
                    print(
                        f"⚠️ infer-only: speaker_id={sid} not found in dataset centroids; "
                        f"falling back to speaker_id={fallback_sid}. "
                        f"(set --infer-speaker-a / --infer-speaker-b explicitly)"
                    )
                    p = spk_pt_by_id[fallback_sid]
                else:
                    raise RuntimeError(
                        f"speaker_id={sid} not found in dataset speaker_embeds map "
                        f"(dataset_json={getattr(args,'dataset_json','')})"
                    )
            return _load_emb_spkdim_pt(p)

        _spkenc_model = None

        def _load_pretrained_spkenc() -> nn.Module:
            nonlocal _spkenc_model
            if _spkenc_model is not None:
                return _spkenc_model
            ckpt = str(getattr(args, "spkenc_ckpt", "")).strip()
            if not ckpt:
                raise RuntimeError("--spkenc-ckpt is required for --infer-ref-wav-* (zero-shot centroid).")
            ckpt_p = Path(ckpt).expanduser()
            try:
                payload = torch.load(str(ckpt_p), map_location="cpu")
            except Exception:
                payload = torch.load(str(ckpt_p), map_location="cpu", weights_only=False)  # type: ignore[call-arg]
            cfg = payload.get("args", {}) if isinstance(payload, dict) else {}

            spkenc_dir = (_THIS_DIR / "Speakder_enkoder")
            if spkenc_dir.is_dir() and str(spkenc_dir) not in sys.path:
                sys.path.append(str(spkenc_dir))
            import spk_enc_model as spkmod  # type: ignore

            vdev = str(getattr(args, "spkenc_device", "auto")).lower().strip()
            if vdev == "auto":
                vdev_t = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            else:
                vdev_t = torch.device(vdev)
            enc = spkmod.MelSpeakerEncoder(
                n_mels=int(cfg.get("n_mels", N_MELS)),
                d_spk=int(cfg.get("spk_dim", spk_d)),
                hidden_dim=int(cfg.get("hidden_dim", 256)),
                num_layers=int(cfg.get("layers", 4)),
                attn_head_dim=int(cfg.get("attn_head_dim", 128)),
            ).to(vdev_t).eval()
            enc.load_state_dict(payload["model"], strict=True)
            _spkenc_model = enc
            return _spkenc_model

        def _spk_from_ref_wav(path: str) -> torch.Tensor:
            if vocos is None:
                raise RuntimeError("Vocos is required for reference-WAV speaker extraction.")
            fe = getattr(vocos, "feature_extractor", None)
            if fe is None:
                raise RuntimeError("Loaded Vocos has no feature_extractor; cannot compute mel from audio.")
            use_dual = bool(getattr(mel_flow, "spk_style_ref_ready", False)) and (getattr(mel_flow, "spk_style_ref", None) is not None)
            if use_dual:
                enc = getattr(mel_flow, "spk_style_ref")
                enc_dev = next(enc.parameters()).device  # type: ignore[union-attr]
            else:
                enc = _load_pretrained_spkenc()
                enc_dev = next(enc.parameters()).device
            wav_1t = _load_wav_mono_24k(
                path,
                max_sec=float(getattr(args, "infer_ref_max_sec", 10.0)),
                start_sec=float(getattr(args, "infer_ref_start_sec", 0.0)),
            )
            with torch.no_grad():
                mel = fe(wav_1t.to(device=device))  # [1, n_mels, T] or [1, T, n_mels]
            if mel.dim() == 3 and int(mel.size(1)) != int(N_MELS) and int(mel.size(2)) == int(N_MELS):
                mel = mel.transpose(1, 2).contiguous()
            if mel.dim() != 3:
                raise RuntimeError(f"Bad mel from vocos.feature_extractor: {tuple(mel.shape)}")
            mel_bct = _ensure_mel_bct(mel).to(device=enc_dev, dtype=torch.float32)
            mask_bt = torch.ones((int(mel_bct.size(0)), int(mel_bct.size(-1))), device=enc_dev, dtype=torch.bool)
            with torch.no_grad():
                if use_dual:
                    z_spk, _z_style = enc(mel_bct, mask_bt=mask_bt)  # type: ignore[misc,call-arg]
                    z = z_spk
                else:
                    z = enc(mel_bct, mask_bt=mask_bt)  # type: ignore[call-arg]
            z = z.detach().to(torch.float32).view(-1)
            if int(z.numel()) != int(spk_d):
                raise RuntimeError(f"ref speaker emb dim mismatch: got {int(z.numel())}, expected {spk_d}")
            if bool(getattr(args, "spk_emb_l2norm", True)):
                z = z / z.norm().clamp_min(1e-12)
            return z

        def _load_ref_mel_pt(path: str) -> Tuple[torch.Tensor, torch.Tensor]:
            p = Path(path).expanduser()
            obj = torch.load(str(p), map_location="cpu")
            if isinstance(obj, dict):
                mel = obj.get("mel", obj.get("mel_bct", obj.get("x", obj.get("features"))))
                t_val = obj.get("T_mel", obj.get("length", obj.get("T", None)))
            else:
                mel = obj
                t_val = None
            if not torch.is_tensor(mel):
                raise RuntimeError(f"Cannot find tensor mel in {p}")
            mel_bct = _ensure_mel_bct(mel).to(device=device, dtype=torch.float32)
            if t_val is None:
                t_len = torch.tensor([int(mel_bct.size(-1))], device=device, dtype=torch.long)
            else:
                if torch.is_tensor(t_val):
                    t_i = int(t_val.view(-1)[0].item())
                else:
                    t_i = int(t_val)
                t_len = torch.tensor([max(1, min(t_i, int(mel_bct.size(-1))))], device=device, dtype=torch.long)
            return mel_bct[:, :, : int(t_len[0].item())].contiguous(), t_len

        def _spk_from_ref_mel_pt(path: str) -> torch.Tensor:
            mel_bct, t_len = _load_ref_mel_pt(path)
            if speaker_encoder is not None:
                enc_dev = next(speaker_encoder.parameters()).device
                mask_bt = _make_tmask_from_Tlen(t_len.to(enc_dev), int(mel_bct.size(-1))).squeeze(1).to(
                    device=enc_dev,
                    dtype=torch.bool,
                )
                ctx = nullcontext() if bool(getattr(args, "speaker_encoder_trainable", True)) else torch.no_grad()
                with ctx:
                    z_spk, _z_style = speaker_encoder(  # type: ignore[misc,call-arg]
                        mel_bct.to(device=enc_dev, dtype=torch.float32),
                        mask_bt=mask_bt,
                    )
                z = z_spk.detach().to(torch.float32).view(-1)
            else:
                if not bool(getattr(mel_flow, "spk_style_ref_ready", False)):
                    raise RuntimeError("--demo-ref-mel requires speaker_encoder or loaded frozen dualhead encoder.")
                enc = getattr(mel_flow, "spk_style_ref")
                enc_dev = next(enc.parameters()).device
                mask_bt = _make_tmask_from_Tlen(t_len.to(enc_dev), int(mel_bct.size(-1))).squeeze(1).to(
                    device=enc_dev,
                    dtype=torch.bool,
                )
                with torch.no_grad():
                    z_spk, _z_style = enc(mel_bct.to(device=enc_dev, dtype=torch.float32), mask_bt=mask_bt)  # type: ignore[misc,call-arg]
                z = z_spk.detach().to(torch.float32).view(-1)
            if bool(getattr(args, "spk_emb_l2norm", True)):
                z = z / z.norm().clamp_min(1e-12)
            return z

        def _report_ref_vs_known(z: torch.Tensor, *, tag: str) -> None:
            k = int(getattr(args, "infer_ref_report_topk", 0))
            if k <= 0 or not spk_pt_by_id:
                return
            # Build centroid matrix once per call (small: typically 36 speakers).
            sids = sorted(spk_pt_by_id.keys())
            C = torch.stack([_load_emb_spkdim_pt(spk_pt_by_id[sid]) for sid in sids], dim=0)  # [S,D]
            C = C.to(dtype=torch.float32)
            z0 = z.detach().cpu().to(torch.float32).view(1, -1)
            Cn = C / C.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            sims = (z0 @ Cn.t()).view(-1)
            kk = int(min(k, int(sims.numel())))
            vals, idx = torch.topk(sims, k=kk)
            pairs = [(int(sids[int(i.item())]), float(s.item())) for i, s in zip(idx, vals)]
            msg = f"🔎 [{tag}] ref->known top{kk}: " + ", ".join([f"{sid}:{sim:.3f}" for sid, sim in pairs])
            if len(pairs) >= 2:
                msg += f" | margin={pairs[0][1] - pairs[1][1]:.4f}"
            print(msg)

        def _save_wav(path: Path, wav_1t: torch.Tensor, sr: int = 24000) -> None:
            wav = wav_1t.detach().cpu()
            if wav.ndim == 1:
                wav = wav.unsqueeze(0)
            try:
                import torchaudio  # type: ignore

                torchaudio.save(str(path), wav, sr)
                return
            except Exception:
                pass
            try:
                import soundfile as sf  # type: ignore

                sf.write(str(path), wav.squeeze(0).numpy(), sr)
                return
            except Exception as exc:
                last_exc = exc
            try:
                import wave

                wav_np = wav.squeeze(0).numpy()
                wav_i16 = np.clip(wav_np, -1.0, 1.0)
                wav_i16 = (wav_i16 * 32767.0).astype(np.int16, copy=False)
                with wave.open(str(path), "wb") as f:
                    f.setnchannels(1)
                    f.setsampwidth(2)
                    f.setframerate(int(sr))
                    f.writeframes(wav_i16.tobytes())
                return
            except Exception as exc:
                raise RuntimeError(f"Cannot save wav (torchaudio/soundfile/wave failed): {exc}; previous={last_exc}")

        import re

        def _split_sentences(par: str) -> List[str]:
            """
            Naive sentence splitter for Polish prose.
            Splits mainly on end punctuation (. ! ? …). Keeps punctuation with the sentence.
            """
            s = str(par).strip()
            if not s:
                return []
            # Normalize whitespace
            s = re.sub(r"\s+", " ", s).strip()
            out: List[str] = []
            # Match minimal chunks ending with sentence-final punctuation (optionally followed by quotes/brackets)
            pat = re.compile(r".*?[.!?…]+(?:[\"”»’\)\]]*)?(?:\s+|$)")
            i = 0
            for m in pat.finditer(s):
                seg = m.group(0).strip()
                if seg:
                    out.append(seg)
                i = m.end()
            # leftover (no terminal punctuation)
            tail = s[i:].strip()
            if tail:
                out.append(tail)
            return out

        def _word_count(s: str) -> int:
            return len(re.findall(r"[A-Za-zĄĆĘŁŃÓŚŹŻąćęłńóśźż0-9]+", str(s)))

        def _merge_short_chunks(parts: List[str], min_words: int = 5) -> List[str]:
            merged: List[str] = []
            i = 0
            while i < len(parts):
                cur = str(parts[i]).strip()
                if not cur:
                    i += 1
                    continue
                if _word_count(cur) < int(min_words):
                    if i + 1 < len(parts):
                        nxt = str(parts[i + 1]).strip()
                        cur = f"{cur} {nxt}".strip() if nxt else cur
                        i += 2
                    elif merged:
                        merged[-1] = f"{merged[-1]} {cur}".strip()
                        i += 1
                        continue
                else:
                    i += 1
                merged.append(cur)
            return merged

        # Auto-chunk: split paragraphs by newlines, then split into sentences by punctuation.
        chunks: List[str] = []
        for par in text_in.splitlines():
            if not str(par).strip():
                continue
            chunks.extend(_split_sentences(par))
        chunks = _merge_short_chunks(chunks, min_words=5)

        # Bound each chunk. Internal mid-sentence pieces end with <sp>, not
        # <EOS>, so inference does not learn an artificial sentence ending at
        # technical split points.
        texts = [
            _ensure_boundary_tokens(
                t,
                continuation_out=(i < len(chunks) - 1 and not _looks_sentence_final(t)),
            )
            for i, t in enumerate(chunks)
            if str(t).strip()
        ]
        if not texts:
            raise SystemExit("--infer-text is empty after stripping")

        sid_a = int(getattr(args, "infer_speaker_a", 0))
        sid_b = int(getattr(args, "infer_speaker_b", -1))
        do_mix = bool(getattr(args, "infer_mix", True)) and (sid_b >= 0)
        alpha = float(getattr(args, "infer_mix_alpha", 0.5))
        alpha = float(max(0.0, min(1.0, alpha)))

        emb_a_pt = str(getattr(args, "infer_speaker_emb_pt", "")).strip()
        if emb_a_pt:
            spkA_vec = _load_emb_spkdim_pt(emb_a_pt)
        else:
            spkA_vec = _centroid_from_dataset(sid_a)
        save_spk_pt = str(getattr(args, "infer_save_speaker_emb_pt", "")).strip()
        if save_spk_pt:
            save_spk_path = Path(save_spk_pt).expanduser()
            save_spk_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(spkA_vec.detach().cpu().to(torch.float32), save_spk_path)
            print(f"💾 saved voice A speaker vector: {save_spk_path}")
        spkA_256 = spkA_vec.to(device=device).unsqueeze(0)
        spkA = spk_adapter(spkA_256).to(dtype=torch.float32)  # [1, D]
        voices = [("A", sid_a, spkA)]
        voice_vec_by_tag: Dict[str, torch.Tensor] = {"A": spkA_256}

        if sid_b >= 0:
            emb_b_pt = str(getattr(args, "infer_speaker_emb_pt_b", "")).strip()
            if emb_b_pt:
                spkB_vec = _load_emb_spkdim_pt(emb_b_pt)
            else:
                spkB_vec = _centroid_from_dataset(sid_b)
            spkB_256 = spkB_vec.to(device=device).unsqueeze(0)
            spkB = spk_adapter(spkB_256).to(dtype=torch.float32)  # [1, D]
            voices.append(("B", sid_b, spkB))
            voice_vec_by_tag["B"] = spkB_256
            if do_mix:
                spkM = (1.0 - alpha) * spkA + alpha * spkB
                voices.append((f"MIX_a{alpha:.2f}", sid_a, spkM))
                voice_vec_by_tag[f"MIX_a{alpha:.2f}"] = (1.0 - alpha) * spkA_256 + alpha * spkB_256

        if vocos is None:
            raise RuntimeError("Vocos is required for --infer-only (dataset has no WAV paths).")
        vocos.eval()

        tok = PLTokenizer(str(getattr(args, "vocab", "")).strip())

        prior_mu.eval()
        dur_predictor.eval()
        mel_flow.eval()
        spk_adapter.eval()
        gender_embed.eval()
        if speaker_encoder is not None:
            speaker_encoder.eval()
        if train_base:
            model.eval()
            bridge.eval()
            spk_embed.eval()

        join_sil_ms = float(getattr(args, "infer_join_sil_ms", 0.0))
        join_sil_ms = float(max(0.0, min(5000.0, join_sil_ms)))
        join_sil = None
        if join_sil_ms > 0.0:
            join_sil = torch.zeros((1, int((join_sil_ms / 1000.0) * 24000)), dtype=torch.float32)

        decode_joined = False
        overlap_ms = 0.0
        overlap_frames = 0

        mel_steps = int(getattr(args, "infer_mel_steps", int(CONFIG.get("mel_flow_steps_demo", 8))))
        dur_steps = int(getattr(args, "infer_dur_steps", int(CONFIG.get("dur_flow_steps", 10))))
        dur_noise = float(getattr(args, "infer_dur_noise_scale", float(CONFIG.get("dur_flow_noise_scale", 1.0))))
        dur_source = str(getattr(args, "infer_dur_source", "flow")).lower().strip()
        speed = float(getattr(args, "infer_speed", 1.0))
        speed = float(max(0.05, min(20.0, speed)))

        def _tok_name_for_debug(tid: int) -> str:
            try:
                if int(tid) in getattr(tok, "id2token", {}):
                    return str(tok.id2token[int(tid)])
            except Exception:
                pass
            try:
                return str(ID2SYMBOL.get(int(tid), f"<id:{int(tid)}>"))
            except Exception:
                return f"<id:{int(tid)}>"

        for tag, sid, spk_hidden in voices:
            tag_out = f"{infer_tag_prefix}__{tag}" if infer_tag_prefix else str(tag)
            wav_joined: List[torch.Tensor] = []
            dur_debug_lines: List[str] = [
                f"tag={tag_out}",
                f"dur_source={dur_source} dur_steps={dur_steps} dur_noise_scale={dur_noise} speed={speed}",
                f"frame_sec={_SECS_PER_FRAME:.9f}",
                "",
            ]
            speaker_ids1 = torch.tensor([int(sid)], device=device, dtype=torch.long)
            gender_ids1 = _gender_ids_from_speaker_ids(speaker_ids1, device=device)
            if bool(getattr(args, "disable_gender_token", True)):
                gender_ids1 = torch.zeros_like(gender_ids1)
            # Prefix-continuity state (per voice): previous chunk's generated mel (for tail clamping).
            prev_mel_out_bct: Optional[torch.Tensor] = None  # on device, float32

            carry_duration_state = bool(getattr(args, "duration_state_carry", True))
            _dur_lstm_hc: "tuple | None" = None  # stan LSTM między chunkami (stateful rhythm)
            for chunk_no, txt in enumerate(texts, 1):
                ids = tok.encode(str(txt))
                if not ids:
                    continue
                tok_pad1 = torch.tensor([ids], device=device, dtype=torch.long)

                x_tok1, ids_full1, special_len1 = encode_text_features(
                    model=model,
                    spk_embed=spk_embed,
                    gender_embed=gender_embed,
                    emotion_token_embed=emotion_token_embed,
                    tok_pad=tok_pad1,
                    speaker_ids=speaker_ids1,
                    gender_ids=gender_ids1,
                    emotion_ids=torch.zeros_like(speaker_ids1),
                    device=device,
                    spk_vec_override=spk_hidden.to(device=device, dtype=torch.float32),
                    require_spk_override=bool(getattr(args, "require_spk_override", True)),
                    use_emotion_token=bool(emotion_conditioning_enabled),
                )

                spk_base1 = spk_hidden.to(device=device, dtype=x_tok1.dtype)
                spk_dur1 = _zero_spk_cond_like(spk_base1)

                # durations (inference): either flow sampler or direct dur-prior
                if dur_source in ("prior_mu", "prior_sample"):
                    dur_pred1, _, _ = _predict_dur_prior_direct(
                        model,
                        x_tok1,
                        tok_pad1,
                        special_len1,
                        source=dur_source,
                        noise_scale=float(dur_noise),
                        dur_prior_logs_min=float(getattr(args, "dur_prior_logs_min", -5.0)),
                        dur_prior_logs_max=float(getattr(args, "dur_prior_logs_max", 2.0)),
                        dur_prior_sigma_min=float(getattr(args, "dur_prior_sigma_min", 0.1)),
                        initial_hc=_dur_lstm_hc if carry_duration_state else None,
                    )
                    _dur_lstm_hc = getattr(_predict_dur_prior_direct, "_last_hc", None) if carry_duration_state else None
                else:
                    dur_pred1, _, _ = _predict_dur(
                        model,
                        x_tok1,
                        tok_pad1,
                        special_len1,
                        spk_embed,
                        speaker_ids1,
                        None,
                        spk_vec_override=spk_dur1,
                        require_spk_override=bool(getattr(args, "require_spk_override", True)),
                        style_vec=None,
                        dur_x0_mode=str(getattr(args, "dur_x0", "prior")),
                        dur_x0_noise_scale=float(getattr(args, "dur_x0_noise_scale", 1.0)),
                        dur_prior_logs_min=float(getattr(args, "dur_prior_logs_min", -5.0)),
                        dur_prior_logs_max=float(getattr(args, "dur_prior_logs_max", 2.0)),
                        dur_prior_sigma_min=float(getattr(args, "dur_prior_sigma_min", 0.1)),
                        dur_sigma0_demo=0.0,
                        steps_override=dur_steps,
                        noise_scale_override=dur_noise,
                        dur_flow_clip_sigma=float(getattr(args, "dur_flow_clip_sigma", 0.0)),
                        dur_flow_clip_abs_min=(
                            None
                            if (not math.isfinite(float(getattr(args, "dur_flow_clip_abs_min", float("nan")))))
                            else float(getattr(args, "dur_flow_clip_abs_min", float("nan")))
                        ),
                        dur_flow_clip_abs_max=(
                            None
                            if (not math.isfinite(float(getattr(args, "dur_flow_clip_abs_max", float("nan")))))
                            else float(getattr(args, "dur_flow_clip_abs_max", float("nan")))
                        ),
                        dur_flow_fix_total=bool(getattr(args, "dur_flow_fix_total", False)),
                        dur_flow_fix_total_mode=str(getattr(args, "dur_flow_fix_total_mode", "prior_mu")),
                        initial_hc=_dur_lstm_hc if carry_duration_state else None,
                    )
                    _dur_lstm_hc = getattr(_predict_dur, "_last_hc", None) if carry_duration_state else None

                # speed ratio: >1 => faster => shorter durations
                if float(speed) != 1.0:
                    dur_pred1 = dur_pred1 / float(speed)
                    dur_pred1 = dur_pred1.clamp_min(0.0)
                    # keep pause tokens minimum 1 frame
                    ids_full_tmp2 = F.pad(tok_pad1, (int(special_len1), 0), value=PAD_ID)
                    pause_mask = _pause_mask_from_ids(ids_full_tmp2)
                    dur_pred1 = torch.where(pause_mask, dur_pred1.clamp_min(1.0), dur_pred1)

                ids_full1 = F.pad(tok_pad1, (int(special_len1), 0), value=PAD_ID)
                dur_allowed1 = _build_dur_allowed_mask(ids_full1, special_len1)
                dur_for_prior1 = torch.where(dur_allowed1, dur_pred1, torch.zeros_like(dur_pred1))
                pause_mask1 = _pause_mask_from_ids(ids_full1)
                text_min1 = float(getattr(args, "dur_pred_text_min_frames", 0.0))
                if text_min1 > 0:
                    text_dur_mask1 = dur_allowed1 & (~pause_mask1)
                    dur_for_prior1 = torch.where(
                        text_dur_mask1,
                        dur_for_prior1.clamp_min(float(text_min1)),
                        dur_for_prior1,
                    )
                sp_max1 = float(getattr(args, "dur_pred_sp_max_frames", 0.0))
                if sp_max1 > 0:
                    dur_for_prior1 = torch.where(
                        pause_mask1,
                        dur_for_prior1.clamp_max(float(sp_max1)),
                        dur_for_prior1,
                    )
                sp_mask_tok1 = _pause_mask_from_ids(ids_full1)
                ids_dbg = [int(x) for x in ids_full1[0].detach().cpu().tolist()]
                dur_dbg = [float(x) for x in dur_for_prior1[0].detach().cpu().tolist()]
                total_frames_dbg = float(sum(dur_dbg))
                dur_debug_lines.append(f"[chunk {chunk_no}] text={str(txt)}")
                dur_debug_lines.append(
                    f"total_frames={total_frames_dbg:.3f} total_sec={total_frames_dbg * _SECS_PER_FRAME:.3f} tokens={len(ids_dbg)}"
                )
                for pos, (tid, dval) in enumerate(zip(ids_dbg, dur_dbg)):
                    dur_debug_lines.append(
                        f"{pos:04d}\tid={tid}\ttok={_tok_name_for_debug(tid)}\tframes={dval:.3f}\tsec={dval * _SECS_PER_FRAME:.4f}"
                    )
                dur_debug_lines.append("")

                prior_cond1 = None

                prior_noise_scale = float(getattr(args, "prior_noise_scale", 1.0))
                t0_btc1, _mu_btc1, _logs_btc1, _ = prior_mu(
                    x_tok1,
                    dur_for_prior1,
                    cond=prior_cond1,
                    T_hint=None,
                    noise_scale=float(prior_noise_scale),
                    sp_mask_tok=sp_mask_tok1,
                    timbre_vec=spk_base1,
                )
                x01 = t0_btc1.transpose(1, 2).contiguous()

                # ---- Prefix continuity (decoder): clamp a fixed prefix window to prev tail, then trim on output ----
                prefix_k = 0
                prefix_tail = None
                if prev_mel_out_bct is not None:
                    _pf = int(_prefix_frames_from_ms(_short_continuity_ms))
                    prefix_k = int(min(int(_pf), int(x01.size(-1)), int(prev_mel_out_bct.size(-1))))
                    if prefix_k > 0:
                        prefix_tail = prev_mel_out_bct[:, :, -prefix_k:].contiguous()

                if prefix_k > 0 and torch.is_tensor(prefix_tail):
                    if bool(getattr(args, "mel_twopass", False)):
                        mel_out1 = sample_mel_flow_with_prefix_twopass(
                            mel_flow,
                            x01,
                            x_tok1 if bool(CONFIG["text_cross_attn"]) else None,
                            speaker_ids1,
                            steps_first=int(getattr(args, "mel_twopass_steps_first", mel_steps)),
                            steps_second=int(getattr(args, "mel_twopass_steps_second", 0)),
                            t_noise=float(getattr(args, "mel_twopass_t_noise", 0.12)),
                            spk_vec_override=spk_base1,
                            prefix_tail_bct=prefix_tail,
                            prefix_k=int(prefix_k),
                        )
                    else:
                        mel_out1 = sample_mel_flow_with_prefix(
                            mel_flow,
                            x01,
                            x_tok1 if bool(CONFIG["text_cross_attn"]) else None,
                            speaker_ids1,
                            steps=int(mel_steps),
                            spk_vec_override=spk_base1,
                            prefix_tail_bct=prefix_tail,
                            prefix_k=int(prefix_k),
                        )
                else:
                    if bool(getattr(args, "mel_twopass", False)):
                        mel_out1 = sample_mel_flow_twopass(
                            mel_flow,
                            x01,
                            x_tok1 if bool(CONFIG["text_cross_attn"]) else None,
                            speaker_ids1,
                            steps_first=int(getattr(args, "mel_twopass_steps_first", mel_steps)),
                            steps_second=int(getattr(args, "mel_twopass_steps_second", 0)),
                            t_noise=float(getattr(args, "mel_twopass_t_noise", 0.12)),
                            spk_vec_override=spk_base1,
                        )
                    else:
                        mel_out1 = sample_mel_flow(
                            mel_flow,
                            x01,
                            x_tok1 if bool(CONFIG["text_cross_attn"]) else None,
                            speaker_ids1,
                            steps=int(mel_steps),
                            spk_vec_override=spk_base1,
                        )

                # Update prefix state for next chunk (store *this* chunk as previous).
                try:
                    prev_mel_out_bct = mel_out1.detach().to(dtype=torch.float32)
                except Exception:
                    prev_mel_out_bct = None

                mel_dec = mel_out1
                silence_mask_dec = None
                if bool(getattr(args, "pause_force_digital_silence_infer", False)):
                    silence_mask_dec = _pause_middle_frame_mask_from_durations(
                        ids_full=ids_full1,
                        dur_values=dur_for_prior1,
                        T=int(mel_out1.size(-1)),
                        edge_frames=int(getattr(args, "pause_edge_frames", 0)),
                    )
                if int(prefix_k) > 0 and int(mel_dec.size(-1)) > int(prefix_k):
                    mel_dec = mel_dec[..., int(prefix_k) :].contiguous()
                    if torch.is_tensor(silence_mask_dec) and int(silence_mask_dec.size(-1)) > int(prefix_k):
                        silence_mask_dec = silence_mask_dec[..., int(prefix_k) :].contiguous()
                elif int(prefix_k) > 0:
                    continue
                wav = _vocos_decode_or_none(mel_dec, tag=f"infer/{tag_out}")
                if wav is None:
                    raise RuntimeError(f"vocos.decode failed in infer-only for tag={tag} (bad/empty/non-finite mel).")
                if bool(getattr(args, "pause_force_digital_silence_infer", False)):
                    wav = _apply_digital_silence_to_wav(wav, silence_mask_dec)
                wav = wav.detach().cpu()
                if wav_joined and (join_sil is not None) and (join_sil.numel() > 0):
                    wav_joined.append(join_sil)
                wav_joined.append(wav)

            if wav_joined:
                _save_wav(out_inf / f"infer__{tag_out}.wav", torch.cat(wav_joined, dim=-1).contiguous(), sr=24000)
                (out_inf / f"infer__{tag_out}__dur_debug.txt").write_text(
                    "\n".join(dur_debug_lines) + "\n",
                    encoding="utf-8",
                )

        (out_inf / "meta.txt").write_text(
            f"resume={getattr(args,'resume','')}\n"
            f"infer_tag={infer_tag_prefix}\n"
            f"speaker_a={sid_a} speaker_b={sid_b} mix={do_mix} alpha={alpha}\n"
            f"prior_noise_scale={float(getattr(args,'prior_noise_scale',1.0))}\n"
            f"mel_steps={mel_steps} dur_steps={dur_steps} dur_noise_scale={dur_noise}\n"
            f"decode_joined=disabled chunkwise_vocos=enabled mel_join_overlap_ms=0.0 frames=0\n"
            f"prefix_continuity={'enabled' if _prefix_frames_from_ms(_short_continuity_ms) > 0 else 'disabled'} "
            f"(fixed clamp+trim; prefix_ms={_short_continuity_ms} frames={_prefix_frames_from_ms(_short_continuity_ms)})\n"
            f"dataset_json={args.dataset_json}\n",
            encoding="utf-8",
        )
        print(f"✅ infer-only saved to: {out_inf}")
        return

    # -------- Training loop --------
    for ep in range(start_epoch, int(args.epochs) + 1):
        # Prefix continuity (training): cache prev tail per (speaker_id, book_id) and
        # enable only when chunk_idx is consecutive and prefix_ms > 0.
        _train_prev_mel_tail: Dict[Tuple[int, str], Tuple[int, torch.Tensor]] = {}
        _train_prev_mel_fill: Dict[Tuple[int, str], Tuple[int, torch.Tensor]] = {}
        _train_prefix_frames = int(_prefix_frames_from_ms(_short_continuity_ms))
        if stateful_sampler is not None:
            try:
                stateful_sampler.set_epoch(int(ep))
            except Exception:
                pass
        if hasattr(ds_train, "set_epoch"):
            try:
                ds_train.set_epoch(int(ep))  # type: ignore[attr-defined]
            except Exception:
                pass
        prior_mu.train()
        dur_predictor.train()
        mel_flow.train()
        spk_adapter.train()
        gender_embed.train()
        if speaker_encoder is not None:
            if bool(getattr(args, "speaker_encoder_trainable", True)):
                speaker_encoder.train()
            else:
                speaker_encoder.eval()
        if train_base:
            model.train()
            bridge.train()
            spk_embed.eval()

        t0 = time.time()
        running = 0.0
        n_batches = 0
        last_loss_mu = None
        last_loss_flow = None
        last_loss_dur = None
        last_loss_dur_flow = None
        last_loss_dur_prior_aux = None
        last_loss_prior = None
        last_loss_spk = None
        last_loss_spk_teacher = None
        last_loss_spk_verifier = None
        last_loss_pitch = None
        last_loss_energy = None
        last_loss_online_ctc = None
        last_loss_total_dur = None
        last_dur_budget_ratio = None
        last_loss_token_floor = None

        for batch_i, batch in enumerate(dl_train):
            (
                mel_pad, T_len, tok_pad, N_tok, L_gt_pad, prosody_pad,
                speaker_ids, chunk_idx, book_ids, author_ids,
                audio_paths,
                speaker_emb_b,
                gender_ids,
                emotion_ids,
                prior_f0_paths,
            ) = batch

            nb = (device.type == "cuda")
            mel_pad = mel_pad.to(device, non_blocking=nb)
            T_len = T_len.to(device, non_blocking=nb)
            tok_pad = tok_pad.to(device, non_blocking=nb)
            N_tok = N_tok.to(device, non_blocking=nb)
            L_gt_pad = L_gt_pad.to(device, non_blocking=nb).float()
            speaker_ids = speaker_ids.to(device, non_blocking=nb)
            speaker_emb_b = speaker_emb_b.to(device, non_blocking=nb).to(dtype=torch.float32)
            gender_ids = gender_ids.to(device, non_blocking=nb)
            emotion_ids = emotion_ids.to(device, non_blocking=nb)
            if bool(getattr(args, "disable_gender_token", True)):
                gender_ids = torch.zeros_like(gender_ids)
            gender_drop_p = float(getattr(args, "gender_dropout_prob", 0.0))
            if gender_drop_p > 0.0:
                gender_drop_p = max(0.0, min(1.0, gender_drop_p))
                gender_drop_mask = torch.rand_like(gender_ids.float()) < gender_drop_p
                gender_ids = gender_ids.masked_fill(gender_drop_mask, 0)

            L_gt_pad = torch.nan_to_num(L_gt_pad, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)

            # -------- Encoder (frozen w trybie z ckpt / trainable w trybie od zera) --------
            book_ids_t = _book_ids_to_tensor(book_ids)
            mel_ref_bct = _ensure_mel_bct(mel_pad)
            spk_256 = learned_spk_table(speaker_ids.clamp(0, int(num_speakers) - 1)).float()
            style_128 = learned_style_table(speaker_ids.clamp(0, int(num_speakers) - 1)).float()
            style_128 = _apply_emotion_style_conditioning(
                style_128,
                emotion_ids,
                enabled=bool(emotion_conditioning_enabled),
                emotion_embed=emotion_embed,
                emotion_to_style=emotion_to_style,
                emotion_style_gate=emotion_style_gate,
            )
            if not bool(torch.isfinite(spk_256).all().item()):
                opt.zero_grad(set_to_none=True)
                _dump_and_maybe_abort(
                    reason="nonfinite_speaker_vector",
                    ep=int(ep),
                    batch_i=int(batch_i),
                    speaker_ids=speaker_ids,
                    audio_paths=audio_paths,
                    tensors={"speaker_emb_b": speaker_emb_b},
                )
                continue
            spk_256 = spk_256 / spk_256.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            spk_base_override = spk_adapter(spk_256)
            last_loss_spk = 0.0
            if train_base:
                with _autocast(enabled=use_amp):
                    x_tok, ids_full_enc, special_len, mem_after = encode_text_features_stateful(
                        model=model,
                        spk_embed=spk_embed,
                        gender_embed=gender_embed,
                        emotion_token_embed=emotion_token_embed,
                        tok_pad=tok_pad,
                        speaker_ids=speaker_ids,
                        gender_ids=gender_ids,
                        emotion_ids=emotion_ids,
                        book_ids=book_ids_t,
                        chunk_idx=chunk_idx,
                        device=device,
                        bridge=bridge,
                        bridge_cache=bridge_cache,
                        spk_vec_override=spk_base_override,
                        require_spk_override=bool(getattr(args, "require_spk_override", True)),
                        use_emotion_token=bool(emotion_conditioning_enabled),
                    )
            else:
                with _autocast(enabled=use_amp):
                    x_tok, ids_full_enc, special_len, mem_after = encode_text_features_stateful(
                        model=model,
                        spk_embed=spk_embed,
                        gender_embed=gender_embed,
                        emotion_token_embed=emotion_token_embed,
                        tok_pad=tok_pad,
                        speaker_ids=speaker_ids,
                        gender_ids=gender_ids,
                        emotion_ids=emotion_ids,
                        book_ids=book_ids_t,
                        chunk_idx=chunk_idx,
                        device=device,
                        bridge=bridge,
                        bridge_cache=bridge_cache,
                        spk_vec_override=spk_base_override,
                        require_spk_override=bool(getattr(args, "require_spk_override", True)),
                        use_emotion_token=bool(emotion_conditioning_enabled),
                    )

            loss_online_ctc = mel_pad.new_zeros(())
            L_gt_pad_sup = L_gt_pad
            if asr_model is not None and aligner_mod is not None:
                ctc_ctx = _autocast(enabled=use_amp) if bool(getattr(args, "online_ctc_trainable", False)) else torch.no_grad()
                with ctc_ctx:
                    asr_log_probs, asr_lens = asr_model(mel_pad, T_len)
                    asr_log_probs = asr_log_probs.float()
                    L_gt_pad_sup = _reconstruct_online_full_durs(
                        aligner_mod,
                        asr_log_probs,
                        tok_pad,
                        N_tok,
                        asr_lens,
                        target_frame_lens=T_len,
                    )
                    if bool(getattr(args, "online_ctc_trainable", False)):
                        ctc_targets, ctc_target_lens = _build_ctc_targets(tok_pad, N_tok)
                        if int(ctc_targets.numel()) > 0 and int(ctc_target_lens.sum().item()) > 0:
                            loss_online_ctc = F.ctc_loss(
                                asr_log_probs.transpose(0, 1),
                                ctc_targets,
                                asr_lens,
                                ctc_target_lens,
                                blank=0,
                                zero_infinity=True,
                            )

            # pad L_gt do prefixu
            L_gt_full = F.pad(L_gt_pad_sup, (int(special_len), 0), value=0.0)
            ids_full = F.pad(tok_pad, (int(special_len), 0), value=PAD_ID)
            dur_allowed = _build_dur_allowed_mask(ids_full, special_len)

            loss_style = x_tok.new_zeros(())

            x_tok_dur = x_tok

            if torch.is_tensor(spk_base_override):
                spk_base = spk_base_override.to(device=x_tok.device, dtype=x_tok.dtype)
            else:
                if bool(getattr(args, "require_spk_override", True)):
                    raise RuntimeError(
                        "require_spk_override=True but spk_base_override is None/invalid in training loop. "
                        "Your dataset likely lacks speaker_embeds, or speaker_embeds failed to load."
                    )
                spk_base = spk_embed(speaker_ids).to(device=x_tok.device, dtype=x_tok.dtype)
            spk_dur = _zero_spk_cond_like(spk_base)
            prior_cond = None

            # -------- duration flow loss --------
            flow_cond_train = _build_flow_cond(
                CONFIG["dur_flow_cond_mode"],
                CONFIG["dur_flow_cond_source_train"],
                L_gt_full_used=L_gt_full,
                T_len=T_len,
                total_len_pred=None,
                rate_pred=None,
                tok_pad=tok_pad,
                special_len=special_len,
            )

            # wymuś: tylko tokeny tekstowe + pauzy mogą mieć duracje
            L_gt_full = torch.where(dur_allowed, L_gt_full, torch.zeros_like(L_gt_full))
            target_log = torch.log1p(L_gt_full.clamp_min(0.0))
            x_mask = dur_allowed.float().unsqueeze(1)  # [B,1,L]
            # Dur-loss licz w FP32 (AMP potrafi pogorszyć stabilność dur_pred).
            with _autocast(enabled=False):
                if hasattr(model.dur, "predict_logdur"):
                    dur_loss = model.dur.loss(  # type: ignore[attr-defined]
                        x_tok_dur.float(),
                        x_mask.float(),
                        target_log.float(),
                        kind=str(getattr(args, "ar_dur_loss", "huber")),
                        style_vec=style_128.to(device=x_tok_dur.device, dtype=torch.float32) if torch.is_tensor(style_128) else None,
                    )
                    last_loss_dur_flow = None
                    last_loss_dur_prior_aux = None
                else:
                    use_dur_prior_x0 = (
                        str(getattr(args, "dur_x0", "none")).lower().strip() == "prior" and
                        hasattr(model.dur, "flow_loss_with_prior_x0")
                    )
                    if use_dur_prior_x0:
                        dur_loss, _dur_flow_loss, _dur_prior_aux = model.dur.flow_loss_with_prior_x0(  # type: ignore[attr-defined]
                            x_tok_bld=x_tok_dur.float(),
                            x_mask_b1l=x_mask.float(),
                            target_log_bl=target_log.float(),
                            spk_emb_bd=spk_dur.float(),
                            style_vec_bd=None,
                            cond=flow_cond_train.float() if torch.is_tensor(flow_cond_train) else None,
                            prior_loss_mode="nll",
                            prior_w=float(getattr(args, "dur_prior_w", 0.10)),
                            prior_kl_w=0.0,
                            prior_logs_min=float(getattr(args, "dur_prior_logs_min", -5.0)),
                            prior_logs_max=float(getattr(args, "dur_prior_logs_max", 2.0)),
                            prior_sigma_min=float(getattr(args, "dur_prior_sigma_min", 0.1)),
                            x0_noise_scale=float(getattr(args, "dur_x0_noise_scale", 1.0)),
                            sigma0_train_min=0.0,
                            sigma0_train_max=0.0,
                        )
                        last_loss_dur_flow = float(_dur_flow_loss.detach().cpu())
                        last_loss_dur_prior_aux = float(_dur_prior_aux.detach().cpu())
                    else:
                        dur_loss = model.dur.flow_loss(
                            x_tok_dur.float(),
                            x_mask.float(),
                            target_log.float(),
                            spk_emb=spk_dur.float(),
                            style_vec=None,
                            cond=flow_cond_train.float() if torch.is_tensor(flow_cond_train) else None,
                        )
                        last_loss_dur_flow = None
                        last_loss_dur_prior_aux = None

            # -------- choose durations for prior --------
            dur_for_prior = _select_dec_durations(L_gt_full, None, CONFIG["dec_dur_source_train"])
            dur_for_prior = torch.where(dur_allowed, dur_for_prior, torch.zeros_like(dur_for_prior))

            sp_mask_tok = _pause_mask_from_ids(ids_full)
            # -------- mel prior (gauss_nll): t0 (noisy), mu/logs (stats) --------
            with _autocast(enabled=use_amp):
                t0_btc, mu_btc, logs_btc, T_vec, prior_h_btd = prior_mu(
                    x_tok,
                    dur_for_prior,
                    cond=prior_cond,
                    T_hint=T_len,
                    noise_scale=float(args.prior_noise_scale),
                    sp_mask_tok=sp_mask_tok,
                    timbre_vec=spk_base,
                    return_hidden=True,
                )
            prior_gauss_align_btl = getattr(prior_mu, "last_gauss_align_btl", None)

            # dopasuj GT do T
            T = int(mu_btc.size(1))
            mel_gt_bct = _ensure_mel_bct(mel_pad)
            mel_gt_bct = _crop_or_pad_bct(mel_gt_bct, T)
            tmask = _make_tmask_from_Tlen(T_len.clamp_max(T), T).to(mel_gt_bct.dtype)

            # ---- Prefix continuity / prefix-fill training ----
            # Short continuity overwrites a small target prefix. Prefix-fill prepends a longer GT
            # acoustic prefix and masks it out, leaving the current text/durations unchanged.
            _prefix_k_per_item: List[int] = [0 for _ in range(int(mel_gt_bct.size(0)))]
            _prefix_tail_cpu: List[Optional[torch.Tensor]] = [None for _ in range(int(mel_gt_bct.size(0)))]
            _prefix_fill_bct: Optional[torch.Tensor] = None
            _prefix_fill_mask: Optional[torch.Tensor] = None
            try:
                spk_ids_cpu = speaker_ids.detach().cpu().tolist()
            except Exception:
                spk_ids_cpu = [int(x) for x in speaker_ids]
            try:
                chunk_ids_cpu = chunk_idx.detach().cpu().tolist() if torch.is_tensor(chunk_idx) else [int(x) for x in chunk_idx]
            except Exception:
                chunk_ids_cpu = [None for _ in range(len(spk_ids_cpu))]
            try:
                book_ids_py = [str(b) for b in book_ids]
            except Exception:
                book_ids_py = ["?" for _ in range(len(spk_ids_cpu))]

            if _prefix_fill_enabled:
                prefix_fill_items: List[Optional[torch.Tensor]] = [None for _ in range(int(mel_gt_bct.size(0)))]
                prefix_fill_ks: List[int] = [0 for _ in range(int(mel_gt_bct.size(0)))]
                for ii in range(int(mel_gt_bct.size(0))):
                    try:
                        sid_i = int(spk_ids_cpu[ii])
                        bid_i = str(book_ids_py[ii])
                        cidx_i = int(chunk_ids_cpu[ii])
                    except Exception:
                        continue
                    prev = _train_prev_mel_fill.get((sid_i, bid_i), None)
                    if prev is None:
                        continue
                    prev_idx, prev_tail = prev
                    if int(prev_idx) != int(cidx_i) - 1:
                        continue
                    if (not torch.is_tensor(prev_tail)) or prev_tail.numel() <= 0:
                        continue
                    k = int(min(int(_prefix_fill_frames), int(prev_tail.size(-1))))
                    if k <= 0:
                        continue
                    tail_k = prev_tail[:, :, -k:].to(dtype=torch.float32, device="cpu").contiguous()
                    prefix_fill_items[ii] = tail_k
                    prefix_fill_ks[ii] = int(k)

                kmax = int(max(prefix_fill_ks) if prefix_fill_ks else 0)
                if kmax > 0:
                    _prefix_fill_bct = torch.zeros(
                        int(mel_gt_bct.size(0)),
                        int(mel_gt_bct.size(1)),
                        kmax,
                        device=mel_gt_bct.device,
                        dtype=mel_gt_bct.dtype,
                    )
                    _prefix_fill_mask = torch.zeros(
                        int(mel_gt_bct.size(0)),
                        1,
                        kmax,
                        device=mel_gt_bct.device,
                        dtype=mel_gt_bct.dtype,
                    )
                    for ii, tail_k in enumerate(prefix_fill_items):
                        if not torch.is_tensor(tail_k):
                            continue
                        k = int(min(int(prefix_fill_ks[ii]), kmax, int(tail_k.size(-1))))
                        if k <= 0:
                            continue
                        # Right-align the available prefix inside the fixed batch prefix window.
                        _prefix_fill_bct[ii : ii + 1, :, kmax - k : kmax] = tail_k[:, :, -k:].to(
                            device=mel_gt_bct.device,
                            dtype=mel_gt_bct.dtype,
                        )
                    mel_gt_bct = torch.cat([_prefix_fill_bct, mel_gt_bct], dim=-1)
                    tmask = torch.cat([_prefix_fill_mask, tmask], dim=-1)

                # Update fill cache from the original current target, not from the prepended tensor.
                for ii in range(int(tmask.size(0))):
                    try:
                        sid_i = int(spk_ids_cpu[ii])
                        bid_i = str(book_ids_py[ii])
                        cidx_i = int(chunk_ids_cpu[ii])
                    except Exception:
                        continue
                    Ti = int(min(int(T), int(T_len[ii].detach().cpu().item() if torch.is_tensor(T_len) else int(T_len[ii]))))
                    if Ti <= 0:
                        continue
                    kk = int(min(int(_prefix_fill_frames), int(Ti)))
                    if kk <= 0:
                        continue
                    target_mel_bct = mel_gt_bct[ii : ii + 1, :, -T:]
                    tail = target_mel_bct[:, :, Ti - kk : Ti].detach().to(device="cpu", dtype=torch.float32).contiguous()
                    _train_prev_mel_fill[(sid_i, bid_i)] = (int(cidx_i), tail)
            else:
                for ii in range(int(mel_gt_bct.size(0))):
                    try:
                        sid_i = int(spk_ids_cpu[ii])
                        bid_i = str(book_ids_py[ii])
                        cidx_i = int(chunk_ids_cpu[ii])
                    except Exception:
                        continue
                    prev = _train_prev_mel_tail.get((sid_i, bid_i), None)
                    if prev is None:
                        continue
                    prev_idx, prev_tail = prev
                    if int(prev_idx) != int(cidx_i) - 1:
                        continue
                    if (not torch.is_tensor(prev_tail)) or prev_tail.numel() <= 0:
                        continue
                    Ti = int(min(int(T), int(T_len[ii].detach().cpu().item() if torch.is_tensor(T_len) else int(T_len[ii]))))
                    if Ti <= 0:
                        continue
                    k = int(min(int(_train_prefix_frames), int(Ti), int(prev_tail.size(-1))))
                    if k <= 0:
                        continue
                    tail_k = prev_tail[:, :, -k:].to(dtype=torch.float32, device="cpu").contiguous()
                    mel_gt_bct[ii : ii + 1, :, :k] = tail_k.to(device=mel_gt_bct.device, dtype=mel_gt_bct.dtype)
                    tmask[ii : ii + 1, :, :k] = 0.0
                    _prefix_k_per_item[ii] = int(k)
                    _prefix_tail_cpu[ii] = tail_k

                # Update cache for next chunk: always store current GT tail (valid frames only).
                for ii in range(int(mel_gt_bct.size(0))):
                    try:
                        sid_i = int(spk_ids_cpu[ii])
                        bid_i = str(book_ids_py[ii])
                        cidx_i = int(chunk_ids_cpu[ii])
                    except Exception:
                        continue
                    Ti = int(min(int(T), int(T_len[ii].detach().cpu().item() if torch.is_tensor(T_len) else int(T_len[ii]))))
                    if Ti <= 0:
                        continue
                    kk = int(min(int(_train_prefix_frames), int(Ti)))
                    if kk <= 0:
                        continue
                    tail = mel_gt_bct[ii : ii + 1, :, Ti - kk : Ti].detach().to(device="cpu", dtype=torch.float32).contiguous()
                    _train_prev_mel_tail[(sid_i, bid_i)] = (int(cidx_i), tail)

            mu_bct = mu_btc.transpose(1, 2).contiguous().float()
            logs_bct = logs_btc.transpose(1, 2).contiguous().float()
            if _prefix_fill_bct is not None:
                prefix_f = _prefix_fill_bct.to(device=mu_bct.device, dtype=mu_bct.dtype)
                mu_bct = torch.cat([prefix_f, mu_bct], dim=-1)
                logs_bct = torch.cat([torch.zeros_like(prefix_f), logs_bct], dim=-1)
            acoustic_tmask = tmask.float()
            pause_mid_w = float(getattr(args, "pause_mid_loss_weight", 1.0))
            pause_edge_k = int(getattr(args, "pause_edge_frames", 0))
            if pause_mid_w < 0.9999 or pause_edge_k > 0:
                pause_weights = _pause_center_loss_weights_from_durations(
                    ids_full=ids_full,
                    dur_values=dur_for_prior,
                    T=T,
                    edge_frames=pause_edge_k,
                    pause_mid_weight=pause_mid_w,
                    dtype=mel_gt_bct.dtype,
                )
                if _prefix_fill_bct is not None:
                    pause_weights = torch.cat(
                        [
                            torch.ones(
                                int(pause_weights.size(0)),
                                1,
                                int(_prefix_fill_bct.size(-1)),
                                device=pause_weights.device,
                                dtype=pause_weights.dtype,
                            ),
                            pause_weights,
                        ],
                        dim=-1,
                    )
                acoustic_tmask = acoustic_tmask * pause_weights
            loss_prior = _gaussian_nll_bct(mu_bct, logs_bct, mel_gt_bct.float(), acoustic_tmask)

            # monitoringu (L1 jest czytelne dla człowieka)
            loss_mu = _masked_l1_bct(mu_bct, mel_gt_bct.float(), acoustic_tmask)

            loss_prior_energy = x_tok.new_zeros((), dtype=torch.float32)
            loss_prior_f0 = x_tok.new_zeros((), dtype=torch.float32)
            if float(getattr(args, "w_prior_energy", 0.0)) > 0.0 or float(getattr(args, "w_prior_f0", 0.0)) > 0.0:
                prior_f0_pred_bt, prior_energy_pred_bt = prior_prosody(prior_h_btd.float())
                if _prefix_fill_bct is not None:
                    zpad = torch.zeros(
                        int(prior_energy_pred_bt.size(0)),
                        int(_prefix_fill_bct.size(-1)),
                        device=prior_energy_pred_bt.device,
                        dtype=prior_energy_pred_bt.dtype,
                    )
                    prior_energy_pred_bt = torch.cat([zpad, prior_energy_pred_bt], dim=-1)
                    prior_f0_pred_bt = torch.cat([zpad, prior_f0_pred_bt], dim=-1)
                if float(getattr(args, "w_prior_energy", 0.0)) > 0.0:
                    energy_target_bt = _styletts_log_norm(mel_gt_bct.float())
                    loss_prior_energy = _masked_smooth_l1_bt(prior_energy_pred_bt, energy_target_bt, acoustic_tmask)
                if float(getattr(args, "w_prior_f0", 0.0)) > 0.0:
                    f0_target_bt = _load_cached_f0_targets(
                        prior_f0_paths,
                        device=prior_f0_pred_bt.device,
                        T=int(mel_gt_bct.size(-1)),
                        log_hz=True,
                    )
                    if torch.is_tensor(f0_target_bt):
                        loss_prior_f0 = _masked_smooth_l1_bt(prior_f0_pred_bt, f0_target_bt, acoustic_tmask)

            x0_bct = t0_btc.transpose(1, 2).contiguous().float()
            if _prefix_fill_bct is not None:
                x0_bct = torch.cat([_prefix_fill_bct.to(device=x0_bct.device, dtype=x0_bct.dtype), x0_bct], dim=-1)

            # match x0 prefix to injected GT prefix (keeps flow tuple consistent); masked out in losses.
            if _prefix_fill_bct is None:
                for ii, k in enumerate(_prefix_k_per_item):
                    if int(k) <= 0:
                        continue
                    tail_k = _prefix_tail_cpu[ii]
                    if not torch.is_tensor(tail_k):
                        continue
                    kk = int(min(int(k), int(x0_bct.size(-1)), int(tail_k.size(-1))))
                    if kk <= 0:
                        continue
                    x0_bct[ii : ii + 1, :, :kk] = tail_k[:, :, -kk:].to(device=x0_bct.device, dtype=x0_bct.dtype)

            # -------- flow-matching --------
            # POPRAWKA: jedno spójne t
            t_b = _sample_flow_t(
                int(x0_bct.size(0)),
                device=device,
                dtype=x0_bct.dtype,
                mode=str(getattr(args, "t_sample_mode", "logit_normal")),
                logit_mu=float(getattr(args, "t_logit_mu", 0.0)),
                logit_sigma=float(getattr(args, "t_logit_sigma", 1.0)),
            )
            xt_bct, target_v_bct = flow_helper.get_flow_tuple(x0_bct, mel_gt_bct, t_b)

            text_cond = x_tok if bool(CONFIG["text_cross_attn"]) else None
            mel_flow_amp = bool(use_amp) and (not bool(getattr(args, "mel_flow_fp32", True)))
            with _autocast(enabled=mel_flow_amp):
                pred_v_bct = mel_flow(
                    xt_bct.float() if bool(getattr(args, "mel_flow_fp32", True)) else xt_bct,
                    t_b.float() if bool(getattr(args, "mel_flow_fp32", True)) else t_b,
                    speaker_ids,
                    text_cond,
                    spk_vec_override=spk_base,
                    gauss_align_btl=prior_gauss_align_btl
                    if bool(getattr(args, "flow_gauss_token_attn", False) or getattr(args, "flow_gauss_cross_attn", False))
                    else None,
                )
            if not bool(torch.isfinite(pred_v_bct).all().item()):
                opt.zero_grad(set_to_none=True)
                _dump_and_maybe_abort(
                    reason="nonfinite_mel_flow_pred_v",
                    ep=int(ep),
                    batch_i=int(batch_i),
                    speaker_ids=speaker_ids,
                    audio_paths=audio_paths,
                    tensors={
                        "xt_bct": xt_bct,
                        "t_b": t_b,
                        "pred_v_bct": pred_v_bct,
                        "target_v_bct": target_v_bct,
                        "tmask": tmask,
                        "speaker_emb_b": speaker_emb_b,
                    },
                )
                continue
            loss_flow = F.l1_loss(pred_v_bct.float() * acoustic_tmask, target_v_bct.float() * acoustic_tmask, reduction="sum") / (
                (acoustic_tmask.sum() * N_MELS).clamp_min(1.0)
            )

            mel_hat_bct: Optional[torch.Tensor] = None
            loss_spk = x_tok.new_zeros((), dtype=torch.float32)
            loss_spk_teacher = x_tok.new_zeros((), dtype=torch.float32)
            loss_spk_verifier = x_tok.new_zeros((), dtype=torch.float32)
            loss_pitch = x_tok.new_zeros((), dtype=torch.float32)
            loss_energy = x_tok.new_zeros((), dtype=torch.float32)
            wav_hat_for_losses = None
            wav_ref_for_losses = None
            if speaker_loss_enabled or speaker_teacher_enabled:
                if not bool(getattr(mel_flow, "spk_style_ref_ready", False)):
                    raise RuntimeError("speaker loss enabled but frozen dualhead encoder is not ready.")
                spk_mask_bt = _make_tmask_from_Tlen(T_len.clamp_max(T), T).squeeze(1).to(dtype=torch.bool, device=mel_gt_bct.device)
                spk_ref_mel_bct = mel_gt_bct[:, :, -T:].contiguous() if _prefix_fill_bct is not None else mel_gt_bct
                with torch.no_grad():
                    z_ref_spk, _z_ref_style = mel_flow.encode_ref_dual(  # type: ignore[attr-defined]
                        spk_ref_mel_bct.float(),
                        mask_bt=spk_mask_bt,
                    )
                    z_ref_spk = z_ref_spk.detach().float()
                if speaker_loss_enabled:
                    mel_hat_bct = (x0_bct.float() + pred_v_bct.float()).contiguous()
                    spk_hat_bct = mel_hat_bct[:, :, -T:].contiguous() if _prefix_fill_bct is not None else mel_hat_bct
                    z_gen_spk, _z_gen_style = mel_flow.encode_ref_dual(  # type: ignore[attr-defined]
                        spk_hat_bct,
                        mask_bt=spk_mask_bt,
                    )
                    loss_spk = (1.0 - F.cosine_similarity(z_gen_spk.float(), z_ref_spk, dim=-1)).mean()
                    last_loss_spk = float(loss_spk.detach().cpu())
                if speaker_teacher_enabled:
                    z_student_spk = spk_adapter.encode_teacher_space(spk_256)
                    loss_spk_teacher = (1.0 - F.cosine_similarity(z_student_spk.float(), z_ref_spk, dim=-1)).mean()
                    last_loss_spk_teacher = float(loss_spk_teacher.detach().cpu())

            verifier_enabled = (
                speaker_verifier is not None
                and float(getattr(args, "w_speaker_verifier", 0.0)) > 0.0
                and int(getattr(args, "speaker_verifier_every", 1)) > 0
                and (int(batch_i) % int(getattr(args, "speaker_verifier_every", 1)) == 0)
            )
            if verifier_enabled:
                if vocos is None:
                    raise RuntimeError("speaker verifier loss requires Vocos")
                if mel_hat_bct is None:
                    mel_hat_bct = (x0_bct.float() + pred_v_bct.float()).contiguous()
                spk_hat_for_verifier = mel_hat_bct[:, :, -T:].contiguous() if _prefix_fill_bct is not None else mel_hat_bct
                spk_ref_for_verifier = mel_gt_bct[:, :, -T:].contiguous() if _prefix_fill_bct is not None else mel_gt_bct
                wav_hat = _decode_mel_for_verifier(
                    vocos,
                    spk_hat_for_verifier,
                    max_sec=float(getattr(args, "speaker_verifier_max_sec", 3.0)),
                )
                with torch.no_grad():
                    wav_ref = _decode_mel_for_verifier(
                        vocos,
                        spk_ref_for_verifier.detach(),
                        max_sec=float(getattr(args, "speaker_verifier_max_sec", 3.0)),
                    )
                    z_ref_ver = None if wav_ref is None else speaker_verifier.encode(wav_ref.to(device), sample_rate=24000).detach()
                if wav_hat is not None and z_ref_ver is not None:
                    wav_hat_for_losses = wav_hat
                    wav_ref_for_losses = wav_ref
                    z_hat_ver = speaker_verifier.encode(wav_hat.to(device), sample_rate=24000)
                    loss_spk_verifier = (1.0 - F.cosine_similarity(z_hat_ver.float(), z_ref_ver.float(), dim=-1)).mean()
                    last_loss_spk_verifier = float(loss_spk_verifier.detach().cpu())

            prosody_enabled = (
                (float(getattr(args, "w_pitch", 0.0)) > 0.0 or float(getattr(args, "w_energy", 0.0)) > 0.0)
                and int(getattr(args, "prosody_loss_every", 1)) > 0
                and (int(batch_i) % int(getattr(args, "prosody_loss_every", 1)) == 0)
            )
            if prosody_enabled:
                if vocos is None:
                    raise RuntimeError("pitch/energy losses require Vocos")
                if mel_hat_bct is None:
                    mel_hat_bct = (x0_bct.float() + pred_v_bct.float()).contiguous()
                spk_hat_for_prosody = mel_hat_bct[:, :, -T:].contiguous() if _prefix_fill_bct is not None else mel_hat_bct
                spk_ref_for_prosody = mel_gt_bct[:, :, -T:].contiguous() if _prefix_fill_bct is not None else mel_gt_bct
                if wav_hat_for_losses is None:
                    wav_hat_for_losses = _decode_mel_for_verifier(
                        vocos,
                        spk_hat_for_prosody,
                        max_sec=float(getattr(args, "prosody_loss_max_sec", 2.0)),
                    )
                if wav_ref_for_losses is None:
                    with torch.no_grad():
                        wav_ref_for_losses = _decode_mel_for_verifier(
                            vocos,
                            spk_ref_for_prosody.detach(),
                            max_sec=float(getattr(args, "prosody_loss_max_sec", 2.0)),
                        )
                if wav_hat_for_losses is not None and wav_ref_for_losses is not None:
                    if float(getattr(args, "w_pitch", 0.0)) > 0.0:
                        loss_pitch = _pitch_consistency_loss(
                            wav_hat_for_losses.to(device),
                            wav_ref_for_losses.to(device),
                            sample_rate=24000,
                        )
                        last_loss_pitch = float(loss_pitch.detach().cpu())
                    if float(getattr(args, "w_energy", 0.0)) > 0.0:
                        loss_energy = _energy_consistency_loss(
                            wav_hat_for_losses.to(device),
                            wav_ref_for_losses.to(device),
                        )
                        last_loss_energy = float(loss_energy.detach().cpu())

            if float(getattr(args, "w_prior_f0", 0.0)) > 0.0:
                last_loss_pitch = float(loss_prior_f0.detach().cpu())
            if float(getattr(args, "w_prior_energy", 0.0)) > 0.0:
                last_loss_energy = float(loss_prior_energy.detach().cpu())

            loss_total_dur, dur_budget_ratio = _duration_budget_loss(
                dur_for_prior,
                dur_allowed,
                T_len,
            )
            loss_token_floor = _token_duration_floor_loss(
                dur_for_prior,
                ids_full,
                dur_allowed,
                min_text_frames=float(getattr(args, "token_dur_floor_frames", 1.5)),
            )
            last_loss_total_dur = float(loss_total_dur.detach().cpu())
            last_dur_budget_ratio = float(dur_budget_ratio.detach().cpu())
            last_loss_token_floor = float(loss_token_floor.detach().cpu())

            # -------- total --------
            total_loss = (
                float(args.w_flow) * loss_flow +
                float(args.w_prior) * loss_prior +
                float(args.w_dur) * dur_loss 
                + float(getattr(args, "w_total_dur", 0.0)) * loss_total_dur.to(loss_flow.dtype)
                + float(getattr(args, "w_token_dur_floor", 0.0)) * loss_token_floor.to(loss_flow.dtype)
                + float(getattr(args, "w_online_ctc", 0.10)) * loss_online_ctc.to(loss_flow.dtype)
                + float(getattr(args, "speaker_loss_w", 0.0)) * loss_spk.to(loss_flow.dtype)
                + float(getattr(args, "speaker_teacher_w", 0.0)) * loss_spk_teacher.to(loss_flow.dtype)
                + float(getattr(args, "w_speaker_verifier", 0.0)) * loss_spk_verifier.to(loss_flow.dtype)
                + float(getattr(args, "w_pitch", 0.0)) * loss_pitch.to(loss_flow.dtype)
                + float(getattr(args, "w_energy", 0.0)) * loss_energy.to(loss_flow.dtype)
                + float(getattr(args, "w_prior_energy", 0.0)) * loss_prior_energy.to(loss_flow.dtype)
                + float(getattr(args, "w_prior_f0", 0.0)) * loss_prior_f0.to(loss_flow.dtype)
            )

            if not bool(torch.isfinite(total_loss).item()):
                opt.zero_grad(set_to_none=True)
                _dump_and_maybe_abort(
                    reason="nonfinite_total_loss",
                    ep=int(ep),
                    batch_i=int(batch_i),
                    speaker_ids=speaker_ids,
                    audio_paths=audio_paths,
                    extra_scalars={
                        "total_loss": float(total_loss.detach().cpu().item()) if torch.is_tensor(total_loss) else None,
                        "loss_mu": float(loss_mu.detach().cpu().item()) if torch.is_tensor(loss_mu) else None,
                        "loss_flow": float(loss_flow.detach().cpu().item()) if torch.is_tensor(loss_flow) else None,
                        "dur_loss": float(dur_loss.detach().cpu().item()) if torch.is_tensor(dur_loss) else None,
                        "loss_prior": float(loss_prior.detach().cpu().item()) if torch.is_tensor(loss_prior) else None,
                        "loss_total_dur": float(loss_total_dur.detach().cpu().item()) if torch.is_tensor(loss_total_dur) else None,
                        "dur_budget_ratio": float(dur_budget_ratio.detach().cpu().item()) if torch.is_tensor(dur_budget_ratio) else None,
                        "loss_token_floor": float(loss_token_floor.detach().cpu().item()) if torch.is_tensor(loss_token_floor) else None,
                    },
                    tensors={
                        "mu_bct": mu_bct,
                        "logs_bct": logs_bct,
                        "mel_gt_bct": mel_gt_bct,
                        "tmask": tmask,
                        "pred_v_bct": pred_v_bct,
                        "target_v_bct": target_v_bct,
                    },
                )
                continue

            opt.zero_grad(set_to_none=True)
            if use_amp:
                scaler.scale(total_loss).backward()
                if bool(getattr(args, "grad_diagnose_once", False)):
                    try:
                        scaler.unscale_(opt)
                    except Exception:
                        pass
                    _print_grad_diagnostics(
                        ep=int(ep),
                        batch_i=int(batch_i),
                        total_loss=total_loss,
                        prior_mu=prior_mu,
                        mel_flow=mel_flow,
                    )
                    return
                if float(getattr(args, "grad_clip_norm", 0.0)) > 0:
                    try:
                        scaler.unscale_(opt)
                    except Exception:
                        pass
                    torch.nn.utils.clip_grad_norm_(unique_params, float(getattr(args, "grad_clip_norm", 0.0)))
                scaler.step(opt)
                scaler.update()
            else:
                total_loss.backward()
                if bool(getattr(args, "grad_diagnose_once", False)):
                    _print_grad_diagnostics(
                        ep=int(ep),
                        batch_i=int(batch_i),
                        total_loss=total_loss,
                        prior_mu=prior_mu,
                        mel_flow=mel_flow,
                    )
                    return
                if float(getattr(args, "grad_clip_norm", 0.0)) > 0:
                    torch.nn.utils.clip_grad_norm_(unique_params, float(getattr(args, "grad_clip_norm", 0.0)))
                opt.step()

            running += float(total_loss.detach().cpu())
            n_batches += 1
            last_loss_mu = float(loss_mu.detach().cpu())
            last_loss_flow = float(loss_flow.detach().cpu())
            last_loss_dur = float(dur_loss.detach().cpu())
            last_loss_prior = float(loss_prior.detach().cpu())
            last_loss_online_ctc = float(loss_online_ctc.detach().cpu())

        dt = time.time() - t0
        avg = float("nan") if n_batches <= 0 else (running / max(1, n_batches))
        lp = float("nan") if last_loss_prior is None else float(last_loss_prior)
        prior_tag = f"prior={lp:.4f} noise_scale={float(args.prior_noise_scale):.3f}"
        mu_v = float("nan") if last_loss_mu is None else float(last_loss_mu)
        flow_v = float("nan") if last_loss_flow is None else float(last_loss_flow)
        dur_v = float("nan") if last_loss_dur is None else float(last_loss_dur)
        dur_flow_v = float("nan") if last_loss_dur_flow is None else float(last_loss_dur_flow)
        dur_prior_aux_v = float("nan") if last_loss_dur_prior_aux is None else float(last_loss_dur_prior_aux)
        spk_v = float("nan") if last_loss_spk is None else float(last_loss_spk)
        spk_teacher_v = float("nan") if last_loss_spk_teacher is None else float(last_loss_spk_teacher)
        spk_verifier_v = float("nan") if last_loss_spk_verifier is None else float(last_loss_spk_verifier)
        pitch_v = float("nan") if last_loss_pitch is None else float(last_loss_pitch)
        energy_v = float("nan") if last_loss_energy is None else float(last_loss_energy)
        online_ctc_v = float("nan") if last_loss_online_ctc is None else float(last_loss_online_ctc)
        total_dur_v = float("nan") if last_loss_total_dur is None else float(last_loss_total_dur)
        dur_ratio_v = float("nan") if last_dur_budget_ratio is None else float(last_dur_budget_ratio)
        token_floor_v = float("nan") if last_loss_token_floor is None else float(last_loss_token_floor)
        if str(getattr(args, "dur_x0", "none")).lower().strip() == "prior":
            dur_tag = f"dur={dur_v:.4f} (flow={dur_flow_v:.4f} prior_aux={dur_prior_aux_v:.4f})"
        else:
            dur_tag = f"dur={dur_v:.4f}"
        print(
            f"[ep{ep:04d}] loss={avg:.4f} | "
            f"mu={mu_v:.4f} flow={flow_v:.4f} ctc={online_ctc_v:.4f} spk={spk_v:.4f} spk_ext={spk_verifier_v:.4f} "
            f"pitch={pitch_v:.4f} energy={energy_v:.4f} spk_teacher={spk_teacher_v:.4f} {dur_tag} | "
            f"budget={total_dur_v:.4f} ratio={dur_ratio_v:.3f} floor={token_floor_v:.4f} | "
            f"{prior_tag} | {dt:.1f}s"
        )
        if _ref_source_counts:
            ref_total = max(1, int(sum(_ref_source_counts.values())))
            ref_parts = [
                f"{k}={int(v)}({100.0 * float(v) / float(ref_total):.1f}%)"
            for k, v in sorted(_ref_source_counts.items())
            ]
            print(f"[ep{ep:04d}] dualhead_ref_source: " + " ".join(ref_parts))

        # -------- save --------
        ckpt_out = out_dir / "chkpts" / "prior_mu_flow_spkprefix_dualhead_last.pt"
        chk = {
            "epoch": ep,
            "train_base": bool(train_base),
            "prior_mu": prior_mu.state_dict(),
            "prior_prosody": prior_prosody.state_dict(),
            "online_ctc": (asr_model.state_dict() if asr_model is not None else None),
            "mel_flow": mel_flow.state_dict(),
            "dur_predictor": dur_predictor.state_dict(),
            "dur_prior": None,
            "context_bridge": bridge.state_dict(),
            "spk_adapter": spk_adapter.state_dict(),
            "gender_embed": gender_embed.state_dict(),
            "emotion_embed": emotion_embed.state_dict(),
            "emotion_token_embed": emotion_token_embed.state_dict(),
            "emotion_to_style": emotion_to_style.state_dict(),
            "emotion_style_gate": emotion_style_gate.detach().cpu(),
            "learned_spk_table": learned_spk_table.state_dict(),
            "learned_style_table": learned_style_table.state_dict(),
            "speaker_encoder": speaker_encoder.state_dict() if speaker_encoder is not None else None,
            "optim": opt.state_dict(),
            "scaler": scaler.state_dict() if use_amp else None,
            "args": vars(args),
            "config": CONFIG,
        }
        if train_base:
            chk["base_model"] = model.state_dict()
            chk["bridge"] = bridge.state_dict()
            chk["spk_embed"] = spk_embed.state_dict()
        torch.save(chk, ckpt_out)
        if bool(use_style_encoder128) and speaker_encoder is not None:
            style128_out = out_dir / "chkpts" / "style_encoder128_tts_finetuned_last.pt"
            enc_mod = getattr(speaker_encoder, "encoder", None)
            style128_chk = {
                "epoch": ep,
                "model": enc_mod.state_dict() if enc_mod is not None else speaker_encoder.state_dict(),
                "wrapper": speaker_encoder.state_dict(),
                "spk_proj": getattr(speaker_encoder, "spk_proj", nn.Identity()).state_dict()
                if hasattr(speaker_encoder, "spk_proj") else None,
                "source_checkpoint": str(getattr(args, "style_encoder128_ckpt", "")),
                "args": vars(args),
            }
            torch.save(style128_chk, style128_out)
        if asr_model is not None:
            online_ctc_ckpt_out = out_dir / "chkpts" / "online_ctc_last.pt"
            online_ctc_chk = {
                "state_dict": asr_model.state_dict(),
            "epoch": ep,
            "train_loss": avg,
            "online_ctc_loss": online_ctc_v,
            "duration_budget_loss": total_dur_v,
            "duration_budget_ratio": dur_ratio_v,
            "token_duration_floor_loss": token_floor_v,
            "args": vars(args),
            "config": CONFIG,
        }
            torch.save(online_ctc_chk, online_ctc_ckpt_out)

        # -------- demo --------
        do_demo = int(getattr(args, "demo_every", 0)) > 0 and (
            (ep % int(getattr(args, "demo_every", 1)) == 0) or (ep == int(args.epochs))
        )
        if do_demo and (not math.isfinite(float(avg))):
            print(f"⚠️ Skipping demos at ep{int(ep):04d}: non-finite train loss ({avg}).")
        elif do_demo:
            prior_mu.eval()
            dur_predictor.eval()
            mel_flow.eval()
            spk_adapter.eval()
            gender_embed.eval()
            if speaker_encoder is not None:
                speaker_encoder.eval()

            _save_wav_demo = save_wav

            demo_batches = max(1, int(getattr(args, "demo_batches", 1)))
            demo_iter = iter(dl_demo)
            demo_ref_l1_list: List[float] = []
            demo_mu_l1_list: List[float] = []

            # zapis demo artefaktów tylko dla pierwszego batcha
            saved_demo_payload = None

            for bi in range(demo_batches):
                try:
                    demo_batch = next(demo_iter)
                except StopIteration:
                    demo_iter = iter(dl_demo)
                    demo_batch = next(demo_iter)

                (
                    mel_pad, T_len, tok_pad, N_tok, L_gt_pad, prosody_pad,
                    speaker_ids, chunk_idx, book_ids, author_ids,
                    audio_paths,
                    speaker_emb_b,
                    gender_ids,
                    emotion_ids,
                    prior_f0_paths,
                ) = demo_batch

                mel_pad = mel_pad.to(device)
                T_len = T_len.to(device)
                tok_pad = tok_pad.to(device)
                L_gt_pad = L_gt_pad.to(device).float()
                speaker_ids = speaker_ids.to(device)
                speaker_emb_b = speaker_emb_b.to(device).to(dtype=torch.float32)
                gender_ids = gender_ids.to(device)
                emotion_ids = emotion_ids.to(device)
                if bool(getattr(args, "disable_gender_token", True)):
                    gender_ids = torch.zeros_like(gender_ids)

                mel_gt_bct_demo = _ensure_mel_bct(mel_pad)
                spk_256 = learned_spk_table(speaker_ids.clamp(0, int(num_speakers) - 1)).float()
                style_128_demo = learned_style_table(speaker_ids.clamp(0, int(num_speakers) - 1)).float()
                style_128_demo = _apply_emotion_style_conditioning(
                    style_128_demo,
                    emotion_ids,
                    enabled=bool(emotion_conditioning_enabled),
                    emotion_embed=emotion_embed,
                    emotion_to_style=emotion_to_style,
                    emotion_style_gate=emotion_style_gate,
                )
                spk_base_override = spk_adapter(spk_256)

                with torch.no_grad():
                    book_ids_t = _book_ids_to_tensor(book_ids)
                    x_tok, ids_full_enc, special_len, mem_after = encode_text_features_stateful(
                        model=model,
                        spk_embed=spk_embed,
                        gender_embed=gender_embed,
                        emotion_token_embed=emotion_token_embed,
                        tok_pad=tok_pad,
                        speaker_ids=speaker_ids,
                        gender_ids=gender_ids,
                        emotion_ids=emotion_ids,
                        book_ids=book_ids_t,
                        chunk_idx=chunk_idx,
                        device=device,
                        bridge=bridge,
                        bridge_cache=bridge_cache,
                        spk_vec_override=spk_base_override,
                        use_emotion_token=bool(emotion_conditioning_enabled),
                    )
                    flow_cond_demo = None

                    x_tok_dur_demo = x_tok

                    if torch.is_tensor(spk_base_override):
                        spk_base_demo = spk_base_override.to(device=x_tok.device, dtype=x_tok.dtype)
                    else:
                        if bool(getattr(args, "require_spk_override", True)):
                            raise RuntimeError(
                                "require_spk_override=True but spk_base_override is None/invalid in demo. "
                                "Speaker centroid bank is missing/invalid for this speaker_id."
                            )
                        spk_base_demo = spk_embed(speaker_ids).to(device=x_tok.device, dtype=x_tok.dtype)
                    spk_dur_demo = _zero_spk_cond_like(spk_base_demo)
                    prior_cond_demo = None

                    dur_pred, _, _ = _predict_dur(
                        model,
                        x_tok_dur_demo,
                        tok_pad,
                        special_len,
                        spk_embed,
                        speaker_ids,
                        flow_cond_demo,
                        spk_vec_override=spk_dur_demo,
                        require_spk_override=bool(getattr(args, "require_spk_override", True)),
                        style_vec=style_128_demo.to(device=x_tok_dur_demo.device, dtype=torch.float32)
                        if torch.is_tensor(style_128_demo)
                        else None,
                        dur_x0_mode=str(getattr(args, "dur_x0", "none")),
                        dur_x0_noise_scale=float(getattr(args, "dur_x0_noise_scale", 1.0)),
                        dur_prior_logs_min=float(getattr(args, "dur_prior_logs_min", -5.0)),
                        dur_prior_logs_max=float(getattr(args, "dur_prior_logs_max", 2.0)),
                        dur_prior_sigma_min=float(getattr(args, "dur_prior_sigma_min", 0.1)),
                        dur_sigma0_demo=0.0,
                        dur_flow_clip_sigma=float(getattr(args, "dur_flow_clip_sigma", 0.0)),
                        dur_flow_clip_abs_min=(
                            None
                            if (not math.isfinite(float(getattr(args, "dur_flow_clip_abs_min", float("nan")))))
                            else float(getattr(args, "dur_flow_clip_abs_min", float("nan")))
                        ),
                        dur_flow_clip_abs_max=(
                            None
                            if (not math.isfinite(float(getattr(args, "dur_flow_clip_abs_max", float("nan")))))
                            else float(getattr(args, "dur_flow_clip_abs_max", float("nan")))
                        ),
                        dur_flow_fix_total=bool(getattr(args, "dur_flow_fix_total", False)),
                        dur_flow_fix_total_mode=str(getattr(args, "dur_flow_fix_total_mode", "prior_mu")),
                    )
                    L_gt_full = F.pad(L_gt_pad, (int(special_len), 0), value=0.0)

                    ids_full = F.pad(tok_pad, (int(special_len), 0), value=PAD_ID)
                    dur_allowed = _build_dur_allowed_mask(ids_full, special_len)
                    L_gt_full = torch.where(dur_allowed, L_gt_full, torch.zeros_like(L_gt_full))
                    dur_pred = torch.where(dur_allowed, dur_pred, torch.zeros_like(dur_pred))
                    dur_for_prior = _select_dec_durations(L_gt_full, dur_pred, CONFIG["dec_dur_source_demo"])
                    dur_for_prior = torch.where(dur_allowed, dur_for_prior, torch.zeros_like(dur_for_prior))

                    t0_btc, mu_btc, _logs_btc, _ = prior_mu(
                        x_tok,
                        dur_for_prior,
                        cond=prior_cond_demo,
                        T_hint=None,
                        noise_scale=float(args.prior_noise_scale),
                        timbre_vec=spk_base_demo,
                    )
                    mu_bct = mu_btc.transpose(1, 2).contiguous()
                    x0 = t0_btc.transpose(1, 2).contiguous()

                    if bool(getattr(args, "mel_twopass", False)):
                        mel_out = sample_mel_flow_twopass(
                            mel_flow,
                            x0,
                            x_tok if bool(CONFIG["text_cross_attn"]) else None,
                            speaker_ids,
                            steps_first=int(getattr(args, "mel_twopass_steps_first", int(CONFIG["mel_flow_steps_demo"]))),
                            steps_second=int(getattr(args, "mel_twopass_steps_second", 0)),
                            t_noise=float(getattr(args, "mel_twopass_t_noise", 0.12)),
                            spk_vec_override=spk_base_demo,
                        )
                    else:
                        mel_out = sample_mel_flow(
                            mel_flow,
                            x0,
                            x_tok if bool(CONFIG["text_cross_attn"]) else None,
                            speaker_ids,
                            steps=int(CONFIG["mel_flow_steps_demo"]),
                            spk_vec_override=spk_base_demo,
                        )

                    # GT mel (dla porównania vocos/quality)
                    mel_gt_bct = _ensure_mel_bct(mel_pad)

                    # metryki mel (nie wymagają vocos)
                    T_common = int(min(mel_out.size(-1), mel_gt_bct.size(-1)))
                    mel_out_c = mel_out[..., :T_common].contiguous()
                    mu_c = mu_bct[..., :T_common].contiguous()
                    mel_gt_c = mel_gt_bct[..., :T_common].contiguous()
                    tmask = _make_tmask_from_Tlen(T_len.clamp_max(T_common), T_common).to(mel_gt_c.dtype)
                    demo_ref_l1 = float(_masked_l1_bct(mel_out_c, mel_gt_c, tmask).detach().cpu())
                    demo_mu_l1 = float(_masked_l1_bct(mu_c, mel_gt_c, tmask).detach().cpu())

                    demo_ref_l1_list.append(demo_ref_l1)
                    demo_mu_l1_list.append(demo_mu_l1)

                    if bi == 0:
                        saved_demo_payload = {
                            "mel_out": mel_out,
                            "mu_bct": mu_bct,
                            "mel_gt_bct": mel_gt_bct,
                            "tok_pad": tok_pad,
                            "T_len": T_len,
                            "ids_full": ids_full.detach(),
                            "dur_for_prior": dur_for_prior.detach(),
                            "speaker_ids": speaker_ids,
                            "spk_256": spk_256.detach().cpu(),
                            "T_common": T_common,
                        }

            demo_dir = out_dir / "demos" / f"ep{ep:04d}"
            demo_dir.mkdir(parents=True, exist_ok=True)

            demo_ref_l1_mean = float(sum(demo_ref_l1_list) / max(1, len(demo_ref_l1_list)))
            demo_mu_l1_mean = float(sum(demo_mu_l1_list) / max(1, len(demo_mu_l1_list)))
            metrics = {
                "epoch": int(ep),
                "demo_refined_mel_l1": demo_ref_l1_mean,
                "demo_mu_mel_l1": demo_mu_l1_mean,
                "demo_batches": int(demo_batches),
                "demo_refined_mel_l1_batches": demo_ref_l1_list,
                "demo_mu_mel_l1_batches": demo_mu_l1_list,
                "prior_layers": int(prior_layers),
                "prior_heads": int(prior_heads),
                "flow_layers": int(flow_layers),
                "flow_heads": int(flow_heads),
                "hidden_dim": int(hidden_dim),
                "encoder_heads": int(n_heads),
                "train_base": bool(train_base),
            }
            try:
                (demo_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
            except Exception:
                pass
            print(
                f"[ep{ep:04d}] demo_ref_l1={demo_ref_l1_mean:.4f} demo_mu_l1={demo_mu_l1_mean:.4f} "
                f"(batches={len(demo_ref_l1_list)})"
            )

            def _save_demo_wav(tag: str, wav_1t: torch.Tensor) -> None:
                try:
                    _save_wav_demo(demo_dir / f"{tag}.wav", wav_1t, 24000)
                except Exception as exc:
                    torch.save(wav_1t.detach().cpu(), str(demo_dir / f"{tag}.pt"))
                    try:
                        (demo_dir / f"{tag}.save_error.txt").write_text(str(exc) + "\n", encoding="utf-8")
                    except Exception:
                        pass

            def _decode_and_save(tag: str, mel_1ct: torch.Tensor) -> None:
                wav_1t = _vocos_decode_or_none(mel_1ct, tag=f"demo/{tag}")
                if wav_1t is None:
                    torch.save(mel_1ct.detach().cpu(), str(demo_dir / f"{tag}.mel.pt"))
                    return
                _save_demo_wav(tag, wav_1t)

            if saved_demo_payload is None:
                saved_demo_payload = {}

            mel_out = saved_demo_payload.get("mel_out")
            mu_bct = saved_demo_payload.get("mu_bct")
            mel_gt_bct = saved_demo_payload.get("mel_gt_bct")
            tok_pad = saved_demo_payload.get("tok_pad")
            T_len = saved_demo_payload.get("T_len")
            ids_full_demo = saved_demo_payload.get("ids_full")
            dur_for_prior_demo = saved_demo_payload.get("dur_for_prior")

            # jeśli z jakiegoś powodu nie udało się zapisać payloadu z pierwszego batcha, pomiń artefakty
            if mel_out is None or mu_bct is None or mel_gt_bct is None or tok_pad is None or T_len is None:
                mel_out = None

            # ile przykładów zapisać (z batcha)
            B = int(tok_pad.size(0)) if tok_pad is not None else 0
            k = max(1, min(B, int(getattr(args, "demo_count", 1)))) if B > 0 else 0
            if mel_out is None:
                k = 0

            # Zamiast zapisywać wiele plików sample00..sampleNN, sklej wszystko w 3 długie audia:
            #   dataset_gt_all / dataset_mu_all / dataset_refined_all
            # Dzięki temu łatwiej odsłuchać przejścia.
            # Small fixed gap between concatenated samples (applied equally to GT/mu/refined).
            # Keep it short so it doesn't mask true per-sample trailing silences.
            join_silence_sec = 0.05
            join_sr = 24000
            join_sil = torch.zeros((1, int(join_sr * join_silence_sec)), dtype=torch.float32)

            wavs_gt: List[torch.Tensor] = []
            wavs_mu: List[torch.Tensor] = []
            wavs_ref: List[torch.Tensor] = []
            mels_gt: List[torch.Tensor] = []
            mels_mu: List[torch.Tensor] = []
            mels_ref: List[torch.Tensor] = []
            tok_report: List[str] = []

            vocos_ok = (vocos is not None)
            for i in range(k):
                stem = f"sample{i:02d}"
                try:
                    ids = tok_pad[i].detach().cpu().tolist()
                    s = " ".join([ID2SYMBOL.get(int(t), "?") for t in ids if int(t) != PAD_ID])
                    tok_report.append(f"{stem}: {s}")
                except Exception:
                    pass

                T_gt = int(T_len[i].item()) if torch.is_tensor(T_len) else int(mel_gt_bct.size(-1))
                T_gt = int(max(1, T_gt))
                mel_gt_i = mel_gt_bct[i:i+1, :, :T_gt].contiguous()
                # IMPORTANT: crop mu/refined to GT length too, otherwise batch padding beyond T_len
                # becomes audible as unnaturally long trailing silence when concatenated.
                mu_i = mu_bct[i:i+1, :, :T_gt].contiguous()
                mel_ref_i = mel_out[i:i+1, :, :T_gt].contiguous()
                silence_mask_i = None
                if (
                    bool(getattr(args, "pause_force_digital_silence_demo", False))
                    and torch.is_tensor(ids_full_demo)
                    and torch.is_tensor(dur_for_prior_demo)
                ):
                    silence_mask_i = _pause_middle_frame_mask_from_durations(
                        ids_full=ids_full_demo[i : i + 1].to(device=dur_for_prior_demo.device),
                        dur_values=dur_for_prior_demo[i : i + 1],
                        T=int(T_gt),
                        edge_frames=int(getattr(args, "pause_edge_frames", 0)),
                    )

                if vocos is None:
                    mels_gt.append(mel_gt_i.detach().cpu())
                    mels_mu.append(mu_i.detach().cpu())
                    mels_ref.append(mel_ref_i.detach().cpu())
                else:
                    wav_gt = _vocos_decode_or_none(mel_gt_i, tag=f"demo/dataset_gt/{stem}")
                    wav_mu = _vocos_decode_or_none(mu_i, tag=f"demo/dataset_mu/{stem}")
                    wav_rf = _vocos_decode_or_none(mel_ref_i, tag=f"demo/dataset_refined/{stem}")
                    if (wav_gt is None) or (wav_mu is None) or (wav_rf is None):
                        vocos_ok = False
                        mels_gt.append(mel_gt_i.detach().cpu())
                        mels_mu.append(mu_i.detach().cpu())
                        mels_ref.append(mel_ref_i.detach().cpu())
                        continue
                    wav_gt = wav_gt.detach().cpu()
                    wav_mu = wav_mu.detach().cpu()
                    wav_rf = wav_rf.detach().cpu()
                    if bool(getattr(args, "pause_force_digital_silence_demo", False)):
                        wav_mu = _apply_digital_silence_to_wav(wav_mu, silence_mask_i)
                        wav_rf = _apply_digital_silence_to_wav(wav_rf, silence_mask_i)

                    if wavs_gt:
                        wavs_gt.append(join_sil)
                        wavs_mu.append(join_sil)
                        wavs_ref.append(join_sil)
                    wavs_gt.append(wav_gt)
                    wavs_mu.append(wav_mu)
                    wavs_ref.append(wav_rf)

            # tokens report
            if tok_report:
                try:
                    (demo_dir / "samples_tokens.txt").write_text("\n".join(tok_report) + "\n", encoding="utf-8")
                except Exception:
                    pass

            if (vocos is None) or (not vocos_ok):
                # fallback: save concatenated mels (no audio)
                try:
                    if mels_gt:
                        torch.save(torch.cat(mels_gt, dim=-1).contiguous(), str(demo_dir / "dataset_gt_all.mel.pt"))
                    if mels_mu:
                        torch.save(torch.cat(mels_mu, dim=-1).contiguous(), str(demo_dir / "dataset_mu_all.mel.pt"))
                    if mels_ref:
                        torch.save(torch.cat(mels_ref, dim=-1).contiguous(), str(demo_dir / "dataset_refined_all.mel.pt"))
                except Exception:
                    pass
            else:
                try:
                    if wavs_gt:
                        _save_wav_demo(demo_dir / "dataset_gt_all.wav", torch.cat(wavs_gt, dim=-1).contiguous(), 24000)
                    if wavs_mu:
                        _save_wav_demo(demo_dir / "dataset_mu_all.wav", torch.cat(wavs_mu, dim=-1).contiguous(), 24000)
                    if wavs_ref:
                        _save_wav_demo(demo_dir / "dataset_refined_all.wav", torch.cat(wavs_ref, dim=-1).contiguous(), 24000)
                except Exception:
                    pass

            # -------- compact speaker audit demo --------
            if bool(getattr(args, "demo_speaker_audit", True)):
                audit_dir = demo_dir / "speaker_audit"
                audit_dir.mkdir(parents=True, exist_ok=True)

                def _sanitize_demo_name(s: str) -> str:
                    s = str(s or "").strip()
                    if not s:
                        return "unknown"
                    s = __import__("re").sub(r"\s+", "_", s)
                    s = __import__("re").sub(r"[^0-9A-Za-zĄąĆćĘęŁłŃńÓóŚśŹźŻż_-]+", "", s)
                    return s[:120] if s else "unknown"

                def _ensure_sp_bounds_audit(s: str) -> str:
                    return _ensure_boundary_tokens(s)

                def _decode_and_save_to_audit(tag: str, mel_1ct: torch.Tensor) -> None:
                    wav_1t = _vocos_decode_or_none(mel_1ct, tag=f"speaker_audit/{tag}")
                    if wav_1t is None:
                        torch.save(mel_1ct.detach().cpu(), str(audit_dir / f"{tag}.mel.pt"))
                        return
                    try:
                        _save_wav_demo(audit_dir / f"{tag}.wav", wav_1t, 24000)
                    except Exception as exc:
                        torch.save(wav_1t.detach().cpu(), str(audit_dir / f"{tag}.pt"))
                        try:
                            (audit_dir / f"{tag}.save_error.txt").write_text(str(exc) + "\n", encoding="utf-8")
                        except Exception:
                            pass

                tok_audit = PLTokenizer(str(getattr(args, "vocab", "")).strip())
                audit_text = str(getattr(args, "demo_speaker_audit_text", "")).strip()
                if audit_text:
                    speaker_to_indices: Dict[int, List[int]] = {}
                    speaker_name_by_id: Dict[int, str] = {}
                    for ii, it in enumerate(ds_demo.items):
                        if not isinstance(it, dict):
                            continue
                        sid = it.get("speaker_id", it.get("speaker", None))
                        if sid is None:
                            continue
                        try:
                            sid_i = int(sid)
                        except Exception:
                            continue
                        speaker_to_indices.setdefault(sid_i, []).append(ii)
                        speaker_name_by_id.setdefault(
                            sid_i,
                            str(it.get("speaker_name", it.get("author", f"speaker_{sid_i}"))),
                        )

                    audit_ids = sorted(speaker_to_indices.keys())
                    audit_n = int(max(0, min(int(getattr(args, "demo_speaker_audit_count", 10)), len(audit_ids))))
                    if audit_n > 0:
                        rng = random.Random(int(getattr(args, "seed", 1234)) + int(ep))
                        audit_pick = rng.sample(audit_ids, audit_n)
                        manifest_lines = ["speaker_id,speaker_name,gt_file,pred_prefixonly_file,pred_flowspk_file"]
                        for rank, sid in enumerate(audit_pick, start=1):
                            idx_pool = speaker_to_indices.get(int(sid), [])
                            if not idx_pool:
                                continue
                            sample_idx = idx_pool[0]
                            item = ds_demo[sample_idx]
                            mel_gt_i = _ensure_mel_bct(item["mel"]).to(device=device, dtype=torch.float32)
                            T_gt_i = int(item["T_mel"]) if "T_mel" in item else int(mel_gt_i.size(-1))
                            mel_gt_i = mel_gt_i[:, :, : max(1, T_gt_i)].contiguous()

                            spk_vec_256 = _load_dataset_speaker_centroid(int(sid)).view(1, -1).to(device=device, dtype=torch.float32)
                            spk_hidden = spk_adapter(spk_vec_256).to(dtype=torch.float32)
                            speaker_ids1 = torch.tensor([int(sid)], device=device, dtype=torch.long)
                            gender_ids1 = _gender_ids_from_speaker_ids(speaker_ids1, device=device)
                            if bool(getattr(args, "disable_gender_token", True)):
                                gender_ids1 = torch.zeros_like(gender_ids1)
                            tok_ids = tok_audit.encode(_ensure_sp_bounds_audit(audit_text))
                            tok_pad1 = torch.tensor([tok_ids], dtype=torch.long, device=device)
                            x_tok1, ids_full1, special_len1 = encode_text_features(
                                model=model,
                                spk_embed=spk_embed,
                                gender_embed=gender_embed,
                                emotion_token_embed=emotion_token_embed,
                                tok_pad=tok_pad1,
                                speaker_ids=speaker_ids1,
                                gender_ids=gender_ids1,
                                emotion_ids=torch.zeros_like(speaker_ids1),
                                device=device,
                                spk_vec_override=spk_hidden,
                                require_spk_override=bool(getattr(args, "require_spk_override", True)),
                                use_emotion_token=bool(emotion_conditioning_enabled),
                            )

                            demo_dur_source = str(getattr(args, "demo_dur_source", "flow")).lower().strip()
                            if demo_dur_source in ("prior_mu", "prior_sample"):
                                dur_pred1, ids_full1, _ = _predict_dur_prior_direct(
                                    model,
                                    x_tok1,
                                    tok_pad1,
                                    special_len1,
                                    source=demo_dur_source,
                                    noise_scale=float(getattr(args, "demo_dur_noise_scale", 0.0)),
                                    dur_prior_logs_min=float(getattr(args, "dur_prior_logs_min", -5.0)),
                                    dur_prior_logs_max=float(getattr(args, "dur_prior_logs_max", 2.0)),
                                    dur_prior_sigma_min=float(getattr(args, "dur_prior_sigma_min", 0.1)),
                                )
                            else:
                                dur_pred1, ids_full1, _ = _predict_dur(
                                    model,
                                    x_tok1,
                                    tok_pad1,
                                    special_len1,
                                    spk_embed,
                                    speaker_ids1,
                                    None,
                                    spk_vec_override=_zero_spk_cond_like(spk_hidden.to(device=x_tok1.device, dtype=x_tok1.dtype)),
                                    require_spk_override=bool(getattr(args, "require_spk_override", True)),
                                    style_vec=None,
                                    dur_x0_mode=str(getattr(args, "dur_x0", "none")),
                                    dur_x0_noise_scale=float(getattr(args, "dur_x0_noise_scale", 1.0)),
                                    dur_prior_logs_min=float(getattr(args, "dur_prior_logs_min", -5.0)),
                                    dur_prior_logs_max=float(getattr(args, "dur_prior_logs_max", 2.0)),
                                    dur_prior_sigma_min=float(getattr(args, "dur_prior_sigma_min", 0.1)),
                                    dur_sigma0_demo=0.0,
                                )
                            dur_allowed1 = _build_dur_allowed_mask(ids_full1, special_len1)
                            dur_for_prior1 = torch.where(dur_allowed1, dur_pred1, torch.zeros_like(dur_pred1))
                            sp_mask_tok1 = _pause_mask_from_ids(ids_full1)
                            t0_btc1, _mu_btc1, _logs_btc1, _ = prior_mu(
                                x_tok1,
                                dur_for_prior1,
                                cond=None,
                                T_hint=None,
                                noise_scale=float(args.prior_noise_scale),
                                sp_mask_tok=sp_mask_tok1,
                                timbre_vec=spk_hidden.to(device=x_tok1.device, dtype=x_tok1.dtype),
                            )
                            x01 = t0_btc1.transpose(1, 2).contiguous()
                            if bool(getattr(args, "mel_twopass", False)):
                                mel_pred_i = sample_mel_flow_twopass(
                                    mel_flow,
                                    x01,
                                    x_tok1 if bool(CONFIG["text_cross_attn"]) else None,
                                    speaker_ids1,
                                    steps_first=int(getattr(args, "mel_twopass_steps_first", int(CONFIG["mel_flow_steps_demo"]))),
                                    steps_second=int(getattr(args, "mel_twopass_steps_second", 0)),
                                    t_noise=float(getattr(args, "mel_twopass_t_noise", 0.12)),
                                    spk_vec_override=spk_hidden.to(device=x01.device, dtype=x01.dtype),
                                )
                                mel_pred_flowspk_i = sample_mel_flow_twopass(
                                    mel_flow,
                                    x01,
                                    x_tok1 if bool(CONFIG["text_cross_attn"]) else None,
                                    speaker_ids1,
                                    steps_first=int(getattr(args, "mel_twopass_steps_first", int(CONFIG["mel_flow_steps_demo"]))),
                                    steps_second=int(getattr(args, "mel_twopass_steps_second", 0)),
                                    t_noise=float(getattr(args, "mel_twopass_t_noise", 0.12)),
                                    spk_vec_override=spk_hidden.to(device=x01.device, dtype=x01.dtype),
                                )
                            else:
                                mel_pred_i = sample_mel_flow(
                                    mel_flow,
                                    x01,
                                    x_tok1 if bool(CONFIG["text_cross_attn"]) else None,
                                    speaker_ids1,
                                    steps=int(CONFIG["mel_flow_steps_demo"]),
                                    spk_vec_override=spk_hidden.to(device=x01.device, dtype=x01.dtype),
                                )
                                mel_pred_flowspk_i = sample_mel_flow(
                                    mel_flow,
                                    x01,
                                    x_tok1 if bool(CONFIG["text_cross_attn"]) else None,
                                    speaker_ids1,
                                    steps=int(CONFIG["mel_flow_steps_demo"]),
                                    spk_vec_override=spk_hidden.to(device=x01.device, dtype=x01.dtype),
                                )

                            speaker_name = speaker_name_by_id.get(int(sid), f"speaker_{int(sid)}")
                            safe_name = _sanitize_demo_name(speaker_name)
                            stem = f"{rank:02d}__{safe_name}__sid{int(sid):04d}"
                            _decode_and_save_to_audit(f"{stem}__gt", mel_gt_i[:1].contiguous())
                            _decode_and_save_to_audit(f"{stem}__pred_prefixonly", mel_pred_i[:1].contiguous())
                            _decode_and_save_to_audit(f"{stem}__pred_flowspk", mel_pred_flowspk_i[:1].contiguous())
                            manifest_lines.append(
                                f'{int(sid)},"{speaker_name}",{stem}__gt.wav,{stem}__pred_prefixonly.wav,{stem}__pred_flowspk.wav'
                            )
                        try:
                            (audit_dir / "audit_text.txt").write_text(audit_text + "\n", encoding="utf-8")
                            (audit_dir / "manifest.csv").write_text("\n".join(manifest_lines) + "\n", encoding="utf-8")
                        except Exception:
                            pass

            # -------- text demo (dur_pred path) --------
            if bool(getattr(args, "demo_long", True)):
                text_demo_dir = demo_dir / "text_demo"
                text_demo_dir.mkdir(parents=True, exist_ok=True)

                # Tokenizer is needed only for text demos.
                tok = PLTokenizer(str(getattr(args, "vocab", "")).strip())

                spk_text_base_override: Optional[torch.Tensor] = None  # [1, D] in model hidden dim

                # speaker: stały dla całego text_demo (żeby kolejne fragmenty nie były "innym lektorem").
                forced = int(getattr(args, "demo_long_speaker_id", 0))
                preferred_demo_lang = str(getattr(args, "demo_long_speaker_lang", "en")).lower().strip()

                def _item_lang_for_text_demo(it: Dict[str, Any]) -> str:
                    lang = str(it.get("lang", it.get("language", "")) or "").lower().strip()
                    if lang in ("eng", "english"):
                        return "en"
                    if lang in ("pol", "polish"):
                        return "pl"
                    txt = str(it.get("text", "") or "").strip().lower()
                    if txt.startswith("<en>"):
                        return "en"
                    if txt.startswith("<pl>"):
                        return "pl"
                    return lang

                def _find_text_demo_ref() -> Tuple[Optional[Any], Optional[int], Optional[int]]:
                    pools = [ds_demo, ds_train]
                    langs = [preferred_demo_lang] if preferred_demo_lang in ("en", "pl") else []
                    langs += [x for x in ("en", "pl") if x not in langs]
                    for lang_want in langs:
                        for ds_obj in pools:
                            try:
                                meta_items = list(getattr(ds_obj, "items", []))
                            except Exception:
                                meta_items = []
                            for jj, meta in enumerate(meta_items):
                                if not isinstance(meta, dict):
                                    continue
                                if _item_lang_for_text_demo(meta) != str(lang_want):
                                    continue
                                try:
                                    sid_j = int(meta.get("speaker_id", meta.get("speaker", 0)))
                                except Exception:
                                    continue
                                if forced >= 0 and int(sid_j) != int(forced):
                                    continue
                                return ds_obj, int(jj), int(sid_j)
                    return None, None, None

                text_demo_ref_ds, text_demo_ref_idx, text_demo_ref_sid = _find_text_demo_ref()
                spk0 = 0
                if forced >= 0:
                    spk0 = int(forced)
                elif text_demo_ref_sid is not None:
                    spk0 = int(text_demo_ref_sid)
                else:
                    try:
                        if saved_demo_payload is not None:
                            spk0 = int(saved_demo_payload["speaker_ids"][0].detach().cpu().item())
                    except Exception:
                        spk0 = 0
                if (
                    str(getattr(args, "speaker_vector_source", "")).lower().strip() == "learned_voice"
                    and int(spk0) == 0
                    and text_demo_ref_sid is not None
                ):
                    # In learned_voice mode the default 0 should not accidentally select a non-dataset/Zosia-like
                    # external reference path. Use the selected dataset speaker unless the user explicitly forces
                    # another id that exists.
                    spk0 = int(text_demo_ref_sid)
                if int(spk0) == 0:
                    if speaker_embeds_pt_by_id:
                        spk0 = int(sorted(speaker_embeds_pt_by_id.keys())[0])
                    elif speaker_chunk_emb_by_id:
                        spk0 = int(sorted(speaker_chunk_emb_by_id.keys())[0])
                speaker_ids1 = torch.tensor([int(spk0)], device=device, dtype=torch.long)
                gender_ids1 = _gender_ids_from_speaker_ids(speaker_ids1, device=device)
                if bool(getattr(args, "disable_gender_token", True)):
                    gender_ids1 = torch.zeros_like(gender_ids1)
                spk_text_256: Optional[torch.Tensor] = None
                text_demo_style_128: Optional[torch.Tensor] = None

                def _text_demo_decode_fn(mel_1ct: torch.Tensor, tag: str) -> Optional[torch.Tensor]:
                    return _vocos_decode_or_none(mel_1ct, tag=tag)

                def _text_demo_silence_fn(
                    wav_1t: torch.Tensor,
                    silence_mask: Optional[torch.Tensor],
                ) -> torch.Tensor:
                    if bool(getattr(args, "pause_force_digital_silence_demo", False)):
                        return _apply_digital_silence_to_wav(wav_1t, silence_mask)
                    return wav_1t

                def _decode_and_save_to(
                    dir_path: Path,
                    tag: str,
                    mel_1ct: torch.Tensor,
                    silence_mask: Optional[torch.Tensor] = None,
                ) -> None:
                    decode_and_save_mel(
                        dir_path=dir_path,
                        tag=tag,
                        mel_1ct=mel_1ct,
                        decode_fn=_text_demo_decode_fn,
                        silence_fn=_text_demo_silence_fn,
                        silence_mask=silence_mask,
                    )

                if str(getattr(args, "speaker_vector_source", "")).lower().strip() == "learned_voice":
                    sid_tab = int(max(0, min(int(num_speakers) - 1, int(spk0))))
                    sid_tensor = torch.tensor([sid_tab], device=device, dtype=torch.long)
                    spk_text_256 = learned_spk_table(sid_tensor).detach().float()
                    text_demo_style_128 = learned_style_table(sid_tensor).detach().float()
                    spk_text_base_override = spk_adapter(spk_text_256.to(device=device)).to(dtype=torch.float32)
                    print(f"🧬 text_demo learned_voice: speaker_id={sid_tab} from learned tables")
                    if text_demo_ref_ds is not None and text_demo_ref_idx is not None:
                        try:
                            ref_item = text_demo_ref_ds[int(text_demo_ref_idx)]
                            mel_ref_demo = _ensure_mel_bct(ref_item["mel"]).to(device=device, dtype=torch.float32)
                            T_ref_demo = int(ref_item["T_mel"]) if "T_mel" in ref_item else int(mel_ref_demo.size(-1))
                            mel_ref_demo = mel_ref_demo[:, :, : max(1, T_ref_demo)].contiguous()
                            _decode_and_save_to(text_demo_dir, f"reference_original_dataset_sid{sid_tab}", mel_ref_demo[:1].contiguous(), None)
                            (text_demo_dir / "reference_original_info.txt").write_text(
                                "\n".join(
                                    [
                                        f"speaker_id={sid_tab}",
                                        f"frames={int(mel_ref_demo.size(-1))}",
                                        f"seconds={float(mel_ref_demo.size(-1)) * float(_SECS_PER_FRAME):.3f}",
                                        "source=learned_voice_dataset_reference_audio_only",
                                        "conditioning=learned_spk_table+learned_style_table",
                                    ]
                                )
                                + "\n",
                                encoding="utf-8",
                            )
                        except Exception as exc:
                            print(f"⚠️ text_demo learned_voice: failed to save dataset reference audio: {exc}")

                def _demo_ref_audio_paths() -> List[Path]:
                    return collect_reference_audio_paths(
                        str(getattr(args, "demo_ref_wav", "") or ""),
                        str(getattr(args, "demo_ref_wav_glob", "") or ""),
                    )

                def _decode_chunks_and_save_to(
                    dir_path: Path,
                    tag: str,
                    mel_chunks_1ct: List[torch.Tensor],
                    silence_masks: Optional[List[Optional[torch.Tensor]]] = None,
                ) -> None:
                    decode_chunks_and_save(
                        dir_path=dir_path,
                        tag=tag,
                        mel_chunks_1ct=mel_chunks_1ct,
                        decode_fn=_text_demo_decode_fn,
                        silence_fn=_text_demo_silence_fn,
                        silence_masks=silence_masks,
                    )

                def _use_external_text_demo_reference(paths: List[Path]) -> bool:
                    nonlocal spk_text_256, text_demo_style_128, spk_text_base_override
                    if not paths:
                        return False
                    if speaker_encoder is None and not bool(getattr(mel_flow, "spk_style_ref_ready", False)):
                        raise RuntimeError("--demo-ref-wav requires speaker_encoder or loaded frozen dualhead encoder.")
                    z_spk_list: List[torch.Tensor] = []
                    z_style_list: List[torch.Tensor] = []
                    mel_refs: List[torch.Tensor] = []
                    max_sec = float(getattr(args, "demo_ref_max_sec", 10.0))
                    for p in paths:
                        mel_ref_i, t_len_i = _ref_wav_to_mel(str(p), max_sec=max_sec)
                        mel_refs.append(mel_ref_i.detach().to(device=device, dtype=torch.float32))
                        mask_i = _make_tmask_from_Tlen(t_len_i.to(device), int(mel_ref_i.size(-1))).squeeze(1).to(device=device, dtype=torch.bool)
                        with torch.no_grad():
                            if speaker_encoder is not None:
                                enc_dev = next(speaker_encoder.parameters()).device
                                z_i, z_style_i = speaker_encoder(  # type: ignore[misc,call-arg]
                                    mel_ref_i.to(device=enc_dev, dtype=torch.float32),
                                    mask_bt=mask_i.to(device=enc_dev),
                                )
                            else:
                                z_i, z_style_i = mel_flow.encode_ref_dual(  # type: ignore[attr-defined]
                                    mel_ref_i.float(),
                                    mask_bt=mask_i,
                                )
                        z_spk_list.append(z_i.detach().float().to(device=device))
                        z_style_list.append(z_style_i.detach().float().to(device=device))

                    z_demo = torch.stack(z_spk_list, dim=0).mean(dim=0)
                    z_demo = z_demo / z_demo.norm(dim=-1, keepdim=True).clamp_min(1e-12)
                    z_style_demo = torch.stack(z_style_list, dim=0).mean(dim=0)
                    spk_text_256 = z_demo.detach().float()
                    text_demo_style_128 = z_style_demo.detach().float()
                    spk_text_base_override = spk_adapter(z_demo.to(device=device)).to(dtype=torch.float32)

                    mel_ref_demo = torch.cat(mel_refs, dim=-1).contiguous()
                    max_ref_frames = int(round(30.0 / float(_SECS_PER_FRAME)))
                    if int(mel_ref_demo.size(-1)) > max_ref_frames:
                        mel_ref_demo = mel_ref_demo[:, :, :max_ref_frames].contiguous()
                    T_ref_demo = int(mel_ref_demo.size(-1))
                    _decode_and_save_to(text_demo_dir, "reference_original_external", mel_ref_demo[:1].contiguous(), None)
                    try:
                        (text_demo_dir / "reference_original_info.txt").write_text(
                            "\n".join(
                                [
                                    "source=external_demo_ref_wav",
                                    f"files={len(paths)}",
                                    f"frames={int(T_ref_demo)}",
                                    f"seconds={float(T_ref_demo) * float(_SECS_PER_FRAME):.3f}",
                                    *[f"file={str(p)}" for p in paths],
                                ]
                            )
                            + "\n",
                            encoding="utf-8",
                        )
                    except Exception:
                        pass
                    return True

                # Prefer a real EN/PL GT sample selected above: it gives text_demo an explicit original voice sample
                # and avoids binding the demo voice to whatever happened to be first in the validation batch.
                if spk_text_base_override is None:
                    external_ref_paths = _demo_ref_audio_paths()
                    if external_ref_paths:
                        try:
                            _use_external_text_demo_reference(external_ref_paths)
                        except Exception as exc:
                            print(f"⚠️ text_demo: failed to use external reference audio: {exc}")
                            spk_text_base_override = None
                            spk_text_256 = None
                            text_demo_style_128 = None

                if spk_text_base_override is None:
                    if (
                        text_demo_ref_ds is not None
                        and text_demo_ref_idx is not None
                        and (
                            speaker_encoder is not None
                            or (
                                bool(getattr(mel_flow, "spk_style_ref_ready", False))
                                and getattr(mel_flow, "spk_style_ref", None) is not None
                            )
                        )
                    ):
                        try:
                            ref_item = text_demo_ref_ds[int(text_demo_ref_idx)]
                            mel_ref_demo = _ensure_mel_bct(ref_item["mel"]).to(device=device, dtype=torch.float32)
                            T_ref_demo = int(ref_item["T_mel"]) if "T_mel" in ref_item else int(mel_ref_demo.size(-1))
                            mel_ref_demo = mel_ref_demo[:, :, : max(1, T_ref_demo)].contiguous()
                            mask_ref_demo = torch.ones((1, int(mel_ref_demo.size(-1))), device=device, dtype=torch.bool)
                            with torch.no_grad():
                                if speaker_encoder is not None:
                                    enc_dev = next(speaker_encoder.parameters()).device
                                    z_demo, z_style_demo = speaker_encoder(  # type: ignore[misc,call-arg]
                                        mel_ref_demo.to(device=enc_dev, dtype=torch.float32),
                                        mask_bt=mask_ref_demo.to(device=enc_dev),
                                    )
                                else:
                                    z_demo, z_style_demo = mel_flow.encode_ref_dual(  # type: ignore[attr-defined]
                                        mel_ref_demo.float(),
                                        mask_bt=mask_ref_demo,
                                    )
                                z_demo = z_demo.detach().float()
                                text_demo_style_128 = z_style_demo.detach().float().to(device=device)
                                spk_text_256 = z_demo
                                spk_text_base_override = spk_adapter(z_demo.to(device=device)).to(dtype=torch.float32)
                            ref_meta = {}
                            try:
                                ref_meta_items = list(getattr(text_demo_ref_ds, "items", []))
                                if 0 <= int(text_demo_ref_idx) < len(ref_meta_items):
                                    ref_meta = ref_meta_items[int(text_demo_ref_idx)] if isinstance(ref_meta_items[int(text_demo_ref_idx)], dict) else {}
                            except Exception:
                                ref_meta = {}
                            ref_lang = _item_lang_for_text_demo(ref_meta)
                            ref_name = str(ref_meta.get("speaker_name", ref_meta.get("author", f"speaker_{int(spk0)}")))
                            ref_book = str(ref_meta.get("book_id", ""))
                            ref_utt = str(ref_meta.get("utt_id", ""))
                            _decode_and_save_to(text_demo_dir, f"reference_original_{ref_lang}_sid{int(spk0)}", mel_ref_demo[:1].contiguous(), None)
                            try:
                                (text_demo_dir / "reference_original_info.txt").write_text(
                                    "\n".join(
                                        [
                                            f"speaker_id={int(spk0)}",
                                            f"speaker_name={ref_name}",
                                            f"lang={ref_lang}",
                                            f"book_id={ref_book}",
                                            f"utt_id={ref_utt}",
                                            f"frames={int(mel_ref_demo.size(-1))}",
                                            f"seconds={float(mel_ref_demo.size(-1)) * float(_SECS_PER_FRAME):.3f}",
                                            "source=dualhead_text_demo_reference",
                                        ]
                                    )
                                    + "\n",
                                    encoding="utf-8",
                                )
                            except Exception:
                                pass
                        except Exception as exc:
                            print(f"⚠️ text_demo: failed to use preferred {preferred_demo_lang} reference sample: {exc}")
                            spk_text_base_override = None
                            spk_text_256 = None

                # Fallback: use GT-dualhead speaker vector from the first val sample, or dataset centroid.
                if spk_text_base_override is None:
                    if str(getattr(args, "speaker_vector_source", "dataset_centroid")).lower().strip() == "gt_dualhead":
                        try:
                            spk_demo_b = saved_demo_payload.get("spk_256") if isinstance(saved_demo_payload, dict) else None
                            if torch.is_tensor(spk_demo_b) and int(spk_demo_b.size(0)) > 0:
                                z_demo = spk_demo_b[:1].to(device=device, dtype=torch.float32)
                                spk_text_256 = z_demo
                                with torch.no_grad():
                                    spk_text_base_override = spk_adapter(z_demo).to(dtype=torch.float32)
                        except Exception:
                            spk_text_base_override = None
                if spk_text_base_override is None:
                    # Robustness: user may force a speaker_id that doesn't exist in the current dataset mapping.
                    # In that case, fall back to the first available speaker_id to avoid crashing training at demo time.
                    sid_req = int(spk0)
                    if sid_req not in speaker_embeds_pt_by_id:
                        avail = sorted(speaker_embeds_pt_by_id.keys())
                        if avail:
                            sid_fallback = int(avail[0])
                            print(
                                f"⚠️ text_demo: requested demo_long_speaker_id={sid_req} not found in dataset centroids; "
                                f"falling back to speaker_id={sid_fallback}."
                            )
                            spk0 = sid_fallback
                            speaker_ids1 = torch.tensor([int(spk0)], device=device, dtype=torch.long)
                            gender_ids1 = _gender_ids_from_speaker_ids(speaker_ids1, device=device)
                            if bool(getattr(args, "disable_gender_token", True)):
                                gender_ids1 = torch.zeros_like(gender_ids1)
                        else:
                            print("⚠️ text_demo: no dataset centroids available; skipping text_demo.")
                            continue
                    try:
                        c = _load_dataset_speaker_centroid(int(spk0)).view(1, -1).to(device=device, dtype=torch.float32)
                        spk_text_256 = c
                        with torch.no_grad():
                            spk_text_base_override = spk_adapter(c).to(dtype=torch.float32)
                    except Exception as exc:
                        print(f"⚠️ text_demo: cannot load centroid for speaker_id={int(spk0)}; skipping text_demo. ({exc})")
                        continue

                if bool(getattr(args, "require_spk_override", True)) and (spk_text_base_override is None):
                    raise RuntimeError(
                        "require_spk_override=True but could not get speaker override for text_demo. "
                        f"demo_long_speaker_id={int(spk0)}. "
                        "Dataset speaker_embeds mapping has no centroid for this speaker_id."
                    )

                demo_dur_steps = int(getattr(args, "demo_dur_steps", -1))
                demo_dur_steps = None if demo_dur_steps <= 0 else int(demo_dur_steps)
                demo_dur_noise = float(getattr(args, "demo_dur_noise_scale", -1.0))
                demo_dur_noise = None if demo_dur_noise < 0.0 else float(demo_dur_noise)
                # Prefix-continuity state for text demos (per book_tag): previous chunk mel.
                _td_prev: Dict[str, torch.Tensor] = {}
                _td_dur_hc: Dict[str, tuple] = {}
                carry_duration_state = bool(getattr(args, "duration_state_carry", True))

                def _ensure_boundary_textdemo(s: str) -> str:
                    return _ensure_boundary_tokens(s, continuation_out=not _looks_sentence_final(s))

                def _gen_text_one(
                    *,
                    txt: str,
                    book_tag: str,
                    chunk_i: int,
                    tag: str,
                    emotion_group: str = "neutral",
                    save_audio: bool = True,
                    save_debug: bool = True,
                ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
                    # Reset prefix continuity at the first chunk of a story/dialog.
                    if int(chunk_i) <= 0:
                        _td_prev.pop(str(book_tag), None)
                        _td_dur_hc.pop(str(book_tag), None)
                    tok_ids = tok.encode(_ensure_boundary_textdemo(txt))
                    tok_pad1 = torch.tensor([tok_ids], dtype=torch.long, device=device)
                    book_id = int(zlib.crc32(str(book_tag).encode("utf-8"))) & 0x7FFFFFFF
                    book_ids1 = torch.tensor([int(book_id)], dtype=torch.long)  # CPU
                    chunk_idx1 = torch.tensor([int(chunk_i)], dtype=torch.long)  # CPU
                    emo_id_for_demo = int(EMOTION_GROUP_TO_ID.get(str(emotion_group).strip().lower(), 0))
                    emo_ids1_for_demo = torch.tensor([emo_id_for_demo], device=device, dtype=torch.long)

                    x_tok1, _ids_full_enc1, special_len1, mem_after1 = encode_text_features_stateful(
                        model=model,
                        spk_embed=spk_embed,
                        gender_embed=gender_embed,
                        emotion_token_embed=emotion_token_embed,
                        tok_pad=tok_pad1,
                        speaker_ids=speaker_ids1,
                        gender_ids=gender_ids1,
                        emotion_ids=emo_ids1_for_demo,
                        book_ids=book_ids1,
                        chunk_idx=chunk_idx1,
                        device=device,
                        bridge=bridge,
                        bridge_cache=bridge_cache,
                        spk_vec_override=spk_text_base_override,
                        require_spk_override=bool(getattr(args, "require_spk_override", True)),
                        use_emotion_token=bool(emotion_conditioning_enabled),
                    )
                    flow_cond_demo1 = None

                    x_tok1_dur = x_tok1

                    if spk_text_base_override is None:
                        if bool(getattr(args, "require_spk_override", True)):
                            raise RuntimeError(
                                "require_spk_override=True but spk_text_base_override=None in text_demo. "
                                "Ensure dataset-json has speaker_embeds for the chosen speaker_id."
                            )
                        spk_base1 = spk_embed(speaker_ids1).to(device=x_tok1.device, dtype=x_tok1.dtype)
                    else:
                        spk_base1 = spk_text_base_override.to(device=x_tok1.device, dtype=x_tok1.dtype)
                    spk_dur1 = _zero_spk_cond_like(spk_base1)
                    prior_cond1 = None
                    text_demo_style_for_emotion = text_demo_style_128
                    if bool(emotion_conditioning_enabled):
                        text_demo_style_for_emotion = _apply_emotion_style_conditioning(
                            text_demo_style_128,
                            emo_ids1_for_demo,
                            enabled=True,
                            emotion_embed=emotion_embed,
                            emotion_to_style=emotion_to_style,
                            emotion_style_gate=emotion_style_gate,
                        )

                    demo_dur_source = str(getattr(args, "demo_dur_source", "flow")).lower().strip()
                    initial_dur_hc = _td_dur_hc.get(str(book_tag), None) if carry_duration_state else None
                    if demo_dur_source in ("prior_mu", "prior_sample"):
                        dur_pred1, ids_full1, _ = _predict_dur_prior_direct(
                            model,
                            x_tok1_dur,
                            tok_pad1,
                            special_len1,
                            source=demo_dur_source,
                            noise_scale=float(0.0 if demo_dur_noise is None else demo_dur_noise),
                            dur_prior_logs_min=float(getattr(args, "dur_prior_logs_min", -5.0)),
                            dur_prior_logs_max=float(getattr(args, "dur_prior_logs_max", 2.0)),
                            dur_prior_sigma_min=float(getattr(args, "dur_prior_sigma_min", 0.1)),
                            style_vec=text_demo_style_for_emotion.to(device=x_tok1_dur.device, dtype=torch.float32) if torch.is_tensor(text_demo_style_for_emotion) else None,
                            initial_hc=initial_dur_hc,
                        )
                        if carry_duration_state:
                            _td_dur_hc[str(book_tag)] = getattr(_predict_dur_prior_direct, "_last_hc", None)
                        else:
                            _td_dur_hc.pop(str(book_tag), None)
                    else:
                        dur_pred1, ids_full1, _ = _predict_dur(
                            model,
                            x_tok1_dur,
                            tok_pad1,
                            special_len1,
                            spk_embed,
                            speaker_ids1,
                            flow_cond_demo1,
                            spk_vec_override=spk_dur1,
                            require_spk_override=bool(getattr(args, "require_spk_override", True)),
                            style_vec=text_demo_style_for_emotion.to(device=x_tok1_dur.device, dtype=torch.float32) if torch.is_tensor(text_demo_style_for_emotion) else None,
                            dur_x0_mode=str(getattr(args, "dur_x0", "none")),
                            dur_x0_noise_scale=float(getattr(args, "dur_x0_noise_scale", 1.0)),
                            dur_prior_logs_min=float(getattr(args, "dur_prior_logs_min", -5.0)),
                            dur_prior_logs_max=float(getattr(args, "dur_prior_logs_max", 2.0)),
                            dur_prior_sigma_min=float(getattr(args, "dur_prior_sigma_min", 0.1)),
                            dur_sigma0_demo=0.0,
                            steps_override=demo_dur_steps,
                            noise_scale_override=demo_dur_noise,
                            dur_flow_clip_sigma=float(getattr(args, "dur_flow_clip_sigma", 0.0)),
                            dur_flow_clip_abs_min=(
                                None
                                if (not math.isfinite(float(getattr(args, "dur_flow_clip_abs_min", float("nan")))))
                                else float(getattr(args, "dur_flow_clip_abs_min", float("nan")))
                            ),
                            dur_flow_clip_abs_max=(
                                None
                                if (not math.isfinite(float(getattr(args, "dur_flow_clip_abs_max", float("nan")))))
                                else float(getattr(args, "dur_flow_clip_abs_max", float("nan")))
                            ),
                            dur_flow_fix_total=bool(getattr(args, "dur_flow_fix_total", False)),
                            dur_flow_fix_total_mode=str(getattr(args, "dur_flow_fix_total_mode", "prior_mu")),
                            initial_hc=initial_dur_hc,
                        )
                        if carry_duration_state:
                            _td_dur_hc[str(book_tag)] = getattr(_predict_dur, "_last_hc", None)
                        else:
                            _td_dur_hc.pop(str(book_tag), None)
                    dur_allowed1 = _build_dur_allowed_mask(ids_full1, special_len1)
                    dur_for_prior1 = torch.where(dur_allowed1, dur_pred1, torch.zeros_like(dur_pred1))
                    sp_mask_tok1 = _pause_mask_from_ids(ids_full1)

                    # text_demo: opcjonalna korekta duracji (skala + minimalna cisza dla pauz)
                    dur_scale = float(getattr(args, "text_demo_dur_scale", 1.0))
                    if dur_scale != 1.0:
                        dur_for_prior1 = dur_for_prior1 * float(dur_scale)
                    sp_min = float(getattr(args, "dur_pred_sp_min_frames", 1.0))
                    if sp_min > 0:
                        pause_mask = _pause_mask_from_ids(ids_full1)
                        dur_for_prior1 = torch.where(
                            pause_mask,
                            dur_for_prior1.clamp_min(float(sp_min)),
                            dur_for_prior1,
                        )
                    sp_max = float(getattr(args, "dur_pred_sp_max_frames", 0.0))
                    text_min = float(getattr(args, "dur_pred_text_min_frames", 0.0))
                    pause_mask = _pause_mask_from_ids(ids_full1)
                    if text_min > 0:
                        text_dur_mask = dur_allowed1 & (~pause_mask)
                        dur_for_prior1 = torch.where(
                            text_dur_mask,
                            dur_for_prior1.clamp_min(float(text_min)),
                            dur_for_prior1,
                        )
                    if sp_max > 0:
                        dur_for_prior1 = torch.where(
                            pause_mask,
                            dur_for_prior1.clamp_max(float(sp_max)),
                            dur_for_prior1,
                        )

                    prior_noise_scale = float(getattr(args, "prior_noise_scale", 1.0))
                    t0_btc1, mu_btc1, _logs_btc1, _ = prior_mu(
                        x_tok1,
                        dur_for_prior1,
                        cond=prior_cond1,
                        T_hint=None,
                        noise_scale=float(prior_noise_scale),
                        sp_mask_tok=sp_mask_tok1,
                        timbre_vec=spk_base1,
                    )
                    x01 = t0_btc1.transpose(1, 2).contiguous()
                    mu1 = mu_btc1.transpose(1, 2).contiguous()
                    silence_mask1 = None
                    if bool(getattr(args, "pause_force_digital_silence_demo", False)):
                        silence_mask1 = _pause_middle_frame_mask_from_durations(
                            ids_full=ids_full1,
                            dur_values=dur_for_prior1,
                            T=int(x01.size(-1)),
                            edge_frames=int(getattr(args, "pause_edge_frames", 0)),
                        )

                    # ---- Prefix continuity (decoder) for multi-chunk text demos ----
                    prefix_k = 0
                    prefix_tail = None
                    try:
                        prev_mel = _td_prev.get(str(book_tag), None)
                        if torch.is_tensor(prev_mel):
                            _pf = int(_prefix_frames_from_ms(_short_continuity_ms))
                            prefix_k = int(min(int(_pf), int(x01.size(-1)), int(prev_mel.size(-1))))
                            if prefix_k > 0:
                                prefix_tail = prev_mel[:, :, -prefix_k:].contiguous()
                    except Exception:
                        prefix_k = 0
                        prefix_tail = None

                    if prefix_k > 0 and torch.is_tensor(prefix_tail):
                        if bool(getattr(args, "mel_twopass", False)):
                            mel_out1 = sample_mel_flow_with_prefix_twopass(
                                mel_flow,
                                x01,
                                x_tok1 if bool(CONFIG["text_cross_attn"]) else None,
                                speaker_ids1,
                                steps_first=int(getattr(args, "mel_twopass_steps_first", int(CONFIG["mel_flow_steps_demo"]))),
                                steps_second=int(getattr(args, "mel_twopass_steps_second", 0)),
                                t_noise=float(getattr(args, "mel_twopass_t_noise", 0.12)),
                                spk_vec_override=spk_base1,
                                prefix_tail_bct=prefix_tail,
                                prefix_k=int(prefix_k),
                            )
                        else:
                            mel_out1 = sample_mel_flow_with_prefix(
                                mel_flow,
                                x01,
                                x_tok1 if bool(CONFIG["text_cross_attn"]) else None,
                                speaker_ids1,
                                steps=int(CONFIG["mel_flow_steps_demo"]),
                                spk_vec_override=spk_base1,
                                prefix_tail_bct=prefix_tail,
                                prefix_k=int(prefix_k),
                            )
                    else:
                        if bool(getattr(args, "mel_twopass", False)):
                            mel_out1 = sample_mel_flow_twopass(
                                mel_flow,
                                x01,
                                x_tok1 if bool(CONFIG["text_cross_attn"]) else None,
                                speaker_ids1,
                                steps_first=int(getattr(args, "mel_twopass_steps_first", int(CONFIG["mel_flow_steps_demo"]))),
                                steps_second=int(getattr(args, "mel_twopass_steps_second", 0)),
                                t_noise=float(getattr(args, "mel_twopass_t_noise", 0.12)),
                                spk_vec_override=spk_base1,
                            )
                        else:
                            mel_out1 = sample_mel_flow(
                                mel_flow,
                                x01,
                                x_tok1 if bool(CONFIG["text_cross_attn"]) else None,
                                speaker_ids1,
                                steps=int(CONFIG["mel_flow_steps_demo"]),
                                spk_vec_override=spk_base1,
                            )

                    # Update prefix state for next chunk in this book_tag
                    try:
                        _td_prev[str(book_tag)] = mel_out1.detach().to(dtype=torch.float32)
                    except Exception:
                        _td_prev.pop(str(book_tag), None)

                    # save token/duration debug
                    if save_debug:
                        try:
                            ids_list = ids_full1[0].detach().cpu().tolist()
                            durs_list = torch.round(dur_for_prior1[0].detach().cpu()).long().tolist()
                            lines = [
                                f"text: {txt}",
                                f"emotion_group={str(emotion_group)}",
                                f"emotion_gate={float(emotion_style_gate.detach().cpu()):.6f}",
                                f"dur_scale={dur_scale} sp_min={sp_min}",
                                "tokens:",
                            ]
                            for ii, (tid, dd) in enumerate(zip(ids_list, durs_list)):
                                if ii >= 200:
                                    break
                                sym = ID2SYMBOL.get(int(tid), str(int(tid)))
                                lines.append(f"{ii:03d} {int(tid)}:{sym} dur={int(dd)}")
                            (text_demo_dir / f"{tag}_dur_debug.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
                        except Exception:
                            pass

                    # save audio
                    if save_audio:
                        _decode_and_save_to(text_demo_dir, f"{tag}_mu", mu1[:1].contiguous(), silence_mask1)
                        _decode_and_save_to(text_demo_dir, f"{tag}_refined", mel_out1[:1].contiguous(), silence_mask1)
                    # Return tensors for concatenation: trim the duplicated prefix (already present at end of previous chunk).
                    mu_ret = mu1[:1].contiguous()
                    mel_ret = mel_out1[:1].contiguous()
                    mask_ret = silence_mask1[:1].contiguous() if torch.is_tensor(silence_mask1) else None
                    if int(prefix_k) > 0:
                        if int(mu_ret.size(-1)) > int(prefix_k):
                            mu_ret = mu_ret[..., int(prefix_k) :].contiguous()
                        else:
                            mu_ret = mu_ret[..., :0].contiguous()
                        if int(mel_ret.size(-1)) > int(prefix_k):
                            mel_ret = mel_ret[..., int(prefix_k) :].contiguous()
                        else:
                            mel_ret = mel_ret[..., :0].contiguous()
                        if torch.is_tensor(mask_ret):
                            if int(mask_ret.size(-1)) > int(prefix_k):
                                mask_ret = mask_ret[..., int(prefix_k) :].contiguous()
                            else:
                                mask_ret = mask_ret[..., :0].contiguous()
                    return mu_ret, mel_ret, mask_ret

                # Optional extra dialog/story prompts.
                if bool(getattr(args, "extra_text_demos", True)):
                    extra_sent = list(EXTRA_DEMO_SENTENCES_PL)
                    news_long = list(NEWS_DEMO_CHUNKS_PL)
                    extra_long = list(EXTRA_LONG_DEMO_CHUNKS_PL)
                    extra_extreme_long = list(EXTRA_EXTREME_LONG_DEMO_CHUNKS_PL)
                else:
                    extra_sent = []
                    news_long = []
                    extra_long = []
                    extra_extreme_long = []

                test_sentences = list(extra_sent) + list(TEST_SENTENCES_PL)

                demo_extra_long_chunks = int(getattr(args, "demo_extra_long_chunks", 0))
                demo_extreme_long_chunks = int(getattr(args, "demo_extreme_long_chunks", 0))

                # 1) Short sentences (independent).
                demo_emotions = [
                    e.strip().lower()
                    for e in str(getattr(args, "demo_emotions", "neutral,happy")).split(",")
                    if e.strip()
                ]
                if not bool(emotion_conditioning_enabled):
                    demo_emotions = [""]
                for i, txt in enumerate(test_sentences[: max(1, int(getattr(args, "demo_count", 6)))]):
                    for emo in demo_emotions:
                        if emo:
                            _gen_text_one(
                                txt=str(txt),
                                book_tag=f"text_demo_sent_{i}_{emo}",
                                chunk_i=0,
                                tag=f"sent{i:02d}_{emo}",
                                emotion_group=emo,
                            )
                        else:
                            _gen_text_one(txt=str(txt), book_tag=f"text_demo_sent_{i}", chunk_i=0, tag=f"sent{i:02d}")

                # 2) Long story chunks (stateful across chunks via book_id + chunk_idx).
                n_chunks = int(getattr(args, "demo_long_chunks", 7))
                n_chunks = int(max(0, min(n_chunks, len(LONG_DEMO_CHUNKS_PL))))
                if n_chunks > 0:
                    mu_chunks: List[torch.Tensor] = []
                    ref_chunks: List[torch.Tensor] = []
                    mask_chunks: List[Optional[torch.Tensor]] = []
                    long_text_lines: List[str] = []
                    for j in range(n_chunks):
                        txt_j = str(LONG_DEMO_CHUNKS_PL[j])
                        long_text_lines.append(f"[{j:02d}] {txt_j}")

                        mu_j, ref_j, mask_j = _gen_text_one(
                            txt=txt_j,
                            book_tag="text_demo_long",
                            chunk_i=j,
                            tag=f"long{j:02d}",
                            save_audio=False,   # save only concatenated long-story audio
                            save_debug=True,
                        )
                        mu_chunks.append(mu_j)
                        ref_chunks.append(ref_j)
                        mask_chunks.append(mask_j)
                    try:
                        if long_text_lines:
                            (text_demo_dir / "long_story_text.txt").write_text("\n".join(long_text_lines) + "\n", encoding="utf-8")
                    except Exception:
                        pass
                    try:
                        _decode_chunks_and_save_to(text_demo_dir, "long_story_mu", mu_chunks, mask_chunks)
                        _decode_chunks_and_save_to(text_demo_dir, "long_story_refined", ref_chunks, mask_chunks)
                    except Exception:
                        pass

                # 2b) News diagnostic chunks: the exact long sentence/regression case.
                if news_long:
                    mu_newsx: List[torch.Tensor] = []
                    ref_newsx: List[torch.Tensor] = []
                    mask_newsx: List[Optional[torch.Tensor]] = []
                    news_text_lines: List[str] = []
                    for j, txt_j in enumerate(news_long):
                        txt_j = str(txt_j)
                        news_text_lines.append(f"[{j:02d}] {txt_j}")

                        mu_j, ref_j, mask_j = _gen_text_one(
                            txt=txt_j,
                            book_tag="text_demo_news",
                            chunk_i=j,
                            tag=f"news{j:02d}",
                            save_audio=False,
                            save_debug=True,
                        )
                        mu_newsx.append(mu_j)
                        ref_newsx.append(ref_j)
                        mask_newsx.append(mask_j)
                    try:
                        (text_demo_dir / "news_story_text.txt").write_text("\n".join(news_text_lines) + "\n", encoding="utf-8")
                    except Exception:
                        pass
                    try:
                        _decode_chunks_and_save_to(text_demo_dir, "news_story_mu", mu_newsx, mask_newsx)
                        _decode_chunks_and_save_to(text_demo_dir, "news_story_refined", ref_newsx, mask_newsx)
                    except Exception:
                        pass

                # 2b) Extra-long dialog/story chunks (separate topic; not mixed with LONG_DEMO_CHUNKS_PL).
                n_extra = int(max(0, min(int(demo_extra_long_chunks), len(extra_long))))
                if n_extra > 0:
                    mu_chunksx: List[torch.Tensor] = []
                    ref_chunksx: List[torch.Tensor] = []
                    mask_chunksx: List[Optional[torch.Tensor]] = []
                    extra_text_lines: List[str] = []
                    for j in range(n_extra):
                        txt_j = str(extra_long[j])
                        extra_text_lines.append(f"[{j:02d}] {txt_j}")

                        mu_j, ref_j, mask_j = _gen_text_one(
                            txt=txt_j,
                            book_tag="text_demo_extra_long",
                            chunk_i=j,
                            tag=f"xlong{j:02d}",
                            save_audio=False,
                            save_debug=True,
                        )
                        mu_chunksx.append(mu_j)
                        ref_chunksx.append(ref_j)
                        mask_chunksx.append(mask_j)
                    try:
                        if extra_text_lines:
                            (text_demo_dir / "extra_long_story_text.txt").write_text("\n".join(extra_text_lines) + "\n", encoding="utf-8")
                    except Exception:
                        pass
                    try:
                        _decode_chunks_and_save_to(text_demo_dir, "extra_long_story_mu", mu_chunksx, mask_chunksx)
                        _decode_chunks_and_save_to(text_demo_dir, "extra_long_story_refined", ref_chunksx, mask_chunksx)
                    except Exception:
                        pass

                # 3) Extreme-long dialog/story chunks (stateful across chunks; extra_text_demos-only by default).
                n_ext = int(max(0, min(int(demo_extreme_long_chunks), len(extra_extreme_long))))
                if n_ext > 0:
                    mu_chunks2: List[torch.Tensor] = []
                    ref_chunks2: List[torch.Tensor] = []
                    mask_chunks2: List[Optional[torch.Tensor]] = []
                    ext_text_lines: List[str] = []
                    for j in range(n_ext):
                        txt_j = str(extra_extreme_long[j])
                        ext_text_lines.append(f"[{j:02d}] {txt_j}")

                        mu_j, ref_j, mask_j = _gen_text_one(
                            txt=txt_j,
                            book_tag="text_demo_extreme_long",
                            chunk_i=j,
                            tag=f"ext{j:02d}",
                            save_audio=False,
                            save_debug=True,
                        )
                        mu_chunks2.append(mu_j)
                        ref_chunks2.append(ref_j)
                        mask_chunks2.append(mask_j)
                    try:
                        if ext_text_lines:
                            (text_demo_dir / "extreme_long_story_text.txt").write_text("\n".join(ext_text_lines) + "\n", encoding="utf-8")
                    except Exception:
                        pass
                    try:
                        _decode_chunks_and_save_to(text_demo_dir, "extreme_long_story_mu", mu_chunks2, mask_chunks2)
                        _decode_chunks_and_save_to(text_demo_dir, "extreme_long_story_refined", ref_chunks2, mask_chunks2)
                    except Exception:
                        pass

        prior_mu.train()
        dur_predictor.train()
        mel_flow.train()
if __name__ == "__main__":
    main()
