#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TTS daemon for the current WęgorzTTS prefix-only speaker-encoder model.
Duration predictor is selected from checkpoint args (BiLSTM or stateful LSTM).

Loads model ONCE, then serves JSON-line synthesis requests via stdin/stdout.

stdin (one JSON per line):
  {
    "text": "Tekst do syntezy.",
    "ref_mel": "/path/to/speaker.pt",   # mel .pt used by trainable speaker_encoder
    "voice_emb": "/path/to/emb.pt",      # optional precomputed spk_256 vector
    "speed": 1.0,
    "mel_steps_first": 8,
    "mel_steps_second": 3,
    "mel_twopass_t_noise": 0.12,
    "seed": 1234,
    "out_dir": "/tmp/tts_out",
    "tag": "abc123",
    "lang": "pl",
    "digital_silence": false,
    "pause_edge_frames": 10
  }

stdout:
  {"wav": "/tmp/tts_out/abc123.wav"}     on success
  {"error": "message"}                   on failure

First stdout line after startup: {"status": "ready"}
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import json
import logging
import math
import os
import re
import sys
import time
import tempfile
import zlib
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

# ---------------------------------------------------------------------------
# sys.path setup — must happen BEFORE importing project modules
# ---------------------------------------------------------------------------
_THIS_DIR = Path(__file__).resolve().parent
_PAGE_DIR = _THIS_DIR.parent
_TRANSLATE_DIR = _PAGE_DIR / "translate"

for _p in [str(_THIS_DIR), str(_PAGE_DIR), str(_TRANSLATE_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import soundfile as sf

# ---------------------------------------------------------------------------
# Import model module (gives access to all class/function definitions)
# ---------------------------------------------------------------------------
_model_path = Path(
    os.environ.get("WEGORZ_TTS_MODEL_MODULE", str(_THIS_DIR / "WęgorzTTS3_dubbing_lstm_styleadapters.py"))
).expanduser()
if not _model_path.exists():
    raise RuntimeError(f"TTS model module not found: {_model_path}")
print(f"[daemon] Runtime model module: {_model_path}", file=sys.stderr, flush=True)
_spec = importlib.util.spec_from_file_location("_wegorz_tts_model", str(_model_path))
_vmod = importlib.util.module_from_spec(_spec)
sys.modules["_wegorz_tts_model"] = _vmod  # must be in sys.modules before exec (for @dataclass)
_spec.loader.exec_module(_vmod)  # type: ignore[union-attr]

# Model classes (all defined at module level in model module)
ARDurationTransformer = getattr(_vmod, "ARDurationTransformer", None)
BiLSTMDurationPredictor = getattr(_vmod, "BiLSTMDurationPredictor", None)
MiniDualPathDurationPredictor = getattr(_vmod, "MiniDualPathDurationPredictor", None)
ProbabilisticPriorDecoder = _vmod.ProbabilisticPriorDecoder
MelFlowDecoder = _vmod.MelFlowDecoder
SpeakerlessMelFlowDecoder = _vmod.SpeakerlessMelFlowDecoder
SpeakerVecAdapter = _vmod.SpeakerVecAdapter
ContextBridge = _vmod.ContextBridge
ContextBridgeCache = _vmod.ContextBridgeCache

# Module-level helper functions
_ensure_mel_bct = _vmod._ensure_mel_bct
_ensure_boundary_tokens = _vmod._ensure_boundary_tokens
_ensure_lang_prefix = _vmod._ensure_lang_prefix
_make_tmask_from_Tlen = _vmod._make_tmask_from_Tlen
_partial_load = _vmod._partial_load
_pause_mask_from_ids = _vmod._pause_mask_from_ids
_build_dur_allowed_mask = _vmod._build_dur_allowed_mask
_zero_spk_cond_like = _vmod._zero_spk_cond_like
_predict_dur_prior_direct = _vmod._predict_dur_prior_direct
_pause_middle_frame_mask_from_durations = _vmod._pause_middle_frame_mask_from_durations
_apply_digital_silence_to_wav = _vmod._apply_digital_silence_to_wav
sample_mel_flow_twopass = _vmod.sample_mel_flow_twopass
sample_mel_flow = _vmod.sample_mel_flow
sample_mel_flow_with_prefix_twopass = _vmod.sample_mel_flow_with_prefix_twopass
sample_mel_flow_with_prefix = _vmod.sample_mel_flow_with_prefix
encode_text_features = _vmod.encode_text_features
encode_text_features_stateful = _vmod.encode_text_features_stateful
set_seed = _vmod.set_seed
_prefix_frames_from_ms = _vmod._prefix_frames_from_ms
EMOTION_GROUP_TO_ID = getattr(_vmod, "EMOTION_GROUP_TO_ID", {"neutral": 0})
_apply_emotion_style_conditioning = getattr(_vmod, "_apply_emotion_style_conditioning", None)

# Constants from model module
N_MELS = _vmod.N_MELS
SYMBOL2ID = _vmod.SYMBOL2ID
ID2SYMBOL = _vmod.ID2SYMBOL
PAD_ID = _vmod.PAD_ID
CONFIG = _vmod.CONFIG
_SECS_PER_FRAME = _vmod._SECS_PER_FRAME


def _ensure_boundary_tokens_for_infer(s: str, *, continuation_out: bool = False) -> str:
    s = str(s).strip()
    if not s:
        return s
    s = re.sub(r"^(?:\s*<(?:sp|BOS|EOS)>\s*)+", "", s, flags=re.IGNORECASE)
    s = re.sub(r"(?:\s*<(?:sp|BOS|EOS)>\s*)+$", "", s, flags=re.IGNORECASE)
    return f"<BOS>{s}<sp>" if bool(continuation_out) else f"<BOS>{s}<EOS>"

# Imports from project modules
from tts_helpers import GenericTTS, SpeakerAdapter, load_checkpoint, maybe_load_vocos  # type: ignore
from benchmark_config import BenchmarkConfig  # type: ignore
from utils_bilingual_bridge import PLTokenizer  # type: ignore

_CONTINUITY_MEL_BY_KEY: dict[str, torch.Tensor] = {}
_TEXT_CHUNK_BY_KEY: dict[str, int] = {}
_DUR_LSTM_HC_BY_KEY: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
_STATE_SNAPSHOTS: dict[str, dict[str, Any]] = {}
_PL_NORMALIZER: Any = None
_EN_NORMALIZER: Any = None


class _ScaledEmotionToken(nn.Module):
    """Amplify the learned emotion-token direction without changing the checkpoint."""

    def __init__(self, base: nn.Module, strength: float):
        super().__init__()
        self.base = base
        self.strength = float(strength)

    def forward(self, emotion_ids: torch.Tensor) -> torch.Tensor:
        values = self.base(emotion_ids)
        neutral = self.base(torch.zeros_like(emotion_ids))
        return neutral + self.strength * (values - neutral)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _load_model(resume: str, device: torch.device) -> Dict[str, Any]:
    """Load model from checkpoint. Returns dict of components."""
    ckpt_path = Path(resume).expanduser()
    payload = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    ckpt_args: Dict[str, Any] = payload.get("args", {}) or {}
    train_base: bool = bool(payload.get("train_base", False))

    # ---- Base model ----
    hidden_dim = int(ckpt_args.get("hidden_dim", 512))
    num_layers = int(ckpt_args.get("layers", 8))
    learned_voice_enabled = (
        str(ckpt_args.get("speaker_vector_source", "")).lower().strip() == "learned_voice"
        or isinstance(payload.get("learned_spk_table"), dict)
        or isinstance(payload.get("learned_style_table"), dict)
    )
    learned_spk_sd = payload.get("learned_spk_table", {}) if isinstance(payload.get("learned_spk_table"), dict) else {}
    learned_style_sd = payload.get("learned_style_table", {}) if isinstance(payload.get("learned_style_table"), dict) else {}
    learned_spk_rows = int(learned_spk_sd.get("weight", torch.empty(0, 0)).shape[0]) if torch.is_tensor(learned_spk_sd.get("weight")) else 0
    learned_style_rows = int(learned_style_sd.get("weight", torch.empty(0, 0)).shape[0]) if torch.is_tensor(learned_style_sd.get("weight")) else 0
    num_speakers = max(int(ckpt_args.get("num_speakers", 256)), learned_spk_rows, learned_style_rows, 1)
    n_heads = int(ckpt_args.get("heads", 8))
    use_sdpa = bool(ckpt_args.get("use_sdpa", False))
    spk_dim = int(ckpt_args.get("spk_dim", 256))

    text_dualpath_projection = bool(ckpt_args.get("text_dualpath_projection", True))
    model = GenericTTS(
        BenchmarkConfig(vocab_size=len(SYMBOL2ID), hidden_dim=hidden_dim,
                        num_layers=num_layers, n_mels=N_MELS),
        n_layers=num_layers,
        n_heads=n_heads,
        use_sdpa=use_sdpa,
        dualpath_branch_mode="projected" if text_dualpath_projection else "split",
        dualpath_attn_dim=int(ckpt_args.get("text_dualpath_attn_dim", hidden_dim // 2)) if text_dualpath_projection else None,
        dualpath_conv_dim=int(ckpt_args.get("text_dualpath_conv_dim", hidden_dim // 2)) if text_dualpath_projection else None,
        dualpath_init_split_identity=bool(ckpt_args.get("dualpath_init_split_identity", True)),
    ).to(device)

    spk_embed = SpeakerAdapter(num_speakers=num_speakers, hidden_dim=hidden_dim).to(device)

    spk_adapter = SpeakerVecAdapter(
        spk_dim, hidden_dim,
        p_drop=float(ckpt_args.get("spk_emb_drop", 0.1)),
    ).to(device)
    learned_spk_table = nn.Embedding(num_speakers, spk_dim).to(device) if learned_voice_enabled else None
    learned_style_table = nn.Embedding(num_speakers, 128).to(device) if learned_voice_enabled else None

    gender_dim = int(ckpt_args.get("gender_emb_dim", hidden_dim))
    if gender_dim == hidden_dim:
        gender_embed = nn.Embedding(3, hidden_dim).to(device)
    else:
        gender_embed = nn.Sequential(
            nn.Embedding(3, gender_dim),
            nn.Linear(gender_dim, hidden_dim),
        ).to(device)

    emotion_enabled = bool(ckpt_args.get("emotion_conditioning", False))
    emotion_emb_dim = int(ckpt_args.get("emotion_emb_dim", 128))
    emotion_embed = nn.Embedding(len(EMOTION_GROUP_TO_ID), emotion_emb_dim).to(device)
    emotion_token_embed = nn.Embedding(len(EMOTION_GROUP_TO_ID), hidden_dim).to(device)
    emotion_to_style = nn.Linear(emotion_emb_dim, 128).to(device)
    emotion_style_gate = nn.Parameter(torch.zeros((), device=device), requires_grad=False)

    mem_dim = int(hidden_dim)
    bridge = ContextBridge(
        text_dim=hidden_dim,
        mem_dim=mem_dim,
        attn_heads=int(CONFIG.get("context_attn_heads", 4)),
        attn_drop=float(CONFIG.get("context_attn_drop", 0.0)),
    ).to(device)

    # ---- Prior mu ----
    prior_layers = max(1, int(ckpt_args.get("prior_layers", 3)))
    prior_heads = max(1, int(ckpt_args.get("prior_heads", 8)))
    prior_mu = ProbabilisticPriorDecoder(
        dim=hidden_dim,
        n_mels=N_MELS,
        n_layers=prior_layers,
        n_heads=prior_heads,
        logs_min=float(ckpt_args.get("prior_logs_min", -7.0)),
        logs_max=float(ckpt_args.get("prior_logs_max", 2.0)),
        dualpath_branch_dim=int(ckpt_args.get("dualpath_branch_dim", hidden_dim // 2)),
        dualpath_attn_dim=int(ckpt_args.get("dualpath_attn_dim", hidden_dim // 2)),
        dualpath_conv_dim=int(ckpt_args.get("dualpath_conv_dim", hidden_dim // 2)),
        dualpath_init_split_identity=bool(ckpt_args.get("dualpath_init_split_identity", True)),
    ).to(device)

    # ---- Duration predictor ----
    duration_model = str(ckpt_args.get("duration_model", "bilstm")).lower().strip()
    if duration_model == "ar_transformer":
        duration_model = "lstm_ar"
    if duration_model == "mini_dualpath":
        if MiniDualPathDurationPredictor is None:
            raise RuntimeError("Checkpoint requests mini_dualpath duration, but runtime model lacks MiniDualPathDurationPredictor.")
        dur_predictor = MiniDualPathDurationPredictor(
            dim=hidden_dim,
            layers=int(ckpt_args.get("mini_dur_layers", 3)),
            attn_dim=int(ckpt_args.get("mini_dur_attn_dim", 128)),
            conv_dim=int(ckpt_args.get("mini_dur_conv_dim", 128)),
            heads=int(ckpt_args.get("mini_dur_heads", 4)),
            dropout=float(ckpt_args.get("mini_dur_dropout", ckpt_args.get("ar_dur_dropout", 0.1))),
        ).to(device)
    else:
        if duration_model == "bilstm" and BiLSTMDurationPredictor is None:
            raise RuntimeError("Checkpoint requests bilstm duration, but runtime model lacks BiLSTMDurationPredictor.")
        if duration_model != "bilstm" and ARDurationTransformer is None:
            raise RuntimeError(f"Checkpoint requests {duration_model!r} duration, but runtime model lacks ARDurationTransformer.")
        dur_cls = BiLSTMDurationPredictor if duration_model == "bilstm" else ARDurationTransformer
        dur_predictor = dur_cls(
            dim=hidden_dim,
            layers=int(ckpt_args.get("ar_dur_layers", 4)),
            heads=int(ckpt_args.get("ar_dur_heads", 4)),
            ffn_mult=int(ckpt_args.get("ar_dur_ffn_mult", 4)),
            dropout=float(ckpt_args.get("ar_dur_dropout", 0.1)),
        ).to(device)

    # ---- Mel flow ----
    flow_layers = max(1, int(ckpt_args.get("flow_layers", 6)))
    flow_heads = max(1, int(ckpt_args.get("flow_heads", 8)))
    mel_flow = SpeakerlessMelFlowDecoder(
        dim=hidden_dim,
        text_dim=hidden_dim,
        n_mels=N_MELS,
        heads=flow_heads,
        layers=flow_layers,
        num_speakers=num_speakers,
        use_timbre_adaln=bool(ckpt_args.get("flow_timbre_adaln", False)),
        dualpath_branch_dim=int(ckpt_args.get("dualpath_branch_dim", hidden_dim // 2)),
        dualpath_attn_dim=int(ckpt_args.get("dualpath_attn_dim", hidden_dim // 2)),
        dualpath_conv_dim=int(ckpt_args.get("dualpath_conv_dim", hidden_dim // 2)),
        dualpath_init_split_identity=bool(ckpt_args.get("dualpath_init_split_identity", True)),
        flow_gauss_token_attn=bool(ckpt_args.get("flow_gauss_token_attn", False)),
        flow_gauss_cross_attn=bool(ckpt_args.get("flow_gauss_cross_attn", False)),
        flow_gauss_token_group_size=int(ckpt_args.get("flow_gauss_token_group_size", 1)),
    ).to(device)

    # ---- Trainable speaker_encoder copy from the checkpoint ----
    from spk_style_dualhead_model import MelDualEncoder  # type: ignore

    speaker_encoder = None
    if not learned_voice_enabled:
        speaker_encoder = MelDualEncoder(
            n_mels=int(N_MELS),
            d_spk=spk_dim,
            d_style=128,
            hidden_dim=256,
            num_layers=4,
            attn_head_dim=128,
            style_project_off_spk=True,
        ).to(device).eval()

    model.dec_mu = prior_mu
    model.dur = dur_predictor

    # ---- Load checkpoint state dicts ----
    if train_base and "base_model" in payload:
        skipped = _partial_load(model, dict(payload.get("base_model", {})))
        if skipped:
            print(f"[daemon] base_model partial load: skipped {len(skipped)}", file=sys.stderr, flush=True)
    if "bridge" in payload:
        bridge.load_state_dict(payload["bridge"], strict=False)
    if "spk_embed" in payload:
        skipped = _partial_load(spk_embed, dict(payload.get("spk_embed", {})))
        if skipped:
            print(f"[daemon] legacy spk_embed partial load skipped {len(skipped)} tensors", file=sys.stderr, flush=True)
    if "spk_adapter" in payload:
        spk_adapter.load_state_dict(payload["spk_adapter"], strict=False)
    if learned_spk_table is not None and isinstance(payload.get("learned_spk_table"), dict):
        learned_spk_table.load_state_dict(payload["learned_spk_table"], strict=False)
        print(f"[daemon] Loaded learned_spk_table rows={learned_spk_table.num_embeddings}", file=sys.stderr, flush=True)
    if learned_style_table is not None and isinstance(payload.get("learned_style_table"), dict):
        learned_style_table.load_state_dict(payload["learned_style_table"], strict=False)
        print(f"[daemon] Loaded learned_style_table rows={learned_style_table.num_embeddings}", file=sys.stderr, flush=True)
    if "gender_embed" in payload:
        gender_embed.load_state_dict(payload["gender_embed"], strict=False)
    if "emotion_embed" in payload:
        emotion_embed.load_state_dict(payload.get("emotion_embed", {}), strict=False)
    if "emotion_token_embed" in payload:
        emotion_token_embed.load_state_dict(payload.get("emotion_token_embed", {}), strict=False)
    if "emotion_to_style" in payload:
        emotion_to_style.load_state_dict(payload.get("emotion_to_style", {}), strict=False)
    if torch.is_tensor(payload.get("emotion_style_gate")):
        emotion_style_gate.data.copy_(
            payload["emotion_style_gate"].to(device=device, dtype=emotion_style_gate.dtype).view(())
        )
    prior_mu.load_state_dict(payload.get("prior_mu", {}), strict=False)
    mel_flow.load_state_dict(payload.get("mel_flow", {}), strict=False)
    dur_predictor.load_state_dict(payload.get("dur_predictor", {}), strict=False)
    if speaker_encoder is not None and "speaker_encoder" in payload and payload.get("speaker_encoder") is not None:
        speaker_encoder.load_state_dict(payload["speaker_encoder"], strict=False)
        print("[daemon] Loaded speaker_encoder from checkpoint", file=sys.stderr, flush=True)
    elif speaker_encoder is not None:
        spk_style_ckpt = str(ckpt_args.get("spk_style_ckpt", "")).strip()
        if not spk_style_ckpt or not Path(spk_style_ckpt).exists():
            raise RuntimeError("Checkpoint has no speaker_encoder and spk_style_ckpt fallback is missing.")
        dh_payload = torch.load(spk_style_ckpt, map_location="cpu", weights_only=False)
        speaker_encoder.load_state_dict(dh_payload["model"], strict=True)
        print(f"[daemon] Loaded speaker_encoder fallback from: {spk_style_ckpt}", file=sys.stderr, flush=True)

    # ---- Eval mode ----
    eval_modules = [
        model, spk_embed, spk_adapter, gender_embed, emotion_embed, emotion_token_embed,
        emotion_to_style, bridge, prior_mu, dur_predictor, mel_flow,
    ]
    if speaker_encoder is not None:
        eval_modules.append(speaker_encoder)
    if learned_spk_table is not None:
        eval_modules.append(learned_spk_table)
    if learned_style_table is not None:
        eval_modules.append(learned_style_table)
    for m in eval_modules:
        m.eval()
        for p in m.parameters():
            p.requires_grad_(False)

    # ---- Vocos vocoder ----
    vocos = maybe_load_vocos(device)
    if vocos is None:
        raise RuntimeError("Vocos vocoder not found. Check tts_helpers.maybe_load_vocos().")

    print(f"[daemon] Model loaded. device={device}", file=sys.stderr, flush=True)

    return {
        "model": model,
        "spk_embed": spk_embed,
        "spk_adapter": spk_adapter,
        "gender_embed": gender_embed,
        "emotion_enabled": emotion_enabled,
        "emotion_embed": emotion_embed,
        "emotion_token_embed": emotion_token_embed,
        "emotion_to_style": emotion_to_style,
        "emotion_style_gate": emotion_style_gate,
        "bridge": bridge,
        "bridge_cache": ContextBridgeCache(),
        "prior_mu": prior_mu,
        "dur_predictor": dur_predictor,
        "mel_flow": mel_flow,
        "speaker_encoder": speaker_encoder,
        "learned_voice_enabled": learned_voice_enabled,
        "learned_spk_table": learned_spk_table,
        "learned_style_table": learned_style_table,
        "vocos": vocos,
        "ckpt_args": ckpt_args,
        "device": device,
    }


# ---------------------------------------------------------------------------
# Helper functions (replicate inner functions from verifier main())
# ---------------------------------------------------------------------------

def _load_ref_mel_pt(path: str, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    p = Path(path).expanduser()
    obj = torch.load(str(p), map_location="cpu")
    if isinstance(obj, dict):
        mel = obj.get("mel", obj.get("mel_bct", obj.get("x", obj.get("features"))))
        t_val = obj.get("T_mel", obj.get("length", obj.get("T", None)))
    else:
        mel = obj
        t_val = None
    if not torch.is_tensor(mel):
        raise RuntimeError(f"Cannot find mel tensor in: {p}")
    mel_bct = _ensure_mel_bct(mel).to(device=device, dtype=torch.float32)
    if t_val is None:
        t_len = torch.tensor([int(mel_bct.size(-1))], device=device, dtype=torch.long)
    else:
        t_i = int(t_val.view(-1)[0].item()) if torch.is_tensor(t_val) else int(t_val)
        t_len = torch.tensor([max(1, min(t_i, int(mel_bct.size(-1))))],
                             device=device, dtype=torch.long)
    return mel_bct[:, :, : int(t_len[0].item())].contiguous(), t_len


def _load_voice_emb(path: str, device: torch.device, expected_dim: int = 256) -> torch.Tensor:
    p = Path(path).expanduser()
    obj = torch.load(str(p), map_location="cpu", weights_only=False)
    if isinstance(obj, dict):
        emb = obj.get("emb", obj.get("speaker_emb", obj.get("spk_emb", obj.get("spk_256", None))))
    else:
        emb = obj
    if not torch.is_tensor(emb):
        raise RuntimeError(f"Cannot find speaker embedding tensor in: {p}")
    z = emb.detach().to(device=device, dtype=torch.float32).view(-1)
    if int(z.numel()) != int(expected_dim):
        raise RuntimeError(f"speaker embedding dim mismatch: got {int(z.numel())}, expected {expected_dim}: {p}")
    return z / z.norm().clamp_min(1e-12)


def _vocos_decode(vocos: Any, mel_1ct: torch.Tensor) -> Optional[torch.Tensor]:
    if not torch.is_tensor(mel_1ct) or mel_1ct.ndim != 3 or mel_1ct.numel() == 0:
        return None
    mel_1ct = mel_1ct.to(dtype=torch.float32)
    if not torch.isfinite(mel_1ct).all():
        return None
    try:
        with torch.no_grad():
            return vocos.decode(mel_1ct)
    except Exception as exc:
        print(f"[daemon] ⚠️ vocos.decode failed: {exc}", file=sys.stderr, flush=True)
        return None


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
    sf.write(str(path), wav.squeeze(0).numpy(), sr)


@torch.no_grad()
def _sample_mel_flow_onepass_daemon(
    mel_flow: nn.Module,
    x0_bct: torch.Tensor,
    text_seq_b: Optional[torch.Tensor],
    speaker_ids: torch.Tensor,
    *,
    steps: int,
    spk_vec_override: Optional[torch.Tensor],
    gauss_align_btl: Optional[torch.Tensor],
) -> torch.Tensor:
    x = x0_bct
    dt = 1.0 / float(max(1, int(steps)))
    for k in range(int(steps)):
        t = torch.full((x.size(0),), float(k) / float(max(1, int(steps))), device=x.device, dtype=x.dtype)
        try:
            v = mel_flow(
                x,
                t,
                speaker_ids,
                text_seq_b,
                spk_vec_override=spk_vec_override,
                gauss_align_btl=gauss_align_btl,
            )
        except TypeError:
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
def _sample_mel_flow_twopass_daemon(
    mel_flow: nn.Module,
    x0_bct: torch.Tensor,
    text_seq_b: Optional[torch.Tensor],
    speaker_ids: torch.Tensor,
    *,
    steps_first: int,
    steps_second: int,
    t_noise: float,
    spk_vec_override: Optional[torch.Tensor],
    gauss_align_btl: Optional[torch.Tensor],
    prefix_tail_bct: Optional[torch.Tensor] = None,
    prefix_k: int = 0,
) -> torch.Tensor:
    def with_prefix(x_bct: torch.Tensor) -> torch.Tensor:
        if prefix_tail_bct is None or int(prefix_k) <= 0:
            return x_bct
        k = int(prefix_k)
        return torch.cat([prefix_tail_bct.to(device=x_bct.device, dtype=x_bct.dtype), x_bct[..., k:]], dim=-1)

    x_start = with_prefix(x0_bct)
    mel_first = _sample_mel_flow_onepass_daemon(
        mel_flow,
        x_start,
        text_seq_b,
        speaker_ids,
        steps=int(steps_first),
        spk_vec_override=spk_vec_override,
        gauss_align_btl=gauss_align_btl,
    )
    if int(steps_second) <= 0 or float(t_noise) <= 0.0:
        return mel_first
    mix = float(max(0.0, min(1.0, float(t_noise))))
    x_second = with_prefix((1.0 - mix) * mel_first + mix * x0_bct)
    return _sample_mel_flow_onepass_daemon(
        mel_flow,
        x_second,
        text_seq_b,
        speaker_ids,
        steps=int(steps_second),
        spk_vec_override=spk_vec_override,
        gauss_align_btl=gauss_align_btl,
    )


def _snapshot_state(components: Dict[str, Any]) -> Dict[str, Any]:
    bridge_cache = components["bridge_cache"]
    sid = f"snap_{int(time.time()*1000)}_{os.getpid()}_{len(_STATE_SNAPSHOTS)}"
    _STATE_SNAPSHOTS[sid] = {
        "mel": {k: v.clone() for k, v in _CONTINUITY_MEL_BY_KEY.items()},
        "text_chunk": dict(_TEXT_CHUNK_BY_KEY),
        "dur_lstm_hc": {
            k: (h.detach().clone(), c.detach().clone())
            for k, (h, c) in _DUR_LSTM_HC_BY_KEY.items()
        },
        "bridge_mem": {k: v.clone() for k, v in bridge_cache._mem.items()},
        "bridge_prev_h": {k: v.clone() for k, v in bridge_cache._prev_h.items()},
        "bridge_prev_mask": {k: v.clone() for k, v in bridge_cache._prev_mask.items()},
        "bridge_last_chunk": dict(bridge_cache._last_chunk),
    }
    return {"snapshot_id": sid}


def _restore_state(components: Dict[str, Any], snapshot_id: str) -> Dict[str, Any]:
    snap = _STATE_SNAPSHOTS.get(str(snapshot_id))
    if not snap:
        raise RuntimeError(f"Unknown daemon state snapshot: {snapshot_id}")
    bridge_cache = components["bridge_cache"]
    _CONTINUITY_MEL_BY_KEY.clear()
    _CONTINUITY_MEL_BY_KEY.update({k: v.clone() for k, v in snap["mel"].items()})
    _TEXT_CHUNK_BY_KEY.clear()
    _TEXT_CHUNK_BY_KEY.update(dict(snap["text_chunk"]))
    _DUR_LSTM_HC_BY_KEY.clear()
    _DUR_LSTM_HC_BY_KEY.update({
        k: (h.clone(), c.clone())
        for k, (h, c) in dict(snap.get("dur_lstm_hc", {})).items()
    })
    bridge_cache._mem = {k: v.clone() for k, v in snap["bridge_mem"].items()}
    bridge_cache._prev_h = {k: v.clone() for k, v in snap["bridge_prev_h"].items()}
    bridge_cache._prev_mask = {k: v.clone() for k, v in snap["bridge_prev_mask"].items()}
    bridge_cache._last_chunk = dict(snap["bridge_last_chunk"])
    return {"restored": True}


def _encode_ref_audio_to_mel(req: Dict[str, Any], components: Dict[str, Any]) -> Dict[str, Any]:
    vocos = components["vocos"]
    device: torch.device = components["device"]
    src = Path(str(req.get("audio", "")).strip()).expanduser()
    if not src.exists():
        raise RuntimeError(f"Reference audio not found: {src}")
    out_dir = Path(str(req.get("out_dir", tempfile.gettempdir())))
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = str(req.get("tag", f"ref_{int(time.time()*1000)%100000}"))
    out_pt = out_dir / f"{tag}.mel.pt"
    start_sec = max(0.0, float(req.get("start_sec", 0.0)))
    max_sec = float(req.get("max_sec", 12.0))

    wav_np, sr = sf.read(str(src), always_2d=False)
    if getattr(wav_np, "ndim", 1) > 1:
        wav_np = wav_np.mean(axis=1)
    if int(sr) != 24000:
        raise RuntimeError(f"Reference WAV must be 24 kHz after server conversion, got sr={sr}: {src}")
    s0 = max(0, int(round(start_sec * sr)))
    s1 = len(wav_np) if max_sec <= 0 else min(len(wav_np), s0 + int(round(max_sec * sr)))
    wav_np = wav_np[s0:s1]
    if wav_np.size < int(0.25 * sr):
        raise RuntimeError("Reference audio is too short after trimming.")

    wav = torch.from_numpy(wav_np).to(device=device, dtype=torch.float32).unsqueeze(0)
    fe = getattr(vocos, "feature_extractor", None)
    if fe is None:
        raise RuntimeError("Loaded Vocos has no feature_extractor; cannot compute mel from reference audio.")
    vocos_dev = next(vocos.parameters()).device
    with torch.no_grad():
        mel = fe(wav.to(device=vocos_dev, dtype=torch.float32))
    if mel.dim() == 3 and int(mel.size(1)) != int(N_MELS) and int(mel.size(2)) == int(N_MELS):
        mel = mel.transpose(1, 2).contiguous()
    mel_bct = _ensure_mel_bct(mel).detach().cpu().to(torch.float32).contiguous()
    torch.save({"mel": mel_bct[0], "T_mel": torch.tensor([int(mel_bct.size(-1))])}, str(out_pt))
    return {
        "mel": str(out_pt),
        "frames": int(mel_bct.size(-1)),
        "duration": round(float(wav_np.size) / float(sr), 3),
    }


def _normalize_en_for_tts(text: str) -> str:
    """English text normalization for TTS inference: numbers/dates/symbols -> words."""
    global _EN_NORMALIZER
    try:
        if _EN_NORMALIZER is None:
            for name in ("nemo_logger", "nemo_text_processing", "NeMo"):
                logging.getLogger(name).setLevel(logging.ERROR)
            logging.getLogger("nemo_text_processing.text_normalization").setLevel(logging.ERROR)
            from nemo_text_processing.text_normalization.normalize import Normalizer  # type: ignore
            with contextlib.redirect_stdout(sys.stderr):
                _EN_NORMALIZER = Normalizer(input_case="cased", lang="en")
        with contextlib.redirect_stdout(sys.stderr):
            return str(_EN_NORMALIZER.normalize(str(text), verbose=False, punct_post_process=True))
    except Exception as exc:
        print(f"[daemon] ⚠️ EN normalization failed, using raw text: {exc}", file=sys.stderr, flush=True)
        return str(text)


def _normalize_pl_for_tts(text: str) -> str:
    """Polish text normalization for TTS inference using the bundled Węgorz normalizer."""
    global _PL_NORMALIZER
    try:
        if _PL_NORMALIZER is None:
            from wegorz_normalizer import PolishTTSPipeline  # type: ignore
            _PL_NORMALIZER = PolishTTSPipeline()
        out = str(_PL_NORMALIZER.process(str(text))).strip()
        return out or str(text)
    except Exception as exc:
        print(f"[daemon] ⚠️ PL normalization failed, using raw text: {exc}", file=sys.stderr, flush=True)
        return str(text)


def _prepare_text_for_tts(text: str, lang: str) -> str:
    """Prepare text in the same format used by the mixed PL-orth + EN-IPA tokenizer."""
    lang_s = str(lang or "pl").strip().lower()
    raw = str(text or "").strip()
    if lang_s == "en":
        try:
            from english_phonemizer import phonemize_en_ipa  # type: ignore
            normalized = _normalize_en_for_tts(raw)
            with contextlib.redirect_stdout(sys.stderr):
                return phonemize_en_ipa(normalized, voice="en-us", backend="misaki_no_trf")
        except Exception as exc:
            print(f"[daemon] ⚠️ EN phonemization failed, using raw text: {exc}", file=sys.stderr, flush=True)
            return raw
    if lang_s == "pl":
        return _normalize_pl_for_tts(raw)
    return raw


# ---------------------------------------------------------------------------
# Single-segment synthesis
# ---------------------------------------------------------------------------

@torch.no_grad()
def synthesize_one(req: Dict[str, Any], components: Dict[str, Any]) -> Dict[str, Any]:
    """Synthesize one text segment. Returns daemon response dict."""
    device: torch.device = components["device"]
    model = components["model"]
    spk_embed = components["spk_embed"]
    spk_adapter = components["spk_adapter"]
    gender_embed = components["gender_embed"]
    emotion_enabled = bool(components.get("emotion_enabled", False))
    emotion_embed = components.get("emotion_embed")
    emotion_token_embed = components.get("emotion_token_embed")
    emotion_to_style = components.get("emotion_to_style")
    emotion_style_gate = components.get("emotion_style_gate")
    bridge = components["bridge"]
    bridge_cache = components["bridge_cache"]
    prior_mu = components["prior_mu"]
    mel_flow = components["mel_flow"]
    vocos = components["vocos"]
    speaker_encoder = components["speaker_encoder"]
    learned_voice_enabled = bool(components.get("learned_voice_enabled", False))
    learned_spk_table = components.get("learned_spk_table")
    learned_style_table = components.get("learned_style_table")
    ckpt_args = components["ckpt_args"]

    text: str = str(req.get("text", "")).strip()
    ref_mel: str = str(req.get("ref_mel", "")).strip()
    voice_emb: str = str(req.get("voice_emb", "")).strip()
    speed: float = float(req.get("speed", 1.0))
    mel_steps_first: int = int(req.get("mel_steps_first", int(ckpt_args.get("mel_twopass_steps_first", 8))))
    mel_steps_second: int = int(req.get("mel_steps_second", int(ckpt_args.get("mel_twopass_steps_second", 3))))
    t_noise: float = float(req.get("mel_twopass_t_noise", float(ckpt_args.get("mel_twopass_t_noise", 0.12))))
    seed: int = int(req.get("seed", 1234))
    out_dir: str = str(req.get("out_dir", tempfile.gettempdir()))
    tag: str = str(req.get("tag", f"daemon_{int(time.time()*1000)%100000}"))
    lang: str = str(req.get("lang", "pl"))
    digital_silence: bool = bool(req.get("digital_silence", True))
    pause_edge: int = int(req.get("pause_edge_frames", int(ckpt_args.get("pause_edge_frames", 10))))
    short_continuity_ms: float = float(req.get("short_continuity_ms", 0.0))
    continuity_key: str = str(req.get("continuity_key", "")).strip()
    continuity_reset: bool = bool(req.get("continuity_reset", False))
    # Text demos reset the duration LSTM for every chunk. Carrying its hidden
    # state was never trained and collapses durations after 2-3 chunks.
    duration_stateful: bool = bool(req.get("duration_stateful", False))
    emotion_group = str(req.get("emotion_group", "neutral")).strip().lower()
    if emotion_group not in EMOTION_GROUP_TO_ID:
        emotion_group = "neutral"
    emotion_strength = max(0.0, min(6.0, float(req.get("emotion_strength", 3.0))))
    emotion_ids1 = torch.tensor(
        [int(EMOTION_GROUP_TO_ID.get(emotion_group, 0))],
        device=device,
        dtype=torch.long,
    )

    if not text:
        raise ValueError("text is empty")
    if voice_emb and not Path(voice_emb).exists():
        raise ValueError(f"voice_emb not found: {voice_emb!r}")
    if (not learned_voice_enabled) and (not voice_emb) and (not ref_mel or not Path(ref_mel).exists()):
        raise ValueError(f"ref_mel not found: {ref_mel!r}")

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    wav_out = out_path / f"{tag}.wav"

    set_seed(seed)
    if continuity_key and continuity_reset:
        _CONTINUITY_MEL_BY_KEY.pop(continuity_key, None)
        _TEXT_CHUNK_BY_KEY.pop(continuity_key, None)
        _DUR_LSTM_HC_BY_KEY.pop(continuity_key, None)

    # ---- Speaker conditioning ----
    spk_dim = int(ckpt_args.get("spk_dim", 256))
    speaker_id_for_model = 0
    if learned_voice_enabled:
        if learned_spk_table is None or learned_style_table is None:
            raise RuntimeError("learned_voice checkpoint loaded without learned voice tables.")
        raw_sid = req.get("speaker_id", 0)
        try:
            speaker_id_for_model = int(raw_sid)
        except Exception:
            speaker_id_for_model = 0
        max_sid = int(learned_spk_table.num_embeddings) - 1
        if speaker_id_for_model < 0 or speaker_id_for_model > max_sid:
            print(
                f"[daemon] ⚠️ learned_voice speaker_id={speaker_id_for_model} outside 0..{max_sid}; using 0",
                file=sys.stderr,
                flush=True,
            )
            speaker_id_for_model = 0
        sid_t = torch.tensor([speaker_id_for_model], device=device, dtype=torch.long)
        spk_vec_256 = learned_spk_table(sid_t).to(device=device, dtype=torch.float32)
        style_vec = learned_style_table(sid_t).to(device=device, dtype=torch.float32)
        if bool(ckpt_args.get("spk_emb_l2norm", True)):
            spk_vec_256 = spk_vec_256 / spk_vec_256.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    elif voice_emb:
        spk_vec = _load_voice_emb(voice_emb, device=device, expected_dim=spk_dim)
        style_vec = None
        spk_vec_256 = spk_vec.to(device=device).unsqueeze(0)
    else:
        if speaker_encoder is None:
            raise RuntimeError("speaker_encoder is unavailable and checkpoint is not learned_voice.")
        mel_ref_bct, t_ref = _load_ref_mel_pt(ref_mel, device)
        enc_dev = next(speaker_encoder.parameters()).device
        mask_bt = _make_tmask_from_Tlen(t_ref.to(enc_dev), int(mel_ref_bct.size(-1))).squeeze(1).to(
            device=enc_dev,
            dtype=torch.bool,
        )
        z_spk, z_style = speaker_encoder(
            mel_ref_bct.to(device=enc_dev, dtype=torch.float32),
            mask_bt=mask_bt,
        )
        spk_vec = z_spk.detach().to(device=device, dtype=torch.float32).view(-1)
        style_vec = z_style.detach().to(device=device, dtype=torch.float32)
        if bool(ckpt_args.get("spk_emb_l2norm", True)):
            spk_vec = spk_vec / spk_vec.norm().clamp_min(1e-12)
        spk_vec_256 = spk_vec.to(device=device).unsqueeze(0)

    spk_hidden = spk_adapter(spk_vec_256).to(dtype=torch.float32)  # [1, D]
    if emotion_enabled:
        if _apply_emotion_style_conditioning is None:
            raise RuntimeError("Emotion checkpoint loaded with a runtime module that lacks emotion conditioning.")
        style_vec = _apply_emotion_style_conditioning(
            style_vec,
            emotion_ids1,
            enabled=True,
            emotion_embed=emotion_embed,
            emotion_to_style=emotion_to_style,
            emotion_style_gate=emotion_style_gate,
        )

    # ---- Tokenize ----
    tok = PLTokenizer(str(_THIS_DIR / "vocab_pl_orth_en_ipa_bridge.json"))

    lang_prefix = f"<{lang}>" if lang in ("pl", "en") else "<pl>"
    text_prepared = _prepare_text_for_tts(text, lang)
    continuation_out = bool(req.get("continuation_out", False))
    text_in = _ensure_lang_prefix(
        _ensure_boundary_tokens_for_infer(text_prepared, continuation_out=continuation_out),
        default_lang=lang_prefix,
    )
    ids = tok.encode(text_in)
    if not ids:
        raise ValueError(f"Tokenization returned empty for: {text_in!r}")

    tok_pad1 = torch.tensor([ids], device=device, dtype=torch.long)
    speaker_ids1 = torch.tensor([speaker_id_for_model], device=device, dtype=torch.long)
    gender_ids1 = torch.zeros(1, device=device, dtype=torch.long)
    text_stream_key = continuity_key or f"single:{tag}"
    text_chunk_idx = int(_TEXT_CHUNK_BY_KEY.get(text_stream_key, 0))
    book_id = int(zlib.crc32(text_stream_key.encode("utf-8")) & 0x7FFFFFFF)
    book_ids1 = torch.tensor([book_id], dtype=torch.long)
    chunk_idx1 = torch.tensor([text_chunk_idx], dtype=torch.long)

    # ---- Text features with the same stateful ContextBridge used by training/text demos ----
    encode_kwargs = dict(
        model=model,
        spk_embed=spk_embed,
        gender_embed=gender_embed,
        tok_pad=tok_pad1,
        speaker_ids=speaker_ids1,
        gender_ids=gender_ids1,
        book_ids=book_ids1,
        chunk_idx=chunk_idx1,
        device=device,
        bridge=bridge,
        bridge_cache=bridge_cache,
        spk_vec_override=spk_hidden.to(device=device, dtype=torch.float32),
        require_spk_override=bool(ckpt_args.get("require_spk_override", True)),
    )
    if emotion_enabled:
        encode_kwargs.update(
            emotion_token_embed=_ScaledEmotionToken(emotion_token_embed, emotion_strength),
            emotion_ids=emotion_ids1,
            use_emotion_token=True,
        )
    x_tok1, ids_full1, special_len1, _mem_after1 = encode_text_features_stateful(**encode_kwargs)
    _TEXT_CHUNK_BY_KEY[text_stream_key] = text_chunk_idx + 1

    # ---- Duration prediction ----
    dur_noise = float(ckpt_args.get("infer_dur_noise_scale", 0.1))
    _dur_lstm_hc = _DUR_LSTM_HC_BY_KEY.get(text_stream_key) if duration_stateful else None
    if _dur_lstm_hc is not None:
        _dur_lstm_hc = tuple(t.to(device=device) for t in _dur_lstm_hc)  # type: ignore[assignment]
    dur_pred1, _, _ = _predict_dur_prior_direct(
        model,
        x_tok1,
        tok_pad1,
        special_len1,
        source="prior_mu",
        noise_scale=dur_noise,
        dur_prior_logs_min=float(ckpt_args.get("dur_prior_logs_min", -5.0)),
        dur_prior_logs_max=float(ckpt_args.get("dur_prior_logs_max", 2.0)),
        dur_prior_sigma_min=float(ckpt_args.get("dur_prior_sigma_min", 0.1)),
        initial_hc=_dur_lstm_hc,
        style_vec=style_vec,
    )
    _last_hc = getattr(_predict_dur_prior_direct, "_last_hc", None)
    if duration_stateful and _last_hc is not None and all(torch.is_tensor(t) for t in _last_hc):
        _DUR_LSTM_HC_BY_KEY[text_stream_key] = tuple(
            t.detach().to(device="cpu", dtype=torch.float32) for t in _last_hc
        )
    elif not duration_stateful:
        _DUR_LSTM_HC_BY_KEY.pop(text_stream_key, None)

    # ---- Speed scaling ----
    if speed != 1.0:
        dur_pred1 = dur_pred1 / speed
        dur_pred1 = dur_pred1.clamp_min(0.0)
        ids_full_tmp = F.pad(tok_pad1, (int(special_len1), 0), value=PAD_ID)
        pause_mask_tmp = _pause_mask_from_ids(ids_full_tmp)
        dur_pred1 = torch.where(pause_mask_tmp, dur_pred1.clamp_min(1.0), dur_pred1)

    ids_full1 = F.pad(tok_pad1, (int(special_len1), 0), value=PAD_ID)
    dur_allowed1 = _build_dur_allowed_mask(ids_full1, special_len1)
    dur_for_prior1 = torch.where(dur_allowed1, dur_pred1, torch.zeros_like(dur_pred1))
    pause_mask1 = _pause_mask_from_ids(ids_full1)
    sp_mask_tok1 = pause_mask1

    # Give the vocoder a short acoustic pre-roll before the first content token.
    # The duration model commonly predicts one <sp> frame (~11 ms), which puts
    # an initial consonant directly on the Vocos boundary and can weaken it.
    _leading_sp_min = int(req.get("leading_sp_min_frames", 4))
    if _leading_sp_min > 0:
        for b in range(int(dur_for_prior1.size(0))):
            content_pos = torch.nonzero(
                dur_allowed1[b] & (~pause_mask1[b]),
                as_tuple=False,
            ).flatten()
            if int(content_pos.numel()) <= 0:
                continue
            first_content = int(content_pos[0].item())
            leading_pause_pos = torch.nonzero(
                pause_mask1[b, :first_content] & dur_allowed1[b, :first_content],
                as_tuple=False,
            ).flatten()
            if int(leading_pause_pos.numel()) > 0:
                pos = int(leading_pause_pos[-1].item())
                dur_for_prior1[b, pos] = dur_for_prior1[b, pos].clamp_min(float(_leading_sp_min))

    # Clamp trailing pause tokens (after the last content token) to avoid over-long
    # end-of-chunk silence. BiLSTM tends to predict 0.3-0.5s EOS pauses; 10 frames
    # (~107ms) is enough to sound natural without wasting timing budget.
    # Clamp ALL pause tokens to suppress BiLSTM over-prediction on leading / mid-sentence <sp>.
    # 18 frames ≈ 192ms — natural comma pause; without this, leading <sp> was ≥0.54s.
    _pause_max = int(req.get("pause_max_frames", 18))
    if _pause_max > 0:
        dur_for_prior1 = torch.where(
            pause_mask1,
            dur_for_prior1.clamp_max(float(_pause_max)),
            dur_for_prior1,
        )

    # Further tighten trailing pauses (after last content token) — already covered by
    # pause_max above, but 10 frames keeps pure end-of-chunk silence from feeling long.
    _trail_max = int(req.get("trailing_sp_max_frames", 10))
    if _trail_max > 0:
        not_pause = (~pause_mask1[0]).nonzero(as_tuple=False)
        if len(not_pause) > 0:
            last_content = int(not_pause[-1].item())
            dur_for_prior1[0, last_content + 1:] = (
                dur_for_prior1[0, last_content + 1:].clamp_max(float(_trail_max))
            )
    dur_debug = []
    ids_cpu = ids_full1[0].detach().cpu().tolist()
    dur_cpu = dur_for_prior1[0].detach().cpu().tolist()
    raw_dur_cpu = dur_pred1[0].detach().cpu().tolist()
    allowed_cpu = dur_allowed1[0].detach().cpu().tolist()
    for pos, tok_id in enumerate(ids_cpu):
        sym = str(ID2SYMBOL.get(int(tok_id), f"<id:{int(tok_id)}>"))
        dur_debug.append({
            "pos": int(pos),
            "id": int(tok_id),
            "token": sym,
            "dur": round(float(dur_cpu[pos]), 3),
            "dur_sec": round(float(dur_cpu[pos]) * float(_SECS_PER_FRAME), 4),
            "raw_dur": round(float(raw_dur_cpu[pos]), 3),
            "raw_dur_sec": round(float(raw_dur_cpu[pos]) * float(_SECS_PER_FRAME), 4),
            "allowed": bool(allowed_cpu[pos]),
            "low_duration": bool(allowed_cpu[pos]) and float(dur_cpu[pos]) <= 1.25,
        })

    # ---- Prior mu sampling ----
    prior_noise_scale = float(ckpt_args.get("prior_noise_scale", 1.0))
    t0_btc1, _, _, _ = prior_mu(
        x_tok1,
        dur_for_prior1,
        cond=None,
        T_hint=None,
        noise_scale=prior_noise_scale,
        sp_mask_tok=sp_mask_tok1,
        timbre_vec=spk_hidden,
    )
    x01 = t0_btc1.transpose(1, 2).contiguous()
    gauss_align_btl = getattr(prior_mu, "last_gauss_align_btl", None)
    if torch.is_tensor(gauss_align_btl):
        gauss_align_btl = gauss_align_btl.to(device=device, dtype=torch.float32)
    else:
        gauss_align_btl = None

    # ---- Mel flow (twopass) ----
    text_cond = x_tok1 if bool(CONFIG.get("text_cross_attn", True)) else None
    prefix_k = 0
    prefix_tail = None
    prev_mel = _CONTINUITY_MEL_BY_KEY.get(continuity_key) if continuity_key else None
    if torch.is_tensor(prev_mel) and float(short_continuity_ms) > 0.0:
        prefix_frames = int(_prefix_frames_from_ms(float(short_continuity_ms)))
        prefix_k = int(min(prefix_frames, int(x01.size(-1)), int(prev_mel.size(-1))))
        if prefix_k > 0:
            prefix_tail = prev_mel[:, :, -prefix_k:].to(device=x01.device, dtype=x01.dtype).contiguous()

    if prefix_k > 0 and torch.is_tensor(prefix_tail):
        mel_out1 = _sample_mel_flow_twopass_daemon(
            mel_flow,
            x01,
            text_cond,
            speaker_ids1,
            steps_first=mel_steps_first,
            steps_second=mel_steps_second,
            t_noise=t_noise,
            spk_vec_override=spk_hidden,
            gauss_align_btl=gauss_align_btl,
            prefix_tail_bct=prefix_tail,
            prefix_k=int(prefix_k),
        )
    else:
        mel_out1 = _sample_mel_flow_twopass_daemon(
            mel_flow,
            x01,
            text_cond,
            speaker_ids1,
            steps_first=mel_steps_first,
            steps_second=mel_steps_second,
            t_noise=t_noise,
            spk_vec_override=spk_hidden,
            gauss_align_btl=gauss_align_btl,
        )

    if continuity_key:
        _CONTINUITY_MEL_BY_KEY[continuity_key] = mel_out1.detach().to(device="cpu", dtype=torch.float32)

    # ---- Digital silence on pauses ----
    mel_dec = mel_out1
    silence_mask_dec = None
    if digital_silence:
        silence_mask_dec = _pause_middle_frame_mask_from_durations(
            ids_full=ids_full1,
            dur_values=dur_for_prior1,
            T=int(mel_out1.size(-1)),
            edge_frames=pause_edge,
        )
    if int(prefix_k) > 0:
        if int(mel_dec.size(-1)) <= int(prefix_k):
            raise RuntimeError("continuity prefix consumed the whole generated mel")
        mel_dec = mel_dec[..., int(prefix_k) :].contiguous()
        if torch.is_tensor(silence_mask_dec) and int(silence_mask_dec.size(-1)) > int(prefix_k):
            silence_mask_dec = silence_mask_dec[..., int(prefix_k) :].contiguous()

    # ---- Decode with vocos ----
    wav = _vocos_decode(vocos, mel_dec)
    if wav is None:
        raise RuntimeError("vocos.decode returned None (bad mel?)")
    if digital_silence and torch.is_tensor(silence_mask_dec):
        wav = _apply_digital_silence_to_wav(wav, silence_mask_dec)

    wav = wav.detach().cpu()
    _save_wav(wav_out, wav)
    return {
        "wav": str(wav_out),
        "debug": {
            "text_raw": text,
            "text_prepared": text_prepared,
            "text_in": text_in,
            "continuation_out": bool(continuation_out),
            "token_count": len(ids_cpu),
            "special_len": int(special_len1),
            "pred_frames": int(dur_for_prior1.sum().round().item()),
            "pred_sec": round(float(dur_for_prior1.sum().item()) * float(_SECS_PER_FRAME), 4),
            "mel_frames": int(mel_out1.size(-1)),
            "mel_sec": round(float(mel_out1.size(-1)) * float(_SECS_PER_FRAME), 4),
            "prefix_frames": int(prefix_k),
            "prefix_sec": round(float(prefix_k) * float(_SECS_PER_FRAME), 4),
            "speed": float(speed),
            "leading_sp_min_frames": int(_leading_sp_min),
            "emotion_group": emotion_group,
            "emotion_strength": emotion_strength,
            "emotion_enabled": emotion_enabled,
            "duration_stateful": duration_stateful,
            "learned_voice": learned_voice_enabled,
            "speaker_id": int(speaker_id_for_model),
            "durations": dur_debug,
        },
    }


# ---------------------------------------------------------------------------
# Daemon main loop
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="WęgorzTTS prefix-only speaker_encoder daemon")
    ap.add_argument("--resume", required=True, help="Path to checkpoint .pt")
    ap.add_argument("--dataset-json", required=True, help="Dataset manifest JSON (for speaker list)")
    ap.add_argument("--vocab", required=True, help="Vocabulary JSON path")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = ap.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"[daemon] Loading model from {args.resume} on {device}...", file=sys.stderr, flush=True)
    t0 = time.time()
    components = _load_model(args.resume, device)
    print(f"[daemon] Model loaded in {time.time()-t0:.1f}s", file=sys.stderr, flush=True)

    # Signal ready to parent process
    sys.stdout.write(json.dumps({"status": "ready"}) + "\n")
    sys.stdout.flush()

    # Service loop
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except Exception as exc:
            sys.stdout.write(json.dumps({"error": f"JSON parse error: {exc}"}) + "\n")
            sys.stdout.flush()
            continue

        if req.get("cmd") == "quit":
            break

        try:
            if str(req.get("op", "")).strip() == "encode_ref_mel":
                resp = _encode_ref_audio_to_mel(req, components)
            elif str(req.get("op", "")).strip() == "state_snapshot":
                resp = _snapshot_state(components)
            elif str(req.get("op", "")).strip() == "state_restore":
                resp = _restore_state(components, str(req.get("snapshot_id", "")))
            else:
                resp = synthesize_one(req, components)
        except Exception as exc:
            import traceback
            tb = traceback.format_exc()
            print(f"[daemon] ⚠️ synthesis error:\n{tb}", file=sys.stderr, flush=True)
            resp = {"error": str(exc)}

        sys.stdout.write(json.dumps(resp) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
