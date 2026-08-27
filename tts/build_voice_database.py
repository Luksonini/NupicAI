#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build a voice database for the current WęgorzTTS speaker_encoder model.

The output is a `voices.pt` dict with:
  - by_id[speaker_id]["emb"] = normalized spk_256 vector
  - groups["PL_F"], groups["PL_M"], ... = speaker ids

Example:
  python tts/build_voice_database.py \
    --checkpoint models/tts/checkpoints/styleenc128_lstm.pt \
    --dataset-json /path/to/manifest.json \
    --out models/tts/voice_banks/selected_top_voices.pt
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

THIS_DIR = Path(__file__).resolve().parent
PAGE_DIR = THIS_DIR.parent

for p in (THIS_DIR, PAGE_DIR):
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

MODEL_PATH = THIS_DIR / "WęgorzTTS3_dubbing_lstm_styleadapters.py"
spec = importlib.util.spec_from_file_location("_wegorz_tts_model_for_voices", str(MODEL_PATH))
model_mod = importlib.util.module_from_spec(spec)
sys.modules["_wegorz_tts_model_for_voices"] = model_mod
spec.loader.exec_module(model_mod)  # type: ignore[union-attr]

N_MELS = int(model_mod.N_MELS)
_ensure_mel_bct = model_mod._ensure_mel_bct
_make_tmask_from_Tlen = model_mod._make_tmask_from_Tlen

from spk_style_dualhead_model import MelDualEncoder  # type: ignore  # noqa: E402


def _load_mel(path: str, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    obj = torch.load(str(Path(path).expanduser()), map_location="cpu", weights_only=False)
    if isinstance(obj, dict):
        mel = obj.get("mel", obj.get("mel_bct", obj.get("x", obj.get("features"))))
        t_val = obj.get("T_mel", obj.get("length", obj.get("T", None)))
    else:
        mel = obj
        t_val = None
    if not torch.is_tensor(mel):
        raise RuntimeError(f"Cannot find mel tensor in {path}")
    mel_bct = _ensure_mel_bct(mel).to(device=device, dtype=torch.float32)
    if t_val is None:
        t_len = torch.tensor([int(mel_bct.size(-1))], device=device, dtype=torch.long)
    elif torch.is_tensor(t_val):
        t_len = torch.tensor([int(t_val.view(-1)[0].item())], device=device, dtype=torch.long)
    else:
        t_len = torch.tensor([int(t_val)], device=device, dtype=torch.long)
    t_len = t_len.clamp(min=1, max=int(mel_bct.size(-1)))
    return mel_bct[:, :, : int(t_len[0].item())].contiguous(), t_len


def _speaker_meta(item: dict[str, Any], sid: int) -> dict[str, Any]:
    raw_name = str(item.get("speaker_name", item.get("author", f"speaker_{sid}"))).strip()
    gender = str(item.get("gender", "U") or "U").upper()
    if gender not in {"F", "M"}:
        if raw_name.endswith("_F"):
            gender = "F"
        elif raw_name.endswith("_M"):
            gender = "M"
        else:
            gender = "U"
    name = raw_name.replace("_F", "").replace("_M", "").strip() or f"speaker_{sid}"
    lang = str(item.get("lang", "unknown") or "unknown").lower()
    return {
        "speaker_id": int(sid),
        "speaker_id_original": int(sid),
        "lang": lang,
        "gender": gender,
        "gender_source": "manifest",
        "name_full": name,
        "name_first": name.split()[0] if name.split() else name,
        "name_last": " ".join(name.split()[1:]) if len(name.split()) > 1 else "",
        "speaker_name_raw": raw_name,
        "synthetic": bool(item.get("synthetic", False)),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Build WęgorzTTS speaker_encoder voices.pt")
    ap.add_argument("--checkpoint", required=True, help="TTS checkpoint containing speaker_encoder")
    ap.add_argument("--dataset-json", required=True, help="Manifest JSON with mel_24k/mel paths")
    ap.add_argument("--out", required=True, help="Output voices.pt")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", choices=["cpu", "cuda"])
    ap.add_argument("--max-mels-per-speaker", type=int, default=8)
    ap.add_argument("--langs", default="pl,en", help="Comma-separated languages to include, or 'all'")
    ap.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=True)
    args = ap.parse_args()

    device = torch.device(args.device)
    checkpoint = Path(args.checkpoint).expanduser()
    dataset_json = Path(args.dataset_json).expanduser()
    out_path = Path(args.out).expanduser()

    payload = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
    ckpt_args = payload.get("args", {}) if isinstance(payload, dict) else {}
    spk_dim = int(ckpt_args.get("spk_dim", 256))

    enc = MelDualEncoder(
        n_mels=N_MELS,
        d_spk=spk_dim,
        d_style=128,
        hidden_dim=256,
        num_layers=4,
        attn_head_dim=128,
        style_project_off_spk=True,
    ).to(device).eval()

    if isinstance(payload, dict) and payload.get("speaker_encoder") is not None:
        enc.load_state_dict(payload["speaker_encoder"], strict=False)
        source = str(checkpoint)
    else:
        spk_style_ckpt = str(ckpt_args.get("spk_style_ckpt", "")).strip()
        if not spk_style_ckpt:
            raise RuntimeError("Checkpoint has no speaker_encoder and no spk_style_ckpt fallback.")
        dh_payload = torch.load(spk_style_ckpt, map_location="cpu", weights_only=False)
        enc.load_state_dict(dh_payload["model"], strict=True)
        source = spk_style_ckpt

    data = json.loads(dataset_json.read_text(encoding="utf-8"))
    allowed_langs = None if str(args.langs).lower().strip() == "all" else {
        x.strip().lower() for x in str(args.langs).split(",") if x.strip()
    }

    by_speaker: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for item in data:
        if not isinstance(item, dict):
            continue
        lang = str(item.get("lang", "") or "").lower()
        if allowed_langs is not None and lang not in allowed_langs:
            continue
        sid_raw = item.get("speaker_id", item.get("speaker", None))
        if sid_raw is None:
            continue
        mel = item.get("mel_24k", item.get("mel", ""))
        if not mel or not Path(str(mel)).exists():
            continue
        by_speaker[int(sid_raw)].append(item)

    by_id: dict[int, dict[str, Any]] = {}
    groups: dict[str, list[int]] = defaultdict(list)

    with torch.no_grad():
        for n, (sid, items) in enumerate(sorted(by_speaker.items()), start=1):
            vecs = []
            chosen = sorted(items, key=lambda it: int(it.get("chunk_idx", it.get("chunk_index", 0)) or 0))
            if int(args.max_mels_per_speaker) > 0:
                chosen = chosen[: int(args.max_mels_per_speaker)]
            for item in chosen:
                try:
                    mel_path = str(item.get("mel_24k", item.get("mel", "")))
                    mel_bct, t_len = _load_mel(mel_path, device)
                    mask_bt = _make_tmask_from_Tlen(t_len, int(mel_bct.size(-1))).squeeze(1).to(
                        device=device,
                        dtype=torch.bool,
                    )
                    z, _ = enc(mel_bct, mask_bt=mask_bt)
                    z = z.detach().float().view(-1)
                    z = z / z.norm().clamp_min(1e-12)
                    vecs.append(z.cpu())
                except Exception as exc:
                    print(f"skip speaker={sid} mel={item.get('mel_24k', item.get('mel', ''))}: {exc}", flush=True)
            if not vecs:
                continue
            emb = torch.stack(vecs, dim=0).mean(dim=0)
            emb = emb / emb.norm().clamp_min(1e-12)
            if bool(args.fp16):
                emb = emb.to(torch.float16)
            meta = _speaker_meta(chosen[0], sid)
            meta["emb"] = emb
            meta["num_mels"] = len(vecs)
            by_id[int(sid)] = meta
            groups[f"{meta['lang'].upper()}_{meta['gender']}"].append(int(sid))
            if n % 100 == 0:
                print(f"encoded speakers: {len(by_id)}", flush=True)

    out = {
        "version": 2,
        "spk_dim": spk_dim,
        "style_dim": 128,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "dataset_json": str(dataset_json),
        "tts_ckpt": str(checkpoint),
        "speaker_encoder_source": source,
        "max_mels_per_spk": int(args.max_mels_per_speaker),
        "groups": {k: sorted(v) for k, v in sorted(groups.items())},
        "by_id": by_id,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, str(out_path))
    print(f"saved: {out_path} speakers={len(by_id)} groups={list(out['groups'].keys())}", flush=True)


if __name__ == "__main__":
    main()
