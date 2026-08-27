#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from pathlib import Path
from typing import Any

import torch


APP_DIR = Path(__file__).resolve().parent
TTS_DIR = APP_DIR.parent / "tts"
sys.path.insert(0, str(APP_DIR))
sys.path.insert(0, str(TTS_DIR))

from grapheme_tokenizer import cap_decode, cap_encode  # noqa: E402
from model_dualpath_v3 import CONFIGS_DUALPATH_V3, WegorzTranslatorDualPathV3  # noqa: E402


_SENT_BOUNDARY_RE = re.compile(r"(?<=[.!?])\s+(?=[\"'(\[]?[A-Z0-9])")


def clean_ws(text: str) -> str:
    return " ".join(str(text or "").split())


def split_by_commas(text: str, max_chars: int) -> list[str]:
    text = clean_ws(text)
    if len(text) <= max_chars:
        return [text] if text else []

    pieces = re.split(r"(?<=[,;:])\s+", text)
    out: list[str] = []
    cur = ""
    for piece in pieces:
        piece = piece.strip()
        if not piece:
            continue
        candidate = f"{cur} {piece}".strip() if cur else piece
        if len(candidate) <= max_chars:
            cur = candidate
            continue
        if cur:
            out.append(cur)
        if len(piece) <= max_chars:
            cur = piece
        else:
            words = piece.split()
            cur = ""
            for word in words:
                candidate = f"{cur} {word}".strip() if cur else word
                if len(candidate) <= max_chars:
                    cur = candidate
                else:
                    if cur:
                        out.append(cur)
                    cur = word
    if cur:
        out.append(cur)
    return out


def split_segment_text(text: str, max_chars: int) -> list[str]:
    text = clean_ws(text)
    if not text:
        return []
    rough = [p.strip() for p in _SENT_BOUNDARY_RE.split(text) if p.strip()]
    out: list[str] = []
    for part in rough:
        out.extend(split_by_commas(part, max_chars=max_chars))
    return out


class EnNormalizer:
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.norm = None
        if not enabled:
            return
        for name in ("nemo_logger", "nemo_text_processing", "NeMo"):
            logging.getLogger(name).setLevel(logging.ERROR)
        logging.getLogger("nemo_text_processing.text_normalization").setLevel(logging.ERROR)
        import normalize_corpus as corpus_norm

        corpus_norm._worker_init()
        self.norm = corpus_norm

    def __call__(self, text: str) -> str:
        if not self.enabled or self.norm is None:
            return text
        if not self.norm._NEEDS_EN.search(text):
            return text
        out = self.norm._normalize_en(text)
        return clean_ws(out) or text


def load_model(ckpt_path: Path, tokenizer_path: Path, device_name: str):
    import sentencepiece as spm

    device = torch.device("cuda" if device_name.startswith("cuda") and torch.cuda.is_available() else "cpu")
    sp = spm.SentencePieceProcessor(model_file=str(tokenizer_path))
    ckpt = torch.load(str(ckpt_path), map_location=device)
    config_name = ckpt.get("config", "base")
    vocab_size = int(ckpt.get("vocab_size", sp.get_piece_size()))
    model = WegorzTranslatorDualPathV3(vocab_size=vocab_size, **CONFIGS_DUALPATH_V3[config_name])
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    return model, sp, device, config_name, vocab_size


@torch.inference_mode()
def translate_texts(
    model,
    sp,
    device: torch.device,
    texts: list[str],
    batch_size: int,
    max_new_tokens: int,
) -> list[str]:
    out: list[str] = []
    for start in range(0, len(texts), batch_size):
        chunk = texts[start : start + batch_size]
        encoded = [sp.encode(cap_encode(text)) for text in chunk]
        max_len = max(len(row) for row in encoded)
        src = torch.zeros(len(encoded), max_len, dtype=torch.long, device=device)
        for i, row in enumerate(encoded):
            src[i, : len(row)] = torch.tensor(row, dtype=torch.long, device=device)
        pred = model.generate_greedy_batch(src, max_new_tokens=max_new_tokens)
        out.extend(cap_decode(sp.decode(row.tolist())) for row in pred)
    return [clean_ws(text) for text in out]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True, type=Path, help="Parakeet JSON with segments[].text")
    ap.add_argument("--ckpt", required=True, type=Path)
    ap.add_argument("--tokenizer", default=str(APP_DIR / "models/wegorz.model"), type=Path)
    ap.add_argument("--out-json", default="", type=Path)
    ap.add_argument("--out-txt", default="", type=Path)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-chars", type=int, default=220)
    ap.add_argument("--max-new-tokens", type=int, default=180)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no-normalize", action="store_true")
    args = ap.parse_args()

    payload = json.loads(args.json.read_text(encoding="utf-8"))
    segments = payload.get("segments", [])
    if not isinstance(segments, list) or not segments:
        raise RuntimeError("Input JSON must contain non-empty segments[].")

    out_base = args.json.with_name(args.json.stem + "_wegorz_sentence_split")
    out_json = args.out_json if str(args.out_json) not in ("", ".") else out_base.with_suffix(".json")
    out_txt = args.out_txt if str(args.out_txt) not in ("", ".") else out_base.with_suffix(".txt")

    t0 = time.perf_counter()
    normalizer = EnNormalizer(enabled=not args.no_normalize)
    model, sp, device, config_name, vocab_size = load_model(args.ckpt, args.tokenizer, args.device)
    load_sec = time.perf_counter() - t0

    units: list[dict[str, Any]] = []
    merged_segments: list[dict[str, Any]] = []
    for seg_idx, seg in enumerate(segments):
        text = clean_ws(seg.get("text", ""))
        parts = split_segment_text(text, max_chars=args.max_chars)
        unit_ids: list[int] = []
        for part in parts:
            normalized = normalizer(part)
            unit_ids.append(len(units))
            units.append(
                {
                    "segment_index": seg_idx,
                    "text": part,
                    "text_norm": normalized,
                }
            )
        merged_segments.append({**seg, "split_count": len(parts), "_unit_ids": unit_ids})

    translate_t0 = time.perf_counter()
    translations = translate_texts(
        model=model,
        sp=sp,
        device=device,
        texts=[u["text_norm"] for u in units],
        batch_size=max(1, args.batch_size),
        max_new_tokens=max(8, args.max_new_tokens),
    )
    translate_sec = time.perf_counter() - translate_t0

    for unit, tr in zip(units, translations):
        unit["translation_pl"] = tr

    for seg in merged_segments:
        unit_ids = seg.pop("_unit_ids")
        sub = [units[i] for i in unit_ids]
        seg["translation_pl"] = clean_ws(" ".join(u.get("translation_pl", "") for u in sub))
        seg["translation_units"] = [
            {
                "text": u["text"],
                "text_norm": u["text_norm"],
                "translation_pl": u.get("translation_pl", ""),
            }
            for u in sub
        ]

    raw_lines = [f"[{i}] {seg.get('translation_pl', '')}" for i, seg in enumerate(merged_segments, start=1)]
    result = {
        "source_json": str(args.json),
        "model": "wegorz_dualpath_v3",
        "checkpoint": str(args.ckpt),
        "tokenizer": str(args.tokenizer),
        "config": config_name,
        "vocab_size": vocab_size,
        "preprocess": "sentence_split+nemo_en_normalize+cap_encode" if not args.no_normalize else "sentence_split+cap_encode",
        "max_chars": args.max_chars,
        "split_units": len(units),
        "load_seconds": load_sec,
        "translate_seconds": translate_sec,
        "segments": merged_segments,
    }

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_txt.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    out_txt.write_text("\n".join(raw_lines).rstrip() + "\n", encoding="utf-8")

    print(f"segments={len(merged_segments)} split_units={len(units)} max_chars={args.max_chars}")
    print(f"load_sec={load_sec:.2f} translate_sec={translate_sec:.2f} ms_unit={translate_sec / max(1, len(units)) * 1000:.1f}")
    print(f"out_json={out_json}")
    print(f"out_txt={out_txt}")
    print("\n--- PREVIEW ---")
    print("\n".join(raw_lines[:20]))


if __name__ == "__main__":
    main()
