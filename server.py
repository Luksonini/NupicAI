#!/usr/bin/env python3
"""
Parakeet Transcription + Translation Server — FastAPI with SSE job streaming.

Run:
    python server.py

Then open http://localhost:8765
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import secrets
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncGenerator

import numpy as np
import soundfile as sf

# ── Paths ─────────────────────────────────────────────────────────────────────
HERE = Path(__file__).resolve().parent

TTS_LOCAL      = HERE / "tts"
TRANSLATE_LOCAL = HERE / "translate"
MODELS_LOCAL   = HERE / "models"
YTDLP_DENO     = HERE / "tools/deno/deno"
LEGACY_GRADIO_CONFIG = HERE / "parakeet_config.json"
ADMIN_CONFIG_PATH = HERE / "admin_config.json"
ADMIN_TOKEN = os.environ.get("WEGORZ_ADMIN_TOKEN", "").strip()

# Production TTS profiles. All runtime artifacts must stay inside this folder.
TTS_DAEMON     = TTS_LOCAL / "tts_daemon.py"
_VOICE_BANK_CANDIDATES = [
    Path(os.environ["WEGORZ_VOICE_BANK"]).expanduser() if os.environ.get("WEGORZ_VOICE_BANK") else None,
    MODELS_LOCAL / "tts/voice_banks/selected_top_voices_current.pt",
]
VOICE_BANK = next((p for p in _VOICE_BANK_CANDIDATES if p is not None and p.exists()), None)
if VOICE_BANK is None:
    raise RuntimeError("No production voice bank found. Set WEGORZ_VOICE_BANK or restore models/tts/voice_banks/selected_top_voices_current.pt.")
_TTS_CKPT_CANDIDATES = [
    Path(os.environ["WEGORZ_TTS_CKPT"]).expanduser() if os.environ.get("WEGORZ_TTS_CKPT") else None,
    MODELS_LOCAL / "tts/checkpoints/mini_dualpath_learnedvoice.pt",
]
TTS_CKPT = next((p for p in _TTS_CKPT_CANDIDATES if p is not None and p.exists()), _TTS_CKPT_CANDIDATES[-1])
if TTS_CKPT is None or not Path(TTS_CKPT).exists():
    raise RuntimeError(
        "No usable TTS checkpoint found. Set WEGORZ_TTS_CKPT or restore one of the configured checkpoints."
    )

_STYLEENC128_COMPETE_CKPT = MODELS_LOCAL / "tts/checkpoints/styleenc128_lstm.pt"
_TTS_MODEL_PROFILES_RAW: dict[str, dict[str, Any]] = {
    "mini_dualpath": {
        "label": "MiniDualPath learned voice",
        "description": "learned speaker/style tables + gauss-cross flow + MiniDualPath duration",
        "checkpoint": Path(TTS_CKPT),
    },
    "styleenc128_lstm": {
        "label": "StyleEnc128 LSTM",
        "description": "trainable style encoder checkpoint + stateful LSTM duration",
        "checkpoint": _STYLEENC128_COMPETE_CKPT,
    },
}
TTS_MODEL_PROFILES: dict[str, dict[str, Any]] = {
    key: rec for key, rec in _TTS_MODEL_PROFILES_RAW.items()
    if Path(rec["checkpoint"]).exists()
}
DEFAULT_TTS_PROFILE = str(os.environ.get("WEGORZ_TTS_PROFILE", "styleenc128_lstm")).strip() or "styleenc128_lstm"
if DEFAULT_TTS_PROFILE not in TTS_MODEL_PROFILES:
    DEFAULT_TTS_PROFILE = next(iter(TTS_MODEL_PROFILES.keys()))
TTS_CKPT = Path(TTS_MODEL_PROFILES[DEFAULT_TTS_PROFILE]["checkpoint"])
print(f"✅ TTS checkpoint [{DEFAULT_TTS_PROFILE}]: {TTS_CKPT}", flush=True)
_TTS_DATASET_CANDIDATES = [
    Path(os.environ["WEGORZ_TTS_DATASET_JSON"]).expanduser() if os.environ.get("WEGORZ_TTS_DATASET_JSON") else None,
    TTS_LOCAL / "manifest_runtime_refs.json",
]
TTS_DATASET_JSON = next((p for p in _TTS_DATASET_CANDIDATES if p is not None and p.exists()), _TTS_DATASET_CANDIDATES[-1])
TTS_VOCAB      = TTS_LOCAL / "vocab_pl_orth_en_ipa_bridge.json"
LEARNED_VOICE_SPEAKER_MAP = TTS_LOCAL / "learned_voice_speaker_map.json"

DAEMON_PYTHON  = sys.executable

for _p in (str(TTS_LOCAL), str(TRANSLATE_LOCAL)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

sys.path.insert(0, str(TRANSLATE_LOCAL))
import parakeet_translation_core as core  # noqa: E402

# ── Speaker list (loaded once from voiceprompt manifest) ─────────────────────
_SPEAKER_MEL: dict[str, str] = {}     # label → mel_24k path
_SPEAKER_VOICE_EMB: dict[str, str] = {}  # label → precomputed spk_256 .pt
_SPEAKER_ID_BY_LABEL: dict[str, int] = {}  # label → dataset/learned_voice speaker_id
_RAW_SPEAKER_ID_TO_DENSE: dict[int, int] = {}
_SPEAKER_LIST: list[dict[str, Any]] = []  # [{label, id}] for /speakers endpoint


def _load_learned_voice_speaker_remap() -> dict[int, int]:
    """Mirror training: sorted unique raw speaker_id -> dense learned_voice table row."""
    global _RAW_SPEAKER_ID_TO_DENSE
    if _RAW_SPEAKER_ID_TO_DENSE:
        return _RAW_SPEAKER_ID_TO_DENSE
    try:
        local_map = json.loads(LEARNED_VOICE_SPEAKER_MAP.read_text(encoding="utf-8"))
        if isinstance(local_map, dict) and local_map:
            _RAW_SPEAKER_ID_TO_DENSE = {int(raw): int(dense) for raw, dense in local_map.items()}
            print(
                f"✅ learned_voice speaker remap: {len(_RAW_SPEAKER_ID_TO_DENSE)} IDs from {LEARNED_VOICE_SPEAKER_MAP}",
                flush=True,
            )
            return _RAW_SPEAKER_ID_TO_DENSE
    except Exception as exc:
        print(f"⚠️ Cannot load local learned_voice speaker map: {exc}", flush=True)
    try:
        obj = json.loads(Path(TTS_DATASET_JSON).read_text(encoding="utf-8"))
        if isinstance(obj, dict):
            items = obj.get("items") or obj.get("data") or obj.get("manifest") or []
        else:
            items = obj
        raw_ids: list[int] = []
        if isinstance(items, list):
            for it in items:
                if not isinstance(it, dict) or "speaker_id" not in it:
                    continue
                try:
                    raw_ids.append(int(it.get("speaker_id")))
                except Exception:
                    pass
        _RAW_SPEAKER_ID_TO_DENSE = {sid: i for i, sid in enumerate(sorted(set(raw_ids)))}
        if _RAW_SPEAKER_ID_TO_DENSE:
            print(
                f"✅ learned_voice speaker remap: {len(_RAW_SPEAKER_ID_TO_DENSE)} raw IDs from {TTS_DATASET_JSON}",
                flush=True,
            )
    except Exception as exc:
        print(f"⚠️ Cannot build learned_voice speaker remap from {TTS_DATASET_JSON}: {exc}", flush=True)
        _RAW_SPEAKER_ID_TO_DENSE = {}
    return _RAW_SPEAKER_ID_TO_DENSE


def _dense_speaker_id(raw_sid: Any) -> int:
    try:
        sid = int(raw_sid)
    except Exception:
        return 0
    return int(_load_learned_voice_speaker_remap().get(sid, sid))


def _init_voice_bank_speakers() -> bool:
    global _SPEAKER_VOICE_EMB, _SPEAKER_ID_BY_LABEL, _SPEAKER_LIST
    if VOICE_BANK is None or not Path(VOICE_BANK).exists():
        return False
    try:
        import torch

        bank = torch.load(str(VOICE_BANK), map_location="cpu", weights_only=False)
        by_id = bank.get("by_id", {}) if isinstance(bank, dict) else {}
    except Exception as exc:
        print(f"⚠️ Cannot load voice bank {VOICE_BANK}: {exc}", flush=True)
        return False

    emb_dir = Path(tempfile.gettempdir()) / "parakeet_server" / "voice_bank_embs"
    emb_dir.mkdir(parents=True, exist_ok=True)

    result: list[dict[str, Any]] = []
    for idx, (sid, rec) in enumerate(sorted(by_id.items(), key=lambda kv: int((kv[1] or {}).get("selected_rank") or 999999))):
        if not isinstance(rec, dict):
            continue
        emb = rec.get("emb")
        if emb is None:
            continue
        rank = int(rec.get("selected_rank") or (idx + 1))
        gender_raw = str(rec.get("gender", "U")).upper()
        gender = "K" if gender_raw == "F" else ("M" if gender_raw == "M" else "U")
        lang = str(rec.get("lang", "") or "").upper()
        name = str(rec.get("speaker_name") or f"speaker_{sid}").replace("_F", "").replace("_M", "").strip()
        quality = rec.get("quality", None)
        qtxt = f", q={float(quality):.2f}" if quality is not None else ""
        label = f"[top {rank:02d}] {name} ({gender}, {lang}{qtxt}) [{sid}]"
        emb_path = emb_dir / f"voice_{sid}.pt"
        torch.save(
            {
                "emb": emb.detach().to(dtype=torch.float32).cpu() if hasattr(emb, "detach") else emb,
                "speaker_id": int(sid),
                "speaker_name": name,
                "gender": gender_raw,
                "lang": str(rec.get("lang", "")),
                "quality": quality,
            },
            str(emb_path),
        )
        _SPEAKER_VOICE_EMB[label] = str(emb_path)
        dense_sid = _dense_speaker_id(sid)
        _SPEAKER_ID_BY_LABEL[label] = dense_sid
        result.append({"label": label, "id": dense_sid, "raw_id": int(sid), "voice_bank": True, "rank": rank})

    if not result:
        return False
    _SPEAKER_LIST = result
    print(f"✅ Voice bank: {len(result)} głosów z {VOICE_BANK}", flush=True)
    return True


def _init_speakers() -> None:
    global _SPEAKER_MEL, _SPEAKER_ID_BY_LABEL, _SPEAKER_LIST
    if _init_voice_bank_speakers():
        return
    try:
        data_obj = json.loads(TTS_DATASET_JSON.read_text(encoding="utf-8"))
        if isinstance(data_obj, dict):
            data = data_obj.get("items") or data_obj.get("data") or data_obj.get("manifest") or []
        else:
            data = data_obj
    except Exception as exc:
        print(f"⚠️ Cannot load speaker manifest: {exc}", flush=True)
        return

    spk_best: dict[int, dict] = {}
    spk_has_nonzero: dict[int, bool] = {}

    for it in data:
        if it.get("lang") != "pl":
            continue
        sid = it.get("speaker_id")
        if sid is None:
            continue
        mel = it.get("mel_24k", "")
        if not mel or not Path(mel).exists():
            continue
        chunk = int(it.get("chunk_idx") or 0)
        if sid not in spk_best:
            spk_best[sid] = it
            spk_has_nonzero[sid] = chunk > 0
        elif chunk > 0 and not spk_has_nonzero[sid]:
            spk_best[sid] = it
            spk_has_nonzero[sid] = True

    result: list[dict[str, Any]] = []
    for idx, (sid, it) in enumerate(sorted(spk_best.items())):
        raw_name = str(it.get("speaker_name", f"ID:{sid}")).replace("_F", "").replace("_M", "").strip()
        gender = "K" if it.get("gender") == "F" else "M"
        label = f"{raw_name} ({gender})"
        mel_path = it["mel_24k"]
        _SPEAKER_MEL[label] = mel_path
        dense_sid = _dense_speaker_id(sid)
        _SPEAKER_ID_BY_LABEL[label] = dense_sid
        result.append({"label": label, "id": dense_sid, "raw_id": int(sid), "list_index": idx})

    result.sort(key=lambda x: str(x["label"]).lower())
    _SPEAKER_LIST = result
    print(f"✅ Lektorzy: {len(result)} PL", flush=True)


_init_speakers()


# ── TTS daemon management ─────────────────────────────────────────────────────
_daemon_procs: dict[str, subprocess.Popen] = {}
_daemon_active_profile: str | None = None
_daemon_lock = threading.Lock()


def _resolve_tts_profile(profile: str | None) -> tuple[str, Path]:
    key = str(profile or DEFAULT_TTS_PROFILE).strip() or DEFAULT_TTS_PROFILE
    if key not in TTS_MODEL_PROFILES:
        key = DEFAULT_TTS_PROFILE
    return key, Path(TTS_MODEL_PROFILES[key]["checkpoint"])


def _stop_daemon_locked() -> None:
    global _daemon_active_profile
    procs = list(_daemon_procs.items())
    _daemon_procs.clear()
    _daemon_active_profile = None
    for _profile, proc in procs:
        try:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()
        except Exception:
            pass


def _start_daemon_locked(profile: str | None = None) -> None:
    global _daemon_active_profile
    profile_key, ckpt_path = _resolve_tts_profile(profile)
    old = _daemon_procs.get(profile_key)
    if old is not None and old.poll() is None:
        _daemon_active_profile = profile_key
        return
    pythonpath = os.pathsep.join([str(TTS_LOCAL), str(TRANSLATE_LOCAL), str(HERE), os.environ.get("PYTHONPATH", "")])
    env = {**os.environ, "PYTHONPATH": pythonpath}
    if "WEGORZ_TTS_MODEL_MODULE" not in env:
        env["WEGORZ_TTS_MODEL_MODULE"] = str(TTS_LOCAL / "wegorz_tts_model.py")
    proc = subprocess.Popen(
        [
            DAEMON_PYTHON, str(TTS_DAEMON),
            "--resume", str(ckpt_path),
            "--dataset-json", str(TTS_DATASET_JSON),
            "--vocab", str(TTS_VOCAB),
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=sys.stderr,
        env=env,
        text=True,
        bufsize=1,
    )
    # Wait for "ready" line (model loading can take 30–60 s)
    print(f"🔄 Uruchamiam daemon TTS [{profile_key}]…", flush=True)
    while True:
        line = proc.stdout.readline()
        if not line:
            proc.kill()
            raise RuntimeError("Daemon exited before sending ready signal")
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except Exception:
            print(f"[daemon-out] {line}", flush=True)
            continue
        if msg.get("status") == "ready":
            print(f"✅ Daemon TTS gotowy [{profile_key}].", flush=True)
            break
        print(f"[daemon] {line}", flush=True)
    _daemon_procs[profile_key] = proc
    _daemon_active_profile = profile_key


def _ensure_daemon_locked(profile: str | None = None) -> None:
    global _daemon_active_profile
    profile_key, _ckpt_path = _resolve_tts_profile(profile)
    proc = _daemon_procs.get(profile_key)
    if proc is not None and proc.poll() is None:
        _daemon_active_profile = profile_key
        return
    _start_daemon_locked(profile_key)


def _daemon_call(req: dict[str, Any]) -> dict[str, Any]:
    """Send request to daemon, return JSON response. Thread-safe."""
    req = dict(req)
    tts_profile = req.pop("tts_model_profile", None)
    with _daemon_lock:
        profile_key, _ckpt_path = _resolve_tts_profile(tts_profile)
        _ensure_daemon_locked(profile_key)
        proc = _daemon_procs.get(profile_key)
        if proc is None or proc.poll() is not None:
            raise RuntimeError(f"TTS daemon for profile={profile_key!r} is not running")
        proc.stdin.write(json.dumps(req, ensure_ascii=False) + "\n")
        proc.stdin.flush()
        # Read response (blocks until synthesis completes). Some lazy-loaded
        # normalizers may write informational lines before the JSON response.
        while True:
            resp_line = proc.stdout.readline()
            if not resp_line:
                raise RuntimeError("Daemon closed stdout unexpectedly")
            line = resp_line.strip()
            if not line:
                continue
            try:
                resp = json.loads(line)
                break
            except Exception:
                print(f"[daemon-out] {line}", flush=True)
                continue
        if "error" in resp:
            raise RuntimeError(f"Daemon error: {resp['error']}")
        return dict(resp)


def _daemon_synth_response(req: dict[str, Any]) -> dict[str, Any]:
    return _daemon_call(req)


def _daemon_synth(req: dict[str, Any]) -> str:
    """Send synthesis request to daemon, return WAV path. Thread-safe."""
    return str(_daemon_synth_response(req)["wav"])


def _daemon_state_snapshot(profile: str | None = None) -> str:
    return str(_daemon_call({"op": "state_snapshot", "tts_model_profile": profile}).get("snapshot_id", ""))


def _daemon_state_restore(snapshot_id: str, profile: str | None = None) -> None:
    _daemon_call({"op": "state_restore", "snapshot_id": str(snapshot_id), "tts_model_profile": profile})


def _daemon_encode_ref_mel(audio_path: Path, out_dir: Path, *, start_sec: float = 0.0, max_sec: float = 12.0) -> dict[str, Any]:
    return _daemon_call({
        "op": "encode_ref_mel",
        "audio": str(audio_path),
        "out_dir": str(out_dir),
        "tag": f"voice_prompt_{uuid.uuid4().hex[:10]}",
        "start_sec": float(start_sec),
        "max_sec": float(max_sec),
    })


# Training data: median=5.7s/76ch, p90=9.6s/176ch, p95=10.7s/197ch.
# Splitting at clause boundaries (commas) causes BiLSTM to predict sentence-initial
# pauses at chunk starts (0.15-0.2s dead air) which sounds like words being swallowed.
# Prefer whole sentences in free-text TTS; PL p99 is ~294 chars, and the old
# Polish-only Gradio used sentence chunks. Mid-sentence chunks are a stronger
# hallucination risk than moderately long sentences.
_TTS_MAX_CHARS = 120
_TTS_HARD_SENTENCE_CHARS = 120
_SENT_END_RE = re.compile(r'(?<=[.!?…])\s+')
_CLAUSE_END_RE = re.compile(r'(?<=[,;:])\s+')
_SHORT_OPENING_MAX_CHARS = 20
_COMMON_ABBREVIATIONS = frozenset({
    "dr.", "mgr.", "inż.", "prof.", "ul.", "al.", "godz.", "min.", "sek.",
    "mr.", "mrs.", "ms.", "prof.", "st.", "vs.", "etc.",
})

# Words that must NOT end a TTS chunk — they require continuation to sound natural.
# BiLSTM hallucinates when it sees these immediately before <EOS>.
_PL_OPEN_WORDS = frozenset({
    # spójniki
    "i", "a", "ale", "oraz", "czy", "więc", "lecz", "jednak", "też", "albo",
    "ani", "bądź", "że", "żeby", "by", "aby", "iż", "gdyby", "jeśli", "jeżeli",
    "choć", "chociaż", "gdyż", "bo", "zaś", "natomiast", "dlatego", "zatem",
    # przyimki
    "w", "we", "z", "ze", "do", "od", "na", "po", "przy", "przed", "nad",
    "pod", "za", "o", "u", "ku", "przez", "między", "poza", "wśród",
    # zaimki względne / spójnikowe
    "który", "która", "które", "których", "którą", "którym", "którymi",
    "gdzie", "kiedy", "jak", "skąd", "dokąd",
    # inne otwierające
    "nie", "się", "to",
})


_TTS_MIN_CHUNK = 20  # chunks shorter than this get merged into previous
_NEWS_DEMO_CHUNKS_PL = [
    "Dzisiejszego wieczoru napływają doniesienia, ",
    "że amerykańscy i irańscy negocjatorzy osiągnęli porozumienie ",
    "w sprawie przedłużenia zawieszenia broni oraz rozpoczęcia negocjacji dotyczących programu nuklearnego Iranu. ",
    "Donald Trump musi jednak wciąż zatwierdzić jakąkolwiek umowę ",
    "w obliczu wymiany ognia między obiema stronami, która zagraża obecnemu rozejmowi. ",
    "Irański Korpus Strażników Rewolucji Islamskiej twierdzi, ",
    "że obrał za cel amerykańską bazę lotniczą w tym regionie. ",
]
_NEWS_DEMO_TEXT_NORM = " ".join("".join(_NEWS_DEMO_CHUNKS_PL).split()).lower()


def _looks_sentence_final(text: str) -> bool:
    return bool(re.search(r'[.!?…]["”’)\]]*\s*$', str(text or "").strip()))


def _fix_open_tails(chunks: list[str]) -> list[str]:
    """Move trailing open words (conjunctions/prepositions) to the start of the next chunk,
    then merge any tiny trailing chunk into its predecessor."""
    out = list(chunks)
    for i in range(len(out) - 1):
        words = out[i].split()
        while len(words) > 1 and words[-1].lower().rstrip(",:;") in _PL_OPEN_WORDS:
            out[i + 1] = words.pop() + " " + out[i + 1]
        out[i] = " ".join(words)
    # Merge tiny chunks (e.g. "regionie.") into the previous chunk
    result: list[str] = []
    for c in out:
        c = c.strip()
        if not c:
            continue
        if result and len(c) < _TTS_MIN_CHUNK:
            result[-1] = result[-1] + " " + c
        else:
            result.append(c)
    return result


def _split_text_for_tts(
    text: str,
    max_chars: int = _TTS_MAX_CHARS,
    hard_sentence_chars: int = _TTS_HARD_SENTENCE_CHARS,
) -> list[str]:
    """
    Split text at sentence boundaries keeping each chunk ≤ max_chars.
    Clause splits (comma/semicolon) are only used as a last resort for very
    long sentences — splitting at clauses causes the duration predictor to
    insert sentence-initial pauses at chunk junctions.
    """
    text = text.strip()
    if not text:
        return []
    if " ".join(text.split()).lower() == _NEWS_DEMO_TEXT_NORM:
        return [c.strip() for c in _NEWS_DEMO_CHUNKS_PL if c.strip()]

    # A very short sentence at the start of a longer request (for example
    # "Tak. Dlaczego nie?") can be acoustically suppressed by the flow model.
    # Render that complete utterance separately; abbreviations are not sentences.
    opening_parts = _SENT_END_RE.split(text, maxsplit=1)
    if len(opening_parts) == 2:
        opening, remainder = (part.strip() for part in opening_parts)
        if (
            opening
            and remainder
            and len(opening) <= _SHORT_OPENING_MAX_CHARS
            and len(opening.split()) <= 3
            and opening.lower() not in _COMMON_ABBREVIATIONS
        ):
            return [opening, *_split_text_for_tts(
                remainder,
                max_chars=max_chars,
                hard_sentence_chars=hard_sentence_chars,
            )]
    if len(text) <= max_chars:
        return [text]

    def _greedy_merge(parts: list[str]) -> list[str]:
        chunks: list[str] = []
        cur = ""
        for p in parts:
            p = p.strip()
            if not p:
                continue
            candidate = f"{cur} {p}".strip() if cur else p
            if len(candidate) <= max_chars or not cur:
                cur = candidate
            else:
                chunks.append(cur)
                cur = p
        if cur:
            chunks.append(cur)
        return chunks

    def _split_words_hard(piece: str, limit: int) -> list[str]:
        piece = piece.strip()
        if not piece:
            return []
        out: list[str] = []
        while len(piece) > limit:
            cut = piece.rfind(" ", 0, limit)
            if cut <= 0:
                cut = limit
            out.append(piece[:cut].strip())
            piece = piece[cut:].strip()
        if piece:
            out.append(piece)
        return out

    # Step 1: try sentence-end splits only. Keep a sentence whole up to the
    # hard limit even when it exceeds max_chars; this matches the older PL TTS
    # path and avoids broken phrase boundaries such as "...broni | oraz...".
    sent_parts = _SENT_END_RE.split(text)
    if len(sent_parts) > 1:
        chunks = _greedy_merge(sent_parts)
        if all(len(c) <= max_chars for c in chunks):
            return _fix_open_tails([c for c in chunks if c.strip()])
        sentence_safe: list[str] = []
        must_split = False
        for ch in chunks:
            if len(ch) <= hard_sentence_chars:
                sentence_safe.append(ch)
            else:
                must_split = True
                break
        if not must_split:
            return _fix_open_tails([c for c in sentence_safe if c.strip()])
        # Some sentence is still too long — split those long pieces at clause boundaries.
        result: list[str] = []
        for ch in chunks:
            if len(ch) <= hard_sentence_chars:
                result.append(ch)
            else:
                clause_parts = _CLAUSE_END_RE.split(ch)
                for c in _greedy_merge(clause_parts):
                    if len(c) > max_chars:
                        result.extend(_split_words_hard(c, max_chars))
                    elif c.strip():
                        result.append(c)
        return _fix_open_tails([c for c in result if c.strip()])

    # Step 2: clause splits for text with no sentence boundaries.
    clause_parts = _CLAUSE_END_RE.split(text)
    if len(clause_parts) > 1:
        chunks = _greedy_merge(clause_parts)
        if all(len(c) <= max_chars for c in chunks):
            return _fix_open_tails([c for c in chunks if c.strip()])
        result = []
        for ch in chunks:
            if len(ch) > max_chars:
                result.extend(_split_words_hard(ch, max_chars))
            elif ch.strip():
                result.append(ch)
        return _fix_open_tails([c for c in result if c.strip()])

    # Hard fallback: cut on word boundary.
    chunks = []
    while len(text) > max_chars:
        cut = text.rfind(" ", 0, max_chars)
        if cut <= 0:
            cut = max_chars
        chunks.append(text[:cut].strip())
        text = text[cut:].strip()
    if text:
        chunks.append(text)
    return _fix_open_tails([c for c in chunks if c.strip()])


def _daemon_synth_chunked_response(base_req: dict[str, Any], out_dir: str, tag: str) -> dict[str, Any]:
    """
    Synthesize potentially long text by splitting into chunks, synthesizing each
    with continuity, and concatenating the resulting WAVs. Returns final WAV path.
    Falls back to direct synthesis when text is short enough.
    """
    text = str(base_req.get("text", "")).strip()
    max_chars = int(base_req.get("tts_max_chars") or _TTS_MAX_CHARS)
    hard_sentence_chars = int(base_req.get("tts_hard_sentence_chars") or _TTS_HARD_SENTENCE_CHARS)
    chunks = _split_text_for_tts(text, max_chars=max_chars, hard_sentence_chars=hard_sentence_chars)
    if len(chunks) <= 1:
        resp = _daemon_synth_response(base_req)
        resp.setdefault("chunks", [{"text": text, "wav": str(resp.get("wav", "")), "debug": resp.get("debug", {})}])
        return resp

    SR = 24000
    audios: list[np.ndarray] = []
    debug_chunks: list[dict[str, Any]] = []
    for i, chunk in enumerate(chunks):
        # If an internal chunk does not end like a sentence, tell the daemon to
        # end it with <sp> rather than <EOS>. This avoids training/inference
        # mismatch for technical mid-sentence splits.
        continuation_out = (i < len(chunks) - 1) and (not _looks_sentence_final(chunk))
        req = {
            **base_req,
            "text": chunk,
            "tag": f"{tag}_c{i:02d}",
            "continuity_reset": (i == 0) or bool(base_req.get("continuity_reset", False)),
            "continuation_out": bool(continuation_out),
        }
        if i > 0:
            req["continuity_reset"] = False
        resp = _daemon_synth_response(req)
        wav_path = str(resp["wav"])
        audio, file_sr = sf.read(wav_path)
        if audio.ndim > 1:
            audio = audio[:, 0]
        audio = audio.astype(np.float32)
        if file_sr != SR:
            ratio = SR / file_sr
            audio = np.interp(
                np.arange(0, len(audio) * ratio, ratio),
                np.arange(len(audio)), audio,
            ).astype(np.float32)
        audios.append(audio)
        debug_chunks.append({
            "text": chunk,
            "continuation_out": bool(continuation_out),
            "wav": wav_path,
            "debug": resp.get("debug", {}),
        })

    combined = np.concatenate(audios)
    out_path = Path(out_dir) / f"{tag}.wav"
    sf.write(str(out_path), combined, SR)
    return {"wav": str(out_path), "chunks": debug_chunks}


def _daemon_synth_chunked(base_req: dict[str, Any], out_dir: str, tag: str) -> str:
    return str(_daemon_synth_chunked_response(base_req, out_dir, tag)["wav"])


def _summarize_tts_debug(debug_info: dict[str, Any], *, text: str, budget_sec: float, wav_sec: float) -> dict[str, Any]:
    """Build a compact, UI-friendly explanation of risky TTS duration decisions."""
    chunks = list(debug_info.get("chunks") or [])
    total_tokens = 0
    low_tokens: list[dict[str, Any]] = []
    chunk_summaries: list[dict[str, Any]] = []

    for ci, ch in enumerate(chunks):
        dbg = dict(ch.get("debug") or {})
        durs = list(dbg.get("durations") or [])
        allowed = [d for d in durs if bool(d.get("allowed"))]
        lows = [
            {
                "chunk": ci + 1,
                "token": str(d.get("token", "")),
                "dur": d.get("dur"),
                "dur_sec": d.get("dur_sec"),
                "pos": d.get("pos"),
            }
            for d in allowed
            if bool(d.get("low_duration")) or float(d.get("dur") or 0.0) <= 1.25
        ]
        total_tokens += len(allowed)
        low_tokens.extend(lows)
        chunk_summaries.append({
            "chunk": ci + 1,
            "text_len": len(str(ch.get("text") or "")),
            "token_count": int(dbg.get("token_count") or 0),
            "allowed_tokens": len(allowed),
            "pred_sec": dbg.get("pred_sec"),
            "mel_sec": dbg.get("mel_sec"),
            "prefix_sec": dbg.get("prefix_sec"),
            "low_token_count": len(lows),
        })

    warnings: list[str] = []
    if len(chunks) > 1:
        warnings.append(f"Tekst podzielony wewnętrznie na {len(chunks)} chunki TTS.")
    if low_tokens:
        preview = ", ".join(str(t["token"]) for t in low_tokens[:12])
        warnings.append(f"{len(low_tokens)} tokenów ma bardzo małą durację: {preview}")
    if budget_sec > 0 and wav_sec > budget_sec + 0.02:
        warnings.append(f"Audio przekracza budżet o {wav_sec - budget_sec:.3f}s.")
    if budget_sec > 0 and wav_sec < max(0.05, budget_sec * 0.55):
        warnings.append("Audio jest podejrzanie krótkie względem budżetu.")

    return {
        "text_len": len(text),
        "chunk_count": len(chunks),
        "total_allowed_tokens": total_tokens,
        "low_token_count": len(low_tokens),
        "low_tokens": low_tokens[:80],
        "chunks": chunk_summaries,
        "warnings": warnings,
    }


def _resolve_speaker_mel(label: str) -> str:
    """Return mel .pt path for speaker label. Falls back to first speaker."""
    mel = _SPEAKER_MEL.get(label, "")
    if mel and Path(mel).exists():
        return mel
    if _SPEAKER_MEL:
        fallback = next(iter(_SPEAKER_MEL.values()))
        print(f"⚠️ Speaker {label!r} not found, using fallback", flush=True)
        return fallback
    raise RuntimeError("No speakers available in manifest")


def _speaker_condition_payload(label: str) -> dict[str, Any]:
    """Return daemon speaker conditioning payload for selected UI label."""
    payload: dict[str, Any] = {}
    if label in _SPEAKER_ID_BY_LABEL:
        payload["speaker_id"] = int(_SPEAKER_ID_BY_LABEL[label])
    voice_emb = _SPEAKER_VOICE_EMB.get(label, "")
    if voice_emb and Path(voice_emb).exists():
        payload["voice_emb"] = voice_emb
        return payload
    payload["ref_mel"] = _resolve_speaker_mel(label)
    return payload


def _convert_media_to_mono24k(src: Path, out_dir: Path) -> Path:
    core.require_executable("ffmpeg")
    out_dir.mkdir(parents=True, exist_ok=True)
    dst = out_dir / f"voice_prompt_{uuid.uuid4().hex[:12]}_mono24k.wav"
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(src), "-vn", "-ac", "1", "-ar", "24000", "-sample_fmt", "s16", str(dst)],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    return dst


# ── Config ──────────────────────────────────────────────────────────────────
_WORK = Path(tempfile.gettempdir()) / "parakeet_server"
_WORK.mkdir(parents=True, exist_ok=True)

core._CONFIG = {
    "device": "cuda",
    "parakeet_model": str(MODELS_LOCAL / "asr/parakeet-tdt-0.6b-v3.nemo"),
    "asr_timestamp_window_seconds": 180.0,
    "asr_timestamp_min_window_seconds": 1.0,
    "asr_final_tail_pad_seconds": float(os.environ.get("PARAKEET_ASR_FINAL_TAIL_PAD_SEC", "1.2")),
    "asr_window_seconds": float(os.environ.get("PARAKEET_ASR_WINDOW_SEC", "180.0")),
    "asr_window_overlap_seconds": float(os.environ.get("PARAKEET_ASR_OVERLAP_SEC", "2.0")),
    "asr_wordseg_max_chars": 180,
    "translation_endpoint": os.environ.get("TRANSLATION_ENDPOINT", "https://ai.nupic.homes/v1"),
    "translation_model": os.environ.get("TRANSLATION_MODEL", "qwen3.5:35b-mtp"),
    "translation_mode": os.environ.get("TRANSLATION_MODE", "qwen_mtp_35b_json_overlap"),
    "translation_batch_segments": int(os.environ.get("TRANSLATION_BATCH_SEGMENTS", "8")),
    "translation_api_key": "",
    "translation_target_lang": "pl",
    "translation_source_lang": "auto",
    "translation_temperature": 0.1,
    "translation_timeout_seconds": 180,
    "translation_retry": 2,
    "work_dir": str(_WORK),
    "outputs_dir": str(_WORK / "outputs"),
    "wegorz_ckpt": str(MODELS_LOCAL / "translate/wegorz_translator_32k_best.pt"),
    "wegorz_tokenizer": str(TRANSLATE_LOCAL / "wegorz.model"),
}


def _load_legacy_translation_api_key() -> str:
    try:
        data = json.loads(LEGACY_GRADIO_CONFIG.read_text(encoding="utf-8"))
        return str(data.get("translation_api_key", "")).strip()
    except Exception:
        return ""


core._CONFIG["translation_api_key"] = (
    os.environ.get("NUPIC_API_KEY", "").strip()
    or _load_legacy_translation_api_key()
)


def _load_admin_config() -> None:
    try:
        data = json.loads(ADMIN_CONFIG_PATH.read_text(encoding="utf-8"))
    except Exception:
        return
    if not isinstance(data, dict):
        return
    for key in ("translation_endpoint", "translation_model", "translation_mode"):
        value = str(data.get(key, "")).strip()
        if value:
            core._CONFIG[key] = value
    if data.get("translation_batch_segments") is not None:
        core._CONFIG["translation_batch_segments"] = max(1, min(20, int(data["translation_batch_segments"])))
    saved_key = str(data.get("translation_api_key", "")).strip()
    if saved_key and not os.environ.get("NUPIC_API_KEY", "").strip():
        core._CONFIG["translation_api_key"] = saved_key


def _save_admin_config() -> None:
    payload = {
        "translation_endpoint": str(core._CONFIG.get("translation_endpoint", "")),
        "translation_model": str(core._CONFIG.get("translation_model", "")),
        "translation_mode": str(core._CONFIG.get("translation_mode", "")),
        "translation_batch_segments": int(core._CONFIG.get("translation_batch_segments", 8)),
        "translation_api_key": str(core._CONFIG.get("translation_api_key", "")),
    }
    tmp = ADMIN_CONFIG_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.chmod(tmp, 0o600)
    tmp.replace(ADMIN_CONFIG_PATH)


_load_admin_config()

from fastapi import BackgroundTasks, FastAPI, File, Form, Header, HTTPException, UploadFile  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.responses import FileResponse, StreamingResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402
from pydantic import BaseModel  # noqa: E402
import uvicorn  # noqa: E402

UI_OUT = HERE / "parakeet-ui" / "out"

# ── Job system ─────────────────────────────────────────────────────────────
_executor = ThreadPoolExecutor(max_workers=2)
_loop: asyncio.AbstractEventLoop | None = None


@dataclass
class Job:
    id: str
    kind: str
    status: str = "pending"
    progress: float = 0.0
    message: str = ""
    result: dict[str, Any] | None = None
    error: str | None = None
    queue: asyncio.Queue = field(default_factory=asyncio.Queue)
    created_at: float = field(default_factory=time.time)


_jobs: dict[str, Job] = {}
_MODEL_READY = asyncio.Event()
_TTS_READY = asyncio.Event()


def _push(job: Job, event: dict[str, Any]) -> None:
    if _loop is not None:
        asyncio.run_coroutine_threadsafe(job.queue.put(event), _loop)


# ── Segmentation ────────────────────────────────────────────────────────────
_SENTENCE_END = re.compile(r"[.!?…]+[\"')\]]*$")
_COMMA_END    = re.compile(r"[,;:]+[\"')\]]*$")


def _wval(w: dict) -> str:
    return str(w.get("word") or w.get("text") or "").strip()


def _wtext(words: list[dict]) -> str:
    return " ".join(_wval(w) for w in words if _wval(w))


def _gap(a: dict, b: dict) -> float:
    return max(0.0, float(b.get("start", 0.0)) - float(a.get("end", 0.0)))


def _validate_segment_timeline(segments: list[dict[str, Any]], *, stage: str) -> None:
    if not segments:
        raise ValueError(f"{stage}: brak segmentów do przetworzenia")
    previous_start = -1.0
    for pos, seg in enumerate(segments):
        start = float(seg.get("start", 0.0))
        end = float(seg.get("end", start))
        if start < -1e-6 or end + 1e-6 < start:
            raise ValueError(f"{stage}: nieprawidłowy czas segmentu {pos}: start={start}, end={end}")
        if start + 1e-6 < previous_start:
            raise ValueError(f"{stage}: segmenty nie są uporządkowane czasowo przy pozycji {pos}")
        if not str(seg.get("source_text") or seg.get("text") or "").strip():
            raise ValueError(f"{stage}: pusty tekst źródłowy w segmencie {pos}")
        previous_start = start


def build_segments(
    words: list[dict],
    *,
    min_sec: float = 2.0,
    soft_max: float = 9.5,
    hard_max: float = 11.5,
    max_words: int = 32,
    gap_split_sec: float = 0.75,
    max_chars: int = 180,
) -> list[dict]:
    segs: list[dict] = []
    buf: list[dict] = []

    def emit(part: list[dict]) -> None:
        text = _wtext(part)
        if text:
            segs.append({
                "index": len(segs),
                "start": round(float(part[0].get("start", 0.0)), 3),
                "end":   round(float(part[-1].get("end", part[0].get("start", 0.0))), 3),
                "text":  text,
                "words": list(part),
            })

    def split_emit() -> None:
        nonlocal buf
        if not buf:
            return
        t0 = float(buf[0].get("start", 0.0))
        last_ok = max(0, len(buf) - 2)

        def ok(i: int) -> bool:
            return i <= last_ok and float(buf[i].get("end", t0)) - t0 >= min_sec

        idx: int | None = None
        for pat in (_SENTENCE_END, _COMMA_END):
            cands = [i for i, w in enumerate(buf[:-1]) if ok(i) and pat.search(_wval(w))]
            if cands:
                idx = cands[-1]
                break
        if idx is None:
            gaps = [(_gap(buf[i], buf[i + 1]), i) for i in range(len(buf) - 1) if ok(i)]
            strong_gaps = [item for item in gaps if item[0] >= float(gap_split_sec)]
            if strong_gaps:
                idx = max(strong_gaps)[1]
            elif gaps:
                idx = max(gaps)[1]
        idx = idx if idx is not None else max(0, len(buf) - 2)
        emit(buf[: idx + 1])
        buf = buf[idx + 1:]

    for raw in sorted(words, key=lambda w: (float(w.get("start", 0.0)), float(w.get("end", 0.0)))):
        if not _wval(raw):
            continue
        buf.append(raw)
        while buf:
            t0 = float(buf[0].get("start", 0.0))
            t1 = float(buf[-1].get("end", t0))
            dur = t1 - t0
            text = _wtext(buf)
            last = _wval(buf[-1])
            if dur >= min_sec and _SENTENCE_END.search(last):
                emit(buf); buf = []; break
            if dur >= soft_max and _COMMA_END.search(last):
                emit(buf); buf = []; break
            if dur >= hard_max or len(buf) >= int(max_words) or len(text) >= int(max_chars):
                if len(buf) <= 1:
                    break
                split_emit()
                if not buf:
                    break
                continue
            break
    if buf:
        emit(buf)
    return segs


def _ffprobe_dur(path: Path) -> float:
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
            capture_output=True, text=True,
        )
        return max(0.0, float(r.stdout.strip()))
    except Exception:
        return 0.0


def _append_wav_tail_silence(src: Path, dst: Path, *, pad_sec: float, sr: int = 16000) -> Path:
    """Create an ASR-only copy with trailing silence so RNNT can flush final tokens."""
    pad_sec = max(0.0, float(pad_sec))
    if pad_sec <= 1e-3:
        return src
    audio, file_sr = sf.read(str(src), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if int(file_sr) != int(sr):
        raise RuntimeError(f"Unexpected ASR wav sample rate: {file_sr}, expected {sr}")
    pad = np.zeros(int(round(pad_sec * sr)), dtype=np.float32)
    out = np.concatenate([audio.astype(np.float32, copy=False), pad], axis=0)
    sf.write(str(dst), out, sr)
    return dst


def _plan_asr_slices(
    duration: float,
    *,
    window: float = 180.0,
    overlap: float = 2.0,
    min_tail: float = 0.05,
) -> list[dict[str, float | bool | int]]:
    """Plan overlapping ASR windows and non-overlapping timestamp ownership."""
    duration = max(0.0, float(duration))
    window = max(1.0, float(window))
    overlap = min(max(0.0, float(overlap)), window - 0.01)
    stride = window - overlap
    starts: list[float] = []
    cur = 0.0
    while cur < duration - float(min_tail):
        starts.append(cur)
        if cur + window >= duration - 1e-6:
            break
        cur += stride

    plans: list[dict[str, float | bool | int]] = []
    ends = [min(duration, start + window) for start in starts]
    for i, (start, end) in enumerate(zip(starts, ends)):
        own_start = 0.0 if i == 0 else 0.5 * (start + ends[i - 1])
        own_end = duration if i + 1 == len(starts) else 0.5 * (starts[i + 1] + end)
        plans.append({
            "index": i,
            "start": start,
            "duration": max(0.0, end - start),
            "ownership_start": own_start,
            "ownership_end": own_end,
            "is_last": i + 1 == len(starts),
        })
    return plans


def _words_owned_by_slice(words: list[dict[str, Any]], plan: dict[str, Any]) -> list[dict[str, Any]]:
    """Keep each overlapped word in exactly one ASR window, based on its midpoint."""
    lo = float(plan["ownership_start"])
    hi = float(plan["ownership_end"])
    is_last = bool(plan["is_last"])
    out: list[dict[str, Any]] = []
    for word in words:
        start = float(word.get("start", 0.0))
        end = float(word.get("end", start))
        midpoint = 0.5 * (start + max(start, end))
        if midpoint >= lo - 1e-6 and (midpoint < hi - 1e-6 or (is_last and midpoint <= hi + 1e-6)):
            out.append(word)
    return out


# ── Worker: transcription ────────────────────────────────────────────────────
def _worker_transcribe(job: Job, upload_path: Path) -> None:
    wav_path: Path | None = None
    slice_paths: list[Path] = []
    uid = job.id[:8]
    try:
        job.status = "running"
        _push(job, {"type": "progress", "progress": 0.05, "message": "Konwertuję audio…"})

        wav_path = core.convert_to_mono16k_wav(upload_path, _WORK)
        duration = _ffprobe_dur(wav_path)
        if duration < 0.5:
            raise ValueError("Audio za krótkie (< 0.5 s)")

        window = float(core._CONFIG.get("asr_window_seconds", 180.0))
        overlap = float(core._CONFIG.get("asr_window_overlap_seconds", 2.0))
        final_tail_pad = max(0.0, float(core._CONFIG.get("asr_final_tail_pad_seconds", 1.2)))
        offsets: list[float] = []
        plans = _plan_asr_slices(duration, window=window, overlap=overlap)
        slice_debug: list[dict[str, Any]] = []
        for plan in plans:
            idx = int(plan["index"])
            cur = float(plan["start"])
            win = float(plan["duration"])
            is_last_slice = bool(plan["is_last"])
            if idx == 0 and is_last_slice:
                if final_tail_pad > 1e-3:
                    sl = _WORK / f"sl_{uid}_{idx:04d}_tailpad.wav"
                    slice_paths.append(_append_wav_tail_silence(wav_path, sl, pad_sec=final_tail_pad))
                else:
                    slice_paths.append(wav_path)
            else:
                sl = _WORK / f"sl_{uid}_{idx:04d}.wav"
                subprocess.run(
                    ["ffmpeg", "-y", "-ss", f"{cur:.3f}", "-t", f"{win:.3f}",
                     "-i", str(wav_path), "-vn", "-ac", "1", "-ar", "16000", str(sl)],
                    check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                )
                if is_last_slice and final_tail_pad > 1e-3:
                    padded = _WORK / f"sl_{uid}_{idx:04d}_tailpad.wav"
                    sl = _append_wav_tail_silence(sl, padded, pad_sec=final_tail_pad)
                slice_paths.append(sl)
            offsets.append(cur)
            slice_debug.append({
                "index": idx,
                "offset": round(cur, 3),
                "source_duration": round(win, 3),
                "ownership_start": round(float(plan["ownership_start"]), 3),
                "ownership_end": round(float(plan["ownership_end"]), 3),
                "tail_padded": bool(is_last_slice and final_tail_pad > 1e-3),
            })

        n_slices = len(slice_paths)
        _push(job, {"type": "progress", "progress": 0.15,
                    "message": f"Transkrybuję {n_slices} okno/okien…"})

        transcriber = core.get_transcriber()
        words_all: list[dict] = []
        lang_counts: dict[str, int] = {}
        for si, (sp, off, plan) in enumerate(zip(slice_paths, offsets, plans)):
            _push(job, {"type": "progress",
                        "progress": 0.15 + 0.75 * (si / n_slices),
                        "message": f"Slice {si + 1}/{n_slices}…"})
            w, meta = transcriber.transcribe_word_timestamps(
                [str(sp)], batch_size=1, offsets=[off],
            )
            owned = _words_owned_by_slice(w, plan)
            words_all.extend(owned)
            slice_debug[si]["raw_word_count"] = len(w)
            slice_debug[si]["owned_word_count"] = len(owned)
            for lang, cnt in meta.get("language_counts", {}).items():
                lang_counts[lang] = lang_counts.get(lang, 0) + cnt

        words_all.sort(key=lambda item: (float(item.get("start", 0.0)), float(item.get("end", 0.0))))
        transcript = _wtext(words_all)
        segments = build_segments(words_all)
        detected = (max(lang_counts, key=lang_counts.get) if lang_counts else "auto")

        result = {
            "transcript": transcript,
            "words": words_all,
            "segments": segments,
            "detected_language": detected,
            "language_counts": lang_counts,
            "duration": round(duration, 3),
            "asr_final_tail_pad_seconds": round(final_tail_pad, 3),
            "asr_slices": slice_debug,
            "word_count": len(words_all),
            "segment_count": len(segments),
            "upload_path": str(upload_path),
        }
        job.result = result
        job.status = "done"
        _push(job, {"type": "done", "progress": 1.0,
                    "message": f"Gotowe — {len(words_all)} słów", "result": result})

    except Exception as exc:
        job.status = "error"
        job.error = str(exc)
        _push(job, {"type": "error", "error": str(exc)})
        upload_path.unlink(missing_ok=True)
    finally:
        for p in slice_paths:
            if p != wav_path:
                p.unlink(missing_ok=True)


def _yt_dlp_extra_args() -> list[str]:
    args: list[str] = []
    if YTDLP_DENO.is_file():
        args += ["--js-runtimes", f"deno:{YTDLP_DENO}"]

    cookies_file = os.environ.get("WEGORZ_YTDLP_COOKIES_FILE", "").strip()
    cookies_browser = os.environ.get("WEGORZ_YTDLP_COOKIES_FROM_BROWSER", "").strip()
    if cookies_file:
        args += ["--cookies", str(Path(cookies_file).expanduser())]
    elif cookies_browser:
        args += ["--cookies-from-browser", cookies_browser]
    return args


def _download_youtube_video(url: str, work_dir: Path) -> Path:
    core.require_executable("ffmpeg")
    token = uuid.uuid4().hex[:12]
    out_template = str(work_dir / f"yt_{token}.%(ext)s")
    cmd = core.yt_dlp_command() + _yt_dlp_extra_args() + [
        "--no-playlist",
        "--no-progress",
        "--retries", "3",
        "--fragment-retries", "3",
        "-f", "bv*+ba/best",
        "--merge-output-format", "mp4",
        "-o", out_template,
        str(url),
    ]
    result = subprocess.run(cmd, check=False, text=True, capture_output=True)
    if result.stdout:
        print(result.stdout, end="", flush=True)
    if result.stderr:
        print(result.stderr, end="", file=sys.stderr, flush=True)
    if result.returncode != 0:
        details = f"{result.stdout}\n{result.stderr}".lower()
        if "sign in" in details or "cookies" in details or "confirm you're not a bot" in details:
            hint = (
                " YouTube wymaga cookies. Ustaw WEGORZ_YTDLP_COOKIES_FROM_BROWSER=firefox "
                "albo WEGORZ_YTDLP_COOKIES_FILE=/sciezka/cookies.txt i uruchom serwer ponownie."
            )
        else:
            hint = " Sprawdź połączenie z siecią oraz aktualność pakietu yt-dlp."
        raise RuntimeError(f"Nie udało się pobrać filmu przez yt-dlp.{hint}")
    candidates = sorted(work_dir.glob(f"yt_{token}.*"), key=lambda p: p.stat().st_mtime)
    mp4s = [p for p in candidates if p.suffix.lower() == ".mp4"]
    if mp4s:
        return mp4s[-1]
    if candidates:
        return candidates[-1]
    raise RuntimeError("yt-dlp zakończył pracę, ale nie utworzył pliku wideo")


def _worker_transcribe_youtube(job: Job, url: str) -> None:
    try:
        job.status = "running"
        _push(job, {"type": "progress", "progress": 0.02, "message": "Pobieram film z YouTube…"})
        video_path = _download_youtube_video(url, _WORK)
        _worker_transcribe(job, video_path)
    except Exception as exc:
        job.status = "error"
        job.error = str(exc)
        _push(job, {"type": "error", "error": str(exc)})


# ── Worker: translation ──────────────────────────────────────────────────────
class TranslateRequest(BaseModel):
    segments: list[dict[str, Any]]
    source_lang: str = "auto"
    target_lang: str = "pl"
    mode: str = ""
    model: str = ""
    api_key: str = ""
    batch_segments: int = 0


def _worker_translate(job: Job, req: TranslateRequest) -> None:
    try:
        job.status = "running"
        _push(job, {"type": "progress", "progress": 0.05, "message": "Inicjalizuję tłumaczenie…"})
        _validate_segment_timeline(req.segments, stage="tłumaczenie")

        selected_model = str(req.model or core._CONFIG.get("translation_model", "qwen3.5:35b-mtp"))
        selected_mode = str(req.mode or core._CONFIG.get("translation_mode", "qwen_mtp_35b_json_overlap"))
        selected_batch = int(req.batch_segments or core._CONFIG.get("translation_batch_segments", 8))
        env_key_name = "GEMINI_API_KEY" if selected_model.startswith("gemini") else "NUPIC_API_KEY"
        key = (
            req.api_key.strip()
            or os.environ.get(env_key_name, "").strip()
            or str(core._CONFIG.get("translation_api_key", "")).strip()
        )
        core._CONFIG.update({
            "translation_source_lang": req.source_lang,
            "translation_target_lang": req.target_lang,
        })

        t0 = time.perf_counter()
        n = len(req.segments)
        _push(job, {"type": "progress", "progress": 0.1,
                    "message": f"Tłumaczę {n} segmentów (tryb={req.mode})…"})
        if req.mode == "wegorz_local_sentence_split":
            _push(job, {
                "type": "progress",
                "progress": 0.15,
                "message": "Lokalny Węgorz: pierwsze użycie ładuje checkpoint do GPU/RAM, kolejne tłumaczenia będą szybkie…",
            })

        def _translation_progress(done_batches: int, total_batches: int, start: int, count: int, phase: str) -> None:
            total_batches = max(1, int(total_batches))
            done_batches = max(0, min(int(done_batches), total_batches))
            first = int(start) + 1
            last = int(start) + int(count)
            progress = 0.12 + 0.82 * (done_batches / total_batches)
            verb = "Gotowe" if phase == "done" else "Tłumaczę"
            _push(job, {
                "type": "progress",
                "progress": min(0.95, progress),
                "message": f"{verb} batch {done_batches if phase == 'done' else done_batches + 1}/{total_batches} (segmenty {first}-{last})…",
            })

        translated, meta = core.translate_segments_to_pl(
            segments=list(req.segments),
            api_key=key,
            endpoint=str(core._CONFIG.get("translation_endpoint", "")),
            model=selected_model,
            mode=selected_mode,
            batch_segments=max(1, min(20, selected_batch)),
            temperature=float(core._CONFIG.get("translation_temperature", 0.1)),
            timeout=float(core._CONFIG.get("translation_timeout_seconds", 600)),
            retry=int(core._CONFIG.get("translation_retry", 2)),
            progress_callback=_translation_progress,
        )
        elapsed = time.perf_counter() - t0

        if len(translated) != len(req.segments):
            raise RuntimeError(
                f"Tłumacz zwrócił {len(translated)} segmentów dla {len(req.segments)} wejściowych; "
                "wynik został zatrzymany, aby nie pominąć fragmentu."
            )

        normalized_segments: list[dict[str, Any]] = []
        for i, src_raw in enumerate(req.segments):
            src = dict(src_raw)
            seg = dict(translated[i])
            source_text = str(src.get("source_text") or src.get("text") or seg.get("source_text") or "").strip()
            translated_text = str(seg.get("translation") or seg.get("text") or "").strip()
            if not translated_text:
                translated_text = source_text
            normalized_segments.append({
                **src,
                **seg,
                "source_text": source_text,
                "text": source_text,
                "translation": translated_text,
            })
        translated = normalized_segments
        _validate_segment_timeline(translated, stage="wynik tłumaczenia")

        translation_text = " ".join(str(s.get("translation", "")) for s in translated).strip()
        result = {
            "translation": translation_text,
            "segments": translated,
            "source_lang": req.source_lang,
            "target_lang": req.target_lang,
            "model": selected_model,
            "elapsed": round(elapsed, 2),
            "meta": meta,
        }
        job.result = result
        job.status = "done"
        _push(job, {"type": "done", "progress": 1.0,
                    "message": f"Przetłumaczono {len(translated)} segmentów ({elapsed:.1f} s)",
                    "result": result})

    except Exception as exc:
        job.status = "error"
        job.error = str(exc)
        _push(job, {"type": "error", "error": str(exc)})


# ── Worker: TTS dubbing ──────────────────────────────────────────────────────
class DubRequest(BaseModel):
    segments: list[dict[str, Any]]
    speaker_label: str
    tts_model_profile: str = DEFAULT_TTS_PROFILE
    transcribe_job_id: str = ""
    target_lang: str = "pl"
    base_speed: float = 1.0
    max_adaptive_speed: float = 1.3
    extra_tail_sec: float = 0.0
    dur_scale: float = 1.0           # unused, kept for compat
    dur_source: str = "prior_mu"     # unused, kept for compat
    mel_steps_first: int = 8
    mel_steps_second: int = 3
    mel_twopass_t_noise: float = 0.12
    digital_silence: bool = True
    pause_edge_frames: int = 10
    short_continuity_ms: float = 0.0
    emotion_group: str = "neutral"
    emotion_strength: float = 3.0
    original_gain: float = 0.22
    dubbing_gain: float = 1.0
    ducking_strength: float = 0.65


def _render_audio_mix(
    source_path: Path,
    dubbed_path: Path,
    out_path: Path,
    *,
    original_gain: float,
    dubbing_gain: float,
    ducking_strength: float,
) -> None:
    """Mix source ambience and dubbed speech with optional sidechain ducking."""
    original_gain = max(0.0, min(1.5, float(original_gain)))
    dubbing_gain = max(0.0, min(1.5, float(dubbing_gain)))
    ducking_strength = max(0.0, min(1.0, float(ducking_strength)))
    if ducking_strength > 0.001:
        ratio = 1.0 + 15.0 * ducking_strength
        threshold = 0.08 - 0.06 * ducking_strength
        graph = (
            f"[0:a]aformat=sample_rates=24000:channel_layouts=mono,volume={original_gain:.4f}[original];"
            f"[1:a]aformat=sample_rates=24000:channel_layouts=mono,volume={dubbing_gain:.4f},"
            "asplit=2[sidechain][voice];"
            f"[original][sidechain]sidechaincompress=threshold={threshold:.4f}:ratio={ratio:.3f}:"
            "attack=15:release=280[ducked];"
            "[ducked][voice]amix=inputs=2:duration=longest:normalize=0,alimiter=limit=0.95[mix]"
        )
    else:
        graph = (
            f"[0:a]aformat=sample_rates=24000:channel_layouts=mono,volume={original_gain:.4f}[original];"
            f"[1:a]aformat=sample_rates=24000:channel_layouts=mono,volume={dubbing_gain:.4f}[voice];"
            "[original][voice]amix=inputs=2:duration=longest:normalize=0,alimiter=limit=0.95[mix]"
        )
    subprocess.run(
        [
            "ffmpeg", "-y", "-i", str(source_path), "-i", str(dubbed_path),
            "-filter_complex", graph, "-map", "[mix]", "-ac", "1", "-ar", "24000",
            "-c:a", "pcm_s16le", str(out_path),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )


def _worker_dub(job: Job, req: DubRequest) -> None:
    try:
        job.status = "running"
        _push(job, {"type": "progress", "progress": 0.01, "message": "Przygotowuję TTS…"})
        _validate_segment_timeline(req.segments, stage="dubbing")

        speaker_payload = _speaker_condition_payload(req.speaker_label)

        SR = 24000
        segs = req.segments
        n = len(segs)
        video_dur = max((float(s.get("end", 0.0)) for s in segs), default=10.0) + 5.0
        full = np.zeros(int(video_dur * SR), dtype=np.float32)
        last_end = 0.0
        meta_segments: list[dict[str, Any]] = []

        out_dir = _WORK / f"dub_{job.id}"
        out_dir.mkdir(parents=True, exist_ok=True)

        for i, seg in enumerate(segs):
            text = str(seg.get("translation") or seg.get("text", "")).strip()
            if not text:
                raise ValueError(f"dubbing: pusty tekst docelowy w segmencie {i}")
            src_start = float(seg.get("start", 0.0))
            src_end = float(seg.get("end", src_start))
            next_start = float(segs[i + 1].get("start", src_end)) if i + 1 < n else src_end + float(req.extra_tail_sec)
            place = max(src_start, last_end + 0.04 if i > 0 else src_start)
            target_budget = max(0.05, next_start - place)

            _push(job, {
                "type": "progress",
                "progress": 0.05 + 0.93 * (i / n),
                "message": f"TTS {i + 1}/{n}: {text[:60]}…",
            })

            def synth_at(speed: float, tag_suffix: str = "") -> str:
                tag = f"seg_{job.id[:8]}_{i:04d}{tag_suffix}"
                base = {
                    "text": text,
                    **speaker_payload,
                    "speed": float(speed),
                    "mel_steps_first": req.mel_steps_first,
                    "mel_steps_second": req.mel_steps_second,
                    "mel_twopass_t_noise": req.mel_twopass_t_noise,
                    "seed": 1234 + i,
                    "out_dir": str(out_dir),
                    "tag": tag,
                    "lang": req.target_lang,
                    "digital_silence": req.digital_silence,
                    "pause_edge_frames": req.pause_edge_frames,
                    "leading_sp_min_frames": 4,
                    "short_continuity_ms": req.short_continuity_ms,
                    "emotion_group": req.emotion_group,
                    "emotion_strength": req.emotion_strength,
                    "tts_model_profile": req.tts_model_profile,
                    # Dubbing segments are independent ASR chunks. Reusing one bridge
                    # state across segments can swallow short leading words.
                    "continuity_key": f"dub:{job.id}:{req.speaker_label}:seg{i:04d}",
                    "continuity_reset": True,
                }
                resp = _daemon_synth_chunked_response(base, out_dir=str(out_dir), tag=tag)
                synth_at.last_debug = resp
                return str(resp["wav"])
            synth_at.last_debug = {}

            actual_speed = float(req.base_speed)
            fit_retries = 0
            snapshot_id = _daemon_state_snapshot(req.tts_model_profile)
            wav_path = synth_at(actual_speed)
            wav_dur = _ffprobe_dur(Path(wav_path))
            debug_info = dict(getattr(synth_at, "last_debug", {}) or {})
            hard_speed = max(float(req.base_speed), min(float(req.max_adaptive_speed), 1.3))
            if wav_dur > target_budget + 0.02 and actual_speed < hard_speed - 1e-6:
                retry_speed = min(hard_speed, actual_speed * (wav_dur / target_budget) * 1.03)
                if retry_speed > actual_speed + 0.01:
                    _daemon_state_restore(snapshot_id, req.tts_model_profile)
                    actual_speed = retry_speed
                    fit_retries = 1
                    wav_path = synth_at(actual_speed, "_fit")
                    wav_dur = _ffprobe_dur(Path(wav_path))
                    debug_info = dict(getattr(synth_at, "last_debug", {}) or {})

            audio, file_sr = sf.read(wav_path)
            if audio.ndim > 1:
                audio = audio[:, 0]
            audio = audio.astype(np.float32)

            if file_sr != SR:
                ratio = SR / file_sr
                audio = np.interp(
                    np.arange(0, len(audio) * ratio, ratio),
                    np.arange(len(audio)),
                    audio,
                ).astype(np.float32)

            s0 = max(0, int(round(place * SR)))
            needed = s0 + len(audio)
            if needed > len(full):
                full = np.pad(full, (0, needed - len(full)))
            full[s0:s0 + len(audio)] += audio
            last_end = max(last_end, place + len(audio) / SR)
            audio_sec = len(audio) / SR
            debug_summary = _summarize_tts_debug(
                debug_info,
                text=text,
                budget_sec=target_budget,
                wav_sec=audio_sec,
            )
            meta_segments.append({
                "index": int(seg.get("index", i)),
                "text": text,
                "start": round(place, 3),
                "source_start": round(src_start, 3),
                "source_end": round(src_end, 3),
                "next_start": round(next_start, 3),
                "target_budget": round(target_budget, 3),
                "audio_duration": round(audio_sec, 3),
                "speed": round(actual_speed, 3),
                "fit_retries": fit_retries,
                "over_budget": round(max(0.0, (place + audio_sec) - next_start), 3),
                "tts_debug_summary": debug_summary,
                "tts_debug": debug_info,
            })

        out_wav = out_dir / "dubbed.wav"
        end_sample = min(len(full), int(last_end * SR) + SR)
        sf.write(str(out_wav), full[:end_sample], SR)
        mixed_wav: Path | None = None
        source_job = _jobs.get(req.transcribe_job_id)
        source_path_value = ((source_job.result or {}).get("upload_path") if source_job and source_job.result else "")
        source_path = Path(str(source_path_value)) if source_path_value else None
        if source_path is not None and source_path.exists():
            mixed_wav = out_dir / "mixed.wav"
            _render_audio_mix(
                source_path,
                out_wav,
                mixed_wav,
                original_gain=req.original_gain,
                dubbing_gain=req.dubbing_gain,
                ducking_strength=req.ducking_strength,
            )
        debug_log = out_dir / "tts_debug.json"
        debug_log.write_text(json.dumps(meta_segments, ensure_ascii=False, indent=2), encoding="utf-8")

        result = {
            "audio_path": str(out_wav),
            "mixed_audio_path": str(mixed_wav) if mixed_wav is not None else "",
            "duration": round(last_end, 3),
            "transcribe_job_id": req.transcribe_job_id,
            "segments": meta_segments,
            "debug_log": str(debug_log),
            "mix": {
                "original_gain": round(float(req.original_gain), 3),
                "dubbing_gain": round(float(req.dubbing_gain), 3),
                "ducking_strength": round(float(req.ducking_strength), 3),
            },
        }
        job.result = result
        job.status = "done"
        _push(job, {"type": "done", "progress": 1.0,
                    "message": f"Dubbing gotowy ({last_end:.1f} s)",
                    "result": result})

    except Exception as exc:
        import traceback
        job.status = "error"
        job.error = str(exc)
        _push(job, {"type": "error", "error": f"{exc}\n{traceback.format_exc()[-1000:]}"})


# ── FastAPI app ──────────────────────────────────────────────────────────────
app = FastAPI(title="Węgorz Dubbing Studio API", version="3.0")
_cors_origins = [
    value.strip() for value in os.environ.get(
        "WEGORZ_CORS_ORIGINS",
        "http://127.0.0.1:8765,http://localhost:8765",
    ).split(",") if value.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def _startup() -> None:
    global _loop
    _loop = asyncio.get_running_loop()
    _warm_model()
    _warm_tts_daemon()


def _warm_model() -> None:
    try:
        print("🔄 Ładuję model ASR…", flush=True)
        core.get_transcriber()
        print("✅ Model ASR gotowy.", flush=True)
        if _loop:
            _loop.call_soon_threadsafe(_MODEL_READY.set)
    except Exception as exc:
        print(f"❌ Błąd ładowania modelu: {exc}", flush=True)


def _warm_tts_daemon() -> None:
    try:
        print(f"🔄 Ładuję checkpoint TTS w tle [{DEFAULT_TTS_PROFILE}]…", flush=True)
        with _daemon_lock:
            _ensure_daemon_locked(DEFAULT_TTS_PROFILE)
        print("✅ Checkpoint TTS gotowy.", flush=True)
        if _loop:
            _loop.call_soon_threadsafe(_TTS_READY.set)
    except Exception as exc:
        print(f"❌ Błąd ładowania TTS: {exc}", flush=True)


# ── SSE helper ───────────────────────────────────────────────────────────────
async def _sse_stream(job: Job) -> AsyncGenerator[str, None]:
    while True:
        try:
            event = await asyncio.wait_for(job.queue.get(), timeout=30.0)
        except asyncio.TimeoutError:
            yield "data: {\"type\":\"ping\"}\n\n"
            if job.status in ("done", "error"):
                break
            continue
        payload = json.dumps(event, ensure_ascii=False)
        yield f"data: {payload}\n\n"
        if event.get("type") in ("done", "error"):
            break


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/health")
async def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "model_ready": _MODEL_READY.is_set(),
        "tts_ready": _TTS_READY.is_set(),
        "tts_profile": _daemon_active_profile or DEFAULT_TTS_PROFILE,
        "tts_loaded_profiles": sorted(
            key for key, proc in _daemon_procs.items()
            if proc is not None and proc.poll() is None
        ),
    }


def _require_admin(token: str) -> None:
    if not ADMIN_TOKEN:
        raise HTTPException(
            status_code=503,
            detail="Panel administratora wymaga WEGORZ_ADMIN_TOKEN w środowisku serwera.",
        )
    if not token or not secrets.compare_digest(token, ADMIN_TOKEN):
        raise HTTPException(status_code=401, detail="Nieprawidłowy token administratora")


def _masked_secret(value: str) -> str:
    value = str(value or "")
    if not value:
        return ""
    suffix = value[-4:] if len(value) >= 4 else value
    return f"••••••••{suffix}"


class AdminSettingsRequest(BaseModel):
    translation_endpoint: str = ""
    translation_model: str = ""
    translation_mode: str = ""
    translation_batch_segments: int = 8
    translation_api_key: str = ""
    clear_translation_api_key: bool = False


@app.get("/admin/settings")
async def admin_settings(x_admin_token: str = Header(default="")) -> dict[str, Any]:
    _require_admin(x_admin_token)
    recent_jobs = []
    for job in list(_jobs.values())[-30:][::-1]:
        result = job.result or {}
        segment_debug = []
        for segment in list(result.get("segments") or [])[:200]:
            summary = dict(segment.get("tts_debug_summary") or {})
            segment_debug.append({
                "index": segment.get("index"),
                "start": segment.get("start"),
                "audio_duration": segment.get("audio_duration"),
                "target_budget": segment.get("target_budget"),
                "speed": segment.get("speed"),
                "over_budget": segment.get("over_budget"),
                "warnings": list(summary.get("warnings") or []),
                "low_token_count": summary.get("low_token_count", 0),
            })
        recent_jobs.append({
            "id": job.id,
            "kind": job.kind,
            "status": job.status,
            "message": job.message,
            "error": job.error,
            "debug_log": str(result.get("debug_log", "")),
            "duration": result.get("duration"),
            "segments": segment_debug,
        })
    api_key = str(core._CONFIG.get("translation_api_key", ""))
    return {
        "translation_endpoint": str(core._CONFIG.get("translation_endpoint", "")),
        "translation_model": str(core._CONFIG.get("translation_model", "")),
        "translation_mode": str(core._CONFIG.get("translation_mode", "")),
        "translation_batch_segments": int(core._CONFIG.get("translation_batch_segments", 8)),
        "translation_api_key_configured": bool(api_key),
        "translation_api_key_masked": _masked_secret(api_key),
        "tts_profile": _daemon_active_profile or DEFAULT_TTS_PROFILE,
        "tts_loaded_profiles": sorted(
            key for key, proc in _daemon_procs.items()
            if proc is not None and proc.poll() is None
        ),
        "model_ready": _MODEL_READY.is_set(),
        "tts_ready": _TTS_READY.is_set(),
        "recent_jobs": recent_jobs,
    }


@app.post("/admin/settings")
async def update_admin_settings(
    req: AdminSettingsRequest,
    x_admin_token: str = Header(default=""),
) -> dict[str, Any]:
    _require_admin(x_admin_token)
    for key, value in (
        ("translation_endpoint", req.translation_endpoint),
        ("translation_model", req.translation_model),
        ("translation_mode", req.translation_mode),
    ):
        clean = str(value or "").strip()
        if clean:
            core._CONFIG[key] = clean
    core._CONFIG["translation_batch_segments"] = max(1, min(20, int(req.translation_batch_segments)))
    if req.clear_translation_api_key:
        core._CONFIG["translation_api_key"] = ""
    elif str(req.translation_api_key or "").strip():
        core._CONFIG["translation_api_key"] = str(req.translation_api_key).strip()
    _save_admin_config()
    return await admin_settings(x_admin_token)


@app.get("/tts_models")
async def tts_models() -> dict[str, Any]:
    return {
        "default": DEFAULT_TTS_PROFILE,
        "active": _daemon_active_profile or DEFAULT_TTS_PROFILE,
        "loaded": sorted(
            key for key, proc in _daemon_procs.items()
            if proc is not None and proc.poll() is None
        ),
        "models": [
            {
                "key": key,
                "label": str(rec.get("label", key)),
                "description": str(rec.get("description", "")),
                "checkpoint": str(rec.get("checkpoint", "")),
                "default": key == DEFAULT_TTS_PROFILE,
                "active": key == (_daemon_active_profile or DEFAULT_TTS_PROFILE),
                "loaded": (
                    key in _daemon_procs
                    and _daemon_procs[key] is not None
                    and _daemon_procs[key].poll() is None
                ),
            }
            for key, rec in TTS_MODEL_PROFILES.items()
        ],
    }


@app.post("/transcribe")
async def transcribe(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
) -> dict[str, str]:
    suffix = Path(file.filename or "audio").suffix or ".wav"
    uid = uuid.uuid4().hex
    upload_path = _WORK / f"up_{uid}{suffix}"
    with upload_path.open("wb") as fh:
        shutil.copyfileobj(file.file, fh)
    job = Job(id=uid, kind="transcribe")
    _jobs[uid] = job
    _executor.submit(_worker_transcribe, job, upload_path)
    return {"job_id": uid}


class YoutubeTranscribeRequest(BaseModel):
    url: str


@app.post("/transcribe_youtube")
async def transcribe_youtube(req: YoutubeTranscribeRequest) -> dict[str, str]:
    url = str(req.url or "").strip()
    if not url:
        raise HTTPException(status_code=400, detail="Brak URL YouTube")
    uid = uuid.uuid4().hex
    job = Job(id=uid, kind="transcribe")
    _jobs[uid] = job
    _executor.submit(_worker_transcribe_youtube, job, url)
    return {"job_id": uid}


@app.post("/translate")
async def translate(
    background_tasks: BackgroundTasks,
    req: TranslateRequest,
) -> dict[str, str]:
    uid = uuid.uuid4().hex
    job = Job(id=uid, kind="translate")
    _jobs[uid] = job
    _executor.submit(_worker_translate, job, req)
    return {"job_id": uid}


@app.get("/jobs/{job_id}")
async def get_job(job_id: str) -> dict[str, Any]:
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return {
        "id": job.id,
        "kind": job.kind,
        "status": job.status,
        "progress": job.progress,
        "message": job.message,
        "result": job.result,
        "error": job.error,
    }


@app.get("/speakers")
async def list_speakers() -> dict[str, Any]:
    return {"speakers": _SPEAKER_LIST}


@app.post("/voice_prompt")
async def upload_voice_prompt(
    file: UploadFile = File(...),
    start_sec: float = Form(0.0),
    max_sec: float = Form(12.0),
) -> dict[str, Any]:
    uid = uuid.uuid4().hex[:10]
    suffix = Path(file.filename or "prompt.wav").suffix or ".wav"
    src = _WORK / f"voice_prompt_upload_{uid}{suffix}"
    with src.open("wb") as fh:
        shutil.copyfileobj(file.file, fh)
    wav24 = _convert_media_to_mono24k(src, _WORK)
    enc = _daemon_encode_ref_mel(wav24, _WORK / "voice_prompts", start_sec=float(start_sec), max_sec=float(max_sec))
    mel_path = str(enc.get("mel", ""))
    if not mel_path or not Path(mel_path).exists():
        raise HTTPException(status_code=500, detail="Nie udało się utworzyć mel promptu")
    raw_name = Path(file.filename or "custom").stem[:48] or "custom"
    label = f"[custom] {raw_name} [{uid}]"
    _SPEAKER_MEL[label] = mel_path
    entry = {"label": label, "id": 100000 + len(_SPEAKER_LIST), "custom": True}
    _SPEAKER_LIST.insert(0, entry)
    return {
        "speaker": entry,
        "mel_path": mel_path,
        "duration": enc.get("duration", 0.0),
        "frames": enc.get("frames", 0),
    }


class TextTTSRequest(BaseModel):
    text: str
    speaker_label: str
    tts_model_profile: str = DEFAULT_TTS_PROFILE
    language: str = "pl"
    speed: float = 1.0
    dur_scale: float = 1.0    # unused, kept for frontend compat
    dur_source: str = "prior_mu"  # unused, kept for frontend compat
    mel_steps_first: int = 8
    mel_steps_second: int = 3
    mel_twopass_t_noise: float = 0.12
    digital_silence: bool = True
    pause_edge_frames: int = 10
    short_continuity_ms: float = 0.0
    emotion_group: str = "neutral"
    emotion_strength: float = 3.0


def _worker_tts_text(job: Job, req: TextTTSRequest) -> None:
    try:
        job.status = "running"
        _push(job, {"type": "progress", "progress": 0.05, "message": "Przygotowuję TTS…"})

        speaker_payload = _speaker_condition_payload(req.speaker_label)
        out_dir = _WORK / f"tts_{job.id}"
        out_dir.mkdir(parents=True, exist_ok=True)

        _push(job, {"type": "progress", "progress": 0.15, "message": "Syntetyzuję (czeka na daemon)…"})

        tag = f"tts_{job.id[:8]}"
        synth_resp = _daemon_synth_chunked_response(
            {
                "text": req.text,
                **speaker_payload,
                "speed": req.speed,
                "mel_steps_first": req.mel_steps_first,
                "mel_steps_second": req.mel_steps_second,
                "mel_twopass_t_noise": req.mel_twopass_t_noise,
                "seed": 1234,
                "out_dir": str(out_dir),
                "tag": tag,
                "lang": req.language,
                "digital_silence": req.digital_silence,
                "pause_edge_frames": req.pause_edge_frames,
                "leading_sp_min_frames": 4,
                "short_continuity_ms": req.short_continuity_ms,
                "emotion_group": req.emotion_group,
                "emotion_strength": req.emotion_strength,
                "tts_model_profile": req.tts_model_profile,
                "continuity_key": f"tts:{job.id}:{req.speaker_label}",
                "continuity_reset": True,
                "tts_max_chars": _TTS_MAX_CHARS,
                "tts_hard_sentence_chars": _TTS_HARD_SENTENCE_CHARS,
            },
            out_dir=str(out_dir),
            tag=tag,
        )
        wav_path = str(synth_resp["wav"])

        audio, sr = sf.read(wav_path)
        dur = len(audio) / sr if sr > 0 else 0.0

        # Build compact per-chunk debug info for the frontend
        debug_chunks: list[dict[str, Any]] = []
        for ci, ch in enumerate(synth_resp.get("chunks") or []):
            dbg = ch.get("debug") or {}
            raw_toks = dbg.get("durations") or []
            tokens = [
                {
                    "token": t["token"],
                    "dur": t["dur"],
                    "dur_sec": t["dur_sec"],
                    "is_pause": t["token"] == "<sp>",
                    "allowed": t.get("allowed", False),
                    "low": t.get("low_duration", False),
                }
                for t in raw_toks
                if t.get("allowed")
            ]
            debug_chunks.append({
                "index": ci,
                "text": ch.get("text", ""),
                "text_prepared": dbg.get("text_prepared", ""),
                "text_in": dbg.get("text_in", ""),
                "pred_sec": dbg.get("pred_sec"),
                "mel_sec": dbg.get("mel_sec"),
                "token_count": len(tokens),
                "tokens": tokens,
            })

        import json as _json
        (out_dir / "tts_debug.json").write_text(
            _json.dumps({"text": req.text, "speaker": req.speaker_label,
                         "duration": round(dur, 3), "chunks": debug_chunks}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        result = {
            "audio_path": wav_path,
            "duration": round(dur, 3),
            "chunks": debug_chunks,
            "debug_log": str(out_dir / "tts_debug.json"),
        }
        job.result = result
        job.status = "done"
        _push(job, {"type": "done", "progress": 1.0,
                    "message": f"Gotowe ({dur:.1f} s)", "result": result})

    except Exception as exc:
        import traceback
        job.status = "error"
        job.error = str(exc)
        _push(job, {"type": "error", "error": f"{exc}\n{traceback.format_exc()[-1500:]}"})


@app.post("/tts_text")
async def tts_text(req: TextTTSRequest) -> dict[str, str]:
    uid = uuid.uuid4().hex
    job = Job(id=uid, kind="tts_text")
    _jobs[uid] = job
    _executor.submit(_worker_tts_text, job, req)
    return {"job_id": uid}


@app.post("/dub")
async def dub(req: DubRequest) -> dict[str, str]:
    uid = uuid.uuid4().hex
    job = Job(id=uid, kind="dub")
    _jobs[uid] = job
    _executor.submit(_worker_dub, job, req)
    return {"job_id": uid}


@app.get("/jobs/{job_id}/audio")
async def job_audio(job_id: str) -> StreamingResponse:
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.status != "done" or job.result is None:
        raise HTTPException(status_code=409, detail="Job not finished")
    audio_path = Path(job.result.get("audio_path", ""))
    if not audio_path.exists():
        raise HTTPException(status_code=404, detail="Audio file not found")
    return StreamingResponse(
        open(audio_path, "rb"),
        media_type="audio/wav",
        headers={"Content-Disposition": f'attachment; filename="tts_{job_id}.wav"'},
    )


@app.get("/jobs/{job_id}/mix_audio")
async def job_mix_audio(job_id: str) -> StreamingResponse:
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.status != "done" or job.result is None:
        raise HTTPException(status_code=409, detail="Job not finished")
    audio_path = Path(str(job.result.get("mixed_audio_path", "")))
    if not audio_path.exists():
        raise HTTPException(status_code=404, detail="Mixed audio file not found")
    return StreamingResponse(
        open(audio_path, "rb"),
        media_type="audio/wav",
        headers={"Content-Disposition": f'attachment; filename="mix_{job_id}.wav"'},
    )


@app.get("/jobs/{job_id}/source")
async def job_source(job_id: str) -> FileResponse:
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.status != "done" or job.result is None:
        raise HTTPException(status_code=409, detail="Job not finished")
    source_path = Path(str(job.result.get("upload_path", "")))
    if not source_path.exists():
        raise HTTPException(status_code=404, detail="Source file not found")
    return FileResponse(source_path)


@app.get("/mix_video")
@app.post("/mix_video")
async def mix_video(dub_job_id: str, transcribe_job_id: str) -> StreamingResponse:
    dub_job = _jobs.get(dub_job_id)
    asr_job = _jobs.get(transcribe_job_id)
    if dub_job is None or asr_job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    if dub_job.status != "done" or dub_job.result is None:
        raise HTTPException(status_code=409, detail="Dub job not finished")
    dubbed_wav = Path(
        str(dub_job.result.get("mixed_audio_path") or dub_job.result.get("audio_path", ""))
    )
    if not dubbed_wav.exists():
        raise HTTPException(status_code=404, detail="Dubbed audio missing")
    upload_path_str = (asr_job.result or {}).get("upload_path", "")
    upload_path = Path(upload_path_str) if upload_path_str else None
    if not upload_path or not upload_path.exists():
        raise HTTPException(status_code=404, detail="Original video file not found")
    out_mp4 = dubbed_wav.parent / "mixed.mp4"
    subprocess.run([
        "ffmpeg", "-y",
        "-i", str(upload_path),
        "-i", str(dubbed_wav),
        "-map", "0:v:0", "-map", "1:a:0",
        "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
        "-shortest",
        str(out_mp4),
    ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    return StreamingResponse(
        open(out_mp4, "rb"),
        media_type="video/mp4",
        headers={"Content-Disposition": 'attachment; filename="dubbed_video.mp4"'},
    )


@app.get("/jobs/{job_id}/stream")
async def stream_job(job_id: str) -> StreamingResponse:
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.status == "done" and job.result is not None:
        await job.queue.put({"type": "done", "progress": 1.0, "result": job.result})
    elif job.status == "error":
        await job.queue.put({"type": "error", "error": job.error or "unknown error"})
    return StreamingResponse(
        _sse_stream(job),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ── Static UI ────────────────────────────────────────────────────────────────
if UI_OUT.exists():
    app.mount("/_next", StaticFiles(directory=UI_OUT / "_next"), name="nextjs-chunks")

    @app.get("/{full_path:path}")
    async def spa(full_path: str) -> FileResponse:
        candidate = UI_OUT / full_path
        if candidate.is_file():
            return FileResponse(candidate)
        html = UI_OUT / full_path / "index.html"
        if html.is_file():
            return FileResponse(html)
        return FileResponse(UI_OUT / "index.html")
else:
    @app.get("/")
    async def index_fallback() -> FileResponse:
        return FileResponse(HERE / "index.html")


if __name__ == "__main__":
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", 8765))
    uvicorn.run(app, host=host, port=port, log_level="info")
