#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid
import wave
from pathlib import Path
from typing import Any


APP_DIR = Path(__file__).resolve().parent
EXPORT_ROOT = APP_DIR.parent
REPO_ROOT = EXPORT_ROOT.parent
TTS_DIR = EXPORT_ROOT / "TTS"
MODELS_DIR = EXPORT_ROOT / "models"
TTS_MODELS_DIR = MODELS_DIR / "tts"
VOICES_DIR = APP_DIR / "voices"
DEFAULT_CONFIG = APP_DIR / "parakeet_config.json"

_CONFIG: dict[str, Any] | None = None
_TRANSCRIBER: "ParakeetTranscriber | None" = None
_WEGORZ_LOCAL: dict[str, Any] | None = None
_TTS_CACHE: dict[str, Any] = {}
_SPEAKER_CACHE: dict[tuple[str, str, str], tuple[list[str], str]] = {}


def _resolve_path(value: str | Path, *, base: Path = APP_DIR) -> Path:
    p = Path(str(value))
    if not p.is_absolute():
        p = base / p
    return p


def load_config(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    cfg = json.loads(Path(path).read_text(encoding="utf-8"))
    cfg.setdefault("work_dir", "work")
    cfg.setdefault("outputs_dir", "outputs")
    cfg.setdefault("device", "cuda")
    cfg.setdefault("parakeet_model", str(MODELS_DIR / "asr" / "parakeet-tdt-0.6b-v3.nemo"))
    cfg.setdefault("translation_endpoint", "https://ai.nupic.homes/v1")
    cfg.setdefault("translation_model", "gpt-oss:120b")
    cfg.setdefault("qwen_mtp_model", "qwen3.5:27b-mtp")
    cfg.setdefault("qwen_mtp_35b_model", "qwen3.5:35b-mtp")
    cfg.setdefault("translation_timeout_seconds", 180)
    cfg.setdefault("translation_retry", 2)
    cfg.setdefault("tts_model_file", str(TTS_DIR / "WęgorzTTS3_bilanguage.py"))
    cfg.setdefault("tts_dataset_json", str(TTS_DIR / "manifest_runtime_refs.json"))
    cfg.setdefault("tts_vocab", str(TTS_DIR / "vocab_pl_orth_en_ipa_bridge.json"))
    cfg.setdefault("tts_ckpt", str(TTS_MODELS_DIR / "wegorz_multilingual.pt"))
    cfg.setdefault("tts_voices", str(VOICES_DIR / "voices_curated_synth_public.pt"))
    cfg.setdefault("tts_vocos_dir", str(TTS_MODELS_DIR / "vocos-mel-24khz"))
    cfg.setdefault("wegorz_ckpt", str(MODELS_DIR / "translate" / "wegorz_translator_32k_best.pt"))
    cfg.setdefault("wegorz_tokenizer", str(APP_DIR / "wegorz.model"))
    return cfg


def require_executable(name: str) -> None:
    if shutil.which(name) is None:
        raise RuntimeError(f"Brak programu w PATH: {name}")


def yt_dlp_command() -> list[str]:
    exe = shutil.which("yt-dlp")
    if exe:
        return [exe]
    return [sys.executable, "-m", "yt_dlp"]


def _ffprobe_duration(path: Path) -> float:
    require_executable("ffprobe")
    out = subprocess.check_output(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        text=True,
    ).strip()
    try:
        return max(0.0, float(out))
    except Exception:
        return 0.0


def convert_to_mono16k_wav(input_path: str | Path, work_dir: str | Path) -> Path:
    require_executable("ffmpeg")
    src = Path(str(input_path))
    out_dir = _resolve_path(work_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dst = out_dir / f"{src.stem}_{uuid.uuid4().hex[:10]}_mono16k.wav"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(src),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            str(dst),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return dst


class ParakeetTranscriber:
    def __init__(self) -> None:
        assert _CONFIG is not None
        import torch
        from nemo.collections.asr.models import EncDecRNNTBPEModel

        device = str(_CONFIG.get("device", "cuda"))
        self.device = "cuda" if device.startswith("cuda") and torch.cuda.is_available() else "cpu"
        model_ref = str(_CONFIG.get("parakeet_model", str(MODELS_DIR / "asr" / "parakeet-tdt-0.6b-v3.nemo")))
        model_path = Path(model_ref).expanduser()
        if model_path.exists():
            self.model = EncDecRNNTBPEModel.restore_from(restore_path=str(model_path), map_location=self.device)
        else:
            self.model = EncDecRNNTBPEModel.from_pretrained(model_ref)
        self.model.eval()
        self.model = self.model.to(self.device)

    @staticmethod
    def _hyp_text(item: Any) -> str:
        if isinstance(item, str):
            return item
        if hasattr(item, "text"):
            return str(item.text)
        if isinstance(item, (list, tuple)) and item:
            return ParakeetTranscriber._hyp_text(item[0])
        return str(item or "")

    def transcribe_batch(self, paths: list[str], batch_size: int = 8) -> list[str]:
        out: list[str] = []
        for start in range(0, len(paths), max(1, int(batch_size))):
            chunk = [str(p) for p in paths[start : start + max(1, int(batch_size))]]
            hyps = self.model.transcribe(chunk, batch_size=max(1, int(batch_size)))
            out.extend(" ".join(self._hyp_text(h).split()) for h in hyps)
        return out

    @staticmethod
    def _first_hyp(item: Any) -> Any:
        if isinstance(item, (list, tuple)) and item:
            return ParakeetTranscriber._first_hyp(item[0])
        return item

    @staticmethod
    def _stamp_text(stamp: dict[str, Any]) -> str:
        for key in ("word", "segment", "char", "text", "value"):
            if key in stamp:
                return str(stamp.get(key) or "")
        return ""

    @staticmethod
    def _wav_duration(path: str | Path) -> float:
        try:
            with wave.open(str(path), "rb") as wav:
                return float(wav.getnframes()) / max(1.0, float(wav.getframerate()))
        except Exception:
            return 0.0

    def _hyp_language(self, hyp: Any, text: str = "") -> str:
        for attr in ("language", "lang", "language_id", "lang_id", "predicted_language", "predicted_lang"):
            value = getattr(hyp, attr, None)
            if isinstance(value, str) and value.strip():
                return self._normalize_lang_code(value)
        token_ids = getattr(hyp, "y_sequence", None)
        if token_ids is not None:
            try:
                if hasattr(token_ids, "detach"):
                    token_ids = token_ids.detach().cpu().tolist()
                token_ids = [int(x) for x in token_ids]
                tokenizer = getattr(self.model, "tokenizer", None)
                token_strings: list[str] = []
                if tokenizer is not None:
                    for tid in token_ids:
                        tok = None
                        for meth in ("ids_to_tokens", "id_to_token"):
                            fn = getattr(tokenizer, meth, None)
                            if fn is None:
                                continue
                            try:
                                got = fn([tid]) if meth == "ids_to_tokens" else fn(tid)
                                tok = got[0] if isinstance(got, list) and got else got
                                break
                            except Exception:
                                pass
                        if tok is not None:
                            token_strings.append(str(tok))
                joined = " ".join(token_strings)
                m = re.search(r"<\\|([a-z]{2,3})\\|>", joined)
                if m:
                    return self._normalize_lang_code(m.group(1))
            except Exception:
                pass
        return self._guess_language_from_text(text)

    @staticmethod
    def _normalize_lang_code(value: str) -> str:
        s = str(value or "").lower().strip()
        s = s.replace("<|", "").replace("|>", "")
        aliases = {
            "eng": "en",
            "english": "en",
            "pol": "pl",
            "polish": "pl",
            "deu": "de",
            "ger": "de",
            "german": "de",
            "fra": "fr",
            "fre": "fr",
            "french": "fr",
            "spa": "es",
            "spanish": "es",
            "ita": "it",
            "italian": "it",
            "ukr": "uk",
            "ukrainian": "uk",
            "rus": "ru",
            "russian": "ru",
        }
        if s in aliases:
            return aliases[s]
        if re.fullmatch(r"[a-z]{2,3}", s):
            return s[:2]
        return "auto"

    @staticmethod
    def _guess_language_from_text(text: str) -> str:
        s = f" {str(text or '').lower()} "
        if not s.strip():
            return "auto"
        scores = {
            "pl": len(re.findall(r"[ąćęłńóśźż]", s)) * 4 + sum(s.count(f" {w} ") for w in ("że", "jest", "nie", "się", "oraz", "dla", "który", "jak")),
            "de": len(re.findall(r"[äöüß]", s)) * 4 + sum(s.count(f" {w} ") for w in ("der", "die", "das", "und", "nicht", "ist", "ein", "eine", "zu", "mit")),
            "fr": len(re.findall(r"[àâçéèêëîïôûùüÿœ]", s)) * 3 + sum(s.count(f" {w} ") for w in (" le", " la", " les", " des", " une", " est", " pas", " pour")),
            "es": len(re.findall(r"[áéíóúñ¿¡]", s)) * 3 + sum(s.count(f" {w} ") for w in (" el", " la", " los", " una", " que", " para", " con", " está")),
            "it": len(re.findall(r"[àèéìíîòóùú]", s)) * 2 + sum(s.count(f" {w} ") for w in (" il", " lo", " gli", " una", " che", " per", " con", " non")),
            "en": sum(s.count(f" {w} ") for w in ("the", "and", "is", "are", "not", "with", "for", "that", "this", "you")),
        }
        lang, score = max(scores.items(), key=lambda kv: kv[1])
        return lang if score >= 2 else "auto"

    def transcribe_word_timestamps(self, paths: list[str], *, batch_size: int = 1, offsets: list[float] | None = None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        offsets = list(offsets or [0.0] * len(paths))
        min_duration = float((_CONFIG or {}).get("asr_timestamp_min_window_seconds", 1.0))
        filtered: list[tuple[str, float, float]] = []
        skipped: list[dict[str, Any]] = []
        for path, offset in zip(paths, offsets):
            duration = self._wav_duration(path)
            if duration < min_duration:
                skipped.append({"path": str(path), "offset": float(offset), "duration": duration})
                continue
            filtered.append((str(path), float(offset), duration))
        paths = [p for p, _offset, _duration in filtered]
        offsets = [offset for _p, offset, _duration in filtered]
        if not paths:
            return [], {"texts": [], "chunks": [], "skipped_short_audio": skipped}
        words: list[dict[str, Any]] = []
        texts: list[str] = []
        raw_counts: list[dict[str, Any]] = []
        language_counts: dict[str, int] = {}
        for start in range(0, len(paths), max(1, int(batch_size))):
            chunk = [str(p) for p in paths[start : start + max(1, int(batch_size))]]
            chunk_offsets = offsets[start : start + max(1, int(batch_size))]
            chunk_durations = [self._wav_duration(p) for p in chunk]
            try:
                hyps = self.model.transcribe(chunk, batch_size=max(1, int(batch_size)), timestamps=True)
            except TypeError:
                self.model.change_decoding_strategy(decoding_cfg={"compute_timestamps": True})
                hyps = self.model.transcribe(chunk, batch_size=max(1, int(batch_size)))
            for hyp_item, offset, audio_duration in zip(hyps, chunk_offsets, chunk_durations):
                hyp = self._first_hyp(hyp_item)
                text = self._hyp_text(hyp)
                texts.append(" ".join(text.split()))
                lang = self._hyp_language(hyp, text)
                if lang and lang != "auto":
                    language_counts[lang] = language_counts.get(lang, 0) + 1
                timestamp = getattr(hyp, "timestamp", {}) if hyp is not None else {}
                word_items = timestamp.get("word", []) if isinstance(timestamp, dict) else []
                raw_counts.append({"offset": float(offset), "words": len(word_items), "language": lang, "timestamp_keys": list(timestamp.keys()) if isinstance(timestamp, dict) else []})
                local_words: list[dict[str, Any]] = []
                for item in word_items:
                    stamp = dict(item) if isinstance(item, dict) else {"word": str(item)}
                    word = self._stamp_text(stamp).strip()
                    if not word:
                        continue
                    if "start" in stamp and "end" in stamp:
                        start_sec = float(stamp["start"])
                        end_sec = float(stamp["end"])
                    elif "start_offset" in stamp and "end_offset" in stamp:
                        try:
                            stride = 8 * float(self.model.cfg.preprocessor.window_stride)
                        except Exception:
                            stride = 0.08
                        start_sec = float(stamp["start_offset"]) * stride
                        end_sec = float(stamp["end_offset"]) * stride
                    else:
                        continue
                    local_words.append(
                        {
                            "word": word,
                            "start": start_sec,
                            "end": end_sec,
                            "lang": lang,
                        }
                    )
                max_end = max((float(w.get("end", 0.0)) for w in local_words), default=0.0)
                timestamp_scale = 1.0
                if audio_duration > 10.0 and max_end > 0.0 and len(local_words) >= 20:
                    ratio = float(audio_duration) / float(max_end)
                    if 2.0 <= ratio <= 50.0:
                        timestamp_scale = ratio
                    elif 0.02 <= ratio <= 0.5:
                        timestamp_scale = ratio
                if timestamp_scale != 1.0:
                    raw_counts[-1]["timestamp_scale"] = round(timestamp_scale, 6)
                    raw_counts[-1]["audio_duration"] = round(float(audio_duration), 3)
                    raw_counts[-1]["raw_max_word_end"] = round(float(max_end), 3)
                for item in local_words:
                    s = float(item["start"]) * timestamp_scale
                    e = float(item["end"]) * timestamp_scale
                    words.append(
                        {
                            "word": item["word"],
                            "start": round(float(offset) + s, 3),
                            "end": round(float(offset) + max(e, s + 0.02), 3),
                            "lang": item.get("lang", lang),
                        }
                    )
        detected_language = max(language_counts.items(), key=lambda kv: kv[1])[0] if language_counts else self._guess_language_from_text(" ".join(texts))
        return words, {"texts": texts, "chunks": raw_counts, "skipped_short_audio": skipped, "language_counts": language_counts, "detected_language": detected_language}


def get_transcriber() -> ParakeetTranscriber:
    global _TRANSCRIBER
    if _TRANSCRIBER is None:
        _TRANSCRIBER = ParakeetTranscriber()
    return _TRANSCRIBER


def _call_chat_text_api(
    *,
    endpoint: str,
    api_key: str,
    model: str,
    messages: list[dict[str, str]],
    temperature: float,
    timeout: float,
) -> str:
    url = endpoint.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "stream": False,
    }
    if _CONFIG is not None:
        payload["max_tokens"] = int(_CONFIG.get("translation_max_tokens", 16000))
        if bool(_CONFIG.get("translation_disable_thinking", True)):
            payload["think"] = False
            payload["chat_template_kwargs"] = {"enable_thinking": False}
        reasoning = str(_CONFIG.get("translation_reasoning_effort", "") or "").strip()
        if reasoning:
            payload["reasoning"] = {"effort": reasoning}
    req = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=float(timeout)) as response:
            data = json.loads(response.read().decode("utf-8", errors="replace"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"API HTTP {exc.code}: {body[:1000]}") from exc
    return str(data["choices"][0]["message"]["content"])


def _call_chat_json_api(
    *,
    endpoint: str,
    api_key: str,
    model: str,
    messages: list[dict[str, str]],
    temperature: float,
    timeout: float,
) -> str:
    url = endpoint.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "stream": False,
        "response_format": {"type": "json_object"},
    }
    if _CONFIG is not None:
        payload["max_tokens"] = int(_CONFIG.get("translation_max_tokens", 16000))
        if bool(_CONFIG.get("translation_disable_thinking", True)):
            payload["think"] = False
            payload["chat_template_kwargs"] = {"enable_thinking": False}
        reasoning = str(_CONFIG.get("translation_reasoning_effort", "") or "").strip()
        if reasoning:
            payload["reasoning"] = {"effort": reasoning}
    req = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=float(timeout)) as response:
            data = json.loads(response.read().decode("utf-8", errors="replace"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"API HTTP {exc.code}: {body[:1000]}") from exc
    message = data["choices"][0]["message"]
    return str(message.get("content") or message.get("reasoning_content") or "")


def _extract_json_object(text: str) -> dict[str, Any]:
    raw = str(text or "").strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.I).strip()
        raw = re.sub(r"\s*```$", "", raw).strip()
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}")
        if start < 0 or end <= start:
            raise
        obj = json.loads(raw[start : end + 1])
    if not isinstance(obj, dict):
        raise RuntimeError(f"API returned non-object JSON: {type(obj).__name__}")
    return obj


def _parse_json_segments_response(text: str) -> dict[int, str]:
    obj = _extract_json_object(text)
    items = obj.get("segments")
    if not isinstance(items, list):
        raise RuntimeError("JSON response has no segments list")
    out: dict[int, str] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        idx = int(item.get("id"))
        value = item.get("translation", item.get("text", ""))
        out[idx] = " ".join(str(value or "").split())
    return out


def _parse_numbered_response(text: str) -> dict[int, str]:
    out: dict[int, str] = {}
    current: int | None = None
    buf: list[str] = []
    pat = re.compile(r"^\s*\[(\d+)\]\s*(.*)$")
    for line in str(text or "").splitlines():
        m = pat.match(line)
        if m:
            if current is not None:
                out[current] = " ".join(" ".join(buf).split())
            current = int(m.group(1))
            buf = [m.group(2).strip()]
        elif current is not None:
            buf.append(line.strip())
    if current is not None:
        out[current] = " ".join(" ".join(buf).split())
    return out


def _check_numbered_indices(parsed: dict[int, str], expected: list[int]) -> dict[str, Any]:
    got_set = set(parsed)
    exp_set = set(expected)
    return {
        "got": len(parsed),
        "expected": len(expected),
        "missing": sorted(exp_set - got_set),
        "extra": sorted(got_set - exp_set),
        "empty": sorted(idx for idx in exp_set if idx in parsed and not str(parsed[idx]).strip()),
    }


def _translate_api_numbered(
    segments: list[dict[str, Any]],
    *,
    api_key: str,
    endpoint: str,
    model: str,
    batch_segments: int,
    temperature: float,
    timeout: float,
    retry: int,
    progress_callback: Any | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not api_key:
        raise RuntimeError("Brak API key. Podaj w UI albo ustaw NUPIC_API_KEY.")
    target_lang = str((_CONFIG or {}).get("translation_target_lang", "pl")).lower().strip()
    source_lang = str((_CONFIG or {}).get("translation_source_lang", "auto")).lower().strip()
    lang_names = {
        "auto": "the source language detected from the transcript",
        "en": "English",
        "pl": "Polish",
        "de": "German",
        "fr": "French",
        "es": "Spanish",
        "it": "Italian",
        "uk": "Ukrainian",
        "ru": "Russian",
    }
    source_name = lang_names.get(source_lang, source_lang.upper())
    if target_lang == "en":
        direction = f"{source_name}->English"
        target_name = "English"
        tts_note = "Normalize only the translated output for English TTS. Write numbers and symbols naturally in English when appropriate."
        abbrev_note = "For English TTS, prefer spoken forms for unclear acronyms when natural, for example AAA -> triple A, i.e. -> that is."
    else:
        direction = f"{source_name}->Polish"
        target_name = "Polish"
        tts_note = "Normalizuj tylko przetlumaczony wynik pod polski TTS: liczby i symbole zapisuj naturalnie po polsku, gdy to pasuje."
        abbrev_note = "For Polish TTS, prefer Polish spoken letter forms for unclear acronyms when natural, for example AAA -> a a a, CNN -> ce en en, API -> a pe i, i.e. -> to znaczy."
    system = (
        f"You are a precise {direction} translator for dubbing/TTS. "
        "The input is an automatic speech transcription from audio, so it can contain ASR mistakes, broken boundaries, "
        "wrong casing, homophones, and misrecognized names. "
        "Translate each numbered input segment into exactly one numbered output segment. "
        "Use nearby segments only as context for meaning; do not merge, skip, summarize, continue, replace, or renumber segments. "
        "You may lightly fix obvious ASR errors only when the intended meaning is clear. "
        "Do not invent facts, numbers, names, or claims that are not supported by the transcript. "
        "Expand or spell out ambiguous abbreviations, acronyms, and letter-by-letter forms when needed for clear TTS pronunciation "
        "(for example i.e., e.g., AAA, CNN, API, ASR), but only when the expansion is obvious from context. "
        f"{abbrev_note} "
        "Return only numbered lines in the exact format [1] text. "
        "Keep exactly the local input numbers. "
        "Do not add markdown, JSON, comments, explanations, titles, or extra text. "
        "Translate idioms by meaning, not literally. "
        "If a segment is clearly a fragment, translate it as a fragment; do not complete it using invented content. "
        f"Translate into {target_name}. {tts_note}"
    )
    meta_chunks: list[dict[str, Any]] = []
    max_split_depth = int((_CONFIG or {}).get("translation_numbered_split_depth", 4))

    def _translate_chunk_numbered(chunk: list[dict[str, Any]], start: int, depth: int = 0) -> list[dict[str, Any]]:
        expected_local = list(range(1, len(chunk) + 1))
        prompt = "\n".join(
            [
                (
                    f"Translate every segment below into {target_name}. "
                    f"Return exactly {len(chunk)} lines numbered [1] to [{len(chunk)}]. "
                    "One input line = one output line. Do not omit, merge, summarize, replace, or change numbers."
                ),
                *[f"[{i}] {seg.get('text', '')}" for i, seg in enumerate(chunk, start=1)],
            ]
        )
        last_error: Exception | None = None
        response_text = ""
        t0 = time.perf_counter()
        check: dict[str, Any] = {}
        for attempt in range(max(1, int(retry) + 1)):
            try:
                user_prompt = prompt
                if attempt > 0:
                    user_prompt = (
                        f"Poprzednia odpowiedz miala bledna numeracje. "
                        f"Teraz zwroc DOKLADNIE {len(chunk)} linii: [1] ... [{len(chunk)}]. "
                        "Nie uzywaj numerow globalnych. Nie pomijaj segmentow.\n\n"
                        + prompt
                    )
                response_text = _call_chat_text_api(
                    endpoint=endpoint,
                    api_key=api_key,
                    model=model,
                    messages=[
                        {"role": "system", "content": system},
                        {
                            "role": "user",
                            "content": (
                                "Przyklad formatu:\n[1] She was beating around the bush.\n[2] The Fourier transform reveals hidden cycles."
                                if target_lang != "en"
                                else "Format example:\n[1] Owijała w bawełnę.\n[2] Transformata Fouriera ujawnia ukryte cykle."
                            ),
                        },
                        {
                            "role": "assistant",
                            "content": (
                                "[1] Owijała w bawełnę.\n[2] Transformata Fouriera ujawnia ukryte cykle."
                                if target_lang != "en"
                                else "[1] She was beating around the bush.\n[2] The Fourier transform reveals hidden cycles."
                            ),
                        },
                        {"role": "user", "content": "/no_think\n" + user_prompt},
                    ],
                    temperature=temperature,
                    timeout=timeout,
                )
                parsed = _parse_numbered_response(response_text)
                check = _check_numbered_indices(parsed, expected_local)
                if not check["missing"] and not check["extra"] and not check["empty"]:
                    break
                raise RuntimeError(f"Niepelna numeracja: {check}")
            except Exception as exc:
                last_error = exc
                if attempt >= int(retry):
                    if len(chunk) > 1 and depth < max_split_depth:
                        mid = max(1, len(chunk) // 2)
                        meta_chunks.append(
                            {
                                "start": start,
                                "count": len(chunk),
                                "seconds": round(time.perf_counter() - t0, 3),
                                "check": check,
                                "fallback": f"split {len(chunk)} -> {mid}+{len(chunk) - mid}",
                                "depth": depth,
                            }
                        )
                        return _translate_chunk_numbered(chunk[:mid], start, depth + 1) + _translate_chunk_numbered(chunk[mid:], start + mid, depth + 1)
                    preview = response_text[:1000] if response_text else ""
                    raise RuntimeError(f"Nie udało się przetłumaczyć batcha. Ostatni błąd: {last_error}\nOdpowiedź preview:\n{preview}") from exc
                time.sleep(1.0 + attempt)
        parsed = _parse_numbered_response(response_text)
        out = [{**seg, "text": parsed[i]} for i, seg in enumerate(chunk, start=1)]
        meta_chunks.append({"start": start, "count": len(chunk), "seconds": round(time.perf_counter() - t0, 3), "check": check, "depth": depth})
        return out

    translated: list[dict[str, Any]] = []
    step = max(1, int(batch_segments))
    total_batches = max(1, (len(segments) + step - 1) // step)
    for batch_idx, start in enumerate(range(0, len(segments), step), start=1):
        chunk = segments[start : start + step]
        if progress_callback is not None:
            progress_callback(batch_idx - 1, total_batches, start, len(chunk), "start")
        translated.extend(_translate_chunk_numbered(chunk, start))
        if progress_callback is not None:
            progress_callback(batch_idx, total_batches, start, len(chunk), "done")
    return translated, {"mode": "api_numbered_export", "model": model, "target_lang": target_lang, "chunks": meta_chunks}


def _translate_api_json_overlap(
    segments: list[dict[str, Any]],
    *,
    api_key: str,
    endpoint: str,
    model: str,
    batch_segments: int,
    temperature: float,
    timeout: float,
    retry: int,
    progress_callback: Any | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not api_key:
        raise RuntimeError("Brak API key. Podaj w UI albo ustaw NUPIC_API_KEY.")
    target_lang = str((_CONFIG or {}).get("translation_target_lang", "pl")).lower().strip()
    source_lang = str((_CONFIG or {}).get("translation_source_lang", "auto")).lower().strip()
    lang_names = {
        "auto": "the source language detected from the transcript",
        "en": "English",
        "pl": "Polish",
        "de": "German",
        "fr": "French",
        "es": "Spanish",
        "it": "Italian",
        "uk": "Ukrainian",
        "ru": "Russian",
    }
    source_name = lang_names.get(source_lang, source_lang.upper())
    if target_lang == "en":
        direction = f"{source_name}->English"
        target_name = "English"
        abbrev_note = "For English TTS, prefer spoken forms for unclear acronyms when natural, for example AAA -> triple A, i.e. -> that is."
    else:
        direction = f"{source_name}->Polish"
        target_name = "Polish"
        abbrev_note = "For Polish TTS, prefer Polish spoken letter forms for unclear acronyms when natural, for example AAA -> a a a, CNN -> ce en en, API -> a pe i, i.e. -> to znaczy."
    system = (
        f"You are a precise {direction} translator for dubbing. "
        "The input is an automatic speech transcription from audio, so it can contain ASR mistakes, broken boundaries, "
        "wrong casing, homophones, and misrecognized names. "
        "Translate only the items in the JSON field segments. "
        "The optional context_before field is previous transcript context only; do not output it. "
        "Use context_before and nearby segments to preserve meaning, references, names, and sentence continuity. "
        "Preserve segmentation strictly: each input segment must produce exactly one output segment with the same local id. "
        "Do not merge, split, skip, summarize, continue, replace, or renumber segments. "
        "You may lightly fix obvious ASR errors only when the intended meaning is clear. "
        "Do not invent facts, numbers, names, or claims not supported by the transcript. "
        "Translate naturally and by meaning. If a segment is clearly a fragment, translate it as a fragment. "
        "Expand or spell out ambiguous abbreviations, acronyms, and letter-by-letter forms when needed for clear TTS pronunciation "
        "(for example i.e., e.g., AAA, CNN, API, ASR), but only when the expansion is obvious from context. "
        f"{abbrev_note} "
        "Do not normalize for TTS; keep ordinary readable translation. A deterministic TTS normalizer will run later. "
        f"Translate into {target_name}. Return only valid JSON: "
        '{"segments":[{"id":1,"translation":"..."}]}'
    )
    prefix_messages = [
        {"role": "system", "content": system},
        {
            "role": "user",
            "content": json.dumps(
                {
                    "context_before": [],
                    "segments": [
                        {"id": 1, "text": "She was beating around the bush."},
                        {"id": 2, "text": "The Fourier transform reveals hidden cycles."},
                    ],
                },
                ensure_ascii=False,
            ),
        },
        {
            "role": "assistant",
            "content": json.dumps(
                {
                    "segments": (
                        [
                            {"id": 1, "translation": "Owijała w bawełnę."},
                            {"id": 2, "translation": "Transformata Fouriera ujawnia ukryte cykle."},
                        ]
                        if target_lang != "en"
                        else [
                            {"id": 1, "translation": "She was beating around the bush."},
                            {"id": 2, "translation": "The Fourier transform reveals hidden cycles."},
                        ]
                    )
                },
                ensure_ascii=False,
            ),
        },
    ]
    meta_chunks: list[dict[str, Any]] = []
    max_split_depth = int((_CONFIG or {}).get("translation_numbered_split_depth", 4))

    def _payload_for_chunk(chunk: list[dict[str, Any]], start: int) -> str:
        context_before = []
        if start > 0 and start - 1 < len(segments):
            prev = segments[start - 1]
            prev_text = " ".join(str(prev.get("text", "")).split())
            if prev_text:
                context_before.append({"id": int(prev.get("index", start - 1)), "text": prev_text})
        payload = {
            "task": "translate_segments_for_dubbing",
            "source": "automatic_audio_transcript",
            "source_language": source_name,
            "target_language": target_name,
            "instructions": {
                "translate_only_segments": True,
                "context_before_is_not_output": True,
                "preserve_segmentation": True,
                "one_output_per_input": True,
                "do_not_merge_or_split": True,
                "return_json_only": True,
                "do_not_normalize_for_tts": True,
                "expand_obvious_abbreviations_for_tts": True,
            },
            "context_before": context_before,
            "segments": [
                {"id": i, "text": " ".join(str(seg.get("text", "")).split())}
                for i, seg in enumerate(chunk, start=1)
            ],
        }
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))

    def _translate_chunk_json(chunk: list[dict[str, Any]], start: int, depth: int = 0) -> list[dict[str, Any]]:
        expected_local = list(range(1, len(chunk) + 1))
        payload = _payload_for_chunk(chunk, start)
        last_error: Exception | None = None
        response_text = ""
        t0 = time.perf_counter()
        check: dict[str, Any] = {}
        for attempt in range(max(1, int(retry) + 1)):
            try:
                user_text = payload
                if attempt > 0:
                    user_text = (
                        f"Previous response had invalid JSON or wrong ids. Return exactly {len(chunk)} objects "
                        f"in segments with local id 1 to {len(chunk)}. Do not output context_before.\n\n"
                        + payload
                    )
                response_text = _call_chat_json_api(
                    endpoint=endpoint,
                    api_key=api_key,
                    model=model,
                    messages=prefix_messages + [{"role": "user", "content": "/no_think\n" + user_text}],
                    temperature=temperature,
                    timeout=timeout,
                )
                parsed = _parse_json_segments_response(response_text)
                check = _check_numbered_indices(parsed, expected_local)
                if not check["missing"] and not check["extra"] and not check["empty"]:
                    break
                raise RuntimeError(f"Niepelna numeracja JSON: {check}")
            except Exception as exc:
                last_error = exc
                if attempt >= int(retry):
                    if len(chunk) > 1 and depth < max_split_depth:
                        mid = max(1, len(chunk) // 2)
                        meta_chunks.append(
                            {
                                "start": start,
                                "count": len(chunk),
                                "seconds": round(time.perf_counter() - t0, 3),
                                "check": check,
                                "fallback": f"split {len(chunk)} -> {mid}+{len(chunk) - mid}",
                                "depth": depth,
                            }
                        )
                        return _translate_chunk_json(chunk[:mid], start, depth + 1) + _translate_chunk_json(chunk[mid:], start + mid, depth + 1)
                    preview = response_text[:1000] if response_text else ""
                    raise RuntimeError(f"Nie udało się przetłumaczyć batcha JSON. Ostatni błąd: {last_error}\nOdpowiedź preview:\n{preview}") from exc
                time.sleep(1.0 + attempt)
        parsed = _parse_json_segments_response(response_text)
        out = [{**seg, "text": parsed[i]} for i, seg in enumerate(chunk, start=1)]
        meta_chunks.append({"start": start, "count": len(chunk), "seconds": round(time.perf_counter() - t0, 3), "check": check, "depth": depth, "context_before": 1 if start > 0 else 0})
        return out

    translated: list[dict[str, Any]] = []
    step = max(1, int(batch_segments))
    total_batches = max(1, (len(segments) + step - 1) // step)
    for batch_idx, start in enumerate(range(0, len(segments), step), start=1):
        chunk = segments[start : start + step]
        if progress_callback is not None:
            progress_callback(batch_idx - 1, total_batches, start, len(chunk), "start")
        translated.extend(_translate_chunk_json(chunk, start))
        if progress_callback is not None:
            progress_callback(batch_idx, total_batches, start, len(chunk), "done")
    return translated, {"mode": "api_json_overlap", "model": model, "target_lang": target_lang, "chunks": meta_chunks, "context_overlap_segments": 1}


def _translate_local_wegorz(segments: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    assert _CONFIG is not None
    if str(_CONFIG.get("translation_target_lang", "pl")).lower().strip() != "pl":
        raise RuntimeError("Lokalny Węgorz w tym pipeline obsługuje tylko EN->PL. Dla PL->EN wybierz Qwen/API numbered.")
    ckpt = _resolve_path(_CONFIG["wegorz_ckpt"])
    tokenizer = _resolve_path(_CONFIG["wegorz_tokenizer"])
    if not ckpt.exists():
        raise RuntimeError(f"Lokalny Węgorz: brak checkpointu translatora: {ckpt}")
    if not tokenizer.exists():
        raise RuntimeError(f"Lokalny Węgorz: brak tokenizera: {tokenizer}")

    global _WEGORZ_LOCAL
    t0 = time.perf_counter()
    cache_key = (str(ckpt), str(tokenizer), str(_CONFIG.get("device", "cuda")))
    if _WEGORZ_LOCAL is None or _WEGORZ_LOCAL.get("cache_key") != cache_key:
        if str(APP_DIR) not in sys.path:
            sys.path.insert(0, str(APP_DIR))
        from translate_wegorz_sentence_split import (  # noqa: WPS433
            EnNormalizer,
            clean_ws,
            load_model,
            split_segment_text,
            translate_texts,
        )

        normalizer = EnNormalizer(enabled=True)
        model, sp, device, config_name, vocab_size = load_model(ckpt, tokenizer, str(_CONFIG.get("device", "cuda")))
        _WEGORZ_LOCAL = {
            "cache_key": cache_key,
            "normalizer": normalizer,
            "model": model,
            "sp": sp,
            "device": device,
            "config_name": config_name,
            "vocab_size": vocab_size,
            "load_seconds": time.perf_counter() - t0,
            "helpers": (clean_ws, split_segment_text, translate_texts),
        }

    assert _WEGORZ_LOCAL is not None
    clean_ws, split_segment_text, translate_texts = _WEGORZ_LOCAL["helpers"]
    normalizer = _WEGORZ_LOCAL["normalizer"]
    load_sec = float(_WEGORZ_LOCAL.get("load_seconds", 0.0)) if (time.perf_counter() - t0) > 1.0 else 0.0

    units: list[dict[str, Any]] = []
    merged_segments: list[dict[str, Any]] = []
    max_chars = int(_CONFIG.get("wegorz_split_max_chars", 220))
    for seg_idx, seg in enumerate(segments):
        text = clean_ws(seg.get("text", ""))
        parts = split_segment_text(text, max_chars=max_chars)
        unit_ids: list[int] = []
        for part in parts:
            normalized = normalizer(part)
            unit_ids.append(len(units))
            units.append({"segment_index": seg_idx, "text": part, "text_norm": normalized})
        merged_segments.append({**seg, "split_count": len(parts), "_unit_ids": unit_ids})

    translate_t0 = time.perf_counter()
    translations = translate_texts(
        model=_WEGORZ_LOCAL["model"],
        sp=_WEGORZ_LOCAL["sp"],
        device=_WEGORZ_LOCAL["device"],
        texts=[u["text_norm"] for u in units],
        batch_size=max(1, int(_CONFIG.get("wegorz_batch_size", 16))),
        max_new_tokens=max(8, int(_CONFIG.get("wegorz_max_new_tokens", 180))),
    )
    translate_sec = time.perf_counter() - translate_t0

    if len(translations) != len(units):
        raise RuntimeError(
            f"Lokalny Węgorz zwrócił {len(translations)} wyników dla {len(units)} części; "
            "tłumaczenie zatrzymano, aby nie pominąć tekstu."
        )
    empty_units = [i for i, value in enumerate(translations) if not str(value).strip()]
    if empty_units:
        preview = ", ".join(str(i) for i in empty_units[:10])
        raise RuntimeError(f"Lokalny Węgorz zwrócił puste tłumaczenia części: {preview}")

    for unit, tr in zip(units, translations):
        unit["translation_pl"] = tr

    for seg in merged_segments:
        unit_ids = seg.pop("_unit_ids")
        sub = [units[i] for i in unit_ids]
        seg["translation_pl"] = clean_ws(" ".join(u.get("translation_pl", "") for u in sub))
        seg["translation_units"] = [
            {"text": u["text"], "text_norm": u["text_norm"], "translation_pl": u.get("translation_pl", "")}
            for u in sub
        ]

    translated = []
    for seg in merged_segments:
        translated.append({**seg, "text": str(seg.get("translation_pl", "")).strip()})
    return translated, {
        "mode": "wegorz_local_sentence_split",
        "seconds": round(time.perf_counter() - t0, 3),
        "load_seconds": round(load_sec, 3),
        "translate_seconds": round(translate_sec, 3),
        "split_units": len(units),
        "checkpoint": str(ckpt),
        "tokenizer": str(tokenizer),
        "config": _WEGORZ_LOCAL.get("config_name"),
        "vocab_size": _WEGORZ_LOCAL.get("vocab_size"),
    }


def translate_segments_to_pl(
    *,
    segments: list[dict[str, Any]],
    api_key: str,
    endpoint: str,
    model: str,
    mode: str,
    break_token: str = "[BREAK]",
    batch_segments: int = 20,
    batch_seconds: float = 0.0,
    temperature: float = 0.1,
    timeout: float = 600.0,
    retry: int = 2,
    progress_callback: Any | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    del break_token, batch_seconds
    if str(mode) == "wegorz_local_sentence_split":
        return _translate_local_wegorz(segments)
    if str(mode) in {"api_json_overlap", "qwen_mtp_json_overlap", "qwen_mtp_35b_json_overlap"}:
        return _translate_api_json_overlap(
            segments,
            api_key=api_key,
            endpoint=endpoint,
            model=model,
            batch_segments=batch_segments,
            temperature=temperature,
            timeout=timeout,
            retry=retry,
            progress_callback=progress_callback,
        )
    return _translate_api_numbered(
        segments,
        api_key=api_key,
        endpoint=endpoint,
        model=model,
        batch_segments=batch_segments,
        temperature=temperature,
        timeout=timeout,
        retry=retry,
        progress_callback=progress_callback,
    )


def _load_manifest_speaker_choices(lang: str) -> list[str]:
    assert _CONFIG is not None
    manifest = Path(str(_CONFIG.get("tts_dataset_json", TTS_DIR / "manifest_runtime_refs.json")))
    if not manifest.exists():
        return []
    data = json.loads(manifest.read_text(encoding="utf-8"))
    seen: dict[int, str] = {}
    del lang
    for row in data:
        row_lang = str(row.get("lang") or row.get("language") or "").lower()
        sid_raw = row.get("speaker_id", row.get("spk_id", row.get("speaker")))
        try:
            sid = int(sid_raw)
        except Exception:
            continue
        if sid in seen:
            continue
        name = str(row.get("speaker_name") or row.get("speaker") or row.get("book_id") or f"speaker_{sid}")
        gender = str(row.get("gender") or "").strip().upper()
        if not gender:
            gender_id = row.get("gender_id", None)
            gender = "F" if gender_id == 1 else "M" if gender_id == 0 else "?"
        lang_label = (row_lang or "?").upper()
        prefix = "[R]"
        seen[sid] = f"{prefix} {lang_label} {gender} {name} [{sid}]"
    return list(seen.values())


def _load_synthetic_voice_choices(lang: str = "PL") -> list[str]:
    assert _CONFIG is not None
    voices_path = Path(str(_CONFIG.get("tts_voices", VOICES_DIR / "voices_curated_synth_public.pt")))
    if not voices_path.exists():
        return []
    import torch

    payload = torch.load(str(voices_path), map_location="cpu", weights_only=False)
    out: list[str] = []
    want_lang = str(lang or "").upper().strip()
    for sid, item in sorted((payload.get("by_id") or {}).items(), key=lambda kv: int(kv[0])):
        if not bool(item.get("synthetic", False)):
            continue
        sid_i = int(item.get("speaker_id", sid))
        lang_label = str(item.get("lang", "?")).upper()
        if want_lang and want_lang not in {"ALL", "*"} and lang_label != want_lang:
            continue
        gender = str(item.get("gender", "?")).upper()
        name = str(item.get("name_full") or item.get("speaker_name_raw") or sid_i)
        out.append(f"[S] {lang_label} {gender} {name} [{sid_i}]")
    return out


def get_tts_speaker_choices(show: str = "synth", lang: str = "PL") -> tuple[list[str], str]:
    assert _CONFIG is not None
    key = (str(show or "all"), str(lang or "PL"), str(_CONFIG.get("tts_dataset_json", "")), str(_CONFIG.get("tts_voices", "")))
    if key in _SPEAKER_CACHE:
        return _SPEAKER_CACHE[key]
    show_s = str(show or "all").lower().strip()
    synth_choices = _load_synthetic_voice_choices(lang) if show_s in {"synth", "all"} else []
    real_choices = _load_manifest_speaker_choices(lang) if show_s in {"real", "all"} else []
    choices = synth_choices + real_choices
    if not choices:
        default = str(_CONFIG.get("tts_default_speaker", "[R] PL speaker [0]"))
        choices = [default]
    default_cfg = str(_CONFIG.get("tts_default_speaker", "")).strip()
    value = default_cfg if default_cfg in choices else choices[0]
    _SPEAKER_CACHE[key] = (choices, value)
    return choices, value
