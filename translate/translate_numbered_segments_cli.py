#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


APP_DIR = Path(__file__).resolve().parent


def build_numbered_prompt(segments: list[dict[str, Any]], start_index: int = 1) -> tuple[str, list[dict[str, Any]]]:
    usable = [seg for seg in segments if str(seg.get("text", "")).strip()]
    lines: list[str] = []
    lines.append("Przetłumacz poniższe segmenty z angielskiego na polski.")
    lines.append("To są segmenty czasowe z transkrypcji audio, niekoniecznie pojedyncze zdania.")
    lines.append("Zasady:")
    lines.append("- tłumacz segment po segmencie")
    lines.append(f"- zachowaj dokładnie nagłówki segmentów w formacie [{start_index}], [{start_index + 1}], [{start_index + 2}]")
    lines.append("- liczba segmentów w odpowiedzi musi być taka sama jak w wejściu")
    lines.append("- nie dodawaj JSON, markdown, komentarzy ani list punktowanych")
    lines.append("- po każdym nagłówku wpisz tylko polskie tłumaczenie tego segmentu")
    lines.append("- zachowuj osobę, liczbę i czas gramatyczny")
    lines.append("- normalizuj tekst pod TTS: liczby i symbole zapisuj naturalnie po polsku, gdy to pasuje")
    lines.append("- idiomy tłumacz znaczeniowo, nie dosłownie")
    lines.append("- zachowuj terminologię specjalistyczną po polsku")
    lines.append("")
    lines.append("SEGMENTY:")
    lines.append("")
    for out_idx, seg in enumerate(usable, start_index):
        text = " ".join(str(seg.get("text", "")).split())
        lines.append(f"[{out_idx}] {text}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n", usable


def parse_numbered_segments(text: str) -> dict[int, str]:
    pattern = re.compile(r"(?ms)^\s*\[(\d+)\]\s*(.*?)(?=^\s*\[\d+\]\s*|\Z)")
    parsed: dict[int, str] = {}
    for match in pattern.finditer(text.strip()):
        parsed[int(match.group(1))] = match.group(2).strip()
    return parsed


def call_openai_compatible(
    endpoint: str,
    api_key: str,
    model: str,
    prompt: str,
    temperature: float,
    max_tokens: int,
    timeout: float,
) -> tuple[str, dict[str, Any]]:
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "Jesteś precyzyjnym tłumaczem EN->PL. Zwracasz tylko ponumerowane segmenty.",
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": float(temperature),
        "max_tokens": int(max_tokens),
    }
    req = urllib.request.Request(
        endpoint.rstrip("/") + "/chat/completions",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
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
    content = str(message.get("content") or "").strip()
    if not content:
        content = str(message.get("reasoning_content") or "").strip()
    if not content:
        raise RuntimeError(f"API returned empty content; usage={data.get('usage')}")
    return content, data.get("usage", {})


def call_gemini(
    api_key: str,
    model: str,
    prompt: str,
    temperature: float,
    max_tokens: int,
    timeout: float,
) -> tuple[str, dict[str, Any]]:
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": float(temperature),
            "maxOutputTokens": int(max_tokens),
        },
    }
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    req = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=float(timeout)) as response:
            data = json.loads(response.read().decode("utf-8", errors="replace"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Gemini HTTP {exc.code}: {body[:1000]}") from exc
    content = (
        data.get("candidates", [{}])[0]
        .get("content", {})
        .get("parts", [{}])[0]
        .get("text", "")
    )
    if not str(content).strip():
        raise RuntimeError(f"Gemini returned empty content; usage={data.get('usageMetadata')}")
    return str(content).strip(), data.get("usageMetadata", {})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True, help="Input JSON with segments[].text.")
    ap.add_argument("--out-txt", default="", help="Raw numbered translation output txt.")
    ap.add_argument("--out-json", default="", help="Merged JSON with translation_segments.")
    ap.add_argument("--prompt-out", default="", help="Save generated prompt to txt.")
    ap.add_argument("--model", default="gpt-oss:120b")
    ap.add_argument("--endpoint", default="https://ai.nupic.homes/v1")
    ap.add_argument("--api-key", default="")
    ap.add_argument("--temperature", type=float, default=0.1)
    ap.add_argument("--max-tokens", type=int, default=16000)
    ap.add_argument("--timeout", type=float, default=600)
    ap.add_argument("--start-index", type=int, default=1)
    ap.add_argument("--dry-run", action="store_true", help="Only write prompt, do not call API.")
    args = ap.parse_args()

    in_path = Path(args.json)
    payload = json.loads(in_path.read_text(encoding="utf-8"))
    segments = payload.get("segments", [])
    if not isinstance(segments, list) or not segments:
        raise RuntimeError("Input JSON must contain non-empty segments list.")

    prompt, usable = build_numbered_prompt(segments, start_index=int(args.start_index))
    prompt_out = Path(args.prompt_out) if args.prompt_out else in_path.with_name(in_path.stem + "_numbered_prompt.txt")
    prompt_out.write_text(prompt, encoding="utf-8")
    print(f"prompt: {prompt_out}")
    print(f"segments: {len(segments)} usable_nonempty: {len(usable)} prompt_chars: {len(prompt)}")
    if args.dry_run:
        return

    model = str(args.model)
    key_env = "GEMINI_API_KEY" if model.startswith("gemini") else "NUPIC_API_KEY"
    api_key = str(args.api_key or os.environ.get(key_env, "")).strip()
    if not api_key:
        raise RuntimeError(f"Missing API key. Pass --api-key or set {key_env}.")

    t0 = time.perf_counter()
    if model.startswith("gemini"):
        content, usage = call_gemini(
            api_key=api_key,
            model=model,
            prompt=prompt,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            timeout=args.timeout,
        )
    else:
        content, usage = call_openai_compatible(
            endpoint=args.endpoint,
            api_key=api_key,
            model=model,
            prompt=prompt,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            timeout=args.timeout,
        )
    elapsed = time.perf_counter() - t0

    out_txt = Path(args.out_txt) if args.out_txt else in_path.with_name(in_path.stem + f"_{model.replace(':', '_')}_numbered_translation.txt")
    out_txt.write_text(content.rstrip() + "\n", encoding="utf-8")
    parsed = parse_numbered_segments(content)
    expected = list(range(int(args.start_index), int(args.start_index) + len(usable)))
    missing = [idx for idx in expected if idx not in parsed]
    print(f"response: {out_txt}")
    print(f"elapsed_sec: {elapsed:.2f}")
    print(f"usage: {usage}")
    print(f"parsed: {len(parsed)} expected: {len(expected)} missing: {missing[:50]} count={len(missing)}")

    out_json = Path(args.out_json) if args.out_json else in_path.with_name(in_path.stem + f"_{model.replace(':', '_')}_numbered_merged.json")
    translation_segments: list[dict[str, Any]] = []
    usable_by_object = {id(seg): out_idx for out_idx, seg in enumerate(usable, int(args.start_index))}
    for seg in segments:
        out_idx = usable_by_object.get(id(seg))
        pl = parsed.get(out_idx, "") if out_idx is not None else ""
        translation_segments.append(
            {
                **seg,
                "source_text": str(seg.get("text", "")),
                "translation_text": pl,
                "text": pl,
            }
        )
    merged = {
        **payload,
        "translation_text": " ".join(seg.get("translation_text", "") for seg in translation_segments).strip(),
        "translation_segments": translation_segments,
        "translation_import": {
            "mode": "numbered_segments_cli",
            "model": model,
            "prompt": str(prompt_out),
            "response": str(out_txt),
            "elapsed_sec": round(elapsed, 3),
            "usage": usage,
            "missing": missing,
        },
    }
    out_json.write_text(json.dumps(merged, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"merged_json: {out_json}")


if __name__ == "__main__":
    main()


'''
/home/rizos/Miniforge3/envs/uvtts2/bin/python translate_numbered_segments_cli.py \
  --json /home/rizos/Downloads/SalmonTTS2/inference_multilanguage/transcription_and_translation/outputs/odpowiedź_clean_merged.json \
'''