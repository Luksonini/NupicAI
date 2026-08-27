#!/usr/bin/env python3
from __future__ import annotations

import importlib.metadata
import json
import shutil
import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent

REQUIRED_FILES = {
    "ASR Parakeet": ("models/asr/parakeet-tdt-0.6b-v3.nemo", 2_000_000_000),
    "translator Wegorz": ("models/translate/wegorz_translator_32k_best.pt", 1_000_000_000),
    "tokenizer translatora": ("translate/wegorz.model", 500_000),
    "TTS MiniDualPath": ("models/tts/checkpoints/mini_dualpath_learnedvoice.pt", 1_500_000_000),
    "TTS StyleEnc LSTM": ("models/tts/checkpoints/styleenc128_lstm.pt", 1_500_000_000),
    "Vocos config": ("models/tts/vocos-mel-24khz/config.yaml", 100),
    "Vocos weights": ("models/tts/vocos-mel-24khz/pytorch_model.bin", 50_000_000),
    "bank glosow": ("models/tts/voice_banks/selected_top_voices_current.pt", 10_000),
    "mapa learned voice": ("tts/learned_voice_speaker_map.json", 1_000),
    "vocab TTS": ("tts/vocab_pl_orth_en_ipa_bridge.json", 500),
    "frontend": ("parakeet-ui/out/index.html", 1_000),
    "Deno dla YouTube": ("tools/deno/deno", 10_000_000),
}

REQUIRED_PACKAGES = (
    "torch",
    "torchaudio",
    "nemo_toolkit",
    "vocos",
    "fastapi",
    "uvicorn",
    "pydantic",
    "numpy",
    "soundfile",
    "sentencepiece",
    "num2words",
    "yt-dlp",
    "yt-dlp-ejs",
)


def main() -> int:
    errors: list[str] = []
    print(f"Wegorz Dubbing Studio: {HERE}")

    for label, (relative, min_size) in REQUIRED_FILES.items():
        path = HERE / relative
        size = path.stat().st_size if path.is_file() else 0
        ok = size >= min_size
        print(f"[{'OK' if ok else 'BRAK'}] {label}: {relative} ({size / 1024**2:.1f} MiB)")
        if not ok:
            errors.append(f"{label}: {path}")

    for executable in ("ffmpeg", "ffprobe"):
        path = shutil.which(executable)
        print(f"[{'OK' if path else 'BRAK'}] program {executable}: {path or '-'}")
        if not path:
            errors.append(f"program: {executable}")

    deno = HERE / "tools/deno/deno"
    try:
        deno_version = subprocess.run(
            [str(deno), "--version"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()[0]
        print(f"[OK] runtime YouTube: {deno_version}")
    except (OSError, subprocess.CalledProcessError, IndexError) as exc:
        print(f"[BLAD] runtime YouTube Deno: {exc}")
        errors.append("runtime YouTube Deno")

    for package in REQUIRED_PACKAGES:
        try:
            version = importlib.metadata.version(package)
            print(f"[OK] Python {package}=={version}")
        except importlib.metadata.PackageNotFoundError:
            print(f"[BRAK] Python {package}")
            errors.append(f"pakiet Python: {package}")

    try:
        import torch

        bank = torch.load(
            str(HERE / "models/tts/voice_banks/selected_top_voices_current.pt"),
            map_location="cpu",
            weights_only=False,
        )
        raw_ids = {int(value) for value in bank.get("by_id", {})}
        speaker_map = json.loads((HERE / "tts/learned_voice_speaker_map.json").read_text(encoding="utf-8"))
        missing_ids = sorted(raw_id for raw_id in raw_ids if str(raw_id) not in speaker_map)
        invalid_rows = sorted(
            raw_id for raw_id in raw_ids
            if str(raw_id) in speaker_map and not 0 <= int(speaker_map[str(raw_id)]) < 1535
        )
        ok = not missing_ids and not invalid_rows
        print(
            f"[{'OK' if ok else 'BLAD'}] bank -> learned voice: "
            f"voices={len(raw_ids)}, missing={missing_ids[:5]}, invalid={invalid_rows[:5]}"
        )
        if not ok:
            errors.append("mapowanie banku glosow na learned voice")
    except Exception as exc:
        print(f"[BLAD] walidacja banku glosow: {exc}")
        errors.append("walidacja banku glosow")

    if errors:
        print("\nProjekt nie jest gotowy do uruchomienia:")
        for error in errors:
            print(f"- {error}")
        return 1

    print("\nOK: wszystkie wymagane pliki, programy i pakiety sa obecne.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
