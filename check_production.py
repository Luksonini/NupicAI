#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
from datetime import date
from pathlib import Path


HERE = Path(__file__).resolve().parent

REQUIRED_FILES = {
    "ASR Parakeet": ("models/asr/parakeet-tdt-0.6b-v3.nemo", 2_000_000_000),
    "translator Wegorz": ("models/translate/wegorz_translator_32k_best.pt", 1_000_000_000),
    "tokenizer translatora": ("translate/wegorz.model", 500_000),
    "TTS MiniDualPath": ("models/tts/checkpoints/mini_dualpath_learnedvoice.pt", 1_500_000_000),
    "TTS StyleEnc LSTM": ("models/tts/checkpoints/styleenc128_lstm.pt", 1_500_000_000),
    "TTS TDA-MaskGIT continuity": ("models/tts/checkpoints/minidualpath_bins_maskgit_continuity_ep742.pt", 1_500_000_000),
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


def env_value(name: str) -> str:
    if os.environ.get(name, "").strip():
        return os.environ[name].strip()
    try:
        for raw_line in (HERE / ".env").read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if line.startswith(f"{name}="):
                return line.split("=", 1)[1].strip().strip("\"'")
    except FileNotFoundError:
        pass
    return ""


def _enabled(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _production_config_findings() -> list[str]:
    findings: list[str] = []
    public_url = env_value("NUPICAI_PUBLIC_URL") or env_value("NEXT_PUBLIC_SITE_URL")
    site_url = env_value("NEXT_PUBLIC_SITE_URL") or public_url
    cors_origins = [value.strip() for value in env_value("WEGORZ_CORS_ORIGINS").split(",") if value.strip()]
    allowed_hosts = [value.strip() for value in env_value("NUPICAI_ALLOWED_HOSTS").split(",") if value.strip()]
    admin_emails = [value.strip().lower() for value in env_value("NUPICAI_ADMIN_EMAILS").split(",") if value.strip()]

    if not _enabled(env_value("NUPICAI_SECURE_COOKIES")):
        findings.append("ustaw NUPICAI_SECURE_COOKIES=1 za HTTPS")
    if not public_url.startswith("https://") or ".example" in public_url:
        findings.append("NUPICAI_PUBLIC_URL musi wskazywac prawdziwy adres HTTPS")
    if not site_url.startswith("https://") or ".example" in site_url:
        findings.append("NEXT_PUBLIC_SITE_URL musi wskazywac prawdziwy adres HTTPS")
    if not allowed_hosts or any(value == "*" or "localhost" in value for value in allowed_hosts):
        findings.append("ustaw scisla produkcyjna allowliste NUPICAI_ALLOWED_HOSTS")
    if not cors_origins or any(
        value == "*" or not value.startswith("https://") or "localhost" in value
        for value in cors_origins
    ):
        findings.append("WEGORZ_CORS_ORIGINS musi zawierac tylko produkcyjne originy HTTPS")
    if not admin_emails or any(value.endswith("@example.com") for value in admin_emails):
        findings.append("NUPICAI_ADMIN_EMAILS musi zawierac prawdziwe konto wlasciciela")
    if not env_value("RESEND_API_KEY") or not env_value("NUPICAI_EMAIL_FROM"):
        findings.append("skonfiguruj RESEND_API_KEY i NUPICAI_EMAIL_FROM")

    env_path = HERE / ".env"
    if env_path.is_file() and env_path.stat().st_mode & 0o077:
        findings.append("ogranicz uprawnienia .env poleceniem chmod 600 .env")

    frontend = HERE / "parakeet-ui/out/index.html"
    if frontend.is_file() and site_url:
        html = frontend.read_text(encoding="utf-8", errors="ignore")
        if f'href="{site_url.rstrip("/")}' not in html:
            findings.append("przebuduj frontend z produkcyjnym NEXT_PUBLIC_SITE_URL")
    return findings


def main() -> int:
    parser = argparse.ArgumentParser(description="Sprawdza kompletność wdrożenia NupicAI")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="uznaje nieprodukcyjna konfiguracje HTTPS/poczty/hostow za blad",
    )
    args = parser.parse_args()
    strict = bool(args.strict) or _enabled(env_value("NUPICAI_PRODUCTION"))
    errors: list[str] = []
    warnings: list[str] = []
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
        raw_version = importlib.metadata.version("yt-dlp").split("+")[0]
        year, month, day = (int(value) for value in raw_version.split(".")[:3])
        age_days = (date.today() - date(year, month, day)).days
        if age_days > 45:
            warnings.append(
                f"yt-dlp ma {age_days} dni; przed diagnozowaniem bledow YouTube zaktualizuj yt-dlp[default]"
            )
    except (ValueError, TypeError):
        warnings.append("nie udalo sie ustalic wieku wersji yt-dlp")

    production_findings = _production_config_findings()
    if production_findings:
        target = errors if strict else warnings
        target.extend(f"konfiguracja produkcyjna: {finding}" for finding in production_findings)

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

    if warnings:
        print("\nOstrzezenia produkcyjne:")
        for warning in warnings:
            print(f"- {warning}")

    if errors:
        print("\nProjekt nie jest gotowy do uruchomienia:")
        for error in errors:
            print(f"- {error}")
        return 1

    suffix = " Konfiguracja produkcyjna rowniez przeszla walidacje." if strict else ""
    print(f"\nOK: wszystkie wymagane pliki, programy i pakiety sa obecne.{suffix}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
