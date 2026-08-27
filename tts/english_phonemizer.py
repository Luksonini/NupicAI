#!/usr/bin/env python3
from __future__ import annotations

import re
import subprocess
from functools import lru_cache


SPACE_RE = re.compile(r"\s+")


def _tidy_ipa_spacing(text: str) -> str:
    out = SPACE_RE.sub(" ", str(text or "")).strip()
    return re.sub(r"\s+([.,!?;:])", r"\1", out)


def _voice_is_british(voice: str) -> bool:
    s = str(voice or "").strip().lower()
    return any(tag in s for tag in ("en-gb", "en-uk", "english_rp", "english-north", "english_wmids"))


def phonemize_en_ipa_espeak(text: str, voice: str = "en-us") -> str:
    proc = subprocess.run(
        ["espeak-ng", "-q", "--ipa=3", "-v", str(voice), "--", str(text)],
        capture_output=True,
        text=True,
        check=True,
    )
    return _tidy_ipa_spacing(proc.stdout)


@lru_cache(maxsize=4)
def _get_misaki_g2p(british: bool):
    from misaki import en as misaki_en
    from misaki import espeak as misaki_espeak

    fallback = misaki_espeak.EspeakFallback(british=bool(british))
    return misaki_en.G2P(trf=False, british=bool(british), fallback=fallback)


def phonemize_en_ipa(text: str, voice: str = "en-us", backend: str = "misaki_no_trf") -> str:
    backend_s = str(backend or "misaki_no_trf").strip().lower()
    british = _voice_is_british(voice)

    if backend_s not in {"auto", "espeak", "misaki_no_trf"}:
        raise ValueError(f"Unsupported EN phonemizer backend: {backend!r}")

    if backend_s == "espeak":
        return phonemize_en_ipa_espeak(text, voice=voice)

    if backend_s in {"auto", "misaki_no_trf"}:
        try:
            g2p = _get_misaki_g2p(bool(british))
            out = g2p(str(text))
            if isinstance(out, tuple):
                phon = out[0]
            else:
                phon = out
            phon_s = _tidy_ipa_spacing(str(phon))
            if phon_s:
                return phon_s
        except Exception:
            if backend_s == "misaki_no_trf":
                return phonemize_en_ipa_espeak(text, voice=voice)

    return phonemize_en_ipa_espeak(text, voice=voice)
