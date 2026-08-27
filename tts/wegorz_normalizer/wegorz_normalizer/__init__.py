"""
WęgorzAI Polish TTS Text Normalizer

Standalone text normalization pipeline for Polish TTS synthesis.
Converts raw Polish text into speech-ready form: numbers → words,
abbreviations → expansions, foreign terms → Polish phonetics.

Usage:
    from wegorz_normalizer import PolishTTSPipeline

    pipe = PolishTTSPipeline()
    result = pipe.process("Spotkanie o godz. 14:30 kosztuje 29,99 zł.")
    # → "spotkanie o godzinie czternastej trzydzieści kosztuje dwadzieścia dziewięć złotych dziewięćdziesiąt dziewięć groszy."
"""

from .tokenize_and_text_norm import (
    FileReader,
    PLTokenizer,
    PolishTTSPipeline,
    _count_unk,
    _process_file,
    _tokenize,
    run_demo,
)

__version__ = "1.0.0"
__all__ = [
    "FileReader",
    "PLTokenizer",
    "PolishTTSPipeline",
    "_count_unk",
    "_process_file",
    "_tokenize",
    "run_demo",
]
