#!/usr/bin/env python3
"""
Normalizacja korpusu EN→PL dla WegorzTranslator.

Strona EN:  NeMo Normalizer (lang='en', input_case='cased')
            liczby → słowa EN, daty, skróty, jednostki
Strona PL:  PolishTTSPipeline z wegorz_normalizer
            liczby → słowa PL, skróty PL, daty PL

Opcja --cap-encode (domyślnie włączona):
  Zamienia wielkie litery na <CAP> + mała litera.
  'Szkoła w Warszawie' → '<CAP>szkoła w <CAP>warszawie'
  Pozwala trenować BPE na czysto lowercase tekście.
  Inference: BPE decode → cap_decode() → encode_pl() → TTS (bez normalizacji)

Użycie:
  conda run -n uvtts2 python normalize_corpus.py \\
      --input  data/train.jsonl \\
      --output data/train_norm.jsonl \\
      --workers 5

  # Bez cap-encode (zwykła normalizacja):
  conda run -n uvtts2 python normalize_corpus.py \\
      --input  data/train.jsonl \\
      --output data/train_norm_plain.jsonl \\
      --workers 5 --no-cap-encode
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import re
import sys
import time
from pathlib import Path

# ── Wzorce decydujące czy para wymaga normalizacji ────────────────────────────

_NEEDS_EN = re.compile(
    r'\d'                                          # cyfry
    r'|[A-Z]{2,}'                                  # akronimy (NATO, NHTSA, GDP)
    r'|\b(?:Mr|Mrs|Ms|Dr|Prof|St|vs|etc'           # skróty z kropką
    r'|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Oct|Nov|Dec'
    r'|approx|govt|dept|no|fig|vol|pg|pp)\.'
    r'|\b\d+(?:\.\d+)?(?:st|nd|rd|th)\b'          # liczby porządkowe: 1st, 2nd
    r'|[$€£¥%]'                                    # waluty/procent
)

_NEEDS_PL = re.compile(
    r'\d'                                          # cyfry
    r'|[A-Z]{2,}'                                  # akronimy
    r'|\b(?:r\.|ul\.|nr\.|str\.|godz\.|min\.|km'  # skróty PL
    r'|kg|mg|ml|tys\.|mln\.|mld\.|zł|PLN|EUR|USD'
    r'|m\.in\.|tj\.|tzn\.|np\.)\b'
    r'|[$€£¥%]'
)


# ── Worker initializer (wywoływany raz na process) ────────────────────────────

_en_norm = None
_pl_pipe = None


def _worker_init():
    global _en_norm, _pl_pipe
    import logging
    for _lg in ("nemo_logger", "nemo_text_processing", "NeMo"):
        logging.getLogger(_lg).setLevel(logging.ERROR)
    # Wycisz też logi z modułów NeMo które logują na poziomie WARNING
    logging.getLogger("nemo_text_processing.text_normalization").setLevel(logging.ERROR)

    from nemo_text_processing.text_normalization.normalize import Normalizer
    _en_norm = Normalizer(input_case="cased", lang="en")

    salmon_root = str(Path(__file__).resolve().parent.parent)
    if salmon_root not in sys.path:
        sys.path.insert(0, salmon_root)
    from wegorz_normalizer.tokenize_and_text_norm import PolishTTSPipeline
    _pl_pipe = PolishTTSPipeline(preserve_case=True)


def _normalize_en(text: str) -> str:
    try:
        return _en_norm.normalize(text, verbose=False, punct_post_process=True)
    except Exception:
        return text


def _normalize_pl(text: str) -> str:
    try:
        return _pl_pipe.process(text)
    except Exception:
        return text


def _cap_encode(text: str) -> str:
    """'Szkoła w Warszawie' → '<CAP>szkoła w <CAP>warszawie'"""
    out = []
    for ch in text:
        if ch.isupper():
            out.append("<CAP>")
            out.append(ch.lower())
        else:
            out.append(ch)
    return "".join(out)


def _process_batch(pairs: list[tuple], cap_encode: bool = True) -> list[dict]:
    """Przetwarza listę (en, pl, extra_fields) → lista {"en": ..., "pl": ..., ...extra}.
    extra_fields to dict z metadanymi (src, doc_id, pos) — przepuszczane bez zmian."""
    results = []
    for item in pairs:
        if len(item) == 3:
            en, pl, extra = item
        else:
            en, pl, extra = item[0], item[1], {}

        was_normalized = bool(extra.get("normalized", False))
        if was_normalized:
            en_out = en
            pl_out = pl
        else:
            en_out = _normalize_en(en) if _NEEDS_EN.search(en) else en
            pl_out = _normalize_pl(pl) if _NEEDS_PL.search(pl) else pl

            if not en_out or not en_out.strip():
                en_out = en
            if not pl_out or not pl_out.strip():
                pl_out = pl

            if cap_encode:
                en_out = _cap_encode(en_out)
                pl_out = _cap_encode(pl_out)

        out_extra = dict(extra)
        if "normalized" in out_extra:
            out_extra["normalized_input"] = was_normalized
            out_extra["normalized"] = True
        results.append({**out_extra, "en": en_out.strip(), "pl": pl_out.strip()})
    return results


# ── Helpers ───────────────────────────────────────────────────────────────────

def _count_lines(path: str) -> int:
    n = 0
    with open(path, "rb") as f:
        for _ in f:
            n += 1
    return n


def _read_jsonl(path: str, skip: int = 0):
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i < skip:
                continue
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                en = obj.get("en", "").strip()
                pl = obj.get("pl", "").strip()
                if en and pl:
                    # Wyciągnij extra pola (src, doc_id, pos) do przepuszczenia
                    extra = {k: v for k, v in obj.items() if k not in ("en", "pl")}
                    yield en, pl, extra
            except json.JSONDecodeError:
                continue


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input",   default="./data/train.jsonl")
    ap.add_argument("--output",  default="./data/train_norm.jsonl")
    ap.add_argument("--workers", type=int, default=4,
                    help="Liczba procesów (default 4, zalecane ≤ n_cpu-1)")
    ap.add_argument("--batch",   type=int, default=500,
                    help="Rozmiar batcha per worker")
    ap.add_argument("--no-resume", action="store_true",
                    help="Zacznij od początku (skasuj istniejący output)")
    ap.add_argument("--no-cap-encode", action="store_true",
                    help="Wyłącz <CAP> encoding (domyślnie włączony)")
    args = ap.parse_args()
    args.cap_encode = not args.no_cap_encode

    in_path  = Path(args.input)
    out_path = Path(args.output)

    if not in_path.exists():
        print(f"Błąd: {in_path} nie istnieje")
        sys.exit(1)

    # Resume: policz ile par już zapisano
    skip_lines = 0
    if out_path.exists() and not args.no_resume:
        skip_lines = _count_lines(str(out_path))
        print(f"Wznawianie: pominięto {skip_lines:,} już zapisanych par")
    elif args.no_resume and out_path.exists():
        out_path.unlink()

    # Policz łącznie (dla paska postępu)
    print(f"Liczę linie w {in_path.name}...")
    total = _count_lines(str(in_path))
    remaining = total - skip_lines
    print(f"Łącznie: {total:,}  Do przetworzenia: {remaining:,}")
    if remaining <= 0:
        print("Nic do zrobienia.")
        return

    # Uruchom pool
    ctx = mp.get_context("spawn")
    pool = ctx.Pool(processes=args.workers, initializer=_worker_init)

    t0 = time.perf_counter()
    written = 0
    norm_en = 0
    norm_pl = 0
    batch: list[tuple[str, str]] = []
    futures = []

    def flush_batch():
        nonlocal batch
        if batch:
            futures.append((pool.apply_async(_process_batch, (batch, args.cap_encode)), len(batch)))
            batch = []

    def collect_futures(fout, max_pending: int = args.workers * 3):
        nonlocal written, norm_en, norm_pl
        while len(futures) > max_pending:
            fut, n = futures.pop(0)
            results = fut.get()
            for r in results:
                fout.write(json.dumps(r, ensure_ascii=False) + "\n")
                written += 1
            # Policz znormalizowane (sprawdź czy en/pl się zmieniły — nie mamy oryginału,
            # więc liczymy te które przeszły przez normalizator tj. miały digits/abbrev)
            norm_en += sum(1 for r in results if _NEEDS_EN.search(r.get("en", "")))
            norm_pl += sum(1 for r in results if _NEEDS_PL.search(r.get("pl", "")))

    log_every = 10_000
    next_log  = log_every

    with open(str(out_path), "a", encoding="utf-8") as fout:
        for en, pl, extra in _read_jsonl(str(in_path), skip=skip_lines):
            batch.append((en, pl, extra))
            if len(batch) >= args.batch:
                flush_batch()
                collect_futures(fout)

            if written + skip_lines >= next_log:
                elapsed = time.perf_counter() - t0
                done    = written + skip_lines
                speed   = written / elapsed if elapsed > 0 else 0
                eta     = (total - done) / speed if speed > 0 else 0
                pct     = done / total * 100
                print(
                    f"  {done:>9,}/{total:,} ({pct:.1f}%)  "
                    f"{speed:>7.0f} par/s  ETA {eta/60:.0f}min",
                    flush=True,
                )
                next_log += log_every

        # Flush reszty
        flush_batch()
        collect_futures(fout, max_pending=0)

    pool.close()
    pool.join()

    elapsed = time.perf_counter() - t0
    print(f"\nGotowe: {written:,} par zapisano → {out_path}")
    print(f"Czas: {elapsed/60:.1f} min  ({written/elapsed:.0f} par/s)")
    print(f"(skip_lines={skip_lines:,}  łącznie w pliku: {written + skip_lines:,})")


if __name__ == "__main__":
    main()
