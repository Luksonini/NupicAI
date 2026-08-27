# WęgorzAI Polish TTS Text Normalizer

Standalone Polish text normalization pipeline for TTS synthesis. Converts raw Polish text into speech-ready form: numbers to words, abbreviations to expansions, foreign terms to Polish phonetics.

## Features

- **Numbers** → Polish words with correct grammatical case (nominative, genitive, locative, etc.)
- **Dates** → "15.03.2024 r." → "piętnastego marca dwa tysiące dwudziestego czwartego roku"
- **Times** → "o 14:30" → "o czternastej trzydzieści"
- **Currencies** → "29,99 zł" → "dwadzieścia dziewięć złotych dziewięćdziesiąt dziewięć groszy"
- **Units** → "120 km/h" → "sto dwadzieścia kilometrów na godzinę"
- **Abbreviations** → "dr", "ul.", "art." expanded with correct case
- **Foreign words** → Wi-Fi, DVD, CEO with Polish phonetic rendering
- **Fractions & percentages** → "3/4" → "trzy czwarte", "25%" → "dwadzieścia pięć procent"

## Usage

### Multi-file package

```python
from wegorz_normalizer import PolishTTSPipeline

pipe = PolishTTSPipeline()
result = pipe.process("Spotkanie o godz. 14:30 kosztuje 29,99 zł.")
# → "spotkanie o godzinie czternastej trzydzieści kosztuje dwadzieścia dziewięć złotych dziewięćdziesiąt dziewięć groszy."
```

### Single-file version

Drop `wegorz_normalizer_single.py` + `slownik_wymowy.json` into any project:

```python
from wegorz_normalizer_single import PolishTTSPipeline

pipe = PolishTTSPipeline()
print(pipe.process("Silnik V8 ma moc 500 KM."))
```

## Installation

```bash
pip install num2words
```

## Testing

```bash
pip install pytest
pytest tests/ -v
```

1553 tests covering dates, times, numbers, currencies, units, fractions, abbreviations, foreign words, edge cases, and full-text normalization.

## Architecture

```
wegorz_normalizer/
├── __init__.py                    # Package entry point
├── tokenize_and_text_norm.py      # Main pipeline (5-stage normalization)
├── inflect_pl.py                  # Polish number inflection (all 6 cases)
├── unit_registry.py               # SI/metric units with inflection
└── slownik_wymowy.json            # Pronunciation dictionary (5600+ entries)

wegorz_normalizer_single.py        # All-in-one single file version
```

### Pipeline stages

1. `raw_clean()` — Unicode normalization, dash variants, foreign scripts, URLs, emails
2. `abbreviation_expand()` — Polish abbreviations, Roman numerals
3. `foreign_expand()` — Acronyms (TTS→"te te es"), loanwords, mixed tokens
4. `num_normalize()` — Dates, times, currencies, units, percentages, fractions
5. `final_filter()` — Character whitelisting for TTS vocabulary

## License

MIT
