"""
Polish number inflection module for TTS normalization.

Provides case-inflected cardinal and ordinal number forms in Polish.
Uses morfeusz2 morphological generator when available (Python 3.11/3.12 in Docker),
falls back to comprehensive static tables on unsupported Python versions.

Public API:
    cardinal_inflect(n, case, gender="m") -> str
    ordinal_inflect(n, case, gender="m") -> str
    noun_gender(word) -> str   ("m", "f", "n")

Cases: "nom", "gen", "dat", "acc", "inst", "loc"
Genders: "m" (masculine), "f" (feminine), "n" (neuter)
"""

import logging
import re
from functools import lru_cache

try:
    from num2words import num2words
except ImportError:
    raise ImportError("num2words is required for inflect_pl")

logger = logging.getLogger(__name__)

# ── Morfeusz2 runtime (Python 3.11/3.12 only) ────────────────────────────────

_HAS_MORFEUSZ = False
_morf = None

# morfeusz2 segfaults on Python >= 3.13 (abi3 wheel incompatible).
# Only attempt import on Python 3.11/3.12 (Docker production environment).
import sys as _sys
if _sys.version_info < (3, 13):
    try:
        import morfeusz2
        _morf = morfeusz2.Morfeusz(generate=True)
        _HAS_MORFEUSZ = True
        logger.info("morfeusz2 loaded — using runtime morphological generation")
    except Exception:
        logger.info("morfeusz2 not available — using static inflection tables")
else:
    logger.info("Python %s — skipping morfeusz2 (requires ≤3.12), using static tables",
                _sys.version_info[:2])


# ── NKJP case abbreviations ──────────────────────────────────────────────────

CASES = ("nom", "gen", "dat", "acc", "inst", "loc")

# ── Known noun genders (for the unit nouns used in TTS normalization) ─────────

_NOUN_GENDER = {
    # Feminine (f)
    "godzina": "f", "minuta": "f", "sekunda": "f", "atmosfera": "f",
    "tona": "f", "ciężarówka": "f", "osoba": "f",
    "korona": "f", "watosekunda": "f", "kilowatogodzina": "f",
    "klatka": "f",  # klatka na sekundę (fps)
    # Masculine (m)
    "metr": "m", "kilometr": "m", "centymetr": "m", "milimetr": "m",
    "litr": "m", "kilogram": "m", "gram": "m", "miligram": "m",
    "bajt": "m", "kilobajt": "m", "megabajt": "m", "gigabajt": "m",
    "terabajt": "m", "petabajt": "m",
    "wat": "m", "kilowat": "m", "megawat": "m",
    "herc": "m", "megaherc": "m", "gigaherc": "m",
    "wolt": "m", "amper": "m", "niuton": "m", "paskal": "m",
    "bar": "m", "decybel": "m", "piksel": "m", "megapiksel": "m",
    "kelwin": "m", "stopień": "m", "hektar": "m",
    "dolar": "m", "funt": "m", "frank": "m", "jen": "m",
    "złoty": "m", "grosz": "m", "cent": "m",
    "procent": "m", "rok": "m", "dzień": "m",
    # Neuter (n)
    "euro": "n",
}


def noun_gender(word: str) -> str:
    """Return gender of a known noun, or 'm' as default."""
    return _NOUN_GENDER.get(word.lower(), "m")


# ══════════════════════════════════════════════════════════════════════════════
# STATIC FALLBACK TABLES — Polish cardinal numbers in all 6 cases
# ══════════════════════════════════════════════════════════════════════════════
#
# Format: _CARD_CASE[case][(tens_digit, ones_digit)] = word
# For compound numbers, components are looked up separately and joined.
#
# Note: For cardinals >= 5 in Polish, gen == dat == loc in most forms.
# Instrumental is unique. Accusative == nominative for inanimates.

# ── Ones: 0-9 ────────────────────────────────────────────────────────────────

_CARD_ONES = {
    "nom": {0: "zero", 1: "jeden", 2: "dwa", 3: "trzy", 4: "cztery",
            5: "pięć", 6: "sześć", 7: "siedem", 8: "osiem", 9: "dziewięć"},
    "gen": {0: "zera", 1: "jednego", 2: "dwóch", 3: "trzech", 4: "czterech",
            5: "pięciu", 6: "sześciu", 7: "siedmiu", 8: "ośmiu", 9: "dziewięciu"},
    "dat": {0: "zeru", 1: "jednemu", 2: "dwóm", 3: "trzem", 4: "czterem",
            5: "pięciu", 6: "sześciu", 7: "siedmiu", 8: "ośmiu", 9: "dziewięciu"},
    "acc": {0: "zero", 1: "jeden", 2: "dwa", 3: "trzy", 4: "cztery",
            5: "pięć", 6: "sześć", 7: "siedem", 8: "osiem", 9: "dziewięć"},
    "inst": {0: "zerem", 1: "jednym", 2: "dwoma", 3: "trzema", 4: "czterema",
             5: "pięcioma", 6: "sześcioma", 7: "siedmioma", 8: "ośmioma", 9: "dziewięcioma"},
    "loc": {0: "zerze", 1: "jednym", 2: "dwóch", 3: "trzech", 4: "czterech",
            5: "pięciu", 6: "sześciu", 7: "siedmiu", 8: "ośmiu", 9: "dziewięciu"},
}

# Feminine ones (only 1 and 2 differ)
_CARD_ONES_F = {
    "nom": {1: "jedna", 2: "dwie"},
    "gen": {1: "jednej", 2: "dwóch"},
    "dat": {1: "jednej", 2: "dwóm"},
    "acc": {1: "jedną", 2: "dwie"},
    "inst": {1: "jedną", 2: "dwiema"},
    "loc": {1: "jednej", 2: "dwóch"},
}

# ── Teens: 10-19 ─────────────────────────────────────────────────────────────

_CARD_TEENS = {
    "nom": {10: "dziesięć", 11: "jedenaście", 12: "dwanaście", 13: "trzynaście",
            14: "czternaście", 15: "piętnaście", 16: "szesnaście", 17: "siedemnaście",
            18: "osiemnaście", 19: "dziewiętnaście"},
    "gen": {10: "dziesięciu", 11: "jedenastu", 12: "dwunastu", 13: "trzynastu",
            14: "czternastu", 15: "piętnastu", 16: "szesnastu", 17: "siedemnastu",
            18: "osiemnastu", 19: "dziewiętnastu"},
    "dat": {10: "dziesięciu", 11: "jedenastu", 12: "dwunastu", 13: "trzynastu",
            14: "czternastu", 15: "piętnastu", 16: "szesnastu", 17: "siedemnastu",
            18: "osiemnastu", 19: "dziewiętnastu"},
    "acc": {10: "dziesięć", 11: "jedenaście", 12: "dwanaście", 13: "trzynaście",
            14: "czternaście", 15: "piętnaście", 16: "szesnaście", 17: "siedemnaście",
            18: "osiemnaście", 19: "dziewiętnaście"},
    "inst": {10: "dziesięcioma", 11: "jedenastoma", 12: "dwunastoma", 13: "trzynastoma",
             14: "czternastoma", 15: "piętnastoma", 16: "szesnastoma", 17: "siedemnastoma",
             18: "osiemnastoma", 19: "dziewiętnastoma"},
    "loc": {10: "dziesięciu", 11: "jedenastu", 12: "dwunastu", 13: "trzynastu",
            14: "czternastu", 15: "piętnastu", 16: "szesnastu", 17: "siedemnastu",
            18: "osiemnastu", 19: "dziewiętnastu"},
}

# ── Tens: 20-90 ──────────────────────────────────────────────────────────────

_CARD_TENS = {
    "nom": {20: "dwadzieścia", 30: "trzydzieści", 40: "czterdzieści",
            50: "pięćdziesiąt", 60: "sześćdziesiąt", 70: "siedemdziesiąt",
            80: "osiemdziesiąt", 90: "dziewięćdziesiąt"},
    "gen": {20: "dwudziestu", 30: "trzydziestu", 40: "czterdziestu",
            50: "pięćdziesięciu", 60: "sześćdziesięciu", 70: "siedemdziesięciu",
            80: "osiemdziesięciu", 90: "dziewięćdziesięciu"},
    "dat": {20: "dwudziestu", 30: "trzydziestu", 40: "czterdziestu",
            50: "pięćdziesięciu", 60: "sześćdziesięciu", 70: "siedemdziesięciu",
            80: "osiemdziesięciu", 90: "dziewięćdziesięciu"},
    "acc": {20: "dwadzieścia", 30: "trzydzieści", 40: "czterdzieści",
            50: "pięćdziesiąt", 60: "sześćdziesiąt", 70: "siedemdziesiąt",
            80: "osiemdziesiąt", 90: "dziewięćdziesiąt"},
    "inst": {20: "dwudziestoma", 30: "trzydziestoma", 40: "czterdziestoma",
             50: "pięćdziesięcioma", 60: "sześćdziesięcioma", 70: "siedemdziesięcioma",
             80: "osiemdziesięcioma", 90: "dziewięćdziesięcioma"},
    "loc": {20: "dwudziestu", 30: "trzydziestu", 40: "czterdziestu",
            50: "pięćdziesięciu", 60: "sześćdziesięciu", 70: "siedemdziesięciu",
            80: "osiemdziesięciu", 90: "dziewięćdziesięciu"},
}

# ── Hundreds: 100-900 ────────────────────────────────────────────────────────

_CARD_HUNDREDS = {
    "nom": {100: "sto", 200: "dwieście", 300: "trzysta", 400: "czterysta",
            500: "pięćset", 600: "sześćset", 700: "siedemset",
            800: "osiemset", 900: "dziewięćset"},
    "gen": {100: "stu", 200: "dwustu", 300: "trzystu", 400: "czterystu",
            500: "pięciuset", 600: "sześciuset", 700: "siedmiuset",
            800: "ośmiuset", 900: "dziewięciuset"},
    "dat": {100: "stu", 200: "dwustu", 300: "trzystu", 400: "czterystu",
            500: "pięciuset", 600: "sześciuset", 700: "siedmiuset",
            800: "ośmiuset", 900: "dziewięciuset"},
    "acc": {100: "sto", 200: "dwieście", 300: "trzysta", 400: "czterysta",
            500: "pięćset", 600: "sześćset", 700: "siedemset",
            800: "osiemset", 900: "dziewięćset"},
    "inst": {100: "stoma", 200: "dwustoma", 300: "trzystoma", 400: "czterystoma",
             500: "pięciusetoma", 600: "sześciusetoma", 700: "siedmiusetoma",
             800: "ośmiusetoma", 900: "dziewięciusetoma"},
    "loc": {100: "stu", 200: "dwustu", 300: "trzystu", 400: "czterystu",
            500: "pięciuset", 600: "sześciuset", 700: "siedmiuset",
            800: "ośmiuset", 900: "dziewięciuset"},
}

# ── Thousands multiplier ─────────────────────────────────────────────────────

_THOUSAND_FORMS = {
    # (singular, plural 2-4, plural 5+, genitive singular for fractions)
    "nom": ("tysiąc", "tysiące", "tysięcy"),
    "gen": ("tysiąca", "tysięcy", "tysięcy"),
    "dat": ("tysiącowi", "tysiącom", "tysiącom"),
    "acc": ("tysiąc", "tysiące", "tysięcy"),
    "inst": ("tysiącem", "tysiącami", "tysiącami"),
    "loc": ("tysiącu", "tysiącach", "tysiącach"),
}

_MILLION_FORMS = {
    "nom": ("milion", "miliony", "milionów"),
    "gen": ("miliona", "milionów", "milionów"),
    "dat": ("milionowi", "milionom", "milionom"),
    "acc": ("milion", "miliony", "milionów"),
    "inst": ("milionem", "milionami", "milionami"),
    "loc": ("milionie", "milionach", "milionach"),
}

_MILLIARD_FORMS = {
    "nom": ("miliard", "miliardy", "miliardów"),
    "gen": ("miliarda", "miliardów", "miliardów"),
    "dat": ("miliardowi", "miliardom", "miliardom"),
    "acc": ("miliard", "miliardy", "miliardów"),
    "inst": ("miliardem", "miliardami", "miliardami"),
    "loc": ("miliardzie", "miliardach", "miliardach"),
}


# ══════════════════════════════════════════════════════════════════════════════
# MORFEUSZ2 RUNTIME INFLECTION
# ══════════════════════════════════════════════════════════════════════════════

@lru_cache(maxsize=2048)
def _morfeusz_inflect_word(word: str, target_case: str, gender: str = "m") -> str | None:
    """Use morfeusz2 to inflect a single word to the target case.
    Returns None if morfeusz2 is unavailable or fails."""
    if not _HAS_MORFEUSZ or _morf is None:
        return None
    try:
        results = _morf.generate(word)
        # results: list of (lemma, tag, form)
        # NKJP tags for cardinals: num:pl:gen:m2.m3.f.n (example)
        # NKJP tags for ordinals: adj:sg:gen:m1.m2.m3:pos
        candidates = []
        for lemma, tag, form in results:
            if f":{target_case}:" in tag or tag.endswith(f":{target_case}"):
                # Check gender match if specified
                if gender == "f" and ":f" in tag:
                    candidates.insert(0, form)  # prioritize feminine
                elif gender == "m" and (":m" in tag or ":m1" in tag or ":m2" in tag or ":m3" in tag):
                    candidates.insert(0, form)
                elif gender == "n" and ":n" in tag:
                    candidates.insert(0, form)
                else:
                    candidates.append(form)
        if candidates:
            return candidates[0]
    except Exception as e:
        logger.debug("morfeusz2 generate failed for %r: %s", word, e)
    return None


def _morfeusz_cardinal(n: int, case: str, gender: str = "m") -> str | None:
    """Try to inflect a cardinal number using morfeusz2 runtime.
    Returns the inflected string or None if unavailable."""
    if not _HAS_MORFEUSZ:
        return None
    # Get nominative from num2words, then inflect each word
    nom = num2words(n, lang="pl")
    words = nom.split()
    inflected = []
    for w in words:
        result = _morfeusz_inflect_word(w, target_case=case, gender=gender)
        if result is None:
            return None  # can't inflect one word → bail to static tables
        inflected.append(result)
    return " ".join(inflected)


# ══════════════════════════════════════════════════════════════════════════════
# STATIC TABLE INFLECTION
# ══════════════════════════════════════════════════════════════════════════════

def _pick_plural(n: int, sg: str, pl234: str, pl5plus: str) -> str:
    """Pick correct plural form for Polish number n."""
    n = abs(n)
    if n % 100 in range(11, 20):
        return pl5plus
    d = n % 10
    if d == 1 and n == 1:
        return sg
    if d in (2, 3, 4):
        return pl234
    return pl5plus


def _static_cardinal_below_1000(n: int, case: str, gender: str = "m") -> str:
    """Inflect cardinal 0-999 using static tables."""
    if n < 0:
        return "minus " + _static_cardinal_below_1000(-n, case, gender)

    # Use feminine forms for 1, 2 when gender is feminine
    if n <= 2 and gender == "f" and case in _CARD_ONES_F and n in _CARD_ONES_F[case]:
        return _CARD_ONES_F[case][n]

    if n <= 9:
        return _CARD_ONES[case].get(n, _CARD_ONES["nom"][n])

    if 10 <= n <= 19:
        return _CARD_TEENS[case].get(n, _CARD_TEENS["nom"][n])

    if n < 100:
        t, o = (n // 10) * 10, n % 10
        tens_word = _CARD_TENS[case].get(t, _CARD_TENS["nom"][t])
        if o == 0:
            return tens_word
        # For compound tens+ones, ones also inflect
        if o <= 2 and gender == "f" and case in _CARD_ONES_F and o in _CARD_ONES_F[case]:
            ones_word = _CARD_ONES_F[case][o]
        else:
            ones_word = _CARD_ONES[case].get(o, _CARD_ONES["nom"][o])
        return f"{tens_word} {ones_word}"

    if n < 1000:
        h, rest = (n // 100) * 100, n % 100
        h_word = _CARD_HUNDREDS[case].get(h, _CARD_HUNDREDS["nom"][h])
        if rest == 0:
            return h_word
        return f"{h_word} {_static_cardinal_below_1000(rest, case, gender)}"

    return num2words(n, lang="pl")  # should not reach here


def _static_cardinal(n: int, case: str, gender: str = "m") -> str:
    """Inflect any cardinal number using static tables + decomposition."""
    if n < 0:
        return "minus " + _static_cardinal(-n, case, gender)

    if n < 1000:
        return _static_cardinal_below_1000(n, case, gender)

    # Thousands
    if n < 1_000_000:
        k, rest = n // 1000, n % 1000
        if k == 1:
            k_word = "tysiąc" if case == "nom" or case == "acc" else \
                     _THOUSAND_FORMS[case][0]
        else:
            # The multiplier (k) is always in genitive when k >= 5
            # For 2-4: "dwa/trzy/cztery tysiące" (nom) or inflected forms
            k_inflected = _static_cardinal_below_1000(k, case if k <= 4 else "gen", gender="m")
            thousand_form = _pick_plural(k, *_THOUSAND_FORMS[case])
            k_word = f"{k_inflected} {thousand_form}"
        if rest == 0:
            return k_word
        return f"{k_word} {_static_cardinal_below_1000(rest, case, gender)}"

    # Millions
    if n < 1_000_000_000:
        m, rest = n // 1_000_000, n % 1_000_000
        if m == 1:
            m_word = _MILLION_FORMS[case][0]
        else:
            m_inflected = _static_cardinal_below_1000(m, "gen" if m >= 5 else case, gender="m")
            million_form = _pick_plural(m, *_MILLION_FORMS[case])
            m_word = f"{m_inflected} {million_form}"
        if rest == 0:
            return m_word
        # Rest of millions is in genitive
        return f"{m_word} {_static_cardinal(rest, case, gender)}"

    # Milliards
    if n < 1_000_000_000_000:
        md, rest = n // 1_000_000_000, n % 1_000_000_000
        if md == 1:
            md_word = _MILLIARD_FORMS[case][0]
        else:
            md_inflected = _static_cardinal_below_1000(md, "gen" if md >= 5 else case, gender="m")
            milliard_form = _pick_plural(md, *_MILLIARD_FORMS[case])
            md_word = f"{md_inflected} {milliard_form}"
        if rest == 0:
            return md_word
        return f"{md_word} {_static_cardinal(rest, case, gender)}"

    # Fallback for very large numbers: nominative from num2words
    return num2words(n, lang="pl")


# ══════════════════════════════════════════════════════════════════════════════
# PUBLIC API
# ══════════════════════════════════════════════════════════════════════════════

def cardinal_inflect(n: int, case: str = "nom", gender: str = "m") -> str:
    """
    Return cardinal number `n` inflected to the given Polish grammatical case.

    Args:
        n: Integer to inflect
        case: One of "nom", "gen", "dat", "acc", "inst", "loc"
        gender: "m" (masculine), "f" (feminine), "n" (neuter)
                Only affects 1 and 2 (jedna/dwie vs jeden/dwa)

    Returns:
        Inflected Polish cardinal number as string.

    Examples:
        cardinal_inflect(5, "gen")   → "pięciu"
        cardinal_inflect(5, "inst")  → "pięcioma"
        cardinal_inflect(2, "nom", "f") → "dwie"
        cardinal_inflect(22, "inst") → "dwudziestoma dwoma"
    """
    if case not in CASES:
        case = "nom"

    # Nominative: use num2words directly (fastest path, always correct)
    if case == "nom":
        w = num2words(n, lang="pl")
        if gender == "f":
            w = re.sub(r"\bdwa\b", "dwie", w)
            w = re.sub(r"\bjeden\b", "jedna", w)
        return w

    # Try morfeusz2 first (handles ALL numbers, including edge cases)
    if _HAS_MORFEUSZ:
        result = _morfeusz_cardinal(n, case, gender)
        if result is not None:
            return result

    # Fallback to static tables
    return _static_cardinal(n, case, gender)


def cardinal_gen(n: int) -> str:
    """Backward-compatible alias: genitive cardinal."""
    return cardinal_inflect(n, "gen")


# ── Ordinal inflection ────────────────────────────────────────────────────────

def _ordinal_suffix_transform(ordinal_nom: str, case: str, gender: str = "m") -> str:
    """Transform ordinal nominative to target case using suffix rules.

    Polish ordinals are adjectives with regular declension patterns:
    - Masculine: -y/-i → -ego (gen), -emu (dat), -ym/-im (inst/loc)
    - Feminine:  -y/-i → -a → -ej (gen/dat/loc), -ą (acc/inst)
    - Neuter:    -y/-i → -e → -ego (gen), -emu (dat), -ym/-im (inst/loc)
    """
    words = ordinal_nom.split()
    result = []
    for w in words:
        if gender == "f":
            if case == "nom":
                # Feminine nominative: pierwszy → pierwsza, czternasty → czternasta
                if w.endswith("y"):
                    w = w[:-1] + "a"
                elif w.endswith("i"):
                    w = w[:-1] + "ia"
            elif case in ("gen", "dat", "loc"):
                if w.endswith("y"):
                    w = w[:-1] + "ej"
                elif w.endswith("i"):
                    w = w[:-1] + "iej"
            elif case == "acc":
                if w.endswith("y"):
                    w = w[:-1] + "ą"
                elif w.endswith("i"):
                    w = w[:-1] + "ią"
            elif case == "inst":
                if w.endswith("y"):
                    w = w[:-1] + "ą"
                elif w.endswith("i"):
                    w = w[:-1] + "ią"
        elif gender == "n":
            if case == "gen":
                if w.endswith("y"):
                    w = w[:-1] + "ego"
                elif w.endswith("i"):
                    w = w[:-1] + "iego"
            elif case == "dat":
                if w.endswith("y"):
                    w = w[:-1] + "emu"
                elif w.endswith("i"):
                    w = w[:-1] + "iemu"
            elif case in ("inst", "loc"):
                if w.endswith("y"):
                    w = w[:-1] + "ym"
                elif w.endswith("i"):
                    w = w[:-1] + "im"
            elif case == "acc":
                if w.endswith("y"):
                    w = w[:-1] + "e"
                elif w.endswith("i"):
                    w = w[:-1] + "ie"
        else:  # masculine
            if case == "gen":
                if w == "sto":
                    w = "stu"
                elif w.endswith("y"):
                    w = w[:-1] + "ego"
                elif w.endswith("i"):
                    w = w[:-1] + "iego"
            elif case == "dat":
                if w == "sto":
                    w = "stu"
                elif w.endswith("y"):
                    w = w[:-1] + "emu"
                elif w.endswith("i"):
                    w = w[:-1] + "iemu"
            elif case in ("inst", "loc"):
                if w == "sto":
                    w = "stu"
                elif w.endswith("y"):
                    w = w[:-1] + "ym"
                elif w.endswith("i"):
                    w = w[:-1] + "im"
            elif case == "acc":
                # Masculine animate: same as genitive
                # Masculine inanimate: same as nominative
                # Default to genitive (safer for TTS — people, animals)
                if w.endswith("y"):
                    w = w[:-1] + "ego"
                elif w.endswith("i"):
                    w = w[:-1] + "iego"
        result.append(w)
    return " ".join(result)


def ordinal_inflect(n: int, case: str = "nom", gender: str = "m") -> str:
    """
    Return ordinal number `n` inflected to the given Polish grammatical case.

    Args:
        n: Integer (the ordinal position)
        case: One of "nom", "gen", "dat", "acc", "inst", "loc"
        gender: "m", "f", "n"

    Examples:
        ordinal_inflect(1, "gen", "m")  → "pierwszego"
        ordinal_inflect(14, "loc", "f") → "czternastej"
        ordinal_inflect(21, "gen", "m") → "dwudziestego pierwszego"
    """
    if case not in CASES:
        case = "nom"

    # Get nominative ordinal from num2words (always masculine)
    nom = num2words(n, lang="pl", to="ordinal")

    if case == "nom" and gender == "m":
        return nom

    # Try morfeusz2 for ordinals too
    if _HAS_MORFEUSZ:
        words = nom.split()
        inflected = []
        for w in words:
            result = _morfeusz_inflect_word(w, target_case=case, gender=gender)
            if result is None:
                # Fall through to suffix rules
                inflected = None
                break
            inflected.append(result)
        if inflected:
            return " ".join(inflected)

    # Suffix transformation (works for all regular ordinals)
    return _ordinal_suffix_transform(nom, case, gender)


# ── Hour ordinals (feminine, used for time) ───────────────────────────────────

def hour_ordinal(hour: int, case: str = "nom") -> str:
    """Return hour as feminine ordinal in the given case.

    'godzina' is feminine, so hours use feminine ordinal forms:
    - nom: "pierwsza", "druga", ..., "dwudziesta trzecia"
    - gen: "pierwszej", "drugiej", ...
    - loc: "pierwszej", "drugiej", ... (same as gen for feminine)
    """
    # Hour 0 (midnight) → use 24 for ordinal: "dwudziesta czwarta" / "zero zero"
    if hour == 0:
        _ZERO_HOUR = {"nom": "zero", "gen": "zero", "loc": "zero",
                      "dat": "zero", "acc": "zero", "inst": "zero"}
        return _ZERO_HOUR.get(case, "zero")
    return ordinal_inflect(hour, case=case, gender="f")


# ── Noun inflection (for unit nouns) ──────────────────────────────────────────

@lru_cache(maxsize=512)
def inflect_noun(nom_sg: str, case: str = "nom", number: str = "sg",
                 gender: str = "auto") -> str:
    """Inflect a Polish noun by case and grammatical number.

    Intended for unit nouns (metr, gram, godzina, etc.) but works for any noun
    that morfeusz2 can handle. Falls back to basic suffix rules when morfeusz2
    is unavailable.

    Args:
        nom_sg: Nominative singular form (e.g., "metr", "godzina")
        case: Target case — "nom", "gen", "dat", "acc", "inst", "loc"
        number: "sg" (singular) or "pl" (plural)
        gender: "m", "f", "n", or "auto" (guess from ending / known dict)

    Returns:
        Inflected form, or nom_sg unchanged if inflection fails.
    """
    if case == "nom" and number == "sg":
        return nom_sg

    if gender == "auto":
        gender = noun_gender(nom_sg)

    # For compound nouns ("metr kwadratowy", "stopień Celsjusza"),
    # inflect the first word and handle the rest separately.
    parts = nom_sg.split(" ", 1)
    head = parts[0]
    tail = parts[1] if len(parts) > 1 else None

    # Try morfeusz2 for the head word
    if _HAS_MORFEUSZ and _morf is not None:
        num_tag = "sg" if number == "sg" else "pl"
        result = _morfeusz_inflect_head(head, case, num_tag, gender)
        if result is not None:
            if tail is not None:
                tail_inflected = _inflect_modifier(tail, case, number, gender)
                return f"{result} {tail_inflected}"
            return result

    # Static fallback — gen sg is the most commonly needed case
    inflected_head = _static_noun_inflect(head, case, number, gender)
    if tail is not None:
        tail_inflected = _inflect_modifier(tail, case, number, gender)
        return f"{inflected_head} {tail_inflected}"
    return inflected_head


@lru_cache(maxsize=512)
def _morfeusz_inflect_head(word: str, case: str, num_tag: str,
                           gender: str) -> str | None:
    """Try morfeusz2 to inflect a single noun to target case/number."""
    if not _HAS_MORFEUSZ or _morf is None:
        return None
    try:
        results = _morf.generate(word)
        for _lemma, tag, form in results:
            if f":{case}:" in tag and f":{num_tag}" in tag:
                if gender == "f" and ":f" in tag:
                    return form
                if gender == "m" and (":m" in tag):
                    return form
                if gender == "n" and ":n" in tag:
                    return form
        # Second pass: accept any gender match
        for _lemma, tag, form in results:
            if f":{case}:" in tag and f":{num_tag}" in tag:
                return form
    except Exception:
        pass
    return None


def _static_noun_inflect(word: str, case: str, number: str,
                         gender: str) -> str:
    """Basic static inflection for Polish nouns (fallback)."""
    # Genitive singular — the most commonly needed case for units
    if case == "gen" and number == "sg":
        if gender == "f" and word.endswith("a"):
            stem = word[:-1]
            return stem + ("i" if stem.endswith(("k", "g")) else "y")
        # Masculine irregular patterns
        if word.endswith("ień"):
            return word[:-3] + "nia"
        if word.endswith("ót"):
            return word[:-2] + "otu"
        if word.endswith("iec"):
            return word[:-3] + "ca"
        return word + "a"

    # For other cases, return nominative (safe fallback — registry provides real forms)
    return word


def _inflect_modifier(modifier: str, case: str, number: str,
                      gender: str) -> str:
    """Inflect an adjective or fixed qualifier that follows the head noun.

    Fixed qualifiers (proper nouns like "Celsjusza", genitives like "słupa rtęci")
    are returned unchanged. Adjectives are inflected by case.
    """
    # Already a fixed genitive qualifier — don't change
    if (modifier[0].isupper() or
            modifier.endswith("a") and not modifier.endswith(("owa", "nna", "na")) or
            " " in modifier):  # multi-word qualifiers like "słupa rtęci"
        return modifier

    # Try morfeusz2 first
    if _HAS_MORFEUSZ and _morf is not None:
        result = _morfeusz_inflect_word(modifier, target_case=case, gender=gender)
        if result is not None:
            return result

    # Static adjective inflection
    if case == "gen":
        if gender in ("m", "n"):
            if modifier.endswith("owy"):
                return modifier[:-1] + "ego"
            if modifier.endswith("nny"):
                return modifier[:-1] + "ego"
            if modifier.endswith("ny"):
                return modifier[:-1] + "ego"
            if modifier.endswith("y"):
                return modifier[:-1] + "ego"
            if modifier.endswith("i"):
                return modifier[:-1] + "iego"
    elif case in ("loc", "inst"):
        if gender in ("m", "n"):
            if modifier.endswith("y"):
                return modifier[:-1] + "ym"
            if modifier.endswith("i"):
                return modifier[:-1] + "im"

    return modifier
