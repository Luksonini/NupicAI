"""
Compositional Polish unit registry for TTS normalization.

Generates Polish unit noun forms (nominative, genitive, locative, instrumental)
from base units and SI prefixes. Drop-in replacement for the manual _UNITS and
_UNIT_CASE_FORMS dictionaries in tokenize_and_text_norm.py.

Usage:
    from model.unit_registry import REGISTRY
    units_dict = REGISTRY.as_units_dict()      # same format as _UNITS
    case_dict  = REGISTRY.as_case_dict()       # same format as _UNIT_CASE_FORMS
"""

from __future__ import annotations

from dataclasses import dataclass


# ═══════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class UnitForms:
    """Complete inflected forms for a unit noun in Polish."""
    nom_sg: str       # nominative singular:   "metr"
    nom_pl234: str    # nominative plural 2-4: "metry"
    nom_pl5: str      # genitive plural 5+:    "metrów"
    gender: str       # "m" (masculine) or "f" (feminine)
    gen_sg: str       # genitive singular:     "metra"
    loc_sg: str       # locative singular:     "metrze"
    inst_sg: str      # instrumental singular: "metrem"
    loc_pl: str       # locative plural:       "metrach"
    inst_pl: str      # instrumental plural:   "metrami"

    def as_units_tuple(self) -> tuple[str, str, str, str]:
        """Compatible with _UNITS dict values: (nom_sg, nom_pl234, nom_pl5, gender)."""
        return (self.nom_sg, self.nom_pl234, self.nom_pl5, self.gender)

    def as_case_entries(self) -> dict[str, tuple[str, str, str]]:
        """Compatible with _UNIT_CASE_FORMS values: {case: (sg, pl234, pl5+)}.

        For oblique cases, pl234 == pl5+ (standard in Polish).
        """
        return {
            "gen":  (self.gen_sg,  self.nom_pl5, self.nom_pl5),
            "loc":  (self.loc_sg,  self.loc_pl,  self.loc_pl),
            "inst": (self.inst_sg, self.inst_pl, self.inst_pl),
        }


@dataclass(frozen=True)
class BaseUnit:
    """Fundamental measurement unit with all inflected forms."""
    nom_sg: str
    nom_pl234: str
    nom_pl5: str
    gender: str       # "m" | "f"
    gen_sg: str
    loc_sg: str
    inst_sg: str
    loc_pl: str
    inst_pl: str
    connect: str      # connecting form for compound-time units (e.g., "wato" → watogodzina)


# ═══════════════════════════════════════════════════════════════════════════════
# BASE UNITS — Polish noun forms for ~25 fundamental measurement units
# ═══════════════════════════════════════════════════════════════════════════════
#
# Each field is explicitly listed (no derivation) to guarantee correctness
# of Polish morphology including ó→o alternation, consonant mutations, etc.

_BASE_UNITS: dict[str, BaseUnit] = {
    # ── Length ────────────────────────────────────────────────────────────────
    "metr": BaseUnit(
        "metr", "metry", "metrów", "m",
        "metra", "metrze", "metrem", "metrach", "metrami", "metro"),
    "cal": BaseUnit(
        "cal", "cale", "cali", "m",
        "cala", "calu", "calem", "calach", "calami", ""),
    # ── Mass ─────────────────────────────────────────────────────────────────
    "gram": BaseUnit(
        "gram", "gramy", "gramów", "m",
        "grama", "gramie", "gramem", "gramach", "gramami", "gramo"),
    "tona": BaseUnit(
        "tona", "tony", "ton", "f",
        "tony", "tonie", "toną", "tonach", "tonami", ""),
    # ── Volume ───────────────────────────────────────────────────────────────
    "litr": BaseUnit(
        "litr", "litry", "litrów", "m",
        "litra", "litrze", "litrem", "litrach", "litrami", "litro"),
    # ── Time ─────────────────────────────────────────────────────────────────
    "godzina": BaseUnit(
        "godzina", "godziny", "godzin", "f",
        "godziny", "godzinie", "godziną", "godzinach", "godzinami", "godzino"),
    "minuta": BaseUnit(
        "minuta", "minuty", "minut", "f",
        "minuty", "minucie", "minutą", "minutach", "minutami", "minuto"),
    "sekunda": BaseUnit(
        "sekunda", "sekundy", "sekund", "f",
        "sekundy", "sekundzie", "sekundą", "sekundach", "sekundami", "sekundo"),
    # ── Digital ──────────────────────────────────────────────────────────────
    "bajt": BaseUnit(
        "bajt", "bajty", "bajtów", "m",
        "bajta", "bajcie", "bajtem", "bajtach", "bajtami", "bajto"),
    "bit": BaseUnit(
        "bit", "bity", "bitów", "m",
        "bita", "bicie", "bitem", "bitach", "bitami", "bito"),
    # ── Power ────────────────────────────────────────────────────────────────
    "wat": BaseUnit(
        "wat", "waty", "watów", "m",
        "wata", "wacie", "watem", "watach", "watami", "wato"),
    # ── Frequency ────────────────────────────────────────────────────────────
    "herc": BaseUnit(
        "herc", "herce", "herców", "m",
        "herca", "hercu", "hercem", "hercach", "hercami", "herco"),
    # ── Electrical ───────────────────────────────────────────────────────────
    "wolt": BaseUnit(
        "wolt", "wolty", "woltów", "m",
        "wolta", "wolcie", "woltem", "woltach", "woltami", "wolto"),
    "amper": BaseUnit(
        "amper", "ampery", "amperów", "m",
        "ampera", "amperze", "amperem", "amperach", "amperami", "ampero"),
    # ── Temperature ──────────────────────────────────────────────────────────
    "kelwin": BaseUnit(
        "kelwin", "kelwiny", "kelwinów", "m",
        "kelwina", "kelwinie", "kelwinem", "kelwinach", "kelwinami", ""),
    "stopień": BaseUnit(
        "stopień", "stopnie", "stopni", "m",
        "stopnia", "stopniu", "stopniem", "stopniach", "stopniami", ""),
    # ── Mechanical ───────────────────────────────────────────────────────────
    "niuton": BaseUnit(
        "niuton", "niutony", "niutonów", "m",
        "niutona", "niutonie", "niutonem", "niutonach", "niutonami", ""),
    # ── Pressure ─────────────────────────────────────────────────────────────
    "paskal": BaseUnit(
        "paskal", "paskale", "paskali", "m",
        "paskala", "paskalu", "paskalem", "paskalach", "paskalami", ""),
    "bar": BaseUnit(
        "bar", "bary", "barów", "m",
        "baru", "barze", "barem", "barach", "barami", ""),
    "atmosfera": BaseUnit(
        "atmosfera", "atmosfery", "atmosfer", "f",
        "atmosfery", "atmosferze", "atmosferą", "atmosferach", "atmosferami", ""),
    # ── Acoustics ────────────────────────────────────────────────────────────
    "bel": BaseUnit(
        "bel", "bele", "beli", "m",
        "bela", "belu", "belem", "belach", "belami", ""),
    # ── Display ──────────────────────────────────────────────────────────────
    "piksel": BaseUnit(
        "piksel", "piksele", "pikseli", "m",
        "piksela", "pikselu", "pikselem", "pikselach", "pikselami", ""),
    # ── Area ─────────────────────────────────────────────────────────────────
    "hektar": BaseUnit(
        "hektar", "hektary", "hektarów", "m",
        "hektara", "hektarze", "hektarem", "hektarach", "hektarami", ""),
    # ── Currency ─────────────────────────────────────────────────────────────
    "grosz": BaseUnit(
        "grosz", "grosze", "groszy", "m",
        "grosza", "groszu", "groszem", "groszach", "groszami", ""),
    # ── Rotation ─────────────────────────────────────────────────────────────
    "obrót": BaseUnit(
        "obrót", "obroty", "obrotów", "m",
        "obrotu", "obrocie", "obrotem", "obrotach", "obrotami", ""),
    # ── Capacitance ─────────────────────────────────────────────────────────
    "farad": BaseUnit(
        "farad", "farady", "faradów", "m",
        "farada", "faradzie", "faradem", "faradach", "faradami", ""),
    # ── Energy (calorie) ────────────────────────────────────────────────────
    "kaloria": BaseUnit(
        "kaloria", "kalorie", "kalorii", "f",
        "kalorii", "kalorii", "kalorią", "kaloriach", "kaloriami", "kalorio"),
    # ── Length (pikometr base) ──────────────────────────────────────────────
    "pikometr": BaseUnit(
        "pikometr", "pikometry", "pikometrów", "m",
        "pikometra", "pikometrze", "pikometrem", "pikometrach", "pikometrami", ""),
    # ── Time (millisecond base) ─────────────────────────────────────────────
    "milisekunda": BaseUnit(
        "milisekunda", "milisekundy", "milisekund", "f",
        "milisekundy", "milisekundzie", "milisekundą", "milisekundach", "milisekundami", ""),
}


# ═══════════════════════════════════════════════════════════════════════════════
# MODIFIER ADJECTIVE FORMS — for ² (kwadratowy) and ³ (sześcienny)
# ═══════════════════════════════════════════════════════════════════════════════

_MODIFIER_FORMS: dict[str, dict[str, str]] = {
    "²": {
        "nom_sg": "kwadratowy",    "nom_pl234": "kwadratowe",
        "nom_pl5": "kwadratowych", "gen_sg": "kwadratowego",
        "loc_sg": "kwadratowym",   "inst_sg": "kwadratowym",
        "loc_pl": "kwadratowych",  "inst_pl": "kwadratowymi",
    },
    "³": {
        "nom_sg": "sześcienny",    "nom_pl234": "sześcienne",
        "nom_pl5": "sześciennych", "gen_sg": "sześciennego",
        "loc_sg": "sześciennym",   "inst_sg": "sześciennym",
        "loc_pl": "sześciennych",  "inst_pl": "sześciennymi",
    },
}

# ═══════════════════════════════════════════════════════════════════════════════
# RATE SUFFIXES — fixed suffixes for per-time units
# ═══════════════════════════════════════════════════════════════════════════════

_RATE_SUFFIXES: dict[str, str] = {
    "/h":   "na godzinę",
    "/s":   "na sekundę",
    "/min": "na minutę",
    "/l":   "na litr",
    "/km":  "na kilometr",
    "/kg":  "na kilogram",
    "/ha":  "na hektar",
}


# ═══════════════════════════════════════════════════════════════════════════════
# GENERATION FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def _gen_prefixed(prefix: str, base_key: str) -> UnitForms:
    """Generate forms for a SI-prefixed unit (e.g., kilo + metr → kilometr).

    For unprefixed units, pass prefix="".
    """
    b = _BASE_UNITS[base_key]
    return UnitForms(
        nom_sg=prefix + b.nom_sg,       nom_pl234=prefix + b.nom_pl234,
        nom_pl5=prefix + b.nom_pl5,     gender=b.gender,
        gen_sg=prefix + b.gen_sg,       loc_sg=prefix + b.loc_sg,
        inst_sg=prefix + b.inst_sg,     loc_pl=prefix + b.loc_pl,
        inst_pl=prefix + b.inst_pl,
    )


def _gen_modified(prefix: str, base_key: str, modifier: str) -> UnitForms:
    """Generate unit + adjective modifier (e.g., metr kwadratowy)."""
    base = _gen_prefixed(prefix, base_key)
    m = _MODIFIER_FORMS[modifier]
    return UnitForms(
        nom_sg=f"{base.nom_sg} {m['nom_sg']}",
        nom_pl234=f"{base.nom_pl234} {m['nom_pl234']}",
        nom_pl5=f"{base.nom_pl5} {m['nom_pl5']}",
        gender=base.gender,
        gen_sg=f"{base.gen_sg} {m['gen_sg']}",
        loc_sg=f"{base.loc_sg} {m['loc_sg']}",
        inst_sg=f"{base.inst_sg} {m['inst_sg']}",
        loc_pl=f"{base.loc_pl} {m['loc_pl']}",
        inst_pl=f"{base.inst_pl} {m['inst_pl']}",
    )


def _gen_rate(prefix: str, base_key: str, rate: str) -> UnitForms:
    """Generate rate unit with fixed suffix (e.g., kilometr na godzinę)."""
    base = _gen_prefixed(prefix, base_key)
    suffix = _RATE_SUFFIXES[rate]
    return UnitForms(
        nom_sg=f"{base.nom_sg} {suffix}",
        nom_pl234=f"{base.nom_pl234} {suffix}",
        nom_pl5=f"{base.nom_pl5} {suffix}",
        gender=base.gender,
        gen_sg=f"{base.gen_sg} {suffix}",
        loc_sg=f"{base.loc_sg} {suffix}",
        inst_sg=f"{base.inst_sg} {suffix}",
        loc_pl=f"{base.loc_pl} {suffix}",
        inst_pl=f"{base.inst_pl} {suffix}",
    )


def _gen_compound_time(prefix: str, base_key: str, time_key: str) -> UnitForms:
    """Generate compound-time unit (e.g., kilo + wato + godzina → kilowatogodzina).

    Uses the base unit's connecting form to fuse with the time unit.
    Result inherits gender/declension from the time unit.
    """
    base = _BASE_UNITS[base_key]
    time = _BASE_UNITS[time_key]
    cp = prefix + base.connect  # e.g., "kilo" + "wato" = "kilowato"
    return UnitForms(
        nom_sg=cp + time.nom_sg,       nom_pl234=cp + time.nom_pl234,
        nom_pl5=cp + time.nom_pl5,     gender=time.gender,
        gen_sg=cp + time.gen_sg,       loc_sg=cp + time.loc_sg,
        inst_sg=cp + time.inst_sg,     loc_pl=cp + time.loc_pl,
        inst_pl=cp + time.inst_pl,
    )


def _gen_qualified(prefix: str, base_key: str, qualifier: str) -> UnitForms:
    """Generate unit + fixed genitive qualifier (e.g., stopień Celsjusza)."""
    base = _gen_prefixed(prefix, base_key)
    return UnitForms(
        nom_sg=f"{base.nom_sg} {qualifier}",
        nom_pl234=f"{base.nom_pl234} {qualifier}",
        nom_pl5=f"{base.nom_pl5} {qualifier}",
        gender=base.gender,
        gen_sg=f"{base.gen_sg} {qualifier}",
        loc_sg=f"{base.loc_sg} {qualifier}",
        inst_sg=f"{base.inst_sg} {qualifier}",
        loc_pl=f"{base.loc_pl} {qualifier}",
        inst_pl=f"{base.inst_pl} {qualifier}",
    )


# ═══════════════════════════════════════════════════════════════════════════════
# ABBREVIATION → GENERATION RULES
# ═══════════════════════════════════════════════════════════════════════════════

# Type 1: Simple or SI-prefixed — (polish_prefix, base_key)
_SIMPLE_UNITS: dict[str, tuple[str, str]] = {
    # Length
    "km": ("kilo", "metr"),   "m": ("", "metr"),    "cm": ("centy", "metr"),
    "mm": ("mili", "metr"),   "nm": ("nano", "metr"), "µm": ("mikro", "metr"),
    # Mass
    "kg": ("kilo", "gram"),   "g": ("", "gram"),    "mg": ("mili", "gram"),
    "µg": ("mikro", "gram"),  "dag": ("deka", "gram"),  "t": ("", "tona"),
    # Volume
    "l": ("", "litr"),        "ml": ("mili", "litr"),
    # Time
    "h": ("", "godzina"),     "min": ("", "minuta"),
    "s": ("", "sekunda"),     "sek": ("", "sekunda"),
    # Power
    "W": ("", "wat"),         "kW": ("kilo", "wat"),  "MW": ("mega", "wat"),
    # Frequency
    "Hz": ("", "herc"),       "MHz": ("mega", "herc"), "GHz": ("giga", "herc"),
    "kHz": ("kilo", "herc"),
    # Data — bytes
    "B": ("", "bajt"),        "kB": ("kilo", "bajt"), "KB": ("kilo", "bajt"),
    "MB": ("mega", "bajt"),   "GB": ("giga", "bajt"),
    "TB": ("tera", "bajt"),   "PB": ("peta", "bajt"),
    # Data — bits
    "Gb": ("giga", "bit"),    "Mb": ("mega", "bit"),  "Kb": ("kilo", "bit"),
    # Temperature
    "K": ("", "kelwin"),      "°": ("", "stopień"),
    # Electrical
    "V": ("", "wolt"),        "A": ("", "amper"),
    # Mechanical
    "N": ("", "niuton"),
    # Pressure
    "Pa": ("", "paskal"),     "hPa": ("hekto", "paskal"), "kPa": ("kilo", "paskal"),
    "bar": ("", "bar"),       "atm": ("", "atmosfera"),
    # Acoustics
    "dB": ("decy", "bel"),
    # Capacitance
    "µF": ("mikro", "farad"),
    # Display
    "px": ("", "piksel"),     "MP": ("mega", "piksel"),  "Mpx": ("mega", "piksel"),
    # Area
    "ha": ("", "hektar"),
    # Currency
    "gr": ("", "grosz"),
    # Mechanical
    "obr": ("", "obrót"),
    # Energy
    "kcal": ("kilo", "kaloria"),
    # Length
    "pm": ("", "pikometr"),
    # Time
    "ms": ("", "milisekunda"),
}

# Type 2: Modified (unit + adjective) — (prefix, base_key, modifier)
_MODIFIED_UNITS: dict[str, tuple[str, str, str]] = {
    "km²": ("kilo", "metr", "²"),   "m²": ("", "metr", "²"),
    "cm²": ("centy", "metr", "²"),  "mm²": ("mili", "metr", "²"),
    "m³": ("", "metr", "³"),        "cm³": ("centy", "metr", "³"),
}

# Type 3: Rate (unit per time) — (prefix, base_key, rate_suffix_key)
_RATE_UNITS_MAP: dict[str, tuple[str, str, str]] = {
    "km/h": ("kilo", "metr", "/h"),
    "m/s": ("", "metr", "/s"),      "km/s": ("kilo", "metr", "/s"),
    "Gb/s": ("giga", "bit", "/s"),  "Mb/s": ("mega", "bit", "/s"),
    "kb/s": ("kilo", "bit", "/s"),
    "MB/s": ("mega", "bajt", "/s"), "GB/s": ("giga", "bajt", "/s"),
    "kB/s": ("kilo", "bajt", "/s"),
    "l/h": ("", "litr", "/h"),
    "kg/ha": ("kilo", "gram", "/ha"),
    "kcal/h": ("kilo", "kaloria", "/h"),
}

# Type 4: Compound-time (base×time fused) — (prefix, base_key, time_key)
_COMPOUND_TIME_UNITS: dict[str, tuple[str, str, str]] = {
    "kWh": ("kilo", "wat", "godzina"),
    "mAh": ("mili", "amper", "godzina"),
    "Ah":  ("", "amper", "godzina"),
    "Ws":  ("", "wat", "sekunda"),
}

# Type 5: Qualified (unit + fixed genitive) — (prefix, base_key, qualifier)
_QUALIFIED_UNITS: dict[str, tuple[str, str, str]] = {
    "°C":   ("", "stopień", "Celsjusza"),
    "°F":   ("", "stopień", "Fahrenheita"),
    "mmHg": ("mili", "metr", "słupa rtęci"),
}

# Type 6: Explicit (non-compositional) — hand-specified UnitForms
_EXPLICIT_UNITS: dict[str, UnitForms] = {
    "fps": UnitForms(
        "klatka na sekundę", "klatki na sekundę", "klatek na sekundę", "f",
        "klatki na sekundę", "klatce na sekundę", "klatką na sekundę",
        "klatkach na sekundę", "klatkami na sekundę"),
    "dpi": UnitForms(
        "punkt na cal", "punkty na cal", "punktów na cal", "m",
        "punktu na cal", "punkcie na cal", "punktem na cal",
        "punktach na cal", "punktami na cal"),
    "str./min.": UnitForms(
        "strona na minutę", "strony na minutę", "stron na minutę", "f",
        "strony na minutę", "stronie na minutę", "stroną na minutę",
        "stronach na minutę", "stronami na minutę"),
    "str./min": UnitForms(  # alias without trailing dot
        "strona na minutę", "strony na minutę", "stron na minutę", "f",
        "strony na minutę", "stronie na minutę", "stroną na minutę",
        "stronach na minutę", "stronami na minutę"),
    "m³/s": UnitForms(
        "metr sześcienny na sekundę", "metry sześcienne na sekundę", "metrów sześciennych na sekundę", "m",
        "metra sześciennego na sekundę", "metrze sześciennym na sekundę", "metrem sześciennym na sekundę",
        "metrach sześciennych na sekundę", "metrami sześciennymi na sekundę"),
    "g/cm³": UnitForms(
        "gram na centymetr sześcienny", "gramy na centymetr sześcienny", "gramów na centymetr sześcienny", "m",
        "grama na centymetr sześcienny", "gramie na centymetr sześcienny", "gramem na centymetr sześcienny",
        "gramach na centymetr sześcienny", "gramami na centymetr sześcienny"),
    "m/s²": UnitForms(
        "metr na sekundę do kwadratu", "metry na sekundę do kwadratu", "metrów na sekundę do kwadratu", "m",
        "metra na sekundę do kwadratu", "metrze na sekundę do kwadratu", "metrem na sekundę do kwadratu",
        "metrach na sekundę do kwadratu", "metrami na sekundę do kwadratu"),
    "mg/kg": UnitForms(
        "miligram na kilogram", "miligramy na kilogram", "miligramów na kilogram", "m",
        "miligrama na kilogram", "miligramie na kilogram", "miligramem na kilogram",
        "miligramach na kilogram", "miligramami na kilogram"),
    "obr./min": UnitForms(
        "obrót na minutę", "obroty na minutę", "obrotów na minutę", "m",
        "obrotu na minutę", "obrocie na minutę", "obrotem na minutę",
        "obrotach na minutę", "obrotami na minutę"),
    "µg/m³": UnitForms(
        "mikrogram na metr sześcienny", "mikrogramy na metr sześcienny", "mikrogramów na metr sześcienny", "m",
        "mikrograma na metr sześcienny", "mikrogramie na metr sześcienny", "mikrogramem na metr sześcienny",
        "mikrogramach na metr sześcienny", "mikrogramami na metr sześcienny"),
    "l/100 km": UnitForms(
        "litr na sto kilometrów", "litry na sto kilometrów", "litrów na sto kilometrów", "m",
        "litra na sto kilometrów", "litrze na sto kilometrów", "litrem na sto kilometrów",
        "litrach na sto kilometrów", "litrami na sto kilometrów"),
}


# ═══════════════════════════════════════════════════════════════════════════════
# REGISTRY BUILD
# ═══════════════════════════════════════════════════════════════════════════════

def _build_registry() -> dict[str, UnitForms]:
    """Build the complete unit registry from all rule types."""
    registry: dict[str, UnitForms] = {}

    for abbrev, (prefix, base_key) in _SIMPLE_UNITS.items():
        registry[abbrev] = _gen_prefixed(prefix, base_key)

    for abbrev, (prefix, base_key, mod) in _MODIFIED_UNITS.items():
        registry[abbrev] = _gen_modified(prefix, base_key, mod)

    for abbrev, (prefix, base_key, rate) in _RATE_UNITS_MAP.items():
        registry[abbrev] = _gen_rate(prefix, base_key, rate)

    for abbrev, (prefix, base_key, time_key) in _COMPOUND_TIME_UNITS.items():
        registry[abbrev] = _gen_compound_time(prefix, base_key, time_key)

    for abbrev, (prefix, base_key, qual) in _QUALIFIED_UNITS.items():
        registry[abbrev] = _gen_qualified(prefix, base_key, qual)

    registry.update(_EXPLICIT_UNITS)

    return registry


# ═══════════════════════════════════════════════════════════════════════════════
# PUBLIC API
# ═══════════════════════════════════════════════════════════════════════════════

class UnitRegistry:
    """Compositional unit registry providing inflected Polish unit forms.

    Drop-in replacement for the manual _UNITS and _UNIT_CASE_FORMS dicts.
    """

    def __init__(self) -> None:
        self._units = _build_registry()
        # Reverse lookup: nominative singular → UnitForms
        self._by_nom: dict[str, UnitForms] = {}
        for forms in self._units.values():
            if forms.nom_sg not in self._by_nom:
                self._by_nom[forms.nom_sg] = forms

    def get(self, abbrev: str) -> tuple[str, str, str, str] | None:
        """Get unit tuple by abbreviation. Same format as _UNITS[key]."""
        forms = self._units.get(abbrev)
        return forms.as_units_tuple() if forms else None

    def get_forms(self, abbrev: str) -> UnitForms | None:
        """Get full unit forms by abbreviation."""
        return self._units.get(abbrev)

    def get_case_forms(self, nom_sg: str, case: str) -> tuple[str, str, str] | None:
        """Get case forms by nominative singular + case name.

        Same lookup pattern as _UNIT_CASE_FORMS[nom_sg][case].
        """
        forms = self._by_nom.get(nom_sg)
        if forms is None:
            return None
        return forms.as_case_entries().get(case)

    def gender(self, abbrev: str) -> str:
        """Return gender of unit noun ('m' or 'f')."""
        forms = self._units.get(abbrev)
        return forms.gender if forms else "m"

    def gen_sg(self, abbrev: str) -> str | None:
        """Return genitive singular form for an abbreviation."""
        forms = self._units.get(abbrev)
        return forms.gen_sg if forms else None

    def as_units_dict(self) -> dict[str, tuple[str, str, str, str]]:
        """Export as _UNITS-compatible dict: {abbrev: (sg, pl234, pl5, gender)}."""
        return {k: v.as_units_tuple() for k, v in self._units.items()}

    def as_case_dict(self) -> dict[str, dict[str, tuple[str, str, str]]]:
        """Export as _UNIT_CASE_FORMS-compatible dict: {nom_sg: {case: (sg, pl, pl)}}.

        Unlike the old manual dict (~20 entries), this covers ALL units.
        """
        return {nom: f.as_case_entries() for nom, f in self._by_nom.items()}

    @property
    def abbreviations(self) -> list[str]:
        """All known abbreviation strings, sorted longest-first (for regex)."""
        return sorted(self._units.keys(), key=len, reverse=True)


# Module-level singleton
REGISTRY = UnitRegistry()
