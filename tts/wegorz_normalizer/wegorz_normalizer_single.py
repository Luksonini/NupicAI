#!/usr/bin/env python3
"""
WęgorzAI Polish TTS Text Normalizer — Single-File Edition

All normalization logic in one file: unit registry, Polish inflection, and the
full PolishTTSPipeline. Drop this file + slownik_wymowy.json into any project.

Usage:
    from wegorz_normalizer_single import PolishTTSPipeline
    pipe = PolishTTSPipeline()
    print(pipe.process("Spotkanie o godz. 14:30 kosztuje 29,99 zł."))

Requirements: num2words
"""

from __future__ import annotations
import re
import unicodedata
import logging
import json as _json
import os as _os
import sys
from functools import lru_cache
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════════
# PART 1: Unit Registry (from unit_registry.py)
# ═══════════════════════════════════════════════════════════════════════════════




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

# ═══════════════════════════════════════════════════════════════════════════════
# PART 2: Polish Number Inflection (from inflect_pl.py)
# ═══════════════════════════════════════════════════════════════════════════════

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

# ═══════════════════════════════════════════════════════════════════════════════
# PART 3: Main Normalization Pipeline (from tokenize_and_text_norm.py)
# ═══════════════════════════════════════════════════════════════════════════════

"""
╔══════════════════════════════════════════════════════════════════════════╗
║       POLISH TTS TEXT NORMALIZATION PIPELINE  v5                         ║
║                                                                          ║
║  Samodzielny potok: czytanie plików → normalizacja → tokenizacja         ║
║                                                                          ║
║  CZYTNIKI:  FileReader.read(path)  /  .read_chapters(path)               ║
║             Obsługuje: .txt  .pdf  .epub  .mobi                          ║
║                                                                          ║
║  NORMALIZACJA (5 kroków w PolishTTSPipeline):                            ║
║    1. raw_clean           — unicode, myślniki, obce skrypty, cudzysłowy  ║
║    2. abbreviation_expand — skróty polskie, cyfry rzymskie, URL/email    ║
║    3. foreign_expand      — skróty międzynarodowe (TTS→te te es),        ║
║                             imiona angielskie (Shakespeare→Szekspir),    ║
║                             zapożyczenia (startup→startap),              ║
║                             tokeny mieszane (GPT-4, Wi-Fi, 5G)           ║
║    4. num_normalize       — daty/%, waluty, jednostki, liczby → słowa PL ║
║    5. final_filter        — tylko znaki z vocab PLTokenizer              ║
║                                                                          ║
║  WBUDOWANY PLTokenizer:                                                  ║
║    pipe.tokenize(text)           → list[int]                             ║
║    pipe.count_unk(text)          → int   (0 = brak <unk>)                ║
║    pipe.process_file(path, ...)  → str | list[int] | rozdziały           ║
║                                                                          ║
║  Gwarancja: 0 tokenów <unk> na wyjściu.                                  ║
╚══════════════════════════════════════════════════════════════════════════╝
"""

# ─────────────────────────────────────────────────────────────────────────────
# CELL 1 ▸ Zależności (bez auto-instalacji)
# ─────────────────────────────────────────────────────────────────────────────
#
# Ten plik jest używany w produkcji / w Gradio i musi być importowalny bez efektów
# ubocznych. Zależności instalujemy w środowisku (pip/conda), a tutaj tylko
# sprawdzamy importy i w razie braku rzucamy czytelny błąd.
#
# Wymagane:
#   - num2words
# Opcjonalne (tylko dla czytników plików):
#   - pymupdf (fitz) dla PDF
#   - ebooklib + html2text dla EPUB
#   - mobi dla MOBI

# ─────────────────────────────────────────────────────────────────────────────
# CELL 2 ▸ Stałe i mapy znaków zgodne z PLTokenizer
# ─────────────────────────────────────────────────────────────────────────────

import re
import unicodedata
import json as _json
import os as _os

logger = logging.getLogger(__name__)

# ── Polish morphological inflection (case-aware number forms) ─────────────────
# Aliases for embedded modules (inflect_pl and unit_registry are above)
_cardinal_inflect = cardinal_inflect
_ordinal_inflect = ordinal_inflect
_hour_ordinal = hour_ordinal
_REGISTRY = REGISTRY

# ── Słowniki wypełniane z slownik_wymowy.json (patrz _load_slownik_wymowy) ──
_ABBREV_MAP: dict = {}            # skróty polskie      (kategoria: skróty)
_LETTER_NAMES_PL: dict = {}       # nazwy liter PL      (kategoria: litery_pl)
_INTL_ABBREV_MAP: dict = {}       # skróty między.      (kategoria: skrótowce)
_ENGLISH_NAMES_PL: dict = {}      # imiona/nazwiska EN  (kategoria: nazwy_własne_en, imiona_*)
_ENGLISH_LOANWORDS_PL: dict = {}  # zapożyczenia EN     (kategoria: zapożyczenia_en)
_DOTTED_ABBREV_MAP: dict = {}     # skróty z kropkami   (kategoria: skróty_kropkowane)

# ── Wczytaj slownik_wymowy.json i wypełnij słowniki ─────────────────────────
def _load_slownik_wymowy() -> None:
    """
    Wczytuje slownik_wymowy.json z tego samego katalogu co ten skrypt
    i wypełnia wszystkie moduł-poziomowe słowniki:
      _ABBREV_MAP        ← kategoria: skróty
      _LETTER_NAMES_PL   ← kategoria: litery_pl
      _INTL_ABBREV_MAP   ← kategoria: skrótowce
      _ENGLISH_NAMES_PL  ← kategoria: nazwy_własne_en, imiona_m_en, imiona_f_en, imiona_m_*, imiona_f_*
      _ENGLISH_LOANWORDS_PL ← kategoria: zapożyczenia_en
      _DOTTED_ABBREV_MAP ← kategoria: skróty_kropkowane
    """
    _here = _os.path.dirname(_os.path.abspath(__file__))
    _path = _os.path.join(_here, "slownik_wymowy.json")
    if not _os.path.exists(_path):
        logger.warning("slownik_wymowy.json not found at %s — skipping", _path)
        return
    try:
        with open(_path, encoding="utf-8") as _f:
            _data = _json.load(_f)
    except Exception as _e:
        logger.warning("Could not load slownik_wymowy.json: %s", _e)
        return

    for _key, _entry in _data.items():
        _pl = _entry.get("pl")
        if not _pl:
            continue
        _cat = _entry.get("kategoria", "")
        if _cat == "skróty":
            _ABBREV_MAP[_key] = _pl
        elif _cat == "litery_pl":
            _LETTER_NAMES_PL[_key] = _pl
        elif _cat == "skrótowce":
            _INTL_ABBREV_MAP[_key.upper()] = _pl
        elif _cat in {"nazwy_własne_en", "imiona_m_en", "imiona_f_en",
                      "imiona_m_ru", "imiona_f_ru", "imiona_m_de", "imiona_f_de",
                      "imiona_m_fr", "imiona_f_fr"}:
            _ENGLISH_NAMES_PL[_key] = _pl
        elif _cat == "zapożyczenia_en":
            _ENGLISH_LOANWORDS_PL[_key] = _pl
        elif _cat == "skróty_kropkowane":
            _DOTTED_ABBREV_MAP[_key] = _pl

    logger.debug("slownik_wymowy.json loaded: %d entries", len(_data))

_load_slownik_wymowy()

# ── Prekompilacja wzorców dla foreign_expand (po załadowaniu słowników) ──────
# Wzorce są kompilowane RAZ przy imporcie, nie przy każdym wywołaniu.
# Każdy wpis to krotka (pattern, replacement) lub (pattern, phonetic) dla nazw.
_COMPILED_LOANWORDS: list = []   # [(compiled_re, replacement), ...]
_COMPILED_NAMES: list = []       # [(compiled_re, phonetic), ...]
_COMPILED_ABBREVS: list = []     # [(compiled_re, expansion, abbrev), ...] — precompiled abbreviation patterns

def _compile_abbrev_patterns() -> None:
    """Kompiluje wzorce dla _ABBREV_MAP RAZ przy załadowaniu modułu."""
    global _COMPILED_ABBREVS
    _COMPILED_ABBREVS = []
    for abbrev, expansion in sorted(_ABBREV_MAP.items(), key=lambda x: -len(x[0])):
        escaped = re.escape(abbrev)
        if abbrev.endswith("."):
            if len(abbrev) == 2 and abbrev[0].isalpha():
                # Jednoliterowe skróty (o. z. s. itp.) — nie matchuj przed wielokropkiem (...)
                pattern = re.compile(rf"(?<![A-Za-z]\.)(?<!')\b{escaped}(?![A-Za-z]\.)(?!\.)")
            else:
                pattern = re.compile(rf"(?<!\.)\b{escaped}(?!\.)", re.IGNORECASE)
        else:
            pattern = re.compile(rf"\b{escaped}\b", re.IGNORECASE)
        _COMPILED_ABBREVS.append((pattern, expansion, abbrev))

def _compile_lookup_patterns() -> None:
    """Kompiluje i cachuje wzorce regex dla _ENGLISH_LOANWORDS_PL i _ENGLISH_NAMES_PL."""
    global _COMPILED_LOANWORDS, _COMPILED_NAMES
    _COMPILED_LOANWORDS = []
    for eng, pol in sorted(_ENGLISH_LOANWORDS_PL.items(), key=lambda x: -len(x[0])):
        if "-" in eng:
            _COMPILED_LOANWORDS.append((re.compile(re.escape(eng), re.IGNORECASE), pol, True))
        else:
            _COMPILED_LOANWORDS.append((re.compile(rf"\b{re.escape(eng)}\b", re.IGNORECASE), pol, False))
    _COMPILED_NAMES = []
    # Sufiksy odmiany polskiej dla angielskich imion:
    # - samogłoska + końcówka wieloznakowa (Johnach, Johnem itd.)
    # - lub samo 'a'/'i'/'y' (gen. masc. -a, gen./dat. fem. -i/-y)
    # NIE dopasowujemy samotnego 'o'/'e'/'u' — to by złapało polskie słowa jak "Dawno"!
    _SUFFIX = r"(?:')?([aeiouyąę](?:ch|mi|em|owi|ie|om|ów|ach)|[aiy]|em|ie|owi|ów|om|ach)?(?=\s|[.,!?:;)\]\-]|$)"
    # Imiona angielskie których rdzeń pokrywa się z polskimi imionami
    # (suffix regex łapie polskie formy deklinacyjne: Jacka, Barbary, Anny)
    _NAME_SKIP_STEMS = frozenset({
        "Jack", "Ann", "Barbara", "Barbar", "Dan", "Dag",
        "Mark", "Martin", "Anton", "Adrian", "Gabriel",
        "Sebastian", "Norbert", "Robert", "Roland",
    })
    for name, phonetic in sorted(_ENGLISH_NAMES_PL.items(), key=lambda x: -len(x[0])):
        if name in _NAME_SKIP_STEMS:
            continue
        pat = re.compile(rf"\b{re.escape(name)}{_SUFFIX}")
        _COMPILED_NAMES.append((pat, phonetic))

_compile_lookup_patterns()
_compile_abbrev_patterns()

# ── Skompilowane wzorce regex dla foreign_expand ─────────────────────────────
# Skróty z kropkami: U.S.A. / e.g. / Ph.D. (2+ liter z kropkami)
_RE_DOTTED_ABBREV = re.compile(
    r"\b(?:[A-Za-z]\.){2,}[A-Za-z]?\.?"
)
# Tokeny mieszane z myślnikiem: GPT-4, Wi-Fi, COVID-19
_RE_MIXED_HYPHEN = re.compile(
    r"\b[A-Za-zĄĆĘŁŃÓŚŹŻąćęłńóśźż0-9]+(?:-[A-Za-zĄĆĘŁŃÓŚŹŻąćęłńóśźż0-9]+)+\b"
)
# Wszystkie wielkie litery (2+ znaków): TTS, API, NATO
_RE_ALLCAPS = re.compile(
    r"\b[A-ZĄĆĘŁŃÓŚŹŻ][A-ZĄĆĘŁŃÓŚŹŻ0-9]{1,}\b"
)
# CamelCase: YouTube, iPhone, ChatGPT, JavaScript
_RE_CAMELCASE = re.compile(
    r"\b(?:[A-Z][a-z]+){2,}\b|\b[a-z]+(?:[A-Z][a-z]*)+\b|\b[A-Z][a-z]+(?:[A-Z][A-Za-z]*)+\b"
)

# Samogłoski do testu wymawialności (ASCII + polskie)
_VOWELS = set("aeiouyAEIOUYąęóÓĄĘ")


# Polskie krótkie słowa (przyimki, spójniki, zaimki) — pisane ALL-CAPS w tytułach
# Powinny być czytane jako słowa, nie literowane
_POLISH_SHORT_WORDS = frozenset([
    "do", "na", "po", "od", "za", "ze", "we", "ku", "bo", "co", "go", "mu",
    "no", "to", "ni", "tu", "ta", "te", "ja", "ty", "on", "my", "wy", "je",
    "im", "jej", "ich", "nam", "wam", "one", "się", "czy", "nie", "ale",
    "lub", "że", "jak", "gdy", "już", "też", "też", "aż", "też",
])

# Polskie słowa, których rdzenie pokrywają się ze skrótami zakończonymi kropką.
# Jeśli taki rdzeń pojawia się przed kropką zdaniową (koniec zdania),
# NIE rozwijaj go jako skrót (np. "dom." ≠ "domowy", "ust." ≠ "ustawa").
_ABBREV_SAFE_WORDS = frozenset({
    "dom", "ust", "rzecz", "tłum", "cel", "but", "par", "port",
    "raj", "dal", "bit", "rym", "pl", "go", "mi", "im", "ci",
    "dam", "mam", "pas", "raz", "rad", "ran", "sam", "gram",
    "sen", "os", "moc", "rok", "las", "lot", "tan", "ton",
    "ból", "pot", "pat", "skok", "most", "post",
    # Dodane po audycie — słowa pojawiające się na końcu zdań w audiobookach
    "ok", "hm", "mat", "kat", "paw", "rys", "tir", "woj",
    "marsz", "scen", "kraj", "miejsc", "lok", "o",
    "ps",  # "ps." = pseudonim, ale na końcu zdania "PS." to skrótowiec (pe es)
    "s",   # "s." na końcu zdania/przed wielką literą = sekunda, nie "strona"
})

# Skróty, które ZAWSZE powinny być rozwijane — nawet przed wielką literą.
# Guard na inicjały (len(stem)<=2 + uppercase after) NIE blokuje tych skrótów.
_ALWAYS_EXPAND_ABBREVS = frozenset({
    "np", "ul", "nr", "dr", "wg", "ok", "tzn", "tzw", "al",
    "prof", "doc", "hab", "inż", "mgr", "gen", "mjr", "ppłk", "płk",
    "godz", "im", "os",
    # Jednoliterowe skróty, które też zawsze rozwijamy:
    "t",   # tom (publishing context: "t. IV")
    # "s" removed — handled as unit "sekunda" by num_normalize (s. with dot = strona handled separately)
    "r",   # rok (handled elsewhere but listed for completeness)
    "w",   # wu (nazwa litery) / inicjał
    "m",   # miasto / em (nazwa litery)
    "p",   # pan
    "d",   # dawny
    "k",   # koło
    "n",   # numer
    "g",   # gie (nazwa litery)
    "j",   # język
    "l",   # liczba
})

_PRONOUNCEABLE_OVERRIDES = {
    "galaxy", "index", "max", "linux", "unix", "latex", "flex", "hex",
    "mix", "fox", "box", "apex", "onyx", "nexus", "pixel", "vortex",
    "matrix", "complex", "reflex", "annex", "arch", "ajax", "proxy",
    "expo", "taxi", "extra", "text", "next", "exit", "oxide", "exam",
    "exact", "excel", "exec", "exist", "export", "express", "extend",
    "extreme", "exchange", "exclude", "except", "excite", "exercise",
    "explore", "exploit", "expose", "external", "extract",
    "probe", "orbit", "node", "code", "mode", "vote", "note",
    "seed", "feed", "speed", "breed", "greed",
    "init", "emit", "admit", "permit", "submit",
    "alert", "insert", "convert", "revert", "assert",
    "status", "campus", "focus", "bonus", "virus", "census",
    "delta", "alpha", "beta", "gamma", "sigma", "omega",
}

# Skróty, które brzmią jak słowa ale ZAWSZE powinny być literowane
_SPELL_OVERRIDES = {
    "api", "ipo", "iso", "esa", "ema", "eta", "ore",
    "ppe", "age", "ace", "ice", "ode", "ore", "ape",
    "acm",  # Association for Computing Machinery — literuj
    "gpu", "cpu",  # Always spell out hardware abbreviations
    "tdi",  # Turbocharged Direct Injection — always spell out
    "eip",  # Ekspresowy InterCity Premium — always spell out
    "pvc",  # Polyvinyl chloride — always spell out
}

def _is_pronounceable(word: str) -> bool:
    """
    Czy słowo (all-caps) da się wymówić jako słowo, czy trzeba literować?
    NATO → True (czytamy "nato"), TTS → False (literujemy "te te es").

    Heurystyka oparta na polskiej fonotaktyce: sprawdza samogłoski,
    proporcje i klastry spółgłoskowe.
    """
    w = word.lower()
    if w in _PRONOUNCEABLE_OVERRIDES:
        return True
    if w in _SPELL_OVERRIDES:
        return False
    # Polskie krótkie słowa w ALL-CAPS (tytuły, nagłówki) — czytaj jako słowa
    if w in _POLISH_SHORT_WORDS:
        return True
    if len(w) <= 2:
        return False  # 2-literowe zawsze literuj (IT, AI, PC)
    # Litera 'q' nie jest polska — tokeny z q raczej literować
    if "q" in w:
        return False
    vowel_count = sum(1 for c in w if c in _VOWELS)
    if vowel_count == 0:
        return False
    ratio = vowel_count / len(w)
    # 3-literowe — przyjmij 1 samogłoska na 3 litery (ratio ≈ 0.33)
    if len(w) == 3 and ratio < 0.30:
        return False
    if len(w) == 4 and ratio < 0.25:
        return False
    if ratio < 0.2:
        return False
    # Sprawdź czy nie ma zbyt długich klastrów spółgłoskowych
    # Polski dopuszcza max 3-4 w praktyce (wstrząs), ale w skrótach
    # klaster >2 na początku/końcu = niewymawialny
    max_consonant_run = 0
    current_run = 0
    for c in w:
        if c not in _VOWELS:
            current_run += 1
            max_consonant_run = max(max_consonant_run, current_run)
        else:
            current_run = 0
    if max_consonant_run > 3:
        return False
    # Sprawdź czy zaczyna się lub kończy na >2 spółgłoski
    start_consonants = 0
    for c in w:
        if c not in _VOWELS:
            start_consonants += 1
        else:
            break
    end_consonants = 0
    for c in reversed(w):
        if c not in _VOWELS:
            end_consonants += 1
        else:
            break
    # Polski dopuszcza 3 spółgłoski na początku: przez, strz, wstr, szcz itp.
    if start_consonants > 3 or end_consonants > 2:
        return False
    return True


def _spell_abbreviation(word: str) -> str:
    """
    Literuje skrót po polsku: TTS → "te te es", API → "a pe i".
    Cyfry w środku zamieniane na polskie słowa.
    """
    parts = []
    for ch in word.upper():
        if ch in _LETTER_NAMES_PL:
            parts.append(_LETTER_NAMES_PL[ch])
        elif ch.isdigit():
            parts.append(num2words(int(ch), lang="pl"))
        else:
            parts.append(ch)
    return " ".join(parts)


def _expand_mixed_token(token: str) -> str:
    """
    Rozwija tokeny mieszane z myślnikiem: GPT-4, Wi-Fi, COVID-19.
    """
    # Najpierw sprawdź cały token w słownikach
    upper = token.upper()
    if upper in _INTL_ABBREV_MAP:
        return _INTL_ABBREV_MAP[upper]
    if token in _ENGLISH_LOANWORDS_PL:
        return _ENGLISH_LOANWORDS_PL[token]

    segments = token.split("-")
    # If left segment contains Polish-specific chars, preserve as Polish word (e.g. węgla-14)
    _POLISH_CHARS = set("ąćęłńóśźżĄĆĘŁŃÓŚŹŻ")
    if len(segments) == 2 and any(c in _POLISH_CHARS for c in segments[0]):
        right = segments[1]
        if right.isdigit():
            return f"{segments[0]} - {num2words(int(right), lang='pl')}"
        return f"{segments[0]} - {right}"
    # Detect model/product designator: CAPS-letter (USB-C, HDMI-A) or letter-3+DIGITS (F-123)
    # Preserve hyphen as separator in these cases
    is_model_designator = (
        len(segments) == 2 and (
            # ACRONYM-letter: USB-C, HDMI-A
            (segments[0].isupper() and len(segments[0]) >= 2 and
             len(segments[1]) == 1 and segments[1].isalpha()) or
            # Letter-3+DIGITS: F-123 (not F-16, B-52 which are known models without dash)
            (len(segments[0]) == 1 and segments[0].isalpha() and
             segments[1].isdigit() and len(segments[1]) >= 3)
        )
    )
    result_parts = []
    for seg in segments:
        seg_upper = seg.upper()
        # Sprawdź w słowniku skrótów
        if seg_upper in _INTL_ABBREV_MAP:
            result_parts.append(_INTL_ABBREV_MAP[seg_upper])
        elif seg.isdigit():
            result_parts.append(num2words(int(seg), lang="pl"))
        elif len(seg) == 1 and seg.isalpha():
            # Jeśli poprzedni segment był akronimem (ALL-CAPS), to ten segment
            # jest prawdopodobnie polskim sufiksem deklinacyjnym (SMS-y, GPS-ie)
            if result_parts and seg.islower():
                # Dołącz sufiks do poprzedniego segmentu zamiast literować
                result_parts[-1] = result_parts[-1] + seg
            else:
                # Pojedyncza litera → polska nazwa litery
                result_parts.append(_LETTER_NAMES_PL.get(seg.upper(), seg.lower()))
        elif seg.isupper() and len(seg) >= 2:
            if _is_pronounceable(seg):
                result_parts.append(seg.lower())
            else:
                result_parts.append(_spell_abbreviation(seg))
        elif seg.islower() and result_parts and seg in (
            "y", "ów", "om", "ach", "ami", "em", "ie", "a", "u",
            "owy", "owe", "owego", "owym", "owych",
        ):
            # Polski sufiks deklinacyjny po akronimie (SMS-ów, GPS-ach, IT-owy)
            result_parts[-1] = result_parts[-1] + seg
        elif seg in _ENGLISH_NAMES_PL:
            result_parts.append(_ENGLISH_NAMES_PL[seg])
        elif seg in _ENGLISH_LOANWORDS_PL:
            result_parts.append(_ENGLISH_LOANWORDS_PL[seg])
        else:
            # Rozbij segmenty mieszane jak "16bit", "1200Ws" na części
            mixed = re.findall(r'(\d+|[A-Za-z]+)', seg)
            if len(mixed) > 1:
                for part in mixed:
                    if part.isdigit():
                        result_parts.append(num2words(int(part), lang="pl"))
                    elif part.upper() in _INTL_ABBREV_MAP:
                        result_parts.append(_INTL_ABBREV_MAP[part.upper()])
                    elif part.upper() in _LETTER_NAMES_PL and len(part) == 1:
                        result_parts.append(_LETTER_NAMES_PL[part.upper()])
                    else:
                        result_parts.append(part)
            else:
                result_parts.append(seg)
    if is_model_designator:
        return " - ".join(result_parts)
    return " ".join(result_parts)


# ── PCRE do dopasowania Roman numerals (bez pojedynczego I) ─────────────────
# Dopasowuje: I, V, X, L, C, D, M w poprawnych combined forms
_ROMAN_PATTERN = re.compile(
    r"\b(?<!\w)([IVXLCDM]+)(?!\w)\b",
    re.IGNORECASE
)
# Ścisły wzorzec — tylko poprawne kanoniczne liczby rzymskie (nie MIDI, CIVIC itp.)
_ROMAN_STRICT = re.compile(
    r"^M{0,4}(?:CM|CD|D?C{0,3})(?:XC|XL|L?X{0,3})(?:IX|IV|V?I{0,3})$",
    re.IGNORECASE
)
_ROMAN_CONTEXT_WORDS = (
    "wiek", "rozdział", "akt", "tom", "część", "księga", "pieśń", "wojna",
    "wieków", "rozdziału", "akty", "tomu", "części", "księgi", "pieśni",
    "wojny", "wojnie", "wojną",
    "w.",   # skrót "wiek" (np. "XVI w.")
    # Nazwy własne wymagające numeru porządkowego
    "rzesz", "rzeczpospolit", "międzynarodówk", "sobór", "soboru",
    "dynastia", "dynastii", "armia", "armii", "front", "frontu",
    "kadr", "symfonii", "symfonia", "koncert", "koncertu",
    "gimnazjum", "liceum", "liga", "ligi",
    # Inne konteksty
    "poł.",  "połow",  # II poł. / II połowa
    "wydział", "wydzia",  # Wydział II
    "departament", "oddział", "dywizj", "brygad", "korpus",
    # Okresy i kwartały
    "kwartale", "kwartał", "kwartały", "kwartałem", "kwartału",
    "kwart",   # prefix — catches kwartale, kwartał etc.
)

# Słowa, które technicznie pasują do wzorca rzymskich liczb, ale są
# polskimi zaimkami/słowami (lub popularnymi skrótami) — NIE interpretuj jako Roman
_ROMAN_BLACKLIST = frozenset({
    "MI", "IM", "CI", "DI", "VI",          # zaimki / przyimki
    "MIX", "MIDI", "MINI", "CIVIL",        # popularne słowa łacińskie/angielskie
    "DIVI", "VICI", "MILD", "MILL", "MIMIC",
    "CLICK", "MILK", "MIND", "FILM",
})

# ── Roman numeral values ─────────────────────────────────────────────────────
_ROMAN_VALUES = {
    "I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000
}


def _roman_to_int(roman: str) -> int:
    """Konwertuje rzymską liczbę na arabską."""
    roman = roman.upper()
    result = 0
    prev = 0
    for char in reversed(roman):
        curr = _ROMAN_VALUES[char]
        if curr < prev:
            result -= curr
        else:
            result += curr
        prev = curr
    return result
import string
try:
    from num2words import num2words
except Exception as exc:  # pragma: no cover
    raise ImportError(
        "Missing dependency: num2words. Install it in your env (e.g. `pip install num2words`)."
    ) from exc

# ── Zbiory znaków przyjmowane przez PLTokenizer ───────────────────────────────

_PL_LETTERS_LOWER = set("abcdefghijklmnopqrstuvwxyząćęłńóśźż")
_PL_LETTERS_UPPER = set("ABCDEFGHIJKLMNOPQRSTUVWXYZĄ ĆĘŁŃÓŚŹŻ".replace(" ", ""))
_PL_LETTERS       = _PL_LETTERS_LOWER | _PL_LETTERS_UPPER

# Dokładnie taka lista jak w PLTokenizer.PUNCT
_TOKENIZER_PUNCT = set(list('.,!?:;-"\'()[]…/') +
                       ["@", "#", "$", "%", "&", "*",
                        "+", "=", "<", ">", "^", "_", "|", "~"])

# Wszystkie znaki dozwolone przez tokenizer (poza spacją)
_ALLOWED_CHARS = _PL_LETTERS | _TOKENIZER_PUNCT | {" "}

# ── Mapy transliteracyjne ─────────────────────────────────────────────────────

# Myślniki i kreski (wszystkie warianty Unicode → ASCII -)
_DASH_MAP = str.maketrans({
    "\u2013": "-",   # en dash  –
    "\u2014": "-",   # em dash  —
    "\u2011": "-",   # non-breaking hyphen ‑
    "\u2012": "-",   # figure dash ‒
    "\u2015": "-",   # horizontal bar ―
    "\u2212": "-",   # minus sign −
    "\uFE58": "-",   # small em dash ﹘
    "\uFE63": "-",   # small hyphen-minus ﹣
    "\uFF0D": "-",   # fullwidth hyphen-minus －
})

# Znaki niemieckie → polskie ekwiwalenty
_GERMAN_MAP = str.maketrans({
    "ö": "o", "Ö": "O",
    "ä": "a", "Ä": "A",
    "ü": "u", "Ü": "U",
})

# Cudzysłowy (wszystkie warianty) → do usunięcia
_QUOTE_CHARS = set([
    '"', '„', '\u201C', '\u201D',  # "  „  "  "
    '«', '»',                       # «  »
    '\u2018', '\u2019',             # '  '
    '\u201A', '\u201B',             # ‚  ‛
    '\u00AB', '\u00BB',             # «  »
    '\u2039', '\u203A',             # ‹  ›
])

# Spacje niestandardowe → zwykła spacja
_SPACE_MAP = str.maketrans({
    "\u00A0": " ",   # no-break space
    "\u202F": " ",   # narrow no-break space
    "\u2009": " ",   # thin space
    "\u2008": " ",   # punctuation space
    "\u2007": " ",   # figure space
    "\u2006": " ",   # six-per-em space
    "\u2005": " ",   # four-per-em space
    "\u2004": " ",   # three-per-em space
    "\u2003": " ",   # em space
    "\u2002": " ",   # en space
    "\u3000": " ",   # ideographic space
    "\t":     " ",
    "\r":     " ",
    "\n":     " ",
    "\u000B": " ",   # vertical tab
    "\u000C": " ",   # form feed
})


# ─────────────────────────────────────────────────────────────────────────────
# CELL 3 ▸ Słowniki gramatyczne (dla normalizatora liczb)
# ─────────────────────────────────────────────────────────────────────────────

_MONTHS_GEN = {
    1:"stycznia", 2:"lutego", 3:"marca", 4:"kwietnia",
    5:"maja", 6:"czerwca", 7:"lipca", 8:"sierpnia",
    9:"września", 10:"października", 11:"listopada", 12:"grudnia",
}

_ORD_GEN_DAYS = {
    1:"pierwszego", 2:"drugiego", 3:"trzeciego", 4:"czwartego",
    5:"piątego", 6:"szóstego", 7:"siódmego", 8:"ósmego",
    9:"dziewiątego", 10:"dziesiątego", 11:"jedenastego", 12:"dwunastego",
    13:"trzynastego", 14:"czternastego", 15:"piętnastego", 16:"szesnastego",
    17:"siedemnastego", 18:"osiemnastego", 19:"dziewiętnastego", 20:"dwudziestego",
    21:"dwudziestego pierwszego", 22:"dwudziestego drugiego",
    23:"dwudziestego trzeciego", 24:"dwudziestego czwartego",
    25:"dwudziestego piątego", 26:"dwudziestego szóstego",
    27:"dwudziestego siódmego", 28:"dwudziestego ósmego",
    29:"dwudziestego dziewiątego", 30:"trzydziestego",
    31:"trzydziestego pierwszego",
}

# Waluty: (mianownik sg, mianownik 2-4, dopełniacz 5+, grosze ×3)
# Waluty: (mianownik sg, mianownik 2-4, dopełniacz 5+, grosze sg, grosze 2-4, grosze 5+, rodzaj)
_CURRENCIES = {
    "zł":  ("złoty",  "złote",   "złotych",  "grosz",  "grosze",  "groszy", "m"),
    "PLN": ("złoty",  "złote",   "złotych",  "grosz",  "grosze",  "groszy", "m"),
    "EUR": ("euro",   "euro",    "euro",     "cent",   "centy",   "centów", "n"),
    "€":   ("euro",   "euro",    "euro",     "cent",   "centy",   "centów", "n"),
    "USD": ("dolar",  "dolary",  "dolarów",  "cent",   "centy",   "centów", "m"),
    "$":   ("dolar",  "dolary",  "dolarów",  "cent",   "centy",   "centów", "m"),
    "GBP": ("funt",   "funty",   "funtów",   "pens",   "pensy",   "pensów", "m"),
    "£":   ("funt",   "funty",   "funtów",   "pens",   "pensy",   "pensów", "m"),
    "CHF": ("frank szwajcarski",  "franki szwajcarskie",  "franków szwajcarskich",  "centym", "centymy", "centymów", "m"),
    "CZK": ("korona czeska", "korony czeskie",  "koron czeskich",    "halerz",  "halerze",  "halerzy", "f"),
    "¥":   ("jen",    "jeny",    "jenów",    "sen",    "seny",    "senów", "m"),
    "JPY": ("jen japoński", "jeny japońskie", "jenów japońskich", "sen", "seny", "senów", "m"),
    "BTC": ("bitcoin","bitcoiny","bitcoinów","satoshi","satoshi",  "satoshi", "m"),
}

# Jednostki: (sg, pl234, plgen, rodzaj 'm'|'f') — generated from unit_registry
_UNITS = _REGISTRY.as_units_dict()

_LARGE = {
    "tys": ("tysiąc","tysiące","tysięcy","tysiąca"),
    "mln": ("milion","miliony","milionów","miliona"),
    "mld": ("miliard","miliardy","miliardów","miliarda"),
    "bln": ("bilion","biliony","bilionów","biliona"),
}


# ─────────────────────────────────────────────────────────────────────────────
# CELL 3b ▸ Dopełniacz liczebników głównych (do/dla/bez/od/z/około + N)
# ─────────────────────────────────────────────────────────────────────────────

# Dopełniacz l. poj. / zbiorowy dla liczebników głównych
_CARD_GEN_ONES = {
    0: "zera",  1: "jednego", 2: "dwóch",  3: "trzech",  4: "czterech",
    5: "pięciu",6: "sześciu", 7: "siedmiu",8: "ośmiu",   9: "dziewięciu",
}
_CARD_GEN_TEENS = {
    10:"dziesięciu",11:"jedenastu",12:"dwunastu",13:"trzynastu",
    14:"czternastu",15:"piętnastu",16:"szesnastu",17:"siedemnastu",
    18:"osiemnastu",19:"dziewiętnastu",
}
_CARD_GEN_TENS = {
    20:"dwudziestu",30:"trzydziestu",40:"czterdziestu",
    50:"pięćdziesięciu",60:"sześćdziesięciu",70:"siedemdziesięciu",
    80:"osiemdziesięciu",90:"dziewięćdziesięciu",
}
_CARD_GEN_HUNDREDS = {
    100:"stu",200:"dwustu",300:"trzystu",400:"czterystu",
    500:"pięciuset",600:"sześciuset",700:"siedmiuset",
    800:"ośmiuset",900:"dziewięciuset",
}

def _cardinal_gen(n: int) -> str:
    """Dopełniacz liczebnika głównego (dla do/dla/bez/od/z/około + N).
    Now delegates to inflect_pl module for all cases."""
    return _cardinal_inflect(n, "gen")


# ─────────────────────────────────────────────────────────────────────────────
# CELL 4 ▸ Funkcje gramatyczne
# ─────────────────────────────────────────────────────────────────────────────

def _pick(n: int, sg: str, pl234: str, plgen: str) -> str:
    """Właściwa forma fleksyjna rzeczownika dla liczby n (reguła polska)."""
    n = abs(int(n))
    if n % 100 in range(11, 20): return plgen
    d = n % 10
    if d == 1 and n == 1:   return sg   # tylko "jeden" używa mianownika l.poj.
    if d in (2, 3, 4):      return pl234
    return plgen

def _n2w(n: int, gender: str = "m") -> str:
    w = num2words(int(n), lang="pl")
    if gender == "f":
        w = re.sub(r"\bdwa\b",  "dwie",  w)
        w = re.sub(r"\bjeden\b","jedna", w)
    elif gender == "n":
        w = re.sub(r"\bjeden\b","jedno", w)
    return w

_DIGIT_WORDS_PL = ["zero", "jeden", "dwa", "trzy", "cztery",
                    "pięć", "sześć", "siedem", "osiem", "dziewięć"]

def _masc_gen_sg(nom: str) -> str:
    """Derive masculine genitive singular from nominative for unit names.

    Handles single-word (metr→metra) and compound (metr sześcienny→metra sześciennego).
    Inflects noun + adjective (if present).
    """
    parts = nom.split(" ", 1)
    first = parts[0]
    # Inflect the noun
    if first.endswith("ień"):
        noun = first[:-3] + "nia"
    elif first.endswith("iec"):
        noun = first[:-3] + "ca"
    else:
        noun = first + "a"
    # Inflect adjective if present (kwadratowy→kwadratowego, sześcienny→sześciennego)
    if len(parts) > 1:
        adj = parts[1]
        # Already genitive (e.g. "Celsjusza", "Fahrenheita", "rtęci") — pass through
        if adj[0].isupper() or adj.endswith("a") or adj.endswith("y") == False:
            return noun + " " + adj
        # -owy → -owego, -owy → -owego
        if adj.endswith("owy"):
            return noun + " " + adj[:-1] + "ego"
        # -nny → -nnego (sześcienny → sześciennego)
        if adj.endswith("nny"):
            return noun + " " + adj[:-1] + "ego"
        # -ny → -nego
        if adj.endswith("ny"):
            return noun + " " + adj[:-1] + "ego"
        return noun + " " + adj
    return noun

def _n2w_float(raw: str, gender: str = "m") -> str:
    raw = raw.replace(",", ".")
    if "." not in raw:
        return _n2w(int(raw), gender)
    int_s, dec_s = raw.split(".", 1)
    int_w = _n2w(int(int_s or 0), gender)
    # Preserve leading zeros: "0,01" → "zero przecinek zero jeden"
    if dec_s.startswith("0"):
        dec_w = " ".join(_DIGIT_WORDS_PL[int(d)] for d in dec_s)
        return f"{int_w} przecinek {dec_w}"
    # Read decimal part as full number: 0,333 → "zero przecinek trzysta trzydzieści trzy"
    dec_w = num2words(int(dec_s), lang='pl')
    if gender == "f":
        dec_w = re.sub(r"\bdwa\b", "dwie", dec_w)
        dec_w = re.sub(r"\bjeden\b", "jedna", dec_w)
    return f"{int_w} przecinek {dec_w}"

def _n2w_float_gen(raw: str) -> str:
    """Genitive form of a decimal number: 1.2 → 'jednego przecinek dwóch'."""
    raw = raw.replace(",", ".")
    if "." not in raw:
        return _cardinal_inflect(int(raw), "gen")
    int_s, dec_s = raw.split(".", 1)
    int_w = _cardinal_inflect(int(int_s or 0), "gen")
    if dec_s.startswith("0"):
        dec_w = " ".join(_DIGIT_WORDS_PL[int(d)] for d in dec_s)
        return f"{int_w} przecinek {dec_w}"
    dec_w = _cardinal_inflect(int(dec_s), "gen")
    return f"{int_w} przecinek {dec_w}"

def _parse_raw(raw: str):
    """Parsuje surowy ciąg liczbowy (separator tysięcy i dziesiętny)."""
    clean = raw.replace("\u00a0", "").replace(" ", "")
    if "." in clean and "," in clean:
        clean = clean.replace(".", "").replace(",", ".")
    else:
        clean = clean.replace(",", ".")
    return float(clean), clean

def _ord_to_gen(word: str) -> str:
    if word.endswith("y"):  return word[:-1] + "ego"
    if word.endswith("i"):  return word[:-1] + "iego"
    return word

# Ordinals femininy dla godzin (pol. "godzina" jest rodzaju żeńskiego)
_HOUR_ORDINALS = {
    0: "zero", 1: "pierwsza", 2: "druga", 3: "trzecia",
    4: "czwarta", 5: "piąta", 6: "szósta", 7: "siódma",
    8: "ósma", 9: "dziewiąta", 10: "dziesiąta", 11: "jedenasta",
    12: "dwunasta", 13: "trzynasta", 14: "czternasta", 15: "piętnasta",
    16: "szesnasta", 17: "siedemnasta", 18: "osiemnasta", 19: "dziewiętnasta",
    20: "dwudziesta", 21: "dwudziesta pierwsza", 22: "dwudziesta druga",
    23: "dwudziesta trzecia",
}

_HOUR_ORDINALS_LOC = {
    0: "zero", 1: "pierwszej", 2: "drugiej", 3: "trzeciej",
    4: "czwartej", 5: "piątej", 6: "szóstej", 7: "siódmej",
    8: "ósmej", 9: "dziewiątej", 10: "dziesiątej", 11: "jedenastej",
    12: "dwunastej", 13: "trzynastej", 14: "czternastej", 15: "piętnastej",
    16: "szesnastej", 17: "siedemnastej", 18: "osiemnastej", 19: "dziewiętnastej",
    20: "dwudziestej", 21: "dwudziestej pierwszej", 22: "dwudziestej drugiej",
    23: "dwudziestej trzeciej",
}


def _year_gen(year: int) -> str:
    ordinal = num2words(year, lang="pl", to="ordinal")
    return " ".join(_ord_to_gen(w) for w in ordinal.split())



def _year_gen_small(year: int) -> str:
    ordinal = num2words(year, lang="pl", to="ordinal")
    return " ".join(_ord_to_gen(w) for w in ordinal.split())

def _year_loc(year: int) -> str:
    """Forma miejscownikowa roku: 'w 2024 roku' → 'w dwa tysiące dwudziestym czwartym roku'."""
    ordinal = num2words(year, lang="pl", to="ordinal")
    words = ordinal.split()
    loc_words = []
    for w in words:
        if w.endswith("y"):
            loc_words.append(w[:-1] + "ym")
        elif w.endswith("i"):
            loc_words.append(w[:-1] + "im")
        else:
            loc_words.append(w)
    return " ".join(loc_words)

def _fraction_words(clean: str, gender: str = "m") -> str:
    """Converts '2.5' → 'dwa i pół', inne ułamki → 'X przecinek Y'."""
    clean = clean.replace(",", ".")
    if "." not in clean:
        return _n2w(int(clean), gender)
    int_s, dec_s = clean.split(".", 1)
    # Specjalny przypadek: .5 → "pół"
    if dec_s == "5":
        int_n = int(int_s or "0")
        if int_n == 0:
            return "pół"
        return f"{_n2w(int_n, gender)} i pół"
    # Reszta: format "przecinek"
    return _n2w_float(clean, gender)


# ─────────────────────────────────────────────────────────────────────────────
# CELL 5 ▸ Normalizator liczb (wewnętrzna klasa)
# ─────────────────────────────────────────────────────────────────────────────

class _NumberNormalizer:
    """Wewnętrzny normalizator liczb — używany przez PolishTTSPipeline."""

    # Preposition → grammatical case mapping for Polish
    _PREP_CASE_MAP = {
        # Genitive prepositions
        "do": "gen", "dla": "gen", "bez": "gen", "od": "gen",
        "z": "gen", "ze": "gen", "powyżej": "gen", "poniżej": "gen",
        "około": "gen", "rzędu": "gen",
        # Accusative (= nominative for inanimates) prepositions
        "ponad": "nom", "na": "nom", "przez": "nom",
        # Locative prepositions
        "o": "loc", "w": "loc", "we": "loc", "po": "loc", "przy": "loc",
        # Instrumental prepositions
        "między": "inst", "pomiędzy": "inst", "przed": "inst",
        "nad": "inst", "pod": "inst",
    }

    # ── Noun-mediated case government ────────────────────────────────────
    # Polish nouns that govern grammatical case on following numbers.
    # Format: noun_form → (case, number_type)
    #   number_type: "cardinal" (inflected cardinal), "ordinal" (ordinal number)
    _NOUN_CASE_MAP = {
        # Nouns that require GENITIVE CARDINAL on following number(+unit)
        "pułap": ("gen", "cardinal"),
        "pojemność": ("gen", "cardinal"), "pojemności": ("gen", "cardinal"),
        "pojemnością": ("gen", "cardinal"),
        "boku": ("gen", "cardinal"),
        "grupy": ("gen", "cardinal"), "grono": ("gen", "cardinal"),
        "szerokość": ("gen", "cardinal"), "szerokości": ("gen", "cardinal"),
        "wysokość": ("gen", "cardinal"), "wysokości": ("gen", "cardinal"),
        "długość": ("gen", "cardinal"), "długości": ("gen", "cardinal"),
        "głębokość": ("gen", "cardinal"), "głębokości": ("gen", "cardinal"),
        "prędkość": ("gen", "cardinal"), "prędkości": ("gen", "cardinal"),
        "moc": ("gen", "cardinal"), "mocy": ("gen", "cardinal"),
        "waga": ("gen", "cardinal"), "wagą": ("gen", "cardinal"),
        "masę": ("gen", "cardinal"), "masa": ("gen", "cardinal"),
        "ciężar": ("gen", "cardinal"),
        # Nouns that require ORDINAL on following number
        # Case is inferred from the noun form (gen → gen ordinal, loc → loc ordinal)
        "toru": ("gen", "ordinal"),        # z toru czwartego (genitive)
        "torze": ("loc", "ordinal"),       # na torze czwartym (locative)
        "peronu": ("gen", "ordinal"),       # z peronu drugiego (genitive)
        "peronie": ("loc", "ordinal"),     # przy peronie drugim (locative)
        "piętrze": ("loc", "ordinal"),     # na piętrze trzecim (locative)
        "linii": ("loc", "ordinal"),       # w linii sto dwudziestej ósmej (locative)
        "linią": ("inst", "ordinal"),
        "stronie": ("loc", "ordinal"),     # na stronie dziesiątej
        "pozycji": ("loc", "ordinal"),     # na pozycji trzeciej
        "miejscu": ("loc", "ordinal"),     # na miejscu piątym
        "arkuszu": ("loc", "ordinal"),
        "rozdziale": ("loc", "ordinal"),
        "tomie": ("loc", "ordinal"),
        "numerze": ("loc", "ordinal"),
        # Additional governing nouns for number case inflection
        "taktowaniu": ("gen", "cardinal"),
        "dawce": ("gen", "cardinal"),
        "dawkę": ("gen", "cardinal"),
        "dawką": ("inst", "cardinal"),
        "kwocie": ("gen", "cardinal"),
        "kwotą": ("inst", "cardinal"),
        "czasie": ("loc", "cardinal"),
        "kątem": ("gen", "cardinal"),
        "miarę": ("gen", "cardinal"),
        "potrzeba": ("gen", "cardinal"),
        "wietrze": ("gen", "cardinal"),
        "dystansie": ("gen", "cardinal"),
        "poziomie": ("gen", "cardinal"),
        "temperaturze": ("gen", "cardinal"),
        "temperatury": ("gen", "cardinal"),
        "stężeniu": ("gen", "cardinal"),
        "napięciem": ("gen", "cardinal"),
        # Additional units context
        "natężeniu": ("gen", "cardinal"),
        "matrycy": ("gen", "cardinal"),
        "przekroju": ("gen", "cardinal"),
        "rozmiarze": ("loc", "cardinal"),
        "kubaturę": ("gen", "cardinal"),
        "powierzchni": ("gen", "cardinal"),
        "próby": ("gen", "cardinal"),
        "prędkością": ("gen", "cardinal"),
        "prędkości": ("gen", "cardinal"),
        "poziom": ("gen", "cardinal"),
        "poziomie": ("gen", "cardinal"),
        # Phase 2A additions — missing noun case governors
        "szybkością": ("gen", "cardinal"),   # z szybkością 35 stron
        "przekątnej": ("gen", "cardinal"),   # o przekątnej 65 cali
        "numerem": ("inst", "ordinal"),      # z numerem 15. → piętnastym
        "odległości": ("gen", "cardinal"),   # w odległości 4 mld
        "zapłacie": ("gen", "cardinal"),     # po zapłacie 150 €
        "mierze": ("gen", "cardinal"),       # o mierze 90° (overrides default)
        "koszt": ("gen", "cardinal"),        # koszt 1 kWh
        "kosztem": ("inst", "cardinal"),
        "kosztów": ("gen", "cardinal"),
        "rzędu": ("gen", "cardinal"),        # rzędu 2 500 000
        "kwarcie": ("gen", "ordinal"),       # w 1. kwarcie → pierwszej (feminine)
        "lidze": ("gen", "ordinal"),         # w 1. lidze → pierwszej (feminine)
    }

    # ── Spelled-out unit names (not abbreviations) ───────────────────────
    # Units that appear as full words after numbers: "256 bit", "5 procent"
    _SPELLED_UNITS = {
        "bit":    ("bit", "bity", "bitów", "m"),
        "bitów":  ("bit", "bity", "bitów", "m"),
        "bajt":   ("bajt", "bajty", "bajtów", "m"),
        "bajtów": ("bajt", "bajty", "bajtów", "m"),
        "piksel":  ("piksel", "piksele", "pikseli", "m"),
        "pikseli": ("piksel", "piksele", "pikseli", "m"),
        "mikrofarad": ("mikrofarad", "mikrofarady", "mikrofaradów", "m"),
        "mikrogram": ("mikrogram", "mikrogramy", "mikrogramów", "m"),
    }

    # Unit inflection: gen, locative, and instrumental forms for ALL units
    # Generated from unit_registry (complete coverage, not just ~20 manual entries)
    # Key: nominative singular → {case: (sg, pl234, pl5+)}
    _UNIT_CASE_FORMS = _REGISTRY.as_case_dict()
    # "procent" is not a unit abbreviation but needs case forms for percent handlers
    _UNIT_CASE_FORMS["procent"] = {
        "gen":  ("procentu", "procentów", "procentów"),
        "loc":  ("procencie", "procentach", "procentach"),
        "inst": ("procentem", "procentami", "procentami"),
    }

    def __init__(self):
        # Wzorce liczbowe: akceptują liczby z separatorem tysięcy (NBSP/spacja)
        # lub zwykłe liczby całkowite (dowolna liczba cyfr)
        _NUM     = r"\d+(?:[\u00a0 ]\d{3})*(?:[,.]\d+)?"
        _NUM_INT = r"\d+(?:[\u00a0 ]\d{3})*"
        # Rozszerzona wersja dla walut — akceptuje też liczby 4+ bez separatora
        _NUM_INT_CURR = r"\d+(?:[\u00a0 ]\d{3})*"
        _SYM_A   = r"(zł|PLN|EUR|USD|GBP|CHF|CZK|JPY|BTC)"
        _SYM_B   = r"([€\$£¥])"
        _UK      = "|".join(re.escape(k) for k in sorted(_UNITS, key=len, reverse=True))

        self._pats = [
            # ── GROUP A: Network/Phone/Version ─────────────────────────────
            # Must be FIRST — consume dots/colons before dates/times/numbers.

            # Version numbers: X.Y.Z or X.Y after "wersj" context → "X kropka Y kropka Z"
            (re.compile(r"\b(wersj[iaęąom]+)\s+(\d+)\.(\d+)\.(\d+)\b", re.IGNORECASE),
             self._version_3),
            (re.compile(r"\b(wersj[iaęąom]+)\s+(\d+)\.(\d+)\b", re.IGNORECASE),
             self._version_2),
            # Adres IP: 192.168.1.1 — przed datami i liczbami
            (re.compile(r"\b(\d{1,3})\.(\d{1,3})\.(\d{1,3})\.(\d{1,3})\b"),
             self._ip_address),
            # Numer telefonu z kodem kraju: +48 123 456 789
            (re.compile(r"\+(\d{2})\s+(\d{3})\s+(\d{3})\s+(\d{3})\b"),
             self._phone_intl),
            # ── GROUP A2: Triple-dash notation (N-N-N where N ≤ 3 digits) ─
            # Must be BEFORE dates — otherwise 10-10-10 is parsed as DD-MM-YY.
            # Negative lookahead: skip if last group is 4 digits (that's a date).
            (re.compile(r"\b(\d{1,3})-(\d{1,3})-(\d{1,3})\b(?!\d)"),
             self._triple_dash),
            # ── GROUP B: Dates ─────────────────────────────────────────
            # Must be before standalone numbers (dots/slashes are ambiguous).

            # Daty z miesiącem rzymskim: 12.XI.1473, 5.IV.2024
            (re.compile(r"\b([012]?\d|3[01])\.(I{1,3}|IV|VI{0,3}|IX|XI{0,2}|XII?)\.(\d{4})\b"),
             self._date_roman_month),
            (re.compile(r"\b([012]?\d|3[01])[./\-]([01]?\d)[./\-](\d{4})(?:\s+r\.)?"),
             self._date_full),
            (re.compile(r"\b([012]?\d|3[01])[./\-]([01]?\d)[./\-](\d{2})\b"),
             self._date_short),
            # ── GROUP C: Scales/Ratios/Times ───────────────────────────
            # Colon-delimited patterns — must be before general numbers.
            # Time patterns must be after scales (colon ambiguity).
            # Scores must be after times.

            # Scale/ratio with context: "w skali 1:10", "skala 1:50 000", "proporcji 2:1"
            (re.compile(r"\b(skal[iaey]|proporcj[iaei]|stosunk[iu])\s+(\d+)\s*:\s*(\d[\d\s]*\d|\d)\b", re.IGNORECASE),
             self._scale_context),
            # Large scale without context: 1:50 000 (second number has spaces/NBSP between digit groups)
            (re.compile(r"\b(\d+)\s*:\s*(\d{1,3}(?:[\s\u00a0]\d{3})+)\b"),
             self._scale_large),
            # Time range with godz prefix: godz. 08:00-22:00 → w godzinach ósma - dwudziesta druga
            (re.compile(r"\b(w?\s*godz\.?|w?\s*godzin(?:ach|y|a(?:ch)?)?)\s+([01]?\d|2[0-3])[.:]([0-5]\d)\s*[-–—]\s*([01]?\d|2[0-3])[.:]([0-5]\d)\b", re.IGNORECASE),
             self._time_range_godz),
            # "O godzina/godzinie 14:30" — colon time after godz prefix (case-aware)
            (re.compile(r"\b(o\s+godz\.?|o\s+godzina|o\s+godzinie|o\s+godziną|godz\.?|godzina|godzinie|godziną)\s+([01]?\d|2[0-3]):([0-5]\d)(?::([0-5]\d))?\b", re.IGNORECASE),
             self._time_prefixed_dot),  # reuse same handler — works for both colon & dot
            # Preposition + colon time: "o 14:35", "od 6:00", "do 22:30", "około 08:00"
            (re.compile(r"\b(o|od|do|około|przed|po|między|na)\s+([01]?\d|2[0-3]):([0-5]\d)(?::([0-5]\d))?\b", re.IGNORECASE),
             self._time_prefixed_colon),
            (re.compile(r"\b([01]?\d|2[0-3]):([0-5]\d)(?::([0-5]\d))?\b"),
             self._time),
            # Zapis godzin z kropką tylko w bezpiecznym kontekście narracyjnym.
            # Przykłady: "o 6.45", "między 18.30 a 19.00", "Wtorek, 7 grudnia, 6.45."
            (re.compile(r"\b(o\s+godz\.?|o\s+godzina|o\s+godzinie|o\s+godziną|godz\.?|godzina|godzinie|godziną)\s+([01]?\d|2[0-3])\.([0-5]\d)(?::([0-5]\d))?\b", re.IGNORECASE),
             self._time_prefixed_dot),
            (re.compile(r"\b(o|od|do|a|między|około|przed|po)\s+([01]?\d|2[0-3])\.([0-5]\d)(?::([0-5]\d))?\b", re.IGNORECASE),
             self._time_prefixed_dot),
            (re.compile(r"\b([01]?\d|2[0-3])\.([0-5]\d)(?::([0-5]\d))?\s+(rano|wieczorem|w nocy|po południu|nad ranem)\b", re.IGNORECASE),
             lambda m: f"{self._time_words(int(m.group(1)), int(m.group(2)), int(m.group(3)) if m.group(3) else None)} {m.group(4)}"),
            (re.compile(r"([,;]\s+)([01]?\d|2[0-3])\.([0-5]\d)(?::([0-5]\d))?(?=($|[)\].,;!?]))"),
             self._time_after_sep_dot),
            # Duration after "w": w 45:30 → w czterdzieści pięć minut trzydzieści sekund
            # Only match when first number is 24+ (can't be a valid hour, so it's a duration)
            (re.compile(r"\bw\s+((?:2[4-9]|[3-9]\d|\d{3})):([0-5]\d)\b", re.IGNORECASE), self._duration_ms),
            # Score: 3:0, 2:1, 25:23 — sport scores (AFTER all time patterns)
            # Optionally capture "wynikiem" before score for 0:0 → "remisem" substitution
            (re.compile(r"\b(?:(wynikiem)\s+)?(\d{1,2}):(\d{1,2})\b(?!\s*:)(?=\s|$|[,;.!?)\]])"), self._score),
            # General ratio: 1:10000 → jeden do dziesięciu tysięcy (AFTER scores)
            (re.compile(r"\b(\d+):(\d+)\b(?!\s*:)"), self._ratio),
            # ── GROUP D: Percentages ──────────────────────────────────

            # Percentage-as-adjective: "5% wzrost" → "pięcioprocentowy wzrost"
            # Only whole numbers (no decimals) — negative lookbehind for comma/dot+digit
            (re.compile(r"(?<![,.\d])(\d+)\s*%\s*-?\s*(wzrost[aueouęą]?|wzroście|wzrostem|spadek|spadk[auięąo]|spadkiem|udział[aueęąom]?|podatek|podatk[auięąo]|rabat|obniżce|obniżc[eę]|obniżk[aęiąou]|zniżce|zniżc[eę]|zniżk[aęiąou]|zwyżce|zwyżc[eę]|zwyżk[aęiąou]|stawce|stawk[aęiąou]|oprocentowani[euaom])\b",
                        re.IGNORECASE),
             self._percent_adjective),

            (re.compile(r"([+-])(\d+(?:[,.]\d+)?)\s*%"),
             self._signed_percent),
            # Preposition + percentage: "od 7,5% do 10%" → genitive
            (re.compile(r"\b(od|do|dla|bez|z|ze|w|we|ponad|powyżej|poniżej|około|rzędu)\s+(\d+(?:[,.]\d+)?)\s*%", re.IGNORECASE),
             self._prep_percent),
            (re.compile(r"(\d+(?:[,.]\d+)?)\s*%"),
             self._percent),
            # ── GROUP E: Years with prepositions ──────────────────────
            # Must be before currency and general number patterns.

            # Rok z przyimkiem 'w/W' → forma miejscownikowa
            (re.compile(r"\b([wW])\s+(\d{4})\s+r\."),
             self._year_abbr_loc),
            # Rok po przyimku 'w/W' z kropką przed słowem "roku": W 1989. roku
            (re.compile(r"\b([wW])\s+(\d{4})\.\s+roku\b"),
             self._year_w_loc_dotted_roku),
            # Rok po przyimku 'w/W' z kropką bez słowa "roku": W 1989. organizował
            (re.compile(r"\b([wW])\s+(\d{4})\.(?!\d)"),
             self._year_w_loc_dotted),
            # Miesiąc + rok bez słowa 'roku': w grudniu 2014
            (re.compile(r"\b(w|we)\s+(styczniu|lutym|marcu|kwietniu|maju|czerwcu|lipcu|sierpniu|wrześniu|październiku|listopadzie|grudniu)\s+(\d{4})(?:\s+(?:r\.|roku))?\b", re.IGNORECASE),
             self._month_year_loc),
            # Rok po przyimku 'w/W' bez 'r.' → forma miejscownikowa (bez "roku")
            # Obsługuje też krótkie lata (496, 19 n.e.) — wymaga "roku" lub "r." po, albo 4 cyfry
            (re.compile(r"\b([wW])\s+(\d{2,3})\s+roku\b"),
             self._year_w_loc_short_roku),
            (re.compile(r"\b([wW])\s+(\d{4})\b"),
             self._year_w_loc),
            (re.compile(r"\b(\d{4})\s+r\."),
             self._year_abbr),
            # ── GROUP F: Currency ──────────────────────────────────────
            # Currency with cents, large currency (mln/mld), prefix ($), suffix (zł).

            (re.compile(rf"({_NUM_INT})[,.](\d{{2}})\s*{_SYM_A}\b"),
             self._curr_cents),
            # Preposition + large currency: do 10 mld zł → genitive
            (re.compile(rf"\b(do|dla|bez|od|z|ze|ponad|powyżej|poniżej)\s+({_NUM})\s*(mln|mld|bln|tys)\.?\s*{_SYM_A}\b", re.IGNORECASE),
             self._prep_large_curr),
            (re.compile(rf"\b({_NUM})\s*(?:mln|mld|bln|tys\.?)\s*(?:{_SYM_A}\b|{_SYM_B})"),
             self._large_curr),
            (re.compile(rf"\b({_NUM_INT})\s*[-]\s*({_NUM_INT})\s*(mln|mld|bln|tys)(\.?)(?!\s*(?:zł|PLN|EUR|USD|GBP|CHF|CZK)\b)(?=\s|$|[,;:!?)])"),
             self._range_large),
            # Large + unit: 9,2 mln ha → dziewięć przecinek dwa miliona hektarów
            (re.compile(rf"\b({_NUM})\s*(mln|mld|bln|tys)\.?\s+({_UK})\b"),
             self._large_unit),
            (re.compile(rf"\b({_NUM})\s*(mln|mld|bln|tys)(\.?)(?!\s*(?:zł|PLN|EUR|USD|GBP|CHF|CZK)\b)(?=\s|$|[,;:!?)])"),
             self._large),
            # Waluta z prefiksem + k/M suffix: $52k, $12.5M
            (re.compile(rf"{_SYM_B}\s*(\d+(?:[,.]\d+)?)\s*([kKmM])\b"),
             self._curr_prefix_suffix),
            # Minus + waluta: -$2 770, +$268
            (re.compile(rf"([+-]){_SYM_B}\s*({_NUM})\b"),
             self._signed_curr_prefix),
            (re.compile(rf"{_SYM_B}\s*({_NUM})\b"),
             self._curr_prefix),
            # Liczba + k/M bez waluty: 103k, 12.5k
            (re.compile(r"\b(\d+(?:[,.]\d+)?)\s*([kK])\b"),
             self._num_k_suffix),
            # Preposition + number + currency: do 4242 zł → genitive
            (re.compile(rf"\b(do|dla|bez|od|z|ze|ponad|powyżej|poniżej|około|w|we|o)\s+({_NUM_INT_CURR})\s*{_SYM_A}\b", re.IGNORECASE),
             self._prep_curr),
            # Preposition + number + postfix symbol currency: około 12 500 £ → genitive
            (re.compile(rf"\b(do|dla|bez|od|z|ze|ponad|powyżej|poniżej|około|w|we|o)\s+({_NUM_INT_CURR})\s*{_SYM_B}", re.IGNORECASE),
             self._prep_curr),
            # Noun-governed number + currency: kwocie 10 000 PLN → genitive
            (re.compile(rf"\b(\w+)\s+({_NUM_INT_CURR})\s*{_SYM_A}\b"),
             self._noun_num_curr),
            # Noun-governed number + postfix symbol currency: poziom 2000 $ → genitive
            (re.compile(rf"\b(\w+)\s+({_NUM_INT_CURR})\s*{_SYM_B}"),
             self._noun_num_curr),
            (re.compile(rf"\b({_NUM_INT_CURR})\s*{_SYM_A}\b"),
             self._curr),
            # ── GROUP G: Units ─────────────────────────────────────────
            # Unit patterns: density, range+unit, signed+unit, prep+unit, plain unit.
            # Order: density → range → pos → prep_neg → neg → prep → plain.
            # _prep_neg_unit MUST be before _neg_unit (prefix match).

            # Gęstość: 22 osób/km² → dwadzieścia dwa osób na kilometr kwadratowy
            (re.compile(r"\b(\d+)\s+(\w+)/km[²2]\b"),
             self._per_km2),
            # ⑪b Zakres + jednostka:  30-50 km/h  →  trzydzieści do pięćdziesięciu km/h
            (re.compile(rf"\b(\d+)\s*[-]\s*(\d+)\s*({_UK})\b"),
             self._range_unit),
            # ⑪a+ Dodatnia liczba z plusem + jednostka: +5°C → plus pięć stopni Celsjusza
            (re.compile(rf"(?<!\w)\+\s*(\d+(?:[,.]\d+)?)\s*({_UK})\b"),
             self._pos_unit),
            # ⑪e1 Przyimek + ujemna liczba + jednostka: do -8°C → do minus ośmiu stopni Celsjusza
            # MUST be before standalone _neg_unit
            (re.compile(rf"\b(do|dla|bez|od|ponad|powyżej|poniżej)\s+-\s*(\d+(?:\u00a0\d{{3}})*(?:[,.]\d+)?)\s*({_UK})\b", re.IGNORECASE),
             self._prep_neg_unit),
            # ⑪a Ujemna liczba + jednostka:  -5°C  →  minus pięć stopni Celsjusza
            (re.compile(rf"(?<!\d)-\s*(\d+(?:[,.]\d+)?)\s*({_UK})\b"),
             self._neg_unit),
            # ⑪c0 Zakres lat z przyimkiem: od 1922 do 1940 [roku] → ordinal genitive
            (re.compile(r"\b(od)\s+(\d{4})\s+(do)\s+(\d{4})(?:\s+roku)?\b", re.IGNORECASE),
             self._prep_year_range_od_do),
            # ⑪c-1 "w wieku N lat" → genitive: w wieku pięćdziesięciu dwóch lat
            (re.compile(r"\b(w\s+wieku)\s+(\d+)\s+(lat)\b", re.IGNORECASE),
             self._prep_wieku_lat),
            # Duration with preposition: około 2h 30m → około dwie godziny i trzydzieści minut
            # MUST be before _prep_unit to prevent "około 2 h" from matching as unit
            (re.compile(r"\b(około|ok\.|do|od|na|przez)\s+(\d+)\s*h\s+(?:i\s+)?(\d+)\s*m\b", re.IGNORECASE),
             self._prep_duration_hm),
            # ⑪c Przyimek + liczba + jednostka:  do 23 kg  →  do dwudziestu trzech kilogramów
            # Czas trwania z ułamkiem: na 2,5 roku → na dwa i pół roku
            (re.compile(r"\b(na|przez|do|dla|bez|od|ponad|powyżej|poniżej|z)\s+(\d+(?:[,.]\d+))\s+(roku|lata|lat)\b", re.IGNORECASE),
             self._prep_year_duration),
            (re.compile(rf"\b(do|dla|bez|od|powyżej|poniżej|z|ze|o|w|we|po|przy|na|między|przed|nad|pod|ponad|około|rzędu)\s+(\d+(?:\u00a0\d{{3}})*(?:[,.]\d+)?)\s*({_UK})\b", re.IGNORECASE),
             self._prep_unit),
            # ── GROUP H: Preposition + number (no unit) ─────────────────
            # Plain prep+number patterns. Must be AFTER prep+unit patterns.

            # Preposition + fraction: do 1/3 → do jednej trzeciej (genitive)
            # MUST be before _prep_num to prevent "do 1" from being consumed separately
            (re.compile(r"\b(w|we|z|ze|do|od|bez|dla|na|po|przy|o)\s+(\d+)/(\d+)\b", re.IGNORECASE),
             self._prep_fraction),

            # ⑪e Przyimek + ujemna liczba: do -15 → do minus piętnastu
            (re.compile(r"\b(do|dla|bez|od|ponad|powyżej|poniżej)\s+-\s*(\d+(?:\u00a0\d{3})*)\b", re.IGNORECASE),
             self._prep_neg_num),
            # ⑪d0 Przyimek + rok (4 cyfry w zakresie 1000-2100): od 1971 → ordinal genitive
            (re.compile(r"\b(od|do|z|ze)\s+(\d{4})\b(?!\s*[-–—]\s*\d)", re.IGNORECASE),
             self._prep_year_standalone),
            # ⑪d-inst Instrumental prepositions + number: z 5 nowymi → z pięcioma
            # "z/ze" is ambiguous: instrumental (with) vs genitive (from).
            # Heuristic: if followed by a noun in instrumental (-ami/-oma/-ymi/-imi/-ą), use instrumental.
            (re.compile(r"\b(z|ze|między|pomiędzy|przed|nad|pod)\s+(\d+(?:\u00a0\d{3})*)\b(?=\s+\w+(?:ami|oma|ymi|imi|ową|owym|owymi|ą)\b)", re.IGNORECASE),
             self._prep_num_inst),
            # ⑪d-dec Przyimek + liczba dziesiętna (np. "około 3,14")
            (re.compile(r"\b(do|dla|bez|od|z|ze|ponad|powyżej|poniżej|około|rzędu)\s+(\d+(?:\u00a0\d{3})*[,.]\d+)\b", re.IGNORECASE),
             self._prep_num_decimal),
            # ⑪d Przyimek + sama liczba całkowita (w tym tysiące z NBSP)
            # Negative lookahead: nie łap "od 1919 roku" (to rok, nie ilość) ani dat
            # Also skip compound adjectives: "z 50-osobowej" — let _compound_adjective handle them
            (re.compile(r"\b(do|dla|bez|od|z|ze|ponad|powyżej|poniżej|około|rzędu)\s+(\d+(?:\u00a0\d{3})*)\b(?!\s*-\s*(?:osobow|metrow|procentow|karatow|godzinn|minutow|kilogramow|litrow|tonow|stopniow|wieczn|leci[aeu]|letn))(?!\s+(?:roku|r\.|stycznia|lutego|marca|kwietnia|maja|czerwca|lipca|sierpnia|września|października|listopada|grudnia)\b)", re.IGNORECASE),
             self._prep_num),
            # Stopnie bez jednostki temperatury: 90° → dziewięćdziesiąt stopni
            (re.compile(r"\b(\d+(?:[,.]\d+)?)\s*°(?![CF])"),
             self._degree_standalone),
            # Czas trwania: 2h i 15m, 1h 30m → dwie godziny i piętnaście minut
            (re.compile(r"\b(\d+)\s*h\s+(?:i\s+)?(\d+)\s*m\b"),
             self._duration_hm),
            # Blood pressure: 120/80 mmHg → sto dwadzieścia na osiemdziesiąt milimetrów słupa rtęci
            (re.compile(r"\b(\d+)/(\d+)\s*(mmHg)\b"),
             self._blood_pressure),
            # Fuel consumption: 6,5 l/100 km → sześć i pół litra na sto kilometrów
            (re.compile(rf"\b({_NUM})\s*l/100\s*km\b"),
             self._fuel_consumption),
            # Rate units: uderzenia/min, obr./min.
            (re.compile(r"\b(\w+)/min\.?\b"),
             self._per_minute),
            # Tire dimensions: 225/45 R17 → dwieście dwadzieścia pięć na czterdzieści pięć er siedemnaście
            (re.compile(r"\b(\d{3})/(\d{2})\s*[Rr](\d{2})\b"),
             self._tire_dimensions),
            # Distance decomposition: 42,195 km → czterdzieści dwa kilometry sto dziewięćdziesiąt pięć metrów
            (re.compile(r"\b(\d+),(\d{3})\s*km\b"),
             self._distance_decompose_km),
            # Decimal range + unit: od 1,5 do 2 h → od półtora do dwóch godzin
            (re.compile(rf"\b(od)\s+(\d+[,.]\d+)\s+(do)\s+(\d+)\s*({_UK})\b", re.IGNORECASE),
             self._range_decimal_unit),
            # Dimensions: 10 x 15 x 20 cm (must be before _unit to prevent partial match)
            (re.compile(rf"\b(\d+)\s*[xX×]\s*(\d+)(?:\s*[xX×]\s*(\d+))?\s*({_UK})\b"),
             self._resolution_with_unit),
            # Noun-governed number + unit: pojemność 32 GB → pojemność trzydziestu dwóch gigabajtów
            (re.compile(rf"\b(\w+)\s+(\d+(?:\u00a0\d{{3}})*(?:[,.]\d+)?)\s*({_UK})\b"),
             self._noun_num_unit),
            # Noun-governed number + spelled-out unit: 256 bit → dwieście pięćdziesiąt sześć bitów
            (re.compile(r"\b(\d+(?:\u00a0\d{3})*)\s+(bit|bitów|bity|bajt|bajtów|bajty|mikrofarad|mikrofarady|mikrofaradów|mikrogram|mikrogramy|mikrogramów)\b"),
             self._num_spelled_unit),
            (re.compile(rf"\b({_NUM})\s*({_UK})\b"),
             self._unit),
            # ── GROUP I: Ordinals, decades, date-with-month, ranges ───
            # Positional numbers, historical periods, calendar dates.

            # Cyfra + mała litera (odniesienia prawne): 49b → czterdzieści dziewięć be
            # Ograniczone do 1-2 liter żeby nie łapać "32bitowy" itp.
            (re.compile(r"\b(\d+)([a-ząćęłńóśźż]{1,2})\b(?![\wąćęłńóśźż])"),
             self._digit_lower_suffix),
            # N-wieczny compound: 18-wiecznym → osiemnastowiecznym
            (re.compile(r"\b(\d+)\s*-\s*(wieczn[yaeoęąi]|wiecznym|wiecznej|wiecznego|wiecznych|wieczni)\b"),
             self._century_compound),
            # Decimal compound half: 2,5-letni → dwuipółletni (BEFORE integer N-lecia)
            (re.compile(r"\b(\d+)[,.]5\s*-\s*(letni[aeoąę]?|letniej|letniego|letnich|letnim|letnimi)\b"),
             self._decimal_compound_half),
            # N-lecia/N-lecie: 40-lecia → czterdziestolecia (PRZED ordinalem!)
            (re.compile(r"\b(\d+)\s*-\s*(leci[aeu]|leciu|letni[aeoąę]?|letniej|letniego|letnich|letnim|letnimi)\b"),
             self._anniversary),
            # N-osobowy/N-metrowy/N-procentowy compound adjectives:
            # 50-osobowej → pięćdziesięcioosobowej, 20-procentowy → dwudziestoprocentowy
            (re.compile(r"\b(\d+)\s*-\s*(osobow[yaeoęąi]|osobowej|osobowego|osobowych|osobowym|osobowymi"
                        r"|metrow[yaeoęąi]|metrowej|metrowego|metrowych|metrowym"
                        r"|procentow[yaeoęąi]|procentowej|procentowego|procentowych|procentowym"
                        r"|karatow[yaeoęąi]|karatowej|karatowego|karatowych|karatowym"
                        r"|godzinn[yaeoęąi]|godzinnej|godzinnego|godzinnych|godzinnym"
                        r"|minutow[yaeoęąi]|minutowej|minutowego|minutowych|minutowym"
                        r"|kilogramow[yaeoęąi]|kilogramowej|kilogramowego|kilogramowych"
                        r"|litrow[yaeoęąi]|litrowej|litrowego|litrowych|litrowym"
                        r"|tonow[yaeoęąi]|tonowej|tonowego|tonowych|tonowym"
                        r"|stopniow[yaeoęąi]|stopniowej|stopniowego|stopniowych|stopniowym)\b"),
             self._compound_adjective),
            # Dekady z koniunkcją: "W latach 50. i 60." → oba w dopełniaczu l.mn.
            (re.compile(r"\b((?:w\s+)?lat(?:a(?:ch)?)?)\s+(\d{2})\.?\s+(i|lub|oraz|czy)\s+(\d{2})\.?(?!\d)", re.IGNORECASE),
             self._decade_conjunction),
            # Dekady: "W latach 80." / "lat 70." → (PRZED ordinalem, bo "70." → nie ordinal!)
            (re.compile(r"\b((?:w\s+)?lat(?:a(?:ch)?)?)\s+(\d{2})\.?(?!\d)", re.IGNORECASE),
             self._decade),
            # "W/Na N. słowo" → ordinal locative (W drugim rozdziale, na 3. piętrze)
            (re.compile(r"\b([Ww]|[Nn]a)\s+(\d{1,3})\.\s+(?=[^\d\s])"),
             self._ordinal_loc_prep),
            (re.compile(r"\b(\d{1,3})\.\s+(?=[^\d\s])"),
             self._ordinal),
            # Conjunction of days with month: 12 i 13 kwietnia → dwunastego i trzynastego kwietnia
            (re.compile(r"\b([12]?\d|3[01])\s+(i|oraz|lub|czy)\s+([12]?\d|3[01])\s+(stycznia|lutego|marca|kwietnia|maja|czerwca|lipca|sierpnia|września|października|listopada|grudnia)\b"),
             self._day_conj_month),
            # Zakres dni z nazwą miesiąca: 10-19 sierpnia [2007] → dziesiątego do dziewiętnastego
            (re.compile(r"\b([12]?\d|3[01])\s*[-]\s*([12]?\d|3[01])\s+(stycznia|lutego|marca|kwietnia|maja|czerwca|lipca|sierpnia|września|października|listopada|grudnia)(?:\s+(\d{4}))?\b"),
             self._day_range_month),
            # Pełna data: 20 listopada 2017 roku → dwudziestego listopada dwa tysiące siedemnastego roku
            (re.compile(r"\b([12]?\d|3[01])\s+(stycznia|lutego|marca|kwietnia|maja|czerwca|lipca|sierpnia|września|października|listopada|grudnia)\s+(\d{4})(?:\s+roku)?\b"),
             self._day_month_year),
            # Dzień z nazwą miesiąca: 24 grudnia → dwudziestego czwartego grudnia
            # Also matches capitalized: "3 Maja" (street names)
            (re.compile(r"\b([12]?\d|3[01])\s+(stycznia|lutego|marca|kwietnia|maja|czerwca|lipca|sierpnia|września|października|listopada|grudnia)\b", re.IGNORECASE),
             self._day_month),
            # Miesiąc (dopełniacz) + rok bez dnia: września 1921 → genitive year
            (re.compile(r"\b(stycznia|lutego|marca|kwietnia|maja|czerwca|lipca|sierpnia|września|października|listopada|grudnia)\s+(\d{3,4})(?:\s+roku)?\b"),
             self._month_bare_year),
            # Pora roku + rok: Latem 1937 → genitive year
            (re.compile(r"\b(latem|jesienią|wiosną|zimą)\s+(\d{3,4})\b", re.IGNORECASE),
             self._season_year),
            # "W latach/Lata/lat/z lat 1918–1939" — zakres lat z prefiksem
            (re.compile(r"\b((?:w\s+)?lat(?:a(?:ch)?)?|z\s+lat)\s+(\d{4})\s*[-]\s*(\d{4})\b", re.IGNORECASE),
             self._year_range_gen),
            # Zakresy lat w nawiasach: (1923-1993) → genitive ordinal
            (re.compile(r"\((\d{4})\s*[-–—]\s*(\d{4})\)"),
             self._year_range_paren),
            # Zakres lat: 1975-1998 → ordinal nominative
            (re.compile(r"\b(\d{4})\s*[-]\s*(\d{4})\b"),
             self._year_range),
            # N i 1/2 → N i pół
            (re.compile(r"\b(\d+)\s+i\s+1/2\b"), self._compound_half),
            # Score with "pkt/punkt": 98/100 pkt → dziewięćdziesiąt osiem na sto punktów
            (re.compile(r"\b(\d+)/(\d+)\s+(?:pkt\.?|punkt[aówyeó]*)\b"),
             self._score_pkt),
            # Fraction-inches: 3/4" → trzy czwarte cala
            (re.compile(r'\b(\d+)/(\d+)\s*"'),
             self._fraction_inch),
            # Equal ratio: 50/50 → pięćdziesiąt do pięćdziesięciu
            (re.compile(r"\b(\d+)/(\d+)\b(?=\s|$|[,;.!?)\]])", re.IGNORECASE),
             self._equal_ratio),
            # Ułamki zwykłe: 1/2 → pół, 3/4 → trzy czwarte
            (re.compile(r"\b(\d+)/(\d+)\b"),
             self._fraction),
            # Kod pocztowy: 00-123 → zero zero sto dwadzieścia trzy
            (re.compile(r"\b(\d{2})-(\d{3})\b"),
             self._postal_code),
            # Ocena szkolna z minusem: 5- → pięć minus
            (re.compile(r"\b([1-6])\s*-(?=\s|$|[,;.!?)])"),
             self._grade_minus),
            # Triple-dash notation (NPK, etc.): 10-20-30 → dziesięć dwadzieścia trzydzieści
            (re.compile(r"\b(\d+)-(\d+)-(\d+)\b"),
             self._triple_dash),
            (re.compile(r"\b(\d+)\s*[-]\s*(\d+)\b"),
             self._range),
            (re.compile(r"\b(\d{1,3}(?:[\u00a0 ]\d{3})+)\b"),
             self._thousands),
            # ⑮a Ujemna sama liczba: -15 → minus piętnaście
            (re.compile(r"(?<![\w°])-(\d+(?:[,.]\d+)?)\b"),
             self._neg_number),
            # ⑮b Dodatnia sama liczba: +49 → plus czterdzieści dziewięć
            (re.compile(r"(?<!\w)\+\s*(\d+(?:[,.]\d+)?)\b"),
             self._pos_number),
            # "Rok 1989" → ordinal nominative (tysiąc dziewięćset osiemdziesiąty dziewiąty)
            (re.compile(r"\b(rok|roku)\s+(\d{3,4})\b", re.IGNORECASE),
             self._year_after_rok),
            # Rok przed słowem "roku": 1982 roku → tysiąc dziewięćset osiemdziesiątego drugiego roku
            (re.compile(r"\b(\d{4})\s+roku\b", re.IGNORECASE),
             self._year_before_roku),
            # Skrótowy rok przed "roku": 82 roku → osiemdziesiątego drugiego roku
            (re.compile(r"\b(\d{2})\s+roku\b", re.IGNORECASE),
             self._year2_before_roku),
            # Godzina bez minut w naturalnym kontekście: około godziny 17 / o godzinie 17
            (re.compile(r"\b(o\s+godzinie|około\s+godziny)\s+([01]?\d|2[0-3])\b", re.IGNORECASE),
             self._hour_context_number),
            # Postfix currency: 100$ → sto dolarów, 50£ → pięćdziesiąt funtów
            (re.compile(r"\b(\d+(?:[,.]\d+)?)\s*(\$|£|€)"),
             self._curr_postfix),
            # Standalone £/€ after number with space: 50 £ → pięćdziesiąt funtów
            (re.compile(r"\b(\d+(?:[,.]\d+)?)\s+(£|€)\b"),
             self._curr_postfix),
            # Resolution: 1920x1080 → tysiąc dziewięćset dwadzieścia na tysiąc osiemdziesiąt
            (re.compile(r"\b(\d+)\s*[xX×]\s*(\d+)(?:\s*[xX×]\s*(\d+))?\b"),
             self._resolution),
            # (scales handled by _scale_context and _scale_large above, before time patterns)
            # Inches: 55" → pięćdziesiąt pięć cali, 2" → dwa cale
            (re.compile(r'\b(\d+(?:[,.]\d+)?)\s*"'),
             self._inches),
            # Drive letter: C: → ce (single uppercase letter + colon not followed by digit)
            (re.compile(r"\b([A-Z]):\s*(?=[/\\]|$|\s|[.])", re.IGNORECASE),
             self._drive_letter),
            # DD.MM date (no year) in context: na 15.04 o ... → na piętnastego kwietnia
            (re.compile(r"\b(na|dnia|do dnia)\s+([012]?\d|3[01])\.([01]?\d)\b", re.IGNORECASE),
             self._date_no_year),
            # Exchange rate: po kursie 3,95 → po kursie trzy złote i dziewięćdziesiąt pięć groszy
            (re.compile(r"\b(po\s+kursie|kursu|kurs)\s+(\d+)[,.](\d{2})\b", re.IGNORECASE),
             self._exchange_rate),
            # Decimal + temporal unit: 5,27 lat → pięć i dwadzieścia siedem setnych lat
            (re.compile(r"\b(\d+),(\d{2})\s+(lat|roku|lata|sekundy|sekund|minuty|minut)\b"),
             self._decimal_temporal),
            # Noun-governed ordinal: w linii 128 → w linii sto dwudziestej ósmej
            (re.compile(r"\b(\w+)\s+(\d+)\b"),
             self._noun_num_ordinal),
            (re.compile(r"\b(\d+(?:[,.]\d+)?)\b"),
             self._number),
        ]

    # ── konwertery ────────────────────────────────────────────────────────────

    def _date_roman_month(self, m):
        """Daty z miesiącem rzymskim: 12.XI.1473 → dwunastego listopada..."""
        d, roman_m, y = int(m.group(1)), m.group(2).upper(), int(m.group(3))
        try:
            mo = _roman_to_int(roman_m)
        except Exception:
            return m.group(0)
        if not (1 <= mo <= 12 and 1 <= d <= 31):
            return m.group(0)
        return f"{_ORD_GEN_DAYS.get(d, str(d))} {_MONTHS_GEN[mo]} {_year_gen(y)} roku"

    def _date_full(self, m):
        d, mo, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if not (1 <= mo <= 12 and 1 <= d <= 31): return m.group(0)
        # Add trailing period if we consumed " r." at end of sentence
        matched = m.group(0)
        suffix = "." if matched.rstrip().endswith("r.") and not m.string[m.end():].strip() else ""
        return f"{_ORD_GEN_DAYS.get(d, str(d))} {_MONTHS_GEN[mo]} {_year_gen(y)} roku{suffix}"

    def _date_short(self, m):
        d, mo, y2 = int(m.group(1)), int(m.group(2)), int(m.group(3))
        y = 2000 + y2 if y2 < 50 else 1900 + y2
        if not (1 <= mo <= 12 and 1 <= d <= 31): return m.group(0)
        return f"{_ORD_GEN_DAYS.get(d, str(d))} {_MONTHS_GEN[mo]} {_year_gen(y)} roku"

    def _year_abbr(self, m):
        suffix = "." if not m.string[m.end():].strip() else ""
        return f"{_year_gen(int(m.group(1)))} roku{suffix}"

    def _year_abbr_loc(self, m):
        """W 2024 r. / w 1965 r. → forma miejscownikowa, zachowując wielkość 'w/W'."""
        prefix = m.group(1)   # 'w' lub 'W'
        year = int(m.group(2))
        suffix = "." if not m.string[m.end():].strip() else ""
        return f"{prefix} {_year_loc(year)} roku{suffix}"

    def _year_w_loc(self, m):
        """W 2024 (bez r.) → forma miejscownikowa bez 'roku'."""
        prefix = m.group(1)   # 'w' lub 'W'
        year = int(m.group(2))
        suffix = "." if not m.string[m.end():].strip() else ""
        return f"{prefix} {_year_loc(year)}{suffix}"

    # Formy łącznikowe dla N-lecie/N-lecia
    _ANNIVERSARY_CONNECTING = {
        5: "pięcio", 6: "sześcio", 7: "siedmio", 8: "ośmio", 9: "dziewięcio",
        10: "dziesięcio", 15: "piętnasto", 20: "dwudziesto", 25: "dwudziestopięcio",
        30: "trzydziesto", 35: "trzydziestopięcio", 40: "czterdziesto",
        45: "czterdziestopięcio", 50: "pięćdziesięcio", 60: "sześćdziesięcio",
        70: "siedemdziesięcio", 75: "siedemdziesięciopięcio", 80: "osiemdziesięcio",
        90: "dziewięćdziesięcio", 100: "stu", 150: "stupięćdziesięcio",
        200: "dwustu", 250: "dwustupięćdziesięcio", 300: "trzechset",
        500: "pięciuset", 1000: "tysiąc",
    }

    def _anniversary(self, m):
        """40-lecia → czterdziestolecia, 100-lecie → stulecie."""
        n = int(m.group(1))
        suffix = m.group(2)  # lecia, lecie, leciu, letni, etc.
        connecting = self._ANNIVERSARY_CONNECTING.get(n)
        if connecting:
            return f"{connecting}{suffix}"
        # Fallback: spell number + suffix
        return f"{_n2w(n)} {suffix}"

    def _decimal_compound_half(self, m):
        """2,5-letni → dwuipółletni, 3,5-letniego → trzyipółletniego."""
        n = int(m.group(1))
        suffix = m.group(2)
        connecting = self._COMPOUND_CONNECTING.get(n)
        if connecting:
            return f"{connecting}ipół{suffix}"
        return f"{_n2w(n)} i pół {suffix}"

    _CENTURY_CONNECTING = {
        1: "jedno", 2: "dwu", 3: "trzy", 4: "cztero", 5: "pięcio",
        6: "sześcio", 7: "siedmio", 8: "ośmio", 9: "dziewięcio",
        10: "dziesięcio", 11: "jedenasto", 12: "dwunasto", 13: "trzynasto",
        14: "czternasto", 15: "piętnasto", 16: "szesnasto", 17: "siedemnasto",
        18: "osiemnasto", 19: "dziewiętnasto", 20: "dwudziesto",
        21: "dwudziestopierwszo",
    }

    # Connecting forms for compound adjectives (N-osobowy, N-metrowy, N-procentowy, etc.)
    _COMPOUND_CONNECTING = {
        1: "jedno", 2: "dwu", 3: "trzy", 4: "cztero", 5: "pięcio",
        6: "sześcio", 7: "siedmio", 8: "ośmio", 9: "dziewięcio",
        10: "dziesięcio", 11: "jedenasto", 12: "dwunasto", 13: "trzynasto",
        14: "czternasto", 15: "piętnasto", 16: "szesnasto", 17: "siedemnasto",
        18: "osiemnasto", 19: "dziewiętnasto", 20: "dwudziesto",
        25: "dwudziestopięcio", 30: "trzydziesto", 40: "czterdziesto",
        50: "pięćdziesięcio", 60: "sześćdziesięcio", 70: "siedemdziesięcio",
        80: "osiemdziesięcio", 90: "dziewięćdziesięcio", 100: "stu",
    }

    def _compound_adjective(self, m):
        """50-osobowej → pięćdziesięcioosobowej, 24-karatowe → dwudziestoczterokaratowe."""
        n = int(m.group(1))
        root_suffix = m.group(2)
        connecting = self._COMPOUND_CONNECTING.get(n)
        if not connecting and 20 < n < 100:
            # Decompose: 24 = 20 + 4 → "dwudziesto" + "cztero" = "dwudziestocztero"
            tens = (n // 10) * 10
            ones = n % 10
            tens_form = self._COMPOUND_CONNECTING.get(tens)
            ones_form = self._COMPOUND_CONNECTING.get(ones)
            if tens_form and ones_form:
                connecting = tens_form + ones_form
        if connecting:
            return f"{connecting}{root_suffix}"
        return f"{_n2w(n)} {root_suffix}"

    def _century_compound(self, m):
        """18-wiecznym → osiemnastowiecznym."""
        n = int(m.group(1))
        suffix = m.group(2)
        connecting = self._CENTURY_CONNECTING.get(n)
        if connecting:
            return f"{connecting}{suffix}"
        return f"{_n2w(n)} {suffix}"

    def _year_range_paren(self, m):
        """(1923-1993) → tysiąc dziewięćset dwudziestego trzeciego do ... (genitive ordinal)."""
        y1, y2 = int(m.group(1)), int(m.group(2))
        return f"({_year_gen(y1)} do {_year_gen(y2)})"

    def _digit_lower_suffix(self, m):
        """49b → czterdzieści dziewięć be (odniesienia prawne, artykuły)."""
        num, suffix = int(m.group(1)), m.group(2)
        # Przeliteruj suffix jako nazwy liter
        spelled = " ".join(_LETTER_NAMES_PL.get(c.upper(), c) for c in suffix)
        return f"{_n2w(num)} {spelled}"

    @staticmethod
    def _ordinal_to_gen_pl(ordinal: str) -> str:
        """Ordinal nominative → genitive plural: osiemdziesiąty → osiemdziesiątych."""
        if ordinal.endswith("y"):
            return ordinal[:-1] + "ych"
        elif ordinal.endswith("i"):
            return ordinal[:-1] + "ich"
        return ordinal + "ych"

    def _decade_conjunction(self, m):
        """W latach 50. i 60. → W latach pięćdziesiątych i sześćdziesiątych."""
        prefix, n1, conj, n2 = m.group(1), int(m.group(2)), m.group(3), int(m.group(4))
        gen1 = self._ordinal_to_gen_pl(num2words(n1, lang="pl", to="ordinal"))
        gen2 = self._ordinal_to_gen_pl(num2words(n2, lang="pl", to="ordinal"))
        return f"{prefix} {gen1} {conj} {gen2}"

    def _decade(self, m):
        """W latach 80. / lat 70. → W latach osiemdziesiątych. / lat siedemdziesiątych."""
        prefix, n = m.group(1), int(m.group(2))
        gen_pl = self._ordinal_to_gen_pl(num2words(n, lang="pl", to="ordinal"))
        return f"{prefix} {gen_pl}"

    def _year_w_loc_short_roku(self, m):
        """W 496 roku → W czterysta dziewięćdziesiątym szóstym roku."""
        prefix, year = m.group(1), int(m.group(2))
        return f"{prefix} {_year_loc(year)} roku"

    def _month_bare_year(self, m):
        """września 1921 → września tysiąc dziewięćset dwudziestego pierwszego roku."""
        month, year = m.group(1), int(m.group(2))
        return f"{month} {_year_gen(year)} roku"

    def _season_year(self, m):
        """Latem 1937 → Latem tysiąc dziewięćset trzydziestego siódmego roku."""
        season, year = m.group(1), int(m.group(2))
        return f"{season} {_year_gen(year)} roku"

    def _day_conj_month(self, m):
        """12 i 13 kwietnia → dwunastego i trzynastego kwietnia."""
        d1 = int(m.group(1))
        conj = m.group(2)
        d2 = int(m.group(3))
        month = m.group(4)
        return f"{_ORD_GEN_DAYS.get(d1, str(d1))} {conj} {_ORD_GEN_DAYS.get(d2, str(d2))} {month}"

    def _day_month(self, m):
        """24 grudnia → dwudziestego czwartego grudnia."""
        d, month = int(m.group(1)), m.group(2)
        return f"{_ORD_GEN_DAYS.get(d, str(d))} {month}"

    def _day_month_year(self, m):
        """20 listopada 2017 roku → dwudziestego listopada dwa tysiące siedemnastego roku."""
        d, month, year = int(m.group(1)), m.group(2), int(m.group(3))
        return f"{_ORD_GEN_DAYS.get(d, str(d))} {month} {_year_gen(year)} roku"

    def _month_year_loc(self, m):
        prefix, month, year = m.group(1), m.group(2), int(m.group(3))
        return f"{prefix} {month} {_year_gen(year)} roku"

    def _year_w_loc_dotted_roku(self, m):
        prefix, year = m.group(1), int(m.group(2))
        return f"{prefix} {_year_loc(year)} roku"

    def _year_w_loc_dotted(self, m):
        prefix, year = m.group(1), int(m.group(2))
        return f"{prefix} {_year_loc(year)}"

    def _day_range_month(self, m):
        """10-19 sierpnia [2007] → dziesiątego do dziewiętnastego sierpnia [dwa tysiące siódmego roku]."""
        d1, d2, month = int(m.group(1)), int(m.group(2)), m.group(3)
        result = f"{_ORD_GEN_DAYS.get(d1, str(d1))} do {_ORD_GEN_DAYS.get(d2, str(d2))} {month}"
        if m.group(4):
            result += f" {_year_gen(int(m.group(4)))} roku"
        return result

    def _year_range(self, m):
        """1975-1998 → tysiąc dziewięćset siedemdziesiąty piąty do ... (ordinal nominative)."""
        y1, y2 = int(m.group(1)), int(m.group(2))
        o1 = num2words(y1, lang="pl", to="ordinal")
        o2 = num2words(y2, lang="pl", to="ordinal")
        return f"{o1} do {o2}"

    def _version_3(self, m):
        """Version X.Y.Z: wersji 10.15.7 → wersji dziesięć kropka piętnaście kropka siedem."""
        prefix = m.group(1)
        parts = [_n2w(int(m.group(i))) for i in range(2, 5)]
        return f"{prefix} {' kropka '.join(parts)}"

    def _version_2(self, m):
        """Version X.Y: wersja 2.0 → wersja dwa kropka zero."""
        prefix = m.group(1)
        major, minor = _n2w(int(m.group(2))), _n2w(int(m.group(3)))
        return f"{prefix} {major} kropka {minor}"

    def _ip_address(self, m):
        """192.168.1.1 → sto dziewięćdziesiąt dwa kropka sto sześćdziesiąt osiem kropka jeden kropka jeden."""
        parts = [_n2w(int(m.group(i))) for i in range(1, 5)]
        return " kropka ".join(parts)

    def _phone_intl(self, m):
        """Telefon z kodem kraju: +48 123 456 789 → plus czterdzieści osiem..."""
        cc = _n2w(int(m.group(1)))
        g1 = _n2w(int(m.group(2)))
        g2 = _n2w(int(m.group(3)))
        g3 = _n2w(int(m.group(4)))
        return f"plus {cc} {g1} {g2} {g3}"

    def _compound_half(self, m):
        """N i 1/2 → N i pół: 3 i 1/2 → trzy i pół."""
        n = int(m.group(1))
        return f"{_n2w(n)} i pół"

    def _equal_ratio(self, m):
        """Equal ratio: 50/50 → pięćdziesiąt na pięćdziesiąt."""
        n1, n2 = int(m.group(1)), int(m.group(2))
        if n1 == n2:
            return f"{_n2w(n1)} na {_n2w(n2)}"
        return m.group(0)  # not equal, leave for _fraction handler

    # Fraction denominator forms by case:
    # nominative (jedna trzecia), genitive (jednej trzeciej), accusative (jedną trzecią)
    _DENOM_NOM = {
        2: ("druga", "drugie", "drugich"),
        3: ("trzecia", "trzecie", "trzecich"),
        4: ("czwarta", "czwarte", "czwartych"),
        5: ("piąta", "piąte", "piątych"),
        6: ("szósta", "szóste", "szóstych"),
        7: ("siódma", "siódme", "siódmych"),
        8: ("ósma", "ósme", "ósmych"),
        9: ("dziewiąta", "dziewiąte", "dziewiątych"),
        10: ("dziesiąta", "dziesiąte", "dziesiątych"),
    }
    _DENOM_GEN = {
        2: ("drugiej", "drugich", "drugich"),
        3: ("trzeciej", "trzecich", "trzecich"),
        4: ("czwartej", "czwartych", "czwartych"),
        5: ("piątej", "piątych", "piątych"),
        6: ("szóstej", "szóstych", "szóstych"),
        7: ("siódmej", "siódmych", "siódmych"),
        8: ("ósmej", "ósmych", "ósmych"),
        9: ("dziewiątej", "dziewiątych", "dziewiątych"),
        10: ("dziesiątej", "dziesiątych", "dziesiątych"),
    }
    _DENOM_ACC = {
        2: ("drugą", "drugie", "drugich"),
        3: ("trzecią", "trzecie", "trzecich"),
        4: ("czwartą", "czwarte", "czwartych"),
        5: ("piątą", "piąte", "piątych"),
        6: ("szóstą", "szóste", "szóstych"),
        7: ("siódmą", "siódme", "siódmych"),
        8: ("ósmą", "ósme", "ósmych"),
        9: ("dziewiątą", "dziewiąte", "dziewiątych"),
        10: ("dziesiątą", "dziesiąte", "dziesiątych"),
    }
    # Numerator forms by case (feminine):
    # nom: jedna, dwie, trzy...; gen: jednej, dwóch, trzech...; acc: jedną, dwie, trzy...
    _NUM_FRAC_GEN = {
        1: "jednej", 2: "dwóch", 3: "trzech", 4: "czterech",
        5: "pięciu", 6: "sześciu", 7: "siedmiu", 8: "ośmiu", 9: "dziewięciu",
    }
    _NUM_FRAC_ACC = {
        1: "jedną", 2: "dwie", 3: "trzy", 4: "cztery",
        5: "pięć", 6: "sześć", 7: "siedem", 8: "osiem", 9: "dziewięć",
    }

    def _prep_fraction(self, m):
        """Preposition + fraction with case inflection:
        w 1/3 → w jednej trzeciej (genitive after w/we/po/przy/o)
        do 2/5 → do dwóch piątych (genitive after do/od/bez/dla/z/ze)
        na 1/3 → na jedną trzecią (accusative after na)
        """
        prep = m.group(1)
        num, den = int(m.group(2)), int(m.group(3))
        case = self._PREP_CASE_MAP.get(prep.lower(), "gen")
        # For fractions: "na" requires accusative (not nominative)
        if prep.lower() in ("na", "przez"):
            case = "acc"
        # "o" + fraction typically means "by" (accusative): spadły o 2/5 → o dwie piąte
        if prep.lower() == "o":
            case = "acc"
        # Map locative to genitive for fractions (same forms)
        if case == "loc":
            case = "gen"
        if case == "inst":
            case = "acc"  # instrumental fractions use accusative-like forms: z jedną trzecią
        if den == 2 and num == 1:
            if case == "gen":
                return f"{prep} połowy"
            elif case == "acc" or case == "nom":
                return f"{prep} połowę" if case == "acc" else f"{prep} połowa"
            return f"{prep} pół"
        if case == "gen":
            denom_table = self._DENOM_GEN
            num_word = self._NUM_FRAC_GEN.get(num, _cardinal_inflect(num, "gen", "f"))
        elif case == "acc" or case == "nom":
            # "na" takes accusative for fractions
            denom_table = self._DENOM_ACC if case == "acc" else self._DENOM_NOM
            num_word = self._NUM_FRAC_ACC.get(num, _n2w(num, "f")) if case == "acc" else _n2w(num, "f")
        else:
            denom_table = self._DENOM_GEN
            num_word = self._NUM_FRAC_GEN.get(num, _cardinal_inflect(num, "gen", "f"))
        if den not in denom_table:
            return f"{prep} {_n2w(num)} łamane {_n2w(den)}"
        d_forms = denom_table[den]
        d_word = _pick(num, d_forms[0], d_forms[1], d_forms[2])
        return f"{prep} {num_word} {d_word}"

    def _fraction_inch(self, m):
        """Fraction-inches: 3/4" → trzech czwartych cala (genitive)."""
        num, den = int(m.group(1)), int(m.group(2))
        if den in self._DENOM_GEN:
            d_forms = self._DENOM_GEN[den]
            d_word = _pick(num, d_forms[0], d_forms[1], d_forms[2])
            num_word = self._NUM_FRAC_GEN.get(num, _cardinal_inflect(num, "gen", "f"))
            return f"{num_word} {d_word} cala"
        return f"{_n2w(num, 'f')} {_n2w(den)} cala"

    def _score_pkt(self, m):
        """Score with pkt: 98/100 pkt → dziewięćdziesiąt osiem na sto punktów."""
        num, den = int(m.group(1)), int(m.group(2))
        pkt_form = _pick(den, "punkt", "punkty", "punktów")
        return f"{_n2w(num)} na {_n2w(den)} {pkt_form}"

    def _fraction(self, m):
        """Ułamki zwykłe: 1/2 → pół, 3/4 → trzy czwarte.
        Context-aware: after transitive verbs → accusative; after genitive nouns → genitive; default → nominative."""
        num, den = int(m.group(1)), int(m.group(2))
        # In formula context (after "równa się"), use full fraction form: 1/2 → "jedna druga"
        before_frac = m.string[max(0, m.start() - 30):m.start()].lower().rstrip()
        in_formula = "równa się" in before_frac or before_frac.endswith("razy")
        # Standalone 1/2 → "pół" (compound "N i 1/2" handled earlier by _compound_half)
        if den == 2 and num == 1 and not in_formula:
            return "pół"
        if den not in self._DENOM_NOM:
            # Check if followed by "pkt" — score context: "98/100 pkt" → "na sto punktów"
            after = m.string[m.end():]
            pkt_m = re.match(r'\s+pkt\.?\b', after)
            if pkt_m:
                return f"{_n2w(num)} na {_n2w(den)}"
            # General fraction/ratio: 15/2026 → "piętnaście ukośnik dwa tysiące..."
            return f"{_n2w(num)} ukośnik {_n2w(den)}"
        # Determine case from context
        before = m.string[max(0, m.start() - 40):m.start()].lower().rstrip()
        # Address context: fraction after street name → "N przez M"
        if re.search(r'(?:ul\.|ulica|al\.|aleja|alei|pl\.|plac|os\.|osiedle|solidarności|niepodległości|piłsudskiego|kościuszki|sikorskiego|chopina|mickiewicza|słowackiego)\s*$', before):
            return f"{_n2w(num)} przez {_n2w(den)}"
        # After genitive-governing nouns → genitive
        if re.search(r'\b(użycia|potrzeba|wymaga)\s*$', before):
            denom_table = self._DENOM_GEN
            num_word = self._NUM_FRAC_GEN.get(num, _cardinal_inflect(num, "gen", "f"))
        # After "i" with preceding genitive number (potrzeba dwóch i 1/4) → genitive
        elif re.search(r'\b(?:dwóch|trzech|czterech|pięciu|sześciu|siedmiu|ośmiu|dziewięciu)\s+i\s*$', before):
            denom_table = self._DENOM_GEN
            num_word = self._NUM_FRAC_GEN.get(num, _cardinal_inflect(num, "gen", "f"))
        # After "N i" with preceding genitive-governing noun (potrzeba 2 i 1/4) → genitive
        # (digit form — _fraction runs before _noun_num_ordinal converts digits to words)
        elif re.search(r'\b(?:użycia|potrzeba|wymaga)\s+\d+\s+i\s*$', before):
            denom_table = self._DENOM_GEN
            num_word = self._NUM_FRAC_GEN.get(num, _cardinal_inflect(num, "gen", "f"))
        # After transitive verbs → accusative (dodał, zużyto, wydał, etc.)
        elif re.search(r'\b(dodał|dodać|zużyto|zużył|wydał|kupił|wlał|wlej|dodaj|spadły)\s*$', before):
            denom_table = self._DENOM_ACC
            num_word = self._NUM_FRAC_ACC.get(num, _n2w(num, "f"))
        else:
            # Default: nominative (jedna trzecia, trzy czwarte)
            denom_table = self._DENOM_NOM
            num_word = _n2w(num, "f")
        d_forms = denom_table[den]
        d_word = _pick(num, d_forms[0], d_forms[1], d_forms[2])
        return f"{num_word} {d_word}"

    def _year_before_roku(self, m):
        """1982 roku → tysiąc dziewięćset osiemdziesiątego drugiego roku."""
        year = int(m.group(1))
        return f"{_year_gen(year)} roku"

    def _year2_before_roku(self, m):
        """82 roku → osiemdziesiątego drugiego roku."""
        year = int(m.group(1))
        ordinal = num2words(year, lang="pl", to="ordinal")
        return f"{_year_gen(year) if year >= 1000 else _year_gen_small(year)} roku"

    def _hour_context_number(self, m):
        prefix = m.group(1)
        hour = int(m.group(2))
        prefix_norm = prefix.lower().strip()
        if prefix_norm.startswith("o "):
            return f"o godzinie {_hour_ordinal(hour, 'loc')}"
        return f"około godziny {_hour_ordinal(hour, 'gen')}"

    def _year_range_gen(self, m):
        """lat/z lat 1918-1939 → tysiąc dziewięćset osiemnaście do ... (cardinal)."""
        prefix, y1, y2 = m.group(1), int(m.group(2)), int(m.group(3))
        return f"{prefix} {_n2w(y1)} do {_n2w(y2)}"

    def _year_after_rok(self, m):
        """Rok 1989 → Rok tysiąc dziewięćset osiemdziesiąty dziewiąty (ordinal nominative).
        W roku 1949 → W roku tysiąc dziewięćset czterdziestym dziewiątym (locative).
        Rok 2000 → Rok dwutysięczny (ordinal, nie kardynalny).
        """
        prefix = m.group(1)  # 'rok' lub 'roku' (preserve case)
        year = int(m.group(2))
        before = m.string[max(0, m.start() - 15):m.start()].lower().rstrip()
        if prefix.lower() == "roku":
            # "W roku" → locative
            if re.search(r'\bw\s*$', before):
                return f"{prefix} {_year_loc(year)}"
            # "Około/do/od/z roku" → genitive
            if re.search(r'\b(?:około|ok\.|do|od|z|ze)\s*$', before):
                return f"{prefix} {_year_gen(year)}"
        ordinal = num2words(year, lang="pl", to="ordinal")
        return f"{prefix} {ordinal}"

    def _pos_unit(self, m):
        """Jednaostka z plusem: +5°C → plus pięć stopni Celsjusza."""
        raw, key = m.group(1), m.group(2)
        unit = _UNITS.get(key)
        if not unit: return m.group(0)
        val, clean = _parse_raw(raw)
        n_int = int(val)
        if val == n_int:
            return f"plus {_n2w(n_int, unit[3])} {_pick(n_int, unit[0], unit[1], unit[2])}"
        gen_sg = self._gen_sg_unit(key)
        return f"plus {_n2w_float(clean, unit[3])} {gen_sg}"

    def _per_km2(self, m):
        """22 osób/km² → dwadzieścia dwa osób na kilometr kwadratowy."""
        n, noun = int(m.group(1)), m.group(2)
        return f"{_n2w(n)} {noun} na kilometr kwadratowy"

    def _time_words(self, h: int, mi: int, sec: int | None = None, *,
                    locative_hour: bool = False, hour_case: str | None = None) -> str:
        # Determine hour case: explicit parameter > locative flag > nominative
        if hour_case:
            h_ord = _hour_ordinal(h, case=hour_case)
        elif locative_hour:
            h_ord = _hour_ordinal(h, case="loc")
        else:
            h_ord = _hour_ordinal(h, case="nom")
        if sec is not None:
            # Full HH:MM:SS format: "dwie godziny piętnaście minut i czterdzieści pięć sekund"
            h_word = _pick(h, "godzinę", "godziny", "godzin")
            h_num = _n2w(h, "f")
            mi_w = num2words(mi, lang="pl") if mi != 0 else "zero"
            mi_word = _pick(mi, "minutę", "minuty", "minut")
            sec_w = num2words(sec, lang="pl") if sec != 0 else "zero"
            sec_word = _pick(sec, "sekundę", "sekundy", "sekund")
            return f"{h_num} {h_word} {mi_w} {mi_word} i {sec_w} {sec_word}"
        if mi == 0:
            return h_ord
        # Leading zero for minutes 1-9: "czternastej zero dwa"
        if 1 <= mi <= 9:
            return f"{h_ord} zero {num2words(mi, lang='pl')}"
        return f"{h_ord} {num2words(mi, lang='pl')}"

    # Preposition → hour case mapping for colon-formatted times
    _TIME_PREP_CASE = {
        "o": "loc",      # o czternastej (locative)
        "od": "gen",     # od szóstej (genitive)
        "do": "gen",     # do dwudziestej drugiej (genitive)
        "około": "gen",  # około ósmej (genitive)
        "przed": "inst", # przed piątą (instrumental — but same as loc for fem ordinals)
        "po": "loc",     # po piątej (locative)
        "między": "inst", # między piątą a szóstą (instrumental)
        "na": "acc",     # na dwunastą (accusative)
    }

    _TIME_SUFFIX_RE = re.compile(r"\s+(rano|wieczorem|w nocy|po południu|nad ranem)\b", re.IGNORECASE)

    # Midnight case forms for "przed północą", "o północy", "do północy" etc.
    _MIDNIGHT_CASE = {
        "nom": "północ", "gen": "północy", "loc": "północy",
        "dat": "północy", "acc": "północ", "inst": "północą",
    }

    def _time_prefixed_colon(self, m):
        """Preposition + HH:MM[:SS] — case-aware hour form.
        Full hours (e.g. 11:00) include 'zero zero' for clarity,
        unless followed by rano/wieczorem/etc.
        HH:MM:SS after clock-time prepositions uses clock format (not duration).
        """
        prefix = m.group(1)
        h, mi = int(m.group(2)), int(m.group(3))
        sec = int(m.group(4)) if m.group(4) else None
        case = self._TIME_PREP_CASE.get(prefix.lower(), "nom")
        # Midnight: 00:00 → "północ" (with case inflection)
        if h == 0 and mi == 0 and sec is None:
            return f"{prefix} {self._MIDNIGHT_CASE.get(case, 'północ')}"
        # HH:MM:SS after clock preposition → clock format: "o trzeciej zero zero piętnaście"
        if sec is not None and prefix.lower() in self._TIME_PREP_CASE:
            h_ord = _hour_ordinal(h, case=case)
            mi_str = f"zero {num2words(mi, lang='pl')}" if 1 <= mi <= 9 else (
                "zero zero" if mi == 0 else num2words(mi, lang="pl"))
            sec_str = f"zero {num2words(sec, lang='pl')}" if 1 <= sec <= 9 else (
                "zero zero" if sec == 0 else num2words(sec, lang="pl"))
            return f"{prefix} {h_ord} {mi_str} {sec_str}"
        tw = self._time_words(h, mi, sec, hour_case=case)
        # For full hours in prefixed context, add "zero zero" only for nominative case
        # (bare times like "godzina 17:00"), not after prepositions (o 17:00 → "o siedemnastej")
        if mi == 0 and sec is None:
            after = m.string[m.end():]
            if case == "nom" and not self._TIME_SUFFIX_RE.match(after):
                tw = f"{tw} zero zero"
        return f"{prefix} {tw}"

    def _time(self, m):
        h, mi = int(m.group(1)), int(m.group(2))
        sec = int(m.group(3)) if m.group(3) else None
        # Midnight: 00:00 → "północ"
        if h == 0 and mi == 0 and sec is None:
            return "północ"
        # HH:MM:SS → full duration format: "dwie godziny piętnaście minut i czterdzieści pięć sekund"
        if sec is not None:
            return self._time_words(h, mi, sec)
        # Check context: if preceded by duration/music words, treat as MM:SS
        before = m.string[max(0, m.start() - 80):m.start()].lower()
        if re.search(r'(?:trwania|czas trwania|długość|utwór|piosenk|nagranie|film|odcinek|w czasie)', before) and h < 60:
            mi_word = _pick(h, "minutę", "minuty", "minut")
            sec_word = _pick(mi, "sekundę", "sekundy", "sekund")
            return f"{_n2w(h, 'f')} {mi_word} i {num2words(mi, lang='pl')} {sec_word}"
        # Full hour without preposition: use ordinal form
        # "15:00" → "piętnasta"
        if mi == 0:
            return _hour_ordinal(h, case="nom")
        # Plain time without preposition: use cardinal numbers
        # "19:45" → "dziewiętnaście czterdzieści pięć"
        parts = [_n2w(h)]
        if mi > 0:
            parts.append(num2words(mi, lang="pl"))
        return " ".join(parts)

    def _time_prefixed_dot(self, m):
        prefix = m.group(1)
        h, mi = int(m.group(2)), int(m.group(3))
        sec = int(m.group(4)) if m.group(4) else None
        prefix_norm = prefix.lower().strip()
        hour_case = "nom"
        if prefix_norm == "o":
            rendered_prefix = prefix
            hour_case = "loc"
        elif prefix_norm.startswith("o godz") or prefix_norm in {"o godzina", "o godzinie", "o godziną"}:
            rendered_prefix = "o godzinie"
            hour_case = "loc"
        elif prefix_norm in {"godz.", "godz", "godzina"}:
            rendered_prefix = "godzina"
        elif prefix_norm in {"godzinie"}:
            rendered_prefix = "godzinie"
            hour_case = "loc"
        elif prefix_norm in {"godziną"}:
            rendered_prefix = "godziną"
        elif prefix_norm in {"od", "do", "około"}:
            rendered_prefix = prefix
            hour_case = "gen"
        elif prefix_norm in {"między", "przed", "po"}:
            rendered_prefix = prefix
            hour_case = self._TIME_PREP_CASE.get(prefix_norm, "nom")
        else:
            rendered_prefix = prefix
        return f"{rendered_prefix} {self._time_words(h, mi, sec, hour_case=hour_case)}"

    def _time_after_sep_dot(self, m):
        sep = m.group(1)
        h, mi = int(m.group(2)), int(m.group(3))
        sec = int(m.group(4)) if m.group(4) else None
        return f"{sep}{self._time_words(h, mi, sec)}"

    def _score(self, m):
        """Sport score: 3:0 → trzy do zera, 25:23 → dwadzieścia pięć do dwudziestu trzech.
        Special: wynikiem 0:0 → remisem zero do zera."""
        prefix = m.group(1)  # "wynikiem" or None
        n1, n2 = int(m.group(2)), int(m.group(3))
        score_text = f"{_n2w(n1)} do {_cardinal_gen(n2)}"
        if prefix and n1 == 0 and n2 == 0:
            return f"remisem {score_text}"
        if prefix:
            return f"{prefix} {score_text}"
        return score_text

    # Map noun form → adjective suffix (appended to "procentow")
    _PERCENT_ADJ_SUFFIX = {
        # Feminine locative/genitive → -ej
        "obniżce": "ej", "obniżki": "ej", "obniżką": "ą", "obniżkę": "ą",
        "zniżce": "ej", "zniżki": "ej", "zniżką": "ą", "zniżkę": "ą",
        "zwyżce": "ej", "zwyżki": "ej", "zwyżką": "ą", "zwyżkę": "ą",
        "stawce": "ej", "stawki": "ej", "stawką": "ą", "stawkę": "ą",
        "obniżka": "a", "zniżka": "a", "zwyżka": "a", "stawka": "a",
        # Masculine genitive → -ego
        "wzrostu": "ego", "spadku": "ego", "udziału": "ego",
        "podatku": "ego", "rabatu": "ego",
        # Masculine instrumental/locative → -ym
        "wzrostem": "ym", "spadkiem": "ym", "wzroście": "ym",
    }

    def _percent_adjective(self, m):
        """5% wzrost → pięcioprocentowy wzrost, 20% obniżce → dwudziestoprocentowej obniżce."""
        n = int(m.group(1))
        noun = m.group(2)
        suffix = self._PERCENT_ADJ_SUFFIX.get(noun.lower(), "y")
        connecting = self._COMPOUND_CONNECTING.get(n)
        if connecting:
            return f"{connecting}procentow{suffix} {noun}"
        return f"{_n2w(n)} procentow{suffix} {noun}"

    def _prep_percent(self, m):
        """Preposition + percentage — case-aware: 'od 7,5%' → 'od siedmiu i pół procent',
        'w 90%' → 'w dziewięćdziesięciu procentach'."""
        prep, raw = m.group(1), m.group(2)
        case = self._PREP_CASE_MAP.get(prep.lower(), "gen")
        val, clean = _parse_raw(raw)
        # Determine percent form based on case
        if case == "loc":
            percent_word = "procentach"
        elif case == "inst":
            percent_word = "procentami"
        else:
            percent_word = "procent"
        if val == int(val):
            n = int(val)
            return f"{prep} {_cardinal_inflect(n, case)} {percent_word}"
        # For decimals with prepositions, use "procent" (genitive plural)
        # except for standalone nominative context which uses "procenta" (gen sg)
        if case == "loc":
            percent_dec = "procentach"
        elif case == "inst":
            percent_dec = "procentami"
        else:
            percent_dec = "procent"
        clean_dot = clean.replace(",", ".")
        if "." in clean_dot and clean_dot.split(".")[1] == "5":
            int_part = int(clean_dot.split(".")[0])
            if int_part == 0:
                return f"{prep} pół {percent_dec}"
            return f"{prep} {_cardinal_inflect(int_part, case)} i pół {percent_dec}"
        if "." in clean_dot:
            int_s, dec_s = clean_dot.split(".", 1)
            int_part = int(int_s or "0")
            if len(dec_s) == 1:
                dec_val = int(dec_s)
                int_w = _cardinal_inflect(int_part, case) if int_part > 0 else "zera"
                dec_w = _n2w(dec_val, "f")
                return f"{prep} {int_w} i {dec_w} dziesiątych {percent_dec}"
        return f"{prep} {_n2w_float(clean)} {percent_dec}"

    def _percent(self, m):
        raw = m.group(1)
        val, clean = _parse_raw(raw)
        # Check for genitive-governing verb context
        before = m.string[max(0, m.start() - 40):m.start()].lower().rstrip()
        use_gen = bool(re.search(r'\b(użyto|zużyto|potrzeba|wymaga|wystarczy|brakuje|zabrakło)\s*$', before))
        if val == int(val):
            n = int(val)
            if use_gen:
                return f"{_cardinal_inflect(n, 'gen')} procent"
            return f"{_n2w(n)} procent"
        clean_dot = clean.replace(",", ".")
        if "." in clean_dot:
            int_s, dec_s = clean_dot.split(".", 1)
            int_part = int(int_s or "0")
            # .5 → "i pół"
            if dec_s == "5":
                if use_gen:
                    percent_word = "procenta" if int_part <= 4 else "procent"
                    int_w = _cardinal_inflect(int_part, "gen") if int_part > 0 else "zera"
                    return f"{int_w} i pół {percent_word}"
                return f"{_fraction_words(clean)} procent"
            # Single decimal: "N i X dziesiąte procent" (nominative)
            if len(dec_s) == 1:
                dec_val = int(dec_s)
                int_w = _n2w(int_part) if int_part > 0 else "zero"
                dec_w = _n2w(dec_val, "f")
                frac_form = _pick(dec_val, "dziesiąta", "dziesiąte", "dziesiątych")
                return f"{int_w} i {dec_w} {frac_form} procent"
        return f"{_n2w_float(clean)} procent"

    def _curr_cents(self, m):
        main_n = int(m.group(1).replace("\u00a0","").replace(" ",""))
        cents_n, sym = int(m.group(2)), m.group(3)
        cur = _CURRENCIES.get(sym)
        if not cur: return m.group(0)
        # BTC: don't split small amounts into satoshi, use decimal format
        if sym == "BTC" and main_n == 0:
            raw = m.group(1) + "," + m.group(2)
            gender = cur[6] if len(cur) > 6 else "m"
            gen_sg = _masc_gen_sg(cur[0])
            return f"{_n2w_float(raw, gender)} {gen_sg}"
        gender = cur[6] if len(cur) > 6 else "m"
        cf = _pick(cents_n, cur[3], cur[4], cur[5])
        if main_n == 0:
            return f"{_n2w(cents_n)} {cf}"
        mf = _pick(main_n,  cur[0], cur[1], cur[2])
        return f"{_n2w(main_n, gender)} {mf} {_n2w(cents_n)} {cf}"

    def _prep_large_curr(self, m):
        """Preposition + large currency: do 10 mld zł → do dziesięciu miliardów złotych."""
        prep = m.group(1)
        num_raw = m.group(2)
        abbr_raw = m.group(3).rstrip(".")
        # Find currency symbol at end
        rest = m.group(0)
        sym_match = re.search(r"(zł|PLN|EUR|USD|GBP|CHF|CZK)\s*$", rest)
        if not sym_match:
            return m.group(0)
        sym = sym_match.group(1)
        val, clean = _parse_raw(num_raw)
        large, cur = _LARGE.get(abbr_raw), _CURRENCIES.get(sym)
        if not large or not cur:
            return m.group(0)
        n_int = int(val)
        if val == n_int:
            return f"{prep} {_cardinal_gen(n_int)} {large[2]} {cur[2]}"
        return f"{prep} {_n2w_float(clean)} {large[3]} {cur[2]}"

    def _large_curr(self, m):
        full = m.group(0)
        inner = re.match(
            r"(\d[\d\u00a0 ,.]*)\s*(mln|mld|bln|tys\.?)\s*(zł|PLN|EUR|USD|GBP|CHF|CZK|[€$£¥])", full)
        if not inner: return full
        num_raw, abbr_raw, sym = inner.group(1), inner.group(2).rstrip("."), inner.group(3)
        val, clean = _parse_raw(num_raw)
        large, cur = _LARGE.get(abbr_raw), _CURRENCIES.get(sym)
        if not large or not cur: return full
        # Check for preceding governing noun (limited set for large currencies)
        _LARGE_CURR_GEN_NOUNS = {"poziomie", "kwocie", "kwotą", "sumie", "sumą"}
        before = m.string[max(0, m.start() - 30):m.start()].rstrip()
        prev_word = before.rsplit(None, 1)[-1].lower() if before else ""
        use_gen = prev_word in _LARGE_CURR_GEN_NOUNS
        n_int = int(val)
        # Półtora logic: 1,5 mln zł → półtora miliona złotych
        clean_dot = clean.replace(",", ".")
        if "." in clean_dot:
            dec_part = clean_dot.split(".")[1]
            int_part = int(clean_dot.split(".")[0])
            if dec_part == "5" and int_part in (0, 1):
                if int_part == 0:
                    return f"pół {large[3]} {cur[2]}"
                return f"półtora {large[3]} {cur[2]}"
        if val == n_int:
            if use_gen:
                return f"{_cardinal_gen(n_int)} {large[2]} {cur[2]}"
            return f"{_n2w(n_int)} {_pick(n_int, large[0], large[1], large[2])} {cur[2]}"
        if use_gen:
            return f"{_n2w_float_gen(clean)} {large[3]} {cur[2]}"
        return f"{_fraction_words(clean)} {large[3]} {cur[2]}"

    def _large_unit(self, m):
        """Large + unit: 9,2 mln ha → dziewięć przecinek dwa miliona hektarów."""
        raw, abbr, key = m.group(1), m.group(2).rstrip("."), m.group(3)
        val, clean = _parse_raw(raw)
        large = _LARGE.get(abbr)
        unit = _UNITS.get(key)
        if not large or not unit:
            return m.group(0)
        n_int = int(val)
        if val == n_int:
            return f"{_n2w(n_int)} {_pick(n_int, large[0], large[1], large[2])} {unit[2]}"
        return f"{_fraction_words(clean)} {large[3]} {unit[2]}"

    def _large(self, m):
        raw, abbr = m.group(1), m.group(2)
        dot = m.group(3) if m.lastindex >= 3 else ""
        val, clean = _parse_raw(raw)
        large = _LARGE.get(abbr)
        if not large: return m.group(0)
        n_int = int(val)
        # Preserve trailing period if it was a sentence-ending dot after abbreviation
        suffix = "." if dot == "." and not m.string[m.end():].strip() else ""
        if val == n_int:
            return f"{_n2w(n_int)} {_pick(n_int, large[0], large[1], large[2])}{suffix}"
        return f"{_fraction_words(clean)} {large[3]}{suffix}"

    def _range_large(self, m):
        raw1, raw2, abbr = m.group(1), m.group(2), m.group(3)
        large = _LARGE.get(abbr)
        if not large:
            return m.group(0)
        n1 = int(raw1.replace("\u00a0", "").replace(" ", ""))
        n2 = int(raw2.replace("\u00a0", "").replace(" ", ""))
        large_form = large[3] if n2 == 1 else large[2]
        return f"{_n2w(n1)} do {_cardinal_gen(n2)} {large_form}"

    def _curr_prefix(self, m):
        sym, raw = m.group(1), m.group(2)
        cur = _CURRENCIES.get(sym)
        if not cur: return m.group(0)
        return self._fmt_curr(raw, cur)

    def _prep_curr(self, m):
        """Preposition + number + currency: do 4242 zł → do czterech tysięcy dwustu czterdziestu dwóch złotych."""
        prep, raw, sym = m.group(1), m.group(2), m.group(3)
        cur = _CURRENCIES.get(sym)
        if not cur:
            return m.group(0)
        case = self._PREP_CASE_MAP.get(prep.lower(), "gen")
        # "o N PLN mniej/więcej" → nominative (accusative = same as nominative)
        after = m.string[m.end():].lstrip()
        if prep.lower() == "o" and re.match(r"(?:mniej|więcej)\b", after):
            case = "nom"
        val, clean = _parse_raw(raw)
        n_int = int(val)
        if val == n_int:
            num_word = _cardinal_inflect(n_int, case)
            # Currency form: genitive/locative/instrumental use gen plural (złotych)
            if case in ("gen", "loc", "inst"):
                return f"{prep} {num_word} {cur[2]}"
            return f"{prep} {num_word} {_pick(n_int, cur[0], cur[1], cur[2])}"
        gen_sg = _masc_gen_sg(cur[0])
        return f"{prep} {_n2w_float(clean)} {gen_sg}"

    def _noun_num_curr(self, m):
        """Noun-governed number + currency: kwocie 10 000 PLN → kwocie dziesięciu tysięcy złotych."""
        noun, raw, sym = m.group(1), m.group(2).replace("\u00a0", ""), m.group(3)
        noun_lower = noun.lower()
        if noun_lower not in self._NOUN_CASE_MAP:
            return m.group(0)  # Not a governing noun → fall through
        if noun_lower in self._PREP_CASE_MAP:
            return m.group(0)  # Preposition → handled by _prep_curr
        cur = _CURRENCIES.get(sym)
        if not cur:
            return m.group(0)
        case, _ = self._NOUN_CASE_MAP[noun_lower]
        val, clean = _parse_raw(raw)
        n_int = int(val)
        if val == n_int:
            num_word = _cardinal_inflect(n_int, case)
            if case in ("gen", "loc", "inst"):
                return f"{noun} {num_word} {cur[2]}"
            return f"{noun} {num_word} {_pick(n_int, cur[0], cur[1], cur[2])}"
        gen_sg = _masc_gen_sg(cur[0])
        return f"{noun} {_n2w_float(clean)} {gen_sg}"

    def _curr(self, m):
        raw, sym = m.group(1), m.group(2)
        cur = _CURRENCIES.get(sym)
        if not cur: return m.group(0)
        return self._fmt_curr(raw, cur)

    def _curr_prefix_suffix(self, m):
        """$52k → pięćdziesiąt dwa tysiące dolarów, $12.5M → dwanaście i pół miliona dolarów."""
        sym, raw, suffix = m.group(1), m.group(2), m.group(3).upper()
        cur = _CURRENCIES.get(sym)
        if not cur: return m.group(0)
        val, clean = _parse_raw(raw)
        if suffix == "K":
            n = val * 1000
        else:  # M
            n = val * 1_000_000
        n_int = int(n)
        if n == n_int:
            return f"{_n2w(n_int)} {_pick(n_int, cur[0], cur[1], cur[2])}"
        return f"{_n2w_float(clean)} {_pick(2, cur[0], cur[1], cur[2])}"

    def _signed_curr_prefix(self, m):
        """−$2 770 → minus dwa tysiące siedemset siedemdziesiąt dolarów."""
        sign, sym, raw = m.group(1), m.group(2), m.group(3)
        cur = _CURRENCIES.get(sym)
        if not cur: return m.group(0)
        prefix = "minus " if sign == "-" else "plus "
        val, clean = _parse_raw(raw)
        n_int = int(val)
        if val == n_int:
            return f"{prefix}{_n2w(n_int)} {_pick(n_int, cur[0], cur[1], cur[2])}"
        return f"{prefix}{_n2w_float(clean)} {cur[1]}"

    def _num_k_suffix(self, m):
        """103k → sto trzy tysiące, 12.5k → dwanaście i pół tysiąca."""
        raw, suffix = m.group(1), m.group(2)
        val, clean = _parse_raw(raw)
        n = val * 1000
        n_int = int(n)
        if n == n_int:
            return _n2w(n_int)
        return _n2w_float(str(n))

    def _signed_percent(self, m):
        """-42% → minus czterdzieści dwa procent, +5.3% → plus pięć i trzy dziesiąte procenta."""
        sign, raw = m.group(1), m.group(2)
        prefix = "minus " if sign == "-" else "plus "
        val, clean = _parse_raw(raw)
        n_int = int(val)
        if val == n_int:
            return f"{prefix}{_n2w(n_int)} {_pick(n_int, 'procent', 'procent', 'procent')}"
        return f"{prefix}{_n2w_float(clean)} procent"

    def _prep_wieku_lat(self, m):
        """w wieku 52 lat → w wieku pięćdziesięciu dwóch lat."""
        prefix, n, suffix = m.group(1), int(m.group(2)), m.group(3)
        return f"{prefix} {_cardinal_gen(n)} {suffix}"

    # Words that indicate the 4-digit number is NOT a year (quantity context)
    _NON_YEAR_FOLLOWERS = re.compile(
        r"\s+(?:przebadanych|osób|ludzi|pracowników|uczestników|"
        r"sztuk|egzemplarzy|metrów|kilometrów|złotych|dolarów|euro|"
        r"żołnierzy|studentów|pacjentów|mieszkańców|"
        r"jednostek|elementów|próbek|przypadków|"
        r"wyprodukowanych|sprzedanych|zamówionych|dostarczonych|"
        r"zebranych|zgłoszonych|zarejestrowanych)\b", re.IGNORECASE)

    def _prep_year_standalone(self, m):
        """od/do/z + rok 4-cyfrowy → ordinal genitive: od 1971 → od tysiąc dziewięćset siedemdziesiątego pierwszego."""
        prep = m.group(1)
        year = int(m.group(2))
        # Tylko lata w zakresie 1000-2100 traktuj jako rok
        if 1000 <= year <= 2100:
            # Check for quantity context (non-year follower)
            after = m.string[m.end():]
            if self._NON_YEAR_FOLLOWERS.match(after):
                return f"{prep} {_cardinal_gen(year)}"
            return f"{prep} {_year_gen(year)}"
        # Poza zakresem → cardinal genitive (jak _prep_num)
        return f"{prep} {_cardinal_gen(year)}"

    def _prep_year_range_od_do(self, m):
        """od 1922 do 1940 [roku] → od tysiąc dziewięćset dwudziestego drugiego do ... roku."""
        _, y1, _, y2 = m.group(1), int(m.group(2)), m.group(3), int(m.group(4))
        return f"od {_year_gen(y1)} do {_year_gen(y2)} roku"

    def _prep_neg_unit(self, m):
        """Przyimek + ujemna liczba + jednostka: do -8°C → do minus ośmiu stopni Celsjusza."""
        prep, raw, key = m.group(1), m.group(2).replace("\u00a0", ""), m.group(3)
        unit = _UNITS.get(key)
        if not unit: return m.group(0)
        gender = unit[3]
        case = self._PREP_CASE_MAP.get(prep.lower(), "gen")
        val, clean = _parse_raw(raw)
        n_int = int(val)
        if val == n_int:
            unit_form = self._format_unit(key, n_int, case)
            return f"{prep} minus {_cardinal_inflect(n_int, case)} {unit_form}"
        gen_sg = self._gen_sg_unit(key)
        clean_dot = clean.replace(",", ".")
        if "." in clean_dot:
            int_s, dec_s = clean_dot.split(".", 1)
            int_part = int(int_s or "0")
            if dec_s == "5":
                if int_part == 0:
                    return f"{prep} minus pół {gen_sg}"
                if int_part == 1:
                    return f"{prep} minus półtora {gen_sg}"
                return f"{prep} minus {_cardinal_inflect(int_part, case, gender)} i pół {gen_sg}"
            if len(dec_s) == 1:
                dec_val = int(dec_s)
                int_w = _cardinal_inflect(int_part, case, gender) if int_part > 0 else "zera"
                dec_w = _n2w(dec_val, "f")
                return f"{prep} minus {int_w} i {dec_w} dziesiątych {gen_sg}"
        return f"{prep} minus {_n2w_float(clean, gender)} {gen_sg}"

    def _fmt_curr(self, raw, cur):
        val, clean = _parse_raw(raw)
        n_int = int(val)
        gender = cur[6] if len(cur) > 6 else "m"
        if val == n_int:
            return f"{_n2w(n_int, gender)} {_pick(n_int, cur[0], cur[1], cur[2])}"
        # Check for cents: exactly 2 decimal places → split into main + cents
        clean_dot = clean.replace(",", ".")
        if "." in clean_dot:
            parts = clean_dot.split(".")
            if len(parts[1]) == 2:
                main_n = int(parts[0])
                cents_n = int(parts[1])
                mf = _pick(main_n, cur[0], cur[1], cur[2])
                cf = _pick(cents_n, cur[3], cur[4], cur[5])
                if main_n == 0:
                    return f"{_n2w(cents_n)} {cf}"
                return f"{_n2w(main_n)} {mf} i {_n2w(cents_n)} {cf}"
        # Decimal amounts use genitive singular: 0,55 dolara, 2,5 euro
        # _masc_gen_sg gives genitive from nominative (dolar→dolara, euro→euro)
        gen_sg = _masc_gen_sg(cur[0])
        return f"{_n2w_float(clean)} {gen_sg}"

    def _neg_unit(self, m):
        """Ujemna liczba + jednostka: -5°C → minus pięć stopni Celsjusza.
        Context-aware: after genitive-governing nouns → genitive case."""
        raw, key = m.group(1), m.group(2)
        unit = _UNITS.get(key)
        if not unit: return m.group(0)
        gender = unit[3]
        val, clean = _parse_raw(raw)
        n_int = int(val)
        # Check preceding context for genitive-governing noun
        before = m.string[max(0, m.start() - 30):m.start()].lower().rstrip()
        prev_word_m = re.search(r'(\w+)\s*$', before)
        use_gen = False
        if prev_word_m:
            prev_word = prev_word_m.group(1)
            if prev_word in self._NOUN_CASE_MAP:
                case, _ = self._NOUN_CASE_MAP[prev_word]
                if case == "gen":
                    use_gen = True
        if val == n_int:
            if use_gen:
                unit_form = self._format_unit(key, n_int, "gen")
                return f"minus {_cardinal_inflect(n_int, 'gen')} {unit_form}"
            return f"minus {_n2w(n_int, gender)} {_pick(n_int, unit[0], unit[1], unit[2])}"
        gen_sg = self._gen_sg_unit(key)
        clean_dot = clean.replace(",", ".")
        if "." in clean_dot:
            int_s, dec_s = clean_dot.split(".", 1)
            int_part = int(int_s or "0")
            if len(dec_s) == 1:
                dec_val = int(dec_s)
                if use_gen:
                    int_w = _cardinal_inflect(int_part, "gen", gender) if int_part > 0 else "zera"
                else:
                    int_w = _n2w(int_part, gender) if int_part > 0 else "zero"
                dec_w = _n2w(dec_val)
                return f"minus {int_w} i {dec_w} dziesiątych {gen_sg}"
        return f"minus {_n2w_float(clean, gender)} {gen_sg}"

    def _range_unit(self, m):
        """Zakres + jednostka: 30-50 km/h → trzydzieści do pięćdziesięciu km/h."""
        n1, n2, key = int(m.group(1)), int(m.group(2)), m.group(3)
        unit = _UNITS.get(key)
        if not unit: return m.group(0)
        # n1 w mianowniku, n2 + jednostka w dopełniaczu
        return f"{_n2w(n1)} do {_cardinal_gen(n2)} {unit[2]}"

    def _prep_year_duration(self, m):
        """na 2,5 roku → na dwa i pół roku."""
        prep, raw, unit = m.group(1), m.group(2), m.group(3)
        _, clean = _parse_raw(raw)
        return f"{prep} {_fraction_words(clean)} {unit}"

    def _format_unit(self, key: str, n_int: int, case: str = "nom") -> str:
        """Pick the correct unit form for number n_int and grammatical case.

        Central helper replacing scattered _pick/_UNIT_CASE_FORMS lookups.
        """
        unit = _UNITS.get(key)
        if not unit:
            return ""
        if case in ("gen", "loc", "inst"):
            case_forms = self._UNIT_CASE_FORMS.get(unit[0])
            if case_forms and case in case_forms:
                cf = case_forms[case]
                return _pick(n_int, cf[0], cf[1], cf[2])
        return _pick(n_int, unit[0], unit[1], unit[2])

    def _gen_sg_unit(self, key: str) -> str:
        """Return genitive singular of a unit (for decimals/fractions).

        Replaces _masc_gen_sg(unit[0]) with registry-backed lookup.
        """
        forms = _REGISTRY.get_forms(key)
        if forms is not None:
            return forms.gen_sg
        unit = _UNITS.get(key)
        if unit is None:
            return ""
        return unit[1] if unit[3] == "f" else _masc_gen_sg(unit[0])

    def _prep_unit(self, m):
        """Przyimek + liczba + jednostka — case-aware inflection."""
        prep, raw, key = m.group(1), m.group(2).replace("\u00a0", ""), m.group(3)
        unit = _UNITS.get(key)
        if not unit: return m.group(0)
        gender = unit[3]
        case = self._PREP_CASE_MAP.get(prep.lower(), "gen")
        val, clean = _parse_raw(raw)
        n_int = int(val)

        # Półtora / "i pół" logic for X.5
        clean_dot = clean.replace(",", ".")
        if "." in clean_dot:
            dec_part = clean_dot.split(".")[1]
            int_part = int(clean_dot.split(".")[0])
            if dec_part == "5":
                gen_sg = self._gen_sg_unit(key)
                if int_part == 0:
                    return f"{prep} pół {gen_sg}"
                if int_part == 1:
                    return f"{prep} półtora {gen_sg}"
                return f"{prep} {_cardinal_inflect(int_part, case, gender)} i pół {gen_sg}"

        if val == n_int:
            num_word = _cardinal_inflect(n_int, case, gender)
            unit_form = self._format_unit(key, n_int, case)
            return f"{prep} {num_word} {unit_form}"
        # Tech units (non-.5): use "N przecinek M" format
        if key in self._TECH_UNITS:
            gen_sg = self._gen_sg_unit(key)
            return f"{prep} {_n2w_float(clean, gender)} {gen_sg}"
        # Decimals with units
        gen_sg = self._gen_sg_unit(key)
        clean_dot = clean.replace(",", ".")
        if "." in clean_dot:
            int_s, dec_s = clean_dot.split(".", 1)
            int_part = int(int_s or "0")
            if len(dec_s) == 1:
                # Single decimal: "N i X dziesiątych unit-gen-sg" (genitive for prep context)
                dec_val = int(dec_s)
                int_w = _cardinal_inflect(int_part, case, gender) if int_part > 0 else "zera"
                dec_w = _cardinal_inflect(dec_val, "gen", "f")
                return f"{prep} {int_w} i {dec_w} dziesiątych {gen_sg}"
        # 2+ decimal digits: "N przecinek M unit-gen-sg"
        return f"{prep} {_n2w_float(clean, gender)} {gen_sg}"

    def _prep_num_inst(self, m):
        """Instrumental preposition + number: z 5 nowymi → z pięcioma nowymi."""
        prep, n = m.group(1), int(m.group(2).replace("\u00a0", ""))
        return f"{prep} {_cardinal_inflect(n, 'inst')}"

    def _prep_num_decimal(self, m):
        """Przyimek + liczba dziesiętna: około 3,14 → około trzy przecinek czternaście."""
        prep, raw = m.group(1), m.group(2)
        case = self._PREP_CASE_MAP.get(prep.lower(), "gen")
        val, clean = _parse_raw(raw)
        clean_dot = clean.replace(",", ".")
        # .5 → "pięciu i pół" etc.
        if "." in clean_dot:
            int_s, dec_s = clean_dot.split(".", 1)
            int_part = int(int_s or "0")
            if dec_s == "5":
                if int_part == 0:
                    return f"{prep} pół"
                if int_part == 1:
                    return f"{prep} półtora"
                return f"{prep} {_cardinal_inflect(int_part, case)} i pół"
        return f"{prep} {_n2w_float(raw)}"

    def _prep_num(self, m):
        """Przyimek + liczba — case-aware: do 15 → do piętnastu, ponad 50 000 → ponad pięćdziesiąt tysięcy."""
        prep, n = m.group(1), int(m.group(2).replace("\u00a0", ""))
        case = self._PREP_CASE_MAP.get(prep.lower(), "gen")
        # "około N do M" (range) → use nominative for first number
        after = m.string[m.end():]
        if prep.lower() in ("około", "ok.") and re.match(r"\s+do\s+\d", after):
            case = "nom"
        return f"{prep} {_cardinal_inflect(n, case)}"

    def _prep_neg_num(self, m):
        """Przyimek + ujemna liczba: do -15 → do minus piętnastu."""
        prep, n = m.group(1), int(m.group(2).replace("\u00a0", ""))
        return f"{prep} minus {_cardinal_gen(n)}"

    def _degree_standalone(self, m):
        """90° → dziewięćdziesiąt stopni (stopień bez jednostki temperatury).
        If preceded by a governing noun (e.g. kątem), inflect accordingly."""
        raw = m.group(1)
        val, clean = _parse_raw(raw)
        n_int = int(val)
        unit = _UNITS["°"]
        gender = unit[3]
        # Check for preceding governing noun
        before = m.string[max(0, m.start() - 30):m.start()].rstrip()
        prev_word = before.rsplit(None, 1)[-1].lower() if before else ""
        if prev_word in self._NOUN_CASE_MAP:
            case, _ = self._NOUN_CASE_MAP[prev_word]
            gen_sg = self._gen_sg_unit("°")
            if val == n_int:
                num_word = _cardinal_inflect(n_int, case, gender)
                unit_form = self._format_unit("°", n_int, case)
                return f"{num_word} {unit_form}"
            # Decimal with governing noun
            clean_dot = clean.replace(",", ".")
            if "." in clean_dot:
                int_s, dec_s = clean_dot.split(".", 1)
                int_part = int(int_s or "0")
                if dec_s == "5":
                    if int_part == 0:
                        return f"pół {gen_sg}"
                    if int_part == 1:
                        return f"półtora {gen_sg}"
                    return f"{_cardinal_inflect(int_part, case, gender)} i pół {gen_sg}"
            return f"{_n2w_float(clean, gender)} {gen_sg}"
        if val == n_int:
            return f"{_n2w(n_int, gender)} {_pick(n_int, unit[0], unit[1], unit[2])}"
        return f"{_n2w_float(clean, gender)} {unit[1]}"

    # Technical units: always use "przecinek" format for decimals (not "i X dziesiątych")
    _TECH_UNITS = {"GB", "MB", "TB", "kB", "GHz", "kHz", "Mb", "Gb", "Mbps", "Gbps"}

    def _unit(self, m):
        raw, key = m.group(1), m.group(2)
        unit = _UNITS.get(key)
        if not unit: return m.group(0)
        gender = unit[3]
        val, clean = _parse_raw(raw)
        n_int = int(val)
        if val == n_int:
            # Check for locative context: "w + adjective(s) + N unit"
            before = m.string[max(0, m.start() - 40):m.start()].lower().rstrip()
            if re.search(r'\bw\s+\w+(?:ych|ich|nym|nej|nych)\s*$', before):
                num_word = _cardinal_inflect(n_int, "loc")
                unit_form = self._format_unit(key, n_int, "loc")
                return f"{num_word} {unit_form}"
            return f"{_n2w(n_int, gender)} {_pick(n_int, unit[0], unit[1], unit[2])}"
        # X,5 → półtora or "X i pół"
        clean_dot = clean.replace(",", ".")
        if "." in clean_dot:
            dec_part = clean_dot.split(".")[1]
            int_part = int(clean_dot.split(".")[0])
            if dec_part == "5":
                gen_sg = self._gen_sg_unit(key)
                if int_part == 0:
                    return f"pół {gen_sg}"
                if int_part == 1:
                    return f"półtora {gen_sg}"
                return f"{_n2w(int_part, gender)} i pół {gen_sg}"
        # Tech units (non-.5): use "N przecinek M" format
        if key in self._TECH_UNITS:
            gen_sg = self._gen_sg_unit(key)
            return f"{_n2w_float(clean, gender)} {gen_sg}"
        # Other decimals with units
        gen_sg = self._gen_sg_unit(key)
        clean_dot = clean.replace(",", ".")
        if "." in clean_dot:
            int_s, dec_s = clean_dot.split(".", 1)
            int_part = int(int_s or "0")
            if len(dec_s) == 1:
                # Single decimal: "N i X dziesiąte unit-gen-sg" (nominative fractional)
                dec_val = int(dec_s)
                int_w = _n2w(int_part, gender) if int_part > 0 else "zero"
                dec_w = _n2w(dec_val, "f")
                frac_form = _pick(dec_val, "dziesiąta", "dziesiąte", "dziesiątych")
                return f"{int_w} i {dec_w} {frac_form} {gen_sg}"
        # 2+ decimal digits: "N przecinek M unit-gen-sg"
        return f"{_n2w_float(clean, gender)} {gen_sg}"

    def _blood_pressure(self, m):
        """Blood pressure: 120/80 mmHg → sto dwadzieścia na osiemdziesiąt milimetrów słupa rtęci."""
        n1, n2, key = int(m.group(1)), int(m.group(2)), m.group(3)
        unit = _UNITS.get(key)
        if not unit:
            return m.group(0)
        return f"{_n2w(n1)} na {_n2w(n2)} {unit[2]}"

    def _fuel_consumption(self, m):
        """Fuel consumption: 6,5 l/100 km → sześć i pół litra na sto kilometrów."""
        raw = m.group(1)
        val, clean = _parse_raw(raw)
        l_unit = _UNITS["l"]
        km_unit = _UNITS["km"]
        n_int = int(val)
        if val == n_int:
            l_form = _pick(n_int, l_unit[0], l_unit[1], l_unit[2])
            return f"{_n2w(n_int, l_unit[3])} {l_form} na sto {km_unit[2]}"
        # Use "i pół" format for .5 decimals, genitive singular for unit
        gen_sg = self._gen_sg_unit("l")
        clean_dot = clean.replace(",", ".")
        if "." in clean_dot and clean_dot.split(".")[1] == "5":
            int_part = int(clean_dot.split(".")[0])
            if int_part == 0:
                return f"pół {gen_sg} na sto {km_unit[2]}"
            if int_part == 1:
                return f"półtora {gen_sg} na sto {km_unit[2]}"
            return f"{_n2w(int_part, l_unit[3])} i pół {gen_sg} na sto {km_unit[2]}"
        return f"{_n2w_float(clean, l_unit[3])} {gen_sg} na sto {km_unit[2]}"

    def _per_minute(self, m):
        """Rate per minute: uderzenia/min → uderzeń na minutę."""
        noun = m.group(1)
        return f"{noun} na minutę"

    # Feminine nouns that require feminine ordinal endings in locative
    _FEMININE_LOC_NOUNS = frozenset({
        "połowie", "połowy", "edycji",
        "lidze", "stronie", "klasie", "grupie", "sekcji", "kategorii",
        "kolejce", "sesji", "kadencji", "turze", "rundzie", "fazie",
        "części", "wojnie", "erze", "epoce",
    })

    def _ordinal_loc_prep(self, m):
        """W 2. słowo → W drugim/drugiej słowie (miejscownik ordinalny, gender-aware)."""
        prep, n = m.group(1), int(m.group(2))
        # Check following noun for gender
        after = m.string[m.end():].lstrip()
        next_word = after.split()[0].lower().rstrip(".,!?;:") if after.split() else ""
        is_feminine = next_word in self._FEMININE_LOC_NOUNS
        gender = "f" if is_feminine else "m"
        return f"{prep} {_ordinal_inflect(n, 'loc', gender)} "

    # Feminine nouns that require feminine ordinal endings
    _FEMININE_NOUNS = frozenset({
        "edycję", "edycja", "edycji",
        "klasę", "klasa", "klasy",
        "grupę", "grupa", "grupy",
        "turę", "tura", "tury",
        "rundę", "runda", "rundy",
        "fazę", "faza", "fazy",
        "wojnę", "wojna", "wojny",
        "erę", "era", "ery",
    })

    # Rzeczowniki nijake — po nich ordinal w formie nijakiej
    _NEUTER_NOUNS = {
        "miejsce", "zadanie", "pytanie", "ćwiczenie", "zdanie",
        "pole", "miasto", "państwo", "okno", "centrum", "biuro",
        "prawo", "słowo", "wyjście", "badanie", "osiągnięcie",
    }

    def _ordinal(self, m):
        n = int(m.group(1))
        if n == 0:
            return "zero "
        ordinal = num2words(n, lang="pl", to="ordinal")
        # Sprawdź czy następne słowo jest rodzaju nijakiego
        after = m.string[m.end():].lstrip()
        next_word = after.split()[0].lower().rstrip(".,!?;:") if after.split() else ""
        if next_word in self._NEUTER_NOUNS:
            # Zamień końcówkę na nijaką: -y → -e, -i → -ie
            if ordinal.endswith("y"):
                ordinal = ordinal[:-1] + "e"
            elif ordinal.endswith("i"):
                ordinal = ordinal[:-1] + "ie"
        elif next_word in self._FEMININE_NOUNS:
            # Feminine: -y → -a, -i → -a (nom: piąty → piąta, trzeci → trzecia)
            words = ordinal.split()
            fem_words = []
            for w in words:
                if w.endswith("y"):
                    fem_words.append(w[:-1] + "a")
                elif w.endswith("i"):
                    fem_words.append(w[:-1] + "a")
                else:
                    fem_words.append(w)
            ordinal = " ".join(fem_words)
        return ordinal + " "

    def _range(self, m):
        return f"{_n2w(int(m.group(1)))} do {_cardinal_gen(int(m.group(2)))}"

    def _neg_number(self, m):
        """Ujemna liczba bez jednostki: -15 → minus piętnaście."""
        raw = m.group(1)
        val, clean = _parse_raw(raw)
        n_int = int(val)
        if val == n_int:
            return f"minus {_n2w(n_int)}"
        return f"minus {_n2w_float(clean)}"

    def _pos_number(self, m):
        """Dodatnia liczba z plusem bez jednostki: +49 → plus czterdzieści dziewięć."""
        raw = m.group(1)
        val, clean = _parse_raw(raw)
        n_int = int(val)
        if val == n_int:
            return f"plus {_n2w(n_int)}"
        return f"plus {_n2w_float(clean)}"

    def _thousands(self, m):
        raw = m.group(1).replace("\u00a0","").replace(" ","")
        try:    return _n2w(int(raw))
        except: return m.group(0)

    def _scale_context(self, m):
        """Scale with context word: 'w skali 1:10' → 'w skali jeden do dziesięciu'."""
        prefix = m.group(1)
        n1 = int(m.group(2))
        raw2 = m.group(3).replace(" ", "").replace("\u00a0", "")
        n2 = int(raw2)
        return f"{prefix} {_n2w(n1)} do {_cardinal_gen(n2)}"

    def _scale_large(self, m):
        """Large scale: 1:50 000 → jeden do pięćdziesięciu tysięcy."""
        n1 = int(m.group(1))
        raw2 = m.group(2).replace(" ", "").replace("\u00a0", "")
        n2 = int(raw2)
        return f"{_n2w(n1)} do {_cardinal_gen(n2)}"

    def _curr_postfix(self, m):
        """Postfix currency: 100$ → sto dolarów, 50£ → pięćdziesiąt funtów."""
        raw, sym = m.group(1), m.group(2)
        cur = _CURRENCIES.get(sym)
        if not cur:
            return m.group(0)
        return self._fmt_curr(raw, cur)

    def _resolution_with_unit(self, m):
        """Dimensions with unit: 10 x 15 x 20 cm → dziesięć na piętnaście na dwadzieścia centymetrów."""
        a, b = int(m.group(1)), int(m.group(2))
        c = int(m.group(3)) if m.group(3) else None
        key = m.group(4)
        unit = _UNITS.get(key)
        last = c if c is not None else b
        if unit:
            unit_word = _pick(last, unit[0], unit[1], unit[2])
        else:
            unit_word = key
        if c is not None:
            return f"{_n2w(a)} na {_n2w(b)} na {_n2w(c)} {unit_word}"
        return f"{_n2w(a)} na {_n2w(b)} {unit_word}"

    def _resolution(self, m):
        """Resolution: 1920x1080 → tysiąc dziewięćset dwadzieścia na tysiąc osiemdziesiąt."""
        a, b = int(m.group(1)), int(m.group(2))
        c = int(m.group(3)) if m.group(3) else None
        if c is not None:
            return f"{_n2w(a)} na {_n2w(b)} na {_n2w(c)}"
        return f"{_n2w(a)} na {_n2w(b)}"

    def _scale_or_score(self, m):
        """Scale 1:50000 → jeden do pięćdziesięciu tysięcy / score 3:0 → trzy do zera.
        Distinguish by: if second number > 99 or has spaces → scale. Otherwise → score."""
        raw2 = m.group(2).replace(" ", "").replace("\u00a0", "")
        n1 = int(m.group(1))
        try:
            n2 = int(raw2)
        except ValueError:
            return m.group(0)
        if n2 > 99:
            # Scale: 1:50000 → jeden do pięćdziesięciu tysięcy
            return f"{_n2w(n1)} do {_cardinal_gen(n2)}"
        # Small numbers: treat as score (3:0 → trzy do zera)
        return f"{_n2w(n1)} do {_cardinal_gen(n2)}"

    def _ratio(self, m):
        """General ratio: 1:10000 → jeden do dziesięciu tysięcy."""
        n1, n2 = int(m.group(1)), int(m.group(2))
        return f"{_n2w(n1)} do {_cardinal_gen(n2)}"

    def _inches(self, m):
        """Inches: 55" → pięćdziesiąt pięć cali."""
        raw = m.group(1)
        val, clean = _parse_raw(raw)
        n = int(val) if val == int(val) else val
        if isinstance(n, int):
            inch = _pick(n, "cal", "cale", "cali")
            return f"{_n2w(n)} {inch}"
        return f"{_n2w_float(clean)} cala"

    def _drive_letter(self, m):
        """Drive letter: C: → ce."""
        letter = m.group(1).upper()
        return _LETTER_NAMES_PL.get(letter, letter.lower())

    def _postal_code(self, m):
        """Polish postal code: 00-123 → zero zero sto dwadzieścia trzy, 31-001 → trzydzieści jeden zero zero jeden."""
        d1, d2 = m.group(1), m.group(2)
        # First part: if starts with 0, spell digit by digit; otherwise as number
        if d1.startswith("0"):
            part1 = " ".join(num2words(int(d), lang="pl") for d in d1)
        else:
            part1 = _n2w(int(d1))
        # Second part: if has leading zeros, spell digit by digit; otherwise as number
        if d2.startswith("0"):
            part2 = " ".join(num2words(int(d), lang="pl") for d in d2)
        else:
            part2 = _n2w(int(d2))
        return f"{part1} - {part2}"

    def _prep_duration_hm(self, m):
        """Preposition + duration: około 2h 30m → około dwie godziny i trzydzieści minut."""
        prep = m.group(1)
        hours, minutes = int(m.group(2)), int(m.group(3))
        h_word = _pick(hours, "godzinę", "godziny", "godzin")
        m_word = _pick(minutes, "minutę", "minuty", "minut")
        h_num = "jedną" if hours == 1 else _n2w(hours, 'f')
        m_num = "jedną" if minutes == 1 else _n2w(minutes, 'f')
        return f"{prep} {h_num} {h_word} i {m_num} {m_word}"

    def _duration_hm(self, m):
        """Duration: 2h i 15m → dwie godziny i piętnaście minut."""
        hours, minutes = int(m.group(1)), int(m.group(2))
        h_word = _pick(hours, "godzinę", "godziny", "godzin")
        m_word = _pick(minutes, "minutę", "minuty", "minut")
        # Accusative: "jedną godzinę" not "jedna godzinę"
        h_num = "jedną" if hours == 1 else _n2w(hours, 'f')
        m_num = "jedną" if minutes == 1 else _n2w(minutes, 'f')
        return f"{h_num} {h_word} i {m_num} {m_word}"

    def _duration_ms(self, m):
        """Duration after 'w': w 45:30 → w czterdzieści pięć minut trzydzieści sekund.
        Only matches when first number > 23 (can't be a time)."""
        minutes, seconds = int(m.group(1)), int(m.group(2))
        m_word = _pick(minutes, "minutę", "minuty", "minut")
        s_word = _pick(seconds, "sekundę", "sekundy", "sekund")
        result = f"w {_n2w(minutes, 'f')} {m_word}"
        if seconds > 0:
            result += f" {_n2w(seconds, 'f')} {s_word}"
        return result

    def _grade_minus(self, m):
        """School grade with minus: 5- → pięć minus."""
        n = int(m.group(1))
        return f"{_n2w(n)} minus"

    def _triple_dash(self, m):
        """Triple-dash notation: 10-10-10 → dziesięć - dziesięć - dziesięć.
        Skip if it looks like a date (DD-MM-YY): middle=1-12, first=1-31, third=0-99 with len==2."""
        n1, n2, n3 = int(m.group(1)), int(m.group(2)), int(m.group(3))
        g3 = m.group(3)
        # Skip if it looks like a date DD-MM-YY (all different, plausible day/month/year)
        if (1 <= n1 <= 31 and 1 <= n2 <= 12 and len(g3) == 2
                and not (n1 == n2 == n3)):  # repeated numbers like 10-10-10 are NOT dates
            return m.group(0)
        return f"{_n2w(n1)} - {_n2w(n2)} - {_n2w(n3)}"

    # ── New handlers for noun-mediated case, special formats ───────────

    def _time_range_godz(self, m):
        """Time range with godz prefix: godz. 08:00-22:00 → w godzinach ósma - dwudziesta druga."""
        h1, m1 = int(m.group(2)), int(m.group(3))
        h2, m2 = int(m.group(4)), int(m.group(5))
        # Use ordinal hour forms for time ranges
        t1 = self._time_words(h1, m1, hour_case="nom")
        t2 = self._time_words(h2, m2, hour_case="nom")
        return f"w godzinach {t1} - {t2}"

    def _tire_dimensions(self, m):
        """Tire dimensions: 225/45 R17 → dwieście dwadzieścia pięć na czterdzieści pięć er siedemnaście."""
        width, profile, rim = int(m.group(1)), int(m.group(2)), int(m.group(3))
        return f"{_n2w(width)} na {_n2w(profile)} er {_n2w(rim)}"

    def _distance_decompose_km(self, m):
        """Distance decomposition: 42,195 km → czterdzieści dwa kilometry sto dziewięćdziesiąt pięć metrów."""
        km_val = int(m.group(1))
        m_val = int(m.group(2))
        km_unit = _UNITS.get("km")
        m_unit = _UNITS.get("m")
        km_form = _pick(km_val, km_unit[0], km_unit[1], km_unit[2])
        m_form = _pick(m_val, m_unit[0], m_unit[1], m_unit[2])
        return f"{_n2w(km_val)} {km_form} {_n2w(m_val)} {m_form}"

    def _range_decimal_unit(self, m):
        """Decimal range + unit: od 1,5 do 2 h → od półtora do dwóch godzin."""
        prep1, raw_dec, prep2, raw_int, key = m.group(1), m.group(2), m.group(3), m.group(4), m.group(5)
        unit = _UNITS.get(key)
        if not unit:
            return m.group(0)
        gender = unit[3]
        gen_sg = self._gen_sg_unit(key)
        # First number: decimal with "pół" format if .5
        val, clean = _parse_raw(raw_dec)
        clean_dot = clean.replace(",", ".")
        if "." in clean_dot and clean_dot.split(".")[1] == "5":
            int_part = int(clean_dot.split(".")[0])
            if int_part == 0:
                first = "pół"
            elif int_part == 1:
                first = "półtora"
            else:
                first = f"{_n2w(int_part)} i pół"
        else:
            first = _n2w_float(clean, gender)
        # Second number: genitive
        n2 = int(raw_int)
        second = _cardinal_inflect(n2, "gen", gender)
        unit_form = self._format_unit(key, n2, "gen")
        return f"{prep1} {first} {prep2} {second} {unit_form}"

    def _noun_num_unit(self, m):
        """Noun-governed number + unit: pojemność 32 GB → pojemność trzydziestu dwóch gigabajtów."""
        noun, raw, key = m.group(1), m.group(2).replace("\u00a0", ""), m.group(3)
        noun_lower = noun.lower()
        if noun_lower not in self._NOUN_CASE_MAP:
            return m.group(0)  # Not a governing noun, let generic _unit handle it
        # Skip if the noun is actually a preposition itself (prep_unit handles those)
        if noun_lower in self._PREP_CASE_MAP:
            return m.group(0)
        case, _ = self._NOUN_CASE_MAP[noun_lower]
        unit = _UNITS.get(key)
        if not unit:
            return m.group(0)
        gender = unit[3]
        val, clean = _parse_raw(raw)
        n_int = int(val)
        if val == n_int:
            num_word = _cardinal_inflect(n_int, case, gender)
            unit_form = self._format_unit(key, n_int, case)
            return f"{noun} {num_word} {unit_form}"
        # Decimals: handle .5 and fractional forms
        gen_sg = self._gen_sg_unit(key)
        clean_dot = clean.replace(",", ".")
        if "." in clean_dot:
            int_s, dec_s = clean_dot.split(".", 1)
            int_part = int(int_s or "0")
            if dec_s == "5":
                if int_part == 0:
                    return f"{noun} pół {gen_sg}"
                if int_part == 1:
                    return f"{noun} półtora {gen_sg}"
                return f"{noun} {_cardinal_inflect(int_part, case, gender)} i pół {gen_sg}"
            # Tech units (non-.5): use "N przecinek M" format
            if key in self._TECH_UNITS:
                return f"{noun} {_n2w_float(clean, gender)} {gen_sg}"
            if len(dec_s) == 1:
                dec_val = int(dec_s)
                int_w = _cardinal_inflect(int_part, case, gender) if int_part > 0 else "zera"
                dec_w = _cardinal_inflect(dec_val, "gen", "f")
                return f"{noun} {int_w} i {dec_w} dziesiątych {gen_sg}"
        return f"{noun} {_n2w_float(clean, gender)} {gen_sg}"

    def _num_spelled_unit(self, m):
        """Number + spelled-out unit: 256 bit → dwieście pięćdziesiąt sześć bitów."""
        raw, unit_word = m.group(1).replace("\u00a0", ""), m.group(2).lower()
        n = int(raw)
        su = self._SPELLED_UNITS.get(unit_word)
        if not su:
            return m.group(0)
        return f"{_n2w(n, su[3])} {_pick(n, su[0], su[1], su[2])}"

    def _date_no_year(self, m):
        """DD.MM date without year: na 15.04 → na piętnastego kwietnia."""
        prep = m.group(1)
        d, mo = int(m.group(2)), int(m.group(3))
        if not (1 <= mo <= 12 and 1 <= d <= 31):
            return m.group(0)
        return f"{prep} {_ORD_GEN_DAYS.get(d, str(d))} {_MONTHS_GEN[mo]}"

    def _exchange_rate(self, m):
        """Exchange rate: po kursie 3,95 → po kursie trzy złote i dziewięćdziesiąt pięć groszy."""
        prefix = m.group(1)
        zl, gr = int(m.group(2)), int(m.group(3))
        cur = _CURRENCIES.get("zł")
        zl_form = _pick(zl, cur[0], cur[1], cur[2])
        gr_form = _pick(gr, cur[3], cur[4], cur[5])
        return f"{prefix} {_n2w(zl)} {zl_form} i {_n2w(gr)} {gr_form}"

    def _decimal_temporal(self, m):
        """Decimal + temporal unit: 5,27 lat → pięć i dwadzieścia siedem setnych lat."""
        int_part, dec_part, unit_word = int(m.group(1)), m.group(2), m.group(3)
        dec_val = int(dec_part)
        # Determine "setnych" denominator based on number of decimal digits
        denom = "setnych" if len(dec_part) == 2 else "tysięcznych"
        return f"{_n2w(int_part)} i {_n2w(dec_val)} {denom} {unit_word}"

    # Hallmark/purity numbers: codes, not quantities — use nominative
    _HALLMARK_NUMBERS = frozenset({333, 375, 500, 585, 750, 900, 999})

    def _noun_num_ordinal(self, m):
        """Noun-governed ordinal: w linii 128 → w linii sto dwudziestej ósmej."""
        noun, raw = m.group(1), m.group(2)
        noun_lower = noun.lower()
        if noun_lower not in self._NOUN_CASE_MAP:
            return m.group(0)  # Not a governing noun, let generic _number handle it
        # Verify the number looks like an ordinal context (not too large, not a year)
        n = int(raw)
        if n > 9999 or (1000 <= n <= 2100):
            return m.group(0)  # Likely a year or very large number
        case, num_type = self._NOUN_CASE_MAP[noun_lower]
        # Hallmark/purity numbers after "próby" → nominative (codes, not quantities)
        if noun_lower == "próby" and n in self._HALLMARK_NUMBERS:
            return f"{noun} {_n2w(n)}"
        n = int(raw)
        if num_type == "ordinal":
            # Determine gender: -ii/-ji → feminine (linii, pozycji), else masculine
            gender = "f" if noun_lower.endswith(("ii", "ji")) else "m"
            return f"{noun} {_ordinal_inflect(n, case, gender)}"
        else:
            # Cardinal in the specified case
            return f"{noun} {_cardinal_inflect(n, case)}"

    def _number(self, m):
        raw = m.group(1)
        val, clean = _parse_raw(raw)
        n_int = int(val)
        if val == n_int:
            # Jawne ".0" lub ",0" — np. wersja 4.0 → cztery kropka zero
            if ("." in clean or "," in clean):
                return f"{_n2w(n_int)} kropka zero"
            # Conjunction continuation: "i N" after genitive context → genitive
            # e.g. "z dwóch atomów wodoru i 1 tlenu" → "i jednego"
            before = m.string[max(0, m.start() - 60):m.start()].lower().rstrip()
            if re.search(r'\bi\s*$', before):
                # Check further back for genitive number forms
                before_i = before[:before.rfind('i')].rstrip()
                if re.search(r'\b(?:dwóch|trzech|czterech|pięciu|sześciu|siedmiu|ośmiu|dziewięciu|dziesięciu|jedenastu|dwunastu|trzynastu|czternastu|piętnastu|szesnastu|siedemnastu|osiemnastu|dziewiętnastu|dwudziestu)\b', before_i):
                    return _cardinal_inflect(n_int, "gen")
            return _n2w(n_int)
        # X,5 → "N i pół" (np. 4,5 → cztery i pół)
        clean_dot = clean.replace(",", ".")
        if "." in clean_dot and clean_dot.split(".")[1] == "5":
            int_part = int(clean_dot.split(".")[0])
            if int_part == 0:
                return "zero przecinek pięć"
            return f"{_n2w(int_part)} i pół"
        # Ułamki dziesiętne: "przecinek" format (99,9 → dziewięćdziesiąt dziewięć przecinek dziewięć)
        return _n2w_float(clean)

    def normalize(self, text: str) -> str:
        result = text
        # Zamień spacje w liczbach na NBSP (nierozdzielającą spację)
        # żeby _NUM pattern (\d{1,3}(?:[\u00a0 ]\d{3})*) je dopasował poprawnie
        # a prep patterns (ponad \d+) nie złapały pierwszej cyfry osobno
        def _merge_thousands(m):
            return m.group(0).replace(" ", "\u00a0")
        result = re.sub(r'\b(\d{1,3}(?:\s\d{3})+)\b', _merge_thousands, result)
        for pat, fn in self._pats:
            result = pat.sub(fn, result)
        result = re.sub(r"\br\.\s*", "", result)   # usuń osierocone "r."
        # Nieskończoność po przyimku → dopełniacz: "do nieskończoność" → "do nieskończoności"
        result = re.sub(
            r"\b(do|bez|od)\s+nieskończoność\b",
            lambda m: f"{m.group(1)} nieskończoności",
            result, flags=re.IGNORECASE
        )
        # Remaining superscript digits (not consumed by unit patterns like m², km²)
        _SUPER_MAP = {"⁰": "0", "¹": "1", "²": "2", "³": "3", "⁴": "4",
                      "⁵": "5", "⁶": "6", "⁷": "7", "⁸": "8", "⁹": "9"}
        def _expand_superscript(m):
            base = m.group(1)
            sup = "".join(_SUPER_MAP.get(c, c) for c in m.group(2))
            if sup == "2":
                return f"{base} kwadrat"
            elif sup == "3":
                return f"{base} sześcian"
            else:
                return f"{base} do potęgi {sup}"
        result = re.sub(r"(\w)([⁰¹²³⁴⁵⁶⁷⁸⁹]+)", _expand_superscript, result)
        # Formula variable spelling: single lowercase letters in math context → Polish names
        _MATH_LETTERS = {
            'x': 'iks', 'y': 'igrek', 'z': 'zet', 'r': 'er',
            'n': 'en', 't': 'te', 'a': 'a', 'b': 'be', 'c': 'ce',
            'f': 'ef', 'g': 'gie', 'h': 'ha', 'k': 'ka', 'p': 'pe',
            's': 'es', 'v': 'fał', 'd': 'de', 'e': 'e', 'l': 'el',
            'm': 'em', 'q': 'ku', 'u': 'u', 'w': 'wu',
        }
        if "kwadrat" in result or "sześcian" in result or "równa się" in result:
            def _spell_math_var(m):
                name = _MATH_LETTERS.get(m.group(1), m.group(1))
                return f"{name} {m.group(2)}"
            result = re.sub(r"\b([a-z])\s+(kwadrat|sześcian|do potęgi)\b", _spell_math_var, result)
            def _spell_eq_var(m):
                return _MATH_LETTERS.get(m.group(1), m.group(1))
            result = re.sub(r"(?<=\s)([a-z])(?=\s+(?:kwadrat|sześcian|plus|minus|razy|równa|do potęgi))", _spell_eq_var, result)
            result = re.sub(r"((?:się|plus|minus|razy)\s)([a-z])(?=[\s.,;:!?)]|$)", lambda m: m.group(1) + _MATH_LETTERS.get(m.group(2), m.group(2)), result)
        return result


# ─────────────────────────────────────────────────────────────────────────────
# CELL 6 ▸ Główny potok: PolishTTSPipeline
# ─────────────────────────────────────────────────────────────────────────────

class PolishTTSPipeline:
    """
    Kompletny potok normalizacji tekstu dla polskiego TTS + PLTokenizer.

    Kolejność kroków:
      1. raw_clean()     — unicode, myślniki, cudzysłowy, znaki obce
      2. num_normalize() — liczby, daty, % → słowa PL
      3. final_filter()  — usunięcie znaków spoza vocab tokenizera

    Użycie:
        pipe = PolishTTSPipeline()
        clean = pipe.process("Dnia 15.03.2024 r. zysk wyniósł 3,5 mld zł.")
        # → "Dnia piętnastego marca dwa tysiące dwudziestego czwartego roku
        #     zysk wyniósł trzy przecinek pięć miliardów złotych."
    """

    def __init__(self, keep_unknown_as_unk: bool = False, preserve_case: bool = True):
        """
        keep_unknown_as_unk : jeśli True, zamiast usuwać nieznane znaki zostawia
                              marker '<unk>' (pomocne przy debugowaniu).
        """
        self._num = _NumberNormalizer()
        self.keep_unknown_as_unk = keep_unknown_as_unk
        self.preserve_case = bool(preserve_case)

    @staticmethod
    def _split_preserved(text: list) -> list:
        return [p for p in re.split(r"(<[^<>]+>)", str(text or "")) if p]

    @staticmethod
    def _canonicalize_tag(tag: str) -> str:
        inner = str(tag or "").strip()[1:-1].strip()
        cf = inner.casefold()
        if cf == "sp":
            return "<sp>"
        if cf == "bos":
            return "<BOS>"
        if cf == "eos":
            return "<EOS>"
        if cf == "cap":
            return "<CAP>"
        if cf in {"nar", "akt"}:
            return f"<{cf}>"
        if cf.startswith("reserved") and cf[8:].isdigit():
            return f"<{cf}>"
        if cf in {"sz", "cz", "dz", "dź", "dż", "ch", "rz"}:
            return f"<{cf}>"
        return f"<{inner}>"

    def _process_plain(self, text: str) -> str:
        s = self.raw_clean(text)
        s = self.abbreviation_expand(s)
        s = self.foreign_expand(s)
        s = self.num_normalize(s)
        s = self.final_filter(s)
        if not self.preserve_case:
            s = s.lower()
        return s

    # ── KROK 1: surowe czyszczenie ────────────────────────────────────────────

    def raw_clean(self, text: str) -> str:
        """
        Normalizacja znaków:
          • NFC unicode
          • entery, tabulatory → spacja
          • wszystkie warianty myślników Unicode → -
          • znaki niemieckie (ö ä ü ß) → polskie odpowiedniki
          • cudzysłowy → usunięcie
          • znaki nie-łacińskie (koreańskie, chińskie, arabskie…) → usunięcie
          • collapse whitespace
        """
        if not text:
            return ""

        # NFC i normalizacja
        s = unicodedata.normalize("NFC", text)

        # Spacje niestandardowe i białe znaki → zwykła spacja
        s = s.translate(_SPACE_MAP)

        # Normalize digit : digit → digit:digit (for time patterns like "03: 14")
        s = re.sub(r'(\d)\s*:\s*(\d)', r'\1:\2', s)

        # Myślniki Unicode → -
        s = s.translate(_DASH_MAP)

        # Myślniki są już znormalizowane do ASCII '-' przez _DASH_MAP.
        # Nie robimy dalszych przekształceń semantycznych — to zadanie wyższych warstw.

        # Znaki niemieckie
        s = s.translate(_GERMAN_MAP)
        s = s.replace("ß", "ss")

        # Ampersand: directly between words (no spaces) → "i" (R&D, AT&T),
        # spaced or standalone → "ampersand" (Znak & nazywamy → Znak ampersand nazywamy)
        s = re.sub(r'(?<=\w)&(?=\w)', ' i ', s)
        s = re.sub(r'&', ' ampersand ', s)

        # Geographic coordinates: 52°13'N, 52° 13' 47" N → DMS format
        _GEO_LETTER = {"N": "en", "S": "es", "E": "e", "W": "wu"}
        def _geo_coord_dms(m):
            deg = m.group(1)
            minutes = m.group(2)
            seconds = m.group(3)
            direction = _GEO_LETTER.get(m.group(4).upper(), m.group(4))
            n_deg = int(deg)
            deg_word = _pick(n_deg, "stopień", "stopnie", "stopni")
            result = f"{deg} {deg_word}"
            if minutes:
                result += f" {minutes} minut"
            if seconds:
                result += f" {seconds} sekund"
            result += f" {direction}"
            return result
        # Full DMS: 52° 13' 47" N
        s = re.sub(r'(\d+)\s*°\s+(\d+)\s*[\'\u2032]\s+(\d+)\s*["\u201D\u201C\u2033]\s*([NSEWnsew])\b', _geo_coord_dms, s)
        # Degrees + minutes: 52°13'N or 52° 13' N
        def _geo_coord(m):
            deg = m.group(1)
            minutes = m.group(2)
            direction = _GEO_LETTER.get(m.group(3).upper(), m.group(3))
            n_deg = int(deg)
            deg_word = _pick(n_deg, "stopień", "stopnie", "stopni")
            if minutes:
                return f"{deg} {deg_word} {minutes} minut {direction}"
            return f"{deg} {deg_word} {direction}"
        s = re.sub(r"(\d+)°(\d+)?['\u2032]?\s*([NSEWnsew])\b", _geo_coord, s)

        # Inches: N" → N cali/cale/cal (before quote removal)
        def _inch_mark(m):
            d = m.group(1)
            # Simple approximation — full form handled by _inches in num_normalize
            # but since " is removed by quote filter, we must expand here
            n = int(d)
            inch = _pick(n, "cal", "cale", "cali")
            return d + ' ' + inch
        # Fraction-inches: 3/4" → 3/4 cala (before quote removal)
        s = re.sub(r'(\d+/\d+)\s*"', r'\1 cala', s)
        s = re.sub(r'(\d)\s*"', _inch_mark, s)
        s = re.sub(r"(\d)\s*[\u201D\u201C]", _inch_mark, s)  # also curly quotes

        # Opening quote after dialogue dash: - "Word → -, Word (comma = pause marker)
        # Before quote removal, insert comma as pause marker for the removed opening quote.
        # The comma survives whitespace collapsing and becomes double-space in final_filter.
        _Q_ALL = ''.join(sorted(_QUOTE_CHARS))
        s = re.sub(rf'(- )[{re.escape(_Q_ALL)}](?=\w)', r'\1, ', s)

        # Cudzysłowy → usunięcie
        s = "".join(ch for ch in s if ch not in _QUOTE_CHARS)

        # Ochrona URL-i przed rozwinięciem symboli specjalnych (// → spacja)
        # ISBN → "numer ISBN" (przed URL-ami i innymi rozwinięciami)
        s = re.sub(r"\bISBN[-: ]*[\dX][\d\-X]{9,16}\b", "numer ISBN", s, flags=re.IGNORECASE)

        def _spell_url(m):
            url = m.group(0)
            # Strip trailing sentence punctuation
            suffix = ''
            while url and url[-1] in '.!?,;:':
                suffix = url[-1] + suffix
                url = url[:-1]
            # Spell out protocol
            url = re.sub(r'^https://', 'ha te te pe es dwukropek ukośnik ukośnik ', url, flags=re.IGNORECASE)
            url = re.sub(r'^http://', 'ha te te pe dwukropek ukośnik ukośnik ', url, flags=re.IGNORECASE)
            # www → wu wu wu (may be at start or after protocol expansion)
            url = re.sub(r'\bwww\b', 'wu wu wu', url, flags=re.IGNORECASE)
            # Spell out dots and slashes
            url = url.replace('.', ' kropka ')
            url = url.replace('/', ' ukośnik ')
            url = url.replace('@', ' małpa ')
            url = url.replace('-', ' - ')
            url = re.sub(r'\s+', ' ', url).strip()
            return url + suffix
        s = re.sub(r"https?://[^\s]+", _spell_url, s, flags=re.IGNORECASE)
        s = re.sub(r"www\.[a-z0-9.-]+\.[a-z]{2,}(?:/[^\s]*)?", _spell_url, s,
                   flags=re.IGNORECASE)
        # Email addresses: user@domain.tld → spelled out
        def _spell_email(m):
            email = m.group(0)
            suffix = ''
            while email and email[-1] in '.!?,;:':
                suffix = email[-1] + suffix
                email = email[:-1]
            email = email.replace('.', ' kropka ')
            email = email.replace('@', ' małpa ')
            email = email.replace('-', ' - ')
            email = re.sub(r'\s+', ' ', email).strip()
            return email + suffix
        s = re.sub(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", _spell_email, s)

        # Filename extensions: word.ext → word kropka ext (before quote/dot removal)
        # Common file extensions
        _FILE_EXTS = r"(?:exe|pdf|doc|docx|txt|jpg|png|gif|mp3|mp4|avi|zip|rar|html|css|js|py|csv|xml|json|xlsx|pptx|wav|ogg|flac|apk|iso|dmg|deb|rpm|tar|gz|bz2)"
        s = re.sub(rf"(\w)\.({_FILE_EXTS})\b", r"\1 kropka \2", s, flags=re.IGNORECASE)
        # Standalone .ext at word boundary: ".exe" → "kropka exe"
        s = re.sub(rf"(?<=\s)\.({_FILE_EXTS})\b", r"kropka \1", s, flags=re.IGNORECASE)

        # Drive letters: C:\ → ce \ (expand BEFORE abbreviation_expand sees "C" as Roman numeral)
        s = re.sub(r"\b([A-Za-z]):\s*(?=[/\\])", lambda m: _LETTER_NAMES_PL.get(m.group(1).upper(), m.group(1).lower()) + " ", s)

        # Color codes (#FF5733) BEFORE special symbols expansion (# → krzyżyk)
        def _spell_color(m):
            digits = m.group(1)
            parts = ["krzyżyk"]
            for ch in digits:
                ch_upper = ch.upper()
                if ch.isdigit():
                    parts.append(_DIGIT_WORDS_PL[int(ch)])
                elif ch_upper in _LETTER_NAMES_PL:
                    parts.append(_LETTER_NAMES_PL[ch_upper])
                else:
                    parts.append(ch.lower())
            return " ".join(parts)
        s = re.sub(r"#([0-9A-Fa-f]{6})\b", _spell_color, s)

        # Rozwinięcie specjalnych symboli PRZED filtrem obcych skryptów
        s = self._expand_special_symbols(s)

        # Usunięcie znaków ze skryptów innych niż łaciński/polski
        # Zachowujemy: litery łacińskie + polskie diakrytyki + cyfry + allowed punct + spacja
        # Cyfry zachowujemy tutaj (zostaną zamienione na słowa w kroku 2)
        s = self._remove_foreign_scripts(s)

        # Add space after sentence-ending punctuation when missing before a letter
        # "psa.Pies" → "psa. Pies"  (but not "..." or "3.14" or "U.S.A." or "m.in.")
        # Wyklucz wzorzec litera-kropka-litera (skróty z kropkami): [A-Za-z].[A-Za-z]
        s = re.sub(r'(?<=[^.!?\s])(?<!\b[A-Za-z])([.!?])(?![A-Za-z]\.)'
                   r'(?=[A-ZĄĆĘŁŃÓŚŻŹ])', r'\1 ', s)

        # Add space after comma/semicolon/colon when followed directly by a letter
        # "szczeknal,spojrzal" → "szczeknal, spojrzal"
        s = re.sub(r'([,;:])([A-ZĄĆĘŁŃÓŚŻŹa-ząćęłńóśżź])', r'\1 \2', s)

        # Normalize whitespace
        s = re.sub(r" {2,}", " ", s).strip()
        return s

    @staticmethod
    def _expand_special_symbols(text: str) -> str:
        """
        Zamienia symbole specjalne na polski tekst PRZED filtrem obcych skryptów.
        Dzięki temu greckie litery, ≈, ∞ itp. nie zostaną usunięte bezśladu.
        """
        s = text

        # Wielokropek … jest w _TOKENIZER_PUNCT — zostaje jako sygnał pauzy/zawieszenia
        # (nie zamieniaj na spację — TTS model zna ten znak)

        # Indeksy dolne → zwykłe cyfry
        s = s.translate(str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789"))

        # µg, µm, µs etc. → micro-units (before Greek letter expansion)
        # Handle both U+00B5 (micro sign) and U+03BC (Greek mu)
        # Compound forms first: µg/m³ → mikrogramów na metr sześcienny
        s = re.sub(r"[µμ]g/m[³3]", "mikrogramów na metr sześcienny", s)
        s = re.sub(r"[µμ]g/l", "mikrogramów na litr", s)
        # Simple forms — must expand BEFORE _remove_foreign_scripts strips µ
        s = re.sub(r"[µμ]F", "mikrofarad", s)
        s = re.sub(r"[µμ]g", "mikrogram", s)
        s = re.sub(r"[µμ]m", "mikrometr", s)
        s = re.sub(r"[µμ]s", "mikrosekunda", s)
        s = re.sub(r"[µμ]l", "mikrolitr", s)

        # Greckie litery → polskie nazwy
        _GREEK = {
            "Δ": " delta ", "δ": " delta ",
            "Ω": " omega ", "ω": " omega ",
            "Ξ": " ksi ", "ξ": " ksi ",
            "α": " alfa ", "Α": " alfa ",
            "β": " beta ", "Β": " beta ",
            "γ": " gamma ", "Γ": " gamma ",
            "π": " pi ", "Π": " pi ",
            "σ": " sigma ", "Σ": " sigma ",
            "μ": " mi ", "Μ": " mi ",
            "Φ": " fi ", "φ": " fi ",
            "Θ": " theta ", "θ": " theta ",
            "Λ": " lambda ", "λ": " lambda ",
            "ε": " epsilon ", "Ε": " epsilon ",
        }
        for greek, pol in _GREEK.items():
            s = s.replace(greek, pol)

        # Litera łacińska bezpośrednio po nazwie greckiej: "delta T" → "delta te"
        _greek_names = r"(?:alfa|beta|gamma|delta|epsilon|theta|lambda|mi|pi|sigma|omega|fi|ksi)"
        s = re.sub(
            rf"\b({_greek_names})\s+([A-Z])\b",
            lambda m: f"{m.group(1)} {_LETTER_NAMES_PL.get(m.group(2), m.group(2).lower())}",
            s,
        )

        # § → paragraf
        s = re.sub(r"§\s*(\d+)", r"paragraf \1", s)
        s = s.replace("§", "paragraf")

        # ∞ → nieskończoność
        s = s.replace("∞", " nieskończoność ")

        # ≠ → nie równa się
        s = s.replace("≠", " nie równa się ")

        # ≤ → mniej lub równe, ≥ → więcej lub równe
        s = s.replace("≤", " mniej lub równe ")
        s = s.replace("≥", " więcej lub równe ")

        # × → razy (znak mnożenia)
        s = s.replace("×", " razy ")

        # ≈ → około (PRZED ~)
        s = s.replace("≈", " około ")

        # ~ przed liczbą → około, reszta → spacja
        s = re.sub(r"~\s*(?=\d)", "około ", s)
        s = s.replace("~", " ")

        # > < przed liczbami → ponad/poniżej, reszta → spacja
        s = re.sub(r">\s*(?=\d)", "ponad ", s)
        s = re.sub(r"<\s*(?=\d)", "poniżej ", s)
        s = s.replace(">", " ").replace("<", " ")

        # → ← strzałki → spacja
        s = s.replace("→", " ").replace("←", " ")

        # ^ → do potęgi (w kontekście liczbowym: 2^10, x^2)
        s = re.sub(r"(\d)\s*\^\s*(\d+)", lambda m: f"{m.group(1)} do potęgi {m.group(2)}", s)
        s = s.replace("^", " ")

        # | → przecinek
        s = s.replace("|", ", ")

        # = → "równa się" between word characters (formulas: E=mc², a=b)
        # Then spell out isolated letter sequences after "równa się" (mc → em ce)
        # Only match when LHS is a standalone word or single letter (not middle of EUR, PLN etc.)
        def _formula_equal(m):
            lhs = m.group(1)
            rhs = m.group(2)
            # Spell single uppercase letter on LHS
            if len(lhs) == 1 and lhs.isupper():
                lhs = _LETTER_NAMES_PL.get(lhs, lhs.lower())
            return f"{lhs} równa się {rhs}"
        s = re.sub(r"(\b\w+)\s*=\s*(\w)", _formula_equal, s)
        # Spell out isolated lowercase letter sequences (2-4 chars) that follow "równa się"
        # E.g., "równa się mc" → "równa się em ce"
        def _spell_formula_letters(m):
            prefix = m.group(1)
            letters = m.group(2)
            spelled = " ".join(_LETTER_NAMES_PL.get(ch.upper(), ch) for ch in letters)
            return f"{prefix}{spelled}"
        s = re.sub(r"(równa się\s+)([a-z]{2,4})(?=\s|[²³⁰¹⁴⁵⁶⁷⁸⁹]|$)", _spell_formula_letters, s)
        # == → spacja (remaining)
        s = s.replace("==", " ")
        # = → spacja (remaining standalone =)
        s = s.replace("=", " ")

        # Error codes: #DIV/0!, #N/A, #REF! → spell components
        # Letters are grouped into words: DIV → "div", REF → "ref"
        def _error_code(m):
            code = m.group(1)
            parts = ["krzyżyk"]
            # Split into segments by / and !
            segments = re.split(r'([/!])', code)
            for seg in segments:
                if seg == '/':
                    parts.append("ukośnik")
                elif seg == '!':
                    continue  # skip exclamation (pause handled by final_filter)
                elif seg.isdigit():
                    parts.append(_DIGIT_WORDS_PL[int(seg)])
                elif seg.isalpha():
                    # Group of letters → pronounce as word
                    if _is_pronounceable(seg):
                        parts.append(seg.lower())
                    else:
                        parts.append(_spell_abbreviation(seg))
                else:
                    # Mixed: spell digit-by-digit
                    for ch in seg:
                        if ch.isdigit():
                            parts.append(_DIGIT_WORDS_PL[int(ch)])
                        elif ch.isalpha():
                            parts.append(ch.lower())
            return " ".join(parts)
        s = re.sub(r"#([A-Z]+/[A-Z0-9]+)", _error_code, s)

        # # → "numer N" for #N, otherwise "krzyżyk"
        s = re.sub(r"#(\d+)", r"krzyżyk \1", s)
        s = s.replace("#", " krzyżyk ")

        # Standalone @ → "małpa" (when not in email context, already handled by _spell_email)
        s = re.sub(r"(?<![a-zA-Z0-9._%+-])@(?![a-zA-Z0-9.-])", " małpa ", s)

        # _ → spacja
        s = s.replace("_", " ")

        # // → spacja (po zamianie greki mogą powstać podwójne /)
        s = re.sub(r"/{2,}", " ", s)

        # Blood type: Rh-, Rh+ → er ha minus/plus (before + and - stripping)
        s = re.sub(r"\bRh([+-])", lambda m: f"er ha {'minus' if m.group(1) == '-' else 'plus'}", s)

        # Timezone offsets: GMT+1 → gie em te plus jeden, UTC-5 → u te ce minus pięć
        # Must be before generic +/- stripping
        s = re.sub(r"\b(GMT|UTC)\s*([+-])\s*(\d{1,2})\b",
            lambda m: f"{_spell_abbreviation(m.group(1))} {'plus' if m.group(2) == '+' else 'minus'} {num2words(int(m.group(3)), lang='pl')}",
            s)

        # + between numbers or with spaces → "plus", but keep +5, +$268 as-is
        s = re.sub(r"(?<=\d)\s*\+\s*(?=\d)", " plus ", s)
        # + after word (GMT+, UTC+) and before digit → "plus"
        s = re.sub(r"(?<=[A-Za-z])\+(?=\d)", " plus ", s)
        # + not immediately before digit/currency → "plus" (keeps +5, +$268 for signed patterns)
        s = re.sub(r"\+(?!\d|[$€¥£])", " plus ", s)

        # * between numbers/letters → "razy" (multiplication in formulas)
        s = re.sub(r"(?<=[\d\w])\s*\*\s*(?=[\d\w])", " razy ", s)
        s = s.replace("*", " ")

        # Myślniki są już znormalizowane do ASCII "-" przez _DASH_MAP.
        # Nie robimy dalszych przekształceń semantycznych — to zadanie wyższych warstw.

        return s

    @staticmethod
    def _remove_foreign_scripts(text: str) -> str:
        """
        Usuwa znaki spoza alfabetu łacińskiego i polskiego.
        Przepuszcza:
          - litery ASCII a-z A-Z
          - polskie diakrytyki: ą ć ę ł ń ó ś ź ż (+ wielkie)
          - cyfry 0-9
          - znaki z _TOKENIZER_PUNCT
          - spacja
        Usuwa: znaki koreańskie, chińskie, arabskie, cyrylicę, emoji itp.
        """
        _SAFE_LETTERS = (
            set(string.ascii_letters) |
            set("ąćęłńóśźżĄĆĘŁŃÓŚŹŻ")
        )
        _SAFE_DIGITS = set("0123456789")
        # Znaki tymczasowe — potrzebne dla num_normalize (°²³%$£€)
        # Jeśli nie zostaną skonwertowane na słowa, final_filter je usunie
        _NUM_NORM_PASS = set("°⁰¹²³⁴⁵⁶⁷⁸⁹%$£€¥")
        _KEEP = _SAFE_LETTERS | _SAFE_DIGITS | _TOKENIZER_PUNCT | _NUM_NORM_PASS | {" "}

        out = []
        for ch in text:
            if ch in _KEEP:
                out.append(ch)
            elif unicodedata.category(ch).startswith("Z"):
                out.append(" ")   # różne rodzaje spacji unicode → spacja
            elif unicodedata.category(ch).startswith("L"):
                # Akcenty łacińskie (é, ê, à, ñ itp.) → zdekompozycja do bazy ASCII
                base = unicodedata.normalize("NFKD", ch)[0]
                if base in _SAFE_LETTERS:
                    out.append(base)
                # Cyrylica, CJK itp. → pomiń
            # Reszta (koreański, chiński, emoji itp.): pomiń
        return "".join(out)

    # ── KROK 1b: rozbudowa skrótów ────────────────────────────────────────────

    def abbreviation_expand(self, text: str) -> str:
        """
        Rozszerza skróty polskie na pełne formy i rozwija rzymskie liczby.
        Wywoływana po raw_clean(), przed num_normalize().
        """
        s = text

        # Special-case abbreviations (before general expansion)
        # Address context: m. N → mieszkania N (before generic "m." → "metrów")
        s = re.sub(r"\bm\.\s+(\d+)", r"mieszkania \1", s)
        s = re.sub(r"\bDz\.U\.", "dziennik ustaw", s, flags=re.IGNORECASE)
        def _obr_min_repl(m):
            after = m.string[m.end():]
            if not after.strip():
                return "obrotów na minutę."
            return "obrotów na minutę"
        s = re.sub(r"\bobr\./min\.", _obr_min_repl, s)

        # Legal references: art. N ust. N pkt N[letter] → ordinal locative forms
        # e.g. "art. 15 ust. 3 pkt 2a" → "artykule piętnastym ustępie trzecim punkcie drugim a"
        def _legal_ref_expand(m):
            art_n = int(m.group(1))
            ust_n = int(m.group(2))
            pkt_raw = m.group(3)
            pkt_match = re.match(r'(\d+)([a-ząćęłńóśźż]*)', pkt_raw)
            pkt_n = int(pkt_match.group(1))
            pkt_suffix = pkt_match.group(2)
            art_ord = _ordinal_inflect(art_n, 'loc', 'm')
            ust_ord = _ordinal_inflect(ust_n, 'loc', 'm')
            pkt_ord = _ordinal_inflect(pkt_n, 'loc', 'm')
            result = f"artykule {art_ord} ustępie {ust_ord} punkcie {pkt_ord}"
            if pkt_suffix:
                result += f" {pkt_suffix}"
            return result
        s = re.sub(
            r"\bart\.\s*(\d+)\s+ust\.\s*(\d+)\s+pkt\s+(\d+[a-ząćęłńóśźż]*)\b",
            _legal_ref_expand, s, flags=re.IGNORECASE
        )

        s = re.sub(r"\bust\.\s+", "ustęp ", s)
        s = re.sub(r"\bpkt\b", "punkt", s)

        # Skróty jednostek — NIE rozwijaj gdy poprzedzone cyfrą (num_normalize obsłuży)
        _UNIT_ABBREV_STEMS = frozenset({"m", "g", "l", "s", "ha", "km", "cm", "mm", "str", "gr", "t", "nm", "mg", "kg", "w", "v"})

        # 1. Rozszerz skróty — używamy prekompilowanych wzorców (_COMPILED_ABBREVS)
        def _make_repl(exp, ab):
            def _repl(m):
                matched = m.group(0)
                # Guard: matched text is a known unit (e.g. "dB") → skip, num_normalize handles it
                if matched in _UNITS:
                    return matched
                # Guard: jednostki po cyfrze (5 m., 3 g.) lub po / (km/s.) → nie rozwijaj
                if ab.endswith("."):
                    stem = ab[:-1].lower()
                    if stem in _UNIT_ABBREV_STEMS:
                        before = m.string[max(0, m.start() - 5):m.start()].rstrip()
                        if before and (before[-1].isdigit() or before.endswith("/")):
                            return matched
                # Guard: jeśli skrót z kropką (np. "dom.") matchuje na końcu zdania,
                # a rdzeń jest częstym polskim słowem → nie rozwijaj
                if ab.endswith("."):
                    stem = ab[:-1].lower()
                    after = m.string[m.end():].strip()
                    # Guard: "t." (tom) — only expand when followed by digit/Roman numeral
                    # (publishing context: "t. IV", "t. 3"), not at end of sentence ("e i t.")
                    if stem == "t":
                        if not after or not (after[0].isdigit() or re.match(r'[IVXLCDM]', after)):
                            return matched
                    # Guard: jednoliterowe skróty (M. St. O.) przed wielką literą
                    # to prawdopodobnie inicjały lub prefiksy nazw własnych — nie rozwijaj
                    # ALE znane skróty (np., ul., t., nr.) ZAWSZE rozwijamy
                    if len(stem) <= 2 and after and after[0].isupper():
                        if stem not in _ALWAYS_EXPAND_ABBREVS:
                            return matched
                    if stem in _ABBREV_SAFE_WORDS:
                        # Koniec zdania/tekstu, lub po kropce jest wielka litera,
                        # myślnik (dialog), wykrzyknik, pytajnik, wielokropek itp.
                        if not after or after[0] in "ABCDEFGHIJKLMNOPQRSTUVWXYZĄĆĘŁŃÓŚŹŻ-—–!?…)]\"\':;":
                            return matched
                result = exp
                if matched and matched[0].isupper() and exp and exp[0].islower():
                    result = exp[0].upper() + exp[1:]
                if ab.endswith('.') and not m.string[m.end():].strip():
                    result = result + '.'
                return result
            return _repl

        for pattern, expansion, abbrev in _COMPILED_ABBREVS:
            s = pattern.sub(_make_repl(expansion, abbrev), s)

        # 1a. św. → święty/świętej/świętego z kontekstem
        _FEMALE_SAINT_NAMES = frozenset({
            "Anna", "Anny", "Annie", "Annę", "Anną",
            "Maria", "Marii", "Marię", "Marią",
            "Katarzyna", "Katarzyny", "Katarzynę", "Katarzyną",
            "Barbara", "Barbary", "Barbarę", "Barbarą",
            "Teresa", "Teresy", "Teresę", "Teresą",
            "Jadwiga", "Jadwigi", "Jadwigę", "Jadwigą",
            "Klara", "Klary", "Klarę", "Klarą",
            "Helena", "Heleny", "Helenę", "Heleną",
            "Agnieszka", "Agnieszki", "Agnieszkę", "Agnieszką",
            "Faustyna", "Faustyny", "Faustynę", "Faustyną",
            "Łucja", "Łucji", "Łucję", "Łucją",
            "Magdalena", "Magdaleny", "Magdalenę", "Magdaleną",
            "Monika", "Moniki", "Monikę", "Moniką",
            "Cecylia", "Cecylii", "Cecylię", "Cecylią",
            "Rita", "Rity", "Ritę", "Ritą",
            "Róża", "Róży", "Różę", "Różą",
            "Zofia", "Zofii", "Zofię", "Zofią",
        })
        def _sw_expand(m):
            before = m.string[max(0, m.start() - 40):m.start()].lower()
            name = m.group(1)
            is_fem = name in _FEMALE_SAINT_NAMES
            # Heurystyka przypadka: po kontekście dopełniaczowym → dopełniacz
            needs_gen = bool(re.search(
                r'\b(?:do|od|bez|u|dla|koło|obok|'
                r'kościół|kościele|kościoła|kaplica|kaplicy|'
                r'szpital|szpitala|parafia|parafii|bazylika|bazyliki|'
                r'klasztor|klasztoru|katedra|katedry|'
                r'plac|placu|ulica|ulicy|obraz|obrazu)\s*$',
                before))
            needs_instr = bool(re.search(r'\b(?:ze?|przed|pod|nad|za|między)\s*$', before))
            if is_fem:
                if needs_gen:
                    return f"świętej {name}"
                elif needs_instr:
                    return f"świętą {name}"
                else:
                    return f"święta {name}"
            else:
                if needs_gen:
                    return f"świętego {name}"
                elif needs_instr:
                    return f"świętym {name}"
                else:
                    return f"święty {name}"
        s = re.sub(r"\bśw\.\s*([A-ZĄĆĘŁŃÓŚŹŻ][a-ząćęłńóśźż]+)", _sw_expand, s)

        # 1b. Daty z miesiącem rzymskim: 12.XI.1473, 03.V.1474
        #     Musi być PRZED rozwojem rzymskich liczb (krok 2)
        def _date_roman_expand(m):
            d, roman_m, y = int(m.group(1)), m.group(2).upper(), int(m.group(3))
            try:
                mo = _roman_to_int(roman_m)
            except Exception:
                return m.group(0)
            if not (1 <= mo <= 12 and 1 <= d <= 31):
                return m.group(0)
            return f"{_ORD_GEN_DAYS.get(d, str(d))} {_MONTHS_GEN[mo]} {_year_gen(y)} roku"
        s = re.sub(
            r"\b([012]?\d|3[01])\.\s*(I{1,3}|IV|VI{0,3}|IX|XI{0,2}|XII?)\.\s*(\d{4})(?:\s+r\.|\s+roku)?\b",
            _date_roman_expand, s
        )

        # 2. Zamień rzymskie liczby na words w bezpiecznych kontekstach
        s = self._expand_roman_numerals(s)

        # 3. Usuń adresy URL i e-maile
        # URLs → "adres internetowy"
        s = re.sub(
            r"https?://[^\s]+",
            "adres internetowy",
            s,
            flags=re.IGNORECASE
        )
        # Emails → "adres email"
        s = re.sub(
            r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b",
            "adres email",
            s,
            flags=re.IGNORECASE
        )

        # 4. Usuń znaczniki list i punktorów
        # Bullet chars: • ◦ ▪ ● ◆ ◇
        bullet_pattern = r"[•◦▪●◆◇]"
        s = re.sub(bullet_pattern, "", s)
        # Numbered list: 1) 2) itp. (z nawiasem) — usuń tylko nawias, nie numer
        # Uwaga: nie usuwaj "1. " (kropka) bo to może być ordinal (1. miejsce)
        number_paren_pattern = r"^\s*(\d+)\)\s+"
        s = re.sub(number_paren_pattern, r"\1. ", s, flags=re.MULTILINE)
        # Letter list: a) b) c) → polska nazwa litery
        # Tylko małe litery — wielka litera na początku linii to nagłówek/inicjał, nie punkt listy
        def _letter_list_expand(m):
            letter = m.group(1).upper()
            name = _LETTER_NAMES_PL.get(letter, letter.lower())
            return f"{name}, "
        # (?!\d) — nie dopasuj "r. 1939" (rok), "s. 42" (strona) itp.
        s = re.sub(r"^\s*([a-z])[.)]\s+(?!\d)", _letter_list_expand, s, flags=re.MULTILINE)

        return s

    def _expand_roman_numerals(self, text: str) -> str:
        """
        Zamienia rzymskie liczby na słowa polskie tylko w bezpiecznych kontekstach.
        Pomija pojedyncze "I" (czyli "i" = "and").
        """
        def replace_roman(match: re.Match) -> str:
            roman = match.group(1).upper()
            # Skip single I (conflicts with Polish "i")
            if len(roman) == 1:
                return match.group(0)
            try:
                val = _roman_to_int(roman)
                # Konwertuj na liczbę porządkową w języku polskim
                ordinal = num2words(val, lang="pl", to="ordinal")
                return ordinal
            except Exception:
                return match.group(0)

        # Dopasuj rzymskie numery (MINUS pojedyncze I) w całości tekstu
        # Zbieramy wszystkie zamiany NAJPIERW, potem aplikujemy od prawej do lewej
        # (żeby przesunięcia pozycji nie psuly późniejszych dopasowań)
        s = text
        replacements = []  # (start, end, ordinal, wiek_fix_pos_or_None)

        for match in _ROMAN_PATTERN.finditer(s):
            roman = match.group(1).upper()
            # Skip single "I" — conflicts with Polish conjunction "i" (and)
            # Other single-letter Roman numerals (V, X, L, C, D, M) are OK when context present
            if roman == "I":
                # Allow single "I" when immediately followed by a quarter/half context word
                # e.g. "W I kwartale" → "W pierwszym kwartale"
                after_stripped = s[match.span()[1]:match.span()[1] + 30].lower().lstrip()
                if not any(after_stripped.startswith(w) for w in ("kwart", "połow", "poł.")):
                    continue
            # Skip blacklisted words (Polish pronouns, common words)
            if roman in _ROMAN_BLACKLIST:
                continue
            # Sprawdź kontekst: czy przed lub po jest słowo kluczowe?
            span = match.span()
            before = s[max(0, span[0] - 20):span[0]].lower()
            after = s[span[1]:span[1] + 30].lower()

            context_found = any(word in before or word in after for word in _ROMAN_CONTEXT_WORDS)
            if not context_found:
                continue

            try:
                val = _roman_to_int(roman)
                ordinal = num2words(val, lang="pl", to="ordinal")
                # Sprawdź kontekst: forma miejscownikowa vs dopełniaczowa
                after_stripped = after.lstrip()
                before_stripped = before.rstrip()
                preceded_by_w = bool(re.search(r'\bw\s*$', before_stripped, re.IGNORECASE))
                wiek_nominative_after = bool(re.match(r"wiek\b", after_stripped))
                wiek_abbrev_after = bool(re.match(r"w\.\s", after_stripped))  # "w." = skrót "wiek"
                # Dopełniacz wymagany po: na początku, pod koniec, z, do, od, końca, połowy
                # Oraz po "lat/lata/latach [NN]" (dekady): "lat 90 XX wieku" → genitive
                # Oraz po formach dopełniacza l.mn. (-ych/-ich)
                needs_genitive = bool(re.search(
                    r'(?:\b(?:początku|początkiem|koniec|końca|końcem|połowy|połowie|'
                    r'przełomie|przełomu|schyłku|do|od|z|lat|lata|latach)'
                    r'(?:\s+\d{2}\.?)?\s*$|(?:ych|ich)\s*$)',
                    before_stripped, re.IGNORECASE
                ))
                wiek_context = (after_stripped.startswith("wieku")
                                or after_stripped.startswith("wiekowi")
                                or wiek_abbrev_after
                                or (wiek_nominative_after and preceded_by_w))
                war_context = (after_stripped.startswith("wojna")
                               or after_stripped.startswith("wojny")
                               or after_stripped.startswith("wojnie")
                               or after_stripped.startswith("wojną"))
                if wiek_context:
                    if needs_genitive:
                        # Dopełniacz: pierwsz-y → pierwsz-ego, drug-i → drug-iego
                        words = ordinal.split()
                        gen_words = []
                        for w in words:
                            if w.endswith("y"):
                                gen_words.append(w[:-1] + "ego")
                            elif w.endswith("i"):
                                gen_words.append(w[:-1] + "iego")
                            else:
                                gen_words.append(w)
                        ordinal = " ".join(gen_words)
                    else:
                        # Miejscownik: pierwsz-y → pierwsz-ym, drug-i → drug-im
                        words = ordinal.split()
                        loc_words = []
                        for w in words:
                            if w.endswith("y"):
                                loc_words.append(w[:-1] + "ym")
                            elif w.endswith("i"):
                                loc_words.append(w[:-1] + "im")
                            else:
                                loc_words.append(w)
                        ordinal = " ".join(loc_words)
                elif preceded_by_w and any(after_stripped.startswith(w) for w in ("kwart", "połow", "poł.")):
                    # Quarter/half context after "W": W I kwartale → w pierwszym kwartale
                    words = ordinal.split()
                    loc_words = []
                    for w in words:
                        if w.endswith("y"):
                            loc_words.append(w[:-1] + "ym")
                        elif w.endswith("i"):
                            loc_words.append(w[:-1] + "im")
                        else:
                            loc_words.append(w)
                    ordinal = " ".join(loc_words)
                elif war_context:
                    words = ordinal.split()
                    war_words = []
                    for w in words:
                        if after_stripped.startswith("wojną"):
                            if w.endswith("y"):
                                war_words.append(w[:-1] + "ą")
                            elif w.endswith("i"):
                                war_words.append(w[:-1] + "ą")
                            else:
                                war_words.append(w)
                        elif after_stripped.startswith("wojnie") or after_stripped.startswith("wojny"):
                            if w.endswith("y"):
                                war_words.append(w[:-1] + "iej")
                            elif w.endswith("i"):
                                war_words.append(w[:-1] + "iej")
                            else:
                                war_words.append(w)
                        else:
                            if w.endswith("y"):
                                war_words.append(w[:-1] + "a")
                            elif w.endswith("i"):
                                war_words.append(w[:-1] + "a")
                            else:
                                war_words.append(w)
                    ordinal = " ".join(war_words)
                else:
                    # Feminine genitive context: III Rzeszy, II Rzeczpospolitej, II poł.
                    fem_gen_context = bool(re.match(
                        r"(?:rzesz|rzeczpospolit|połow|poł\.|międzynarodówk|dynastii|armii|ligi|brygad|dywizj)",
                        after_stripped))
                    if fem_gen_context:
                        words = ordinal.split()
                        fem_words = []
                        for w in words:
                            if w.endswith("y"):
                                fem_words.append(w[:-1] + "iej")
                            elif w.endswith("i"):
                                fem_words.append(w[:-1] + "iej")
                            else:
                                fem_words.append(w)
                        ordinal = " ".join(fem_words)
                # Gdy po numerale następuje 'wiek' (nominativus) lub 'w.' (skrót),
                # zapamiętaj pozycję do późniejszej zamiany na 'wieku'
                wiek_fix_pos = None
                wiek_fix_len = 4  # len("wiek") domyślnie
                if wiek_nominative_after and preceded_by_w:
                    after_end = span[1]
                    wiek_m = re.match(r"wiek\b", s[after_end:].lstrip())
                    if wiek_m:
                        ws_pos = after_end + (len(s[after_end:]) - len(s[after_end:].lstrip()))
                        wiek_fix_pos = ws_pos
                elif wiek_abbrev_after:
                    after_end = span[1]
                    wiek_m = re.match(r"w\.", s[after_end:].lstrip())
                    if wiek_m:
                        ws_pos = after_end + (len(s[after_end:]) - len(s[after_end:].lstrip()))
                        wiek_fix_pos = ws_pos
                        wiek_fix_len = 2  # len("w.")
                replacements.append((span[0], span[1], ordinal, wiek_fix_pos, wiek_fix_len))
            except Exception:
                pass

        # Aplikuj zamiany od prawej do lewej (zachowuje poprawność pozycji)
        for start, end, ordinal, wiek_fix_pos, wiek_fix_len in sorted(replacements, key=lambda x: x[0], reverse=True):
            if wiek_fix_pos is not None:
                s = s[:wiek_fix_pos] + "wieku" + s[wiek_fix_pos + wiek_fix_len:]
            s = s[:start] + ordinal + s[end:]

        return s

    # ── KROK 1c: rozbudowa obcych skrótów, nazw i zapożyczeń ────────────────

    def foreign_expand(self, text: str) -> str:
        """
        Rozwija zagraniczne skróty, imiona angielskie i zapożyczenia na
        fonetyczny zapis polski zrozumiały dla TTS.

        Kolejność:
          1. Skróty z kropkami (U.S.A., e.g.)
          2. Zapożyczenia angielskie (case-insensitive)
          3. Znane nazwy własne (case-sensitive)
          4. Tokeny mieszane z myślnikiem (GPT-4, Wi-Fi)
          5. CamelCase (YouTube, iPhone)
          6. Skróty ALL-CAPS (TTS, API, NATO)
        """
        s = text
        s_before = s

        # Zbiór symboli walut/jednostek — nie ruszać, num_normalize je obsłuży
        _CURRENCY_UNIT_SKIP = set(_CURRENCIES.keys()) | set(
            k for k in _UNITS if k.isupper() or len(k) <= 2 or any(c.isupper() for c in k)
        )

        # 0a0. Tire dimensions: 225/45 R17 → tire_225_45_R17 (placeholder, expanded in num_normalize)
        def _tire_pre(m):
            return f"{m.group(1)} na {m.group(2)} er {m.group(3)}"
        s = re.sub(r"\b(\d{3})/(\d{2})\s*[Rr](\d{2})\b", _tire_pre, s)

        # 0a-pesel. PESEL/PIN/NIP/REGON digit-by-digit: PESEL 850312... → osiem pięć zero...
        def _spell_digits_context(m):
            keyword = m.group(1)
            sep = m.group(2) or ""
            digits = m.group(3)
            spelled = " ".join(_DIGIT_WORDS_PL[int(d)] for d in digits)
            return f"{keyword}{sep}{spelled}"
        s = re.sub(r"\b(PESEL|PIN|NIP|REGON|kod|numer\s+seryjny|numer\s+identyfikacyjny|cyfr|cyfry)(\s*:?\s*)(\d{4,})\b",
                   _spell_digits_context, s, flags=re.IGNORECASE)

        # 0a-hex. Hex codes: 0x000000ED → zero iks zero zero zero zero zero zero e de
        # Also color codes: #FF5733 → hash ef ef pięć siedem trzy trzy
        def _spell_hex(m):
            prefix = m.group(1)  # "0x" or "#"
            digits = m.group(2)
            parts = []
            if prefix.lower() == "0x":
                parts.append("zero iks")
            else:
                parts.append("krzyżyk")
            for ch in digits:
                ch_upper = ch.upper()
                if ch.isdigit():
                    parts.append(_DIGIT_WORDS_PL[int(ch)])
                elif ch_upper in _LETTER_NAMES_PL:
                    parts.append(_LETTER_NAMES_PL[ch_upper])
                else:
                    parts.append(ch.lower())
            return " ".join(parts)
        s = re.sub(r"\b(0x)([0-9A-Fa-f]+)\b", _spell_hex, s)
        s = re.sub(r"(#)([0-9A-Fa-f]{6})\b", _spell_hex, s)

        # 0a. Filename detection: dane_v2.csv → dane podkreślnik fał dwa kropka ce es fał
        _KNOWN_EXTS = {"csv", "txt", "pdf", "json", "xml", "html", "py", "js", "ts",
                        "doc", "docx", "xls", "xlsx", "png", "jpg", "jpeg", "gif", "zip",
                        "tar", "gz", "log", "cfg", "ini", "yml", "yaml", "md", "sql",
                        "sh", "bat", "exe", "dll", "so", "jar", "war", "mp3", "mp4",
                        "wav", "avi", "mov", "mkv", "pdf", "rtf", "odt", "ods",
                        "sys", "dat", "bak", "tmp", "conf"}
        def _spell_filename(m):
            filename = m.group(0)
            parts = []
            for ch in filename:
                if ch == '_':
                    parts.append("podkreślnik")
                elif ch == '.':
                    parts.append("kropka")
                elif ch == '-':
                    parts.append("-")
                elif ch.isdigit():
                    parts.append(_n2w(int(ch)))
                elif ch.isalpha():
                    name = _LETTER_NAMES_PL.get(ch.upper(), ch.lower())
                    parts.append(name)
                else:
                    parts.append(ch)
            return " ".join(parts)
        # Match word_word.ext patterns with known extensions (underscore required)
        _ext_pattern = "|".join(re.escape(e) for e in _KNOWN_EXTS)
        s = re.sub(
            r"\b(\w+(?:_\w+)+\.(?:" + _ext_pattern + r"))\b",
            _spell_filename, s, flags=re.IGNORECASE)

        # 0. Liczby rzymskie po imionach: Gates III → Gates trzeci
        #    Wymagaj aby po numerale było: interpunkcja, koniec tekstu, lub kolejne imię
        #    (żeby "Literka I odpadła" nie dawało "Literka pierwszy odpadła")
        def _name_roman(m):
            roman = m.group(2).upper()
            try:
                val = _roman_to_int(roman)
                ordinal = num2words(val, lang="pl", to="ordinal")
                name = m.group(1)
                # Dopełniacz: jeśli imię kończy się na -a/-ego (gen.), odmień ordinal
                if name.endswith(("a", "ego")):
                    ordinal = " ".join(_ord_to_gen(w) for w in ordinal.split())
                return name + " " + ordinal
            except Exception:
                return m.group(0)
        def _name_roman_between_words(m):
            left, roman, right = m.group(1), m.group(2).upper(), m.group(3)
            try:
                val = _roman_to_int(roman)
                ordinal = num2words(val, lang="pl", to="ordinal")
                if left.endswith("a") or right.endswith(("ego", "iej")):
                    ordinal = " ".join(_ord_to_gen(w) for w in ordinal.split())
                return f"{left} {ordinal} {right}"
            except Exception:
                return m.group(0)
        s = re.sub(
            r"\b([A-ZĄĆĘŁŃÓŚŹŻ][a-ząćęłńóśźż]+)\s+(I{1,3}|IV|VI{0,3}|IX|XI{0,3})\b(?=\s*[.,;:!?)\]\-]|\s*$|\s+[a-ząćęłńóśźż])",
            _name_roman, s
        )
        s = re.sub(
            r"\b([A-ZĄĆĘŁŃÓŚŹŻ][a-ząćęłńóśźż]+)\s+(I{1,3}|IV|VI{0,3}|IX|XI{0,3})\s+([A-ZĄĆĘŁŃÓŚŹŻ][a-ząćęłńóśźż]+)\b",
            _name_roman_between_words, s
        )

        # 0a. Kwartały: Q1→"kwartał pierwszy", Q4→"kwartał czwarty"
        _QUARTER_ORDINALS = {
            "1": "pierwszy", "2": "drugi", "3": "trzeci", "4": "czwarty",
        }
        def _quarter_expand(m):
            return f"kwartał {_QUARTER_ORDINALS.get(m.group(1), m.group(1))}"
        s = re.sub(r"\bQ([1-4])\b", _quarter_expand, s)

        # 0b. Skróty rozdzielone ukośnikiem: TCP/IP, AC/DC, B2B/B2C
        def _slash_expand(m):
            parts = m.group(0).split("/")
            expanded = []
            for p in parts:
                pu = p.upper()
                if pu in _INTL_ABBREV_MAP:
                    expanded.append(_INTL_ABBREV_MAP[pu])
                elif pu in _CURRENCY_UNIT_SKIP:
                    expanded.append(p)
                elif p.isupper() and len(p) >= 2:
                    if _is_pronounceable(p):
                        expanded.append(p.lower())
                    else:
                        expanded.append(_spell_abbreviation(p))
                else:
                    expanded.append(p)
            return " ".join(expanded)
        s = re.sub(r"\b[A-Z][A-Z0-9]+(?:/[A-Z][A-Z0-9]+)+\b", _slash_expand, s)

        # 0c. Samodzielne waluty BEZ liczby: "w EUR", "na USD"
        #     (currency_skip chroni je gdy SĄ z liczbą, ale samodzielne trzeba rozwinąć)
        _CURRENCY_NAMES_STANDALONE = {
            "PLN": "złotych", "EUR": "euro", "USD": "dolarach",
            "GBP": "funtach", "CHF": "frankach", "CZK": "koronach",
            "BTC": "bitcoinów",
        }
        def _standalone_currency(m):
            sym = m.group(1)
            if sym in _CURRENCY_NAMES_STANDALONE:
                # Nie zamieniaj jeśli poprzedza mnożnik (mln, mld, bln, tys) — wtedy _large_curr obsłuży
                before = m.string[:m.start()].rstrip()
                if re.search(r'\b(?:mln|mld|bln|tys|tys\.)\s*$', before):
                    return m.group(0)
                # W nawiasach → literuj zamiast rozwijać (kurs funta brytyjskiego (GBP))
                if before.endswith("("):
                    return _spell_abbreviation(sym)
                return m.group(0).replace(sym, _CURRENCY_NAMES_STANDALONE[sym])
            return m.group(0)
        # Dopasuj walutę NIEPOPRZEDZONĄ cyfrą
        s = re.sub(r"(?<!\d\s)(?<!\d)\b(PLN|EUR|USD|GBP|CHF|CZK|BTC)\b(?!\s*\d)", _standalone_currency, s)

        # 0c2. Wersja po nazwie CamelCase: "AmigaOS 4.0" → "AmigaOS v4.0"
        def _sw_version(m):
            word, ver = m.group(1), m.group(2)
            # Tylko CamelCase z wielką literą PO pierwszym znaku — nie ALL-CAPS ani zwykłe Słowa
            # "AmigaOS" → OK (O/S uppercase after position 0); "Silnik" → skip (plain Polish word)
            if any(c.isupper() for c in word[1:]) and any(c.islower() for c in word):
                return f"{word} v{ver}"
            return m.group(0)
        s = re.sub(
            r"\b([A-Za-z][A-Za-z0-9]+)\s+(\d{1,2}\.\d{1,2})\b"
            r"(?!\.\d{4}\b)"
            r"(?!\s*(?:GHz|MHz|kHz|km|cm|mm|m\b|kg|g\b|%|°|ls|ms\b))",
            _sw_version, s
        )

        # 0d. Numery wersji: v1.0, v2.3.1, v0.0.7, v1
        def _version_expand(m):
            parts = m.group(1).split(".")
            words = [num2words(int(p), lang="pl") for p in parts]
            return "wersja " + " ".join(words)
        s = re.sub(r"\bv(\d+(?:\.\d+)+)\b", _version_expand, s, flags=re.IGNORECASE)
        # v1, v2 (bez kropek) — tylko gdy po v następuje sama cyfra
        # Dopasuj: v2 i v2. (kropka zdaniowa też OK)
        def _version_simple(m):
            return f"wersja {num2words(int(m.group(1)), lang='pl')}"
        # v1, v2 (lowercase only) — uppercase V + small digit = engine type (V8, V6, V12)
        s = re.sub(r"\bv(\d+)(?=\b|\.(?!\d))", _version_simple, s)  # removed re.IGNORECASE

        # Engine types: V8, V6, V12 → "fał osiem", "fał sześć" (Polish phonetic for V)
        def _engine_type(m):
            token = m.group(0)
            # Check dictionary first (V8 etc.)
            if token.upper() in _INTL_ABBREV_MAP:
                return _INTL_ABBREV_MAP[token.upper()]
            n = int(m.group(2))
            return f"fał {num2words(n, lang='pl')}"
        s = re.sub(r"\b(V)(\d{1,2})\b", _engine_type, s)

        # 0d2. Zakresy wersji z wildcardem: 3.x, 4.x
        def _version_wildcard_x(m):
            return f"{num2words(int(m.group(1)), lang='pl')} iks"
        s = re.sub(r"\b(\d+)\.x\b", _version_wildcard_x, s, flags=re.IGNORECASE)

        # 1. Skróty z kropkami — najdłuższe najpierw
        for dotted, expansion in sorted(
            _DOTTED_ABBREV_MAP.items(), key=lambda x: -len(x[0])
        ):
            # Gdy skrót jest na końcu zdania, zachowaj kropkę
            def _dotted_repl(m, exp=expansion, ab=dotted):
                if ab.endswith('.') and not m.string[m.end():].strip():
                    return exp + '.'
                return exp
            s = re.sub(re.escape(dotted), _dotted_repl, s)
        # Fallback: nieznane skróty z kropkami → usuń kropki i przeliteruj
        def _dotted_fallback(m):
            full = m.group(0)
            letters = full.replace(".", "")
            if letters.upper() in _INTL_ABBREV_MAP:
                exp = _INTL_ABBREV_MAP[letters.upper()]
            elif _is_pronounceable(letters):
                exp = letters.lower()
            else:
                exp = _spell_abbreviation(letters)
            # Zachowaj zdaniową kropkę gdy skrót jest na końcu
            if full.endswith('.') and not m.string[m.end():].strip():
                exp = exp + '.'
            return exp
        s = _RE_DOTTED_ABBREV.sub(_dotted_fallback, s)

        # 2. Zapożyczenia angielskie (case-insensitive, dłuższe najpierw)
        #    Używa prekompilowanych wzorców z _COMPILED_LOANWORDS
        #    Zachowuje wielką literę z oryginału (Software → Softłer)
        def _loanword_with_case(m, pol, hyphen):
            orig = m.group(0)
            # Zapożyczenia z łącznikiem (Wi-Fi) i nazwy własne z wbudowaną wielką literą
            # (ActiveX, iPhone) nigdy nie kapitalizujemy
            if hyphen:
                return pol
            if not (orig and orig[0].isupper() and pol and pol[0].islower()):
                return pol
            # Nazwa własna / brand (CamelCase: ActiveX, iPhone) → bez kapitalizacji
            if any(c.isupper() for c in orig[1:]):
                return pol
            # Pospolite słowo z wielkiej (Software, Startup) → zachowaj wielkość na pocz. zdania
            pos = m.start()
            before = m.string[:pos].rstrip()
            if pos == 0 or (before and before[-1] in ".!?"):
                return pol[0].upper() + pol[1:]
            return pol
        for pat, pol, _hyphen in _COMPILED_LOANWORDS:
            s = pat.sub(lambda m, p=pol, h=_hyphen: _loanword_with_case(m, p, h), s)

        # 3. Znane imiona/nazwiska (case-sensitive, dłuższe najpierw)
        #    Używa prekompilowanych wzorców z _COMPILED_NAMES
        def _name_repl(m, ph):
            suffix = m.group(1) or ""
            return ph + suffix
        for pat, phonetic in _COMPILED_NAMES:
            s = pat.sub(lambda m, ph=phonetic: ph + (m.group(1) or ""), s)

        # 3b. Sufiksy k/M przy walutach: $52k, $120M, €38M
        #     Musi być PRZED digit_letter (5b) żeby $120M nie stał się "sto dwadzieścia em"
        _KM_LARGE = {"k": 1_000, "K": 1_000, "m": 1_000_000, "M": 1_000_000}
        def _curr_km_expand(m):
            sym, raw, suffix = m.group(1), m.group(2), m.group(3)
            cur = _CURRENCIES.get(sym)
            if not cur:
                return m.group(0)
            val, clean = _parse_raw(raw)
            multiplier = _KM_LARGE.get(suffix, 1)
            n = val * multiplier
            n_int = int(n)
            if n == n_int:
                return f"{_n2w(n_int)} {_pick(n_int, cur[0], cur[1], cur[2])}"
            return f"{_n2w_float(clean)} {_pick(2, cur[0], cur[1], cur[2])}"
        s = re.sub(r"([€$£¥])\s*(\d+(?:[,.]\d+)?)\s*([kKmM])\b", _curr_km_expand, s)
        # Signed: +$268, -$2 770
        def _signed_curr_expand(m):
            sign, sym, raw = m.group(1), m.group(2), m.group(3)
            cur = _CURRENCIES.get(sym)
            if not cur:
                return m.group(0)
            prefix = "minus " if sign == "-" else "plus "
            val, clean = _parse_raw(raw)
            n_int = int(val)
            if val == n_int:
                return f"{prefix}{_n2w(n_int)} {_pick(n_int, cur[0], cur[1], cur[2])}"
            return f"{prefix}{_n2w_float(clean)} {cur[1]}"
        s = re.sub(r"([+-])([€$£¥])\s*(\d[\d\s,.]*\d|\d)", _signed_curr_expand, s)
        # Standalone k suffix: 103k → sto trzy tysiące
        # Ale nie 4K (rozdzielczość) — pomijaj 1-cyfrowe + wielkie K
        def _k_suffix_expand(m):
            raw, letter = m.group(1), m.group(2)
            # 4K, 8K → to rozdzielczość, nie tysiące
            if letter == "K" and len(raw) == 1:
                return m.group(0)
            val, _ = _parse_raw(raw)
            n = int(val * 1000)
            return _n2w(n)
        s = re.sub(r"\b(\d+(?:[,.]\d+)?)\s*([kK])\b", _k_suffix_expand, s)

        # 3c. Mixed-case abbreviations (PhD, MSc, etc.) — before CamelCase split
        _MIXED_CASE_ABBREVS = {
            "PhD": "pe ejcz de", "MSc": "em es ce", "BSc": "be es ce",
            "MBA": "em be a", "MPhil": "em fil",
            "XMLHttpRequest": "iks em el ha te te pe rikłest",
        }
        for mc_abbr, mc_exp in _MIXED_CASE_ABBREVS.items():
            s = re.sub(rf"\b{re.escape(mc_abbr)}\b", mc_exp, s)

        # 3d. Dokładne trafienia słownikowe dla alfanumeryków nieobsługiwanych
        # przez prostsze reguły typu A17 / 5G / CamelCase (np. M68x00, IE11).
        def _dict_alnum_exact(m):
            token = m.group(0)
            return _INTL_ABBREV_MAP.get(token.upper(), token)
        s = re.sub(r"\b[A-Za-z][A-Za-z0-9]*\d[A-Za-z0-9]*\b", _dict_alnum_exact, s)

        # 4. Tokeny mieszane z myślnikiem (GPT-4, Wi-Fi, COVID-19)
        # Pomiń czysto numeryczne zakresy (5-10, 1975-1998) — obsłuży je num_normalize
        _RE_ANNIVERSARY = re.compile(r"^\d+-(?:leci[aeu]|leciu|letni[aeoąę]?|letniej|letniego|letnich|letnim|letnimi|wieczn[yaeoęąi]|wiecznym|wiecznej|wiecznego|wiecznych|wieczni)$")
        # Compound adjective pattern for foreign_expand (digit-adjective)
        _RE_COMPOUND_ADJ = re.compile(
            r"^\d+-(?:osobow|metrow|procentow|karatow|godzinn|minutow|kilogramow"
            r"|litrow|tonow|stopniow|wieczn|piętrow|kondygnacyjn|cylindrow"
            r"|calow|centymetrow|kilogramow)", re.IGNORECASE)

        def _mixed_hyphen_repl(m):
            token = m.group(0)
            # N-osobowej, N-metrowy etc. → let num_normalize handle compound adjectives
            if _RE_COMPOUND_ADJ.match(token):
                return token
            # N-lecia/N-letni: 40-lecia, 100-lecie → niech num_normalize to obsłuży
            if _RE_ANNIVERSARY.match(token):
                return token
            # Najpierw sprawdź cały token w słowniku zapożyczeń
            if token in _ENGLISH_LOANWORDS_PL:
                return _ENGLISH_LOANWORDS_PL[token]
            if token.lower() in _ENGLISH_LOANWORDS_PL:
                return _ENGLISH_LOANWORDS_PL[token.lower()]
            segments = token.split("-")
            if all(seg.isdigit() for seg in segments):
                return token
            # Polskie złożenia przymiotnikowe — wszystkie segmenty małymi literami
            # (z opcjonalną wielką literą na początku — początek zdania)
            lower_segs = [s.lower() for s in segments]
            if all(seg.isalpha() for seg in lower_segs):
                # Sprawdź czy to polskie złożenie: segmenty małymi literami,
                # ewentualnie pierwszy z wielką (początek zdania)
                if all(seg.islower() or (seg[0].isupper() and seg[1:].islower())
                       for seg in segments):
                    return token  # zachowaj z myślnikiem (polskie złożenia i nazwiska)
            # Arabskie/perskie nazwy z al-/el-/ad-/ibn- — zachowaj bez zmian
            if segments[0].lower() in ("al", "el", "ad", "ibn", "bin", "abu"):
                return token
            return _expand_mixed_token(token)
        s = _RE_MIXED_HYPHEN.sub(_mixed_hyphen_repl, s)

        # 5. CamelCase (YouTube, iPhone, ChatGPT)
        # Pomiń krótkie tokeny będące jednostkami (dB, pH, kB, etc.)
        _CAMEL_SKIP = set(k for k in _UNITS if any(c.isupper() for c in k))
        def _camel_expand(m):
            word = m.group(0)
            if word in _CAMEL_SKIP:
                return word
            # Sprawdź w słowniku skrótowców (case-insensitive)
            if word.upper() in _INTL_ABBREV_MAP:
                return _INTL_ABBREV_MAP[word.upper()]
            if word in _ENGLISH_LOANWORDS_PL:
                return _ENGLISH_LOANWORDS_PL[word]
            # Celtyckie prefiksy nazwisk (Mc/Mac) → zachowaj jako jedno słowo, lowercase
            if re.match(r'^(?:Mc|Mac)[A-Z]', word):
                return word.lower()
            # Jeśli nie w słowniku, podziel na segmenty
            parts = re.findall(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+", word)
            result = []
            for p in parts:
                if p.upper() in _INTL_ABBREV_MAP:
                    result.append(_INTL_ABBREV_MAP[p.upper()])
                elif p in _ENGLISH_LOANWORDS_PL:
                    result.append(_ENGLISH_LOANWORDS_PL[p])
                elif p in _ENGLISH_NAMES_PL:
                    result.append(_ENGLISH_NAMES_PL[p])
                else:
                    result.append(p.lower())
            return " ".join(result)
        s = _RE_CAMELCASE.sub(_camel_expand, s)

        # 5a_mul. Wzorce mnożnikowe: 2x, 3x, 10x, x5, x10 → "dwa razy"
        def _multiplier_expand(m):
            n = int(m.group(1))
            return f"{num2words(n, lang='pl')} razy"
        s = re.sub(r"\b(\d+)\s*[xX]\b(?!\s*\d)", _multiplier_expand, s)
        # Prefix: x5, x10, x20
        def _multiplier_prefix(m):
            n = int(m.group(1))
            return f"razy {num2words(n, lang='pl')}"
        s = re.sub(r"\b[xX](\d+)\b", _multiplier_prefix, s)

        # 5b. Tokeny cyfra+litera: 3D, 4K, 5G, 2D
        _RE_DIGIT_LETTER = re.compile(r"\b(\d+)([A-Z]{1,3})\b")
        _DIGIT_LETTER_SKIP = set(k.upper() for k in _UNITS if k[0].isdigit()) if any(k[0].isdigit() for k in _UNITS) else set()
        def _digit_letter_expand(m):
            digits, letters = m.group(1), m.group(2)
            full = digits + letters
            # Sprawdź cały token w słowniku
            if full in _INTL_ABBREV_MAP:
                return _INTL_ABBREV_MAP[full]
            # Nie ruszaj jednostek
            if full in _CURRENCY_UNIT_SKIP:
                return m.group(0)
            num_part = num2words(int(digits), lang="pl")
            if len(letters) == 1:
                letter_part = _LETTER_NAMES_PL.get(letters, letters.lower())
            elif letters in _INTL_ABBREV_MAP:
                letter_part = _INTL_ABBREV_MAP[letters]
            else:
                letter_part = _spell_abbreviation(letters)
            return f"{num_part} {letter_part}"
        s = _RE_DIGIT_LETTER.sub(_digit_letter_expand, s)

        # 5c. Tokeny alfanumeryczne bez myślnika: A17, B2, G20, H2O
        _RE_ALPHANUM = re.compile(r"\b([A-Z]{1,5})(\d+)\b")
        def _alphanum_expand(m):
            letters, digits = m.group(1), m.group(2)
            full = letters + digits
            # Sprawdź cały token w słowniku
            if full in _INTL_ABBREV_MAP:
                return _INTL_ABBREV_MAP[full]
            # Literuj litery + zamień cyfry na słowa
            if len(letters) == 1:
                letter_part = _LETTER_NAMES_PL.get(letters, letters.lower())
            elif letters in _INTL_ABBREV_MAP:
                letter_part = _INTL_ABBREV_MAP[letters]
            elif _is_pronounceable(letters):
                letter_part = letters.lower()
            else:
                letter_part = _spell_abbreviation(letters)
            num_part = num2words(int(digits), lang="pl")
            return f"{letter_part} {num_part}"
        s = _RE_ALPHANUM.sub(_alphanum_expand, s)

        # 5d. Chemical formulas: H2SO4, CO2, NaCl, H2O
        # Pattern: letter(s) + digit(s) repeating, e.g. H2SO4, C6H12O6
        # Known no-digit chemical formulas
        _KNOWN_CHEM_FORMULAS = frozenset({
            "NaCl", "KCl", "CaCO", "NaOH", "KOH", "HCl",
            "NaBr", "KBr", "CaO", "MgO", "FeO", "ZnO",
            "AgCl", "AgBr", "CuO", "PbO", "AlCl",
        })
        _RE_CHEM = re.compile(r"\b((?:[A-Z][a-z]?\d*){2,})\b")
        def _chem_expand(m):
            formula = m.group(0)
            # Must contain at least one digit OR be a known chemical formula
            if not any(c.isdigit() for c in formula) and formula not in _KNOWN_CHEM_FORMULAS:
                return formula
            # Split into letter+digit segments and spell them
            parts = re.findall(r"([A-Z][a-z]?)(\d*)", formula)
            result = []
            for letters, digits in parts:
                if not letters:
                    continue
                letter_name = _LETTER_NAMES_PL.get(letters.upper(), letters.lower())
                if len(letters) == 2:
                    # Two-letter element: Na, Ca, etc. — spell both
                    letter_name = _LETTER_NAMES_PL.get(letters[0], letters[0].lower()) + " " + _LETTER_NAMES_PL.get(letters[1].upper(), letters[1].lower())
                result.append(letter_name)
                if digits:
                    result.append(num2words(int(digits), lang="pl"))
            return " ".join(result)
        s = _RE_CHEM.sub(_chem_expand, s)

        # 5e. Standalone lowercase abbreviations known to dictionary: ip → a i pi
        def _lowercase_abbrev_expand(m):
            word = m.group(0)
            upper = word.upper()
            if upper in _INTL_ABBREV_MAP and upper not in _CURRENCY_UNIT_SKIP:
                return _INTL_ABBREV_MAP[upper]
            return word
        s = re.sub(r"\b(ip)\b", _lowercase_abbrev_expand, s)

        # 5f. Mixed-case element symbols from dictionary: Zn → cynk
        # Only match specific known element abbreviations, not Polish words
        _ELEMENT_SYMBOLS = {"Zn", "Fe", "Cu", "Ag", "Au", "Pb", "Sn", "Hg", "Mn", "Cr", "Ni", "Ti"}
        def _element_symbol_expand(m):
            sym = m.group(0)
            upper = sym.upper()
            if sym in _ELEMENT_SYMBOLS and upper in _INTL_ABBREV_MAP:
                return _INTL_ABBREV_MAP[upper]
            return sym
        s = re.sub(r"\b([A-Z][a-z])\b(?!\w)", _element_symbol_expand, s)

        # 5g. Standalone uppercase single letter → Polish letter name
        # "osi X" → "osi iks", "Vit. C" → "witamina ce"
        # Skip: units after digits, formula context, Polish prepositions/words
        _UNIT_SINGLE_LETTERS = frozenset(k for k in _UNITS if len(k) == 1 and k.isupper())
        _SKIP_SINGLE_LETTERS = frozenset("W I O A U Z")  # Polish prepositions/conjunctions
        def _standalone_single_letter(m):
            letter = m.group(2)
            if letter in _SKIP_SINGLE_LETTERS:
                return m.group(0)
            before = m.string[max(0, m.start() - 20):m.start()].rstrip()
            # Don't expand if preceded by digit (it's a unit: 12 V, 16 A)
            if before and before[-1].isdigit() and letter in _UNIT_SINGLE_LETTERS:
                return m.group(0)
            # Don't expand in formula/equation context
            if re.search(r'(?:równa się|nie równa|plus|minus|razy)\s*$', before):
                return m.group(0)
            name = _LETTER_NAMES_PL.get(letter, letter.lower())
            return f"{m.group(1)}{name}"
        s = re.sub(r"(\s)([A-Z])(?=\s|$|[.,;:!?)\]])", _standalone_single_letter, s)

        # 6. ALL-CAPS (2+ liter) — TTS, API, NATO
        def _allcaps_expand(m):
            word = m.group(0)
            upper = word.upper()
            # Nie ruszaj symboli walut/jednostek — num_normalize je obsłuży
            if upper in _CURRENCY_UNIT_SKIP:
                return word
            # Sprawdź w słowniku
            if upper in _INTL_ABBREV_MAP:
                return _INTL_ABBREV_MAP[upper]
            # Liczby rzymskie: VIII, IX, XIV itp. → liczebnik porządkowy
            # (tylko realistyczne numery rozdziałów/sekcji: 2–499)
            # Wymagamy kontekstu (rozdział, wiek, akt…) żeby nie mylić XL=40 z rozmiarem odzieży
            if upper in _ROMAN_BLACKLIST:
                pass  # skip — to polskie słowo, nie rzymska liczba
            elif _ROMAN_STRICT.fullmatch(upper):
                try:
                    val = _roman_to_int(upper)
                    if 2 <= val <= 499:
                        _start, _end = m.start(), m.end()
                        _before = m.string[max(0, _start - 20):_start].lower()
                        _after = m.string[_end:_end + 30].lower()
                        if any(w in _before or w in _after for w in _ROMAN_CONTEXT_WORDS):
                            return num2words(val, lang="pl", to="ordinal")
                except Exception:
                    pass
            # Polskie litery diakrytyczne (Ą,Ć,Ę,Ł,Ń,Ó,Ś,Ź,Ż) → polskie słowo pisane caps
            # Jeśli wymawialny → lowercase, jeśli nie → literuj (np. ŚFN → eś ef en)
            if any(c in "ĄĆĘŁŃÓŚŹŻ" for c in upper):
                if _is_pronounceable(word):
                    return word.lower()
                return _spell_abbreviation(word)
            # Wymawialny → czytaj jako słowo
            if _is_pronounceable(word):
                return word.lower()
            # Niewymawialny → literuj
            return _spell_abbreviation(word)
        s = _RE_ALLCAPS.sub(_allcaps_expand, s)

        if s != s_before:
            logger.debug("foreign_expand: %d changes", sum(1 for a, b in zip(s.split(), s_before.split()) if a != b))
        return s

    # ── KROK 2: normalizacja liczb ────────────────────────────────────────────

    def num_normalize(self, text: str) -> str:
        """
        Zamienia liczby, daty, procenty, waluty i jednostki na słowa polskie.
        """
        return self._num.normalize(text)

    # ── KROK 3: filtr końcowy ─────────────────────────────────────────────────

    def final_filter(self, text: str) -> str:
        """
        Usuwa wszelkie znaki których nie ma w vocab PLTokenizer.
        Po tym kroku tekst można bezpiecznie przekazać do PLTokenizer.encode().
        Cyfry, które przeżyły (np. nie rozpoznane przez normalizator)
        są usuwane — nie trafią jako <unk>.
        """
        out = []
        for ch in text:
            if ch in _ALLOWED_CHARS:
                out.append(ch)
            elif self.keep_unknown_as_unk:
                out.append("?")   # marker do debugowania
            # else: pomiń
        result = "".join(out)
        # Zachowaj normalną interpunkcję; czyścimy tylko spacing i artefakty.
        # To nie wpływa na zamiany dat/liczb, bo dzieje się po normalizacji.
        result = re.sub(r"\.{4,}", "...", result)
        # Usuń spacje przed znakami interpunkcyjnymi.
        result = re.sub(r"\s+([,.;:!?])", r"\1", result)
        # Usuń spacje bezpośrednio po otwierających nawiasach/cudzysłowach.
        result = re.sub(r"([(\[\"'])\s+", r"\1", result)
        # Usuń spacje bezpośrednio przed zamykającymi nawiasami/cudzysłowami.
        result = re.sub(r"\s+([)\]\"'])", r"\1", result)
        # Dodaj pojedynczą spację po interpunkcji, jeśli dalej idzie zwykłe słowo.
        result = re.sub(r"([,;:!?])(?=[^\s\"')\]\-])", r"\1 ", result)
        result = re.sub(r"(\.)(?![\.\s\"')\]\-])", r"\1 ", result)
        # Sklej dialogowy myślnik po końcu zdania: ". - słowo" zostaje bez zmian.
        result = re.sub(r"\s{2,}", " ", result).strip()
        return result

    # ── Detekcja struktury tekstu ────────────────────────────────────────────

    # Typy segmentów i odpowiadające im pauzy w milisekundach
    SEG_CHAPTER   = "chapter"     # 1800 ms
    SEG_TITLE     = "title"       # 1200 ms
    SEG_DIVIDER   = "divider"     # 1000 ms
    SEG_PARAGRAPH = "paragraph"   #  700 ms
    SEG_TEXT      = "text"        #    0 ms (domyślna pauza modelu)

    PAUSE_MS = {
        "chapter":   1800,
        "title":     1200,
        "divider":   1000,
        "paragraph":  700,
        "text":         0,
    }

    # Wzorce nagłówków rozdziałów
    _RE_CHAPTER = re.compile(
        r'^(?:ROZDZIAŁ|CHAPTER|PART|CZĘŚĆ|PROLOG|EPILOG|WSTĘP|ZAKOŃCZENIE'
        r'|PODSUMOWANIE|POSŁOWIE|PRZEDMOWA|ANEKS|DODATEK|SPIS\s+TREŚCI)'
        r'(?:\s+[IVXLCDM0-9]+\.?\s*(?:[–—-]\s*.*)?)?$',
        re.IGNORECASE
    )

    # Separatory sekcji: ---, ***, ===, ___  (≥3 znaki)
    _RE_DIVIDER = re.compile(r'^[\-\*=_~]{3,}\s*$')

    @staticmethod
    def detect_structure(text: str) -> list:
        """
        Analizuje surowy tekst i dzieli go na segmenty z typem strukturalnym.

        Zwraca listę krotek (typ, tekst):
          - ("chapter",   "ROZDZIAŁ I — Geneza")
          - ("title",     "Tytuł sekcji")
          - ("divider",   "---")
          - ("paragraph", "Blok tekstu akapitu...")
          - ("text",      "Kontynuacja bez przerwy")

        Uruchamiany PRZED normalizacją — pracuje na surowym tekście.
        """
        pipe = PolishTTSPipeline
        lines = text.split('\n')
        segments = []
        current_para_lines = []

        def _flush_para():
            if current_para_lines:
                joined = ' '.join(current_para_lines)
                if joined.strip():
                    segments.append((pipe.SEG_PARAGRAPH, joined.strip()))
                current_para_lines.clear()

        i = 0
        while i < len(lines):
            line = lines[i]
            stripped = line.strip()

            # Pusta linia = granica akapitu
            if not stripped:
                _flush_para()
                i += 1
                continue

            # Separator sekcji: ---, ***, ===
            if pipe._RE_DIVIDER.match(stripped):
                _flush_para()
                segments.append((pipe.SEG_DIVIDER, stripped))
                i += 1
                continue

            # Nagłówek rozdziału (pasuje do wzorca)
            if pipe._RE_CHAPTER.match(stripped):
                _flush_para()
                segments.append((pipe.SEG_CHAPTER, stripped))
                i += 1
                continue

            # Tytuł: krótka linia (< 60 znaków), nie kończy się przecinkiem,
            # po której jest pusta linia lub koniec tekstu,
            # ALBO linia pisana CAPS (> 60% wielkich liter, ≥ 3 litery)
            is_short = len(stripped) < 60 and not stripped.endswith(',')
            next_empty = (i + 1 >= len(lines) or not lines[i + 1].strip())
            letters = [c for c in stripped if c.isalpha()]
            is_caps = (len(letters) >= 3 and
                       sum(1 for c in letters if c.isupper()) / len(letters) > 0.6)

            starts_upper = stripped[0].isupper() if stripped else False
            starts_digit = stripped[0].isdigit() if stripped else False
            starts_dash = stripped[0] in ('-', '–', '—') if stripped else False

            if is_caps and len(stripped) < 80 and not starts_digit and not starts_dash:
                # Linia ALL CAPS (≥60% wielkich) = tytuł/nagłówek
                _flush_para()
                segments.append((pipe.SEG_TITLE, stripped))
                i += 1
                continue

            if (is_short and next_empty and not stripped.endswith('.')
                    and starts_upper and not starts_digit and not starts_dash):
                # Krótka linia + pusta po niej + nie kończy się kropką
                # + zaczyna się wielką literą = tytuł
                _flush_para()
                segments.append((pipe.SEG_TITLE, stripped))
                i += 1
                continue

            # Element listy: linia zaczynająca się od numeru, litery z nawiasem,
            # myślnika/em-dasha → osobny akapit (pauza między elementami listy)
            is_list_item = bool(re.match(
                r'^(?:\d+[.)]\s|[a-z][.)]\s|[—–-]\s|[*•◦▪●◆◇]\s)', stripped
            ))
            if is_list_item:
                _flush_para()
                segments.append((pipe.SEG_PARAGRAPH, stripped))
                i += 1
                continue

            # Jeśli linia zaczyna się wielką literą a w buforze już jest
            # tekst — traktuj jako nowy akapit (pauza między wierszami)
            if current_para_lines and starts_upper:
                _flush_para()

            # Zwykły tekst — dodaj do bieżącego akapitu
            current_para_lines.append(stripped)
            i += 1

        _flush_para()

        # Scal kolejne tytuły w jeden (np. wieloliniowy tytuł książki)
        merged = []
        i = 0
        while i < len(segments):
            seg_type, seg_text = segments[i]
            if seg_type == pipe.SEG_TITLE:
                # Zbieraj kolejne tytuły
                title_parts = [seg_text]
                while i + 1 < len(segments) and segments[i + 1][0] == pipe.SEG_TITLE:
                    i += 1
                    title_parts.append(segments[i][1])
                merged.append((pipe.SEG_TITLE, ' '.join(title_parts)))
            else:
                merged.append((seg_type, seg_text))
            i += 1
        return merged

    def _process_title(self, text: str) -> str:
        """
        Normalizacja tytułów/nagłówków — bez foreign_expand.
        Tytuły pisane CAPS (np. "NOC RÓŻNA OD WSZYSTKICH INNYCH NOCY")
        nie powinny być literowane jako skróty.
        """
        parts = self._split_preserved(text)
        if len(parts) > 1:
            out = []
            for part in parts:
                if re.fullmatch(r"<[^<>]+>", part):
                    out.append(self._canonicalize_tag(part))
                else:
                    out.append(self._process_title(part))
            merged = "".join(out)
            merged = re.sub(r"\s*(<[^<>]+>)\s*", r" \1 ", merged)
            return re.sub(r"\s+", " ", merged).strip()

        s = self.raw_clean(text)
        s = self.abbreviation_expand(s)

        # Zamień rzymskie numery w nagłówkach rozdziałów na liczebniki porządkowe
        # "ROZDZIAŁ I" → "ROZDZIAŁ pierwszy", "CZĘŚĆ III" → "CZĘŚĆ trzeci"
        def _chapter_roman_to_ordinal(m):
            prefix, roman = m.group(1), m.group(2).upper()
            try:
                val = _roman_to_int(roman)
                ordinal = num2words(val, lang="pl", to="ordinal")
                return f"{prefix} {ordinal}"
            except Exception:
                return m.group(0)
        s = re.sub(
            r'\b(ROZDZIAŁ|CHAPTER|CZĘŚĆ|PART|TOM|AKT|KSIĘGA|PIEŚŃ)\s+([IVXLCDM]+)\b',
            _chapter_roman_to_ordinal, s, flags=re.IGNORECASE
        )

        # Pomijamy foreign_expand — tytuły to nazwy własne, nie skróty
        s = self.num_normalize(s)
        s = self.final_filter(s)
        if not self.preserve_case:
            s = s.lower()
        return s

    def process_structured(self, text: str) -> list:
        """
        Dzieli tekst na segmenty strukturalne, normalizuje każdy osobno.

        Zwraca listę krotek (typ, znormalizowany_tekst, pauza_ms):
          [("chapter", "rozdział pierwszy geneza", 1800),
           ("paragraph", "wszystko zaczęło się...", 700),
           ...]
        """
        segments = self.detect_structure(text)
        result = []
        for seg_type, seg_text in segments:
            if seg_type == self.SEG_DIVIDER:
                # Separator — nie generujemy mowy, tylko pauzę
                result.append((seg_type, "", self.PAUSE_MS[seg_type]))
                continue
            if seg_type in (self.SEG_TITLE, self.SEG_CHAPTER):
                normalized = self._process_title(seg_text)
            else:
                normalized = self.process(seg_text)
            if normalized.strip():
                result.append((seg_type, normalized.strip(),
                               self.PAUSE_MS[seg_type]))
        return result

    # ── Główne API ────────────────────────────────────────────────────────────

    def process(self, text: str) -> str:
        """
        Pełny potok: raw_clean → abbreviation_expand → foreign_expand
                      → num_normalize → final_filter.
        Zwraca tekst gotowy dla PLTokenizer.
        """
        logger.debug("process() input (%d chars): %.100s", len(text), text)
        parts = self._split_preserved(text)
        if len(parts) > 1:
            out = []
            for part in parts:
                if re.fullmatch(r"<[^<>]+>", part):
                    out.append(self._canonicalize_tag(part))
                else:
                    out.append(self._process_plain(part))
            s = "".join(out)
            s = re.sub(r"\s*(<[^<>]+>)\s*", r" \1 ", s)
            s = re.sub(r"\s+", " ", s).strip()
        else:
            s = self.raw_clean(text)
            logger.debug("after raw_clean: %.100s", s)
            s = self.abbreviation_expand(s)
            logger.debug("after abbreviation_expand: %.100s", s)
            s = self.foreign_expand(s)
            logger.debug("after foreign_expand: %.100s", s)
            s = self.num_normalize(s)
            logger.debug("after num_normalize: %.100s", s)
            s = self.final_filter(s)
            if not self.preserve_case:
                s = s.lower()
        logger.debug("process() output (%d chars): %.100s", len(s), s)
        return s

    def process_batch(self, texts: list) -> list:
        """Przetwarza listę tekstów."""
        return [self.process(t) for t in texts]

    def debug(self, text: str) -> dict:
        """
        Zwraca wyniki pośrednie każdego kroku — przydatne do debugowania.
        Includes tokenization step showing boundary tokens per chunk.
        Splits on newlines (like the actual synthesis pipeline) to show
        how each paragraph/line becomes a separate chunk with BOS/EOS.
        """
        s0 = text

        # Split on newlines first (mirrors _normalize_plain behavior)
        paragraphs = [p.strip() for p in re.split(r'\n+', s0) if p.strip()]
        if not paragraphs:
            paragraphs = [s0]

        # Run pipeline on each paragraph separately
        per_paragraph = []
        for para in paragraphs:
            s1 = self.raw_clean(para)
            s1b = self.abbreviation_expand(s1)
            s1c = self.foreign_expand(s1b)
            s2 = self.num_normalize(s1c)
            s3 = self.final_filter(s2)
            if not self.preserve_case:
                s3 = s3.lower()
            per_paragraph.append(s3)

        # Also run whole text through pipeline (single pass) for comparison
        s1_full = self.raw_clean(s0)
        s1b_full = self.abbreviation_expand(s1_full)
        s1c_full = self.foreign_expand(s1b_full)
        s2_full = self.num_normalize(s1c_full)
        s3_full = self.final_filter(s2_full)
        if not self.preserve_case:
            s3_full = s3_full.lower()

        # Tokenization step: show boundary tokens per chunk
        tok = PLTokenizer()
        tokenized_chunks = []
        for chunk_text in per_paragraph:
            if not chunk_text.strip():
                continue
            ids = tok.encode(chunk_text)
            decoded = tok.decode(ids)
            tokenized_chunks.append({
                "text": chunk_text,
                "tokens": decoded,
                "token_ids": ids,
                "num_tokens": len(ids),
            })

        return {
            "input": s0,
            "after_raw_clean": s1_full,
            "after_abbreviation_expand": s1b_full,
            "after_foreign_expand": s1c_full,
            "after_num_normalize": s2_full,
            "output": s3_full,
            "tokenized_chunks": tokenized_chunks,
            "total_chunks": len(tokenized_chunks),
            "total_tokens": sum(c["num_tokens"] for c in tokenized_chunks),
        }


# ─────────────────────────────────────────────────────────────────────────────
# CELL 6b ▸ Wbudowany PLTokenizer
# ─────────────────────────────────────────────────────────────────────────────

import json
import unicodedata as _ud
from pathlib import Path as _Path
from typing import List as _List, Dict as _Dict, Optional as _Optional

class PLTokenizer:
    """
    Wbudowany tokenizer języka polskiego (centralna implementacja).
    Vocab automatycznie tworzony przy pierwszym uruchomieniu
    i zapisywany do polish_vocab_cap_reserved.json (jeśli podano ścieżkę).

    Cechy:
      • lowercase vocab z markerem <CAP> przed wyrazami pisanymi wielką literą
      • <sp> jako separator słów wewnątrz tekstu
      • digrafy: sz cz dz dź dż ch rz jako pojedyncze tokeny
      • domyślnie zachowuje boundary jako <sp> (bez automatycznego BOS/EOS)
      • opcjonalnie może pracować w trybie legacy BOS/EOS na granicach chunka
      • <reserved5> zostaje wolny jako mask token dla MLM/BERT
    """

    SPECIALS = [
        "<pad>", "<unk>", "<sp>",
        "<CAP>",
        "<BOS>", "<EOS>", "<reserved3>", "<reserved4>", "<reserved5>",
        "<nar>", "<akt>",
    ]
    PUNCT = list('.,!?:;-\"\'()[]…/') + [
        "→","←","@","#","$","%","&","*","+","=","<",">","^","_","|","~"
    ]
    LETTERS = list("abcdefghijklmnopqrstuvwxyząćęłńóśźż")

    def __init__(self, vocab_path: _Optional[str] = None, boundary_mode: str = "sp"):
        self.vocab_path = _Path(vocab_path) if vocab_path else _Path(__file__).with_name("polish_vocab_cap_reserved.json")
        if boundary_mode not in {"sp", "bos_eos"}:
            raise ValueError(f"Unsupported boundary_mode={boundary_mode!r}; expected 'sp' or 'bos_eos'")
        self.boundary_mode = boundary_mode
        self.digraph_map = {
            "dź": "<dź>", "dż": "<dż>",
            "sz": "<sz>", "cz": "<cz>", "dz": "<dz>",
            "ch": "<ch>", "rz": "<rz>",
        }
        self.rev_digraph_map = {v: k for k, v in self.digraph_map.items()}

        if self.vocab_path.exists():
            self._load_vocab()
            for group in (self.SPECIALS, list(self.digraph_map.values()),
                          self.LETTERS, self.PUNCT):
                self._ensure_tokens(group)
        else:
            self._create_vocab()
            self._save_vocab()

        self.pad_id = self.token2id["<pad>"]
        self.unk_id = self.token2id["<unk>"]
        self.sp_id  = self.token2id["<sp>"]
        self.cap_id = self.token2id["<CAP>"]
        self.bos_id = self.token2id["<BOS>"]
        self.eos_id = self.token2id["<EOS>"]

    # ── normalizacja wewnętrzna ───────────────────────────────────────────────

    @staticmethod
    def _nfc(text: str) -> str:
        text = _ud.normalize("NFC", text)
        for bad, good in [
            ("\u00A0"," "),("\u202F"," "),("\u2009"," "),
            ("\u2011","-"),("–","-"),("—","-"),
        ]:
            text = text.replace(bad, good)
        return re.sub(r"\s+", " ", text.strip())

    @staticmethod
    def _is_upper_pl(ch: str) -> bool:
        return len(ch) == 1 and ch.isalpha() and ch.isupper()

    def _insert_cap_markers(self, text: str) -> str:
        out, prev_alpha = [], False
        for c in text:
            if c.isalpha():
                if not prev_alpha and self._is_upper_pl(c):
                    out.append("<CAP>")
                    out.append(c.lower())
                else:
                    out.append(c)
                prev_alpha = True
            else:
                out.append(c)
                prev_alpha = False
        return "".join(out)

    @staticmethod
    def _lower_preserve_angle(text: str) -> str:
        out, inside = [], False
        for c in text:
            if c == "<":   inside = True;  out.append(c); continue
            if c == ">":   inside = False; out.append(c); continue
            out.append(c if inside else c.lower())
        return "".join(out)

    def _apply_digraphs(self, text: str) -> str:
        for pat, tok in sorted(self.digraph_map.items(), key=lambda kv: len(kv[0]), reverse=True):
            text = text.replace(pat, tok)
        return text

    def _pretokenize(self, text: str) -> _List[str]:
        text = self._nfc(text)
        text = self._insert_cap_markers(text)
        text = self._lower_preserve_angle(text)
        text = self._apply_digraphs(text)
        text = re.sub(r" ", " <sp> ", text)

        toks, i = [], 0
        while i < len(text):
            if text[i] == "<":
                j = text.find(">", i + 1)
                if j != -1:
                    toks.append(text[i:j+1]); i = j + 1; continue
            toks.append(text[i]); i += 1
        return [t.strip() for t in toks if t.strip()]

    def _collapse_special_runs(self, ids: _List[int]) -> _List[int]:
        if not ids:
            return ids
        special_run_ids = {self.sp_id, self.bos_id, self.eos_id}
        out = [ids[0]]
        for tok_id in ids[1:]:
            if tok_id in special_run_ids and out[-1] == tok_id:
                continue
            out.append(tok_id)
        return out

    def _apply_boundary_mode(self, ids: _List[int]) -> _List[int]:
        if not ids:
            return ids
        ids = [self.sp_id if tok_id in (self.bos_id, self.eos_id) else tok_id for tok_id in ids]
        ids = self._collapse_special_runs(ids)
        if self.boundary_mode == "sp":
            return ids
        if ids[0] == self.sp_id:
            ids[0] = self.bos_id
        else:
            ids = [self.bos_id] + ids
        if ids[-1] == self.sp_id:
            ids[-1] = self.eos_id
        else:
            ids = ids + [self.eos_id]
        return self._collapse_special_runs(ids)

    # ── vocab ─────────────────────────────────────────────────────────────────

    def _create_vocab(self) -> None:
        vocab: _Dict[str, int] = {}
        idx = 0
        for group in (self.SPECIALS, self.LETTERS,
                      list(self.digraph_map.values()), self.PUNCT):
            for t in group:
                if t not in vocab:
                    vocab[t] = idx; idx += 1
        self.token2id = vocab
        self.id2token = {i: t for t, i in vocab.items()}

    def _ensure_tokens(self, toks: _List[str]) -> None:
        max_id = max(self.id2token.keys()) if self.id2token else -1
        for t in toks:
            if t not in self.token2id:
                max_id += 1
                self.token2id[t] = max_id
                self.id2token[max_id] = t

    def _save_vocab(self) -> None:
        self.vocab_path.write_text(
            json.dumps(self.token2id, ensure_ascii=False, indent=2), encoding="utf-8")

    def _load_vocab(self) -> None:
        self.token2id = json.loads(self.vocab_path.read_text(encoding="utf-8"))
        legacy_inverse = {int(v): k for k, v in self.token2id.items()}
        changed = False
        for special_id, new_name in ((4, "<BOS>"), (5, "<EOS>")):
            old_name = legacy_inverse.get(special_id)
            if old_name is not None and old_name != new_name:
                self.token2id.pop(old_name, None)
                self.token2id[new_name] = special_id
                changed = True
        self.id2token = {int(v): k for k, v in self.token2id.items()}
        if changed:
            try:
                self._save_vocab()
            except OSError:
                pass  # read-only filesystem (Docker :ro mount)

    # ── publiczne API ─────────────────────────────────────────────────────────

    def encode(self, text: str) -> _List[int]:
        """Tekst → lista ID tokenów; domyślnie zachowuje boundary jako <sp>."""
        text = re.sub(r"(?i)<\s*(?:bos|eos)\s*>", "<sp>", text)
        text = re.sub(r"(?i)<\s*sp\s*>", "<sp>", text)
        ids = [self.token2id.get(t, self.unk_id) for t in self._pretokenize(text)]
        return self._apply_boundary_mode(ids)

    def decode(self, ids: _List[int]) -> str:
        """Lista ID tokenów → tekst."""
        out = []
        for i in ids:
            t = self.id2token.get(int(i), "<unk>")
            if   t == "<sp>":             out.append(" ")
            elif t == "<BOS>":            out.append("<BOS>")
            elif t == "<EOS>":            out.append("<EOS>")
            elif t in self.rev_digraph_map: out.append(self.rev_digraph_map[t])
            elif t == "<CAP>":            out.append("<CAP>")
            else:                         out.append(t)
        return "".join(out)

    @property
    def vocab_size(self) -> int:
        return len(self.token2id)

    def count_unk(self, text: str) -> int:
        """Zwraca liczbę tokenów <unk> w zakodowanym tekście."""
        return self.encode(text).count(self.unk_id)

    def split_words(self, text: str) -> _List[str]:
        return [w for w in self._nfc(text).split(" ") if w]


# ─────────────────────────────────────────────────────────────────────────────
# CELL 6c ▸ Czytniki plików: TXT, PDF, EPUB, MOBI
# ─────────────────────────────────────────────────────────────────────────────

import tempfile
import os
from pathlib import Path

class FileReader:
    """
    Ekstrakcja czystego tekstu z plików: .txt .pdf .epub .mobi

    Użycie:
        reader = FileReader()
        text   = reader.read("ksiazka.epub")
        # lub rozdzielony na rozdziały:
        parts  = reader.read_chapters("ksiazka.epub")
    """

    SUPPORTED = {".txt", ".pdf", ".epub", ".mobi"}

    # ── publiczne API ─────────────────────────────────────────────────────────

    def read(self, path: str | Path) -> str:
        """Zwraca cały tekst pliku jako jeden string."""
        path = Path(path)
        self._check(path)
        ext = path.suffix.lower()
        if   ext == ".txt":  return self._read_txt(path)
        elif ext == ".pdf":  return self._read_pdf(path)
        elif ext == ".epub": return self._read_epub(path)
        elif ext == ".mobi": return self._read_mobi(path)

    def read_chapters(self, path: str | Path) -> list[dict]:
        """
        Zwraca listę słowników: [{"title": str, "text": str}, ...]
        Dla TXT/PDF zwraca jeden element bez tytułu.
        """
        path = Path(path)
        self._check(path)
        ext = path.suffix.lower()
        if   ext == ".txt":  return [{"title": path.stem, "text": self._read_txt(path)}]
        elif ext == ".pdf":  return self._pdf_chapters(path)
        elif ext == ".epub": return self._epub_chapters(path)
        elif ext == ".mobi": return self._mobi_chapters(path)

    # ── walidacja ─────────────────────────────────────────────────────────────

    def _check(self, path: Path) -> None:
        if not path.exists():
            raise FileNotFoundError(f"Plik nie istnieje: {path}")
        if path.suffix.lower() not in self.SUPPORTED:
            raise ValueError(
                f"Nieobsługiwany format: {path.suffix!r}. "
                f"Obsługiwane: {self.SUPPORTED}")

    # ── TXT ───────────────────────────────────────────────────────────────────

    @staticmethod
    def _read_txt(path: Path) -> str:
        for enc in ("utf-8", "utf-8-sig", "cp1250", "iso-8859-2", "latin-1"):
            try:
                return path.read_text(encoding=enc)
            except (UnicodeDecodeError, LookupError):
                continue
        return path.read_bytes().decode("utf-8", errors="replace")

    # ── PDF ───────────────────────────────────────────────────────────────────

    @staticmethod
    def _read_pdf(path: Path) -> str:
        import fitz  # pymupdf
        doc = fitz.open(str(path))
        pages = []
        for page in doc:
            pages.append(page.get_text("text"))
        doc.close()
        return "\n".join(pages)

    @staticmethod
    def _pdf_chapters(path: Path) -> list[dict]:
        import fitz
        doc = fitz.open(str(path))
        toc = doc.get_toc()  # [[level, title, page], ...]
        pages_text = [page.get_text("text") for page in doc]
        doc.close()

        if not toc:
            return [{"title": path.stem, "text": "\n".join(pages_text)}]

        chapters = []
        for i, (_, title, start_page) in enumerate(toc):
            end_page = toc[i + 1][2] if i + 1 < len(toc) else len(pages_text) + 1
            text = "\n".join(pages_text[start_page - 1: end_page - 1])
            chapters.append({"title": title.strip(), "text": text})
        return chapters

    # ── EPUB ──────────────────────────────────────────────────────────────────

    @staticmethod
    def _html_to_text(html_bytes: bytes) -> str:
        import html2text
        h = html2text.HTML2Text()
        h.ignore_links      = True
        h.ignore_images     = True
        h.ignore_emphasis   = True
        h.ignore_tables     = True
        h.body_width        = 0    # nie zawijaj
        h.unicode_snob      = True
        raw = html_bytes.decode("utf-8", errors="replace")
        text = h.handle(raw)
        # Usuń znaczniki Markdown które html2text zostawia
        text = re.sub(r"#+\s*", "", text)
        text = re.sub(r"\*{1,2}(.*?)\*{1,2}", r"\1", text)
        text = re.sub(r"_{1,2}(.*?)_{1,2}", r"\1", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    @classmethod
    def _read_epub(cls, path: Path) -> str:
        chapters = cls._epub_chapters(path)
        return "\n\n".join(c["text"] for c in chapters)

    @staticmethod
    def _epub_chapters(path: Path) -> list[dict]:
        from ebooklib import epub, ITEM_DOCUMENT
        import html2text

        book = epub.read_epub(str(path), options={"ignore_ncx": True})
        chapters = []

        h = html2text.HTML2Text()
        h.ignore_links = h.ignore_images = h.ignore_emphasis = h.ignore_tables = True
        h.body_width = 0
        h.unicode_snob = True

        for item in book.get_items_of_type(ITEM_DOCUMENT):
            raw = item.get_body_content()
            if not raw or len(raw.strip()) < 50:
                continue
            text = h.handle(raw.decode("utf-8", errors="replace"))
            text = re.sub(r"#+\s*", "", text)
            text = re.sub(r"\*{1,3}(.*?)\*{1,3}", r"\1", text, flags=re.S)
            text = re.sub(r"_{1,3}(.*?)_{1,3}", r"\1", text, flags=re.S)
            text = re.sub(r"\n{3,}", "\n\n", text).strip()
            if not text:
                continue

            # Spróbuj wyciągnąć tytuł z pierwszej linii lub metadanych
            title = item.get_name().rsplit("/", 1)[-1].replace(".html","").replace(".xhtml","")
            lines = [l.strip() for l in text.split("\n") if l.strip()]
            if lines and len(lines[0]) < 80 and not lines[0].endswith("."):
                title = lines[0]
                text = "\n".join(lines[1:])

            chapters.append({"title": title, "text": text})

        return chapters if chapters else [{"title": path.stem, "text": ""}]

    # ── MOBI ──────────────────────────────────────────────────────────────────

    @classmethod
    def _read_mobi(cls, path: Path) -> str:
        chapters = cls._mobi_chapters(path)
        return "\n\n".join(c["text"] for c in chapters)

    @classmethod
    def _mobi_chapters(cls, path: Path) -> list[dict]:
        """
        MOBI → rozpakuj do EPUB (mobi.extract) → czytaj jak EPUB.
        Działa z plikami .mobi i .azw (niezaszyfrowanymi).
        """
        from mobi import extract
        with tempfile.TemporaryDirectory() as tmpdir:
            # extract() zwraca (outdir, epub_path)
            try:
                _, epub_path = extract(str(path))
                epub_p = Path(epub_path)
                if epub_p.exists():
                    # Skopiuj do tmpdir żeby nie zaśmiecać
                    dest = Path(tmpdir) / epub_p.name
                    import shutil
                    shutil.copy2(epub_p, dest)
                    return cls._epub_chapters(dest)
            except Exception as e:
                # Fallback: spróbuj czytać jako surowy HTML z pliku MOBI
                pass

        # Ostateczny fallback — surowy tekst z bytes
        raw = path.read_bytes()
        text = raw.decode("utf-8", errors="replace")
        # Usuń nagłówek binarny — znajdź pierwszą sensowną treść HTML
        html_start = text.find("<html")
        if html_start == -1:
            html_start = text.find("<HTML")
        if html_start != -1:
            text = cls._html_to_text(text[html_start:].encode("utf-8", errors="replace"))
        return [{"title": path.stem, "text": text}]


# ─────────────────────────────────────────────────────────────────────────────
# CELL 6d ▸ Rozszerzone API PolishTTSPipeline: process_file()
# ─────────────────────────────────────────────────────────────────────────────

# Dodaj metody do PolishTTSPipeline przez monkey-patching (żeby nie duplikować klasy)

def _process_file(self, path: str | Path, *,
                  by_chapter: bool = False,
                  tokenize: bool = False,
                  vocab_path: str | None = None) -> "str | list":
    """
    Wczytuje plik (txt/pdf/epub/mobi), normalizuje tekst i opcjonalnie tokenizuje.

    Parametry
    ---------
    path        : ścieżka do pliku
    by_chapter  : jeśli True → zwraca listę słowników
                  [{"title": str, "text": str, "tokens": list|None}]
    tokenize    : jeśli True → dołącza pole "tokens" (lista int ID)
    vocab_path  : ścieżka do pliku vocab PLTokenizer (domyślnie auto)

    Zwraca
    ------
    str  (gdy by_chapter=False) — cały znormalizowany tekst
    list (gdy by_chapter=True)  — lista rozdziałów
    """
    reader = FileReader()
    tok    = PLTokenizer(vocab_path) if tokenize else None

    if by_chapter:
        chapters = reader.read_chapters(path)
        out = []
        for ch in chapters:
            normalized = self.process(ch["text"])
            entry = {"title": ch["title"], "text": normalized}
            if tok is not None:
                entry["tokens"] = tok.encode(normalized)
            out.append(entry)
        return out
    else:
        raw = reader.read(path)
        normalized = self.process(raw)
        if tok:
            return tok.encode(normalized)
        return normalized

def _tokenize(self, text: str, vocab_path: str | None = None) -> list[int]:
    """
    Normalizuje i tokenizuje tekst.
    Skrót: pipeline.tokenize("Zysk wyniósł 3,5 mld zł.")
    """
    tok = PLTokenizer(vocab_path)
    return tok.encode(self.process(text))

def _count_unk(self, text: str, vocab_path: str | None = None) -> int:
    """Liczba tokenów <unk> po normalizacji — 0 oznacza brak nieznanych znaków."""
    tok = PLTokenizer(vocab_path)
    return tok.count_unk(self.process(text))

PolishTTSPipeline.process_file = _process_file
PolishTTSPipeline.tokenize     = _tokenize
PolishTTSPipeline.count_unk    = _count_unk


# ─────────────────────────────────────────────────────────────────────────────
# CELL 7 ▸ Testy i demo
# ─────────────────────────────────────────────────────────────────────────────

def run_demo():
    pipe = PolishTTSPipeline()
    W = 74

    # ── Testy jednostkowe ────────────────────────────────────────────────────
    test_cases = [
        # FORMAT: (opis, wejście, oczekiwany wynik)

        # ── Surowe czyszczenie ────────────────────────────────────────────────
        ("Entery i wielokrotne spacje",
         "Ala\nma\r\nkota.\t  Kot  ma  Alę.",
         "Ala ma kota. Kot ma Alę."),

        ("Myślniki Unicode",
         "Dialog\u2013krótki, myśl\u2014długa, minus\u2212pięć.",
         "Dialog-krótki, myśl-długa, minus-pięć."),

        ("Znaki niemieckie",
         "Über die Größe: Müller, Schröder.",
         "uber die Grosse: Muller, Schroder."),

        ("Cudzysłowy",
         "Powiedział \u201eDobrze\u201d i \u00abĆwierć\u00bb.",
         "Powiedział Dobrze i Ćwierć."),

        ("Znaki koreańskie/chińskie",
         "Produkt 한국어 z ceną 50 zł oraz 字典 słownik.",
         "Produkt z ceną pięćdziesiąt złotych oraz słownik."),

        ("Emoji i symbole",
         "Świetnie! 😀🔥 Cena: 99 zł.",
         "Świetnie! Cena: dziewięćdziesiąt dziewięć złotych."),

        # ── Daty ──────────────────────────────────────────────────────────────
        ("Data pełna",
         "Spotkanie 15.03.2024",
         "Spotkanie piętnastego marca dwa tysiące dwudziestego czwartego roku"),

        ("Data z r.",
         "W 2023 r. wyniki były dobre.",
         "W dwa tysiące dwudziestego trzeciego roku wyniki były dobre."),

        # ── Procenty ──────────────────────────────────────────────────────────
        ("Procenty całkowite",
         "Wzrost o 22%",
         "Wzrost o dwadzieścia dwa procenty"),

        ("Procenty ułamkowe",
         "Inflacja 3,5%",
         "Inflacja trzy przecinek pięć procenta"),

        # ── Waluty ────────────────────────────────────────────────────────────
        ("Waluta z groszami",
         "Cena 12,99 zł.",
         "Cena dwanaście złotych dziewięćdziesiąt dziewięć groszy."),

        ("Waluta z separatorem",
         "Zarobił 1 000 000 PLN",
         "Zarobił milion złotych"),

        ("Skrót mld + waluta",
         "Zysk 2,5 mld zł",
         None),  # zależy od kolejności abbreviation_expand vs num_normalize

        # ── Jednostki ─────────────────────────────────────────────────────────
        ("Jednostki odległości",
         "Trasa 42 km",
         "Trasa czterdzieści dwa kilometry"),

        ("Temperatura",
         "Było -5°C",
         "Było minus pięć stopni Celsjusza"),

        ("Separatorem tysięcy + jednostka",
         "Odległość 1 000 km",
         "Odległość tysiąc kilometrów"),

        # ── Godziny ───────────────────────────────────────────────────────────
        ("Godzina pełna",
         "Pociąg o 8:00",
         "Pociąg o osiem godzin"),

        ("Godzina z minutami",
         "Alarm o 7:01",
         "Alarm o siedem godzin jedna minuta"),

        # ── Myślniki dialogowe ────────────────────────────────────────────────
        ("Myślnik dialogowy en-dash",
         "\u2013 Dzień dobry \u2013 powiedział.",
         "- Dzień dobry - powiedział."),

        # ── Filtr końcowy ─────────────────────────────────────────────────────
        ("Cyfry resztkowe usunięte",
         "Tekst abc 999xyz",
         "Tekst abc xyz"),

        # ── Skróty międzynarodowe ────────────────────────────────────────────
        ("Skrót TTS",
         "System TTS jest ważny",
         "System te te es jest ważny"),

        ("Skrót API",
         "Użyj API do połączenia",
         "Użyj a pe i do połączenia"),

        ("NATO jako słowo",
         "Pakt NATO obowiązuje",
         "Pakt nato obowiązuje"),

        ("COVID-19 mieszany",
         "Pandemia COVID-19 trwała",
         "Pandemia kowid dziewiętnaście trwała"),

        ("GPT-4 mieszany",
         "Model GPT-4 jest nowy",
         "Model gie pe te cztery jest nowy"),

        ("Angielskie imię",
         "William Shakespeare napisał",
         "Uiliam Szekspir napisał"),

        ("Wi-Fi mieszany",
         "Sieć Wi-Fi działa",
         "Sieć łaj faj działa"),

        ("Zapożyczenie angielskie",
         "Nowy startup z online marketingu",
         "Nowy startap z onlajn marketingu"),

        ("Nieznany skrót - literowanie",
         "Protokół XMPP jest używany",
         "Protokół iks em pe pe jest używany"),

        ("Skrót z kropkami U.S.A.",
         "Flaga U.S.A. jest kolorowa",
         "Flaga u es a jest kolorowa"),

        ("Skrót e.g.",
         "Używaj e.g. tego narzędzia",
         "Używaj na przykład tego narzędzia"),

        # ── Złożone ───────────────────────────────────────────────────────────
        ("Zdanie złożone",
         "W 3. kwartale 2023 r. sprzedaż wzrosła o 22% do 4,7 mld zł.",
         None),  # None = tylko wyświetl

        ("Pełny tekst z obcymi znakami",
         "Firma 한국 GmbH osiągnęła 3,5 mld zł\u2013 to 12% więcej niż w 2022 r.",
         None),
    ]

    print(f"\n{'═'*W}")
    print("  POLISH TTS PIPELINE — TESTY")
    print(f"{'═'*W}")

    total = passed = failed = 0
    for desc, inp, expected in test_cases:
        got = pipe.process(inp)
        total += 1

        if expected is None:
            print(f"\n  📝 {desc}")
            print(f"     WE:  {inp!r}")
            print(f"     WY:  {got!r}")
            continue

        ok = got.strip() == expected.strip()
        passed += (1 if ok else 0)
        failed += (0 if ok else 1)
        sym = "✅" if ok else "❌"
        print(f"\n  {sym}  {desc}")
        print(f"     WE:  {inp!r}")
        print(f"     WY:  {got!r}")
        if not ok:
            print(f"     OK:  {expected!r}")

    print(f"\n{'═'*W}")
    print(f"  WYNIKI: {passed}/{total-2}  ({failed} nieudanych)")
    print(f"{'═'*W}\n")

    # ── Demo debugowania krok po kroku ───────────────────────────────────────
    print("\n🔍 DEBUG — kroki pośrednie\n" + "─"*W)
    debug_cases = [
        "Dnia 15.03.2024 r. firma 한국 osiągnęła 3,5 mld zł – wzrost o 12%.",
        "„Cena\u201d to tylko 29,99 zł… Sprawdź na www!",
        "Temperatura: -20°C, wiatr 30\u201350 km/h, śnieg do 15 cm.",
    ]
    for s in debug_cases:
        d = pipe.debug(s)
        print(f"\n  INPUT:       {d['input']!r}")
        print(f"  raw_clean:   {d['after_raw_clean']!r}")
        print(f"  num_norm:    {d['after_num_normalize']!r}")
        print(f"  OUTPUT:      {d['output']!r}")

    # ── Weryfikacja tokenizera (jeśli dostępny) ───────────────────────────────
    print("\n\n🔤 WERYFIKACJA z PLTokenizer\n" + "─"*W)
    tok = PLTokenizer()
    verify_texts = [
        "Zysk wyniósł trzy miliardy złotych i dwanaście procent wzrostu.",
        "Spotkanie piętnastego marca dwa tysiące dwudziestego czwartego roku.",
        "Alarm o siedem godzin jedna minuta.",
        "Dnia 15.03.2024 r. firma osiągnęła 3,5 mld zł – wzrost o 12%.",
        "Pociąg o 08:15, bagaż do 23 kg, cena £180.",
    ]
    print(f"  Vocab: {tok.vocab_size} tokenów\n")
    all_ok = True
    for t in verify_texts:
        processed = pipe.process(t)
        ids = tok.encode(processed)
        unk_count = ids.count(tok.unk_id)
        ok = unk_count == 0
        if not ok: all_ok = False
        status = f"✅ {len(ids):3d} tokenów" if ok else f"❌ {unk_count} UNK"
        print(f"  {status}  {processed[:65]!r}")
    print(f"\n  {chr(9989) + ' Wszystkie zdania: 0 UNK' if all_ok else chr(10060) + ' Są tokeny UNK!'}\n")


# ─────────────────────────────────────────────────────────────────────────────
# CELL 8 ▸ Uruchomienie
# ─────────────────────────────────────────────────────────────────────────────

def _demo_files():
    """Demonstracja FileReader + PLTokenizer na syntetycznych plikach."""
    import tempfile, os
    from ebooklib import epub as _epub
    import fitz

    pipe   = PolishTTSPipeline()
    reader = FileReader()
    tok    = PLTokenizer()
    W      = 74

    print(f"\n\n{chr(9552)*W}")
    print("  📂 DEMO CZYTNIKÓW PLIKÓW (TXT / PDF / EPUB / MOBI)")
    print(f"{chr(9552)*W}\n")

    # ── TXT ──────────────────────────────────────────────────────────────────
    with tempfile.NamedTemporaryFile(suffix=".txt", mode="w", encoding="utf-8", delete=False) as f:
        f.write("Wyniki za 2023 r.\nZysk netto: 3,5 mld zl.\nWzrost o 22%.\nWiatr 30-50 km/h.")
        tmp_txt = f.name
    text = pipe.process_file(tmp_txt)
    ids  = tok.encode(text)
    unk  = ids.count(tok.unk_id)
    print(f"  📄 TXT  tokens={len(ids)}  unk={unk}  {chr(9989) if unk==0 else chr(10060)}")
    print(f"     {text[:100]!r}")
    os.unlink(tmp_txt)

    # ── PDF ──────────────────────────────────────────────────────────────────
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
        tmp_pdf = f.name
    doc  = fitz.open()
    font = fitz.Font("helv")
    page = doc.new_page()
    tw   = fitz.TextWriter(page.rect)
    tw.append((50, 70),  "Raport Q1 2024 r.", font=font, fontsize=14)
    tw.append((50, 110), "Przychody: 1,2 mld zl, wzrost o 15%.", font=font, fontsize=12)
    tw.append((50, 140), "Temperatura zewnetrzna -5 stopni Celsjusza.", font=font, fontsize=12)
    tw.write_text(page)
    doc.save(tmp_pdf); doc.close()
    text = pipe.process_file(tmp_pdf)
    ids  = tok.encode(text)
    unk  = ids.count(tok.unk_id)
    print(f"\n  📄 PDF  tokens={len(ids)}  unk={unk}  {chr(9989) if unk==0 else chr(10060)}")
    print(f"     {text[:100]!r}")
    os.unlink(tmp_pdf)

    # ── EPUB ─────────────────────────────────────────────────────────────────
    with tempfile.NamedTemporaryFile(suffix=".epub", delete=False) as f:
        tmp_epub = f.name
    book = _epub.EpubBook()
    book.set_title("Demo"); book.set_language("pl")
    ch1 = _epub.EpubHtml(title="Wstep", file_name="ch1.xhtml")
    ch1.content = ("<html><body><h1>Wstep</h1>"
                   "<p>Zysk 2,5 mld zl, wzrost o 22% w 2023 r.</p></body></html>")
    ch2 = _epub.EpubHtml(title="Rozdzial 1", file_name="ch2.xhtml")
    ch2.content = ("<html><body><h1>Rozdzial 1</h1>"
                   "<p>Bagaz do 23 kg, wiatr 30-50 km/h, cena 299 zl.</p></body></html>")
    for item in (ch1, ch2, _epub.EpubNcx(), _epub.EpubNav()):
        book.add_item(item)
    book.spine = ["nav", ch1, ch2]
    _epub.write_epub(tmp_epub, book)
    chapters = pipe.process_file(tmp_epub, by_chapter=True, tokenize=True)
    print(f"\n  📗 EPUB ({len([c for c in chapters if c['text']])} rozdzialy z tekstem)")
    for ch in chapters:
        if not ch["text"]: continue
        unk = ch["tokens"].count(tok.unk_id)
        print(f"     [{ch['title']}] tokens={len(ch['tokens'])}  unk={unk}  {chr(9989) if unk==0 else chr(10060)}")
        print(f"       {ch['text'][:80]!r}")
    os.unlink(tmp_epub)

    print(f"\n  Obsługiwane formaty : {sorted(FileReader.SUPPORTED)}")
    print(f"  Vocab tokenizera    : {tok.vocab_size} tokenów")
    print(f"\n{chr(9552)*W}\n")


if __name__ == "__main__":
    run_demo()
    _demo_files()

    # Przykład użycia w pipeline danych (jak w oryginalnym skrypcie)
    print("\n── Przykład batch pipeline (styl JSON dataset) ──")
    import json

    sample_items = [
        {"utt_id": "utt_001", "text": "Cena produktu to 29,99 zł – promocja do 31.12.2024 r."},
        {"utt_id": "utt_002", "text": "Temperatura: -5°C, opady 한국 śniegu, widoczność 2\u20135 km."},
        {"utt_id": "utt_003", "text": "„Zysk\u201d wyniósł 2,5 mld zł (wzrost o 22%)."},
        {"utt_id": "utt_004", "text": "Pociąg o 08:15, bagaż do 23 kg, cena £180."},
    ]

    pipe = PolishTTSPipeline()
    out_items = []
    for item in sample_items:
        item_out = dict(item)
        item_out["text_normalized"] = pipe.process(item["text"])
        out_items.append(item_out)

    for it in out_items:
        print(f"\n  [{it['utt_id']}]")
        print(f"  IN:  {it['text']}")
        print(f"  OUT: {it['text_normalized']}")


# ═══════════════════════════════════════════════════════════════════════════════
# CLI Entry Point
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    pipe = PolishTTSPipeline()
    if len(sys.argv) > 1:
        text = " ".join(sys.argv[1:])
    else:
        text = input("Enter Polish text: ")
    print(pipe.process(text))
