"""
GraphemeTokenizer — char-level tokenizer kompatybilny z pipeline TTS.

EN input : char-by-char + <CAP> przed wielką literą + <sp> dla spacji
PL output: digraph-aware (sz,cz,ch,rz,dz,dź,dż → single token) + <CAP> + <sp>

Vocab ~70 tokenów (vs BPE 32k). Output PL jest bezpośrednio kompatybilny
z vocab_pl_orth_en_ipa_bridge.json używanym przez TTS text encoder.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import List

# Digrafy PL — kolejność ważna: dłuższe/specyficzne PRZED krótszymi
PL_DIGRAPHS = ["dź", "dż", "dz", "sz", "cz", "ch", "rz"]
PL_DIGRAPH_TOKENS = {d: f"<{d}>" for d in PL_DIGRAPHS}

# Interpunkcja wspólna EN+PL
PUNCT = list(".,!?:;-\"'()[]…/—""")

# Znaki specjalne EN (nie są digramami PL)
EN_EXTRA = list("@#$%&*+=<>^_|~")


def _build_vocab() -> dict[str, int]:
    vocab: dict[str, int] = {}

    def add(tok):
        if tok not in vocab:
            vocab[tok] = len(vocab)

    # Specjalne (muszą być pierwsze żeby pad=0, bos=1, eos=2, unk=3)
    for s in ["<pad>", "<bos>", "<eos>", "<unk>", "<sp>", "<CAP>", "<en>", "<pl>"]:
        add(s)

    # Digrafy PL
    for d in PL_DIGRAPHS:
        add(f"<{d}>")

    # Litery a-z (EN + PL base)
    for c in "abcdefghijklmnopqrstuvwxyz":
        add(c)

    # PL diacritics
    for c in "ąćęłńóśźż":
        add(c)

    # Interpunkcja
    for c in PUNCT:
        add(c)

    # EN extras (rzadkie, ale żeby nie były <unk>)
    for c in EN_EXTRA:
        add(c)

    # Cyfry
    for c in "0123456789":
        add(c)

    return vocab


class GraphemeTokenizer:
    """
    Interface kompatybilny z sentencepiece.SentencePieceProcessor:
      encode(text) -> List[int]
      decode(ids)  -> str
      get_piece_size() -> int

    Dodatkowe metody:
      encode_en(text) -> List[int]
      encode_pl(text) -> List[int]
    """

    PAD_ID = 0
    BOS_ID = 1
    EOS_ID = 2
    UNK_ID = 3

    def __init__(self, vocab: dict[str, int] | None = None):
        self.tok2id: dict[str, int] = vocab if vocab is not None else _build_vocab()
        self.id2tok: dict[int, str] = {v: k for k, v in self.tok2id.items()}

    # ── Serialization ─────────────────────────────────────────────────────────

    def save(self, path: str | Path):
        Path(path).write_text(json.dumps(self.tok2id, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "GraphemeTokenizer":
        vocab = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(vocab)

    @classmethod
    def build_and_save(cls, path: str | Path) -> "GraphemeTokenizer":
        tok = cls()
        tok.save(path)
        return tok

    # ── Core ──────────────────────────────────────────────────────────────────

    def get_piece_size(self) -> int:
        return len(self.tok2id)

    def _tok(self, s: str) -> int:
        return self.tok2id.get(s, self.UNK_ID)

    def encode_pl(self, text: str) -> List[int]:
        """Tokenizuje polski tekst z detekcją digramów i <CAP>."""
        ids: List[int] = []
        i = 0
        text = text.strip()
        while i < len(text):
            c = text[i]
            if c == " ":
                ids.append(self._tok("<sp>"))
                i += 1
                continue
            # Spróbuj digramów (2 znaki)
            digraph_found = False
            if i + 1 < len(text):
                pair = text[i:i+2].lower()
                if pair in PL_DIGRAPH_TOKENS:
                    if c.isupper():
                        ids.append(self._tok("<CAP>"))
                    ids.append(self._tok(PL_DIGRAPH_TOKENS[pair]))
                    i += 2
                    digraph_found = True
            if not digraph_found:
                if c.isupper():
                    ids.append(self._tok("<CAP>"))
                    ids.append(self._tok(c.lower()))
                else:
                    ids.append(self._tok(c))
                i += 1
        return ids

    def encode_en(self, text: str) -> List[int]:
        """Tokenizuje angielski tekst char-by-char (bez digrafów PL)."""
        ids: List[int] = []
        text = text.strip()
        for c in text:
            if c == " ":
                ids.append(self._tok("<sp>"))
            elif c.isupper():
                ids.append(self._tok("<CAP>"))
                ids.append(self._tok(c.lower()))
            else:
                ids.append(self._tok(c))
        return ids

    def encode(self, text: str, lang: str = "pl") -> List[int]:
        """Domyślnie koduje jako PL (dla targetu). lang='en' dla źródła."""
        if lang == "en":
            return self.encode_en(text)
        return self.encode_pl(text)

    def decode(self, ids: List[int]) -> str:
        """Rekonstruuje tekst z tokenów."""
        parts: List[str] = []
        cap_next = False
        for i in ids:
            tok = self.id2tok.get(i, "")
            if tok in ("<pad>", "<bos>", "<eos>", "<en>", "<pl>", "<unk>"):
                continue
            if tok == "<CAP>":
                cap_next = True
                continue
            if tok == "<sp>":
                parts.append(" ")
                cap_next = False
                continue
            # Digraph token np. <sz> → "sz"
            if tok.startswith("<") and tok.endswith(">"):
                ch = tok[1:-1]
            else:
                ch = tok
            if cap_next:
                ch = ch[0].upper() + ch[1:]
                cap_next = False
            parts.append(ch)
        return "".join(parts)


# ── CAP encoding helpers ──────────────────────────────────────────────────────

def cap_encode(text: str) -> str:
    """Konwertuje wielkie litery na <CAP> + mała litera.

    'Szkoła w Warszawie' → '<CAP>szkoła w <CAP>warszawie'
    Wynik jest spójny z formatem TTS (token <CAP> poprzedza literę).
    """
    out: List[str] = []
    for ch in text:
        if ch.isupper():
            out.append("<CAP>")
            out.append(ch.lower())
        else:
            out.append(ch)
    return "".join(out)


def cap_decode(text: str) -> str:
    """Odwraca cap_encode: '<CAP>szkoła' → 'Szkoła'.

    Bezpieczne na tekście który nie zawiera <CAP> — zwraca go bez zmian.
    """
    import re as _re
    decoded = _re.sub(r"<CAP>(.)", lambda m: m.group(1).upper(), text)
    return decoded.replace("<CAP>", "")  # usuń ewentualne osierocone <CAP> na końcu


# ── Corpus tokenization helper ────────────────────────────────────────────────

def tokenize_corpus(
    src_path: str,
    out_path: str,
    tokenizer: GraphemeTokenizer,
    max_lines: int = 0,
):
    """
    Przepisuje train.jsonl tokenizując EN+PL do list int-ów.
    Wynikowy plik ma pola 'en_ids' i 'pl_ids'.
    """
    import json as _json

    n = 0
    with open(src_path, encoding="utf-8") as fin, \
         open(out_path, "w", encoding="utf-8") as fout:
        for line in fin:
            if max_lines and n >= max_lines:
                break
            obj = _json.loads(line)
            en = obj.get("en", "").strip()
            pl = obj.get("pl", "").strip()
            if not en or not pl:
                continue
            en_ids = tokenizer.encode_en(en)
            pl_ids = tokenizer.encode_pl(pl)
            if len(en_ids) > 512 or len(pl_ids) > 512:
                continue
            fout.write(_json.dumps({"en_ids": en_ids, "pl_ids": pl_ids}) + "\n")
            n += 1
    print(f"Zapisano {n:,} par → {out_path}")
    return n


if __name__ == "__main__":
    tok = GraphemeTokenizer()
    print(f"Vocab size: {tok.get_piece_size()}")
    print("Tokens:", list(tok.tok2id.keys()))

    tests_pl = [
        "To jest przykładowe zdanie po polsku.",
        "Szczególnie trudne słowa: dżdżownica, chrząszcz, źródło.",
        "Rzeka płynie przez Szczecin.",
    ]
    tests_en = [
        "Hello, this is a test sentence.",
        "The algorithm computes the shortest path.",
    ]
    print("\nTest PL:")
    for t in tests_pl:
        ids = tok.encode_pl(t)
        dec = tok.decode(ids)
        print(f"  IN : {t}")
        print(f"  IDs: {ids[:15]}... ({len(ids)} tok)")
        print(f"  OUT: {dec}")
        print()
    print("Test EN:")
    for t in tests_en:
        ids = tok.encode_en(t)
        dec = tok.decode(ids)
        print(f"  IN : {t}")
        print(f"  IDs: {ids[:15]}... ({len(ids)} tok)")
        print(f"  OUT: {dec}")
        print()
