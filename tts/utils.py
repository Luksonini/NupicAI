#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
utils.py — dataset + collate + drobne bloki TTS
Wersja z PLTokenizer (tokeny: litery PL, digrafy <sz>/<cz>/..., separator słów = <sp>).
Wyczyszczona z logiki Word-Level CLIP (word_spans).
"""

import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
import zlib

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

try:
    from wegorz_normalizer.tokenize_and_text_norm import PLTokenizer
except Exception:
    try:
        from wegorz_normalizer.wegorz_normalizer_single import PLTokenizer
    except Exception:
        from text_normalyzer.tokenize_and_normalize_new import PLTokenizer
from semantic_bridge import SemanticBridgeState
from benchmark_config import BenchmarkConfig

# ==============================
# Konfiguracja stałych
# ==============================

SAMPLE_RATE = 24000
HOP_LENGTH  = 256
N_MELS      = 100
FRAMES_PER_SEC = SAMPLE_RATE / HOP_LENGTH  # ~93.75

# ==============================
# Tokenizer (GLOBAL)
# ==============================

_TOK = PLTokenizer()                       # wczyta/utworzy słownik
SYMBOL2ID = _TOK.token2id                  # map[str->int]
ID2SYMBOL = _TOK.id2token                  # map[int->str]
PAD_ID    = SYMBOL2ID["<pad>"]
_SP_ID    = SYMBOL2ID["<sp>"]              # separator słów

# Tokeny prozodyczne/interpunkcyjne – widoczne w encoderze, maskowane w duracjach
PROSODY_TOKENS = list(getattr(_TOK, 'PUNCT', list(".,!?:;-\"'()[]…/")))
PROSODY_IDS = {SYMBOL2ID[t] for t in PROSODY_TOKENS if t in SYMBOL2ID}

# Tokeny ról (narrator/aktor) – używane w tekście, ale bez duration
ROLE_TOKENS = ["<nar>", "<akt>", "<CAP>", "<cap>", "<BOS>", "<EOS>", "<reserved3>", "<reserved4>", "<reserved5>"]
ROLE_IDS = {SYMBOL2ID[t] for t in ROLE_TOKENS if t in SYMBOL2ID}

# API helper: kodowanie tekstu
def encode_text(text: str, add_eos: bool = False, add_continue: bool = False) -> List[int]:
    """Koduje tekst na ID. Parametry add_eos/add_continue zostawione dla kompatybilności (ignorowane)."""
    return _TOK.encode(text)
def safe_log1p(x: torch.Tensor) -> torch.Tensor:
    return torch.log1p(torch.clamp(x, min=0.0))


def book_ids_to_tensor(book_ids: object) -> torch.Tensor:
    """
    `collate_fn` zwraca `book_ids` jako listę wartości (często str/int).
    Część kodu (np. cache mostów) potrzebuje stabilnych intów, więc:
      - jeśli da się zrzutować do int -> użyj int,
      - inaczej -> deterministyczny hash (crc32).
    Zwraca CPU tensor [B] dtype long.
    """
    if isinstance(book_ids, torch.Tensor):
        return book_ids.detach().cpu().long()
    if not isinstance(book_ids, (list, tuple)):
        book_ids = [book_ids]
    out: List[int] = []
    for b in book_ids:
        if isinstance(b, (int,)):
            out.append(int(b))
            continue
        s = str(b)
        try:
            out.append(int(s))
        except Exception:
            out.append(int(zlib.crc32(s.encode("utf-8"))) & 0x7FFFFFFF)
    return torch.tensor(out, dtype=torch.long)


class BridgeStateCache:
    """
    Przechowuje SemanticBridgeState między chunkami, per (speaker_id, book_id).

    Zasady resetu:
      - chunk_idx == 0  => reset stanu
      - albo brak ciągłości (last_chunk+1 != chunk_idx) => reset
    Przechowywanie stanu na CPU (float32), żeby nie pompować VRAM.
    """
    def __init__(self):
        self._state: Dict[Tuple[int, int], SemanticBridgeState] = {}
        self._last_chunk: Dict[Tuple[int, int], int] = {}

    def _reset_needed(self, key: Tuple[int, int], chunk_i: int) -> bool:
        if chunk_i == 0:
            return True
        last = self._last_chunk.get(key, None)
        if last is None:
            return True
        return (last + 1) != chunk_i

    def get_batch_state(
        self,
        *,
        bridge: nn.Module,
        speaker_ids: torch.Tensor,  # CPU tensor [B]
        book_ids: torch.Tensor,     # CPU tensor [B]
        chunk_idx: torch.Tensor,    # CPU tensor [B]
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[SemanticBridgeState, List[Tuple[int, int]], List[int]]:
        keys: List[Tuple[int, int]] = []
        chunks: List[int] = []
        short_list: List[torch.Tensor] = []
        long_list: List[torch.Tensor] = []

        sids = speaker_ids.detach().cpu().tolist()
        bids = book_ids.detach().cpu().tolist()
        cidx = chunk_idx.detach().cpu().tolist()

        for sid, bid, ci in zip(sids, bids, cidx):
            key = (int(sid), int(bid))
            ci = int(ci)
            keys.append(key)
            chunks.append(ci)

            if self._reset_needed(key, ci) or (key not in self._state):
                st = bridge.init_state(batch_size=1, device=device, dtype=dtype)
            else:
                st_cpu = self._state[key]
                # przenieś na device/dtype
                st = SemanticBridgeState(
                    short=st_cpu.short.to(device=device, dtype=dtype),
                    long=st_cpu.long.to(device=device, dtype=dtype),
                )

            short_list.append(st.short)
            long_list.append(st.long)

        short = torch.cat(short_list, dim=0)  # [B, D_mem]
        long = torch.cat(long_list, dim=0)
        return SemanticBridgeState(short=short, long=long), keys, chunks

    def set_batch_state(
        self,
        *,
        keys: List[Tuple[int, int]],
        chunks: List[int],
        state_after: SemanticBridgeState,
    ) -> None:
        # Zapis per-sample na CPU
        for i, (key, ci) in enumerate(zip(keys, chunks)):
            st_cpu = SemanticBridgeState(
                short=state_after.short[i:i+1].detach().to("cpu", dtype=torch.float32),
                long=state_after.long[i:i+1].detach().to("cpu", dtype=torch.float32),
            )
            self._state[key] = st_cpu
            self._last_chunk[key] = int(ci)

# ==============================
# Dataset
# ==============================

class AlignedTTSDataset(Dataset):
    """
    Czyta dataset.json.
    Oblicza duracje tokenów na podstawie wyrównania słów (jeśli dostępne w 'words').
    """
    def __init__(
        self,
        json_path: str,
        max_items: Optional[int] = None,
        auto_add_default_role: bool = False,
        default_role_token: str = "<nar>",
        dur_source: str = "auto",
    ):
        super().__init__()
        self.manifest_path = Path(json_path)
        with open(self.manifest_path, "r", encoding="utf-8") as f:
            self.items: List[Dict[str, Any]] = json.load(f)
        if max_items is not None:
            self.items = self.items[:max_items]
        if not isinstance(self.items, list):
            raise ValueError("dataset.json powinien zawierać listę rekordów (list).")
        self.auto_add_default_role = auto_add_default_role
        self.default_role_token = default_role_token
        self.dur_source = str(dur_source)

    def __len__(self): 
        return len(self.items)
    
    def _features_path_for_audio(self, audio_path: Path) -> Path:
        # /.../segments/<stem>.wav -> /.../features/<stem>.pt
        p = Path(audio_path)
        parent = p.parent
        if parent.name != "segments":
            try:
                idx = p.parts.index("segments")
                base = Path(*p.parts[:idx])
                stem = Path(*p.parts[idx+1:]).with_suffix(".pt").name
                return base / "features" / stem
            except ValueError:
                return parent.with_name("features") / p.with_suffix(".pt").name
        return parent.with_name("features") / p.with_suffix(".pt").name

    def _split_ids_by_words(self, ids: List[int]) -> List[List[int]]:
        """Zwraca listę słów jako listy tokenów (bez <sp> i bez tokenów ról)."""
        words_ids: List[List[int]] = []
        cur: List[int] = []
        for tid in ids:
            if tid == _SP_ID or tid in ROLE_IDS or tid in PROSODY_IDS:
                if len(cur) > 0:
                    words_ids.append(cur); cur = []
            else:
                cur.append(tid)
        if len(cur) > 0:
            words_ids.append(cur)
        return words_ids

    def _compute_word_frame_spans(self, words: Optional[List[Dict]], T: int) -> List[Tuple[int, int]]:
        """
        Pomocnicza funkcja do mapowania słów (WhisperX) na ramki audio.
        Potrzebna do wyliczenia Ground Truth Duration (L_gt).
        """
        if not words or len(words) == 0 or T <= 0:
            return []

        # 1) czasy słów
        starts: List[float] = []
        ends: List[float] = []
        for w in words:
            s = max(0.0, float(w.get("start", 0.0)))
            e = max(s, float(w.get("end", 0.0)))
            starts.append(s)
            ends.append(e)

        word_frames: List[int] = []
        for s, e in zip(starts, ends):
            dur_s = max(0.0, e - s)
            word_frames.append(max(0, int(round(dur_s * FRAMES_PER_SEC))))

        # pauzy między słowami
        gap_frames: List[int] = []
        for i in range(len(words) - 1):
            gap_s = max(0.0, starts[i + 1] - ends[i])
            gap_frames.append(max(0, int(round(gap_s * FRAMES_PER_SEC))))

        total_fw = sum(word_frames)
        total_fg = sum(gap_frames)
        total_all = total_fw + total_fg

        if total_all <= 0:
            # fallback: równy podział
            W = len(words)
            base = T // W
            rest = T - base * W
            word_frames = [base + (1 if i < rest else 0) for i in range(W)]
            gap_frames = [0] * (W - 1)
        else:
            if total_all != T:
                scale = T / float(total_all)
                word_frames = [max(0, int(round(f * scale))) for f in word_frames]
                gap_frames  = [max(0, int(round(g * scale))) for g in gap_frames]
                curr = sum(word_frames) + sum(gap_frames)
                diff = T - curr
                if diff != 0:
                    all_frames = word_frames + gap_frames
                    if len(all_frames) > 0:
                        step = 1 if diff > 0 else -1
                        for i in range(abs(diff)):
                            j = i % len(all_frames)
                            if all_frames[j] + step >= 0:
                                all_frames[j] += step
                        word_frames = all_frames[:len(word_frames)]
                        gap_frames  = all_frames[len(word_frames):]

        # 2) spany słów po ramkach (start, end)
        spans: List[Tuple[int, int]] = []
        cursor = 0
        for wi, wf in enumerate(word_frames):
            start = cursor
            end = min(T, start + wf)
            if end <= start:
                end = min(T, start + 1)
            spans.append((start, end))
            cursor = end
            if wi < len(gap_frames):
                cursor = min(T, cursor + gap_frames[wi])

        return spans

    def _compute_token_durations(self, text_ids: List[int], words: Optional[List[Dict]], T: int) -> List[int]:
        """
        Zwraca listę długości w ramkach dla KAŻDEGO tokenu (L_gt).
        Rozdziela czas słów na tokeny literowe, a czas pauz na tokeny <sp>.
        """
        NO_DUR_IDS = set(ROLE_IDS) | set(PROSODY_IDS)
        keep_mask = [tid not in NO_DUR_IDS for tid in text_ids]
        filtered_ids = [tid for tid, keep in zip(text_ids, keep_mask) if keep]
        num_keep = len(filtered_ids)

        if num_keep == 0:
            return [0] * len(text_ids)

        if not words or len(words) == 0:
            # Fallback: uniform
            base = T // num_keep
            rest = T - base * num_keep
            token_durations = [base + (1 if i < rest else 0) for i in range(num_keep)]
        else:
            word_frames_spans = self._compute_word_frame_spans(words, T)
            # Długość trwania samego słowa (bez pauzy po nim)
            word_lengths = [max(1, end - start) for (start, end) in word_frames_spans]
            
            # Pauzy po słowach
            gap_frames = []
            if len(word_frames_spans) > 1:
                for i in range(len(word_frames_spans) - 1):
                    gap = max(0, word_frames_spans[i+1][0] - word_frames_spans[i][1])
                    gap_frames.append(gap)

            word_slices = self._split_ids_by_words(filtered_ids)
            token_durations: List[int] = []

            if len(word_slices) == 0:
                base = T // num_keep
                rest = T - base * num_keep
                token_durations = [base + (1 if i < rest else 0) for i in range(num_keep)]
            else:
                for wi, word_ids in enumerate(word_slices):
                    tok_count = max(1, len(word_ids))
                    wf = word_lengths[wi] if wi < len(word_lengths) else word_lengths[-1]
                    
                    # Dziel czas słowa na tokeny
                    base = wf // tok_count
                    rest = wf - base * tok_count
                    for j in range(tok_count):
                        token_durations.append(base + (1 if j < rest else 0))

                    # Daj czas pauzy dla <sp>
                    if wi < len(word_slices) - 1:
                        sp_frames = gap_frames[min(wi, len(gap_frames) - 1)] if gap_frames else 0
                        sp_frames = max(1, sp_frames)
                        token_durations.append(sp_frames)

            # Wyrównanie długości listy (padding <sp> na końcu itp.)
            if len(token_durations) < num_keep:
                token_durations += [0] * (num_keep - len(token_durations))
            elif len(token_durations) > num_keep:
                token_durations = token_durations[:num_keep]

        # Odtwórz pełną listę z zerami dla ROLE_TOKENS
        full_durations: List[int] = []
        it = iter(token_durations)
        for keep in keep_mask:
            if keep:
                full_durations.append(next(it))
            else:
                full_durations.append(0)

        # Ostateczna korekta sumy do T
        diff_total = T - sum(d for d, keep in zip(full_durations, keep_mask) if keep)
        if any(keep_mask) and diff_total != 0:
            step = 1 if diff_total > 0 else -1
            adjusted = 0
            i = 0
            max_iter = len(full_durations) * 4 + abs(diff_total)
            while adjusted < abs(diff_total) and i < max_iter:
                idx = i % len(full_durations)
                i += 1
                if not keep_mask[idx]:
                    continue
                if full_durations[idx] + step >= 0:
                    full_durations[idx] += step
                    adjusted += 1

        return full_durations

    def __getitem__(self, idx: int):
        item = self.items[idx]
        audio_val = item.get("audio")
        audio_pt = item.get("audio_pt")
        # Prefer precomputed token ids from the manifest when available (e.g. dataset_ctc_aligned2.json),
        # to keep exact alignment with `dur_tok_frames`.
        tokens_val = item.get("tokens", None)
        text = item.get("text_with_roles") or item.get("text", "")
        if self.auto_add_default_role and isinstance(tokens_val, (list, tuple)) is False and not any(tok in str(text) for tok in ROLE_TOKENS):
            text = f"{self.default_role_token} {text}".strip()
        words = item.get("words", None)

        book_id    = item.get("book_id", "unknown")
        author_id  = item.get("author", "unknown")
        speaker_id = int(item.get("speaker_id", 0))
        chunk_idx  = int(item.get("chunk_idx", item.get("chunk_index", idx)))

        if audio_val:
            audio_path = Path(audio_val)
            feat_path = self._features_path_for_audio(audio_path)
        elif audio_pt:
            feat_path = Path(audio_pt)
        else:
            raise RuntimeError("Missing audio/audio_pt in dataset item.")
        
        if not feat_path.exists():
            raise FileNotFoundError(f"Brak features PT: {feat_path}")

        d = torch.load(feat_path, map_location="cpu")
        mel = d["mel"]
        if isinstance(mel, torch.Tensor):
            mel = mel.clone().detach()
        else:
            mel = torch.as_tensor(mel, dtype=torch.float32)
        mel = torch.nan_to_num(mel, nan=0.0, posinf=0.0, neginf=0.0).to(dtype=torch.float32)

        # Optional: per-chunk speaker embedding (e.g. SpeechBrain ECAPA) stored alongside mel.
        spk_emb_chunk = None
        if isinstance(d, dict):
            spk_emb_chunk = d.get("spk_emb_chunk", d.get("speaker_emb_chunk", None))
        if spk_emb_chunk is not None:
            spk_emb_chunk = torch.as_tensor(spk_emb_chunk, dtype=torch.float32).view(-1).clone().detach()

        if mel.ndim == 3 and mel.shape[0] == 1:
            mel = mel.squeeze(0)

        T = int(mel.shape[1])

        # Tekst i duracje
        text_ids: List[int]
        if isinstance(tokens_val, (list, tuple)) and len(tokens_val) > 0:
            try:
                text_ids = [int(t) for t in tokens_val]
            except Exception:
                text_ids = encode_text(str(text), add_eos=False)
        else:
            text_ids = encode_text(str(text), add_eos=False)
        dur_source = getattr(self, "dur_source", "auto")
        L_gt = None
        if dur_source in ("ctc", "auto"):
            ctc_dur = item.get("dur_tok_frames")
            if isinstance(ctc_dur, (list, tuple)) and len(ctc_dur) == len(text_ids):
                L_gt = list(ctc_dur)
        if L_gt is None:
            L_gt = self._compute_token_durations(text_ids, words, T)

        return {
            "mel": mel,
            "text_ids": torch.tensor(text_ids, dtype=torch.long),
            "L_gt": torch.tensor(L_gt, dtype=torch.long),
            "T_len": torch.tensor(T, dtype=torch.long),
            "book_id": book_id,
            "author_id": author_id,
            "speaker_id": speaker_id,
            "chunk_idx": chunk_idx,
            "spk_emb_chunk": spk_emb_chunk,
        }

# ==============================
# Collate
# ==============================

def _pad_1d_list(tensors: List[torch.Tensor], pad_value: int=0) -> Tuple[torch.Tensor, torch.Tensor]:
    lengths = torch.tensor([t.numel() for t in tensors], dtype=torch.long)
    N = int(lengths.max()) if len(lengths) else 0
    B = len(tensors)
    out = tensors[0].new_full((B, N), fill_value=pad_value)
    for i, t in enumerate(tensors):
        out[i, :t.numel()] = t
    return out, lengths

def _pad_mel_list(mels: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    lengths = torch.tensor([m.shape[1] for m in mels], dtype=torch.long)
    Tm = int(lengths.max()) if len(lengths) else 0
    B = len(mels)
    out = mels[0].new_zeros((B, N_MELS, Tm))
    for i, m in enumerate(mels):
        out[i, :, :m.shape[1]] = m
    return out, lengths

def collate_fn(batch: List[Dict]):
    """
    Zwraca tuple tensorów przygotowanych do batcha.
    Usunięto zwracanie list spanów słownych (word_*).
    """
    mel_list = [b["mel"] for b in batch]
    ids_list = [b["text_ids"] for b in batch]
    Lgt_list = [b["L_gt"] for b in batch]

    prosody_list = [
        torch.tensor(
            [1 if int(i) in PROSODY_IDS else 0 for i in ids],
            dtype=torch.float32,
        )
        for ids in ids_list
    ]

    mel_pad, T_len = _pad_mel_list(mel_list)
    tok_pad, N_tok = _pad_1d_list(ids_list, pad_value=PAD_ID)
    L_gt_pad, _    = _pad_1d_list(Lgt_list, pad_value=0)
    prosody_pad, _ = _pad_1d_list(prosody_list, pad_value=0)

    speaker_ids = torch.tensor([int(b.get("speaker_id", 0)) for b in batch], dtype=torch.long)
    chunk_idx = torch.tensor([int(b.get("chunk_idx", b.get("chunk_index", i))) for i, b in enumerate(batch)], dtype=torch.long)
    book_ids = [b.get("book_id", "") for b in batch]
    author_ids = [b.get("author_id", "") for b in batch]

    return (
        mel_pad,
        T_len,
        tok_pad,
        N_tok,
        L_gt_pad,
        prosody_pad,
        speaker_ids,
        chunk_idx,
        book_ids,
        author_ids,
    )

__all__ = [
    'AlignedTTSDataset','collate_fn','BenchmarkConfig',
    'safe_log1p','encode_text',
    'book_ids_to_tensor','BridgeStateCache',
    'SYMBOL2ID','ID2SYMBOL','N_MELS','PAD_ID',
    'PROSODY_TOKENS','PROSODY_IDS','ROLE_TOKENS','ROLE_IDS'
]
