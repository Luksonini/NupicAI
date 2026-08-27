#!/usr/bin/env python3
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch

try:
    from utils_bilingual_bridge import PROSODY_IDS, ROLE_IDS, SYMBOL2ID
except Exception:  # pragma: no cover - imported through training PYTHONPATH
    PROSODY_IDS = set()
    ROLE_IDS = set()
    SYMBOL2ID = {"<sp>": 2}


SP_ID = int(SYMBOL2ID.get("<sp>", 2))


@dataclass(frozen=True)
class _Piece:
    ids: torch.Tensor
    dur: torch.Tensor
    mel: torch.Tensor


class BoundaryJitterDataset(torch.utils.data.Dataset):
    """
    On-the-fly boundary jitter for stateful chunk training.

    The wrapper keeps the original item order and metadata, but for consecutive
    chunks from the same (speaker_id, book_id) it can deterministically move
    1..N words across the boundary. The same boundary decision is seen from both
    neighboring chunks, so the stateful sampler still observes a coherent
    sequence without writing augmented mels to disk.
    """

    def __init__(
        self,
        base: torch.utils.data.Dataset,
        *,
        prob: float = 0.25,
        max_words: int = 2,
        seed: int = 1234,
        epoch_vary: bool = True,
        min_frames: int = 24,
        max_frames: int = 0,
    ) -> None:
        self.base = base
        self.items = getattr(base, "items", [])
        self.prob = float(max(0.0, min(1.0, prob)))
        self.max_words = int(max(0, max_words))
        self.seed = int(seed)
        self.epoch_vary = bool(epoch_vary)
        self.min_frames = int(max(0, min_frames))
        self.max_frames = int(max(0, max_frames))
        self.epoch = 0
        self._index: Dict[Tuple[int, str, int], int] = {}
        for i, it in enumerate(self.items):
            if not isinstance(it, dict):
                continue
            self._index[self._key_from_item(it, i)] = int(i)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)
        if hasattr(self.base, "set_epoch"):
            try:
                self.base.set_epoch(epoch)  # type: ignore[attr-defined]
            except Exception:
                pass

    def __len__(self) -> int:
        return len(self.base)

    @staticmethod
    def _sid(it: Dict[str, Any]) -> int:
        try:
            return int(it.get("speaker_id", it.get("speaker", 0)))
        except Exception:
            return 0

    @staticmethod
    def _book(it: Dict[str, Any]) -> str:
        return str(it.get("book_id", ""))

    @staticmethod
    def _chunk(it: Dict[str, Any], fallback: int) -> int:
        try:
            return int(it.get("chunk_idx", it.get("chunk_index", fallback)))
        except Exception:
            return int(fallback)

    def _key_from_item(self, it: Dict[str, Any], fallback: int) -> Tuple[int, str, int]:
        return (self._sid(it), self._book(it), self._chunk(it, fallback))

    def _neighbor_index(self, idx: int, delta: int) -> Optional[int]:
        if idx < 0 or idx >= len(self.items):
            return None
        it = self.items[idx]
        if not isinstance(it, dict):
            return None
        sid, book, chunk = self._key_from_item(it, idx)
        return self._index.get((sid, book, int(chunk) + int(delta)))

    def _boundary_shift(self, idx_left: Optional[int]) -> int:
        if idx_left is None or self.prob <= 0.0 or self.max_words <= 0:
            return 0
        idx_right = self._neighbor_index(int(idx_left), +1)
        if idx_right is None:
            return 0
        it = self.items[int(idx_left)]
        sid, book, chunk = self._key_from_item(it, int(idx_left))
        epoch_key = int(self.epoch) if self.epoch_vary else 0
        key = f"{self.seed}|{epoch_key}|{sid}|{book}|{chunk}".encode("utf-8", errors="ignore")
        h = hashlib.blake2b(key, digest_size=16).digest()
        r = int.from_bytes(h[:8], "little") / float(2**64)
        if r >= self.prob:
            return 0
        raw = int.from_bytes(h[8:12], "little")
        n_words = 1 + (raw % int(self.max_words))
        sign = 1 if (int.from_bytes(h[12:13], "little") & 1) == 0 else -1
        return int(sign * n_words)

    @staticmethod
    def _leading_end(ids: torch.Tensor) -> int:
        n = int(ids.numel())
        i = 0
        while i < n and int(ids[i]) in ROLE_IDS:
            i += 1
        return i

    @staticmethod
    def _trailing_start(ids: torch.Tensor) -> int:
        i = int(ids.numel())
        while i > 0 and (int(ids[i - 1]) in ROLE_IDS or int(ids[i - 1]) == SP_ID):
            i -= 1
        return i

    @staticmethod
    def _word_spans(ids: torch.Tensor) -> list[Tuple[int, int]]:
        spans: list[Tuple[int, int]] = []
        n = int(ids.numel())
        i = 0
        separators = set(ROLE_IDS) | {SP_ID}
        while i < n:
            while i < n and int(ids[i]) in separators:
                i += 1
            if i >= n:
                break
            start = i
            has_content = False
            while i < n and int(ids[i]) not in separators:
                tid = int(ids[i])
                if tid not in PROSODY_IDS:
                    has_content = True
                i += 1
            end = i
            if has_content and end > start:
                spans.append((int(start), int(end)))
        return spans

    @staticmethod
    def _frame_bounds(dur: torch.Tensor, tok_start: int, tok_end: int) -> Tuple[int, int]:
        tok_start = int(max(0, min(tok_start, int(dur.numel()))))
        tok_end = int(max(tok_start, min(tok_end, int(dur.numel()))))
        if tok_start <= 0:
            f0 = 0
        else:
            f0 = int(torch.clamp(dur[:tok_start], min=0).sum().item())
        f1 = int(torch.clamp(dur[:tok_end], min=0).sum().item())
        return f0, max(f0, f1)

    def _piece(self, sample: Dict[str, Any], tok_start: int, tok_end: int) -> _Piece:
        ids = torch.as_tensor(sample["text_ids"], dtype=torch.long)
        dur = torch.as_tensor(sample["L_gt"], dtype=torch.long)
        mel = torch.as_tensor(sample["mel"], dtype=torch.float32)
        tok_start = int(max(0, min(tok_start, int(ids.numel()))))
        tok_end = int(max(tok_start, min(tok_end, int(ids.numel()))))
        f0, f1 = self._frame_bounds(dur, tok_start, tok_end)
        f0 = int(max(0, min(f0, int(mel.size(-1)))))
        f1 = int(max(f0, min(f1, int(mel.size(-1)))))
        return _Piece(
            ids=ids[tok_start:tok_end].clone(),
            dur=dur[tok_start:tok_end].clone(),
            mel=mel[:, f0:f1].clone(),
        )

    @staticmethod
    def _prefix_range(sample: Dict[str, Any], n_words: int) -> Optional[Tuple[int, int]]:
        ids = torch.as_tensor(sample["text_ids"], dtype=torch.long)
        spans = BoundaryJitterDataset._word_spans(ids)
        if not spans:
            return None
        n = int(min(max(1, n_words), len(spans)))
        start = int(spans[0][0])
        end = int(spans[n - 1][1])
        if end < int(ids.numel()) and int(ids[end]) == SP_ID:
            end += 1
        return start, end

    @staticmethod
    def _suffix_range(sample: Dict[str, Any], n_words: int) -> Optional[Tuple[int, int]]:
        ids = torch.as_tensor(sample["text_ids"], dtype=torch.long)
        spans = BoundaryJitterDataset._word_spans(ids)
        if not spans:
            return None
        n = int(min(max(1, n_words), len(spans)))
        start = int(spans[-n][0])
        end = int(spans[-1][1])
        if start > 0 and int(ids[start - 1]) == SP_ID:
            start -= 1
        return start, end

    @staticmethod
    def _concat_pieces(pieces: list[_Piece]) -> _Piece:
        ids = torch.cat([p.ids for p in pieces if int(p.ids.numel()) > 0], dim=0)
        dur = torch.cat([p.dur for p in pieces if int(p.dur.numel()) > 0], dim=0)
        mel_parts = [p.mel for p in pieces if int(p.mel.size(-1)) > 0]
        if mel_parts:
            mel = torch.cat(mel_parts, dim=-1)
        else:
            ref = pieces[0].mel
            mel = ref[:, :0].clone()
        return _Piece(ids=ids, dur=dur, mel=mel)

    @staticmethod
    def _fix_duration_sum(dur: torch.Tensor, target_frames: int) -> torch.Tensor:
        dur = torch.clamp(torch.as_tensor(dur, dtype=torch.long).clone(), min=0)
        diff = int(target_frames) - int(dur.sum().item())
        if diff == 0 or int(dur.numel()) == 0:
            return dur
        positive = torch.nonzero(dur > 0, as_tuple=False).view(-1)
        if int(positive.numel()) > 0:
            idx = int(positive[torch.argmax(dur[positive])].item())
        else:
            idx = int(dur.numel() - 1)
        dur[idx] = max(0, int(dur[idx].item()) + int(diff))
        return dur

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        base_sample = self.base[idx]
        if self.prob <= 0.0 or self.max_words <= 0:
            return base_sample

        left_shift = self._boundary_shift(self._neighbor_index(idx, -1))
        right_shift = self._boundary_shift(idx)
        if left_shift == 0 and right_shift == 0:
            return base_sample

        prev_idx = self._neighbor_index(idx, -1)
        next_idx = self._neighbor_index(idx, +1)
        prev_sample = self.base[prev_idx] if prev_idx is not None and left_shift < 0 else None
        next_sample = self.base[next_idx] if next_idx is not None and right_shift > 0 else None

        ids = torch.as_tensor(base_sample["text_ids"], dtype=torch.long)
        lead_end = self._leading_end(ids)
        trail_start = self._trailing_start(ids)
        core_start = int(lead_end)
        core_end = int(trail_start)

        prepend: Optional[_Piece] = None
        append: Optional[_Piece] = None

        # Previous boundary: positive means previous chunk took current prefix.
        if left_shift > 0:
            r = self._prefix_range(base_sample, abs(left_shift))
            if r is not None:
                core_start = max(core_start, int(r[1]))
        elif left_shift < 0 and prev_sample is not None:
            r = self._suffix_range(prev_sample, abs(left_shift))
            if r is not None:
                prepend = self._piece(prev_sample, int(r[0]), int(r[1]))

        # Right boundary: positive means current chunk takes next prefix.
        if right_shift > 0 and next_sample is not None:
            r = self._prefix_range(next_sample, abs(right_shift))
            if r is not None:
                append = self._piece(next_sample, int(r[0]), int(r[1]))
        elif right_shift < 0:
            r = self._suffix_range(base_sample, abs(right_shift))
            if r is not None:
                core_end = min(core_end, int(r[0]))

        if core_start > core_end:
            return base_sample

        pieces = [
            self._piece(base_sample, 0, lead_end),
        ]
        if prepend is not None:
            pieces.append(prepend)
        pieces.append(self._piece(base_sample, core_start, core_end))
        if append is not None:
            pieces.append(append)
        pieces.append(self._piece(base_sample, trail_start, int(ids.numel())))

        merged = self._concat_pieces(pieces)
        T = int(merged.mel.size(-1))
        if T < self.min_frames:
            return base_sample
        if self.max_frames > 0 and T > self.max_frames:
            return base_sample

        out = dict(base_sample)
        out["text_ids"] = merged.ids.to(dtype=torch.long)
        out["L_gt"] = self._fix_duration_sum(merged.dur, T)
        out["mel"] = merged.mel.to(dtype=torch.float32)
        out["T_len"] = torch.tensor(T, dtype=torch.long)
        out["boundary_jitter"] = {
            "left_shift": int(left_shift),
            "right_shift": int(right_shift),
            "frames": int(T),
            "tokens": int(merged.ids.numel()),
        }
        return out
