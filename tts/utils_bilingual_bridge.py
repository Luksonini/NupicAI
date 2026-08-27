#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
utils_bilingual_bridge.py

Compat shim for Węgorz TTS forks that need a mixed PL-orthography + EN-IPA
token space without rewriting the original training code.
"""

from __future__ import annotations

import json
import re
import zlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset

from semantic_bridge import SemanticBridgeState
from benchmark_config import BenchmarkConfig


SAMPLE_RATE = 24000
HOP_LENGTH = 256
N_MELS = 100
FRAMES_PER_SEC = SAMPLE_RATE / HOP_LENGTH

DEFAULT_VOCAB = str(Path(__file__).resolve().parent / "vocab_pl_orth_en_ipa_bridge.json")


class PLTokenizer:
    """
    Backward-compatible name expected by Węgorz code.
    Internally this is a bridge tokenizer over mixed PL orthography + EN IPA vocab.
    """

    def __init__(self, vocab_json: str = ""):
        vocab_path = str(vocab_json or DEFAULT_VOCAB)
        try:
            from text_normalyzer.tokenize_and_normalize_new import PLTokenizer as BasePLTokenizer
        except Exception:
            try:
                from wegorz_normalizer.tokenize_and_text_norm import PLTokenizer as BasePLTokenizer
            except Exception:
                BasePLTokenizer = None  # type: ignore[assignment]
        self._base = BasePLTokenizer(vocab_path=vocab_path, boundary_mode="sp") if BasePLTokenizer is not None else None
        if self._base is not None:
            self.token2id = dict(self._base.token2id)
            self.id2token = dict(self._base.id2token)
            self.sp_id = int(self._base.sp_id)
            self.unk_id = int(self._base.unk_id)
            self.PUNCT = list(getattr(self._base, "PUNCT", list(".,!?:;-\"'()[]…/")))
            self._punct_set = set(self.PUNCT)
            return

        vocab = json.loads(Path(vocab_path).expanduser().read_text(encoding="utf-8"))
        self.token2id = {str(k): int(v) for k, v in dict(vocab).items()}
        self.id2token = {int(v): str(k) for k, v in self.token2id.items()}
        self.sp_id = int(self.token2id.get("<sp>", 2))
        self.unk_id = int(self.token2id.get("<unk>", 1))
        self.PUNCT = list(".,!?:;-\"'()[]…/“”„,")
        self._punct_set = set(self.PUNCT)

    def encode(self, text: str) -> List[int]:
        if self._base is not None:
            def _lower_first_alpha(seg: str) -> str:
                chars = list(seg)
                for idx, ch in enumerate(chars):
                    if ch.isalpha():
                        chars[idx] = ch.lower()
                        break
                return "".join(chars)

            s = str(text or "")
            s = re.sub(r"(?i)<\s*(?:bos|eos)\s*>", "<sp>", s)
            s = re.sub(r"(?i)<\s*sp\s*>", "<sp>", s)

            toks: List[str] = []
            prev_tok: Optional[str] = None
            for m in re.finditer(r"<[^>]+>|[^<]+", s):
                chunk = m.group(0)
                if not chunk:
                    continue
                if chunk.startswith("<") and chunk.endswith(">") and chunk in self.token2id:
                    toks.append(chunk)
                    prev_tok = chunk
                    continue
                if prev_tok == "<CAP>":
                    chunk = _lower_first_alpha(chunk)
                toks.extend(self._base._pretokenize(chunk))
                if toks:
                    prev_tok = toks[-1]

            ids = [int(self.token2id.get(t, self.unk_id)) for t in toks]
            return list(self._base._apply_boundary_mode(ids))
        s = str(text or "")
        ids: List[int] = []
        i = 0
        while i < len(s):
            ch = s[i]
            if ch.isspace():
                ids.append(self.sp_id)
                i += 1
                while i < len(s) and s[i].isspace():
                    i += 1
                continue
            if ch == "<":
                j = s.find(">", i + 1)
                if j > i:
                    tok = s[i : j + 1]
                    if tok in self.token2id:
                        ids.append(int(self.token2id[tok]))
                        i = j + 1
                        continue
            ids.append(int(self.token2id.get(ch, self.unk_id)))
            i += 1
        return ids


_TOK = PLTokenizer()
SYMBOL2ID = _TOK.token2id
ID2SYMBOL = _TOK.id2token
PAD_ID = SYMBOL2ID["<pad>"]
_SP_ID = SYMBOL2ID["<sp>"]

PROSODY_TOKENS = list(getattr(_TOK, "PUNCT", list(".,!?:;-\"'()[]…/")))
PROSODY_IDS = {SYMBOL2ID[t] for t in PROSODY_TOKENS if t in SYMBOL2ID}
ROLE_TOKENS = [
    "<nar>",
    "<akt>",
    "<CAP>",
    "<cap>",
    "<BOS>",
    "<EOS>",
    "<reserved3>",
    "<reserved4>",
    "<reserved5>",
    "<pl>",
    "<en>",
]
ROLE_IDS = {SYMBOL2ID[t] for t in ROLE_TOKENS if t in SYMBOL2ID}


def encode_text(text: str, add_eos: bool = False, add_continue: bool = False) -> List[int]:
    return _TOK.encode(text)


def safe_log1p(x: torch.Tensor) -> torch.Tensor:
    return torch.log1p(torch.clamp(x, min=0.0))


def book_ids_to_tensor(book_ids: object) -> torch.Tensor:
    if isinstance(book_ids, torch.Tensor):
        return book_ids.detach().cpu().long()
    if not isinstance(book_ids, (list, tuple)):
        book_ids = [book_ids]
    out: List[int] = []
    for b in book_ids:
        if isinstance(b, int):
            out.append(int(b))
            continue
        s = str(b)
        try:
            out.append(int(s))
        except Exception:
            out.append(int(zlib.crc32(s.encode("utf-8"))) & 0x7FFFFFFF)
    return torch.tensor(out, dtype=torch.long)


class BridgeStateCache:
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
        bridge,
        speaker_ids: torch.Tensor,
        book_ids: torch.Tensor,
        chunk_idx: torch.Tensor,
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
                st = SemanticBridgeState(
                    short=st_cpu.short.to(device=device, dtype=dtype),
                    long=st_cpu.long.to(device=device, dtype=dtype),
                )

            short_list.append(st.short)
            long_list.append(st.long)

        short = torch.cat(short_list, dim=0)
        long = torch.cat(long_list, dim=0)
        return SemanticBridgeState(short=short, long=long), keys, chunks

    def set_batch_state(
        self,
        *,
        keys: List[Tuple[int, int]],
        chunks: List[int],
        state_after: SemanticBridgeState,
    ) -> None:
        for i, (key, ci) in enumerate(zip(keys, chunks)):
            st_cpu = SemanticBridgeState(
                short=state_after.short[i:i+1].detach().to("cpu", dtype=torch.float32),
                long=state_after.long[i:i+1].detach().to("cpu", dtype=torch.float32),
            )
            self._state[key] = st_cpu
            self._last_chunk[key] = int(ci)


class AlignedTTSDataset(Dataset):
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
            raise ValueError("dataset.json should contain a list of records.")
        self.auto_add_default_role = auto_add_default_role
        self.default_role_token = default_role_token
        self.dur_source = str(dur_source)

    def __len__(self):
        return len(self.items)

    def _features_path_for_audio(self, audio_path: Path) -> Path:
        p = Path(audio_path)
        parent = p.parent
        if parent.name != "segments":
            try:
                idx = p.parts.index("segments")
                base = Path(*p.parts[:idx])
                stem = Path(*p.parts[idx + 1 :]).with_suffix(".pt").name
                return base / "features" / stem
            except ValueError:
                return parent.with_name("features") / p.with_suffix(".pt").name
        return parent.with_name("features") / p.with_suffix(".pt").name

    def _split_ids_by_words(self, ids: List[int]) -> List[List[int]]:
        words_ids: List[List[int]] = []
        cur: List[int] = []
        for tid in ids:
            if tid == _SP_ID or tid in ROLE_IDS or tid in PROSODY_IDS:
                if cur:
                    words_ids.append(cur)
                    cur = []
            else:
                cur.append(tid)
        if cur:
            words_ids.append(cur)
        return words_ids

    def _compute_word_frame_spans(self, words: Optional[List[Dict[str, Any]]], T: int) -> List[Tuple[int, int]]:
        if not words or T <= 0:
            return []

        starts: List[float] = []
        ends: List[float] = []
        for w in words:
            s = max(0.0, float(w.get("start", 0.0)))
            e = max(s, float(w.get("end", 0.0)))
            starts.append(s)
            ends.append(e)

        word_frames = [max(0, int(round(max(0.0, e - s) * FRAMES_PER_SEC))) for s, e in zip(starts, ends)]
        gap_frames = []
        for i in range(len(words) - 1):
            gap_s = max(0.0, starts[i + 1] - ends[i])
            gap_frames.append(max(0, int(round(gap_s * FRAMES_PER_SEC))))

        total_all = sum(word_frames) + sum(gap_frames)
        if total_all <= 0:
            W = len(words)
            base = T // W
            rest = T - base * W
            word_frames = [base + (1 if i < rest else 0) for i in range(W)]
            gap_frames = [0] * (W - 1)
        elif total_all != T:
            scale = T / float(total_all)
            word_frames = [max(0, int(round(f * scale))) for f in word_frames]
            gap_frames = [max(0, int(round(g * scale))) for g in gap_frames]
            curr = sum(word_frames) + sum(gap_frames)
            diff = T - curr
            if diff != 0:
                all_frames = word_frames + gap_frames
                step = 1 if diff > 0 else -1
                for i in range(abs(diff)):
                    j = i % len(all_frames)
                    if all_frames[j] + step >= 0:
                        all_frames[j] += step
                word_frames = all_frames[: len(word_frames)]
                gap_frames = all_frames[len(word_frames) :]

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

    def _compute_token_durations(self, text_ids: List[int], words: Optional[List[Dict[str, Any]]], T: int) -> List[int]:
        no_dur_ids = set(ROLE_IDS) | set(PROSODY_IDS)
        keep_mask = [tid not in no_dur_ids for tid in text_ids]
        filtered_ids = [tid for tid, keep in zip(text_ids, keep_mask) if keep]
        num_keep = len(filtered_ids)
        if num_keep == 0:
            return [0] * len(text_ids)

        if not words:
            base = T // num_keep
            rest = T - base * num_keep
            token_durations = [base + (1 if i < rest else 0) for i in range(num_keep)]
        else:
            word_spans = self._compute_word_frame_spans(words, T)
            word_lengths = [max(1, end - start) for (start, end) in word_spans]
            gap_frames = []
            if len(word_spans) > 1:
                for i in range(len(word_spans) - 1):
                    gap_frames.append(max(0, word_spans[i + 1][0] - word_spans[i][1]))
            word_slices = self._split_ids_by_words(filtered_ids)
            token_durations = []
            if not word_slices:
                base = T // num_keep
                rest = T - base * num_keep
                token_durations = [base + (1 if i < rest else 0) for i in range(num_keep)]
            else:
                for wi, word_ids in enumerate(word_slices):
                    tok_count = max(1, len(word_ids))
                    wf = word_lengths[wi] if wi < len(word_lengths) else word_lengths[-1]
                    base = wf // tok_count
                    rest = wf - base * tok_count
                    for j in range(tok_count):
                        token_durations.append(base + (1 if j < rest else 0))
                    if wi < len(word_slices) - 1:
                        sp_frames = gap_frames[min(wi, len(gap_frames) - 1)] if gap_frames else 0
                        token_durations.append(max(1, sp_frames))

            if len(token_durations) < num_keep:
                token_durations += [0] * (num_keep - len(token_durations))
            elif len(token_durations) > num_keep:
                token_durations = token_durations[:num_keep]

        full_durations: List[int] = []
        it = iter(token_durations)
        for keep in keep_mask:
            full_durations.append(next(it) if keep else 0)

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
        tokens_val = item.get("tokens", None)
        text = item.get("text_with_roles") or item.get("text", "")
        if self.auto_add_default_role and not isinstance(tokens_val, (list, tuple)) and not any(tok in str(text) for tok in ROLE_TOKENS):
            text = f"{self.default_role_token} {text}".strip()
        words = item.get("words", None)

        book_id = item.get("book_id", "unknown")
        author_id = item.get("author", "unknown")
        speaker_id = int(item.get("speaker_id", 0))
        chunk_idx = int(item.get("chunk_idx", item.get("chunk_index", idx)))

        if audio_val:
            feat_path = self._features_path_for_audio(Path(audio_val))
        elif audio_pt:
            feat_path = Path(audio_pt)
        else:
            raise RuntimeError("Missing audio/audio_pt in dataset item.")

        if not feat_path.exists():
            raise FileNotFoundError(f"Missing features PT: {feat_path}")

        d = torch.load(feat_path, map_location="cpu")
        mel = d["mel"]
        mel = torch.as_tensor(mel, dtype=torch.float32).clone().detach()
        mel = torch.nan_to_num(mel, nan=0.0, posinf=0.0, neginf=0.0).to(dtype=torch.float32)

        spk_emb_chunk = None
        if isinstance(d, dict):
            spk_emb_chunk = d.get("spk_emb_chunk", d.get("speaker_emb_chunk", None))
        if spk_emb_chunk is not None:
            spk_emb_chunk = torch.as_tensor(spk_emb_chunk, dtype=torch.float32).view(-1).clone().detach()

        if mel.ndim == 3 and mel.shape[0] == 1:
            mel = mel.squeeze(0)
        T = int(mel.shape[1])

        if isinstance(tokens_val, (list, tuple)) and tokens_val:
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
            elif dur_source == "ctc":
                utt = str(item.get("utt_id", item.get("id", idx)))
                raise RuntimeError(
                    f"CTC durations required but missing/mismatched for item {utt}: "
                    f"len(text_ids)={len(text_ids)} dur_tok_frames="
                    f"{0 if ctc_dur is None else len(ctc_dur)}"
                )
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


def _pad_1d_list(tensors: List[torch.Tensor], pad_value: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
    lengths = torch.tensor([t.numel() for t in tensors], dtype=torch.long)
    N = int(lengths.max()) if len(lengths) else 0
    B = len(tensors)
    out = tensors[0].new_full((B, N), fill_value=pad_value)
    for i, t in enumerate(tensors):
        out[i, : t.numel()] = t
    return out, lengths


def _pad_mel_list(mels: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    lengths = torch.tensor([m.shape[1] for m in mels], dtype=torch.long)
    Tm = int(lengths.max()) if len(lengths) else 0
    B = len(mels)
    out = mels[0].new_zeros((B, N_MELS, Tm))
    for i, m in enumerate(mels):
        out[i, :, : m.shape[1]] = m
    return out, lengths


def collate_fn(batch: List[Dict[str, Any]]):
    mel_list = [b["mel"] for b in batch]
    ids_list = [b["text_ids"] for b in batch]
    Lgt_list = [b["L_gt"] for b in batch]

    prosody_list = [
        torch.tensor([1 if int(i) in PROSODY_IDS else 0 for i in ids], dtype=torch.float32)
        for ids in ids_list
    ]

    mel_pad, T_len = _pad_mel_list(mel_list)
    tok_pad, N_tok = _pad_1d_list(ids_list, pad_value=PAD_ID)
    L_gt_pad, _ = _pad_1d_list(Lgt_list, pad_value=0)
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
    "AlignedTTSDataset",
    "BenchmarkConfig",
    "BridgeStateCache",
    "ID2SYMBOL",
    "N_MELS",
    "PAD_ID",
    "PLTokenizer",
    "PROSODY_IDS",
    "PROSODY_TOKENS",
    "ROLE_IDS",
    "ROLE_TOKENS",
    "SYMBOL2ID",
    "book_ids_to_tensor",
    "collate_fn",
    "encode_text",
    "safe_log1p",
]


'''
wnloads/SalmonTTS2 \
> /home/rizos/Miniforge3/envs/uvtts2/bin/python \
> /home/rizos/Downloads/SalmonTTS2/test_paraqueet/wegorz_ablations/WęgorzTTS3_paperctc_online_durab_bilingual_pseudot.py \
>   --resume /home/rizos/Downloads/SalmonTTS2/test_paraqueet/wegorz_ablations/run_wegorz_bilingual_onlinectc_1k_continue/chkpts/prior_mu_flow_spkprefix_dualhead_last.pt \
>   --dataset-json /home/rizos/Downloads/SalmonTTS2/test_paraqueet/wegorz_ablations/run_wegorz_bilingual_onlinectc_1k_continue/manifest_text_lang_prefixed_pl_orth_en_ipa__stateful_mixed_full.json \
>   --vocab /home/rizos/Downloads/SalmonTTS2/test_paraqueet/interlanguage_bridge/data_pl_orth_en_ipa_bridge/vocab_pl_orth_en_ipa_bridge.json \
>   --asr-ckpt /home/rizos/Downloads/SalmonTTS2/test_paraqueet/runs/asr_nano_4layers_split80_160_max60min/ckpt_best.pt \
>   --max-items 1000 \
>   --epochs 600 \
>   --batch-size 7 \
>   --workers 2 \
>   --mu-noise-sigma 0.10 \
>   --mu-noise-prob 1.0 \
>   --out-dir /home/rizos/Downloads/SalmonTTS2/test_paraqueet/wegorz_ablations/run_wegorz_bilingual_onlinectc_pseudot_mu_noise10_1k
📦 Output: /home/rizos/Downloads/SalmonTTS2/test_paraqueet/wegorz_ablations/run_wegorz_bilingual_onlinectc_pseudot_mu_noise10_1k
🧩 prefix-continuity: enabled (train+infer) prefix_ms=30.0 frames=3 (applied only when chunk_idx is consecutive for same speaker_id+book_id)
🎯 speaker-conditioning: prefix-token-only (speaker_loss_w=0.1000, speaker_teacher_w=0.0000, speaker_vector_source=gt_dualhead, decoder/prior/dur speaker projections disabled)
🔇 pause-loss-shaping: disabled
🔁 mel two-pass: enabled=True steps_first=8 steps_second=3 t_noise=0.120
⏱️ flow t-sampling: mode=logit_normal mu=0.000 sigma=1.000
🧭 online CTC durations: enabled (impl=WegorzASRNanoV2 align=x2->mel trainable=True w=0.100 asr_ckpt=/home/rizos/Downloads/SalmonTTS2/test_paraqueet/runs/asr_nano_4layers_split80_160_max60min/ckpt_best.pt)
🗂️ dataset dur source: auto (online_ctc=on)
🔪 train/val split: last 4 chunks per (speaker,book) from last part -> train_items=940 val_items=60 speakers=15 books=15
✅ Loaded frozen online ASR V2: /home/rizos/Downloads/SalmonTTS2/test_paraqueet/runs/asr_nano_4layers_split80_160_max60min/ckpt_best.pt (skipped=2 deep_branch=new x2->x4->x2 fusion ctc_rows_reused=84)
✅ Loaded frozen spk_style dualhead: /home/rizos/Downloads/SalmonTTS2/Speakder_enkoder/_runs/spk_style_dualhead_v2/last.pt
[dur] MC two-pass enabled (n_samples=5 std_thr=0.25 t_noise=0.4 steps=1). Active when --dur-x0 prior.
ℹ️ base_model partial load: skipped=40 (including duration-head mismatches)
⚠️ Nie udało się wczytać optim: loaded state dict contains a parameter group that doesn't match the size of optimizer's group
✅ Resumed from: /home/rizos/Downloads/SalmonTTS2/test_paraqueet/wegorz_ablations/run_wegorz_bilingual_onlinectc_1k_continue/chkpts/prior_mu_flow_spkprefix_dualhead_last.pt (start_epoch=392)
/home/rizos/Downloads/SalmonTTS2/CTC aligner3/CTC_aligner_v3.py:896: UserWarning: torchaudio.functional._alignment.forced_align has been deprecated. This deprecation is part of a large refactoring effort to transition TorchAudio into a maintenance phase. Please see https://github.com/pytorch/audio/issues/3902 for more information. It will be removed from the 2.9 release. 
  paths, _ = TAF.forced_align(
[ep0392] loss=2.5017 | mu=2.0605 flow=0.4226 ctc=0.1728 spk=0.2118 dur=-0.0284 | prior=2.0605 | 54.8s
[ep0392] pseudo_t noisy=true:0.438 pred:0.374 | mu_pred:0.302 | l1 noisy_gt=0.9245 mu=2.7187
[ep0392] demo_ref_l1=1.5170 demo_mu_l1=1.6426 (batches=10)
/home/rizos/Downloads/SalmonTTS2/CTC aligner3/CTC_aligner_v3.py:896: UserWarning: torchaudio.functional._alignment.forced_align has been deprecated. This deprecation is part of a large refactoring effort to transition TorchAudio into a maintenance phase. Please see https://github.com/pytorch/audio/issues/3902 for more information. It will be removed from the 2.9 release. 
  paths, _ = TAF.forced_align(
[ep0393] loss=2.3863 | mu=1.8234 flow=0.3729 ctc=0.1883 spk=0.1530 dur=0.0995 | prior=1.8234 | 42.5s
[ep0393] pseudo_t noisy=true:0.452 pred:0.415 | mu_pred:0.343 | l1 noisy_gt=0.9871 mu=2.7810
[ep0393] demo_ref_l1=1.2551 demo_mu_l1=1.6376 (batches=10)
[ep0394] loss=2.3030 | mu=1.8050 flow=0.3301 ctc=0.1834 spk=0.1721 dur=0.0410 | prior=1.8050 | 43.4s
[ep0394] pseudo_t noisy=true:0.453 pred:0.421 | mu_pred:0.313 | l1 noisy_gt=0.9866 mu=2.7836
[ep0394] demo_ref_l1=1.2073 demo_mu_l1=1.6307 (batches=10)
[ep0395] loss=2.2659 | mu=1.8653 flow=0.3587 ctc=0.1107 spk=0.2404 dur=-0.0041 | prior=1.8653 | 44.2s
[ep0395] pseudo_t noisy=true:0.446 pred:0.425 | mu_pred:0.294 | l1 noisy_gt=0.9716 mu=2.7349
[ep0395] demo_ref_l1=1.2097 demo_mu_l1=1.6373 (batches=10)
[ep0396] loss=2.2259 | mu=1.8142 flow=0.2573 ctc=0.0242 spk=0.1053 dur=0.2015 | prior=1.8142 | 44.3s
[ep0396] pseudo_t noisy=true:0.452 pred:0.433 | mu_pred:0.241 | l1 noisy_gt=1.0084 mu=2.7534
[ep0396] demo_ref_l1=1.2090 demo_mu_l1=1.6127 (batches=10)
[ep0397] loss=2.2081 | mu=1.8292 flow=0.9192 ctc=0.0832 spk=0.0863 dur=-0.0551 | prior=1.8292 | 43.1s
[ep0397] pseudo_t noisy=true:0.450 pred:0.443 | mu_pred:0.213 | l1 noisy_gt=0.9820 mu=2.7185
[ep0397] demo_ref_l1=1.2279 demo_mu_l1=1.6201 (batches=10)
[ep0398] loss=2.1682 | mu=1.7851 flow=0.3129 ctc=0.0112 spk=0.2145 dur=-0.0215 | prior=1.7851 | 47.4s
[ep0398] pseudo_t noisy=true:0.451 pred:0.445 | mu_pred:0.187 | l1 noisy_gt=0.9810 mu=2.6882
[ep0398] demo_ref_l1=1.2342 demo_mu_l1=1.6212 (batches=10)
[ep0399] loss=2.1544 | mu=1.7783 flow=0.7079 ctc=0.0303 spk=0.0971 dur=0.1018 | prior=1.7783 | 44.7s
[ep0399] pseudo_t noisy=true:0.446 pred:0.440 | mu_pred:0.172 | l1 noisy_gt=0.9532 mu=2.7077
[ep0399] demo_ref_l1=1.1619 demo_mu_l1=1.6116 (batches=10)
[ep0400] loss=2.1348 | mu=1.8359 flow=0.2081 ctc=0.0231 spk=0.0857 dur=0.0982 | prior=1.8359 | 43.4s
[ep0400] pseudo_t noisy=true:0.444 pred:0.439 | mu_pred:0.159 | l1 noisy_gt=0.9652 mu=2.6387
[ep0400] demo_ref_l1=1.1701 demo_mu_l1=1.6210 (batches=10)
[ep0401] loss=2.1222 | mu=1.7881 flow=0.6297 ctc=0.0270 spk=0.0676 dur=0.0290 | prior=1.7881 | 43.9s
[ep0401] pseudo_t noisy=true:0.448 pred:0.445 | mu_pred:0.169 | l1 noisy_gt=0.9708 mu=2.6652
[ep0401] demo_ref_l1=1.2196 demo_mu_l1=1.6065 (batches=10)
[ep0402] loss=2.0739 | mu=1.8028 flow=0.2255 ctc=0.0152 spk=0.0795 dur=-0.0260 | prior=1.8028 | 49.7s
[ep0402] pseudo_t noisy=true:0.455 pred:0.451 | mu_pred:0.174 | l1 noisy_gt=0.9982 mu=2.6778
[ep0402] demo_ref_l1=1.3296 demo_mu_l1=1.6171 (batches=10)


PYTHONPATH=/home/rizos/Downloads/SalmonTTS2 \
/home/rizos/Miniforge3/envs/uvtts2/bin/python \
/home/rizos/Downloads/SalmonTTS2/test_paraqueet/wegorz_ablations/WęgorzTTS3_paperctc_online_durab_bilingual_pseudot_residual.py \
  --resume /home/rizos/Downloads/SalmonTTS2/test_paraqueet/wegorz_ablations/run_wegorz_bilingual_onlinectc_pseudot_mu_noise10_durmse_1k/chkpts/prior_mu_flow_spkprefix_dualhead_last.pt \
  --dataset-json /home/rizos/Downloads/SalmonTTS2/test_paraqueet/wegorz_ablations/run_wegorz_bilingual_onlinectc_1k_continue/manifest_text_lang_prefixed_pl_orth_en_ipa__stateful_mixed_full.json \
  --vocab /home/rizos/Downloads/SalmonTTS2/test_paraqueet/interlanguage_bridge/data_pl_orth_en_ipa_bridge/vocab_pl_orth_en_ipa_bridge.json \
  --asr-ckpt /home/rizos/Downloads/SalmonTTS2/test_paraqueet/runs/asr_nano_4layers_split80_160_max60min/ckpt_best.pt \
  --max-items 10000 \
  --epochs 600 \
  --batch-size 7 \
  --workers 2 \
  --mu-noise-sigma 0.10 \
  --mu-noise-prob 1.0 \
  --dur-prior-loss-mode mse \
  --dur-sigma0-train-min 0.03 \
  --dur-sigma0-train-max 0.03 \
  --demo-dur-steps 1 \
  --mel-flow-output-mode residual \
  --out-dir /home/rizos/Downloads/SalmonTTS2/test_paraqueet/wegorz_ablations/run_wegorz_bilingual_onlinectc_pseudot_residual_1k

'''
