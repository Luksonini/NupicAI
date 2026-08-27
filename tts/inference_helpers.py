#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared inference/demo helpers for WęgorzTTS training scripts."""
from __future__ import annotations

import glob
import wave
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import torch


def collect_reference_audio_paths(one: str = "", pattern: str = "") -> List[Path]:
    paths: List[Path] = []
    one = str(one or "").strip()
    if one:
        paths.append(Path(one).expanduser())
    pattern = str(pattern or "").strip()
    if pattern:
        paths.extend(Path(p).expanduser() for p in sorted(glob.glob(str(Path(pattern).expanduser()))))

    out: List[Path] = []
    seen = set()
    for p in paths:
        key = str(p)
        if key in seen:
            continue
        seen.add(key)
        if p.exists():
            out.append(p)
    return out


def load_ref_mel_pt(
    path: str,
    *,
    ensure_mel_bct: Callable[[torch.Tensor], torch.Tensor],
    device: torch.device | str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    p = Path(path).expanduser()
    obj = torch.load(str(p), map_location="cpu")
    if isinstance(obj, dict):
        mel = obj.get("mel", obj.get("mel_bct", obj.get("x", obj.get("features"))))
        t_val = obj.get("T_mel", obj.get("length", obj.get("T", None)))
    else:
        mel = obj
        t_val = None
    if not torch.is_tensor(mel):
        raise RuntimeError(f"Cannot find tensor mel in {p}")
    mel_bct = ensure_mel_bct(mel).to(device=device, dtype=torch.float32)
    if t_val is None:
        t_len = torch.tensor([int(mel_bct.size(-1))], device=device, dtype=torch.long)
    else:
        if torch.is_tensor(t_val):
            t_i = int(t_val.view(-1)[0].item())
        else:
            t_i = int(t_val)
        t_len = torch.tensor([max(1, min(t_i, int(mel_bct.size(-1))))], device=device, dtype=torch.long)
    return mel_bct[:, :, : int(t_len[0].item())].contiguous(), t_len


def load_wav_mono_24k(path: str, *, max_sec: float, start_sec: float = 0.0) -> torch.Tensor:
    p = Path(str(path)).expanduser()
    if not p.exists():
        raise RuntimeError(f"Reference audio not found: {p}")
    try:
        import torchaudio  # type: ignore
    except Exception:
        torchaudio = None
    try:
        import soundfile as sf  # type: ignore
    except Exception:
        sf = None

    wav = None
    sr = None
    if sf is not None:
        import numpy as np

        try:
            arr, sr = sf.read(str(p), dtype="float32", always_2d=False)
            if arr.ndim == 2:
                arr = arr.mean(axis=1)
            wav = torch.from_numpy(np.asarray(arr)).view(1, -1).to(torch.float32)
        except Exception:
            wav = None
            sr = None
    if wav is None and torchaudio is not None:
        try:
            wav, sr = torchaudio.load(str(p))
            if wav.dim() == 2 and int(wav.size(0)) > 1:
                wav = wav.mean(dim=0, keepdim=True)
            wav = wav.to(torch.float32)
        except Exception as exc:
            msg = str(exc).splitlines()[0] if str(exc).strip() else repr(exc)
            raise RuntimeError(f"Cannot load reference audio with soundfile or torchaudio: {p} ({msg})") from exc
    if wav is None or sr is None:
        raise RuntimeError("Cannot load reference audio: install torchaudio or soundfile.")
    if int(sr) != 24000:
        if torchaudio is None:
            raise RuntimeError(f"Need torchaudio to resample ({sr} -> 24000) for reference audio: {p}")
        wav = torchaudio.functional.resample(wav, int(sr), 24000)

    start_samp = max(0, int(float(start_sec) * 24000.0))
    if start_samp > 0 and start_samp < int(wav.size(-1)):
        wav = wav[..., start_samp:]
    if float(max_sec) > 0.0:
        max_samp = int(float(max_sec) * 24000.0)
        if int(wav.size(-1)) > max_samp:
            wav = wav[..., :max_samp]
    return wav.contiguous()


def wav_to_vocos_mel(
    path: str,
    *,
    vocos,
    ensure_mel_bct: Callable[[torch.Tensor], torch.Tensor],
    n_mels: int,
    device: torch.device | str,
    max_sec: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if vocos is None:
        raise RuntimeError("Vocos is required to compute mel from --demo-ref-wav.")
    fe = getattr(vocos, "feature_extractor", None)
    if fe is None:
        raise RuntimeError("Loaded Vocos has no feature_extractor; cannot compute mel from --demo-ref-wav.")
    wav_1t = load_wav_mono_24k(path, max_sec=float(max_sec))
    with torch.no_grad():
        mel = fe(wav_1t.to(device=device))
    if mel.dim() == 3 and int(mel.size(1)) != int(n_mels) and int(mel.size(2)) == int(n_mels):
        mel = mel.transpose(1, 2).contiguous()
    mel_bct = ensure_mel_bct(mel).to(device=device, dtype=torch.float32)
    t_len = torch.tensor([int(mel_bct.size(-1))], device=device, dtype=torch.long)
    return mel_bct.contiguous(), t_len


def save_wav(path: Path, wav_1t: torch.Tensor, sr: int = 24000) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    wav = wav_1t.detach().cpu()
    if wav.ndim == 1:
        wav = wav.view(1, -1)
    if wav.ndim != 2:
        raise RuntimeError(f"Bad wav shape for save: {tuple(wav.shape)}")
    if int(wav.size(0)) > 1:
        wav = wav.mean(dim=0, keepdim=True)

    last_exc = None
    try:
        import torchaudio  # type: ignore

        torchaudio.save(str(path), wav.clamp(-1, 1), int(sr))
        return
    except Exception as exc:
        last_exc = exc
    try:
        import soundfile as sf  # type: ignore

        sf.write(str(path), wav.squeeze(0).numpy(), int(sr))
        return
    except Exception as exc:
        last_exc = exc

    try:
        wav_i16 = (wav.squeeze(0).clamp(-1, 1).numpy() * 32767.0).astype("<i2")
        with wave.open(str(path), "wb") as f:
            f.setnchannels(1)
            f.setsampwidth(2)
            f.setframerate(int(sr))
            f.writeframes(wav_i16.tobytes())
        return
    except Exception as exc:
        raise RuntimeError(f"Cannot save wav (torchaudio/soundfile/wave failed): {exc}; previous={last_exc}")


def save_wav_or_tensor(dir_path: Path, tag: str, wav_1t: torch.Tensor, sr: int = 24000) -> None:
    try:
        save_wav(Path(dir_path) / f"{tag}.wav", wav_1t, sr)
    except Exception as exc:
        torch.save(wav_1t.detach().cpu(), str(Path(dir_path) / f"{tag}.pt"))
        try:
            (Path(dir_path) / f"{tag}.save_error.txt").write_text(str(exc) + "\n", encoding="utf-8")
        except Exception:
            pass


def decode_and_save_mel(
    *,
    dir_path: Path,
    tag: str,
    mel_1ct: torch.Tensor,
    decode_fn: Callable[[torch.Tensor, str], Optional[torch.Tensor]],
    silence_fn: Optional[Callable[[torch.Tensor, Optional[torch.Tensor]], torch.Tensor]] = None,
    silence_mask: Optional[torch.Tensor] = None,
    sr: int = 24000,
) -> None:
    wav_1t = decode_fn(mel_1ct, tag=f"text_demo/{tag}")
    if wav_1t is None:
        torch.save(mel_1ct.detach().cpu(), str(Path(dir_path) / f"{tag}.mel.pt"))
        return
    if silence_fn is not None:
        wav_1t = silence_fn(wav_1t, silence_mask)
    save_wav_or_tensor(Path(dir_path), tag, wav_1t, sr)


def decode_chunks_and_save(
    *,
    dir_path: Path,
    tag: str,
    mel_chunks_1ct: List[torch.Tensor],
    decode_fn: Callable[[torch.Tensor, str], Optional[torch.Tensor]],
    silence_fn: Optional[Callable[[torch.Tensor, Optional[torch.Tensor]], torch.Tensor]] = None,
    silence_masks: Optional[List[Optional[torch.Tensor]]] = None,
    sr: int = 24000,
) -> None:
    wavs: List[torch.Tensor] = []
    for idx, mel_1ct in enumerate(mel_chunks_1ct):
        if mel_1ct is None or int(mel_1ct.numel()) <= 0:
            continue
        wav_1t = decode_fn(mel_1ct, tag=f"text_demo/{tag}/chunk{idx:03d}")
        if wav_1t is None:
            valid = [m.detach().cpu() for m in mel_chunks_1ct if m is not None and int(m.numel()) > 0]
            if valid:
                torch.save(torch.cat(valid, dim=-1).contiguous(), str(Path(dir_path) / f"{tag}.mel.pt"))
            return
        if silence_fn is not None:
            sm = silence_masks[idx] if silence_masks is not None and idx < len(silence_masks) else None
            wav_1t = silence_fn(wav_1t, sm)
        wavs.append(wav_1t.detach().cpu())
    if wavs:
        save_wav_or_tensor(Path(dir_path), tag, torch.cat(wavs, dim=-1).contiguous(), sr)
