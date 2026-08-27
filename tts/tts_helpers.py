#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tts_helpers.py — wspólne moduły do treningu TTS + mostu pamięci

Zawiera:
- GenericTTS (alias na BackboneTTS z backbone_tts.py),
- loss'y melowe (L1 + delty + TV),
- EMA do uśredniania wag,
- helpery do demówek (audio, tekst),
- helper make_out_dir.

Uwaga:
- Dataset, tokenizer, BenchmarkConfig, N_MELS itd. są w utils.py.
"""

import datetime
import hashlib
import os
from pathlib import Path
from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda import amp

# Backbone TTS (tu używamy wariantu linear)
# from backbone_tts import BackboneTTS as GenericTTS
from backbone_tts import BackboneTTS as GenericTTS
from utils import (
    N_MELS,
    encode_text,
)

# ========================= Speaker Conditioning =========================


class SpeakerAdapter(nn.Module):
    def __init__(self, num_speakers: int, hidden_dim: int):
        super().__init__()
        self.table = nn.Embedding(int(num_speakers), int(hidden_dim))

    def forward(self, speaker_ids: torch.LongTensor, ref_emb: torch.Tensor | None = None) -> torch.Tensor:
        """
        Dziś: używamy tylko speaker_ids -> embedding z tabeli.
        Jutro: ref_emb z voice encodera można tu domieszać.
        """
        spk = self.table(speaker_ids)  # [B, D]
        if ref_emb is not None:
            spk = spk  # placeholder
        return spk


# ========================= Mel / TV / delty =========================


def pad_to_max_time(
    mel_post: torch.Tensor,  # [B,T_pred,D]
    mel_pad: torch.Tensor,   # [B,N_MELS,T_gt]
    T_len: torch.Tensor      # [B]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Wyrównanie mel_pred i mel_gt do wspólnego Tmax z maską czasową.

    mel_post: [B,T_pred,D]
    mel_pad : [B,N_MELS,T_gt]
    T_len   : [B] długości klipów (w klatkach)
    """
    B = mel_post.size(0)
    C = mel_post.size(-1)
    T_pred_max = mel_post.size(1)
    T_gt_max = mel_pad.size(2)
    T_len_max = int(T_len.max().item())
    Tmax = min(T_pred_max, T_gt_max, T_len_max)

    mel_fake = mel_post.new_zeros(B, Tmax, C)
    mel_real = mel_post.new_zeros(B, Tmax, C)
    mask = mel_post.new_zeros(B, 1, Tmax)

    for b in range(B):
        T_eff = min(int(T_len[b].item()), T_pred_max, T_gt_max)
        if T_eff > 0:
            mel_fake[b, :T_eff, :] = mel_post[b, :T_eff, :]
            mel_real[b, :T_eff, :] = mel_pad[b, :, :T_eff].transpose(0, 1)
            mask[b, 0, :T_eff] = 1.0
    return mel_fake, mel_real, mask


def masked_deltas_tv(
    mel_btc: torch.Tensor,  # [B,T,C]
    mask: torch.Tensor      # [B,1,T]
):
    """
    Delty Δ1, Δ2 i TV po czasie z maską.
    """
    B, T, C = mel_btc.shape
    if T <= 1:
        zero = mel_btc.sum() * 0.0
        return zero, zero, zero

    m = mask  # [B,1,T]

    # Δ1
    d1 = mel_btc[:, 1:, :] - mel_btc[:, :-1, :]         # [B,T-1,C]
    m1 = (m[:, :, 1:] * m[:, :, :-1]).unsqueeze(-1)     # [B,1,T-1,1]
    d1_abs = d1.abs().unsqueeze(1)                      # [B,1,T-1,C]
    d1_sum = (d1_abs * m1).sum()
    d1_den = m1.sum() * C
    d1_mean = d1_sum / d1_den.clamp_min(1.0)

    # Δ2
    if (T - 1) <= 1:
        d2_mean = d1_mean.new_tensor(0.0)
    else:
        d2 = d1[:, 1:, :] - d1[:, :-1, :]               # [B,T-2,C]
        m2 = (m1[:, :, 1:, :] * m1[:, :, :-1, :])       # [B,1,T-2,1]
        d2_abs = d2.abs().unsqueeze(1)                  # [B,1,T-2,C]
        d2_sum = (d2_abs * m2).sum()
        d2_den = m2.sum() * C
        d2_mean = d2_sum / d2_den.clamp_min(1.0)

    tv_mean = d1_mean
    return d1_mean, d2_mean, tv_mean


def masked_l1_mel(
    mel_pred: torch.Tensor,  # [B,T,D]
    mel_gt: torch.Tensor,    # [B,N_MELS,T_gt]
    T_len: torch.Tensor      # [B]
) -> torch.Tensor:
    """
    L1 po melach z maską czasu.
    """
    mel_fake, mel_real, mask = pad_to_max_time(mel_pred, mel_gt, T_len)
    diff = (mel_fake - mel_real).abs().unsqueeze(1)  # [B,1,Tmax,D]
    m = mask.unsqueeze(-1)                           # [B,1,Tmax,1]
    num = (diff * m).sum()
    den = (m.sum() * mel_fake.size(-1)).clamp_min(1.0)
    return num / den


# ========================= EMA =========================


class EMA:
    """
    Proste EMA po wagach (tylko floatowe tensory).
    Trzyma:
      - shadow: uśrednione wagi
      - _backup: kopię bieżących wag na czas apply()/restore()
      - model: referencję do modelu (dla wygody, np. ema.model)
    """
    def __init__(self, model: nn.Module, decay: float = 0.9995):
        self.decay = decay
        self.model = model  # <<< TO JEST KLUCZOWE, ŻEBY DZIAŁAŁO ema.model >>>

        # kopiujemy tylko tensory floatowe (wagi, biasy)
        self.shadow = {
            k: v.detach().clone()
            for k, v in model.state_dict().items()
            if v.dtype.is_floating_point
        }
        self._backup = None

    def update(self, model: nn.Module):
        """
        Aktualizacja EMA na podstawie bieżących wag modelu.
        """
        with torch.no_grad():
            for k, v in model.state_dict().items():
                if k in self.shadow and v.dtype.is_floating_point:
                    # shadow = decay * shadow + (1-decay) * new_value
                    self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1.0 - self.decay)

    def apply_to(self, model: nn.Module):
        """
        Tymczasowo nadpisuje wagi modelu uśrednionymi (shadow).
        Zapisuje bieżący stan do _backup, żeby można było zrobić restore().
        """
        self._backup = {
            k: v.detach().clone()
            for k, v in model.state_dict().items()
            if v.dtype.is_floating_point
        }
        # łączymy: bieżący state_dict + shadow (nadpisuje floatowe klucze)
        model.load_state_dict({**model.state_dict(), **self.shadow}, strict=False)

    def restore(self, model: nn.Module):
        """
        Przywraca wagi z _backup po wcześniejszym apply_to().
        """
        if self._backup is not None:
            model.load_state_dict({**model.state_dict(), **self._backup}, strict=False)
            self._backup = None

    def state_dict(self):
        """
        Zapisujemy tylko decay i shadow (EMA wag).
        Model trzymamy osobno w checkpointach jako 'model'.
        """
        return {
            "decay": self.decay,
            "shadow": {k: v.clone() for k, v in self.shadow.items()},
        }

    def load_state_dict(self, state: dict):
        """
        Odtworzenie EMA z checkpointu.
        Uwaga: model MUSI już być załadowany osobno (nn.Module).
        """
        self.decay = float(state.get("decay", self.decay))
        shadow = state.get("shadow", {})
        # nic nie kombinujemy z device – dostosuje się przy apply_to(model)
        self.shadow = {k: v.clone() for k, v in shadow.items()}




# ========================= Demówki =========================


def save_demo_audio(
    out_dir: Path,
    vocos,
    mel_hat_b: torch.Tensor,  # [B,100,T]
    mel_gt_b: torch.Tensor,   # [B,100,T]
    step_tag: str,
    sr: int = 24000,
):
    """
    Zapis pred/gt dla jednego batcha do WAV.
    """
    import torchaudio

    out_dir.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        wav_pred = vocos.decode(mel_hat_b)
        wav_gt = vocos.decode(mel_gt_b)
    torchaudio.save(str(out_dir / f"pred_{step_tag}.wav"), wav_pred.cpu(), sr)
    torchaudio.save(str(out_dir / f"gt_{step_tag}.wav"),   wav_gt.cpu(),   sr)


TEST_SENTENCES_PL = [
    "Dzisiejszego wieczoru napływają doniesienia, że amerykańscy i irańscy negocjatorzy osiągnęli porozumienie w sprawie przedłużenia zawieszenia broni oraz rozpoczęcia negocjacji dotyczących programu nuklearnego Iranu.",
    "Chrząszcz w trzcinie brzęczy, a w Szczebrzeszynie wszyscy słyszą to wyraźnie.",
    "Wczoraj późnym wieczorem ktoś cicho szepnął, że w domu obok dzieje się coś dziwnego.",
    "Przeczytaj spokojnie i wyraźnie, bez pośpiechu, tak jak w zwykłej rozmowie.",
    "Na biurku leżał notes, a w nim krótkie zdanie, które brzmiało jak ostrzeżenie.",
    "Zakończ zdanie miękko, bez urywania, i zostaw krótką pauzę po kropce.",
]

LONG_DEMO_CHUNKS_PL = [
    "Nocą w redakcji było cicho, tylko wentylator mruczał, a neon za oknem mrugał na niebiesko. ",
    "Marta przepisała notatki z dyktafonu i zatrzymała się przy jednym zdaniu, którego nie chciała wypowiedzieć głośno. ",
    "W korytarzu skrzypnęła podłoga, jakby ktoś stanął tuż za drzwiami i nasłuchiwał w bezruchu. ",
    "Telefon zadzwonił nagle, a po krótkim powitaniu zapadła cisza, która trwała niepokojąco długo. ",
    "W końcu usłyszała szept, spokojny i pewny, proszący, żeby nie otwierała, tylko sprawdziła zamek. ",
    "Marta wstała powoli, zrobiła kilka kroków i poczuła, że serce bije szybciej, choć starała się oddychać równo. ",
    "Gdy światło na moment przygasło, a potem wróciło, postanowiła zakończyć tę nocną pracę i wrócić do domu. ",
]

NEWS_DEMO_CHUNKS_PL = [
    "Dzisiejszego wieczoru napływają doniesienia, ",
    "że amerykańscy i irańscy negocjatorzy osiągnęli porozumienie ",
    "w sprawie przedłużenia zawieszenia broni oraz rozpoczęcia negocjacji dotyczących programu nuklearnego Iranu. ",
    "Donald Trump musi jednak wciąż zatwierdzić jakąkolwiek umowę ",
    "w obliczu wymiany ognia między obiema stronami, która zagraża obecnemu rozejmowi. ",
    "Irański Korpus Strażników Rewolucji Islamskiej twierdzi, ",
    "że obrał za cel amerykańską bazę lotniczą w tym regionie. ",
]

EXTRA_DEMO_SENTENCES_PL = [
    "– Co ty wyprawiasz?! – syknęła, ale zaraz się opanowała. ",
    "– Spokojnie. Najpierw posłuchaj. – odpowiedział cicho. ",
    "– Nie. Tym razem nie odpuszczę. ",
    "– Wiem. I właśnie dlatego tu jestem. ",
    "– Nie mów mi, że to przypadek. To nie był przypadek. ",
    "– Zaufaj mi. Jeszcze minutę. Potem zrobisz, co chcesz. ",
    "– Jeśli kłamiesz, przysięgam… ",
    "– Nie kłamię. Po prostu się boisz. ",
]

EXTRA_LONG_DEMO_CHUNKS_PL = [
    "– Słyszysz to? – zapytała nagle, jakby bała się własnego głosu. ",
    "– Nic nie słyszę. Tylko ciszę. ",
    "– Właśnie. To jest najgorsze. ",
    "– Dobra. Oddychaj. Mów powoli. Co się stało? ",
    "– Ktoś był pod drzwiami. Przed chwilą. ",
    "– Nie otwieraj. Sprawdź zamek i światło. ",
    "– Już. Zamek trzyma… ale ja mam złe przeczucie. ",
    "– To nie przeczucie. To pamięć. ",
]

EXTRA_EXTREME_LONG_DEMO_CHUNKS_PL = [
    "– Nie patrz na mnie w ten sposób. – jej głos zadrżał, ale nie pękł. ",
    "– W jaki sposób mam patrzeć, kiedy ty robisz to znowu? ",
    "– To nie jest znowu. Tym razem wiem, co robię. ",
    "– Nie wiesz. I właśnie dlatego mnie tu nie było. ",
    "– Kłamiesz. Byłeś obok. Zawsze byłeś obok. ",
    "– Jeśli byłem obok, to tylko po to, żebyś nie zrobiła głupstwa. ",
    "– A może ja już chcę to głupstwo. Może mam dość bycia grzeczną. ",
    "– Cicho. Słyszysz? ",
    "– Co…? ",
    "– Kroki. W korytarzu. Bardzo wolne. ",
    "– To sąsiedzi. ",
    "– Nie. To nie sąsiedzi. Sąsiedzi oddychają głośniej. ",
    "– Przestań… ",
    "– Zamknij oczy. Licz do trzech. I nie oddychaj. ",
    "– Raz… dwa… ",
    "– Trzy. Teraz słuchaj. ",
]

def save_text_demo(out_dir: Path, vocos, model: nn.Module, device: torch.device, epoch: int, sr: int = 24000):
    """
    Demówki na nowych polskich zdaniach (spoza datasetu) + jedna długa sekwencja.

    Część 1: krótkie, niezależne zdania (sanity-check generalizacji).
    Część 2: LONG_DEMO_CHUNKS_PL złożone w jeden długi WAV
             (spójny fragment narracji, inny tekst niż w treningu).
    """
    import torchaudio

    text_demo_dir = out_dir / "text_demo"
    text_demo_dir.mkdir(parents=True, exist_ok=True)

    model.eval()
    with torch.no_grad():
        # ------- 1) Krótkie, niezależne zdania -------
        for i, txt in enumerate(TEST_SENTENCES_PL):
            ids = torch.tensor(
                [encode_text(txt, add_eos=False)],
                dtype=torch.long,
                device=device,
            )
            # infer_pred: [B, T, N_MELS] (BTC)
            mel_hat = model.infer_pred(ids)                   # [1,T,100]
            mel_hat_b = mel_hat.transpose(1, 2).contiguous()  # [1,100,T]
            wav = vocos.decode(mel_hat_b)
            fname = text_demo_dir / f"ep{epoch:04d}_sent{i:02d}.wav"
            torchaudio.save(str(fname), wav.cpu(), sr)

        # ------- 2) Długi fragment (long-form) -------
        mel_chunks = []
        for j, txt in enumerate(LONG_DEMO_CHUNKS_PL):
            ids = torch.tensor(
                [encode_text(txt, add_eos=False)],
                dtype=torch.long,
                device=device,
            )
            mel_hat = model.infer_pred(ids)                   # [1,T_j,100]
            mel_chunks.append(mel_hat)

        if mel_chunks:
            # sklej po osi czasu: [1, T_total, 100]
            mel_long = torch.cat(mel_chunks, dim=1)
            mel_long_b = mel_long.transpose(1, 2).contiguous()   # [1,100,T_total]
            wav_long = vocos.decode(mel_long_b)
            fname_long = text_demo_dir / f"ep{epoch:04d}_long_story.wav"
            torchaudio.save(str(fname_long), wav_long.cpu(), sr)


# ========================= Katalog wyjściowy =========================


def make_out_dir(dataset_json: str, out_dir: str | None) -> Path:
    """
    Jeśli out_dir jest None → automatyczny _runs/tts_linear_TIMESTAMP.
    """
    if out_dir:
        return Path(out_dir)
    p = Path(dataset_json)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    auto = p.parent / "_runs" / f"tts_linear_{stamp}"
    return auto


__all__ = [
    "GenericTTS",
    "MelPatchDiscriminator",
    "pad_to_max_time",
    "masked_deltas_tv",
    "masked_l1_mel",
    "EMA",
    "hinge_d_loss_masked",
    "hinge_g_loss_masked",
    "d_step_only",
    "save_demo_audio",
    "save_text_demo",
    "make_out_dir",
    "TEST_SENTENCES_PL",
    "LONG_DEMO_CHUNKS_PL",
    "NEWS_DEMO_CHUNKS_PL",
    "EXTRA_DEMO_SENTENCES_PL",
    "EXTRA_LONG_DEMO_CHUNKS_PL",
    "EXTRA_EXTREME_LONG_DEMO_CHUNKS_PL",
    "SpeakerAdapter",
    "load_checkpoint",
    "maybe_load_vocos",
    "save_wav_fallback",
    "stable_seed",
    "randn_like_deterministic",
    "randn_like_deterministic_btc",
]


# ========================= Checkpointing / Vocos =========================

def load_checkpoint(
    *,
    path: str,
    device: torch.device,
    model: torch.nn.Module,
    bridge: Optional["SemanticBridge"],
    spk_embed: Optional["SpeakerAdapter"],
    tempo_predictor: torch.nn.Module,
    style_head: Optional[torch.nn.Module],
    style_to_token: Optional[torch.nn.Module] = None,
    opt: Optional[torch.optim.Optimizer] = None,
    scaler: Optional[amp.GradScaler] = None,
) -> Tuple[int, float]:
    """Wspólny loader checkpointów (przeniesiony z WegorzTTS3).

    - Ładuje częściowo, ignorując brakujące klucze.
    - Dla mismatch w 1. wymiarze (np. vocab / speaker table) robi merge: kopiuje wspólny prefix.
    - Opcjonalnie ładuje optimizer i czyści jego stan przy mismatchach.
    """
    # Importy tu, żeby uniknąć cykli importów przy ładowaniu plików.
    from semantic_bridge import SemanticBridge  # noqa: F401

    try:
        ckpt = torch.load(path, map_location=device)
    except Exception:
        ckpt = torch.load(path, map_location=device, weights_only=False)

    had_shape_mismatch = False

    def _load(module: Optional[torch.nn.Module], key: str, label: str) -> None:
        nonlocal had_shape_mismatch
        if module is None:
            return
        if key not in ckpt:
            print(f"⚠️ Pomijam {label}: brak klucza '{key}' w checkpoint")
            return
        try:
            incoming = ckpt[key]
            if isinstance(incoming, dict):
                current = module.state_dict()
                filtered = {}
                skipped = 0
                partially_loaded = 0
                for k, v in incoming.items():
                    if k not in current:
                        continue
                    cv = current[k]
                    if not (torch.is_tensor(v) and torch.is_tensor(cv)):
                        continue
                    if v.shape == cv.shape:
                        filtered[k] = v
                    # merge po pierwszym wymiarze (np. embedding table / vocab)
                    elif v.ndim >= 1 and cv.ndim == v.ndim and tuple(v.shape[1:]) == tuple(cv.shape[1:]):
                        merged = cv.clone()
                        n0 = min(int(v.shape[0]), int(cv.shape[0]))
                        merged[:n0].copy_(v[:n0])
                        filtered[k] = merged
                        partially_loaded += 1
                        had_shape_mismatch = True
                    else:
                        skipped += 1
                        had_shape_mismatch = True

                res = module.load_state_dict(filtered, strict=False)
                missing = getattr(res, "missing_keys", [])
                unexpected = getattr(res, "unexpected_keys", [])
                print(
                    f"✅ Załadowano {label} z '{key}' (missing={len(missing)}, unexpected={len(unexpected)}, "
                    f"partial={partially_loaded}, skipped={skipped})"
                )
            else:
                res = module.load_state_dict(incoming, strict=False)
                missing = getattr(res, "missing_keys", [])
                unexpected = getattr(res, "unexpected_keys", [])
                print(f"✅ Załadowano {label} z '{key}' (missing={len(missing)}, unexpected={len(unexpected)})")
        except Exception as exc:
            if "size mismatch" in str(exc):
                had_shape_mismatch = True
            print(f"⚠️ Nie udało się załadować {label} z '{key}': {type(exc).__name__}: {exc}")

    _load(model, "model", "model")
    _load(bridge, "bridge", "bridge")
    _load(spk_embed, "spk_embed", "spk_embed")
    _load(tempo_predictor, "tempo_predictor", "tempo_predictor")
    _load(style_head, "role_head", "style_head")
    _load(style_to_token, "style_to_token", "style_to_token")

    def _sanitize_optimizer_state(opt_in: torch.optim.Optimizer) -> int:
        cleared = 0
        to_del = []
        for group in opt_in.param_groups:
            for p in group.get("params", []):
                st = opt_in.state.get(p, None)
                if not isinstance(st, dict) or not st:
                    continue
                bad = False
                for _k, v in st.items():
                    if torch.is_tensor(v) and v.shape != p.shape:
                        bad = True
                        break
                if bad:
                    to_del.append(p)
        for p in to_del:
            try:
                del opt_in.state[p]
                cleared += 1
            except Exception:
                pass
        return cleared

    if opt is not None and "optim" in ckpt:
        try:
            opt.load_state_dict(ckpt["optim"])
            cleared = _sanitize_optimizer_state(opt)
            if cleared > 0:
                print(f"⚠️ Optimizer: zresetowano stan dla {cleared} parametrów (shape mismatch).")
        except Exception as exc:
            print(f"⚠️ Nie udało się wczytać optimizer: {type(exc).__name__}: {exc}")

    if scaler is not None and "scaler" in ckpt:
        try:
            if not had_shape_mismatch:
                scaler.load_state_dict(ckpt["scaler"])
        except Exception:
            pass

    best = float(ckpt.get("best", float("inf")))
    start_epoch = int(ckpt.get("epoch", 0)) + 1
    print(f"🔁 Resume z {path} (start_epoch={start_epoch}, best={best:.4f})")
    return start_epoch, best


def maybe_load_vocos(device: torch.device):
    """Opcjonalny Vocos do demówek (bez twardej zależności)."""
    try:
        from vocos import Vocos
        this_dir = Path(__file__).resolve().parent
        page_dir = this_dir.parent
        local_candidates = [
            page_dir / "models/tts/vocos-mel-24khz",
        ]
        for local_dir in local_candidates:
            try:
                cfg = local_dir / "config.yaml"
                bin_ckpt = local_dir / "pytorch_model.bin"
                safe_ckpt = local_dir / "model.safetensors"
                if cfg.exists() and (bin_ckpt.exists() or safe_ckpt.exists()):
                    model = Vocos.from_hparams(str(cfg))
                    if bin_ckpt.exists():
                        state = torch.load(str(bin_ckpt), map_location="cpu")
                        model.load_state_dict(state, strict=True)
                    else:
                        from safetensors.torch import load_file
                        state = load_file(str(safe_ckpt), device="cpu")
                        model.load_state_dict(state, strict=True)
                    return model.to(device).eval()
            except Exception:
                pass
        raise FileNotFoundError(f"Brak lokalnego Vocos w: {local_candidates[0]}")
    except Exception as exc:
        print(f"⚠️ Vocos niedostępny: {type(exc).__name__}: {exc}")
        return None


def save_wav_fallback(wav_path: Path, wav_1t: torch.Tensor, sr: int = 24000) -> None:
    """
    Minimalny zapis WAV bez torchaudio/soundfile.
    wav_1t: [1, T] lub [T] float w zakresie ~[-1,1].
    """
    import wave
    import struct

    x = wav_1t.detach().float().cpu()
    if x.dim() == 2:
        x = x.squeeze(0)
    x = x.clamp(-1.0, 1.0)
    x_i16 = (x * 32767.0).round().to(torch.int16)
    samples = x_i16.tolist()

    wav_path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(wav_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # int16
        wf.setframerate(int(sr))
        wf.writeframes(struct.pack("<" + "h" * len(samples), *samples))


def stable_seed(speaker_id: int, book_id: int, extra: str = "") -> int:
    """
    Stabilny seed (nie używa hash()).
    """
    s = f"{int(speaker_id)}|{int(book_id)}|{str(extra)}".encode("utf-8")
    h = hashlib.blake2b(s, digest_size=8).digest()
    return int.from_bytes(h, "little") % (2**31 - 1)


def randn_like_deterministic(x: torch.Tensor, *, speaker_id: int, book_id: int, extra: str = "") -> torch.Tensor:
    """
    Deterministyczny odpowiednik torch.randn_like, seedowany (speaker_id, book_id, extra).
    """
    seed = stable_seed(int(speaker_id), int(book_id), str(extra))
    try:
        g = torch.Generator(device=x.device)
        g.manual_seed(int(seed))
        return torch.randn(x.shape, device=x.device, dtype=x.dtype, generator=g)
    except Exception:
        g = torch.Generator()
        g.manual_seed(int(seed))
        eps = torch.randn(x.shape, device="cpu", dtype=torch.float32, generator=g)
        return eps.to(device=x.device, dtype=x.dtype)


def randn_like_deterministic_btc(
    x_btc: torch.Tensor,
    *,
    speaker_ids: torch.Tensor,
    book_ids: torch.Tensor,
    extras: Optional[list[str]] = None,
) -> torch.Tensor:
    """
    Deterministyczny eps dla batcha (BTC) per-sample.
    """
    B = int(x_btc.size(0))
    sids = speaker_ids.detach().cpu().tolist()
    bids = book_ids.detach().cpu().tolist()
    if extras is None:
        extras = [""] * B
    outs: list[torch.Tensor] = []
    for i in range(B):
        outs.append(
            randn_like_deterministic(
                x_btc[i : i + 1],
                speaker_id=int(sids[i]),
                book_id=int(bids[i]),
                extra=str(extras[i]),
            )
        )
    return torch.cat(outs, dim=0)
