#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
semantic_bridge.py — most pamięci (prefix/state) oparty wyłącznie na tekście.
Wersja CLEAN: usunięto CLIP, Word-Level logic, Audio Encoders.
Została tylko logika zarządzania stanem pamięci (Short/Long) i projekcje.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Konfiguracja + stan pamięci
# ============================================================

@dataclass
class SemanticBridgeConfig:
    text_dim: int             # wymiar encodera tekstu (D_text)
    mel_dim: int = 100        # (używane tylko informacyjnie w configu)
    mem_dim: int = 128        # wymiar pamięci (short/long)


@dataclass
class SemanticBridgeState:
    """
    Stan pamięci między chunkami:
      - short: lokalna pamięć ostatniego chunka
      - long:  wolno zmieniająca się pamięć globalna (gated-EMA)
    """
    short: torch.Tensor  # [B, D_mem]
    long: torch.Tensor   # [B, D_mem]

    @staticmethod
    def init(
        batch_size: int,
        mem_dim: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> "SemanticBridgeState":
        short = torch.zeros(batch_size, mem_dim, device=device, dtype=dtype)
        long = torch.zeros(batch_size, mem_dim, device=device, dtype=dtype)
        return SemanticBridgeState(short=short, long=long)


# ============================================================
# Bloczki pomocnicze
# ============================================================

class ProjectionHead(nn.Module):
    """
    Prosta główka projekcyjna: Linear -> GELU -> (LayerNorm) -> Linear
    Używana do rzutowania: text_emb -> memory_emb
    """
    def __init__(self, in_dim: int, out_dim: int, use_ln: bool = False):
        super().__init__()
        hidden = max(in_dim, out_dim)
        layers = [nn.Linear(in_dim, hidden), nn.GELU()]
        if use_ln:
            layers.append(nn.LayerNorm(hidden))
        layers.append(nn.Linear(hidden, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ============================================================
# Główna klasa: SemanticBridge
# ============================================================

class SemanticBridge(nn.Module):
    """
    Most pamięciowy:
      1. Rzutuje zpoolowany tekst (D_text) -> przestrzeń pamięci (D_mem).
      2. Aktualizuje pamięć Short (bieżąca) i Long (gated EMA).
      3. Generuje tokeny prefixowe [short, long] z powrotem do przestrzeni D_text.
    """

    def __init__(self, cfg: SemanticBridgeConfig):
        super().__init__()
        self.cfg = cfg

        # Projekcja wejściowa: text_pool -> memory
        self.proj_text = ProjectionHead(cfg.text_dim, cfg.mem_dim, use_ln=True)

        # Mieszanie z_text przed wstawieniem do pamięci (opcjonalna warstwa transformacji)
        self.mix_text = ProjectionHead(cfg.mem_dim, cfg.mem_dim, use_ln=False)

        # Bramka dla long memory (gated-EMA)
        self.long_gate = nn.Linear(cfg.mem_dim * 2, cfg.mem_dim)

        # Projekcje wyjściowe: memory -> prefix tokens (do wstrzyknięcia do encodera)
        self.proj_short_to_ctx = nn.Linear(cfg.mem_dim, cfg.text_dim)
        self.proj_long_to_ctx  = nn.Linear(cfg.mem_dim, cfg.text_dim)

    def init_state(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> SemanticBridgeState:
        return SemanticBridgeState.init(
            batch_size=batch_size,
            mem_dim=self.cfg.mem_dim,
            device=device,
            dtype=dtype,
        )

    def make_prefix_tokens(self, state_before: SemanticBridgeState) -> torch.Tensor:
        """
        Buduje 2 prefix tokeny w przestrzeni D_text: [short_prev, long_prev]
        Używane na początku forwardu encodera.
        """
        short_prev = state_before.short     # [B, D_mem]
        long_prev  = state_before.long      # [B, D_mem]

        tok_short = self.proj_short_to_ctx(short_prev)  # [B, D_text]
        tok_long  = self.proj_long_to_ctx(long_prev)    # [B, D_text]

        # Stack: [B, 2, D_text]
        prefix_tokens = torch.stack([tok_short, tok_long], dim=1)
        return prefix_tokens

    def update_state(self, state_before: SemanticBridgeState, z_text: torch.Tensor) -> SemanticBridgeState:
        """
        Aktualizacja pamięci po przetworzeniu chunka tekstu.
        z_text: [B, D_mem] (wyjście z self.proj_text(h_pool_enc))
        """
        short_prev = state_before.short
        long_prev  = state_before.long

        # 1. Przetworzenie wejścia
        z_mix = self.mix_text(z_text)   # [B, D_mem]

        # 2. Obliczenie bramki dla Long Memory
        # Decyduje, ile nowego kontekstu (z_mix) wpuścić do pamięci długotrwałej
        gate_in = torch.cat([z_mix, long_prev], dim=-1)   # [B, 2*D_mem]
        gate = torch.sigmoid(self.long_gate(gate_in))     # [B, D_mem]

        # 3. Update
        # Short memory to po prostu bieżący kontekst
        short_new = z_mix
        # Long memory to średnia ważona (EMA) sterowana bramką
        long_new  = gate * z_mix + (1.0 - gate) * long_prev

        return SemanticBridgeState(short=short_new, long=long_new)