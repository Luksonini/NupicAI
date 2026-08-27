#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
benchmark_config.py

Shared tiny config dataclasses used across SalmonTTS2/SalmonTTS3 codepaths.

This exists to avoid importing anything from `legacy/` in production code.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class BenchmarkConfig:
    vocab_size: int
    hidden_dim: int = 256
    num_layers: int = 4
    n_mels: int = 100
    max_pos: int = 2000

