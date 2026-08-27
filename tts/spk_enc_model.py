#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvNeXtBlock1d(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        dim = int(dim)
        self.dwconv = nn.Conv1d(dim, dim, kernel_size=7, padding=3, groups=dim, bias=False)
        self.norm = nn.LayerNorm(dim)
        self.pwconv1 = nn.Conv1d(dim, 4 * dim, kernel_size=1, bias=False)
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv1d(4 * dim, dim, kernel_size=1, bias=False)

    def forward(self, x_bct: torch.Tensor) -> torch.Tensor:
        residual = x_bct
        x = self.dwconv(x_bct)
        x = x.transpose(1, 2).contiguous()
        x = self.norm(x)
        x = x.transpose(1, 2).contiguous()
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        return residual + x


class AttentiveStatsPooling(nn.Module):
    """
    ECAPA-like attentive statistics pooling.
    Returns concatenated weighted mean and weighted std: [B, 2*C].
    """

    def __init__(self, in_dim: int, head_dim: int = 128):
        super().__init__()
        in_dim = int(in_dim)
        head_dim = int(head_dim)
        self.attn = nn.Sequential(
            nn.Conv1d(in_dim, head_dim, kernel_size=1, bias=False),
            nn.Tanh(),
            nn.Conv1d(head_dim, in_dim, kernel_size=1, bias=False),
        )

    def forward(self, x_bct: torch.Tensor, mask_bt: Optional[torch.Tensor] = None) -> torch.Tensor:
        logits = self.attn(x_bct)
        if mask_bt is not None:
            m = mask_bt.to(dtype=torch.bool, device=x_bct.device).unsqueeze(1)
            neg_inf = torch.finfo(logits.dtype).min
            logits = logits.masked_fill(~m, neg_inf)
            valid = m.any(dim=-1, keepdim=True)
            logits = torch.where(valid, logits, torch.zeros_like(logits))
        w = F.softmax(logits, dim=-1)
        mu = (x_bct * w).sum(dim=-1)
        x2 = ((x_bct * x_bct) * w).sum(dim=-1)
        var = (x2 - mu * mu).clamp_min(1e-9)
        std = torch.sqrt(var)
        return torch.cat([mu, std], dim=-1)


class MelSpeakerEncoder(nn.Module):
    """
    mel[B,C,T] -> z[B,d] (L2-normalized)
    """

    def __init__(
        self,
        *,
        n_mels: int = 100,
        d_spk: int = 256,
        hidden_dim: int = 256,
        num_layers: int = 4,
        attn_head_dim: int = 128,
    ):
        super().__init__()
        n_mels = int(n_mels)
        self.d_spk = int(d_spk)
        hidden_dim = int(hidden_dim)
        num_layers = int(max(1, num_layers))
        attn_head_dim = int(max(8, attn_head_dim))

        self.pre_conv = nn.Sequential(
            nn.Conv1d(n_mels, hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.GELU(),
        )
        self.blocks = nn.ModuleList([ConvNeXtBlock1d(hidden_dim) for _ in range(num_layers)])
        self.pooling = AttentiveStatsPooling(hidden_dim * num_layers, head_dim=attn_head_dim)

        pooled_dim = 2 * (hidden_dim * num_layers)
        hid = int(max(256, self.d_spk * 2))
        self.mlp = nn.Sequential(
            nn.LayerNorm(pooled_dim),
            nn.Linear(pooled_dim, hid),
            nn.GELU(),
            nn.Linear(hid, self.d_spk),
        )

    def forward(self, mel_bct: torch.Tensor, *, mask_bt: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.pre_conv(mel_bct)
        outs = []
        for blk in self.blocks:
            x = blk(x)
            outs.append(x)
        x_agg = torch.cat(outs, dim=1)
        feat = self.pooling(x_agg, mask_bt=mask_bt)
        z = self.mlp(feat)
        z = z / z.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        return z


@dataclass(frozen=True)
class ArcFaceConfig:
    s: float = 30.0
    m: float = 0.30
    easy_margin: bool = False


class ArcFaceHead(nn.Module):
    """
    ArcFace / AAM-Softmax head.
    """

    def __init__(self, emb_dim: int, num_classes: int, *, cfg: ArcFaceConfig):
        super().__init__()
        self.emb_dim = int(emb_dim)
        self.num_classes = int(num_classes)
        self.cfg = cfg
        self.W = nn.Parameter(torch.randn(self.num_classes, self.emb_dim) * 0.01)

        self.cos_m = math.cos(float(cfg.m))
        self.sin_m = math.sin(float(cfg.m))
        self.th = math.cos(math.pi - float(cfg.m))
        self.mm = math.sin(math.pi - float(cfg.m)) * float(cfg.m)

    def forward(self, z_be: torch.Tensor, y_b: torch.Tensor) -> torch.Tensor:
        z = F.normalize(z_be, dim=-1)
        W = F.normalize(self.W, dim=-1)
        cosine = torch.matmul(z, W.t()).clamp(-1.0, 1.0)
        sine = torch.sqrt((1.0 - cosine * cosine).clamp_min(1e-9))
        phi = cosine * self.cos_m - sine * self.sin_m
        if bool(self.cfg.easy_margin):
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        y = y_b.to(dtype=torch.long)
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, y.view(-1, 1), 1.0)
        logits = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        return float(self.cfg.s) * logits


def speaker_top1_acc(logits_bc: torch.Tensor, y_b: torch.Tensor) -> torch.Tensor:
    pred = torch.argmax(logits_bc, dim=1)
    return (pred == y_b.to(dtype=torch.long)).float().mean()
