"""Inference-only shared reference conditioning used by zero-shot checkpoints."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from dualpath_projected_block import DualPathProjectedBlock


def _resize_mask(mask_bt: torch.Tensor, length: int) -> torch.Tensor:
    mask = F.interpolate(
        mask_bt.to(dtype=torch.float32).unsqueeze(1),
        size=int(length),
        mode="nearest",
    )
    return mask.squeeze(1).to(dtype=torch.bool)


class _ReferenceTaskAdapter(nn.Module):
    def __init__(self, dim: int, bottleneck: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(int(dim))
        self.net = nn.Sequential(
            nn.Linear(int(dim), int(bottleneck)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(bottleneck), int(dim)),
        )
        self.scale = nn.Parameter(torch.tensor(0.10))

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        return frames + self.scale * self.net(self.norm(frames))


class SharedReferenceEncoder(nn.Module):
    """One mel backbone producing speaker, style and local prompt tokens."""

    def __init__(
        self,
        *,
        n_mels: int = 100,
        dim: int = 256,
        token_count: int = 16,
        layers: int = 4,
        heads: int = 4,
        attn_dim: int = 128,
        conv_dim: int = 128,
        speaker_dim: int = 256,
        style_dim: int = 128,
        adapter_dim: int = 96,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        self.dim = int(dim)
        self.token_count = int(token_count)
        self.mel_stem = nn.Sequential(
            nn.Conv1d(int(n_mels), self.dim, kernel_size=5, stride=2, padding=2),
            nn.GELU(),
            nn.Conv1d(self.dim, self.dim, kernel_size=5, stride=2, padding=2),
            nn.GELU(),
        )
        self.blocks = nn.ModuleList(
            [
                DualPathProjectedBlock(
                    self.dim,
                    num_heads=int(heads),
                    attn_dim=int(attn_dim),
                    conv_dim=int(conv_dim),
                    use_sdpa=True,
                    init_split_identity=True,
                )
                for _ in range(int(layers))
            ]
        )
        self.local_adapter = _ReferenceTaskAdapter(self.dim, adapter_dim, dropout)
        self.queries = nn.Parameter(torch.randn(self.token_count, self.dim) * 0.02)
        self.resampler = nn.MultiheadAttention(self.dim, int(heads), batch_first=True)
        self.token_norm = nn.LayerNorm(self.dim)

        self.speaker_adapter = _ReferenceTaskAdapter(self.dim, adapter_dim, dropout)
        self.speaker_attention = nn.Sequential(
            nn.LayerNorm(self.dim),
            nn.Linear(self.dim, self.dim // 2),
            nn.Tanh(),
            nn.Linear(self.dim // 2, 1),
        )
        self.speaker_head = nn.Sequential(
            nn.LayerNorm(self.dim * 2),
            nn.Linear(self.dim * 2, self.dim * 2),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.dim * 2, int(speaker_dim)),
        )

        self.style_adapter = _ReferenceTaskAdapter(self.dim, adapter_dim, dropout)
        self.style_query = nn.Parameter(torch.randn(1, self.dim) * 0.02)
        self.style_pool = nn.MultiheadAttention(self.dim, int(heads), batch_first=True)
        self.style_head = nn.Sequential(
            nn.LayerNorm(self.dim),
            nn.Linear(self.dim, self.dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.dim, int(style_dim)),
        )

    def forward(
        self,
        mel_bct: torch.Tensor,
        *,
        mask_bt: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        frames = self.mel_stem(mel_bct.float()).transpose(1, 2).contiguous()
        frame_mask = None
        key_padding_mask = None
        if mask_bt is not None:
            frame_mask = _resize_mask(mask_bt, frames.size(1)).to(device=frames.device)
            key_padding_mask = ~frame_mask
        for block in self.blocks:
            frames = block(frames, key_padding_mask=key_padding_mask)

        local_frames = self.local_adapter(frames)
        queries = self.queries.unsqueeze(0).expand(frames.size(0), -1, -1)
        tokens, _ = self.resampler(
            queries,
            local_frames,
            local_frames,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        tokens = self.token_norm(tokens)

        speaker_frames = self.speaker_adapter(frames)
        speaker_logits = self.speaker_attention(speaker_frames).squeeze(-1)
        if key_padding_mask is not None:
            speaker_logits = speaker_logits.masked_fill(key_padding_mask, -1e4)
        speaker_weights = speaker_logits.softmax(dim=-1).unsqueeze(-1)
        speaker_mean = (speaker_frames * speaker_weights).sum(dim=1)
        speaker_var = (
            (speaker_frames - speaker_mean.unsqueeze(1)).square() * speaker_weights
        ).sum(dim=1)
        speaker = self.speaker_head(
            torch.cat([speaker_mean, (speaker_var + 1e-5).sqrt()], dim=-1)
        )
        speaker = F.normalize(speaker.float(), dim=-1).to(dtype=mel_bct.dtype)

        style_frames = self.style_adapter(frames)
        style_query = self.style_query.unsqueeze(0).expand(frames.size(0), -1, -1)
        style, _ = self.style_pool(
            style_query,
            style_frames,
            style_frames,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        style = self.style_head(style.squeeze(1)).to(dtype=mel_bct.dtype)
        return speaker, style, tokens.to(dtype=mel_bct.dtype), frames


class _PromptTextFusionBlock(nn.Module):
    def __init__(self, text_dim: int, prompt_dim: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.text_norm = nn.LayerNorm(int(text_dim))
        self.prompt_norm = nn.LayerNorm(int(prompt_dim))
        self.prompt_proj = (
            nn.Identity()
            if int(prompt_dim) == int(text_dim)
            else nn.Linear(int(prompt_dim), int(text_dim))
        )
        self.cross_attn = nn.MultiheadAttention(
            int(text_dim), int(heads), dropout=float(dropout), batch_first=True
        )
        self.ffn_norm = nn.LayerNorm(int(text_dim))
        self.ffn = nn.Sequential(
            nn.Linear(int(text_dim), int(text_dim) * 4),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(text_dim) * 4, int(text_dim)),
        )
        self.cross_scale = nn.Parameter(torch.tensor(0.05))
        self.ffn_scale = nn.Parameter(torch.tensor(0.10))

    def forward(self, text_bld: torch.Tensor, prompt_bkd: torch.Tensor) -> torch.Tensor:
        query = self.text_norm(text_bld)
        prompt = self.prompt_proj(self.prompt_norm(prompt_bkd))
        attended, _ = self.cross_attn(query, prompt, prompt, need_weights=False)
        text_bld = text_bld + self.cross_scale * attended
        return text_bld + self.ffn_scale * self.ffn(self.ffn_norm(text_bld))


class PromptTextFusion(nn.Module):
    def __init__(
        self,
        *,
        text_dim: int = 512,
        prompt_dim: int = 256,
        layers: int = 1,
        heads: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                _PromptTextFusionBlock(text_dim, prompt_dim, heads, dropout)
                for _ in range(int(layers))
            ]
        )

    def forward(self, text_bld: torch.Tensor, prompt_bkd: torch.Tensor) -> torch.Tensor:
        output = text_bld
        for block in self.blocks:
            output = block(output, prompt_bkd)
        return output


def fuse_prompt_preserving_prefix(
    fusion: PromptTextFusion,
    text_bld: torch.Tensor,
    prompt_bkd: torch.Tensor,
    *,
    special_len: int,
) -> torch.Tensor:
    fused = fusion(text_bld, prompt_bkd)
    prefix_len = max(0, min(int(special_len), int(text_bld.size(1))))
    if prefix_len == 0:
        return fused
    return torch.cat([text_bld[:, :prefix_len], fused[:, prefix_len:]], dim=1)
