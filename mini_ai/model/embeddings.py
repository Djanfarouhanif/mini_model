"""Embeddings de tokens et de position (apprises ou RoPE)."""

from __future__ import annotations

import torch
import torch.nn as nn


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, n_embd: int):
        super().__init__()
        self.weight_table = nn.Embedding(vocab_size, n_embd)

    @property
    def weight(self) -> torch.Tensor:
        return self.weight_table.weight

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        return self.weight_table(idx)


class LearnedPositionalEmbedding(nn.Module):
    def __init__(self, block_size: int, n_embd: int):
        super().__init__()
        self.block_size = block_size
        self.table = nn.Embedding(block_size, n_embd)

    def forward(self, t: int, device: torch.device) -> torch.Tensor:
        if t > self.block_size:
            raise ValueError(f"séquence de {t} tokens > block_size {self.block_size}")
        pos = torch.arange(0, t, dtype=torch.long, device=device)
        return self.table(pos)  # (T, C)


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (Su et al., 2021), appliqué à q et k."""

    def __init__(self, head_dim: int, block_size: int, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        t = torch.arange(block_size).float()
        freqs = torch.outer(t, inv_freq)  # (T, head_dim/2)
        emb = torch.cat([freqs, freqs], dim=-1)  # (T, head_dim)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        half = x.shape[-1] // 2
        return torch.cat([-x[..., half:], x[..., :half]], dim=-1)

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # q, k : (B, n_head, T, head_dim)
        t = q.shape[-2]
        cos = self.cos_cached[:t].to(q.dtype)
        sin = self.sin_cached[:t].to(q.dtype)
        q = q * cos + self._rotate_half(q) * sin
        k = k * cos + self._rotate_half(k) * sin
        return q, k
