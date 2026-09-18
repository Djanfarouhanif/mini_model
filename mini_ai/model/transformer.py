"""Bloc Transformer : pre-LayerNorm, attention causale, MLP GELU."""

from __future__ import annotations

import torch
import torch.nn as nn

from .attention import CausalSelfAttention
from .config import GPTConfig


class LayerNorm(nn.Module):
    """LayerNorm avec biais optionnel (nn.LayerNorm ne permet pas bias=False partout)."""

    def __init__(self, ndim: int, bias: bool = True):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)


class MLP(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        hidden = config.mlp_ratio * config.n_embd
        self.c_fc = nn.Linear(config.n_embd, hidden, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(hidden, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.c_proj(self.gelu(self.c_fc(x))))


class TransformerBlock(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, config.bias)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    """Pile de blocs + LayerNorm final (sans embeddings ni tête de sortie)."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.blocks = nn.ModuleList(TransformerBlock(config) for _ in range(config.n_layer))
        self.ln_f = LayerNorm(config.n_embd, config.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return self.ln_f(x)
