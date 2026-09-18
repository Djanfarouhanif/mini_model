"""Modèle GPT complet : embeddings → Transformer → logits, + génération."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import GPTConfig
from .embeddings import LearnedPositionalEmbedding, TokenEmbedding
from .transformer import Transformer


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad or not trainable_only)


class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        self.wte = TokenEmbedding(config.vocab_size, config.n_embd)
        self.wpe = LearnedPositionalEmbedding(config.block_size, config.n_embd) if config.pos_type == "learned" else None
        self.drop = nn.Dropout(config.dropout)
        self.transformer = Transformer(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        if config.tie_weights:
            self.lm_head.weight = self.wte.weight_table.weight

        self.apply(self._init_weights)
        # Init GPT-2 : les projections résiduelles sont réduites en 1/√(2L).
        for name, p in self.named_parameters():
            if name.endswith("c_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    # ------------------------------------------------------------------ utils
    def num_parameters(self, non_embedding: bool = False) -> int:
        n = count_parameters(self)
        if non_embedding and self.wpe is not None:
            n -= self.wpe.table.weight.numel()
        return n

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    # ---------------------------------------------------------------- forward
    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        B, T = idx.shape
        if T > self.config.block_size:
            raise ValueError(f"séquence de {T} tokens > block_size {self.config.block_size}")

        x = self.wte(idx)
        if self.wpe is not None:
            x = x + self.wpe(T, idx.device)
        x = self.drop(x)
        x = self.transformer(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
            return logits, loss

        # En inférence, seul le dernier pas est utile.
        logits = self.lm_head(x[:, [-1], :])
        return logits, None

    # -------------------------------------------------------------- optimizer
    def configure_optimizers(self, weight_decay: float, learning_rate: float, betas: tuple[float, float] = (0.9, 0.95)):
        decay = [p for n, p in self.named_parameters() if p.requires_grad and p.dim() >= 2]
        no_decay = [p for n, p in self.named_parameters() if p.requires_grad and p.dim() < 2]
        groups = [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        return torch.optim.AdamW(groups, lr=learning_rate, betas=betas)

    # ------------------------------------------------------------- generation
    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
        eos_id: int | None = None,
        repetition_penalty: float = 1.0,
    ) -> torch.Tensor:
        """Échantillonne ``max_new_tokens`` tokens à la suite de ``idx`` (B, T)."""
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size :]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]

            if repetition_penalty != 1.0:
                for b in range(idx.size(0)):
                    seen = torch.unique(idx[b])
                    penal = logits[b, seen]
                    logits[b, seen] = torch.where(penal > 0, penal / repetition_penalty, penal * repetition_penalty)

            if temperature <= 0:
                next_id = logits.argmax(dim=-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k is not None and top_k > 0:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = float("-inf")
                if top_p is not None and 0 < top_p < 1.0:
                    sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                    probs = F.softmax(sorted_logits, dim=-1)
                    cum = torch.cumsum(probs, dim=-1)
                    remove = cum - probs > top_p  # garde toujours le 1er token
                    sorted_logits[remove] = float("-inf")
                    logits = torch.full_like(logits, float("-inf")).scatter(1, sorted_idx, sorted_logits)
                probs = F.softmax(logits, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1)

            idx = torch.cat([idx, next_id], dim=1)
            if eos_id is not None and bool((next_id == eos_id).all()):
                break
        return idx
