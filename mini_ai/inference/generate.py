"""Génération de texte avec le vrai modèle GPT + tokenizer BPE.

    prompt → token ids → model.generate() → generated tokens → texte
"""

from __future__ import annotations

from pathlib import Path

import torch

from ..model import GPT
from ..tokenizer import Tokenizer
from ..training.checkpoint import load_model


class TextGenerator:
    name = "gpt"

    def __init__(self, model: GPT, tokenizer: Tokenizer, device: str = "cpu"):
        self.model = model.to(device).eval()
        self.tokenizer = tokenizer
        self.device = device

    def encode(self, prompt: str, add_bos: bool = True) -> torch.Tensor:
        ids = self.tokenizer.encode(prompt, add_bos=add_bos)
        # On garde la fin du prompt si celui-ci dépasse le contexte.
        ids = ids[-self.model.config.block_size :]
        return torch.tensor([ids], dtype=torch.long, device=self.device)

    @torch.no_grad()
    def generate_tokens(
        self,
        prompt: str,
        max_tokens: int = 100,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
        stop_at_eos: bool = True,
        repetition_penalty: float = 1.0,
    ) -> list[int]:
        idx = self.encode(prompt)
        n_prompt = idx.shape[1]
        out = self.model.generate(
            idx,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            eos_id=self.tokenizer.eos_id if stop_at_eos else None,
            repetition_penalty=repetition_penalty,
        )
        return out[0, n_prompt:].tolist()

    def generate(
        self,
        prompt: str,
        max_tokens: int = 100,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
        stop_at_eos: bool = True,
        repetition_penalty: float = 1.0,
    ) -> str:
        """Retourne uniquement le texte généré (sans le prompt)."""
        ids = self.generate_tokens(prompt, max_tokens, temperature, top_k, top_p, stop_at_eos, repetition_penalty)
        if stop_at_eos and self.tokenizer.eos_id in ids:
            ids = ids[: ids.index(self.tokenizer.eos_id)]
        return self.tokenizer.decode(ids)

    def complete(self, prompt: str, **kwargs) -> str:
        """Prompt + texte généré."""
        return prompt + self.generate(prompt, **kwargs)


def load_generator(checkpoint_path: str | Path, tokenizer_path: str | Path, device: str = "cpu") -> TextGenerator:
    model, _ = load_model(checkpoint_path, device)
    tokenizer = Tokenizer.load(tokenizer_path)
    if tokenizer.vocab_size > model.config.vocab_size:
        raise ValueError(
            f"tokenizer ({tokenizer.vocab_size} tokens) incompatible avec le modèle ({model.config.vocab_size})"
        )
    return TextGenerator(model, tokenizer, device)
