"""Interface publique du tokenizer : encode / decode / save / load / train."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from .bpe import BOS_ID, EOS_ID, PAD_ID, SPECIAL_TOKENS, UNK_ID, BPE, normalize
from .trainer import BPETrainer


class Tokenizer:
    pad_id = PAD_ID
    unk_id = UNK_ID
    bos_id = BOS_ID
    eos_id = EOS_ID
    special_tokens = SPECIAL_TOKENS

    def __init__(self, bpe: BPE | None = None):
        self.bpe = bpe or BPE()

    # ------------------------------------------------------------- properties
    @property
    def vocab_size(self) -> int:
        return self.bpe.vocab_size

    def id_to_token(self, idx: int) -> str:
        if idx < len(SPECIAL_TOKENS):
            return SPECIAL_TOKENS[idx]
        return self.bpe.vocab[idx].decode("utf-8", errors="replace")

    def token_to_id(self, token: str) -> int:
        if token in SPECIAL_TOKENS:
            return SPECIAL_TOKENS.index(token)
        ids = self.bpe.encode(token)
        return ids[0] if len(ids) == 1 else UNK_ID

    # ---------------------------------------------------------- encode/decode
    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        ids = self.bpe.encode(text)
        if add_bos:
            ids.insert(0, BOS_ID)
        if add_eos:
            ids.append(EOS_ID)
        return ids

    def decode(self, ids: Iterable[int], skip_special: bool = True) -> str:
        if skip_special:
            return self.bpe.decode(i for i in ids if i >= len(SPECIAL_TOKENS))
        parts: list[str] = []
        buffer: list[int] = []
        for i in ids:
            if i < len(SPECIAL_TOKENS):
                if buffer:
                    parts.append(self.bpe.decode(buffer))
                    buffer = []
                parts.append(SPECIAL_TOKENS[i])
            else:
                buffer.append(i)
        if buffer:
            parts.append(self.bpe.decode(buffer))
        return "".join(parts)

    def tokens(self, text: str) -> list[str]:
        """Représentation lisible des tokens (utile pour le debug)."""
        return [self.id_to_token(i) for i in self.encode(text)]

    # ------------------------------------------------------------ persistence
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "version": 1,
            "type": "byte-level-bpe",
            "special_tokens": SPECIAL_TOKENS,
            "vocab_size": self.vocab_size,
            "merges": [list(p) for p in self.bpe.merge_list()],
        }
        path.write_text(json.dumps(data), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "Tokenizer":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(BPE(tuple(p) for p in data["merges"]))

    # --------------------------------------------------------------- training
    @classmethod
    def train(
        cls,
        texts: Iterable[str] | str,
        vocab_size: int = 4096,
        min_frequency: int = 2,
        verbose: bool = False,
    ) -> "Tokenizer":
        trainer = BPETrainer(vocab_size=vocab_size, min_frequency=min_frequency, verbose=verbose)
        return cls(trainer.train(texts))

    # --------------------------------------------------------------- helpers
    @staticmethod
    def normalize(text: str) -> str:
        return normalize(text)

    def __len__(self) -> int:
        return self.vocab_size

    def __repr__(self) -> str:
        return f"Tokenizer(vocab_size={self.vocab_size})"
