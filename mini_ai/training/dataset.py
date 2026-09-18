"""Pipeline de données.

    Raw dataset → Cleaning → Deduplication → Tokenizer → Token IDs → Train / Validation

Les tokens sont stockés en uint16 dans ``train.bin`` / ``validation.bin`` et
lus via numpy.memmap ; un batch est un ensemble de fenêtres aléatoires de
``block_size`` tokens, la cible étant la fenêtre décalée d'un token.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np
import torch

from ..tokenizer import Tokenizer

TOKEN_DTYPE = np.uint16  # suffisant tant que vocab_size < 65 536


# ---------------------------------------------------------------- cleaning
def clean_text(text: str) -> str:
    text = Tokenizer.normalize(text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)  # espaces multiples
    text = re.sub(r"\n{3,}", "\n\n", text)  # lignes vides multiples
    return "\n".join(line.strip() for line in text.split("\n")).strip()


def split_documents(text: str) -> list[str]:
    """Un document = un paragraphe séparé par une ligne vide."""
    return [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]


def deduplicate(docs: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for doc in docs:
        key = hashlib.sha1(doc.lower().encode("utf-8")).hexdigest()
        if key not in seen:
            seen.add(key)
            out.append(doc)
    return out


def read_raw_corpus(raw_dir: str | Path) -> list[str]:
    raw_dir = Path(raw_dir)
    docs: list[str] = []
    for path in sorted(raw_dir.rglob("*")):
        if path.suffix.lower() in {".txt", ".md", ".py"} and path.is_file():
            docs.extend(split_documents(clean_text(path.read_text(encoding="utf-8", errors="replace"))))
    return deduplicate(docs)


# ----------------------------------------------------------------- encoding
def encode_documents(docs: Iterable[str], tokenizer: Tokenizer) -> np.ndarray:
    ids: list[int] = []
    for doc in docs:
        ids.extend(tokenizer.encode(doc, add_bos=True, add_eos=True))
    return np.array(ids, dtype=TOKEN_DTYPE)


def prepare_dataset(
    raw_dir: str | Path,
    out_dir: str | Path,
    tokenizer_path: str | Path,
    vocab_size: int = 4096,
    val_ratio: float = 0.1,
    tokenizer: Tokenizer | None = None,
    verbose: bool = True,
) -> dict:
    """Exécute tout le pipeline et écrit train.bin / validation.bin."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    docs = read_raw_corpus(raw_dir)
    if not docs:
        raise FileNotFoundError(f"aucun fichier .txt/.md/.py dans {raw_dir}")
    n_chars = sum(len(d) for d in docs)
    if verbose:
        print(f"[data] {len(docs)} documents, {n_chars:,} caractères après nettoyage/dédoublonnage")

    if tokenizer is None:
        tokenizer = Tokenizer.train(docs, vocab_size=vocab_size, verbose=verbose)
        tokenizer.save(tokenizer_path)
        if verbose:
            print(f"[data] tokenizer sauvegardé : {tokenizer_path} ({tokenizer.vocab_size} tokens)")

    # Mélange déterministe puis découpe train/val au niveau des documents.
    rng = np.random.default_rng(42)
    order = rng.permutation(len(docs))
    n_val = max(1, int(len(docs) * val_ratio)) if len(docs) > 1 else 0
    val_docs = [docs[i] for i in order[:n_val]]
    train_docs = [docs[i] for i in order[n_val:]]

    train_ids = encode_documents(train_docs, tokenizer)
    val_ids = encode_documents(val_docs, tokenizer) if val_docs else train_ids[-max(1, len(train_ids) // 10) :]

    train_path = out_dir / "train.bin"
    val_path = out_dir / "validation.bin"
    train_ids.tofile(train_path)
    val_ids.tofile(val_path)
    if verbose:
        print(f"[data] train : {len(train_ids):,} tokens → {train_path}")
        print(f"[data] val   : {len(val_ids):,} tokens → {val_path}")
    return {
        "documents": len(docs),
        "chars": n_chars,
        "train_tokens": int(len(train_ids)),
        "val_tokens": int(len(val_ids)),
        "vocab_size": tokenizer.vocab_size,
    }


# ------------------------------------------------------------------ dataset
class TokenDataset:
    """Accès par batch aléatoire aux fichiers .bin (train / validation)."""

    def __init__(self, train_path: str | Path, val_path: str | Path, block_size: int, device: str = "cpu"):
        self.block_size = block_size
        self.device = device
        self.splits = {
            "train": np.memmap(train_path, dtype=TOKEN_DTYPE, mode="r"),
            "val": np.memmap(val_path, dtype=TOKEN_DTYPE, mode="r"),
        }
        for name, data in self.splits.items():
            if len(data) <= block_size + 1:
                raise ValueError(f"split '{name}' trop petit ({len(data)} tokens) pour block_size={block_size}")

    def close(self) -> None:
        """Libère les memmaps (nécessaire sous Windows avant de supprimer les fichiers)."""
        for name in list(self.splits):
            mm = self.splits.pop(name)
            if hasattr(mm, "_mmap") and mm._mmap is not None:
                mm._mmap.close()
            del mm

    def __enter__(self) -> "TokenDataset":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __len__(self) -> int:
        return len(self.splits["train"])

    def size(self, split: str) -> int:
        return len(self.splits[split])

    def get_batch(self, split: str, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        data = self.splits[split]
        ix = torch.randint(len(data) - self.block_size - 1, (batch_size,))
        x = torch.stack([torch.from_numpy(data[i : i + self.block_size].astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy(data[i + 1 : i + 1 + self.block_size].astype(np.int64)) for i in ix])
        return x.to(self.device), y.to(self.device)

    def iter_batches(self, split: str, batch_size: int, n: int) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        for _ in range(n):
            yield self.get_batch(split, batch_size)
