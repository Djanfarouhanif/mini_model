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


def iter_paragraphs(path: Path) -> Iterator[str]:
    """Lit un fichier paragraphe par paragraphe (séparés par une ligne vide),
    sans jamais charger le fichier entier en mémoire."""
    buffer: list[str] = []
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.strip():
                buffer.append(line)
            elif buffer:
                yield "".join(buffer)
                buffer = []
    if buffer:
        yield "".join(buffer)


def read_raw_corpus(raw_dir: str | Path) -> list[str]:
    """Tous les documents nettoyés et dédoublonnés de data/raw/ (en streaming)."""
    raw_dir = Path(raw_dir)
    seen: set[str] = set()
    docs: list[str] = []
    for path in sorted(raw_dir.rglob("*")):
        if path.suffix.lower() not in {".txt", ".md", ".py"} or not path.is_file():
            continue
        for raw in iter_paragraphs(path):
            for doc in split_documents(clean_text(raw)):
                key = hashlib.sha1(doc.lower().encode("utf-8")).hexdigest()
                if key not in seen:
                    seen.add(key)
                    docs.append(doc)
    return docs


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
    tokenizer_sample_chars: int = 50_000_000,
    verbose: bool = True,
) -> dict:
    """Exécute tout le pipeline et écrit train.bin / validation.bin.

    Conçu pour des corpus de plusieurs Go : lecture en streaming, tokenizer
    appris sur un échantillon de ``tokenizer_sample_chars`` caractères, et
    tokens écrits au fil de l'eau dans les fichiers .bin.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    docs = read_raw_corpus(raw_dir)
    if not docs:
        raise FileNotFoundError(f"aucun fichier .txt/.md/.py dans {raw_dir}")
    n_chars = sum(len(d) for d in docs)
    if verbose:
        print(f"[data] {len(docs):,} documents, {n_chars:,} caractères après nettoyage/dédoublonnage")

    # Mélange déterministe : sert à la fois à l'échantillon du tokenizer et à
    # la découpe train/val.
    rng = np.random.default_rng(42)
    order = rng.permutation(len(docs))

    if tokenizer is None:
        sample: list[str] = []
        sample_chars = 0
        for i in order:
            sample.append(docs[i])
            sample_chars += len(docs[i])
            if sample_chars >= tokenizer_sample_chars:
                break
        if verbose and sample_chars < n_chars:
            print(f"[data] tokenizer appris sur un échantillon de {sample_chars:,} caractères ({len(sample):,} documents)")
        tokenizer = Tokenizer.train(sample, vocab_size=vocab_size, verbose=verbose)
        del sample
        tokenizer.save(tokenizer_path)
        if verbose:
            print(f"[data] tokenizer sauvegardé : {tokenizer_path} ({tokenizer.vocab_size} tokens)")

    # Découpe train/val par quantité de caractères (≈ tokens), à la frontière
    # des documents ; les tokens sont écrits directement dans les fichiers.
    train_path = out_dir / "train.bin"
    val_path = out_dir / "validation.bin"
    target_val_chars = int(n_chars * val_ratio) if len(docs) > 1 else 0
    counts = {"train": 0, "val": 0}
    with train_path.open("wb") as f_train, val_path.open("wb") as f_val:
        val_chars = 0
        for n, i in enumerate(order, 1):
            doc = docs[i]
            ids = np.array(tokenizer.encode(doc, add_bos=True, add_eos=True), dtype=TOKEN_DTYPE)
            if val_chars < target_val_chars and n < len(docs):
                f_val.write(ids.tobytes())
                val_chars += len(doc)
                counts["val"] += len(ids)
            else:
                f_train.write(ids.tobytes())
                counts["train"] += len(ids)
            if verbose and n % 20_000 == 0:
                print(f"[data] {n:,}/{len(docs):,} documents encodés ({counts['train'] + counts['val']:,} tokens)")
    if counts["val"] == 0:  # corpus d'un seul document : on recopie la fin du train
        train_ids = np.fromfile(train_path, dtype=TOKEN_DTYPE)
        train_ids[-max(1, len(train_ids) // 10):].tofile(val_path)
        counts["val"] = max(1, len(train_ids) // 10)

    if verbose:
        print(f"[data] train : {counts['train']:,} tokens → {train_path}")
        print(f"[data] val   : {counts['val']:,} tokens → {val_path}")
    return {
        "documents": len(docs),
        "chars": n_chars,
        "train_tokens": counts["train"],
        "val_tokens": counts["val"],
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
