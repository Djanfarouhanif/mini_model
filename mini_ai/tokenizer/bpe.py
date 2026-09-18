"""Cœur du tokenizer : BPE au niveau octet (byte-level), à la GPT-2.

Pipeline :  Text → Normalization (NFC) → Pré-tokenisation → BPE → Token IDs

Travailler sur les octets garantit que n'importe quel texte est encodable
(pas de <UNK> réel) et que ``decode(encode(text)) == text`` pour tout texte
déjà normalisé en NFC.

Disposition des identifiants :

    0..3      tokens spéciaux  <PAD> <UNK> <BOS> <EOS>
    4..259    les 256 octets
    260..     fusions BPE, dans l'ordre d'apprentissage
"""

from __future__ import annotations

import re
import unicodedata
from typing import Iterable

SPECIAL_TOKENS = ["<PAD>", "<UNK>", "<BOS>", "<EOS>"]
PAD_ID, UNK_ID, BOS_ID, EOS_ID = 0, 1, 2, 3
BYTE_OFFSET = len(SPECIAL_TOKENS)  # premier id d'octet
N_BYTES = 256
BASE_VOCAB_SIZE = BYTE_OFFSET + N_BYTES  # 260

# Découpe le texte en morceaux "mots" avant le BPE : un mot (avec son espace
# précédent), un nombre, une suite de ponctuation, ou des blancs.
PRETOKENIZE_RE = re.compile(r" ?[^\W\d_]+| ?\d+| ?[^\s\w]+|\s+(?!\S)|\s+")


def normalize(text: str) -> str:
    """Normalisation Unicode NFC (forme canonique composée)."""
    return unicodedata.normalize("NFC", text)


def pretokenize(text: str) -> list[str]:
    return PRETOKENIZE_RE.findall(text)


def merge_pair(ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    """Remplace chaque occurrence consécutive de ``pair`` par ``new_id``."""
    out: list[int] = []
    i = 0
    a, b = pair
    n = len(ids)
    while i < n:
        if i < n - 1 and ids[i] == a and ids[i + 1] == b:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


class BPE:
    """Table de fusions BPE + vocabulaire (id → bytes)."""

    def __init__(self, merges: Iterable[tuple[int, int]] | None = None):
        self.vocab: dict[int, bytes] = {
            BYTE_OFFSET + b: bytes([b]) for b in range(N_BYTES)
        }
        # pair → nouvel id ; l'id croît avec l'ordre des fusions, il sert
        # donc directement de rang de priorité.
        self.merges: dict[tuple[int, int], int] = {}
        self._cache: dict[str, list[int]] = {}
        for pair in merges or []:
            self.add_merge(tuple(pair))  # type: ignore[arg-type]

    # ------------------------------------------------------------------ vocab
    @property
    def vocab_size(self) -> int:
        return BASE_VOCAB_SIZE + len(self.merges)

    def add_merge(self, pair: tuple[int, int]) -> int:
        if pair in self.merges:
            return self.merges[pair]
        new_id = BASE_VOCAB_SIZE + len(self.merges)
        self.merges[pair] = new_id
        self.vocab[new_id] = self.vocab[pair[0]] + self.vocab[pair[1]]
        self._cache.clear()
        return new_id

    def merge_list(self) -> list[tuple[int, int]]:
        return sorted(self.merges, key=self.merges.__getitem__)

    # ----------------------------------------------------------------- encode
    def encode_bytes(self, data: bytes) -> list[int]:
        ids = [b + BYTE_OFFSET for b in data]
        while len(ids) >= 2:
            best: tuple[int, int] | None = None
            best_rank = 0
            for pair in zip(ids, ids[1:]):
                rank = self.merges.get(pair)
                if rank is not None and (best is None or rank < best_rank):
                    best, best_rank = pair, rank
            if best is None:
                break
            ids = merge_pair(ids, best, best_rank)
        return ids

    def encode_chunk(self, chunk: str) -> list[int]:
        cached = self._cache.get(chunk)
        if cached is None:
            cached = self.encode_bytes(chunk.encode("utf-8"))
            self._cache[chunk] = cached
        return list(cached)

    def encode(self, text: str) -> list[int]:
        ids: list[int] = []
        for chunk in pretokenize(normalize(text)):
            ids.extend(self.encode_chunk(chunk))
        return ids

    # ----------------------------------------------------------------- decode
    def decode_bytes(self, ids: Iterable[int]) -> bytes:
        return b"".join(self.vocab[i] for i in ids if i in self.vocab)

    def decode(self, ids: Iterable[int]) -> str:
        return self.decode_bytes(ids).decode("utf-8", errors="replace")
