"""Apprentissage des fusions BPE.

Algorithme classique (Sennrich et al.) avec comptage incrémental : à chaque
fusion on ne met à jour que les mots qui contiennent la paire fusionnée, et
la meilleure paire est tirée d'un tas avec invalidation paresseuse.
"""

from __future__ import annotations

import heapq
import time
from collections import Counter, defaultdict
from typing import Iterable

from .bpe import BASE_VOCAB_SIZE, BYTE_OFFSET, BPE, merge_pair, normalize, pretokenize


class BPETrainer:
    def __init__(self, vocab_size: int = 4096, min_frequency: int = 2, verbose: bool = False):
        if vocab_size <= BASE_VOCAB_SIZE:
            raise ValueError(f"vocab_size doit être > {BASE_VOCAB_SIZE}")
        self.vocab_size = vocab_size
        self.min_frequency = min_frequency
        self.verbose = verbose

    def count_words(self, texts: Iterable[str]) -> Counter[str]:
        word_freqs: Counter[str] = Counter()
        for text in texts:
            word_freqs.update(pretokenize(normalize(text)))
        return word_freqs

    def train(self, texts: Iterable[str] | str) -> BPE:
        if isinstance(texts, str):
            texts = [texts]
        word_freqs = self.count_words(texts)
        return self.train_from_counts(word_freqs)

    def train_from_counts(self, word_freqs: Counter[str]) -> BPE:
        bpe = BPE()
        words: list[list[int]] = [
            [b + BYTE_OFFSET for b in w.encode("utf-8")] for w in word_freqs
        ]
        freqs: list[int] = list(word_freqs.values())

        pair_counts: dict[tuple[int, int], int] = defaultdict(int)
        pair_words: dict[tuple[int, int], set[int]] = defaultdict(set)
        for wi, ids in enumerate(words):
            f = freqs[wi]
            for pair in zip(ids, ids[1:]):
                pair_counts[pair] += f
                pair_words[pair].add(wi)

        heap = [(-c, pair) for pair, c in pair_counts.items()]
        heapq.heapify(heap)

        n_merges = self.vocab_size - BASE_VOCAB_SIZE
        t0 = time.time()
        for i in range(n_merges):
            # Récupère la meilleure paire encore valide.
            pair = None
            while heap:
                neg_count, cand = heapq.heappop(heap)
                if pair_counts.get(cand, 0) == -neg_count:
                    pair, count = cand, -neg_count
                    break
            if pair is None or count < self.min_frequency:
                if self.verbose:
                    print(f"[bpe] arrêt anticipé après {i} fusions (plus de paire fréquente)")
                break

            new_id = bpe.add_merge(pair)
            affected = pair_words.pop(pair)
            changed: set[tuple[int, int]] = set()
            for wi in affected:
                ids = words[wi]
                f = freqs[wi]
                for p in zip(ids, ids[1:]):
                    pair_counts[p] -= f
                    pair_words[p].discard(wi)
                    changed.add(p)
                new_ids = merge_pair(ids, pair, new_id)
                words[wi] = new_ids
                for p in zip(new_ids, new_ids[1:]):
                    pair_counts[p] += f
                    pair_words[p].add(wi)
                    changed.add(p)
            pair_counts.pop(pair, None)
            changed.discard(pair)
            for p in changed:
                c = pair_counts.get(p, 0)
                if c > 0:
                    heapq.heappush(heap, (-c, p))
                else:
                    pair_counts.pop(p, None)
                    pair_words.pop(p, None)

            if self.verbose and (i + 1) % 500 == 0:
                print(
                    f"[bpe] {i + 1}/{n_merges} fusions  "
                    f"dernière={bpe.vocab[new_id]!r} (freq {count})  "
                    f"{time.time() - t0:.1f}s"
                )

        if self.verbose:
            print(f"[bpe] vocabulaire final : {bpe.vocab_size} tokens en {time.time() - t0:.1f}s")
        return bpe
