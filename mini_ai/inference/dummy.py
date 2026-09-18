"""Interface commune des modèles de langage + un modèle factice.

Le ``DummyModel`` ne dépend pas de PyTorch : il permet de tester toute la
couche agents / communication / orchestration sans checkpoint entraîné.
"""

from __future__ import annotations

import random
import re
from typing import Protocol, runtime_checkable


@runtime_checkable
class LanguageModel(Protocol):
    def generate(
        self,
        prompt: str,
        max_tokens: int = 100,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
    ) -> str: ...


class DummyModel:
    """Produit une réponse déterministe à partir des mots du prompt.

    Utile pour valider le fonctionnement structurel du système multi-agent
    (délégation, messages, mémoire, synthèse) indépendamment de la qualité
    du vrai modèle 1M.
    """

    name = "dummy"

    def __init__(self, seed: int = 0, responses: dict[str, str] | None = None):
        self.rng = random.Random(seed)
        # Réponses fixes par mot-clé (regex insensible à la casse) → texte.
        self.responses = responses or {}
        self.calls: list[str] = []

    @staticmethod
    def extract_task(prompt: str) -> str:
        """Retrouve la tâche dans un prompt construit par Agent.build_prompt."""
        m = re.search(r"^Tâche\s*:\s*(.*)$", prompt, re.MULTILINE)
        if m:
            return m.group(1).strip()
        lines = [l.strip() for l in prompt.strip().splitlines() if l.strip()]
        while lines and re.match(r"^(Réponse|Answer)\s*:\s*$", lines[-1], re.IGNORECASE):
            lines.pop()
        return lines[-1] if lines else ""

    def generate(self, prompt: str, max_tokens: int = 100, temperature: float = 1.0, top_k=None, top_p=None) -> str:
        self.calls.append(prompt)
        task = self.extract_task(prompt)
        # Les patterns sont testés sur la tâche seule : le reste du prompt
        # (mémoire, historique) contient souvent les instructions d'autres agents.
        for pattern, answer in self.responses.items():
            if re.search(pattern, task, re.IGNORECASE):
                return answer
        words = re.findall(r"\w+", task)
        picked = words[: max(3, max_tokens // 10)]
        return "Réponse générée à propos de : " + " ".join(picked) + "."
