"""Researcher : cherche et synthétise les informations disponibles dans son
environnement — mémoire partagée (SQLite) et fichiers du dossier de travail
(outil filesystem) — avant de solliciter le modèle."""

from __future__ import annotations

import re

from .base import Agent


class Researcher(Agent):
    default_role = "Chercheur"
    default_goal = "Rassembler et synthétiser les informations utiles à la tâche."
    instruction = "Résume les informations pertinentes trouvées et ce qu'il faut retenir."

    def __init__(self, name: str = "researcher", max_sources: int = 5, **kwargs):
        super().__init__(name, **kwargs)
        self.max_sources = max_sources

    # ---------------------------------------------------------- environment
    def search_environment(self, query: str) -> list[str]:
        """Faits en mémoire longue + lignes de fichiers contenant les mots-clés."""
        findings = list(self.recall(query, limit=self.max_sources))
        if "filesystem" in self.tools:
            keywords = [w for w in re.findall(r"\w{4,}", query.lower())][:3]
            for kw in keywords:
                result = self.tools.run("filesystem", f"search {kw}")
                if result.ok and not result.output.startswith("("):
                    findings.extend(result.output.splitlines()[:3])
        # Dédoublonnage en gardant l'ordre.
        seen: set[str] = set()
        return [f for f in findings if not (f in seen or seen.add(f))][: self.max_sources]

    def build_prompt(self, task: str) -> str:
        parts = [self.identity, self.instruction]
        sources = self.search_environment(task)
        if sources:
            parts.append("Informations trouvées :\n" + "\n".join(f"- {s}" for s in sources))
        history = self.short_term.render(max_chars=300, n=4)
        if history:
            parts.append("Historique récent :\n" + history)
        parts.append(f"Tâche : {task}\nRéponse :")
        return "\n\n".join(parts)

    def think(self, task: str) -> str:
        sources = self.search_environment(task)
        answer = super().think(task)
        if sources:
            answer = "Sources : " + " | ".join(s[:80] for s in sources) + "\n" + answer
        self.remember(f"Recherche sur « {task[:80]} » : {answer[:200]}", tags=["research"])
        return answer
