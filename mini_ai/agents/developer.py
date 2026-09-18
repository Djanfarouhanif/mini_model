"""Developer : propose des solutions techniques."""

from __future__ import annotations

from .base import Agent


class Developer(Agent):
    default_role = "Développeur Python"
    default_goal = "Résoudre les problèmes techniques en proposant une solution concrète."
    instruction = "Propose une solution technique structurée (architecture, composants, étapes)."

    def __init__(self, name: str = "developer", **kwargs):
        super().__init__(name, **kwargs)

    def think(self, task: str) -> str:
        answer = super().think(task)
        self.remember(f"Proposition pour « {task[:80]} » : {answer[:200]}", tags=["proposal"])
        return answer
