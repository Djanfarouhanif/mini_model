"""Mémoire longue : faits persistants d'un agent, adossés au MemoryStore.

    memory.remember("Django utilise l'architecture MTV.")
    memory.recall("architecture Django")  → [faits pertinents]
"""

from __future__ import annotations

from typing import Any

from .store import MemoryStore


class LongTermMemory:
    def __init__(self, store: MemoryStore, agent_name: str, shared: bool = True):
        self.store = store
        self.agent_name = agent_name
        self.shared = shared  # consulter aussi les faits des autres agents

    def remember(self, fact: str, tags: list[str] | None = None) -> int:
        return self.store.add_fact(self.agent_name, fact, tags)

    def recall(self, query: str, limit: int = 5) -> list[dict[str, Any]]:
        return self.store.search_facts(query, agent=self.agent_name, limit=limit, shared=self.shared)

    def recall_text(self, query: str, limit: int = 5) -> list[str]:
        return [f["content"] for f in self.recall(query, limit)]

    def all(self, limit: int | None = None) -> list[dict[str, Any]]:
        return self.store.list_facts(self.agent_name, limit)

    def forget_all(self) -> None:
        self.store.delete_facts(self.agent_name)

    def render(self, query: str, limit: int = 5) -> str:
        facts = self.recall_text(query, limit)
        return "\n".join(f"- {f}" for f in facts)

    def __len__(self) -> int:
        return len(self.all())
