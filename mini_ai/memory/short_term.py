"""Mémoire courte : l'historique récent de la conversation d'un agent."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass
class MemoryItem:
    speaker: str
    content: str
    kind: str = "message"  # message | thought | tool | result
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds"))


class ShortTermMemory:
    def __init__(self, max_items: int = 20):
        self.max_items = max_items
        self._items: deque[MemoryItem] = deque(maxlen=max_items)

    def add(self, speaker: str, content: str, kind: str = "message") -> MemoryItem:
        item = MemoryItem(speaker=speaker, content=content.strip(), kind=kind)
        self._items.append(item)
        return item

    def last(self, n: int = 5) -> list[MemoryItem]:
        return list(self._items)[-n:]

    @property
    def items(self) -> list[MemoryItem]:
        return list(self._items)

    def render(self, max_chars: int = 600, n: int | None = None) -> str:
        """Texte compact injectable dans un prompt (les plus récents en dernier)."""
        items = self.items if n is None else self.last(n)
        lines = [f"{it.speaker}: {it.content}" for it in items]
        text = "\n".join(lines)
        if len(text) > max_chars:
            text = text[-max_chars:]
            text = text[text.find("\n") + 1 :] if "\n" in text else text
        return text

    def clear(self) -> None:
        self._items.clear()

    def __len__(self) -> int:
        return len(self._items)

    def __iter__(self):
        return iter(self._items)
