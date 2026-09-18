"""Format de message standard échangé entre agents.

    {
      "sender": "manager",
      "receiver": "developer",
      "type": "task",
      "content": "Propose une architecture Django",
      "timestamp": "2026-09-18T10:00:00+00:00"
    }
"""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any


class MessageType:
    TASK = "task"
    RESPONSE = "response"
    INFO = "info"
    CRITIQUE = "critique"
    TOOL = "tool"
    FINAL = "final"
    ERROR = "error"

    ALL = (TASK, RESPONSE, INFO, CRITIQUE, TOOL, FINAL, ERROR)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class Message:
    sender: str
    receiver: str
    content: str
    type: str = MessageType.TASK
    timestamp: str = field(default_factory=now_iso)
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    reply_to: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.type not in MessageType.ALL:
            raise ValueError(f"type de message inconnu : {self.type!r}")

    # ------------------------------------------------------------- helpers
    def reply(self, content: str, type: str = MessageType.RESPONSE, **metadata: Any) -> "Message":
        return Message(
            sender=self.receiver,
            receiver=self.sender,
            content=content,
            type=type,
            reply_to=self.id,
            metadata=metadata,
        )

    @property
    def is_broadcast(self) -> bool:
        return self.receiver == "*"

    # ------------------------------------------------------- serialisation
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Message":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def from_json(cls, raw: str) -> "Message":
        return cls.from_dict(json.loads(raw))

    def __str__(self) -> str:
        return f"[{self.type}] {self.sender} → {self.receiver}: {self.content}"
