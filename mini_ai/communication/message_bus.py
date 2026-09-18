"""Bus de messages in-memory (première version, cf. PRD §13).

    message_bus.send(sender="manager", receiver="developer", content="...")
    message = message_bus.receive("developer")

Une file par destinataire, un historique global, et un verrou pour rester
utilisable depuis plusieurs threads. Le remplacement par Redis se fera en
gardant cette interface.
"""

from __future__ import annotations

import queue
import threading
from typing import Callable, Iterable

from .message import Message, MessageType

Subscriber = Callable[[Message], None]


class MessageBus:
    BROADCAST = "*"

    def __init__(self, keep_history: bool = True):
        self._queues: dict[str, queue.Queue[Message]] = {}
        self._subscribers: dict[str, list[Subscriber]] = {}
        self._history: list[Message] = []
        self._keep_history = keep_history
        self._lock = threading.RLock()

    # ---------------------------------------------------------- registry
    def register(self, name: str) -> None:
        with self._lock:
            self._queues.setdefault(name, queue.Queue())
            self._subscribers.setdefault(name, [])

    def unregister(self, name: str) -> None:
        with self._lock:
            self._queues.pop(name, None)
            self._subscribers.pop(name, None)

    def is_registered(self, name: str) -> bool:
        return name in self._queues

    @property
    def agents(self) -> list[str]:
        return list(self._queues)

    def subscribe(self, name: str, callback: Subscriber) -> None:
        """Appelle ``callback`` dès qu'un message arrive pour ``name``."""
        self.register(name)
        self._subscribers[name].append(callback)

    # ------------------------------------------------------------ sending
    def send(
        self,
        sender: str,
        receiver: str,
        content: str,
        type: str = MessageType.TASK,
        reply_to: str | None = None,
        **metadata,
    ) -> Message:
        message = Message(sender=sender, receiver=receiver, content=content, type=type, reply_to=reply_to, metadata=metadata)
        return self.deliver(message)

    def deliver(self, message: Message) -> Message:
        with self._lock:
            if self._keep_history:
                self._history.append(message)
            targets: Iterable[str]
            if message.is_broadcast:
                targets = [n for n in self._queues if n != message.sender]
            else:
                if message.receiver not in self._queues:
                    raise KeyError(f"destinataire inconnu : {message.receiver!r} (agents : {self.agents})")
                targets = [message.receiver]
            for name in targets:
                self._queues[name].put(message)
                for cb in self._subscribers.get(name, []):
                    cb(message)
        return message

    def broadcast(self, sender: str, content: str, type: str = MessageType.INFO, **metadata) -> Message:
        return self.send(sender, self.BROADCAST, content, type=type, **metadata)

    # ---------------------------------------------------------- receiving
    def receive(self, name: str, timeout: float | None = 0) -> Message | None:
        """Retourne le prochain message de ``name`` ou None si la file est vide.

        ``timeout=None`` bloque jusqu'à réception (usage multi-thread).
        """
        if name not in self._queues:
            raise KeyError(f"agent non enregistré : {name!r}")
        q = self._queues[name]
        try:
            if timeout is None:
                return q.get(block=True)
            if timeout <= 0:
                return q.get_nowait()
            return q.get(block=True, timeout=timeout)
        except queue.Empty:
            return None

    def receive_all(self, name: str) -> list[Message]:
        out: list[Message] = []
        while True:
            m = self.receive(name)
            if m is None:
                return out
            out.append(m)

    def pending(self, name: str) -> int:
        return self._queues[name].qsize() if name in self._queues else 0

    # ------------------------------------------------------------ history
    @property
    def history(self) -> list[Message]:
        return list(self._history)

    def history_for(self, name: str) -> list[Message]:
        return [m for m in self._history if name in (m.sender, m.receiver) or m.is_broadcast]

    def conversation(self, a: str, b: str) -> list[Message]:
        return [m for m in self._history if {m.sender, m.receiver} == {a, b}]

    def clear(self) -> None:
        with self._lock:
            self._history.clear()
            for q in self._queues.values():
                while not q.empty():
                    q.get_nowait()

    def __len__(self) -> int:
        return len(self._history)
