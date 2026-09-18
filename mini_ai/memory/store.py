"""Stockage persistant SQLite partagé par tous les agents.

Deux tables :
  * facts    — mémoire long terme ("Django utilise l'architecture MTV.")
  * messages — journal des messages échangés (historique persistant)
"""

from __future__ import annotations

import re
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS facts (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    agent      TEXT NOT NULL,
    content    TEXT NOT NULL,
    tags       TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    UNIQUE(agent, content)
);
CREATE INDEX IF NOT EXISTS idx_facts_agent ON facts(agent);

CREATE TABLE IF NOT EXISTS messages (
    id        TEXT PRIMARY KEY,
    sender    TEXT NOT NULL,
    receiver  TEXT NOT NULL,
    type      TEXT NOT NULL,
    content   TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    reply_to  TEXT
);
CREATE INDEX IF NOT EXISTS idx_messages_ts ON messages(timestamp);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _keywords(text: str) -> set[str]:
    return {w for w in re.findall(r"\w+", text.lower()) if len(w) > 2}


class MemoryStore:
    def __init__(self, path: str | Path = ":memory:"):
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        with self._lock:
            self._conn.executescript(SCHEMA)

    # --------------------------------------------------------------- facts
    def add_fact(self, agent: str, content: str, tags: list[str] | None = None) -> int:
        content = content.strip()
        with self._lock:
            cur = self._conn.execute(
                "INSERT OR IGNORE INTO facts(agent, content, tags, created_at) VALUES (?, ?, ?, ?)",
                (agent, content, ",".join(tags or []), _now()),
            )
            self._conn.commit()
            if cur.lastrowid:
                return int(cur.lastrowid)
            row = self._conn.execute("SELECT id FROM facts WHERE agent=? AND content=?", (agent, content)).fetchone()
            return int(row["id"])

    def list_facts(self, agent: str | None = None, limit: int | None = None) -> list[dict[str, Any]]:
        sql = "SELECT * FROM facts"
        params: tuple = ()
        if agent is not None:
            sql += " WHERE agent=?"
            params = (agent,)
        sql += " ORDER BY id DESC"
        if limit:
            sql += f" LIMIT {int(limit)}"
        with self._lock:
            return [dict(r) for r in self._conn.execute(sql, params).fetchall()]

    def search_facts(self, query: str, agent: str | None = None, limit: int = 5, shared: bool = True) -> list[dict[str, Any]]:
        """Recherche par recouvrement de mots-clés (score = mots communs).

        ``shared=True`` inclut les faits des autres agents (mémoire partagée).
        """
        q = _keywords(query)
        if not q:
            return []
        rows = self.list_facts(None if shared else agent)
        scored: list[tuple[int, dict[str, Any]]] = []
        for row in rows:
            score = len(q & _keywords(row["content"] + " " + row["tags"]))
            if score:
                if agent is not None and row["agent"] == agent:
                    score += 1  # léger bonus pour ses propres souvenirs
                scored.append((score, row))
        scored.sort(key=lambda s: (-s[0], -s[1]["id"]))
        return [r for _, r in scored[:limit]]

    def delete_facts(self, agent: str | None = None) -> None:
        with self._lock:
            if agent is None:
                self._conn.execute("DELETE FROM facts")
            else:
                self._conn.execute("DELETE FROM facts WHERE agent=?", (agent,))
            self._conn.commit()

    # ------------------------------------------------------------ messages
    def log_message(self, message: dict[str, Any]) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO messages(id, sender, receiver, type, content, timestamp, reply_to) VALUES (?,?,?,?,?,?,?)",
                (
                    message["id"],
                    message["sender"],
                    message["receiver"],
                    message["type"],
                    message["content"],
                    message["timestamp"],
                    message.get("reply_to"),
                ),
            )
            self._conn.commit()

    def get_messages(self, agent: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
        sql = "SELECT * FROM messages"
        params: tuple = ()
        if agent is not None:
            sql += " WHERE sender=? OR receiver=?"
            params = (agent, agent)
        sql += f" ORDER BY timestamp DESC, rowid DESC LIMIT {int(limit)}"
        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()
        return [dict(r) for r in reversed(rows)]

    # --------------------------------------------------------------- misc
    def close(self) -> None:
        with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:  # noqa: BLE001 — pas d'erreur à la destruction
            pass

    def __enter__(self) -> "MemoryStore":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
