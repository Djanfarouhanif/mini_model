"""Abstraction d'outil.

    class Tool:
        name
        description
        def execute(self, input) -> str

Un agent invoque un outil quand un message contient la syntaxe :

    [tool:calculator] 15 * 20
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass

TOOL_CALL_RE = re.compile(r"\[tool:([a-zA-Z_][\w-]*)\]\s*(.*)", re.DOTALL)


@dataclass
class ToolCall:
    name: str
    input: str


@dataclass
class ToolResult:
    tool: str
    input: str
    output: str
    ok: bool = True

    def __str__(self) -> str:
        status = "ok" if self.ok else "erreur"
        return f"[{self.tool}:{status}] {self.output}"


def parse_tool_call(text: str) -> ToolCall | None:
    m = TOOL_CALL_RE.search(text)
    if not m:
        return None
    return ToolCall(name=m.group(1).lower(), input=m.group(2).strip())


class Tool(ABC):
    name: str = "tool"
    description: str = ""

    @abstractmethod
    def execute(self, input: str) -> str: ...

    def run(self, input: str) -> ToolResult:
        try:
            return ToolResult(self.name, input, self.execute(input), ok=True)
        except Exception as exc:  # noqa: BLE001 — l'erreur est renvoyée à l'agent
            return ToolResult(self.name, input, f"{type(exc).__name__}: {exc}", ok=False)

    def __call__(self, input: str) -> str:
        return self.execute(input)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r})"


class ToolRegistry:
    def __init__(self, tools: list[Tool] | None = None):
        self._tools: dict[str, Tool] = {}
        for t in tools or []:
            self.register(t)

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name.lower())

    @property
    def names(self) -> list[str]:
        return list(self._tools)

    def describe(self) -> str:
        return "\n".join(f"- {t.name}: {t.description}" for t in self._tools.values())

    def run(self, name: str, input: str) -> ToolResult:
        tool = self.get(name)
        if tool is None:
            return ToolResult(name, input, f"outil inconnu : {name!r} (disponibles : {', '.join(self.names)})", ok=False)
        return tool.run(input)

    def run_call(self, call: ToolCall) -> ToolResult:
        return self.run(call.name, call.input)

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def __len__(self) -> int:
        return len(self._tools)

    def __iter__(self):
        return iter(self._tools.values())
