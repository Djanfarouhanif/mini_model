"""Manager : coordonne les autres agents.

    Task → Manager → Delegation → Collect → Critic → Synthesis
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..communication import Message, MessageType
from .base import Agent


@dataclass
class Plan:
    task: str
    steps: list[tuple[str, str]] = field(default_factory=list)  # (agent, instruction)


class Manager(Agent):
    default_role = "Chef de projet"
    default_goal = "Coordonner les agents pour produire la meilleure réponse finale."
    instruction = "Synthétise les contributions en une réponse finale claire."

    def __init__(self, name: str = "manager", **kwargs):
        super().__init__(name, **kwargs)
        self.responses: dict[str, Message] = {}

    # ------------------------------------------------------------ planning
    def analyse(self, task: str, available: list[str]) -> Plan:
        """Découpe la tâche en instructions par agent (déterministe pour le MVP)."""
        plan = Plan(task=task)
        if "researcher" in available:
            plan.steps.append(("researcher", f"Rassemble les informations utiles pour : {task}"))
        if "developer" in available:
            plan.steps.append(("developer", f"Propose une solution technique pour : {task}"))
        # Tout autre agent (hors critique) reçoit la tâche telle quelle.
        for name in available:
            if name not in ("researcher", "developer", "critic", self.name):
                plan.steps.append((name, task))
        return plan

    # ---------------------------------------------------------- delegation
    def delegate(self, receiver: str, instruction: str, context: str = "", **metadata) -> Message:
        content = instruction if not context else f"{instruction}\n\nContexte :\n{context}"
        return self.send(receiver, content, type=MessageType.TASK, **metadata)

    def collect(self) -> list[Message]:
        """Récupère les réponses en attente et les indexe par expéditeur."""
        collected: list[Message] = []
        while True:
            m = self.receive()
            if m is None:
                break
            self.handle(m)
            if m.type in (MessageType.RESPONSE, MessageType.CRITIQUE):
                self.responses[m.sender] = m
                collected.append(m)
        return collected

    def request_review(self, critic: str, proposal: str, task: str) -> Message:
        return self.send(critic, proposal, type=MessageType.TASK, task=task)

    # ----------------------------------------------------------- synthesis
    def synthesize(self, task: str, responses: dict[str, Message], critique: Message | None = None) -> str:
        """Produit la réponse finale à partir des contributions reçues."""
        sections = [f"Tâche : {task}"]
        for name, m in responses.items():
            if m.type == MessageType.CRITIQUE:
                continue
            sections.append(f"[{name}]\n{m.content}")
        if critique is not None:
            sections.append(f"[critique]\n{critique.content}")

        summary = ""
        if self.model is not None:
            digest = "\n".join(f"{n}: {m.content[:150]}" for n, m in responses.items())
            summary = self.think(f"Synthétise en une conclusion : {task}\n{digest}")
        if summary:
            sections.append(f"[{self.name}] Conclusion\n{summary}")
        return "\n\n".join(sections)
