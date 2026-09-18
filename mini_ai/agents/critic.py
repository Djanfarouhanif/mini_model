"""Critic : analyse une réponse et identifie les problèmes.

En plus de l'avis du modèle, le critique applique quelques vérifications
déterministes (réponse vide, trop courte, répétitive, appel d'outil en échec)
afin d'apporter un signal fiable même avec un modèle de 1M paramètres.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..communication import Message, MessageType
from .base import Agent


@dataclass
class Critique:
    issues: list[str] = field(default_factory=list)
    comment: str = ""
    score: float = 1.0  # 1.0 = aucun problème détecté

    @property
    def ok(self) -> bool:
        return not self.issues

    def render(self) -> str:
        lines = [f"Score : {self.score:.2f}"]
        if self.issues:
            lines.append("Problèmes :")
            lines.extend(f"- {i}" for i in self.issues)
        else:
            lines.append("Aucun problème bloquant détecté.")
        if self.comment:
            lines.append("Avis : " + self.comment)
        return "\n".join(lines)


class Critic(Agent):
    default_role = "Revieweur de code"
    default_goal = "Analyser les réponses et identifier les problèmes."
    instruction = "Identifie les faiblesses, risques et manques de la proposition, puis suggère une amélioration."

    def __init__(self, name: str = "critic", min_length: int = 20, **kwargs):
        super().__init__(name, **kwargs)
        self.min_length = min_length

    # ------------------------------------------------------- deterministic
    def check(self, text: str) -> list[str]:
        issues: list[str] = []
        stripped = text.strip()
        if not stripped:
            issues.append("réponse vide")
            return issues
        if len(stripped) < self.min_length:
            issues.append(f"réponse très courte ({len(stripped)} caractères)")
        words = re.findall(r"\w+", stripped.lower())
        if len(words) >= 8 and len(set(words)) / len(words) < 0.4:
            issues.append("réponse répétitive (peu de mots distincts)")
        if re.search(r"\[\w+:erreur\]", stripped):
            issues.append("un appel d'outil a échoué")
        if "�" in stripped:
            issues.append("caractères invalides dans la sortie")
        return issues

    def evaluate(self, proposal: str, context: str = "") -> Critique:
        issues = self.check(proposal)
        task = f"Analyse cette proposition{(' pour : ' + context) if context else ''}\n{proposal}"
        comment = self.think(task) if self.model is not None else ""
        score = max(0.0, 1.0 - 0.3 * len(issues))
        critique = Critique(issues=issues, comment=comment, score=score)
        self.remember(f"Critique ({score:.2f}) de « {proposal[:60]} » : {'; '.join(issues) or 'ok'}", tags=["critique"])
        return critique

    # ---------------------------------------------------------- messaging
    def act(self, content: str) -> str:
        return self.evaluate(content).render()

    def handle(self, message: Message) -> Message | None:
        self.short_term.add(f"{message.sender}→{self.name}", message.content)
        if self.store is not None:
            self.store.log_message(message.to_dict())
        if message.type != MessageType.TASK:
            return None
        critique = self.evaluate(message.content, context=message.metadata.get("task", ""))
        return self.send(
            message.sender,
            critique.render(),
            type=MessageType.CRITIQUE,
            reply_to=message.id,
            score=critique.score,
            issues=critique.issues,
        )
