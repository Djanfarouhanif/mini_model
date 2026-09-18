"""Agent : une abstraction au-dessus du modèle de langage partagé.

    class Agent:
        name, role, goal      identité
        memory                mémoire courte + longue
        model                 modèle de langage partagé
        tools                 outils disponibles

    developer = Agent(name="developer", role="Développeur Python",
                      goal="Résoudre les problèmes techniques")

Cycle de vie d'un message :

    receive()  →  handle(message)  →  act(content)  →  think() | tool  →  send(reply)
"""

from __future__ import annotations

import re
from typing import Any

from ..communication import Message, MessageBus, MessageType
from ..inference.dummy import LanguageModel
from ..memory import LongTermMemory, MemoryStore, ShortTermMemory
from ..tools import ToolRegistry, ToolResult, parse_tool_call


class Agent:
    default_role = "Assistant"
    default_goal = "Aider au mieux à accomplir la tâche."
    # Consigne courte insérée en tête de prompt, spécialisée par sous-classe.
    instruction = "Réponds de façon claire et concise."

    def __init__(
        self,
        name: str,
        role: str | None = None,
        goal: str | None = None,
        model: LanguageModel | None = None,
        bus: MessageBus | None = None,
        store: MemoryStore | None = None,
        tools: ToolRegistry | None = None,
        max_tokens: int = 80,
        temperature: float = 0.8,
        top_k: int | None = 40,
        top_p: float | None = 0.9,
        short_term_size: int = 20,
    ):
        self.name = name
        self.role = role or self.default_role
        self.goal = goal or self.default_goal
        self.model = model
        self.tools = tools or ToolRegistry()
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p

        self.short_term = ShortTermMemory(max_items=short_term_size)
        self.store = store
        self.long_term = LongTermMemory(store, name) if store is not None else None

        self.bus: MessageBus | None = None
        if bus is not None:
            self.attach(bus)

    # ----------------------------------------------------------- identity
    @property
    def identity(self) -> str:
        return f"Nom : {self.name}\nRôle : {self.role}\nObjectif : {self.goal}"

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r}, role={self.role!r})"

    # -------------------------------------------------------------- memory
    def remember(self, fact: str, tags: list[str] | None = None) -> None:
        """Stocke une information durable (mémoire longue, si un store existe)."""
        if self.long_term is not None:
            self.long_term.remember(fact, tags)

    def recall(self, query: str, limit: int = 3) -> list[str]:
        if self.long_term is None:
            return []
        return self.long_term.recall_text(query, limit)

    # -------------------------------------------------------------- prompt
    def build_prompt(self, task: str) -> str:
        """Assemble : identité + faits rappelés + historique récent + tâche."""
        parts = [self.identity, self.instruction]
        facts = self.recall(task)
        if facts:
            parts.append("Ce que je sais :\n" + "\n".join(f"- {f}" for f in facts))
        history = self.short_term.render(max_chars=400, n=6)
        if history:
            parts.append("Historique récent :\n" + history)
        parts.append(f"Tâche : {task}\nRéponse :")
        return "\n\n".join(parts)

    def postprocess(self, text: str) -> str:
        """Nettoyage minimal de la sortie brute du modèle."""
        text = text.strip()
        # Un petit modèle continue souvent après la réponse : on coupe au
        # premier marqueur de "nouveau tour".
        cut = re.split(r"\n(?:Tâche|Nom|Rôle|Réponse)\s*:", text, maxsplit=1)[0]
        return cut.strip() or text

    # --------------------------------------------------------------- think
    def think(self, task: str) -> str:
        """Interroge le modèle partagé et mémorise l'échange."""
        if self.model is None:
            raise RuntimeError(f"l'agent {self.name!r} n'a pas de modèle")
        prompt = self.build_prompt(task)
        raw = self.model.generate(
            prompt,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_k=self.top_k,
            top_p=self.top_p,
        )
        answer = self.postprocess(raw)
        self.short_term.add("task", task, kind="message")
        self.short_term.add(self.name, answer, kind="thought")
        return answer

    def use_tool(self, name: str, input: str) -> ToolResult:
        result = self.tools.run(name, input)
        self.short_term.add(f"tool:{name}", f"{input} → {result.output}", kind="tool")
        return result

    def act(self, content: str) -> str:
        """Point d'entrée : outil si la syntaxe [tool:x] est présente, sinon modèle."""
        call = parse_tool_call(content)
        if call is not None and call.name in self.tools:
            return str(self.use_tool(call.name, call.input))
        return self.think(content)

    # ------------------------------------------------------- communication
    def attach(self, bus: MessageBus) -> None:
        self.bus = bus
        bus.register(self.name)

    def send(
        self,
        receiver: str,
        content: str,
        type: str = MessageType.TASK,
        reply_to: str | None = None,
        **metadata: Any,
    ) -> Message:
        if self.bus is None:
            raise RuntimeError(f"l'agent {self.name!r} n'est relié à aucun bus")
        message = self.bus.send(self.name, receiver, content, type=type, reply_to=reply_to, **metadata)
        self.short_term.add(f"{self.name}→{receiver}", content)
        if self.store is not None:
            self.store.log_message(message.to_dict())
        return message

    def receive(self, timeout: float | None = 0) -> Message | None:
        if self.bus is None:
            raise RuntimeError(f"l'agent {self.name!r} n'est relié à aucun bus")
        return self.bus.receive(self.name, timeout=timeout)

    def handle(self, message: Message) -> Message | None:
        """Traite un message entrant et renvoie la réponse envoyée (ou None)."""
        self.short_term.add(f"{message.sender}→{self.name}", message.content)
        if self.store is not None:
            self.store.log_message(message.to_dict())
        if message.type in (MessageType.RESPONSE, MessageType.INFO, MessageType.FINAL, MessageType.CRITIQUE):
            return None  # simple accusé : on mémorise, pas de réponse automatique
        answer = self.act(message.content)
        return self.send(message.sender, answer, type=MessageType.RESPONSE, reply_to=message.id)

    def step(self) -> list[Message]:
        """Traite tous les messages en attente ; retourne les réponses émises."""
        replies: list[Message] = []
        while True:
            message = self.receive()
            if message is None:
                return replies
            reply = self.handle(message)
            if reply is not None:
                replies.append(reply)

    def ask(self, receiver: str, content: str, other: "Agent") -> Message | None:
        """Raccourci synchrone : envoie à ``other`` et récupère sa réponse.

        Sert au premier test du PRD : Agent A → Agent B → Agent A.
        """
        self.send(receiver, content)
        other.step()
        reply = self.receive()
        if reply is not None:
            self.handle(reply)
        return reply
