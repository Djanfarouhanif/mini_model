"""Orchestrateur : la boucle multi-agent du PRD (§16, §18).

    Task → Manager analyses → Delegate → Agents execute → Collect
         → Critic evaluates → Manager synthesizes → Final response

Tous les agents partagent le même modèle, le même bus et le même store.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

from ..agents import Agent, Critic, Developer, Manager, Researcher
from ..communication import Message, MessageBus, MessageType
from ..inference.dummy import LanguageModel
from ..memory import MemoryStore
from ..tools import ToolRegistry, default_tools

USER = "user"


@dataclass
class OrchestrationResult:
    task: str
    final: str
    responses: dict[str, str] = field(default_factory=dict)
    critique: str | None = None
    transcript: list[Message] = field(default_factory=list)
    duration: float = 0.0

    def render_transcript(self) -> str:
        return "\n".join(str(m) for m in self.transcript)


class Orchestrator:
    def __init__(
        self,
        model: LanguageModel,
        store: MemoryStore | None = None,
        tools: ToolRegistry | None = None,
        bus: MessageBus | None = None,
        agents: list[Agent] | None = None,
        verbose: bool = False,
        on_message: Callable[[Message], None] | None = None,
        max_rounds: int = 1,
        agent_kwargs: dict | None = None,
    ):
        self.model = model
        self.store = store
        self.tools = tools if tools is not None else default_tools()
        self.bus = bus or MessageBus()
        self.verbose = verbose
        self.on_message = on_message
        self.max_rounds = max_rounds
        self.bus.register(USER)

        kw = dict(model=model, bus=self.bus, store=store, tools=self.tools)
        kw.update(agent_kwargs or {})
        if agents is None:
            agents = [Manager(**kw), Researcher(**kw), Developer(**kw), Critic(**kw)]
        self.agents: dict[str, Agent] = {}
        for a in agents:
            if a.bus is not self.bus:
                a.attach(self.bus)
            self.agents[a.name] = a

        managers = [a for a in self.agents.values() if isinstance(a, Manager)]
        if not managers:
            raise ValueError("l'orchestrateur a besoin d'un Manager")
        self.manager: Manager = managers[0]
        self.critic: Critic | None = next((a for a in self.agents.values() if isinstance(a, Critic)), None)

        if verbose or on_message:
            for name in self.bus.agents:
                self.bus.subscribe(name, self._trace)

    # --------------------------------------------------------------- utils
    def _trace(self, message: Message) -> None:
        if self.on_message:
            self.on_message(message)
        if self.verbose:
            print(f"  {message}")

    def _run(self, name: str) -> list[Message]:
        """Fait traiter à l'agent ``name`` tous ses messages en attente."""
        return self.agents[name].step()

    # ----------------------------------------------------------------- run
    def run(self, task: str) -> OrchestrationResult:
        t0 = time.time()
        start = len(self.bus)
        manager = self.manager
        manager.responses.clear()

        # 1. USER → MANAGER : le manager retire la tâche de sa file et la mémorise ;
        #    c'est l'orchestrateur qui pilote ensuite le cycle (pas handle()).
        incoming = self.bus.send(USER, manager.name, task, type=MessageType.TASK)
        manager.receive()
        manager.short_term.add(f"{USER}→{manager.name}", task)
        if self.store is not None:
            self.store.log_message(incoming.to_dict())

        # 2. Le manager analyse et délègue.
        workers = [n for n in self.agents if n not in (manager.name, getattr(self.critic, "name", None))]
        plan = manager.analyse(task, workers)
        context = ""
        for receiver, instruction in plan.steps:
            manager.delegate(receiver, instruction, context=context, task=task)
            self._run(receiver)
            replies = manager.collect()
            # Le résultat du researcher sert de contexte au developer.
            if receiver == "researcher" and replies:
                context = replies[-1].content

        responses = {n: m.content for n, m in manager.responses.items() if m.type == MessageType.RESPONSE}

        # 3. Le critique évalue la proposition principale.
        critique_msg: Message | None = None
        if self.critic is not None:
            proposal = responses.get("developer") or (next(iter(responses.values())) if responses else "")
            manager.request_review(self.critic.name, proposal, task)
            self._run(self.critic.name)
            for m in manager.collect():
                if m.type == MessageType.CRITIQUE:
                    critique_msg = m

        # 4. Synthèse finale → USER
        final = manager.synthesize(task, manager.responses, critique_msg)
        manager.send(USER, final, type=MessageType.FINAL)
        manager.remember(f"Tâche traitée : {task[:100]}", tags=["task"])
        self.bus.receive(USER)  # vide la file utilisateur

        return OrchestrationResult(
            task=task,
            final=final,
            responses=responses,
            critique=critique_msg.content if critique_msg else None,
            transcript=self.bus.history[start:],
            duration=time.time() - t0,
        )
