import tempfile
import unittest

from mini_ai.agents import Critic, Developer, Manager, Researcher
from mini_ai.communication import MessageBus, MessageType
from mini_ai.inference import DummyModel
from mini_ai.memory import MemoryStore
from mini_ai.orchestration import Orchestrator
from mini_ai.tools import default_tools

TASK = "Construis une architecture pour une API Django."


class TestRoles(unittest.TestCase):
    def test_default_roles(self):
        self.assertEqual(Manager().role, "Chef de projet")
        self.assertEqual(Developer().role, "Développeur Python")
        self.assertEqual(Researcher().role, "Chercheur")
        self.assertEqual(Critic().role, "Revieweur de code")
        self.assertEqual(Developer(role="Dev Go", goal="Aller vite").goal, "Aller vite")

    def test_critic_checks(self):
        critic = Critic()
        self.assertIn("réponse vide", critic.check("   "))
        self.assertTrue(any("courte" in i for i in critic.check("ok")))
        self.assertTrue(any("répétitive" in i for i in critic.check("bla bla bla bla bla bla bla bla bla bla")))
        self.assertEqual(critic.check("Une proposition claire avec plusieurs composants distincts et des étapes."), [])
        c = critic.evaluate("")
        self.assertFalse(c.ok)
        self.assertLess(c.score, 1.0)

    def test_researcher_uses_environment(self):
        store = MemoryStore(":memory:")
        store.add_fact("developer", "Django utilise l'architecture MTV.")
        model = DummyModel()
        with tempfile.TemporaryDirectory() as d:
            tools = default_tools(d)
            tools.run("filesystem", "write notes.txt :: Django REST Framework fournit des sérialiseurs.")
            r = Researcher(model=model, store=store, tools=tools)
            answer = r.think("architecture Django")
        self.assertIn("Sources", answer)
        self.assertIn("MTV", answer)
        self.assertIn("Informations trouvées", model.calls[0])

    def test_agent_uses_tool(self):
        with tempfile.TemporaryDirectory() as d:
            dev = Developer(model=DummyModel(), tools=default_tools(d))
            self.assertIn("300", dev.act("[tool:calculator] 15 * 20"))
            self.assertEqual(dev.short_term.items[-1].kind, "tool")


class TestOrchestrator(unittest.TestCase):
    def setUp(self):
        self.model = DummyModel(
            responses={
                r"Propose une solution technique": "Je propose des modèles, des sérialiseurs DRF, des ViewSets, un routeur et une authentification JWT.",
                r"Rassemble les informations": "Django utilise l'architecture MTV et DRF fournit les briques d'une API.",
                r"Analyse cette proposition": "Il manque la pagination et le versionnage.",
                r"Synthétise": "Architecture DRF solide, à compléter par la pagination et le versionnage.",
            }
        )
        self.store = MemoryStore(":memory:")

    def test_full_cycle(self):
        with tempfile.TemporaryDirectory() as d:
            orch = Orchestrator(self.model, store=self.store, tools=default_tools(d))
            result = orch.run(TASK)

        # Délégation, collecte, critique, synthèse.
        self.assertEqual(set(result.responses), {"researcher", "developer"})
        self.assertIn("ViewSets", result.responses["developer"])
        self.assertIsNotNone(result.critique)
        self.assertIn("pagination", result.critique)
        self.assertIn(TASK, result.final)
        self.assertIn("[developer]", result.final)
        self.assertIn("[critique]", result.final)
        self.assertIn("Conclusion", result.final)

        # Le contexte du chercheur a été transmis au développeur.
        dev_task = next(m for m in result.transcript if m.receiver == "developer" and m.type == MessageType.TASK)
        self.assertIn("MTV", dev_task.content)

        # Flux : user → manager → researcher → manager → developer → manager → critic → manager → user
        senders = [m.sender for m in result.transcript]
        self.assertEqual(senders[0], "user")
        self.assertEqual(result.transcript[-1].type, MessageType.FINAL)
        self.assertEqual(result.transcript[-1].receiver, "user")
        self.assertEqual(len(result.transcript), 8)

        # Mémoire longue alimentée et journal persistant.
        self.assertTrue(self.store.list_facts("developer"))
        self.assertTrue(self.store.list_facts("critic"))
        self.assertTrue(self.store.list_facts("manager"))
        self.assertGreaterEqual(len(self.store.get_messages()), 7)

    def test_without_critic_and_custom_agents(self):
        bus = MessageBus()
        agents = [Manager(model=self.model, bus=bus), Developer(model=self.model, bus=bus)]
        orch = Orchestrator(self.model, bus=bus, agents=agents, tools=None)
        result = orch.run(TASK)
        self.assertIsNone(result.critique)
        self.assertEqual(list(result.responses), ["developer"])

    def test_two_consecutive_tasks_share_memory(self):
        with tempfile.TemporaryDirectory() as d:
            orch = Orchestrator(self.model, store=self.store, tools=default_tools(d))
            orch.run(TASK)
            second = orch.run("Ajoute l'authentification à l'API Django.")
        self.assertIn("[developer]", second.final)
        # Le chercheur retrouve les faits mémorisés lors de la première tâche.
        self.assertIn("Sources", second.responses["researcher"])


if __name__ == "__main__":
    unittest.main()
