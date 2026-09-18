import unittest

from mini_ai.agents import Agent
from mini_ai.communication import Message, MessageBus, MessageType
from mini_ai.inference import DummyModel


class TestMessage(unittest.TestCase):
    def test_serialisation(self):
        m = Message(sender="manager", receiver="developer", content="Propose une architecture Django")
        d = m.to_dict()
        for key in ("sender", "receiver", "type", "content", "timestamp"):
            self.assertIn(key, d)
        self.assertEqual(Message.from_json(m.to_json()), m)

    def test_reply(self):
        m = Message(sender="a", receiver="b", content="?")
        r = m.reply("!")
        self.assertEqual((r.sender, r.receiver, r.type, r.reply_to), ("b", "a", MessageType.RESPONSE, m.id))

    def test_invalid_type(self):
        with self.assertRaises(ValueError):
            Message(sender="a", receiver="b", content="", type="nope")


class TestMessageBus(unittest.TestCase):
    def test_send_receive(self):
        bus = MessageBus()
        bus.register("manager")
        bus.register("developer")
        bus.send(sender="manager", receiver="developer", content="Analyse cette tâche")
        m = bus.receive("developer")
        self.assertEqual(m.content, "Analyse cette tâche")
        self.assertIsNone(bus.receive("developer"))
        self.assertEqual(len(bus.history), 1)

    def test_unknown_receiver(self):
        bus = MessageBus()
        bus.register("a")
        with self.assertRaises(KeyError):
            bus.send("a", "ghost", "hello")

    def test_broadcast_and_subscribe(self):
        bus = MessageBus()
        for n in ("a", "b", "c"):
            bus.register(n)
        seen = []
        bus.subscribe("c", lambda m: seen.append(m.content))
        bus.broadcast("a", "salut")
        self.assertEqual(bus.pending("a"), 0)
        self.assertEqual(bus.pending("b"), 1)
        self.assertEqual(seen, ["salut"])
        self.assertEqual(len(bus.history_for("b")), 1)


class TestAgentCommunication(unittest.TestCase):
    def test_agent_a_to_b_to_a(self):
        """Premier test du PRD (phase 6) : Agent A → Agent B → Agent A."""
        model = DummyModel()
        bus = MessageBus()
        a = Agent("agent_a", role="Développeur Python", model=model, bus=bus)
        b = Agent("agent_b", role="Revieweur de code", model=model, bus=bus)

        reply = a.ask("agent_b", "Relis mon code", b)
        self.assertIsNotNone(reply)
        self.assertEqual(reply.sender, "agent_b")
        self.assertEqual(reply.receiver, "agent_a")
        self.assertEqual(reply.type, MessageType.RESPONSE)
        self.assertIn("Relis mon code", reply.content)

        history = bus.conversation("agent_a", "agent_b")
        self.assertEqual([m.sender for m in history], ["agent_a", "agent_b"])
        self.assertEqual(history[1].reply_to, history[0].id)
        # Chaque agent a mémorisé l'échange dans sa mémoire courte.
        self.assertGreaterEqual(len(a.short_term), 2)
        self.assertGreaterEqual(len(b.short_term), 2)
        # Le prompt envoyé au modèle contenait le rôle de B.
        self.assertIn("Revieweur de code", model.calls[0])

    def test_agent_identity(self):
        a = Agent("dev", role="Développeur Python", goal="Résoudre les problèmes techniques")
        self.assertIn("Développeur Python", a.identity)
        self.assertIn("Résoudre les problèmes techniques", a.build_prompt("x"))
        with self.assertRaises(RuntimeError):
            a.think("x")  # pas de modèle
        with self.assertRaises(RuntimeError):
            a.send("x", "y")  # pas de bus


if __name__ == "__main__":
    unittest.main()
