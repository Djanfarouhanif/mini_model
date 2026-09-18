import tempfile
import unittest
from pathlib import Path

from mini_ai.memory import LongTermMemory, MemoryStore, ShortTermMemory
from mini_ai.tools import CalculatorTool, FilesystemTool, PythonTool, ToolRegistry, default_tools, parse_tool_call


class TestShortTermMemory(unittest.TestCase):
    def test_bounded_history(self):
        mem = ShortTermMemory(max_items=3)
        for i in range(5):
            mem.add("a", f"message {i}")
        self.assertEqual(len(mem), 3)
        self.assertEqual([m.content for m in mem.items], ["message 2", "message 3", "message 4"])
        self.assertIn("message 4", mem.render())
        self.assertNotIn("message 1", mem.render())


class TestLongTermMemory(unittest.TestCase):
    def test_remember_recall(self):
        store = MemoryStore(":memory:")
        mem = LongTermMemory(store, "developer")
        mem.remember("Django utilise l'architecture MTV.", tags=["django"])
        mem.remember("Python est un langage interprété.")
        mem.remember("Django utilise l'architecture MTV.")  # doublon ignoré
        self.assertEqual(len(mem), 2)
        found = mem.recall_text("architecture Django")
        self.assertEqual(found[0], "Django utilise l'architecture MTV.")
        self.assertEqual(mem.recall("rien à voir"), [])

    def test_shared_between_agents(self):
        store = MemoryStore(":memory:")
        LongTermMemory(store, "researcher").remember("FastAPI génère la documentation OpenAPI.")
        dev = LongTermMemory(store, "developer")
        self.assertTrue(dev.recall("documentation FastAPI"))
        self.assertFalse(LongTermMemory(store, "developer", shared=False).recall("documentation FastAPI"))

    def test_persistence_on_disk(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "memory.db"
            with MemoryStore(path) as store:
                store.add_fact("manager", "Tâche traitée : API Django")
                store.log_message({"id": "m1", "sender": "a", "receiver": "b", "type": "task", "content": "x", "timestamp": "t"})
            with MemoryStore(path) as store:
                self.assertEqual(len(store.list_facts("manager")), 1)
                self.assertEqual(store.get_messages("a")[0]["id"], "m1")


class TestTools(unittest.TestCase):
    def test_calculator(self):
        calc = CalculatorTool()
        self.assertEqual(calc.execute("15 * 20"), "300")
        self.assertEqual(calc.execute("2 ^ 10"), "1024")
        self.assertEqual(calc.execute("sqrt(16) + 1"), "5")
        self.assertEqual(calc.execute("(1 + 2) * 3.5"), "10.5")
        with self.assertRaises(ValueError):
            calc.execute("__import__('os')")
        self.assertFalse(calc.run("1 /").ok)

    def test_python(self):
        py = PythonTool(timeout=10)
        self.assertEqual(py.execute("print(sum(range(10)))"), "45")
        self.assertIn("ZeroDivisionError", py.execute("1/0"))

    def test_filesystem_sandbox(self):
        with tempfile.TemporaryDirectory() as d:
            fs = FilesystemTool(root=d)
            self.assertIn("écrit", fs.execute("write notes/a.txt :: Django utilise MTV"))
            self.assertEqual(fs.execute("read notes/a.txt"), "Django utilise MTV")
            self.assertIn("notes", fs.execute("list ."))
            self.assertIn("a.txt:1", fs.execute("search mtv"))
            with self.assertRaises(PermissionError):
                fs.execute("read ../../secret.txt")

    def test_registry_and_parsing(self):
        call = parse_tool_call("Calcule ceci : [tool:calculator] 6 * 7")
        self.assertEqual((call.name, call.input), ("calculator", "6 * 7"))
        self.assertIsNone(parse_tool_call("pas d'outil ici"))
        reg = ToolRegistry([CalculatorTool()])
        self.assertEqual(reg.run_call(call).output, "42")
        self.assertFalse(reg.run("inconnu", "x").ok)
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(sorted(default_tools(d).names), ["calculator", "filesystem", "python"])


if __name__ == "__main__":
    unittest.main()
