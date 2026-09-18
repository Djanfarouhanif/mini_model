import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from mini_ai.dashboard import read_log
from mini_ai.model import GPT, GPTConfig, count_parameters
from mini_ai.model.config import PRESETS
from mini_ai.tokenizer import Tokenizer
from mini_ai.training import Probe, TokenDataset, TrainConfig, Trainer, load_model, save_checkpoint
from mini_ai.inference import TextGenerator

TINY = PRESETS["tiny"]


class TestGPT(unittest.TestCase):
    def test_parameter_count_matches_estimate(self):
        for name, cfg in PRESETS.items():
            model = GPT(cfg)
            self.assertEqual(count_parameters(model), cfg.estimated_parameters(), name)

    def test_default_preset_is_about_one_million(self):
        n = GPT(PRESETS["mini"]).num_parameters()
        self.assertTrue(900_000 <= n <= 1_200_000, n)

    def test_forward_and_loss(self):
        model = GPT(TINY)
        x = torch.randint(0, TINY.vocab_size, (2, 16))
        logits, loss = model(x, x)
        self.assertEqual(logits.shape, (2, 16, TINY.vocab_size))
        self.assertIsNotNone(loss)
        self.assertTrue(torch.isfinite(loss))
        logits, loss = model(x)
        self.assertEqual(logits.shape, (2, 1, TINY.vocab_size))
        self.assertIsNone(loss)

    def test_rope_variant(self):
        cfg = GPTConfig(vocab_size=128, block_size=32, n_layer=1, n_head=2, n_embd=16, pos_type="rope")
        model = GPT(cfg)
        x = torch.randint(0, 128, (1, 10))
        self.assertEqual(model(x)[0].shape, (1, 1, 128))

    def test_causality(self):
        """Changer un token futur ne modifie pas les logits des positions passées."""
        model = GPT(TINY).eval()
        x = torch.randint(0, TINY.vocab_size, (1, 12))
        y = x.clone()
        y[0, -1] = (y[0, -1] + 1) % TINY.vocab_size
        with torch.no_grad():
            lx, _ = model(x, x)
            ly, _ = model(y, y)
        self.assertTrue(torch.allclose(lx[0, :-1], ly[0, :-1], atol=1e-5))

    def test_generate(self):
        model = GPT(TINY)
        x = torch.randint(0, TINY.vocab_size, (2, 4))
        out = model.generate(x, max_new_tokens=6, temperature=0.7, top_k=10, top_p=0.9)
        self.assertEqual(out.shape, (2, 10))
        out = model.generate(x, max_new_tokens=3, temperature=0)  # greedy
        self.assertEqual(out.shape, (2, 7))

    def test_context_overflow_raises(self):
        model = GPT(TINY)
        with self.assertRaises(ValueError):
            model(torch.zeros(1, TINY.block_size + 1, dtype=torch.long))

    def test_checkpoint_roundtrip(self):
        model = GPT(TINY)
        x = torch.randint(0, TINY.vocab_size, (1, 8))
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "m.pt"
            save_checkpoint(path, model, step=7)
            loaded, ckpt = load_model(path)
        self.assertEqual(ckpt["step"], 7)
        with torch.no_grad():
            self.assertTrue(torch.allclose(model.eval()(x)[0], loaded(x)[0]))


class TestTraining(unittest.TestCase):
    def test_short_training_reduces_loss(self):
        rng = np.random.default_rng(0)
        # Séquence très régulière : le modèle doit l'apprendre vite.
        pattern = np.tile(np.arange(4, 20, dtype=np.uint16), 400)
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            pattern.tofile(d / "train.bin")
            pattern[: len(pattern) // 4].tofile(d / "val.bin")
            model = GPT(TINY)
            cfg = TrainConfig(max_steps=40, batch_size=8, eval_interval=20, eval_iters=2, log_interval=100, warmup_steps=5, checkpoint_dir=str(d / "ckpt"))
            with TokenDataset(d / "train.bin", d / "val.bin", block_size=TINY.block_size) as ds:
                trainer = Trainer(model, ds, cfg)
                before = trainer.estimate_loss()
                state = trainer.train()
                after = trainer.estimate_loss()
            self.assertLess(after["train"], before["train"])
            self.assertGreater(after["val_accuracy"], before["val_accuracy"])
            self.assertAlmostEqual(after["val_perplexity"], math.exp(after["val"]), places=3)
            self.assertEqual(state.step, 40)
            self.assertTrue((d / "ckpt" / "best.pt").exists())
            self.assertTrue((d / "ckpt" / "last.pt").exists())
            # Le log contient run_start, des évals avec les métriques, run_end.
            records = read_log(d / "ckpt" / "train_log.jsonl")
            types = [r["type"] for r in records]
            self.assertEqual(types[0], "run_start")
            self.assertEqual(types[-1], "run_end")
            evals = [r for r in records if r["type"] == "eval"]
            self.assertEqual(len(evals), 3)  # steps 0, 20, 39
            for key in ("val_loss", "val_perplexity", "val_accuracy", "probe_score", "sample", "lr"):
                self.assertIn(key, evals[0])

    def test_probes_and_samples_with_tokenizer(self):
        corpus = ["Django utilise l'architecture MTV. Python est un langage de programmation."] * 40
        tok = Tokenizer.train(corpus, vocab_size=320)
        ids = np.array(sum((tok.encode(t, add_eos=True) for t in corpus), []), dtype=np.uint16)
        cfg_model = GPTConfig(vocab_size=320, block_size=32, n_layer=2, n_head=2, n_embd=32, dropout=0.0)
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            ids.tofile(d / "train.bin")
            ids[:200].tofile(d / "val.bin")
            probes = [Probe("Django utilise l'architecture", "MTV"), Probe("Python est un langage de", "programmation")]
            cfg = TrainConfig(max_steps=150, batch_size=8, eval_interval=75, eval_iters=2, log_interval=1000, warmup_steps=10,
                              learning_rate=3e-3, checkpoint_dir=str(d / "ckpt"), sample_prompt="Django", sample_tokens=8)
            with TokenDataset(d / "train.bin", d / "val.bin", block_size=32) as ds:
                trainer = Trainer(GPT(cfg_model), ds, cfg, tokenizer=tok, probes=probes)
                trainer.train()
            evals = [r for r in read_log(d / "ckpt" / "train_log.jsonl") if r["type"] == "eval"]
        self.assertEqual(len(evals[0]["probes"]), 2)
        self.assertIsInstance(evals[-1]["sample"], str)
        # Sur un corpus aussi répétitif, les sondes doivent finir par réussir.
        self.assertGreaterEqual(evals[-1]["probe_score"], evals[0]["probe_score"])
        self.assertEqual(evals[-1]["probe_score"], 1.0)


class TestTextGenerator(unittest.TestCase):
    def test_generate_returns_text(self):
        tok = Tokenizer.train(["bonjour le monde python django"] * 5, vocab_size=300)
        model = GPT(GPTConfig(vocab_size=300, block_size=32, n_layer=1, n_head=2, n_embd=16, dropout=0.0))
        gen = TextGenerator(model, tok)
        out = gen.generate("bonjour", max_tokens=5, temperature=1.0)
        self.assertIsInstance(out, str)
        self.assertTrue(gen.complete("bonjour", max_tokens=3).startswith("bonjour"))


if __name__ == "__main__":
    unittest.main()
