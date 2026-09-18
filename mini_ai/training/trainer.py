"""Boucle d'entraînement (causal language modeling).

    Load batch → Forward → Loss → Backward → Optimizer step → Validation → Checkpoint

À chaque évaluation, en plus de la loss, on mesure « l'intelligence » du
modèle de plusieurs façons, écrites dans ``checkpoints/train_log.jsonl`` et
affichées par le tableau de bord (``python main.py dashboard``) :

  * perplexité de validation          exp(val_loss)
  * précision du token suivant        % de tokens de validation prédits (argmax)
  * score de sondes                   % de « questions à trous » complétées
  * un échantillon de texte généré    pour juger la qualité à l'œil
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import torch

from ..model import GPT
from ..tokenizer import Tokenizer
from .checkpoint import save_checkpoint
from .dataset import TokenDataset


@dataclass
class TrainConfig:
    max_steps: int = 2000
    batch_size: int = 32
    learning_rate: float = 1e-3
    min_lr: float = 1e-4
    warmup_steps: int = 100
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    eval_interval: int = 200
    eval_iters: int = 20
    log_interval: int = 20
    checkpoint_dir: str = "checkpoints"
    device: str = "cpu"
    seed: int = 1337
    grad_accum_steps: int = 1
    # Mesures qualitatives (nécessitent un tokenizer)
    sample_prompt: str = "Python est"
    sample_tokens: int = 40
    probe_tokens: int = 12
    run_name: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Probe:
    """Question à trous : le modèle doit compléter ``prompt`` par ``expected``."""

    prompt: str
    expected: str

    @staticmethod
    def load(path: str | Path) -> list["Probe"]:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return [Probe(p["prompt"], p["expected"]) for p in data]


@dataclass
class TrainState:
    step: int = 0
    best_val_loss: float = float("inf")
    history: list[dict] = field(default_factory=list)


class Trainer:
    def __init__(
        self,
        model: GPT,
        dataset: TokenDataset,
        config: TrainConfig,
        tokenizer: Tokenizer | None = None,
        probes: list[Probe] | None = None,
        on_eval: Callable[[dict], None] | None = None,
    ):
        self.model = model.to(config.device)
        self.dataset = dataset
        self.config = config
        self.tokenizer = tokenizer
        self.probes = probes or []
        self.on_eval = on_eval
        self.optimizer = model.configure_optimizers(config.weight_decay, config.learning_rate)
        self.state = TrainState()
        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.checkpoint_dir / "train_log.jsonl"
        self.run_name = config.run_name or datetime.now().strftime("run-%Y%m%d-%H%M%S")
        self._generator = None
        torch.manual_seed(config.seed)

    # -------------------------------------------------------------- schedule
    def lr_at(self, step: int) -> float:
        c = self.config
        if step < c.warmup_steps:
            return c.learning_rate * (step + 1) / c.warmup_steps
        if step >= c.max_steps:
            return c.min_lr
        progress = (step - c.warmup_steps) / max(1, c.max_steps - c.warmup_steps)
        coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
        return c.min_lr + coeff * (c.learning_rate - c.min_lr)

    # ------------------------------------------------------------ evaluation
    @torch.no_grad()
    def estimate_loss(self) -> dict[str, float]:
        """Loss moyenne train/val + précision du token suivant sur la validation."""
        self.model.eval()
        out: dict[str, float] = {}
        for split in ("train", "val"):
            losses = torch.zeros(self.config.eval_iters)
            correct = total = 0
            for k in range(self.config.eval_iters):
                x, y = self.dataset.get_batch(split, self.config.batch_size)
                logits, loss = self.model(x, y)
                losses[k] = loss.item()
                if split == "val":
                    correct += (logits.argmax(-1) == y).sum().item()
                    total += y.numel()
            out[split] = losses.mean().item()
            if split == "val":
                out["val_accuracy"] = correct / max(1, total)
        out["val_perplexity"] = math.exp(min(out["val"], 50))
        self.model.train()
        return out

    def _gen(self):
        if self._generator is None and self.tokenizer is not None:
            from ..inference.generate import TextGenerator

            self._generator = TextGenerator(self.model, self.tokenizer, self.config.device)
        return self._generator

    @torch.no_grad()
    def evaluate_probes(self) -> tuple[float | None, list[dict]]:
        """Complétion gloutonne de chaque sonde ; réussite si ``expected`` apparaît."""
        gen = self._gen()
        if gen is None or not self.probes:
            return None, []
        self.model.eval()
        results = []
        for p in self.probes:
            out = gen.generate(p.prompt, max_tokens=self.config.probe_tokens, temperature=0)
            ok = p.expected.lower() in out.lower()
            results.append({"prompt": p.prompt, "expected": p.expected, "output": out, "ok": ok})
        self.model.train()
        return sum(r["ok"] for r in results) / len(results), results

    @torch.no_grad()
    def sample(self) -> str | None:
        gen = self._gen()
        if gen is None or not self.config.sample_prompt:
            return None
        self.model.eval()
        text = gen.generate(self.config.sample_prompt, max_tokens=self.config.sample_tokens, temperature=0.8, top_k=40)
        self.model.train()
        return text

    def evaluate(self, step: int, lr: float) -> dict:
        losses = self.estimate_loss()
        probe_score, probe_results = self.evaluate_probes()
        record = {
            "type": "eval",
            "run": self.run_name,
            "step": step,
            "time": time.time(),
            "train_loss": losses["train"],
            "val_loss": losses["val"],
            "val_perplexity": losses["val_perplexity"],
            "val_accuracy": losses["val_accuracy"],
            "probe_score": probe_score,
            "probes": probe_results,
            "sample": self.sample(),
            "lr": lr,
        }
        return record

    def _log(self, record: dict) -> None:
        self.state.history.append(record)
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # ---------------------------------------------------------------- resume
    def load_state(self, ckpt: dict) -> None:
        if ckpt.get("optimizer_state"):
            self.optimizer.load_state_dict(ckpt["optimizer_state"])
        self.state.step = ckpt.get("step", 0)
        self.state.best_val_loss = ckpt.get("best_val_loss", float("inf"))

    # ------------------------------------------------------------------ train
    def train(self) -> TrainState:
        c = self.config
        model = self.model
        model.train()
        step = self.state.step
        self._log({
            "type": "run_start",
            "run": self.run_name,
            "time": time.time(),
            "step": step,
            "train_config": c.to_dict(),
            "model_config": model.config.to_dict(),
            "parameters": model.num_parameters(),
            "train_tokens": self.dataset.size("train"),
            "val_tokens": self.dataset.size("val"),
            "n_probes": len(self.probes),
        })
        t0 = time.time()

        while step < c.max_steps:
            lr = self.lr_at(step)
            for g in self.optimizer.param_groups:
                g["lr"] = lr

            # Validation + mesures + checkpoint
            if step % c.eval_interval == 0 or step == c.max_steps - 1:
                record = self.evaluate(step, lr)
                self._log(record)
                probe = f"  sondes {record['probe_score'] * 100:.0f}%" if record["probe_score"] is not None else ""
                print(
                    f"[eval] step {step:>6}  train {record['train_loss']:.4f}  val {record['val_loss']:.4f}  "
                    f"ppl {record['val_perplexity']:.1f}  acc {record['val_accuracy'] * 100:.1f}%{probe}  lr {lr:.2e}"
                )
                if record["sample"] is not None:
                    print(f"       sample: {c.sample_prompt!r} -> {record['sample'][:120]!r}")
                if self.on_eval:
                    self.on_eval(record)
                extra = {"train_config": c.to_dict(), "run": self.run_name}
                if record["val_loss"] < self.state.best_val_loss:
                    self.state.best_val_loss = record["val_loss"]
                    save_checkpoint(self.checkpoint_dir / "best.pt", model, self.optimizer, step, self.state.best_val_loss, extra)
                save_checkpoint(self.checkpoint_dir / "last.pt", model, self.optimizer, step, self.state.best_val_loss, extra)

            # Étape d'optimisation (avec accumulation de gradient optionnelle)
            self.optimizer.zero_grad(set_to_none=True)
            total = 0.0
            for _ in range(c.grad_accum_steps):
                x, y = self.dataset.get_batch("train", c.batch_size)
                _, loss = model(x, y)
                loss = loss / c.grad_accum_steps
                loss.backward()
                total += loss.item()
            if c.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), c.grad_clip)
            self.optimizer.step()

            if step % c.log_interval == 0:
                dt = time.time() - t0
                t0 = time.time()
                self._log({"type": "train", "run": self.run_name, "step": step, "time": time.time(), "loss": total, "lr": lr})
                print(f"[train] step {step:>6}  loss {total:.4f}  lr {lr:.2e}  ({dt * 1000 / max(1, c.log_interval):.0f} ms/step)")

            step += 1
            self.state.step = step

        save_checkpoint(self.checkpoint_dir / "last.pt", model, self.optimizer, step, self.state.best_val_loss, {"train_config": c.to_dict(), "run": self.run_name})
        self._log({"type": "run_end", "run": self.run_name, "time": time.time(), "step": step, "best_val_loss": self.state.best_val_loss})
        return self.state
