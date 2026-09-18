"""Boucle d'entraînement (causal language modeling).

    Load batch → Forward → Loss → Backward → Optimizer step → Validation → Checkpoint
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

import torch

from ..model import GPT
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

    def to_dict(self) -> dict:
        return asdict(self)


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
        on_eval: Callable[[int, float, float], None] | None = None,
    ):
        self.model = model.to(config.device)
        self.dataset = dataset
        self.config = config
        self.on_eval = on_eval
        self.optimizer = model.configure_optimizers(config.weight_decay, config.learning_rate)
        self.state = TrainState()
        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.checkpoint_dir / "train_log.jsonl"
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
        self.model.eval()
        out: dict[str, float] = {}
        for split in ("train", "val"):
            losses = torch.zeros(self.config.eval_iters)
            for k in range(self.config.eval_iters):
                x, y = self.dataset.get_batch(split, self.config.batch_size)
                _, loss = self.model(x, y)
                losses[k] = loss.item()
            out[split] = losses.mean().item()
        self.model.train()
        return out

    def _log(self, record: dict) -> None:
        self.state.history.append(record)
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

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
        t0 = time.time()
        step = self.state.step

        while step < c.max_steps:
            lr = self.lr_at(step)
            for g in self.optimizer.param_groups:
                g["lr"] = lr

            # Validation + checkpoint
            if step % c.eval_interval == 0 or step == c.max_steps - 1:
                losses = self.estimate_loss()
                self._log({"step": step, "train_loss": losses["train"], "val_loss": losses["val"], "lr": lr, "type": "eval"})
                print(f"[eval] step {step:>6}  train {losses['train']:.4f}  val {losses['val']:.4f}  lr {lr:.2e}")
                if self.on_eval:
                    self.on_eval(step, losses["train"], losses["val"])
                if losses["val"] < self.state.best_val_loss:
                    self.state.best_val_loss = losses["val"]
                    save_checkpoint(self.checkpoint_dir / "best.pt", model, self.optimizer, step, self.state.best_val_loss, {"train_config": c.to_dict()})
                save_checkpoint(self.checkpoint_dir / "last.pt", model, self.optimizer, step, self.state.best_val_loss, {"train_config": c.to_dict()})

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
                self._log({"step": step, "loss": total, "lr": lr, "type": "train"})
                print(f"[train] step {step:>6}  loss {total:.4f}  lr {lr:.2e}  ({dt * 1000 / max(1, c.log_interval):.0f} ms/step)")

            step += 1
            self.state.step = step

        save_checkpoint(self.checkpoint_dir / "last.pt", model, self.optimizer, step, self.state.best_val_loss, {"train_config": c.to_dict()})
        return self.state
