"""Sauvegarde / restauration du modèle et de l'état d'entraînement."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from ..model import GPT, GPTConfig


def save_checkpoint(
    path: str | Path,
    model: GPT,
    optimizer: torch.optim.Optimizer | None = None,
    step: int = 0,
    best_val_loss: float = float("inf"),
    extra: dict[str, Any] | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_config": model.config.to_dict(),
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "step": step,
        "best_val_loss": best_val_loss,
        "extra": extra or {},
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)  # écriture atomique


def load_checkpoint(path: str | Path, device: str = "cpu") -> dict[str, Any]:
    return torch.load(Path(path), map_location=device, weights_only=False)


def load_model(path: str | Path, device: str = "cpu") -> tuple[GPT, dict[str, Any]]:
    ckpt = load_checkpoint(path, device)
    config = GPTConfig.from_dict(ckpt["model_config"])
    model = GPT(config)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()
    return model, ckpt
