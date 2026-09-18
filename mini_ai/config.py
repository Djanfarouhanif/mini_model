"""Configuration globale : chemins et valeurs par défaut du projet."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"

TOKENIZER_PATH = PROCESSED_DIR / "tokenizer.json"
TRAIN_BIN = DATA_DIR / "train.bin"
VAL_BIN = DATA_DIR / "validation.bin"
MEMORY_DB = DATA_DIR / "memory.db"
LAST_CHECKPOINT = CHECKPOINT_DIR / "last.pt"
BEST_CHECKPOINT = CHECKPOINT_DIR / "best.pt"

# Taille de vocabulaire cible du tokenizer BPE. Le PRD vise ~8 000, mais
# avec un embedding de 128 cela dépasse à lui seul 1M de paramètres ;
# 4 096 × 96 permet de rester autour de 1M au total (voir model/config.py).
DEFAULT_VOCAB_SIZE = 4096

# Ratio train / validation lors de la préparation des données.
VAL_RATIO = 0.1
