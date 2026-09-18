"""Point d'entrée CLI de MiniAI.

    python main.py params        [--preset mini]
    python main.py prepare-data  [--vocab-size 4096]
    python main.py train         [--max-steps 2000 --batch-size 32 ...]
    python main.py generate      --prompt "Python est" [--max-tokens 100]
    python main.py run           "Construis une architecture pour une API Django" [--dummy]
    python main.py demo-bus                 # Agent A → Agent B → Agent A
    python main.py tool calculator "15 * 20"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from mini_ai import config as C


# ----------------------------------------------------------------- helpers
def _device(name: str) -> str:
    if name != "auto":
        return name
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def _load_model_or_dummy(args) -> object:
    """Charge le vrai modèle si un checkpoint existe, sinon le DummyModel."""
    from mini_ai.inference import DummyModel

    ckpt = Path(args.checkpoint)
    if getattr(args, "dummy", False) or not ckpt.exists() or not Path(args.tokenizer).exists():
        if not getattr(args, "dummy", False):
            print(f"[info] checkpoint {ckpt} ou tokenizer absent → DummyModel")
        return DummyModel()
    from mini_ai.inference import load_generator

    return load_generator(ckpt, args.tokenizer, device=_device(args.device))


# ---------------------------------------------------------------- commands
def cmd_params(args) -> None:
    from mini_ai.model import GPT
    from mini_ai.model.config import PRESETS

    presets = PRESETS if args.preset == "all" else {args.preset: PRESETS[args.preset]}
    for name, cfg in presets.items():
        model = GPT(cfg)
        print(f"{name:>8} : {model.num_parameters():>10,} paramètres  "
              f"(hors positions : {model.num_parameters(non_embedding=True):,})  {cfg.to_dict()}")


def cmd_prepare_data(args) -> None:
    from mini_ai.training import prepare_dataset

    prepare_dataset(args.raw, args.out, args.tokenizer, vocab_size=args.vocab_size, val_ratio=args.val_ratio)


def cmd_train(args) -> None:
    from mini_ai.model import GPT, GPTConfig
    from mini_ai.model.config import PRESETS
    from mini_ai.tokenizer import Tokenizer
    from mini_ai.training import TokenDataset, TrainConfig, Trainer, load_checkpoint

    device = _device(args.device)
    tokenizer = Tokenizer.load(args.tokenizer)
    cfg = PRESETS[args.preset]
    overrides = {k: getattr(args, k) for k in ("n_layer", "n_head", "n_embd", "block_size", "dropout", "pos_type") if getattr(args, k) is not None}
    cfg = GPTConfig.from_dict({**cfg.to_dict(), **overrides, "vocab_size": max(cfg.vocab_size, tokenizer.vocab_size)})

    resume = None
    if args.resume and Path(args.resume).exists():
        resume = load_checkpoint(args.resume, device)
        cfg = GPTConfig.from_dict(resume["model_config"])
        print(f"[train] reprise depuis {args.resume} (step {resume['step']})")

    model = GPT(cfg)
    if resume:
        model.load_state_dict(resume["model_state"])
    print(f"[train] modèle : {model.num_parameters():,} paramètres  {cfg.to_dict()}")

    dataset = TokenDataset(args.train_bin, args.val_bin, cfg.block_size, device=device)
    print(f"[train] train {dataset.size('train'):,} tokens / val {dataset.size('val'):,} tokens  device={device}")

    tc = TrainConfig(
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        min_lr=args.min_lr,
        warmup_steps=args.warmup,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        eval_interval=args.eval_interval,
        eval_iters=args.eval_iters,
        log_interval=args.log_interval,
        checkpoint_dir=args.checkpoint_dir,
        device=device,
        grad_accum_steps=args.grad_accum,
    )
    trainer = Trainer(model, dataset, tc)
    if resume:
        trainer.load_state(resume)
    state = trainer.train()
    print(f"[train] terminé : step {state.step}, meilleure val loss {state.best_val_loss:.4f}")

    if args.sample:
        from mini_ai.inference import TextGenerator

        gen = TextGenerator(model, tokenizer, device)
        print("\n[sample]", repr(gen.complete(args.sample, max_tokens=60, temperature=0.8, top_k=40)))


def cmd_generate(args) -> None:
    from mini_ai.inference import load_generator

    gen = load_generator(args.checkpoint, args.tokenizer, device=_device(args.device))
    for i in range(args.n):
        text = gen.generate(
            args.prompt,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
        )
        print(f"--- échantillon {i + 1} ---")
        print(args.prompt + text)


def cmd_run(args) -> None:
    from mini_ai.memory import MemoryStore
    from mini_ai.orchestration import Orchestrator
    from mini_ai.tools import default_tools

    model = _load_model_or_dummy(args)
    store = MemoryStore(":memory:" if args.no_memory else args.memory_db)
    orch = Orchestrator(model, store=store, tools=default_tools(args.workspace), verbose=not args.quiet)
    print(f"\n=== Tâche : {args.task}\n")
    result = orch.run(args.task)
    print("\n=== Réponse finale ===\n")
    print(result.final)
    print(f"\n({len(result.transcript)} messages, {result.duration:.2f}s, modèle={getattr(model, 'name', type(model).__name__)})")


def cmd_demo_bus(args) -> None:
    """Premier test du PRD : Agent A → Agent B → Agent A."""
    from mini_ai.agents import Agent
    from mini_ai.communication import MessageBus

    model = _load_model_or_dummy(args)
    bus = MessageBus()
    a = Agent("agent_a", role="Développeur Python", goal="Poser des questions", model=model, bus=bus)
    b = Agent("agent_b", role="Revieweur de code", goal="Répondre aux questions", model=model, bus=bus)
    reply = a.ask("agent_b", args.message, b)
    for m in bus.history:
        print(m)
    print("\nréponse reçue par A :", reply.content if reply else None)


def cmd_tool(args) -> None:
    from mini_ai.tools import default_tools

    result = default_tools(args.workspace).run(args.name, args.input)
    print(result)


# ------------------------------------------------------------------ parser
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="mini_ai", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    def add_model_args(sp):
        sp.add_argument("--checkpoint", default=str(C.BEST_CHECKPOINT))
        sp.add_argument("--tokenizer", default=str(C.TOKENIZER_PATH))
        sp.add_argument("--device", default="auto")

    sp = sub.add_parser("params", help="affiche le nombre de paramètres")
    sp.add_argument("--preset", default="all")
    sp.set_defaults(func=cmd_params)

    sp = sub.add_parser("prepare-data", help="nettoyage + tokenizer + train.bin/validation.bin")
    sp.add_argument("--raw", default=str(C.RAW_DIR))
    sp.add_argument("--out", default=str(C.DATA_DIR))
    sp.add_argument("--tokenizer", default=str(C.TOKENIZER_PATH))
    sp.add_argument("--vocab-size", type=int, default=C.DEFAULT_VOCAB_SIZE)
    sp.add_argument("--val-ratio", type=float, default=C.VAL_RATIO)
    sp.set_defaults(func=cmd_prepare_data)

    sp = sub.add_parser("train", help="entraîne le modèle")
    sp.add_argument("--preset", default="mini")
    sp.add_argument("--tokenizer", default=str(C.TOKENIZER_PATH))
    sp.add_argument("--train-bin", default=str(C.TRAIN_BIN))
    sp.add_argument("--val-bin", default=str(C.VAL_BIN))
    sp.add_argument("--checkpoint-dir", default=str(C.CHECKPOINT_DIR))
    sp.add_argument("--resume", default=None, help="checkpoint à reprendre (ex: checkpoints/last.pt)")
    sp.add_argument("--device", default="auto")
    sp.add_argument("--max-steps", type=int, default=2000)
    sp.add_argument("--batch-size", type=int, default=32)
    sp.add_argument("--grad-accum", type=int, default=1)
    sp.add_argument("--lr", type=float, default=1e-3)
    sp.add_argument("--min-lr", type=float, default=1e-4)
    sp.add_argument("--warmup", type=int, default=100)
    sp.add_argument("--weight-decay", type=float, default=0.1)
    sp.add_argument("--grad-clip", type=float, default=1.0)
    sp.add_argument("--eval-interval", type=int, default=200)
    sp.add_argument("--eval-iters", type=int, default=20)
    sp.add_argument("--log-interval", type=int, default=20)
    sp.add_argument("--sample", default="Python est", help="prompt d'exemple généré en fin d'entraînement ('' pour désactiver)")
    for name, typ in (("n_layer", int), ("n_head", int), ("n_embd", int), ("block_size", int), ("dropout", float), ("pos_type", str)):
        sp.add_argument(f"--{name.replace('_', '-')}", dest=name, type=typ, default=None)
    sp.set_defaults(func=cmd_train)

    sp = sub.add_parser("generate", help="génère du texte")
    add_model_args(sp)
    sp.add_argument("--prompt", required=True)
    sp.add_argument("--max-tokens", type=int, default=100)
    sp.add_argument("--temperature", type=float, default=0.8)
    sp.add_argument("--top-k", type=int, default=40)
    sp.add_argument("--top-p", type=float, default=0.9)
    sp.add_argument("--repetition-penalty", type=float, default=1.0)
    sp.add_argument("-n", type=int, default=1, help="nombre d'échantillons")
    sp.set_defaults(func=cmd_generate)

    sp = sub.add_parser("run", help="lance l'orchestrateur multi-agent sur une tâche")
    add_model_args(sp)
    sp.add_argument("task")
    sp.add_argument("--dummy", action="store_true", help="force le modèle factice")
    sp.add_argument("--memory-db", default=str(C.MEMORY_DB))
    sp.add_argument("--no-memory", action="store_true", help="mémoire longue en RAM uniquement")
    sp.add_argument("--workspace", default=str(C.DATA_DIR / "workspace"))
    sp.add_argument("--quiet", action="store_true")
    sp.set_defaults(func=cmd_run)

    sp = sub.add_parser("demo-bus", help="Agent A → Agent B → Agent A")
    add_model_args(sp)
    sp.add_argument("--message", default="Peux-tu relire mon architecture Django ?")
    sp.add_argument("--dummy", action="store_true")
    sp.set_defaults(func=cmd_demo_bus)

    sp = sub.add_parser("tool", help="exécute un outil directement")
    sp.add_argument("name", choices=["calculator", "python", "filesystem"])
    sp.add_argument("input")
    sp.add_argument("--workspace", default=str(C.DATA_DIR / "workspace"))
    sp.set_defaults(func=cmd_tool)
    return p


def main(argv: list[str] | None = None) -> int:
    # La console Windows est souvent en cp1252 : on force l'UTF-8 pour les accents et flèches.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    args = build_parser().parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
