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


def cmd_inspect(args) -> None:
    """Affiche le contenu d'un checkpoint : config, étape, et chaque tenseur de poids."""
    import torch

    from mini_ai.training import load_checkpoint

    ckpt = load_checkpoint(args.checkpoint)
    print(f"fichier        : {args.checkpoint}")
    print(f"étape          : {ckpt['step']}   meilleure val loss : {ckpt['best_val_loss']:.4f}   run : {ckpt.get('extra', {}).get('run', '?')}")
    print(f"config modèle  : {ckpt['model_config']}")
    state = ckpt["model_state"]
    total = 0
    print(f"\n{'nom du tenseur':<42} {'forme':<18} {'nb':>10} {'moyenne':>9} {'écart-type':>11} {'min':>8} {'max':>8}")
    print("-" * 112)
    for name, t in state.items():
        t = t.float()
        total += t.numel()
        print(f"{name:<42} {str(tuple(t.shape)):<18} {t.numel():>10,} {t.mean():>9.4f} {t.std():>11.4f} {t.min():>8.3f} {t.max():>8.3f}")
    print("-" * 112)
    print(f"{'total':<42} {'':<18} {total:>10,}")
    if args.show:
        t = state[args.show]
        torch.set_printoptions(precision=4, sci_mode=False, edgeitems=4)
        print(f"\n{args.show} {tuple(t.shape)} :\n{t}")


def cmd_download_data(args) -> None:
    from mini_ai.training.download import PRESETS, download

    if args.list:
        for name, p in PRESETS.items():
            print(f"{name:>14} : {p.dataset}" + (f" ({p.config})" if p.config else "") + f" — {p.description}")
        return
    preset = PRESETS.get(args.preset) if args.preset else None
    dataset = args.dataset or (preset.dataset if preset else None)
    if not dataset:
        raise SystemExit("précisez --preset <nom> ou --dataset <org/nom>  (--list pour voir les presets)")
    config = args.config if args.config is not None else (preset.config if preset else None)
    text_field = args.text_field or (preset.text_field if preset else "text")
    split = args.split or (preset.split if preset else "train")
    filename = args.out or (preset.filename if preset else dataset.replace("/", "_") + ".txt")
    out_path = Path(filename) if Path(filename).is_absolute() or "/" in filename or "\\" in filename else C.RAW_DIR / filename
    download(dataset, out_path, config=config, split=split, text_field=text_field, max_chars=args.max_chars, max_docs=args.max_docs)


def cmd_prepare_data(args) -> None:
    from mini_ai.training import prepare_dataset

    tokenizer = None
    if args.keep_tokenizer:
        from mini_ai.tokenizer import Tokenizer

        if not Path(args.tokenizer).exists():
            raise SystemExit(f"--keep-tokenizer : aucun tokenizer trouvé à {args.tokenizer}")
        tokenizer = Tokenizer.load(args.tokenizer)
        print(f"[data] tokenizer existant réutilisé ({tokenizer.vocab_size} tokens) — les checkpoints restent compatibles")
    prepare_dataset(args.raw, args.out, args.tokenizer, vocab_size=args.vocab_size, val_ratio=args.val_ratio,
                    tokenizer=tokenizer, tokenizer_sample_chars=args.tokenizer_sample_chars)


def cmd_train(args) -> None:
    from mini_ai.model import GPT, GPTConfig
    from mini_ai.model.config import PRESETS
    from mini_ai.tokenizer import Tokenizer
    from mini_ai.training import Probe, TokenDataset, TrainConfig, Trainer, load_checkpoint

    device = _device(args.device)
    log_path = Path(args.checkpoint_dir) / "train_log.jsonl"
    if args.dashboard:
        from mini_ai.dashboard import start_in_background

        start_in_background(log_path, port=args.port)
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
        sample_prompt=args.sample,
        sample_tokens=args.sample_tokens,
        run_name=args.run_name or "",
    )
    probes = Probe.load(args.probes) if args.probes and Path(args.probes).exists() else []
    if probes:
        print(f"[train] {len(probes)} sondes chargées depuis {args.probes}")
    trainer = Trainer(model, dataset, tc, tokenizer=tokenizer, probes=probes)
    if resume:
        trainer.load_state(resume)
    state = trainer.train()
    print(f"[train] terminé : step {state.step}, meilleure val loss {state.best_val_loss:.4f}")
    print(f"[train] visualiser : python main.py dashboard  (log : {log_path})")
    if args.dashboard:
        print("[dashboard] serveur toujours actif — Ctrl+C pour quitter")
        try:
            import time

            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass


def cmd_dashboard(args) -> None:
    from mini_ai.dashboard import serve

    serve(args.log, port=args.port, open_browser=not args.no_browser)


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

    sp = sub.add_parser("inspect", help="liste les paramètres appris d'un checkpoint")
    sp.add_argument("--checkpoint", default=str(C.BEST_CHECKPOINT))
    sp.add_argument("--show", default=None, help="affiche les valeurs d'un tenseur, ex: wte.weight_table.weight")
    sp.set_defaults(func=cmd_inspect)

    sp = sub.add_parser("download-data", help="télécharge un corpus Hugging Face (streaming) dans data/raw/")
    sp.add_argument("--preset", default=None, help="wiki-fr, wiki-en, tinystories, french-books, python-code")
    sp.add_argument("--dataset", default=None, help="identifiant HF, ex: wikimedia/wikipedia")
    sp.add_argument("--config", default=None, help="configuration/sous-ensemble, ex: 20231101.fr")
    sp.add_argument("--split", default=None)
    sp.add_argument("--text-field", default=None, help="colonne contenant le texte (défaut : text)")
    sp.add_argument("--max-chars", type=int, default=5_000_000, help="taille max du corpus en caractères")
    sp.add_argument("--max-docs", type=int, default=None)
    sp.add_argument("--out", default=None, help="nom du fichier de sortie dans data/raw/")
    sp.add_argument("--list", action="store_true", help="liste les presets")
    sp.set_defaults(func=cmd_download_data)

    sp = sub.add_parser("prepare-data", help="nettoyage + tokenizer + train.bin/validation.bin")
    sp.add_argument("--raw", default=str(C.RAW_DIR))
    sp.add_argument("--out", default=str(C.DATA_DIR))
    sp.add_argument("--tokenizer", default=str(C.TOKENIZER_PATH))
    sp.add_argument("--vocab-size", type=int, default=C.DEFAULT_VOCAB_SIZE)
    sp.add_argument("--val-ratio", type=float, default=C.VAL_RATIO)
    sp.add_argument("--keep-tokenizer", action="store_true", help="réutilise tokenizer.json au lieu de le réapprendre (garde les checkpoints compatibles)")
    sp.add_argument("--tokenizer-sample-chars", type=int, default=50_000_000, help="taille max de l'échantillon pour apprendre le tokenizer")
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
    sp.add_argument("--sample", default="Python est", help="prompt généré à chaque évaluation ('' pour désactiver)")
    sp.add_argument("--sample-tokens", type=int, default=40)
    sp.add_argument("--probes", default=str(C.DATA_DIR / "probes.json"), help="fichier JSON de questions à trous ('' pour désactiver)")
    sp.add_argument("--run-name", default=None, help="nom du run dans le log (défaut : horodatage)")
    sp.add_argument("--dashboard", action="store_true", help="ouvre le tableau de bord pendant l'entraînement")
    sp.add_argument("--port", type=int, default=8765)
    for name, typ in (("n_layer", int), ("n_head", int), ("n_embd", int), ("block_size", int), ("dropout", float), ("pos_type", str)):
        sp.add_argument(f"--{name.replace('_', '-')}", dest=name, type=typ, default=None)
    sp.set_defaults(func=cmd_train)

    sp = sub.add_parser("dashboard", help="tableau de bord d'entraînement (loss, perplexité, précision, sondes, échantillons)")
    sp.add_argument("--log", default=str(C.CHECKPOINT_DIR / "train_log.jsonl"))
    sp.add_argument("--port", type=int, default=8765)
    sp.add_argument("--no-browser", action="store_true")
    sp.set_defaults(func=cmd_dashboard)

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
