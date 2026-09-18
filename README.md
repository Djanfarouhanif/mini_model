# MiniAI — Mini Multi-Agent AI

Implémentation du [PRD](prd.md) : un Transformer d'environ **1M de paramètres**, écrit
de zéro en PyTorch, qui sert de cerveau à plusieurs agents (Manager, Researcher,
Developer, Critic) capables de communiquer, mémoriser et collaborer.

```text
Dataset → Tokenizer BPE → Transformer ~1M → Inference → Agent → Memory
        → Communication (MessageBus) → Multi-Agent → Tools → Orchestration
```

## Installation

```bash
python -m venv venv
venv\Scripts\activate            # Windows   (source venv/bin/activate sous Linux/macOS)
pip install -r requirements.txt  # torch + numpy (CPU suffit)
```

## Démarrage rapide

```bash
python main.py params                      # nombre de paramètres par preset
python main.py prepare-data                # data/raw/*.txt → tokenizer + train.bin / validation.bin
python main.py train --max-steps 2000      # entraînement (checkpoints/best.pt, last.pt)
python main.py train --dashboard           # idem, avec le tableau de bord ouvert dans le navigateur
python main.py dashboard                   # tableau de bord seul (relit checkpoints/train_log.jsonl)
python main.py generate --prompt "Python est" --max-tokens 60 --temperature 0.8
python main.py run "Construis une architecture pour une API Django."
python main.py demo-bus                    # Agent A → Agent B → Agent A
python main.py tool calculator "15 * 20"
python -m unittest discover -s tests       # 41 tests
```

`run` et `demo-bus` utilisent `checkpoints/best.pt` s'il existe, sinon le `DummyModel`
(réponses déterministes) — pratique pour tester la couche agents sans modèle entraîné
(`--dummy` pour le forcer).

Pour un vrai entraînement, déposez davantage de texte (`.txt`, `.md`, `.py`) dans
`data/raw/` : le corpus d'exemple (14 Ko) ne sert qu'à valider le pipeline.

## Structure

```text
mini_ai/
├── config.py            chemins et valeurs par défaut
├── tokenizer/           bpe.py (byte-level BPE), trainer.py, tokenizer.py
├── model/               config.py, embeddings.py (learned + RoPE), attention.py, transformer.py, gpt.py
├── training/            dataset.py (pipeline + batches), trainer.py (AdamW, cosine LR), checkpoint.py
├── inference/           generate.py (TextGenerator : temperature / top_k / top_p), dummy.py
├── dashboard/           server.py (HTTP local) + page.html (graphiques SVG)
├── agents/              base.py, manager.py, researcher.py, developer.py, critic.py
├── memory/              short_term.py, long_term.py, store.py (SQLite)
├── communication/       message.py, message_bus.py (in-memory)
├── tools/               base.py, calculator.py, python.py, filesystem.py
└── orchestration/       orchestrator.py
data/raw/                corpus brut          checkpoints/   modèles sauvegardés
tests/                   tests unitaires      main.py, train.py
```

## Modèle

| Preset    | vocab | d   | couches | têtes | ctx | paramètres |
|-----------|------:|----:|--------:|------:|----:|-----------:|
| `mini` *  | 4096  | 96  | 6       | 4     | 256 | 1 089 024  |
| `mini-8k` | 8000  | 64  | 6       | 4     | 256 |   828 416  |
| `prd`     | 8000  | 128 | 6       | 4     | 256 | 2 246 656  |

\* preset par défaut. La configuration littérale du PRD (`prd`) dépasse 2M car
l'embedding 8000 × 128 fait à lui seul 1,02M ; conformément au §6 du PRD, la
configuration a été ajustée pour rester proche de 1M. `--preset`, `--n-layer`,
`--n-embd`, `--pos-type rope`… permettent de changer cela à l'entraînement.

Le tokenizer est un BPE au niveau octet : `decode(encode(text)) == text` pour tout
texte normalisé NFC, tokens spéciaux `<PAD> <UNK> <BOS> <EOS>` (ids 0–3).

## Mesurer si le modèle « devient plus intelligent »

`python main.py dashboard` ouvre http://127.0.0.1:8765/ — une page locale, sans
dépendance, qui se rafraîchit toutes les 3 s pendant l'entraînement. Elle lit
`checkpoints/train_log.jsonl`, où le trainer écrit à chaque évaluation :

| Mesure | Ce qu'elle dit |
|---|---|
| **train / val loss** | entropie croisée ; la validation est la seule qui compte pour la généralisation |
| **perplexité val** | `exp(val_loss)` : « entre combien de tokens le modèle hésite » (4096 = hasard) |
| **précision token** | % de tokens de validation prédits exactement (argmax) |
| **score de sondes** | % de questions à trous de [data/probes.json](data/probes.json) complétées correctement (« Django utilise l'architecture → MTV »), génération gloutonne |
| **échantillons** | le même prompt (`--sample`) généré à chaque évaluation, pour juger à l'œil |

Un encart « Le modèle apprend-il ? » résume la tendance et signale le
surapprentissage (val qui remonte alors que train descend) et le meilleur
checkpoint. Plusieurs runs sont conservés dans le log (`--run-name`) et
sélectionnables dans la page. Adaptez `data/probes.json` à votre corpus : les
sondes ne valent que si la réponse attendue figure dans les données.

## Cycle multi-agent

```text
USER → MANAGER → "Researcher, rassemble les informations"   → RESEARCHER (mémoire + fichiers)
              → "Developer, propose une solution" + contexte → DEVELOPER
              → "Critic, analyse cette proposition"          → CRITIC (vérifications + avis)
              → synthèse finale → USER
```

Tous les agents partagent le même modèle, le même `MessageBus` et le même
`MemoryStore` (SQLite : faits + journal des messages). Un agent appelle un outil
quand un message contient `[tool:calculator] 15 * 20`.

## Utilisation en Python

```python
from mini_ai.agents import Agent
from mini_ai.communication import MessageBus
from mini_ai.inference import load_generator          # ou DummyModel
from mini_ai.memory import MemoryStore
from mini_ai.orchestration import Orchestrator

model = load_generator("checkpoints/best.pt", "data/processed/tokenizer.json")

# Agent A → Agent B → Agent A
bus = MessageBus()
dev = Agent("developer", role="Développeur Python", model=model, bus=bus)
critic = Agent("critic", role="Revieweur de code", model=model, bus=bus)
reply = dev.ask("critic", "Relis mon architecture", critic)

# Orchestration complète
result = Orchestrator(model, store=MemoryStore("data/memory.db")).run("Construis une API Django")
print(result.final)
```

## Roadmap (PRD §20)

- [x] Phase 1 — Tokenizer BPE
- [x] Phase 2 — Transformer ~1M
- [x] Phase 3 — Training (dataset, loop, validation, checkpoints, log)
- [x] Phase 4 — Inference
- [x] Phase 5 — Agent
- [x] Phase 6 — Communication
- [x] Phase 7 — Multi-agent (Manager / Researcher / Developer / Critic)
- [x] Phase 8 — Memory (courte + longue SQLite)
- [x] Phase 9 — Tools (calculator, python, filesystem)
- [ ] Phase 10 — Experiments (1 vs 2 vs 4 vs 8 agents)
