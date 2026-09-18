Oui. Je te propose un PRD orienté **projet technique / R&D**, avec un MVP réaliste : construire un modèle de langage d’environ **1M de paramètres**, puis créer une architecture permettant à plusieurs agents partageant ce modèle de communiquer entre eux.

# PRD — Mini Multi-Agent AI

**Nom de travail :** `MiniAI`
**Type :** Projet R&D / framework expérimental
**Objectif principal :** Construire un petit modèle de langage (~1M paramètres) capable d’alimenter plusieurs agents autonomes pouvant communiquer, mémoriser des informations et collaborer sur une tâche.

---

## 1. Vision du projet

Construire un environnement local dans lequel un petit modèle de langage est utilisé comme **cerveau de plusieurs agents spécialisés**.

L'objectif n'est pas de reproduire ChatGPT avec 1M paramètres.

L'objectif est de comprendre et implémenter soi-même les briques fondamentales :

```text
Dataset
   ↓
Tokenizer
   ↓
Language Model ~1M
   ↓
Inference
   ↓
Agent
   ↓
Memory
   ↓
Communication
   ↓
Multi-Agent System
```

À terme :

```text
                    ┌─────────────────┐
                    │      USER       │
                    └────────┬────────┘
                             ↓
                    ┌─────────────────┐
                    │     MANAGER     │
                    └────────┬────────┘
                             │
             ┌───────────────┼───────────────┐
             ↓               ↓               ↓
       ┌───────────┐   ┌───────────┐   ┌───────────┐
       │ RESEARCHER│   │ DEVELOPER │   │  CRITIC   │
       └─────┬─────┘   └─────┬─────┘   └─────┬─────┘
             │               │               │
             └───────────────┼───────────────┘
                             ↓
                    ┌─────────────────┐
                    │  SHARED MEMORY  │
                    └─────────────────┘
```

---

# 2. Problème

Les frameworks d'agents actuels permettent de construire rapidement des systèmes multi-agents, mais cachent souvent les mécanismes internes.

Le projet cherche donc à répondre à la question :

> **Peut-on construire de zéro un petit système multi-agent dans lequel un modèle de seulement ~1M paramètres constitue le cerveau de plusieurs agents ?**

Le projet permettra d'expérimenter directement avec :

* tokenisation ;
* Transformer ;
* entraînement ;
* génération ;
* mémoire ;
* rôles ;
* communication inter-agents ;
* outils ;
* orchestration.

---

# 3. Objectifs

## Objectif principal

Construire un système fonctionnel composé de :

### A. Un tokenizer

Tokenizer BPE développé pour le projet.

### B. Un modèle de langage

Un Transformer causal d'environ **1M paramètres**.

### C. Un moteur d'inférence

Capable de prendre :

```text
prompt
```

et produire :

```text
generated tokens
```

puis :

```text
generated text
```

### D. Un système d'agents

Permettre de transformer le modèle en plusieurs agents.

Exemple :

```python
developer = Agent(
    name="developer",
    role="Développeur Python"
)

critic = Agent(
    name="critic",
    role="Revieweur de code"
)
```

### E. Un système de communication

Permettre :

```text
Agent A
   ↓
Message
   ↓
Agent B
   ↓
Response
```

### F. Une mémoire

Chaque agent doit pouvoir conserver :

* historique récent ;
* informations importantes ;
* résultats précédents.

### G. Un orchestrateur

Gérer le cycle :

```text
Task
 ↓
Manager
 ↓
Agents
 ↓
Messages
 ↓
Responses
 ↓
Evaluation
 ↓
Final answer
```

---

# 4. Non-objectifs du MVP

Le MVP ne cherche pas à :

* concurrencer GPT ;
* créer un modèle généraliste ;
* avoir plusieurs milliards de paramètres ;
* entraîner un modèle sur Internet à grande échelle ;
* créer immédiatement une interface SaaS ;
* avoir une intelligence comparable aux LLM modernes.

Le projet est avant tout **pédagogique et expérimental**.

---

# 5. Architecture globale

```text
                    ┌───────────────┐
                    │    Dataset    │
                    └───────┬───────┘
                            ↓
                    ┌───────────────┐
                    │   Tokenizer   │
                    │      BPE      │
                    └───────┬───────┘
                            ↓
                    ┌───────────────┐
                    │ Transformer   │
                    │    ~1M        │
                    └───────┬───────┘
                            ↓
                    ┌───────────────┐
                    │  Inference    │
                    └───────┬───────┘
                            ↓
              ┌─────────────┴─────────────┐
              ↓                           ↓
       ┌─────────────┐             ┌─────────────┐
       │    Agent    │             │    Agent    │
       └──────┬──────┘             └──────┬──────┘
              │                           │
              └─────────────┬─────────────┘
                            ↓
                    ┌───────────────┐
                    │ Message Bus   │
                    └───────┬───────┘
                            ↓
                    ┌───────────────┐
                    │     Memory    │
                    └───────────────┘
```

---

# 6. Architecture du modèle

Le premier modèle sera un **Decoder-only Transformer**, inspiré de l'architecture GPT.

Configuration cible :

| Paramètre           |                         Cible |
| ------------------- | ----------------------------: |
| Architecture        |      Decoder-only Transformer |
| Paramètres          |                           ~1M |
| Context length      |                    256 tokens |
| Vocabulary          |                 ~8 000 tokens |
| Embedding dimension |                          ~128 |
| Attention heads     |                             4 |
| Transformer layers  |                             6 |
| Activation          |                          GELU |
| Normalisation       |                     LayerNorm |
| Position            | RoPE ou positional embeddings |
| Precision initiale  |                          FP32 |

La configuration exacte devra être ajustée après calcul du nombre réel de paramètres.

**Critère :** rester proche de 1M plutôt que forcer artificiellement exactement 1 000 000.

---

# 7. Tokenizer

Le tokenizer sera développé avant le modèle.

### Fonctionnalités

```text
Text
 ↓
Normalization
 ↓
BPE
 ↓
Token IDs
```

Exemple :

```text
"Bonjour le monde"
```

devient :

```text
[421, 87, 1942]
```

Le tokenizer devra supporter :

* `<PAD>`
* `<UNK>`
* `<BOS>`
* `<EOS>`

et permettre :

```python
tokens = tokenizer.encode(text)

text = tokenizer.decode(tokens)
```

### Critère d'acceptation

```python
text == tokenizer.decode(
    tokenizer.encode(text)
)
```

pour les cas compatibles avec la normalisation du tokenizer.

---

# 8. Dataset

Le modèle devra être entraîné sur un dataset texte relativement petit.

Pipeline :

```text
Raw dataset
    ↓
Cleaning
    ↓
Deduplication
    ↓
Tokenizer
    ↓
Token IDs
    ↓
Train / Validation
```

### Dataset initial

Le MVP peut commencer avec un corpus volontairement limité afin de faciliter les expérimentations.

Par exemple :

```text
data/
├── raw/
├── processed/
├── train.bin
└── validation.bin
```

---

# 9. Entraînement

Le modèle utilise un objectif de **causal language modeling**.

Exemple :

```text
Input:

"Python est un langage"

Target:

"est un langage de"
```

Le modèle apprend :

```text
P(token_next | tokens_previous)
```

### Training loop

```text
Load batch
    ↓
Forward
    ↓
Calculate loss
    ↓
Backward
    ↓
Optimizer step
    ↓
Validation
    ↓
Checkpoint
```

Technologies :

* Python
* PyTorch
* NumPy

---

# 10. Génération

Le modèle devra supporter :

```python
model.generate(
    prompt,
    max_tokens=100
)
```

Paramètres :

```text
temperature
top_k
top_p
max_tokens
```

Exemple :

```text
Prompt:
"Python est"

Output:
"Python est un langage..."
```

La qualité de génération sera évaluée séparément des fonctionnalités d'agent.

---

# 11. Système Agent

L'agent sera une abstraction au-dessus du modèle.

```python
class Agent:
    name
    role
    goal
    memory
    model
```

Exemple :

```python
developer = Agent(
    name="developer",
    role="Développeur Python",
    goal="Résoudre les problèmes techniques"
)
```

Chaque agent possède :

### Identité

```text
name
role
goal
```

### Mémoire

```text
short-term memory
long-term memory
```

### Modèle

```text
shared language model
```

### Outils

```text
calculator
filesystem
python
etc.
```

---

# 12. Communication entre agents

Le système devra utiliser un format de message standard.

```json
{
  "sender": "manager",
  "receiver": "developer",
  "type": "task",
  "content": "Propose une architecture Django",
  "timestamp": "..."
}
```

Réponse :

```json
{
  "sender": "developer",
  "receiver": "manager",
  "type": "response",
  "content": "Je propose...",
  "timestamp": "..."
}
```

---

# 13. Message Bus

Créer une couche indépendante permettant aux agents de communiquer.

```python
message_bus.send(
    sender="manager",
    receiver="developer",
    content="Analyse cette tâche"
)
```

Puis :

```python
message = message_bus.receive("developer")
```

### Première version

Le message bus peut simplement être **in-memory**.

Plus tard :

```text
In-memory
   ↓
Redis
   ↓
Distributed agents
```

---

# 14. Mémoire

Deux niveaux.

## Short-term memory

Historique de la conversation :

```text
message 1
message 2
message 3
...
```

## Long-term memory

Informations persistantes :

```text
Agent learned:
"Django utilise MTV architecture."
```

Structure possible :

```text
memory/
├── short_term.py
├── long_term.py
└── store.py
```

Pour le MVP, une base SQLite peut suffire.

---

# 15. Rôles des agents

Le MVP devra proposer au minimum quatre rôles :

### Manager

Coordonne les autres agents.

```text
Task
 ↓
Manager
 ↓
Delegation
```

### Researcher

Cherche et synthétise des informations disponibles dans son environnement.

### Developer

Propose des solutions techniques.

### Critic

Analyse les réponses et identifie les problèmes.

---

# 16. Exemple de scénario

Utilisateur :

> Construis une architecture pour une API Django.

Le système :

```text
USER
 ↓
MANAGER
 ↓
"Developer, propose une architecture"
 ↓
DEVELOPER
 ↓
Architecture
 ↓
MANAGER
 ↓
"Critic, analyse cette architecture"
 ↓
CRITIC
 ↓
Critique
 ↓
MANAGER
 ↓
Final response
```

Le résultat final doit être produit par le manager à partir des réponses reçues.

---

# 17. Outils

Le système devra progressivement permettre aux agents d'utiliser des outils.

MVP :

```text
Calculator
Filesystem
Python execution
```

Architecture :

```python
class Tool:
    name
    description

    def execute(self, input):
        ...
```

Exemple :

```python
calculator.execute("15 * 20")
```

---

# 18. Orchestrateur

L'orchestrateur est responsable de la boucle multi-agent.

```python
orchestrator.run(task)
```

Flux :

```text
Task
 ↓
Create Manager
 ↓
Manager analyses task
 ↓
Delegate
 ↓
Agents execute
 ↓
Collect responses
 ↓
Critic evaluates
 ↓
Manager synthesizes
 ↓
Final response
```

---

# 19. Structure du projet

```text
mini_ai/
│
├── tokenizer/
│   ├── bpe.py
│   ├── trainer.py
│   └── tokenizer.py
│
├── model/
│   ├── config.py
│   ├── embeddings.py
│   ├── attention.py
│   ├── transformer.py
│   └── gpt.py
│
├── training/
│   ├── dataset.py
│   ├── trainer.py
│   └── checkpoint.py
│
├── inference/
│   └── generate.py
│
├── agents/
│   ├── base.py
│   ├── manager.py
│   ├── developer.py
│   ├── researcher.py
│   └── critic.py
│
├── memory/
│   ├── short_term.py
│   ├── long_term.py
│   └── store.py
│
├── communication/
│   ├── message.py
│   └── message_bus.py
│
├── tools/
│   ├── base.py
│   ├── calculator.py
│   ├── python.py
│   └── filesystem.py
│
├── orchestration/
│   └── orchestrator.py
│
├── data/
├── checkpoints/
├── tests/
│
├── config.py
├── train.py
└── main.py
```

---

# 20. Roadmap

## Phase 1 — Tokenizer

**Objectif :** créer le BPE.

Livrables :

* tokenizer ;
* vocabulaire ;
* encode/decode ;
* tests.

---

## Phase 2 — Transformer

**Objectif :** construire le modèle ~1M.

Livrables :

```text
Embedding
Attention
MLP
TransformerBlock
Transformer
```

Puis :

```python
print(count_parameters(model))
```

Résultat attendu :

```text
~1,000,000 parameters
```

---

## Phase 3 — Training

**Objectif :** faire apprendre le modèle.

Livrables :

* dataset ;
* dataloader ;
* training loop ;
* validation ;
* checkpoints ;
* loss monitoring.

---

## Phase 4 — Inference

**Objectif :** générer du texte.

Livrables :

```python
generate("Bonjour")
```

---

## Phase 5 — Agent

Transformer :

```text
Model
```

en :

```text
Agent
```

avec :

* identité ;
* rôle ;
* objectif ;
* mémoire ;
* historique.

---

## Phase 6 — Communication

Créer :

```text
Message
MessageBus
Agent.send()
Agent.receive()
```

Premier test :

```text
Agent A → Agent B → Agent A
```

---

## Phase 7 — Multi-Agent

Créer :

```text
Manager
Developer
Researcher
Critic
```

et les faire collaborer.

---

## Phase 8 — Memory

Ajouter :

```text
Short-term memory
Long-term memory
```

---

## Phase 9 — Tools

Ajouter :

```text
Python
Calculator
Filesystem
```

---

## Phase 10 — Experiments

Tester différentes architectures :

```text
1 agent
      vs
2 agents
      vs
4 agents
      vs
8 agents
```

et différents mécanismes de communication.

---

# 21. Critères de réussite du MVP

Le MVP sera considéré comme fonctionnel lorsque les conditions suivantes seront remplies :

### Modèle

* [ ] modèle Transformer fonctionnel ;
* [ ] environ 1M paramètres ;
* [ ] entraînement possible ;
* [ ] sauvegarde/restauration du modèle ;
* [ ] génération de texte.

### Agent

* [ ] création d'un agent ;
* [ ] rôle configurable ;
* [ ] objectif configurable ;
* [ ] mémoire courte ;
* [ ] utilisation du modèle.

### Communication

* [ ] Agent A peut envoyer un message à B ;
* [ ] B peut répondre ;
* [ ] historique des messages conservé ;
* [ ] plusieurs agents peuvent participer à une tâche.

### Orchestration

* [ ] manager ;
* [ ] délégation ;
* [ ] collecte des réponses ;
* [ ] critique ;
* [ ] synthèse finale.

---

# 22. Contraintes techniques

Le projet doit fonctionner **localement**, avec une priorité donnée à la simplicité.

### Stack

```text
Python
PyTorch
NumPy
SQLite
```

Optionnel plus tard :

```text
Redis
FastAPI
Angular
Docker
```

L'interface web n'est **pas prioritaire dans le MVP**.

---

# 23. Architecture finale envisagée

À terme :

```text
                       ┌───────────────┐
                       │    Angular    │
                       │      UI       │
                       └───────┬───────┘
                               ↓
                       ┌───────────────┐
                       │    FastAPI    │
                       └───────┬───────┘
                               ↓
                    ┌─────────────────────┐
                    │    Orchestrator     │
                    └──────────┬──────────┘
                               ↓
             ┌─────────────────┼─────────────────┐
             ↓                 ↓                 ↓
        ┌─────────┐       ┌─────────┐       ┌─────────┐
        │Manager  │       │Developer│       │ Critic  │
        └────┬────┘       └────┬────┘       └────┬────┘
             │                 │                 │
             └─────────────────┼─────────────────┘
                               ↓
                       ┌───────────────┐
                       │  Message Bus  │
                       └───────┬───────┘
                               ↓
                       ┌───────────────┐
                       │    Memory     │
                       └───────────────┘
                               │
                               ↓
                       ┌───────────────┐
                       │ Transformer   │
                       │    ~1M        │
                       └───────────────┘
```

### Principe fondamental

Le projet doit progresser dans cet ordre :

**`1M Model → Inference → Agent → Memory → Communication → Multi-Agent → Tools → Orchestration`**

C'est important : **on ne cherche pas à résoudre l'intelligence multi-agent avant d'avoir un modèle de base que l'on comprend entièrement.**
