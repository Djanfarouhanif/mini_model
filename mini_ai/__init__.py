"""MiniAI — un petit modèle de langage (~1M paramètres) qui sert de cerveau
à plusieurs agents capables de communiquer, mémoriser et collaborer.

Ordre de construction (voir prd.md) :

    Dataset → Tokenizer → Model → Inference → Agent → Memory
            → Communication → Multi-Agent → Tools → Orchestration
"""

__version__ = "0.1.0"
