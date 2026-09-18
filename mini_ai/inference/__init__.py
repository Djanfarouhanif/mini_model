"""Moteur d'inférence : prompt → tokens générés → texte.

``TextGenerator`` (PyTorch) et ``DummyModel`` (sans dépendance, pour les
tests et les démonstrations d'agents) exposent la même interface :

    model.generate(prompt: str, max_tokens=..., temperature=..., top_k=..., top_p=...) -> str
"""

from .dummy import DummyModel, LanguageModel


def __getattr__(name):  # import paresseux : évite de charger torch inutilement
    if name in ("TextGenerator", "load_generator"):
        from . import generate as _g

        return getattr(_g, name)
    raise AttributeError(name)


__all__ = ["LanguageModel", "DummyModel", "TextGenerator", "load_generator"]
