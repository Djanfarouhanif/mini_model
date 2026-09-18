"""Hyper-paramètres du Transformer.

Budget de paramètres (poids de sortie liés à l'embedding) :

    tokens   : vocab_size × n_embd
    position : block_size × n_embd          (si pos_type == "learned")
    par bloc : 12 × n_embd²  + biais/LayerNorm
    total    ≈ vocab_size×d + 12×L×d²

Preset par défaut → ~1,09M :  vocab 4096, d 96, 6 couches, 4 têtes, ctx 256.
Le preset "prd" (vocab 8000, d 128, 6 couches) dépasse 2M : l'embedding seul
fait déjà 1,02M, c'est pourquoi la configuration a été ajustée comme le
PRD le prévoit (§6).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class GPTConfig:
    vocab_size: int = 4096
    block_size: int = 256  # longueur de contexte
    n_layer: int = 6
    n_head: int = 4
    n_embd: int = 96
    dropout: float = 0.1
    bias: bool = True
    tie_weights: bool = True  # lm_head partage la matrice d'embedding
    pos_type: str = "learned"  # "learned" | "rope"
    mlp_ratio: int = 4

    def __post_init__(self) -> None:
        if self.n_embd % self.n_head != 0:
            raise ValueError("n_embd doit être divisible par n_head")
        if self.pos_type not in ("learned", "rope"):
            raise ValueError("pos_type doit valoir 'learned' ou 'rope'")
        if self.pos_type == "rope" and (self.n_embd // self.n_head) % 2:
            raise ValueError("RoPE requiert une dimension de tête paire")

    @property
    def head_dim(self) -> int:
        return self.n_embd // self.n_head

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "GPTConfig":
        known = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**known)

    def estimated_parameters(self) -> int:
        """Estimation analytique, à comparer avec count_parameters(model)."""
        d = self.n_embd
        b = 1 if self.bias else 0
        emb = self.vocab_size * d
        pos = self.block_size * d if self.pos_type == "learned" else 0
        attn = (d * 3 * d + 3 * d * b) + (d * d + d * b)
        mlp = (d * self.mlp_ratio * d + self.mlp_ratio * d * b) + (self.mlp_ratio * d * d + d * b)
        ln = 2 * (d + d * b)
        block = attn + mlp + ln
        ln_f = d + d * b
        head = 0 if self.tie_weights else self.vocab_size * d
        return emb + pos + self.n_layer * block + ln_f + head


PRESETS: dict[str, GPTConfig] = {
    # ~1,09M — configuration par défaut du projet.
    "mini": GPTConfig(),
    # ~0,83M — garde les 8 000 tokens du PRD en réduisant l'embedding.
    "mini-8k": GPTConfig(vocab_size=8000, n_embd=64, n_layer=6, n_head=4),
    # Configuration littérale du PRD (~2,2M) : gardée pour comparaison.
    "prd": GPTConfig(vocab_size=8000, n_embd=128, n_layer=6, n_head=4),
    # Tout petit modèle pour les tests unitaires.
    "tiny": GPTConfig(vocab_size=512, block_size=64, n_layer=2, n_head=2, n_embd=32, dropout=0.0),
}
