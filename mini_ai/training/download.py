"""Téléchargement d'un corpus depuis Hugging Face (``datasets``) vers data/raw/.

Le jeu de données est lu en *streaming* : on ne rapatrie que ce qu'on écrit,
jusqu'à ``max_chars`` caractères — pas besoin de télécharger 20 Go de
Wikipédia pour entraîner un modèle de 1M de paramètres.

    python main.py download-data --preset wiki-fr --max-chars 5000000
    python main.py download-data --dataset roneneldan/TinyStories --text-field text
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class DatasetPreset:
    dataset: str
    config: str | None
    split: str
    text_field: str
    filename: str
    description: str


PRESETS: dict[str, DatasetPreset] = {
    "wiki-fr": DatasetPreset(
        "wikimedia/wikipedia", "20231101.fr", "train", "text", "wikipedia_fr.txt",
        "Wikipédia en français (articles encyclopédiques)",
    ),
    "wiki-en": DatasetPreset(
        "wikimedia/wikipedia", "20231101.en", "train", "text", "wikipedia_en.txt",
        "Wikipédia en anglais",
    ),
    "tinystories": DatasetPreset(
        "roneneldan/TinyStories", None, "train", "text", "tinystories.txt",
        "Histoires courtes en anglais simple — le jeu de données de référence pour les modèles < 10M",
    ),
    "french-books": DatasetPreset(
        "PleIAs/French-PD-Books", None, "train", "complete_text", "french_books.txt",
        "Livres français du domaine public",
    ),
    "python-code": DatasetPreset(
        "bigcode/the-stack-smol", "data/python", "train", "content", "python_code.txt",
        "Extraits de code Python (The Stack, petit échantillon)",
    ),
}


def download(
    dataset: str,
    out_path: str | Path,
    config: str | None = None,
    split: str = "train",
    text_field: str = "text",
    max_chars: int = 5_000_000,
    max_docs: int | None = None,
    min_doc_chars: int = 200,
    verbose: bool = True,
) -> dict:
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover
        raise ImportError("installez d'abord : pip install datasets") from exc

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"[download] {dataset}" + (f" ({config})" if config else "") + f" split={split} champ={text_field!r}")
        print(f"[download] streaming jusqu'à {max_chars:,} caractères → {out_path}")

    ds = load_dataset(dataset, config, split=split, streaming=True)
    written = n_docs = skipped = 0
    with out_path.open("w", encoding="utf-8") as f:
        for row in ds:
            text = row.get(text_field)
            if not isinstance(text, str) or len(text.strip()) < min_doc_chars:
                skipped += 1
                continue
            text = text.strip()
            # Un document = un bloc séparé par une ligne vide (cf. dataset.split_documents).
            f.write(text.replace("\n\n", "\n") + "\n\n")
            written += len(text)
            n_docs += 1
            if verbose and n_docs % 500 == 0:
                print(f"[download] {n_docs:,} documents, {written:,} caractères")
            if written >= max_chars or (max_docs and n_docs >= max_docs):
                break

    if verbose:
        print(f"[download] terminé : {n_docs:,} documents, {written:,} caractères ({skipped} ignorés) → {out_path}")
        print("[download] étape suivante : python main.py prepare-data")
    return {"documents": n_docs, "chars": written, "skipped": skipped, "path": str(out_path)}
