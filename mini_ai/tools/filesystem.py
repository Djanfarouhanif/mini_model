"""Accès au système de fichiers, confiné à un dossier racine (sandbox).

    filesystem.execute("list .")
    filesystem.execute("read notes.txt")
    filesystem.execute("write notes.txt :: contenu du fichier")
    filesystem.execute("search django")   # recherche texte dans les fichiers
"""

from __future__ import annotations

from pathlib import Path

from .base import Tool


class FilesystemTool(Tool):
    name = "filesystem"
    description = "Lit, liste, cherche ou écrit des fichiers dans un dossier de travail (list/read/write/search)."

    def __init__(self, root: str | Path | None = None, max_read: int = 4000, allow_write: bool = True):
        self.root = Path(root or Path.cwd() / "data" / "workspace").resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.max_read = max_read
        self.allow_write = allow_write

    def _resolve(self, rel: str) -> Path:
        path = (self.root / rel.strip().strip("'\"")).resolve()
        if path != self.root and self.root not in path.parents:
            raise PermissionError(f"chemin hors du dossier autorisé : {rel}")
        return path

    def execute(self, input: str) -> str:
        parts = input.strip().split(None, 1)
        if not parts:
            raise ValueError("commande vide (list|read|write|search)")
        cmd = parts[0].lower()
        arg = parts[1] if len(parts) > 1 else ""

        if cmd in ("list", "ls"):
            path = self._resolve(arg or ".")
            if not path.exists():
                raise FileNotFoundError(arg)
            entries = sorted(path.iterdir(), key=lambda p: (p.is_file(), p.name))
            return "\n".join(f"{'📄' if p.is_file() else '📁'} {p.relative_to(self.root)}" for p in entries) or "(vide)"

        if cmd in ("read", "cat"):
            path = self._resolve(arg)
            if not path.is_file():
                raise FileNotFoundError(arg)
            text = path.read_text(encoding="utf-8", errors="replace")
            return text[: self.max_read] + ("…" if len(text) > self.max_read else "")

        if cmd == "write":
            if not self.allow_write:
                raise PermissionError("écriture désactivée")
            if "::" not in arg:
                raise ValueError("syntaxe : write <chemin> :: <contenu>")
            rel, content = arg.split("::", 1)
            path = self._resolve(rel)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content.strip(), encoding="utf-8")
            return f"écrit {len(content.strip())} caractères dans {path.relative_to(self.root)}"

        if cmd in ("search", "grep"):
            needle = arg.lower()
            if not needle:
                raise ValueError("syntaxe : search <texte>")
            hits: list[str] = []
            for p in sorted(self.root.rglob("*")):
                if p.is_file() and p.suffix.lower() in {".txt", ".md", ".py", ".json", ".csv"}:
                    for i, line in enumerate(p.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
                        if needle in line.lower():
                            hits.append(f"{p.relative_to(self.root)}:{i}: {line.strip()[:120]}")
                            if len(hits) >= 20:
                                return "\n".join(hits)
            return "\n".join(hits) or "(aucun résultat)"

        raise ValueError(f"commande inconnue : {cmd} (list|read|write|search)")
