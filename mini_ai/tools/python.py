"""Exécution de code Python dans un sous-processus isolé avec délai maximal.

    python.execute("print(sum(range(10)))")  → "45"
"""

from __future__ import annotations

import subprocess
import sys

from .base import Tool


class PythonTool(Tool):
    name = "python"
    description = "Exécute un extrait de code Python et renvoie sa sortie standard."

    def __init__(self, timeout: float = 5.0, max_output: int = 2000):
        self.timeout = timeout
        self.max_output = max_output

    def execute(self, input: str) -> str:
        code = input.strip()
        if code.startswith("```"):  # accepte les blocs markdown
            code = code.strip("`")
            if code.startswith("python"):
                code = code[len("python") :]
        if not code:
            raise ValueError("code vide")
        try:
            proc = subprocess.run(
                [sys.executable, "-I", "-c", code],  # -I : mode isolé (pas de site/env utilisateur)
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
        except subprocess.TimeoutExpired:
            raise TimeoutError(f"exécution interrompue après {self.timeout}s")
        out = proc.stdout
        if proc.returncode != 0:
            err = proc.stderr.strip().splitlines()
            out += ("\n" if out else "") + (err[-1] if err else f"code de sortie {proc.returncode}")
        out = out.strip()
        if len(out) > self.max_output:
            out = out[: self.max_output] + "…"
        return out or "(aucune sortie)"
