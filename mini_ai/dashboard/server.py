"""Tableau de bord d'entraînement : petit serveur HTTP local, sans dépendance.

    python main.py dashboard            # lit checkpoints/train_log.jsonl
    python main.py train --dashboard    # l'ouvre pendant l'entraînement

Routes :
    /            la page (page.html)
    /api/log     tous les enregistrements du log, en JSON
    /api/status  taille du log, dernière modification
"""

from __future__ import annotations

import json
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

PAGE_PATH = Path(__file__).with_name("page.html")


def read_log(path: str | Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    records: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # ligne en cours d'écriture
    return records


def make_handler(log_path: Path):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, body: bytes, content_type: str, status: int = 200) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802
            route = self.path.split("?", 1)[0]
            if route in ("/", "/index.html"):
                self._send(PAGE_PATH.read_bytes(), "text/html; charset=utf-8")
            elif route == "/api/log":
                body = json.dumps(read_log(log_path), ensure_ascii=False).encode("utf-8")
                self._send(body, "application/json; charset=utf-8")
            elif route == "/api/status":
                st = log_path.stat() if log_path.exists() else None
                body = json.dumps({"exists": st is not None, "size": st.st_size if st else 0, "mtime": st.st_mtime if st else 0, "path": str(log_path)})
                self._send(body.encode("utf-8"), "application/json")
            else:
                self._send(b"not found", "text/plain", 404)

        def log_message(self, *args) -> None:  # silence les logs HTTP
            pass

    return Handler


def serve(log_path: str | Path, host: str = "127.0.0.1", port: int = 8765, open_browser: bool = True) -> None:
    server = ThreadingHTTPServer((host, port), make_handler(Path(log_path)))
    url = f"http://{host}:{port}/"
    print(f"[dashboard] {url}  (log : {log_path})  Ctrl+C pour arrêter")
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def start_in_background(log_path: str | Path, host: str = "127.0.0.1", port: int = 8765, open_browser: bool = True) -> ThreadingHTTPServer:
    """Démarre le serveur dans un thread daemon (utilisé par ``train --dashboard``)."""
    server = ThreadingHTTPServer((host, port), make_handler(Path(log_path)))
    thread = threading.Thread(target=server.serve_forever, daemon=True, name="dashboard")
    thread.start()
    url = f"http://{host}:{port}/"
    print(f"[dashboard] {url}")
    if open_browser:
        webbrowser.open(url)
    return server
