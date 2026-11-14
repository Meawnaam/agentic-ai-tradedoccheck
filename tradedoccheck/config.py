"""Settings and a tiny .env loader (no python-dotenv dependency)."""
from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_dotenv(path: str | Path | None = None) -> None:
    """Load KEY=VALUE lines from .env into os.environ (existing variables win)."""
    path = Path(path) if path else PROJECT_ROOT / ".env"
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip().removeprefix("export ").strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


DEFAULT_MODELS = {
    "gemini": "gemini-3.8-flash",
    "anthropic": "claude-sonnet-5-5",
    "ollama": "llama3.1",
}
