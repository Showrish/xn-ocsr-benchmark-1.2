"""Load the benchmark configuration and resolve paths and credentials.

Paths in config/config.yaml are resolved against the repo root, so the code works
on any machine without hardcoded paths. Credentials are read only from environment
variables (or an untracked .env file), never from files in the repo.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "config" / "config.yaml"


def setup_logging() -> None:
    """Use one consistent log format across every pipeline step."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def resolve_path(value: str | os.PathLike) -> Path:
    """Expand ~ and $VARS, then anchor relative paths at the repo root."""
    path = Path(os.path.expandvars(os.path.expanduser(str(value))))
    return path if path.is_absolute() else REPO_ROOT / path


def load_config(config_path: str | os.PathLike = DEFAULT_CONFIG) -> dict[str, Any]:
    """Read the YAML config and convert every entry under `paths` to an absolute Path."""
    with open(resolve_path(config_path), "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    config["paths"] = {name: resolve_path(value) for name, value in config["paths"].items()}
    return config


def load_env_file(env_path: Path = REPO_ROOT / ".env") -> None:
    """Load KEY=VALUE lines from an untracked .env file, without overriding real env vars."""
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def require_env(name: str) -> str:
    """Return a required environment variable, failing with a clear message if it is missing."""
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"Environment variable {name} is not set. See .env.example.")
    return value
