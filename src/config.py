"""Config loading. Single source of truth = configs/default.yaml."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent


def load_config(path: str | Path = "configs/default.yaml") -> Dict[str, Any]:
    path = Path(path)
    if not path.is_absolute():
        path = REPO_ROOT / path
    with open(path) as fh:
        return yaml.safe_load(fh)


def resolve(rel: str | Path) -> Path:
    """Repo-relative path -> absolute, creating parents for outputs."""
    p = Path(rel)
    return p if p.is_absolute() else REPO_ROOT / p