"""Config loading. Single source of truth = configs/default.yaml."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent


# --- generators -----------------------------------------------------------
# The reference model keeps the flat results/ layout every existing notebook
# reads. Every other generator writes under results/<tag>/, so resume=True
# can never find another model's rows under a shared setting label.
# Adding a model = one entry here + one overlay in configs/models/.
REFERENCE_GENERATOR = "Qwen/Qwen2.5-3B-Instruct"
MODEL_TAGS = {
    "Qwen/Qwen2.5-3B-Instruct": "qwen25_3b",
    "meta-llama/Llama-3.2-3B-Instruct": "llama32_3b",
    "Qwen/Qwen2.5-7B-Instruct": "qwen25_7b",
}


def _deep_merge(base: Dict[str, Any], over: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: str | Path = "configs/default.yaml",
                model: str | None = None) -> Dict[str, Any]:
    """default.yaml, optionally overlaid with configs/models/<model>.yaml.

    default.yaml stays frozen; a model arm changes only what its overlay names.
    """
    path = Path(path)
    if not path.is_absolute():
        path = REPO_ROOT / path
    with open(path) as fh:
        cfg = yaml.safe_load(fh)
    if model is None:
        return cfg

    with open(REPO_ROOT / "configs" / "models" / f"{model}.yaml") as fh:
        cfg = _deep_merge(cfg, yaml.safe_load(fh))
    gen = cfg["models"]["generator"]
    if MODEL_TAGS.get(gen) != model:
        raise ValueError(
            f"overlay {model!r} sets generator {gen!r}, tagged "
            f"{MODEL_TAGS.get(gen)!r} in MODEL_TAGS. Fix one of them.")
    return cfg


def resolve(rel: str | Path) -> Path:
    """Repo-relative path -> absolute, creating parents for outputs."""
    p = Path(rel)
    return p if p.is_absolute() else REPO_ROOT / p


def results_dir(generator: str) -> Path:
    """Directory holding this generator's result files."""
    if generator not in MODEL_TAGS:
        raise ValueError(
            f"unknown generator {generator!r}. Add it to MODEL_TAGS in "
            f"src/config.py before writing any results.")
    if generator == REFERENCE_GENERATOR:
        return REPO_ROOT / "results"
    return REPO_ROOT / "results" / MODEL_TAGS[generator]


def check_out_path(out_csv: str | Path | None, generator: str) -> None:
    """Fail BEFORE generating if out_csv would put two generators in one file.

    resume=True keys on (setting, qid). Llama rows aimed at a Qwen CSV under a
    shared label like 'baseline' would be skipped as already done — silently,
    with the header check passing.
    """
    if out_csv is None:
        return
    got = Path(out_csv).resolve().parent
    want = results_dir(generator).resolve()
    if got != want:
        raise ValueError(
            f"out_csv={str(out_csv)!r} resolves to {got}, but {generator!r} "
            f"writes to {want}. Build the path with results_dir(generator).")