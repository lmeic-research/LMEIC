"""
Path resolver for the LMEIC pipeline.
Version: v1 | Date: 2026-05-26

Loads configs/paths.yaml, resolves ${var} substitutions, and exposes
a typed accessor. All scripts SHOULD use this module
instead of hardcoding any absolute path.
"""

from __future__ import annotations

import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

_REPO_ROOT_DEFAULT = Path(__file__).resolve().parents[2]
_VAR_RE = re.compile(r"\$\{([a-zA-Z_][a-zA-Z0-9_]*)\}")


def _substitute(value: Any, registry: dict[str, str]) -> Any:
    if isinstance(value, str):

        def repl(match: re.Match[str]) -> str:
            key = match.group(1)
            if key not in registry:
                raise KeyError(f"Undefined variable in paths.yaml: ${{{key}}}")
            return str(registry[key])

        prev = None
        out = value
        while prev != out:
            prev = out
            out = _VAR_RE.sub(repl, out)
        return out
    if isinstance(value, dict):
        return {k: _substitute(v, registry) for k, v in value.items()}
    if isinstance(value, list):
        return [_substitute(v, registry) for v in value]
    return value


def load_paths(config_path: Path | str | None = None) -> dict[str, Any]:
    """Load and resolve paths.yaml."""
    if config_path is None:
        env_override = os.environ.get("LMEIC_PATHS_YAML")
        if env_override:
            config_path = Path(env_override)
        else:
            config_path = _REPO_ROOT_DEFAULT / "configs" / "paths.yaml"
    config_path = Path(config_path).resolve()

    with config_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    registry: dict[str, str] = {"repo_root": str(raw.get("repo_root", _REPO_ROOT_DEFAULT))}
    resolved = _substitute(raw, registry)
    resolved["_config_path"] = str(config_path)
    return resolved


def make_run_dir(paths: dict[str, Any], run_id: str | None = None) -> Path:
    """Create a timestamped run directory under runs_base_dir."""
    if run_id is None:
        run_id = "run_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    runs_base = Path(paths["output"]["runs_base_dir"])
    run_dir = runs_base / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def phase0_subdir(run_dir: Path, paths: dict[str, Any], stage_key: str) -> Path:
    """Resolve a phase0 sub-step directory under a run."""
    rel = paths["output"]["phase0_subdirs"].get(stage_key)
    if rel is None:
        raise KeyError(f"Unknown phase0 stage: {stage_key}")
    sub = run_dir / rel
    sub.mkdir(parents=True, exist_ok=True)
    return sub


def latest_run_dir(paths: dict[str, Any]) -> Path | None:
    """Return the most recent run directory under runs_base_dir, or None."""
    runs_base = Path(paths["output"]["runs_base_dir"])
    if not runs_base.exists():
        return None
    candidates = sorted(
        (p for p in runs_base.glob("run_*") if p.is_dir()),
        reverse=True,
    )
    return candidates[0] if candidates else None
