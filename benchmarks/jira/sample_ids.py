"""Locate and read the sampled Jira issue identifiers shipped with the repository.

The default is the shipped file ``data/jira/sample_issue_ids.txt``. A relative name is looked up
as given (current directory), then inside the ``--processed`` directory, then relative to the
repository root, so older invocations that copy the file into ``processed/`` keep working.
"""

from __future__ import annotations

from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SAMPLE_IDS = "data/jira/sample_issue_ids.txt"


def resolve_sample_ids(processed: Path, name: str = DEFAULT_SAMPLE_IDS) -> Path:
    p = Path(name)
    candidates = [p] if p.is_absolute() else [Path.cwd() / p, Path(processed) / p, _REPO_ROOT / p]
    for c in candidates:
        if c.is_file():
            return c
    raise FileNotFoundError(f"sample id file {name!r} not found; looked in: " + ", ".join(str(c) for c in candidates))


def read_sample_ids(processed: Path, name: str = DEFAULT_SAMPLE_IDS) -> list[str]:
    return [l.strip() for l in resolve_sample_ids(processed, name).read_text().splitlines() if l.strip()]
