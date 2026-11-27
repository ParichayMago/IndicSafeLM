"""Lightweight run-ID and metadata helpers for reproducible experiment records."""

import datetime as dt
import json
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional


def git_sha(root: Path) -> str:
    """Return a short Git revision, or ``nogit`` outside a Git worktree."""

    try:
        value = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "nogit"
    return value or "nogit"


def make_run_id(stage: str, language: str, seed: int, root: Path, now: Optional[dt.datetime] = None) -> str:
    """Create the project run ID: timestamp, stage, language, seed, and Git SHA."""

    now = now or dt.datetime.now(dt.timezone.utc)
    timestamp = now.strftime("%Y%m%d-%H%M%S")
    return f"{timestamp}_{stage}_{language}_{seed}_{git_sha(root)}"


def write_metadata(path: Path, metadata: Dict[str, Any]) -> None:
    """Write stable, human-readable JSON metadata, creating parent directories."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")

