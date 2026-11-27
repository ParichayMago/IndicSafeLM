"""Supervised instruction tuning from the base checkpoint."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from indicsafelm.training.sft import SFTConfig, run_sft  # noqa: E402

if __name__ == "__main__":
    path = ROOT / "configs/train_sft.json"
    cfg = SFTConfig(**json.loads(path.read_text())) if path.exists() else SFTConfig()
    manifest = run_sft(cfg, ROOT, log=lambda m: print(m, flush=True))
    print(json.dumps({k: manifest[k] for k in ("run_id", "completed_steps", "wall_minutes", "best")}, indent=2))
