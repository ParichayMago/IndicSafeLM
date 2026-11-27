"""PPO safety alignment with a frozen SFT reference and the validated reward model."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from indicsafelm.alignment.ppo import PPOConfig, run_ppo  # noqa: E402

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int)
    args = parser.parse_args()
    path = ROOT / "configs/train_ppo.json"
    cfg = PPOConfig(**json.loads(path.read_text())) if path.exists() else PPOConfig()
    if args.iterations:
        cfg.iterations = args.iterations
    manifest = run_ppo(cfg, ROOT, log=lambda m: print(m, flush=True))
    print(json.dumps({k: manifest[k] for k in ("run_id", "stop_reason", "best", "wall_minutes")}, indent=2))
