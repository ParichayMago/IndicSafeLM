"""Train the pairwise reward model from the SFT checkpoint and apply the PPO quality gate."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from indicsafelm.reward.train import RewardConfig, run_reward  # noqa: E402

if __name__ == "__main__":
    path = ROOT / "configs/train_reward.json"
    cfg = RewardConfig(**json.loads(path.read_text())) if path.exists() else RewardConfig()
    report = run_reward(cfg, ROOT, log=lambda m: print(m, flush=True))
    print("quality gate passed:", report["quality_gate"]["passed"])
