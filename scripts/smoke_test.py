#!/usr/bin/env python3
"""Run a dependency-light Phase 0 end-to-end reproducibility smoke test."""

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from indicsafelm.reproducibility import seed_python, stable_sample  # noqa: E402
from indicsafelm.run_tracking import make_run_id, write_metadata  # noqa: E402


def load(relative_path: str):
    with (ROOT / relative_path).open(encoding="utf-8") as handle:
        return json.load(handle)


def main() -> int:
    project = load("configs/project.json")
    experiments = load("configs/experiments.json")
    evaluation = load("configs/evaluation.json")

    assert project["languages"] == ["hi", "bn", "ta"]
    experiment_ids = {item["id"] for item in experiments["matrix"]}
    assert {"smoke", "base-pretrain", "sft", "reward-model", "ppo", "keyword-filter"} <= experiment_ids
    assert set(evaluation["evaluation_set_policy"]["languages"]) == set(project["languages"])
    assert evaluation["evaluation_set_policy"]["frozen_before_training"] is True

    seed = project["reproducibility"]["stage_seeds"]["evaluation"]
    seed_python(seed, set_hash_seed=True)
    first = stable_sample(seed)
    second = stable_sample(seed)
    assert first == second, "seed policy is not deterministic"

    run_id = make_run_id("smoke", "multi", seed, ROOT)
    metadata = {
        "run_id": run_id,
        "stage": "smoke",
        "languages": project["languages"],
        "seed": seed,
        "sample": first,
        "config_files": [
            "configs/project.json",
            "configs/experiments.json",
            "configs/evaluation.json",
        ],
    }
    output = ROOT / "artifacts" / "smoke" / "metadata.json"
    write_metadata(output, metadata)
    loaded = json.loads(output.read_text(encoding="utf-8"))
    assert loaded == metadata
    print(f"Smoke test passed; metadata written to {output.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

