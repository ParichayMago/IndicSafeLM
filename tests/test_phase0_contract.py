"""Dependency-free contract tests; full ML tests are added in later phases."""

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read_json(path):
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def test_project_languages_and_split_sum():
    project = read_json("configs/project.json")
    assert project["languages"] == ["hi", "bn", "ta"]
    assert sum(project["data"]["split_ratios"].values()) == 1.0


def test_experiment_matrix_has_required_checkpoints():
    experiments = read_json("configs/experiments.json")
    ids = {item["id"] for item in experiments["matrix"]}
    assert {"base-pretrain", "sft", "reward-model", "ppo", "keyword-filter"} <= ids


def test_evaluation_is_frozen_and_has_benign_context():
    evaluation = read_json("configs/evaluation.json")
    assert evaluation["evaluation_set_policy"]["frozen_before_training"] is True
    categories = {item["id"] for item in evaluation["categories"]}
    assert "benign_sensitive_context" in categories

