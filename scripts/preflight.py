#!/usr/bin/env python3
"""Check the Phase 0 contract without requiring ML dependencies."""

import argparse
import json
import platform
import shutil
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from indicsafelm.data.inventory import load_manifest, validate_manifest  # noqa: E402


def load_json(relative_path: str):
    path = ROOT / relative_path
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strict", action="store_true", help="fail on training-host requirements")
    args = parser.parse_args()

    errors = []
    warnings = []

    for path in (
        "configs/project.json",
        "configs/experiments.json",
        "configs/evaluation.json",
        "configs/data_pipeline.json",
    ):
        try:
            document = load_json(path)
            if document.get("schema_version") != 1:
                errors.append(f"{path}: unsupported schema_version")
        except (OSError, ValueError) as exc:
            errors.append(f"{path}: {exc}")

    project = load_json("configs/project.json")
    required_languages = {"hi", "bn", "ta"}
    actual_languages = set(project.get("languages", []))
    if actual_languages != required_languages:
        errors.append(f"project languages must be {sorted(required_languages)}; got {sorted(actual_languages)}")

    manifest_path = ROOT / project["paths"]["source_manifest"]
    try:
        validate_manifest(load_manifest(manifest_path))
        print(f"corpus manifest: valid ({manifest_path.relative_to(ROOT)})")
    except (OSError, ValueError, KeyError) as exc:
        errors.append(f"corpus manifest: {exc}")

    python_ok = sys.version_info >= (3, 11)
    if not python_ok:
        warnings.append(f"Python {platform.python_version()} is below the training target 3.11+")

    try:
        import torch  # type: ignore

        print(f"torch: {torch.__version__}; CUDA available: {torch.cuda.is_available()}")
        if not torch.cuda.is_available():
            warnings.append("CUDA is unavailable; this host cannot run the planned full training workload")
    except ImportError:
        warnings.append("PyTorch is not installed; only configuration and lightweight smoke checks can run")

    if shutil.which("git") is None:
        warnings.append("git executable not found; run metadata cannot include a commit SHA")

    for directory in ("configs", "docs", "src/indicsafelm", "scripts", "data/manifests", "tests"):
        if not (ROOT / directory).is_dir():
            errors.append(f"missing required directory: {directory}")

    for message in warnings:
        print(f"WARNING: {message}")
    for message in errors:
        print(f"ERROR: {message}")

    if errors or (args.strict and (not python_ok or warnings)):
        return 1
    print("Phase 0 preflight passed (non-strict mode).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
