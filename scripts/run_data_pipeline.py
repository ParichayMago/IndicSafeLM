#!/usr/bin/env python3
"""Run the deterministic Phase 2 cleaning pipeline on an explicit JSONL input."""

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from indicsafelm.data.pipeline import PipelineError, run_pipeline  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Clean, deduplicate, split, and balance JSONL records. No network access is performed."
    )
    parser.add_argument("--input", required=True, help="JSONL records with source_id/source_document_id/language/text")
    parser.add_argument("--config", default="configs/data_pipeline.json")
    parser.add_argument("--output", required=True, help="Empty or previously completed output directory")
    args = parser.parse_args()
    try:
        report = run_pipeline(ROOT / args.input, ROOT / args.config, ROOT / args.output)
    except (OSError, PipelineError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    print(f"Phase 2 pipeline completed: {report['selected_document_count']} documents retained.")
    print(f"Exact duplicates: {report['duplicate_counts']['exact']}; near duplicates: {report['duplicate_counts']['near']}.")
    print(f"Output: {args.output}; external corpus downloads: none.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
