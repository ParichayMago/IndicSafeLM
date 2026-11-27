#!/usr/bin/env python3
"""Validate and summarize the Phase 1 corpus manifest without downloading data."""

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from indicsafelm.data.inventory import load_manifest, summarize_manifest  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create a metadata-only Phase 1 inventory; this command never makes network requests."
    )
    parser.add_argument("--manifest", default="data/manifests/sources.jsonl")
    parser.add_argument("--output", default="data/manifests/inventory.json")
    parser.add_argument("--target-tokens", type=int, default=100_000_000)
    args = parser.parse_args()

    manifest_path = ROOT / args.manifest
    output_path = ROOT / args.output
    records = load_manifest(manifest_path)
    inventory = summarize_manifest(records, target_tokens=args.target_tokens)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(inventory, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    selected = len(inventory["selected_source_ids"])
    print(f"Validated {inventory['record_count']} manifest records across {len(inventory['languages'])} languages.")
    print(f"Selected cloud-pilot records: {selected}; local corpus downloads: none.")
    print(f"Metadata inventory written to {output_path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
