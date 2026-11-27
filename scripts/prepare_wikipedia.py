"""Convert dated Wikipedia parquet snapshots into Phase 2 pipeline input.

Input: one shard per language from the Hugging Face ``wikimedia/wikipedia``
dataset, snapshot ``20231101`` (CC BY-SA 4.0 / GFDL). The shards are verified
against the SHA-256 recorded in ``data/manifests/local_run_sources.jsonl``
before any text is read (FR-D01: a source without an approved record is
rejected).

Output: JSONL records with ``source_id``, ``source_document_id``, ``language``,
``domain`` and ``text`` that ``scripts/run_data_pipeline.py`` consumes. Articles
are taken in shard order until a per-language character budget is met, so the
selection is deterministic for a fixed shard and budget.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
LANGUAGES = ("hi", "bn", "ta")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_sources(path: Path) -> dict:
    sources = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            sources[record["language"]] = record
    return sources


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=ROOT / "data/raw/wikipedia-20231101")
    parser.add_argument("--sources", type=Path, default=ROOT / "data/manifests/local_run_sources.jsonl")
    parser.add_argument("--output", type=Path, default=ROOT / "data/interim/wikipedia-20231101.jsonl")
    parser.add_argument("--max-chars-per-language", type=int, default=60_000_000)
    args = parser.parse_args()

    sources = load_sources(args.sources)
    summary = {}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".jsonl.tmp")
    with temporary.open("w", encoding="utf-8") as out:
        for language in LANGUAGES:
            source = sources.get(language)
            if source is None or source.get("selection_status") != "selected_for_local_run":
                print(f"[{language}] no approved source record; skipping", file=sys.stderr)
                return 2
            shard = args.raw_dir / source["local_filename"]
            actual = sha256_file(shard)
            if actual != source["checksum_sha256"]:
                print(f"[{language}] checksum mismatch for {shard}: {actual}", file=sys.stderr)
                return 3
            parquet = pq.ParquetFile(shard)
            chars = docs = 0
            for group in range(parquet.num_row_groups):
                table = parquet.read_row_group(group, columns=["id", "title", "text"])
                for doc_id, text in zip(table.column("id").to_pylist(), table.column("text").to_pylist()):
                    record = {
                        "source_id": source["source_id"],
                        "source_document_id": str(doc_id),
                        "language": language,
                        "domain": "wikipedia",
                        "text": text,
                    }
                    out.write(json.dumps(record, ensure_ascii=False) + "\n")
                    chars += len(text)
                    docs += 1
                if chars >= args.max_chars_per_language:
                    break
            summary[language] = {"documents": docs, "characters": chars, "shard_sha256": actual}
            print(f"[{language}] {docs} documents, {chars:,} characters")
    temporary.replace(args.output)
    (args.output.with_suffix(".summary.json")).write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
