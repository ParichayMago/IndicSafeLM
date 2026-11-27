#!/usr/bin/env python3
"""Inspect a bounded remote sample and write aggregate metadata only.

This command is intentionally guarded for a cloud host. It streams a pinned
dataset revision, never writes raw text, and emits only aggregate quality/PII/
script/domain counters. Do not run it on the development machine.
"""

import argparse
import hashlib
import json
import os
import re
import sys
import unicodedata
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse


EMAIL_RE = re.compile(r"\b[^\s@]+@[^\s@]+\.[^\s@]+\b")
PHONE_RE = re.compile(r"(?<!\d)(?:\+?\d[\s-]?){8,14}\d(?!\d)")
IP_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
URL_RE = re.compile(r"https?://|www\.", re.IGNORECASE)
HTML_RE = re.compile(r"</?[a-z][^>]*>", re.IGNORECASE)
BOILERPLATE_RE = re.compile(
    r"\b(cookie policy|privacy policy|subscribe|javascript|all rights reserved)\b", re.IGNORECASE
)


SCRIPT_RANGES = {
    "hi": (0x0900, 0x097F),
    "bn": (0x0980, 0x09FF),
    "ta": (0x0B80, 0x0BFF),
}


def script_fraction(text: str, language: str) -> float:
    start, end = SCRIPT_RANGES[language]
    letters = [character for character in text if unicodedata.category(character).startswith("L")]
    if not letters:
        return 0.0
    return sum(start <= ord(character) <= end for character in letters) / len(letters)


def domain_for(row: dict) -> str:
    for field in ("domain", "url", "source"):
        value = row.get(field)
        if not value:
            continue
        parsed = urlparse(str(value) if "://" in str(value) else f"https://{value}")
        if parsed.netloc:
            return parsed.netloc.lower()
        return str(value)[:120]
    return "unknown"


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the guarded Phase 1 sample inspection on a cloud host.")
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--revision", required=True, help="Pinned dataset revision or commit SHA")
    parser.add_argument("--language", choices=("hi", "bn", "ta"), required=True)
    parser.add_argument("--sample-docs", type=int, default=1000)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    if os.environ.get("INDICSAFE_CLOUD_RUN") != "1":
        parser.error("refusing to access a dataset payload: set INDICSAFE_CLOUD_RUN=1 only on the approved cloud host")
    if args.sample_docs <= 0:
        parser.error("--sample-docs must be positive")

    try:
        from datasets import load_dataset  # type: ignore
    except ImportError as exc:
        parser.error(f"install the cloud-data dependencies on the remote host: {exc}")

    dataset = load_dataset(args.repo_id, split="train", streaming=True, revision=args.revision)
    seen_hashes = set()
    counters = Counter()
    domain_counts = Counter()
    script_fractions = []
    total_chars = 0
    total_bytes = 0
    inspected = 0

    for row in dataset:
        if row.get("lang") != args.language:
            continue
        text = row.get("text")
        if not isinstance(text, str):
            counters["missing_or_non_string_text"] += 1
            continue
        inspected += 1
        try:
            text_bytes = text.encode("utf-8", errors="strict")
        except UnicodeEncodeError:
            counters["malformed_utf8_documents"] += 1
            continue
        digest = hashlib.sha256(text_bytes).hexdigest()
        counters["exact_duplicate_documents"] += digest in seen_hashes
        seen_hashes.add(digest)
        total_chars += len(text)
        total_bytes += len(text_bytes)
        counters["empty_documents"] += not bool(text.strip())
        counters["very_short_documents_lt_20_chars"] += len(text.strip()) < 20
        counters["very_long_documents_gt_10000_chars"] += len(text) > 10_000
        counters["unicode_normalization_changes"] += unicodedata.normalize("NFC", text) != text
        counters["html_like_documents"] += bool(HTML_RE.search(text))
        counters["url_like_documents"] += bool(URL_RE.search(text))
        counters["boilerplate_marker_documents"] += bool(BOILERPLATE_RE.search(text))
        counters["email_like_documents"] += bool(EMAIL_RE.search(text))
        counters["phone_like_documents"] += bool(PHONE_RE.search(text))
        counters["ipv4_like_documents"] += bool(IP_RE.search(text))
        script_fractions.append(script_fraction(text, args.language))
        domain_counts[domain_for(row)] += 1
        if inspected >= args.sample_docs:
            break

    if inspected == 0:
        raise SystemExit(f"no rows for language {args.language!r} were observed at the pinned revision")

    counters["manual_safety_review_queue"] = inspected
    report = {
        "schema_version": 1,
        "report_type": "phase1_remote_sample_aggregate_only",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "repo_id": args.repo_id,
        "revision": args.revision,
        "language": args.language,
        "documents_inspected": inspected,
        "total_characters": total_chars,
        "total_utf8_bytes": total_bytes,
        "mean_script_fraction": sum(script_fractions) / len(script_fractions),
        "counters": dict(sorted(counters.items())),
        "domain_counts": dict(domain_counts),
        "raw_text_written": False,
        "network_payload_saved_locally": False,
        "safety_note": "Counters do not replace human review; inspect safety samples only in the approved cloud environment.",
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Inspected {inspected} remote {args.language} documents; aggregate report written to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
