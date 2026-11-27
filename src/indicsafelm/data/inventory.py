"""Validate and summarize the Phase 1 corpus manifest without data access."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List


SUPPORTED_LANGUAGES = ("hi", "bn", "ta")
REQUIRED_FIELDS = (
    "manifest_schema_version",
    "source_id",
    "language",
    "language_name",
    "source_release",
    "manifest_accessed_at",
    "acquisition_date",
    "landing_url",
    "download_method",
    "format",
    "source_type",
    "role",
    "permission_basis",
    "license",
    "license_status",
    "attribution_requirements",
    "redistribution_restrictions",
    "checksum_sha256",
    "access_probe_status",
    "representative_sample_status",
    "estimated_documents",
    "estimated_tokens",
    "estimated_bytes",
    "estimated_volume_scope",
    "selection_status",
)


class ManifestError(ValueError):
    """Raised when a corpus manifest violates the Phase 1 contract."""


def load_manifest(path: Path) -> List[Dict[str, Any]]:
    """Load JSONL records from *path* without contacting any remote URL."""

    records: List[Dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if not raw_line.strip():
                continue
            try:
                record = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise ManifestError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(record, dict):
                raise ManifestError(f"{path}:{line_number}: record must be a JSON object")
            record["_line_number"] = line_number
            records.append(record)
    return records


def validate_manifest(records: Iterable[Dict[str, Any]]) -> None:
    """Validate required metadata, language coverage, and unique identifiers."""

    seen_ids = set()
    seen_languages = set()
    for record in records:
        line_number = record.get("_line_number", "?")
        missing = [field for field in REQUIRED_FIELDS if field not in record]
        if missing:
            raise ManifestError(f"line {line_number}: missing fields: {', '.join(missing)}")
        source_id = record["source_id"]
        if not isinstance(source_id, str) or not source_id:
            raise ManifestError(f"line {line_number}: source_id must be a non-empty string")
        if source_id in seen_ids:
            raise ManifestError(f"line {line_number}: duplicate source_id: {source_id}")
        seen_ids.add(source_id)
        language = record["language"]
        if language not in SUPPORTED_LANGUAGES:
            raise ManifestError(f"line {line_number}: unsupported language: {language}")
        seen_languages.add(language)
        if record["manifest_schema_version"] != 1:
            raise ManifestError(f"line {line_number}: unsupported manifest schema version")
        if not str(record["landing_url"]).startswith("https://"):
            raise ManifestError(f"line {line_number}: landing_url must use HTTPS")
        for nullable_number in ("estimated_documents", "estimated_tokens", "estimated_bytes"):
            value = record[nullable_number]
            if value is not None and (not isinstance(value, int) or value < 0):
                raise ManifestError(f"line {line_number}: {nullable_number} must be null or a non-negative integer")
        if record["checksum_sha256"] is not None:
            checksum = str(record["checksum_sha256"])
            if len(checksum) != 64 or any(character not in "0123456789abcdef" for character in checksum.lower()):
                raise ManifestError(f"line {line_number}: checksum_sha256 must be a 64-character hex string or null")
    if seen_languages != set(SUPPORTED_LANGUAGES):
        missing_languages = sorted(set(SUPPORTED_LANGUAGES) - seen_languages)
        raise ManifestError(f"manifest is missing language records: {', '.join(missing_languages)}")


def _sum_known(records: Iterable[Dict[str, Any]], field: str) -> int:
    return sum(record[field] for record in records if isinstance(record.get(field), int))


def summarize_manifest(records: List[Dict[str, Any]], target_tokens: int = 100_000_000) -> Dict[str, Any]:
    """Build a deterministic inventory summary from manifest metadata only."""

    validate_manifest(records)
    by_language: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_language[record["language"]].append(record)

    language_summary: Dict[str, Dict[str, Any]] = {}
    for language in SUPPORTED_LANGUAGES:
        language_records = by_language[language]
        selected = [item for item in language_records if item["selection_status"] == "selected_for_cloud_pilot"]
        language_summary[language] = {
            "language_name": language_records[0]["language_name"],
            "record_count": len(language_records),
            "selected_record_count": len(selected),
            "selected_source_ids": [item["source_id"] for item in selected],
            "target_training_tokens": target_tokens,
            "selected_published_token_estimate": _sum_known(selected, "estimated_tokens"),
            "all_candidate_published_token_estimate": _sum_known(language_records, "estimated_tokens"),
            "known_document_estimate": _sum_known(language_records, "estimated_documents"),
            "known_byte_estimate": _sum_known(
                [item for item in language_records if item.get("estimated_volume_scope") == "language_only"],
                "estimated_bytes",
            ),
            "records_without_document_estimate": sum(item["estimated_documents"] is None for item in language_records),
            "records_without_token_estimate": sum(item["estimated_tokens"] is None for item in language_records),
            "records_without_byte_estimate": sum(item["estimated_bytes"] is None for item in language_records),
            "sample_inspection_statuses": dict(Counter(item["representative_sample_status"] for item in language_records)),
        }

    source_overview = []
    for record in records:
        source_overview.append(
            {
                "source_id": record["source_id"],
                "language": record["language"],
                "source_release": record["source_release"],
                "selection_status": record["selection_status"],
                "license": record["license"],
                "license_status": record["license_status"],
                "access_probe_status": record["access_probe_status"],
                "representative_sample_status": record["representative_sample_status"],
                "estimated_documents": record["estimated_documents"],
                "estimated_tokens": record["estimated_tokens"],
                "estimated_bytes": record["estimated_bytes"],
                "estimated_volume_scope": record["estimated_volume_scope"],
            }
        )

    manifest_dates = sorted({record["manifest_accessed_at"] for record in records})
    return {
        "schema_version": 1,
        "inventory_type": "phase1_metadata_only",
        "manifest_access_dates": manifest_dates,
        "execution_policy": {
            "network_access": False,
            "corpus_payload_access": False,
            "raw_data_downloaded_locally": False,
            "checksums_pending_cloud_acquisition": True,
        },
        "record_count": len(records),
        "languages": list(SUPPORTED_LANGUAGES),
        "selected_source_ids": [item["source_id"] for item in records if item["selection_status"] == "selected_for_cloud_pilot"],
        "language_summary": language_summary,
        "source_overview": source_overview,
        "cloud_gate": {
            "required_before_acquisition": [
                "pin_dataset_revision_or_dated_dump",
                "inspect_remote_sample_for_encoding_boilerplate_pii_unsafe_content_and_domain_balance",
                "record_source_checksum_and_access_date",
                "approve_license_and_source_rights",
            ],
            "current_status": "sample_inspection_and_checksums_pending_cloud_only_gate",
        },
    }
