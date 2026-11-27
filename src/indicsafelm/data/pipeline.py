"""Deterministic corpus cleaning, deduplication, and split pipeline.

The implementation is intentionally dependency-free so that it can be smoke
tested on the development machine. It accepts JSONL records, writes text only
to an explicitly supplied output directory, and keeps all split manifests
ID-only. Large corpus acquisition remains a separate cloud-only operation.
"""

from __future__ import annotations

import copy
import hashlib
import html
import json
import math
import os
import re
import shutil
import tempfile
import unicodedata
from collections import Counter, defaultdict
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:  # numpy is optional: it only accelerates MinHash on real corpora.
    import numpy as _np
except ImportError:  # pragma: no cover - exercised on minimal installs
    _np = None


PIPELINE_VERSION = "phase2-v1.2"
SUPPORTED_LANGUAGES = ("hi", "bn", "ta")
LANGUAGE_SCRIPT_RANGES = {
    "hi": (0x0900, 0x097F),
    "bn": (0x0980, 0x09FF),
    "ta": (0x0B80, 0x0BFF),
}
TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)
EMAIL_RE = re.compile(r"\b[^\s@]+@[^\s@]+\.[^\s@]+\b")
PHONE_RE = re.compile(r"(?<!\d)(?:\+?\d[\s-]?){8,14}\d(?!\d)")
IP_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
CREDIT_CARD_RE = re.compile(r"(?<!\d)(?:\d[ -]?){13,19}(?!\d)")
URL_RE = re.compile(r"(?:https?://|www\.)", re.IGNORECASE)


class PipelineError(ValueError):
    """Raised when the data pipeline contract cannot be satisfied."""


class _MarkupTextExtractor(HTMLParser):
    """Small HTML-to-text extractor used before quality checks."""

    BLOCK_TAGS = {"br", "div", "p", "li", "section", "article", "header", "footer"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: List[str] = []

    def handle_starttag(self, tag: str, attrs: list) -> None:  # noqa: ARG002
        if tag.lower() in self.BLOCK_TAGS:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() in self.BLOCK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        self.parts.append(data)

    def text(self) -> str:
        return "".join(self.parts)


def stable_document_id(source_id: str, source_document_id: str) -> str:
    """Return an ID stable across input ordering and preprocessing runs."""

    payload = f"{source_id}\0{source_document_id}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def estimate_tokens(text: str, characters_per_token: float) -> int:
    non_whitespace = sum(not character.isspace() for character in text)
    if non_whitespace == 0:
        return 0
    return max(1, math.ceil(non_whitespace / characters_per_token))


def _metrics(records: Iterable[Dict[str, Any]], characters_per_token: float) -> Dict[str, int]:
    records = list(records)
    characters = sum(len(record.get("text", "")) for record in records)
    byte_count = sum(len(record.get("text", "").encode("utf-8")) for record in records)
    return {
        "documents": len(records),
        "characters": characters,
        "utf8_bytes": byte_count,
        "estimated_tokens": sum(
            estimate_tokens(record.get("text", ""), characters_per_token) for record in records
        ),
    }


def _stage_report(
    stage: str,
    before: Iterable[Dict[str, Any]],
    after: Iterable[Dict[str, Any]],
    characters_per_token: float,
    rejected: Optional[Iterable[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    rejected = list(rejected or [])
    return {
        "stage": stage,
        "before": _metrics(before, characters_per_token),
        "after": _metrics(after, characters_per_token),
        "rejected_documents": len(rejected),
        "rejection_reasons": dict(sorted(Counter(item["reason"] for item in rejected).items())),
    }


def _rejection(record: Optional[Dict[str, Any]], stage: str, reason: str, duplicate_of: Optional[str] = None) -> Dict[str, Any]:
    record = record or {}
    item: Dict[str, Any] = {
        "stage": stage,
        "reason": reason,
        "document_id": record.get("document_id"),
        "source_id": record.get("source_id"),
        "source_document_id": record.get("source_document_id"),
        "language": record.get("language", "unknown"),
        "domain": record.get("domain", "unknown"),
    }
    if duplicate_of is not None:
        item["duplicate_of"] = duplicate_of
    if isinstance(record.get("text"), str):
        item["_review_text"] = record["text"]
    return item


def _load_config(path: Path) -> Dict[str, Any]:
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PipelineError(f"cannot read pipeline config {path}: {exc}") from exc
    if config.get("schema_version") != 1:
        raise PipelineError("unsupported data pipeline schema_version")
    return config


def _read_jsonl(path: Path, config: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []
    nonempty_lines = 0
    raw_bytes = 0
    allowed_sources = set(config["source_policy"].get("selected_source_ids", []))
    allow_fixture = bool(config["source_policy"].get("allow_fixture_records", False))

    with path.open("rb") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            raw_bytes += len(raw_line)
            if not raw_line.strip():
                continue
            nonempty_lines += 1
            raw_digest = hashlib.sha256(raw_line).hexdigest()
            try:
                decoded = raw_line.decode("utf-8")
            except UnicodeDecodeError:
                rejected.append(
                    {
                        "stage": "decode_and_schema",
                        "reason": "invalid_utf8",
                        "document_id": raw_digest,
                        "source_id": None,
                        "source_document_id": None,
                        "language": "unknown",
                        "domain": "unknown",
                    }
                )
                continue
            try:
                payload = json.loads(decoded)
            except json.JSONDecodeError:
                rejected.append(
                    {
                        "stage": "decode_and_schema",
                        "reason": "invalid_json",
                        "document_id": raw_digest,
                        "source_id": None,
                        "source_document_id": None,
                        "language": "unknown",
                        "domain": "unknown",
                    }
                )
                continue
            if not isinstance(payload, dict):
                rejected.append(
                    {
                        "stage": "decode_and_schema",
                        "reason": "record_not_object",
                        "document_id": raw_digest,
                        "source_id": None,
                        "source_document_id": None,
                        "language": "unknown",
                        "domain": "unknown",
                    }
                )
                continue
            missing = [field for field in ("source_id", "source_document_id", "language", "text") if field not in payload]
            source_id = payload.get("source_id")
            source_document_id = payload.get("source_document_id")
            language = payload.get("language", "unknown")
            if missing:
                rejected.append(_rejection({"document_id": raw_digest, "language": language}, "decode_and_schema", "missing_required_field"))
                continue
            if language not in SUPPORTED_LANGUAGES:
                rejected.append(_rejection({"document_id": raw_digest, "language": language}, "decode_and_schema", "unsupported_language"))
                continue
            if not isinstance(payload["text"], str):
                rejected.append(_rejection({"document_id": raw_digest, "language": language}, "decode_and_schema", "text_not_string"))
                continue
            try:
                payload["text"].encode("utf-8", errors="strict")
            except UnicodeEncodeError:
                rejected.append(_rejection({"document_id": raw_digest, "language": language}, "decode_and_schema", "invalid_unicode_scalar"))
                continue
            if not isinstance(source_id, str) or not isinstance(source_document_id, str):
                rejected.append(_rejection({"document_id": raw_digest, "language": language}, "decode_and_schema", "source_id_not_string"))
                continue
            if not source_id or not source_document_id:
                rejected.append(_rejection({"document_id": raw_digest, "language": language}, "decode_and_schema", "empty_source_identifier"))
                continue
            is_fixture = bool(payload.get("is_fixture", False))
            if source_id not in allowed_sources and not (allow_fixture and is_fixture):
                rejected.append(
                    _rejection(
                        {"document_id": raw_digest, "source_id": source_id, "language": language},
                        "decode_and_schema",
                        "source_not_selected",
                    )
                )
                continue
            document_id = stable_document_id(source_id, source_document_id)
            records.append(
                {
                    "document_id": document_id,
                    "source_id": source_id,
                    "source_document_id": source_document_id,
                    "language": language,
                    "domain": str(payload.get("domain") or "unknown"),
                    "code_mixed": bool(payload.get("code_mixed", False)),
                    "is_fixture": is_fixture,
                    "text": payload["text"],
                    "input_line_number": line_number,
                }
            )

    input_stats = {
        "nonempty_input_lines": nonempty_lines,
        "raw_input_bytes": raw_bytes,
    }
    return records, rejected, input_stats


def _normalize_text(text: str, config: Dict[str, Any]) -> str:
    normalization = config["normalization"]
    text = unicodedata.normalize(normalization["unicode_form"], text)
    if normalization.get("remove_bom", True):
        text = text.replace("\ufeff", "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    cleaned = []
    for character in text:
        category = unicodedata.category(character)
        if category == "Cc" and character not in "\n\t":
            cleaned.append(" ")
        else:
            cleaned.append(character)
    return "".join(cleaned)


def _clean_text(text: str, config: Dict[str, Any]) -> str:
    if re.search(r"</?[A-Za-z][^>]*>", text):
        parser = _MarkupTextExtractor()
        parser.feed(text)
        parser.close()
        text = parser.text()
    text = html.unescape(text)
    patterns = [re.compile(pattern, re.IGNORECASE) for pattern in config["quality"]["boilerplate_patterns"]]
    lines = []
    for line in text.split("\n"):
        stripped = line.strip()
        if not stripped:
            continue
        if any(pattern.search(stripped) for pattern in patterns):
            continue
        lines.append(re.sub(r"[ \t]+", " ", stripped))
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def _target_script_fraction(text: str, language: str) -> float:
    start, end = LANGUAGE_SCRIPT_RANGES[language]
    letters = [character for character in text if unicodedata.category(character).startswith("L")]
    if not letters:
        return 0.0
    return sum(start <= ord(character) <= end for character in letters) / len(letters)


def _quality_reason(record: Dict[str, Any], config: Dict[str, Any]) -> Optional[str]:
    text = record["text"]
    quality = config["quality"]
    if len(text) < quality["min_characters"]:
        return "too_short"
    if len(text) > quality["max_characters"]:
        return "too_long"
    if "\ufffd" in text:
        return "replacement_character"
    max_run = quality["max_repeated_character_run"]
    if re.search(r"(.)\1{" + str(max_run) + r",}", text, re.DOTALL):
        return "extreme_character_repetition"
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if lines and (len(lines) - len(set(lines))) / len(lines) > quality["max_repeated_line_fraction"]:
        return "repeated_lines"
    target_fraction = _target_script_fraction(text, record["language"])
    minimum = quality["min_target_script_fraction"]
    if record.get("code_mixed"):
        minimum = quality["min_code_mixed_target_script_fraction"]
    if target_fraction < minimum:
        return "script_mismatch"
    url_fraction = len(URL_RE.findall(text)) / max(1, len(text))
    if url_fraction > quality["max_url_like_fraction"]:
        return "url_heavy"
    return None


def _pii_reason(text: str, config: Dict[str, Any]) -> Optional[str]:
    privacy = config["privacy"]
    checks = (
        ("pii_email", privacy.get("reject_email", True), EMAIL_RE),
        ("pii_phone", privacy.get("reject_phone", True), PHONE_RE),
        ("pii_ipv4", privacy.get("reject_ipv4", True), IP_RE),
        ("pii_credit_card", privacy.get("reject_credit_card", True), CREDIT_CARD_RE),
    )
    for reason, enabled, pattern in checks:
        if enabled and pattern.search(text):
            return reason
    return None


def _source_sort_key(record: Dict[str, Any], source_priority: Sequence[str]) -> Tuple[int, str]:
    try:
        priority = source_priority.index(record["source_id"])
    except ValueError:
        priority = len(source_priority)
    return priority, record["document_id"]


def _token_shingles(text: str, shingle_size: int) -> frozenset:
    tokens = [token.casefold() for token in TOKEN_RE.findall(text)]
    if not tokens:
        return frozenset()
    if len(tokens) < shingle_size:
        return frozenset({"\x1f".join(tokens)})
    return frozenset("\x1f".join(tokens[index : index + shingle_size]) for index in range(len(tokens) - shingle_size + 1))


def _hash64(seed: int, value: str) -> int:
    payload = seed.to_bytes(4, "big") + value.encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big")


_MASK64 = (1 << 64) - 1


def _mix64(value: int) -> int:
    """splitmix64 finaliser: a cheap bijective 64-bit mixer."""

    value = (value ^ (value >> 30)) * 0xBF58476D1CE4E5B9 & _MASK64
    value = (value ^ (value >> 27)) * 0x94D049BB133111EB & _MASK64
    return value ^ (value >> 31)


def _minhash(shingles: frozenset, permutations: int) -> Tuple[int, ...]:
    """MinHash signature.

    Each shingle is hashed once with BLAKE2b; the ``permutations`` hash functions
    are then derived by XOR-ing a per-permutation seed and applying the
    splitmix64 mixer. The numpy path computes exactly the same values as the
    pure-Python path, only vectorised, so real corpora stay tractable.
    """

    if not shingles:
        return tuple()
    seeds = [_mix64(seed + 1) for seed in range(permutations)]
    if _np is not None and len(shingles) > 16:
        base = _np.fromiter((_hash64(0, shingle) for shingle in shingles), dtype=_np.uint64, count=len(shingles))
        seed_array = _np.array(seeds, dtype=_np.uint64)[:, None]
        with _np.errstate(over="ignore"):
            value = base[None, :] ^ seed_array
            value = (value ^ (value >> _np.uint64(30))) * _np.uint64(0xBF58476D1CE4E5B9)
            value = (value ^ (value >> _np.uint64(27))) * _np.uint64(0x94D049BB133111EB)
            value = value ^ (value >> _np.uint64(31))
        return tuple(int(item) for item in value.min(axis=1))
    base_values = [_hash64(0, shingle) for shingle in shingles]
    return tuple(min(_mix64(base ^ seed) for base in base_values) for seed in seeds)


def _signature_similarity(first: Tuple[int, ...], second: Tuple[int, ...]) -> float:
    if not first or not second:
        return 1.0 if first == second else 0.0
    return sum(a == b for a, b in zip(first, second)) / len(first)


def _jaccard(first: frozenset, second: frozenset) -> float:
    union = first | second
    if not union:
        return 1.0
    return len(first & second) / len(union)


def _near_deduplicate(records: List[Dict[str, Any]], config: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    settings = config["deduplication"]
    shingle_size = int(settings["shingle_size"])
    permutations = int(settings["minhash_permutations"])
    bands = int(settings["minhash_bands"])
    if permutations % bands != 0:
        raise PipelineError("minhash_permutations must be divisible by minhash_bands")
    threshold = float(settings["jaccard_threshold"])
    accepted: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []
    band_index: Dict[Tuple[str, int, Tuple[int, ...]], List[str]] = defaultdict(list)
    prefix_index: Dict[Tuple[str, Tuple[str, ...]], List[str]] = defaultdict(list)
    state: Dict[str, Tuple[frozenset, Tuple[int, ...], Tuple[str, ...]]] = {}
    rows_per_band = permutations // bands
    # "exact_jaccard" (default) verifies LSH candidates on the full shingle sets;
    # "signature_estimate" uses the MinHash agreement rate for large corpora.
    estimate_only = settings.get("verification", "exact_jaccard") == "signature_estimate"

    for record in sorted(records, key=lambda item: item["document_id"]):
        shingles = _token_shingles(record["text"], shingle_size)
        signature = _minhash(shingles, permutations)
        tokens = tuple(token.casefold() for token in TOKEN_RE.findall(record["text"]))
        candidate_ids = set()
        for band in range(bands):
            start = band * rows_per_band
            key = (record["language"], band, signature[start : start + rows_per_band])
            candidate_ids.update(band_index.get(key, []))
        if len(tokens) >= shingle_size:
            candidate_ids.update(prefix_index.get((record["language"], tokens[: min(8, len(tokens))]), []))
        matches = []
        for candidate_id in sorted(candidate_ids):
            candidate_shingles, candidate_signature, _ = state[candidate_id]
            if estimate_only:
                similarity = _signature_similarity(signature, candidate_signature)
            else:
                similarity = _jaccard(shingles, candidate_shingles)
            if similarity >= threshold:
                matches.append(candidate_id)
        if matches:
            rejected.append(_rejection(record, "near_deduplication", "near_duplicate", duplicate_of=matches[0]))
            continue
        accepted.append(record)
        # In signature mode only the signature is kept, so memory stays bounded.
        state[record["document_id"]] = (frozenset() if estimate_only else shingles, signature, tokens[:8])
        for band in range(bands):
            start = band * rows_per_band
            band_index[(record["language"], band, signature[start : start + rows_per_band])].append(record["document_id"])
        if len(tokens) >= shingle_size:
            prefix_index[(record["language"], tokens[: min(8, len(tokens))])].append(record["document_id"])
    return accepted, rejected


def _exact_deduplicate(records: List[Dict[str, Any]], config: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    source_priority = config["source_policy"].get("selected_source_ids", [])
    accepted: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []
    by_text: Dict[str, str] = {}
    by_document_id: Dict[str, str] = {}
    for record in sorted(records, key=lambda item: _source_sort_key(item, source_priority)):
        digest = text_sha256(record["text"])
        record["text_sha256"] = digest
        record["estimated_tokens"] = estimate_tokens(
            record["text"], config["token_estimation"]["characters_per_token"]
        )
        existing_id = by_document_id.get(record["document_id"])
        if existing_id is not None:
            if existing_id == record["document_id"] and by_text.get(digest) == existing_id:
                rejected.append(_rejection(record, "exact_deduplication", "exact_duplicate", duplicate_of=existing_id))
            else:
                rejected.append(_rejection(record, "exact_deduplication", "conflicting_document_id", duplicate_of=existing_id))
            continue
        duplicate_id = by_text.get(digest)
        if duplicate_id is not None:
            rejected.append(_rejection(record, "exact_deduplication", "exact_duplicate", duplicate_of=duplicate_id))
            continue
        by_document_id[record["document_id"]] = record["document_id"]
        by_text[digest] = record["document_id"]
        accepted.append(record)
    return accepted, rejected


def _allocate_counts(total: int, ratios: Dict[str, float], ensure_nonempty: bool) -> Dict[str, int]:
    names = ("train", "validation", "test")
    raw = {name: total * float(ratios[name]) for name in names}
    counts = {name: int(math.floor(raw[name])) for name in names}
    remaining = total - sum(counts.values())
    for name in sorted(names, key=lambda item: (-(raw[item] - counts[item]), names.index(item))):
        if remaining <= 0:
            break
        counts[name] += 1
        remaining -= 1
    if ensure_nonempty and total >= 3:
        for name in ("validation", "test"):
            if float(ratios[name]) <= 0 or counts[name] > 0:
                continue
            donor = max((item for item in names if counts[item] > 1), key=lambda item: counts[item])
            counts[donor] -= 1
            counts[name] = 1
    return counts


def _split_records(records: List[Dict[str, Any]], config: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    split_config = config["split"]
    ratios = split_config["ratios"]
    seed = str(split_config["seed"])
    by_language: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_language[record["language"]].append(record)
    result = {"train": [], "validation": [], "test": []}
    for language in SUPPORTED_LANGUAGES:
        language_records = sorted(by_language[language], key=lambda item: item["document_id"])
        ordered = sorted(
            language_records,
            key=lambda item: (
                hashlib.sha256(f"{seed}\0{item['document_id']}".encode("utf-8")).digest(),
                item["document_id"],
            ),
        )
        counts = _allocate_counts(
            len(ordered), ratios, bool(split_config.get("ensure_nonempty_for_small_fixture", False))
        )
        cursor = 0
        for split in ("train", "validation", "test"):
            for record in ordered[cursor : cursor + counts[split]]:
                record["split"] = split
                result[split].append(record)
            cursor += counts[split]
    for split in result:
        result[split].sort(key=lambda item: (item["language"], item["document_id"]))
    return result


def _balanced_view(train_records: List[Dict[str, Any]], config: Dict[str, Any]) -> List[Dict[str, Any]]:
    totals = Counter()
    for record in train_records:
        totals[record["language"]] += estimate_tokens(record["text"], config["token_estimation"]["characters_per_token"])
    target = max(totals.values(), default=0)
    view = []
    for record in sorted(train_records, key=lambda item: (item["language"], item["document_id"])):
        language_total = totals[record["language"]]
        weight = target / language_total if language_total else 0.0
        view.append(
            {
                "document_id": record["document_id"],
                "language": record["language"],
                "source_id": record["source_id"],
                "estimated_tokens": estimate_tokens(record["text"], config["token_estimation"]["characters_per_token"]),
                "sampling_weight": round(weight, 12),
            }
        )
    return view


def _redact_preview(text: str) -> str:
    text = EMAIL_RE.sub("[EMAIL]", text)
    text = PHONE_RE.sub("[PHONE]", text)
    text = IP_RE.sub("[IP]", text)
    text = CREDIT_CARD_RE.sub("[NUMBER]", text)
    return text[:240]


def _review_samples(records: List[Dict[str, Any]], rejections: List[Dict[str, Any]], config: Dict[str, Any]) -> List[Dict[str, Any]]:
    limit = int(config["review"]["samples_per_reason_and_language"])
    candidates = []
    for record in records:
        candidates.append(
            {
                "status": "kept",
                "language": record["language"],
                "reason": "kept",
                "document_id": record["document_id"],
                "preview": _redact_preview(record["text"]),
            }
        )
    for item in rejections:
        if not item.get("_review_text") or not item.get("document_id"):
            continue
        candidates.append(
            {
                "status": "rejected",
                "language": item.get("language", "unknown"),
                "reason": item["reason"],
                "document_id": item["document_id"],
                "preview": _redact_preview(item["_review_text"]),
            }
        )
    selected = []
    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for item in candidates:
        grouped[(item["language"], item["reason"])].append(item)
    for key in sorted(grouped):
        selected.extend(sorted(grouped[key], key=lambda item: item["document_id"])[:limit])
    return selected


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _public_rejection(item: Dict[str, Any]) -> Dict[str, Any]:
    return {key: value for key, value in item.items() if not key.startswith("_")}


def _split_manifest_row(record: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "document_id": record["document_id"],
        "source_id": record["source_id"],
        "source_document_id": record["source_document_id"],
        "language": record["language"],
        "domain": record["domain"],
        "text_sha256": record["text_sha256"],
    }


def _processed_row(record: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "document_id": record["document_id"],
        "source_id": record["source_id"],
        "source_document_id": record["source_document_id"],
        "language": record["language"],
        "domain": record["domain"],
        "code_mixed": record["code_mixed"],
        "split": record["split"],
        "text_sha256": record["text_sha256"],
        "estimated_tokens": record["estimated_tokens"],
        "text": record["text"],
    }


def _leakage_report(split_records: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    fingerprints = {split: {record["text_sha256"] for record in records} for split, records in split_records.items()}
    pairs = {}
    for first, second in (("train", "validation"), ("train", "test"), ("validation", "test")):
        pairs[f"{first}_vs_{second}"] = len(fingerprints[first] & fingerprints[second])
    return {"exact_fingerprint_intersections": pairs, "exact_leakage_free": all(value == 0 for value in pairs.values())}


def run_pipeline(input_path: Path, config_path: Path, output_dir: Path) -> Dict[str, Any]:
    """Run the pipeline atomically, or reuse an identical completed run."""

    config = _load_config(config_path)
    input_digest = hashlib.sha256(input_path.read_bytes()).hexdigest()
    config_digest = hashlib.sha256(config_path.read_bytes()).hexdigest()
    run_fingerprint = hashlib.sha256(
        f"{PIPELINE_VERSION}\0{input_digest}\0{config_digest}".encode("utf-8")
    ).hexdigest()
    completion_path = output_dir / "completion.json"
    if completion_path.exists():
        completion = json.loads(completion_path.read_text(encoding="utf-8"))
        if completion.get("run_fingerprint") == run_fingerprint:
            return json.loads((output_dir / "report.json").read_text(encoding="utf-8"))
        raise PipelineError(f"output directory already contains a different run: {output_dir}")
    if output_dir.exists():
        raise PipelineError(f"output directory already exists without a matching completion record: {output_dir}")

    characters_per_token = float(config["token_estimation"]["characters_per_token"])
    records, initial_rejections, input_stats = _read_jsonl(input_path, config)
    reports = [
        {
            "stage": "decode_and_schema",
            "before": {
                "documents": input_stats["nonempty_input_lines"],
                "characters": _metrics(records, characters_per_token)["characters"],
                "utf8_bytes": input_stats["raw_input_bytes"],
                "estimated_tokens": _metrics(records, characters_per_token)["estimated_tokens"],
            },
            "after": _metrics(records, characters_per_token),
            "rejected_documents": len(initial_rejections),
            "rejection_reasons": dict(sorted(Counter(item["reason"] for item in initial_rejections).items())),
        }
    ]

    before = copy.deepcopy(records)
    for record in records:
        record["text"] = _normalize_text(record["text"], config)
    reports.append(_stage_report("unicode_normalization", before, records, characters_per_token))

    before = copy.deepcopy(records)
    for record in records:
        record["text"] = _clean_text(record["text"], config)
    reports.append(_stage_report("markup_and_boilerplate_cleaning", before, records, characters_per_token))

    quality_kept = []
    quality_rejections = []
    for record in records:
        reason = _quality_reason(record, config)
        if reason:
            quality_rejections.append(_rejection(record, "quality_filter", reason))
        else:
            quality_kept.append(record)
    reports.append(_stage_report("quality_filter", records, quality_kept, characters_per_token, quality_rejections))

    pii_kept = []
    pii_rejections = []
    for record in quality_kept:
        reason = _pii_reason(record["text"], config)
        if reason:
            pii_rejections.append(_rejection(record, "privacy_filter", reason))
        else:
            pii_kept.append(record)
    reports.append(_stage_report("privacy_filter", quality_kept, pii_kept, characters_per_token, pii_rejections))

    exact_kept, exact_rejections = _exact_deduplicate(pii_kept, config)
    reports.append(_stage_report("exact_deduplication", pii_kept, exact_kept, characters_per_token, exact_rejections))

    if config["deduplication"].get("near_duplicate", True):
        near_kept, near_rejections = _near_deduplicate(exact_kept, config)
    else:
        near_kept, near_rejections = exact_kept, []
    reports.append(_stage_report("near_deduplication", exact_kept, near_kept, characters_per_token, near_rejections))

    split_records = _split_records(near_kept, config)
    reports.append(_stage_report("document_level_split", near_kept, near_kept, characters_per_token))
    all_rejections = initial_rejections + quality_rejections + pii_rejections + exact_rejections + near_rejections
    balanced_view = _balanced_view(split_records["train"], config)
    selected_records = [record for split in ("train", "validation", "test") for record in split_records[split]]
    leakage = _leakage_report(split_records)

    language_distribution = Counter(record["language"] for record in selected_records)
    domain_distribution = Counter((record["language"], record["domain"]) for record in selected_records)
    rejection_by_reason = Counter(item["reason"] for item in all_rejections)
    rejection_by_language = Counter(item.get("language", "unknown") for item in all_rejections)
    report = {
        "schema_version": 1,
        "pipeline_version": PIPELINE_VERSION,
        "input_path": str(input_path),
        "input_sha256": input_digest,
        "config_path": str(config_path),
        "config_sha256": config_digest,
        "run_fingerprint": run_fingerprint,
        "execution_policy": {
            "external_corpus_downloaded": not all(record.get("is_fixture", False) for record in records),
            "input_is_synthetic_fixture": all(record.get("is_fixture", False) for record in records),
            "raw_text_written_to_output": True,
        },
        "stage_reports": reports,
        "rejection_counts_by_reason": dict(sorted(rejection_by_reason.items())),
        "rejection_counts_by_language": dict(sorted(rejection_by_language.items())),
        "duplicate_counts": {
            "exact": sum(item["reason"] == "exact_duplicate" for item in exact_rejections),
            "near": sum(item["reason"] == "near_duplicate" for item in near_rejections),
            "conflicting_document_id": sum(item["reason"] == "conflicting_document_id" for item in exact_rejections),
        },
        "leakage": leakage,
        "test_split_policy": {
            "frozen": True,
            "used_for_model_selection": False,
            "assignment_seed": config["split"]["seed"],
        },
        "split_counts": {split: len(items) for split, items in split_records.items()},
        "split_counts_by_language": {
            language: {split: sum(item["language"] == language for item in split_records[split]) for split in split_records}
            for language in SUPPORTED_LANGUAGES
        },
        "language_distribution": dict(sorted(language_distribution.items())),
        "domain_distribution": {
            f"{language}:{domain}": count for (language, domain), count in sorted(domain_distribution.items())
        },
        "manual_review_samples": len(_review_samples(near_kept, all_rejections, config)),
        "selected_document_count": len(selected_records),
    }

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f"{output_dir.name}.tmp-", dir=str(output_dir.parent)))
    try:
        (temporary / "splits").mkdir()
        _write_json(temporary / "report.json", report)
        _write_json(temporary / "balanced_train_view.json", balanced_view)
        _write_jsonl(temporary / "processed.jsonl", [_processed_row(record) for record in selected_records])
        for split in ("train", "validation", "test"):
            _write_jsonl(temporary / "splits" / f"{split}.jsonl", [_split_manifest_row(record) for record in split_records[split]])
        _write_jsonl(
            temporary / "rejections.jsonl",
            [
                _public_rejection(item)
                for item in sorted(
                    all_rejections,
                    key=lambda item: (item.get("document_id") or "", item["stage"], item["reason"]),
                )
            ],
        )
        _write_jsonl(temporary / "review_samples.jsonl", _review_samples(near_kept, all_rejections, config))
        _write_json(
            temporary / "completion.json",
            {
                "pipeline_version": PIPELINE_VERSION,
                "run_fingerprint": run_fingerprint,
                "input_sha256": input_digest,
                "config_sha256": config_digest,
                "raw_corpus_downloaded": not all(record.get("is_fixture", False) for record in records),
            },
        )
        os.replace(str(temporary), str(output_dir))
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return report
