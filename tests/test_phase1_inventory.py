"""Phase 1 manifest tests; they do not access the network or corpus payloads."""

import json
from pathlib import Path

from indicsafelm.data.inventory import load_manifest, summarize_manifest, validate_manifest


ROOT = Path(__file__).resolve().parents[1]


def test_phase1_manifest_contract_and_language_coverage():
    records = load_manifest(ROOT / "data/manifests/sources.jsonl")
    validate_manifest(records)
    assert len(records) == 15
    assert {record["language"] for record in records} == {"hi", "bn", "ta"}
    assert all(record["acquisition_date"] is None for record in records)
    assert all(record["checksum_sha256"] is None for record in records)


def test_phase1_inventory_is_metadata_only_and_selects_hplt():
    records = load_manifest(ROOT / "data/manifests/sources.jsonl")
    inventory = summarize_manifest(records)
    assert inventory["execution_policy"]["network_access"] is False
    assert inventory["execution_policy"]["raw_data_downloaded_locally"] is False
    assert inventory["selected_source_ids"] == ["hplt-v2-hi", "hplt-v2-bn", "hplt-v2-ta"]

    saved = json.loads((ROOT / "data/manifests/inventory.json").read_text(encoding="utf-8"))
    assert saved["record_count"] == inventory["record_count"]
    assert saved["cloud_gate"]["current_status"].startswith("sample_inspection")
