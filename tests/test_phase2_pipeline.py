"""Phase 2 pipeline tests using only the synthetic sanitized fixture."""

import json
import tempfile
from pathlib import Path

from indicsafelm.data.pipeline import run_pipeline


ROOT = Path(__file__).resolve().parents[1]


def _run(output: Path):
    return run_pipeline(
        ROOT / "data/samples/phase2_input.jsonl",
        ROOT / "configs/data_pipeline.json",
        output,
    )


def test_phase2_smoke_filters_deduplicates_and_freezes_leakage_free_splits():
    with tempfile.TemporaryDirectory() as temporary:
        report = _run(Path(temporary) / "phase2")
        assert report["selected_document_count"] == 15
        assert report["duplicate_counts"] == {"conflicting_document_id": 0, "exact": 3, "near": 3}
        assert report["leakage"]["exact_leakage_free"] is True
        assert report["split_counts_by_language"] == {
            "hi": {"train": 3, "validation": 1, "test": 1},
            "bn": {"train": 3, "validation": 1, "test": 1},
            "ta": {"train": 3, "validation": 1, "test": 1},
        }


def test_phase2_restart_reuses_identical_run_and_writes_no_raw_text_to_id_manifests():
    with tempfile.TemporaryDirectory() as temporary:
        output = Path(temporary) / "phase2"
        first = _run(output)
        second = _run(output)
        assert second == first
        rows = [json.loads(line) for line in (output / "splits" / "test.jsonl").read_text().splitlines()]
        assert rows and all("text" not in row for row in rows)
        assert json.loads((output / "completion.json").read_text())["raw_corpus_downloaded"] is False
