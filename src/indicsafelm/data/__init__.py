"""Metadata-only corpus inventory helpers.

Phase 1 deliberately keeps acquisition separate from inventory.  The helpers in
this package read the JSONL manifest and never fetch a URL or open a corpus
payload.
"""

from .inventory import load_manifest, summarize_manifest, validate_manifest
from .pipeline import PipelineError, run_pipeline

__all__ = ["PipelineError", "load_manifest", "run_pipeline", "summarize_manifest", "validate_manifest"]
