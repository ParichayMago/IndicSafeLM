"""Checkpoint bundles (report section 4.8, Table 4.3; NFR-03).

A bundle is a directory::

    <name>/
      weights.pt        model state_dict
      bundle.json       architecture, role, tokenizer fingerprint, lineage, digests
      COMPLETE          written last; bundles without it are ignored

Bundles are written to a temporary sibling directory and moved into place with
an atomic rename, so an interrupted write never looks like a finished
checkpoint. Loading checks the completion marker, the tokenizer fingerprint and
the tensor shapes and raises an explicit error on any mismatch.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch

from ..tokenizer import TokenizerBundle, TokenizerMismatch
from .transformer import ModelConfig, ScalarHeadModel, TransformerLM


class CheckpointError(RuntimeError):
    """Raised for incomplete, incompatible or corrupt bundles."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_dir(target: Path, writer) -> None:
    """Call ``writer(tmp_dir)`` then atomically move tmp_dir to ``target``."""

    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=str(target.parent)))
    try:
        writer(tmp)
        (tmp / "COMPLETE").write_text(dt.datetime.now(dt.timezone.utc).isoformat() + "\n")
        backup = None
        if target.exists():
            backup = target.with_name(f".{target.name}.old-{os.getpid()}")
            os.replace(target, backup)
        os.replace(tmp, target)
        if backup is not None:
            shutil.rmtree(backup, ignore_errors=True)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise


def save_bundle(
    target: Path,
    model: torch.nn.Module,
    *,
    role: str,
    model_config: ModelConfig,
    tokenizer_fingerprint: str,
    run_id: str,
    step: int,
    parent: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Write an inference bundle. ``role`` is base, sft, ppo, reward or value."""

    kind = "scalar_head" if isinstance(model, ScalarHeadModel) else "lm"
    state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    meta: Dict[str, Any] = {
        "schema_version": 1,
        "checkpoint_id": f"{role}-{run_id}-step{step}",
        "role": role,
        "model_kind": kind,
        "model_config": model_config.to_dict(),
        "tokenizer_fingerprint": tokenizer_fingerprint,
        "run_id": run_id,
        "parent_checkpoint": parent,
        "step": step,
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "num_parameters": sum(p.numel() for p in {id(p): p for p in model.parameters()}.values()),
    }
    if extra:
        meta.update(extra)

    def writer(tmp: Path) -> None:
        torch.save(state, tmp / "weights.pt")
        meta["weights_sha256"] = _sha256(tmp / "weights.pt")
        (tmp / "bundle.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    atomic_write_dir(Path(target), writer)
    return meta


def read_bundle_meta(path: Path) -> Dict[str, Any]:
    path = Path(path)
    if not (path / "COMPLETE").exists():
        raise CheckpointError(f"{path} is not a completed checkpoint bundle (missing COMPLETE marker)")
    return json.loads((path / "bundle.json").read_text(encoding="utf-8"))


def load_bundle(
    path: Path,
    tokenizer: Optional[TokenizerBundle] = None,
    device: str | torch.device = "cpu",
    verify_digest: bool = True,
) -> Tuple[torch.nn.Module, Dict[str, Any]]:
    """Load a bundle, validating completion, tokenizer and tensor shapes."""

    path = Path(path)
    meta = read_bundle_meta(path)
    if tokenizer is not None and meta["tokenizer_fingerprint"] != tokenizer.fingerprint:
        raise TokenizerMismatch(
            f"checkpoint {meta['checkpoint_id']} was trained with tokenizer {meta['tokenizer_fingerprint'][:12]}, "
            f"but tokenizer {tokenizer.fingerprint[:12]} is loaded"
        )
    weights = path / "weights.pt"
    if verify_digest and _sha256(weights) != meta.get("weights_sha256"):
        raise CheckpointError(f"{weights} digest does not match bundle.json")
    cfg = ModelConfig.from_dict(meta["model_config"])
    if tokenizer is not None and cfg.vocab_size != tokenizer.vocab_size:
        raise TokenizerMismatch(f"model vocabulary {cfg.vocab_size} != tokenizer vocabulary {tokenizer.vocab_size}")
    backbone = TransformerLM(cfg)
    model: torch.nn.Module = ScalarHeadModel(backbone) if meta["model_kind"] == "scalar_head" else backbone
    state = torch.load(weights, map_location="cpu", weights_only=True)
    expected = model.state_dict()
    bad = [k for k, v in state.items() if k in expected and tuple(expected[k].shape) != tuple(v.shape)]
    missing = [k for k in expected if k not in state]
    if bad or missing:
        raise CheckpointError(f"incompatible checkpoint: shape mismatch {bad[:3]}, missing {missing[:3]}")
    model.load_state_dict(state)
    return model.to(device), meta


def lm_from_scalar_or_lm(path: Path, tokenizer: TokenizerBundle, device="cpu") -> TransformerLM:
    """Return the language-model backbone of any bundle (used to initialise heads)."""

    model, _ = load_bundle(path, tokenizer, device)
    return model.backbone if isinstance(model, ScalarHeadModel) else model
