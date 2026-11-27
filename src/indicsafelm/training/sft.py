"""Supervised instruction tuning (section 6.6-6.7, FR-M04).

Initialises from the base bundle, formats every record with the shared chat
template and trains only on the assistant span (prompt tokens are masked from
the loss). Dropout is disabled for this stage. The checkpoint with the lowest
SFT-validation loss is exported as the SFT bundle, which later serves as the
PPO policy initialisation *and* the frozen reference.
"""

from __future__ import annotations

import json
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List

import torch

from ..model.checkpoint import load_bundle, save_bundle
from ..model.generation import DecodingConfig, chat_example, chat_prompt_ids, generate_batch
from ..model.transformer import TransformerLM
from ..tokenizer import TokenizerBundle
from ..utils import (LANGUAGES, environment_record, git_state, make_run_id, peak_memory_bytes, pick_device, read_jsonl,
                     seed_everything, sha256_file, write_json)
from .core import MixedPrecisionTrainer, OptimConfig, masked_token_nll, right_pad


@dataclass
class SFTConfig:
    data: str = "data/instructions/sft.jsonl"
    tokenizer_dir: str = "artifacts/tokenizer/v1"
    base_bundle: str = "artifacts/checkpoints/base"
    export_dir: str = "artifacts/checkpoints/sft"
    output_dir: str = "artifacts/runs"
    seed: int = 1731
    epochs: int = 3
    batch_size: int = 32
    max_seq_len: int = 512
    precision: str = "bf16"
    device: str = "auto"
    eval_every: int = 50
    optim: Dict = field(default_factory=lambda: asdict(OptimConfig(lr=2e-4, warmup_steps=30, weight_decay=0.0)))


def encode_examples(tok: TokenizerBundle, rows: List[dict], max_len: int) -> List[dict]:
    out = []
    for row in rows:
        ids, mask = chat_example(tok, row["prompt"], row["response"])
        if len(ids) <= max_len:
            out.append({"ids": ids, "mask": mask, "language": row["language"], "intent": row["intent"]})
    return out


def batches(examples: List[dict], batch_size: int, rng: random.Random) -> List[List[dict]]:
    """Length-bucketed shuffled batches (less padding, same examples per epoch)."""

    order = sorted(examples, key=lambda e: len(e["ids"]) + rng.random())
    chunks = [order[i: i + batch_size] for i in range(0, len(order), batch_size)]
    rng.shuffle(chunks)
    return chunks


def batch_loss(model: torch.nn.Module, batch: List[dict], pad_id: int, device: torch.device):
    ids, _ = right_pad([e["ids"] for e in batch], pad_id, device)
    mask, _ = right_pad([e["mask"] for e in batch], 0, device)
    # Right padding with a causal mask: valid queries never see padding keys,
    # so the fast causal path is exact for every position that is scored.
    logits = model(ids[:, :-1])
    return masked_token_nll(logits, ids[:, 1:], mask[:, 1:])


@torch.no_grad()
def evaluate(model: torch.nn.Module, examples: List[dict], pad_id: int, device: torch.device) -> dict:
    model.eval()
    sums: Dict[str, List[float]] = {lang: [0.0, 0.0] for lang in LANGUAGES}
    for i in range(0, len(examples), 32):
        chunk = examples[i: i + 32]
        for lang in LANGUAGES:
            part = [e for e in chunk if e["language"] == lang]
            if part:
                nll, count = batch_loss(model, part, pad_id, device)
                sums[lang][0] += float(nll)
                sums[lang][1] += float(count)
    result = {lang: {"loss": s / max(1, c), "tokens": int(c)} for lang, (s, c) in sums.items()}
    total_nll = sum(s for s, _ in sums.values())
    total = sum(c for _, c in sums.values())
    result["micro_loss"] = total_nll / max(1, total)
    return result


def readiness_checks(model: TransformerLM, tok: TokenizerBundle, prompts: List[dict], device: torch.device) -> dict:
    """Table 6.6: assistant-role start, termination, language adherence."""

    from ..evaluation.labeler import script_fraction

    cfg = DecodingConfig(max_new_tokens=128, temperature=0.0)
    gens = generate_batch(model, tok, [chat_prompt_ids(tok, p["prompt"]) for p in prompts], cfg, device)
    terminated = sum(g.status == "ok" for g in gens)
    role_marker = sum(g.status == "role_marker" for g in gens)
    lang_ok = sum(script_fraction(g.text, p["language"]) >= 0.5 for g, p in zip(gens, prompts))
    return {"prompts": len(prompts), "terminated": terminated, "role_marker_failures": role_marker,
            "language_adherent": lang_ok, "samples": [{"prompt": p["prompt"], "output": g.text, "status": g.status}
                                                      for p, g in list(zip(prompts, gens))[:6]]}


def run_sft(cfg: SFTConfig, root: Path, log=print) -> dict:
    device = pick_device(cfg.device)
    seed_everything(cfg.seed)
    rng = random.Random(cfg.seed)
    tok = TokenizerBundle.load(root / cfg.tokenizer_dir)
    base, base_meta = load_bundle(root / cfg.base_bundle, tok, device)
    base.cfg.dropout = 0.0
    for module in base.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.0
    rows = read_jsonl(root / cfg.data)
    train = encode_examples(tok, [r for r in rows if r["split"] == "sft_train"], cfg.max_seq_len)
    val = encode_examples(tok, [r for r in rows if r["split"] == "sft_validation"], cfg.max_seq_len)
    steps_per_epoch = (len(train) + cfg.batch_size - 1) // cfg.batch_size
    optim_cfg = OptimConfig(**cfg.optim)
    optim_cfg.total_steps = steps_per_epoch * cfg.epochs
    trainer = MixedPrecisionTrainer(base, optim_cfg, cfg.precision)
    run_id = make_run_id("sft", cfg.seed)
    run_dir = root / cfg.output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    log(f"SFT: {len(train)} train / {len(val)} validation examples, {optim_cfg.total_steps} steps")

    manifest = {
        "run_id": run_id, "stage": "sft", "parent_checkpoint": base_meta["checkpoint_id"], "config": asdict(cfg),
        "resolved_optim": asdict(optim_cfg), "loss": "assistant span only (prompt masked)", "dropout": 0.0,
        "data_sha256": sha256_file(root / cfg.data), "train_examples": len(train), "validation_examples": len(val),
        "tokenizer_fingerprint": tok.fingerprint, "environment": environment_record(device, cfg.precision), "git": git_state(root),
    }
    write_json(run_dir / "manifest.json", manifest)
    best = {"loss": float("inf"), "step": None}
    step, started = 0, time.time()
    history = []

    def maybe_eval():
        nonlocal best
        ev = evaluate(trainer.master, val, tok.pad_id, device)
        ev["step"] = step
        history.append(ev)
        log(f"  step {step} val " + " ".join(f"{l} {ev[l]['loss']:.3f}" for l in LANGUAGES) + f" micro {ev['micro_loss']:.3f}")
        if ev["micro_loss"] < best["loss"]:
            best = {"loss": ev["micro_loss"], "step": step}
            save_bundle(root / cfg.export_dir, trainer.master, role="sft", model_config=base.cfg,
                        tokenizer_fingerprint=tok.fingerprint, run_id=run_id, step=step,
                        parent=base_meta["checkpoint_id"], extra={"validation": ev, "selection_rule": "lowest SFT-validation loss"})

    maybe_eval()
    for epoch in range(cfg.epochs):
        for batch in batches(train, cfg.batch_size, rng):
            trainer.train(True)
            nll, count = batch_loss(trainer.compute, batch, tok.pad_id, device)
            (nll / count).backward()
            info = trainer.step(step)
            step += 1
            if step % 20 == 0:
                log(f"epoch {epoch} step {step}/{optim_cfg.total_steps} loss {float(nll / count):.3f} lr {info['lr']:.2e}")
            if step % cfg.eval_every == 0:
                maybe_eval()
    maybe_eval()

    sft_model, _ = load_bundle(root / cfg.export_dir, tok, device)
    dev = read_jsonl(root / "data/eval/dev_prompts.jsonl")
    sample = [p for p in dev if p["intent"] != "harmful"][:30] + [p for p in dev if p["intent"] == "harmful"][:15]
    readiness = readiness_checks(sft_model, tok, sample, device)
    log(f"readiness: {readiness['terminated']}/{readiness['prompts']} terminated, "
        f"{readiness['language_adherent']} language-adherent, {readiness['role_marker_failures']} role-marker failures")
    manifest.update({"completed_steps": step, "wall_minutes": (time.time() - started) / 60, "best": best,
                     "validation_history": history, "readiness_checks": readiness,
                     "peak_device_memory_bytes": peak_memory_bytes(device)})
    write_json(run_dir / "manifest.json", manifest)
    write_json(root / cfg.export_dir / "readiness.json", readiness)
    return manifest
