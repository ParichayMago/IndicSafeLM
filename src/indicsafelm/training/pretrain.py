"""Base pretraining: causal next-token prediction from random weights.

* FR-M01 random initialisation, seed recorded in the run manifest.
* FR-M02 one-token shift; packed windows have no padding, so every target counts.
* FR-M03 validation loss and target-token counts per language.
* FR-M05 resumable snapshots (master weights, optimizer, sampler RNG, step).
* NFR-03 atomic checkpoint writes; the best checkpoint is chosen by a rule fixed
  in the config before training (lowest macro-average validation loss).

Sequences are sampled language-balanced: each window first draws a language
uniformly, then a random offset in that language's token stream. Documents are
packed with ``</s> <s>`` boundaries and attention may cross a boundary; this
packing choice is recorded in the manifest (section 6.3).
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch

from ..model.checkpoint import atomic_write_dir, save_bundle
from ..model.transformer import ModelConfig, TransformerLM
from ..tokenizer import TokenizerBundle
from ..utils import LANGUAGES, environment_record, git_state, make_run_id, peak_memory_bytes, sha256_file, write_json
from .core import MixedPrecisionTrainer, OptimConfig, masked_token_nll


@dataclass
class PretrainConfig:
    tokens_dir: str = "data/processed/tokens-v1"
    tokenizer_dir: str = "artifacts/tokenizer/v1"
    output_dir: str = "artifacts/runs"
    export_dir: str = "artifacts/checkpoints/base"
    seed: int = 1729
    micro_batch: int = 8
    grad_accum: int = 4
    seq_len: int = 512
    precision: str = "bf16"
    device: str = "auto"
    time_budget_minutes: float = 60.0
    max_steps: Optional[int] = None
    eval_every: int = 150
    eval_windows_per_language: int = 48
    checkpoint_every: int = 150
    selection_rule: str = "lowest macro-average validation loss across hi/bn/ta"
    model: Dict = field(default_factory=lambda: ModelConfig().to_dict())
    optim: Dict = field(default_factory=lambda: asdict(OptimConfig(lr=6e-4, warmup_steps=150)))


class BalancedSampler:
    """Uniform-over-languages random windows from packed token streams."""

    def __init__(self, streams: Dict[str, np.ndarray], window: int, seed: int) -> None:
        self.streams = streams
        self.window = window
        self.languages = sorted(streams)
        self.rng = np.random.default_rng(seed)
        self.exposure = {lang: 0 for lang in self.languages}

    def sample(self, batch: int) -> tuple:
        langs = self.rng.choice(self.languages, size=batch)
        rows = []
        for lang in langs:
            stream = self.streams[lang]
            start = int(self.rng.integers(0, len(stream) - self.window))
            rows.append(stream[start: start + self.window].astype(np.int64))
            self.exposure[lang] += self.window - 1
        return torch.from_numpy(np.stack(rows)), list(langs)

    def state_dict(self) -> dict:
        return {"rng": self.rng.bit_generator.state, "exposure": dict(self.exposure)}

    def load_state_dict(self, state: dict) -> None:
        self.rng.bit_generator.state = state["rng"]
        self.exposure = dict(state["exposure"])


def load_streams(tokens_dir: Path, split: str) -> Dict[str, np.ndarray]:
    return {lang: np.fromfile(tokens_dir / f"{split}_{lang}.bin", dtype=np.uint16) for lang in LANGUAGES}


def validation_windows(streams: Dict[str, np.ndarray], window: int, count: int) -> Dict[str, torch.Tensor]:
    """Deterministic, non-overlapping windows from the start of each stream."""

    out = {}
    for lang, stream in streams.items():
        n = min(count, len(stream) // window)
        out[lang] = torch.from_numpy(stream[: n * window].astype(np.int64).reshape(n, window))
    return out


@torch.no_grad()
def evaluate_lm(model: TransformerLM, windows: Dict[str, torch.Tensor], device: torch.device, batch: int = 8) -> dict:
    """Sum of NLL and scored-token count per language; micro/macro averages."""

    model.eval()
    result = {}
    total_nll = total_tokens = 0.0
    for lang, data in windows.items():
        nll = tokens = 0.0
        for start in range(0, len(data), batch):
            chunk = data[start: start + batch].to(device)
            logits = model(chunk[:, :-1])
            s, c = masked_token_nll(logits, chunk[:, 1:], torch.ones_like(chunk[:, 1:]))
            nll += float(s)
            tokens += float(c)
        result[lang] = {"nll_sum": nll, "tokens": int(tokens), "loss": nll / max(1, tokens), "ppl": math.exp(nll / max(1, tokens))}
        total_nll += nll
        total_tokens += tokens
    result["micro_loss"] = total_nll / max(1, total_tokens)
    result["macro_loss"] = sum(result[lang]["loss"] for lang in windows) / len(windows)
    return result


def run_pretraining(cfg: PretrainConfig, root: Path, resume: Optional[Path] = None, log=print) -> dict:
    from ..utils import pick_device, seed_everything

    device = pick_device(cfg.device)
    seed_everything(cfg.seed)
    tok = TokenizerBundle.load(root / cfg.tokenizer_dir)
    model_cfg = ModelConfig.from_dict(cfg.model)
    if model_cfg.vocab_size != tok.vocab_size:
        raise ValueError("model vocab_size must equal the tokenizer vocabulary")
    tokens_dir = root / cfg.tokens_dir
    train = load_streams(tokens_dir, "train")
    val = validation_windows(load_streams(tokens_dir, "validation"), cfg.seq_len + 1, cfg.eval_windows_per_language)
    sampler = BalancedSampler(train, cfg.seq_len + 1, cfg.seed)

    master = TransformerLM(model_cfg).to(device)
    optim_cfg = OptimConfig(**cfg.optim)
    trainer = MixedPrecisionTrainer(master, optim_cfg, cfg.precision)

    run_id = make_run_id("pretrain", cfg.seed)
    step = 0
    best = {"macro_loss": float("inf"), "step": None}
    history = []
    if resume is not None:
        state = torch.load(resume / "state.pt", map_location="cpu", weights_only=False)
        trainer.load_state_dict(state["trainer"])
        sampler.load_state_dict(state["sampler"])
        torch.set_rng_state(state["torch_rng"])
        step, run_id, best, history = state["step"], state["run_id"], state["best"], state["history"]
        optim_cfg = trainer.cfg = OptimConfig(**state["optim"])
        log(f"resumed {run_id} at step {step}")
    run_dir = root / cfg.output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    tokens_per_step = cfg.micro_batch * cfg.grad_accum * cfg.seq_len

    def train_step() -> float:
        trainer.train(True)
        loss_total = 0.0
        for _ in range(cfg.grad_accum):
            batch, _ = sampler.sample(cfg.micro_batch)
            batch = batch.to(device)
            logits = trainer.compute(batch[:, :-1])
            nll, count = masked_token_nll(logits, batch[:, 1:], torch.ones_like(batch[:, 1:]))
            loss = nll / count
            (loss / cfg.grad_accum).backward()
            loss_total += float(loss.detach()) / cfg.grad_accum
        info = trainer.step(step)
        if not info["finite"]:
            raise FloatingPointError(f"non-finite gradients at step {step}")
        return loss_total, info

    # Choose the step budget from measured throughput unless max_steps is fixed.
    started = time.time()
    if cfg.max_steps is None and resume is None:
        probe_start = time.time()
        for _ in range(6):
            loss, info = train_step()
            step += 1
        per_step = (time.time() - probe_start) / 6
        total = int(cfg.time_budget_minutes * 60 / per_step)
        optim_cfg.total_steps = trainer.cfg.total_steps = max(total, step + 50)
        log(f"throughput probe: {per_step:.2f}s/step ({tokens_per_step / per_step:,.0f} tok/s) -> total_steps={optim_cfg.total_steps}")
    elif cfg.max_steps is not None:
        optim_cfg.total_steps = trainer.cfg.total_steps = cfg.max_steps

    manifest = {
        "run_id": run_id,
        "stage": "pretrain",
        "initialisation": "random (no pretrained checkpoint)",
        "seed": cfg.seed,
        "config": asdict(cfg),
        "resolved_optim": asdict(optim_cfg),
        "tokens_per_optimizer_step": tokens_per_step,
        "num_parameters": master.num_parameters(),
        "tokenizer_fingerprint": tok.fingerprint,
        "tokens_meta_sha256": sha256_file(tokens_dir / "meta.json"),
        "train_tokens_available": {lang: int(len(s)) for lang, s in train.items()},
        "packing": "documents concatenated as <s> doc </s>; attention may cross document boundaries",
        "sampling": "language drawn uniformly per window, then a uniform random offset",
        "environment": environment_record(device, cfg.precision),
        "git": git_state(root),
    }
    if resume is not None:
        previous = {}
        if (run_dir / "manifest.json").exists():
            previous = json.loads((run_dir / "manifest.json").read_text())
        events = previous.get("resume_events", [])
        events.append({"resumed_at_step": step, "new_total_steps": optim_cfg.total_steps,
                       "at": time.strftime("%Y-%m-%dT%H:%M:%S%z")})
        manifest["resume_events"] = events
        manifest["resume_note"] = (
            "The throughput probe (taken while the machine was idle) predicted 2706 steps for 70 minutes. Real "
            "throughput was lower under memory pressure and an idle-sleep pause, so the run was resumed from "
            "snapshots with a smaller total_steps. Warm-up (150 steps) is unaffected; the cosine decay targets "
            "the final total. resume_events lists each change."
        )
        for key in ("wall_minutes_before_resume",):
            if key in previous:
                manifest[key] = previous[key]
    write_json(run_dir / "manifest.json", manifest)

    def snapshot() -> None:
        state = {
            "trainer": trainer.state_dict(), "sampler": sampler.state_dict(), "torch_rng": torch.get_rng_state(),
            "step": step, "run_id": run_id, "best": best, "history": history, "optim": asdict(optim_cfg),
        }
        atomic_write_dir(run_dir / "resume", lambda tmp: torch.save(state, tmp / "state.pt"))

    log_path = run_dir / "train_log.jsonl"
    while step < optim_cfg.total_steps:
        t0 = time.time()
        loss, info = train_step()
        step += 1
        if step % 10 == 0:
            rec = {"step": step, "loss": loss, "lr": info["lr"], "grad_norm": info["grad_norm"],
                   "sec_per_step": time.time() - t0, "elapsed_min": (time.time() - started) / 60}
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(rec) + "\n")
        if step % 50 == 0:
            log(f"step {step}/{optim_cfg.total_steps} loss {loss:.3f} lr {info['lr']:.2e} "
                f"{tokens_per_step / (time.time() - t0):,.0f} tok/s elapsed {(time.time() - started) / 60:.1f} min")
        if step % cfg.eval_every == 0 or step == optim_cfg.total_steps:
            ev = evaluate_lm(master, val, device)
            ev.update({"step": step, "train_loss": loss, "exposure_tokens": dict(sampler.exposure)})
            history.append(ev)
            with (run_dir / "validation_log.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(ev) + "\n")
            log("  val " + "  ".join(f"{lang} {ev[lang]['loss']:.3f}" for lang in LANGUAGES) + f"  macro {ev['macro_loss']:.3f}")
            if ev["macro_loss"] < best["macro_loss"]:
                best = {"macro_loss": ev["macro_loss"], "step": step}
                save_bundle(root / cfg.export_dir, master, role="base", model_config=model_cfg,
                            tokenizer_fingerprint=tok.fingerprint, run_id=run_id, step=step,
                            extra={"validation": ev, "selection_rule": cfg.selection_rule})
        if step % cfg.checkpoint_every == 0:
            snapshot()

    manifest.update({
        "completed_steps": step,
        "wall_minutes": (time.time() - started) / 60,
        "train_tokens_seen": step * tokens_per_step,
        "batch_exposure_tokens_by_language": dict(sampler.exposure),
        "epochs_by_language": {lang: sampler.exposure[lang] / len(train[lang]) for lang in train},
        "peak_device_memory_bytes": peak_memory_bytes(device),
        "best_checkpoint": best,
        "exported_bundle": cfg.export_dir,
    })
    write_json(run_dir / "manifest.json", manifest)
    snapshot()
    return manifest
