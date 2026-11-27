"""Reward model: SFT backbone + scalar head trained on preference pairs.

* Objective: minimise ``-log sigmoid(r_chosen - r_rejected)`` (section 7.3).
* Score position: the closing ``</s>`` of ``<s> <|user|> prompt <|assistant|>
  response </s>``; padding is never scored (Table 7.3).
* Quality gate before PPO (Table 7.4): held-out pair accuracy overall, by
  language, by category and by rejected-response type, plus the correlation
  between score and response length (a length shortcut check). The gate rule
  is fixed here, before training, and written into the bundle.
"""

from __future__ import annotations

import math
import random
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn.functional as F

from ..model.checkpoint import load_bundle, save_bundle
from ..model.generation import chat_example
from ..model.transformer import ScalarHeadModel
from ..tokenizer import TokenizerBundle
from ..utils import (LANGUAGES, environment_record, git_state, make_run_id, pick_device, read_jsonl, seed_everything,
                     sha256_file, write_json)
from ..training.core import MixedPrecisionTrainer, OptimConfig, right_pad

GATE = {
    "overall_accuracy_min": 0.80,
    "per_language_accuracy_min": 0.70,
    "per_category_accuracy_min": 0.60,
    # Length check (Table 7.4): on held-out pairs whose rejected reply is at least
    # as long as the chosen reply, the reward must still prefer the chosen one.
    # The raw score/length correlation is reported as a diagnostic only, because
    # preferred replies in this data are legitimately longer.
    "counter_length_accuracy_min": 0.60,
    "counter_length_min_pairs": 20,
}


@dataclass
class RewardConfig:
    train_pairs: str = "data/preferences/pairs_train.jsonl"
    validation_pairs: str = "data/preferences/pairs_validation.jsonl"
    tokenizer_dir: str = "artifacts/tokenizer/v1"
    sft_bundle: str = "artifacts/checkpoints/sft"
    export_dir: str = "artifacts/checkpoints/reward"
    output_dir: str = "artifacts/runs"
    seed: int = 1733
    epochs: int = 2
    pairs_per_batch: int = 16
    precision: str = "bf16"
    device: str = "auto"
    optim: Dict = field(default_factory=lambda: asdict(OptimConfig(lr=3e-5, warmup_steps=20, weight_decay=0.0)))


def encode_sequence(tok: TokenizerBundle, prompt: str, response: str) -> List[int]:
    ids, _ = chat_example(tok, prompt, response)
    return ids


def score_sequences(model: ScalarHeadModel, seqs: List[List[int]], pad_id: int, device) -> torch.Tensor:
    ids, mask = right_pad(seqs, pad_id, device)
    hidden, _ = model.backbone.hidden_states(ids)  # right padding: causal path is exact for valid positions
    values = model.head(hidden).squeeze(-1)
    last = mask.sum(1) - 1
    return values.gather(1, last.unsqueeze(1)).squeeze(1).float()


@torch.no_grad()
def evaluate_pairs(model: ScalarHeadModel, tok: TokenizerBundle, pairs: List[dict], device) -> dict:
    model.eval()
    groups: Dict[str, List[int]] = defaultdict(lambda: [0, 0])
    margins, lengths, scores = [], [], []
    records = []
    for i in range(0, len(pairs), 32):
        chunk = pairs[i: i + 32]
        chosen = score_sequences(model, [encode_sequence(tok, p["prompt"], p["chosen"]) for p in chunk], tok.pad_id, device)
        rejected = score_sequences(model, [encode_sequence(tok, p["prompt"], p["rejected"]) for p in chunk], tok.pad_id, device)
        for p, c, r in zip(chunk, chosen.tolist(), rejected.tolist()):
            correct = int(c > r)
            for key in ("all", f"lang:{p['language']}", f"cat:{p['category']}", f"rej:{p['rejected_reason']}"):
                groups[key][0] += correct
                groups[key][1] += 1
            margins.append(c - r)
            for text, s in ((p["chosen"], c), (p["rejected"], r)):
                lengths.append(len(tok.encode(text)))
                scores.append(s)
            lc, lr = len(tok.encode(p["chosen"])), len(tok.encode(p["rejected"]))
            if lr >= lc:
                groups["counter_length"][0] += correct
                groups["counter_length"][1] += 1
            records.append({"pair_id": p["pair_id"], "language": p["language"], "category": p["category"],
                            "rejected_reason": p["rejected_reason"], "chosen_score": c, "rejected_score": r,
                            "chosen_tokens": lc, "rejected_tokens": lr})
    acc = {k: {"correct": v[0], "total": v[1], "accuracy": v[0] / v[1]} for k, v in sorted(groups.items())}
    n = len(scores)
    mean_l, mean_s = sum(lengths) / n, sum(scores) / n
    cov = sum((l - mean_l) * (s - mean_s) for l, s in zip(lengths, scores))
    var_l = sum((l - mean_l) ** 2 for l in lengths)
    var_s = sum((s - mean_s) ** 2 for s in scores)
    corr = cov / math.sqrt(var_l * var_s) if var_l and var_s else 0.0
    return {"accuracy": acc, "mean_margin": sum(margins) / len(margins), "length_score_correlation": corr,
            "score_mean": mean_s, "score_std": math.sqrt(var_s / n), "records": records}


def check_gate(report: dict) -> dict:
    acc = report["accuracy"]
    checks = {
        "overall": acc["all"]["accuracy"] >= GATE["overall_accuracy_min"],
        "per_language": all(acc[f"lang:{l}"]["accuracy"] >= GATE["per_language_accuracy_min"] for l in LANGUAGES if f"lang:{l}" in acc),
        "per_category": all(v["accuracy"] >= GATE["per_category_accuracy_min"] for k, v in acc.items() if k.startswith("cat:")),
    }
    counter = acc.get("counter_length", {"total": 0, "accuracy": 0.0})
    if counter["total"] >= GATE["counter_length_min_pairs"]:
        checks["length_shortcut"] = counter["accuracy"] >= GATE["counter_length_accuracy_min"]
    else:
        checks["length_shortcut_not_testable"] = False  # too few counter-length pairs: fail closed
    return {"rule": GATE, "checks": checks, "passed": all(checks.values())}


def run_reward(cfg: RewardConfig, root: Path, log=print) -> dict:
    device = pick_device(cfg.device)
    seed_everything(cfg.seed)
    rng = random.Random(cfg.seed)
    tok = TokenizerBundle.load(root / cfg.tokenizer_dir)
    sft, sft_meta = load_bundle(root / cfg.sft_bundle, tok, device)
    model = ScalarHeadModel(sft).to(device)
    train = read_jsonl(root / cfg.train_pairs)
    val = read_jsonl(root / cfg.validation_pairs)
    steps_per_epoch = (len(train) + cfg.pairs_per_batch - 1) // cfg.pairs_per_batch
    optim_cfg = OptimConfig(**cfg.optim)
    optim_cfg.total_steps = steps_per_epoch * cfg.epochs
    trainer = MixedPrecisionTrainer(model, optim_cfg, cfg.precision)
    run_id = make_run_id("reward", cfg.seed)
    run_dir = root / cfg.output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    log(f"reward model: {len(train)} train pairs, {len(val)} held-out pairs, {optim_cfg.total_steps} steps")
    before = evaluate_pairs(model, tok, val, device)
    log(f"  held-out accuracy before training: {before['accuracy']['all']['accuracy']:.3f}")
    step, started = 0, time.time()
    for epoch in range(cfg.epochs):
        order = train[:]
        rng.shuffle(order)
        for i in range(0, len(order), cfg.pairs_per_batch):
            chunk = order[i: i + cfg.pairs_per_batch]
            trainer.train(True)
            seqs = [encode_sequence(tok, p["prompt"], p["chosen"]) for p in chunk] + \
                   [encode_sequence(tok, p["prompt"], p["rejected"]) for p in chunk]
            scores = score_sequences(trainer.compute, seqs, tok.pad_id, device)
            chosen, rejected = scores[: len(chunk)], scores[len(chunk):]
            loss = -F.logsigmoid(chosen - rejected).mean()
            loss.backward()
            trainer.step(step)
            step += 1
            if step % 25 == 0:
                acc = float((chosen > rejected).float().mean())
                log(f"epoch {epoch} step {step}/{optim_cfg.total_steps} loss {float(loss):.4f} batch-acc {acc:.2f}")
    after = evaluate_pairs(trainer.master, tok, val, device)
    train_scores = evaluate_pairs(trainer.master, tok, train[:600], device)
    gate = check_gate(after)
    log(f"  held-out accuracy after training: {after['accuracy']['all']['accuracy']:.3f}  gate passed: {gate['passed']}")
    for key, value in after["accuracy"].items():
        log(f"    {key:38s} {value['accuracy']:.3f}  ({value['correct']}/{value['total']})")
    normaliser = {"mean": train_scores["score_mean"], "std": max(train_scores["score_std"], 1e-3)}
    save_bundle(root / cfg.export_dir, trainer.master, role="reward", model_config=sft.cfg, tokenizer_fingerprint=tok.fingerprint,
                run_id=run_id, step=step, parent=sft_meta["checkpoint_id"],
                extra={"quality_gate": gate, "score_normaliser": normaliser,
                       "score_position": "closing </s> of the chat-formatted sequence",
                       "heldout_accuracy": {k: v for k, v in after["accuracy"].items()}})
    report = {"run_id": run_id, "stage": "reward", "parent_checkpoint": sft_meta["checkpoint_id"], "config": asdict(cfg),
              "resolved_optim": asdict(optim_cfg), "train_pairs_sha256": sha256_file(root / cfg.train_pairs),
              "validation_pairs_sha256": sha256_file(root / cfg.validation_pairs), "before_training": {k: v for k, v in before.items() if k != "records"},
              "after_training": {k: v for k, v in after.items() if k != "records"}, "quality_gate": gate,
              "score_normaliser": normaliser, "wall_minutes": (time.time() - started) / 60,
              "environment": environment_record(device, cfg.precision), "git": git_state(root)}
    write_json(run_dir / "manifest.json", report)
    write_json(root / cfg.export_dir / "heldout_report.json", {**report, "records": after["records"]})
    return report
