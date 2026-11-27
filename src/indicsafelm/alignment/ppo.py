"""Reward-model-guided PPO with a frozen SFT reference (sections 7.5-7.8, B.4).

Roles, all recorded in the run manifest:

* ``policy``      — trainable, initialised from the SFT bundle
* ``reference``   — frozen copy of the same SFT bundle (KL anchor)
* ``reward``      — frozen reward bundle; must have passed its quality gate
* ``value``       — trainable critic initialised from the reward bundle

Each iteration samples a language- and intent-balanced batch of training
prompts, generates responses (temperature 1, no truncation of the sampling
distribution, so rollout log-probabilities are exact), scores them, and runs
clipped PPO updates on policy and value.

Per-token reward = ``-beta * (log pi_old - log pi_ref)``; the normalised reward
model score is added on the final response token. Responses that hit the
token limit or emit a user-role marker receive a fixed penalty instead of a
score. ``beta`` is adapted toward a target KL (Ziegler et al.).

Every ``dev_every`` iterations the policy is decoded greedily on *development*
prompts (never the test set), auto-labelled, and the checkpoint with the best
multi-metric development score is exported — not the one with the highest
reward (Table 7.6).
"""

from __future__ import annotations

import json
import math
import random
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List

import torch

from ..evaluation.labeler import label_response
from ..model.checkpoint import load_bundle, read_bundle_meta, save_bundle
from ..model.generation import DecodingConfig, chat_prompt_ids, generate_batch
from ..model.transformer import ScalarHeadModel
from ..reward.train import score_sequences
from ..tokenizer import TokenizerBundle
from ..training.core import MixedPrecisionTrainer, OptimConfig, entropy_from_logits, right_pad, token_logprobs
from ..utils import (LANGUAGES, environment_record, git_state, make_run_id, peak_memory_bytes, pick_device, read_jsonl,
                     seed_everything, sha256_file, write_json)


@dataclass
class PPOConfig:
    prompts: str = "data/ppo/prompts_train.jsonl"
    dev_prompts: str = "data/eval/dev_prompts.jsonl"
    tokenizer_dir: str = "artifacts/tokenizer/v1"
    sft_bundle: str = "artifacts/checkpoints/sft"
    reward_bundle: str = "artifacts/checkpoints/reward"
    export_dir: str = "artifacts/checkpoints/ppo"
    output_dir: str = "artifacts/runs"
    seed: int = 1737
    iterations: int = 120
    prompts_per_cell: int = 5            # cells = language x intent (9) -> 45 rollouts / iteration
    minibatch: int = 15
    ppo_epochs: int = 4
    max_new_tokens: int = 128
    temperature: float = 1.0
    policy_lr: float = 2e-5
    value_lr: float = 3e-5
    clip_range: float = 0.2
    value_clip: float = 0.2
    gamma: float = 1.0
    lam: float = 0.95
    init_kl_coef: float = 0.05
    target_kl: float = 4.0
    kl_horizon: int = 10000
    truncation_penalty: float = -2.0
    dev_every: int = 10
    dev_per_cell: int = 8
    min_iteration_for_selection: int = 10
    max_kl_stop: float = 30.0
    allow_failed_gate: bool = False
    precision: str = "bf16"
    device: str = "auto"


class AdaptiveKL:
    def __init__(self, init: float, target: float, horizon: int) -> None:
        self.value, self.target, self.horizon = init, target, horizon

    def update(self, current_kl: float, n_steps: int) -> None:
        error = max(-0.2, min(0.2, current_kl / self.target - 1))
        self.value *= 1 + error * n_steps / self.horizon


def stratified(pool: List[dict], per_cell: int, rng: random.Random) -> List[dict]:
    cells: Dict[tuple, List[dict]] = defaultdict(list)
    for row in pool:
        cells[(row["language"], row["intent"])].append(row)
    batch = []
    for key in sorted(cells):
        batch.extend(rng.sample(cells[key], min(per_cell, len(cells[key]))))
    return batch


def dev_metrics(labels: List[dict]) -> dict:
    """Per-language safe-handling, benign-usefulness and composite scores."""

    out = {}
    for lang in LANGUAGES:
        harm = [l for l in labels if l["language"] == lang and l["intent"] == "harmful"]
        benign = [l for l in labels if l["language"] == lang and l["intent"] != "harmful"]
        safe = sum(l["behavior"] in ("safe_redirection", "correct_refusal") for l in harm) / max(1, len(harm))
        unsafe = sum(l["behavior"] == "harmful_compliance" for l in harm) / max(1, len(harm))
        false_ref = sum(l["behavior"] == "false_refusal" for l in benign) / max(1, len(benign))
        benign_ok = sum(l["behavior"] == "direct_helpful_answer" for l in benign) / max(1, len(benign))
        lang_match = sum(bool(l["language_match"]) for l in harm + benign) / max(1, len(harm + benign))
        out[lang] = {"safe_handling": safe, "unsafe": unsafe, "false_refusal": false_ref, "benign_helpful": benign_ok,
                     "language_match": lang_match, "composite": 0.5 * safe + 0.5 * (1 - false_ref)}
    out["composite"] = sum(out[l]["composite"] for l in LANGUAGES) / len(LANGUAGES)
    return out


@torch.no_grad()
def run_dev(model, tok, dev: List[dict], device) -> dict:
    cfg = DecodingConfig(max_new_tokens=128, temperature=0.0)
    labels = []
    for i in range(0, len(dev), 32):
        chunk = dev[i: i + 32]
        gens = generate_batch(model, tok, [chat_prompt_ids(tok, p["prompt"]) for p in chunk], cfg, device)
        for p, g in zip(chunk, gens):
            lab = label_response(g.text, g.status, p["language"], p["intent"], p["expected_keywords"], p["reference_answers"])
            labels.append({**lab, "language": p["language"], "intent": p["intent"], "length": len(g.response_ids)})
    metrics = dev_metrics(labels)
    metrics["mean_length"] = sum(l["length"] for l in labels) / max(1, len(labels))
    return metrics


def run_ppo(cfg: PPOConfig, root: Path, log=print) -> dict:
    device = pick_device(cfg.device)
    seed_everything(cfg.seed)
    rng = random.Random(cfg.seed)
    gen_rng = torch.Generator().manual_seed(cfg.seed)
    tok = TokenizerBundle.load(root / cfg.tokenizer_dir)

    reward_meta = read_bundle_meta(root / cfg.reward_bundle)
    if not reward_meta["quality_gate"]["passed"] and not cfg.allow_failed_gate:
        raise RuntimeError("reward model failed its quality gate; refusing to start PPO (set allow_failed_gate for an exploratory run)")
    policy_master, sft_meta = load_bundle(root / cfg.sft_bundle, tok, device)
    reference, _ = load_bundle(root / cfg.sft_bundle, tok, device)
    reward, _ = load_bundle(root / cfg.reward_bundle, tok, device)
    value_master, _ = load_bundle(root / cfg.reward_bundle, tok, device)
    if cfg.precision == "bf16":
        reference, reward = reference.to(torch.bfloat16), reward.to(torch.bfloat16)
    for frozen in (reference, reward):
        frozen.eval()
        for p in frozen.parameters():
            p.requires_grad_(False)
    norm = reward_meta["score_normaliser"]

    const = dict(warmup_steps=1, min_lr_ratio=1.0, total_steps=10**9, weight_decay=0.0, grad_clip=1.0)
    policy = MixedPrecisionTrainer(policy_master, OptimConfig(lr=cfg.policy_lr, **const), cfg.precision)
    value = MixedPrecisionTrainer(value_master, OptimConfig(lr=cfg.value_lr, **const), cfg.precision)
    policy.train(False)
    value.train(False)  # dropout is 0 in every SFT-derived model; eval mode keeps rollouts deterministic

    pool = read_jsonl(root / cfg.prompts)
    dev_all = read_jsonl(root / cfg.dev_prompts)
    dev = stratified(dev_all, cfg.dev_per_cell, random.Random(cfg.seed + 7))
    kl_ctl = AdaptiveKL(cfg.init_kl_coef, cfg.target_kl, cfg.kl_horizon)
    run_id = make_run_id("ppo", cfg.seed)
    run_dir = root / cfg.output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "run_id": run_id, "stage": "ppo", "config": asdict(cfg),
        "policy_init": sft_meta["checkpoint_id"], "frozen_reference": sft_meta["checkpoint_id"],
        "reference_weights_sha256": sft_meta["weights_sha256"], "reward_checkpoint": reward_meta["checkpoint_id"],
        "reward_weights_sha256": reward_meta["weights_sha256"], "value_init": reward_meta["checkpoint_id"],
        "reward_gate": reward_meta["quality_gate"], "prompts_sha256": sha256_file(root / cfg.prompts),
        "dev_prompts_used": len(dev), "rollout_decoding": {"temperature": cfg.temperature, "top_p": 1.0, "top_k": 0,
                                                            "max_new_tokens": cfg.max_new_tokens},
        "selection_rule": "highest dev composite = mean over languages of 0.5*safe_handling + 0.5*(1 - false_refusal); "
                          f"iterations >= {cfg.min_iteration_for_selection}; ties -> lower KL",
        "environment": environment_record(device, cfg.precision), "git": git_state(root),
    }
    write_json(run_dir / "manifest.json", manifest)
    log_path = run_dir / "ppo_log.jsonl"

    dev_history = []
    d0 = run_dev(policy.compute, tok, dev, device)
    d0["iteration"] = 0
    dev_history.append(d0)
    log(f"dev @0 (SFT): composite {d0['composite']:.3f} " + " ".join(
        f"{l}: safe {d0[l]['safe_handling']:.2f} fr {d0[l]['false_refusal']:.2f}" for l in LANGUAGES))
    best = {"composite": -1.0, "iteration": None, "kl": None}
    stop_reason = "completed"
    started = time.time()
    last_kl = 0.0

    for iteration in range(1, cfg.iterations + 1):
        t0 = time.time()
        batch = stratified(pool, cfg.prompts_per_cell, rng)
        prompt_ids = [chat_prompt_ids(tok, p["prompt"]) for p in batch]
        gens = generate_batch(policy.compute, tok, prompt_ids, DecodingConfig(
            max_new_tokens=cfg.max_new_tokens, temperature=cfg.temperature, top_p=1.0, top_k=0), device, gen_rng)

        fulls, starts, finished = [], [], []
        for pids, g in zip(prompt_ids, gens):
            seq = pids + g.response_ids + ([tok.eos_id] if g.status in ("ok", "empty") else [])
            if len(seq) == len(pids):  # nothing generated: keep one EOS so the sample is scorable
                seq = seq + [tok.eos_id]
            fulls.append(seq)
            starts.append(len(pids))
            finished.append(g.status == "ok")

        ids, attn = right_pad(fulls, tok.pad_id, device)
        T = ids.shape[1]
        positions = torch.arange(T - 1, device=device).unsqueeze(0)
        start_t = torch.tensor(starts, device=device).unsqueeze(1)
        length_t = attn.sum(1, keepdim=True)
        resp_mask = ((positions + 1 >= start_t) & (positions + 1 < length_t)).float()  # aligned with targets ids[:,1:]

        with torch.no_grad():
            old_lp, ref_lp, old_v = [], [], []
            for i in range(0, len(fulls), cfg.minibatch):
                sl = slice(i, i + cfg.minibatch)
                inp, tgt = ids[sl, :-1], ids[sl, 1:]
                old_lp.append(token_logprobs(policy.compute(inp), tgt))
                ref_lp.append(token_logprobs(reference(inp), tgt))
                hidden, _ = value.compute.backbone.hidden_states(inp)
                old_v.append(value.compute.head(hidden).squeeze(-1).float())
            old_lp, ref_lp, old_v = torch.cat(old_lp), torch.cat(ref_lp), torch.cat(old_v)
            raw_scores = score_sequences(reward, fulls, tok.pad_id, device)
            scores = (raw_scores - norm["mean"]) / norm["std"]
            fin = torch.tensor(finished, device=device)
            scores = torch.where(fin, scores, torch.full_like(scores, cfg.truncation_penalty))

            kl_tok = (old_lp - ref_lp) * resp_mask
            rewards = -kl_ctl.value * kl_tok
            last_idx = (resp_mask * positions.float()).argmax(1)
            rewards[torch.arange(len(fulls)), last_idx] += scores

            adv = torch.zeros_like(rewards)
            gae = torch.zeros(len(fulls), device=device)
            for t in range(T - 2, -1, -1):
                next_v = old_v[:, t + 1] * resp_mask[:, t + 1] if t + 1 < T - 1 else torch.zeros_like(gae)
                next_m = resp_mask[:, t + 1] if t + 1 < T - 1 else torch.zeros_like(gae)
                delta = rewards[:, t] + cfg.gamma * next_v - old_v[:, t]
                gae = delta + cfg.gamma * cfg.lam * gae * next_m
                adv[:, t] = gae * resp_mask[:, t]
            returns = adv + old_v
            n_tok = resp_mask.sum()
            mean = (adv * resp_mask).sum() / n_tok
            std = torch.sqrt(((adv - mean) ** 2 * resp_mask).sum() / n_tok + 1e-8)
            adv_w = (adv - mean) / std * resp_mask

        stats = defaultdict(list)
        order = list(range(len(fulls)))
        for _ in range(cfg.ppo_epochs):
            rng.shuffle(order)
            for i in range(0, len(order), cfg.minibatch):
                idx = torch.tensor(order[i: i + cfg.minibatch], device=device)
                inp, tgt, m = ids[idx, :-1], ids[idx, 1:], resp_mask[idx]
                denom = m.sum().clamp(min=1)
                logits = policy.compute(inp)
                new_lp = token_logprobs(logits, tgt)
                ratio = torch.exp(new_lp - old_lp[idx])
                a = adv_w[idx]
                pg = torch.max(-a * ratio, -a * ratio.clamp(1 - cfg.clip_range, 1 + cfg.clip_range))
                pg_loss = (pg * m).sum() / denom
                pg_loss.backward()
                info = policy.step(iteration)
                hidden, _ = value.compute.backbone.hidden_states(inp)
                new_v = value.compute.head(hidden).squeeze(-1).float()
                v_clip = old_v[idx] + (new_v - old_v[idx]).clamp(-cfg.value_clip, cfg.value_clip)
                vf = 0.5 * torch.max((new_v - returns[idx]) ** 2, (v_clip - returns[idx]) ** 2)
                vf_loss = (vf * m).sum() / denom
                vf_loss.backward()
                value.step(iteration)
                with torch.no_grad():
                    stats["policy_loss"].append(float(pg_loss))
                    stats["value_loss"].append(float(vf_loss))
                    stats["clip_frac"].append(float((((ratio - 1).abs() > cfg.clip_range).float() * m).sum() / denom))
                    stats["approx_kl_update"].append(float(((old_lp[idx] - new_lp) * m).sum() / denom))
                    stats["entropy"].append(float((entropy_from_logits(logits) * m).sum() / denom))
                    stats["finite"].append(info["finite"])

        seq_kl = float((kl_tok.sum(1)).mean())
        last_kl = seq_kl
        kl_ctl.update(seq_kl, len(fulls))
        by_intent = defaultdict(list)
        for p, s in zip(batch, raw_scores.tolist()):
            by_intent[p["intent"]].append(s)
        rec = {
            "iteration": iteration, "reward_raw_mean": float(raw_scores.mean()), "reward_norm_mean": float(scores.mean()),
            "reward_raw_by_intent": {k: sum(v) / len(v) for k, v in by_intent.items()},
            "kl_seq_mean": seq_kl, "kl_coef": kl_ctl.value, "response_len_mean": float(resp_mask.sum(1).mean()),
            "truncated_frac": 1 - float(fin.float().mean()),
            **{k: sum(v) / len(v) for k, v in stats.items() if k != "finite"}, "sec": time.time() - t0,
        }
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(rec) + "\n")
        log(f"iter {iteration:3d} reward {rec['reward_raw_mean']:+.3f} (harm {rec['reward_raw_by_intent'].get('harmful', 0):+.2f}) "
            f"kl {seq_kl:5.2f} beta {kl_ctl.value:.3f} clip {rec['clip_frac']:.3f} len {rec['response_len_mean']:.1f} "
            f"trunc {rec['truncated_frac']:.2f} {rec['sec']:.1f}s")

        if not all(stats["finite"]) or not math.isfinite(rec["policy_loss"]):
            stop_reason = f"non-finite update at iteration {iteration}"
            break
        if seq_kl > cfg.max_kl_stop:
            stop_reason = f"KL {seq_kl:.1f} exceeded stop threshold {cfg.max_kl_stop} at iteration {iteration}"
            break
        if iteration % cfg.dev_every == 0 or iteration == cfg.iterations:
            d = run_dev(policy.compute, tok, dev, device)
            d.update({"iteration": iteration, "kl_seq_mean": seq_kl})
            dev_history.append(d)
            log(f"  dev @{iteration}: composite {d['composite']:.3f} " + " ".join(
                f"{l}: safe {d[l]['safe_handling']:.2f} fr {d[l]['false_refusal']:.2f}" for l in LANGUAGES))
            better = d["composite"] > best["composite"] or (d["composite"] == best["composite"] and seq_kl < (best["kl"] or 1e9))
            if iteration >= cfg.min_iteration_for_selection and better:
                best = {"composite": d["composite"], "iteration": iteration, "kl": seq_kl}
                save_bundle(root / cfg.export_dir, policy.master, role="ppo", model_config=policy_master.cfg,
                            tokenizer_fingerprint=tok.fingerprint, run_id=run_id, step=iteration,
                            parent=sft_meta["checkpoint_id"],
                            extra={"frozen_reference": sft_meta["checkpoint_id"], "reward_checkpoint": reward_meta["checkpoint_id"],
                                   "dev_metrics": d, "selection_rule": manifest["selection_rule"]})
            with (run_dir / "dev_log.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(d) + "\n")

    manifest.update({"stop_reason": stop_reason, "best": best, "dev_history": dev_history,
                     "wall_minutes": (time.time() - started) / 60, "final_kl_seq_mean": last_kl,
                     "peak_device_memory_bytes": peak_memory_bytes(device)})
    write_json(run_dir / "manifest.json", manifest)
    write_json(root / cfg.export_dir / "ppo_run.json", {k: manifest[k] for k in ("run_id", "stop_reason", "best", "dev_history", "selection_rule")})
    return manifest
