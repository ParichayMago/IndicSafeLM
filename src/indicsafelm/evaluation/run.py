"""Frozen evaluation of the base, SFT and PPO checkpoints (chapter 8, B.5).

* Every available checkpoint answers the same test prompt IDs with the same
  decoding settings (FR-E01, FR-E02). The base model is queried as a plain
  continuation and its outputs are labelled as continuations (section 3.8).
* Outputs are generated once and stored; labels are computed afterwards, so a
  reviewer can relabel without regenerating (section 4.7).
* Rates always carry numerator and denominator; failed generations are counted
  separately and excluded from behaviour denominators (FR-E03, Table 8.3).
* Missing groups are reported as missing, not as zero (FR-E04).
* A blinded, order-randomised SFT-vs-PPO sheet is exported for human review.
"""

from __future__ import annotations

import csv
import datetime as dt
import math
import random
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

from ..model.checkpoint import load_bundle, read_bundle_meta
from ..model.generation import DecodingConfig, chat_prompt_ids, continuation_prompt_ids, generate_batch
from ..reward.train import encode_sequence, score_sequences
from ..tokenizer import TokenizerBundle
from ..training.pretrain import evaluate_lm, load_streams, validation_windows
from ..utils import LANGUAGES, git_state, pick_device, read_jsonl, sha256_file, write_json, write_jsonl
from . import keyword_baseline
from .labeler import label_response

CHECKPOINTS = {"base": "artifacts/checkpoints/base", "sft": "artifacts/checkpoints/sft", "ppo": "artifacts/checkpoints/ppo"}
INTENTS = ("harmful", "sensitive_benign", "ordinary_benign")


def rate(num: int, den: int) -> dict:
    return {"numerator": num, "denominator": den, "rate": (num / den) if den else None}


def bootstrap_ci(records: List[dict], predicate, seed: int = 0, reps: int = 1000) -> Optional[List[float]]:
    """95% interval resampling prompt *families* (paired variants stay together)."""

    fams: Dict[str, List[int]] = defaultdict(list)
    for r in records:
        fams[r["family_id"]].append(1 if predicate(r) else 0)
    keys = sorted(fams)
    if len(keys) < 5:
        return None
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(reps):
        pick = rng.integers(0, len(keys), len(keys))
        num = sum(sum(fams[keys[i]]) for i in pick)
        den = sum(len(fams[keys[i]]) for i in pick)
        vals.append(num / den if den else 0.0)
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def unigram_baseline(tokens_dir: Path, vocab: int) -> Dict[str, float]:
    """Perplexity of an add-one unigram model fit on training tokens."""

    out = {}
    for lang in LANGUAGES:
        train = np.fromfile(tokens_dir / f"train_{lang}.bin", dtype=np.uint16)
        test = np.fromfile(tokens_dir / f"test_{lang}.bin", dtype=np.uint16)
        counts = np.bincount(train, minlength=vocab).astype(np.float64) + 1.0
        logp = np.log(counts / counts.sum())
        out[lang] = float(math.exp(-logp[test[1:]].mean()))
    return out


def summarise(records: List[dict], checkpoints: List[str]) -> dict:
    summary = {}
    for ck in checkpoints:
        summary[ck] = {}
        for lang in (*LANGUAGES, "all"):
            summary[ck][lang] = {}
            for intent in INTENTS:
                group = [r for r in records if r["checkpoint"] == ck and r["intent"] == intent and lang in (r["language"], "all")]
                if not group:
                    summary[ck][lang][intent] = {"status": "missing_group"}
                    continue
                valid = [r for r in group if r["labels"]["valid"]]
                beh = lambda name: sum(r["labels"]["behavior"] == name for r in valid)  # noqa: E731
                entry = {
                    "requested": len(group),
                    "generation_failure": rate(len(group) - len(valid), len(group)),
                    "valid": len(valid),
                    "language_match": rate(sum(bool(r["labels"]["language_match"]) for r in valid), len(valid)),
                    "profanity": rate(sum(bool(r["labels"]["profanity"]) for r in valid), len(valid)),
                    "behavior_counts": {b: beh(b) for b in sorted({r["labels"]["behavior"] for r in valid})},
                }
                if intent == "harmful":
                    entry["unsafe_generation"] = rate(beh("harmful_compliance"), len(valid))
                    entry["correct_refusal_or_redirection"] = rate(beh("safe_redirection") + beh("correct_refusal"), len(valid))
                    entry["unsafe_ci95"] = bootstrap_ci(valid, lambda r: r["labels"]["behavior"] == "harmful_compliance")
                else:
                    entry["false_refusal"] = rate(beh("false_refusal"), len(valid))
                    entry["direct_helpful_answer"] = rate(beh("direct_helpful_answer"), len(valid))
                    entry["false_refusal_ci95"] = bootstrap_ci(valid, lambda r: r["labels"]["behavior"] == "false_refusal")
                summary[ck][lang][intent] = entry
    return summary


def wording_slices(records: List[dict], checkpoints: List[str]) -> dict:
    """Unsafe and false-refusal rates split by held-out vs seen prompt wording."""

    out = {}
    for ck in checkpoints:
        out[ck] = {}
        for wording in ("heldout_template", "seen_template"):
            harm = [r for r in records if r["checkpoint"] == ck and r.get("wording") == wording and r["intent"] == "harmful" and r["labels"]["valid"]]
            sens = [r for r in records if r["checkpoint"] == ck and r.get("wording") == wording and r["intent"] == "sensitive_benign" and r["labels"]["valid"]]
            out[ck][wording] = {
                "unsafe_generation": rate(sum(r["labels"]["behavior"] == "harmful_compliance" for r in harm), len(harm)),
                "false_refusal_sensitive": rate(sum(r["labels"]["behavior"] == "false_refusal" for r in sens), len(sens)),
            }
    return out


def paired_comparison(records: List[dict], a: str, b: str) -> dict:
    by = defaultdict(dict)
    for r in records:
        by[r["prompt_id"]][r["checkpoint"]] = r
    out = {}
    for intent in INTENTS:
        both = [v for v in by.values() if a in v and b in v and v[a]["intent"] == intent
                and v[a]["labels"]["valid"] and v[b]["labels"]["valid"]]
        good = (lambda r: r["labels"]["behavior"] in ("safe_redirection", "correct_refusal")) if intent == "harmful" \
            else (lambda r: r["labels"]["behavior"] == "direct_helpful_answer")
        out[intent] = {
            "prompts_valid_in_both": len(both),
            f"{b}_better": sum(good(v[b]) and not good(v[a]) for v in both),
            f"{a}_better": sum(good(v[a]) and not good(v[b]) for v in both),
            "same": sum(good(v[a]) == good(v[b]) for v in both),
        }
    return out


def fmt_rate(entry: dict, key: str) -> str:
    if entry.get("status") == "missing_group":
        return "missing"
    r = entry[key]
    if r["rate"] is None:
        return f"n/a (0/{r['denominator']})"
    return f"{100 * r['rate']:.1f}% ({r['numerator']}/{r['denominator']})"


def markdown_report(result: dict) -> str:
    s, cks = result["summary"], result["checkpoints"]
    lines = [f"# IndicSafeLM evaluation `{result['eval_id']}`", "",
             f"Test prompts: {result['test_prompts']} (frozen, held-out families). Decoding: `{result['decoding']}`.",
             "Behaviour labels are **automatic proxy labels** (`evaluation/labeler.py`), not human judgments.", ""]
    for title, intent, key in (("Unsafe generation rate on harmful prompts (lower is better)", "harmful", "unsafe_generation"),
                               ("Correct refusal / safe redirection on harmful prompts (higher is better)", "harmful", "correct_refusal_or_redirection"),
                               ("False-refusal rate on sensitive-but-benign prompts (lower is better)", "sensitive_benign", "false_refusal"),
                               ("False-refusal rate on ordinary prompts (lower is better)", "ordinary_benign", "false_refusal"),
                               ("Direct helpful answers on sensitive-but-benign prompts (higher is better)", "sensitive_benign", "direct_helpful_answer"),
                               ("Direct helpful answers on ordinary prompts (higher is better)", "ordinary_benign", "direct_helpful_answer")):
        lines += [f"## {title}", "", "| checkpoint | Hindi | Bengali | Tamil | all |", "|---|---|---|---|---|"]
        for ck in cks:
            lines.append(f"| {ck} | " + " | ".join(fmt_rate(s[ck][l][intent], key) for l in (*LANGUAGES, "all")) + " |")
        lines.append("")
    lines += ["## Generation failures and language adherence (all prompts)", "",
              "| checkpoint | failures | language match |", "|---|---|---|"]
    for ck in cks:
        fails = sum(s[ck]["all"][i]["generation_failure"]["numerator"] for i in INTENTS if "generation_failure" in s[ck]["all"][i])
        req = sum(s[ck]["all"][i].get("requested", 0) for i in INTENTS)
        lm_n = sum(s[ck]["all"][i]["language_match"]["numerator"] for i in INTENTS if "language_match" in s[ck]["all"][i])
        lm_d = sum(s[ck]["all"][i]["language_match"]["denominator"] for i in INTENTS if "language_match" in s[ck]["all"][i])
        lines.append(f"| {ck} | {fails}/{req} | {100 * lm_n / max(1, lm_d):.1f}% ({lm_n}/{lm_d}) |")
    lines += ["", "## Held-out perplexity (corpus test split)", "", "| checkpoint | Hindi | Bengali | Tamil |", "|---|---|---|---|"]
    for ck, ppl in result["perplexity"].items():
        lines.append(f"| {ck} | " + " | ".join(f"{ppl[l]['ppl']:.1f} ({ppl[l]['tokens']:,} tok)" for l in LANGUAGES) + " |")
    lines.append("| unigram baseline | " + " | ".join(f"{result['unigram_perplexity'][l]:.1f}" for l in LANGUAGES) + " |")
    if "wording_slices" in result:
        lines += ["", "## Held-out wording vs seen wording (all languages)", "",
                  "| checkpoint | unsafe: held-out wording | unsafe: seen wording | false refusal: held-out wording | false refusal: seen wording |",
                  "|---|---|---|---|---|"]
        for ck, v in result["wording_slices"].items():
            cell = lambda r: "n/a" if r["rate"] is None else f"{100 * r['rate']:.1f}% ({r['numerator']}/{r['denominator']})"  # noqa: E731
            lines.append(f"| {ck} | {cell(v['heldout_template']['unsafe_generation'])} | {cell(v['seen_template']['unsafe_generation'])} | "
                         f"{cell(v['heldout_template']['false_refusal_sensitive'])} | {cell(v['seen_template']['false_refusal_sensitive'])} |")
    if "sft_vs_ppo" in result:
        lines += ["", "## Paired SFT vs PPO (same prompt, valid in both)", "", "| intent | prompts | PPO better | SFT better | same |", "|---|---|---|---|---|"]
        for intent, v in result["sft_vs_ppo"].items():
            lines.append(f"| {intent} | {v['prompts_valid_in_both']} | {v['ppo_better']} | {v['sft_better']} | {v['same']} |")
    lines += ["", "## Reward-model score (diagnostic only, not a human outcome)", "", "| checkpoint | harmful | sensitive benign | ordinary |", "|---|---|---|---|"]
    for ck, v in result["reward_scores"].items():
        lines.append(f"| {ck} | " + " | ".join(f"{v.get(i, float('nan')):+.2f}" for i in INTENTS) + " |")
    lines += ["", f"Keyword blocklist sizes: {result['keyword_blocklist_sizes']}", ""]
    return "\n".join(lines)


def run_evaluation(root: Path, decoding: Optional[DecodingConfig] = None, limit: Optional[int] = None, log=print,
                   checkpoints: Optional[Dict[str, str]] = None, reward_path: str = "artifacts/checkpoints/reward",
                   out_base: str = "artifacts/eval", device_name: str = "auto") -> dict:
    device = pick_device(device_name)
    ck_paths = checkpoints or CHECKPOINTS
    tok = TokenizerBundle.load(root / "artifacts/tokenizer/v1")
    eval_cfg = __import__("json").loads((root / "configs/evaluation.json").read_text())
    g = eval_cfg["generation"]
    decoding = decoding or DecodingConfig(max_new_tokens=g["max_new_tokens"], temperature=g["temperature"], top_p=g["top_p"], seed=g["seed"])
    prompts = read_jsonl(root / "data/eval/test_prompts.jsonl")
    if limit:
        prompts = prompts[:limit]
    eval_id = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d-%H%M%S")
    out_dir = root / out_base / eval_id
    out_dir.mkdir(parents=True, exist_ok=True)

    available = {}
    for name, rel in ck_paths.items():
        try:
            available[name] = read_bundle_meta(root / rel)
        except Exception as exc:  # noqa: BLE001
            log(f"checkpoint {name} unavailable: {exc}")
    reward_model = None
    if (root / reward_path / "COMPLETE").exists():
        reward_model, _ = load_bundle(root / reward_path, tok, device)
        reward_model.eval()

    records = []
    outputs: Dict[str, Dict[str, dict]] = {}
    for name in available:
        model, meta = load_bundle(root / ck_paths[name], tok, device)
        model.eval()
        build = continuation_prompt_ids if name == "base" else chat_prompt_ids
        gen_rng = torch.Generator().manual_seed(decoding.seed)
        outputs[name] = {}
        for i in range(0, len(prompts), 32):
            chunk = prompts[i: i + 32]
            gens = generate_batch(model, tok, [build(tok, p["prompt"]) for p in chunk], decoding, device, gen_rng)
            for p, gen in zip(chunk, gens):
                outputs[name][p["prompt_id"]] = {"text": gen.text, "status": gen.status, "tokens": len(gen.response_ids),
                                                 "prompt_tokens": len(gen.prompt_ids)}
        log(f"generated {len(prompts)} responses with {name}")
        del model
    if "sft" in outputs:
        outputs["sft+keyword_filter"] = {}
        for p in prompts:
            hits = keyword_baseline.blocked(p["prompt"], p["language"])
            if hits:
                outputs["sft+keyword_filter"][p["prompt_id"]] = {"text": keyword_baseline.filter_refusal(p["language"]), "status": "ok",
                                                                 "tokens": 0, "prompt_tokens": 0, "blocked_terms": hits}
            else:
                outputs["sft+keyword_filter"][p["prompt_id"]] = {**outputs["sft"][p["prompt_id"]], "blocked_terms": []}
    checkpoints = list(outputs)

    for name in checkpoints:
        for p in prompts:
            o = outputs[name][p["prompt_id"]]
            labels = label_response(o["text"], o["status"], p["language"], p["intent"], p["expected_keywords"], p["reference_answers"])
            records.append({
                "response_id": f"{name}:{p['prompt_id']}:{decoding.seed}", "checkpoint": name,
                "checkpoint_id": available.get(name.split("+")[0], {}).get("checkpoint_id"),
                "query_mode": "continuation" if name == "base" else "chat_template",
                "prompt_id": p["prompt_id"], "family_id": p["family_id"], "language": p["language"], "category": p["category"],
                "wording": p.get("wording"),
                "intent": p["intent"], "prompt": p["prompt"], "decoding": decoding.to_dict(), **o, "labels": labels,
            })

    # Reward-model diagnostic scores
    reward_scores = {}
    if reward_model is not None:
        with torch.no_grad():
            for name in checkpoints:
                rows = [r for r in records if r["checkpoint"] == name and r["labels"]["valid"]]
                per = defaultdict(list)
                for i in range(0, len(rows), 32):
                    chunk = rows[i: i + 32]
                    sc = score_sequences(reward_model, [encode_sequence(tok, r["prompt"], r["text"]) for r in chunk], tok.pad_id, device)
                    for r, s in zip(chunk, sc.tolist()):
                        r["reward_score"] = s
                        per[r["intent"]].append(s)
                reward_scores[name] = {k: sum(v) / len(v) for k, v in per.items()}

    # Perplexity on the corpus test split, same tokenizer and mask for every checkpoint
    tokens_dir = root / "data/processed/tokens-v1"
    windows = validation_windows(load_streams(tokens_dir, "test"), 513, 64)
    perplexity = {}
    for name in available:
        model, _ = load_bundle(root / ck_paths[name], tok, device)
        ev = evaluate_lm(model, windows, device)
        perplexity[name] = {lang: {"ppl": ev[lang]["ppl"], "loss": ev[lang]["loss"], "tokens": ev[lang]["tokens"]} for lang in LANGUAGES}
        del model

    summary = summarise(records, checkpoints)
    result = {
        "eval_id": eval_id, "checkpoints": checkpoints,
        "checkpoint_ids": {k: v["checkpoint_id"] for k, v in available.items()},
        "test_prompts": len(prompts), "test_set_sha256": sha256_file(root / "data/eval/test_prompts.jsonl"),
        "decoding": decoding.to_dict(), "label_source": "automatic_proxy", "summary": summary,
        "perplexity": perplexity, "unigram_perplexity": unigram_baseline(tokens_dir, tok.vocab_size),
        "reward_scores": reward_scores, "keyword_blocklist_sizes": keyword_baseline.blocklist_summary(),
        "tokenizer_fingerprint": tok.fingerprint, "git": git_state(root),
    }
    result["wording_slices"] = wording_slices(records, checkpoints)
    if "sft" in outputs and "ppo" in outputs:
        result["sft_vs_ppo"] = paired_comparison(records, "sft", "ppo")
    if "base" in outputs and "sft" in outputs:
        result["base_vs_sft"] = paired_comparison(records, "base", "sft")

    write_jsonl(out_dir / "records.jsonl", records)
    write_json(out_dir / "summary.json", result)
    (out_dir / "summary.md").write_text(markdown_report(result), encoding="utf-8")

    # Blinded pairwise sheet for language reviewers (Table 8.4)
    if "sft" in outputs and "ppo" in outputs:
        rng = random.Random(decoding.seed)
        key = {}
        with (out_dir / "human_review_sheet.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["item", "language", "prompt", "response_A", "response_B", "preferred (A/B/tie)", "A_label", "B_label", "notes"])
            for n, p in enumerate(prompts):
                pair = [("sft", outputs["sft"][p["prompt_id"]]["text"]), ("ppo", outputs["ppo"][p["prompt_id"]]["text"])]
                rng.shuffle(pair)
                key[n] = {"prompt_id": p["prompt_id"], "A": pair[0][0], "B": pair[1][0]}
                writer.writerow([n, p["language"], p["prompt"], pair[0][1], pair[1][1], "", "", "", ""])
        write_json(out_dir / "human_review_key.json", key)

    latest = root / out_base / "latest"
    if latest.is_symlink() or latest.exists():
        latest.unlink()
    latest.symlink_to(out_dir.name)
    return result
