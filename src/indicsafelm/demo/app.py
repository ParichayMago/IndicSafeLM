"""Local side-by-side comparison interface (sections 3.8, 8.8; FR-U01-U05).

Run with ``python scripts/demo.py`` and open http://127.0.0.1:7860.

* Same prompt, same decoding settings and the same seed for every checkpoint.
* The base model is queried as a plain continuation and labelled as such.
* Load and generation errors are shown as errors, never as refusals.
* Prompt length is capped (visible limit) with a request to shorten.
* "Classroom mode" masks abusive terms in displayed outputs (NFR-10).
* Reset clears the prompt and every output for the next participant.
"""

from __future__ import annotations

import hashlib
import html
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch

from ..evaluation import keyword_baseline
from ..evaluation.labeler import all_harm_markers, label_response, script_fraction
from ..model.checkpoint import load_bundle
from ..model.generation import DecodingConfig, chat_prompt_ids, continuation_prompt_ids, generate_batch
from ..reward.train import encode_sequence, score_sequences
from ..tokenizer import TokenizerBundle
from ..utils import LANGUAGE_NAMES, LANGUAGES, ROOT, pick_device, read_jsonl

MAX_PROMPT_TOKENS = 256
MAX_NEW_TOKENS_CAP = 160
CKPT_DIR = os.environ.get("INDICSAFELM_CKPT_DIR", "artifacts/checkpoints")
SYSTEMS = {
    "base": ("Base (pretrained only)", f"{CKPT_DIR}/base"),
    "sft": ("SFT (instruction-tuned)", f"{CKPT_DIR}/sft"),
    "ppo": ("PPO (safety-aligned)", f"{CKPT_DIR}/ppo"),
    "kw": ("SFT + keyword filter (baseline)", None),
}
LANG_CHOICES = [(f"{LANGUAGE_NAMES[l]} ({l})", l) for l in LANGUAGES]

CSS = """
:root { --indic-font: "Noto Sans Devanagari", "Kohinoor Devanagari", "Noto Sans Bengali", "Kohinoor Bangla",
        "Bangla Sangam MN", "Noto Sans Tamil", "Tamil Sangam MN", system-ui, sans-serif; }
.gradio-container { font-family: var(--indic-font) !important; max-width: 1500px !important; }
#prompt textarea { font-size: 1.3rem !important; line-height: 1.6 !important; }
.card { border: 1px solid var(--border-color-primary); border-radius: 10px; padding: 14px 16px; min-height: 260px;
        background: var(--background-fill-primary); }
.card h3 { margin: 0 0 4px 0; font-size: 1.05rem; }
.card .cid { font-family: ui-monospace, monospace; font-size: .72rem; opacity: .7; word-break: break-all; }
.card .out { font-size: 1.28rem; line-height: 1.65; margin: 12px 0; white-space: pre-wrap; }
.badge { display: inline-block; padding: 2px 10px; border-radius: 999px; font-size: .8rem; font-weight: 600; }
.b-answer { background: #dcfce7; color: #14532d; } .b-refusal { background: #dbeafe; color: #1e3a8a; }
.b-unsafe { background: #fee2e2; color: #7f1d1d; } .b-error { background: #fde68a; color: #78350f; }
.b-cont { background: #e5e7eb; color: #111827; } .b-amb { background: #f3e8ff; color: #581c87; }
.meta { font-size: .8rem; opacity: .8; }
.warn { padding: 10px 14px; border-radius: 8px; background: #fff7ed; color: #7c2d12; border: 1px solid #fed7aa; }
.warn b { color: #7c2d12 !important; }
"""


class Backend:
    """Loads every available checkpoint once and serves generations."""

    def __init__(self, root: Path = ROOT) -> None:
        self.root = root
        self.device = pick_device(os.environ.get("INDICSAFELM_DEVICE", "auto"))
        self.tok = TokenizerBundle.load(root / "artifacts/tokenizer/v1")
        self.models: Dict[str, torch.nn.Module] = {}
        self.meta: Dict[str, dict] = {}
        self.errors: Dict[str, str] = {}
        for key, (_, rel) in SYSTEMS.items():
            if rel is None:
                continue
            try:
                model, meta = load_bundle(root / rel, self.tok, self.device)
                model.eval()
                self.models[key], self.meta[key] = model, meta
            except Exception as exc:  # noqa: BLE001 — shown to the presenter
                self.errors[key] = f"{type(exc).__name__}: {exc}"
        self.reward = None
        try:
            self.reward, _ = load_bundle(root / f"{CKPT_DIR}/reward", self.tok, self.device)
            self.reward.eval()
        except Exception as exc:  # noqa: BLE001
            self.errors["reward"] = f"{type(exc).__name__}: {exc}"
        if "sft" not in self.models:
            self.errors.setdefault("kw", "needs the SFT checkpoint")
        demo_path = root / "data/eval/demo_cases.jsonl"
        self.cases = read_jsonl(demo_path) if demo_path.exists() else []

    def available(self) -> List[str]:
        return [k for k in SYSTEMS if k in self.models or (k == "kw" and "sft" in self.models)]

    def run(self, system: str, prompt: str, lang: str, cfg: DecodingConfig, intent: str) -> dict:
        started = time.time()
        if system == "kw":
            hits = keyword_baseline.blocked(prompt, lang)
            if hits:
                text = keyword_baseline.filter_refusal(lang)
                return {"text": text, "status": "ok", "tokens": 0, "seconds": time.time() - started,
                        "note": f"blocked by keyword filter (matched: {', '.join(hits[:5])})", "mode": "filter"}
            out = self.run("sft", prompt, lang, cfg, intent)
            out["note"] = "passed keyword filter → SFT output"
            return out
        model = self.models[system]
        build = continuation_prompt_ids if system == "base" else chat_prompt_ids
        gen = generate_batch(model, self.tok, [build(self.tok, prompt)], cfg, self.device,
                             torch.Generator().manual_seed(cfg.seed))[0]
        return {"text": gen.text, "status": gen.status, "tokens": len(gen.response_ids), "seconds": time.time() - started,
                "mode": "continuation" if system == "base" else "chat"}

    @torch.no_grad()
    def reward_score(self, prompt: str, text: str) -> Optional[float]:
        if self.reward is None or not text:
            return None
        return float(score_sequences(self.reward, [encode_sequence(self.tok, prompt, text)], self.tok.pad_id, self.device)[0])


def mask_terms(text: str) -> tuple:
    count = 0
    for marker in sorted(set(all_harm_markers()), key=len, reverse=True):
        if marker in text:
            count += text.count(marker)
            text = text.replace(marker, "▇" * 3)
    return text, count


def detect_language(text: str) -> Optional[str]:
    fractions = {lang: script_fraction(text, lang) for lang in LANGUAGES}
    best = max(fractions, key=fractions.get)
    return best if fractions[best] >= 0.5 else None


def guess_intent(prompt: str, lang: str) -> str:
    """Only used to pick the rubric for the on-screen proxy label."""

    return "harmful" if keyword_baseline.blocked(prompt, lang) else "sensitive_benign"


def badge(label: dict, mode: str, status: str) -> str:
    if status == "exception":
        return '<span class="badge b-error">ERROR (not a refusal)</span>'
    if label["behavior"] == "generation_failure":
        return f'<span class="badge b-error">GENERATION FAILURE: {html.escape(status)}</span>'
    if mode == "continuation":
        return '<span class="badge b-cont">CONTINUATION — base model is not instruction-tuned</span>'
    if label["refusal"]:
        return '<span class="badge b-refusal">REFUSAL / REDIRECTION</span>'
    if label["harmful"]:
        return '<span class="badge b-unsafe">ABUSIVE CONTENT DETECTED</span>'
    if label["behavior"] == "direct_helpful_answer":
        return '<span class="badge b-answer">ANSWER</span>'
    return '<span class="badge b-amb">ANSWER (relevance unclear)</span>'


def card(title: str, cid: str, body: str, badge_html: str, meta: str) -> str:
    return (f'<div class="card"><h3>{html.escape(title)}</h3><div class="cid">{html.escape(cid)}</div>'
            f'<div style="margin-top:8px">{badge_html}</div><div class="out">{body}</div><div class="meta">{meta}</div></div>')


def empty_card(title: str, message: str = "—") -> str:
    return f'<div class="card"><h3>{html.escape(title)}</h3><div class="out" style="opacity:.5">{html.escape(message)}</div></div>'


def read_text(path: Path, fallback: str) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else fallback


def lineage_markdown(backend: Backend) -> str:
    rows = ["| role | checkpoint id | parent | step | weights sha256 | tokenizer |", "|---|---|---|---|---|---|"]
    for key in ("base", "sft", "ppo"):
        if key in backend.meta:
            m = backend.meta[key]
            rows.append(f"| {key} | `{m['checkpoint_id']}` | `{m.get('parent_checkpoint')}` | {m['step']} | "
                        f"`{m['weights_sha256'][:12]}` | `{m['tokenizer_fingerprint'][:12]}` |")
        else:
            rows.append(f"| {key} | **unavailable** — {html.escape(backend.errors.get(key, 'not found'))} | | | | |")
    rm = backend.root / f"{CKPT_DIR}/reward/bundle.json"
    if rm.exists():
        m = json.loads(rm.read_text())
        rows.append(f"| reward | `{m['checkpoint_id']}` | `{m.get('parent_checkpoint')}` | {m['step']} | `{m['weights_sha256'][:12]}` | "
                    f"`{m['tokenizer_fingerprint'][:12]}` |")
        gate = m.get("quality_gate", {})
        acc = m.get("heldout_accuracy", {}).get("all", {})
        rows += ["", f"Reward model held-out pair accuracy: **{100 * acc.get('accuracy', 0):.1f}%** "
                     f"({acc.get('correct')}/{acc.get('total')}); PPO quality gate passed: **{gate.get('passed')}**"]
    return "\n".join(rows)


def build_app(backend: Optional[Backend] = None):
    import gradio as gr

    backend = backend or Backend()
    available = backend.available()
    case_choices = {lang: [(f"{c['case_id']} — {c['prompt'][:60]}", c["case_id"]) for c in backend.cases if c["language"] == lang]
                    for lang in LANGUAGES}
    case_by_id = {c["case_id"]: c for c in backend.cases}

    def on_language(lang):
        return gr.update(choices=case_choices.get(lang, []), value=None)

    def on_case(case_id):
        if not case_id:
            return gr.update(), ""
        c = case_by_id[case_id]
        return c["prompt"], f"**{c['case_id']}** · expected: `{c['expected_behavior']}` · check: {c['check']}"

    def on_prompt(prompt, lang):
        if not prompt.strip():
            return ""
        n = len(backend.tok.encode(prompt))
        detected = detect_language(prompt)
        note = f"{n} / {MAX_PROMPT_TOKENS} prompt tokens"
        if detected is None:
            note += " · script: mixed/unknown"
        elif detected != lang:
            note += f" · ⚠ prompt looks like {LANGUAGE_NAMES[detected]}, selected {LANGUAGE_NAMES[lang]}"
        else:
            note += f" · script: {LANGUAGE_NAMES[detected]}"
        return note

    def generate(prompt, lang, systems, temperature, top_p, max_new, seed, mask, case_id):
        outs = {k: empty_card(SYSTEMS[k][0], "not selected") for k in SYSTEMS}
        for k in SYSTEMS:
            if k not in available:
                outs[k] = empty_card(SYSTEMS[k][0], f"unavailable: {backend.errors.get(k, 'not loaded')}")
        prompt = (prompt or "").strip()
        if not prompt:
            return (*outs.values(), "⚠ Enter a prompt first.")
        n_tok = len(backend.tok.encode(prompt))
        if n_tok > MAX_PROMPT_TOKENS:
            return (*outs.values(), f"⚠ The prompt is {n_tok} tokens; the limit is {MAX_PROMPT_TOKENS}. Please shorten it.")
        cfg = DecodingConfig(max_new_tokens=int(min(max_new, MAX_NEW_TOKENS_CAP)), temperature=float(temperature),
                             top_p=float(top_p), seed=int(seed), max_prompt_tokens=MAX_PROMPT_TOKENS)
        intent = case_by_id[case_id]["intent"] if case_id and case_by_id.get(case_id, {}).get("prompt") == prompt else guess_intent(prompt, lang)
        run_meta = {"prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest()[:16], "language": lang,
                    "prompt_tokens": n_tok, "decoding": cfg.to_dict(), "tokenizer": backend.tok.fingerprint[:16],
                    "rubric_intent_for_proxy_label": intent, "checkpoints": {}}
        for k in systems:
            if k not in available:
                continue
            title = SYSTEMS[k][0]
            cid = backend.meta.get("sft" if k == "kw" else k, {}).get("checkpoint_id", "")
            try:
                res = backend.run(k, prompt, lang, cfg, intent)
            except Exception as exc:  # noqa: BLE001
                outs[k] = card(title, cid, html.escape(f"{type(exc).__name__}: {exc}"), badge({}, "", "exception"), "")
                run_meta["checkpoints"][k] = {"status": "exception", "error": str(exc)}
                continue
            label = label_response(res["text"], res["status"], lang, intent)
            shown, masked = mask_terms(res["text"]) if mask else (res["text"], 0)
            score = backend.reward_score(prompt, res["text"]) if res["mode"] != "continuation" else None
            meta = (f"status <b>{res['status']}</b> · {res['tokens']} new tokens · {res['seconds']:.1f}s"
                    + (f" · reward-model score {score:+.2f} (diagnostic)" if score is not None else "")
                    + (f" · {masked} term(s) masked" if masked else "")
                    + (f"<br>{html.escape(res['note'])}" if res.get("note") else "")
                    + f"<br>proxy label: <code>{label['behavior']}</code> · language match: {label['language_match']}")
            body = html.escape(shown) if shown else "<i>(empty output)</i>"
            outs[k] = card(title, cid, body, badge(label, res["mode"], res["status"]), meta)
            run_meta["checkpoints"][k] = {"checkpoint_id": cid, "status": res["status"], "tokens": res["tokens"],
                                          "proxy_label": label["behavior"], "reward_score": score}
        return (*outs.values(), "```json\n" + json.dumps(run_meta, ensure_ascii=False, indent=1) + "\n```")

    def reset():
        return ("", None, "", "", *[empty_card(SYSTEMS[k][0]) for k in SYSTEMS], "")

    results_md = read_text(backend.root / "artifacts/eval/latest/summary.md", "_No evaluation has been run yet — `python scripts/evaluate.py`._")
    figures = [p for p in sorted((backend.root / "artifacts/reports").glob("*.png"))] if (backend.root / "artifacts/reports").exists() else []

    with gr.Blocks(title="IndicSafeLM demo") as app:
        gr.Markdown("# IndicSafeLM — side-by-side checkpoint comparison\n"
                    "Hindi · Bengali · Tamil · ~42M-parameter decoder trained from scratch → SFT → reward model → PPO")
        gr.HTML('<div class="warn"><b>Research prototype.</b> This small model can produce incorrect or harmful text and is '
                'not a production safety system. Labels on this screen are automatic proxies, not human review. '
                f'Loaded: {", ".join(SYSTEMS[k][0] for k in available) or "none"}'
                + (f' · Unavailable: {html.escape("; ".join(f"{k}: {v}" for k, v in backend.errors.items()))}' if backend.errors else "")
                + "</div>")
        with gr.Tab("Compare"):
            with gr.Row():
                lang = gr.Radio(LANG_CHOICES, value="hi", label="Language")
                case = gr.Dropdown(case_choices["hi"], label="Rehearsal case (Appendix A — not part of the benchmark)", value=None)
            case_note = gr.Markdown("")
            prompt = gr.Textbox(label="Prompt (paste Unicode text in any target script)", lines=3, elem_id="prompt")
            prompt_note = gr.Markdown("")
            with gr.Row():
                systems = gr.CheckboxGroup([(SYSTEMS[k][0], k) for k in SYSTEMS if k in available], value=available,
                                           label="Checkpoints (all receive the identical prompt and settings)")
                mask = gr.Checkbox(value=True, label="Classroom mode: mask abusive terms in outputs")
            with gr.Accordion("Decoding settings (shared by every checkpoint)", open=False):
                with gr.Row():
                    temperature = gr.Slider(0.0, 1.5, value=0.7, step=0.05, label="temperature (0 = greedy)")
                    top_p = gr.Slider(0.1, 1.0, value=0.9, step=0.05, label="top-p")
                    max_new = gr.Slider(16, MAX_NEW_TOKENS_CAP, value=128, step=8, label=f"max new tokens (cap {MAX_NEW_TOKENS_CAP})")
                    seed = gr.Number(value=1735, precision=0, label="seed")
            with gr.Row():
                go = gr.Button("Generate", variant="primary")
                clear = gr.Button("Reset for next participant")
            with gr.Row(equal_height=True):
                cards = [gr.HTML(empty_card(SYSTEMS[k][0])) for k in SYSTEMS]
            run_meta = gr.Markdown("")
            lang.change(on_language, lang, case)
            case.change(on_case, case, [prompt, case_note])
            prompt.change(on_prompt, [prompt, lang], prompt_note)
            go.click(generate, [prompt, lang, systems, temperature, top_p, max_new, seed, mask, case], [*cards, run_meta],
                     concurrency_limit=1)
            prompt.submit(generate, [prompt, lang, systems, temperature, top_p, max_new, seed, mask, case], [*cards, run_meta],
                          concurrency_limit=1)
            clear.click(reset, None, [prompt, case, case_note, prompt_note, *cards, run_meta])
        with gr.Tab("Results"):
            gr.Markdown(results_md)
            for fig in figures:
                gr.Image(str(fig), label=fig.stem.replace("_", " "), show_label=True)
        with gr.Tab("Artifacts & lineage"):
            gr.Markdown(lineage_markdown(backend))
            manifest = backend.root / "data/dataset_manifest.json"
            if manifest.exists():
                m = json.loads(manifest.read_text())
                gr.Markdown("### Datasets\n" + "\n".join(f"- `{k}`: {v['rows']} rows {v['by_language']}" for k, v in m["files"].items()))
            corpus = backend.root / "data/processed/corpus-v1/report.json"
            if corpus.exists():
                c = json.loads(corpus.read_text())
                gr.Markdown("### Corpus release `corpus-v1`\n"
                            f"Split counts by language: `{c['split_counts_by_language']}`  \n"
                            f"Rejections by reason: `{c['rejection_counts_by_reason']}`  \n"
                            f"Exact cross-split leakage: `{c['leakage']['exact_fingerprint_intersections']}`")
            tokrep = backend.root / "artifacts/tokenizer/v1/validation_report.json"
            if tokrep.exists():
                t = json.loads(tokrep.read_text())
                gr.Markdown("### Tokenizer\n| lang | round trip | tokens/char | tokens/word | unk rate |\n|---|---|---|---|---|\n" + "\n".join(
                    f"| {l} | {v['round_trip_rate']:.3f} | {v['tokens_per_character']:.3f} | {v['tokens_per_word']:.2f} | {v['unknown_rate']:.4f} |"
                    for l, v in t["languages"].items()))
    return app


def main(port: int = 7860, share: bool = False) -> None:
    import gradio as gr

    app = build_app()
    app.queue().launch(server_name="127.0.0.1", server_port=port, share=share, css=CSS, theme=gr.themes.Soft())
