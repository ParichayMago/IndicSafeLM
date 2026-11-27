"""IndicSafe Sahayak: an application built on the aligned checkpoints.

A classroom messaging assistant for Hindi, Bengali and Tamil. A student asks
for a reminder, a thank-you note, a birthday wish, a polite version of an
angry message, or advice after a rude message. Requests to insult, mock,
threaten or exclude someone are declined, with a respectful alternative.

Serving pipeline (every step is shown in the UI):

1. The PPO policy drafts ``n`` candidate replies (sampling, one batch).
2. Drafts not written in the user's script are dropped (if any remain), then
   the trained reward model scores each draft and the highest-scoring one is
   kept (best-of-n reranking).
3. An output safety check scans the chosen reply for abusive terms outside
   quotation marks. A flagged reply is replaced by a safe redirection in the
   user's language, and the replacement is reported, never hidden.
"""

from __future__ import annotations

import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import torch

from ..evaluation.labeler import bank, has_harm, is_refusal, script_fraction
from ..model.checkpoint import load_bundle
from ..model.generation import DecodingConfig, chat_prompt_ids, generate_batch
from ..reward.train import encode_sequence, score_sequences
from ..tokenizer import TokenizerBundle
from ..utils import LANGUAGES, ROOT, pick_device

CKPT_DIR = os.environ.get("INDICSAFELM_CKPT_DIR", "artifacts/checkpoints")


@dataclass
class Reply:
    text: str
    kind: str            # helpful | declined | replaced | error
    candidates: int
    reward: Optional[float]
    seconds: float
    note: str = ""


class Assistant:
    def __init__(self, root: Path = ROOT, policy: str = "ppo", candidates: int = 8) -> None:
        self.device = pick_device(os.environ.get("INDICSAFELM_DEVICE", "auto"))
        self.tok = TokenizerBundle.load(root / "artifacts/tokenizer/v1")
        self.policy, self.policy_meta = load_bundle(root / f"{CKPT_DIR}/{policy}", self.tok, self.device)
        self.reward, self.reward_meta = load_bundle(root / f"{CKPT_DIR}/reward", self.tok, self.device)
        self.policy.eval()
        self.reward.eval()
        self.candidates = candidates
        self.rng = random.Random(0)

    def detect_language(self, text: str) -> Optional[str]:
        scores = {lang: script_fraction(text, lang) for lang in LANGUAGES}
        best = max(scores, key=scores.get)
        return best if scores[best] >= 0.5 else None

    def safe_fallback(self, lang: str) -> str:
        b = bank(lang)
        return f"{b['refusal_openers'][0]} {b['concepts']['insult_message']['alternative'].replace('{name}', '').replace('  ', ' ')}".strip()

    @torch.no_grad()
    def respond(self, message: str, lang: Optional[str] = None, seed: Optional[int] = None) -> Reply:
        started = time.time()
        message = message.strip()
        lang = lang or self.detect_language(message) or "hi"
        prompt_ids = chat_prompt_ids(self.tok, message)
        if len(prompt_ids) > 256:
            return Reply("", "error", 0, None, 0.0, "The message is too long (limit 256 tokens). Please shorten it.")
        cfg = DecodingConfig(max_new_tokens=128, temperature=0.8, top_p=0.95, seed=seed if seed is not None else self.rng.randrange(10**6))
        gens = generate_batch(self.policy, self.tok, [prompt_ids] * self.candidates, cfg, self.device)
        usable = [g for g in gens if g.status == "ok" and g.text]
        # Prefer drafts written in the user's script; fall back to all drafts.
        in_language = [g for g in usable if script_fraction(g.text, lang) >= 0.85]
        usable = in_language or usable
        if not usable:
            return Reply("", "error", len(gens), None, time.time() - started, "The model did not produce a usable reply. Please try again.")
        scores = score_sequences(self.reward, [encode_sequence(self.tok, message, g.text) for g in usable], self.tok.pad_id, self.device)
        best_index = int(torch.argmax(scores))
        best, best_score = usable[best_index], float(scores[best_index])
        text = best.text
        if has_harm(text):
            return Reply(self.safe_fallback(lang), "replaced", len(gens), best_score, time.time() - started,
                         "The safety check found abusive wording in the draft and replaced it.")
        kind = "declined" if is_refusal(text) else "helpful"
        return Reply(text, kind, len(gens), best_score, time.time() - started)


def examples(lang: str) -> List[str]:
    """Example requests built from the language bank (training-name slot values)."""

    from ..instructions.build import fill, load_banks, slot_values

    banks = load_banks()
    values = slot_values(banks[lang], {"name": 1, "relation": 0, "region": 0, "word": 0, "day": 0, "time": 1, "subject": 0, "case": 1}, False)
    picks = [("class_reminder", 0), ("thank_you", 0), ("birthday_wish", 0), ("ask_notebook", 0), ("rewrite_angry", 1),
             ("insult_message", 0), ("scare_message", 0)]
    return [fill(banks[lang]["concepts"][c]["prompts"][t], values) for c, t in picks]
