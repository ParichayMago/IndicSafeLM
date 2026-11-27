"""Keyword-filter baseline (section 7.8, Table 8.1).

A fixed term list is derived *mechanically* from the training prompt templates:
every word that occurs in a harmful-concept training template (or in the
training insult-word list) and never in an ordinary-task template. Prompts
containing any listed term are blocked with a fixed refusal; all other prompts
go to the SFT model unchanged. This is the lexical blocking a developer might
ship without a learned safety stage, and it shows what context-blind filtering
misses (harmful prompts with unlisted words) and what it over-blocks (benign
prompts that quote or discuss abuse).
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Dict, List, Tuple

from .labeler import bank

# Split on whitespace and punctuation. (\w would break Indic words at vowel
# signs and viramas, which Python does not treat as word characters.)
WORD = re.compile(r"[^\s.,!?।॥:;'\"“”‘’()\[\]{}…/\-–—]+")
HARMFUL = {"direct_insult", "harassment", "escalation_request", "hate_or_toxicity", "profanity"}


def _words(text: str) -> set:
    return set(WORD.findall(re.sub(r"\{[a-z]+\}", " ", text)))


@lru_cache(maxsize=None)
def blocklist(lang: str) -> Tuple[str, ...]:
    b = bank(lang)
    harmful, ordinary = set(), set()
    slot_words = set()
    for key in ("names_train", "names_test", "relations_train", "relations_test", "regions_train", "regions_test",
                "days", "times", "subjects"):
        for value in b[key]:
            slot_words |= _words(value)
    for concept in b["concepts"].values():
        words = set()
        for template in concept["prompts"][:2]:  # training templates only
            words |= _words(template)
        if concept["category"] in HARMFUL:
            harmful |= words
        elif concept["category"] == "ordinary_helpful":
            ordinary |= words
    terms = (harmful - ordinary - slot_words) | set(b["insult_words_train"])
    return tuple(sorted(t for t in terms if len(t) >= 2))


def blocked(prompt: str, lang: str) -> List[str]:
    words = _words(prompt)
    return [term for term in blocklist(lang) if term in words]


def filter_refusal(lang: str) -> str:
    return bank(lang)["false_refusals"][1]


def blocklist_summary() -> Dict[str, int]:
    return {lang: len(blocklist(lang)) for lang in ("hi", "bn", "ta")}
