"""Build the SFT, preference, PPO-prompt and evaluation datasets.

Design (sections 6.6, 7.1-7.2, 8.2 and Appendix A.1):

* A **prompt family** is one intent with one set of slot values, rendered in
  all three languages and all paraphrases (translations share a family ID
  because slot values are chosen by *index*, the same index in every language).
* **Training families** use paraphrase templates 0-1 and the training slot
  values. They feed SFT, the preference pairs and the PPO prompt pool.
* **Held-out families** use paraphrase template 2 and held-out slot values
  (names, relations, regions, quoted words, rewrite cases, Wikipedia titles
  from the validation/test corpus split). Each held-out family is assigned by
  hash to exactly one of ``rm_validation``, ``dev`` (PPO monitoring) or
  ``test`` (final frozen benchmark). No held-out prompt string occurs in any
  training file; ``build_all`` asserts this.
* SFT data mimics *uncurated* instruction data: a configurable share of harmful
  requests is answered compliantly and a share of sensitive-but-benign
  requests is refused. This gives the SFT checkpoint measurable safety and
  over-refusal errors, so the preference/PPO stage can be tested against it.
  The rates are recorded in the dataset manifest.
"""

from __future__ import annotations

import hashlib
import importlib
import random
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ..utils import LANGUAGES, iter_jsonl, write_json, write_jsonl

HARMFUL = {"direct_insult", "harassment", "escalation_request", "hate_or_toxicity", "profanity"}
SENSITIVE = {"benign_sensitive_context"}
ORDINARY = {"ordinary_helpful"}
SCRIPT_TAG = {"hi": "Deva", "bn": "Beng", "ta": "Taml"}


def intent_of(category: str) -> str:
    if category in HARMFUL:
        return "harmful"
    if category in SENSITIVE:
        return "sensitive_benign"
    return "ordinary_benign"


def expected_behavior(category: str) -> str:
    return "refuse_or_redirect" if category in HARMFUL else "answer"


def load_banks() -> Dict[str, dict]:
    return {lang: importlib.import_module(f"indicsafelm.instructions.lang_{lang}").LANG for lang in LANGUAGES}


@dataclass
class BuildConfig:
    seed: int = 2027
    sft_families_per_concept: int = 50
    pref_families_per_concept: int = 60
    heldout_families_per_concept: int = 30
    tell_about_train_titles: int = 120
    tell_about_heldout_titles: int = 30
    sft_harmful_compliance_rate: float = 0.5
    sft_sensitive_refusal_rate: float = 0.2
    sft_unclear_per_language: int = 30


# ---------------------------------------------------------------- slot values
SLOT_POOLS = {
    # slot: (train list key, held-out list key) — held-out uses a disjoint list
    "name": ("names_train", "names_test"),
    "relation": ("relations_train", "relations_test"),
    "region": ("regions_train", "regions_test"),
    "word": ("insult_words_train", "insult_words_test"),
}
SHARED_POOLS = {"day": "days", "time": "times", "subject": "subjects"}


def _pool_size(banks: Dict[str, dict], key: str) -> int:
    sizes = {len(bank[key]) for bank in banks.values()}
    if len(sizes) != 1:
        raise ValueError(f"slot list {key} differs in length across languages: {sizes}")
    return sizes.pop()


def sample_family_indices(rng: random.Random, banks: Dict[str, dict], held_out: bool) -> Dict[str, int]:
    idx = {}
    for slot, (train_key, test_key) in SLOT_POOLS.items():
        idx[slot] = rng.randrange(_pool_size(banks, test_key if held_out else train_key))
    for slot, key in SHARED_POOLS.items():
        idx[slot] = rng.randrange(_pool_size(banks, key))
    cases = _pool_size(banks, "rewrite_cases")
    idx["case"] = rng.randrange(cases - 2, cases) if held_out else rng.randrange(0, cases - 2)
    return idx


def slot_values(bank: dict, idx: Dict[str, int], held_out: bool) -> Dict[str, str]:
    values = {}
    for slot, (train_key, test_key) in SLOT_POOLS.items():
        values[slot] = bank[test_key if held_out else train_key][idx[slot]]
    for slot, key in SHARED_POOLS.items():
        values[slot] = bank[key][idx[slot]]
    case = bank["rewrite_cases"][idx["case"]]
    values["angry"] = case["angry"].format(**values)
    values["polite"] = case["polite"].format(**values)
    return values


def fill(template: str, values: Dict[str, str]) -> str:
    return template.format(**{k: v for k, v in values.items()})


# ------------------------------------------------------------ Wikipedia titles
FIRST_SENTENCE = re.compile(r"^(.+?[।.!?])(\s|$)", re.DOTALL)


def wiki_facts(root: Path, corpus_dir: Path, raw_dir: Path, splits: Tuple[str, ...], per_language: int, seed: int) -> Dict[str, List[dict]]:
    """Title + first sentence of short, clean articles from the given splits."""

    import pyarrow.parquet as pq

    wanted = defaultdict(dict)
    for row in iter_jsonl(corpus_dir / "processed.jsonl"):
        if row["split"] in splits:
            wanted[row["language"]][row["source_document_id"]] = row["text"]
    facts = {}
    rng = random.Random(seed)
    for lang in LANGUAGES:
        table = pq.read_table(raw_dir / f"{lang}-train-00000-of-00002.parquet", columns=["id", "title"])
        titles = dict(zip(table.column("id").to_pylist(), table.column("title").to_pylist()))
        candidates = []
        for doc_id, text in wanted[lang].items():
            title = titles.get(doc_id)
            match = FIRST_SENTENCE.match(text.strip())
            if not title or not match or "(" in title or len(title) > 30:
                continue
            sentence = match.group(1).strip()
            if 40 <= len(sentence) <= 220 and title.split()[0] in sentence:
                candidates.append({"doc_id": doc_id, "title": title, "summary": sentence})
        candidates.sort(key=lambda item: item["doc_id"])
        rng.shuffle(candidates)
        facts[lang] = candidates[:per_language]
    return facts


# ----------------------------------------------------------- response builders
def refusal_text(rng: random.Random, bank: dict, concept: dict, values: Dict[str, str]) -> str:
    opener = rng.choice(bank["refusal_openers"])
    reason = fill(concept["reason"], values)
    alternative = fill(concept["alternative"], values)
    style = rng.randrange(3)
    if style == 0:
        return f"{opener} {reason} {alternative}"
    if style == 1:
        return f"{reason} {opener} {alternative}"
    return f"{opener} {alternative}"


def helpful_text(rng: random.Random, concept: dict, values: Dict[str, str]) -> str:
    return fill(rng.choice(concept["answers"]), values)


def compliance_text(rng: random.Random, concept: dict, values: Dict[str, str]) -> str:
    return fill(rng.choice(concept["compliance"]), values)


def family_split(family_id: str) -> str:
    bucket = int(hashlib.sha256(family_id.encode()).hexdigest(), 16) % 4
    return ("rm_validation", "dev", "test", "test")[bucket]


def prompt_id(*parts: str) -> str:
    return hashlib.sha256("\x1f".join(parts).encode()).hexdigest()[:16]


# ------------------------------------------------------------------ builders
def build_all(root: Path, cfg: Optional[BuildConfig] = None) -> dict:
    cfg = cfg or BuildConfig()
    banks = load_banks()
    concepts = list(banks["hi"]["concepts"])
    for lang, bank in banks.items():
        if list(bank["concepts"]) != concepts:
            raise ValueError(f"{lang} bank concepts differ from Hindi")
    rng = random.Random(cfg.seed)
    corpus = root / "data/processed/corpus-v1"
    raw = root / "data/raw/wikipedia-20231101"
    train_facts = wiki_facts(root, corpus, raw, ("train",), cfg.tell_about_train_titles, cfg.seed)
    heldout_facts = wiki_facts(root, corpus, raw, ("validation", "test"), cfg.tell_about_heldout_titles, cfg.seed + 1)

    sft_rows, pref_rows, ppo_rows, heldout_rows = [], [], [], []

    def render(lang: str, concept_key: str, template_index: int, values: Dict[str, str]) -> str:
        return fill(banks[lang]["concepts"][concept_key]["prompts"][template_index], values)

    def training_families(concept_key: str, count: int, tag: str):
        """Yield (family_id, lang, template_index, values) for training families."""

        if concept_key == "tell_about":
            for lang in LANGUAGES:
                for fact in train_facts[lang][:count]:
                    values = {"title": fact["title"], "summary": fact["summary"]}
                    yield f"{concept_key}:{lang}:{fact['doc_id']}", lang, rng.randrange(2), values, None
            return
        for _ in range(count):
            idx = sample_family_indices(rng, banks, held_out=False)
            fam = f"{concept_key}:{tag}:" + ",".join(f"{k}{v}" for k, v in sorted(idx.items()))
            for lang in LANGUAGES:
                yield fam, lang, rng.randrange(2), slot_values(banks[lang], idx, held_out=False), idx

    # SFT ------------------------------------------------------------------
    for concept_key in concepts:
        for fam, lang, t_idx, values, _ in training_families(concept_key, cfg.sft_families_per_concept, "sft"):
            bank = banks[lang]
            concept = bank["concepts"][concept_key]
            category = concept["category"]
            prompt = render(lang, concept_key, t_idx, values)
            if category in HARMFUL:
                if rng.random() < cfg.sft_harmful_compliance_rate:
                    response, behavior = compliance_text(rng, concept, values), "harmful_compliance"
                else:
                    response, behavior = refusal_text(rng, bank, concept, values), "safe_redirection"
            elif category in SENSITIVE and rng.random() < cfg.sft_sensitive_refusal_rate:
                response, behavior = rng.choice(bank["false_refusals"]), "false_refusal"
            else:
                response, behavior = helpful_text(rng, concept, values), "direct_helpful_answer"
            sft_rows.append({
                "example_id": prompt_id("sft", fam, lang, prompt, response), "family_id": fam, "language": lang,
                "concept": concept_key, "category": category, "intent": intent_of(category),
                "template_index": t_idx, "prompt": prompt, "response": response, "response_behavior": behavior,
                "source": "authored_template", "split": "sft_train",
            })
    for lang in LANGUAGES:
        unclear = banks[lang]["unclear"]
        for i in range(cfg.sft_unclear_per_language):
            prompt = unclear["prompts"][i % len(unclear["prompts"])]
            response = unclear["answers"][i % len(unclear["answers"])]
            sft_rows.append({
                "example_id": prompt_id("sft", "unclear", lang, prompt, response), "family_id": f"unclear:{lang}:{i % 3}",
                "language": lang, "concept": "unclear", "category": "unclear_request", "intent": "ordinary_benign",
                "template_index": i % 3, "prompt": prompt, "response": response, "response_behavior": "clarification",
                "source": "authored_template", "split": "sft_train",
            })
    # de-duplicate identical (prompt, response) pairs, then hold out 5% of families for SFT validation
    seen, unique = set(), []
    for row in sft_rows:
        key = (row["prompt"], row["response"])
        if key not in seen:
            seen.add(key)
            unique.append(row)
    sft_rows = unique
    for row in sft_rows:
        if int(hashlib.sha256(row["family_id"].encode()).hexdigest(), 16) % 20 == 0:
            row["split"] = "sft_validation"

    # Preference pairs + PPO prompt pool -----------------------------------
    other = {lang: [l for l in LANGUAGES if l != lang] for lang in LANGUAGES}
    for concept_key in concepts:
        fams = list(training_families(concept_key, cfg.pref_families_per_concept, "pref"))
        for fam, lang, t_idx, values, idx in fams:
            bank = banks[lang]
            concept = bank["concepts"][concept_key]
            category = concept["category"]
            prompt = render(lang, concept_key, t_idx, values)
            if category in HARMFUL:
                chosen = refusal_text(rng, bank, concept, values)
                if rng.random() < 0.9:
                    rejected, reason = compliance_text(rng, concept, values), "harmful_compliance"
                else:
                    wrong = rng.choice(other[lang])
                    wrong_values = slot_values(banks[wrong], idx, held_out=False)
                    rejected = refusal_text(rng, banks[wrong], banks[wrong]["concepts"][concept_key], wrong_values)
                    reason = "wrong_language"
            else:
                chosen = helpful_text(rng, concept, values)
                draw = rng.random()
                refusal_share = 0.6 if category in SENSITIVE else 0.4
                if draw < refusal_share:
                    rejected, reason = rng.choice(bank["false_refusals"]), "false_refusal"
                elif draw < refusal_share + 0.2 and concept_key != "tell_about":
                    wrong = rng.choice(other[lang])
                    wrong_values = slot_values(banks[wrong], idx, held_out=False)
                    rejected, reason = helpful_text(rng, banks[wrong]["concepts"][concept_key], wrong_values), "wrong_language"
                else:
                    distractors = [c for c in concepts if c != concept_key and intent_of(bank["concepts"][c]["category"]) != "harmful" and c not in ("tell_about", "rewrite_angry")]
                    distract = bank["concepts"][rng.choice(distractors)]
                    dvalues = values if idx is not None else slot_values(bank, sample_family_indices(rng, banks, False), False)
                    rejected, reason = helpful_text(rng, distract, dvalues), "irrelevant"
            if chosen == rejected:
                continue
            pref_rows.append({
                "pair_id": prompt_id("pair", fam, lang, prompt, chosen, rejected), "prompt_id": prompt_id("prompt", lang, prompt),
                "family_id": fam, "language": lang, "concept": concept_key, "category": category,
                "intent": intent_of(category), "prompt": prompt, "chosen": chosen, "rejected": rejected,
                "rejected_reason": reason, "review_status": "author_reviewed_template", "split": "pref_train",
            })
            ppo_rows.append({
                "prompt_id": prompt_id("prompt", lang, prompt), "family_id": fam, "language": lang, "concept": concept_key,
                "category": category, "intent": intent_of(category), "prompt": prompt, "split": "ppo_train",
            })

    # Held-out families -----------------------------------------------------
    # Every held-out family uses held-out slot values. It is rendered with the
    # held-out wording (template 2) *and* the training wordings (templates 0-1);
    # any rendering that exactly matches a training prompt is dropped. The two
    # wordings are reported as separate slices ("heldout_template" / "seen_template").
    training_prompt_set = {r["prompt"] for r in sft_rows} | {r["prompt"] for r in pref_rows}
    for concept_key in concepts:
        if concept_key == "tell_about":
            families = []
            for lang in LANGUAGES:
                for fact in heldout_facts[lang]:
                    families.append((f"tell_about:{lang}:{fact['doc_id']}", {lang: {"title": fact["title"], "summary": fact["summary"]}}))
        else:
            families = []
            for _ in range(cfg.heldout_families_per_concept):
                idx = sample_family_indices(rng, banks, held_out=True)
                fam = f"{concept_key}:heldout:" + ",".join(f"{k}{v}" for k, v in sorted(idx.items()))
                families.append((fam, {lang: slot_values(banks[lang], idx, held_out=True) for lang in LANGUAGES}))
        for fam, per_lang in families:
            split = family_split(fam)
            for lang, values in per_lang.items():
              bank = banks[lang]
              concept = bank["concepts"][concept_key]
              category = concept["category"]
              for t_idx in (2, 0, 1):
                prompt = render(lang, concept_key, t_idx, values)
                if prompt in training_prompt_set:
                    continue
                references = [fill(a, values) for a in concept.get("answers", [])]
                keywords = [values[k] for k in ("name", "day", "time", "title") if k in values and "{" + k + "}" in concept["prompts"][t_idx]]
                row = {
                    "prompt_id": prompt_id("prompt", lang, prompt), "family_id": fam, "language": lang,
                    "script": SCRIPT_TAG[lang], "concept": concept_key, "category": category, "intent": intent_of(category),
                    "expected_behavior": expected_behavior(category), "prompt": prompt, "reference_answers": references,
                    "expected_keywords": keywords, "template_index": t_idx,
                    "wording": "heldout_template" if t_idx == 2 else "seen_template", "source": "authored_template",
                    "review_disposition": "approved_by_author_pending_native_review", "split": split,
                }
                if category in HARMFUL:
                    row["reference_refusal"] = refusal_text(rng, bank, concept, values)
                    row["reference_compliance"] = compliance_text(rng, concept, values)
                heldout_rows.append(row)

    # Unique held-out prompts per split; duplicate wording is dropped.
    dedup, seen = [], set()
    for row in heldout_rows:
        if row["prompt"] not in seen:
            seen.add(row["prompt"])
            dedup.append(row)
    heldout_rows = dedup

    # RM validation pairs from rm_validation families
    rm_val_pairs = []
    for row in heldout_rows:
        if row["split"] != "rm_validation":
            continue
        lang = row["language"]
        bank = banks[lang]
        if row["intent"] == "harmful":
            chosen, rejected, reason = row["reference_refusal"], row["reference_compliance"], "harmful_compliance"
        else:
            chosen = row["reference_answers"][0] if row["concept"] != "tell_about" else None
            if chosen is None:
                continue
            rejected, reason = rng.choice(bank["false_refusals"]), "false_refusal"
        rm_val_pairs.append({
            "pair_id": prompt_id("pair", row["family_id"], lang, row["prompt"], chosen, rejected), "prompt_id": row["prompt_id"],
            "family_id": row["family_id"], "language": lang, "concept": row["concept"], "category": row["category"],
            "intent": row["intent"], "prompt": row["prompt"], "chosen": chosen, "rejected": rejected,
            "rejected_reason": reason, "review_status": "author_reviewed_template", "split": "pref_validation",
        })
    # tell_about held-out pairs (chosen = summary, rejected = refusal)
    for row in heldout_rows:
        if row["split"] == "rm_validation" and row["concept"] == "tell_about":
            chosen = row["reference_answers"][0]
            rejected = rng.choice(banks[row["language"]]["false_refusals"])
            rm_val_pairs.append({**{k: row[k] for k in ("prompt_id", "family_id", "language", "concept", "category", "intent", "prompt")},
                                 "pair_id": prompt_id("pair", row["prompt"], chosen, rejected), "chosen": chosen, "rejected": rejected,
                                 "rejected_reason": "false_refusal", "review_status": "author_reviewed_template", "split": "pref_validation"})

    # Length-control pairs (Table 7.4, "scores versus response length"): the
    # rejected reply is a full-length answer that is irrelevant or in another
    # language, so length alone cannot pick the chosen reply.
    siblings = {(r["family_id"], r["template_index"], r["language"]): r for r in heldout_rows}
    benign_concepts = [c for c in concepts if c not in ("tell_about", "rewrite_angry")
                       and intent_of(banks["hi"]["concepts"][c]["category"]) != "harmful"]
    for row in heldout_rows:
        if row["split"] != "rm_validation" or row["concept"] == "tell_about":
            continue
        lang = row["language"]
        base = {k: row[k] for k in ("prompt_id", "family_id", "language", "concept", "category", "intent", "prompt")}
        chosen = row["reference_refusal"] if row["intent"] == "harmful" else row["reference_answers"][0]
        other_lang = rng.choice([l for l in LANGUAGES if l != lang])
        sib = siblings.get((row["family_id"], row["template_index"], other_lang))
        if sib is not None:
            wrong = sib["reference_refusal"] if row["intent"] == "harmful" else sib["reference_answers"][0]
            rm_val_pairs.append({**base, "pair_id": prompt_id("pair", row["prompt"], chosen, wrong), "chosen": chosen,
                                 "rejected": wrong, "rejected_reason": "wrong_language",
                                 "review_status": "author_reviewed_template", "split": "pref_validation"})
        if row["intent"] != "harmful":
            distract = banks[lang]["concepts"][rng.choice([c for c in benign_concepts if c != row["concept"]])]
            dvalues = slot_values(banks[lang], sample_family_indices(rng, banks, True), True)
            irrelevant = helpful_text(rng, distract, dvalues)
            rm_val_pairs.append({**base, "pair_id": prompt_id("pair", row["prompt"], chosen, irrelevant), "chosen": chosen,
                                 "rejected": irrelevant, "rejected_reason": "irrelevant",
                                 "review_status": "author_reviewed_template", "split": "pref_validation"})

    # Leakage checks ------------------------------------------------------
    train_prompts = {r["prompt"] for r in sft_rows} | {r["prompt"] for r in pref_rows}
    heldout_prompts = {r["prompt"] for r in heldout_rows}
    leaked = train_prompts & heldout_prompts
    if leaked:
        raise AssertionError(f"{len(leaked)} held-out prompts also appear in training data, e.g. {next(iter(leaked))}")
    train_families = {r["family_id"] for r in sft_rows} | {r["family_id"] for r in pref_rows}
    if train_families & {r["family_id"] for r in heldout_rows}:
        raise AssertionError("a prompt family crosses the training / held-out boundary")
    split_of_family = {}
    for row in heldout_rows:
        if split_of_family.setdefault(row["family_id"], row["split"]) != row["split"]:
            raise AssertionError("a held-out family spans several splits")

    # Demo cases (Appendix A) — separate split, excluded from all scoring.
    demo_rows = []
    for lang in LANGUAGES:
        for case in banks[lang]["demo_cases"]:
            category = banks[lang]["concepts"][case["concept"]]["category"]
            demo_rows.append({**case, "language": lang, "category": category, "intent": intent_of(category),
                              "expected_behavior": expected_behavior(category), "split": "demo"})
    demo_prompts = {r["prompt"] for r in demo_rows}
    if demo_prompts & (train_prompts | heldout_prompts):
        raise AssertionError("a demo rehearsal prompt also appears in training or benchmark data")

    out = root / "data"
    ppo_unique = list({r["prompt_id"]: r for r in ppo_rows}.values())
    files = {
        "instructions/sft.jsonl": sft_rows,
        "preferences/pairs_train.jsonl": pref_rows,
        "preferences/pairs_validation.jsonl": rm_val_pairs,
        "ppo/prompts_train.jsonl": ppo_unique,
        "eval/dev_prompts.jsonl": [r for r in heldout_rows if r["split"] == "dev"],
        "eval/test_prompts.jsonl": [r for r in heldout_rows if r["split"] == "test"],
        "eval/demo_cases.jsonl": demo_rows,
    }
    manifest = {"schema_version": 1, "config": asdict(cfg), "files": {}, "leakage_checks": {
        "heldout_prompt_overlap_with_training": 0, "heldout_family_overlap_with_training": 0,
        "demo_prompt_overlap": 0, "families_span_one_split": True}}
    for rel, rows in files.items():
        write_jsonl(out / rel, rows)
        manifest["files"][rel] = {
            "rows": len(rows),
            "by_language": dict(Counter(r["language"] for r in rows)),
            "by_category": dict(Counter(r["category"] for r in rows)),
        }
    manifest["sft_response_behavior"] = dict(Counter(r["response_behavior"] for r in sft_rows))
    manifest["pref_rejected_reason"] = dict(Counter(r["rejected_reason"] for r in pref_rows))
    write_json(out / "dataset_manifest.json", manifest)
    return manifest
