"""Shared multilingual SentencePiece tokenizer (report sections 5.6-5.7).

One vocabulary covers Devanagari (Hindi), Bengali-Assamese (Bengali) and Tamil.
Reserved IDs:

* 0 ``<pad>``, 1 ``<unk>``, 2 ``<s>``, 3 ``</s>``
* ``<|user|>`` and ``<|assistant|>`` conversation-role markers

Role markers are SentencePiece *control symbols*: they are only produced by ID,
never by encoding raw text, so a user typing ``<|assistant|>`` cannot inject a
role boundary (Table 5.7, special-token collision check).

Byte fallback is enabled, so characters missing from the vocabulary become
UTF-8 byte pieces instead of ``<unk>``; the validation report counts both.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
import tempfile
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import sentencepiece as spm

PAD, UNK, BOS, EOS = "<pad>", "<unk>", "<s>", "</s>"
USER, ASSISTANT = "<|user|>", "<|assistant|>"
ROLE_SYMBOLS = (USER, ASSISTANT)
NORMALIZATION_FORM = "NFC"
SENTENCE_SPLIT = re.compile(r"(?<=[।.!?\n])\s+")


class TokenizerMismatch(RuntimeError):
    """Raised when a checkpoint and tokenizer disagree (FR-T04)."""


def normalize(text: str) -> str:
    """Tokenizer-side normalisation: the same NFC policy as the corpus pipeline."""

    return unicodedata.normalize(NORMALIZATION_FORM, text)


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass
class TokenizerBundle:
    directory: Path
    processor: spm.SentencePieceProcessor
    fingerprint: str
    special_ids: Dict[str, int]

    @classmethod
    def load(cls, directory: Path) -> "TokenizerBundle":
        directory = Path(directory)
        model_path = directory / "tokenizer.model"
        meta = json.loads((directory / "tokenizer.json").read_text(encoding="utf-8"))
        fingerprint = file_sha256(model_path)
        if fingerprint != meta["fingerprint"]:
            raise TokenizerMismatch(f"tokenizer.model digest {fingerprint} != recorded {meta['fingerprint']}")
        processor = spm.SentencePieceProcessor(model_file=str(model_path))
        special = {name: processor.piece_to_id(name) for name in (PAD, UNK, BOS, EOS, *ROLE_SYMBOLS)}
        if special != meta["special_ids"]:
            raise TokenizerMismatch(f"special IDs {special} differ from recorded {meta['special_ids']}")
        return cls(directory, processor, fingerprint, special)

    # -- basic properties -------------------------------------------------
    @property
    def vocab_size(self) -> int:
        return self.processor.get_piece_size()

    @property
    def pad_id(self) -> int:
        return self.special_ids[PAD]

    @property
    def bos_id(self) -> int:
        return self.special_ids[BOS]

    @property
    def eos_id(self) -> int:
        return self.special_ids[EOS]

    @property
    def user_id(self) -> int:
        return self.special_ids[USER]

    @property
    def assistant_id(self) -> int:
        return self.special_ids[ASSISTANT]

    # -- encode / decode ----------------------------------------------------
    def encode(self, text: str) -> List[int]:
        return self.processor.encode(normalize(text), out_type=int)

    def decode(self, ids: Sequence[int]) -> str:
        reserved = set(self.special_ids.values())
        return self.processor.decode([int(i) for i in ids if int(i) not in reserved])

    def is_byte_piece(self, token_id: int) -> bool:
        return self.processor.is_byte(int(token_id))


def _sentences(text: str) -> Iterable[str]:
    for piece in SENTENCE_SPLIT.split(text):
        piece = piece.strip()
        if 2 <= len(piece) <= 1200:
            yield piece


def train_tokenizer(
    texts_by_language: Dict[str, List[str]],
    output_dir: Path,
    vocab_size: int = 32000,
    chars_per_language: int = 12_000_000,
    seed: int = 1729,
    model_type: str = "unigram",
    character_coverage: float = 0.9995,
    num_threads: int = 8,
) -> dict:
    """Fit the shared vocabulary on an equal-character sample of each language.

    Only training-split text may be passed in (FR-T01). The sample is balanced by
    characters so that no script dominates the vocabulary budget (Table 5.5).
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    sample_stats = {}
    with tempfile.TemporaryDirectory() as tmp:
        sample_path = Path(tmp) / "sample.txt"
        with sample_path.open("w", encoding="utf-8") as handle:
            for language in sorted(texts_by_language):
                docs = list(texts_by_language[language])
                rng.shuffle(docs)
                used_chars = used_docs = lines = 0
                for doc in docs:
                    if used_chars >= chars_per_language:
                        break
                    used_docs += 1
                    for sentence in _sentences(normalize(doc)):
                        handle.write(sentence + "\n")
                        used_chars += len(sentence)
                        lines += 1
                sample_stats[language] = {"documents": used_docs, "characters": used_chars, "lines": lines}
        prefix = output_dir / "tokenizer"
        spm.SentencePieceTrainer.train(
            input=str(sample_path),
            model_prefix=str(prefix),
            model_type=model_type,
            vocab_size=vocab_size,
            character_coverage=character_coverage,
            byte_fallback=True,
            split_digits=True,
            normalization_rule_name="identity",
            remove_extra_whitespaces=False,
            add_dummy_prefix=True,
            pad_id=0,
            unk_id=1,
            bos_id=2,
            eos_id=3,
            pad_piece=PAD,
            unk_piece=UNK,
            bos_piece=BOS,
            eos_piece=EOS,
            control_symbols=list(ROLE_SYMBOLS),
            input_sentence_size=3_000_000,
            shuffle_input_sentence=True,
            max_sentence_length=8192,
            num_threads=num_threads,
            seed_sentencepiece_size=1_000_000,
            minloglevel=2,
        )
    model_path = output_dir / "tokenizer.model"
    processor = spm.SentencePieceProcessor(model_file=str(model_path))
    special = {name: processor.piece_to_id(name) for name in (PAD, UNK, BOS, EOS, *ROLE_SYMBOLS)}
    if len(set(special.values())) != len(special):
        raise TokenizerMismatch(f"reserved symbols share IDs: {special}")
    meta = {
        "schema_version": 1,
        "fingerprint": file_sha256(model_path),
        "vocab_size": processor.get_piece_size(),
        "model_type": model_type,
        "character_coverage": character_coverage,
        "byte_fallback": True,
        "normalization": {"form": NORMALIZATION_FORM, "sentencepiece_rule": "identity"},
        "special_ids": special,
        "role_symbols_are_control_symbols": True,
        "training_sample": sample_stats,
        "training_split": "train",
        "seed": seed,
    }
    (output_dir / "tokenizer.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return meta


SCRIPT_RANGES = {"hi": (0x0900, 0x097F), "bn": (0x0980, 0x09FF), "ta": (0x0B80, 0x0BFF)}


def validate_tokenizer(
    tokenizer: TokenizerBundle,
    texts_by_language: Dict[str, List[str]],
    context_length: int = 512,
    max_docs: Optional[int] = 400,
) -> dict:
    """Round trip, fragmentation, unknown/byte-fallback and collision checks (Table 5.7)."""

    report: Dict[str, dict] = {"tokenizer_fingerprint": tokenizer.fingerprint, "languages": {}}
    for language, docs in sorted(texts_by_language.items()):
        docs = docs[:max_docs] if max_docs else docs
        totals = dict(documents=0, characters=0, words=0, tokens=0, unknown=0, byte_fallback=0,
                      round_trip_ok=0, over_context=0)
        failures = []
        for doc in docs:
            expected = normalize(doc)
            ids = tokenizer.encode(expected)
            decoded = tokenizer.decode(ids)
            totals["documents"] += 1
            totals["characters"] += len(expected)
            totals["words"] += len(expected.split())
            totals["tokens"] += len(ids)
            totals["unknown"] += sum(1 for i in ids if i == tokenizer.special_ids[UNK])
            totals["byte_fallback"] += sum(1 for i in ids if tokenizer.is_byte_piece(i))
            totals["over_context"] += int(len(ids) + 2 > context_length)
            if decoded == expected:
                totals["round_trip_ok"] += 1
            elif len(failures) < 3:
                position = next((k for k, (a, b) in enumerate(zip(decoded, expected)) if a != b), min(len(decoded), len(expected)))
                failures.append({"first_difference_at": position, "expected_len": len(expected), "decoded_len": len(decoded)})
        report["languages"][language] = {
            **totals,
            "round_trip_rate": totals["round_trip_ok"] / max(1, totals["documents"]),
            "tokens_per_character": totals["tokens"] / max(1, totals["characters"]),
            "tokens_per_word": totals["tokens"] / max(1, totals["words"]),
            "unknown_rate": totals["unknown"] / max(1, totals["tokens"]),
            "byte_fallback_rate": totals["byte_fallback"] / max(1, totals["tokens"]),
            "documents_longer_than_context": totals["over_context"],
            "round_trip_failure_examples": failures,
        }
    injected = tokenizer.encode(f"{USER} नमस्ते {ASSISTANT} হ্যালো")
    report["special_token_collision"] = {
        "text_containing_role_markers_maps_to_role_ids": any(i in (tokenizer.user_id, tokenizer.assistant_id) for i in injected),
        "passed": not any(i in (tokenizer.user_id, tokenizer.assistant_id) for i in injected),
    }
    report["reserved_ids_unique"] = len(set(tokenizer.special_ids.values())) == len(tokenizer.special_ids)
    return report
