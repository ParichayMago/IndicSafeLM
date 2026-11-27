"""Train and validate the shared tokenizer, then write tokenized shards.

1. Fit SentencePiece (32k, unigram, byte fallback) on an equal-character
   sample of the *training* split of each language (FR-T01, FR-T02).
2. Validate round trips, fragmentation, unknown/byte-fallback rates and
   special-token collisions on the *validation* split (FR-T05, Table 5.7).
3. Encode every split into ``uint16`` token streams per language, each
   document wrapped as ``<s> ... </s>``, for pretraining and perplexity.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from indicsafelm.tokenizer import TokenizerBundle, train_tokenizer, validate_tokenizer  # noqa: E402
from indicsafelm.utils import LANGUAGES, iter_jsonl, sha256_file, write_json  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=ROOT / "data/processed/corpus-v1")
    parser.add_argument("--tokenizer-dir", type=Path, default=ROOT / "artifacts/tokenizer/v1")
    parser.add_argument("--tokens-dir", type=Path, default=ROOT / "data/processed/tokens-v1")
    parser.add_argument("--vocab-size", type=int, default=32000)
    parser.add_argument("--chars-per-language", type=int, default=12_000_000)
    args = parser.parse_args()

    by_split = {split: defaultdict(list) for split in ("train", "validation", "test")}
    for row in iter_jsonl(args.corpus / "processed.jsonl"):
        by_split[row["split"]][row["language"]].append(row["text"])
    for split, langs in by_split.items():
        print(split, {lang: len(docs) for lang, docs in langs.items()})

    if not (args.tokenizer_dir / "tokenizer.model").exists():
        meta = train_tokenizer(by_split["train"], args.tokenizer_dir, args.vocab_size, args.chars_per_language)
        print("trained tokenizer", meta["fingerprint"][:16], "vocab", meta["vocab_size"])
    tokenizer = TokenizerBundle.load(args.tokenizer_dir)

    report = validate_tokenizer(tokenizer, by_split["validation"], max_docs=None)
    report["corpus_release"] = str(args.corpus.relative_to(ROOT))
    report["corpus_report_sha256"] = sha256_file(args.corpus / "report.json")
    write_json(args.tokenizer_dir / "validation_report.json", report)
    for lang, stats in report["languages"].items():
        print(
            f"[{lang}] round-trip {stats['round_trip_rate']:.3f}  tokens/char {stats['tokens_per_character']:.3f}  "
            f"tokens/word {stats['tokens_per_word']:.2f}  unk {stats['unknown_rate']:.5f}  bytes {stats['byte_fallback_rate']:.5f}"
        )
    print("special-token collision check passed:", report["special_token_collision"]["passed"])

    args.tokens_dir.mkdir(parents=True, exist_ok=True)
    meta = {"tokenizer_fingerprint": tokenizer.fingerprint, "dtype": "uint16", "format": "<s> doc </s> concatenated", "files": {}}
    for split, langs in by_split.items():
        for lang in LANGUAGES:
            ids = []
            for doc in langs[lang]:
                ids.append(tokenizer.bos_id)
                ids.extend(tokenizer.encode(doc))
                ids.append(tokenizer.eos_id)
            array = np.asarray(ids, dtype=np.uint16)
            path = args.tokens_dir / f"{split}_{lang}.bin"
            array.tofile(path)
            meta["files"][f"{split}_{lang}"] = {"path": path.name, "tokens": int(array.size), "documents": len(langs[lang])}
            print(f"{split}_{lang}: {array.size:,} tokens")
    write_json(args.tokens_dir / "meta.json", meta)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
