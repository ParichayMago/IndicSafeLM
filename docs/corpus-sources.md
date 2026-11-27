# Corpus source plan

This document explains the links in
[`data/manifests/sources.jsonl`](../data/manifests/sources.jsonl). The manifest
is the machine-readable source of truth; this page records why each source is
there and how it should be acquired.

## Recommended first acquisition

Use **Indic HPLT v2** as the first primary candidate for all three languages.
The Hugging Face card reports approximately 2.43B Hindi, 1.85B Bengali, and
1.94B Tamil tokens, with about 218GB for the complete multilingual dataset.
That is much larger than this project's 100M-token-per-language target, so use
the streaming API and stop after the target sample has passed filtering. The
card describes the package as CC0, while the HPLT catalog notes that HPLT does
not claim ownership of the underlying web text; keep the source-rights and
notice/takedown review in the data pipeline.

The first run should produce no more than approximately 100M accepted tokens
per language after language identification, document quality checks, PII
screening, normalization, and exact/near-duplicate removal. Keep the source
row metadata and a manifest of rejected records so that the sample can be
reproduced.

Example acquisition pattern (do not run until the revision, filters, and cloud
destination are approved):

```python
from datasets import load_dataset

ds = load_dataset(
    "ashtok897/indic-hplt-v2",
    split="train",
    streaming=True,
    revision="<pin-a-reviewed-revision>",
)
for row in ds:
    if row["lang"] in {"hi", "bn", "ta"}:
        text = row["text"]
        # project language, quality, PII, normalization, and dedup filters
```

## Alternatives and their intended roles

**IndicCorp v1** has direct text links for each language and is a convenient
research baseline. It is licensed CC BY-NC-SA 4.0, so it should remain
non-commercial and share-alike compliant; it is not the default source for a
future redistributable model or dataset. A non-authenticated range probe of
the three object-store URLs returned HTTP 403 on 2026-09-30, so treat these as
references until provider access is confirmed; do not put access keys in this
repository.

**IndicCorpV2** is a much larger AI4Bharat release (the card estimates about
275GB). Its card states CC0, but the language directory must be resolved from
the repository listing and pinned before the first run. Use only a streamed,
bounded subset.

**Sangraha** is another very large AI4Bharat collection (about 251B tokens
across languages). Start with `verified/hin`, `verified/ben`, and `verified/tam`
only if the project is prepared to provide CC BY 4.0 attribution and to review
the source terms for its web, OCR, and transcription components.

**Wikimedia dumps** are useful quality-enrichment data, not the main volume
source. The `latest` links are intentionally dynamic; replace them with a
dated dump directory and record the checksum before a reproducible run. Strip
MediaWiki markup and preserve the applicable attribution/share-alike notices.

## Acquisition checklist

1. Select a row in `sources.jsonl` and pin its dataset revision, dated dump, or
   object version. Never rely on a moving `latest` URL for a final experiment.
2. Verify the license and source-rights notice, then record the decision in the
   acquisition manifest together with the access date and SHA-256 checksum.
3. Stream or download only the approved sample into cloud storage. Keep raw
   data outside git and do not expose credentials in manifests or logs.
4. Run language identification, Unicode normalization, quality/boilerplate
   checks, PII screening, and exact/near-deduplication before the train/validation/test
   split.
5. Save the accepted/rejected counts and preprocessing configuration next to
   the resulting dataset artifact.

The links were prepared on 2026-09-30. No corpus was downloaded by this step.
