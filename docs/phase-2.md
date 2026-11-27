# Phase 2: Cleaning, deduplication, and splitting

Phase 2 is implemented as a deterministic JSONL pipeline in
[`src/indicsafelm/data/pipeline.py`](../src/indicsafelm/data/pipeline.py). It
does not fetch URLs or know about cloud credentials. The input must be an
explicit JSONL path supplied by the caller.

## Pipeline order

1. Decode UTF-8 and validate the record schema.
2. Normalize Unicode to NFC, preserve Indic joiners, normalize line endings,
   and remove unsafe control characters.
3. Strip markup, unescape entities, remove configured boilerplate, and collapse
   whitespace.
4. Reject unusably short/long, malformed, repeated, URL-heavy, or wrong-script
   records. Explicitly labelled code-mixed records use a lower target-script
   threshold.
5. Reject obvious email, phone, IPv4, and credit-card-like values.
6. Remove exact duplicates by SHA-256 of normalized text.
7. Remove near duplicates using deterministic token shingles, MinHash bands, and
   exact Jaccard verification.
8. Freeze document-level train/validation/test assignments using a seeded hash,
   stratified by language.
9. Emit an ID-only balanced training view with language sampling weights; the
   frozen source records are not changed.

## Outputs

Each run writes an atomic output directory containing:

- `report.json`: before/after document, character, byte, and estimated-token
  counts for every stage; rejection counts by reason/language; duplicate and
  leakage statistics; split and domain distributions.
- `splits/{train,validation,test}.jsonl`: stable document IDs and metadata only,
  never raw text.
- `balanced_train_view.json`: ID-only sampling weights.
- `processed.jsonl`: normalized text for the explicit input run; this is ignored
  by git and should be written on the cloud host for real corpora.
- `rejections.jsonl` and `review_samples.jsonl`: stable rejection metadata and
  redacted review previews.
- `completion.json`: input/config hashes and the pipeline fingerprint, enabling
  safe restart/reuse of an identical run.

## Local smoke run

The following uses only the synthetic, sanitized fixture in
`data/samples/phase2_input.jsonl`; it does not download a corpus:

```bash
python3 scripts/run_data_pipeline.py \
  --input data/samples/phase2_input.jsonl \
  --config configs/data_pipeline.json \
  --output artifacts/phase2/smoke-final
```

For a real run, perform the same command in the approved cloud environment on
a pinned, licensed input stream or staged shard. Keep the output directory in
cloud storage and copy back only aggregate reports and ID-only manifests.

## Determinism and leakage gates

The run fingerprint is derived from the exact input bytes, pipeline config, and
pipeline version. Re-running an identical completed run reuses its report;
reusing an output directory for a different input/config fails rather than
overwriting it. Exact text fingerprints are checked across all split pairs and
must have zero intersections before training.

The checked-in smoke fixture exercises all three languages, HTML/boilerplate
cleaning, PII rejection, script rejection, code-mixed acceptance, exact and
near-duplicate rejection, and non-empty validation/test splits for a small
dataset. Its text is synthetic and is not a corpus sample.
