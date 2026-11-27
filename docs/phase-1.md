# Phase 1: Corpus inventory and licensing

## Scope and constraint

Phase 1 is complete as a **metadata-only, cloud-ready inventory**. No corpus
payload, dump, or dataset shard was downloaded to this development machine. The
inventory command reads only the versioned JSONL manifest and makes no network
requests:

```bash
python3 scripts/phase1_inventory.py
```

It writes [`data/manifests/inventory.json`](../data/manifests/inventory.json),
which is deterministic for a fixed manifest and records the no-download policy.

## Source decisions

| Source family | Languages | Decision | Reason |
| --- | --- | --- | --- |
| Indic HPLT v2 | hi, bn, ta | Selected for cloud-only pilot | CC0 package, streaming support, published language token estimates; upstream source-rights and takedown review still required |
| IndicCorp v1 | hi, bn, ta | Excluded by default | CC BY-NC-SA non-commercial/share-alike terms and HTTP 403 from unauthenticated object-store probes |
| IndicCorpV2 | hi, bn, ta | Hold | Large CC0 candidate; exact language paths, pinned revision, and provenance review are still required |
| Sangraha verified | hi, bn, ta | Hold | CC BY 4.0 attribution plus source-specific web/OCR/transcription review |
| Wikimedia latest dumps | hi, bn, ta | Hold | Replace moving `latest` URLs with dated dumps and record applicable project terms/checksums |

The per-language records, links, permission basis, attribution requirements,
redistribution restrictions, and current access status are in
[`sources.jsonl`](../data/manifests/sources.jsonl). Records with unclear or
unaccepted rights are not part of the selected cloud-pilot set.

## Published inventory

The selected HPLT records report approximately 2.43B Hindi, 1.85B Bengali,
and 1.94B Tamil tokens. The project target is 100M accepted post-filtering
tokens per language, so the cloud job must stream and stop after the target is
met. HPLT publishes about 34.6M documents and 218GB for the complete
multilingual package; those document/byte values are deliberately marked as
all-language estimates in the inventory rather than incorrectly assigned to
each language.

IndicCorp v1 contributes published article/token estimates in the manifest. The
other candidates publish only all-language totals or no comparable per-language
volume, so the inventory marks those fields as unknown instead of fabricating
precision. The first cloud sample must measure actual bytes, documents, and
accepted tokens by language and source.

## Cloud-only sample gate

Before any record moves from `selected_for_cloud_pilot` to an acquired training
source, run the sampler in the cloud environment against a pinned revision. The
sampler must inspect a bounded sample for:

1. UTF-8 decoding and Unicode normalization failures.
2. Markup, boilerplate, navigation, repeated-page, and extreme-length rates.
3. Language/script correctness and intentional code-mixing labels.
4. Obvious personal or sensitive information and unsafe-content prevalence.
5. Source/domain balance and duplicate fingerprints.

The guarded sampler is checked in at
[`scripts/phase1_cloud_sample.py`](../scripts/phase1_cloud_sample.py). It
requires `INDICSAFE_CLOUD_RUN=1`, a pinned dataset revision, and a cloud
environment with the Hugging Face `datasets` package. For example, run this on
the approved remote host (not here):

```bash
INDICSAFE_CLOUD_RUN=1 python scripts/phase1_cloud_sample.py \
  --repo-id ashtok897/indic-hplt-v2 \
  --revision <reviewed-commit-sha> \
  --language hi \
  --sample-docs 1000 \
  --output artifacts/phase1/cloud-sample-hi.json
```

The same command is repeated for Bengali and Tamil. The output contains no raw
text and is not a substitute for a human safety review.

The cloud run must write only aggregate counts, stable document IDs, source
metadata, and checksums back to the project artifact location. Do not copy raw
sample text to this machine. Keep `representative_sample_status` as pending
until this gate has run.

## Phase 1 exit record

- [x] Candidate sources and links recorded for Hindi, Bengali, and Tamil.
- [x] Release/version, manifest date, permission basis, license, attribution,
  redistribution restrictions, and checksum placeholders recorded per record.
- [x] Default selection and hold/exclusion decisions documented.
- [x] Published document/token/byte estimates summarized without downloading
  corpus payloads.
- [x] Target set to approximately 100M post-tokenization tokens per language.
- [ ] Representative corpus samples inspected — deferred to the cloud-only gate
  by explicit user constraint.
- [ ] Final source checksums recorded — deferred until cloud acquisition.

This means the local Phase 1 preparation is complete, while the two payload-
dependent checks remain explicit gates for the remote environment.
