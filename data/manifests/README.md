# Data manifests

`sources.jsonl` is the curated download manifest for the Hindi (`hi`), Bengali
(`bn`), and Tamil (`ta`) data sources. It records one source-language pair per
line, including the landing/download URL, release/version, manifest access date,
license and permission basis, attribution and redistribution restrictions,
checksum placeholder, access status, estimated volume, and the role that source
is allowed to play in the project.

`inventory.json` is the generated Phase 1 aggregate. It contains expected
document/token/byte estimates, selected source IDs, and the cloud-only gate
status; it does not contain corpus text.

The manifest deliberately contains links and metadata only. `acquisition_date`
and `checksum_sha256` remain `null` until a cloud acquisition has actually
completed. Do not place raw corpus text in this directory. Before acquiring a
source, pin the release (or record the dump date), record a checksum, and
complete the license/source-rights review. Large sources should be streamed and
sampled rather than downloaded in full.

Initial selection guidance:

- HPLT v2 is selected for the cloud-only pilot, targeting a filtered,
  approximately 100M-token sample per language. Apply deduplication, quality
  filtering, PII checks, and the upstream rights/takedown review.
- IndicCorp v1 is excluded from the default selection: its CC BY-NC-SA license
  is non-commercial/share-alike, and its object-store URLs returned HTTP 403 to
  an unauthenticated probe.
- IndicCorpV2 and Sangraha are high-volume alternatives on hold. Use
  streaming/subsets and finish the corresponding revision, provenance, and
  license/attribution review first.
- Wikimedia dumps are smaller, high-quality enrichment sources on hold. Replace
  each `latest` URL with a dated dump before a reproducible run.
