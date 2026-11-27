# Phase 0: Reproducibility and Experiment Contract

## Decisions

- **Languages:** Hindi (`hi`), Bengali (`bn`), and Tamil (`ta`).
- **Model checkpoints:** `base`, `sft`, and `ppo`; all remain independently loadable.
- **Tracking backend:** TensorBoard is the default because it is local, scriptable, and does not require sending sensitive samples to an external service. A run may additionally mirror aggregate metrics to Weights & Biases only after the data-handling review.
- **Configuration format:** JSON for the initial contract, so configuration and smoke validation work on the current Python 3.9 development host without third-party imports. Training code may consume the same files through a typed loader.
- **Seed policy:** Stage-specific seeds are versioned in `configs/project.json`; evaluation uses a fixed seed and deterministic settings wherever supported.
- **Run naming:** `YYYYMMDD-HHMMSS_stage_language_seed_gitsha`.
- **Data policy:** Raw corpora, generated responses, checkpoints, and unreviewed safety examples are local/ignored. Manifests, checksums, aggregate statistics, and sanitized fixtures are tracked.
- **Evaluation policy:** Evaluation prompt families are frozen before training and never used for model selection during training.
- **Cloud strategy:** Local macOS is for development and smoke tests. Use a persistent Linux GPU pod for full training; see [`cloud-compute.md`](cloud-compute.md).

## Phase 0 completion checklist

- [x] Repository layout created.
- [x] Python 3.11+ target environment declared in `pyproject.toml`.
- [x] Top-level dependency versions recorded in `requirements.lock`.
- [x] Project, experiment-matrix, and evaluation configurations created.
- [x] Seed policy and run naming documented.
- [x] Environment requirements and current-host limitations recorded.
- [x] Cloud-compute strategy and fallback options recorded.
- [x] Evaluation categories and metrics defined.
- [x] Annotation and human-rating guidance written.
- [x] Preflight and standard-library smoke-test scripts added.
- [ ] Training GPU environment provisioned (required before Phase 4 full pretraining).
- [ ] Licensed corpus manifest populated (begins in Phase 1).

## Commands

```bash
python3 scripts/preflight.py
python3 scripts/smoke_test.py
python3 scripts/preflight.py --strict
```

The non-strict preflight is expected to warn on a development machine without Python 3.11, PyTorch, or CUDA. The strict form is the gate for full training hosts.

