# Environment Record

## Target training environment

These are the requirements for the full project run:

- Python 3.11+ (the project metadata allows 3.11–3.13).
- PyTorch 2.5.1 with a compatible CUDA build and driver.
- One NVIDIA GPU with approximately 24 GB VRAM minimum; 48 GB is preferred for PPO headroom.
- 32 GB RAM recommended (16 GB is a reduced-workflow minimum).
- 100–200 GB free storage for corpora, shards, checkpoints, and logs.
- Git and a reproducible environment installed from `requirements.lock`.
- Mixed precision, gradient accumulation, checkpoint resume, and deterministic evaluation enabled where supported.

## Phase 0 development host

Recorded on 2026-09-30T03:46:11Z:

| Item | Value |
|---|---|
| OS | macOS 27.0 (Darwin 27.0), ARM64 |
| CPU | 10 logical cores |
| RAM | 16 GiB |
| Python | 3.9.6 (`/Library/Developer/CommandLineTools/usr/bin/python3`) |
| Git | 2.54.0 |
| CUDA/NVIDIA GPU | Not available |
| Free space at record time | Approximately 279 GiB |
| PyTorch/NumPy/etc. | Not installed in the system Python |
| PyYAML | 6.0.3 (available; smoke path does not require it) |

This host is suitable for repository, configuration, documentation, and small standard-library smoke checks. It is not the target host for full Transformer pretraining or PPO. Install Python 3.11+ and the locked dependencies on the selected cloud GPU before Phase 4.

## Run environment requirements

Every run manifest must capture:

- Git revision and working-tree state.
- Python, OS, PyTorch, CUDA, and driver versions.
- GPU model, count, and VRAM.
- Resolved configuration and random seed.
- Input artifact IDs/checksums.
- Start/end time, token count, batch/accumulation settings, and peak memory.

