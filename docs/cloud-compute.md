# Cloud Compute Decision

This project will be developed locally but trained on rented or sponsored Linux GPU compute. Prices and availability are dynamic; the figures below are a decision snapshot recorded on 2026-09-30, not a quotation.

## Recommendation

Use **RunPod Pods** for the real training runs, with an on-demand 24 GB RTX 4090 for the first pilot or a 48 GB RTX A6000/A40 when PPO memory headroom is more important than peak speed. Use a persistent volume, save checkpoints to an external object store as well, and terminate the Pod when it is not in use.

Why this is the best default for this project:

- A 42M-parameter model should fit within 24 GB with mixed precision and gradient accumulation; 48 GB gives safer room for PPO rollouts, reward/reference models, and larger batches.
- RunPod provides ordinary Linux/SSH-style Pods suitable for long-running training, rather than forcing the project into a notebook-only workflow.
- The current published rates are roughly $0.50/hr for RTX 3090, $0.74/hr for RTX 4090, $0.53/hr for RTX A6000 (48 GB), $0.49/hr for A40 (48 GB), and $1.59/hr for A100 80 GB. Confirm the live rate and region before launch.
- RunPod bills compute and storage separately, so using a persistent volume and explicitly stopping/terminating resources is important.

Start with a one-hour pilot that measures tokens/second, peak VRAM, data-loader throughput, and checkpoint upload time. Use that measurement to set the full-run budget rather than trusting a generic FLOPs estimate.

## Options by budget and convenience

| Option | Use it for | Strengths | Main risks/limitations | Decision |
|---|---|---|---|---|
| Kaggle free | Smoke tests and small pilots | Free; official notebook environment; P100 or T4×2; 30 GPU-hours/week is documented, with 12-hour notebook sessions and 20 GB auto-saved working disk | Quota/session limits; small disk; checkpoint and data handling must be designed around notebook restarts | Use first if zero budget |
| Colab free | Notebook development and smoke tests | Very easy setup and Google Drive/GitHub integration | GPU type/availability and limits are dynamic; free notebooks are at most 12 hours and resources are not guaranteed | Do not use as the only full-training backend |
| Colab Pro/Pro+ or Google AI plan | Convenient notebook-based pilots or short runs | Paid plans provide more compute availability; Pro+ supports up to 24-hour continuous execution when units are available | Exact GPU availability and usage limits vary; units can be exhausted; still less predictable than a dedicated Pod | Buy only if notebook convenience is worth the premium or you already have the plan |
| RunPod Pod | Full pretraining, SFT, reward model, PPO | SSH/Linux workflow, per-second billing, broad GPU choice, persistent storage options, straightforward checkpoint/resume | Must manage volumes and stop/terminate resources; capacity and prices vary | **Recommended default** |
| Vast.ai on-demand | Lowest-cost experiments when comfortable with infrastructure | Marketplace pricing can be materially cheaper and offers many GPUs | Host reliability, network/storage performance, max duration, and data-trust vary; interruptible jobs can be killed | Good cost fallback for checkpointed, non-sensitive runs |
| Modal | Short jobs or sponsored research use | Per-second billing, no idle-resource charge, $30/month starter compute, and an academic-credit program | Serverless/container workflow needs more setup; A100 is listed at about $2.50/hr, so it is not the cheapest self-funded choice | Apply for academic credits before considering paid use |
| Paperspace | Managed alternative | Simple managed machines and published GPU catalog | Listed A6000 pricing is about $1.89/hr; low A100 price shown is tied to a long commitment | Not the value default for this project |

## Why not rely on Colab alone?

Google's own FAQ says Colab resources are not guaranteed, GPU types vary, free runtimes can last at most 12 hours depending on availability and usage, and paid runtimes can still terminate when compute units are exhausted. Colab Pro+ can run continuously for up to 24 hours with sufficient units, but this is still a quota-based notebook service rather than a predictable training VM.

Colab is excellent for making the project easy to demonstrate and for short experiments. It is a poor single backend for a from-scratch run that needs many restarts, a large processed corpus, PPO checkpoints, and a stable Linux environment.

## Why not use Vast.ai as the default?

Vast.ai is attractive on price and offers on-demand and interruptible rentals. Its own documentation warns that interruptible instances stop when outbid and recommends frequent external checkpoint saves. Host bandwidth, disk speed, and trust also vary. Use it when cost is the binding constraint and the job is fully restartable; do not place unencrypted sensitive safety data on an untrusted host.

## Cost envelope

The exact training time is unknown until the pilot. Budget for approximately 30–100 GPU-hours across pretraining, SFT, reward modeling, PPO, evaluation, failed runs, and profiling:

- At $0.53–$0.74/hr (A6000/A40/4090 class), this is roughly $16–$74 of GPU time before storage and safety buffer.
- At $1.59/hr (A100 80 GB), this is roughly $48–$159 before storage and buffer.
- Add at least 30% for debugging, interrupted jobs, data transfer, and reruns.

These are planning ranges, not performance guarantees. The pilot must replace them with measured throughput.

## Required cloud operating procedure

1. Use a Linux CUDA image with the exact Python and dependency lock documented in the run manifest.
2. Keep code in Git; keep raw data/checkpoints on a persistent volume and an independent backup.
3. Run `scripts/preflight.py --strict` before any expensive job.
4. Run the one-hour pilot and record tokens/second, VRAM, and storage throughput.
5. Enable checkpointing at a cadence that loses no more than 15–30 minutes of work.
6. Use a spend cap, low-balance alert, and a written shutdown checklist.
7. Stop or terminate the Pod immediately after a run; separately remove unused volumes.
8. Back up final checkpoints, configs, metrics, and manifests before deleting the instance.

## Sources consulted

- Google Colab FAQ: https://research.google.com/colaboratory/intl/en-GB/faq.html
- Kaggle notebook technical specifications: https://www.kaggle.com/docs/notebooks
- RunPod pricing: https://www.runpod.io/pricing
- RunPod billing and persistent-storage behavior: https://github.com/runpod/docs/blob/main/accounts-billing/billing.mdx
- Vast.ai FAQ and rental types: https://console.vast.ai/faq/
- Modal pricing and academic credits: https://frontend.modal.com/pricing
- Paperspace pricing: https://www.paperspace.com/pricing

