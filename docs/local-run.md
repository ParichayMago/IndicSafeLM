# Local run record (2026-11-27)

This is the complete, scaled-down run of the IndicSafeLM design on one laptop.
Every number below comes from a manifest or report written by the code. Paths
are given so the numbers can be checked.

## Host

Apple M5 (10 CPU cores, 8-core GPU), 16 GB unified memory, macOS 27.
Python 3.12 (`.venv`), PyTorch 2.15 using the Apple GPU (MPS). Other apps were
running, including an Android emulator, so the machine was under memory
pressure: 8–17 GB of swap was in use.

## Stage by stage

| Stage | What ran | Evidence |
|---|---|---|
| Sources | Wikipedia `20231101` snapshot, shard 0 of 2 per language, CC BY-SA 4.0. SHA-256 verified against the dataset's LFS object IDs. | `data/manifests/local_run_sources.jsonl` |
| Corpus | 70,000 articles in → 61,910 kept. Rejected: 1,094 exact duplicates, 3,209 near duplicates, 1,449 script mismatch, 1,144 too short, 1,107 with PII-like strings, and a few others. Split 98/1/1 per language with zero cross-split fingerprint overlap. | `data/processed/corpus-v1/report.json` |
| Tokenizer | SentencePiece unigram, 32,000 pieces, byte fallback, NFC. Fit on 12M characters per language from the training split. 100% round trip in hi/bn/ta; unknown-token rate 0; tokens per character 0.29 / 0.26 / 0.25. `<|user|>` and `<|assistant|>` typed as text cannot produce role IDs. | `artifacts/tokenizer/v1/validation_report.json` |
| Tokens | Training split: 14.6M (hi), 14.1M (bn), 14.1M (ta) tokens. | `data/processed/tokens-v1/meta.json` |
| Base model | 41,567,232 parameters: 8 blocks × 512, 8 heads, FFN 2,048, context 512, RoPE, GELU, tied embeddings, dropout 0.1. Trained for 750 optimizer steps of 16,384 tokens (12.3M tokens, about 0.29 epoch) with AdamW (6e-4, warm-up 150, β = 0.9/0.95, weight decay 0.1, clip 1.0), bf16 compute and fp32 master weights. Validation loss (macro over languages) fell from 10.4 at initialisation to 6.70 at step 150 and **5.23 at step 750**. | `artifacts/runs/*_pretrain_*/` |
| Datasets | 2,553 SFT examples; 4,140 training preference pairs; 1,135 held-out pairs; 1,876 PPO prompts; 340 dev prompts; 809 frozen test prompts; 12 Appendix-A demo cases. | `data/dataset_manifest.json` |
| SFT | 8 epochs (616 steps, batch 32, lr 2e-4, dropout 0), prompt masked from the loss. Validation loss 0.565. Readiness: 45/45 dev prompts terminated, 45/45 in the requested language, 0 role-marker failures. (A first 2-epoch run reached only 2.09 and was replaced.) | `artifacts/checkpoints/sft/`, `artifacts/runs/*_sft_*/` |
| Reward model | SFT backbone + scalar head; 1 epoch over 4,140 pairs (lr 3e-5). Held-out accuracy 95.6% (1085/1135): hi 93.7, bn 95.6, ta 97.6; harmful-compliance 100%, false-refusal 100%, irrelevant 90.2%, wrong-language 93.4%, counter-length 91.2%. Gate passed. | `artifacts/checkpoints/reward/heldout_report.json` |
| PPO | 40 iterations × 45 rollouts, temperature 1, ≤128 tokens; adaptive KL (β≈0.05), clip 0.2, GAE λ 0.95, policy lr 2e-5, value lr 3e-5. Dev composite: SFT 0.698 → iteration 10: 0.792 → **20: 0.812 (selected)** → 30: 0.698 → 40: 0.792. Reward kept rising (−5.6 → +1.2) while Hindi dev safety fell after iteration 20 (Table 7.6 pattern). KL at the selected step: 10.1. | `artifacts/runs/*_ppo_*/`, `artifacts/reports/2_ppo_training.png` |
| Evaluation | 809 frozen test prompts × {Base, SFT, PPO, SFT + keyword filter}, temperature 0.7, top-p 0.9, seed 1735. Results are in the README table and `artifacts/eval/latest/summary.md`. | `artifacts/eval/latest/` |

## Deviations from the report

| Report plan | This run | Why |
|---|---|---|
| ~100M tokens per language (Table 5.1) | ~14M per language available; 12.3M seen in pretraining | One Wikipedia shard per language and a laptop compute budget. The pipeline streams, so a larger corpus needs no code change. |
| Full pretraining run | Stopped at 750 steps (lr still 4.4e-4) | Demonstration deadline. The macOS keep-awake hold also lapsed for about 39 minutes mid-run. Snapshots and resume handled both. |
| BF16/FP16 autocast | A full bf16 compute copy with fp32 master weights | Per-op autocast is slower than fp32 on MPS; bf16 matmuls are about 6× faster. |
| Dropout inside attention | Dropout on embeddings and residual branches only | The fused MPS attention kernel has no dropout. |
| RoFormer interleaved RoPE | "Rotate-half" RoPE pairing | Mathematically equivalent up to a fixed feature permutation, and about 4× faster on MPS. Tested for relative-position invariance. |
| Human-collected SFT data and 3–6k reviewed preference pairs | Template-generated hi/bn/ta banks (23 intents × 3 paraphrases); 4,140 training pairs | No reviewers were available in the time frame. Hindi was written by the author, and Bengali and Tamil were translated to match it. **Native-speaker review is pending.** |
| Curated SFT data | SFT data imitates *uncurated* instruction data: 50% of harmful requests are answered compliantly and 20% of sensitive-benign requests are refused (`configs`/`build.py`) | This gives SFT measurable safety and over-refusal errors, so the PPO stage can be tested against them. It is a controlled-experiment choice and is recorded in `dataset_manifest.json`. |
| Human labels for safety metrics | Automatic proxy labeler; blinded human-review sheet exported | It agrees with the intended label on all 3,705 reference responses, but real outputs need human review. |
| Reward gate: score vs length | Accuracy on 399 held-out *counter-length* pairs, where the rejected reply is at least as long as the chosen one | Preferred replies in this data are legitimately longer, so raw correlation would block PPO for the wrong reason (Table 7.4 asks for matched-length pairs). |
| Long PPO run | 40 PPO iterations × 45 rollouts | Demonstration deadline. |
| SFT keeps language-modelling ability | SFT/PPO perplexity on Wikipedia rose far above the base model (and above unigram) | 8 epochs on short, templated replies overwrote general text modelling. Fix: mix pretraining text into SFT batches or use fewer epochs. |

## Known limitations

- The base model is badly under-trained, so its open-ended text, such as the
  Wikipedia "tell me about" answers, is not reliable. Template-shaped replies
  are what the later stages learn reliably.
- Test prompts are held out by family, names and paraphrase. They still come
  from the same template banks as the training data, so the results measure
  generalisation within this prompt distribution, not real-world safety.
- The Phase 2 near-duplicate shingler splits Indic words at vowel signs
  because `\w` does not cover combining marks. It still works as a fingerprint,
  but the shingles are finer than whole words. The evaluation code uses
  whitespace/punctuation splitting instead.
