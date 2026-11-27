# IndicSafeLM Implementation Plan

## 1. Project Goal

Build and evaluate a compact multilingual decoder-only language model for Hindi, Bengali, and Tamil, then determine whether reward-model-guided PPO alignment reduces unsafe generations without materially damaging language quality, helpfulness, or correct handling of benign prompts containing sensitive terms.

The project is an academic research prototype. Its main output is a reproducible experiment and a transparent analysis of the safety–capability trade-off, not a production-ready or commercial-scale language model.

## 2. Research Questions

1. Does reward-model-guided PPO reduce toxic, abusive, and profane generations relative to the base and supervised-fine-tuned models?
2. Are the safety improvements consistent across Hindi, Bengali, and Tamil?
3. What effect does alignment have on perplexity, fluency, relevance, and helpfulness?
4. Does stronger alignment increase false refusals on educational, quoted, translated, or descriptive uses of sensitive language?

## 3. Scope

### Included

- Hindi, Bengali, and Tamil in their native scripts.
- Licensed or research-permitted public text corpora.
- Corpus collection, normalization, filtering, deduplication, and document-level splitting.
- A shared multilingual subword tokenizer of approximately 32,000 tokens.
- A decoder-only Transformer trained from random initialization.
- Supervised instruction tuning (SFT).
- A multilingual safety preference dataset of approximately 3,000–6,000 verified pairs.
- Pairwise reward-model training.
- PPO alignment with a KL penalty against a frozen SFT reference model.
- Capability, safety, human-preference, and false-refusal evaluation.
- A keyword-filter baseline and a local side-by-side comparison interface.

### Excluded from the initial project

- Commercial-scale model quality or reliability.
- Languages beyond Hindi, Bengali, and Tamil.
- Broad safety coverage such as hallucination, medical advice, privacy leakage, or self-harm.
- Large-scale public deployment or an online inference service.
- Unlicensed redistribution of source corpora.
- Transliterated and code-mixed language as primary training targets; these may be retained only in explicitly labelled experiments.

## 4. Planned System

```text
Licensed corpora
    -> normalize/filter/deduplicate
    -> frozen document-level train/validation/test split
    -> balanced tokenizer sample
    -> shared tokenizer
    -> tokenized pretraining data
    -> base decoder-only LM
    -> multilingual SFT
    -> preference data + reward model
    -> PPO with frozen SFT reference and KL control
    -> frozen multilingual evaluation
    -> comparison demo and report
```

The three primary checkpoints must remain separately loadable and evaluable:

1. `base`: pretrained only with causal next-token prediction.
2. `sft`: base model tuned on multilingual prompt-response examples.
3. `ppo`: SFT model aligned using the learned reward model and PPO.

## 5. Target Model Configuration

| Component | Initial target |
|---|---|
| Architecture | Decoder-only Transformer |
| Parameters | Approximately 42 million |
| Transformer blocks | 8 |
| Hidden dimension | 512 |
| Attention heads | 8 |
| Feed-forward dimension | 2,048 |
| Context length | 512 tokens |
| Vocabulary | Approximately 32,000 tokens |
| Position representation | Rotary Position Embeddings (RoPE) |
| Activation | GELU |
| Pretraining dropout | 0.10 |
| Embedding/output weights | Tied |
| Optimizer | AdamW |
| Precision | BF16 when supported; otherwise FP16 with loss scaling |

The configuration may be reduced after a measured GPU memory test, but any change must be recorded with the reason, parameter count, and effect on comparability. Before full pretraining, verify the parameter count, causal mask, RoPE behavior, weight tying, padding/loss masking, checkpoint resume, and deterministic validation.

## 6. Repository Layout

Use the following layout unless implementation constraints justify a documented change:

```text
configs/                 # versioned YAML/TOML experiment configurations
data/
  manifests/             # source URLs, versions, licenses, checksums
  samples/               # small, safe fixtures committed for tests
  raw/                   # ignored; immutable downloaded data
  interim/               # ignored; normalized/deduplicated documents
  processed/             # ignored; frozen splits and tokenized shards
docs/                    # annotation rubric, data card, model card, reports
src/indicsafelm/
  data/                  # ingestion, normalization, filtering, deduplication
  tokenizer/             # training and tokenizer analysis
  model/                 # Transformer, RoPE, generation, configuration
  training/              # pretraining and SFT loops
  reward/                # preference dataset and reward-model training
  alignment/             # PPO, KL controller, rollout generation
  evaluation/            # automatic metrics and comparison reports
  demo/                  # Gradio or Streamlit interface
scripts/                 # thin entry points for reproducible runs
tests/                   # unit, integration, and smoke tests
artifacts/               # ignored; checkpoints, logs, plots, generated reports
```

Large datasets, checkpoints, generated responses, credentials, and raw harmful-text data must not be committed to Git. Commit manifests, configurations, code, aggregate statistics, and only sanitized test fixtures.

## 7. Execution Plan

### Phase 0 — Reproducibility and experiment contract (Week 1)

Tasks:

- Create the repository structure, dependency lock, configuration system, logging convention, and random-seed policy.
- Choose TensorBoard or Weights & Biases and define a run-naming convention.
- Record the available CPU, RAM, GPU, VRAM, storage, CUDA, Python, and PyTorch versions.
- Write the experiment matrix before training: base, SFT, PPO, and keyword-filter baseline.
- Define the held-out evaluation sets and ensure they are not used for model selection.
- Draft annotation guidance covering harmful prompts, safe redirection, legitimate sensitive-language discussion, fluency, and relevance.

Exit criteria:

- A clean environment can run tests and a tiny end-to-end smoke experiment.
- Every run records the Git revision, config, seed, dataset/tokenizer versions, and output path.
- Evaluation and preference splits are defined before model training begins.

### Phase 1 — Corpus inventory and licensing (Weeks 1–2)

Tasks:

- Identify candidate Hindi, Bengali, and Tamil sources such as compatible Indic corpora and Wikipedia dumps.
- Create one manifest entry per source containing URL, release/version, acquisition date, language, license, attribution requirements, checksum, and redistribution restrictions.
- Inspect representative samples for encoding problems, boilerplate, personal information, unsafe content, and domain imbalance.
- Set a target of roughly 100 million post-tokenization training tokens per language, subject to licensing and compute constraints.
- Estimate storage and preprocessing cost before downloading the full corpus.

Exit criteria:

- Every included source has a documented permission basis.
- The corpus inventory shows expected document/byte/token volume by language and source.
- Sources with unclear licenses or unacceptable data risk are excluded.

### Phase 2 — Cleaning, deduplication, and splitting (Week 3)

Implement a deterministic, restartable pipeline in this order:

1. Decode and validate Unicode.
2. Apply documented Unicode normalization.
3. Remove markup, boilerplate, malformed text, extreme repetition, and unusably short records.
4. Perform language and script checks, retaining intentional code-mixed data only under an explicit label.
5. Filter obvious personal or sensitive information where feasible.
6. Deduplicate exact and near-duplicate documents before splitting.
7. Freeze document-level train/validation/test splits at approximately 98%/1%/1%.
8. Produce balanced training views without altering the frozen source records.

Required outputs:

- Before/after document, character, byte, and estimated-token counts for every processing stage.
- Rejection counts by reason and language.
- Duplicate/leakage statistics.
- Versioned split manifests containing stable document identifiers, not raw text.
- Manual samples from kept and rejected records for quality review.

Exit criteria:

- Re-running the pipeline with the same inputs and config produces identical split manifests.
- No exact duplicates cross train, validation, or test boundaries.
- Language proportions and major source domains are documented.
- The test split is frozen and excluded from tuning decisions.

### Phase 3 — Tokenizer training and validation (Week 4)

Tasks:

- Train SentencePiece or an equivalent BPE/Unigram tokenizer on an equal or explicitly weighted sample from all three languages.
- Reserve and document BOS, EOS, padding, unknown, and conversation-role tokens.
- Verify loss masking for padding and define one canonical chat serialization format before SFT data is created.
- Measure tokens per character and tokens per word by language and source domain.
- Inspect unknown-token behavior, round-trip encode/decode, Unicode normalization interactions, long words, punctuation, digits, and mixed-script samples.
- Compare at least two candidate tokenizer settings if one language is disproportionately fragmented.

Exit criteria:

- The tokenizer round-trips representative text from all three scripts correctly.
- Unknown-token use is effectively absent for ordinary valid text.
- Fragmentation statistics show no unexplained major disadvantage for one language.
- Tokenizer files, config, training manifest, and evaluation report are versioned together.

### Phase 4 — Transformer implementation and base pretraining (Weeks 5–8)

Implementation tasks:

- Build embeddings, RMSNorm or the chosen normalization layer, masked multi-head attention, RoPE, feed-forward layers, residual paths, final normalization, and tied output projection.
- Implement packed/fixed-length causal batches with document-boundary handling documented.
- Add AdamW, warmup plus decay, gradient clipping, mixed precision, gradient accumulation, validation, checkpointing, and resume support.
- Add text generation with temperature, top-k/top-p, maximum-new-token, EOS, and seed controls.

Verification sequence:

1. Unit-test tensor shapes, causal masking, RoPE, weight tying, and loss masking.
2. Overfit one tiny batch to verify the optimization path.
3. Run a small multi-language training job and resume it from a checkpoint.
4. Profile peak VRAM and throughput, then freeze the full-training configuration.
5. Run full pretraining while logging overall and per-language validation loss/perplexity.

Checkpoint policy:

- Keep periodic recoverable checkpoints plus the best validation checkpoint.
- Save model, optimizer, scheduler, gradient-scaler, step, RNG state, and exact configuration.
- Generate a fixed qualitative sample set periodically, but do not use the final test set to choose a checkpoint.

Exit criteria:

- Training is numerically stable and can resume with no unexplained state loss.
- Validation loss improves over a unigram/random baseline in each language.
- Per-language curves and representative samples are reviewed for severe imbalance or collapse.
- The selected base checkpoint has a model card recording data, configuration, compute, and known limitations.

### Phase 5 — Multilingual supervised instruction tuning (Weeks 8–9)

Tasks:

- Prepare a versioned prompt-response dataset with ordinary helpful tasks and safety-relevant examples in all three languages.
- Define language, task, source, safety category, and provenance fields.
- Deduplicate against preference and final evaluation prompts to prevent leakage.
- Serialize all examples using the tokenizer's fixed conversation template.
- Train from the selected base checkpoint, masking prompt tokens if response-only loss is chosen.
- Compare the base and SFT checkpoints on prompt adherence, fluency, per-language quality, and held-out perplexity.

Exit criteria:

- The SFT model reliably follows the expected response format in all three languages.
- Instruction following improves without catastrophic language-modeling regression.
- Training/evaluation prompt overlap checks pass.
- The chosen SFT checkpoint is frozen as the PPO reference model.

### Phase 6 — Safety preference dataset (Week 10)

Each record should contain at minimum:

- Stable pair and prompt identifiers.
- Language and script.
- Safety category and benign/harmful intent label.
- Prompt, preferred response, and dispreferred response.
- Response provenance and annotator/reviewer status.
- Split and quality-control flags.

Required categories:

- Profanity.
- Direct insults.
- Harassment.
- Hateful or toxic language.
- Requests to escalate abusive content.
- Benign educational, quoted, translated, or descriptive uses of sensitive terms.

Tasks:

- Create approximately 3,000–6,000 verified pairs with tracked language/category balance.
- Write a rubric that defines preference based on safety, helpfulness, relevance, and naturalness—not refusal alone.
- Use independent review for ambiguous or high-risk pairs and record disagreements.
- Split by prompt or prompt family so paraphrases cannot cross train and evaluation sets.
- Remove overlaps with SFT and final test prompts.
- Protect raw offensive examples and publish only what licensing and ethical review allow.

Exit criteria:

- All retained pairs pass the written rubric and schema validation.
- Language/category distributions and inter-reviewer agreement or disagreement counts are reported.
- Held-out preference pairs are frozen before reward-model training.
- Benign sensitive-context examples are sufficiently represented to measure over-refusal.

### Phase 7 — Reward-model training (Week 11)

Tasks:

- Initialize from SFT weights where practical and attach a scalar reward head.
- Score the prompt-response sequence at a documented terminal-token position.
- Train with pairwise logistic ranking loss so the preferred response scores above the dispreferred response.
- Track ranking accuracy, score margin, calibration/score distributions, and overfitting by language and category.
- Inspect high-reward and low-reward examples manually.
- Test for shortcuts such as rewarding response length, refusal phrases, a language, or formatting artifacts.

Mandatory PPO quality gate:

- Held-out ranking accuracy must be meaningfully above the 50% random baseline overall.
- No target language or required safety category may show an unexplained collapse.
- Human spot checks must not reveal a systematic refusal, length, formatting, or language shortcut.
- If the gate fails, revise preference data or reward-model training; do not start PPO merely because training loss decreased.

Exit criteria:

- A frozen reward-model checkpoint and evaluation report pass the quality gate.
- Known reward-model failure modes are documented for later PPO monitoring.

### Phase 8 — PPO safety alignment (Week 12)

Tasks:

- Initialize the policy and frozen reference from the selected SFT checkpoint.
- Generate rollouts from a balanced multilingual prompt sampler.
- Compute learned reward, token-level KL against the reference, and any explicitly documented shaping/length terms.
- Implement PPO clipping, value estimation, advantage normalization, gradient clipping, and early stopping.
- Begin with a short low-learning-rate run and conservative KL target before scaling the experiment.
- Save frequent checkpoints and evaluate them on a small validation-only safety/capability suite.

Log at minimum:

- Raw reward, non-score reward/penalties, and total reward.
- KL divergence, entropy, response length, clip fraction, policy loss, value loss, and explained variance.
- Refusal rate, false-refusal rate, unsafe-generation rate, and language distribution.
- Fixed-prompt samples from harmful and benign sensitive-context categories.

Stop or roll back when reward rises while any of the following materially deteriorates for repeated evaluations:

- Fluency or relevance.
- False-refusal rate.
- Output diversity or response length.
- KL stability.
- Per-language behavior.
- Human assessment of actual safety.

Exit criteria:

- Select the checkpoint on a documented multi-metric validation rule, not reward alone.
- The final PPO checkpoint and reference checkpoint can be evaluated with identical generation settings.
- The training report documents hyperparameters, compute, instability, stopped runs, and checkpoint-selection rationale.

### Phase 9 — Final evaluation and ablation (Week 13)

Freeze all models and evaluation inputs before generating final results. Use the same prompt serialization and generation settings wherever the comparison permits.

Required comparisons:

- Base vs. SFT.
- SFT vs. PPO.
- SFT plus keyword filter vs. PPO, emphasizing context-sensitive cases.
- Hindi vs. Bengali vs. Tamil.
- Profanity, insult/harassment, hate/toxicity, escalation, and benign sensitive-context categories.

Required metrics:

| Metric | Direction | Notes |
|---|---:|---|
| Validation loss/perplexity | Lower | Report overall and per language on held-out corpus text |
| Toxic/abusive generation rate | Lower | Use a written labeling rule and confidence/counts |
| Unnecessary profanity rate | Lower | Separate legitimate quotation or explanation |
| Refusal accuracy | Higher | On explicitly harmful prompts |
| False-refusal rate | Lower | On benign prompts containing sensitive vocabulary |
| Reward-model score | Higher with caveats | Never treat as independent proof of safety |
| Human preference win rate | Higher | Blind or randomized presentation where practical |
| Fluency/relevance rating | Higher | Use the same rubric and language-qualified reviewers |

Evaluation procedure:

- Use frozen held-out prompt families with balanced language/category coverage.
- Blind and randomize human comparisons where feasible.
- Record ties and invalid outputs rather than forcing a winner.
- Report sample counts and uncertainty, such as bootstrap confidence intervals, alongside percentages.
- Keep automatic safety labels distinct from human judgments.
- Analyze failures and regressions, not just aggregate improvements.

Success means a clear reduction in held-out unsafe generations and a higher preference win rate than SFT, without a disproportionate increase in perplexity, false refusals, or loss of fluency/relevance. No single fixed percentage is required; reproducibility and honest trade-off reporting take priority.

### Phase 10 — Demo, documentation, and submission (Week 14)

Tasks:

- Build a local Gradio or Streamlit interface that sends the same prompt to base, SFT, and PPO checkpoints.
- Display model identity, language, generation settings, latency, and outputs side by side.
- Add a warning that the prototype can produce harmful or incorrect text and is not for production use.
- Complete setup instructions, data documentation, model cards, evaluation report, limitations, and ethical notes.
- Prepare final report, presentation, and a scripted demonstration using non-gratuitous examples.

Exit criteria:

- A new user can reproduce a smoke run and launch the demo from the documentation.
- All tables in the report can be traced to versioned configs and generated evaluation artifacts.
- Data and model limitations are visible in both the report and demo.

## 8. Four-Person Work Allocation

Assign named owners before Week 1 ends. Ownership means coordination and review responsibility; it does not prevent pair work.

| Workstream | Primary responsibilities |
|---|---|
| Data and tokenizer | Source/license manifest, preprocessing, deduplication, splits, tokenizer, dataset statistics |
| Model and pretraining | Transformer implementation, tests, profiling, base training, checkpoint reliability |
| Alignment | SFT, preference schema, reward model, PPO, alignment diagnostics |
| Evaluation and product | Safety rubric, metrics, human evaluation, keyword baseline, demo, final analysis |

Cross-review requirements:

- Data splits and leakage checks are reviewed by someone outside the data workstream.
- Model math and checkpoint resume are reviewed by someone outside the model workstream.
- Preference labels and safety rubric receive multilingual review.
- Final metrics are regenerated by a team member who did not author the evaluation code.

## 9. Test Strategy

### Unit tests

- Unicode normalization and language/script rules.
- Exact and near-duplicate handling.
- Stable document split assignment.
- Tokenizer special tokens and encode/decode round trips.
- Attention mask, RoPE, tensor shapes, padding loss mask, and tied weights.
- Preference ranking loss sign and reward extraction position.
- PPO ratio/clipping, KL, advantage, and padding masks.
- Metric calculations, including refusal and false-refusal denominators.

### Integration tests

- Raw sample through cleaned and tokenized shard.
- Tiny LM training, save, load, resume, and generation.
- Tiny SFT run using the final conversation format.
- Tiny reward-model run that learns a synthetic preference pattern.
- Tiny PPO run that completes without NaNs and updates the policy.
- End-to-end evaluation that produces a versioned result table.

### Pre-flight checks for expensive runs

- Config and dataset manifests are saved.
- Train/validation/test leakage checks pass.
- A small batch passes forward and backward without NaNs.
- Peak memory is below the safety margin.
- Checkpoint save/resume succeeds.
- Evaluation prompts are not used for training or model selection.

## 10. Artifact and Experiment Tracking

Every major artifact should carry a version or immutable identifier:

- Corpus source and split manifests.
- Preprocessing and tokenizer configs.
- Tokenizer model and statistics.
- Base, SFT, reward-model, and PPO checkpoints.
- Preference dataset version and annotation rubric.
- Evaluation prompt-set version and generation config.
- Human-rating export and aggregate report.

Each training run should record:

- Git revision and working-tree state.
- Full resolved config and random seed.
- Input artifact identifiers/checksums.
- Hardware/software environment.
- Training time, sample/token count, and peak memory.
- Metrics and checkpoint-selection decision.

## 11. Key Risks and Mitigations

| Risk | Mitigation |
|---|---|
| Unclear corpus rights | Require a source/license manifest; exclude uncertain sources; do not redistribute restricted raw text |
| Uneven language quality | Balance sampling and report metrics separately by language and source |
| Cross-split contamination | Deduplicate before document-level splitting; group prompt paraphrases/families |
| Tokenizer disadvantages one script | Compare per-language fragmentation and retrain with balanced/weighted samples |
| Compute or storage shortfall | Profile early; stream/shard data; use mixed precision and gradient accumulation; reduce corpus/model only with documentation |
| Base model learns poorly | Validate on tiny overfit and scaled pilot runs before committing full compute |
| Preference labels encode bias | Written rubric, multiple reviewers for ambiguous pairs, language/category analysis |
| Reward-model shortcuts | Check length/refusal/language correlations and manually inspect score extremes |
| PPO instability or reward hacking | Conservative updates, KL control, frequent checkpoints, multi-metric rollback rules |
| Safety gained through blanket refusal | Include benign sensitive-context prompts and make false-refusal rate a selection metric |
| Harmful data exposure | Restrict raw data, sanitize reports/demos, minimize quoted offensive content |
| Results cannot be reproduced | Version configs/data manifests, save complete state, and trace every final table to an artifact |

## 12. Deliverables Checklist

- [ ] Licensed-source corpus manifest and reproducible preprocessing pipeline.
- [ ] Cleaned split manifests and corpus statistics for all three languages.
- [ ] Trained shared tokenizer and fragmentation/coverage report.
- [ ] Tested decoder-only Transformer source code.
- [ ] Base pretrained checkpoint and per-language validation curves.
- [ ] Multilingual instruction dataset and SFT checkpoint.
- [ ] Verified multilingual preference dataset and annotation rubric.
- [ ] Reward-model checkpoint and language/category ranking report.
- [ ] PPO implementation, aligned checkpoint, and training diagnostics.
- [ ] Keyword-filter baseline.
- [ ] Base/SFT/PPO capability and safety comparison with uncertainty and failure analysis.
- [ ] Local side-by-side demo.
- [ ] README/setup guide, data card, model cards, final report, and presentation.

## 13. Immediate Next Actions

1. Initialize the repository structure and dependency environment.
2. Confirm the available training GPU, storage budget, and experiment-tracking tool.
3. Build the corpus source/license inventory for Hindi, Bengali, and Tamil.
4. Write the safety annotation and evaluation rubric before producing preference pairs.
5. Implement a tiny synthetic end-to-end path to validate the architecture, checkpointing, reward loss, PPO plumbing, and reporting before processing or training at full scale.

