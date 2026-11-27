#!/bin/sh
# Reproduce the full IndicSafeLM local run, stage by stage.
# Each stage validates its inputs and writes a manifest under artifacts/runs/.
set -eu
cd "$(dirname "$0")/.."
PY=.venv/bin/python
STAGE="${1:-all}"

run() { echo "==> $*"; "$@"; }

case "$STAGE" in
  all|data)
    run $PY scripts/prepare_wikipedia.py --max-chars-per-language 60000000
    run $PY scripts/run_data_pipeline.py --input data/interim/wikipedia-20231101.jsonl \
        --config configs/data_pipeline_local.json --output data/processed/corpus-v1
    run $PY scripts/train_tokenizer.py
    run $PY scripts/build_datasets.py
    [ "$STAGE" = data ] && exit 0 ;;
esac
case "$STAGE" in
  all|pretrain) run $PY scripts/pretrain.py --minutes 70; [ "$STAGE" = pretrain ] && exit 0 ;;
esac
case "$STAGE" in
  all|align)
    run $PY scripts/sft.py
    run $PY scripts/train_reward.py
    run $PY scripts/ppo.py
    run $PY scripts/evaluate.py
    run $PY scripts/make_figures.py ;;
esac
