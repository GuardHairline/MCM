#!/bin/bash
set -e

configs=(
  "scripts/configs/kaggle_quick_test/none/kaggle_quick_none.json"
  "scripts/configs/kaggle_quick_test/replay/kaggle_quick_replay.json"
  "scripts/configs/kaggle_quick_test/ewc/kaggle_quick_ewc.json"
  "scripts/configs/kaggle_quick_test/si/kaggle_quick_si.json"
  "scripts/configs/kaggle_quick_test/gem/kaggle_quick_gem.json"
  "scripts/configs/kaggle_quick_test/mas/kaggle_quick_mas.json"
  "scripts/configs/kaggle_quick_test/moe_adapters/kaggle_quick_moe_adapters.json"
  "scripts/configs/kaggle_quick_test/lwf/kaggle_quick_lwf.json"
  "scripts/configs/kaggle_quick_test/tam_cl/kaggle_quick_tam_cl.json"
  "scripts/configs/kaggle_quick_test/none_8heads/kaggle_quick_none_8heads.json"
)

for cfg in "${configs[@]}"; do
  echo ">>> Running $cfg"
  python -m scripts.train_with_zero_shot --config "$cfg"
done
