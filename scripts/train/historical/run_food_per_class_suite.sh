#!/usr/bin/env bash
set -euo pipefail

# Evaluate the available, complete FOOD checkpoints under the same test split
# and entity threshold. These are SPSR-Net configurations/ablations, not
# external baselines.
PROJECT=/root/shared-nvme/projects/SPSR-Net
PY=/root/.conda/envs/difinet/bin/python
MODEL=/root/.cache/modelscope/hub/AI-ModelScope/bert-base-chinese
OUT_DIR="$PROJECT/runs/food_per_class_suite_${1:-$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$OUT_DIR"
printf '%s\n' "$OUT_DIR" > "$PROJECT/runs/latest_food_per_class_suite_dir.txt"

evaluate() {
  local name=$1 checkpoint=$2 use_snsa=$3 use_hsr=$4
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "$PY" -u scripts/eval_per_class.py \
      --checkpoint "$checkpoint" --dataset-name food --model-name "$MODEL" \
      --threshold 0.48 --batch-size 4 --num-workers 4 \
      --use-snsa "$use_snsa" --use-hsr "$use_hsr" \
      --output "$OUT_DIR/${name}.json" \
      > "$OUT_DIR/${name}.log" 2>&1
}

cd "$PROJECT"
evaluate pgd_full \
  "$PROJECT/_saved_models/2026-07-03-00_07_48_429272/model-epoch_15-batch_45015-f#f#test_97.98/fastnlp_model.pkl.tar" 1 1
evaluate wo_snsa_hsr \
  "$PROJECT/_saved_models/2026-07-04-16_10_44_078110/model-epoch_11-batch_33011-f#f#test_97.74/fastnlp_model.pkl.tar" 0 0
evaluate wo_hsr \
  "$PROJECT/_saved_models/2026-07-05-22_12_33_633670/model-epoch_8-batch_24008-f#f#test_95.87/fastnlp_model.pkl.tar" 1 0
evaluate wo_pgd \
  "$PROJECT/_saved_models/2026-07-06-02_11_33_112790/model-epoch_10-batch_30010-f#f#test_96.59/fastnlp_model.pkl.tar" 1 1
