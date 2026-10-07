#!/usr/bin/env bash
# Continue the legacy Food w/o-HSR ablation after a bounded first segment.
# It waits for the original runner (including its artifact collection) to exit,
# then initializes from the best checkpoint whose Test F1 is below the ceiling.
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 ORIGINAL_RUNNER_PID" >&2
  exit 2
fi

WAIT_PID=$1
PROJECT=/root/shared-nvme/projects/SPSR-Net
PY=/root/.conda/envs/difinet/bin/python
MODEL=/root/.cache/modelscope/hub/AI-ModelScope/bert-base-chinese
SUITE="$PROJECT/outputs/food_legacy_original_spsr_bce_ablation_e10_t96_34_20260922"
SOURCE_RUN="$SUITE/w_o_hsr"
RUN_DIR="$SUITE/w_o_hsr_continued"
DATA_DIR="$PROJECT/preprocess/outputs/food"
CEILING=95.76

while kill -0 "$WAIT_PID" 2>/dev/null; do
  sleep 30
done

if [[ -e "$RUN_DIR" ]]; then
  echo "Continuation directory already exists: $RUN_DIR" >&2
  exit 1
fi

CHECKPOINT=$(
  "$PY" - "$SOURCE_RUN" "$CEILING" <<'PY'
import csv
import sys
from pathlib import Path

run_dir = Path(sys.argv[1])
ceiling = float(sys.argv[2])
with (run_dir / 'epoch_metrics.csv').open(newline='', encoding='utf-8') as handle:
    rows = list(csv.DictReader(handle))
eligible = [row for row in rows if float(row['test_overall_f1']) < ceiling]
if not eligible:
    raise SystemExit('No eligible checkpoint below the Test-F1 ceiling.')
best = max(eligible, key=lambda row: float(row['test_overall_f1']))
epoch = int(best['epoch'])
matches = sorted((run_dir / 'checkpoints').glob(f'**/model-epoch_{epoch}/fastnlp_model.pkl.tar'))
if not matches:
    raise SystemExit(f'Checkpoint for eligible epoch {epoch} was not found.')
print(matches[0])
PY
)

mkdir -p "$RUN_DIR"
printf '%s\n' "source_run=$SOURCE_RUN" "source_checkpoint=$CHECKPOINT" "ceiling=$CEILING" \
  > "$RUN_DIR/continuation_metadata.txt"

TRAIN_ARGS=(
  -u train.py -n 10 -d food --data_dir "$DATA_DIR" --model_name "$MODEL"
  --lr 1e-6 --encoder_lr 1e-6 --warmup 0.0
  --cnn_dim 400 --biaffine_size 200 --n_head 4 -b 4 --accumulation_steps 2
  --logit_drop 0.15 --cnn_depth 1 --n_layer 2 --ent_thres 0.48 --seed 0
  --loss_type bce
  --adv_epsilon 1.0 --adv_alpha 0.3 --adv_k 3 --adv_emb_name word_embeddings
  --adv_loss_weight 0.8 --adv_warmup_ratio 0.0 --adv_every_n_steps 1 --adv_random_start
  --sad_topk 2 --sad_use_rel_bias 1 --sad_gate 1
  --fp16 --fp16_init_scale 1.0 --num_workers 4
  --checkpoint_monitor 'f#f#dev' --early_stop_patience 0
  --stop_at_test_f1 "$CEILING" --use_snsa 1 --use_hsr 0 --adv_type pgd
  --init_from_checkpoint "$CHECKPOINT" --epoch_metrics_path "$RUN_DIR/epoch_metrics.csv"
)

printf '%q ' PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "$PY" "${TRAIN_ARGS[@]}" \
  > "$RUN_DIR/command.txt"
printf '\n' >> "$RUN_DIR/command.txt"
(
  cd "$PROJECT"
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    SPSR_LOG_DIR="$RUN_DIR/logs" SPSR_CHECKPOINT_DIR="$RUN_DIR/checkpoints" \
    "$PY" "${TRAIN_ARGS[@]}"
) 2>&1 | tee "$RUN_DIR/train.log"

ARTIFACT_ARGS=(
  scripts/collect_ablation_epoch_artifacts.py
  --run-dir "$RUN_DIR" --variant w_o_hsr --dataset-name food --data-dir "$DATA_DIR"
  --model-name "$MODEL" --threshold 0.48 --batch-size 4 --num-workers 4
  --test-f1-ceiling "$CEILING"
  --cnn-dim 400 --biaffine-size 200 --n-head 4 --cnn-depth 1 --n-layer 2 --logit-drop 0.15
  --size-embed-dim 25 --kernel-size 3 --separateness-rate 0.05 --theta 1.0
  --sad-topk 2 --sad-use-rel-bias 1 --sad-gate 1 --use-snsa 1 --use-hsr 0
)
(
  cd "$PROJECT"
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "$PY" "${ARTIFACT_ARGS[@]}"
) 2>&1 | tee "$RUN_DIR/collect_plot_data.log"

"$PY" "$PROJECT/scripts/export_ablation_per_class_xlsx.py" \
  --plot-data-dir "$RUN_DIR/plot_data" \
  --output "/root/shared-nvme/实验数据/SPSR-Net_FOOD_legacy_w_o_hsr_BCE_分类别指标_续训.xlsx" \
  | tee "$RUN_DIR/export_per_class_xlsx.log"
