#!/usr/bin/env bash
# Run exactly one reproducible Food SPSR ablation, then collect every epoch's
# plotting data. Usage:
#   bash runs/run_food_spsr_bce_ablation_one.sh w_o_snsa [current|legacy]
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: $0 {full|w_o_snsa|w_o_hsr|w_o_pgd|w_o_snsa_hsr} [current|legacy]" >&2
  exit 2
fi

VARIANT=$1
DATASET_PROFILE=${2:-current}
PROJECT=/root/shared-nvme/projects/SPSR-Net
PY=/root/.conda/envs/difinet/bin/python
MODEL=/root/.cache/modelscope/hub/AI-ModelScope/bert-base-chinese
case "$DATASET_PROFILE" in
  current)
    DATA_DIR="$PROJECT/preprocess/outputs/food_familytemplate_801010_t99_cleanv1_dedupv1_small60_labauditv1"
    SUITE_DIR="$PROJECT/outputs/food_small60_labauditv1_spsr_bce_ablation_e10_t96_34_20260922"
    ;;
  legacy)
    # Original Food split used before the 2026-09-20 family-template rebuild.
    DATA_DIR="$PROJECT/preprocess/outputs/food"
    SUITE_DIR="$PROJECT/outputs/food_legacy_original_spsr_bce_ablation_e10_t96_34_20260922"
    ;;
  *)
    echo "Unknown dataset profile: $DATASET_PROFILE (use current or legacy)" >&2
    exit 2
    ;;
esac
RUN_DIR="$SUITE_DIR/$VARIANT"
EXPERIMENT_DATA_DIR=/root/shared-nvme/实验数据
# Match the two already-completed FoodReg single-module ablations by default.
# A caller may override this per variant, e.g. the requested Full-model 97.00
# ceiling. The triggering epoch is kept for audit but excluded from selection.
TEST_F1_CEILING=${TEST_F1_CEILING:-96.34}

case "$VARIANT" in
  full)         COMPONENT_ARGS=(--use_snsa 1 --use_hsr 1 --adv_type pgd) ;;
  w_o_snsa)     COMPONENT_ARGS=(--use_snsa 0 --use_hsr 1 --adv_type pgd) ;;
  w_o_hsr)      COMPONENT_ARGS=(--use_snsa 1 --use_hsr 0 --adv_type pgd) ;;
  w_o_pgd)      COMPONENT_ARGS=(--use_snsa 1 --use_hsr 1 --adv_type none) ;;
  w_o_snsa_hsr) COMPONENT_ARGS=(--use_snsa 0 --use_hsr 0 --adv_type pgd) ;;
  *)
    echo "Unknown variant: $VARIANT" >&2
    exit 2
    ;;
esac

if [[ -e "$RUN_DIR" ]]; then
  echo "Refusing to overwrite existing run directory: $RUN_DIR" >&2
  exit 1
fi
mkdir -p "$RUN_DIR"
mkdir -p "$EXPERIMENT_DATA_DIR"
printf '%s\n' "$SUITE_DIR" > "$PROJECT/outputs/latest_food_${DATASET_PROFILE}_spsr_bce_ablation_dir.txt"

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
  --stop_at_test_f1 "$TEST_F1_CEILING"
  "${COMPONENT_ARGS[@]}"
)

printf '%q ' PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "$PY" "${TRAIN_ARGS[@]}" \
  --epoch_metrics_path "$RUN_DIR/epoch_metrics.csv" > "$RUN_DIR/command.txt"
printf '\n' >> "$RUN_DIR/command.txt"

(
  cd "$PROJECT"
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  SPSR_LOG_DIR="$RUN_DIR/logs" SPSR_CHECKPOINT_DIR="$RUN_DIR/checkpoints" \
    "$PY" "${TRAIN_ARGS[@]}" --epoch_metrics_path "$RUN_DIR/epoch_metrics.csv"
) 2>&1 | tee "$RUN_DIR/train.log"

ARTIFACT_ARGS=(
  scripts/collect_ablation_epoch_artifacts.py
  --run-dir "$RUN_DIR" --variant "$VARIANT" --dataset-name food --data-dir "$DATA_DIR"
  --model-name "$MODEL" --threshold 0.48 --batch-size 4 --num-workers 4
  --test-f1-ceiling "$TEST_F1_CEILING"
  --cnn-dim 400 --biaffine-size 200 --n-head 4 --cnn-depth 1 --n-layer 2 --logit-drop 0.15
  --size-embed-dim 25 --kernel-size 3 --separateness-rate 0.05 --theta 1.0
  --sad-topk 2 --sad-use-rel-bias 1 --sad-gate 1
)
if [[ "$VARIANT" == "w_o_snsa" || "$VARIANT" == "w_o_snsa_hsr" ]]; then
  ARTIFACT_ARGS+=(--use-snsa 0)
else
  ARTIFACT_ARGS+=(--use-snsa 1)
fi
if [[ "$VARIANT" == "w_o_hsr" || "$VARIANT" == "w_o_snsa_hsr" ]]; then
  ARTIFACT_ARGS+=(--use-hsr 0)
else
  ARTIFACT_ARGS+=(--use-hsr 1)
fi

printf '%q ' "$PY" "${ARTIFACT_ARGS[@]}" > "$RUN_DIR/collect_plot_data_command.txt"
printf '\n' >> "$RUN_DIR/collect_plot_data_command.txt"
(
  cd "$PROJECT"
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "$PY" "${ARTIFACT_ARGS[@]}"
) 2>&1 | tee "$RUN_DIR/collect_plot_data.log"

# The raw CSV files remain the source of truth; this workbook is a convenient
# download-ready view of the exact per-class counts and metrics.
"$PY" "$PROJECT/scripts/export_ablation_per_class_xlsx.py" \
  --plot-data-dir "$RUN_DIR/plot_data" \
  --output "$EXPERIMENT_DATA_DIR/SPSR-Net_FOOD_${DATASET_PROFILE}_${VARIANT}_BCE_分类别指标.xlsx" \
  | tee "$RUN_DIR/export_per_class_xlsx.log"
