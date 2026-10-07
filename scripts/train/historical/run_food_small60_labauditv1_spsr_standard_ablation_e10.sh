#!/usr/bin/env bash
set -euo pipefail

# Standard SPSR-Net component ablations on FOOD. Every variant uses PGD;
# only SNSA and/or HSR differ.
PROJECT=/root/shared-nvme/projects/SPSR-Net
PY=/root/.conda/envs/difinet/bin/python
MODEL=/root/.cache/modelscope/hub/AI-ModelScope/bert-base-chinese
DATA_DIR="$PROJECT/preprocess/outputs/food_familytemplate_801010_t99_cleanv1_dedupv1_small60_labauditv1"
SUITE_DIR="$PROJECT/outputs/food_small60_labauditv1_spsr_standard_ablation_e10_20260921"

mkdir -p "$SUITE_DIR"
printf '%s\n' "$SUITE_DIR" > "$PROJECT/outputs/latest_food_small60_labauditv1_spsr_standard_ablation_dir.txt"

BASE_ARGS=(
  -u train.py -n 5 -d food --data_dir "$DATA_DIR" --model_name "$MODEL"
  --lr 1e-6 --encoder_lr 1e-6 --warmup 0.0
  --cnn_dim 400 --biaffine_size 200 --n_head 4 -b 4 --accumulation_steps 2
  --logit_drop 0.15 --cnn_depth 1 --n_layer 2 --ent_thres 0.48 --seed 0
  --adv_type pgd --adv_epsilon 1.0 --adv_alpha 0.3 --adv_k 3
  --adv_emb_name word_embeddings --adv_loss_weight 0.8 --adv_warmup_ratio 0.0
  --adv_every_n_steps 1 --adv_random_start
  --sad_topk 2 --sad_use_rel_bias 1 --sad_gate 1
  --fp16 --fp16_init_scale 1.0 --num_workers 4 --checkpoint_monitor 'f#f#dev'
)

run_variant() {
  local name=$1
  shift
  local run_dir="$SUITE_DIR/$name"
  mkdir -p "$run_dir"
  printf '%q ' PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "$PY" "${BASE_ARGS[@]}" "$@" \
    --epoch_metrics_path "$run_dir/epoch_metrics.csv" > "$run_dir/command.txt"
  printf '\n' >> "$run_dir/command.txt"
  (
    cd "$PROJECT"
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    SPSR_LOG_DIR="$run_dir/logs" SPSR_CHECKPOINT_DIR="$run_dir/checkpoints" \
      "$PY" "${BASE_ARGS[@]}" "$@" --epoch_metrics_path "$run_dir/epoch_metrics.csv"
  ) 2>&1 | tee "$run_dir/train.log"
}

# The w/o SNSA 10-epoch run has already completed.  Subsequent variants use
# the 5-epoch budget selected after observing its convergence.
# Run sequentially to give each PGD variant the whole GPU.
run_variant w_o_hsr --use_snsa 1 --use_hsr 0
run_variant w_o_snsa_hsr --use_snsa 0 --use_hsr 0
