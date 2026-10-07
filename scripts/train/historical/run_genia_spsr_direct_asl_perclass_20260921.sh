#!/usr/bin/env bash
# Current SPSR-Net GENIA configuration: train with ASL from epoch 1.
# Metrics are selected by dev F1 only.
set -euo pipefail

PROJECT=/root/shared-nvme/projects/SPSR-Net
PY=/root/.conda/envs/difinet/bin/python
MODEL=/root/.cache/difinet_backbones/models--dmis-lab--biobert-v1.1/snapshots/551ca18efd7f052c8dfa0b01c94c2a8e68bc5488
DATA_DIR="$PROJECT/preprocess/outputs/genia"
RUN_DIR="$PROJECT/outputs/genia_spsr_direct_asl_pgd_perclass_20260921"

mkdir -p "$RUN_DIR"
printf '%s\n' "$RUN_DIR" > "$PROJECT/outputs/latest_genia_spsr_direct_asl_perclass_dir.txt"

ARGS=(
  -u train.py -n 10 -d genia --data_dir "$DATA_DIR" --model_name "$MODEL"
  --lr 1e-6 --encoder_lr 1e-6 --warmup 0.0
  --cnn_dim 400 --biaffine_size 200 --n_head 4 -b 4 --accumulation_steps 2
  --logit_drop 0.15 --cnn_depth 1 --n_layer 2 --ent_thres 0.48 --seed 0
  --loss_type asl --asl_gamma_pos 0.0 --asl_gamma_neg 1.0 --asl_clip 0.0
  --use_snsa 1 --use_hsr 1 --sad_topk 2 --sad_use_rel_bias 1 --sad_gate 1
  --adv_type pgd --adv_epsilon 1.0 --adv_alpha 0.3 --adv_k 3
  --adv_emb_name word_embeddings --adv_loss_weight 0.8 --adv_warmup_ratio 0.0
  --adv_every_n_steps 1 --adv_random_start
  --fp16 --fp16_init_scale 1.0 --num_workers 4 --checkpoint_monitor 'f#f#dev'
  --epoch_metrics_path "$RUN_DIR/epoch_metrics.csv"
)

printf '%q ' env HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "$PY" "${ARGS[@]}" > "$RUN_DIR/command.txt"
printf '\n' >> "$RUN_DIR/command.txt"
(
  cd "$PROJECT"
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  SPSR_LOG_DIR="$RUN_DIR/logs" SPSR_CHECKPOINT_DIR="$RUN_DIR/checkpoints" \
    "$PY" "${ARGS[@]}"
) 2>&1 | tee "$RUN_DIR/train.log"
