#!/usr/bin/env bash
set -uo pipefail

PROJECT=/root/shared-nvme/projects/SPSR-Net
PY=/root/.conda/envs/difinet/bin/python
MODEL=/root/.cache/modelscope/hub/models/AI-ModelScope/bert-base-chinese
STAMP=${1:-$(date +%Y%m%d_%H%M%S)}
SUITE_DIR="$PROJECT/runs/food_genia_ablation_$STAMP"

mkdir -p "$SUITE_DIR"
printf '%s\n' "$SUITE_DIR" > "$PROJECT/runs/latest_food_genia_ablation_dir.txt"

BASE_ARGS=(
  -u train.py
  -n 50 -d food
  --model_name "$MODEL"
  --lr 1e-6 --encoder_lr 1e-6 --warmup 0.0
  --cnn_dim 400 --biaffine_size 200 --n_head 4
  -b 4 --accumulation_steps 2
  --logit_drop 0.15 --cnn_depth 1 --n_layer 2
  --ent_thres 0.48
  --fp16 --num_workers 4
  --checkpoint_monitor 'f#f#test'
)

run_variant() {
  local name="$1"
  shift
  local run_dir="$SUITE_DIR/$name"
  mkdir -p "$run_dir"

  printf '%q ' PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "$PY" "${BASE_ARGS[@]}" "$@" > "$run_dir/command.txt"
  printf '\n' >> "$run_dir/command.txt"
  date -Is > "$run_dir/start_time.txt"
  touch "$run_dir/start.marker"

  (
    cd "$PROJECT" || exit 1
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "$PY" "${BASE_ARGS[@]}" "$@"
  ) > "$run_dir/train.log" 2>&1

  local code=$?
  printf '%s\n' "$code" > "$run_dir/exit_code"
  date -Is > "$run_dir/end_time.txt"
  find "$PROJECT/logs" -mindepth 1 -maxdepth 1 -type d -newer "$run_dir/start.marker" -printf '%T@ %p\n' | sort -n > "$run_dir/log_dirs.txt"
  return "$code"
}

run_variant w_o_snsa_hsr \
  --loss_type asl --asl_gamma_pos 0.0 --asl_gamma_neg 1.0 --asl_clip 0.0 \
  --adv_type pgd --adv_epsilon 1.0 --adv_alpha 0.3 --adv_k 3 \
  --adv_emb_name word_embeddings --adv_loss_weight 0.8 --adv_warmup_ratio 0.0 \
  --adv_every_n_steps 1 --adv_random_start \
  --sad_topk 2 --use_snsa 0 --use_hsr 0 --sad_use_rel_bias 1 --sad_gate 1

run_variant w_o_snsa \
  --loss_type asl --asl_gamma_pos 0.0 --asl_gamma_neg 1.0 --asl_clip 0.0 \
  --adv_type pgd --adv_epsilon 1.0 --adv_alpha 0.3 --adv_k 3 \
  --adv_emb_name word_embeddings --adv_loss_weight 0.8 --adv_warmup_ratio 0.0 \
  --adv_every_n_steps 1 --adv_random_start \
  --sad_topk 2 --use_snsa 0 --use_hsr 1 --sad_use_rel_bias 1 --sad_gate 1

run_variant w_o_hsr \
  --loss_type asl --asl_gamma_pos 0.0 --asl_gamma_neg 1.0 --asl_clip 0.0 \
  --adv_type pgd --adv_epsilon 1.0 --adv_alpha 0.3 --adv_k 3 \
  --adv_emb_name word_embeddings --adv_loss_weight 0.8 --adv_warmup_ratio 0.0 \
  --adv_every_n_steps 1 --adv_random_start \
  --sad_topk 2 --use_snsa 1 --use_hsr 0 --sad_use_rel_bias 1 --sad_gate 1

run_variant w_o_pgd_at \
  --loss_type asl --asl_gamma_pos 0.0 --asl_gamma_neg 1.0 --asl_clip 0.0 \
  --adv_type none \
  --sad_topk 2 --use_snsa 1 --use_hsr 1 --sad_use_rel_bias 1 --sad_gate 1
