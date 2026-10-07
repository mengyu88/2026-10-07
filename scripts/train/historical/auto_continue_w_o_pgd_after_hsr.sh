#!/usr/bin/env bash
set -uo pipefail

PROJECT=/root/shared-nvme/projects/SPSR-Net
PY=/root/.conda/envs/difinet/bin/python
MODEL=/root/.cache/modelscope/hub/models/AI-ModelScope/bert-base-chinese
SUITE_DIR="$PROJECT/runs/food_genia_ablation_remaining_e10_20260704_204323"
HSR_RUN="$SUITE_DIR/w_o_hsr_continue_from_epoch1_20260705_2202"
WATCH_LOG="$SUITE_DIR/auto_continue_w_o_pgd_after_hsr.log"

log() {
  printf '[%s] %s\n' "$(date -Is)" "$*" >> "$WATCH_LOG"
}

log "watcher started"

is_alive_non_zombie() {
  local pid="$1"
  local stat
  stat=$(ps -p "$pid" -o stat= 2>/dev/null | awk '{print $1}')
  [ -n "$stat" ] || return 1
  case "$stat" in
    Z*) return 1 ;;
    *) return 0 ;;
  esac
}

HSR_PID=""
if [ -f "$HSR_RUN/wrapper.pid" ]; then
  HSR_PID=$(tr -dc '0-9' < "$HSR_RUN/wrapper.pid")
fi

if [ -n "$HSR_PID" ]; then
  log "waiting for w_o_hsr_continue wrapper pid=$HSR_PID"
  while is_alive_non_zombie "$HSR_PID"; do
    sleep 120
  done
  log "w_o_hsr_continue wrapper pid=$HSR_PID exited"
else
  log "no wrapper pid found for w_o_hsr_continue; continuing immediately"
fi

if ps -eo cmd | grep -F -- '--use_snsa 1 --use_hsr 0' | grep -F 'train.py' | grep -v grep >/dev/null; then
  log "w_o_hsr train process still detected after wrapper exit; refusing to start next run"
  exit 2
fi

if ps -eo cmd | grep -F -- '--adv_type none' | grep -F -- '--use_snsa 1 --use_hsr 1' | grep -F 'train.py' | grep -v grep >/dev/null; then
  log "w_o_pgd_at appears to already be running; exiting"
  exit 0
fi

STAMP=$(date +%Y%m%d_%H%M%S)
RUN_DIR="$SUITE_DIR/w_o_pgd_at_$STAMP"
mkdir -p "$RUN_DIR"
date -Is > "$RUN_DIR/start_time.txt"
touch "$RUN_DIR/start.marker"

printf '%q ' PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "$PY" -u train.py \
  -n 10 -d food \
  --model_name "$MODEL" \
  --lr 1e-6 --encoder_lr 1e-6 --warmup 0.0 \
  --cnn_dim 400 --biaffine_size 200 --n_head 4 \
  -b 4 --accumulation_steps 2 \
  --logit_drop 0.15 --cnn_depth 1 --n_layer 2 \
  --ent_thres 0.48 \
  --fp16 --num_workers 4 \
  --checkpoint_monitor 'f#f#test' \
  --loss_type asl --asl_gamma_pos 0.0 --asl_gamma_neg 1.0 --asl_clip 0.0 \
  --adv_type none \
  --sad_topk 2 --use_snsa 1 --use_hsr 1 --sad_use_rel_bias 1 --sad_gate 1 \
  > "$RUN_DIR/command.txt"
printf '\n' >> "$RUN_DIR/command.txt"

log "starting w_o_pgd_at in $RUN_DIR"
(
  cd "$PROJECT" || exit 1
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "$PY" -u train.py \
    -n 10 -d food \
    --model_name "$MODEL" \
    --lr 1e-6 --encoder_lr 1e-6 --warmup 0.0 \
    --cnn_dim 400 --biaffine_size 200 --n_head 4 \
    -b 4 --accumulation_steps 2 \
    --logit_drop 0.15 --cnn_depth 1 --n_layer 2 \
    --ent_thres 0.48 \
    --fp16 --num_workers 4 \
    --checkpoint_monitor 'f#f#test' \
    --loss_type asl --asl_gamma_pos 0.0 --asl_gamma_neg 1.0 --asl_clip 0.0 \
    --adv_type none \
    --sad_topk 2 --use_snsa 1 --use_hsr 1 --sad_use_rel_bias 1 --sad_gate 1
) > "$RUN_DIR/train.log" 2>&1

code=$?
printf '%s\n' "$code" > "$RUN_DIR/exit_code"
date -Is > "$RUN_DIR/end_time.txt"
find "$PROJECT/logs" -mindepth 1 -maxdepth 1 -type d -newer "$RUN_DIR/start.marker" -printf '%T@ %p\n' | sort -n > "$RUN_DIR/log_dirs.txt"
log "w_o_pgd_at finished with exit_code=$code"
exit "$code"
