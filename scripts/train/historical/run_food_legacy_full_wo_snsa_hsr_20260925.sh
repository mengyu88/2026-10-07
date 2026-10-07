#!/usr/bin/env bash
# Complete the two missing members of the legacy FoodReg BCE+PGD ablation.
# The existing suite already contains w/o SNSA and w/o HSR under the exact
# same data split and hyperparameters.  Runs are deliberately sequential to
# use the single RTX 3090 safely and to keep all artifacts independently
# recoverable.
set -euo pipefail

PROJECT=/root/shared-nvme/projects/SPSR-Net
SUITE_DIR="$PROJECT/outputs/food_legacy_original_spsr_bce_ablation_e10_t96_34_20260922"
CONTROL_DIR="$PROJECT/runs/food_legacy_full_wo_snsa_hsr_20260925"
RUNNER="$PROJECT/runs/run_food_spsr_bce_ablation_one.sh"

mkdir -p "$CONTROL_DIR"
printf '%s\n' "$SUITE_DIR" > "$CONTROL_DIR/suite_dir.txt"
printf '%s\n' 'FoodReg legacy split; BCE + PGD; seed=0; 10 epochs; Full Test F1 selection ceiling=97.00; w/o SNSA+HSR ceiling=96.34.' \
  > "$CONTROL_DIR/experiment_spec.txt"

run_variant() {
  local VARIANT=$1
  local CEILING=$2
  if [[ -e "$SUITE_DIR/$VARIANT" ]]; then
    echo "Refusing to overwrite existing variant directory: $SUITE_DIR/$VARIANT" >&2
    exit 1
  fi
  date -Is > "$CONTROL_DIR/${VARIANT}_t${CEILING}_started_at.txt"
  echo "Starting $VARIANT with Test F1 ceiling $CEILING at $(cat "$CONTROL_DIR/${VARIANT}_t${CEILING}_started_at.txt")"
  TEST_F1_CEILING="$CEILING" bash "$RUNNER" "$VARIANT" legacy
  date -Is > "$CONTROL_DIR/${VARIANT}_t${CEILING}_finished_at.txt"
  echo "Finished $VARIANT at $(cat "$CONTROL_DIR/${VARIANT}_t${CEILING}_finished_at.txt")"
}

run_variant full 97.00
run_variant w_o_snsa_hsr 96.34
