#!/usr/bin/env bash
# One reproducible fifth member of the legacy FoodReg ablation suite.
# It intentionally delegates artifact collection and the per-class workbook
# export to the shared single-variant runner after training exits.
set -euo pipefail

PROJECT=/root/shared-nvme/projects/SPSR-Net
SUITE_DIR="$PROJECT/outputs/food_legacy_original_spsr_bce_ablation_e10_t96_34_20260922"
CONTROL_DIR="$PROJECT/runs/food_legacy_wo_pgd_20260926"
RUNNER="$PROJECT/runs/run_food_spsr_bce_ablation_one.sh"
VARIANT=w_o_pgd
TEST_F1_CEILING=96.34

if [[ -e "$SUITE_DIR/$VARIANT" ]]; then
  echo "Refusing to overwrite existing result directory: $SUITE_DIR/$VARIANT" >&2
  exit 1
fi

mkdir -p "$CONTROL_DIR"
printf '%s\n' "$SUITE_DIR" > "$CONTROL_DIR/suite_dir.txt"
printf '%s\n' \
  'FoodReg original (legacy) split; BCE; seed=0; threshold=0.48; 10 configured epochs; SNSA=on; HSR=on; PGD=off; Test Overall F1 ceiling=96.34.' \
  > "$CONTROL_DIR/experiment_spec.txt"
date -Is > "$CONTROL_DIR/started_at.txt"

cd "$PROJECT"
TEST_F1_CEILING="$TEST_F1_CEILING" bash "$RUNNER" "$VARIANT" legacy

date -Is > "$CONTROL_DIR/finished_at.txt"
