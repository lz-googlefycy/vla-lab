#!/bin/bash
# ============================================================
# LIBERO 评测启动脚本
#
# 用法:
#   bash run_libero_eval.sh <suite> <ckpt_path> [num_trials_per_task]
# 例:
#   bash run_libero_eval.sh spatial /workspace/output/EXP-.../checkpoint-50000 50
# ============================================================
set -euo pipefail

SUITE=${1:?"usage: $0 <spatial|object|goal|long> <ckpt_path> [num_trials]"}
CKPT=${2:?"need ckpt path"}
N_TRIAL=${3:-50}

case "$SUITE" in
  spatial) TASK_SUITE=libero_spatial ;;
  object)  TASK_SUITE=libero_object  ;;
  goal)    TASK_SUITE=libero_goal    ;;
  long)    TASK_SUITE=libero_10      ;;
  *) echo "Unknown suite: $SUITE"; exit 1 ;;
esac

EVAL_DIR=$(dirname "$CKPT")/eval_${TASK_SUITE}_$(date +%Y%m%d_%H%M%S)
mkdir -p "$EVAL_DIR"

echo "=========================================="
echo "  LIBERO Evaluation"
echo "=========================================="
echo "  Suite          : $SUITE ($TASK_SUITE)"
echo "  Checkpoint     : $CKPT"
echo "  Trials/task    : $N_TRIAL"
echo "  Output dir     : $EVAL_DIR"
echo "=========================================="

cd /workspace/openvla

python experiments/robot/libero/run_libero_eval.py \
    --pretrained_checkpoint "$CKPT" \
    --task_suite_name "$TASK_SUITE" \
    --num_trials_per_task "$N_TRIAL" \
    --center_crop True \
    --local_log_dir "$EVAL_DIR" \
    2>&1 | tee "$EVAL_DIR/eval.log"

echo "=== Eval done. See $EVAL_DIR/eval.log ==="
echo ""
grep -E "task_name|success_rate|TOTAL|avg" "$EVAL_DIR/eval.log" | tail -20
