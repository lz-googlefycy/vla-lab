#!/bin/bash
# ============================================================
# LIBERO 评测启动脚本（兼容官方 ckpt + 自训 ckpt）
#
# 用法:
#   bash run_libero_eval.sh <suite> <ckpt_path> [num_trials_per_task] [extra args]
#
# 例:
#   # 官方 ckpt
#   bash run_libero_eval.sh spatial /workspace/models/openvla-7b-finetuned-libero-spatial 50
#   # 4-bit 推理（更快）
#   bash run_libero_eval.sh spatial <ckpt> 50 --load_in_4bit True
# ============================================================
set -euo pipefail

SUITE=${1:?"usage: $0 <spatial|object|goal|long> <ckpt_path> [num_trials] [extra]"}
CKPT=${2:?"need ckpt path"}
N_TRIAL=${3:-50}
shift 3 || true

case "$SUITE" in
  spatial) TASK_SUITE=libero_spatial ;;
  object)  TASK_SUITE=libero_object  ;;
  goal)    TASK_SUITE=libero_goal    ;;
  long|10) TASK_SUITE=libero_10      ;;
  *) echo "Unknown suite: $SUITE"; exit 1 ;;
esac

EVAL_ID="EVAL-$(date +%Y%m%d_%H%M%S)-${TASK_SUITE}"
EVAL_DIR="/workspace/output/$EVAL_ID"
mkdir -p "$EVAL_DIR"

echo "=========================================="
echo "  LIBERO Evaluation"
echo "=========================================="
echo "  Suite          : $SUITE ($TASK_SUITE)"
echo "  Checkpoint     : $CKPT"
echo "  Trials/task    : $N_TRIAL"
echo "  Output dir     : $EVAL_DIR"
echo "  Extra args     : $*"
echo "=========================================="

cd /workspace/openvla

/opt/conda/bin/python experiments/robot/libero/run_libero_eval.py \
    --pretrained_checkpoint "$CKPT" \
    --task_suite_name "$TASK_SUITE" \
    --num_trials_per_task "$N_TRIAL" \
    --center_crop True \
    --local_log_dir "$EVAL_DIR" \
    --run_id_note "$EVAL_ID" \
    --seed 7 \
    "$@" 2>&1 | tee "$EVAL_DIR/eval.log"

echo ""
echo "=== Eval done. See $EVAL_DIR/eval.log ==="
echo ""
grep -E "task_name|success_rate|TOTAL|avg|Total|Average" "$EVAL_DIR/eval.log" 2>/dev/null | tail -25 || true
