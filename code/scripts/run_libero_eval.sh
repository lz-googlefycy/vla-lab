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

# Render config: osmesa works in our k8s container (EGL fails, see env_setup.md)
export MUJOCO_GL=osmesa

# Force HF transformers to use local code, never call out to huggingface.co
# (dev machine has no internet)
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

# Auto-fix finetuned ckpts that reference remote auto_map ("openvla/openvla-7b--xxx")
# These configs are baked into HF release files; we patch them to local-only
# refs and copy the *.py files from the base model. Idempotent.
if [ -f "$CKPT/config.json" ]; then
    if grep -q 'openvla/openvla-7b--' "$CKPT/config.json" 2>/dev/null \
       || grep -q 'openvla/openvla-7b--' "$CKPT/preprocessor_config.json" 2>/dev/null; then
        echo "[fix] patching auto_map to local refs in $CKPT"
        sed -i 's|openvla/openvla-7b--configuration_prismatic|configuration_prismatic|g; s|openvla/openvla-7b--modeling_prismatic|modeling_prismatic|g' \
            "$CKPT/config.json" 2>/dev/null || true
        sed -i 's|openvla/openvla-7b--processing_prismatic|processing_prismatic|g' \
            "$CKPT/preprocessor_config.json" 2>/dev/null || true
    fi
    BASE_PY_DIR=/workspace/models/openvla-7b
    if [ -d "$BASE_PY_DIR" ] && [ ! -f "$CKPT/configuration_prismatic.py" ]; then
        echo "[fix] copying *.py from $BASE_PY_DIR to $CKPT"
        cp "$BASE_PY_DIR"/*.py "$CKPT/" 2>/dev/null || true
    fi
fi

# cd into EVAL_DIR so save_rollout_video drops MP4 in EVAL_DIR/rollouts/<DATE>/
cd "$EVAL_DIR"

# But import paths need to find /workspace/openvla
export PYTHONPATH=/workspace/openvla:${PYTHONPATH:-}

/opt/conda/bin/python /workspace/openvla/experiments/robot/libero/run_libero_eval.py \
    --pretrained_checkpoint "$CKPT" \
    --task_suite_name "$TASK_SUITE" \
    --num_trials_per_task "$N_TRIAL" \
    --center_crop True \
    --local_log_dir "$EVAL_DIR" \
    --run_id_note "$EVAL_ID" \
    --seed 7 \
    --use_wandb False \
    "$@" 2>&1 | tee "$EVAL_DIR/eval.log"

echo ""
echo "=== Eval done. See $EVAL_DIR/eval.log ==="
echo ""
grep -E "task_name|success_rate|TOTAL|avg|Total|Average" "$EVAL_DIR/eval.log" 2>/dev/null | tail -25 || true
