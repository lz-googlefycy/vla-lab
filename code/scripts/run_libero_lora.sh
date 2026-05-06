#!/bin/bash
# ============================================================
# LIBERO LoRA 微调启动脚本 (OpenVLA-7B)
#
# 用法:
#   bash run_libero_lora.sh <suite> [extra args]
# 例:
#   bash run_libero_lora.sh spatial
#   bash run_libero_lora.sh object
#   bash run_libero_lora.sh goal
#   bash run_libero_lora.sh long
#
# 在开发机上跑（H20-3e, 144 GB）
# ============================================================
set -euo pipefail

SUITE=${1:?"usage: $0 <spatial|object|goal|long> [extra args]"}
shift || true

case "$SUITE" in
  spatial) DATASET=libero_spatial_no_noops ; MAX_STEPS=50000 ;;
  object)  DATASET=libero_object_no_noops  ; MAX_STEPS=50000 ;;
  goal)    DATASET=libero_goal_no_noops    ; MAX_STEPS=50000 ;;
  long)    DATASET=libero_10_no_noops      ; MAX_STEPS=100000 ;;
  *) echo "Unknown suite: $SUITE"; exit 1 ;;
esac

# 默认路径（容器内）
VLA_PATH=${VLA_PATH:-/workspace/models/openvla-7b}
DATA_ROOT=${DATA_ROOT:-/workspace/datasets/modified_libero_rlds}
OUTPUT_ROOT=${OUTPUT_ROOT:-/workspace/output}

EXP_ID="EXP-$(date +%Y%m%d)-libero_${SUITE}_lora_r32"
RUN_DIR="$OUTPUT_ROOT/$EXP_ID"
mkdir -p "$RUN_DIR"

echo "=========================================="
echo "  LIBERO LoRA Fine-tuning"
echo "=========================================="
echo "  Suite       : $SUITE ($DATASET)"
echo "  Max steps   : $MAX_STEPS"
echo "  VLA path    : $VLA_PATH"
echo "  Data root   : $DATA_ROOT"
echo "  Output dir  : $RUN_DIR"
echo "  Exp ID      : $EXP_ID"
echo "=========================================="

# Sanity: make sure the dataset dir exists
if [ ! -d "$DATA_ROOT/$DATASET" ]; then
    echo "ERROR: dataset $DATA_ROOT/$DATASET does not exist."
    ls "$DATA_ROOT/" 2>/dev/null || true
    exit 1
fi

cd /workspace/openvla

# Use conda's torchrun explicitly (avoid /usr/bin/python3 shebang issue)
TORCHRUN=${TORCHRUN:-/opt/conda/bin/torchrun}

# OpenVLA appendix E config (real flags from finetune.py)
"$TORCHRUN" --standalone --nnodes 1 --nproc-per-node 1 \
    vla-scripts/finetune.py \
    --vla_path "$VLA_PATH" \
    --data_root_dir "$DATA_ROOT" \
    --dataset_name "$DATASET" \
    --run_root_dir "$RUN_DIR" \
    --batch_size 16 \
    --grad_accumulation_steps 4 \
    --learning_rate 5e-4 \
    --max_steps $MAX_STEPS \
    --save_steps 5000 \
    --save_latest_checkpoint_only False \
    --image_aug False \
    --use_lora True \
    --lora_rank 32 \
    --lora_dropout 0.0 \
    --use_quantization False \
    --shuffle_buffer_size 100000 \
    --wandb_project openvla-libero \
    --wandb_entity liuzhi7 \
    --run_id_note "$EXP_ID" \
    "$@" 2>&1 | tee "$RUN_DIR/train.log"

echo "=== Training done. Check $RUN_DIR/train.log ==="
