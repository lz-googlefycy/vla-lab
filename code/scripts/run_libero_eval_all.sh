#!/bin/bash
# ============================================================
# LIBERO 4 suite 全量评测一键脚本（用于官方 ckpt 复现）
#
# 假定：
#   - 4 个官方 ckpt 已在 /workspace/models/openvla-7b-finetuned-libero-{spatial,object,goal,10}
#   - ./run_libero_eval.sh 已在同目录
#
# 用法：
#   bash run_libero_eval_all.sh [num_trials] [mode]
#     num_trials  默认 50（论文配置）
#     mode        smoke=quick (5 trial)|full=50 trial（默认）
# ============================================================
set -uo pipefail  # 注意：去掉 -e，让 1 个 suite 失败不影响其他

N_TRIAL=${1:-50}
MODE=${2:-full}

if [ "$MODE" = "smoke" ]; then
    N_TRIAL=5
    echo "=== SMOKE MODE: 5 trials per task ==="
fi

OUTPUT_BASE=/workspace/output
SUMMARY_FILE="$OUTPUT_BASE/EVAL_SUMMARY_$(date +%Y%m%d_%H%M%S).md"

mkdir -p "$OUTPUT_BASE"

cat > "$SUMMARY_FILE" << EOF
# LIBERO 4-suite Evaluation Summary

- Date: $(date '+%Y-%m-%d %H:%M:%S')
- Trials per task: $N_TRIAL
- Mode: $MODE
- GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1)

| Suite | Checkpoint | SR % | Total Trials | Status |
|---|---|---|---|---|
EOF

declare -A SUITE_CKPT
SUITE_CKPT[spatial]=/workspace/models/openvla-7b-finetuned-libero-spatial
SUITE_CKPT[object]=/workspace/models/openvla-7b-finetuned-libero-object
SUITE_CKPT[goal]=/workspace/models/openvla-7b-finetuned-libero-goal
SUITE_CKPT[long]=/workspace/models/openvla-7b-finetuned-libero-10

for SUITE in spatial object goal long; do
    CKPT="${SUITE_CKPT[$SUITE]}"
    echo ""
    echo "========================================================"
    echo "  [$SUITE] starting eval @ $(date '+%H:%M:%S')"
    echo "========================================================"

    if [ ! -d "$CKPT" ]; then
        echo "[skip] checkpoint not found: $CKPT"
        echo "| $SUITE | NOT_FOUND | - | - | ❌ skip |" >> "$SUMMARY_FILE"
        continue
    fi

    # Sanity: must contain at least 4 safetensors shards (full ckpt)
    NUM_SHARDS=$(ls "$CKPT"/model-*-of-*.safetensors 2>/dev/null | wc -l)
    if [ "$NUM_SHARDS" -lt 4 ]; then
        echo "[skip] $CKPT only has $NUM_SHARDS/4 shards (incomplete)"
        echo "| $SUITE | INCOMPLETE($NUM_SHARDS/4) | - | - | ❌ skip |" >> "$SUMMARY_FILE"
        continue
    fi

    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    bash "$SCRIPT_DIR/run_libero_eval.sh" "$SUITE" "$CKPT" "$N_TRIAL"
    EVAL_RC=$?

    # Find the latest EVAL- dir
    LAST_EVAL_DIR=$(ls -td "$OUTPUT_BASE"/EVAL-*-libero_${SUITE}* 2>/dev/null | head -1)
    if [ -n "$LAST_EVAL_DIR" ] && [ -f "$LAST_EVAL_DIR/eval.log" ]; then
        SR=$(tr '\r' '\n' < "$LAST_EVAL_DIR/eval.log" | grep -E 'Current total success rate' | tail -1 | grep -oE '[0-9]+\.[0-9]+' | tail -1)
        TOT=$(tr '\r' '\n' < "$LAST_EVAL_DIR/eval.log" | grep -E '# episodes completed so far' | tail -1 | grep -oE '[0-9]+')
        if [ -n "$SR" ]; then
            SR_PCT=$(python3 -c "print(f'{float(\"$SR\")*100:.1f}')")
            echo "| $SUITE | $(basename $CKPT) | **$SR_PCT** | ${TOT:-?} | ✅ done |" >> "$SUMMARY_FILE"
        else
            echo "| $SUITE | $(basename $CKPT) | ? | ? | ⚠️ no SR found |" >> "$SUMMARY_FILE"
        fi
    else
        echo "| $SUITE | $(basename $CKPT) | - | - | ❌ rc=$EVAL_RC |" >> "$SUMMARY_FILE"
    fi
done

echo ""
echo "========================================================"
echo "  Summary saved to: $SUMMARY_FILE"
echo "========================================================"
cat "$SUMMARY_FILE"
