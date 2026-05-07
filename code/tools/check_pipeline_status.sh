#!/bin/bash
# 一键检查所有正在跑的 pipeline 状态
# 用法：bash code/tools/check_pipeline_status.sh
set -uo pipefail

echo "=========================================="
echo "  Pipeline status @ $(date '+%H:%M:%S')"
echo "=========================================="
echo ""

echo "=== Local downloads ==="
du -sh ~/openvla_assets/finetuned_libero/openvla-7b-finetuned-libero-{spatial,object,goal,10} 2>/dev/null
DL_PROCS=$(ps aux | grep 'hf.*download.*finetuned' | grep -v grep | wc -l)
echo "active hf download procs: $DL_PROCS"
echo ""

echo "=== Local scp procs ==="
ps aux | grep 'scp.*libero-' | grep -v grep | awk '{printf "  PID=%s elapsed=%s\n", $2, $11}' | head
echo ""

echo "=== Dev machine models ==="
ssh -p <dev-port> <dev-machine> 'du -sh /workspace/jfs/models/openvla-7b-finetuned-libero-* 2>/dev/null' 2>/dev/null
echo ""

echo "=== Dev eval running? ==="
ssh -p <dev-port> <dev-machine> 'pgrep -af run_libero_eval | grep -v grep | head -3' 2>/dev/null
echo ""

echo "=== Latest eval logs (per suite) ==="
for s in spatial object goal long; do
    LATEST=$(ssh -p <dev-port> <dev-machine> "ls -t /workspace/jfs/output/eval_${s}_*.log 2>/dev/null | head -1" 2>/dev/null)
    if [ -n "$LATEST" ]; then
        echo "--- $s :: $LATEST ---"
        ssh -p <dev-port> <dev-machine> "tr '\r' '\n' < $LATEST 2>&1 | grep -E 'Current task success rate|Current total success rate|task_id' | tail -3" 2>/dev/null
    fi
done
echo ""

echo "=== auto_pipeline log ==="
tail -8 /tmp/auto_pipeline_v3.log 2>/dev/null

echo ""
echo "=== Dev GPU ==="
ssh -p <dev-port> <dev-machine> 'nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv' 2>/dev/null
