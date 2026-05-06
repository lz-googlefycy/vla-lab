#!/bin/bash
# ============================================================
# 把本机已下载的 OpenVLA 官方 LIBERO ckpt rsync 到开发机
#
# 假设本机：~/openvla_assets/finetuned_libero/
# 假设开发机：/workspace/jfs/models/
#
# 用法（在本机跑）：
#   bash sync_official_ckpts.sh [SUITE]
#   SUITE = spatial / object / goal / 10 / all (default)
# ============================================================
set -euo pipefail

SUITE=${1:-all}
LOCAL_DIR=~/openvla_assets/finetuned_libero
REMOTE=<dev-machine>
PORT=<dev-port>
REMOTE_DIR=/workspace/jfs/models

if [ "$SUITE" = "all" ]; then
    SUITES="spatial object goal 10"
else
    SUITES="$SUITE"
fi

for s in $SUITES; do
    NAME="openvla-7b-finetuned-libero-$s"
    if [ ! -d "$LOCAL_DIR/$NAME" ]; then
        echo "[skip] $NAME not yet downloaded"
        continue
    fi
    SIZE=$(du -sh "$LOCAL_DIR/$NAME" | awk '{print $1}')
    echo "=== Syncing $NAME ($SIZE) ==="
    rsync -aP -e "ssh -p $PORT -o StrictHostKeyChecking=no" \
        "$LOCAL_DIR/$NAME/" \
        "$REMOTE:$REMOTE_DIR/$NAME/"
done

echo ""
echo "=== Verify on remote ==="
ssh -p $PORT $REMOTE "du -sh $REMOTE_DIR/openvla-7b-finetuned-libero-* 2>&1"
