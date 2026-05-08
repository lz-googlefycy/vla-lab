#!/bin/bash
# ============================================================
# Push Spirit v1.5 image + ckpts from local workstation to dev machine.
# Pattern: docker push to registries + scp checkpoints over SSH.
#
# Sanitised for public mirror — replace <private-registry-N> and
# <dev-root> with your own values before use.
#
# Prerequisites:
# - Spirit image built locally: `spirit-v1.1-cu128-py311`
# - Spirit-v1.5 + Qwen3-VL-4B checkpoints in ./spirit_ckpts/
# - Dev machine: ssh -p <DEV_PORT> root@<DEV_HOST>
# - Dev machine destination: <dev-root>
# ============================================================
set -euo pipefail

DEV_PORT="${DEV_PORT:-4163}"
DEV_USER="${DEV_USER:-root}"
DEV_HOST="${DEV_HOST:-127.0.0.1}"
DEV_ROOT="${DEV_ROOT:-/workspace/jfs/ro_planning}"

CKPT_SRC="${CKPT_SRC:-./spirit_ckpts}"
IMAGE_TAG="${IMAGE_TAG:-spirit-v1.1-cu128-py311}"

# Replace with your own registries (e.g. harbor/docker-hub/ecr).
# Multiple entries → image is tagged + pushed to all in parallel.
REGISTRIES=(
    "<private-registry-1>/planningmodel"
    "<private-registry-2>/planningmodel"
    "<private-registry-3>/planning"
)

step() { echo ""; echo "=============================================="; echo "  $1"; echo "=============================================="; }

# ------------------------------------------------------------
# Step 1: Push docker image to all registries
# ------------------------------------------------------------
step "Step 1: Tag + push Spirit image to registries"

for reg in "${REGISTRIES[@]}"; do
    dst="$reg:$IMAGE_TAG"
    echo "  Tagging: $dst"
    docker tag "$IMAGE_TAG:latest" "$dst"
done

for reg in "${REGISTRIES[@]}"; do
    dst="$reg:$IMAGE_TAG"
    echo "  Pushing: $dst"
    docker push "$dst" &
done
wait
echo "All registries pushed in parallel."

# ------------------------------------------------------------
# Step 2: scp ckpts to dev machine
# ------------------------------------------------------------
step "Step 2: scp Spirit-v1.5 + Qwen3-VL-4B to dev machine"

ssh -p "$DEV_PORT" "$DEV_USER@$DEV_HOST" "mkdir -p $DEV_ROOT/models"

echo "  Copying Spirit-v1.5 (21 GB)..."
scp -P "$DEV_PORT" -r \
    "$CKPT_SRC/Spirit-v1.5" \
    "$DEV_USER@$DEV_HOST:$DEV_ROOT/models/" &
SPIRIT_PID=$!

echo "  Copying Qwen3-VL-4B-Instruct (8 GB)..."
scp -P "$DEV_PORT" -r \
    "$CKPT_SRC/Qwen3-VL-4B-Instruct" \
    "$DEV_USER@$DEV_HOST:$DEV_ROOT/models/" &
QWEN_PID=$!

wait $SPIRIT_PID && echo "  Spirit done"
wait $QWEN_PID && echo "  Qwen done"

# ------------------------------------------------------------
# Step 3: Create patched config on dev
# ------------------------------------------------------------
step "Step 3: Build Spirit-v1.5-patched with local-path backbone on dev"

ssh -p "$DEV_PORT" "$DEV_USER@$DEV_HOST" bash -s <<EOF
set -e
DEV_ROOT=$DEV_ROOT
SRC=\$DEV_ROOT/models/Spirit-v1.5
DST=\$DEV_ROOT/models/Spirit-v1.5-patched

mkdir -p \$DST
ln -sf \$SRC/model.safetensors \$DST/model.safetensors
ln -sf \$SRC/README.md \$DST/README.md 2>/dev/null || true

python3 - <<PY
import json
src = "\$SRC/config.json"
dst = "\$DST/config.json"
with open(src) as f:
    d = json.load(f)
d["backbone"] = "/workspace/models/Qwen3-VL-4B-Instruct"  # path inside container
with open(dst, "w") as f:
    json.dump(d, f, indent=2)
print(f"patched config written: {dst}")
PY

ls -la \$DST/
EOF

step "Done."
echo ""
echo "Next on dev machine:"
echo "  docker pull <registry>/planningmodel:$IMAGE_TAG"
echo ""
echo "  ssh -p $DEV_PORT $DEV_USER@$DEV_HOST"
echo "  cd $DEV_ROOT"
echo "  docker run -it --rm --gpus all \\"
echo "    -v \$PWD/models:/workspace/models \\"
echo "    -v \$PWD/code/spirit_adapter:/workspace/spirit_adapter \\"
echo "    -e TRANSFORMERS_OFFLINE=1 \\"
echo "    $IMAGE_TAG \\"
echo "    python /workspace/spirit_adapter/smoke_test.py \\"
echo "    --spirit_ckpt /workspace/models/Spirit-v1.5-patched"
