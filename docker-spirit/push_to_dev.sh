#!/bin/bash
# ============================================================
# Push Spirit v1.5 image + ckpts from local workstation to dev machine.
# Pattern: match what we did for OpenVLA (docker push to MICR +
# scp checkpoints over SSH).
#
# Prerequisites:
# - Spirit image built locally: `spirit-v1.0-cu128-py310`
# - Spirit-v1.5 + Qwen3-VL-4B checkpoints in /home/ubuntu/openvla_assets/spirit_ckpts/
# - Dev machine: ssh -p <DEV_PORT> root@127.0.0.1
# - Dev machine destination: /ad-alg/planning-users/liuzhi7/ro_planning/
# ============================================================
set -euo pipefail

DEV_PORT="${DEV_PORT:-4163}"
DEV_USER="${DEV_USER:-root}"
DEV_HOST="${DEV_HOST:-127.0.0.1}"
DEV_ROOT="${DEV_ROOT:-/ad-alg/planning-users/liuzhi7/ro_planning}"

CKPT_SRC="/home/ubuntu/openvla_assets/spirit_ckpts"
IMAGE_TAG="spirit-v1.0-cu128-py310"

REGISTRIES=(
    "micr.cloud.mioffice.cn/world-model-lyk/planningmodel"
    "test-lab-instance-cn-beijing.cr.volces.com/evad-infra-compute/planningmodel"
    "evad-ml-cn-bjdb-1-vecps.cr.cloud.vnet.com/infra_public/planning"
)

step() { echo ""; echo "=============================================="; echo "  $1"; echo "=============================================="; }

# ------------------------------------------------------------
# Step 1: Push docker image to all 3 registries (like we did for OpenVLA)
# ------------------------------------------------------------
step "Step 1: Tag + push Spirit image to 3 registries"

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

ssh -p "$DEV_PORT" "$DEV_USER@$DEV_HOST" bash -s <<'EOF'
set -e
DEV_ROOT=/ad-alg/planning-users/liuzhi7/ro_planning
SRC=$DEV_ROOT/models/Spirit-v1.5
DST=$DEV_ROOT/models/Spirit-v1.5-patched

mkdir -p $DST
ln -sf $SRC/model.safetensors $DST/model.safetensors
ln -sf $SRC/README.md $DST/README.md 2>/dev/null || true

# Patch config: backbone = local Qwen dir (absolute path on dev)
python3 - <<PY
import json
src = "$SRC/config.json"
dst = "$DST/config.json"
with open(src) as f:
    d = json.load(f)
d["backbone"] = "/workspace/models/Qwen3-VL-4B-Instruct"  # ← path inside container
# Keep device as cuda — on H20 it works without the cpu-load trick
with open(dst, "w") as f:
    json.dump(d, f, indent=2)
print(f"patched config written: {dst}")
PY

ls -la $DST/
EOF

step "Done."
echo ""
echo "Next on dev machine:"
echo "  docker pull <registry>/planningmodel:spirit-v1.0-cu128-py310"
echo ""
echo "  ssh -p $DEV_PORT $DEV_USER@$DEV_HOST"
echo "  cd $DEV_ROOT"
echo "  docker run -it --rm --gpus all \\"
echo "    -v \$PWD/models:/workspace/models \\"
echo "    -v \$PWD/code/spirit_adapter:/workspace/spirit_adapter \\"
echo "    -e TRANSFORMERS_OFFLINE=1 \\"
echo "    spirit-v1.0-cu128-py310 \\"
echo "    python /workspace/spirit_adapter/smoke_test.py \\"
echo "    --spirit_ckpt /workspace/models/Spirit-v1.5-patched"
