#!/bin/bash
# Build Spirit v1.5 inference image.
# Clones Spirit source next to the Dockerfile if needed.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ ! -d spirit-v1.5 ]; then
    echo "Cloning spirit-v1.5 into $(pwd)/spirit-v1.5"
    git clone --depth 1 https://github.com/Spirit-AI-Team/spirit-v1.5.git
fi

TAG="${TAG:-spirit-v1.0-cu128-py310}"
echo "Building $TAG ..."
docker build --network=host -t "$TAG" .
echo "Done. Try: docker run --rm --gpus all $TAG python -c 'import model; print(\"OK\")'"
