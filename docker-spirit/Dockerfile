# ============================================================
# Spirit v1.5 Inference Image
# Base:     pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel (torch 2.8.0 + cu12.8 + py3.10)
# Add:      transformers 4.57.1, diffusers 0.35.2, flash-attn 2.8.3
# Source:   Spirit-AI-Team/spirit-v1.5 (vendored)
#
# Usage:
#   docker build -t spirit-v1.0-cu128-py310 .
#   docker run --rm --gpus all -v <ckpt-dir>:/workspace/models \
#       spirit-v1.0-cu128-py310 bash
# ============================================================

FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV PYTHONUNBUFFERED=1

# ============================================================
# 0. System deps (LeRobot/SO-100 needs these for camera & render,
#    plus SSH tooling for reverse tunnel from pod to user desktop)
# ============================================================
RUN apt-get update && apt-get install -y --no-install-recommends \
        git wget curl ca-certificates ffmpeg \
        libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev \
        v4l-utils libv4l-dev \
        build-essential \
        openssh-server openssh-client autossh tmux \
    && rm -rf /var/lib/apt/lists/*

# ============================================================
# 0b. Bootstrap script to restore SSH keys + start reverse tunnel.
#     Keys live on JuiceFS (/ad-alg/.../ssh_backup/), not in image.
#     Usage after pod rebuild: bash /opt/vla-lab/bootstrap-ssh.sh
# ============================================================
COPY bootstrap-ssh.sh /opt/vla-lab/bootstrap-ssh.sh
RUN chmod +x /opt/vla-lab/bootstrap-ssh.sh

# ============================================================
# 1. Spirit v1.5 dependencies (pinned for reproducibility)
#    Note: torch/torchvision already pinned in base image (2.8.0 + 0.23.0)
# ============================================================
RUN pip install \
        transformers==4.57.1 \
        diffusers==0.35.2 \
        safetensors==0.5.3 \
        numpy==2.2.6 \
        pillow==10.4.0 \
        requests==2.32.5 \
        scipy==1.15.2 \
        huggingface_hub \
        einops

# ============================================================
# 2. flash-attn 2.8.3 (from source, ~10-15 min)
#    Requires torch==2.8.0 which is already in base image.
# ============================================================
RUN pip install packaging ninja
RUN MAX_JOBS=4 pip install flash-attn==2.8.3 --no-build-isolation

# ============================================================
# 3. LeRobot + robotics stack (for future XLeRobot integration)
# ============================================================
RUN pip install \
        lerobot \
        opencv-python \
        pyrealsense2 \
        pyserial \
        mujoco \
        gymnasium \
    || echo "Some packages optional, continuing"

# ============================================================
# 4. Training deps (optional, only needed for finetune)
# ============================================================
RUN pip install \
        opencv-python>=4.8.0 \
        wandb \
        tqdm \
        einops==0.8.1 \
    || true

# ============================================================
# 5. COPY Spirit v1.5 source
# ============================================================
COPY spirit-v1.5/ /workspace/spirit-v1.5/

# ============================================================
# 6. Environment defaults
# ============================================================
ENV HF_ENDPOINT=https://hf-mirror.com
ENV HF_HOME=/workspace/models/hf_cache
ENV TRANSFORMERS_CACHE=/workspace/models/hf_cache
ENV PYTHONPATH=/workspace/spirit-v1.5:${PYTHONPATH}

RUN mkdir -p /workspace/models /workspace/datasets /workspace/output

WORKDIR /workspace/spirit-v1.5

CMD ["/bin/bash"]
