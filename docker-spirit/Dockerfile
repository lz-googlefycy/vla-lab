# ============================================================
# Spirit v1.5 Unified Image (inference + sim + LoRA finetune)
#
# Base:     pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel
#           → CUDA 12.8 + cuDNN 9 + torch 2.8.0 + Python 3.11
# Adds:     transformers 4.57.1, diffusers 0.35.2, flash-attn 2.8.3
#           mani_skill 3.0.1, sapien 3.0.3, LIBERO (vendored)
#           Vulkan loader + mesa drivers (for SAPIEN headless render)
# Source:   Spirit-AI-Team/spirit-v1.5 (vendored under spirit-v1.5/)
#
# History:  v1.0 = infer only (no sim)
#           v1.0-sim = infer + sim (two images)
#           v1.1 = merged (this file) — one image, everything
#
# Usage:
#   docker build -t spirit-v1.1-cu128-py311 .
#   docker run --rm --gpus all -v <ckpt-dir>:/workspace/models \
#       spirit-v1.1-cu128-py311 bash
# ============================================================

FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV PYTHONUNBUFFERED=1

# ============================================================
# 0. System deps
#    - LeRobot/SO-100: opencv, v4l, ffmpeg
#    - SSH tooling:    reverse tunnel from pod to user desktop
#    - SAPIEN render:  vulkan loader + mesa drivers + GL libs
#    - osmesa fallback: for mujoco headless when vulkan unavailable
# ============================================================
RUN apt-get update && apt-get install -y --no-install-recommends \
        git wget curl ca-certificates ffmpeg \
        libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev \
        v4l-utils libv4l-dev \
        build-essential \
        openssh-server openssh-client autossh tmux \
        libvulkan1 libvulkan-dev mesa-vulkan-drivers vulkan-tools \
        libglu1-mesa \
        libxrandr2 libxinerama1 libxcursor1 libxi6 \
        x11-xserver-utils xauth \
        libosmesa6 libosmesa6-dev \
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
#    Note: torch/torchvision already pinned in base (2.8.0 + 0.23.0)
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
# 4. Training deps (only needed for LoRA finetune)
# ============================================================
RUN pip install \
        wandb \
        tqdm \
        einops==0.8.1 \
    || true

# ============================================================
# 5. Sim stack: Maniskill + SAPIEN + imageio
# ============================================================
RUN pip install \
        mani_skill==3.0.1 \
        sapien==3.0.3 \
        h5py==3.16.0 \
        matplotlib \
        imageio[ffmpeg] \
        lxml

# ============================================================
# 6. LIBERO (for "Spirit sees a robot scene" demo + closed-loop eval)
# ============================================================
RUN pip install \
        robosuite==1.4.1 \
        bddl \
        easydict \
        cloudpickle \
        gym \
        hydra-core
RUN cd /workspace && git clone --depth 1 https://github.com/Lifelong-Robot-Learning/LIBERO.git \
    && cd LIBERO && pip install -e . --no-deps

# libero has no __init__.py at top-level; force path inclusion via site-packages .pth
RUN echo '/workspace/LIBERO' > /opt/conda/lib/python3.11/site-packages/libero_local.pth

# Initialise LIBERO config non-interactively
RUN echo 'N' | /opt/conda/bin/python -c "import libero; from libero.libero import benchmark" || true

# ============================================================
# 7. COPY Spirit v1.5 source
# ============================================================
COPY spirit-v1.5/ /workspace/spirit-v1.5/

# ============================================================
# 8. Environment defaults
# ============================================================
ENV HF_ENDPOINT=https://hf-mirror.com
ENV HF_HOME=/workspace/models/hf_cache
ENV TRANSFORMERS_CACHE=/workspace/models/hf_cache
ENV PYTHONPATH=/workspace/spirit-v1.5:${PYTHONPATH}

# Sim defaults (headless):
#   - No X server → DISPLAY empty
#   - mujoco: prefer osmesa when EGL device absent
#   - SAPIEN: headless mode (no GUI window)
#
# Note: SAPIEN will auto-pick vulkan when available (H20 nvidia ICD).
# If vulkan fails, user can set VK_ICD_FILENAMES manually.
ENV DISPLAY=""
ENV MUJOCO_GL=osmesa
ENV PYOPENGL_PLATFORM=osmesa
ENV SAPIEN_HEADLESS=1

RUN mkdir -p /workspace/models /workspace/datasets /workspace/output

WORKDIR /workspace/spirit-v1.5

CMD ["/bin/bash"]
