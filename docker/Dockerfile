# ============================================================
# OpenVLA v1.0 - 继承 trajflow-moe-v1.4-cu118-py310
# 基础: torch 2.2.0 + cu118 + py3.10 + cuda dev toolkit + ninja
# 增量: OpenVLA + LIBERO + flash-attn
# ============================================================

FROM <registry>/planningmodel:trajflow-moe-v1.4-cu118-py310

ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

# ============================================================
# 0. 系统依赖 (LIBERO + mujoco 渲染)
# ============================================================
RUN apt-get update && apt-get install -y --no-install-recommends \
        libosmesa6-dev libglew-dev libgl1-mesa-glx libegl1-mesa libgles2-mesa \
        patchelf libglfw3 libglfw3-dev xvfb ffmpeg \
        git wget curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# ============================================================
# 1. 修正 tensorflow 版本到 OpenVLA 要求
#    base 是 TF 2.12.0 / TFDS 4.9.9, OpenVLA 要 2.15.0 / 4.9.3
# ============================================================
RUN pip uninstall -y tensorflow tensorflow-datasets tensorflow-estimator \
        tensorflow-probability tensorflow-addons tensorflow-metadata \
        tensorflow-io-gcs-filesystem || true \
    && pip install \
        tensorflow==2.15.0 \
        tensorflow-datasets==4.9.3 \
        tensorflow_graphics==2021.12.3

# ============================================================
# 2. OpenVLA 主干依赖
#    关键: 必须 pin torch / torchvision / torchaudio 版本，否则
#    peft 等会拖最新 torch 2.11+cu130 上来，破坏 cu118 环境
# ============================================================
RUN pip install \
        torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 \
        --index-url https://download.pytorch.org/whl/cu118 \
        --force-reinstall

# 安装 OpenVLA 依赖时用 constraints 文件锁住 torch 不被升级
RUN echo "torch==2.2.0+cu118" > /tmp/constraints.txt && \
    echo "torchvision==0.17.0+cu118" >> /tmp/constraints.txt && \
    echo "torchaudio==2.2.0+cu118" >> /tmp/constraints.txt && \
    pip uninstall -y accelerate huggingface_hub || true && \
    pip install -c /tmp/constraints.txt \
        transformers==4.40.1 \
        tokenizers==0.19.1 \
        peft==0.11.1 \
        timm==0.9.10 \
        sentencepiece==0.1.99 \
        accelerate==0.30.1 \
        huggingface_hub==0.23.5 \
        draccus==0.8.0 \
        einops \
        json-numpy \
        jsonlines \
        matplotlib \
        rich \
        wandb \
        bitsandbytes==0.43.1

# ============================================================
# 3. flash-attn 2.5.5 (从源码编译, 用 ninja 限并发避免 OOM)
#    base 已有 ninja + nvcc 11.8 + torch 2.2.0+cu118
#    --no-build-isolation 让它用容器内的 torch 2.2.0
# ============================================================
RUN pip install -c /tmp/constraints.txt packaging
RUN MAX_JOBS=4 pip install -c /tmp/constraints.txt --no-build-isolation flash-attn==2.5.5

# ============================================================
# 4. dlimp (OpenVLA 自家 fork)
# ============================================================
RUN pip install -c /tmp/constraints.txt "dlimp @ git+https://github.com/moojink/dlimp_openvla"

# ============================================================
# 5. LIBERO 评测依赖
# ============================================================
RUN pip install -c /tmp/constraints.txt \
        "imageio[ffmpeg]" \
        robosuite==1.4.1 \
        bddl \
        easydict \
        cloudpickle \
        gym

# ============================================================
# 6. clone LIBERO + editable install
# ============================================================
RUN cd /workspace && git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git \
    && cd LIBERO && pip install -c /tmp/constraints.txt -e . \
    && pip install -c /tmp/constraints.txt hydra-core

# ============================================================
# 7. COPY OpenVLA 源码 + editable install
# ============================================================
COPY openvla/ /workspace/openvla/
RUN cd /workspace/openvla && pip install -c /tmp/constraints.txt -e .

# ============================================================
# 8. 工作目录 + 国内镜像
# ============================================================
RUN mkdir -p /workspace/models /workspace/datasets /workspace/output

ENV HF_ENDPOINT=https://hf-mirror.com
ENV HF_HOME=/workspace/models/hf_cache
ENV TRANSFORMERS_CACHE=/workspace/models/hf_cache
ENV PYTHONUNBUFFERED=1

WORKDIR /workspace/openvla

CMD ["/bin/bash"]
