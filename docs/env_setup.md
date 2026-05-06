# 环境与开发机使用说明

> 创建：2026-05-06

---

## 1. 镜像

### 镜像 tag
```
openvla-v1.0-cu118-py310
```

### 三仓库（任选一处拉，digest 一致）
```
<registry>/planningmodel:openvla-v1.0-cu118-py310
<registry-volc>/planningmodel:openvla-v1.0-cu118-py310
<registry-vnet>/planning:openvla-v1.0-cu118-py310
```

digest: `sha256:d97dfb2220c63e27cf9832c4bef0fea813301d46057c620e0a449dba019c4939`
size: 32.3 GB

### 镜像继承关系
```
nvidia/cuda:11.8 base
  └── trajflow-moe-v1.4-cu118-py310 (TrajFlow 主干, 22 GB)
       └── openvla-v1.0-cu118-py310 (我们, +10 GB)
```

### 镜像内核心包版本
| 包 | 版本 | 备注 |
|---|---|---|
| Python | 3.10.13 | |
| torch | 2.2.0+cu118 | |
| transformers | 4.40.1 | OpenVLA 严格要求 |
| peft | 0.11.1 | OpenVLA 严格要求 |
| tokenizers | 0.19.1 | |
| timm | 0.9.10 | |
| sentencepiece | 0.1.99 | |
| accelerate | 0.30.1 | |
| huggingface_hub | 0.23.5 | |
| flash-attn | 2.5.5 | 从源码编译 |
| bitsandbytes | 0.43.1 | 4-bit 量化 |
| robosuite | 1.4.1 | LIBERO 依赖 |
| LIBERO | 0.1.0 | git clone + editable |
| OpenVLA (prismatic) | 0.0.3 | git clone + editable |

---

## 2. 开发机访问

```bash
# SSH（端口可能变化，最近一次：<dev-port>）
ssh -p <dev-port> <dev-machine>

# GPU
NVIDIA H20-3e × 1, 144 GB HBM

# Python
/opt/conda/bin/python  # 注意：PATH 没自动设，要用全路径
```

---

## 3. 文件系统布局

| 路径 | 用途 | 容量 | 说明 |
|---|---|---|---|
| `/` (overlay) | root 盘 | 20 GB | **不要往这里写大文件** |
| `/nix` | nix store | 100 GB | 系统用 |
| `/workspace/jfs/` | **主工作目录** | JuiceFS 744T 可用 | ⭐ 数据/模型/输出全在这 |
| `/workspace/jfs` | TrajFlow 旧数据 | JuiceFS 4 PB 可用 | 备用 |
| `/workspace/jfs` | | JuiceFS 5.5 PB 可用 | 备用 |

### 主工作目录详细
```
/workspace/jfs/
├── code/                       # rsync 上来的源码
│   ├── Dockerfile
│   ├── smoke_test.py
│   ├── .dockerignore
│   └── openvla/                # OpenVLA 源码
├── models/openvla-7b/          # 15 GB OpenVLA-7B 权重
├── datasets/modified_libero_rlds/  # 9.6 GB LIBERO RLDS
│   ├── libero_spatial_no_noops/
│   ├── libero_object_no_noops/
│   ├── libero_goal_no_noops/
│   └── libero_10_no_noops/
├── output/                     # 训练 ckpt（创建后）
└── hf_cache/                   # HuggingFace cache
```

### 容器内挂载约定（已配软链）
```bash
ROOT=/workspace/jfs
ln -sf $ROOT/datasets /workspace/datasets
ln -sf $ROOT/models   /workspace/models
ln -sf $ROOT/output   /workspace/output
ln -sf $ROOT/hf_cache ~/.cache/huggingface
```

---

## 4. 启动训练流程

### 进容器（如果用 docker run 而非 k8s）
```bash
docker run -it --rm --gpus all --shm-size=32g \
  -v /workspace/jfs:/workspace/jfs \
  -e HF_HOME=/workspace/jfs/hf_cache \
  <registry>/planningmodel:openvla-v1.0-cu118-py310 \
  bash
```

### 当前开发机已是 k8s pod
直接 `ssh -p <dev-port> <dev-machine>` 进入即可，环境已是镜像内的环境。

---

## 5. 已知坑 / 修复

### Q1：libero import 失败 `MAPPING is empty`
**原因**：editable install 在 docker build 时 mapping 没生成
**修复**：开发机执行
```bash
echo '/workspace/LIBERO' > /opt/conda/lib/python3.10/site-packages/libero_local.pth
```

### Q2：LIBERO 首次启动卡在 input prompt
**原因**：`libero/libero/__init__.py` 启动时问 dataset path
**修复**：自动确认
```bash
echo 'N' | /opt/conda/bin/python -c 'import libero; from libero.libero import benchmark'
```
之后 `~/.libero/config.yaml` 会生成。

### Q3：4-bit 加载报 `.to is not supported for 4-bit`
**原因**：transformers 4.40 + accelerate >0.30 兼容
**修复**：`from_pretrained(..., device_map={"": 0})` 而非 `low_cpu_mem_usage=True`

### Q4：HF cache 占满 root 盘
**修复**：
```bash
ln -sf /workspace/jfs/hf_cache ~/.cache/huggingface
```

---

## 6. 镜像重建流程（如需）

```bash
# 本地构建机
cd ~/Trajflow_workspace/openvla-build
docker build --network=host -t openvla-v1.0-cu118-py310:latest .

# Tag + Push 三仓库
TAG=openvla-v1.0-cu118-py310
for repo in \
  <registry>/planningmodel \
  <registry-volc>/planningmodel \
  <registry-vnet>/planning; do
  docker tag openvla-v1.0-cu118-py310:latest $repo:$TAG
  docker push $repo:$TAG
done
```

---

## 7. SSH key / GitLab

```bash
# 配置 git identity
cd /workspace/ro_planning
git config --local user.name "刘志"
git config --local user.email "liuzhi7 (Independent)"

# 远端
git remote add origin <private-gitlab>/ro_planning.git

# 首次 push
git push --set-upstream origin main
```
