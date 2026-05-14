# 🚀 在笔记本上跑 pi0.5 JAX → PyTorch 转换

> 大约耗时：环境装 ~5min + ckpt 下载 ~30min（如果直接走 GCS）+ conversion ~5-10min
> 显存需求：转换需要 **24GB+ GPU**（在 GPU 上 instantiate PI0Pytorch + 加载 JAX 权重）
>
> 如果你笔记本没 24GB 卡，可以加 `--cpu_only` 或用 `CUDA_VISIBLE_DEVICES=""` 在 CPU 跑（慢一些，但 conversion 是一次性的）。

## 步骤 1：装环境

```bash
# 假设笔记本是 Linux + conda（如是 mac M 系列芯片，jax 0.5.3 需特殊处理）
conda create -n pi05-convert python=3.11 -y
source activate pi05-convert  # 或 conda activate pi05-convert

pip install \
  "torch>=2.4" \
  "jax==0.5.3" "jaxlib==0.5.3" \
  "flax==0.10.2" \
  "orbax-checkpoint==0.11.13" \
  "transformers==4.53.2" \
  "jaxtyping==0.2.36" \
  "ml_dtypes==0.4.1" "tensorstore==0.1.74" \
  safetensors numpy einops sentencepiece beartype tyro \
  augmax chex tqdm tqdm-loggable filelock fsspec ml-collections \
  numpydantic treescope
```

> 如果你笔记本有 NVIDIA GPU 并要走 cuda，把 `jax==0.5.3` 改成 `"jax[cuda12]==0.5.3"`。

## 步骤 2：拿代码 + ckpt + tokenizer

### 选项 A（推荐）：从本机 SSH 拉

本机 IP: `10.189.148.41`（同一个 ubuntu/lan，应该 ssh 通）

```bash
mkdir -p ~/pi05_work
cd ~/pi05_work

# 1. openpi 源码 (~50MB)
scp -r ubuntu@10.189.148.41:/home/ubuntu/openpi .
# (如果 ssh 公钥没设，把 ubuntu 改成你的用户名 + 用密码)

# 2. PaliGemma tokenizer (4.2MB)
scp ubuntu@10.189.148.41:/home/ubuntu/pi_assets/tokenizer/paligemma_tokenizer.model .

# 3. JAX ckpt (12GB, ~30-40min over LAN)
scp -r ubuntu@10.189.148.41:/home/ubuntu/pi_assets/pi05_libero .
```

### 选项 B：直接走公网 GCS（笔记本能直连 storage.googleapis.com 才行）

```bash
# 装 gsutil
curl https://sdk.cloud.google.com | bash && exec -l $SHELL
gcloud auth login --no-launch-browser  # 任意 Google 账号即可，bucket 公开

# 下载（~10-15min over public internet）
mkdir -p ~/pi05_work/openpi/pi05_libero
cd ~/pi05_work/openpi/pi05_libero
gsutil -m cp -r gs://openpi-assets/checkpoints/pi05_libero/* .

# tokenizer
mkdir -p ~/pi05_work/tokenizer
cd ~/pi05_work/tokenizer
gsutil cp gs://big_vision/paligemma_tokenizer.model .

# openpi src
cd ~/pi05_work
git clone https://github.com/Physical-Intelligence/openpi.git openpi-src
```

## 步骤 3：跑 conversion

```bash
cd ~/pi05_work/openpi  # 或 openpi-src

# Settings
export JAX_PLATFORMS=cpu       # 转换用 CPU 即可，省显存（jax 不上 GPU）
                               # 如果想用 GPU 加速，删除这行
mkdir -p ~/pi05_work/pi05_libero_pytorch

PYTHONPATH=$PWD/src:$PWD/packages/openpi-client/src \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  python examples/convert_jax_model_to_pytorch.py \
    --config_name pi05_libero \
    --checkpoint_dir ~/pi05_work/pi05_libero \
    --output_path ~/pi05_work/pi05_libero_pytorch
```

期望输出（最后几行）：
```
Saving PyTorch checkpoint...
Successfully converted JAX checkpoint to PyTorch format
Output: ~/pi05_work/pi05_libero_pytorch/model.safetensors
```

输出文件大小约 **7GB**（bf16 safetensors）。

## 步骤 4：把 PyTorch ckpt 推回 dev pod 共享盘

```bash
# 直接 scp 到 dev pod（通过本机的 4321 reverse tunnel）
ssh ubuntu@10.189.148.41 "ssh -p 4321 root@127.0.0.1 'mkdir -p /e2e-data/users/liuzhi7/vla_workspace/models/pi05_libero_pytorch'"

# Tar | ssh push（最快）
cd ~/pi05_work
tar cf - pi05_libero_pytorch/ | \
    ssh ubuntu@10.189.148.41 \
    "ssh -p 4321 root@127.0.0.1 'cd /e2e-data/users/liuzhi7/vla_workspace/models && rm -rf pi05_libero_pytorch && tar xf -'"

# 等 tar push 完后，验证：
ssh ubuntu@10.189.148.41 "ssh -p 4321 root@127.0.0.1 'du -sh /e2e-data/users/liuzhi7/vla_workspace/models/pi05_libero_pytorch'"
# 期望：~7GB
```

## 步骤 5：完成后告诉我，我接着跑 sanity test + 给你 MLP 命令

把 `pi05_libero_pytorch/` 推到 dev pod 共享盘后告诉我 "pi05 ckpt ready"，
我会在 dev pod 上跑：
1. 1-task LIBERO eval（验证 ckpt + adapter end-to-end）
2. 3-step GRPO smoke（K=4 都行，96GB 充裕）
3. 输出 paper-grade MLP 命令清单（4 SFT-eval × 4 DPO × 4 GRPO = 12 个单卡任务）

---

## 如果你笔记本是 Mac (M 系列)

`jax==0.5.3 + jaxlib==0.5.3` 在 Apple Silicon 上需要特殊处理。两条路径：

A. **conda 用 jax-metal**：
```bash
pip install jax-metal "jax==0.5.3" "jaxlib==0.5.3"
```
（cpu mode 跑就行）

B. **不用 mac，找个 Linux 机器**（你台式机？）。

## 如果完全跑不通

回报给我 `--inspect_only` 输出 + 错误日志，我重新规划。

```bash
# 这能跑表示 ckpt 至少能 load
PYTHONPATH=$PWD/src:$PWD/packages/openpi-client/src \
  python examples/convert_jax_model_to_pytorch.py \
    --config_name pi05_libero \
    --checkpoint_dir ~/pi05_work/pi05_libero \
    --inspect_only
```

应该看到大量 JAX 参数 key 树状结构。
