# MLP 训练命令清单 — pi0.5 (Physical Intelligence)

> 状态：dev pod 4321 上 PI0Pytorch 已验证 random-init forward + GRPO surrogate logp 全通。等 pi05_libero ckpt 转 PyTorch + push 到共享盘后就能跑真任务。

## 📦 镜像

```
evad-ml-cn-bjdb-1-vecps.cr.cloud.vnet.com/infra_public/planning:vla-lab-v1.0-cu128-py311
```

⚠️ **首次启动 pod 后必须 patch 镜像里的 transformers**（pi0.5 要求 4.53.2 + 自定义 paligemma/gemma/siglip）：

```bash
# Run once per pod (镜像里 transformers 是 4.57.1，我们要降到 4.53.2 + 替换 4 个文件)
pip install -q --no-cache-dir "transformers==4.53.2"
cp -r /e2e-data/users/liuzhi7/vla_workspace/openpi/src/openpi/models_pytorch/transformers_replace/* \
    /opt/conda/lib/python3.11/site-packages/transformers/

# Install missing pkgs (走 mirrors.ivolces.com 内网 pypi)
pip install -q --no-cache-dir sentencepiece einops safetensors numpydantic \
    chex jaxtyping tyro beartype treescope flax filelock

# Stub tqdm-loggable (mirror 没这个包，stub 一下足矣)
mkdir -p /opt/conda/lib/python3.11/site-packages/tqdm_loggable
echo "from tqdm.auto import *" > /opt/conda/lib/python3.11/site-packages/tqdm_loggable/auto.py
echo "from tqdm import auto" > /opt/conda/lib/python3.11/site-packages/tqdm_loggable/__init__.py
```

我会把这段封装成 `/e2e-data/users/liuzhi7/.persist/install-pi05-deps.sh`，pod 起来后跑一次即可。

## 🗂️ 路径（共享盘原生路径，挂 `/e2e-data/users/liuzhi7`）

| 资源 | 位置 |
|---|---|
| 代码（含 pi05 adapter） | `/e2e-data/users/liuzhi7/ro_planning/code/` |
| openpi 源码 | `/e2e-data/users/liuzhi7/vla_workspace/openpi/` |
| pi0.5 LIBERO PyTorch ckpt | `/e2e-data/users/liuzhi7/vla_workspace/models/pi05_libero_pytorch/` |
| PaliGemma tokenizer | `/e2e-data/users/liuzhi7/vla_workspace/paligemma_tokenizer.model` |

## 🎯 任务清单（4 suites × {SFT eval, DPO, GRPO}）

### 通用环境变量

```bash
export PYTHONPATH=/e2e-data/users/liuzhi7/ro_planning/code:/e2e-data/users/liuzhi7/vla_workspace/openpi/src
export PALIGEMMA_TOKENIZER_PATH=/e2e-data/users/liuzhi7/vla_workspace/paligemma_tokenizer.model
export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

### 任务 P-1 ~ P-4: pi0.5 SFT eval × 4 suite (paper baseline 验证)

每条命令一个 MLP 任务（单卡 H20 96GB），suite 替换：

```bash
SUITE=spatial   # spatial / object / goal / 10 (= long10)
SUITE_NAME=libero_$SUITE

bash -c "
$ENV_PRELUDE
cd /e2e-data/users/liuzhi7/ro_planning/code
python -m post_training.eval_libero \
    --base pi05 \
    --base_ckpt /e2e-data/users/liuzhi7/vla_workspace/models/pi05_libero_pytorch \
    --suites $SUITE_NAME \
    --n_tasks 10 --n_trials_per_task 5 --seeds 42 \
    --output_dir /e2e-data/users/liuzhi7/vla_workspace/output/mlp_pi05_sft_${SUITE}_eval
"
```

预期（Paper §4.2）: spatial 98.8 / object 98.2 / goal 98.0 / long10 92.4. 耗时 ~1-2h/cell.

### 任务 P-5 ~ P-8: pi0.5 + DPO × 4 suite

需要先收集 DPO pair（用 pi0.5 SFT rollout）。

**Step 1 (per suite)**: rollout pair 数据
```bash
SUITE=spatial
python -m post_training.rollout \
    --base pi05 \
    --base_ckpt /e2e-data/users/liuzhi7/vla_workspace/models/pi05_libero_pytorch \
    --suite libero_$SUITE \
    --n_tasks 10 --n_inits_per_task 5 --n_candidates_per_init 4 \
    --output_path /e2e-data/users/liuzhi7/vla_workspace/datasets/pi05_${SUITE}_pairs.pt
# ETA: ~3-4h/suite (pi0.5 是 flow matching, sample_actions 比 OpenVLA greedy 快约 30%)
```

**Step 2 (per suite)**: DPO train
```bash
python -m post_training.train_dpo \
    --base pi05 \
    --base_ckpt /e2e-data/users/liuzhi7/vla_workspace/models/pi05_libero_pytorch \
    --suite $SUITE \
    --pairs_file /e2e-data/users/liuzhi7/vla_workspace/datasets/pi05_${SUITE}_pairs.pt \
    --output_dir /e2e-data/users/liuzhi7/vla_workspace/output/mlp_pi05_dpo_${SUITE} \
    --batch_size 1 --max_steps 500 --warmup 100 \
    --log_every 10 --save_every 500 \
    --lora_r 32 --lora_alpha 64 --lr 5e-5 --beta 0.1
# pi0.5 是 3.62B (vs OpenVLA 7B), DPO 显存只要 ~30GB → bs=4 也能跑
```

**Step 3 (per suite)**: eval
```bash
python -m post_training.eval_libero \
    --base pi05 \
    --base_ckpt /e2e-data/users/liuzhi7/vla_workspace/models/pi05_libero_pytorch \
    --lora_ckpt /e2e-data/users/liuzhi7/vla_workspace/output/mlp_pi05_dpo_${SUITE}/checkpoint-500.pt \
    --suites libero_$SUITE \
    --n_tasks 10 --n_trials_per_task 5 --seeds 42 1337 2026 \
    --output_dir /e2e-data/users/liuzhi7/vla_workspace/output/mlp_pi05_dpo_${SUITE}_eval
```

### 任务 P-9 ~ P-12: pi0.5 + GRPO × 4 suite

Pi0.5 比 OpenVLA 小一半，GRPO K=4 都能跑：

```bash
SUITE=spatial
python -m post_training.train_grpo \
    --base pi05 \
    --base_ckpt /e2e-data/users/liuzhi7/vla_workspace/models/pi05_libero_pytorch \
    --suite $SUITE \
    --output_dir /e2e-data/users/liuzhi7/vla_workspace/output/mlp_pi05_grpo_$SUITE \
    --n_tasks 10 --n_inits_per_task 3 --group_size 4 \
    --max_steps 500 --warmup 50 \
    --log_every 10 --save_every 500 \
    --lora_r 16 --lr 3e-5 \
    --beta 0.1 --epsilon 0.2
```

## 推荐起任务顺序

1. **先 P-1 ~ P-4 SFT eval × 4 suite**（验证 ckpt 加载 + 跟 paper baseline 对齐，~6h，4 个并行）
2. **再 P-5 ~ P-8 DPO × 4 suite**（rollout 拿数据 + DPO 训练 + eval）
3. **再 P-9 ~ P-12 GRPO × 4 suite**

paper §4.2 第二行 (pi0.5) 全填满需要 12 个 MLP 单卡任务，**总耗时 ~30-40h（4 个并行 ~10-12h）**。

---

## sanity 测试（起任务前在 dev pod 上验证一次）

```bash
ssh -p 4321 root@127.0.0.1
bash /e2e-data/users/liuzhi7/.persist/install-pi05-deps.sh   # 安装依赖（如果还没装）

# 1. 1-task eval（5 trial, ~5min）
cd /e2e-data/users/liuzhi7/ro_planning/code
export PYTHONPATH=/e2e-data/users/liuzhi7/ro_planning/code:/e2e-data/users/liuzhi7/vla_workspace/openpi/src
export PALIGEMMA_TOKENIZER_PATH=/e2e-data/users/liuzhi7/vla_workspace/paligemma_tokenizer.model
export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1

python -m post_training.eval_libero \
    --base pi05 \
    --base_ckpt /e2e-data/users/liuzhi7/vla_workspace/models/pi05_libero_pytorch \
    --suites libero_spatial \
    --n_tasks 1 --n_trials_per_task 5 --seeds 42 \
    --output_dir /tmp/pi05_smoke_eval
```

期望：`task 0: 4-5/5 success` (pi0.5 在 spatial 上 paper 报 98.8%)

如果 task 0 跑通就放心起 4 suite × 5 trial paper-grade eval。
