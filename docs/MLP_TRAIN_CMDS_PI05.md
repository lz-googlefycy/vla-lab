# MLP 任务清单 — pi0.5 (Physical Intelligence) — **READY TO RUN**

> 验证状态（2026-05-14 15:30）：
> ✅ Dev pod 4321 上端到端 PASS：load 6.8GB ckpt → select_action [-0.99, 1.00] → policy_logp_with_ref → 22GB GPU peak / 96GB
> ✅ Pi0.5 LIBERO PyTorch ckpt 在 dev pod 共享盘
> ✅ Pi05Adapter 不依赖 lerobot.common（绕开 broken dep）
>
> 现在你可以起 MLP 任务了。

## 📦 镜像 + 路径

```
镜像: evad-ml-cn-bjdb-1-vecps.cr.cloud.vnet.com/infra_public/planning:vla-lab-v1.0-cu128-py311
路径: /e2e-data/users/liuzhi7/...（MLP pod 原生挂载）
```

## 🏗️ Pod 启动后必跑（一次）

```bash
# 1. 一键装 pi0.5 依赖（基础 transformers/sentencepiece/flax/jax/orbax 等）
bash /e2e-data/users/liuzhi7/.persist/install-pi05-deps.sh

# 2. 装 pi0.5 必需的精确版本（jax 0.5.3, flax 0.10.2, orbax 0.11.13 等）
WHL=/e2e-data/users/liuzhi7/.persist/wheels
pip install --force-reinstall --no-cache-dir --no-deps \
    $WHL/jax-0.5.3-py3-none-any.whl \
    $WHL/jaxlib-0.5.3-cp311-cp311-manylinux2014_x86_64.whl \
    $WHL/flax-0.10.2-py3-none-any.whl \
    $WHL/orbax_checkpoint-0.11.13-py3-none-any.whl \
    $WHL/ml_dtypes-0.4.1-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl \
    $WHL/tensorstore-0.1.74-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl \
    $WHL/numpy-1.26.4-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
```

## 🌳 通用环境变量

```bash
export PYTHONPATH=/e2e-data/users/liuzhi7/ro_planning/code:/e2e-data/users/liuzhi7/vla_workspace/openpi/src:/e2e-data/users/liuzhi7/vla_workspace/openpi/packages/openpi-client/src
export PALIGEMMA_TOKENIZER_PATH=/e2e-data/users/liuzhi7/vla_workspace/paligemma_tokenizer.model
export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

---

## 📋 任务 1-4：pi0.5 SFT eval × 4 suite（验证 paper baseline）

每条一个单卡任务，预期 ~1.5h 一个 cell。

### P-1: pi0.5 spatial SFT eval

```bash
bash -c '
set -euo pipefail
export PYTHONPATH=/e2e-data/users/liuzhi7/ro_planning/code:/e2e-data/users/liuzhi7/vla_workspace/openpi/src:/e2e-data/users/liuzhi7/vla_workspace/openpi/packages/openpi-client/src
export PALIGEMMA_TOKENIZER_PATH=/e2e-data/users/liuzhi7/vla_workspace/paligemma_tokenizer.model
export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1

cd /e2e-data/users/liuzhi7/ro_planning/code
python -m post_training.eval_libero \
    --base pi05 \
    --base_ckpt /e2e-data/users/liuzhi7/vla_workspace/models/pi05_libero_pytorch \
    --suites libero_spatial \
    --n_tasks 10 --n_trials_per_task 5 --seeds 42 \
    --output_dir /e2e-data/users/liuzhi7/vla_workspace/output/mlp_pi05_sft_spatial_eval
'
```

### P-2: pi0.5 spatial → **object**

```bash
bash -c '
set -euo pipefail
bash /e2e-data/users/liuzhi7/.persist/install-pi05-deps.sh
export PYTHONPATH=/e2e-data/users/liuzhi7/ro_planning/code:/e2e-data/users/liuzhi7/vla_workspace/openpi/src:/e2e-data/users/liuzhi7/vla_workspace/openpi/packages/openpi-client/src
export PALIGEMMA_TOKENIZER_PATH=/e2e-data/users/liuzhi7/vla_workspace/paligemma_tokenizer.model
export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1

cd /e2e-data/users/liuzhi7/ro_planning/code
python -m post_training.eval_libero \
    --base pi05 \
    --base_ckpt /e2e-data/users/liuzhi7/vla_workspace/models/pi05_libero_pytorch \
    --suites libero_object \
    --n_tasks 10 --n_trials_per_task 5 --seeds 42 \
    --output_dir /e2e-data/users/liuzhi7/vla_workspace/output/mlp_pi05_sft_object_eval
'
```

### P-3: pi0.5 → **goal**

```bash
bash -c '
set -euo pipefail
bash /e2e-data/users/liuzhi7/.persist/install-pi05-deps.sh
export PYTHONPATH=/e2e-data/users/liuzhi7/ro_planning/code:/e2e-data/users/liuzhi7/vla_workspace/openpi/src:/e2e-data/users/liuzhi7/vla_workspace/openpi/packages/openpi-client/src
export PALIGEMMA_TOKENIZER_PATH=/e2e-data/users/liuzhi7/vla_workspace/paligemma_tokenizer.model
export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1

cd /e2e-data/users/liuzhi7/ro_planning/code
python -m post_training.eval_libero \
    --base pi05 \
    --base_ckpt /e2e-data/users/liuzhi7/vla_workspace/models/pi05_libero_pytorch \
    --suites libero_goal \
    --n_tasks 10 --n_trials_per_task 5 --seeds 42 \
    --output_dir /e2e-data/users/liuzhi7/vla_workspace/output/mlp_pi05_sft_goal_eval
'
```

### P-4: pi0.5 → **long10**（注意 `libero_10`，不是 `libero_long10`）

```bash
bash -c '
set -euo pipefail
bash /e2e-data/users/liuzhi7/.persist/install-pi05-deps.sh
export PYTHONPATH=/e2e-data/users/liuzhi7/ro_planning/code:/e2e-data/users/liuzhi7/vla_workspace/openpi/src:/e2e-data/users/liuzhi7/vla_workspace/openpi/packages/openpi-client/src
export PALIGEMMA_TOKENIZER_PATH=/e2e-data/users/liuzhi7/vla_workspace/paligemma_tokenizer.model
export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1

cd /e2e-data/users/liuzhi7/ro_planning/code
python -m post_training.eval_libero \
    --base pi05 \
    --base_ckpt /e2e-data/users/liuzhi7/vla_workspace/models/pi05_libero_pytorch \
    --suites libero_10 \
    --n_tasks 10 --n_trials_per_task 5 --seeds 42 \
    --output_dir /e2e-data/users/liuzhi7/vla_workspace/output/mlp_pi05_sft_long10_eval
'
```

期望（paper §5.3）: spatial 98.8 / object 98.2 / goal 98.0 / long10 92.4

---

## 📋 任务 5-8：pi0.5 DPO 收 pair + train + eval

每个 suite 一个完整任务（rollout + train + eval 串行，单卡 ~10h）：

```bash
bash -c '
set -euo pipefail
export PYTHONPATH=/e2e-data/users/liuzhi7/ro_planning/code:/e2e-data/users/liuzhi7/vla_workspace/openpi/src:/e2e-data/users/liuzhi7/vla_workspace/openpi/packages/openpi-client/src
export PALIGEMMA_TOKENIZER_PATH=/e2e-data/users/liuzhi7/vla_workspace/paligemma_tokenizer.model
export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

SUITE=spatial   # spatial / object / goal / long10
[ "$SUITE" = "long10" ] && SUITE_FULL=libero_10 || SUITE_FULL=libero_$SUITE
cd /e2e-data/users/liuzhi7/ro_planning/code

# Stage 1: rollout pair (~3-4h)
python -m post_training.rollout \
    --base pi05 \
    --base_ckpt /e2e-data/users/liuzhi7/vla_workspace/models/pi05_libero_pytorch \
    --suite $SUITE_FULL \
    --n_tasks 10 --n_inits_per_task 5 --n_candidates_per_init 4 \
    --output_path /e2e-data/users/liuzhi7/vla_workspace/datasets/pi05_${SUITE}_pairs.pt

# Stage 2: DPO train (~30 min)
python -m post_training.train_dpo \
    --base pi05 \
    --base_ckpt /e2e-data/users/liuzhi7/vla_workspace/models/pi05_libero_pytorch \
    --suite $SUITE \
    --pairs_file /e2e-data/users/liuzhi7/vla_workspace/datasets/pi05_${SUITE}_pairs.pt \
    --output_dir /e2e-data/users/liuzhi7/vla_workspace/output/mlp_pi05_dpo_$SUITE \
    --batch_size 1 --max_steps 500 --warmup 100 \
    --log_every 10 --save_every 500 \
    --lora_r 32 --lora_alpha 64 --lr 5e-5 --beta 0.1

# Stage 3: eval (~1.5h)
python -m post_training.eval_libero \
    --base pi05 \
    --base_ckpt /e2e-data/users/liuzhi7/vla_workspace/models/pi05_libero_pytorch \
    --lora_ckpt /e2e-data/users/liuzhi7/vla_workspace/output/mlp_pi05_dpo_$SUITE/checkpoint-500.pt \
    --lora_r 32 --lora_alpha 64 \
    --suites $SUITE_FULL \
    --n_tasks 10 --n_trials_per_task 5 --seeds 42 \
    --output_dir /e2e-data/users/liuzhi7/vla_workspace/output/mlp_pi05_dpo_${SUITE}_eval
'
```

## 📋 任务 9-12：pi0.5 GRPO × 4 suite

```bash
bash -c '
set -euo pipefail
export PYTHONPATH=/e2e-data/users/liuzhi7/ro_planning/code:/e2e-data/users/liuzhi7/vla_workspace/openpi/src:/e2e-data/users/liuzhi7/vla_workspace/openpi/packages/openpi-client/src
export PALIGEMMA_TOKENIZER_PATH=/e2e-data/users/liuzhi7/vla_workspace/paligemma_tokenizer.model
export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

SUITE=spatial   # spatial / object / goal / long10
[ "$SUITE" = "long10" ] && SUITE_FULL=libero_10 || SUITE_FULL=libero_$SUITE
cd /e2e-data/users/liuzhi7/ro_planning/code

# Train (~10-15h)
python -m post_training.train_grpo \
    --base pi05 \
    --base_ckpt /e2e-data/users/liuzhi7/vla_workspace/models/pi05_libero_pytorch \
    --suite $SUITE \
    --output_dir /e2e-data/users/liuzhi7/vla_workspace/output/mlp_pi05_grpo_$SUITE \
    --n_tasks 10 --n_inits_per_task 3 --group_size 4 \
    --max_chunk_len 10 \
    --max_steps 500 --warmup 50 \
    --log_every 10 --save_every 500 \
    --lora_r 16 --lr 3e-5 \
    --beta 0.1 --epsilon 0.2

# Eval (~1.5h)
python -m post_training.eval_libero \
    --base pi05 \
    --base_ckpt /e2e-data/users/liuzhi7/vla_workspace/models/pi05_libero_pytorch \
    --lora_ckpt /e2e-data/users/liuzhi7/vla_workspace/output/mlp_pi05_grpo_$SUITE/checkpoint-500.pt \
    --lora_r 16 --lora_alpha 32 \
    --suites $SUITE_FULL \
    --n_tasks 10 --n_trials_per_task 5 --seeds 42 \
    --output_dir /e2e-data/users/liuzhi7/vla_workspace/output/mlp_pi05_grpo_${SUITE}_eval
'
```

> `--max_chunk_len 10` 因为 pi0.5 action_horizon=10（不像 OpenVLA 220），不需要截断；写明确以防 default 值变。

---

## 推荐起任务顺序

1. **先 P-1 spatial SFT eval**（1.5h）：验证 ckpt + adapter 在 MLP pod 上能跑 → 拿到 paper baseline 数字
2. P-2/3/4 SFT eval 并行（3 个单卡任务，~1.5h 全完）
3. **再起 P-5 spatial DPO**（10h）：rollout + train + eval 串行
4. P-6/7/8 DPO 并行（3 卡 ~10h）
5. **再起 P-9~12 GRPO**（4 卡，~12h）

**总耗时**：~24-30h（4 卡并行），明早全部完成。

## 🚨 第一个任务起来后**立刻验证**

P-1 spatial SFT eval 是最快的 sanity check（1.5h 内出结果，预期 ≥ 90% success），验证：
- `bash install-pi05-deps.sh` + wheel 装通了
- `pi05_libero_pytorch` ckpt 共享盘可读
- Pi05Adapter 在 MLP pod 96GB 上跑得通

如果 P-1 跑出来 < 50%，停下，肯定哪里崩了 — 把 stdout 给我看。
