# 8×H20 世纪互联起任务清单（2026-05-12 02:00 版）

cloudml 这边正在用 1 张 H20 跑 Long10 rollout（~5h），不占世纪互联。
你拿到 8×H20 pod 后，按下面复制粘贴即可。

## 前提

- ssh 进 pod，pod 看得到 8 张 H20 96GB
- Pod 能连外网（拉镜像 + HF ckpt + GitHub）
- Overlay ≥ 50 GB（装镜像 22.4 GB + ckpts ~60 GB）

## Step 0: 准备（一次性，~30 min）

```bash
# 1. 拉镜像
docker pull evad-ml-cn-bjdb-1-vecps.cr.cloud.vnet.com/infra_public/planning:vla-lab-v1.0-cu128-py311

# 2. Clone 代码
cd /e2e-data/users/liuzhi7
git clone https://github.com/lz-googlefycy/vla-lab.git
git clone https://github.com/openvla/openvla.git

# 3. 拉 4 个 OpenVLA LIBERO ckpt (~60 GB total, ~10 min)
mkdir -p models
for S in spatial object goal 10; do
  huggingface-cli download openvla/openvla-7b-finetuned-libero-$S \
    --local-dir models/openvla-7b-finetuned-libero-$S &
done
wait

# 4. 关键：patch 所有 ckpt（解决 transformers>=4.50 generate 退化）
docker run --rm \
    -v /e2e-data/users/liuzhi7/models:/models \
    -v /e2e-data/users/liuzhi7/vla-lab/code:/code \
    -e PYTHONPATH=/code \
    evad-ml-cn-bjdb-1-vecps.cr.cloud.vnet.com/infra_public/planning:vla-lab-v1.0-cu128-py311 \
    python /code/post_training/patch_modeling_prismatic.py --model-root /models
```

## Step 1: 单卡 sanity（5 min）

```bash
docker run --rm --gpus '"device=0"' \
    -v /e2e-data/users/liuzhi7/openvla:/openvla:ro \
    -v /e2e-data/users/liuzhi7/models/openvla-7b-finetuned-libero-spatial:/model:ro \
    -v /e2e-data/users/liuzhi7/vla-lab/code:/code:ro \
    -v /e2e-data/users/liuzhi7/output_test:/output \
    -e PYTHONPATH=/code:/openvla \
    -e MUJOCO_GL=osmesa -e PYOPENGL_PLATFORM=osmesa \
    -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e PYTHONUNBUFFERED=1 \
    evad-ml-cn-bjdb-1-vecps.cr.cloud.vnet.com/infra_public/planning:vla-lab-v1.0-cu128-py311 \
    python -m post_training.eval_libero \
        --base openvla --base_ckpt /model \
        --suites libero_spatial \
        --n_tasks 1 --n_trials_per_task 5 --seeds 42 \
        --output_dir /output
```
期望：1/1 task 4-5/5 success = 80-100%。

## Step 2: 并行 4 个 DPO cells（8 GPU, ~1.5h 全完）

从 cloudml 拷贝已收集的 pair 数据：

```bash
mkdir -p /e2e-data/users/liuzhi7/datasets
for S in spatial goal; do
  scp -P 4163 root@<cloudml_ip>:/ad-alg/planning-users/liuzhi7/ro_planning/output/h20_rollout_$S/${S}_pairs.pt \
      /e2e-data/users/liuzhi7/datasets/openvla_${S}_pairs.pt
done
# Object 和 Long10 的 pair 数据现在 cloudml 也没，或在跑中（long10）
# Object pair 要新收集，或从 cloudml 把 old 拷过来
```

**4 个 DPO 训练并行**（bs=1，每个占 1 张 GPU，~110 GB / 96 GB... 不行，H20 只有 96GB）：

⚠️ **注意**：世纪互联 H20 是 96GB，不是 cloudml 的 144GB。 DPO bs=1 在 Spatial（T=220）上 cloudml 占 103 GB → **96 GB H20 会 OOM**。

需要进一步压缩。两个选择：
- 用 `--max_chunk_len 180`（更短，显存降到 ~70 GB）
- 切 fp16（可能精度掉，不稳）

建议先试 `--max_chunk_len 180`：

```bash
for I in 0 1 2 3; do
  SUITES=(spatial object goal long10)
  S=${SUITES[$I]}
  [[ $S == "long10" ]] && CKPT_DIR=openvla-7b-finetuned-libero-10 || CKPT_DIR=openvla-7b-finetuned-libero-$S
  nohup docker run --rm --gpus "\"device=$I\"" \
      -v /e2e-data/users/liuzhi7/openvla:/openvla:ro \
      -v /e2e-data/users/liuzhi7/models:/models:ro \
      -v /e2e-data/users/liuzhi7/vla-lab/code:/code:ro \
      -v /e2e-data/users/liuzhi7/datasets:/datasets:ro \
      -v /e2e-data/users/liuzhi7/output/dpo_$S:/output \
      -e PYTHONPATH=/code:/openvla \
      -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e PYTHONUNBUFFERED=1 \
      -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      evad-ml-cn-bjdb-1-vecps.cr.cloud.vnet.com/infra_public/planning:vla-lab-v1.0-cu128-py311 \
      python -m post_training.train_dpo \
          --base openvla --base_ckpt /models/$CKPT_DIR \
          --suite $S \
          --pairs_file /datasets/openvla_${S}_pairs.pt \
          --output_dir /output \
          --max_chunk_len 180 \
          --batch_size 1 --max_steps 500 --warmup 100 \
          --log_every 10 --save_every 500 \
          --lora_r 32 --lr 5e-5 --beta 0.1 \
      > /e2e-data/users/liuzhi7/logs/dpo_$S.log 2>&1 &
done
wait
```

## Step 3: 并行 4 个 eval（8 GPU，~1h 全完）

```bash
for I in 0 1 2 3; do
  SUITES=(spatial object goal long10)
  FULL=(libero_spatial libero_object libero_goal libero_10)
  S=${SUITES[$I]}; F=${FULL[$I]}
  [[ $S == "long10" ]] && CKPT_DIR=openvla-7b-finetuned-libero-10 || CKPT_DIR=openvla-7b-finetuned-libero-$S
  CKPT=$(ls -t /e2e-data/users/liuzhi7/output/dpo_$S/checkpoint-*.pt | head -1)
  nohup docker run --rm --gpus "\"device=$I\"" \
      -v /e2e-data/users/liuzhi7/openvla:/openvla:ro \
      -v /e2e-data/users/liuzhi7/models:/models:ro \
      -v /e2e-data/users/liuzhi7/vla-lab/code:/code:ro \
      -v /e2e-data/users/liuzhi7/output:/output \
      -e PYTHONPATH=/code:/openvla -e MUJOCO_GL=osmesa -e PYOPENGL_PLATFORM=osmesa \
      -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e PYTHONUNBUFFERED=1 \
      evad-ml-cn-bjdb-1-vecps.cr.cloud.vnet.com/infra_public/planning:vla-lab-v1.0-cu128-py311 \
      python -m post_training.eval_libero \
          --base openvla --base_ckpt /models/$CKPT_DIR \
          --lora_ckpt $CKPT \
          --suites $F \
          --n_tasks 10 --n_trials_per_task 5 --seeds 42 \
          --output_dir /output/dpo_${S}_eval \
      > /e2e-data/users/liuzhi7/logs/eval_$S.log 2>&1 &
done
wait
```

## Step 4: 回传结果到 cloudml

```bash
rsync -az /e2e-data/users/liuzhi7/output/dpo_*_eval \
    root@<cloudml_ip>:/ad-alg/planning-users/liuzhi7/ro_planning/output/
```

## 总时长估算（8×H20 96GB）

- Step 0: 30 min
- Step 1: 5 min
- Step 2 (4 并行 DPO): ~20 min
- Step 3 (4 并行 eval): ~50 min
- Step 4: 几分钟

**~2 小时**完成 4 个 OpenVLA × DPO × 4-suite，paper §4.2 整个 OpenVLA 行就满了。

## 已经在 cloudml 跑的不用再跑

- OpenVLA SFT 全 4 suite baseline ✅ (用 v1.5 结果)
- OpenVLA + DPO Spatial ✅ 78%
- OpenVLA + DPO Object ✅ 62%
- OpenVLA + DPO Goal ✅ 74%
- **OpenVLA + DPO Long10 正在 cloudml 跑**（ETA 06:00）

所以 8×H20 上**只需要**：
- GRPO 4 个 cells（online rollout 要 ~8-10h each, 但 8 GPU 可以 4 个并行，~10h 全完）
- Spirit v1.5 SFT 训练（LIBERO 没 ckpt, ~10-15h）
- π0.5 的路径决策
