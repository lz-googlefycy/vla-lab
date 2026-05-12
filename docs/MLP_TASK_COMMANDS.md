# MLP 单卡任务启动命令清单（每 cell 一条，世纪互联 8×H20 96GB）

## 📦 镜像地址（MLP 任务选这一个）

```
evad-ml-cn-bjdb-1-vecps.cr.cloud.vnet.com/infra_public/planning:vla-lab-v1.0-cu128-py311
```

digest: `sha256:4ac541a377810e6bd5af7620b21e8b5428103493afe7e0192a8c5929179f1d7d`

备选（同内容，任选一个能 pull 的）：
- `micr.cloud.mioffice.cn/world-model-lyk/planningmodel:vla-lab-v1.0-cu128-py311`
- `test-lab-instance-cn-beijing.cr.volces.com/evad-infra-compute/planningmodel:vla-lab-v1.0-cu128-py311`

---

## 🔧 挂载 / 前置配置（所有任务通用）

从 MLP 任务配置里挂载共享盘，把：
```
/e2e-data/users/liuzhi7  →  /workspace_data
```
（或者 MLP 默认已挂载，用你熟悉的写法即可）

所有任务的**共用环境变量**：
```
PYTHONPATH=/workspace_data/ro_planning/code:/workspace_data/vla_workspace/openvla
MUJOCO_GL=osmesa
PYOPENGL_PLATFORM=osmesa
HF_HUB_OFFLINE=1
TRANSFORMERS_OFFLINE=1
PYTHONUNBUFFERED=1
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

所有任务单卡：`nproc_per_node=1`, 或 MLP UI 选 1 GPU 就行（不要 torch.distributed.run，代码没 DDP）。

---

## 🎯 批 D — multi-seed eval (最简单，已有 DPO ckpt)

**目的**：给 Spatial+Object DPO 补 seed 1337+2026，4 suite 全齐 3-seed paper-grade noise band。
**依赖**：`vla_workspace/output/h20_dpo_{spatial,object}/checkpoint-500.pt`（需要从 cloudml 额外拷进来，见 § 附录 B）
**时长**：Spatial ~2.5h，Object ~4.2h。**建议 2 个单卡任务并行。**

### 任务 D-1: Spatial DPO multi-seed eval

```bash
cd /workspace_data/ro_planning/code && \
python -m post_training.eval_libero \
    --base openvla \
    --base_ckpt /workspace_data/vla_workspace/models/openvla-7b-finetuned-libero-spatial \
    --lora_ckpt /workspace_data/vla_workspace/output/h20_dpo_spatial/checkpoint-500.pt \
    --suites libero_spatial \
    --n_tasks 10 --n_trials_per_task 5 --seeds 1337 2026 \
    --output_dir /workspace_data/vla_workspace/output/h20_dpo_spatial_eval_multiseed
```

### 任务 D-2: Object DPO multi-seed eval

```bash
cd /workspace_data/ro_planning/code && \
python -m post_training.eval_libero \
    --base openvla \
    --base_ckpt /workspace_data/vla_workspace/models/openvla-7b-finetuned-libero-object \
    --lora_ckpt /workspace_data/vla_workspace/output/h20_dpo_object/checkpoint-500.pt \
    --suites libero_object \
    --n_tasks 10 --n_trials_per_task 5 --seeds 1337 2026 \
    --output_dir /workspace_data/vla_workspace/output/h20_dpo_object_eval_multiseed
```

---

## ⭐⭐ 批 B — OpenVLA × GRPO × 4 suite (paper §4.2 整行新数据)

**目的**：填 paper §4.2 OpenVLA + GRPO 这一整行。和 DPO 对照，验证 "on-policy 是否能救 DPO 在强 baseline 上的退步"。
**依赖**：只需 `pairs.pt`（已在 dev pod）+ OpenVLA SFT ckpt（已在 dev pod）
**时长**：每 cell 约 4-5h（on-policy rollout 是瓶颈），**建议 4 个单卡任务并行**，~4-5h 全完。

### 任务 B-1: Spatial GRPO

```bash
cd /workspace_data/ro_planning/code && \
python -m post_training.train_grpo \
    --base openvla \
    --base_ckpt /workspace_data/vla_workspace/models/openvla-7b-finetuned-libero-spatial \
    --suite spatial \
    --output_dir /workspace_data/vla_workspace/output/h20_grpo_spatial \
    --n_rollout_tasks 10 --n_inits_per_task 3 --k_samples 2 \
    --max_chunk_len 180 \
    --batch_size 1 --max_steps 500 --warmup 50 \
    --log_every 10 --save_every 500 \
    --lora_r 16 --lr 3e-5 \
    --grpo_beta 0.1 --grpo_epsilon 0.2 && \
python -m post_training.eval_libero \
    --base openvla \
    --base_ckpt /workspace_data/vla_workspace/models/openvla-7b-finetuned-libero-spatial \
    --lora_ckpt /workspace_data/vla_workspace/output/h20_grpo_spatial/checkpoint-500.pt \
    --suites libero_spatial \
    --n_tasks 10 --n_trials_per_task 5 --seeds 42 \
    --output_dir /workspace_data/vla_workspace/output/h20_grpo_spatial_eval
```

### 任务 B-2: Object GRPO

```bash
cd /workspace_data/ro_planning/code && \
python -m post_training.train_grpo \
    --base openvla \
    --base_ckpt /workspace_data/vla_workspace/models/openvla-7b-finetuned-libero-object \
    --suite object \
    --output_dir /workspace_data/vla_workspace/output/h20_grpo_object \
    --n_rollout_tasks 10 --n_inits_per_task 3 --k_samples 2 \
    --max_chunk_len 180 \
    --batch_size 1 --max_steps 500 --warmup 50 \
    --log_every 10 --save_every 500 \
    --lora_r 16 --lr 3e-5 \
    --grpo_beta 0.1 --grpo_epsilon 0.2 && \
python -m post_training.eval_libero \
    --base openvla \
    --base_ckpt /workspace_data/vla_workspace/models/openvla-7b-finetuned-libero-object \
    --lora_ckpt /workspace_data/vla_workspace/output/h20_grpo_object/checkpoint-500.pt \
    --suites libero_object \
    --n_tasks 10 --n_trials_per_task 5 --seeds 42 \
    --output_dir /workspace_data/vla_workspace/output/h20_grpo_object_eval
```

### 任务 B-3: Goal GRPO

```bash
cd /workspace_data/ro_planning/code && \
python -m post_training.train_grpo \
    --base openvla \
    --base_ckpt /workspace_data/vla_workspace/models/openvla-7b-finetuned-libero-goal \
    --suite goal \
    --output_dir /workspace_data/vla_workspace/output/h20_grpo_goal \
    --n_rollout_tasks 10 --n_inits_per_task 3 --k_samples 2 \
    --max_chunk_len 180 \
    --batch_size 1 --max_steps 500 --warmup 50 \
    --log_every 10 --save_every 500 \
    --lora_r 16 --lr 3e-5 \
    --grpo_beta 0.1 --grpo_epsilon 0.2 && \
python -m post_training.eval_libero \
    --base openvla \
    --base_ckpt /workspace_data/vla_workspace/models/openvla-7b-finetuned-libero-goal \
    --lora_ckpt /workspace_data/vla_workspace/output/h20_grpo_goal/checkpoint-500.pt \
    --suites libero_goal \
    --n_tasks 10 --n_trials_per_task 5 --seeds 42 \
    --output_dir /workspace_data/vla_workspace/output/h20_grpo_goal_eval
```

### 任务 B-4: Long10 GRPO

```bash
cd /workspace_data/ro_planning/code && \
python -m post_training.train_grpo \
    --base openvla \
    --base_ckpt /workspace_data/vla_workspace/models/openvla-7b-finetuned-libero-10 \
    --suite long10 \
    --output_dir /workspace_data/vla_workspace/output/h20_grpo_long10 \
    --n_rollout_tasks 10 --n_inits_per_task 3 --k_samples 2 \
    --max_chunk_len 180 \
    --batch_size 1 --max_steps 500 --warmup 50 \
    --log_every 10 --save_every 500 \
    --lora_r 16 --lr 3e-5 \
    --grpo_beta 0.1 --grpo_epsilon 0.2 && \
python -m post_training.eval_libero \
    --base openvla \
    --base_ckpt /workspace_data/vla_workspace/models/openvla-7b-finetuned-libero-10 \
    --lora_ckpt /workspace_data/vla_workspace/output/h20_grpo_long10/checkpoint-500.pt \
    --suites libero_10 \
    --n_tasks 10 --n_trials_per_task 5 --seeds 42 \
    --output_dir /workspace_data/vla_workspace/output/h20_grpo_long10_eval
```

---

## 附录 A: 起任务前 sanity check（在 dev pod 4321 上跑一次，验证一切就绪）

```bash
# ssh -p 4321 root@<dev_pod>
docker run --rm --gpus '"device=0"' \
    -v /e2e-data/users/liuzhi7/ro_planning:/workspace_data/ro_planning:ro \
    -v /e2e-data/users/liuzhi7/vla_workspace:/workspace_data/vla_workspace \
    -e PYTHONPATH=/workspace_data/ro_planning/code:/workspace_data/vla_workspace/openvla \
    -e MUJOCO_GL=osmesa -e PYOPENGL_PLATFORM=osmesa \
    -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e PYTHONUNBUFFERED=1 \
    evad-ml-cn-bjdb-1-vecps.cr.cloud.vnet.com/infra_public/planning:vla-lab-v1.0-cu128-py311 \
    python -m post_training.eval_libero \
        --base openvla \
        --base_ckpt /workspace_data/vla_workspace/models/openvla-7b-finetuned-libero-spatial \
        --suites libero_spatial \
        --n_tasks 1 --n_trials_per_task 5 --seeds 42 \
        --output_dir /workspace_data/vla_workspace/output/sanity_test
```

期望：**4-5/5 success** on the one task（验证 ckpt + patch + env 全 OK）。

---

## 附录 B: 补充 DPO ckpt（批 D 需要）

批 D 需要 `h20_dpo_{spatial,object}/checkpoint-500.pt`，目前在 cloudml 上。转运方法（和 ckpt 同一路子，cloudml→本机→dev pod）：

```bash
# 在本机跑：
for S in spatial object goal long10; do
  mkdir -p /tmp/dpo_ckpt_transit/h20_dpo_$S
  scp -P 4163 -o StrictHostKeyChecking=no \
      root@127.0.0.1:/ad-alg/planning-users/liuzhi7/ro_planning/output/h20_dpo_$S/checkpoint-500.pt \
      /tmp/dpo_ckpt_transit/h20_dpo_$S/checkpoint-500.pt
  scp -rp -P 4321 -o StrictHostKeyChecking=no \
      /tmp/dpo_ckpt_transit/h20_dpo_$S \
      root@127.0.0.1:/e2e-data/users/liuzhi7/vla_workspace/output/
done
```

每个 67 MB，4 个共 268 MB，几分钟搞定。

---

## 附录 C: 已有资源清单（dev pod /e2e-data/users/liuzhi7）

| 路径 | 内容 | 大小 |
|---|---|---|
| `ro_planning/` | 代码（已 pull 最新 commit 1ea4120，含 --max_chunk_len）| ~500 MB |
| `vla_workspace/datasets/openvla_{spatial,object,goal,long10}_pairs.pt` | DPO pair data | 4 × ~60 MB |
| `vla_workspace/models/openvla-7b-finetuned-libero-{spatial,object,goal,10}/` | SFT ckpt，全 patched | 4 × 15 GB |
| `vla_workspace/openvla/` | OpenVLA 源码（从 cloudml 拷）| ~50 MB |

Docker image 会在每个 MLP 任务启动时 pull（有 `~/.docker/config.json` 凭证）。

---

## 推荐起任务顺序

1. **先起批 D**（2 个单卡任务）：简单、快、保险。约 4.2h 全完。
2. **批 D 跑着的时候起批 B**（4 个单卡任务）：8×H20 恰好 2+4=6 卡占掉，留 2 卡备用。
3. 所有跑完后，我这边（cloudml + 本机）把所有结果 scp 回来合并进 paper §4.2。

**总预估**：批 D (4.2h) + 批 B (4-5h) = **明早所有 cell 全完**。
