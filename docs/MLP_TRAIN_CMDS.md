# MLP 训练命令（仅训练版，单卡任务，每任务一条）

> Eval 独立跑（见 `MLP_EVAL_CMDS.md`），如需 train+eval 串行见 `MLP_TASK_COMMANDS.md` 批 B。
>
> **镜像**：`evad-ml-cn-bjdb-1-vecps.cr.cloud.vnet.com/infra_public/planning:vla-lab-v1.0-cu128-py311`
> **挂载**：`/e2e-data/users/liuzhi7  →  /workspace_data`
> **显存**：96GB H20 × 1 卡/任务

---

## 通用环境变量（每条命令都包含）

```
PYTHONPATH=/workspace_data/ro_planning/code:/workspace_data/vla_workspace/openvla
MUJOCO_GL=osmesa
PYOPENGL_PLATFORM=osmesa
HF_HUB_OFFLINE=1
TRANSFORMERS_OFFLINE=1
PYTHONUNBUFFERED=1
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

---

## 🔧 DPO 训练（用已有 pair data 重跑 DPO，验证/重训）

如果你只想让 DPO 再跑一遍（不 eval），用已有 pair data：

### DPO Spatial

```bash
bash -c '
set -euo pipefail
cd /workspace_data/ro_planning/code
export PYTHONPATH=/workspace_data/ro_planning/code:/workspace_data/vla_workspace/openvla
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python -m post_training.train_dpo \
    --base openvla \
    --base_ckpt /workspace_data/vla_workspace/models/openvla-7b-finetuned-libero-spatial \
    --suite spatial \
    --pairs_file /workspace_data/vla_workspace/datasets/openvla_spatial_pairs.pt \
    --output_dir /workspace_data/vla_workspace/output/h20_dpo_spatial_retrain \
    --max_chunk_len 180 \
    --batch_size 1 --max_steps 500 --warmup 100 \
    --log_every 10 --save_every 500 \
    --lora_r 32 --lora_alpha 64 --lr 5e-5 --beta 0.1
'
```

时长 ~25 min。ckpt → `output/h20_dpo_spatial_retrain/checkpoint-500.pt`。

### DPO Object / Goal / Long10

只把 spatial/libero_spatial 换成对应：

| suite | SFT ckpt dir 后缀 | pair file |
|---|---|---|
| object | `-libero-object` | `openvla_object_pairs.pt` |
| goal | `-libero-goal` | `openvla_goal_pairs.pt` |
| long10 | `-libero-10` | `openvla_long10_pairs.pt` |

完整模板：

```bash
bash -c '
set -euo pipefail
cd /workspace_data/ro_planning/code
export PYTHONPATH=/workspace_data/ro_planning/code:/workspace_data/vla_workspace/openvla
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

SUITE=object    # 改这里: spatial / object / goal / long10
SFT_DIR=openvla-7b-finetuned-libero-object  # 改这里
PAIR=openvla_object_pairs.pt  # 改这里

python -m post_training.train_dpo \
    --base openvla \
    --base_ckpt /workspace_data/vla_workspace/models/$SFT_DIR \
    --suite $SUITE \
    --pairs_file /workspace_data/vla_workspace/datasets/$PAIR \
    --output_dir /workspace_data/vla_workspace/output/h20_dpo_${SUITE}_retrain \
    --max_chunk_len 180 \
    --batch_size 1 --max_steps 500 --warmup 100 \
    --log_every 10 --save_every 500 \
    --lora_r 32 --lora_alpha 64 --lr 5e-5 --beta 0.1
'
```

---

## ⭐⭐ GRPO 训练（4 个任务，每个单卡）

GRPO 是 on-policy，不用 pair data，实时 env rollout。每 cell 训练 ~3-4h。

### GRPO Spatial

```bash
bash -c '
set -euo pipefail
cd /workspace_data/ro_planning/code
export PYTHONPATH=/workspace_data/ro_planning/code:/workspace_data/vla_workspace/openvla
export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python -m post_training.train_grpo \
    --base openvla \
    --base_ckpt /workspace_data/vla_workspace/models/openvla-7b-finetuned-libero-spatial \
    --suite spatial \
    --output_dir /workspace_data/vla_workspace/output/h20_grpo_spatial \
    --n_tasks 10 --n_inits_per_task 3 --group_size 2 \
    --max_chunk_len 180 \
    --max_steps 500 --warmup 50 \
    --log_every 10 --save_every 500 \
    --lora_r 16 --lr 3e-5 \
    --beta 0.1 --epsilon 0.2
'
```

### GRPO Object

```bash
bash -c '
set -euo pipefail
cd /workspace_data/ro_planning/code
export PYTHONPATH=/workspace_data/ro_planning/code:/workspace_data/vla_workspace/openvla
export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python -m post_training.train_grpo \
    --base openvla \
    --base_ckpt /workspace_data/vla_workspace/models/openvla-7b-finetuned-libero-object \
    --suite object \
    --output_dir /workspace_data/vla_workspace/output/h20_grpo_object \
    --n_tasks 10 --n_inits_per_task 3 --group_size 2 \
    --max_chunk_len 180 \
    --max_steps 500 --warmup 50 \
    --log_every 10 --save_every 500 \
    --lora_r 16 --lr 3e-5 \
    --beta 0.1 --epsilon 0.2
'
```

### GRPO Goal

```bash
bash -c '
set -euo pipefail
cd /workspace_data/ro_planning/code
export PYTHONPATH=/workspace_data/ro_planning/code:/workspace_data/vla_workspace/openvla
export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python -m post_training.train_grpo \
    --base openvla \
    --base_ckpt /workspace_data/vla_workspace/models/openvla-7b-finetuned-libero-goal \
    --suite goal \
    --output_dir /workspace_data/vla_workspace/output/h20_grpo_goal \
    --n_tasks 10 --n_inits_per_task 3 --group_size 2 \
    --max_chunk_len 180 \
    --max_steps 500 --warmup 50 \
    --log_every 10 --save_every 500 \
    --lora_r 16 --lr 3e-5 \
    --beta 0.1 --epsilon 0.2
'
```

### GRPO Long10

```bash
bash -c '
set -euo pipefail
cd /workspace_data/ro_planning/code
export PYTHONPATH=/workspace_data/ro_planning/code:/workspace_data/vla_workspace/openvla
export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python -m post_training.train_grpo \
    --base openvla \
    --base_ckpt /workspace_data/vla_workspace/models/openvla-7b-finetuned-libero-10 \
    --suite long10 \
    --output_dir /workspace_data/vla_workspace/output/h20_grpo_long10 \
    --n_tasks 10 --n_inits_per_task 3 --group_size 2 \
    --max_chunk_len 180 \
    --max_steps 500 --warmup 50 \
    --log_every 10 --save_every 500 \
    --lora_r 16 --lr 3e-5 \
    --beta 0.1 --epsilon 0.2
'
```

---

## 🧪 30 秒 smoke test（起正式任务前验证环境 OK）

```bash
bash -c '
set -euo pipefail
export PYTHONPATH=/workspace_data/ro_planning/code:/workspace_data/vla_workspace/openvla
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export OPENVLA_CKPT=/workspace_data/vla_workspace/models/openvla-7b-finetuned-libero-spatial

cd /workspace_data/ro_planning/code
python -m post_training.debug_logp
'
```

期望 `Done.` + 打印 `diff = +0.000000`。

---

## 📋 推荐起 4 个训练任务（今晚）

1. **B-1 GRPO Spatial**
2. **B-2 GRPO Object**
3. **B-3 GRPO Goal**
4. **B-4 GRPO Long10**

4 个 GPU 并行 ~3-4h 全训完。训完后 eval 另起（或用 MLP_TASK_COMMANDS.md 里的 train+eval 合并版）。

---

## ⚠️ 关键注意

1. **`--max_chunk_len 180`**：96GB H20 必须，不加会 OOM（已在命令里）
2. **`--group_size 2`**：GRPO K=4 在 96GB 会 OOM
3. **`MUJOCO_GL=osmesa`**：GRPO 的 rollout 要走 MuJoCo off-screen render，不设就崩
4. **首次启动需 ~15-30s 加载 ckpt**（HF auto_map 编译 modeling_prismatic.py）
5. **训练完的 ckpt**：`{output_dir}/checkpoint-500.pt`（67MB，LoRA only）
