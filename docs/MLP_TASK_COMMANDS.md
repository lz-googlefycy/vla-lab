# MLP 单卡任务启动命令清单（最终版，dev pod 数据已就绪）

> 状态：2026-05-13 00:35 · **所有资源已在 `/e2e-data/users/liuzhi7/` 就位**。
> 你只需在 MLP 控制台新建任务，填镜像 + 挂载 + 这里的启动命令。

---

## 📦 镜像地址（MLP 任务三选一）

```
evad-ml-cn-bjdb-1-vecps.cr.cloud.vnet.com/infra_public/planning:vla-lab-v1.0-cu128-py311
```

备选：
- `micr.cloud.mioffice.cn/world-model-lyk/planningmodel:vla-lab-v1.0-cu128-py311`
- `test-lab-instance-cn-beijing.cr.volces.com/evad-infra-compute/planningmodel:vla-lab-v1.0-cu128-py311`

三者 digest 一致：`sha256:4ac541a377810e6bd5af7620b21e8b5428103493afe7e0192a8c5929179f1d7d`

---

## 🗂️ 挂载（每个任务统一）

在 MLP 任务的"数据挂载"配置：

```
/e2e-data/users/liuzhi7  →  /workspace_data
```

（你起 TrajFlow 任务时用的数据挂载一样）

**已就绪的数据** `/e2e-data/users/liuzhi7/`：

| 路径 | 内容 |
|---|---|
| `ro_planning/` | 代码（最新 commit `1ea4120`，含 --max_chunk_len）|
| `vla_workspace/models/openvla-7b-finetuned-libero-{spatial,object,goal,10}/` | 4 × 15GB，已 patched |
| `vla_workspace/datasets/openvla_{spatial,object,goal,long10}_pairs.pt` | 4 × ~60MB DPO pairs |
| `vla_workspace/output/h20_dpo_{spatial,object,goal,long10}/checkpoint-500.pt` | 4 × 67MB DPO ckpts |
| `vla_workspace/openvla/` | OpenVLA 源码 |

---

## 🎯 批 D — DPO multi-seed eval × 4 suite（paper §4.2 noise band）

**目的**：给 paper §4.2 的 4 个 DPO cells 每个补 seed 1337 + 2026，消除单 seed 攻击。
**依赖**：都在 dev pod ✅
**时间**：Spatial 2.5h, Object 4.2h, Goal 3.7h, Long10 6.2h。4 个单卡任务并行。
**回收**：结果写到 `/workspace_data/vla_workspace/output/h20_dpo_{suite}_eval_multiseed/`

---

### 任务 D-1: Spatial DPO multi-seed（~2.5h）

```bash
bash -c '
set -euo pipefail
cd /workspace_data/ro_planning/code
export PYTHONPATH=/workspace_data/ro_planning/code:/workspace_data/vla_workspace/openvla
export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python -m post_training.eval_libero \
    --base openvla \
    --base_ckpt /workspace_data/vla_workspace/models/openvla-7b-finetuned-libero-spatial \
    --lora_ckpt /workspace_data/vla_workspace/output/h20_dpo_spatial/checkpoint-500.pt \
    --suites libero_spatial \
    --n_tasks 10 --n_trials_per_task 5 --seeds 1337 2026 \
    --output_dir /workspace_data/vla_workspace/output/h20_dpo_spatial_eval_multiseed
'
```

### 任务 D-2: Object DPO multi-seed（~4.2h）

```bash
bash -c '
set -euo pipefail
cd /workspace_data/ro_planning/code
export PYTHONPATH=/workspace_data/ro_planning/code:/workspace_data/vla_workspace/openvla
export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python -m post_training.eval_libero \
    --base openvla \
    --base_ckpt /workspace_data/vla_workspace/models/openvla-7b-finetuned-libero-object \
    --lora_ckpt /workspace_data/vla_workspace/output/h20_dpo_object/checkpoint-500.pt \
    --suites libero_object \
    --n_tasks 10 --n_trials_per_task 5 --seeds 1337 2026 \
    --output_dir /workspace_data/vla_workspace/output/h20_dpo_object_eval_multiseed
'
```

### 任务 D-3: Goal DPO multi-seed（~3.7h，cloudml 正在跑 seed 42 已有；这里补 1337 2026）

```bash
bash -c '
set -euo pipefail
cd /workspace_data/ro_planning/code
export PYTHONPATH=/workspace_data/ro_planning/code:/workspace_data/vla_workspace/openvla
export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python -m post_training.eval_libero \
    --base openvla \
    --base_ckpt /workspace_data/vla_workspace/models/openvla-7b-finetuned-libero-goal \
    --lora_ckpt /workspace_data/vla_workspace/output/h20_dpo_goal/checkpoint-500.pt \
    --suites libero_goal \
    --n_tasks 10 --n_trials_per_task 5 --seeds 1337 2026 \
    --output_dir /workspace_data/vla_workspace/output/h20_dpo_goal_eval_multiseed
'
```

**注意**：cloudml 正在跑这个，可以**跳过 D-3**免得重复。

### 任务 D-4: Long10 DPO multi-seed（~6.2h）

```bash
bash -c '
set -euo pipefail
cd /workspace_data/ro_planning/code
export PYTHONPATH=/workspace_data/ro_planning/code:/workspace_data/vla_workspace/openvla
export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python -m post_training.eval_libero \
    --base openvla \
    --base_ckpt /workspace_data/vla_workspace/models/openvla-7b-finetuned-libero-10 \
    --lora_ckpt /workspace_data/vla_workspace/output/h20_dpo_long10/checkpoint-500.pt \
    --suites libero_10 \
    --n_tasks 10 --n_trials_per_task 5 --seeds 1337 2026 \
    --output_dir /workspace_data/vla_workspace/output/h20_dpo_long10_eval_multiseed
'
```

**注意**：cloudml 正在跑这个，**跳过 D-4** 免得重复。

**推荐起 D-1 + D-2**（Spatial + Object, 2 个任务，世纪互联承担；Goal + Long10 cloudml 承担）。

---

## ⭐⭐ 批 B — OpenVLA × GRPO × 4 suite（paper §4.2 整行新数据）

**目的**：paper §4.2 OpenVLA + GRPO 这一整行（和 DPO 对照）。
**依赖**：SFT ckpt + pair data 全在 dev pod ✅
**时间**：每 cell ~4-5h（on-policy rollout），4 个单卡并行 → 4-5h 总
**注意**：GRPO 是 on-policy，**不用 pairs file**，实时 rollout，但 LIBERO env 需要 mujoco 渲染

---

### 任务 B-1: Spatial GRPO train + eval

```bash
bash -c '
set -euo pipefail
cd /workspace_data/ro_planning/code
export PYTHONPATH=/workspace_data/ro_planning/code:/workspace_data/vla_workspace/openvla
export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Stage 1: GRPO train
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

# Stage 2: eval
python -m post_training.eval_libero \
    --base openvla \
    --base_ckpt /workspace_data/vla_workspace/models/openvla-7b-finetuned-libero-spatial \
    --lora_ckpt /workspace_data/vla_workspace/output/h20_grpo_spatial/checkpoint-500.pt \
    --suites libero_spatial \
    --n_tasks 10 --n_trials_per_task 5 --seeds 42 \
    --output_dir /workspace_data/vla_workspace/output/h20_grpo_spatial_eval
'
```

### 任务 B-2: Object GRPO

同 B-1，把 `spatial`→`object`, `libero_spatial`→`libero_object`。完整命令：

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

python -m post_training.eval_libero \
    --base openvla \
    --base_ckpt /workspace_data/vla_workspace/models/openvla-7b-finetuned-libero-object \
    --lora_ckpt /workspace_data/vla_workspace/output/h20_grpo_object/checkpoint-500.pt \
    --suites libero_object \
    --n_tasks 10 --n_trials_per_task 5 --seeds 42 \
    --output_dir /workspace_data/vla_workspace/output/h20_grpo_object_eval
'
```

### 任务 B-3: Goal GRPO

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

python -m post_training.eval_libero \
    --base openvla \
    --base_ckpt /workspace_data/vla_workspace/models/openvla-7b-finetuned-libero-goal \
    --lora_ckpt /workspace_data/vla_workspace/output/h20_grpo_goal/checkpoint-500.pt \
    --suites libero_goal \
    --n_tasks 10 --n_trials_per_task 5 --seeds 42 \
    --output_dir /workspace_data/vla_workspace/output/h20_grpo_goal_eval
'
```

### 任务 B-4: Long10 GRPO

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

python -m post_training.eval_libero \
    --base openvla \
    --base_ckpt /workspace_data/vla_workspace/models/openvla-7b-finetuned-libero-10 \
    --lora_ckpt /workspace_data/vla_workspace/output/h20_grpo_long10/checkpoint-500.pt \
    --suites libero_10 \
    --n_tasks 10 --n_trials_per_task 5 --seeds 42 \
    --output_dir /workspace_data/vla_workspace/output/h20_grpo_long10_eval
'
```

---

## 📋 推荐起任务顺序

### 今晚起（6 任务 = 6 张 H20）

| # | 任务 | 时长 | 用途 |
|---|---|---|---|
| 1 | D-1 Spatial DPO multi-seed | 2.5h | noise band |
| 2 | D-2 Object DPO multi-seed | 4.2h | noise band |
| 3 | B-1 Spatial GRPO | 4-5h | paper §4.2 新行 |
| 4 | B-2 Object GRPO | 4-5h | paper §4.2 新行 |
| 5 | B-3 Goal GRPO | 4-5h | paper §4.2 新行 |
| 6 | B-4 Long10 GRPO | 4-5h | paper §4.2 新行 |

**D-3 / D-4（Goal/Long10 DPO multi-seed）** cloudml 上已经在跑，不用世纪互联重复。

早上（~5-6h 后）所有任务应该都完成。

---

## 🧪 Sanity check（可选，起正式任务前验证一次）

先起一个 1 task × 5 trial 的快速测试（~3 min）确认镜像+挂载+代码路径都对：

```bash
bash -c '
set -euo pipefail
cd /workspace_data/ro_planning/code
export PYTHONPATH=/workspace_data/ro_planning/code:/workspace_data/vla_workspace/openvla
export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1

python -m post_training.eval_libero \
    --base openvla \
    --base_ckpt /workspace_data/vla_workspace/models/openvla-7b-finetuned-libero-spatial \
    --suites libero_spatial \
    --n_tasks 1 --n_trials_per_task 5 --seeds 42 \
    --output_dir /workspace_data/vla_workspace/output/sanity_test
'
```

期望：**4-5/5 success** on task 0 → 一切通路 OK，起正式任务。

---

## ⚠️ 已知坑

1. **OOM**：`--max_chunk_len 180` 已是 96GB H20 的保守值。如果 Object/Goal/Long10 GRPO 还 OOM，降到 150
2. **PYTHONUNBUFFERED=1 必须**：不然 log 不出东西看似 hang
3. **GRPO K=2**：H20 96GB 跑不动 K=4（显存 2× 不够）
4. **LIBERO env 初始化第一次慢**：3-5 min 加载 BDDL + MuJoCo，之后每 task 快

---

## 📊 任务完成后我这边做什么

你 MLP 任务跑完，我从 dev pod rsync 回本机 → 合进 paper `§4.2`：

```bash
# 我会在本机跑：
for S in spatial object goal long10; do
  scp -rp -P 4321 root@127.0.0.1:/e2e-data/users/liuzhi7/vla_workspace/output/h20_dpo_${S}_eval_multiseed \
      /tmp/; cp -r /tmp/h20_dpo_${S}_eval_multiseed \
      ~/ro_planning/assets/paper_v1.5_eval/
  # 同理 GRPO
done
# 然后重新生成 chart + 更新 paper skeleton + push 公开仓
```

---

## ⚡ 30-秒 smoke test（MLP 任务起来先跑这个）

如果你想在起正式任务前最快验证镜像+挂载没问题：

```bash
bash -c '
set -euo pipefail
cd /workspace_data/ro_planning/code
export PYTHONPATH=/workspace_data/ro_planning/code:/workspace_data/vla_workspace/openvla
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1

# Just load the model + compute one logp. ~30s, no mujoco.
python -c "
import sys; sys.path.insert(0, \"/workspace_data/ro_planning/code\")
import os; os.environ[\"OPENVLA_CKPT\"] = \"/workspace_data/vla_workspace/models/openvla-7b-finetuned-libero-spatial\"
from post_training.debug_logp import main
main()
"
'
```

期望：最后一行 `Done. (At step 0 of training, cur ≈ ref because LoRA init is 0.)`
意思：ckpt + code + openvla src + GPU 全 OK，可以起正式任务。
