# 世纪互联 8×H20 部署清单

为 v1.5 paper 实验在世纪互联 8×H20 (96 GB) 上跑 OpenVLA / Spirit / π0.5 × DPO/GRPO × 4 LIBERO suites 准备的"复制粘贴"命令。

**前提**：
- Pod 端口 4321，工作目录 `/e2e-data/users/liuzhi7/`
- 与 cloudml 开发机一样的 SSH/git 配置
- Pod 默认 overlay 通常 ≥ 50 GB，足够装 vla-lab-v1.0 (22.4 GB)

---

## ⭐ 重要更新（2026-05-11 早晨实证）

我们在 cloudml 1×H20 上完整验证了 pipeline，**实测数据**：

| Cell | 数字 | Δ |
|---|---|---|
| OpenVLA SFT Spatial | 72% | baseline |
| OpenVLA + DPO Spatial | **78%** | **+6** ✅ |
| OpenVLA SFT Object | 62% | baseline |
| OpenVLA + DPO Object | **62%** | **+0** ⚠ |
| OpenVLA SFT Goal | 82% | baseline (paper 79.2%) |
| OpenVLA SFT Long10 | 60% | baseline (paper 53.7%) |

**关键发现**：DPO 是 **suite-dependent** 的：Spatial 上 +6 显著，Object 上 +0 但 per-task 大变化（biggest +2 / -3）。

**关键参数**（实测过最优）：

```
DPO 配置:
  --batch_size 1                  ← bs>=4 OOM on 144GB H20
  --max_steps 500                 ← 100 pair 时 step 500 已 overfit
  --warmup 100
  --lora_r 32 --lora_alpha 64
  --lr 5e-5
  --beta 0.1
  --max_chunk_len 220             ← 关键！Goal/Long10/Object rollout 出的 chunk 是 T=300
                                     会 OOM（attention O(n²)，1.86× Spatial 显存）
                                     截到 220 和 Spatial 一致就能跑通
                                     Spatial pairs 本来就 220，加这个也不会截

  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  PYTHONUNBUFFERED=1               ← 不加这个 log 会 buffer 几分钟无输出
  HF_HUB_OFFLINE=1
  TRANSFORMERS_OFFLINE=1
```

每个 cell 在 H20 上耗时：
- rollout 200 ep × 75s = **~4h**
- DPO train 500 step × bs=1 × ~3s = **~25 min**
- Eval 50 trial × 60s = **~50 min**
- **每 cell 完整 ~5h**

3 个剩余 OpenVLA × DPO cells（Object 已完成）×  4-5h on 8 GPUs 并行 = **~5h 跑完**。

---

## Phase 0: Pod 初始化（每个 pod 一次）

```bash
# 1. SSH 进 pod
ssh -p 4321 root@<pod_ip>

# 2. 设置工作目录
mkdir -p /e2e-data/users/liuzhi7/vla-lab && cd /e2e-data/users/liuzhi7

# 3. 拉镜像（已 push 到 vnet registry，与 cloudml 同 digest）
docker pull evad-ml-cn-bjdb-1-vecps.cr.cloud.vnet.com/infra_public/planning:vla-lab-v1.0-cu128-py311
# 备选 registry（任选一）：
# docker pull micr.cloud.mioffice.cn/world-model-lyk/planningmodel:vla-lab-v1.0-cu128-py311
# docker pull test-lab-instance-cn-beijing.cr.volces.com/evad-infra-compute/planningmodel:vla-lab-v1.0-cu128-py311

# 4. Clone 代码（公开仓即可）
cd /e2e-data/users/liuzhi7
git clone https://github.com/lz-googlefycy/vla-lab.git
# 或者用 GitLab 私仓：
# git clone https://git.n.xiaomi.com/liuzhi7/ro_planning.git

# 5. Clone OpenVLA 源码（adapter PYTHONPATH 需要）
git clone https://github.com/openvla/openvla.git

# 6. 下载所有 4 个 OpenVLA-LIBERO ckpts (~60 GB total)
cd /e2e-data/users/liuzhi7
mkdir -p models
huggingface-cli download openvla/openvla-7b-finetuned-libero-spatial \
    --local-dir models/openvla-7b-finetuned-libero-spatial
huggingface-cli download openvla/openvla-7b-finetuned-libero-object \
    --local-dir models/openvla-7b-finetuned-libero-object
huggingface-cli download openvla/openvla-7b-finetuned-libero-goal \
    --local-dir models/openvla-7b-finetuned-libero-goal
huggingface-cli download openvla/openvla-7b-finetuned-libero-10 \
    --local-dir models/openvla-7b-finetuned-libero-10

# 7. **关键**：patch 所有 ckpt 的 modeling_prismatic.py（解决 transformers>=4.50 generate 退化）
docker run --rm \
    -v /e2e-data/users/liuzhi7/models:/models \
    -v /e2e-data/users/liuzhi7/vla-lab/code:/code \
    -e PYTHONPATH=/code \
    evad-ml-cn-bjdb-1-vecps.cr.cloud.vnet.com/infra_public/planning:vla-lab-v1.0-cu128-py311 \
    python -c "
import sys; sys.path.insert(0, '/code')
import os
# Patch the patcher script's ckpt root to match our layout
import re
from pathlib import Path
ROOT = Path('/models')
PATTERN = re.compile(
    r'    @property\s*\n'
    r'    def _supports_sdpa\(self\) -> bool:\s*\n'
    r'        \"\"\"Check LLM supports SDPA Attention\"\"\"\s*\n'
    r'        return self\.language_model\._supports_sdpa',
    re.MULTILINE,
)
REPLACEMENT = '    _supports_sdpa = False  # patched for transformers>=4.50'
for ckpt_dir in ROOT.iterdir():
    if 'openvla' not in ckpt_dir.name.lower(): continue
    for f in [ckpt_dir / 'modeling_prismatic.py']:
        if not f.exists(): continue
        src = f.read_text()
        new = PATTERN.sub(REPLACEMENT, src)
        if new != src:
            f.write_text(new)
            print(f'patched: {f}')
        elif '_supports_sdpa = False' in src:
            print(f'already: {f}')
        else:
            print(f'NO MATCH (regex needs update): {f}')
"
```

---

## Phase 1: 单卡 sanity check（每个 pod 第一次跑前 5 min）

```bash
# 验证 1 张卡能加载 OpenVLA + 跑 1 个 LIBERO trial
docker run --rm --gpus '"device=0"' \
    -v /e2e-data/users/liuzhi7/openvla:/openvla:ro \
    -v /e2e-data/users/liuzhi7/models/openvla-7b-finetuned-libero-spatial:/model:ro \
    -v /e2e-data/users/liuzhi7/vla-lab/code:/code:ro \
    -v /e2e-data/users/liuzhi7/output_test:/output \
    -e PYTHONPATH=/code:/openvla \
    -e MUJOCO_GL=osmesa \
    -e PYOPENGL_PLATFORM=osmesa \
    -e HF_HUB_OFFLINE=1 \
    -e TRANSFORMERS_OFFLINE=1 \
    -e PYTHONUNBUFFERED=1 \
    evad-ml-cn-bjdb-1-vecps.cr.cloud.vnet.com/infra_public/planning:vla-lab-v1.0-cu128-py311 \
    python -m post_training.eval_libero \
        --base openvla --base_ckpt /model \
        --suites libero_spatial \
        --n_tasks 1 --n_trials_per_task 5 --seeds 42 \
        --output_dir /output

# 期望输出：4-5/5 = 80-100% success
# 如果 0%，patch 没生效，回 Phase 0 step 7
```

---

## Phase 2: 8×H20 并行实验矩阵

### 实验设计

每张 H20 跑一个 cell，8 个 cells 同时跑：

| GPU | 任务 | 预计时间 |
|---|---|---|
| 0 | OpenVLA + DPO Spatial（已有 200 pairs from cloudml H20） | 5-7h |
| 1 | OpenVLA + DPO Object | 5-7h |
| 2 | OpenVLA + DPO Goal | 5-7h |
| 3 | OpenVLA + DPO Long10 | 6-8h |
| 4 | OpenVLA + GRPO Spatial（online rollout） | 8-10h |
| 5 | OpenVLA + GRPO Object | 8-10h |
| 6 | OpenVLA SFT 4-suite eval（baseline 完整 50×3 seed） | 4h |
| 7 | π0.5 SFT 4-suite eval（baseline） | 4h |

一晚上 (~10h) 跑完所有 OpenVLA 4 cells × 2 algs + 2 baseline 完整 eval。

---

### 启动模板（per-GPU）

每个 GPU 跑一个 docker container，用 `--gpus '"device=N"'` 隔离：

```bash
# Helper function for launching a cell
launch_cell() {
    local gpu=$1
    local cell_name=$2
    shift 2
    local cmd="$@"

    nohup docker run --rm --gpus "\"device=$gpu\"" \
        -v /e2e-data/users/liuzhi7/openvla:/openvla:ro \
        -v /e2e-data/users/liuzhi7/models:/models:ro \
        -v /e2e-data/users/liuzhi7/vla-lab/code:/code:ro \
        -v /e2e-data/users/liuzhi7/datasets:/datasets:ro \
        -v /e2e-data/users/liuzhi7/output/$cell_name:/output \
        -e PYTHONPATH=/code:/openvla \
        -e MUJOCO_GL=osmesa \
        -e PYOPENGL_PLATFORM=osmesa \
        -e HF_HUB_OFFLINE=1 \
        -e TRANSFORMERS_OFFLINE=1 \
        -e PYTHONUNBUFFERED=1 \
        evad-ml-cn-bjdb-1-vecps.cr.cloud.vnet.com/infra_public/planning:vla-lab-v1.0-cu128-py311 \
        bash -c "$cmd" \
        > /e2e-data/users/liuzhi7/logs/$cell_name.log 2>&1 &
    echo "GPU $gpu  $cell_name  PID=$!"
}

mkdir -p /e2e-data/users/liuzhi7/logs /e2e-data/users/liuzhi7/output /e2e-data/users/liuzhi7/datasets
```

### Step A: 收集 DPO pair datasets（4 suites × 1 GPU each = 4 hours）

8 张卡同时跑 4 个 rollouts（占 4 张），剩 4 张同时跑 baseline eval。

```bash
# GPU 0: rollout Spatial — 跳过（cloudml H20 已经收集，scp 来即可）
scp -P 4163 root@<cloudml_ip>:/ad-alg/planning-users/liuzhi7/ro_planning/output/h20_rollout_spatial/spatial_pairs.pt \
    /e2e-data/users/liuzhi7/datasets/openvla_spatial_pairs.pt

# GPU 0: rollout Object
launch_cell 0 rollout_object \
    "python -m post_training.rollout \
        --base openvla \
        --base_ckpt /models/openvla-7b-finetuned-libero-object \
        --suite libero_object \
        --n_tasks 10 --n_inits_per_task 5 --n_candidates_per_init 4 \
        --output_path /datasets/openvla_object_pairs.pt \
        --resolution 256"

# GPU 1: rollout Goal
launch_cell 1 rollout_goal \
    "python -m post_training.rollout \
        --base openvla --base_ckpt /models/openvla-7b-finetuned-libero-goal \
        --suite libero_goal --n_tasks 10 --n_inits_per_task 5 --n_candidates_per_init 4 \
        --output_path /datasets/openvla_goal_pairs.pt --resolution 256"

# GPU 2: rollout Long10
launch_cell 2 rollout_long10 \
    "python -m post_training.rollout \
        --base openvla --base_ckpt /models/openvla-7b-finetuned-libero-10 \
        --suite libero_10 --n_tasks 10 --n_inits_per_task 5 --n_candidates_per_init 4 \
        --output_path /datasets/openvla_long10_pairs.pt --resolution 256"

# GPU 3: 同时跑 OpenVLA SFT 完整 baseline eval (50 trial × 3 seed) on Spatial
launch_cell 3 baseline_spatial_full \
    "python -m post_training.eval_libero \
        --base openvla --base_ckpt /models/openvla-7b-finetuned-libero-spatial \
        --suites libero_spatial --n_tasks 10 --n_trials_per_task 50 --seeds 42 1337 2026 \
        --output_dir /output"

# GPU 4: SFT baseline Object full
launch_cell 4 baseline_object_full \
    "python -m post_training.eval_libero \
        --base openvla --base_ckpt /models/openvla-7b-finetuned-libero-object \
        --suites libero_object --n_tasks 10 --n_trials_per_task 50 --seeds 42 1337 2026 \
        --output_dir /output"

# GPU 5: SFT baseline Goal full
launch_cell 5 baseline_goal_full \
    "python -m post_training.eval_libero \
        --base openvla --base_ckpt /models/openvla-7b-finetuned-libero-goal \
        --suites libero_goal --n_tasks 10 --n_trials_per_task 50 --seeds 42 1337 2026 \
        --output_dir /output"

# GPU 6: SFT baseline Long10 full
launch_cell 6 baseline_long10_full \
    "python -m post_training.eval_libero \
        --base openvla --base_ckpt /models/openvla-7b-finetuned-libero-10 \
        --suites libero_10 --n_tasks 10 --n_trials_per_task 50 --seeds 42 1337 2026 \
        --output_dir /output"

# GPU 7: 留给 GRPO 第一个 cell（rollout 完成后再用）
```

### Step B: DPO 训练（rollout 完成后，4 GPUs × 1.5h each）

```bash
# GPU 0: DPO Spatial（pair file 已经 scp 过来）
launch_cell 0 dpo_spatial \
    "python -m post_training.train_dpo \
        --base openvla --base_ckpt /models/openvla-7b-finetuned-libero-spatial \
        --suite spatial \
        --pairs_file /datasets/openvla_spatial_pairs.pt \
        --output_dir /output \
        --batch_size 1 --max_steps 500 --warmup 100 \
        --log_every 10 --save_every 500 \
        --lora_r 32 --lr 5e-5 --beta 0.1 \
        --max_chunk_len 220"

# GPU 1: DPO Object（必须加 --max_chunk_len 220，Object rollout chunk=300 会 OOM）
launch_cell 1 dpo_object \
    "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
     python -m post_training.train_dpo \
        --base openvla --base_ckpt /models/openvla-7b-finetuned-libero-object \
        --suite object --pairs_file /datasets/openvla_object_pairs.pt \
        --output_dir /output \
        --batch_size 1 --max_steps 500 --warmup 100 \
        --log_every 10 --save_every 500 --lora_r 32 --lr 5e-5 --beta 0.1 \
        --max_chunk_len 220"

# GPU 2-3 同理 Goal / Long10
# ...
```

### Step C: DPO eval（训完后用 LoRA ckpt 跑 4-suite eval）

```bash
# GPU 0: eval DPO Spatial ckpt
launch_cell 0 eval_dpo_spatial \
    "python -m post_training.eval_libero \
        --base openvla --base_ckpt /models/openvla-7b-finetuned-libero-spatial \
        --lora_ckpt /output/dpo_spatial/checkpoint-5000.pt \
        --suites libero_spatial libero_object libero_goal libero_10 \
        --n_tasks 10 --n_trials_per_task 50 --seeds 42 1337 2026 \
        --output_dir /output_eval"
```

---

## Monitor 命令

```bash
# 1. 当前所有 cells 进度（在世纪互联 pod 上）
bash /e2e-data/users/liuzhi7/vla-lab/scripts/monitor_8gpu.sh
# (我会在 phase 4 写这个)

# 2. 单看一个 cell
tail -f /e2e-data/users/liuzhi7/logs/dpo_spatial.log

# 3. 总览 GPU 占用
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv

# 4. 同步 output 回 cloudml 开发机（看 paper 用）
rsync -az --progress /e2e-data/users/liuzhi7/output \
    cloudml-host:/ad-alg/planning-users/liuzhi7/ro_planning/output/shijihulian/
```

---

## 关键预防：常见坑

### 1. 模型加载时 `_supports_sdpa` 报错

→ **没跑 Phase 0 step 7 的 patch**。重跑这一步即可。

### 2. LIBERO env 卡死无输出

→ **没设 `PYTHONUNBUFFERED=1`**。print 默认 buffered，要等几分钟才 flush。launch_cell 模板里已带。

### 3. docker pull 失败 "no space left on device"

→ pod overlay 太小（<25 GB）。要么换大 overlay 模板创建 pod，要么先删旧镜像 (`docker image prune -af`)。

### 4. DPO loss 起步是 0.6914 (=ln(2)) 但很快变 NaN

→ LoRA `_LoRALinear` 第一步初始化 effect = 0（B 矩阵初始为 0），cur policy = ref policy → 数学上 loss = ln(2) 正常。如果 NaN 出现，检查 grad clip 或 lr 太大。

### 5. 跑了 10 分钟没出第一行 group 0 k=0 的进度

→ 99% 是 `PYTHONUNBUFFERED=1` 没设。kill + 重启加上。

---

## 时间表（8×H20 一晚预期）

```
T+0:    Phase 0 setup (拉镜像 + ckpts ~30 min on first pod)
T+1:    Step A 启动 — 4 rollouts + 4 baseline eval 并行
T+5h:   Step A 完成 — paper §4.2 SFT 行 4 cells 完整 (50×3 seed)
                    + 4 个 DPO pair dataset 就绪
T+5h:   Step B 启动 — 4 DPO 训练并行
T+7h:   Step B 完成 — 4 个 LoRA checkpoint
T+7h:   Step C 启动 — 4 个 LoRA × 4-suite eval (cross-suite generalization)
T+12h:  Step C 完成 — paper §4.2 OpenVLA + DPO 行 4 cells × 4 suites = 16 cells

可选 T+12h~24h:  GRPO 同样架构，4 cells，online rollout 慢，~12h
```

**一个晚上 + 半天 = OpenVLA × DPO 完整 paper §4.2 数据 + 完整 baseline 行**。

---

## Spirit / π0.5 cells（次日 / 后续）

Spirit baseline 在 LIBERO 没 ckpt — 需要先训 Spirit-LIBERO SFT。这是新工作量，预计：
- 用 LIBERO-RLDS 数据训 Spirit + LoRA：~10-15 H20 hours
- 然后同样跑 DPO/GRPO

π0.5 LoRA 路径未确定（PyTorch 不支持 LoRA） — 三个备选我们要先做选择再启动：
1. JAX 路径（重新做镜像）
2. PyTorch full-finetune（吃显存）
3. 跳过 π0.5 这一家，paper 改成"OpenVLA + Spirit"两家

我建议：**OpenVLA 完整跑完后，看数据再决定 Spirit/π0.5 怎么处理**。

---

*更新：2026-05-10 凌晨。OpenVLA Spatial baseline (我们的) 72%, OpenVLA × LIBERO Spatial DPO rollout 76% success rate, 200 pairs 即将就绪。世纪互联部署的工作流已在 cloudml H20 充分验证。*
