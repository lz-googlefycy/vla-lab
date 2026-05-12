# 世纪互联 8×H20 · 立即起任务（2026-05-12 23:30 最新版）

**状态**：cloudml 正在跑 Goal+Long10 DPO multi-seed eval。世纪互联这边同步起下面剩余任务。

---

## 📦 镜像地址（选一个能连上的）

| Registry | 镜像 Tag | 用途 |
|---|---|---|
| **evad-ml-cn-bjdb-1-vecps.cr.cloud.vnet.com/infra_public/planning** | `vla-lab-v1.0-cu128-py311` | ⭐ 世纪互联内网最快（**推荐**）|
| micr.cloud.mioffice.cn/world-model-lyk/planningmodel | `vla-lab-v1.0-cu128-py311` | 小米内网备用 |
| test-lab-instance-cn-beijing.cr.volces.com/evad-infra-compute/planningmodel | `vla-lab-v1.0-cu128-py311` | 火山 registry 备用 |

**digest 三个一致**：`sha256:4ac541a377810e6bd5af7620b21e8b5428103493afe7e0192a8c5929179f1d7d`，任选一个 pull。

```bash
docker pull evad-ml-cn-bjdb-1-vecps.cr.cloud.vnet.com/infra_public/planning:vla-lab-v1.0-cu128-py311
# 别名
docker tag evad-ml-cn-bjdb-1-vecps.cr.cloud.vnet.com/infra_public/planning:vla-lab-v1.0-cu128-py311 vla-lab:v1.0
```

---

## 🏗️ Phase 0: Pod 初始化（一次）

```bash
cd /e2e-data/users/liuzhi7

# 1. Clone 代码（公开仓，含最新 --max_chunk_len fix）
git clone https://github.com/lz-googlefycy/vla-lab.git
git clone https://github.com/openvla/openvla.git

# 2. 下 4 个 OpenVLA ckpts（~60 GB total, ~10 min）
mkdir -p models
for S in spatial object goal 10; do
  huggingface-cli download openvla/openvla-7b-finetuned-libero-$S \
      --local-dir models/openvla-7b-finetuned-libero-$S &
done
wait

# 3. Patch ckpt 源文件（transformers>=4.50 generate 退化的规避）
docker run --rm \
    -v /e2e-data/users/liuzhi7/models:/models \
    -v /e2e-data/users/liuzhi7/vla-lab/code:/code \
    -e PYTHONPATH=/code \
    vla-lab:v1.0 \
    bash -c "cd /code && python -c \"
import re
from pathlib import Path
ROOT = Path('/models')
PATTERN = re.compile(r'    @property\s*\n    def _supports_sdpa\(self\) -> bool:\s*\n        \\\"\\\"\\\"Check LLM supports SDPA Attention\\\"\\\"\\\"\s*\n        return self\.language_model\._supports_sdpa', re.MULTILINE)
REPLACEMENT = '    _supports_sdpa = False  # patched for transformers>=4.50'
for ck in ROOT.iterdir():
    if 'openvla' not in ck.name.lower(): continue
    f = ck / 'modeling_prismatic.py'
    if not f.exists(): continue
    src = f.read_text()
    new = PATTERN.sub(REPLACEMENT, src)
    if new != src:
        f.write_text(new); print(f'patched: {f}')
    elif '_supports_sdpa = False' in src:
        print(f'already: {f}')
    else:
        print(f'NO MATCH: {f}')
\""

# 4. 从 cloudml 拷 DPO pair 数据（~5 min, 4 个 .pt 文件各 ~80MB）
mkdir -p datasets
for S in spatial object goal long10; do
  scp -P 4163 root@<cloudml_ip>:/ad-alg/planning-users/liuzhi7/ro_planning/output/h20_rollout_$S/${S}_pairs.pt \
      datasets/openvla_${S}_pairs.pt
done

# 5. 建目录
mkdir -p output logs
```

---

## 🧪 Phase 1: 单卡 sanity（5 min, 确认 pod OK 才跑下面）

```bash
docker run --rm --gpus '"device=0"' \
    -v /e2e-data/users/liuzhi7/openvla:/openvla:ro \
    -v /e2e-data/users/liuzhi7/models/openvla-7b-finetuned-libero-spatial:/model:ro \
    -v /e2e-data/users/liuzhi7/vla-lab/code:/code:ro \
    -v /e2e-data/users/liuzhi7/output/sanity:/output \
    -e PYTHONPATH=/code:/openvla \
    -e MUJOCO_GL=osmesa -e PYOPENGL_PLATFORM=osmesa \
    -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e PYTHONUNBUFFERED=1 \
    vla-lab:v1.0 \
    python -m post_training.eval_libero \
        --base openvla --base_ckpt /model \
        --suites libero_spatial \
        --n_tasks 1 --n_trials_per_task 5 --seeds 42 \
        --output_dir /output
# 期望：4-5/5 success（验证 patch + eval 通路都 OK）
```

---

## 🎯 剩余任务矩阵（按优先级排序）

### ⭐ 批 A — Long10 `max_chunk_len` ablation（paper Factor 2 验证，4 cell）

这是 **paper 现在最缺** 的实验。cloudml 144GB 跑不动（大 chunk OOM），只有 8×H20 96GB 分片或 gradient checkpointing 能跑。本批把 `--max_chunk_len ∈ {220, 300, 400, 520}` 都跑一次。

**注意**：T>220 在 96GB 上仍会 OOM。**所以这批需要先上 gradient checkpointing**（代码改动见文末 Appendix）。如果暂时没空改代码，先跑其他批。

### ⭐⭐ 批 B — OpenVLA × GRPO × 4 suite（paper 新一行，8 cell）

GRPO 是 on-policy，每 step 需要 K rollouts，比 DPO 贵。H20 96GB 单卡跑不动满 K=4，需要 K=2。

**单 cell 时间**：rollout 5 group × 4 candidate × ~2 min = ~40 min + 1000 step GRPO train ~1.5h + eval 50 trial ~1h = **~3.5h/cell**

**8 GPU 并行 4 cell**（一半卡跑 spatial+object，一半跑 goal+long10，各 2 GPU）：**~3.5h 一轮**

### ⭐⭐⭐ 批 C — Spirit v1.5 SFT baseline（paper 第二个 base，4 cell）

Spirit 在 LIBERO 上没官方 ckpt，要从 Qwen3-VL-4B base 出发自训 LoRA。这是新工作量，先跑不急。

### 批 D — Object/Goal DPO seed variance（paper §4.2 noise band）

cloudml 今晚跑 Goal+Long10 两 suite 的 seed 1337+2026，你这边可以同时跑 Object+Spatial 的 multi-seed（补齐 4 suite 都有 3 seed）。

---

## 📋 复制粘贴命令（按批次）

### 批 D（最简单，立刻能跑，用已有 DPO ckpt）

先从 cloudml 拷 Spatial+Object DPO ckpt：
```bash
for S in spatial object; do
  scp -P 4163 root@<cloudml_ip>:/ad-alg/planning-users/liuzhi7/ro_planning/output/h20_dpo_$S/checkpoint-500.pt \
      output/h20_dpo_$S/checkpoint-500.pt
done
```

然后并行起 2 个 seed eval（1 GPU 一个 suite）：
```bash
# GPU 0: Spatial multi-seed
nohup docker run --rm --gpus '"device=0"' \
    -v /e2e-data/users/liuzhi7/openvla:/openvla:ro \
    -v /e2e-data/users/liuzhi7/models:/models:ro \
    -v /e2e-data/users/liuzhi7/vla-lab/code:/code:ro \
    -v /e2e-data/users/liuzhi7/output/h20_dpo_spatial:/dpo_in:ro \
    -v /e2e-data/users/liuzhi7/output/h20_dpo_spatial_eval_multiseed:/output \
    -e PYTHONPATH=/code:/openvla \
    -e MUJOCO_GL=osmesa -e PYOPENGL_PLATFORM=osmesa \
    -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e PYTHONUNBUFFERED=1 \
    vla-lab:v1.0 \
    python -m post_training.eval_libero \
        --base openvla --base_ckpt /models/openvla-7b-finetuned-libero-spatial \
        --lora_ckpt /dpo_in/checkpoint-500.pt \
        --suites libero_spatial \
        --n_tasks 10 --n_trials_per_task 5 --seeds 1337 2026 \
        --output_dir /output \
    > logs/spatial_multiseed.log 2>&1 &

# GPU 1: Object multi-seed
nohup docker run --rm --gpus '"device=1"' \
    -v /e2e-data/users/liuzhi7/openvla:/openvla:ro \
    -v /e2e-data/users/liuzhi7/models:/models:ro \
    -v /e2e-data/users/liuzhi7/vla-lab/code:/code:ro \
    -v /e2e-data/users/liuzhi7/output/h20_dpo_object:/dpo_in:ro \
    -v /e2e-data/users/liuzhi7/output/h20_dpo_object_eval_multiseed:/output \
    -e PYTHONPATH=/code:/openvla \
    -e MUJOCO_GL=osmesa -e PYOPENGL_PLATFORM=osmesa \
    -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e PYTHONUNBUFFERED=1 \
    vla-lab:v1.0 \
    python -m post_training.eval_libero \
        --base openvla --base_ckpt /models/openvla-7b-finetuned-libero-object \
        --lora_ckpt /dpo_in/checkpoint-500.pt \
        --suites libero_object \
        --n_tasks 10 --n_trials_per_task 5 --seeds 1337 2026 \
        --output_dir /output \
    > logs/object_multiseed.log 2>&1 &

wait  # 等 2 个都跑完
```

**时间**：Spatial ~2.5h，Object ~4.2h。一张卡一个 suite，**4.2h 整批完成**。

### 批 B（OpenVLA × GRPO × 4 suite, 8 GPU 并行）

**注意**：首先从 cloudml 拷 pair 数据 + OpenVLA SFT base ckpt，确认代码中 `train_grpo.py` 存在（已有，~400 行）。

GRPO 一个 cell 一张 GPU，4 个 suite 用 4 张 GPU 跑训练+eval；剩 4 张 GPU 留给 rollout collection（若 GRPO 是 on-policy 则不需要预先 pair）。

```bash
# GPU 0-3: 4 个 suite 的 GRPO 训练（--max_chunk_len 180 保证 96GB 不 OOM）
SUITES=(spatial object goal long10)
FULL=(libero_spatial libero_object libero_goal libero_10)
CKPT_DIRS=(openvla-7b-finetuned-libero-spatial openvla-7b-finetuned-libero-object openvla-7b-finetuned-libero-goal openvla-7b-finetuned-libero-10)

for I in 0 1 2 3; do
  S=${SUITES[$I]}; F=${FULL[$I]}; CK=${CKPT_DIRS[$I]}
  nohup docker run --rm --gpus "\"device=$I\"" \
      -v /e2e-data/users/liuzhi7/openvla:/openvla:ro \
      -v /e2e-data/users/liuzhi7/models:/models:ro \
      -v /e2e-data/users/liuzhi7/vla-lab/code:/code:ro \
      -v /e2e-data/users/liuzhi7/output/h20_grpo_$S:/output \
      -e PYTHONPATH=/code:/openvla \
      -e MUJOCO_GL=osmesa -e PYOPENGL_PLATFORM=osmesa \
      -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e PYTHONUNBUFFERED=1 \
      -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      vla-lab:v1.0 \
      python -m post_training.train_grpo \
          --base openvla --base_ckpt /models/$CK \
          --suite $S \
          --output_dir /output \
          --n_rollout_tasks 10 --n_inits_per_task 3 --k_samples 2 \
          --max_chunk_len 180 \
          --batch_size 1 --max_steps 500 --warmup 50 \
          --log_every 10 --save_every 500 \
          --lora_r 16 --lr 3e-5 \
          --grpo_beta 0.1 --grpo_epsilon 0.2 \
      > logs/grpo_$S.log 2>&1 &
done
wait  # ~4h 全部完成（on-policy rollout 是瓶颈）

# GPU 0-3: 4 个 GRPO ckpt 的 eval
for I in 0 1 2 3; do
  S=${SUITES[$I]}; F=${FULL[$I]}; CK=${CKPT_DIRS[$I]}
  LORA=$(ls -t /e2e-data/users/liuzhi7/output/h20_grpo_$S/checkpoint-*.pt | head -1)
  nohup docker run --rm --gpus "\"device=$I\"" \
      -v /e2e-data/users/liuzhi7/openvla:/openvla:ro \
      -v /e2e-data/users/liuzhi7/models:/models:ro \
      -v /e2e-data/users/liuzhi7/vla-lab/code:/code:ro \
      -v /e2e-data/users/liuzhi7/output:/output \
      -e PYTHONPATH=/code:/openvla \
      -e MUJOCO_GL=osmesa -e PYOPENGL_PLATFORM=osmesa \
      -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e PYTHONUNBUFFERED=1 \
      vla-lab:v1.0 \
      python -m post_training.eval_libero \
          --base openvla --base_ckpt /models/$CK \
          --lora_ckpt $LORA \
          --suites $F \
          --n_tasks 10 --n_trials_per_task 5 --seeds 42 1337 2026 \
          --output_dir /output/h20_grpo_${S}_eval \
      > logs/grpo_${S}_eval.log 2>&1 &
done
wait  # ~5h 全部完成（Long10 最慢）
```

**总时间**：~9h 一批完成 4 个 GRPO cells + 3-seed eval。**正好一晚**。

---

## 🔬 批 A（Long10 chunk ablation，需要先改代码）

要跑 T ∈ {220, 300, 400, 520}，就必须先加 **gradient checkpointing** 到 `adapters/openvla.py`。

### 代码改动（Appendix A）

在 `OpenVLAAdapter.__init__` 最后加：
```python
# Enable gradient checkpointing to fit long chunks on 96GB H20
if getattr(self.model, "gradient_checkpointing_enable", None):
    self.model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    self.model.config.use_cache = False
```

改完后显存预期（大致，`use_reentrant=False` 省 ~40%）：
- T=220: ~60 GB（安全）
- T=300: ~115 GB → **还 OOM on 96GB**，需 bs=0.5（不可能）或 ZeRO-offload
- T=520: 不可行

**结论**：96GB H20 只能跑 T=220-280。Long10 ablation 真正需要的 T=520 **必须等 cloudml 144GB 开 grad ckpt 后才能跑**。先不做这批。

### 折中：在 cloudml 上改代码 + 跑 T=300/400

如果你愿意，我直接在 cloudml 修 `adapters/openvla.py` 开 gradient checkpointing，然后再起 T=300 的 Long10 DPO。现在 Goal+Long10 multi-seed 在跑没空 GPU，等它 ~10h 后完。

---

## 🧭 推荐起任务顺序

1. **现在（pod 就绪后 10 min 内）**：起**批 D**（Spatial+Object multi-seed, 4.2h）。简单、有价值、立刻能跑。
2. **批 D 跑起来后（另 6 张卡空闲）**：起**批 B**（GRPO × 4 suite, ~9h）。用 GPU 4-7 训 GRPO，GPU 0-1 继续做批 D，GPU 2-3 闲置（GRPO 需要的 on-policy rollout 会占用）。
3. **明早起来 + cloudml 完成 Goal/Long10 multi-seed**：整合所有 seed，更新 paper §4.2 noise band。
4. **之后**：gradient checkpointing 上 + Long10 T-sweep。或 Spirit SFT 训练。

---

## 📞 状态同步 cheat-sheet

```bash
# 本机看所有 GPU 状态
nvidia-smi

# 看单个任务日志
tail -f logs/grpo_spatial.log

# 检查某 cell 进度
ls -la output/h20_grpo_spatial/

# 一键看 8 张卡占用 + 4 个任务进度
bash vla-lab/scripts/monitor_overnight.sh   # （cloudml 上有这个脚本，可以拷过来）
```

---

## ⚠️ 已知坑

1. **OOM**：`--max_chunk_len 180`（96GB 版）是保守值；`--max_chunk_len 220`（144GB 版）只在 cloudml 能跑
2. **PYTHONUNBUFFERED=1 必须**：否则 log 无输出看似 hang
3. **ckpt 必须先 patch**（Phase 0 step 3）：否则 `_supports_sdpa` 报错
4. **scp 跨 pod 慢**：如果有共享 PVC 直接 symlink 就免 scp
5. **GRPO 在 single-GPU 的 K=4 会 OOM**：用 K=2

---

## 📊 完成后更新位置

完成某批后，把结果 scp 回 cloudml 的 paper asset 位：
```bash
# 在 8×H20 上
for S in spatial object; do
  scp -P 4163 output/h20_dpo_${S}_eval_multiseed/summary.json \
      root@<cloudml_ip>:/ad-alg/planning-users/liuzhi7/ro_planning/output/
done
```

然后我在 cloudml/本机会 pull 下来合并进 paper §4.2。
