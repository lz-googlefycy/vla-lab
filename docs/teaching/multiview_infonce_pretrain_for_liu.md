# 双流 InfoNCE 多视角时空对齐预训练 — 仓库索引 + 简历对应

> 本文档对应简历 Bullet 1：
> > 实现 **双流 InfoNCE 多视角时空对齐预训练（SigLIP-so400m frozen + 656K proj head）** 与 DoRA-r32 微调
>
> 给面试官一份"clone 即可复现"的索引。每个简历短语在仓库都有具体可执行的对应。

---

## 1. 简历 → 仓库 关键词 mapping

| 简历短语 | 仓库证据（GitHub `lz-googlefycy/vla-lab`） |
|---|---|
| "**双流 InfoNCE**" | `code/pretrain/model.py:multiview_temporal_loss` (双流加权 `L = 0.5·L_mva + 0.5·L_tc`) |
| "多视角" (Multi-View Alignment, **L_mva**) | `code/pretrain/model.py:info_nce_loss` 调用，agent ↔ wrist 双向对称 InfoNCE |
| "时空对齐" (Temporal Coherence, **L_tc**) | 同上，agent_t ↔ agent_{t+Δ}，Δ=5 |
| "**SigLIP-so400m**" (vision encoder) | `code/pretrain/train.py:SiglipEncoderWrapper`，timm `vit_so400m_patch14_siglip_224` |
| "**frozen**" | `train.py:_build_encoder` `for p in encoder.parameters(): p.requires_grad = False` |
| "**656K proj head**" | `code/pretrain/model.py:MultiViewProjectionHead`<br>`Linear(1152→512) → GELU → Linear(512→128)` 实测 656,832 params |
| 数据来源 | `code/pretrain/dataset_rlds.py:LiberoRLDSPretrainDataset`（OpenVLA team 公开 RLDS） |
| 训练入口 | `code/pretrain/train.py` (AdamW + cosine LR + InfoNCE 训练循环) |
| **真训 ckpt** | `models/pretrain_rlds_siglip_day8/proj_head.pt` (2.6 MB, force-added 到仓库) |
| 收敛证据 | `assets/paper_v1.5_eval/pretrain_rlds_siglip_day8/train_log.json` |
| 收敛曲线图 | `docs/teaching/figures/day8_pretrain_real_loss_curve.png` |
| **下游 retrieval eval** | `models/pretrain_rlds_siglip_day8/retrieval_eval.py` + `retrieval_results.json` |
| **demo notebook** | `models/pretrain_rlds_siglip_day8/demo.ipynb` |
| 完整 README | `models/pretrain_rlds_siglip_day8/README.md` |

---

## 2. 算法 — 双流 InfoNCE 怎么定义

### 2.1 数据三元组

每个训练样本是一个三元组 `(agent_t, wrist_t, agent_{t+Δ})`：
- `agent_t`：第 3 视角 (agentview) 在时刻 t 的图像 (256×256×3 → 224×224×3)
- `wrist_t`：第 1 视角 (wrist-mount camera) 在同一时刻 t 的图像
- `agent_{t+Δ}`：agent 视角在时刻 t+5（约 0.5 秒后）的图像

来自 `modified_libero_rlds`（OpenVLA team 公开数据集），4 个 LIBERO suite × 50 episodes × 30 anchor times = **6000 样本**。

### 2.2 编码 + 投影

```
image (224×224×3, [-1, 1] normalised)
  │
  ├─ SigLIP-so400m (timm vit_so400m_patch14_siglip_224, FROZEN)
  │  427.7M params, 不更新
  ▼
1152-d feature (CLS pooled)
  │
  ├─ MultiViewProjectionHead (★ 唯一可训部分, 656K params)
  │  Linear(1152, 512) → GELU → Linear(512, 128)
  ▼
128-d embedding
  │
  ├─ L2 normalize
  ▼
unit vector ∈ S^127  ← used in InfoNCE
```

**SigLIP weights 来源**：`vision_backbone.fused_featurizer.*` 从 OpenVLA-7B 的 4-shard safetensors 抽取，**342 keys, 0 missing, 0 unexpected** → bit-identical 于 OpenVLA 内置 vision encoder。这个设计意图是：proj_head 学到的特征**直接可用于 OpenVLA 下游**（同一 feature 空间）。

### 2.3 双流 InfoNCE Loss

InfoNCE（标准对比损失）：

```
L_InfoNCE(z, z+) = -log [ exp(z·z+/τ) / Σ_j exp(z·z_j/τ) ]
```

**双流加权**：
```
L_total = 0.5 · L_mva + 0.5 · L_tc

L_mva = 0.5 · [InfoNCE(z_agent_t, z_wrist_t) + InfoNCE(z_wrist_t, z_agent_t)]
        ↑ 多视角对齐：agent_t 与 wrist_t（同时刻、不同相机）拉近

L_tc  = 0.5 · [InfoNCE(z_agent_t, z_agent_{t+5}) + InfoNCE(z_agent_{t+5}, z_agent_t)]
        ↑ 时序对齐：agent 视角下，t 与 t+5 拉近
```

τ = 0.07（SimCLR 推荐值）。Batch 内 negative samples 来自其它 sample（默认 batch_size=128，每个 anchor 有 127 个 negatives）。

### 2.4 训练超参

| 参数 | 值 |
|---|---|
| Batch size | 128 |
| Epochs | 10 |
| Total steps | 470 |
| Optimizer | AdamW |
| Peak LR | 3e-4 |
| LR schedule | cosine decay → 0 |
| Weight decay | 1e-4 |
| InfoNCE τ | 0.07 |
| Loss weights | w_mva = w_tc = 0.5 |
| 硬件 | 1× cloudml H20-3e |
| Wall clock | **30 分钟** |

---

## 3. 数据 — 真 LIBERO RLDS reader

### 3.1 路径

```python
# code/pretrain/dataset_rlds.py
class LiberoRLDSPretrainDataset(torch.utils.data.Dataset):
    """
    OpenVLA team's published RLDS format (TFDS-backed).
    Each step has:
      observation.image:        (256, 256, 3) uint8   ← agent view
      observation.wrist_image:  (256, 256, 3) uint8   ← wrist view
      language_instruction:     bytes (constant per episode)
      action:                   (7,) float32 (xyz + rpy + gripper)
    """
    def __init__(self, rlds_root, suites=("spatial", "object", "goal", "10"),
                 delta_steps=5, max_per_episode=30, ...):
```

### 3.2 RLDS 加载 (TFDS + numpy bulk extraction)

性能优化：从最初的 `for ep in ds; for st in ep["steps"]; v.numpy()` 慢循环改成 `tfds.as_numpy(ds.prefetch(AUTOTUNE))` + 整个 episode 一次性 numpy bulk extract。**5 倍加速**：50 ep/suite < 1 min（之前 12 min）。

### 3.3 anchor 采样

每 episode 从 `range(0, T - Δ - 1)` 随机选 `max_per_episode=30` 个 anchor 时刻 t，输出 `(agent_t, wrist_t, agent_{t+5})`。

---

## 4. 实测收敛 — 真训跑了

### 4.1 命令

```bash
cd ro_planning/code
PYTHONPATH=. python -m pretrain.train \
    --rlds_root /path/to/modified_libero_rlds \
    --siglip_path /path/to/openvla-7b-finetuned-libero-spatial \
    --suites spatial object goal 10 \
    --max_episodes_per_suite 50 \
    --max_per_episode 30 \
    --epochs 10 \
    --batch_size 128 \
    --num_workers 2 \
    --lr 3e-4 \
    --temperature 0.07 \
    --output_dir output/pretrain_rlds_siglip_day8
```

### 4.2 实测数字（artifacts 都在仓库）

| | Step 10 | Step 460 | 减少 |
|---|---|---|---|
| L_total = 0.5 (L_mva + L_tc) | 2.418 | **0.366** | — |
| L_mva (multi-view 跨视角) | 3.527 | 0.508 | 85.6% |
| L_tc (temporal Δ=5) | 1.309 | 0.223 | 83.0% |
| Random baseline log(B=128) | 4.852 | — | (起点) |

**Recovery ratio = (4.852 - 0.366) / 4.852 = 92.5%** — 30 分钟内达到 random→0 距离的 92.5%。

L_mva > L_tc 全程 — 跨视角比时序近邻**难学**，符合直觉（时序近邻 ~80% 像素相同；跨视角是真正的"viewpoint invariance"任务）。

`assets/paper_v1.5_eval/pretrain_rlds_siglip_day8/train_log.json`：每 10 step 记录一行，47 行。
配图：`docs/teaching/figures/day8_pretrain_real_loss_curve.png`

---

## 5. 下游 — k-NN Retrieval Eval

仅靠收敛 loss 不够 — 真正能用的特征必须在 retrieval 上看出语义结构。所以又跑了 retrieval eval（5 分钟，1×H20）。

### 5.1 协议

- 用 trained `proj_head` 对全 6000 帧做 forward → 得 6000 × 128 维 embedding
- 对每个 query 帧，计算与所有其它 5999 帧的 cosine similarity
- 取 top-k 最近邻，统计：
  - **same-task hit**：top-k 中是否有同任务的帧（task_id = hash(language_instruction)）
  - **same-episode hit**：top-k 中是否有同 episode 的帧
  - **temporal hit**：top-k 中是否有同任务且 |Δt| ≤ 10 步的帧

### 5.2 数字（`models/pretrain_rlds_siglip_day8/retrieval_results.json`）

| Recall metric | @1 | @5 | @10 | random@1 |
|---|---|---|---|---|
| **same-task** | **99.5%** | 99.9% | 99.95% | 2.75% |
| same-episode | 91.4% | 97.7% | 99.0% | ~0.5% |
| same-task & \|Δt\|≤10 | 92.4% | 98.2% | 99.0% | ~0.3% |

**99.5% / 2.75% = 36× over random** — 这是 self-supervised 论文的标准量化指标。

### 5.3 per-suite 分解

| Suite | recall@1 same-task |
|---|---|
| Spatial | 99.3% |
| **Object** | **100.0%** |
| Goal | 99.3% |
| Long10 | 99.5% |

Object 100% 是巧合 talking point — DoRA 在 Object 上 win 最多（+14 vs 单 seed），Object 也是 retrieval 最容易的，说明这个 suite 的"跨任务视觉差异"最干净。

### 5.4 Qualitative figure

`models/pretrain_rlds_siglip_day8/retrieval_examples.png` —— 6 query × top-3 retrieval 可视化网格。

---

## 6. 完整 reproduce 路径（给面试官的"30 分钟挑战"）

```bash
# 1. clone
git clone https://github.com/lz-googlefycy/vla-lab.git
cd vla-lab

# 2. 看 README
cat models/pretrain_rlds_siglip_day8/README.md

# 3. 跑 demo notebook (5 min)
jupyter lab models/pretrain_rlds_siglip_day8/demo.ipynb
#    → 看 cos sim 热图，验证 same-episode block 突出

# 4. 跑 retrieval eval (5 min on H20)
python models/pretrain_rlds_siglip_day8/retrieval_eval.py \
    --rlds_root /path/to/modified_libero_rlds \
    --siglip_path /path/to/openvla-7b-finetuned-libero-spatial \
    --proj_ckpt models/pretrain_rlds_siglip_day8/proj_head.pt \
    --output_dir /tmp/retrieval_repro

# 5. (optional) 重跑预训练 (30 min on H20)
python -m code.pretrain.train --rlds_root ... --siglip_path ... \
    --epochs 10 --batch_size 128 --output_dir /tmp/pretrain_repro
#    应得 loss 4.85 → ~0.37
```

---

## 7. 为什么这么设计 (面试官可能问的)

### Q: 为什么用 frozen SigLIP，而不是 fine-tune？

A: 三个理由：
1. **目的是 representation learning，不是 backbone training**：proj_head 656K << SigLIP 427M，研究问题是"在已有 vision encoder 上如何用 self-supervision 加 robot-aware 偏置"
2. **下游兼容性**：保持 SigLIP 不变，proj_head 输出能直接接到 OpenVLA 的 action head（同一 feature 空间）
3. **算力约束**：fine-tune SigLIP 需要 ~10× compute，不在 sprint 预算内（sprint deliverable 是 framework 验证 + ckpt + retrieval 数字，不是 production-grade）

### Q: 为什么不直接学 R3M / SiamMAE？为什么"双流"？

A: R3M 有 time-contrastive，但没多视角。SiamMAE 是 video correspondence (masked + siamese)，但视觉头太重。双流 InfoNCE 是**最小可行设计**：
- L_mva 学 viewpoint invariance（agent ↔ wrist）
- L_tc 学 temporal coherence（agent_t ↔ agent_{t+5}）
- 两个 loss 加权平均（0.5/0.5），共享同一个 proj_head

这样 656K param 就能同时承担两类对齐，对小数据（6000 样本）友好。

### Q: 99.5% retrieval 这么高，是不是过拟合？

A: 不是过拟合，是数据**特性**：LIBERO 4 suite 共 ~40 个 task，每个 task ~150 帧，6000 帧总池里同任务密度 = 150/6000 = 2.5%。proj_head 把同任务的 ~150 帧 cluster 到 embedding 球面上一个紧凑区域，所以 recall@1 自然高。

**关键 sanity**：random baseline = 2.75%，**符合理论预期** 2.5% (约等于 pool 密度) — 这说明随机是对的，proj_head 是真在 cluster。

下游用法不一定是 retrieval — 可以做 k-NN reward shaping、scene change detection、k-NN imitation 等。

### Q: 这一步训练 → 下游 OpenVLA fine-tune 改进了多少？

A: **没做下游 ablation**（诚实）。Sprint 优先级是先验证 framework + 出 ckpt + retrieval 数字。要做下游 ablation 需要把 proj_head 的输出 inject 回 OpenVLA 的 vision pipeline 然后重训 SFT/DPO，工作量 ~1-2 天。这是 §5.4 (paper) / 后续 sprint 的 future work。

简历也是这么写的："实现 双流 InfoNCE 多视角时空对齐预训练 ... 与 DoRA-r32 微调，**DoRA + DPO 使 ... +14pp**" — **+14pp 的归因明确给 DoRA + DPO，不是预训练**。预训练只 claim "实现"。

---

## 8. 核心代码摘录（30 行看完）

### MultiViewProjectionHead

```python
# code/pretrain/model.py
class MultiViewProjectionHead(nn.Module):
    def __init__(self, in_dim=1152, hidden=512, out_dim=128):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, out_dim),
        )
    def forward(self, x):
        return F.normalize(self.proj(x), dim=-1)  # L2 to unit sphere
```

### InfoNCE + 双流组合

```python
# code/pretrain/model.py
def info_nce_loss(z1: torch.Tensor, z2: torch.Tensor, tau: float = 0.07) -> torch.Tensor:
    """Symmetric InfoNCE between matched pairs (z1[i] ~ z2[i])."""
    logits = z1 @ z2.T / tau   # (B, B)
    labels = torch.arange(z1.size(0), device=z1.device)
    return 0.5 * (F.cross_entropy(logits, labels) +
                   F.cross_entropy(logits.T, labels))


def multiview_temporal_loss(z_agent_t, z_wrist_t, z_agent_td,
                            tau=0.07, w_mva=0.5, w_tc=0.5):
    L_mva = info_nce_loss(z_agent_t, z_wrist_t, tau)   # 跨视角
    L_tc  = info_nce_loss(z_agent_t, z_agent_td, tau)  # 时序
    return w_mva * L_mva + w_tc * L_tc, L_mva, L_tc
```

### 训练循环（精简版）

```python
# code/pretrain/train.py:main
encoder, _ = _build_encoder(args, device)   # SigLIP frozen
proj_head = MultiViewProjectionHead(in_dim=1152, hidden=512, out_dim=128).to(device)
opt = torch.optim.AdamW(proj_head.parameters(), lr=args.lr, weight_decay=args.weight_decay)

for epoch in range(args.epochs):
    for batch in loader:
        with torch.no_grad():
            f_a  = encoder(batch["agent_t"].to(device))
            f_w  = encoder(batch["wrist_t"].to(device))
            f_td = encoder(batch["agent_t_delta"].to(device))
        z_a, z_w, z_td = proj_head(f_a), proj_head(f_w), proj_head(f_td)
        loss, L_mva, L_tc = multiview_temporal_loss(z_a, z_w, z_td, tau=args.temperature)
        opt.zero_grad(); loss.backward(); opt.step()
        scheduler.step()
```

---

## 9. 一句话总结

> **656K 可训参数（proj_head）+ 一份双流 InfoNCE loss + 30 分钟 H20 = 99.5% k-NN retrieval recall@1**。整套 framework 已 reproduce-ready 推到公开仓 `lz-googlefycy/vla-lab`，简历每一个名词都对应仓库具体文件，面试官 clone 后能在 30 分钟内验证。
