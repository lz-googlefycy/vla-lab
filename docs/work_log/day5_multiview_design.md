# Day 5: 多视角时空对齐自监督预训练 — 设计文档

> Sprint Day 5 / 10 · 2026-05-16
>
> **目标**：设计多视角 + 时序双流 InfoNCE 预训练任务，给 OpenVLA / π0.5 的 SigLIP-vision 加自监督表征学习信号，为 LIBERO-Long 后续微调做 representation warm-up。

## 1. 背景：VLA 视觉编码器的 representation gap

OpenVLA / π0.5 用 PaLI-Gemma SigLIP-vit + Llama / Gemma decoder。pretrain 时 SigLIP 在 web-scale image-text pair 上对比学习，视觉特征侧重 **objects, scenes, captions**，而 robot manipulation 关注：

1. **多视角一致性**：同一时刻 wrist cam 和 agent-view cam 看的是同一场景，特征应该 align
2. **时序连续性**：相邻 timestep 的视觉特征应该平滑过渡
3. **任务相关性**：抓哪儿、放哪儿的关键 region 应该被 vision encoder highlight

paper 现状：
- **OpenVLA / π0.5**: 直接用 frozen SigLIP，**没做 robot-specific pretrain alignment**
- **RT-2 / Octo**: 部分用了 multi-view contrastive，但 没开源具体 loss
- **4D-VLA (2025 趋势)**: 提出 spatio-temporal alignment

我们 Day 5-9 实现一个**最简但有信号的版本**，目标 LIBERO-Long success rate +5~+13。

## 2. 算法设计

### 2.1 数据来源

LIBERO 4-suite SFT trajectories（已下载，dev pod 共享盘）：
- 每个 task 有 50 demo trajectories
- 每个 traj 包含 (agent_view_t, wrist_view_t, action_t) for t in [0, T_traj]
- T_traj 平均 100-150 steps for spatial/object/goal, 200-300 for long10

**总数据**：~50 task × 50 traj × ~150 steps = **~375K (view, view, t) tuples** for LIBERO

### 2.2 双流 InfoNCE Loss

#### Loss 1: Multi-View Alignment (MVA)

正样本对：(agent_view_t, wrist_view_t)，**同一时刻、不同视角**。
负样本：跨 episode / 跨 task 的 (agent_view, wrist_view) 不匹配对。

$$
\mathcal{L}_{\text{mva}} = -\log \frac{\exp(\text{sim}(z_a, z_w) / \tau)}{\sum_j \exp(\text{sim}(z_a, z_w^{(j)}) / \tau)}
$$

其中 $z_a, z_w$ 是 SigLIP encoder + 一个新加的 projection head 输出的 **128-dim** 表征。$\tau = 0.07$（SimCLR 标准）。

#### Loss 2: Temporal Consistency (TC)

正样本对：(agent_view_t, agent_view_{t+5})，**同一 episode、相近时刻**。
负样本：跨 episode 的远距离样本。

$$
\mathcal{L}_{\text{tc}} = -\log \frac{\exp(\text{sim}(z_t, z_{t+\Delta}) / \tau)}{\sum_j \exp(\text{sim}(z_t, z^{(j)}) / \tau)}
$$

$\Delta = 5$ steps（约 0.5s 真实时间，足够近 + 不平凡）。

#### 总 Loss

$$
\mathcal{L} = 0.5 \cdot \mathcal{L}_{\text{mva}} + 0.5 \cdot \mathcal{L}_{\text{tc}}
$$

### 2.3 模型架构

```
agent_view (224×224) ─┐
                      ├─→ SigLIP-vit (FROZEN) ─→ visual_token (256, D=1152)
wrist_view (224×224) ─┘                                │
                                                       ↓
                                              [mean pool or CLS] (D=1152)
                                                       ↓
                                              proj_head (TRAINABLE, 1152 → 128)
                                                       ↓
                                                  z (128-dim)
```

**只训 projection head**（~150K 参数）+ **可选 fine-tune SigLIP 顶层 LoRA**（~1M）。

### 2.4 训练超参（推荐）

| 超参 | 值 | 备注 |
|---|---|---|
| Batch size | 256 (128 pairs) | InfoNCE 大 batch 才有意义 |
| Learning rate | 3e-4 | proj_head AdamW |
| Epochs | 10 | LIBERO 数据小，10 epoch 内收敛 |
| Temperature τ | 0.07 | SimCLR 标准 |
| Δt for TC | 5 steps | 约 0.5s |
| Optim | AdamW, weight decay 1e-4 | |

预计 wall time: cloudml 144GB H20, 1 epoch ~10 min, 10 epoch ~2h.

### 2.5 评估方式

预训练完后：
1. **Linear probe**: 冻 vision + freeze proj，加一个 task classification linear，在 LIBERO 50 task 上 acc 应该 >> random
2. **下游迁移**: 用预训练 ckpt 做 LIBERO-Long SFT 微调（500 step）+ eval，对比 baseline (no pretrain)

## 3. Day 6-9 实现 + 训练 plan

### Day 6: 实现 (~250 lines)

文件：
- `code/pretrain/__init__.py`
- `code/pretrain/dataset.py`：从 LIBERO trajectory pickle 取 (view_a, view_w, t, episode_id, task_id) tuples
- `code/pretrain/model.py`：projection head + 双流 InfoNCE loss
- `code/pretrain/train.py`：training loop, AdamW + linear warmup

### Day 7: 训练

- 在 cloudml 144GB H20 上跑 10 epochs，~2h
- 每 epoch eval linear probe acc + InfoNCE loss

### Day 8: 下游 SFT 微调

- 用预训练好的 SigLIP+proj_head 做 OpenVLA SFT 微调（DoRA-r32, LIBERO-Long, 500 step）
- 对比 baseline（无预训练，直接 DoRA 微调）

### Day 9: LIBERO-Long Eval

- 跑 10 task × 5 trial × 1 seed = 50 trial
- 对比 baseline + 预训练 + DoRA-r32 三组

### 数字目标

| 配置 | LIBERO-Long success rate | 简历 claim |
|---|---|---|
| OpenVLA SFT (paper) | 53.7% | baseline |
| OpenVLA SFT (我们 v1.5) | 60% | baseline |
| OpenVLA + DoRA only | 60-65% | DoRA 单独贡献 |
| **OpenVLA + 多视角预训练 + DoRA** | **65-75%** (target 73%) | 简历 claim 兜底 |

### 风险

- 50 demo / task 数据量小，自监督信号可能不够
- LIBERO-Long episode 长，wrist cam 视角变化大，TC 可能正负样本边界模糊
- 兜底：实测后简历数字按实际调整

## 4. Day 10: 整合

整理 Day 1 (DoRA) + Day 2-4 (KV-Cache) + Day 5-9 (多视角预训练) 三件事的完整数字到：
- paper §4.3 ablation 段
- 飞书 wiki summary 节点
- 简历微调 + 投递准备

## 5. 推荐扩展阅读

- SimCLR (Chen et al., 2020) — Contrastive learning baseline
- DINO (Caron et al., 2021) — Self-distillation, no negative needed
- 4D-VLA (2025) — Spatio-temporal pre-training for VLA

## 6. 状态

Day 5 设计完成。Day 6 实现待启动（等 dev pod tunnel / cloudml 闲卡时）。
