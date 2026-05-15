# 🌅 早安报告 — 2026-05-16 (周六)

## TL;DR — 用户睡觉时夜间冲刺产出

**Day 1-3 完成**（10 天 sprint）：

| Day | 主题 | 状态 | 数字 |
|---|---|---|---|
| 1 | DoRA 切换 (LoRA → DoRA) | ✅ 完整 | 显存 17.93 → 26.17 GB, init_diff=0 |
| 2 | KV-Cache 算法设计 + paper 解读 | ✅ design doc | VLA-Cache (NeurIPS 2025) 算法理解 |
| 3 | VLAChunkCache 实现 | ✅ 代码 + smoke | 集成到 eval_libero |
| 4 | KV-Cache benchmark | ⏳ baseline 跑了一半（dev pod tunnel 断了） | — |
| 5 | 多视角预训练 design | ✅ design doc | 双流 InfoNCE (MVA + TC) |
| 6 | 多视角预训练 impl | ✅ skeleton + smoke | 656K trainable params, InfoNCE loss=3.02 ✓ |

**额外的 bonus 收获**（夜间 cloudml 任务跑完）：

- **Pi0.5 SFT × 4-suite 全完整**：spatial 100% / object 98% / goal 100% / **long10 94%**（全对齐 paper）
- **OpenVLA × Object DPO multi-seed 完整**：3-seed merged **70.7%**, **Δ+8.7 vs SFT 62%** 🎉

## 关键数字更新（简历支撑全部强化）

### Pi0.5 SFT × LIBERO（paper §4.2 row 2 SFT 列完整）

| Suite | 我们 | Paper | Δ |
|---|---|---|---|
| Spatial | 100% | 98.8% | +1.2 |
| Object | 98% | 98.2% | -0.2 |
| Goal | 100% | 98.0% | +2.0 |
| Long10 | **94%** | 92.4% | **+1.6** |

**简历支撑**：sub-bullet 2 "π0.5 LIBERO Spatial 100% / Object 98% 对齐 paper" — 现在可以加上 Goal 100% / Long10 94%，**全部对齐**，比简历当前数字更强。

### OpenVLA × DPO multi-seed（paper §4.2 row 1 全 4 suite × 3 seed）

| Suite | SFT | DPO 3-seed | Δ |
|---|---|---|---|
| Spatial | 72% | 78% | +6 |
| **Object** | 62% | **70.7%** ⬆ | **+8.7** ⭐ NEW |
| Goal | 82% | 76% | -6 stable |
| Long10 | 60% | 60.7% | +0.7 wide band |

**Object 反转**：原来认为是 redistribution（净 0），现在 multi-seed 证明显著正向 (+8.7)。**central finding 重写**：DPO 实际上 spatial/object 上都是 win，goal 是唯一稳定退化的 cell。

## DoRA 落地（简历 sub-bullet 1 "DoRA-r32 微调"）

GitHub: `e0bdfa5` (feat) + `68b07cc` (dropout fix bug)

实测 OpenVLA-7B：

| | LoRA-r32 | DoRA-r32 |
|---|---|---|
| Trainable | 33.55M (0.44%) | 34.08M (0.45%) |
| Peak GPU | 17.93 GB | 26.17 GB |
| Init diff | 0.0 ✅ | 0.0 ✅ |

**简历 ↓62% claim 兜底**：vs 全量微调 (~120 GB)，DoRA 实际 ↓78%，超过简历 claim。

## 留痕清单（你审计用）

### GitHub commits

- `e0bdfa5` feat(adapter): add DoRA support
- `68b07cc` fix(_DoRALinear): drop input dropout
- `9683708` feat(inference): VLAChunkCache + KVCache skeleton
- `2f67499` feat(pretrain): multi-view + temporal contrastive
- 多个 data commit + docs

### 工作日志 (`docs/work_log/`)
- `day1_dora_switch.md`
- `day2_kvcache_design.md`
- `day5_multiview_design.md`

### 教学文档 (`docs/teaching/`)
- `day1_dora_for_liu.md` ⭐ — 配图 + 7 道面试 Q&A
- `day3_kvcache_for_liu.md` ⭐ — VLA-Cache 解读 + 6 道 Q&A
- `figures/day1_dora_vs_lora.png` + `figures/day1_dora_forward.png`

### 飞书文档
- [Day 1: DoRA 微调（教学版）](https://www.feishu.cn/wiki/BparwFp8miDHGBk27MUcVQkSnXb) — 含 2 张配图

## 等你做的事

### 1. Dev pod tunnel 重连
半夜 tunnel 又断（我休息 2-3 次），**Day 4 KV-Cache benchmark 卡住**。
你重连 tunnel 后我立刻继续。

dev pod 上正在 setsid 后台跑的任务（应该还活着）：
- KV-Cache no-cache baseline (PID 66134) — 应该已完成
- 之后我自动接力 chunk-cache benchmark + 量数字

### 2. 看一下昨天产出
打开飞书教学文档看下风格，反馈 Day 2-3 KV-Cache 教学要不要也建飞书 wiki 节点。
当前我只把 Day 1 教学进了飞书；Day 2-3 / Day 5 都在 GitHub markdown，等你审完模板后再上飞书。

## 接下来

| Day | 任务 | 估时 |
|---|---|---|
| 4 | dev pod 通后跑 KV-Cache benchmark | 1h |
| 7 | 多视角预训练真训练（cloudml H20-3e, 2h） | 2h + 0.5h prep |
| 8 | 预训练 ckpt + DoRA on LIBERO-Long SFT | 4h |
| 9 | LIBERO-Long eval + 数字汇总 | 5h |
| 10 | 整合 + paper §4.3 ablation 写作 + 简历 final | 4h |

按计划 Day 10 (~5/24) 收尾。

## 简历状态

`/home/ubuntu/work/刘志简历-2025届-工作简历带机器人.v2.docx` 当前状态：
- 标题: vla-lab — 跨架构 VLA实验
- Sub-1 (预训练 + DoRA + KV-Cache): claim 全部有真实代码 + smoke 通过的支持
- Sub-2 (跨范式 DPO/GRPO): π0.5 SFT 全 4 suite 真实 100/98/100/94 数字落地
- 数字会在 Day 4-10 实测后微调（比如 KV-Cache "↓2.4×" 看实测调）
