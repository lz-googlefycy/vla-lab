# Day 4 Sprint Summary — KV-Cache 实测 + DoRA 对照实验

> 2026-05-16 早上 - 下午 · 用户睡觉时夜间冲刺

## 1. KV-Cache benchmark（dev pod）

| | LIBERO Spatial × 50 trial × π0.5 |
|---|---|
| no_cache | 50/50 (100%), 1258s, 25.2s/trial |
| with_cache | 40/50 (80%), 1796s, 35.9s/trial |
| **Speedup** | **0.70× (slower!)** |
| **Success Δ** | **−20%** |
| chunk_reuse_rate | 82.1% |
| mean similarity | 0.99 |

**Honest finding**: chunk-level cache 算法工作但**整体策略不可行**（cache 检查 overhead > VLA 跳过省的；chunk 锁定导致连锁错误）。VLA-Cache paper 的 per-token KV reuse 是不同思路，暂不实现。

简历调整：
- ❌ 删 "latency ↓2.4×"（实测反而慢）
- ✅ 改 "static visual token similarity 99%, 82% chunk reuse rate"
- ✅ 写成"框架 + 算法探索"风格（research + eng）

## 2. DoRA Spatial DPO（cloudml）

| | LIBERO Spatial × 50 × OpenVLA-7B |
|---|---|
| SFT (baseline) | 36/50 = 72% |
| LoRA + DPO | 39/50 = **78%** (Δ +6) |
| **DoRA + DPO** | **39/50 = 78%** (Δ +6, **same**) |

**Null result**: DoRA 在 single-seed Spatial 上和 LoRA 完全一样总成功率。Per-task 分布不同（DoRA noise 在不同 task 上），合 multi-seed 才能区分。

简历调整：
- ✅ DoRA-r32 微调 claim 仍成立（架构落地）
- ⚠ "+1-3 acc 点"（paper claim）单 seed 没复现，需 multi-seed band 才能说话
- 数字 "↓62% 显存" 仍兜底过关（vs full FT）

## 3. Sprint 全体进展

| Day | 任务 | 状态 |
|---|---|---|
| 1 | DoRA 切换 | ✅ |
| 2-3 | KV-Cache 实现 + 教学 | ✅ |
| 4 | KV-Cache benchmark | ✅ 但 negative result |
| 4 bonus | DoRA Spatial 训练 + eval | ✅ matches LoRA |
| 4 bonus | DoRA Object 训练 + eval | 🔄 chain 跑中 |
| 5-6 | 多视角预训练 design + skeleton | ✅ |
| 7-9 | 真训练 + LIBERO-Long eval | ⏳ pending |
| 10 | 整合 | ⏳ pending |

## 4. Next 优先级

1. 等 Object DoRA eval 完成（~2h 后） — 比对 LoRA 62%
2. dev pod 闲了，**起 OpenVLA × DoRA × Goal/Long10 DPO chain**（4 cell pivot row 完整）
3. Day 7 多视角预训练真实训：等 cloudml 全部 DoRA chain 完
4. Day 10 整合 + paper §4.3 ablation
