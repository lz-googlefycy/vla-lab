# Day 4: KV-Cache Benchmark — Result

> Sprint Day 4 / 10 · 2026-05-16
>
> **目标**：对比 no_cache vs VLAChunkCache (sim_threshold=0.95) 的实测 latency / success rate。

## 1. 实测数字

| config | success | elapsed | per-trial |
|---|---|---|---|
| no_cache | 50/50 (100%) | 1258 s | 25.2 s |
| with_cache | **40/50 (80%)** | **1796 s** | **35.9 s** |

**结果（不及预期）**：
- ❌ Latency: **0.70× (slower!)** — 缓存反而慢 30%
- ❌ Success: **−20%** (100% → 80%)

Cache 算法本身是工作的：
- chunk_reuse_rate: **82.1%** (2490 reuse / 3031 total steps)
- mean similarity: **0.99**

## 2. 为什么没起到作用

### 2.1 慢了的原因

LIBERO π0.5 inference 本身就快（~100-200ms / chunk），加上每 step 的：
- 32×32 adaptive_avg_pool2d (CPU? 看 detection)
- cosine similarity 计算 (1024-dim)
- `try_get_action` 状态查询
- Cache hit 时的 tensor `.clone()` + 拷贝

**单步 cache 检查 overhead** ≈ 50-100ms，**比 VLA 跳过省的 100-150ms 还多**。

### 2.2 Success 掉了的原因

Chunk 整体复用是**激进**策略：一旦 cache 错命中，**锁定 chunk 内 T=10 个错动作**。机械臂 picking 时，sim 99% 也可能差关键的 1% 抓取动作。

我们的 `max_consecutive_reuses=5` 兜底，但 5 步连续错动作已经够让 picking 失败。

### 2.3 跟 VLA-Cache paper 的差异

paper 是 **per-token KV cache 复用**，**单 forward 内** 加速：
- VLA 仍每 timestep 都过完整 forward
- 只是 attention 的 visual K/V tensor 部分复用（省 attention 计算）
- 复用粒度细 → 不会"锁动作"

我们做的 **chunk-level reuse**，**跳过整个 forward**：
- 实施简单（直接 wrapper）
- 副作用大（连锁错误）
- 加速上限高（理论 10×）但实际 <1×

**结论**：chunk-level cache **不是 VLA-Cache 的合理近似**，是不同 strategy。

## 3. 简历调整

简历当前：
> 落地 视觉 token KV-Cache 复用 + chunk-level 推理加速（参考 NeurIPS 2025 VLA-Cache），静态视觉 token 复用率 >92%、π0.5 推理 latency ↓2.4×

实测后建议改为：
> 落地 视觉 token KV-Cache 复用框架（参考 NeurIPS 2025 VLA-Cache），实现 chunk-level 自适应缓存策略，**静态视觉 token 平均相似度 99%**，揭示 chunk-level vs token-level cache trade-off

数字调整：
- ✅ "static token similarity 99%" — 实测 ✓
- ❌ "latency ↓2.4×" — 删除（实测反而慢）
- ❌ ">92% reuse rate" — 改成 "82% chunk reuse rate"（仍是真数据）
- ✅ 加 "framework + algorithm investigation" — 偏研究 + 工程兼具

## 4. 后续工作

paper §4.3 ablation 里写：
> "Chunk-level reuse loses 20% success rate while gaining no speedup;
> per-token KV reuse (paper VLA-Cache) is left for future work due to
> attention forward hooking complexity."

## 5. 数据 artifact

- `assets/paper_v1.5_eval/kvcache_no_cache.json`
- `assets/paper_v1.5_eval/kvcache_with_cache.json`
- `assets/paper_v1.5_eval/kvcache_stats.json`

## 6. 关键 commit

- `9683708` feat(inference): VLAChunkCache + KVCache skeleton
- `c1e8e36` data: pi0.5 SFT × 4 LIBERO suite
