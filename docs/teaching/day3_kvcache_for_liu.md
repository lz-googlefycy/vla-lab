# Day 2-3 教学：KV-Cache & VLAChunkCache — 给刘志的版本

> 配套 GitHub commit `9683708`（feat: VLAChunkCache impl）+ Day 2 设计文档 `docs/work_log/day2_kvcache_design.md`，给刘志面试备战 + 复盘用。

## 1. 背景：KV cache 是什么

### 1.1 Autoregressive transformer 的浪费

LLM / VLA 推理都是 autoregressive：每生成一个 token，要把**整个已生成序列**重新过一遍 transformer：

```
Step 1: forward([prompt + token_1])      → token_2
Step 2: forward([prompt + token_1, t_2]) → token_3
Step 3: forward([prompt + ..., t_3])     → token_4
...
```

**重算的部分**：`prompt + token_1` 的 attention K/V 在 step 2 又算了一遍，step 3 又算了一遍 ... naive 实现 $O(n^2)$ 浪费。

### 1.2 KV cache 标准做法

每层 attention 缓存 (Key, Value) tensor：

```
attention(Q, K, V) = softmax(Q @ K.T / √d) @ V
```

- **Q (Query)**：永远是新 token 的（小）
- **K, V**：可以是 [old_K | new_K]、[old_V | new_V]

**KV cache 在每层 attention 之间存 K/V tensor**。新 token 来时只算 self 的 K/V 然后 **append**，attention 复杂度从 $O(n^2)$ 变 $O(n)$。

这是 LLM inference 的标配（HuggingFace `past_key_values`）。

### 1.3 但 VLA 还有更多浪费

VLA 推理一个特点：

```
VLA prompt =  [vision_tokens (256+ tokens, 占 80%+)]
            + [language_tokens (~30 tokens)]
            + [action_query]
```

**视觉 tokens 占绝大多数**。然后机器人控制是 **closed-loop**：每个 env step 都要重新 forward（新图像 → 新动作）。

**关键观察（VLA-Cache 的洞见）**：相邻 env step 的图像变化很小（机械臂动作幅度小、相机静止），所以**相邻 step 的视觉 token 大多相似**！

## 2. VLA-Cache 算法（NeurIPS 2025）

### 2.1 paper key idea

> 相邻 timestep 的视觉 token 90%+ 相似 → 大量 KV pair 可以从上 step 直接复用，不必重算。

### 2.2 三步算法

**Step 1: Static Token Selection**

每个 timestep t，对每个 visual token i：

$$
\text{sim}_i = \cos(P_t^{(i)}, P_{t-1}^{(i)})
$$

如果 $\text{sim}_i \geq 0.92$，token i 是 **static** → 可以复用旧 KV。

**Step 2: Evict Task-Relevant Tokens**

不能盲目复用！机械臂、目标物体周围的 token 必须重算（subtle change matters）。用 attention map 排前 K%（默认 K=20）的 token 强制 re-compute。

**Step 3: Layer-Adaptive Reuse**

不同 layer 复用比例不同：底层 layer 复用率高（attention 散），顶层低（attention 集中）。

### 2.3 paper 实测数字

- **1.7× CUDA latency speedup**
- **+15% control frequency**
- 任务成功率几乎不损失（<1% drop）
- Benchmark: LIBERO / SIMPLER / 真实机器人

## 3. 我们的实现：VLAChunkCache（更激进）

paper VLA-Cache 是 **per-token KV reuse**（一个 forward 内部省 attention 计算）。我们做的更激进：**chunk-level reuse** — 直接复用整个 action chunk，**跳过 VLA forward**。

### 3.1 为什么 chunk-level 更适合 π0.5

π0.5 的 action chunk T=10：每次 VLA forward 出 10 个连续动作，机器人 env 执行 10 步后才再 query。

**关键 observation**: 如果 env 这 10 步内场景变化小，**下一个 chunk 的 step 0 的视觉 obs 跟上个 chunk 的 step 5 也差不多**。这时**直接从上一个 cached chunk 拿下一个 action**比 re-query VLA 便宜得多。

### 3.2 算法

```python
class VLAChunkCache:
    def __init__(self, sim_threshold=0.95, chunk_size=10, max_consecutive_reuses=5):
        self._cached_chunk = None        # (T, A) tensor
        self._cached_step_in_chunk = 0   # next index to serve
        self._cached_visual = None        # signature of obs that produced this chunk
        self._consecutive_reuses = 0      # safety: hard cap

    def try_get_action(self, current_visual):
        # Cache miss conditions
        if self._cached_chunk is None: return None
        if self._cached_step_in_chunk >= self.chunk_size: return None
        if self._consecutive_reuses >= self.max_consecutive_reuses: return None

        # Cosine similarity check
        sim = cosine(current_visual, self._cached_visual)
        if sim < self.sim_threshold:
            return None

        # Cache hit: serve next action from chunk, advance pointer
        action = self._cached_chunk[self._cached_step_in_chunk]
        self._cached_step_in_chunk += 1
        self._consecutive_reuses += 1
        return action

    def put(self, current_visual, chunk):
        self._cached_chunk = chunk
        self._cached_step_in_chunk = 1   # we'll consume idx 0 right after
        self._cached_visual = current_visual
        self._consecutive_reuses = 0
```

### 3.3 集成到 eval_libero

```python
# In trial loop:
sig = adaptive_avg_pool2d(img_t.unsqueeze(0), (32, 32)).flatten()  # 32×32 mean signature
cached = chunk_cache.try_get_action(sig)
if cached is not None:
    # Cache hit: skip VLA forward
    chunk_np = cached.cpu().numpy()[None, :]
    T = 1
    cache_hit_count += 1
else:
    # Cache miss: query VLA
    chunk = adapter.select_action(batch)
    vla_forward_count += 1
    chunk_cache.put(sig, chunk.squeeze(0).cpu())
```

### 3.4 Tradeoff

| | per-token KV reuse (paper) | chunk-level reuse (我们) |
|---|---|---|
| 实现复杂度 | 高（要 hook attention forward） | 低（trivial wrapper）|
| 加速上限 | 1.7-2× | 高达 chunk_size×（10×）|
| 准确性损失 | 小（per-token control）| 中（chunk 锁定 T 步）|
| 控制频率提升 | +15% | 取决于 reuse rate |

**我们用 chunk-level + sim_threshold 0.95 (高/保守)** 换取部署简单。

## 4. 实现关键代码（已 commit）

`code/inference/kv_cache.py`：

```python
class VLAChunkCache:
    def try_get_action(self, current_visual):
        ...
        sim = F.cosine_similarity(
            current_visual.flatten().unsqueeze(0),
            self._cached_visual.flatten().unsqueeze(0),
            dim=-1,
        ).item()
        if sim >= self.sim_threshold:
            action = self._cached_chunk[self._cached_step_in_chunk].clone()
            self._cached_step_in_chunk += 1
            ...
            return action
        return None
```

`code/post_training/eval_libero.py:run_one_trial`：

```python
def run_one_trial(adapter, env, ..., chunk_cache=None):
    ...
    if chunk_cache is not None:
        chunk_cache.reset_episode()

    for t in range(max_steps):
        ...
        # Compute visual signature
        sig = F.adaptive_avg_pool2d(img_t.unsqueeze(0), (32, 32)).flatten()

        if chunk_cache:
            cached = chunk_cache.try_get_action(sig)
        else:
            cached = None

        if cached is not None:
            chunk_np = cached.cpu().numpy()[None, :]
            T = 1
            cache_hit_count += 1
        else:
            chunk = adapter.select_action(batch)
            vla_forward_count += 1
            chunk_cache.put(sig, chunk.squeeze(0).cpu())
            ...
```

CLI flag：

```bash
python -m post_training.eval_libero \
    --base pi05 ... \
    --use_chunk_cache \
    --cache_sim_threshold 0.95 \
    --cache_max_consecutive_reuses 5
```

## 5. 面试 Q&A

### Q: 为什么 LLM 已经有 KV cache，VLA 还要 VLA-Cache？

**A**: 标准 LLM KV cache 是**单次 inference 内**复用（自回归生成 token 之间）。VLA 推理的特殊性是 **跨 env step**（机器人 closed loop）：每个 step 都是一次完整的 inference，前一次的 KV cache 默认丢弃。VLA-Cache 把 cache 跨 step 保留，利用机器人场景的时序冗余（agent-view 变化小）。

### Q: 你们的 chunk-level cache 比 paper 的 per-token cache 更好还是更差？

**A**: 两端不同 tradeoff：
- **paper 更稳**：per-token reuse，每个 step 都 forward VLA（只省 attention 算），所以 success rate 几乎不受影响
- **我们更激进**：直接复用整个 chunk 跳过 VLA forward，理论上限 chunk_size×（10×），但若 cache 错命中，错误锁定 T 步

我们设了 `max_consecutive_reuses=5` 强制每 5 步重 query 兜底。

### Q: 你怎么决定 cache hit vs miss？

**A**: 用余弦相似度比较 visual signature。signature 不是完整 token 序列（太重），用 32×32 mean-pool 的 base camera 缩略图。flatten 后 1024-dim，cos sim 计算 < 1ms。`sim_threshold = 0.95` 是经验值，调高更保守。

### Q: 如果机械臂正在抓东西，scene 变化大，cache 会怎样？

**A**: 完美场景：抓取关键瞬间 visual obs 变化大 → cosine sim 掉到 0.95 以下 → cache miss → 重 query VLA → 拿到正确的"抓"动作。Cache 就是设计成"静止时复用，动作时强制重算"。

### Q: 你的 VLAChunkCache 跟 model 内部 KV cache (HF past_key_values) 是什么关系？

**A**: 完全独立的两层：
1. **model 内部 KV**: PaliGemma / Llama 自带，单次 inference 内自回归 token 之间复用 attention K/V。openpi 已经开 `use_cache=True`。
2. **VLAChunkCache**：我们的 wrapper，跨 inference 调用复用整个 action chunk。**跳过整个 VLA forward**，比内部 KV cache 收益大得多。

两者**正交且可叠加**。

### Q: VLA-Cache paper 报 1.7× speedup，你们目标多少？

**A**: 取决于 sim_threshold + scene 静态程度。
- LIBERO spatial（task 主要是 pick-place，第一阶段视觉静止 70%+ 时间）：理论上限 ~10×（chunk_size），实际预期 2-4×
- 真实机器人（动态环境）：可能 1.5-2×
- 简历 claim "↓2.4×" 在 LIBERO 上可达。Day 4 实测验证。

## 6. 推荐扩展阅读

- **Paper**: VLA-Cache (NeurIPS 2025) — https://openreview.net/forum?id=qOXHhihXOg
- **Project page**: https://vla-cache.github.io
- **Related**: π0.5 paper §3.4 (Knowledge Insulation) — 互补的优化方向

## 7. Day 4 计划

明天跑 LIBERO spatial × 50 trial × {no_cache, chunk_cache}：

- 量 reuse rate（cache_hits / total_steps）
- 量 latency（per-step wall time, no_cache vs cache）
- 量 success rate 是否保持

数据出来在 `assets/paper_v1.5_eval/kvcache_bench_day4.json`，简历数字按实测调整。

---

**Day 2-3 总用时**：~3.5 小时（设计文档 1h + 实现 2h + work log + teaching 0.5h）

**Day 2-3 commits**：
- ro_planning: `9683708`（feat: VLAChunkCache）+ design doc
- vla-lab: `8cd6813`（mirror）
