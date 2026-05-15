# Day 2: KV-Cache 算法理解 + 设计

> Sprint Day 2 / 10 · 2026-05-16
>
> **目标**：基于 VLA-Cache (NeurIPS 2025) 设计 vla-lab 自家的 KV-Cache 实现，给 Day 3 写代码做准备。

## 1. VLA-Cache paper 摘要

**Title**: "VLA-Cache: Efficient Vision-Language-Action Manipulation via Adaptive Token Caching"
**Venue**: NeurIPS 2025
**Insight**: VLA 的视觉 prompt 占 LLM input 90%+ tokens，但**相邻 timestep 视觉变化小** → 大量 visual token KV pair 可复用。

### 算法（3 步）

#### Step 1: Static Token Selection

对每个 timestep t，计算每个 visual token 的余弦相似度：
$$
\text{sim}_i = \frac{\langle P_t^{(i)}, P_{t-1}^{(i)} \rangle}{\|P_t^{(i)}\| \cdot \|P_{t-1}^{(i)}\|}
$$

如果 $\text{sim}_i \geq \tau$（默认 $\tau=0.92$），token i 标记为 **static**，可以复用 KV cache。

#### Step 2: Evict Task-Relevant Tokens

从 static 集合里**剔除** attention 高的 token（机械臂、目标物体等关键区域）。具体：

- 取 VLA decoder 的 attention map（最后一层 / 平均所有层）
- 把 attention 排前 K%（默认 K=20）的 visual token 强制 re-compute
- 防止"环境敏感区域"误用旧 KV，让 action 失准

#### Step 3: Layer-Adaptive Token Reuse

不同 decoder layer 对视觉敏感度不同：

- **底层 layer**（vision-text fusion 早期）：attention 散在所有 visual token，**reuse 比例可以高**（~80%）
- **顶层 layer**（决策 head 附近）：attention 集中在关键 region，**reuse 比例降低**（~30-50%）

### 实测数字

- LIBERO / SIMPLER / 真实机器人
- **1.7× CUDA latency speedup**
- **+15% control frequency**
- 任务成功率几乎不损失（<1% drop）

## 2. 我们的实现设计

### 2.1 范围 + 限制

- **目标平台**：π0.5 (PaliGemma + Gemma-Expert dual-tower) + OpenVLA (Llama)
- **场景**：LIBERO inference path（`adapter.select_action`）
- **优先级**：先做 Step 1 (static selection) + Step 3 简化版（global reuse rate, no per-layer），Step 2 task-relevant eviction 作为 v2

> 简历 claim：「视觉 token KV-Cache 复用 + chunk-level 推理加速」+ 数字「>92% 复用率，latency ↓2.4×」 — 我们目标更激进，做完看实测。

### 2.2 接口设计

```python
# code/inference/kv_cache.py (新文件)

class VLAKVCache:
    """Visual-token KV cache manager for VLA inference acceleration.

    Maintains the KV state of visual tokens from the previous timestep,
    and uses cosine similarity to decide which tokens can reuse cached KV
    in the next timestep's forward pass.
    """

    def __init__(
        self,
        sim_threshold: float = 0.92,
        reuse_rate_cap: float = 0.95,   # max % tokens we'll reuse even if all are static
        evict_top_k_attn: float = 0.0,  # Step 2: top-K% high-attention tokens always recompute
        per_layer: bool = False,        # Step 3: per-layer reuse rate (v2)
    ):
        self.sim_threshold = sim_threshold
        self.reuse_rate_cap = reuse_rate_cap
        self.evict_top_k_attn = evict_top_k_attn
        self.per_layer = per_layer
        # State (cleared per episode):
        self.prev_visual_tokens: torch.Tensor | None = None    # (T_v, D)
        self.prev_kv_cache: list[tuple[K, V]] | None = None     # per-layer
        self.episode_steps = 0
        self.reuse_history: list[float] = []                    # reuse rate per step

    def step_compute_reuse_mask(
        self, curr_visual_tokens: torch.Tensor
    ) -> torch.Tensor:
        """Return (T_v,) bool mask: True for tokens whose KV can be reused."""
        if self.prev_visual_tokens is None:
            return torch.zeros(curr_visual_tokens.shape[0], dtype=torch.bool)
        sim = F.cosine_similarity(
            curr_visual_tokens, self.prev_visual_tokens, dim=-1
        )  # (T_v,)
        mask = sim >= self.sim_threshold

        # Cap total reuse rate
        if mask.float().mean() > self.reuse_rate_cap:
            # keep top reuse_rate_cap fraction by similarity
            n_keep = int(self.reuse_rate_cap * len(sim))
            topk_idx = sim.topk(n_keep).indices
            new_mask = torch.zeros_like(mask)
            new_mask[topk_idx] = True
            mask = new_mask
        self.reuse_history.append(mask.float().mean().item())
        return mask

    def update(self, curr_visual_tokens, kv_cache_per_layer):
        """Cache current state for next-step reuse."""
        self.prev_visual_tokens = curr_visual_tokens.detach()
        self.prev_kv_cache = kv_cache_per_layer
        self.episode_steps += 1

    def reset(self):
        """Call at episode boundary."""
        self.prev_visual_tokens = None
        self.prev_kv_cache = None
        self.episode_steps = 0
        # Don't reset reuse_history (we want full-episode aggregation)


def reuse_kv_in_attention(
    layer_kv_cache: tuple[torch.Tensor, torch.Tensor],   # (K_prev, V_prev)
    new_kv: tuple[torch.Tensor, torch.Tensor],           # (K_new, V_new)
    reuse_mask: torch.Tensor,                            # (T_v,) bool
    visual_token_indices: torch.Tensor,                  # (T_v,) positions in seq
) -> tuple[torch.Tensor, torch.Tensor]:
    """For visual tokens with mask=True, replace K_new/V_new with K_prev/V_prev.

    Returns the merged KV that should be used by the attention forward.
    """
    K, V = new_kv
    K_prev, V_prev = layer_kv_cache
    keep_idx = visual_token_indices[reuse_mask]
    K[..., keep_idx, :] = K_prev[..., keep_idx, :]
    V[..., keep_idx, :] = V_prev[..., keep_idx, :]
    return K, V
```

### 2.3 集成到 Pi05Adapter / OpenVLAAdapter

最小侵入：在 `select_action` 路径上插一个 wrapper。

```python
# pi05.py
class Pi05Adapter:
    def __init__(self, cfg):
        ...
        self.kv_cache = VLAKVCache(sim_threshold=0.92) if cfg.use_kv_cache else None

    def select_action(self, batch):
        if self.kv_cache is None:
            return self._select_action_uncached(batch)
        return self._select_action_with_cache(batch)

    def _select_action_with_cache(self, batch):
        # 1. encode visual tokens
        visual_tokens = self._encode_visuals(batch)
        # 2. compute reuse mask
        mask = self.kv_cache.step_compute_reuse_mask(visual_tokens)
        # 3. forward with selective KV reuse
        actions, kv = self._forward_with_kv_reuse(batch, visual_tokens, mask)
        # 4. update cache
        self.kv_cache.update(visual_tokens, kv)
        return actions

    def reset_episode(self):
        if self.kv_cache:
            self.kv_cache.reset()
```

### 2.4 风险

| 风险 | 应对 |
|---|---|
| HuggingFace transformers KV cache API 在 PaliGemma 上太复杂 | 兜底：自己 wrap PaliGemma forward，手动管理 K/V tensors（已有 manual greedy 经验） |
| Reuse 后 success rate 显著掉 | 调 `sim_threshold`（0.92 → 0.95 更保守）+ 加 Step 2 task-relevant evict |
| Latency speedup 不到 1.5× | 接受实测数字，简历调整 |

### 2.5 简历数字调整方向（基于 paper）

| 简历当前 | VLA-Cache paper 报告 | 我们目标 |
|---|---|---|
| 复用率 >92% | 没明确报，但 threshold=0.92 | 把"92"理解成 sim threshold（不是复用率），实测复用率应该 70-85% |
| latency ↓2.4× | **1.7× speedup** | 简历改 ~↓1.7× 更稳；或保 2× 留改 paper claim 余地 |

**Day 4 实测后**调整简历。

## 3. Day 3 任务（明天做）

1. 写 `code/inference/kv_cache.py` (~150 行)
2. 在 `pi05.py:select_action` 加 cache 路径
3. Smoke test：1 task LIBERO eval，对比 with/without cache 输出**一致性**（数字 success rate 相近）
4. 如果数字差距大，调 threshold

## 4. Day 4 任务

跑 LIBERO spatial × 50 trial × {no_cache, with_cache} bench：
- per-step latency
- reuse rate
- success rate
- 写到 `assets/paper_v1.5_eval/kvcache_bench_day4.json`

数据出来后写 result.md + 教学版 + 飞书 wiki。

## 5. 配套教学（Day 3 写）

`docs/teaching/day3_kvcache_for_liu.md`：
- KV cache 原理（autoregressive transformer 为什么能 cache）
- VLA-Cache 的 novelty（visual token 的特殊性 + adaptive selection）
- 我们 vla-lab 实现的 walkthrough
- 面试 Q&A

EOF
echo "Day 2 design doc written: $(wc -l < /home/ubuntu/ro_planning/docs/work_log/day2_kvcache_design.md) lines"