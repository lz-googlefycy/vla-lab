"""VLA-Cache style inference acceleration for chunk-action VLA models.

Inspired by VLA-Cache (NeurIPS 2025): adaptive token caching via visual
token similarity. Robot manipulation has high temporal redundancy in the
agent-view image, so consecutive timesteps' visual tokens are >90% similar
on average — perfect for KV reuse.

Two-tier caching strategy implemented here:

1. **Chunk-level cache (lightweight)**: action chunks have horizon T=10;
   if current obs's visual encoding is highly similar to the obs that
   produced the last chunk, reuse the next steps from that chunk
   instead of re-querying the VLA. Works because action chunks already
   amortize VLA cost over T env steps.

2. **Visual KV cache (full VLA-Cache)**: per-timestep, compute static
   token mask via cosine similarity, replace cached K/V for static
   tokens before the attention forward. Requires hooking into the
   underlying transformer attention. Skeleton API provided; not fully
   wired to PaliGemma/Llama yet (Day 4 if Day 3 results encourage).

Public surface:
    - VLAChunkCache: chunk-level reuse (battle-tested first)
    - VLAKVCache: per-token KV reuse (skeleton)
    - SimilarityStats: dataclass for benchmark metrics
"""
from __future__ import annotations

import dataclasses
from typing import Optional

import torch
import torch.nn.functional as F


@dataclasses.dataclass
class SimilarityStats:
    """Aggregate metrics for benchmark logging."""
    n_steps: int = 0
    n_reused_chunks: int = 0          # chunks where we skipped VLA forward
    n_recomputed_chunks: int = 0      # chunks that triggered VLA forward
    similarities: list[float] = dataclasses.field(default_factory=list)
    static_token_rates: list[float] = dataclasses.field(default_factory=list)

    @property
    def chunk_reuse_rate(self) -> float:
        total = self.n_reused_chunks + self.n_recomputed_chunks
        return self.n_reused_chunks / max(total, 1)

    @property
    def avg_static_token_rate(self) -> float:
        if not self.static_token_rates:
            return 0.0
        return float(sum(self.static_token_rates) / len(self.static_token_rates))


class VLAChunkCache:
    """Chunk-level cache: reuse last action chunk if visual obs unchanged.

    Strategy:
      1. Each VLA forward produces a T-step action chunk.
      2. Cache (visual_embedding, action_chunk, current_step_idx).
      3. Next env step:
         - If still steps left in cached chunk AND similarity ≥ threshold,
           pop next action from cache (skip VLA forward entirely).
         - Else, re-query VLA, cache the new chunk.

    This is the most aggressive form of caching: a single VLA forward
    serves up to T env steps when scene changes are minor. Combined with
    π0.5's T=10 action horizon, this gives up to 10× speedup in the
    static-scene limit.

    Tradeoff: any imperfection in the cached chunk persists for up to
    `chunk_size` steps (you can't react quickly). Mitigated by setting
    similarity threshold high (0.95+) so we always re-query when scene
    actually changes.
    """

    def __init__(
        self,
        sim_threshold: float = 0.95,
        chunk_size: int = 10,
        max_consecutive_reuses: int = 5,
    ):
        """
        Args:
            sim_threshold: cosine sim required to reuse cached chunk
            chunk_size: action horizon T (10 for π0.5, 1 for OpenVLA — but
                        OpenVLA's effective chunk_size is 1 so cache is no-op)
            max_consecutive_reuses: hard cap on reuses without recompute
                                    (safety: even if sim is high, force
                                    re-query every N steps)
        """
        self.sim_threshold = sim_threshold
        self.chunk_size = chunk_size
        self.max_consecutive_reuses = max_consecutive_reuses
        self._cached_chunk: Optional[torch.Tensor] = None       # (T, A)
        self._cached_step_in_chunk: int = 0
        self._cached_visual: Optional[torch.Tensor] = None      # (D,) flattened
        self._consecutive_reuses: int = 0
        self.stats = SimilarityStats()

    def reset_episode(self):
        """Call at episode boundary (env.reset)."""
        self._cached_chunk = None
        self._cached_step_in_chunk = 0
        self._cached_visual = None
        self._consecutive_reuses = 0
        # NOTE: stats persist across episodes for full-suite benchmarking

    def try_get_action(self, current_visual: torch.Tensor) -> Optional[torch.Tensor]:
        """Attempt to fetch next action from cache.

        Args:
            current_visual: (D,) embedding of current observation.
                Recommended: flatten of normalized base camera embed,
                or just pixel-level feature mean.

        Returns:
            (A,) action tensor if cache hit, None if miss (caller must
            re-query VLA and call ``put`` with the new chunk).
        """
        self.stats.n_steps += 1

        # Cache must exist + still have steps remaining + below reuse cap
        if (
            self._cached_chunk is None or
            self._cached_step_in_chunk >= self.chunk_size or
            self._consecutive_reuses >= self.max_consecutive_reuses
        ):
            return None

        # Check similarity
        sim = F.cosine_similarity(
            current_visual.flatten().unsqueeze(0),
            self._cached_visual.flatten().unsqueeze(0),
            dim=-1,
        ).item()
        self.stats.similarities.append(sim)

        if sim >= self.sim_threshold:
            action = self._cached_chunk[self._cached_step_in_chunk].clone()
            self._cached_step_in_chunk += 1
            self._consecutive_reuses += 1
            self.stats.n_reused_chunks += 1
            return action

        return None

    def put(self, current_visual: torch.Tensor, chunk: torch.Tensor):
        """Store newly-computed chunk + visual ref for future reuse.

        Args:
            current_visual: (D,) embedding (must match dim of try_get_action input)
            chunk: (T, A) action chunk
        """
        self._cached_chunk = chunk.detach().clone()
        self._cached_step_in_chunk = 1   # we're about to consume step 0
        self._cached_visual = current_visual.detach().clone()
        self._consecutive_reuses = 0
        self.stats.n_recomputed_chunks += 1


class VLAKVCache:
    """Per-token visual-KV reuse (full VLA-Cache style). SKELETON.

    To wire fully, we need to:
      1. Intercept transformer attention forward (PaliGemma + Llama)
      2. Identify which input positions correspond to visual tokens
      3. After computing K/V, selectively replace visual-token K/V with
         cached values from previous timestep (per static_mask)

    For now this class records the hooks and metrics needed for
    benchmarking; concrete attention patching is left for Day 4 if Day 3
    chunk-cache results suggest more speedup is needed.
    """

    def __init__(self, sim_threshold: float = 0.92):
        self.sim_threshold = sim_threshold
        self.prev_visual_tokens: Optional[torch.Tensor] = None
        self.prev_kv_per_layer: Optional[list] = None
        self.stats = SimilarityStats()

    def reset_episode(self):
        self.prev_visual_tokens = None
        self.prev_kv_per_layer = None

    def compute_static_mask(self, current_visual_tokens: torch.Tensor) -> torch.Tensor:
        """Return (T_v,) bool mask of static (reusable) tokens.

        Args:
            current_visual_tokens: (T_v, D) per-token embeddings at current step.
        """
        if self.prev_visual_tokens is None:
            return torch.zeros(current_visual_tokens.shape[0], dtype=torch.bool,
                              device=current_visual_tokens.device)
        sim = F.cosine_similarity(
            current_visual_tokens, self.prev_visual_tokens, dim=-1
        )
        mask = sim >= self.sim_threshold
        self.stats.static_token_rates.append(mask.float().mean().item())
        return mask

    def update(self, visual_tokens: torch.Tensor, kv_per_layer: list):
        self.prev_visual_tokens = visual_tokens.detach()
        self.prev_kv_per_layer = kv_per_layer
        self.stats.n_steps += 1
