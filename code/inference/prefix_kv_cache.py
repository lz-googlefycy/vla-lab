"""Token-level prefix KV-Cache for π0.5 (real VLA-Cache approximation).

Day 7 redesign after Day 4 chunk-level cache failed.

----
WHY CHUNK-LEVEL FAILED (Day 4 result)
  - π0.5 already caches prefix KV across the 10 denoise steps within a single
    sample_actions() call. So the only cost we'd skip is one expert forward,
    which costs ~50ms. Cache check overhead (cosine sim + clone) costs the
    same. Net: 0.7× speedup, -20% success.
  - VLA-Cache (NeurIPS 2025) actually caches at a different scope:
    cross-timestep reuse of the prefix's vision-tower + paligemma KV.

WHAT THIS MODULE DOES
  Hook sample_actions(observation, ...) such that, on every env-step:
    1. Compute a cheap signature of `observation.images` (mean-pooled feature)
    2. If similarity to last step > threshold, REUSE the cached
       past_key_values from the previous timestep — skipping the entire
       PaliGemma vision-tower + language-tower forward, which is ~70% of
       per-step inference cost on H20.
    3. Only the action expert (suffix) re-runs each step.

  The past_key_values are valid as long as the prefix (image + lang tokens)
  hasn't changed. For LIBERO, the prompt is constant per-task and the
  visual scene is highly correlated step-to-step (>95% similarity for most
  trajectory segments — confirmed empirically on Day 4: mean sim 0.99,
  82% chunks above 0.95).

WHY THIS SHOULD WORK WHERE CHUNK-LEVEL DIDN'T
  - We DON'T skip action prediction (no "frozen action chunk" risk).
    The action expert still runs every timestep with fresh state proprio,
    so reactive control is preserved.
  - We DO skip the expensive part: vision tower (~half of forward time) +
    paligemma prefix forward (~third of forward time).
  - Cache-hit cost: 1 vision-tower forward per trial start + cosine sim
    on a small embedding (cheap, <1ms).

USAGE
    from inference.prefix_kv_cache import install_prefix_cache
    install_prefix_cache(adapter.model, sim_threshold=0.92)
    # adapter.model.sample_actions is now patched; transparent to caller.
    # Call adapter.model._prefix_cache.reset_episode() between trials.
"""
from __future__ import annotations

import dataclasses
import time
from typing import Any, Optional

import torch
import torch.nn.functional as F


@dataclasses.dataclass
class PrefixCacheStats:
    n_steps: int = 0
    n_hits: int = 0
    n_misses: int = 0
    similarities: list[float] = dataclasses.field(default_factory=list)
    prefix_forward_ms: list[float] = dataclasses.field(default_factory=list)
    sample_actions_ms: list[float] = dataclasses.field(default_factory=list)

    @property
    def hit_rate(self) -> float:
        total = self.n_hits + self.n_misses
        return self.n_hits / max(total, 1)

    @property
    def avg_prefix_ms(self) -> float:
        if not self.prefix_forward_ms:
            return 0.0
        return sum(self.prefix_forward_ms) / len(self.prefix_forward_ms)

    @property
    def avg_sample_actions_ms(self) -> float:
        if not self.sample_actions_ms:
            return 0.0
        return sum(self.sample_actions_ms) / len(self.sample_actions_ms)

    def to_dict(self) -> dict:
        return {
            "n_steps": self.n_steps,
            "n_hits": self.n_hits,
            "n_misses": self.n_misses,
            "hit_rate": self.hit_rate,
            "mean_similarity": (sum(self.similarities) / len(self.similarities)) if self.similarities else 0.0,
            "avg_prefix_ms": self.avg_prefix_ms,
            "avg_sample_actions_ms": self.avg_sample_actions_ms,
        }


class PrefixKVCache:
    """Cross-timestep prefix KV-Cache state holder.

    Stores past_key_values from the most recent sample_actions() prefix
    forward, plus a cheap visual signature for similarity check.
    """

    def __init__(self, sim_threshold: float = 0.92, max_reuses: int = 50):
        self.sim_threshold = sim_threshold
        self.max_reuses = max_reuses
        self._cached_pkv: Any = None              # past_key_values list
        self._cached_signature: Optional[torch.Tensor] = None   # (D,) flat
        self._cached_prefix_pad_masks: Optional[torch.Tensor] = None
        self._consecutive_reuses: int = 0
        self.stats = PrefixCacheStats()

    def reset_episode(self) -> None:
        self._cached_pkv = None
        self._cached_signature = None
        self._cached_prefix_pad_masks = None
        self._consecutive_reuses = 0

    @torch.no_grad()
    def _signature(self, images: list[torch.Tensor]) -> torch.Tensor:
        """Cheap visual signature: mean-pool of base camera image as (D,)."""
        # images is list of (B, C, H, W) tensors. Use first camera (base).
        if not images:
            return torch.zeros(1)
        img = images[0]
        # Pool 224x224 → 8x8 → flat (192-dim)
        pooled = F.adaptive_avg_pool2d(img.float(), output_size=8)
        return pooled.flatten(start_dim=1).mean(dim=0)  # (C*8*8,)

    def try_hit(self, images: list[torch.Tensor]) -> tuple[bool, float]:
        """Check if cache is usable for this step.

        Returns: (hit, similarity_score)
        """
        if (
            self._cached_pkv is None or
            self._cached_signature is None or
            self._consecutive_reuses >= self.max_reuses
        ):
            return False, 0.0

        sig = self._signature(images)
        cached_sig = self._cached_signature
        sim = F.cosine_similarity(
            sig.flatten().unsqueeze(0),
            cached_sig.flatten().unsqueeze(0),
            dim=-1,
        ).item()
        return sim >= self.sim_threshold, sim

    def store(self, images: list[torch.Tensor], pkv: Any,
              prefix_pad_masks: torch.Tensor) -> None:
        """Cache freshly-computed prefix state for next step."""
        self._cached_pkv = pkv
        self._cached_signature = self._signature(images).detach()
        self._cached_prefix_pad_masks = prefix_pad_masks.detach()
        self._consecutive_reuses = 0

    def mark_reuse(self) -> None:
        self._consecutive_reuses += 1


def install_prefix_cache(
    model: torch.nn.Module,
    sim_threshold: float = 0.92,
    max_reuses: int = 50,
) -> PrefixKVCache:
    """Monkey-patch model.sample_actions to skip prefix forward on cache hit.

    Args:
        model: PI0Pytorch instance (has .sample_actions and .paligemma_with_expert)
        sim_threshold: cosine sim required to reuse cached past_key_values
        max_reuses: hard cap; force re-compute every N hits

    Returns: PrefixKVCache instance (also stored at model._prefix_cache).
    """
    from openpi.models_pytorch.pi0_pytorch import (  # noqa: WPS433
        make_att_2d_masks,
    )

    cache = PrefixKVCache(sim_threshold=sim_threshold, max_reuses=max_reuses)
    model._prefix_cache = cache  # noqa: SLF001  attached for caller access

    original_sample_actions = model.sample_actions

    @torch.no_grad()
    def cached_sample_actions(device, observation, noise=None, num_steps=10):
        """Drop-in replacement for sample_actions with cross-timestep prefix KV reuse."""
        t0 = time.perf_counter()
        bsize = observation.state.shape[0]
        if noise is None:
            actions_shape = (bsize, model.config.action_horizon, model.config.action_dim)
            noise = model.sample_noise(actions_shape, device)

        images, img_masks, lang_tokens, lang_masks, state = (
            model._preprocess_observation(observation, train=False)  # noqa: SLF001
        )

        cache.stats.n_steps += 1
        hit, sim = cache.try_hit(images)
        cache.stats.similarities.append(sim)

        if hit:
            past_key_values = cache._cached_pkv  # noqa: SLF001
            prefix_pad_masks = cache._cached_prefix_pad_masks  # noqa: SLF001
            cache.stats.n_hits += 1
            cache.mark_reuse()
        else:
            t_prefix = time.perf_counter()
            prefix_embs, prefix_pad_masks, prefix_att_masks = model.embed_prefix(
                images, img_masks, lang_tokens, lang_masks
            )
            prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
            prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

            prefix_att_2d_masks_4d = model._prepare_attention_masks_4d(prefix_att_2d_masks)  # noqa: SLF001
            (
                model.paligemma_with_expert.paligemma.language_model.config._attn_implementation  # noqa: SLF001
            ) = "eager"

            _, past_key_values = model.paligemma_with_expert.forward(
                attention_mask=prefix_att_2d_masks_4d,
                position_ids=prefix_position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, None],
                use_cache=True,
            )
            cache.stats.prefix_forward_ms.append((time.perf_counter() - t_prefix) * 1000)
            cache.store(images, past_key_values, prefix_pad_masks)
            cache.stats.n_misses += 1

        # ----- denoise loop (same as original) -----
        dt = -1.0 / num_steps
        dt = torch.tensor(dt, dtype=torch.float32, device=device)

        x_t = noise
        time_t = torch.tensor(1.0, dtype=torch.float32, device=device)
        while time_t >= -dt / 2:
            expanded_time = time_t.expand(bsize)
            v_t = model.denoise_step(
                state,
                prefix_pad_masks,
                past_key_values,
                x_t,
                expanded_time,
            )
            x_t = x_t + dt * v_t
            time_t += dt

        cache.stats.sample_actions_ms.append((time.perf_counter() - t0) * 1000)
        return x_t

    model.sample_actions = cached_sample_actions
    return cache


def uninstall_prefix_cache(model: torch.nn.Module) -> None:
    """Restore the original sample_actions if it was monkey-patched."""
    if hasattr(model, "_prefix_cache"):
        delattr(model, "_prefix_cache")
    # Restore from class default (since cached_sample_actions is a closure)
    cls = type(model)
    if hasattr(cls, "sample_actions"):
        # Bind to instance — but we need the original. Pull from cls.
        original = cls.sample_actions
        # Avoid re-binding: assign closure that calls cls.sample_actions
        model.sample_actions = original.__get__(model, cls)
