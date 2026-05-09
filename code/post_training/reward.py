"""LIBERO reward signal builder + DPO pair generator.

Two functions used across DPO / GRPO training:

1. ``rollout_chunk(env, chunk)``: replay a 60-step action chunk in a
   LIBERO env, return a scalar reward in [-1, 1].

2. ``build_dpo_pairs(samples, top_k, bottom_k)``: from K candidate
   chunks per instruction, pick top-k as chosen and bottom-k as
   rejected → (B*pairs) tuples for DPO.

Reward design (paper-grade, deliberate):

    r = w_succ * task_success
      + w_prog * task_progress
      + w_smooth * (-mean_jerk)

    defaults: w_succ = 0.7, w_prog = 0.2, w_smooth = 0.1

These three weights are exposed via PostTrainConfig so we can ablate
them in the paper. Rationale:
  - success is the primary signal but binary, hence sparse
  - progress (LIBERO env reports 0..1) gives dense gradient
  - smoothness penalises jerky chunks that *technically* succeed but
    would be unstable on a real robot — keeps the alignment signal
    correlated with deployable policies, not just LIBERO-overfit ones
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch


@dataclass
class RewardConfig:
    success_weight: float = 0.7
    progress_weight: float = 0.2
    smoothness_weight: float = 0.1
    smoothness_normaliser: float = 1.0   # scales |a_{t+1} - a_t| sum into ~1
    fail_terminal_penalty: float = 0.0   # extra -X if env terminated unsuccessfully
    success_terminal_bonus: float = 0.0  # extra +X if env terminated successfully


def chunk_reward_components(
    chunk: np.ndarray,
    success: bool,
    progress: float = 0.0,
    cfg: RewardConfig = RewardConfig(),
) -> dict:
    """Compute the three components of the reward (without combining).

    Useful for paper figures (showing how the reward decomposes per task).

    Args:
        chunk:    (T, A) numpy action chunk that was actually executed
        success:  whether the env reported terminated == success
        progress: env-reported progress in [0, 1]; pass 0 if env doesn't expose it
        cfg:      RewardConfig

    Returns:
        dict with keys: success_score, progress_score, smoothness_score
    """
    # 1. Success: binary
    success_score = 1.0 if success else 0.0

    # 2. Progress: env-reported, default 0
    progress_score = float(np.clip(progress, 0.0, 1.0))

    # 3. Smoothness: -mean(|a_{t+1} - a_t|)
    if chunk.shape[0] >= 2:
        diffs = chunk[1:] - chunk[:-1]
        jerk = float(np.linalg.norm(diffs, axis=-1).mean())
        smoothness_score = -jerk / cfg.smoothness_normaliser
    else:
        smoothness_score = 0.0

    return {
        "success_score": success_score,
        "progress_score": progress_score,
        "smoothness_score": smoothness_score,
    }


def chunk_reward(
    chunk: np.ndarray,
    success: bool,
    progress: float = 0.0,
    cfg: RewardConfig = RewardConfig(),
) -> float:
    """Combined scalar reward in [-1, 1]-ish."""
    c = chunk_reward_components(chunk, success, progress, cfg)
    r = (
        cfg.success_weight * c["success_score"]
        + cfg.progress_weight * c["progress_score"]
        + cfg.smoothness_weight * c["smoothness_score"]
    )
    if success:
        r += cfg.success_terminal_bonus
    elif cfg.fail_terminal_penalty:
        r -= cfg.fail_terminal_penalty
    return r


def rollout_chunk(env: Any, chunk: np.ndarray, max_steps: int | None = None) -> dict:
    """Replay one chunk on a LIBERO env, return raw outcome.

    Args:
        env: LIBERO OffScreenRenderEnv (or compatible). Must have been
             reset to the desired initial state by the caller.
        chunk: (T, A) numpy actions
        max_steps: cap T at this; default = chunk length

    Returns:
        dict with keys:
            success: bool — env terminated as success
            progress: float in [0, 1] — env-reported progress
            chunk_executed: (T_actual, A) actions that were actually applied
                            (may be shorter than chunk if env terminated early)
    """
    T = chunk.shape[0] if max_steps is None else min(max_steps, chunk.shape[0])
    success = False
    progress = 0.0

    for t in range(T):
        a = chunk[t]
        result = env.step(a)
        # LIBERO returns 4-tuple (obs, reward, done, info) in old gym API,
        # 5-tuple (obs, reward, terminated, truncated, info) in newer.
        if len(result) == 4:
            obs, _, done, info = result
            terminated, truncated = done, False
        else:
            obs, _, terminated, truncated, info = result

        if isinstance(info, dict):
            success = bool(info.get("success", success))
            progress = float(info.get("progress", progress))

        if terminated or truncated:
            return {
                "success": success,
                "progress": progress,
                "chunk_executed": chunk[: t + 1],
            }

    return {"success": success, "progress": progress, "chunk_executed": chunk[:T]}


def build_dpo_pairs(
    chunks: torch.Tensor,
    rewards: torch.Tensor,
    n_pairs: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build DPO (chosen, rejected) pairs from K candidate chunks per prompt.

    Args:
        chunks:  (B, K, T, A) candidate chunks
        rewards: (B, K) reward per chunk
        n_pairs: number of pairs to extract per prompt

    Returns:
        chosen_chunks:   (B, n_pairs, T, A)
        rejected_chunks: (B, n_pairs, T, A)
        score_margins:   (B, n_pairs) reward gap between chosen and rejected
    """
    B, K, T, A = chunks.shape
    assert n_pairs <= K // 2, f"n_pairs={n_pairs} but K={K}"

    # sort each row of rewards descending → top-k are chosen, bottom-k rejected
    sorted_rewards, sort_idx = torch.sort(rewards, dim=-1, descending=True)
    top_idx = sort_idx[:, :n_pairs]              # (B, n_pairs)
    bottom_idx = sort_idx[:, -n_pairs:]          # (B, n_pairs)

    chosen = torch.gather(
        chunks, dim=1,
        index=top_idx[..., None, None].expand(-1, -1, T, A),
    )
    rejected = torch.gather(
        chunks, dim=1,
        index=bottom_idx[..., None, None].expand(-1, -1, T, A),
    )
    margins = (
        torch.gather(rewards, dim=1, index=top_idx)
        - torch.gather(rewards, dim=1, index=bottom_idx)
    )

    return chosen, rejected, margins


__all__ = [
    "RewardConfig",
    "chunk_reward_components",
    "chunk_reward",
    "rollout_chunk",
    "build_dpo_pairs",
]
