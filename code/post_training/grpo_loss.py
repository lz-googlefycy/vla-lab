"""GRPO loss for VLA — independent reimplementation.

Background
----------
GRPO (Shao et al. 2024, DeepSeek-Math; popularized by DeepSeek-R1) uses
group-relative advantage normalisation in place of a learned value
function. For a group of K candidate samples per prompt:

    advantage_i = (r_i - mean_k(r)) / std_k(r)
    ratio_i = exp(logp_i - logp_old.detach())
    L_PG_i = -min(ratio_i * advantage_i,
                   clip(ratio_i, 1±ε) * advantage_i)
    L_KL_i = β * KL(π || π_ref)
    L_GRPO = E[ L_PG + L_KL ]

Compared to PPO, GRPO drops the value-function critic. Compared to DPO,
GRPO needs online rollout (the K samples must come from current
policy), but doesn't need a paired (chosen, rejected) dataset.

This file is a clean room reimplementation. The math/structure mirrors
both the GRPO paper and the well-tested form used in production
trajectory-RLHF projects elsewhere; the constants (β=0.1, ε=0.2) are
the standard published GRPO hyperparameters.

For VLA specifically:
  - openvla: logp = sum log p(token | prefix) — exact and cheap.
  - spirit / π0.5: logp via flow-matching surrogate (see adapter).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .dpo_loss import kl_estimate_k1, kl_estimate_k2, kl_estimate_k3


@dataclass
class GRPOConfig:
    beta: float = 0.1                    # KL coefficient
    epsilon: float = 0.2                 # ratio clip
    kl_estimator: str = "k3"             # "k1" | "k2" | "k3"
    advantage_normalization: str = "group"   # "group" | "global" | "none"
    advantage_jitter_std: float = 1e-4
    clip_advantage_log_ratio: bool = False  # PPO-style log_ratio clipping (alt formulation)


@dataclass
class GRPOOutput:
    loss: torch.Tensor                   # scalar — total loss
    pg_loss: torch.Tensor                # scalar — policy-gradient component
    kl_loss: torch.Tensor                # scalar — KL component
    mean_advantage: torch.Tensor         # scalar
    mean_ratio: torch.Tensor             # scalar
    mean_kl: torch.Tensor                # scalar
    clip_fraction: torch.Tensor          # scalar — fraction of samples where ratio was clipped


def _normalize_advantage(reward: torch.Tensor, mode: str, jitter: float) -> torch.Tensor:
    """Group-relative or global advantage normalisation.

    Args:
        reward: (B, K) reward per sample
        mode: "group" | "global" | "none"
        jitter: noise std added to break degenerate cases (all-equal reward)

    Returns:
        advantage: (B, K)
    """
    reward = reward + torch.randn_like(reward) * jitter

    if mode == "group":
        # standardize within each group of K samples (per prompt)
        std, mean = torch.std_mean(reward, dim=-1, keepdim=True)
        return (reward - mean) / (std + 1e-10)

    if mode == "global":
        # subtract group mean (centering per prompt) then standardize
        # globally across the batch — matches some production setups
        # that treat the full batch as one population for variance.
        local_centered = reward - reward.mean(dim=-1, keepdim=True)
        std, mean = torch.std_mean(local_centered)
        return (local_centered - mean) / (std + 1e-10)

    if mode == "none":
        return reward - reward.mean(dim=-1, keepdim=True)

    raise ValueError(f"Unknown advantage normalization: {mode}")


def grpo_loss(
    logp: torch.Tensor,
    logp_old: torch.Tensor,
    logp_ref: torch.Tensor,
    reward: torch.Tensor,
    cfg: GRPOConfig,
    mask: torch.Tensor | None = None,
) -> GRPOOutput:
    """Compute GRPO loss.

    Args:
        logp:      (B, K) log π_θ(chunk_i | prompt) — current policy,
                  REQUIRES_GRAD (this is the parameter we optimise over)
        logp_old:  (B, K) log π_old(chunk_i | prompt) — sampling policy
                  (usually = current policy at the start of the step;
                  for one-step trust region GRPO this is logp.detach())
        logp_ref:  (B, K) log π_ref(chunk_i | prompt) — frozen reference
        reward:    (B, K) scalar reward per sample (e.g. from LIBERO env)
        cfg:       GRPOConfig
        mask:      (B,) optional bool mask. If False for sample b, that
                  prompt is dropped from the loss (e.g. when no successful
                  candidate exists in the group).

    Returns:
        GRPOOutput with total loss + per-component diagnostics.
    """
    advantage = _normalize_advantage(reward, cfg.advantage_normalization, cfg.advantage_jitter_std)

    # Importance-sampling ratio — current policy vs sampling policy
    log_ratio = logp - logp_old.detach()
    if cfg.clip_advantage_log_ratio:
        log_ratio = torch.clamp(log_ratio, max=10.0)  # numerical stability
    ratio = torch.exp(log_ratio)

    # PPO clipping
    ratio_clipped = torch.clamp(ratio, 1.0 - cfg.epsilon, 1.0 + cfg.epsilon)
    surr1 = ratio * advantage
    surr2 = ratio_clipped * advantage
    pg_per_sample = torch.minimum(surr1, surr2)            # (B, K)
    pg_loss_per_prompt = -pg_per_sample.mean(dim=-1)        # (B,)

    # KL term
    if cfg.kl_estimator == "k1":
        kl_per_sample = kl_estimate_k1(logp, logp_ref)
    elif cfg.kl_estimator == "k2":
        kl_per_sample = kl_estimate_k2(logp, logp_ref)
    elif cfg.kl_estimator == "k3":
        kl_per_sample = kl_estimate_k3(logp, logp_ref)
    else:
        raise ValueError(f"Unknown KL estimator: {cfg.kl_estimator}")
    kl_loss_per_prompt = cfg.beta * kl_per_sample.mean(dim=-1)   # (B,)

    # Apply mask + reduce
    if mask is not None:
        valid = mask.float()
        valid_sum = valid.sum().clamp(min=1.0)
        pg_loss = (pg_loss_per_prompt * valid).sum() / valid_sum
        kl_loss = (kl_loss_per_prompt * valid).sum() / valid_sum
    else:
        pg_loss = pg_loss_per_prompt.mean()
        kl_loss = kl_loss_per_prompt.mean()

    total_loss = pg_loss + kl_loss

    # Diagnostics
    with torch.no_grad():
        clip_fraction = ((ratio - ratio_clipped).abs() > 1e-6).float().mean()

    return GRPOOutput(
        loss=total_loss,
        pg_loss=pg_loss,
        kl_loss=kl_loss,
        mean_advantage=advantage.mean(),
        mean_ratio=ratio.mean(),
        mean_kl=kl_per_sample.mean(),
        clip_fraction=clip_fraction,
    )


__all__ = [
    "GRPOConfig",
    "GRPOOutput",
    "grpo_loss",
]
