"""DPO loss for VLA — independent reimplementation.

Background
----------
DPO (Rafailov et al. 2023, https://arxiv.org/abs/2305.18290) optimises:

    L_DPO = -E[ log σ( β * (logπ_θ(y_w|x) - logπ_ref(y_w|x))
                       - β * (logπ_θ(y_l|x) - logπ_ref(y_l|x)) ) ]

where (x, y_w, y_l) are (prompt, chosen, rejected). The implicit reward
model is r̂(x, y) = β * log(π_θ(y|x) / π_ref(y|x)).

For VLA, the "prompt" is the (instruction + image + state) batch, and
"chunk" replaces "completion". β = 0.1 by default (matches LLM literature).

This module is independent — no torch.compile, no peft, no trl. It
takes pre-computed log-probs from the adapter, computes the loss, and
returns. Adapter responsibility:
  - openvla: sum log p(token_t,d | prefix) over chunk tokens
  - spirit / π0.5: ELBO-style surrogate via score-matching residual
                   (see adapter docstring)
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class DPOConfig:
    beta: float = 0.1
    label_smoothing: float = 0.0  # IPO-style if > 0; not yet used in v1.5
    reference_free: bool = False  # if True, drop the reference term (degenerate but useful for ablation)


@dataclass
class DPOOutput:
    """Return container — keeps logits and accuracies for monitoring."""

    loss: torch.Tensor                     # scalar
    chosen_rewards: torch.Tensor           # (B,) — implicit reward for chosen
    rejected_rewards: torch.Tensor         # (B,) — implicit reward for rejected
    accuracy: torch.Tensor                 # scalar — fraction where chosen > rejected
    margin: torch.Tensor                   # scalar — mean(chosen_reward - rejected_reward)


def dpo_loss(
    logp_chosen: torch.Tensor,
    logp_rejected: torch.Tensor,
    logp_ref_chosen: torch.Tensor,
    logp_ref_rejected: torch.Tensor,
    cfg: DPOConfig,
) -> DPOOutput:
    """Compute DPO loss given pre-computed log-probs.

    Args:
        logp_chosen:        (B,) log π_θ(y_w | x) under current policy
        logp_rejected:      (B,) log π_θ(y_l | x) under current policy
        logp_ref_chosen:    (B,) log π_ref(y_w | x) under frozen reference
        logp_ref_rejected:  (B,) log π_ref(y_l | x) under frozen reference
        cfg: DPOConfig

    Returns:
        DPOOutput with loss + diagnostic stats.

    Notes
    -----
    All four inputs should be on the same device & dtype. Adapters that
    can only produce a surrogate log-prob estimator should still pass
    those through here — DPO doesn't care whether logp is exact or a
    surrogate, as long as the surrogate is self-consistent for both
    chosen and rejected (same estimator across the 4 args).
    """
    if cfg.reference_free:
        # Degenerate variant: drop ref term, treat π_θ as the reward model.
        # Only useful for ablation — should converge to maximizing
        # logp_chosen - logp_rejected, which is a noisy form of
        # weighted BC.
        diff = cfg.beta * (logp_chosen - logp_rejected)
    else:
        diff_current = logp_chosen - logp_rejected            # (B,)
        diff_ref = logp_ref_chosen - logp_ref_rejected        # (B,)
        diff = cfg.beta * (diff_current - diff_ref)

    # IPO if label smoothing > 0; standard DPO otherwise.
    if cfg.label_smoothing > 0.0:
        # IPO: square-error variant, more conservative
        loss = (diff - 1.0 / (2.0 * cfg.beta)) ** 2
        loss = loss.mean()
    else:
        loss = -F.logsigmoid(diff).mean()

    # Diagnostics
    with torch.no_grad():
        chosen_rewards = cfg.beta * (logp_chosen - logp_ref_chosen)
        rejected_rewards = cfg.beta * (logp_rejected - logp_ref_rejected)
        accuracy = (chosen_rewards > rejected_rewards).float().mean()
        margin = (chosen_rewards - rejected_rewards).mean()

    return DPOOutput(
        loss=loss,
        chosen_rewards=chosen_rewards,
        rejected_rewards=rejected_rewards,
        accuracy=accuracy,
        margin=margin,
    )


def kl_estimate_k1(logp: torch.Tensor, logp_ref: torch.Tensor) -> torch.Tensor:
    """K1 (low-variance, biased) KL estimator: KL ≈ ref - cur (positive iff
    current >> ref).

    Schulman's 2020 blog post:
    http://joschu.net/blog/kl-approx.html
    """
    return logp - logp_ref


def kl_estimate_k2(logp: torch.Tensor, logp_ref: torch.Tensor) -> torch.Tensor:
    """K2 (unbiased, lower-variance) KL estimator: 0.5 * (log_x)^2."""
    log_x = logp_ref - logp
    return 0.5 * log_x ** 2


def kl_estimate_k3(logp: torch.Tensor, logp_ref: torch.Tensor) -> torch.Tensor:
    """K3 (unbiased) KL estimator: e^{-x} - 1 + x where x = ref - cur.

    Most often used in TRL, RLHF stacks, and DeepSeek-R1 GRPO.
    """
    log_x = logp_ref - logp
    return torch.exp(log_x) - log_x - 1.0


__all__ = [
    "DPOConfig",
    "DPOOutput",
    "dpo_loss",
    "kl_estimate_k1",
    "kl_estimate_k2",
    "kl_estimate_k3",
]
