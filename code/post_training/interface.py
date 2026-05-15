"""Common interface for all VLA bases (OpenVLA / Spirit v1.5 / π0.5).

Goal: let the same DPO / GRPO trainer work across three architecturally
different VLA models without per-base if/else inside the trainer.

Key design constraint: log-probability and sampling have very different
shapes across bases:

  - OpenVLA: action is 7-bin discretised tokens × 7 DoF × T steps.
            log p(chunk) = sum log p(token_t,d | prefix)
            naturally well-defined.
  - Spirit / π0.5: action chunk is continuous, predicted by a flow-matching
            head over T steps. log p(chunk) requires an ODE solver
            (probability flow ODE) or an ELBO surrogate. The clean
            approach for DPO is to pass through `policy_logp` an
            estimator interface and let each adapter implement the
            best estimator for its head.

Therefore `VLABase.policy_logp()` returns a *scalar log-prob estimate
per chunk*. The estimator's choice (exact vs surrogate) is the
adapter's responsibility, not the trainer's.

Future-compat note: when running GRPO, we additionally need
`policy_sample()` to roll out K candidate chunks under the current
policy. The interface intentionally separates `_logp` and `_sample`
so adapters that can't implement one can still implement the other.

References
----------
- `docs/plan_v1.5.md`: research plan
- `docs/openpi_onboard.md`: openpi/π0.5 specifics
- `code/spirit_adapter/`: Spirit-specific code reused by SpiritAdapter
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterator, Literal, Optional, Protocol, runtime_checkable

import torch


# ---------------------------------------------------------------------- #
# 1. PostTrainConfig — top-level config, dataclass-based per openpi style
# ---------------------------------------------------------------------- #


@dataclass
class PostTrainConfig:
    """Top-level config for one post-training experiment cell.

    Attributes are deliberately a flat dataclass so we can serialize to
    yaml/json and inspect easily. One PostTrainConfig = one row in the
    experiment matrix (3 bases × 2 algorithms = 6 rows for v1.5 paper).
    """

    # Identity
    name: str                          # e.g. "openvla_dpo_spatial"
    base: Literal["openvla", "spirit", "pi05"]
    algorithm: Literal["dpo", "grpo", "sft"]

    # Base model
    base_ckpt_path: str                # local path or HF ID
    base_dtype: str = "bfloat16"

    # Dataset
    libero_suite: Literal["spatial", "object", "goal", "long10", "all4"] = "spatial"
    n_rollouts_for_pair_gen: int = 8   # K candidates per instruction (also GRPO group size)

    # Training hyperparameters
    batch_size: int = 4
    learning_rate: float = 1e-4
    max_train_steps: int = 5000
    warmup_steps: int = 200
    weight_decay: float = 1e-4
    grad_clip_norm: float = 1.0

    # LoRA / DoRA (parameter-efficient fine-tuning)
    use_lora: bool = True
    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    lora_target_modules: tuple[str, ...] = ("to_q", "to_k", "to_v", "to_out.0")
    also_train_proj: bool = True       # full-train state/action projection layers
    # DoRA = Weight-Decomposed LoRA (NVIDIA, ICML 2024 spotlight).
    # When True, peft will decompose W = m * (W_dir / ||W_dir||) and apply LoRA
    # only to the direction component; magnitude `m` is trained as a separate
    # vector. Same memory footprint as LoRA, training ~10% slower, +1~3 acc points
    # on LLM/VL benchmarks. Recommended for VLA where pretrain prior matters.
    use_dora: bool = False

    # DPO-specific
    dpo_beta: float = 0.1
    dpo_pair_size: int = 4             # number of (chosen, rejected) pairs per instruction

    # GRPO-specific
    grpo_beta: float = 0.1             # KL coefficient
    grpo_epsilon: float = 0.2          # clip ratio
    grpo_kl_estimator: Literal["k1", "k2"] = "k2"
    grpo_advantage_normalization: Literal["group", "global"] = "global"

    # Reward signal
    reward_success_weight: float = 0.7
    reward_progress_weight: float = 0.2
    reward_smoothness_weight: float = 0.1

    # Output / logging
    output_dir: str = "./output"
    log_interval: int = 10
    save_steps: int = 1000

    # Eval
    eval_after_train: bool = True
    eval_n_trials_per_task: int = 50   # paper-grade: 50 × 10 tasks × 3 seeds

    def __post_init__(self):
        if self.base == "spirit" and self.algorithm == "dpo":
            # Spirit's flow-matching log-prob requires ODE solver; we
            # may need to fall back to a surrogate. Loud warning, no fail.
            import warnings
            warnings.warn(
                "Spirit + DPO uses a surrogate log-prob estimator; see "
                "adapter docstring for the trade-off."
            )


# ---------------------------------------------------------------------- #
# 2. VLABase — runtime protocol every adapter must implement
# ---------------------------------------------------------------------- #


@runtime_checkable
class VLABase(Protocol):
    """Common protocol for all VLA base adapters.

    Each concrete adapter (OpenVLAAdapter, SpiritAdapter, Pi05Adapter)
    wraps the upstream policy class and exposes the methods below.

    Lifecycle:
        1. ``__init__(cfg)`` loads the base ckpt, applies LoRA if
           requested, moves to GPU.
        2. Trainer calls ``policy_logp(batch)`` and/or
           ``policy_sample(batch, K)`` repeatedly during training.
        3. ``save(path)`` / ``load(path)`` for checkpointing.
    """

    cfg: PostTrainConfig

    # ----- training-time interface ----- #

    def policy_logp(self, batch: dict, chunk: torch.Tensor) -> torch.Tensor:
        """Log-probability of `chunk` under current policy.

        Args:
            batch: standard Spirit-style batch dict
                   (observation.state, observation.images.*, task,
                   robot_type, ...). NOT containing actions.
            chunk: (B, T, A) action tensor.

        Returns:
            (B,) tensor of log-prob estimates. Estimator quality varies
            by adapter; see each adapter's docstring.
        """
        ...

    def policy_logp_with_ref(
        self, batch: dict, chunk: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Joint log-prob under (current, frozen-ref) policies.

        Used for DPO loss computation; computing both in one pass shares
        the forward computation.

        Returns:
            (logp_current, logp_ref), each shape (B,)
        """
        ...

    @torch.no_grad()
    def policy_sample(
        self,
        batch: dict,
        n_samples: int,
        deterministic: bool = False,
    ) -> torch.Tensor:
        """Sample K candidate chunks under current policy.

        Args:
            batch: as above.
            n_samples: K.
            deterministic: if True, returns the policy's single-best
                output (no sampling). Useful for eval.

        Returns:
            (B, K, T, A) action tensor, with K samples per element.
        """
        ...

    # ----- weight management ----- #

    def trainable_parameters(self) -> Iterator[torch.nn.Parameter]:
        """Yield all parameters currently set to ``requires_grad=True``."""
        ...

    def freeze_reference(self) -> None:
        """Snapshot current policy as the frozen reference for DPO/GRPO.

        Most implementations clone state_dict of LoRA params + base
        weights into a separate `self.reference_model` and call
        ``requires_grad_(False)`` on it.
        """
        ...

    def save(self, path: str) -> None:
        """Save trainable params + config to ``path``."""
        ...

    def load(self, path: str) -> None:
        """Restore trainable params from ``path``."""
        ...

    # ----- inference / eval ----- #

    @torch.no_grad()
    def select_action(self, batch: dict) -> torch.Tensor:
        """Run one chunk inference for env stepping.

        Returns:
            (B, T, A) action chunk, ready to feed into the LIBERO env.
        """
        ...


# ---------------------------------------------------------------------- #
# 3. SamplingResult — type-safe container for K candidate samples
# ---------------------------------------------------------------------- #


@dataclass
class SamplingResult:
    """One round of sampling K candidates per instruction.

    Used both for DPO pair generation (where we keep top-2/bottom-2 by
    reward) and for GRPO rollout (where we keep all K with their
    log-probs and rewards).
    """

    chunks: torch.Tensor      # (B, K, T, A)
    logps: torch.Tensor       # (B, K) — log-prob under sampling policy
    rewards: Optional[torch.Tensor] = None   # (B, K), filled after env eval
    success: Optional[torch.Tensor] = None   # (B, K) bool, filled after env eval


__all__ = [
    "PostTrainConfig",
    "VLABase",
    "SamplingResult",
]
