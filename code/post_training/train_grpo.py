"""GRPO trainer for VLA — base-agnostic.

Differs from DPO trainer in two essential ways:

1. **Online rollout**: GRPO needs K candidate samples drawn from the
   current policy at each training step. We use the rollout.py
   collector but in "single batch" mode rather than full dataset
   pre-collection. Each step:
       a. Sample K=group_size candidate chunks from current policy
       b. Roll each out in LIBERO env, compute reward
       c. Compute log-prob of each chunk under current policy + ref
       d. GRPO loss + backprop

2. **No paired (chosen, rejected) — group of K**: advantage is
   computed group-relative within each prompt, so all K samples
   contribute to the loss simultaneously.

Wall-clock implication: each step is **much** more expensive than
DPO (which uses pre-collected pairs). Per-step cost ≈ K × episode_time
+ K × forward. For LIBERO Spatial @ 220 steps × ~8 step/s combined
≈ 30s/episode × 8 candidates = 4 min/step.

For v1.5 paper, GRPO experiments are deliberately smaller scale:
  - 500-1000 steps (vs 5000 for DPO)
  - K = 4 (group size)
  - subset of LIBERO tasks per cell

Usage:
    python -m post_training.train_grpo \\
        --base openvla \\
        --base_ckpt /workspace/models/openvla-7b-finetuned-libero-spatial \\
        --suite libero_spatial \\
        --output_dir /workspace/output/openvla_grpo_spatial \\
        --max_steps 500 \\
        --group_size 4
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from . import PostTrainConfig
from .grpo_loss import GRPOConfig, grpo_loss


# ---------------------------------------------------------------------- #
# Adapter factory + scheduler (shared with train_dpo)
# ---------------------------------------------------------------------- #


def build_adapter(cfg: PostTrainConfig):
    if cfg.base == "openvla":
        from .adapters.openvla import OpenVLAAdapter
        return OpenVLAAdapter(cfg)
    if cfg.base == "spirit":
        from .adapters.spirit import SpiritAdapter
        return SpiritAdapter(cfg)
    raise ValueError(f"unsupported base for GRPO: {cfg.base}")


def build_cosine_scheduler(
    optimizer, warmup_steps: int, total_steps: int, base_lr: float, final_lr: float
):
    decay_steps = max(1, total_steps - warmup_steps)
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = min(max((step - warmup_steps) / decay_steps, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return final_lr / base_lr + (1.0 - final_lr / base_lr) * cosine
    return LambdaLR(optimizer, lr_lambda)


# ---------------------------------------------------------------------- #
# GRPO single-step rollout + train
# ---------------------------------------------------------------------- #


def grpo_step_rollout(
    adapter,
    env_factory,
    instruction: str,
    initial_state,
    group_size: int,
    cfg: PostTrainConfig,
):
    """Roll out group_size candidates for one (instruction, init).

    Args:
        adapter: VLABase
        env_factory: callable () → fresh env (so we can roll K times
                     from the same init_state)
        instruction: language prompt
        initial_state: env.set_init_state input
        group_size: K
        cfg: PostTrainConfig (used for reward weights, suite)

    Returns:
        chunks: (K, T_max, A) tensor — variable-length action sequences
                                       padded with last-action repeat
        rewards: (K,) tensor — chunk_reward per candidate
        first_image: (3, H, W) — first decision-step obs (shared across K)
        masks: (K,) bool — True if episode produced at least 1 action
    """
    from .rollout import rollout_one_episode, compute_episode_reward, RolloutConfig

    # LIBERO suite key normalisation (long10 → 10 in the benchmark dict)
    _suite_short = "10" if cfg.libero_suite == "long10" else cfg.libero_suite
    rcfg = RolloutConfig(
        suite=f"libero_{_suite_short}",
        n_candidates_per_init=group_size,
        sample_temperature=1.0,
        deterministic_first=False,    # for GRPO all K should be stochastic
        print_progress=False,
    )

    chunks_per_k = []
    rewards_per_k = []
    first_image = None
    masks = []

    for k in range(group_size):
        env = env_factory()
        episode = rollout_one_episode(
            adapter, env, instruction, initial_state, rcfg
        )
        env.close()
        reward = compute_episode_reward(episode, rcfg)
        actions = episode["actions"]
        if first_image is None:
            first_image = episode["first_image"]
        chunks_per_k.append(torch.from_numpy(actions).float())
        rewards_per_k.append(reward)
        masks.append(actions.shape[0] > 0)

    # Pad to T_max
    T_max = max((c.shape[0] for c in chunks_per_k if c.shape[0] > 0), default=1)
    A = chunks_per_k[0].shape[-1] if chunks_per_k[0].shape[0] > 0 else 7
    def _pad(c: torch.Tensor) -> torch.Tensor:
        if c.shape[0] == 0:
            return torch.zeros(T_max, A)
        if c.shape[0] == T_max:
            return c
        return torch.cat([c, c[-1:].expand(T_max - c.shape[0], -1)], dim=0)

    chunks = torch.stack([_pad(c) for c in chunks_per_k])     # (K, T_max, A)
    rewards = torch.tensor(rewards_per_k, dtype=torch.float32)
    masks_t = torch.tensor(masks, dtype=torch.bool)
    return chunks, rewards, first_image, masks_t


def parse_args():
    p = argparse.ArgumentParser(description="GRPO trainer for VLA")
    p.add_argument("--base", choices=["openvla", "spirit"], required=True)
    p.add_argument("--base_ckpt", required=True)
    p.add_argument(
        "--suite",
        default="spatial",
        choices=["spatial", "object", "goal", "long10"],
    )
    p.add_argument("--n_tasks", type=int, default=10)
    p.add_argument("--n_inits_per_task", type=int, default=2,
                   help="GRPO uses fewer inits per task to keep total cells "
                        "manageable; balance via more steps.")
    p.add_argument("--group_size", type=int, default=4,
                   help="K — number of candidate samples per step")
    p.add_argument("--output_dir", required=True)
    p.add_argument(
        "--max_chunk_len",
        type=int,
        default=0,
        help="Truncate rollout chunks to at most this many steps before "
             "policy_logp forward (0 = no truncation). Recommended: 220 on "
             "144GB H20, 180 on 96GB H20. Goal/Long10 episodes reach 300/520 "
             "steps and will OOM without truncation.",
    )

    # Training
    p.add_argument("--max_steps", type=int, default=500)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--save_every", type=int, default=100)
    p.add_argument("--log_every", type=int, default=5)

    # GRPO
    p.add_argument("--beta", type=float, default=0.1)
    p.add_argument("--epsilon", type=float, default=0.2)
    p.add_argument("--kl", choices=["k1", "k2", "k3"], default="k3")
    p.add_argument("--adv_norm", choices=["group", "global", "none"], default="group")

    # LoRA
    p.add_argument("--lora_r", type=int, default=32)
    p.add_argument("--lora_alpha", type=int, default=64)
    p.add_argument("--no_lora", action="store_true")

    return p.parse_args()


def main():
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    log_path = out / "train_log.jsonl"
    log_f = open(log_path, "a")

    # Config
    cfg = PostTrainConfig(
        name=f"{args.base}_grpo_{args.suite}",
        base=args.base,
        algorithm="grpo",
        base_ckpt_path=args.base_ckpt,
        libero_suite=args.suite,
        max_train_steps=args.max_steps,
        warmup_steps=args.warmup,
        learning_rate=args.lr,
        grad_clip_norm=args.grad_clip,
        use_lora=not args.no_lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        n_rollouts_for_pair_gen=args.group_size,
        grpo_beta=args.beta,
        grpo_epsilon=args.epsilon,
        grpo_kl_estimator=args.kl,
        grpo_advantage_normalization=args.adv_norm,
        save_steps=args.save_every,
        log_interval=args.log_every,
    )
    print(f"[cfg] {cfg.name}")

    # Build adapter
    adapter = build_adapter(cfg)
    adapter.freeze_reference()

    # Build env factory + task pool (for sampling per step)
    import torch as _t
    _orig_load = _t.load
    _t.load = lambda f, *a, **kw: _orig_load(f, *a, **{**kw, "weights_only": False})
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    # LIBERO benchmark dict uses "libero_10" (not "libero_long10"); normalise.
    _suite_short = "10" if cfg.libero_suite == "long10" else cfg.libero_suite
    suite_name = f"libero_{_suite_short}"
    bench = benchmark.get_benchmark_dict()
    task_suite = bench[suite_name]()
    task_pool = []
    for task_id in range(min(args.n_tasks, task_suite.n_tasks)):
        task = task_suite.get_task(task_id)
        bddl = os.path.join(get_libero_path("bddl_files"),
                            task.problem_folder, task.bddl_file)
        inits = task_suite.get_task_init_states(task_id)
        for init_idx in range(min(args.n_inits_per_task, len(inits))):
            task_pool.append({
                "bddl": bddl,
                "instruction": task.language,
                "init_state": inits[init_idx],
            })
    print(f"[data] {len(task_pool)} (task, init) pairs in pool")

    def env_factory_for(bddl, resolution=224):
        return OffScreenRenderEnv(
            bddl_file_name=bddl,
            camera_heights=resolution, camera_widths=resolution,
        )

    # Optimiser
    trainable = list(adapter.trainable_parameters())
    n_trainable = sum(p.numel() for p in trainable)
    print(f"[optim] {n_trainable/1e6:.2f}M trainable params")

    optim = AdamW(trainable, lr=cfg.learning_rate, betas=(0.9, 0.95),
                  weight_decay=cfg.weight_decay)
    sched = build_cosine_scheduler(
        optim, cfg.warmup_steps, cfg.max_train_steps,
        cfg.learning_rate, cfg.learning_rate * 0.1,
    )

    grpo_cfg = GRPOConfig(
        beta=cfg.grpo_beta,
        epsilon=cfg.grpo_epsilon,
        kl_estimator=cfg.grpo_kl_estimator,
        advantage_normalization=cfg.grpo_advantage_normalization,
    )

    # Train loop
    print(f"[train] starting {cfg.max_train_steps} steps, K={args.group_size}")
    t_start = time.time()
    rng = np.random.default_rng(42)

    for step in range(cfg.max_train_steps):
        # Pick random (task, init)
        prompt_data = task_pool[rng.integers(0, len(task_pool))]

        # 1. Rollout K candidates (no grad)
        with torch.no_grad():
            chunks, rewards, first_image, masks = grpo_step_rollout(
                adapter,
                lambda b=prompt_data["bddl"]: env_factory_for(b),
                prompt_data["instruction"],
                prompt_data["init_state"],
                args.group_size,
                cfg,
            )
        if not masks.any():
            print(f"[step {step}] all rollouts failed, skipping")
            continue

        # 2. Compute logp of each chunk under current + ref policy
        chunks = chunks.to(adapter.device)
        rewards = rewards.to(adapter.device)
        # Optional chunk truncation to avoid OOM on Goal/Long10 with 96GB H20
        if args.max_chunk_len > 0 and chunks.shape[1] > args.max_chunk_len:
            chunks = chunks[:, : args.max_chunk_len]
        # Build a per-K batch (each K shares same image + instruction)
        K = chunks.shape[0]
        batch = {
            "instruction": [prompt_data["instruction"]] * K,
            "image": first_image.unsqueeze(0).expand(K, -1, -1, -1).contiguous(),
        }

        logp_cur, logp_ref = adapter.policy_logp_with_ref(batch, chunks)
        # logp_old = logp_cur.detach() at this step (one-step trust region)
        logp_old = logp_cur.detach()

        # GRPO loss expects (B=1, K) shape
        # NOTE: name is `out_grpo`, not `out`, to avoid shadowing the
        # `out = Path(args.output_dir)` defined at the top of main(),
        # which we need below for checkpoint paths.
        out_grpo = grpo_loss(
            logp_cur.unsqueeze(0),
            logp_old.unsqueeze(0),
            logp_ref.unsqueeze(0),
            rewards.unsqueeze(0),
            grpo_cfg,
            mask=masks.any().unsqueeze(0).to(adapter.device),
        )
        loss = out_grpo.loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, cfg.grad_clip_norm)
        optim.step()
        sched.step()
        optim.zero_grad()

        if step % cfg.log_interval == 0:
            entry = {
                "step": step,
                "loss": float(loss.item()),
                "pg_loss": float(out_grpo.pg_loss.item()),
                "kl_loss": float(out_grpo.kl_loss.item()),
                "mean_advantage": float(out_grpo.mean_advantage.item()),
                "mean_ratio": float(out_grpo.mean_ratio.item()),
                "mean_kl": float(out_grpo.mean_kl.item()),
                "clip_fraction": float(out_grpo.clip_fraction.item()),
                "mean_reward": float(rewards.mean().item()),
                "max_reward": float(rewards.max().item()),
                "lr": sched.get_last_lr()[0],
                "elapsed_s": time.time() - t_start,
            }
            log_f.write(json.dumps(entry) + "\n")
            log_f.flush()
            print(
                f"[step {step:4d}] loss={loss.item():+.4f}  "
                f"pg={out_grpo.pg_loss.item():+.4f}  kl={out_grpo.kl_loss.item():+.4f}  "
                f"r̄={rewards.mean().item():+.3f}  r↑={rewards.max().item():+.3f}  "
                f"adv̄={out_grpo.mean_advantage.item():+.3f}  "
                f"clip={out_grpo.clip_fraction.item():.2f}"
            )

        if (step + 1) % cfg.save_steps == 0:
            ckpt = out / f"checkpoint-{step+1}.pt"
            adapter.save(str(ckpt))
            print(f"[ckpt] → {ckpt}")

    final = out / f"checkpoint-{cfg.max_train_steps}.pt"
    adapter.save(str(final))
    log_f.close()
    print(f"[done] {cfg.max_train_steps} steps, {time.time()-t_start:.1f}s")


if __name__ == "__main__":
    main()
