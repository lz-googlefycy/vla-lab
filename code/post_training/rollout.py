"""LIBERO env-coupled rollout collection.

Builds the (chosen, rejected) DPO pair dataset for one suite by:
  1. For each (task, initial_state):
       a. Roll out K candidate episodes under current policy + sampling
          temperature ∈ {0.5, 1.0, 1.5} (or stochastic)
       b. Compute scalar reward per episode (success * 0.7 + progress *
          0.2 + smoothness * 0.1 — see reward.py)
       c. Save observation at first decision step + chunk of K rollouts
  2. After all (task, init) tuples processed:
       a. For each tuple, pick top-2 episodes as 'chosen', bottom-2 as
          'rejected', form 4 pairs (or fewer if rewards tied)
       b. Save bundle to .pt file in DPOPairDataset format

Important architectural choice:
  For OpenVLA single-step action, we record the FULL trajectory action
  sequence (length T = up-to-220 for spatial) as the "chunk". DPO logp
  is computed teacher-force-style over this full chunk in
  OpenVLAAdapter.policy_logp.

  For Spirit / π0.5 60-step chunk models, we'd save only the first
  60-step chunk produced by the policy at each (task, init). Different
  semantics, same .pt schema.

Reference
---------
  - openvla's experiments/robot/libero/run_libero_eval.py: env loop
  - reward.py: chunk_reward + build_dpo_pairs
  - interface.py: SamplingResult container
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from PIL import Image

# Make local libero submodule importable when running inside container
_LIBERO_CANDIDATES = [
    os.environ.get("LIBERO_SRC"),
    "/workspace/LIBERO",
    str(Path.home() / "openpi" / "third_party" / "libero"),
]
for _p in _LIBERO_CANDIDATES:
    if _p and Path(_p).exists() and _p not in sys.path:
        sys.path.insert(0, _p)


# ---------------------------------------------------------------------- #
# Config
# ---------------------------------------------------------------------- #


@dataclass
class RolloutConfig:
    """Knobs for the rollout collector."""

    # LIBERO
    suite: str = "libero_spatial"           # libero_spatial / libero_object / libero_goal / libero_10
    n_tasks: int = 10                        # tasks per suite (LIBERO has 10)
    n_inits_per_task: int = 5                # initial states to sample per task
    # Render at 224 directly so we skip the resize step. OpenVLA's
    # official eval uses 256 → resize 224 with tf.image.lanczos3 +
    # JPEG round-trip, which is hard to reproduce exactly. Direct 224
    # render avoids that distribution shift, at the cost of a slightly
    # different camera FOV vs official eval.
    resolution: int = 224
    model_input_size: int = 224
    num_steps_wait: int = 10                 # initial settling steps
    max_steps_override: Optional[int] = None  # override default per-suite max_steps

    # Sampling
    n_candidates_per_init: int = 4           # K candidates per (task, init)
    sample_temperature: float = 1.0
    sample_top_p: float = 0.9
    deterministic_first: bool = True         # one of the K is greedy as anchor

    # Output
    output_path: str = ""                    # where to save the .pt bundle
    save_videos: bool = False
    video_dir: str = ""

    # Diagnostics
    print_progress: bool = True


# ---------------------------------------------------------------------- #
# LIBERO env helper
# ---------------------------------------------------------------------- #


def get_libero_env(task, resolution: int = 256):
    """Adapted from openvla/experiments/robot/libero/libero_utils.py.

    OpenVLA's official LIBERO eval uses resolution=256 (env render) +
    resize 224 (model input). Match that for direct comparability.
    """
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    bddl_dir = get_libero_path("bddl_files")
    bddl_file = os.path.join(bddl_dir, task.problem_folder, task.bddl_file)
    env_args = {
        "bddl_file_name": bddl_file,
        "camera_heights": resolution,
        "camera_widths": resolution,
    }
    env = OffScreenRenderEnv(**env_args)
    env.seed(0)
    task_description = task.language
    return env, task_description


def get_libero_dummy_action() -> np.ndarray:
    """7-dof zero action used during initial wait steps."""
    return np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0], dtype=np.float32)


# Per-suite max steps (matching openvla eval defaults)
_MAX_STEPS = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_90": 400,
}


# ---------------------------------------------------------------------- #
# One episode rollout
# ---------------------------------------------------------------------- #


def rollout_one_episode(
    adapter,
    env,
    task_description: str,
    initial_state: Any,
    cfg: RolloutConfig,
    sample_kwargs: Optional[dict] = None,
) -> dict:
    """Roll out one full episode, return outcome dict.

    Args:
        adapter: VLABase
        env: LIBERO OffScreenRenderEnv (already constructed)
        task_description: language string for the task
        initial_state: env.set_init_state(initial_state) input
        cfg: RolloutConfig
        sample_kwargs: dict of sampling kwargs forwarded to adapter
                       (temperature, top_p, do_sample, ...)

    Returns:
        {
            "actions":        (T, A) np.float32 — actually executed
            "first_image":    (3, H, W) tensor — observation at first
                              decision step (after settling)
            "success":        bool — env reported terminated as success
            "n_steps":        int — actual length T
            "task":           task_description
            "initial_state":  initial_state (for re-rollout if needed)
        }
    """
    sample_kwargs = sample_kwargs or {}

    # Toggle stochastic sampling on the adapter (if it supports it).
    # Used to make K candidates differ; greedy → all K identical.
    deterministic = sample_kwargs.get("deterministic", False)
    if hasattr(adapter, "_sample_do_sample"):
        adapter._sample_do_sample = not deterministic
    elif not deterministic:
        adapter._sample_do_sample = True
    if not deterministic:
        adapter._sample_temperature = float(sample_kwargs.get("temperature", 1.0))
        adapter._sample_top_p = float(sample_kwargs.get("top_p", 0.95))

    env.reset()
    obs = env.set_init_state(initial_state)

    suite_max = cfg.max_steps_override or _MAX_STEPS.get(cfg.suite, 220)
    actions: list[np.ndarray] = []
    first_image: Optional[torch.Tensor] = None
    success = False
    n_steps = 0
    t = 0

    # Settling phase
    for _ in range(cfg.num_steps_wait):
        obs, _, done, _ = env.step(get_libero_dummy_action().tolist())
        t += 1

    # Main loop
    while t < suite_max + cfg.num_steps_wait:
        # Get image from obs — LIBERO returns images keyed by camera name
        img_arr = obs.get("agentview_image")
        if img_arr is None:
            # Try other common keys
            for k in ("image", "rgb_static", "agent_view_image"):
                if k in obs:
                    img_arr = obs[k]
                    break
        if img_arr is None:
            raise RuntimeError(f"No image found in obs keys: {list(obs.keys())}")

        # LIBERO renders agentview rotated 180° relative to OpenVLA's
        # training distribution. See OpenVLA's libero_utils.get_libero_image:
        # `img = img[::-1, ::-1]`. Same rotation needed here.
        img = img_arr[::-1, ::-1]
        if img.dtype != np.uint8:
            img = (img * 255).astype(np.uint8)
        # Resize from env-render resolution to model input size (224 for
        # OpenVLA). PIL bilinear matches OpenVLA's training pipeline.
        if img.shape[0] != cfg.model_input_size:
            from PIL import Image
            img = np.asarray(
                Image.fromarray(img).resize(
                    (cfg.model_input_size, cfg.model_input_size),
                    Image.BILINEAR,
                )
            )
        img_t = torch.from_numpy(img.copy()).permute(2, 0, 1).float() / 255.0   # (3, H, W)

        if first_image is None:
            first_image = img_t.clone()

        # Build batch and sample one action
        batch = {
            "instruction": [task_description],
            "image": img_t.unsqueeze(0),
        }
        with torch.no_grad():
            # OpenVLA single-step prediction
            action_chunk = adapter.select_action(batch)        # (1, 1, 7) for OpenVLA
        action = action_chunk.squeeze(0).squeeze(0).cpu().numpy()  # (7,)

        # OpenVLA: gripper in [0, 1] → binarise to {-1, +1}, then invert
        # because OpenVLA's RLDS dataloader maps 0=close, 1=open but
        # LIBERO env expects -1=open, +1=close. See
        # openvla/experiments/robot/robot_utils.py.
        if hasattr(adapter, 'cfg') and getattr(adapter.cfg, 'base', '') == 'openvla':
            # Step 1: normalize gripper to [-1, +1] then binarise
            action[-1] = 1.0 if action[-1] > 0.5 else -1.0
            # Step 2: invert sign for LIBERO
            action[-1] = -action[-1]
        else:
            # Spirit / π0.5 / others: simple binarise
            action[-1] = 1.0 if action[-1] > 0.5 else -1.0

        actions.append(action.copy())
        try:
            obs, _, done, info = env.step(action.tolist())
        except Exception as e:
            if cfg.print_progress:
                print(f"  [warn] env.step exception: {e}")
            break

        n_steps = t - cfg.num_steps_wait + 1
        t += 1

        # LIBERO's `done` flag isn't always set on success — use the explicit
        # check_success() method as the canonical signal.
        if done or (hasattr(env, "check_success") and env.check_success()):
            success = True
            break

    if first_image is None:
        first_image = torch.zeros(3, cfg.resolution, cfg.resolution)

    return {
        "actions": np.stack(actions) if actions else np.zeros((0, 7), dtype=np.float32),
        "first_image": first_image,
        "success": bool(success),
        "n_steps": n_steps,
        "task": task_description,
        "initial_state": initial_state,
    }


# ---------------------------------------------------------------------- #
# Compute chunk reward
# ---------------------------------------------------------------------- #


def compute_episode_reward(episode: dict, cfg: RolloutConfig) -> float:
    """Reward for one episode using reward.py weights (default 0.7/0.2/0.1).

    LIBERO env doesn't expose a continuous progress signal, so we use a
    simple binary mapping: success → progress=1.0, failure → progress=0.0.
    Earlier draft used length-fraction as a partial-credit proxy, but
    that produced identical rewards for all failed episodes (they all
    run to max_steps), wiping out the smoothness-based signal that
    distinguishes within the failure cluster.
    """
    from .reward import chunk_reward, RewardConfig

    success = episode["success"]
    actions = episode["actions"]
    progress = 1.0 if success else 0.0
    return chunk_reward(actions, success=success, progress=progress, cfg=RewardConfig())


# ---------------------------------------------------------------------- #
# Main collection loop
# ---------------------------------------------------------------------- #


def collect_pair_dataset(adapter, cfg: RolloutConfig) -> dict:
    """Collect (task, init, K episodes) → DPO pair bundle.

    Returns the dict in DPOPairDataset format, ready to torch.save().
    """
    from libero.libero import benchmark

    # LIBERO ships init_states as torch pickles that aren't safe-loadable
    # under torch>=2.6 default weights_only=True. Patch torch.load locally
    # to allow numpy globals before LIBERO touches them.
    import torch as _t
    try:
        _t.serialization.add_safe_globals(
            [_t.serialization.add_safe_globals.__globals__.get("Tensor")]  # noqa
        )
    except Exception:
        pass
    _orig_load = _t.load
    def _patched_load(f, *a, **kw):
        if "weights_only" not in kw:
            kw["weights_only"] = False
        return _orig_load(f, *a, **kw)
    _t.load = _patched_load

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.suite]()
    n_tasks = min(cfg.n_tasks, task_suite.n_tasks)

    all_chunks: list[torch.Tensor] = []      # each (T, A) — variable T per episode
    all_rewards: list[float] = []
    all_first_images: list[torch.Tensor] = []
    all_instructions: list[str] = []
    all_success: list[bool] = []
    group_ids: list[int] = []                 # which (task, init) group each ep belongs to

    group_counter = 0
    t_total_start = time.time()

    for task_id in range(n_tasks):
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        env, task_description = get_libero_env(task, resolution=cfg.resolution)

        if cfg.print_progress:
            print(f"\n=== Task {task_id+1}/{n_tasks}: {task_description}")

        for init_idx in range(min(cfg.n_inits_per_task, len(initial_states))):
            init_state = initial_states[init_idx]

            for k in range(cfg.n_candidates_per_init):
                deterministic = (k == 0 and cfg.deterministic_first)
                t0 = time.time()
                episode = rollout_one_episode(
                    adapter, env, task_description, init_state, cfg,
                    sample_kwargs=dict(
                        deterministic=deterministic,
                        temperature=cfg.sample_temperature,
                        top_p=cfg.sample_top_p,
                    ),
                )
                reward = compute_episode_reward(episode, cfg)

                all_chunks.append(torch.from_numpy(episode["actions"]).float())
                all_rewards.append(reward)
                all_first_images.append(episode["first_image"])
                all_instructions.append(episode["task"])
                all_success.append(episode["success"])
                group_ids.append(group_counter)

                if cfg.print_progress:
                    print(
                        f"  group {group_counter:3d}  k={k}  "
                        f"len={episode['n_steps']:3d}  "
                        f"success={episode['success']!s:5s}  reward={reward:+.3f}  "
                        f"({time.time()-t0:.1f}s)"
                    )
            group_counter += 1
        env.close()

    # Now build (chosen, rejected) pairs per group
    chosen_chunks: list[torch.Tensor] = []
    rejected_chunks: list[torch.Tensor] = []
    chosen_imgs: list[torch.Tensor] = []
    chosen_instrs: list[str] = []
    chosen_rs: list[float] = []
    rejected_rs: list[float] = []

    n_groups = max(group_ids) + 1 if group_ids else 0
    for gid in range(n_groups):
        members = [i for i, g in enumerate(group_ids) if g == gid]
        if len(members) < 2:
            continue
        rewards_g = [all_rewards[i] for i in members]
        # Sort descending by reward
        order = sorted(members, key=lambda i: -all_rewards[i])
        # n_pairs = floor(K/2), pair top with bottom
        n_pairs = len(order) // 2
        for p in range(n_pairs):
            top_i = order[p]
            bot_i = order[-1 - p]
            if all_rewards[top_i] <= all_rewards[bot_i]:
                continue  # tie, skip
            chosen_chunks.append(all_chunks[top_i])
            rejected_chunks.append(all_chunks[bot_i])
            chosen_imgs.append(all_first_images[top_i])
            chosen_instrs.append(all_instructions[top_i])
            chosen_rs.append(all_rewards[top_i])
            rejected_rs.append(all_rewards[bot_i])

    # Variable-length chunks → pad to max length T_max with last action repeated
    if chosen_chunks:
        T_max = max(c.shape[0] for c in chosen_chunks + rejected_chunks)
        A = chosen_chunks[0].shape[-1]
        def _pad(c: torch.Tensor) -> torch.Tensor:
            T = c.shape[0]
            if T == T_max:
                return c
            if T == 0:
                return torch.zeros(T_max, A, dtype=c.dtype)
            pad = c[-1:].expand(T_max - T, -1)
            return torch.cat([c, pad], dim=0)
        chosen_chunks_t = torch.stack([_pad(c) for c in chosen_chunks])
        rejected_chunks_t = torch.stack([_pad(c) for c in rejected_chunks])
    else:
        chosen_chunks_t = torch.empty(0)
        rejected_chunks_t = torch.empty(0)

    bundle = {
        "instructions": chosen_instrs,
        "images": torch.stack(chosen_imgs) if chosen_imgs else torch.empty(0),
        "chosen_chunks": chosen_chunks_t,
        "rejected_chunks": rejected_chunks_t,
        "chosen_rewards": torch.tensor(chosen_rs, dtype=torch.float32),
        "rejected_rewards": torch.tensor(rejected_rs, dtype=torch.float32),
        # Diagnostics (not used by trainer)
        "_collection_stats": {
            "n_tasks": n_tasks,
            "n_inits_per_task": cfg.n_inits_per_task,
            "n_candidates_per_init": cfg.n_candidates_per_init,
            "n_episodes_total": len(all_chunks),
            "n_pairs": len(chosen_chunks),
            "success_rate": float(sum(all_success) / max(1, len(all_success))),
            "elapsed_s": time.time() - t_total_start,
        },
    }
    return bundle


# ---------------------------------------------------------------------- #
# CLI
# ---------------------------------------------------------------------- #


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", choices=["openvla", "spirit", "pi05"], required=True)
    p.add_argument("--base_ckpt", required=True)
    p.add_argument(
        "--suite",
        default="libero_spatial",
        choices=["libero_spatial", "libero_object", "libero_goal", "libero_10"],
    )
    p.add_argument("--n_tasks", type=int, default=10)
    p.add_argument("--n_inits_per_task", type=int, default=5)
    p.add_argument("--n_candidates_per_init", type=int, default=4)
    p.add_argument("--resolution", type=int, default=224)
    p.add_argument("--output_path", required=True)
    p.add_argument("--max_steps_override", type=int, default=None)
    args = p.parse_args()

    # Build adapter
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from post_training import PostTrainConfig
    from post_training.train_dpo import build_adapter

    cfg_pt = PostTrainConfig(
        name=f"{args.base}_rollout_{args.suite}",
        base=args.base,
        algorithm="dpo",
        base_ckpt_path=args.base_ckpt,
        libero_suite=args.suite.replace("libero_", ""),
        use_lora=False,   # no LoRA for rollout — we want base policy
    )
    adapter = build_adapter(cfg_pt)

    # Configure rollout
    cfg = RolloutConfig(
        suite=args.suite,
        n_tasks=args.n_tasks,
        n_inits_per_task=args.n_inits_per_task,
        n_candidates_per_init=args.n_candidates_per_init,
        resolution=args.resolution,
        max_steps_override=args.max_steps_override,
        output_path=args.output_path,
    )

    bundle = collect_pair_dataset(adapter, cfg)

    out = Path(args.output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bundle, out)
    stats = bundle["_collection_stats"]
    print(f"\n[done] saved {stats['n_pairs']} pairs from {stats['n_episodes_total']} episodes")
    print(f"  success rate: {stats['success_rate']:.1%}")
    print(f"  elapsed: {stats['elapsed_s']/60:.1f} min")
    print(f"  → {out}")


if __name__ == "__main__":
    main()
