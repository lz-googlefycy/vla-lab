"""LIBERO RLDS dataset reader for multi-view + temporal pretraining.

Replaces the synthetic mode with a real LIBERO RLDS reader (TFRecord +
TFDS). The OpenVLA team's ``modified_libero_rlds`` dataset is the canonical
source — same data they used for OpenVLA SFT, so any feature we learn
with this is transferable to OpenVLA / π0.5 fine-tunes.

Schema (per step):
    observation.image:        (256, 256, 3) uint8 — agent view
    observation.wrist_image:  (256, 256, 3) uint8 — wrist view
    language_instruction:     bytes (constant per episode)
    action:                   (7,) float32 (xyz + rpy + gripper)

Output per __getitem__ (keeps compatibility with synthetic mode):
    agent_t:        (3, 224, 224) float32 in [-1, 1]
    wrist_t:        (3, 224, 224)
    agent_t_delta:  (3, 224, 224)
    task_id:        int (hash of language_instruction)
    episode_id:     int

Design choices:
  * Δ_steps default 5 (~0.5s @ 10 Hz, matches paper convention)
  * max_per_episode 30 (each episode contributes 30 random anchor times,
    keeping data balanced across tasks)
  * For 4 LIBERO suites × 432 ep × 30 = ~50K samples per epoch, plenty
    for a 656K param projection head.
  * Pre-load entire dataset into RAM as numpy uint8 (cheap: 4 suite ×
    432 ep × 200 step × 2 view × 256² × 3 bytes ≈ 130 GB worst-case;
    we lazy-load + use TFDS prefetch instead).

Why RLDS / TFRecord (not h5py):
  * That's what OpenVLA team published. Using same source = directly
    comparable. OpenVLA's official LIBERO HDF5 format is a different
    dataset prepared for the LIBERO env, which is harder to align
    cross-suite.
"""
from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

# Force CPU for TF (we only use TFDS to load TFRecords; image processing
# is on CPU/GPU via torch). NOTE: This must NOT touch CUDA_VISIBLE_DEVICES
# since torch may already be using GPU. We hide GPU only from TF via
# tf.config.set_visible_devices below at first use.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")


def _to_tensor_minus1_1(img_uint8: np.ndarray, target_size: int = 224) -> torch.Tensor:
    """uint8 (H, W, 3) → float32 (3, target_size, target_size) in [-1, 1]."""
    if img_uint8.shape[0] != target_size:
        import cv2
        img_uint8 = cv2.resize(img_uint8, (target_size, target_size), interpolation=cv2.INTER_LANCZOS4)
    t = torch.from_numpy(img_uint8).float() / 255.0
    t = t * 2.0 - 1.0
    return t.permute(2, 0, 1).contiguous()


class LiberoRLDSPretrainDataset(Dataset):
    """LIBERO RLDS-backed multi-view + temporal contrastive dataset.

    Args:
        rlds_root: directory containing ``libero_{spatial,object,goal,10}_no_noops``.
            Default cloudml path: ``/ad-alg/planning-users/liuzhi7/ro_planning/datasets/modified_libero_rlds``.
            Default shijihulian path: TBD (rsync prerequisite).
        suites: which suites to include (default all 4)
        delta_steps: temporal stride (default 5)
        max_per_episode: anchor times per episode (default 30)
        max_episodes_per_suite: cap (default None = all 432). Useful for
            quick smoke runs (set 50 for ~5K samples).
    """

    def __init__(
        self,
        rlds_root: str,
        suites: tuple[str, ...] = ("spatial", "object", "goal", "10"),
        delta_steps: int = 5,
        max_per_episode: int = 30,
        max_episodes_per_suite: int | None = None,
        target_size: int = 224,
        verbose: bool = True,
    ):
        # Lazy import — TF is heavy and only used here. Keep tf module
        # accessible so we can use tf.data.AUTOTUNE below.
        import tensorflow as tf
        tf.config.set_visible_devices([], "GPU")
        import tensorflow_datasets as tfds
        self._tf_module = tf  # for tf.data.AUTOTUNE

        self.delta_steps = delta_steps
        self.max_per_episode = max_per_episode
        self.target_size = target_size

        # Per-episode storage: list of dicts with
        #   {"agent": (T, 256, 256, 3) uint8 ndarray, "wrist": same, "instr": bytes}
        # Resolved once at __init__, kept in RAM (each suite ~1.8 GB).
        self.episodes: list[dict] = []
        self.samples: list[dict] = []   # (ep_idx, t_idx)

        import time as _time
        for suite in suites:
            suite_dir = Path(rlds_root) / f"libero_{suite}_no_noops" / "1.0.0"
            if not suite_dir.exists():
                if verbose:
                    print(f"[rlds] WARNING: {suite_dir} not found, skipping suite {suite}")
                continue
            if verbose:
                print(f"[rlds] loading suite {suite} from {suite_dir}")
            t_start = _time.time()

            ds = tfds.builder_from_directory(str(suite_dir)).as_dataset(split="train")
            if max_episodes_per_suite is not None:
                ds = ds.take(max_episodes_per_suite)
            # TFDS prefetch + parallel: tfds.as_numpy iterates faster than
            # plain for-loop with .numpy() on each tensor (avoids per-step
            # Eager → numpy roundtrips).
            ds = ds.prefetch(tf.data.AUTOTUNE)

            n_eps = 0
            for ep_np in tfds.as_numpy(ds):
                # ep_np: dict with "steps" containing a sub-dict where each
                # value is an ndarray with leading T axis.
                steps = ep_np["steps"]
                # tfds.as_numpy on a nested DatasetSpec returns generators
                # for the sub-dataset; we materialise them.
                if isinstance(steps, dict) and isinstance(steps["observation"], dict):
                    agent_frames = steps["observation"]["image"]  # (T, 256, 256, 3)
                    wrist_frames = steps["observation"]["wrist_image"]
                    instrs = steps["language_instruction"]
                else:
                    # Fallback: legacy tfds where steps is iterable
                    agent_list, wrist_list, instr = [], [], b""
                    for st in steps:
                        agent_list.append(st["observation"]["image"])
                        wrist_list.append(st["observation"]["wrist_image"])
                        instr = st["language_instruction"]
                    agent_frames = np.stack(agent_list, axis=0)
                    wrist_frames = np.stack(wrist_list, axis=0)
                    instrs = np.array([instr])

                T = agent_frames.shape[0]
                if T <= delta_steps + 1:
                    continue
                instr0 = instrs[0] if hasattr(instrs, "__len__") else instrs
                if isinstance(instr0, (bytes, np.bytes_)):
                    instr0 = instr0.decode("utf-8", errors="ignore")
                else:
                    instr0 = str(instr0)
                ep_idx = len(self.episodes)
                self.episodes.append({
                    "agent": agent_frames,
                    "wrist": wrist_frames,
                    "instr": instr0,
                    "suite": suite,
                })
                rng = random.Random(ep_idx)
                valid_ts = list(range(0, T - delta_steps - 1))
                rng.shuffle(valid_ts)
                for t in valid_ts[:max_per_episode]:
                    self.samples.append({"ep_idx": ep_idx, "t": t})
                n_eps += 1
                if verbose and n_eps % 50 == 0:
                    elapsed = _time.time() - t_start
                    print(f"[rlds]   {suite}: {n_eps} eps in {elapsed:.0f}s "
                          f"({n_eps/elapsed:.1f} ep/s)")
            if verbose:
                elapsed = _time.time() - t_start
                print(f"[rlds] {suite}: {n_eps} episodes, "
                      f"{sum(1 for s in self.samples if self.episodes[s['ep_idx']]['suite']==suite)} samples "
                      f"({elapsed:.0f}s)")

        if verbose:
            print(f"[rlds] total: {len(self.episodes)} episodes, {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx) -> dict:
        s = self.samples[idx]
        ep = self.episodes[s["ep_idx"]]
        t = s["t"]
        td = t + self.delta_steps

        agent_t = ep["agent"][t]
        wrist_t = ep["wrist"][t]
        agent_t_delta = ep["agent"][td]

        # Hash instr → task_id
        task_id = hash(ep["instr"]) % 1000

        return {
            "agent_t": _to_tensor_minus1_1(agent_t, self.target_size),
            "wrist_t": _to_tensor_minus1_1(wrist_t, self.target_size),
            "agent_t_delta": _to_tensor_minus1_1(agent_t_delta, self.target_size),
            "task_id": task_id,
            "episode_id": s["ep_idx"],
        }
