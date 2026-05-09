"""
Synthetic dataset for Spirit v1.5 LoRA fine-tune pipeline validation.

Purpose
-------
Phase B-1 needs to validate the training pipeline (data → model → loss →
backward → optim step) BEFORE real XLeRobot teleop data exists. This file
produces a deterministic, geometrically-plausible dataset that:

  - matches the Spirit batch dict schema 1:1 (drops into train_lora.py
    unchanged when real data arrives)
  - has nontrivial structure (target loss should drop, not stay random)
  - is small (a few seconds to load, no I/O)
  - covers all 5 instructions we already used in Phase A demos so the
    instruction-conditional path through the VLM is exercised

What "synthetic" means here
---------------------------
For each (instruction, episode) we:

  1. Sample a random START 12-DoF joint config in a safe SO-100 range
  2. Pick a TARGET 12-DoF joint config (also safe range)
  3. Linear-interpolate START→TARGET over `episode_length` frames
  4. Add small smooth noise (low-freq sine perturbation) to make it
     non-degenerate
  5. Render a "fake" head-camera image: a colored solid block for each
     instruction (red/blue/yellow/green/purple), so the visual encoder
     gets a unique input per instruction. Real-looking enough to drive
     the VLM down a deterministic path; obviously fake enough that the
     model cannot accidentally solve the task by visual recognition.

This is enough to verify:
  - dataset.__getitem__ produces correct shapes
  - DataLoader collates correctly
  - model forward returns finite loss
  - loss decreases with training (the model SHOULD learn to map "red
    block visual + instruction X" → trajectory toward target X)
  - LoRA gradients flow into the right modules

It is NOT enough to deploy on a real robot. That's Phase B-3.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset


# Five instructions — same set we used in Phase A demos
INSTRUCTIONS = [
    "pick up the red cube and place it on the blue plate",
    "put the coffee cup into the cabinet",
    "fold the white towel in half",
    "open the drawer and put the apple inside",
    "pour the contents of the bottle into the glass",
]

# A unique solid color per instruction (RGB in [0,1]) — used to render
# the fake head camera image so each instruction has a distinct visual
# input.
INSTRUCTION_COLORS = [
    (0.85, 0.20, 0.20),   # red    — cube/plate
    (0.30, 0.40, 0.80),   # blue   — cup/cabinet
    (0.95, 0.95, 0.85),   # white  — towel
    (0.30, 0.70, 0.30),   # green  — drawer/apple
    (0.65, 0.40, 0.85),   # purple — bottle/glass
]


# SO-100 single-arm joint limits (rough, in rad except gripper which is 0..1)
# (matches XLeRobot/software/.../SO101Robot.py rough ranges)
SO100_JOINT_LO = np.array([-2.0, -1.5, -2.0, -1.5, -3.0, 0.0])
SO100_JOINT_HI = np.array([+2.0, +1.5, +2.0, +1.5, +3.0, 1.0])


@dataclass
class SyntheticDataConfig:
    """Configuration for SyntheticSpiritDataset."""

    # Spirit hyperparams (must match policy)
    action_horizon: int = 60
    chunk_size: int = 60
    state_history: int = 1

    # Synthesis
    n_episodes_per_instruction: int = 4   # 5 × 4 = 20 episodes total
    episode_length: int = 120             # frames per episode
    img_size: tuple[int, int] = (240, 320)
    noise_amplitude: float = 0.05         # rad / unitless joint noise
    seed: int = 42

    # Match the real dataset's batch dict
    robot_type: str = "Franka"            # Spirit was trained on Franka
    tile_head_cam: bool = True

    # Encoding — only "raw_joint" is supported here for simplicity.
    # The synthetic data lives in joint space; real EE-pose encoding
    # requires URDF FK and is exercised by the real LeRobot dataset.
    encoding: str = "raw_joint"


class SyntheticSpiritDataset(Dataset):
    """A small in-memory dataset matching Spirit's batch dict schema."""

    def __init__(self, config: Optional[SyntheticDataConfig] = None):
        self.cfg = config or SyntheticDataConfig()
        self.rng = np.random.default_rng(self.cfg.seed)

        # Build episodes: list of dicts with state[T,12], action[T,12], task_idx
        self._episodes: list[dict] = []
        for task_idx, instruction in enumerate(INSTRUCTIONS):
            for ep_in_task in range(self.cfg.n_episodes_per_instruction):
                self._episodes.append(self._make_episode(task_idx, ep_in_task))

        # Flat (ep_idx, frame_idx) index — leave 2 frames at end like Spirit
        self._index: list[tuple[int, int]] = []
        for ep_idx, ep in enumerate(self._episodes):
            ep_len = len(ep["state"])
            for fr in range(max(0, ep_len - 2)):
                self._index.append((ep_idx, fr))

        # Pre-render one head-camera image per instruction (cached)
        self._cam_cache: dict[int, torch.Tensor] = {}
        for task_idx in range(len(INSTRUCTIONS)):
            self._cam_cache[task_idx] = self._render_fake_camera(task_idx)

    def _make_episode(self, task_idx: int, ep_in_task: int) -> dict:
        """Generate one synthetic episode for a task."""
        T = self.cfg.episode_length

        # Each task has its own START → TARGET region so an instruction
        # truly maps to a different motion (not just same trajectory with
        # different label). Use task_idx to bias the corner of the
        # safe-range box.
        bias = (task_idx / max(1, len(INSTRUCTIONS) - 1))  # 0..1
        start_lo = SO100_JOINT_LO + 0.3 * (SO100_JOINT_HI - SO100_JOINT_LO) * bias
        start_hi = SO100_JOINT_LO + 0.5 * (SO100_JOINT_HI - SO100_JOINT_LO) * bias
        target_lo = SO100_JOINT_LO + 0.5 * (SO100_JOINT_HI - SO100_JOINT_LO) * (1 - bias)
        target_hi = SO100_JOINT_LO + 0.7 * (SO100_JOINT_HI - SO100_JOINT_LO) * (1 - bias)

        # Per-arm random start/target
        start_l = self.rng.uniform(start_lo, start_hi)
        start_r = self.rng.uniform(start_lo, start_hi)
        target_l = self.rng.uniform(target_lo, target_hi)
        target_r = self.rng.uniform(target_lo, target_hi)

        # Linear interpolation
        t = np.linspace(0.0, 1.0, T).reshape(-1, 1)  # (T, 1)
        traj_l = (1 - t) * start_l + t * target_l    # (T, 6)
        traj_r = (1 - t) * start_r + t * target_r

        # Smooth low-freq noise so the trajectory is not perfectly linear
        # — we use 2 sin components per joint
        freqs = self.rng.uniform(0.5, 2.0, size=(2, 6))   # (2 freqs, 6 joints)
        phases = self.rng.uniform(0.0, 2 * np.pi, size=(2, 6))
        time_axis = np.linspace(0.0, 2 * np.pi, T)   # (T,)
        noise_l = self.cfg.noise_amplitude * (
            np.sin(time_axis[:, None] * freqs[0:1] + phases[0:1])
            + 0.5 * np.sin(time_axis[:, None] * freqs[1:2] + phases[1:2])
        )
        noise_r = self.cfg.noise_amplitude * (
            np.cos(time_axis[:, None] * freqs[0:1] + phases[0:1])
            + 0.5 * np.cos(time_axis[:, None] * freqs[1:2] + phases[1:2])
        )
        traj_l = traj_l + noise_l
        traj_r = traj_r + noise_r

        # Clip to safe range
        traj_l = np.clip(traj_l, SO100_JOINT_LO, SO100_JOINT_HI)
        traj_r = np.clip(traj_r, SO100_JOINT_LO, SO100_JOINT_HI)

        # Concatenate into 12-DoF state per frame
        state_12 = np.concatenate([traj_l, traj_r], axis=1).astype(np.float32)
        # Action at frame t = state at t+1 (next-step joint command)
        action_12 = np.zeros_like(state_12)
        action_12[:-1] = state_12[1:]
        action_12[-1] = state_12[-1]

        return {
            "state": state_12,        # (T, 12)
            "action": action_12,      # (T, 12)
            "task_idx": task_idx,
            "task": INSTRUCTIONS[task_idx],
        }

    def _render_fake_camera(self, task_idx: int) -> torch.Tensor:
        """Render a 3×H×W solid-color image with a small distinguishing block.

        The base color encodes which instruction this is. A small offset
        block in the lower-right corner gives the visual encoder some
        non-trivial structure to attend to.
        """
        H, W = self.cfg.img_size
        color = INSTRUCTION_COLORS[task_idx]
        img = np.full((H, W, 3), color, dtype=np.float32)
        # Add a darker block in the lower-right quarter — simulates
        # "object on a surface" structure
        h2, w2 = H // 2, W // 2
        block_color = np.array(color) * 0.5
        img[h2:, w2:, :] = block_color
        # Add gradient across width — gives the encoder spatial cue
        ramp = np.linspace(0.9, 1.1, W).reshape(1, W, 1)
        img = np.clip(img * ramp, 0.0, 1.0)
        # to (3, H, W) torch float32 in [0, 1]
        return torch.from_numpy(img).permute(2, 0, 1).contiguous().float()

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> dict:
        ep_idx, frame_idx = self._index[idx]
        ep = self._episodes[ep_idx]

        # State (current frame), zero-pad 12→14
        state_12 = ep["state"][frame_idx]
        state_14 = np.zeros(14, dtype=np.float32)
        state_14[:12] = state_12
        state_mask = np.zeros(14, dtype=bool)
        state_mask[:12] = True

        # Action chunk: next chunk_size frames as delta (joint-space).
        # End-pad with last valid action like Spirit's convention.
        T = len(ep["state"])
        actions_14 = np.zeros((self.cfg.action_horizon, 14), dtype=np.float32)
        action_mask = np.zeros((self.cfg.action_horizon, 14), dtype=bool)
        action_is_pad = np.ones(self.cfg.action_horizon, dtype=bool)

        last_valid = None
        num_steps = min(self.cfg.chunk_size, self.cfg.action_horizon)
        for i in range(num_steps):
            tgt = frame_idx + i + 1
            if tgt < T:
                tgt_state_12 = ep["state"][tgt]
                # Joint-space delta for non-gripper joints, absolute for grippers
                delta = tgt_state_12 - state_12
                delta[5] = tgt_state_12[5]    # left gripper absolute
                delta[11] = tgt_state_12[11]  # right gripper absolute
                actions_14[i, :12] = delta
                action_mask[i, :12] = True
                action_is_pad[i] = False
                last_valid = actions_14[i].copy()
            elif last_valid is not None:
                actions_14[i] = last_valid
                action_mask[i, :12] = True
                action_is_pad[i] = True

        # Camera (cached per task)
        cam_high = self._cam_cache[ep["task_idx"]]
        if self.cfg.tile_head_cam:
            cam_left = cam_high.clone()
            cam_right = cam_high.clone()
        else:
            # Synthetic dataset only ever has one cam, but we expose the
            # flag for symmetry with the real LeRobot dataset.
            cam_left = cam_high.clone()
            cam_right = cam_high.clone()

        return {
            "observation.state": torch.from_numpy(state_14).unsqueeze(0),
            "observation.state.mask": torch.from_numpy(state_mask).unsqueeze(0),
            "observation.images.cam_high": cam_high,
            "observation.images.cam_left_wrist": cam_left,
            "observation.images.cam_right_wrist": cam_right,
            "action": torch.from_numpy(actions_14),
            "action_mask": torch.from_numpy(action_mask),
            "action_is_pad": torch.from_numpy(action_is_pad),
            "task": ep["task"],
            "robot_type": self.cfg.robot_type,
        }

    @staticmethod
    def collate_fn(batch: list[dict]) -> dict:
        """Same convention as XLeRobotSpiritDataset.collate_fn."""
        result = {}
        for key in batch[0]:
            values = [b[key] for b in batch]
            if isinstance(values[0], torch.Tensor):
                result[key] = torch.stack(values)
            else:
                result[key] = values
        return result


# --------------------------------------------------------------------- #
# CLI sanity check
# --------------------------------------------------------------------- #

if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=2)
    p.add_argument("--save_camera_png", type=str, default="",
                   help="If set, dump the 5 fake camera images here")
    args = p.parse_args()

    ds = SyntheticSpiritDataset()
    print(f"Dataset: {len(ds)} samples / {len(ds._episodes)} episodes / "
          f"{len(INSTRUCTIONS)} instructions")
    for i in range(min(args.n, len(ds))):
        b = ds[i]
        print(f"\n--- sample {i} (task='{b['task'][:40]}...') ---")
        for k, v in b.items():
            if isinstance(v, torch.Tensor):
                stat = ""
                if v.dtype.is_floating_point:
                    stat = f"  μ={v.float().mean().item():+.3f}  σ={v.float().std().item():.3f}"
                print(f"  {k:40s}  {tuple(v.shape)}  {v.dtype}{stat}")
            else:
                print(f"  {k:40s}  {v!r}")

    if args.save_camera_png:
        out = Path(args.save_camera_png)
        out.mkdir(parents=True, exist_ok=True)
        from torchvision.utils import save_image
        for task_idx, color in enumerate(INSTRUCTION_COLORS):
            img = ds._cam_cache[task_idx]
            save_image(img, out / f"task_{task_idx}_{INSTRUCTIONS[task_idx][:20].replace(' ','_')}.png")
        print(f"\nSaved {len(INSTRUCTIONS)} fake camera images to {out}")
