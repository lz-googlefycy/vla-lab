"""
LeRobot → Spirit v1.5 dataset adapter.

Background
----------
Spirit's official `RoboChallengeDataset` (`spirit-v1.5/dataset/dataset.py`)
expects:
  - state encoding: 7D end-effector pose
        [x, y, z,  rx, ry, rz,  gripper]
        (rotvec from a 3D quat, gripper width as 1D scalar)
        zero-padded to 14D (left half = valid 7D, right half = 0)
  - action encoding (delta):
        [Δx, Δy, Δz,  Δrx, Δry, Δrz,  target_gripper]
        zero-padded to 14D, padded along chunk_size with action_is_pad mask
  - 3 cameras: cam_high + cam_left_wrist + cam_right_wrist (3×240×320 RGB)
  - robot_type:  the open-sourced ckpt was trained on a single-arm Franka
                 ("Franka"), not ALOHA. The 14D = 7D + zeros(7) is the
                 forward-compat dual-arm padding.

XLeRobot reality (SO-100 bimanual)
-----------------------------------
  - 12-DoF joint space (6 per arm, no forearm_roll)
  - 1 head camera (no wrist cams)
  - lerobot dataset format: per-frame {state[12], action[12], cam_high}
  - frequency: 30 Hz teleop demos

Two adaptation strategies
-------------------------
  (a) **dual-EE**: forward-kinematics each arm to a 7D EE pose, stack ⇒ 14D
                   (mirrors Franka encoding for both arms)
                   Pros: most faithful to Spirit's training distribution
                   Cons: needs URDF + working FK, 2×7D not exactly 14D from
                         Spirit's perspective
  (c) **single-arm-as-Franka**: use only the left arm, FK to 7D EE,
                                zero-pad right half to 14D
                                Pros: minimal changes, matches Franka
                                      encoding exactly
                                Cons: ignores right arm + assumes
                                      pick-place-style single-arm tasks

Strategy is selected by `XLeRobotDataConfig.encoding`.

This file is the pure dataset; LoRA training script lives next door at
`train_lora.py` and uses this dataset's `__getitem__` output as Spirit
batch dicts.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


# --------------------------------------------------------------------- #
# 1. Config
# --------------------------------------------------------------------- #


@dataclass
class XLeRobotDataConfig:
    """Configuration for XLeRobotSpiritDataset."""

    # Dataset location — a LeRobot-format dataset root, see lerobot docs:
    # https://huggingface.co/docs/lerobot
    data_root: str = ""

    # Encoding strategy — see module docstring §"Two adaptation strategies"
    encoding: str = "single_arm"  # "single_arm" | "dual_ee" | "raw_joint"

    # Spirit dataset hyperparams (must match Spirit's policy config)
    action_horizon: int = 60
    chunk_size: int = 60
    state_history: int = 1

    # Frequency of the source teleop data (Hz)
    fps: float = 30.0

    # Image processing
    img_size: tuple[int, int] = (240, 320)  # (H, W)

    # When SO-100 has only a head cam, tile it to all 3 slots Spirit expects.
    # Set False once we have wrist cams (the future).
    tile_head_cam: bool = True

    # robot_type string sent to Spirit's prompt.
    # Options seen in source: "Franka", "ARX5", "aloha", "ur5".
    # We default to "Franka" to match the open-sourced ckpt's training data.
    robot_type: str = "Franka"

    # Optional: which arm to use in single_arm mode ("left" | "right")
    primary_arm: str = "left"

    # URDF path for FK (only needed when encoding != "raw_joint").
    # Defaults to a sibling-checkout layout — set explicitly if elsewhere.
    # Try in order: $XLEROBOT_URDF, repo-sibling, ~/XLeRobot.
    urdf_path: str = ""

    # Names of the kinematic chains in the URDF.
    # Used for FK end-effector pose extraction.
    left_arm_ee_link: str = "left_gripper"
    right_arm_ee_link: str = "right_gripper"

    # SO-100 single-arm joint order in the LeRobot state vector
    # (matches XLeRobot/software/.../SO101Robot.py:JOINT_NAMES)
    so100_joint_names_left: tuple[str, ...] = (
        "left_shoulder_pan",
        "left_shoulder_lift",
        "left_elbow_flex",
        "left_wrist_flex",
        "left_wrist_roll",
        "left_gripper",
    )
    so100_joint_names_right: tuple[str, ...] = (
        "right_shoulder_pan",
        "right_shoulder_lift",
        "right_elbow_flex",
        "right_wrist_flex",
        "right_wrist_roll",
        "right_gripper",
    )


# --------------------------------------------------------------------- #
# 2. FK helpers
# --------------------------------------------------------------------- #


class _FKHelper:
    """Lightweight FK wrapper. Lazy-loads pinocchio or PyKDL.

    Expected libs: pinocchio (pip install pinocchio) or
    pytorch_kinematics. Falls back to a stub that returns zeros if neither
    is available — the caller should validate at config time.
    """

    def __init__(self, urdf_path: str, ee_link: str):
        self.urdf_path = urdf_path
        self.ee_link = ee_link
        self._impl = None
        self._chain = None

    def _ensure_chain(self):
        if self._impl is not None:
            return
        try:
            import pytorch_kinematics as pk

            self._impl = "pytorch_kinematics"
            with open(self.urdf_path, "rb") as f:
                self._chain = pk.build_serial_chain_from_urdf(
                    f.read(), end_link_name=self.ee_link
                )
            return
        except ImportError:
            pass

        try:
            import pinocchio as pin

            self._impl = "pinocchio"
            self._chain = pin.buildModelFromUrdf(self.urdf_path)
            self._data = self._chain.createData()
            self._frame_id = self._chain.getFrameId(self.ee_link)
            return
        except ImportError:
            pass

        # No FK lib — return identity. Caller must check via .available().
        self._impl = "none"
        self._chain = None

    def available(self) -> bool:
        self._ensure_chain()
        return self._impl in ("pytorch_kinematics", "pinocchio")

    def fk(self, joint_angles: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Forward kinematics.

        Args:
            joint_angles: shape (n_joints,)
        Returns:
            xyz:  shape (3,)
            quat: shape (4,) in (qx, qy, qz, qw) order  (scipy convention)
        """
        self._ensure_chain()
        if self._impl == "pytorch_kinematics":
            import pytorch_kinematics as pk

            t = torch.tensor(joint_angles, dtype=torch.float32).unsqueeze(0)
            tf = self._chain.forward_kinematics(t)
            mat = tf.get_matrix()[0].numpy()  # (4,4)
            xyz = mat[:3, 3]
            from scipy.spatial.transform import Rotation as R

            quat = R.from_matrix(mat[:3, :3]).as_quat()  # (qx, qy, qz, qw)
            return xyz, quat

        if self._impl == "pinocchio":
            import pinocchio as pin

            q = np.asarray(joint_angles, dtype=np.float64)
            pin.forwardKinematics(self._chain, self._data, q)
            pin.updateFramePlacements(self._chain, self._data)
            tf = self._data.oMf[self._frame_id]
            xyz = tf.translation
            from scipy.spatial.transform import Rotation as R

            quat = R.from_matrix(tf.rotation).as_quat()
            return xyz, quat

        # Fallback identity — useful for unit-testing the encoding pipeline
        # without an FK lib installed; caller should NOT use this in training.
        return np.zeros(3), np.array([0.0, 0.0, 0.0, 1.0])


# --------------------------------------------------------------------- #
# 3. The dataset
# --------------------------------------------------------------------- #


class XLeRobotSpiritDataset(Dataset):
    """LeRobot dataset wrapped to Spirit v1.5's batch dict format.

    Expected on-disk layout (LeRobot v2 format)::

        data_root/
        ├── meta/
        │   ├── info.json          # global metadata
        │   ├── episodes.jsonl     # per-episode meta
        │   ├── tasks.jsonl        # task descriptions
        │   └── stats.json         # mean/std per feature (optional)
        ├── data/
        │   └── chunk-XXX/
        │       └── episode_NNNNNN.parquet
        └── videos/
            └── chunk-XXX/
                └── observation.images.cam_high/
                    └── episode_NNNNNN.mp4

    Each parquet row is one frame::

        {
            "observation.state": List[float],          # 12 for SO-100 bimanual
            "action": List[float],                     # 12
            "task_index": int,
            "episode_index": int,
            "frame_index": int,
            "timestamp": float,
            ... (camera fields stored as video files, not in parquet)
        }
    """

    def __init__(self, config: XLeRobotDataConfig):
        self.cfg = config
        self.root = Path(config.data_root)

        # Lazy: don't import pyarrow / cv2 until __init__ runs in actual env
        import pandas as pd  # noqa: F401  (validated)

        self.episodes_meta = self._load_episodes_meta()
        self.tasks_meta = self._load_tasks_meta()

        # Build (ep_idx, frame_idx) flat index for __getitem__
        self.index: list[tuple[int, int]] = []
        for ep in self.episodes_meta:
            ep_idx = ep["episode_index"]
            ep_len = ep["length"]
            # leave 2 frames at the end so we always have a valid future state
            for fr in range(max(0, ep_len - 2)):
                self.index.append((ep_idx, fr))

        # FK setup if needed
        self._fk_left = None
        self._fk_right = None
        if self.cfg.encoding in ("single_arm", "dual_ee"):
            urdf = self.cfg.urdf_path or self._find_urdf()
            if not urdf or not Path(urdf).exists():
                raise FileNotFoundError(
                    "FK encoding needs a valid xlerobot URDF. Either set "
                    "XLeRobotDataConfig.urdf_path explicitly, set the "
                    "XLEROBOT_URDF env var, or check out the XLeRobot repo "
                    "as a sibling of this one."
                )
            self._fk_left = _FKHelper(urdf, self.cfg.left_arm_ee_link)
            self._fk_right = _FKHelper(urdf, self.cfg.right_arm_ee_link)
            if not self._fk_left.available():
                raise RuntimeError(
                    "FK encoding requires pytorch_kinematics or pinocchio. "
                    "pip install pytorch_kinematics  OR  pip install pinocchio"
                )

    @staticmethod
    def _find_urdf() -> str:
        """Try common XLeRobot URDF locations; return first that exists."""
        import os
        candidates = [
            os.environ.get("XLEROBOT_URDF"),
            str(Path.home() / "XLeRobot/simulation/Maniskill/assets/xlerobot/xlerobot.urdf"),
            "/workspace/XLeRobot/simulation/Maniskill/assets/xlerobot/xlerobot.urdf",
        ]
        for c in candidates:
            if c and Path(c).exists():
                return c
        return ""

        # Cache of decoded video frames per (ep, cam) — avoid re-opening
        self._video_cache: dict[tuple[int, str], Any] = {}

        # Image transform — Resize to 240×320 to match Spirit's input
        from torchvision.transforms import Resize  # noqa: F811

        self._resize = Resize(self.cfg.img_size, antialias=True)

    # ---------- meta loaders ---------- #

    def _load_episodes_meta(self) -> list[dict]:
        path = self.root / "meta" / "episodes.jsonl"
        if not path.exists():
            raise FileNotFoundError(
                f"LeRobot meta missing: {path}\n"
                f"Expected layout: <data_root>/meta/episodes.jsonl"
            )
        out = []
        with open(path) as f:
            for line in f:
                out.append(json.loads(line))
        return out

    def _load_tasks_meta(self) -> dict[int, str]:
        """Return {task_index: prompt_str}."""
        path = self.root / "meta" / "tasks.jsonl"
        out = {}
        if not path.exists():
            return out  # tasks may be inlined in info.json instead
        with open(path) as f:
            for line in f:
                d = json.loads(line)
                out[d["task_index"]] = d.get("task", d.get("description", ""))
        return out

    # ---------- frame I/O ---------- #

    def _load_episode_parquet(self, ep_idx: int):
        """Find the parquet file for this episode."""
        import pandas as pd

        # LeRobot v2 stores chunk-XXX/episode_NNNNNN.parquet
        # We do a glob to be format-tolerant.
        candidates = list(
            self.root.glob(f"data/chunk-*/episode_{ep_idx:06d}.parquet")
        )
        if not candidates:
            raise FileNotFoundError(
                f"No parquet for episode {ep_idx} under {self.root}/data/"
            )
        return pd.read_parquet(candidates[0])

    def _load_video_frame(
        self, ep_idx: int, cam_name: str, frame_idx: int
    ) -> torch.Tensor:
        """Load a single frame from a LeRobot video file. Returns (3,H,W)."""
        import cv2

        cache_key = (ep_idx, cam_name)
        if cache_key not in self._video_cache:
            candidates = list(
                self.root.glob(
                    f"videos/chunk-*/{cam_name}/episode_{ep_idx:06d}.mp4"
                )
            )
            if not candidates:
                raise FileNotFoundError(
                    f"No video for ep {ep_idx} cam {cam_name}"
                )
            self._video_cache[cache_key] = str(candidates[0])

        cap = cv2.VideoCapture(self._video_cache[cache_key])
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        cap.release()
        if not ok:
            raise RuntimeError(
                f"Failed to read frame {frame_idx} from {self._video_cache[cache_key]}"
            )

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # (H, W, 3) → (3, H, W) float in [0, 1]
        t = torch.from_numpy(frame).permute(2, 0, 1).contiguous().float() / 255.0
        return self._resize(t)

    # ---------- encoding ---------- #

    def _encode_state(
        self, joint_state_12d: np.ndarray
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """SO-100 12-DoF joint state → Spirit 14D state + mask.

        Returns:
            state_14d:  (1, 14) float32
            state_mask: (1, 14) bool   (True = valid)
        """
        if self.cfg.encoding == "raw_joint":
            # Just put the 12 joint angles into the first 12 slots, zero-pad to 14
            state_14d = np.zeros(14, dtype=np.float32)
            state_14d[:12] = joint_state_12d
            mask = np.zeros(14, dtype=bool)
            mask[:12] = True
            return (
                torch.from_numpy(state_14d).unsqueeze(0),
                torch.from_numpy(mask).unsqueeze(0),
            )

        from scipy.spatial.transform import Rotation as R

        left_q = joint_state_12d[:6]   # 6 joints
        right_q = joint_state_12d[6:]  # 6 joints

        if self.cfg.encoding == "single_arm":
            primary = left_q if self.cfg.primary_arm == "left" else right_q
            xyz, quat = self._fk_left.fk(primary)
            rotvec = R.from_quat(quat).as_rotvec()
            gripper = primary[-1:]  # last joint = gripper
            state_7d = np.concatenate([xyz, rotvec, gripper]).astype(np.float32)
            state_14d = np.zeros(14, dtype=np.float32)
            state_14d[:7] = state_7d
            mask = np.zeros(14, dtype=bool)
            mask[:7] = True
            return (
                torch.from_numpy(state_14d).unsqueeze(0),
                torch.from_numpy(mask).unsqueeze(0),
            )

        if self.cfg.encoding == "dual_ee":
            xyz_l, quat_l = self._fk_left.fk(left_q)
            xyz_r, quat_r = self._fk_right.fk(right_q)
            rv_l = R.from_quat(quat_l).as_rotvec()
            rv_r = R.from_quat(quat_r).as_rotvec()
            grip_l = left_q[-1:]
            grip_r = right_q[-1:]
            state_14d = np.concatenate(
                [xyz_l, rv_l, grip_l, xyz_r, rv_r, grip_r]
            ).astype(np.float32)
            mask = np.ones(14, dtype=bool)
            return (
                torch.from_numpy(state_14d).unsqueeze(0),
                torch.from_numpy(mask).unsqueeze(0),
            )

        raise ValueError(f"Unknown encoding: {self.cfg.encoding}")

    def _encode_action_chunk(
        self, df, frame_idx: int, ep_len: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode delta-action chunk over the next `chunk_size` frames.

        Returns:
            actions:        (action_horizon, 14) float32
            action_mask:    (action_horizon, 14) bool  (True = valid dim)
            action_is_pad:  (action_horizon,) bool     (True = padded step)
        """
        from scipy.spatial.transform import Rotation as R

        cur_state_12 = np.asarray(df.iloc[frame_idx]["observation.state"])

        if self.cfg.encoding == "raw_joint":
            # Action is delta joint position (this is what LeRobot stores
            # by convention for SO-100 anyway).
            actions = np.zeros((self.cfg.action_horizon, 14), dtype=np.float32)
            mask = np.zeros((self.cfg.action_horizon, 14), dtype=bool)
            is_pad = np.ones(self.cfg.action_horizon, dtype=bool)

            num_steps = min(self.cfg.chunk_size, self.cfg.action_horizon)
            last_valid = None
            for i in range(num_steps):
                tgt = frame_idx + i + 1
                if tgt < ep_len:
                    target_state_12 = np.asarray(
                        df.iloc[tgt]["observation.state"]
                    )
                    # delta state for first 11 dims, absolute gripper for both arms
                    delta = target_state_12 - cur_state_12
                    delta[5] = target_state_12[5]   # left gripper absolute
                    delta[11] = target_state_12[11] # right gripper absolute
                    actions[i, :12] = delta
                    mask[i, :12] = True
                    is_pad[i] = False
                    last_valid = actions[i].copy()
                elif last_valid is not None:
                    actions[i] = last_valid
                    mask[i, :12] = True
                    is_pad[i] = True
            return (
                torch.from_numpy(actions),
                torch.from_numpy(mask),
                torch.from_numpy(is_pad),
            )

        # EE-pose encodings: compute current EE pose once, then delta to each future
        if self.cfg.encoding == "single_arm":
            primary_q = (
                cur_state_12[:6]
                if self.cfg.primary_arm == "left"
                else cur_state_12[6:]
            )
            cur_xyz, cur_quat = self._fk_left.fk(primary_q)
            cur_rot = R.from_quat(cur_quat)

            actions = np.zeros((self.cfg.action_horizon, 14), dtype=np.float32)
            mask = np.zeros((self.cfg.action_horizon, 14), dtype=bool)
            is_pad = np.ones(self.cfg.action_horizon, dtype=bool)

            num_steps = min(self.cfg.chunk_size, self.cfg.action_horizon)
            last_valid = None
            for i in range(num_steps):
                tgt = frame_idx + i + 1
                if tgt < ep_len:
                    tgt_state_12 = np.asarray(df.iloc[tgt]["observation.state"])
                    tgt_q = (
                        tgt_state_12[:6]
                        if self.cfg.primary_arm == "left"
                        else tgt_state_12[6:]
                    )
                    tgt_xyz, tgt_quat = self._fk_left.fk(tgt_q)
                    tgt_rot = R.from_quat(tgt_quat)
                    d_xyz = tgt_xyz - cur_xyz
                    d_rot = (tgt_rot * cur_rot.inv()).as_rotvec()
                    target_grip = tgt_q[-1:]
                    actions[i, :7] = np.concatenate([d_xyz, d_rot, target_grip])
                    mask[i, :7] = True
                    is_pad[i] = False
                    last_valid = actions[i].copy()
                elif last_valid is not None:
                    actions[i] = last_valid
                    mask[i, :7] = True
                    is_pad[i] = True
            return (
                torch.from_numpy(actions),
                torch.from_numpy(mask),
                torch.from_numpy(is_pad),
            )

        if self.cfg.encoding == "dual_ee":
            cur_l = self._fk_left.fk(cur_state_12[:6])
            cur_r = self._fk_right.fk(cur_state_12[6:])
            cur_rot_l = R.from_quat(cur_l[1])
            cur_rot_r = R.from_quat(cur_r[1])

            actions = np.zeros((self.cfg.action_horizon, 14), dtype=np.float32)
            mask = np.zeros((self.cfg.action_horizon, 14), dtype=bool)
            is_pad = np.ones(self.cfg.action_horizon, dtype=bool)

            num_steps = min(self.cfg.chunk_size, self.cfg.action_horizon)
            last_valid = None
            for i in range(num_steps):
                tgt = frame_idx + i + 1
                if tgt < ep_len:
                    tgt_state_12 = np.asarray(df.iloc[tgt]["observation.state"])
                    tgt_l = self._fk_left.fk(tgt_state_12[:6])
                    tgt_r = self._fk_right.fk(tgt_state_12[6:])
                    rot_l = R.from_quat(tgt_l[1])
                    rot_r = R.from_quat(tgt_r[1])
                    d_xyz_l = tgt_l[0] - cur_l[0]
                    d_xyz_r = tgt_r[0] - cur_r[0]
                    d_rv_l = (rot_l * cur_rot_l.inv()).as_rotvec()
                    d_rv_r = (rot_r * cur_rot_r.inv()).as_rotvec()
                    grip_l = tgt_state_12[5:6]
                    grip_r = tgt_state_12[11:12]
                    actions[i, :7] = np.concatenate([d_xyz_l, d_rv_l, grip_l])
                    actions[i, 7:14] = np.concatenate([d_xyz_r, d_rv_r, grip_r])
                    mask[i, :] = True
                    is_pad[i] = False
                    last_valid = actions[i].copy()
                elif last_valid is not None:
                    actions[i] = last_valid
                    mask[i, :] = True
                    is_pad[i] = True
            return (
                torch.from_numpy(actions),
                torch.from_numpy(mask),
                torch.from_numpy(is_pad),
            )

        raise ValueError(f"Unknown encoding: {self.cfg.encoding}")

    # ---------- pytorch Dataset interface ---------- #

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> dict:
        ep_idx, frame_idx = self.index[idx]
        ep_meta = next(e for e in self.episodes_meta if e["episode_index"] == ep_idx)
        ep_len = ep_meta["length"]
        task_idx = ep_meta.get("tasks", [0])[0]
        task_str = self.tasks_meta.get(task_idx, "")

        df = self._load_episode_parquet(ep_idx)

        # State (current frame)
        joint_12 = np.asarray(df.iloc[frame_idx]["observation.state"], dtype=np.float32)
        state, state_mask = self._encode_state(joint_12)

        # Action chunk (delta over next chunk_size frames)
        actions, action_mask, action_is_pad = self._encode_action_chunk(
            df, frame_idx, ep_len
        )

        # Images
        cam_high = self._load_video_frame(
            ep_idx, "observation.images.cam_high", frame_idx
        )
        if self.cfg.tile_head_cam:
            cam_left_wrist = cam_high.clone()
            cam_right_wrist = cam_high.clone()
        else:
            cam_left_wrist = self._load_video_frame(
                ep_idx, "observation.images.cam_left_wrist", frame_idx
            )
            cam_right_wrist = self._load_video_frame(
                ep_idx, "observation.images.cam_right_wrist", frame_idx
            )

        return {
            "observation.state": state,           # (1, 14)
            "observation.state.mask": state_mask, # (1, 14)
            "observation.images.cam_high": cam_high,
            "observation.images.cam_left_wrist": cam_left_wrist,
            "observation.images.cam_right_wrist": cam_right_wrist,
            "action": actions,                # (60, 14)
            "action_mask": action_mask,       # (60, 14)
            "action_is_pad": action_is_pad,   # (60,)
            "task": task_str,
            "robot_type": self.cfg.robot_type,
        }

    # ---------- helpers for batching ---------- #

    @staticmethod
    def collate_fn(batch: list[dict]) -> dict:
        """Same convention as Spirit's RoboChallengeDataset.collate_fn."""
        result = {}
        for key in batch[0]:
            values = [b[key] for b in batch]
            if isinstance(values[0], torch.Tensor):
                result[key] = torch.stack(values)
            else:
                result[key] = values
        return result


# --------------------------------------------------------------------- #
# 4. Quick CLI sanity-check
# --------------------------------------------------------------------- #

if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--data_root", required=True)
    p.add_argument(
        "--encoding",
        default="single_arm",
        choices=["single_arm", "dual_ee", "raw_joint"],
    )
    p.add_argument("--n", type=int, default=3, help="Print first N items")
    args = p.parse_args()

    cfg = XLeRobotDataConfig(data_root=args.data_root, encoding=args.encoding)
    ds = XLeRobotSpiritDataset(cfg)
    print(f"Loaded dataset: {len(ds)} samples / {len(ds.episodes_meta)} episodes")
    for i in range(min(args.n, len(ds))):
        b = ds[i]
        print(f"\n--- sample {i} ---")
        for k, v in b.items():
            if isinstance(v, torch.Tensor):
                print(f"  {k:40s}  {tuple(v.shape)}  {v.dtype}")
            else:
                print(f"  {k:40s}  {v!r}")
