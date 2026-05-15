"""LIBERO pretraining dataset: yields (agent_view, wrist_view, agent_view_t+Δ).

Source: LIBERO SFT demo trajectories (50 demo per task × 50 tasks ≈ 2500 traj).

Format:
  Each LIBERO task has demo HDF5 files under
  /workspace/LIBERO/libero/libero/datasets/{spatial,object,goal,10}/...
  with keys: data/demo_X/obs/agentview_rgb (N, H, W, 3), wrist_rgb, etc.

Output per __getitem__:
  agent_view_t:        (3, 224, 224) float32 in [-1, 1]
  wrist_view_t:        (3, 224, 224)
  agent_view_t_delta:  (3, 224, 224)
  task_id:             int
  episode_id:          int

For InfoNCE batching, sample without replacement per (task, episode) to
guarantee in-batch positive uniqueness.
"""
from __future__ import annotations
from pathlib import Path
import random
import numpy as np
import torch
from torch.utils.data import Dataset


def _to_tensor_minus1_1(img_uint8: np.ndarray) -> torch.Tensor:
    """uint8 (H, W, 3) → float32 (3, 224, 224) in [-1, 1]."""
    if img_uint8.shape[0] != 224:
        # Resize via cv2 (lanczos4) — assumes cv2 available
        import cv2
        img_uint8 = cv2.resize(img_uint8, (224, 224), interpolation=cv2.INTER_LANCZOS4)
    t = torch.from_numpy(img_uint8).float() / 255.0     # [0, 1]
    t = t * 2.0 - 1.0                                    # [-1, 1]
    return t.permute(2, 0, 1).contiguous()               # (3, H, W)


class LiberoPretrainDataset(Dataset):
    """Iterates (agent_t, wrist_t, agent_{t+Δ}) tuples from LIBERO demos.

    Args:
        root: base path containing demo HDF5 files. Auto-detects layout.
        suites: which LIBERO suites to include (default all 4).
        delta_steps: Δ for temporal consistency (default 5 = ~0.5s).
        max_per_episode: cap on (t) samples per episode (default 30).

    Note: This is a "dummy" implementation suitable for code review; in
    Day 7 we'll wire it to actual HDF5 reading. For Day 6 smoke test we
    use the synthetic mode to validate model + loss + dataloader API.
    """

    def __init__(
        self,
        root: str | None = None,
        suites: tuple[str, ...] = ("spatial", "object", "goal", "10"),
        delta_steps: int = 5,
        max_per_episode: int = 30,
        synthetic: bool = False,
        synthetic_n: int = 1000,
    ):
        self.root = Path(root) if root else None
        self.suites = suites
        self.delta_steps = delta_steps
        self.max_per_episode = max_per_episode
        self.synthetic = synthetic
        self.synthetic_n = synthetic_n
        self.samples: list[dict] = []

        if synthetic:
            self._build_synthetic()
        else:
            self._build_from_hdf5()

    def _build_synthetic(self):
        """Quick smoke-test data: random uint8 images."""
        for i in range(self.synthetic_n):
            self.samples.append({
                "task_id": i % 50,
                "episode_id": (i // 50) % 50,
                "step_t": i % 100,
            })

    def _build_from_hdf5(self):
        """Walk LIBERO demo HDF5 files, build sample index.

        Each sample is dict (task_id, episode_id, step_t). Real arrays are
        loaded lazily in __getitem__.
        """
        try:
            import h5py  # noqa: F401
        except ImportError:
            print("[pretrain] WARNING: h5py not available, using synthetic mode")
            self.synthetic = True
            self._build_synthetic()
            return

        # Implementation TODO Day 7 — hook to libero.libero.datasets path
        # For now: also fall back to synthetic so tests pass.
        if self.root is None or not self.root.exists():
            print(f"[pretrain] WARNING: root {self.root} not found, using synthetic")
            self.synthetic = True
            self._build_synthetic()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx) -> dict:
        s = self.samples[idx]
        if self.synthetic:
            agent_t = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
            wrist_t = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
            agent_t_delta = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
        else:
            # TODO Day 7: load from HDF5
            agent_t = wrist_t = agent_t_delta = np.zeros((224, 224, 3), dtype=np.uint8)
        return {
            "agent_t": _to_tensor_minus1_1(agent_t),
            "wrist_t": _to_tensor_minus1_1(wrist_t),
            "agent_t_delta": _to_tensor_minus1_1(agent_t_delta),
            "task_id": s["task_id"],
            "episode_id": s["episode_id"],
        }
