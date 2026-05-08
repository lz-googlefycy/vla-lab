"""
XLeRobotSpiritAdapter — bridge XLeRobot (SO-100 dual-arm, 12 DoF) observation
and action space to Spirit v1.5's ALOHA-style 14 DoF expected format.

Joint order mapping:

  ALOHA (Spirit expects, 7 per arm):
    [waist, shoulder, elbow, forearm_roll, wrist_angle, wrist_rotate, gripper]

  SO-100 (XLeRobot uses, 6 per arm):
    [shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper]

  Mapping (SO-100 idx → ALOHA slot):
    waist         <- shoulder_pan   (axial rotation at base)
    shoulder      <- shoulder_lift
    elbow         <- elbow_flex
    forearm_roll  <- (no counterpart, zero-pad)
    wrist_angle   <- wrist_flex
    wrist_rotate  <- wrist_roll
    gripper       <- gripper

The missing forearm_roll DoF is the largest semantic gap; for fine-tuning we
zero-fill that slot in state and ignore it in action output.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np

try:
    import torch
    import torch.nn.functional as F
    _TORCH = True
except ImportError:
    _TORCH = False

SPIRIT_ACTION_DIM = 14   # ALOHA 7 per arm × 2
XLE_ACTION_DIM = 12      # SO-100 6 per arm × 2

# XLE index at which each ALOHA slot sources from (None = zero-pad)
# Length 7: [waist, shoulder, elbow, forearm_roll, wrist_angle, wrist_rotate, gripper]
XLE_TO_SPIRIT_ARM = [0, 1, 2, None, 3, 4, 5]

# Inverse: ALOHA indices to keep when projecting back to SO-100
SPIRIT_TO_XLE_ARM_KEEP = [0, 1, 2, 4, 5, 6]


def pad_arm_state(xle_arm_6: np.ndarray) -> np.ndarray:
    """12 SO-100 joints (6 per arm) → Spirit 14-dim padded.

    Args:
        xle_arm_6: shape (6,) single-arm SO-100 joint state.
    Returns:
        shape (7,) ALOHA-ordered state with forearm_roll zero-padded.
    """
    assert xle_arm_6.shape == (6,), f"expected (6,), got {xle_arm_6.shape}"
    out = np.zeros(7, dtype=xle_arm_6.dtype)
    for slot, src in enumerate(XLE_TO_SPIRIT_ARM):
        if src is not None:
            out[slot] = xle_arm_6[src]
    return out


def unpad_arm_action(spirit_arm_7: np.ndarray) -> np.ndarray:
    """7-DoF ALOHA action → 6-DoF SO-100 (drop forearm_roll).

    Args:
        spirit_arm_7: shape (7,).
    Returns:
        shape (6,).
    """
    assert spirit_arm_7.shape == (7,), f"expected (7,), got {spirit_arm_7.shape}"
    return np.stack([spirit_arm_7[i] for i in SPIRIT_TO_XLE_ARM_KEEP], axis=0)


def pad_state_12_to_14(state_12: np.ndarray) -> np.ndarray:
    """Dual-arm: (12,) XLeRobot state → (14,) Spirit state."""
    assert state_12.shape == (12,), f"expected (12,), got {state_12.shape}"
    left = pad_arm_state(state_12[:6])
    right = pad_arm_state(state_12[6:])
    return np.concatenate([left, right])


def unpad_action_14_to_12(action_14: np.ndarray) -> np.ndarray:
    """Dual-arm: (14,) Spirit action → (12,) XLeRobot action."""
    assert action_14.shape == (14,), f"expected (14,), got {action_14.shape}"
    left = unpad_arm_action(action_14[:7])
    right = unpad_arm_action(action_14[7:])
    return np.concatenate([left, right])


class XLeRobotSpiritAdapter:
    """Wrap XLeRobot observations to Spirit v1.5 input format and vice versa.

    Example:
        adapter = XLeRobotSpiritAdapter(tile_cam_high=True)
        obs_spirit = adapter.wrap_observation({
            "state": np.zeros(12),
            "image_head": np.zeros((480, 640, 3), dtype=np.uint8),
            "task": "pick up the red cup",
        })
        action_14 = policy.select_action(obs_spirit)  # [1, 60, 14]
        action_12 = adapter.unwrap_action(action_14)
    """

    def __init__(
        self,
        tile_cam_high: bool = True,
        image_left_wrist: Optional[np.ndarray] = None,
        image_right_wrist: Optional[np.ndarray] = None,
        target_cam_size: tuple[int, int] = (240, 320),
        device: str = "cuda",
        robot_type: str = "aloha",  # Spirit knows: ARX5, aloha, Franka, UR5 — we impersonate aloha (dual-arm)
        dtype: str = "bf16",         # must match the policy's dtype
    ):
        self.tile_cam_high = tile_cam_high
        self.image_left_wrist = image_left_wrist
        self.image_right_wrist = image_right_wrist
        self.H, self.W = target_cam_size
        self.device = device
        self.robot_type = robot_type
        self.dtype_str = dtype
        if _TORCH:
            if dtype in ("bf16", "bfloat16"):
                self.torch_dtype = torch.bfloat16
            elif dtype in ("fp16", "half", "float16"):
                self.torch_dtype = torch.float16
            else:
                self.torch_dtype = torch.float32

    # ------------------------------------------------------------------
    # Image helpers
    # ------------------------------------------------------------------
    def _to_tensor_image(self, img_uint8_hwc: np.ndarray) -> "torch.Tensor":
        """(H, W, 3) uint8 → (3, 240, 320) float [0,1] torch tensor."""
        if not _TORCH:
            raise ImportError("torch is required for image handling")
        t = torch.from_numpy(img_uint8_hwc).permute(2, 0, 1).float() / 255.0
        t = F.interpolate(t.unsqueeze(0), size=(self.H, self.W), mode="bilinear",
                          align_corners=False).squeeze(0)
        return t

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def wrap_observation(self, obs_xle: Dict) -> Dict:
        """XLeRobot observation → Spirit-compatible batch dict.

        Args:
            obs_xle: {
                "state": np.ndarray (12,) float,
                "image_head": np.ndarray (H, W, 3) uint8,
                "image_left_wrist": optional np.ndarray (H, W, 3) uint8,
                "image_right_wrist": optional np.ndarray (H, W, 3) uint8,
                "task": str,
            }
        Returns:
            Spirit-compatible batch dict with leading batch dim = 1.
        """
        if not _TORCH:
            raise ImportError("torch is required at runtime; install it or skip inference")

        # State must match policy dtype (bf16). But images stay float32 because
        # Spirit's preprocess_rb_batch internally does (img*255).astype(uint8),
        # which doesn't support bfloat16 tensors.
        spirit_state = pad_state_12_to_14(obs_xle["state"])
        state_t = torch.from_numpy(spirit_state).unsqueeze(0).to(self.device).to(self.torch_dtype)

        head = self._to_tensor_image(obs_xle["image_head"]).to(self.device)  # float32 for preprocess

        # Decide wrist cameras
        if obs_xle.get("image_left_wrist") is not None:
            left = self._to_tensor_image(obs_xle["image_left_wrist"]).to(self.device)
        elif self.tile_cam_high:
            left = head.clone()
        else:
            raise ValueError("No left wrist image and tile_cam_high=False")

        if obs_xle.get("image_right_wrist") is not None:
            right = self._to_tensor_image(obs_xle["image_right_wrist"]).to(self.device)
        elif self.tile_cam_high:
            right = head.clone()
        else:
            raise ValueError("No right wrist image and tile_cam_high=False")

        return {
            "observation.state": state_t,
            "observation.images.cam_high": head.unsqueeze(0),
            "observation.images.cam_left_wrist": left.unsqueeze(0),
            "observation.images.cam_right_wrist": right.unsqueeze(0),
            "task": [obs_xle["task"]],
            "robot_type": [self.robot_type],
        }

    def unwrap_action(self, spirit_action: "torch.Tensor", step_idx: int = 0) -> np.ndarray:
        """Spirit's (B, chunk, 14) → (12,) single-step for XLeRobot.

        Args:
            spirit_action: output of policy.select_action, shape (B, T, 14).
            step_idx: which step in the chunk to extract. 0 = immediate next.
        Returns:
            np.ndarray (12,).
        """
        if hasattr(spirit_action, "cpu"):
            # Cast to fp32 before numpy: numpy doesn't support bf16/fp16 directly
            a = spirit_action[0, step_idx].detach().float().cpu().numpy()
        else:
            a = np.asarray(spirit_action)[0, step_idx]
        return unpad_action_14_to_12(a)

    def unwrap_action_chunk(self, spirit_action: "torch.Tensor") -> np.ndarray:
        """Full chunk: (B, T, 14) → (T, 12)."""
        if hasattr(spirit_action, "cpu"):
            a = spirit_action[0].detach().float().cpu().numpy()
        else:
            a = np.asarray(spirit_action)[0]
        T = a.shape[0]
        out = np.zeros((T, XLE_ACTION_DIM), dtype=a.dtype)
        for t in range(T):
            out[t] = unpad_action_14_to_12(a[t])
        return out


class SpiritLeRobotPolicy:
    """Thin wrapper to make SpiritVLAPolicy look like a LeRobot Policy.

    Usage:
        policy = SpiritLeRobotPolicy("/workspace/models/Spirit-v1.5")
        action = policy.select_action(obs_dict)  # returns (12,) np.ndarray

    Note: at construction time this loads Qwen3-VL + DiT head fully into VRAM.
    """

    def __init__(
        self,
        spirit_ckpt_path: str,
        tile_cam_high: bool = True,
        device: str = "cuda",
        dtype: str = "bf16",           # fp32 needs ~42 GB on 3090; bf16 halves it
        load_to_cpu_first: bool = True, # avoid OOM by loading to CPU then casting & moving
    ):
        import sys
        import json
        # Make sure Spirit's source is on the path
        spirit_src = "/workspace/spirit-v1.5"
        if spirit_src not in sys.path:
            sys.path.insert(0, spirit_src)

        # --- Monkey-patch Spirit's sample_noise/sample_time to honor bf16 ---
        # These are hardcoded to float32 in utils/sampling.py, causing dtype
        # mismatches downstream when the model is cast to bf16.
        import torch
        if dtype in ("bf16", "bfloat16"):
            torch_dtype_target = torch.bfloat16
        elif dtype in ("fp16", "half", "float16"):
            torch_dtype_target = torch.float16
        else:
            torch_dtype_target = torch.float32

        from utils import sampling as _spirit_sampling
        _orig_sample_noise = _spirit_sampling.sample_noise
        _orig_sample_time = _spirit_sampling.sample_time
        def _patched_sample_noise(shape, dev):
            return torch.normal(mean=0.0, std=1.0, size=shape, dtype=torch_dtype_target, device=dev)
        def _patched_sample_time(bsize, dev):
            t = _orig_sample_time(bsize, dev)
            return t.to(dtype=torch_dtype_target)
        _spirit_sampling.sample_noise = _patched_sample_noise
        _spirit_sampling.sample_time = _patched_sample_time
        # Also patch the already-bound references inside modeling_spirit_vla
        import model.modeling_spirit_vla as _msv
        _msv.sample_noise = _patched_sample_noise
        _msv.sample_time = _patched_sample_time

        from model import SpiritVLAPolicy  # noqa

        # Spirit's from_pretrained reads config.device and decides to load state_dict
        # directly to that device. To avoid OOM on 24 GB GPUs, we transiently rewrite
        # config.device to 'cpu', load, cast dtype, then move to target device.
        config_path = f"{spirit_ckpt_path}/config.json"
        if load_to_cpu_first:
            with open(config_path, "r") as f:
                original_cfg = f.read()
            cfg = json.loads(original_cfg)
            cfg["device"] = "cpu"
            with open(config_path, "w") as f:
                json.dump(cfg, f, indent=2)
            try:
                self.inner = SpiritVLAPolicy.from_pretrained(spirit_ckpt_path, train=False)
            finally:
                # restore
                with open(config_path, "w") as f:
                    f.write(original_cfg)
        else:
            self.inner = SpiritVLAPolicy.from_pretrained(spirit_ckpt_path, train=False)

        # Cast dtype
        if dtype in ("bf16", "bfloat16"):
            self.inner = self.inner.to(torch.bfloat16)
        elif dtype in ("fp16", "half", "float16"):
            self.inner = self.inner.to(torch.float16)

        # Move to target device
        self.inner = self.inner.to(device).eval()
        self.adapter = XLeRobotSpiritAdapter(
            tile_cam_high=tile_cam_high, device=device, dtype=dtype
        )
        self._cached_chunk = None
        self._cached_step = 0
        self._chunk_horizon = 12  # how many action-chunk steps to reuse before re-inferring
        self._dtype = dtype
        self._device = device

    def select_action(self, obs_xle: Dict) -> np.ndarray:
        """Return next 12-DoF action for XLeRobot."""
        import torch
        if self._cached_chunk is None or self._cached_step >= self._chunk_horizon:
            batch = self.adapter.wrap_observation(obs_xle)
            # Pre-generate noise in the model's dtype to avoid Spirit's internal
            # float32-only sample_noise() which crashes with bf16 weights.
            # Shape: (B, n_action_steps, max_action_dim)
            cfg = self.inner.config
            noise_shape = (
                batch["observation.state"].shape[0],
                cfg.n_action_steps,
                cfg.max_action_dim,
            )
            if self._dtype in ("bf16", "bfloat16"):
                noise_dtype = torch.bfloat16
            elif self._dtype in ("fp16", "half", "float16"):
                noise_dtype = torch.float16
            else:
                noise_dtype = torch.float32
            noise = torch.randn(noise_shape, dtype=noise_dtype, device=self._device)
            # autocast wrapper forces mixed precision across Spirit's internal
            # DiT layers (where some tensors drift back to fp32 and mismatch weights)
            autocast_dtype = noise_dtype if noise_dtype != torch.float32 else torch.bfloat16
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=autocast_dtype):
                self._cached_chunk = self.inner.select_action(batch, noise=noise)
            self._cached_step = 0
        a = self.adapter.unwrap_action(self._cached_chunk, step_idx=self._cached_step)
        self._cached_step += 1
        return a

    def reset(self):
        """Force re-inference on next select_action."""
        self._cached_chunk = None
        self._cached_step = 0


# ----------------------------------------------------------------------
# CLI sanity checks (no torch, no Spirit needed)
# ----------------------------------------------------------------------
def _selftest():
    """Basic shape / mapping checks runnable without torch / Spirit."""
    # round-trip: pad then unpad a dual-arm state/action should lose the
    # forearm_roll slot (which we zero-pad) but keep everything else.
    x = np.arange(12).astype(np.float32)
    padded = pad_state_12_to_14(x)
    assert padded.shape == (14,)
    # ALOHA slot 3 and 10 should be zero (the forearm_roll padding)
    assert padded[3] == 0.0
    assert padded[10] == 0.0

    # Unpad full dual-arm
    a14 = np.zeros(14, dtype=np.float32)
    for i, slot_src in enumerate(XLE_TO_SPIRIT_ARM):
        if slot_src is not None:
            a14[i] = 10 + slot_src     # left arm
            a14[7 + i] = 100 + slot_src  # right arm
    a12 = unpad_action_14_to_12(a14)
    assert a12.shape == (12,)
    # Recover each SO-100 joint
    for i in range(6):
        assert a12[i] == 10 + i, f"left[{i}] wrong: {a12[i]}"
        assert a12[6 + i] == 100 + i, f"right[{i}] wrong: {a12[6+i]}"

    print("OK: adapter shape / mapping self-test passed")


if __name__ == "__main__":
    _selftest()
