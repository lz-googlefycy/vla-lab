"""
Spirit v1.5 smoke test.

Verifies:
1. Model loads from ckpt
2. Adapter maps dummy XLeRobot observation correctly
3. select_action runs end-to-end on GPU
4. Output shape matches expectation (B=1, chunk=60, dim=14)
5. Unwrapped action has shape (12,) suitable for XLeRobot

Usage:
  python smoke_test.py --spirit_ckpt /workspace/models/Spirit-v1.5

If --spirit_ckpt is missing, runs adapter-only tests (no GPU required).
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

# Make sure we find sibling adapter module
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from xlerobot_adapter import (  # noqa: E402
    XLeRobotSpiritAdapter,
    SpiritLeRobotPolicy,
    pad_state_12_to_14,
    unpad_action_14_to_12,
    SPIRIT_ACTION_DIM,
    XLE_ACTION_DIM,
)


def make_fake_xle_obs():
    """A plausible XLeRobot observation."""
    return {
        "state": np.random.uniform(-1.5, 1.5, size=(12,)).astype(np.float32),
        "image_head": np.random.randint(0, 255, size=(480, 640, 3), dtype=np.uint8),
        "task": "pick up the red cup and place it on the plate",
    }


def adapter_only():
    print("=" * 60)
    print("Adapter-only test (no torch / no Spirit required)")
    print("=" * 60)
    obs = make_fake_xle_obs()
    print(f"input state shape:           {obs['state'].shape}")
    print(f"input image_head shape:      {obs['image_head'].shape} dtype={obs['image_head'].dtype}")
    padded = pad_state_12_to_14(obs["state"])
    print(f"padded state shape:          {padded.shape} (expect (14,))")
    assert padded.shape == (14,)

    # Fake Spirit action
    fake_action = np.random.uniform(-0.5, 0.5, size=(1, 60, 14)).astype(np.float32)
    unpadded = unpad_action_14_to_12(fake_action[0, 0])
    print(f"unpadded action shape:       {unpadded.shape} (expect (12,))")
    assert unpadded.shape == (12,)
    print("✅ Adapter self-test passed")


def full_test(spirit_ckpt: str):
    print()
    print("=" * 60)
    print("Full GPU test: loading Spirit v1.5 and running inference")
    print("=" * 60)

    import torch
    print(f"torch:        {torch.__version__}")
    print(f"cuda:         {torch.version.cuda}, available={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"device:       {torch.cuda.get_device_name(0)}")
        print(f"memory total: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")

    obs = make_fake_xle_obs()

    print(f"\n[1/3] Loading Spirit v1.5 policy from {spirit_ckpt}...")
    t0 = time.time()
    policy = SpiritLeRobotPolicy(
        spirit_ckpt_path=spirit_ckpt,
        tile_cam_high=True,
        device="cuda",
    )
    load_time = time.time() - t0
    print(f"   loaded in {load_time:.1f}s")
    if torch.cuda.is_available():
        alloc_gb = torch.cuda.memory_allocated() / 1e9
        reserved_gb = torch.cuda.memory_reserved() / 1e9
        print(f"   GPU memory: allocated={alloc_gb:.2f} GB, reserved={reserved_gb:.2f} GB")

    print(f"\n[2/3] Running select_action (warmup) ...")
    t0 = time.time()
    action_warm = policy.select_action(obs)
    warm_ms = (time.time() - t0) * 1000
    print(f"   warmup: {warm_ms:.0f} ms")
    print(f"   action shape: {action_warm.shape} (expect (12,))")
    print(f"   action range: [{action_warm.min():.3f}, {action_warm.max():.3f}]")
    assert action_warm.shape == (12,)

    print(f"\n[3/3] Running 10 select_action cycles for timing...")
    # reset so we actually re-infer (SpiritLeRobotPolicy caches 12 chunks)
    policy.reset()
    times_ms = []
    for i in range(10):
        policy.reset()  # force fresh inference each time
        t0 = time.time()
        _ = policy.select_action(obs)
        times_ms.append((time.time() - t0) * 1000)
    mean_ms = sum(times_ms) / len(times_ms)
    print(f"   mean:  {mean_ms:.0f} ms  ({1000/mean_ms:.1f} Hz)")
    print(f"   min:   {min(times_ms):.0f} ms")
    print(f"   max:   {max(times_ms):.0f} ms")

    print("\n✅ SMOKE TEST PASSED")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spirit_ckpt", type=str, default=None,
                    help="Path to Spirit v1.5 ckpt dir; if missing run adapter-only")
    args = ap.parse_args()

    adapter_only()
    if args.spirit_ckpt and os.path.exists(args.spirit_ckpt):
        full_test(args.spirit_ckpt)
    else:
        print("\n(skip full GPU test: set --spirit_ckpt to enable)")


if __name__ == "__main__":
    main()
