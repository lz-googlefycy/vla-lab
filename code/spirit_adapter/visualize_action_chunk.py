"""
Visualize Spirit v1.5's predicted action chunk given dummy / fixed observations.

Useful for blog #2 ("here's what Spirit thinks it should do given zero
context") even before we have a sim or real robot wired up.

Outputs:
- action_chunk.png   60-step action trajectory plots (one subplot per DoF)
- action_chunk.json  raw action values per step
- side-by-side grid of input images + action timeline

Usage:
  python visualize_action_chunk.py \
      --spirit_ckpt /workspace/models/Spirit-v1.5-patched \
      --instruction "pick up the red cube and place it on the blue plate" \
      --out_dir /workspace/output/viz
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from xlerobot_adapter import SpiritLeRobotPolicy  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spirit_ckpt", required=True)
    ap.add_argument("--instruction", type=str,
                    default="pick up the red cube and place it on the blue plate")
    ap.add_argument("--out_dir", type=str, default="./viz")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num_samples", type=int, default=3,
                    help="Infer this many times (noise draws) to see sample variance.")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    print(f"Loading Spirit policy from {args.spirit_ckpt} ...")
    t0 = time.time()
    policy = SpiritLeRobotPolicy(
        spirit_ckpt_path=args.spirit_ckpt,
        tile_cam_high=True,
        device="cuda",
    )
    print(f"  loaded in {time.time()-t0:.1f}s")
    # Don't reuse the chunk cache so each call re-infers
    policy._chunk_horizon = 1

    # Build a fixed dummy observation (so different inferences come from noise)
    dummy_obs = {
        "state": np.zeros(12, dtype=np.float32),
        "image_head": rng.integers(0, 255, size=(480, 640, 3), dtype=np.uint8),
        "task": args.instruction,
    }

    print(f"\nInstruction: {args.instruction}")
    print(f"Sampling {args.num_samples} action chunks ...\n")

    all_chunks = []
    for i in range(args.num_samples):
        policy.reset()
        # Invoke once to trigger re-inference; grab the cached chunk
        _ = policy.select_action(dummy_obs)
        chunk = policy._cached_chunk  # (1, 60, 14)
        chunk_np = chunk[0].detach().float().cpu().numpy()  # (60, 14)
        all_chunks.append(chunk_np)
        print(f"  sample {i+1}: shape={chunk_np.shape}, range=[{chunk_np.min():.3f}, {chunk_np.max():.3f}]")

    all_chunks_np = np.stack(all_chunks, axis=0)  # (num_samples, 60, 14)

    # --- Save JSON ---
    json_path = os.path.join(args.out_dir, "action_chunks.json")
    with open(json_path, "w") as f:
        json.dump({
            "instruction": args.instruction,
            "num_samples": args.num_samples,
            "chunk_shape": list(all_chunks_np.shape),
            "chunks": all_chunks_np.tolist(),
        }, f)
    print(f"\nSaved raw actions: {json_path}")

    # --- Plot ---
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        DoF_NAMES = [
            "L.waist", "L.shoulder", "L.elbow", "L.forearm_roll",
            "L.wrist_angle", "L.wrist_rotate", "L.gripper",
            "R.waist", "R.shoulder", "R.elbow", "R.forearm_roll",
            "R.wrist_angle", "R.wrist_rotate", "R.gripper",
        ]

        fig, axes = plt.subplots(7, 2, figsize=(14, 16), sharex=True)
        for dof in range(14):
            ax = axes[dof % 7, dof // 7]
            for s in range(args.num_samples):
                ax.plot(all_chunks_np[s, :, dof], alpha=0.65, linewidth=1.4,
                        label=f"sample {s+1}" if dof == 0 else None)
            ax.axhline(0, color="gray", linewidth=0.5, alpha=0.4)
            ax.set_title(f"DoF {dof}: {DoF_NAMES[dof]}", fontsize=10)
            ax.set_ylabel("normalized action")
            if dof % 7 == 6:
                ax.set_xlabel("step in chunk (0–59)")
        axes[0, 0].legend(loc="upper right", fontsize=8)
        fig.suptitle(
            f'Spirit v1.5 zero-shot action chunk on XLeRobot \n'
            f'"{args.instruction}"  (tile_cam_high=True, no real obs)',
            fontsize=12,
        )
        fig.tight_layout()
        png_path = os.path.join(args.out_dir, "action_chunks.png")
        fig.savefig(png_path, dpi=110, bbox_inches="tight")
        print(f"Saved plot: {png_path}")
    except ImportError:
        print("[warn] matplotlib not installed, skipping plot")

    print("\n✅ Done. Inspect outputs in:", args.out_dir)


if __name__ == "__main__":
    main()
