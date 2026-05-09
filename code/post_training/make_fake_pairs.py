"""Generate a fake DPO pair dataset for pipeline smoke-testing.

Real DPO pairs come from env-coupled rollout (rollout.py — TODO Week 1
end). Before that lands, we want to verify the train_dpo pipeline runs
end-to-end on the actual model. This module produces a small (B=20)
synthetic dataset shaped exactly like the real one:

    {
        "instructions": List[str]   length B
        "images":           (B, 3, 224, 224)
        "chosen_chunks":    (B, T, 7)   small smooth motion towards a goal
        "rejected_chunks":  (B, T, 7)   random motion (low reward)
        "chosen_rewards":   (B,)
        "rejected_rewards": (B,)
    }

The chunks are NOT physically meaningful — they're synthetic. The point
is only to:
  - exercise the train_dpo loop with real OpenVLA logp computation
  - verify loss decreases
  - verify checkpoint save/load works

Result we expect: loss dropping over a few hundred steps, accuracy → 1
(model learns to prefer the synthetic "chosen" pattern over "rejected").
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch


# 5 instructions matching synthetic_dataset.py for consistency
INSTRUCTIONS = [
    "pick up the red cube and place it on the blue plate",
    "put the coffee cup into the cabinet",
    "fold the white towel in half",
    "open the drawer and put the apple inside",
    "pour the contents of the bottle into the glass",
]

# Same color scheme so the fake camera images match
COLORS = [
    (0.85, 0.20, 0.20),
    (0.30, 0.40, 0.80),
    (0.95, 0.95, 0.85),
    (0.30, 0.70, 0.30),
    (0.65, 0.40, 0.85),
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n_per_instruction", type=int, default=4)
    p.add_argument("--chunk_steps", type=int, default=1,
                   help="OpenVLA single-step actions; set 1 for OpenVLA, "
                        "60 for Spirit/π0.5")
    p.add_argument("--action_dim", type=int, default=7)
    p.add_argument("--image_size", type=int, default=224)
    p.add_argument("--out", required=True, help="path to save .pt file")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    B = len(INSTRUCTIONS) * args.n_per_instruction

    instructions = []
    images = []
    chosen_chunks = []
    rejected_chunks = []
    chosen_rewards = []
    rejected_rewards = []

    H, W = args.image_size, args.image_size
    T, A = args.chunk_steps, args.action_dim

    for task_idx, (instruction, color) in enumerate(zip(INSTRUCTIONS, COLORS)):
        for ep_in_task in range(args.n_per_instruction):
            instructions.append(instruction)

            # Image: solid color background + diagonal gradient
            img = np.full((H, W, 3), color, dtype=np.float32)
            ramp = np.linspace(0.85, 1.15, W).reshape(1, W, 1)
            img = np.clip(img * ramp, 0.0, 1.0)
            # Add a darker block in the lower-right quarter to mimic an object
            h2, w2 = H // 2, W // 2
            img[h2:, w2:] = np.array(color) * 0.5
            images.append(torch.from_numpy(img).permute(2, 0, 1).contiguous())

            # Chosen chunk: small linear motion centred at "task_idx-relative"
            # offset (so different instructions induce different "good"
            # actions, exercising the language conditioning)
            target_per_dim = (task_idx / max(1, len(INSTRUCTIONS) - 1) - 0.5) * 0.4
            chosen = np.full((T, A), target_per_dim, dtype=np.float32)
            chosen += rng.normal(0, 0.02, size=(T, A)).astype(np.float32)
            chosen_chunks.append(torch.from_numpy(chosen))

            # Rejected chunk: random in [-1, 1]
            rejected = rng.uniform(-0.8, 0.8, size=(T, A)).astype(np.float32)
            rejected_chunks.append(torch.from_numpy(rejected))

            # Synthetic rewards (just for logging consistency; trainer
            # doesn't use them — DPO is preference-only)
            chosen_rewards.append(0.85)
            rejected_rewards.append(0.15)

    bundle = {
        "instructions": instructions,
        "images": torch.stack(images),                   # (B, 3, H, W) float32 in [0, 1]
        "chosen_chunks": torch.stack(chosen_chunks),     # (B, T, A)
        "rejected_chunks": torch.stack(rejected_chunks), # (B, T, A)
        "chosen_rewards": torch.tensor(chosen_rewards),
        "rejected_rewards": torch.tensor(rejected_rewards),
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bundle, out_path)
    print(f"saved {B} fake pairs → {out_path}")
    print(f"  shapes: images={tuple(bundle['images'].shape)}, "
          f"chosen={tuple(bundle['chosen_chunks'].shape)}, "
          f"rejected={tuple(bundle['rejected_chunks'].shape)}")


if __name__ == "__main__":
    main()
