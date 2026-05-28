"""Standalone debug entry: load OpenVLA + call policy_logp once.

This is the SHORTEST path to hit a breakpoint inside the adapter. No
trainer, no dataloader, no optimizer. Just:

    model = load(...)                # ~15s
    batch = {instruction, image}     # 1 item
    chunk = tensor(T=5, 7)           # fake continuous actions
    logp = adapter.policy_logp(batch, chunk)   # <-- set breakpoint here

Expected output at end: a single scalar logp per sample. If you set a
breakpoint inside `adapters/openvla.py:policy_logp`, you can step
through:
  1. action continuous → 256-bin discretisation
  2. bin_id → vocab_id (last 256 tokens of Llama vocab)
  3. prompt template building (with image placeholder)
  4. tokenization of full sequence
  5. model forward (teacher forcing)
  6. log-softmax over vocab
  7. gather log-probs at the action-token positions
  8. (mean, not sum) aggregation across (T, 7) token positions

Use case in VSCode:
    Launch config ④ → program starts → hits `breakpoint()` below.
    F10 step over line by line, OR
    Place breakpoint in `adapters/openvla.py:policy_logp` and F5 continue.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch


def main() -> None:
    # Allow --base / --base_ckpt overrides; defaults keep old behavior (OpenVLA).
    p = argparse.ArgumentParser()
    p.add_argument("--base", choices=["openvla", "pi05", "spirit"], default="openvla",
                   help="Which adapter to instantiate. Default: openvla")
    p.add_argument("--base_ckpt", default="",
                   help="Path to base ckpt directory. If empty, falls back to "
                        "$OPENVLA_CKPT (openvla) or default ~/pi_assets path (pi05).")
    args = p.parse_args()

    # Resolve ckpt path per --base
    if args.base_ckpt:
        ckpt = args.base_ckpt
    elif args.base == "openvla":
        ckpt = os.environ.get(
            "OPENVLA_CKPT",
            str(Path.home() / "openvla_assets/finetuned_libero/openvla-7b-finetuned-libero-spatial"),
        )
    elif args.base == "pi05":
        ckpt = str(Path.home() / "pi_assets/pi05_libero_pytorch")
    else:
        raise SystemExit(f"--base {args.base} requires --base_ckpt")

    if not Path(ckpt).exists():
        raise SystemExit(f"{args.base} ckpt not found at {ckpt}. "
                         f"Set --base_ckpt or place the model there.")

    print(f"[base] {args.base}  ckpt={ckpt}")

    # Build config
    from post_training.interface import PostTrainConfig
    cfg = PostTrainConfig(
        name="debug_logp",
        base=args.base,
        algorithm="dpo",
        base_ckpt_path=ckpt,
        libero_suite="spatial",
        batch_size=1,
        use_lora=True,
        lora_r=8,
        lora_alpha=16,
        output_dir="/tmp/debug_logp_out",
    )

    # Build adapter
    if args.base == "openvla":
        from post_training.adapters.openvla import OpenVLAAdapter as Adapter
    elif args.base == "pi05":
        from post_training.adapters.pi05 import Pi05Adapter as Adapter
    elif args.base == "spirit":
        from post_training.adapters.spirit import SpiritAdapter as Adapter
    else:
        raise SystemExit(f"Unknown --base {args.base}")

    print(f"[1/4] Loading {Adapter.__name__} (~15s first time)...")
    adapter = Adapter(cfg)
    adapter.freeze_reference()

    # Synthetic batch: 1 instruction + 1 image
    instruction = "pick up the black bowl from table center and place it on the plate"
    image = torch.randn(3, 224, 224)  # will be normalized inside adapter
    # Scale to [0, 1] to look like a real float image
    image = (image - image.min()) / (image.max() - image.min())

    batch = {
        "instruction": [instruction],
        "image": image.unsqueeze(0),  # (B=1, 3, 224, 224)
    }
    # Tiny chunk: 5 steps × 7 action dims
    chunk = torch.zeros(1, 5, 7)  # (B, T, A)
    chunk[0, :, 0] = torch.linspace(-0.1, 0.1, 5)   # move along x
    chunk[0, :, 6] = torch.linspace(0, 1, 5)        # close gripper

    print(f"[2/4] batch ready: B=1, instruction len={len(instruction)}, "
          f"image={tuple(image.shape)}, chunk={tuple(chunk.shape)}")

    # >>> PUT A BREAKPOINT ON THE NEXT LINE <<<
    # Step into this call to walk through policy_logp internals.
    print("[3/4] Calling adapter.policy_logp(batch, chunk)...")
    logp = adapter.policy_logp(batch, chunk)
    print(f"[4/4] policy_logp result: {logp}  (shape={tuple(logp.shape)})")

    # Second call for reference policy
    print("[5/5] Calling policy_logp_with_ref(batch, chunk)...")
    logp_cur, logp_ref = adapter.policy_logp_with_ref(batch, chunk)
    print(f"       cur = {logp_cur}")
    print(f"       ref = {logp_ref}")
    print(f"       diff = {(logp_cur - logp_ref).item():+.6f}")
    print("\nDone. (At step 0 of training, cur ≈ ref because LoRA init is 0.)")


if __name__ == "__main__":
    main()
