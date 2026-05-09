"""
Eval script: load LoRA fine-tuned Spirit + run on 5 Phase-A instructions.

Compares before/after LoRA on the same instruction set:
  - "pick up the red cube and place it on the blue plate"
  - "put the coffee cup into the cabinet"
  - "fold the white towel in half"
  - "open the drawer and put the apple inside"
  - "pour the contents of the bottle into the glass"

Outputs:
  - <out>/before/s{N}_chunk.npy  (60×14 vanilla Spirit)
  - <out>/after/s{N}_chunk.npy   (60×14 LoRA-tuned)
  - <out>/comparison_grid.png    (5 rows × 2 cols, side-by-side)
  - <out>/eval_summary.json      (chunk diff stats)

Usage:
  python eval_lora_chunks.py \
      --pretrained_path /workspace/models/Spirit-v1.5-patched \
      --lora_ckpt /workspace/output/lora_smoke_v1/checkpoint-300 \
      --output_dir /workspace/output/lora_smoke_v1/eval
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import types
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
import os
_SPIRIT_CANDIDATES = [
    os.environ.get("SPIRIT_SRC"),
    "/workspace/spirit-v1.5",
    str(HERE.parent.parent / "spirit-v1.5"),
    str(Path.home() / "spirit-v1.5"),
]
for p in (str(HERE), *(_p for _p in _SPIRIT_CANDIDATES if _p)):
    if p not in sys.path and Path(p).exists():
        sys.path.insert(0, p)


INSTRUCTIONS = [
    "pick up the red cube and place it on the blue plate",
    "put the coffee cup into the cabinet",
    "fold the white towel in half",
    "open the drawer and put the apple inside",
    "pour the contents of the bottle into the glass",
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pretrained_path", required=True)
    p.add_argument("--lora_ckpt", required=True,
                   help="Path to checkpoint dir containing trainable_params.pt")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--num_samples", type=int, default=2,
                   help="Num noise samples per instruction (showcases stochastic head)")
    return p.parse_args()


def load_lora_state_dict(path: Path):
    """Load _LoRALinear state dict saved by train_lora.py manual mode."""
    pt = path / "trainable_params.pt"
    if not pt.exists():
        raise FileNotFoundError(
            f"Expected {pt} (saved by train_lora.py manual mode). "
            f"For peft mode, use peft.PeftModel.from_pretrained instead."
        )
    return torch.load(str(pt), map_location="cpu")


def apply_lora_to_model(model, lora_sd: dict, args):
    """Walk the saved state dict; for each lora_A/B param, find the
    corresponding nn.Linear in the model and replace with _LoRALinear.

    Mirrors train_lora.py:inject_lora's manual mode.
    """
    # Reconstruct the _LoRALinear class shape from the saved keys
    from train_lora import _LoRALinear, _replace_module_by_path  # noqa

    # Group keys by module path: "dit.transformer_blocks.0.attn1.to_q.lora_A" →
    # base path "dit.transformer_blocks.0.attn1.to_q", suffix "lora_A"
    groups: dict[str, dict[str, torch.Tensor]] = {}
    for k, v in lora_sd.items():
        if ".lora_A" in k or ".lora_B" in k:
            base = k.rsplit(".", 1)[0]
            sfx = k.rsplit(".", 1)[1]
            groups.setdefault(base, {})[sfx] = v

    print(f"[lora] {len(groups)} LoRA modules to apply")

    # Determine r and alpha from the first lora_A
    first_A = next(iter(groups.values()))["lora_A"]
    r = first_A.shape[0]

    # Walk the model, find each Linear at base path and replace
    n_replaced = 0
    for base, params in groups.items():
        # Find the Linear at this path
        parent = model
        parts = base.split(".")
        try:
            for p in parts:
                if p.isdigit() and hasattr(parent, "__getitem__"):
                    parent = parent[int(p)]
                else:
                    parent = getattr(parent, p)
        except AttributeError as e:
            print(f"[lora]  ! could not navigate to {base}: {e}")
            continue
        if not isinstance(parent, torch.nn.Linear):
            print(f"[lora]  ! {base} is {type(parent).__name__}, not Linear; skipping")
            continue
        new_m = _LoRALinear(parent, r=r, alpha=2 * r, dropout=0.0)
        # Load A/B weights — alpha is encoded as scale = alpha/r;
        # we don't know alpha exactly, but standard is 2× r so r=32, α=64.
        with torch.no_grad():
            new_m.lora_A.data.copy_(params["lora_A"].to(new_m.lora_A.dtype))
            new_m.lora_B.data.copy_(params["lora_B"].to(new_m.lora_B.dtype))
        _replace_module_by_path(model, base, new_m)
        n_replaced += 1
    print(f"[lora] applied {n_replaced} LoRA modules")

    # Also restore any state_proj / action_in_proj / action_out_proj weights
    proj_keys = [k for k in lora_sd
                 if any(p in k for p in ["state_proj", "action_in_proj",
                                          "action_out_proj"])
                 and ".lora_" not in k]
    if proj_keys:
        own_sd = model.state_dict()
        applied = 0
        for k in proj_keys:
            if k in own_sd:
                own_sd[k].data.copy_(lora_sd[k].to(own_sd[k].dtype))
                applied += 1
        print(f"[lora] also restored {applied} proj weights")


def make_dummy_batch(instruction: str, device, dtype=torch.bfloat16):
    """Construct a minimal Spirit batch for inference on a fake scene.

    Uses the same fake camera image style as synthetic_dataset (so
    LoRA-tuned model sees similar input distribution to what it was
    trained on).

    Note: images stay fp32 (Spirit's internal preprocess_rb_batch calls
    numpy on them which doesn't support bf16). Only state/proj go bf16.
    """
    from synthetic_dataset import INSTRUCTIONS, SyntheticSpiritDataset, SyntheticDataConfig

    # Reuse synthetic dataset's pre-rendered camera image + a
    # zeroed state. We just need ONE sample, not a full episode.
    ds = SyntheticSpiritDataset(SyntheticDataConfig(n_episodes_per_instruction=1))
    task_idx = INSTRUCTIONS.index(instruction)
    cam = ds._cam_cache[task_idx]   # (3, H, W) float32 in [0,1]

    state_14 = torch.zeros(1, 1, 14, dtype=torch.float32)

    batch = {
        "observation.state": state_14.to(device).to(dtype),
        # IMPORTANT: images stay fp32 — see docs/troubleshooting.md bug 6
        "observation.images.cam_high": cam.unsqueeze(0).to(device),
        "observation.images.cam_left_wrist": cam.unsqueeze(0).to(device),
        "observation.images.cam_right_wrist": cam.unsqueeze(0).to(device),
        "task": [instruction],
        "robot_type": ["Franka"],
    }
    return batch


def predict_chunk(model, batch) -> np.ndarray:
    """Run model.select_action() and return chunk as numpy (60, 14).

    Spirit returns shape (B, 60, 14); we squeeze the batch dim so
    callers see a clean (60, 14) for plotting.
    """
    with torch.no_grad():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model.select_action(batch)
    if isinstance(out, dict) and "action" in out:
        chunk = out["action"]
    else:
        chunk = out
    arr = chunk.detach().float().cpu().numpy()
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    return arr


def main():
    args = parse_args()
    out = Path(args.output_dir)
    (out / "before").mkdir(parents=True, exist_ok=True)
    (out / "after").mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Apply same bf16 patch as training
    from train_lora import patch_spirit_for_bf16
    patch_spirit_for_bf16()

    # ---- 1. Vanilla Spirit (before LoRA) ----
    print("\n=== [1/2] Vanilla Spirit ===")
    from model import SpiritVLAPolicy
    config_path = Path(args.pretrained_path) / "config.json"
    with open(config_path) as f:
        raw_config = types.SimpleNamespace(**json.load(f))

    t0 = time.time()
    model_v = SpiritVLAPolicy.from_pretrained(
        ckpt_path=args.pretrained_path, strict=False, train=False
    )
    model_v = model_v.to(device).to(torch.bfloat16)
    model_v.eval()
    print(f"[load] vanilla loaded in {time.time()-t0:.1f}s")

    chunks_before = {}
    for i, instr in enumerate(INSTRUCTIONS):
        batch = make_dummy_batch(instr, device)
        chunk = predict_chunk(model_v, batch)
        chunks_before[i] = chunk
        np.save(out / "before" / f"s{i}_chunk.npy", chunk)
        print(f"  s{i}: shape={chunk.shape}  μ={chunk.mean():+.3f}  σ={chunk.std():.3f}")

    # Free vanilla model before loading LoRA model
    del model_v
    torch.cuda.empty_cache()

    # ---- 2. LoRA-tuned Spirit ----
    print(f"\n=== [2/2] LoRA-tuned Spirit (ckpt={args.lora_ckpt}) ===")
    t0 = time.time()
    model_l = SpiritVLAPolicy.from_pretrained(
        ckpt_path=args.pretrained_path, strict=False, train=False
    )
    model_l = model_l.to(device).to(torch.bfloat16)
    print(f"[load] base re-loaded in {time.time()-t0:.1f}s")

    lora_sd = load_lora_state_dict(Path(args.lora_ckpt))
    print(f"[lora] state_dict has {len(lora_sd)} tensors")
    apply_lora_to_model(model_l, lora_sd, args)
    model_l.eval()

    chunks_after = {}
    for i, instr in enumerate(INSTRUCTIONS):
        batch = make_dummy_batch(instr, device)
        chunk = predict_chunk(model_l, batch)
        chunks_after[i] = chunk
        np.save(out / "after" / f"s{i}_chunk.npy", chunk)
        print(f"  s{i}: shape={chunk.shape}  μ={chunk.mean():+.3f}  σ={chunk.std():.3f}")

    # ---- 3. Comparison + summary ----
    print("\n=== [3/3] Comparison ===")
    summary = {"per_instruction": [], "args": vars(args)}
    for i, instr in enumerate(INSTRUCTIONS):
        b = chunks_before[i]
        a = chunks_after[i]
        l1_diff = np.mean(np.abs(a - b))
        cosine_per_dof = []
        for d in range(14):
            denom = np.linalg.norm(a[:, d]) * np.linalg.norm(b[:, d]) + 1e-9
            c = float(np.dot(a[:, d], b[:, d]) / denom)
            cosine_per_dof.append(c)
        summary["per_instruction"].append({
            "id": i,
            "instruction": instr,
            "before_mean": float(b.mean()),
            "before_std": float(b.std()),
            "after_mean": float(a.mean()),
            "after_std": float(a.std()),
            "l1_diff": float(l1_diff),
            "cosine_per_dof_mean": float(np.mean(cosine_per_dof)),
            "cosine_per_dof_min": float(np.min(cosine_per_dof)),
        })
        print(f"  s{i}: l1Δ={l1_diff:.4f}  cos_avg={np.mean(cosine_per_dof):+.3f}")

    with open(out / "eval_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[done] summary → {out}/eval_summary.json")

    # ---- 4. Comparison plot (5 rows × 2 cols) ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(5, 2, figsize=(14, 12), sharey="row")
        for i, instr in enumerate(INSTRUCTIONS):
            b = chunks_before[i]
            a = chunks_after[i]
            for j, (data, title_pre) in enumerate(((b, "vanilla"), (a, "LoRA"))):
                ax = axes[i, j]
                # Plot 14 DoFs as light traces, mean as bold
                for d in range(14):
                    ax.plot(data[:, d], color="#888", alpha=0.4, linewidth=0.7)
                ax.plot(data.mean(axis=1), color="#d62728", linewidth=1.5,
                        label="mean across 14 DoFs")
                if i == 0:
                    ax.set_title(f"{title_pre}", fontsize=11, fontweight="bold")
                if j == 0:
                    ax.set_ylabel(f"s{i}\n{instr[:25]}…", fontsize=8)
                ax.grid(alpha=0.3)
                ax.set_xlim(0, 60)
        axes[-1, 0].set_xlabel("step in chunk (0–59)")
        axes[-1, 1].set_xlabel("step in chunk (0–59)")
        plt.suptitle(
            "Spirit v1.5 action chunk: vanilla (left) vs LoRA-tuned (right)\n"
            "60 timesteps × 14 DoFs per chunk · same instruction & fake-camera input",
            fontsize=12, y=1.00,
        )
        plt.tight_layout()
        plt.savefig(out / "comparison_grid.png", dpi=130, bbox_inches="tight")
        print(f"[done] plot → {out}/comparison_grid.png")
    except Exception as e:
        print(f"[warn] plot failed: {e}")


if __name__ == "__main__":
    main()
