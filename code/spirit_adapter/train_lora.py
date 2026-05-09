"""
LoRA fine-tune for Spirit v1.5 on XLeRobot data.

Adapted from spirit-v1.5/train.py with these changes:

1. **Single-GPU only.** No FSDP, no DistributedSampler. Phase B-1 runs
   on one server card; multi-GPU comes later when we have enough real
   data to justify it.

2. **PEFT LoRA wrapping.** We freeze:
     - the entire Qwen3-VL-4B backbone (text + vision)
     - the BaseDiT body (we'll only LoRA-adapt the attention modules)
   and we LoRA-adapt:
     - DiT attention (`to_q`, `to_k`, `to_v`, `to_out.0`)
     - state/action projections (`state_proj`, `action_in_proj`,
       `action_out_proj`) — full-finetune since they're small (~50M)
       and they're the most embodiment-specific layers.

3. **Pluggable dataset.** `--dataset {synthetic, xlerobot, robochallenge}`
   switches between:
     - synthetic_dataset.SyntheticSpiritDataset (no real data, validates
       pipeline)
     - xlerobot_dataset.XLeRobotSpiritDataset (real LeRobot teleop data)
     - dataset.RoboChallengeDataset (Spirit's original Franka data, for
       sanity reproduction)

4. **Small-run defaults.** batch=2, max_steps=500, save_steps=200,
   log_interval=10. Override via CLI.

5. **Apply our 6 dtype/load workarounds** from inference (see
   docs/troubleshooting.md). For training we ALSO need to undo the
   sample_noise monkey-patch when it conflicts — flow-matching loss
   needs noise sampling during forward.

Usage
-----

  # Synthetic data smoke test (5 minutes)
  python train_lora.py \
      --pretrained_path /workspace/models/Spirit-v1.5-patched \
      --dataset synthetic \
      --max_train_steps 500 \
      --batch_size 2 \
      --output_dir /workspace/output/lora_smoke

  # Real XLeRobot data (when available)
  python train_lora.py \
      --pretrained_path /workspace/models/Spirit-v1.5-patched \
      --dataset xlerobot --data_root /workspace/datasets/xlerobot_pick_place \
      --max_train_steps 5000 \
      --batch_size 4
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import types
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

# Ensure both Spirit source and our adapter dir are importable.
# Searches a few common spots; override with SPIRIT_SRC env var.
HERE = Path(__file__).resolve().parent
_SPIRIT_CANDIDATES = [
    os.environ.get("SPIRIT_SRC"),
    "/workspace/spirit-v1.5",
    str(HERE.parent.parent / "spirit-v1.5"),     # ../../spirit-v1.5 (sibling)
    str(Path.home() / "spirit-v1.5"),            # ~/spirit-v1.5
]
for p in (str(HERE), *(_p for _p in _SPIRIT_CANDIDATES if _p)):
    if p not in sys.path and Path(p).exists():
        sys.path.insert(0, p)


# --------------------------------------------------------------------- #
# Args
# --------------------------------------------------------------------- #


def parse_args():
    p = argparse.ArgumentParser(
        description="Spirit v1.5 LoRA fine-tune (single-GPU, XLeRobot-friendly)"
    )

    # Required
    p.add_argument("--pretrained_path", required=True,
                   help="Path to Spirit-v1.5(-patched) checkpoint dir")

    # Dataset
    p.add_argument("--dataset", choices=["synthetic", "xlerobot", "robochallenge"],
                   default="synthetic")
    p.add_argument("--data_root", default="",
                   help="Required for --dataset=xlerobot or robochallenge")
    p.add_argument("--encoding", default="raw_joint",
                   choices=["single_arm", "dual_ee", "raw_joint"],
                   help="Only used for --dataset=xlerobot")

    # LoRA
    p.add_argument("--lora_r", type=int, default=32)
    p.add_argument("--lora_alpha", type=int, default=64)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--lora_target_modules", default="to_q,to_k,to_v,to_out.0",
                   help="Comma-separated module name suffixes for LoRA wrap")
    p.add_argument("--also_train_proj", action="store_true", default=True,
                   help="Full-finetune state_proj/action_in_proj/action_out_proj")

    # Training
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--max_train_steps", type=int, default=500)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--warmup_steps", type=int, default=20)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--grad_clip_norm", type=float, default=1.0)

    # Logging / IO
    p.add_argument("--output_dir", default="./outputs/lora")
    p.add_argument("--log_interval", type=int, default=10)
    p.add_argument("--save_steps", type=int, default=200)
    p.add_argument("--log_jsonl", default="train_log.jsonl",
                   help="Per-step jsonl (relative to output_dir)")
    p.add_argument("--num_workers", type=int, default=0)

    # Norm stats
    p.add_argument("--norm_num_samples", type=int, default=200,
                   help="Reduced from 20000 — small_run friendly")
    p.add_argument("--skip_norm_stats", action="store_true",
                   help="Skip norm-stat computation (synthetic data)")

    return p.parse_args()


# --------------------------------------------------------------------- #
# Norm stats — needed by Spirit's Normalize layer
# --------------------------------------------------------------------- #


def compute_norm_stats_from_loader(dataset, num_samples: int, batch_size: int = 8):
    """A simpler version of Spirit's compute_norm_stats — works on any
    Dataset that exposes ``__getitem__`` returning ``observation.state``
    and ``action`` torch.Tensors of shape (1,14) and (T,14)."""
    dl = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=0,
        collate_fn=getattr(dataset, "collate_fn", None),
    )
    state_chunks, action_chunks = [], []
    seen = 0
    for batch in dl:
        state_chunks.append(batch["observation.state"].reshape(-1, 14))
        action_chunks.append(batch["action"].reshape(-1, 14))
        seen += batch["observation.state"].shape[0]
        if seen >= num_samples:
            break
    state = torch.cat(state_chunks, dim=0)[:num_samples]
    action = torch.cat(action_chunks, dim=0)[:num_samples]
    eps = 1e-6

    return {
        "state_min": state.min(dim=0).values - eps,
        "state_max": state.max(dim=0).values + eps,
        "action_min": action.min(dim=0).values - eps,
        "action_max": action.max(dim=0).values + eps,
    }


def set_norm_stats(model, norm_stats, device):
    """Same as Spirit's set_norm_stats — copy-paste for self-containment."""
    model.normalize_inputs.buffer_observation_state["min"].data.copy_(
        norm_stats["state_min"].to(device)
    )
    model.normalize_inputs.buffer_observation_state["max"].data.copy_(
        norm_stats["state_max"].to(device)
    )
    model.normalize_targets.buffer_action["min"].data.copy_(
        norm_stats["action_min"].to(device)
    )
    model.normalize_targets.buffer_action["max"].data.copy_(
        norm_stats["action_max"].to(device)
    )
    model.unnormalize_outputs.buffer_action["min"].data.copy_(
        norm_stats["action_min"].to(device)
    )
    model.unnormalize_outputs.buffer_action["max"].data.copy_(
        norm_stats["action_max"].to(device)
    )


# --------------------------------------------------------------------- #
# LoRA injection
# --------------------------------------------------------------------- #


def freeze_module(m: torch.nn.Module):
    for p in m.parameters():
        p.requires_grad = False


class _LoRALinear(torch.nn.Module):
    """Minimal LoRA-wrapped Linear, replaces an existing nn.Linear in-place.

    Behavior: y = orig(x) + (alpha/r) * x @ A.T @ B.T
    Where A: (r, in), B: (out, r). Both A,B are trainable. `orig` is frozen.

    This is functionally equivalent to peft's Linear LoRA at rank r,
    no peft import required. ~30 lines of code.
    """

    def __init__(self, orig: torch.nn.Linear, r: int = 32, alpha: int = 64,
                 dropout: float = 0.0):
        super().__init__()
        self.orig = orig
        for p in self.orig.parameters():
            p.requires_grad = False
        self.r = r
        self.scale = alpha / r
        in_f = orig.in_features
        out_f = orig.out_features
        # Match parent's dtype/device so optimizer/autocast see the right thing
        device = orig.weight.device
        dtype = orig.weight.dtype
        self.lora_A = torch.nn.Parameter(
            torch.zeros(r, in_f, device=device, dtype=dtype)
        )
        self.lora_B = torch.nn.Parameter(
            torch.zeros(out_f, r, device=device, dtype=dtype)
        )
        # Standard LoRA init: A ~ Kaiming, B = 0  → starting effect = 0
        torch.nn.init.kaiming_uniform_(self.lora_A, a=5 ** 0.5)
        # B stays 0
        self.dropout = torch.nn.Dropout(dropout) if dropout > 0 else torch.nn.Identity()

    def forward(self, x):
        out = self.orig(x)
        # LoRA path
        lora_path = self.dropout(x) @ self.lora_A.T @ self.lora_B.T
        return out + self.scale * lora_path


def _replace_module_by_path(root: torch.nn.Module, path: str, new_module: torch.nn.Module):
    """Replace root.<path> with new_module.

    `path` is dot-separated, e.g. 'layers.0.attn1.to_q'.
    Numeric components (e.g. '0' in to_out.0) are tried as both attr
    and integer index (for nn.Sequential / nn.ModuleList).
    """
    parts = path.split(".")
    parent = root
    for p in parts[:-1]:
        if p.isdigit() and hasattr(parent, "__getitem__"):
            parent = parent[int(p)]
        else:
            parent = getattr(parent, p)
    last = parts[-1]
    if last.isdigit() and hasattr(parent, "__setitem__"):
        parent[int(last)] = new_module
    else:
        setattr(parent, last, new_module)


def inject_lora(model, args):
    """Wrap DiT attention sub-Linears with LoRA.

    Two paths:
    - **manual** (default, no peft dependency): walk model.named_modules,
      replace Linear with _LoRALinear when name matches a target suffix.
    - **peft** (if available): use peft.get_peft_model.

    Returns (model, list_of_trainable_param_names).
    """
    target_modules = [m.strip() for m in args.lora_target_modules.split(",") if m.strip()]

    # ---- 1. Freeze everything ----
    for p in model.parameters():
        p.requires_grad = False

    # ---- 2. Try peft first ----
    injection = None
    try:
        from peft import LoraConfig, get_peft_model
        peft_targets = sorted({t.split(".")[0] for t in target_modules})
        lora_cfg = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            target_modules=peft_targets,
            task_type=None,
        )
        model = get_peft_model(model, lora_cfg)
        injection = "peft"
    except ImportError:
        injection = "manual"
    except Exception as e:
        print(f"[lora] peft path failed ({e}); falling back to manual")
        injection = "manual"

    # ---- 2b. Manual injection ----
    if injection == "manual":
        # Build name → module map first to avoid mutating during iteration
        replacements: list[tuple[str, torch.nn.Linear]] = []
        for name, m in model.named_modules():
            if not isinstance(m, torch.nn.Linear):
                continue
            # Match by suffix of the dotted name. Targets like "to_out.0"
            # require the full suffix to match.
            for t in target_modules:
                if name.endswith("." + t) or name == t:
                    replacements.append((name, m))
                    break
        n_replaced = 0
        for name, orig in replacements:
            new_m = _LoRALinear(orig, r=args.lora_r,
                                alpha=args.lora_alpha,
                                dropout=args.lora_dropout)
            _replace_module_by_path(model, name, new_m)
            n_replaced += 1
        print(f"[lora] manual: replaced {n_replaced} Linears with _LoRALinear")

    # ---- 3. Optionally also full-train state/action projections ----
    if args.also_train_proj:
        proj_count = 0
        for name, m in model.named_modules():
            if any(name.endswith(sfx) for sfx in
                   ["state_proj", "action_in_proj", "action_out_proj"]):
                for p in m.parameters():
                    p.requires_grad = True
                    proj_count += p.numel()
        print(f"[lora] + full-train state/action proj: {proj_count/1e6:.2f}M params")

    # ---- 4. Report ----
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"[lora] injection={injection}, trainable={n_trainable/1e6:.2f}M / "
          f"total={n_total/1e6:.0f}M ({100*n_trainable/n_total:.2f}%)")
    return model, trainable


# --------------------------------------------------------------------- #
# 6-bug workaround layer (from inference adapter, must be applied at
# training time too — DiT internals don't know about training mode)
# --------------------------------------------------------------------- #


def patch_spirit_for_bf16():
    """Apply the bf16 monkey patches from xlerobot_adapter.py.

    Spirit's `utils.sampling.sample_noise` and `sample_time` hardcode
    fp32 — at training (where we use bf16 autocast) this triggers
    dtype mismatch in DiT. We wrap them to follow whatever the model
    expects via a small dtype-aware shim.
    """
    try:
        from utils import sampling
    except ImportError:
        # Spirit code lives somewhere else
        import importlib
        sampling = importlib.import_module("utils.sampling")

    if getattr(sampling, "_spirit_noise_patched", False):
        return  # idempotent

    orig_sample_noise = sampling.sample_noise
    orig_sample_time = sampling.sample_time

    def sample_noise_dtype_aware(*a, **kw):
        out = orig_sample_noise(*a, **kw)
        # If we're inside autocast(bf16), cast to bf16 to avoid mismatch
        if torch.is_autocast_enabled():
            out = out.to(torch.get_autocast_gpu_dtype())
        return out

    def sample_time_dtype_aware(*a, **kw):
        out = orig_sample_time(*a, **kw)
        if torch.is_autocast_enabled():
            out = out.to(torch.get_autocast_gpu_dtype())
        return out

    sampling.sample_noise = sample_noise_dtype_aware
    sampling.sample_time = sample_time_dtype_aware
    sampling._spirit_noise_patched = True
    print("[patch] sample_noise / sample_time dtype-aware wrappers installed")


# --------------------------------------------------------------------- #
# Build dataset
# --------------------------------------------------------------------- #


def build_dataset(args):
    if args.dataset == "synthetic":
        from synthetic_dataset import SyntheticSpiritDataset, SyntheticDataConfig
        cfg = SyntheticDataConfig()
        return SyntheticSpiritDataset(cfg)
    if args.dataset == "xlerobot":
        from xlerobot_dataset import XLeRobotSpiritDataset, XLeRobotDataConfig
        if not args.data_root:
            raise ValueError("--data_root required for --dataset=xlerobot")
        cfg = XLeRobotDataConfig(data_root=args.data_root, encoding=args.encoding)
        return XLeRobotSpiritDataset(cfg)
    if args.dataset == "robochallenge":
        from dataset import RoboChallengeDataset, DataConfig
        if not args.data_root:
            raise ValueError("--data_root required for --dataset=robochallenge")
        return RoboChallengeDataset(DataConfig(data_root=args.data_root))
    raise ValueError(args.dataset)


# --------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------- #


def build_cosine_scheduler(optimizer, warmup_steps, total_steps, base_lr, final_lr):
    decay_steps = max(1, total_steps - warmup_steps)
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = min(max((step - warmup_steps) / decay_steps, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return final_lr / base_lr + (1.0 - final_lr / base_lr) * cosine
    return LambdaLR(optimizer, lr_lambda)


def main():
    args = parse_args()
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    log_path = out / args.log_jsonl
    log_f = open(log_path, "a")
    print(f"[io] output_dir={out}  log={log_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}  bf16_supported={torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False}")

    # ---- Load model with our cpu-load + bf16 cast workaround ----
    print(f"[load] from {args.pretrained_path}")
    config_path = Path(args.pretrained_path) / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"No config.json at {config_path}")
    with open(config_path) as f:
        raw_config = types.SimpleNamespace(**json.load(f))

    from model import SpiritVLAPolicy
    t0 = time.time()
    model = SpiritVLAPolicy.from_pretrained(
        ckpt_path=args.pretrained_path, strict=False, train=True
    )
    print(f"[load] model loaded in {time.time()-t0:.1f}s")

    # Apply patch BEFORE training (sample_noise dtype-aware)
    patch_spirit_for_bf16()

    # Move to device, cast to bf16 (consumer GPU friendly + same as inference)
    model = model.to(device).to(torch.bfloat16)
    print(f"[device] model on {device}, dtype=bf16")

    # ---- Norm stats ----
    print("[data] building dataset...")
    dataset = build_dataset(args)
    print(f"[data] {type(dataset).__name__}: {len(dataset)} samples")

    if not args.skip_norm_stats:
        print(f"[norm] computing stats over {min(args.norm_num_samples, len(dataset))} samples...")
        norm_stats = compute_norm_stats_from_loader(
            dataset, num_samples=min(args.norm_num_samples, len(dataset))
        )
        set_norm_stats(model, norm_stats, device)
        print(f"[norm] state range [{norm_stats['state_min'].min():.3f}, "
              f"{norm_stats['state_max'].max():.3f}]")

    # ---- LoRA ----
    model, trainable_names = inject_lora(model, args)
    print(f"[lora] sample trainable param names:")
    for n in trainable_names[:5]:
        print(f"        {n}")
    if len(trainable_names) > 5:
        print(f"        ... ({len(trainable_names)-5} more)")

    # ---- DataLoader ----
    collate = getattr(dataset, "collate_fn", None)
    dl = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=collate, pin_memory=True,
    )

    # ---- Optim ----
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optim = AdamW(trainable_params, lr=args.lr,
                  betas=(0.9, 0.95), eps=1e-8,
                  weight_decay=args.weight_decay)
    sched = build_cosine_scheduler(
        optim, warmup_steps=args.warmup_steps,
        total_steps=args.max_train_steps,
        base_lr=args.lr, final_lr=args.lr * 0.1,
    )

    # ---- Train loop ----
    print(f"[train] starting {args.max_train_steps} steps, batch={args.batch_size}")
    model.train()

    step = 0
    epoch = 0
    data_iter = iter(dl)
    t_start = time.time()
    losses = []
    while step < args.max_train_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            epoch += 1
            data_iter = iter(dl)
            batch = next(data_iter)

        # Move tensors to device + bf16 (Spirit's normalize layers accept fp32,
        # but DiT body is bf16 — autocast handles the rest)
        batch = {
            k: (v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v)
            for k, v in batch.items()
        }

        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss, log_dict = model(batch)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable_params, args.grad_clip_norm)
        optim.step()
        sched.step()
        optim.zero_grad()

        losses.append(loss.item())
        if step % args.log_interval == 0:
            mean_recent = sum(losses[-args.log_interval:]) / max(1, len(losses[-args.log_interval:]))
            lr_now = sched.get_last_lr()[0]
            elapsed = time.time() - t_start
            steps_per_sec = (step + 1) / max(1.0, elapsed)
            mem = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
            entry = {
                "step": step, "epoch": epoch, "loss": loss.item(),
                "loss_mean": mean_recent, "lr": lr_now,
                "steps_per_sec": steps_per_sec, "gpu_mem_gb": mem,
                "elapsed_s": elapsed,
            }
            log_f.write(json.dumps(entry) + "\n")
            log_f.flush()
            print(
                f"[step {step:5d}] loss={loss.item():.4f}  mean={mean_recent:.4f}  "
                f"lr={lr_now:.2e}  {steps_per_sec:.2f} step/s  mem={mem:.1f}GB"
            )

        if (step + 1) % args.save_steps == 0:
            save_dir = out / f"checkpoint-{step+1}"
            save_dir.mkdir(parents=True, exist_ok=True)
            # If peft, use peft's save (saves only adapter)
            if hasattr(model, "save_pretrained"):
                model.save_pretrained(str(save_dir))
            else:
                torch.save(
                    {n: p.detach().cpu() for n, p in model.named_parameters()
                     if p.requires_grad},
                    save_dir / "trainable_params.pt",
                )
            print(f"[ckpt] saved → {save_dir}")

        step += 1

    # Final save
    final = out / f"checkpoint-{step}"
    final.mkdir(parents=True, exist_ok=True)
    if hasattr(model, "save_pretrained"):
        model.save_pretrained(str(final))
    else:
        torch.save(
            {n: p.detach().cpu() for n, p in model.named_parameters()
             if p.requires_grad},
            final / "trainable_params.pt",
        )
    print(f"[done] final ckpt → {final}")
    print(f"[done] {step} steps in {time.time()-t_start:.1f}s")
    log_f.close()


if __name__ == "__main__":
    main()
