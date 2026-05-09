"""DPO trainer for VLA — base-agnostic.

Drop-in for any adapter implementing VLABase. The trainer:

  1. Builds pre-collected DPO dataset via env-coupled sampling
     (see rollout.py — TODO next module). For first-pass smoke test
     we accept a synthetic dataset of (chunk_chosen, chunk_rejected)
     pairs precomputed by the user.
  2. Runs DPO loss + AdamW + cosine LR + grad clipping.
  3. Logs per-step metrics to JSONL for offline analysis.
  4. Optionally evals on LIBERO at end of training.

Usage (smoke):

    python -m post_training.train_dpo \
        --base openvla \
        --base_ckpt /workspace/models/openvla-7b-finetuned-libero-spatial \
        --suite spatial \
        --pairs_file /workspace/datasets/spatial_pairs.pt \
        --output_dir /workspace/output/openvla_dpo_spatial \
        --max_steps 500
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset

# Local
from . import PostTrainConfig
from .dpo_loss import DPOConfig, dpo_loss


# ---------------------------------------------------------------------- #
# Dataset
# ---------------------------------------------------------------------- #


class DPOPairDataset(Dataset):
    """Pre-collected DPO pair dataset.

    Format (saved by rollout.py via torch.save):

        {
            "instructions": List[str],            # length B
            "images": Tensor (B, 3, H, W),
            "chosen_chunks": Tensor (B, T, A),
            "rejected_chunks": Tensor (B, T, A),
            "chosen_rewards": Tensor (B,),         # for diagnostics
            "rejected_rewards": Tensor (B,),
        }
    """

    def __init__(self, pairs_file: str):
        d = torch.load(pairs_file, map_location="cpu", weights_only=False)
        self.instructions: list[str] = d["instructions"]
        self.images: torch.Tensor = d["images"]
        self.chosen_chunks: torch.Tensor = d["chosen_chunks"]
        self.rejected_chunks: torch.Tensor = d["rejected_chunks"]
        self.chosen_rewards: torch.Tensor = d.get(
            "chosen_rewards", torch.zeros(len(self.instructions))
        )
        self.rejected_rewards: torch.Tensor = d.get(
            "rejected_rewards", torch.zeros(len(self.instructions))
        )
        assert (
            len(self.instructions)
            == self.images.shape[0]
            == self.chosen_chunks.shape[0]
            == self.rejected_chunks.shape[0]
        )

    def __len__(self) -> int:
        return len(self.instructions)

    def __getitem__(self, idx: int) -> dict:
        return {
            "instruction": self.instructions[idx],
            "image": self.images[idx],
            "chosen_chunk": self.chosen_chunks[idx],
            "rejected_chunk": self.rejected_chunks[idx],
            "chosen_reward": self.chosen_rewards[idx].item(),
            "rejected_reward": self.rejected_rewards[idx].item(),
        }

    @staticmethod
    def collate_fn(batch: list[dict]) -> dict:
        return {
            "instruction": [b["instruction"] for b in batch],
            "image": torch.stack([b["image"] for b in batch]),
            "chosen_chunk": torch.stack([b["chosen_chunk"] for b in batch]),
            "rejected_chunk": torch.stack([b["rejected_chunk"] for b in batch]),
            "chosen_reward": torch.tensor([b["chosen_reward"] for b in batch]),
            "rejected_reward": torch.tensor([b["rejected_reward"] for b in batch]),
        }


# ---------------------------------------------------------------------- #
# Adapter factory
# ---------------------------------------------------------------------- #


def build_adapter(cfg: PostTrainConfig):
    if cfg.base == "openvla":
        from .adapters.openvla import OpenVLAAdapter
        return OpenVLAAdapter(cfg)
    if cfg.base == "spirit":
        from .adapters.spirit import SpiritAdapter
        return SpiritAdapter(cfg)
    if cfg.base == "pi05":
        raise NotImplementedError(
            "π0.5 adapter scheduled for Week 2; needs JAX/PyTorch path "
            "decision (see docs/openpi_onboard.md)"
        )
    raise ValueError(f"unknown base: {cfg.base}")


def build_cosine_scheduler(
    optimizer, warmup_steps: int, total_steps: int, base_lr: float, final_lr: float
):
    decay_steps = max(1, total_steps - warmup_steps)
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = min(max((step - warmup_steps) / decay_steps, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return final_lr / base_lr + (1.0 - final_lr / base_lr) * cosine
    return LambdaLR(optimizer, lr_lambda)


# ---------------------------------------------------------------------- #
# Main trainer
# ---------------------------------------------------------------------- #


def parse_args():
    p = argparse.ArgumentParser(description="DPO trainer for VLA")
    p.add_argument("--base", choices=["openvla", "spirit", "pi05"], required=True)
    p.add_argument("--base_ckpt", required=True)
    p.add_argument(
        "--suite",
        default="spatial",
        choices=["spatial", "object", "goal", "long10", "all4"],
    )
    p.add_argument("--pairs_file", required=True,
                   help="Pre-collected DPO pair dataset (.pt)")
    p.add_argument("--output_dir", required=True)

    # Training
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--max_steps", type=int, default=500)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--save_every", type=int, default=200)
    p.add_argument("--log_every", type=int, default=10)

    # DPO
    p.add_argument("--beta", type=float, default=0.1)
    p.add_argument("--label_smoothing", type=float, default=0.0)
    p.add_argument("--reference_free", action="store_true")

    # LoRA
    p.add_argument("--lora_r", type=int, default=32)
    p.add_argument("--lora_alpha", type=int, default=64)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--no_lora", action="store_true")

    return p.parse_args()


def main():
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    log_path = out / "train_log.jsonl"
    log_f = open(log_path, "a")
    print(f"[io] output_dir={out}  log={log_path}")

    # Build config
    cfg = PostTrainConfig(
        name=f"{args.base}_dpo_{args.suite}",
        base=args.base,
        algorithm="dpo",
        base_ckpt_path=args.base_ckpt,
        libero_suite=args.suite,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        max_train_steps=args.max_steps,
        warmup_steps=args.warmup,
        grad_clip_norm=args.grad_clip,
        use_lora=not args.no_lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        dpo_beta=args.beta,
        output_dir=str(out),
        log_interval=args.log_every,
        save_steps=args.save_every,
    )
    print(f"[cfg] {cfg.name}")

    # Build adapter (loads the base model + applies LoRA)
    adapter = build_adapter(cfg)

    # Snapshot reference policy BEFORE any training step
    adapter.freeze_reference()

    # Build dataset
    dataset = DPOPairDataset(args.pairs_file)
    print(f"[data] {len(dataset)} DPO pairs from {args.pairs_file}")

    dl = DataLoader(
        dataset, batch_size=cfg.batch_size, shuffle=True,
        num_workers=0, collate_fn=DPOPairDataset.collate_fn,
        pin_memory=True,
    )

    # Optimiser (only over trainable params — LoRA + projections)
    trainable = list(adapter.trainable_parameters())
    n_trainable = sum(p.numel() for p in trainable)
    print(f"[optim] {n_trainable/1e6:.2f}M trainable params")

    optim = AdamW(
        trainable, lr=cfg.learning_rate, betas=(0.9, 0.95), eps=1e-8,
        weight_decay=cfg.weight_decay,
    )
    sched = build_cosine_scheduler(
        optim, warmup_steps=cfg.warmup_steps,
        total_steps=cfg.max_train_steps,
        base_lr=cfg.learning_rate, final_lr=cfg.learning_rate * 0.1,
    )

    dpo_cfg = DPOConfig(
        beta=cfg.dpo_beta,
        label_smoothing=args.label_smoothing,
        reference_free=args.reference_free,
    )

    # Train loop
    print(f"[train] starting {cfg.max_train_steps} steps, batch={cfg.batch_size}")
    step = 0
    epoch = 0
    data_iter = iter(dl)
    t_start = time.time()

    while step < cfg.max_train_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            epoch += 1
            data_iter = iter(dl)
            batch = next(data_iter)

        # Compute logp for chosen and rejected, both under cur and ref
        # Adapter is responsible for batching + dtype handling.
        logp_cur_c, logp_ref_c = adapter.policy_logp_with_ref(
            batch, batch["chosen_chunk"]
        )
        logp_cur_r, logp_ref_r = adapter.policy_logp_with_ref(
            batch, batch["rejected_chunk"]
        )

        out_dpo = dpo_loss(
            logp_cur_c, logp_cur_r, logp_ref_c, logp_ref_r, dpo_cfg
        )
        loss = out_dpo.loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, cfg.grad_clip_norm)
        optim.step()
        sched.step()
        optim.zero_grad()

        if step % cfg.log_interval == 0:
            entry = {
                "step": step,
                "epoch": epoch,
                "loss": float(loss.item()),
                "accuracy": float(out_dpo.accuracy.item()),
                "margin": float(out_dpo.margin.item()),
                "chosen_reward_mean": float(out_dpo.chosen_rewards.mean().item()),
                "rejected_reward_mean": float(out_dpo.rejected_rewards.mean().item()),
                "lr": sched.get_last_lr()[0],
                "elapsed_s": time.time() - t_start,
            }
            log_f.write(json.dumps(entry) + "\n")
            log_f.flush()
            print(
                f"[step {step:5d}] loss={loss.item():.4f}  "
                f"acc={out_dpo.accuracy.item():.3f}  "
                f"margin={out_dpo.margin.item():+.4f}  "
                f"lr={sched.get_last_lr()[0]:.2e}"
            )

        if (step + 1) % cfg.save_steps == 0:
            ckpt = out / f"checkpoint-{step+1}.pt"
            adapter.save(str(ckpt))
            print(f"[ckpt] saved → {ckpt}")

        step += 1

    final = out / f"checkpoint-{step}.pt"
    adapter.save(str(final))
    log_f.close()
    print(f"[done] {step} steps in {time.time()-t_start:.1f}s, final → {final}")


if __name__ == "__main__":
    main()
