"""Multi-view + temporal contrastive pretraining trainer.

Usage:
    python -m pretrain.train --root /path/to/libero/demos \
                             --epochs 10 --batch_size 256 \
                             --output_dir /path/to/save

For Day 6 smoke test (no real data):
    python -m pretrain.train --synthetic --epochs 1 --batch_size 16 \
                             --output_dir /tmp/pretrain_smoke
"""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from pretrain.dataset import LiberoPretrainDataset
from pretrain.model import MultiViewProjectionHead, multiview_temporal_loss


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default=None)
    p.add_argument("--suites", nargs="+", default=["spatial", "object", "goal", "10"])
    p.add_argument("--delta_steps", type=int, default=5)
    p.add_argument("--max_per_episode", type=int, default=30)
    p.add_argument("--synthetic", action="store_true")
    p.add_argument("--synthetic_n", type=int, default=2000)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--temperature", type=float, default=0.07)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--proj_in_dim", type=int, default=1152,
                   help="SigLIP feature dim (1152 for shape-optimized siglip-vit)")
    p.add_argument("--proj_out_dim", type=int, default=128)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--encoder_path", default=None,
                   help="Path to a pretrained SigLIP encoder. None → random for smoke.")
    p.add_argument("--smoke", action="store_true",
                   help="Use mock encoder (random projection 3*224*224 → 1152) "
                        "for fast Day 6 verification.")
    return p.parse_args()


class MockEncoder(nn.Module):
    """Tiny encoder for smoke test (replaces SigLIP).

    Real training will substitute SigLIP-vit (frozen).
    """
    def __init__(self, in_pixels: int = 3 * 224 * 224, out_dim: int = 1152):
        super().__init__()
        self.proj = nn.Linear(in_pixels, out_dim, bias=False)

    def forward(self, x):  # x: (B, 3, 224, 224)
        return self.proj(x.flatten(1))


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    print(f"[pretrain] device={device}, smoke={args.smoke}")

    # ----- Data -----
    dataset = LiberoPretrainDataset(
        root=args.root, suites=tuple(args.suites),
        delta_steps=args.delta_steps,
        max_per_episode=args.max_per_episode,
        synthetic=args.synthetic or args.smoke,
        synthetic_n=args.synthetic_n,
    )
    print(f"[pretrain] dataset: {len(dataset)} samples")
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )

    # ----- Model -----
    if args.smoke:
        encoder = MockEncoder(out_dim=args.proj_in_dim).to(device)
        for p in encoder.parameters():
            p.requires_grad = False
    else:
        # TODO Day 7: load actual SigLIP-vit from openvla / pi05 ckpt
        print("[pretrain] non-smoke mode TODO: hook SigLIP encoder")
        encoder = MockEncoder(out_dim=args.proj_in_dim).to(device)
        for p in encoder.parameters():
            p.requires_grad = False

    proj = MultiViewProjectionHead(
        in_dim=args.proj_in_dim, out_dim=args.proj_out_dim
    ).to(device)
    print(f"[pretrain] proj_head trainable: "
          f"{sum(p.numel() for p in proj.parameters())/1e3:.1f}K params")

    # ----- Optim -----
    optim = torch.optim.AdamW(
        proj.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim, T_max=args.epochs * len(loader)
    )

    # ----- Train -----
    log = []
    t0 = time.time()
    step = 0
    for epoch in range(args.epochs):
        for batch in loader:
            agent_t = batch["agent_t"].to(device, non_blocking=True)
            wrist_t = batch["wrist_t"].to(device, non_blocking=True)
            agent_t_d = batch["agent_t_delta"].to(device, non_blocking=True)

            with torch.no_grad():
                f_at = encoder(agent_t)
                f_wt = encoder(wrist_t)
                f_atd = encoder(agent_t_d)

            z_at = proj(f_at)
            z_wt = proj(f_wt)
            z_atd = proj(f_atd)

            loss_dict = multiview_temporal_loss(
                z_at, z_wt, z_atd, tau=args.temperature
            )
            loss = loss_dict["total_loss"]
            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(proj.parameters(), 1.0)
            optim.step()
            sched.step()
            step += 1

            if step % 10 == 0:
                entry = {
                    "epoch": epoch, "step": step,
                    "total_loss": float(loss.item()),
                    "mva_loss": float(loss_dict["mva_loss"].item()),
                    "tc_loss": float(loss_dict["tc_loss"].item()),
                    "lr": optim.param_groups[0]["lr"],
                    "elapsed_s": time.time() - t0,
                }
                log.append(entry)
                print(f"[step {step:5d}] e={epoch} "
                      f"loss={entry['total_loss']:.4f} "
                      f"(mva={entry['mva_loss']:.4f}, "
                      f"tc={entry['tc_loss']:.4f}) "
                      f"lr={entry['lr']:.2e}")

    # ----- Save -----
    torch.save({
        "proj_head": proj.state_dict(),
        "config": vars(args),
    }, out_dir / "proj_head.pt")
    with open(out_dir / "train_log.json", "w") as f:
        json.dump(log, f, indent=2)
    print(f"[pretrain] saved to {out_dir}")


if __name__ == "__main__":
    main()
