"""Multi-view + temporal contrastive pretraining trainer.

Three operating modes:

1. ``--smoke``: Mock encoder (random-init Linear) + structured-synthetic
   data. ~2 min on H20. Validates loss + dataloader API.
   ``python -m pretrain.train --smoke --output_dir /tmp/smoke``

2. ``--rlds_root <path>``: real LIBERO RLDS dataset + SigLIP encoder.
   Used for full Day 8 run.
   ``python -m pretrain.train --rlds_root /path/to/modified_libero_rlds \\
        --siglip_path google/siglip-so400m-patch14-384 \\
        --epochs 10 --batch_size 256 --output_dir <out>``

3. Legacy ``--synthetic`` flag: kept for backwards compat (= --smoke).
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
    # Data
    p.add_argument("--root", default=None,
                   help="(legacy/HDF5 path; not used in RLDS mode)")
    p.add_argument("--rlds_root", default=None,
                   help="Path to modified_libero_rlds directory (for real training)")
    p.add_argument("--suites", nargs="+", default=["spatial", "object", "goal", "10"])
    p.add_argument("--delta_steps", type=int, default=5)
    p.add_argument("--max_per_episode", type=int, default=30)
    p.add_argument("--max_episodes_per_suite", type=int, default=None,
                   help="Cap episodes per suite (None=all 432). Set 50 for quick smoke.")
    # Training
    p.add_argument("--synthetic", action="store_true",
                   help="(legacy) use structured-noise dataset; same as --smoke")
    p.add_argument("--synthetic_n", type=int, default=2000)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--temperature", type=float, default=0.07)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    # Model
    p.add_argument("--proj_in_dim", type=int, default=1152,
                   help="Encoder output dim. SigLIP-so400m: 1152, SigLIP-base: 768.")
    p.add_argument("--proj_out_dim", type=int, default=128)
    p.add_argument("--siglip_path", default=None,
                   help="HF model id or local path to SigLIP. None → MockEncoder.")
    p.add_argument("--smoke", action="store_true",
                   help="Use mock encoder + structured-synthetic data for fast verification.")
    # Output
    p.add_argument("--output_dir", required=True)
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


class SiglipEncoderWrapper(nn.Module):
    """Wraps timm SigLIP-so400m vision tower (frozen) for pretraining.

    Why timm not HF transformers:
      OpenVLA uses timm `vit_so400m_patch14_siglip_224` and we want to
      reuse the exact same weights (so our pretrained features stack
      cleanly on top of OpenVLA's vision encoder downstream). Loading
      via timm + state_dict from OpenVLA safetensors avoids the timm→HF
      key remap and keeps weights bit-identical.

    Args:
        siglip_path: either
          (a) "timm" → timm-init random weights (sanity)
          (b) path to OpenVLA safetensors directory → extract
              vision_backbone.featurizer.<...> keys and load.
          (c) None → MockEncoder used instead (handled by _build_encoder)

    Output: (B, 1152) pooled features (timm forward_features→head).
    """
    TIMM_ARCH = "vit_so400m_patch14_siglip_224"

    def __init__(self, siglip_path: str):
        super().__init__()
        import timm
        self.model = timm.create_model(self.TIMM_ARCH, pretrained=False, num_classes=0)
        self.image_size = 224

        if siglip_path != "timm" and Path(siglip_path).exists():
            self._load_from_openvla(siglip_path)
            print(f"[encoder] loaded SigLIP from OpenVLA ckpt: {siglip_path}")
        else:
            print(f"[encoder] SigLIP timm-init RANDOM weights (siglip_path={siglip_path}); "
                  f"this is for sanity/smoke ONLY — features won't be meaningful.")

        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
        n_params = sum(p.numel() for p in self.model.parameters())
        print(f"[encoder] SigLIP frozen, {n_params/1e6:.1f}M params, output_dim=1152")

    def _load_from_openvla(self, openvla_dir: str):
        """Extract vision_backbone.featurizer.<x> from OpenVLA safetensors.

        OpenVLA stores the SigLIP under prefix `vision_backbone.featurizer.`
        in the same timm key naming as the underlying timm model — so we
        just strip the prefix and load.

        OpenVLA also has DINOv2 fused with SigLIP (dinosiglip backbone).
        We pick only the SigLIP half. The fused backbone uses
        `vision_backbone.featurizer` for the FIRST encoder (SigLIP per
        OpenVLA config order) — but timm keys are `blocks.0.norm1.weight`
        etc. We filter by looking for keys that start with
        `vision_backbone.featurizer.` and remap.

        Reference: openvla/prismatic/models/backbones/vision/dinosiglip_vit.py
        """
        from safetensors.torch import load_file
        oroot = Path(openvla_dir)
        # OpenVLA-7B = 4 shards. Iterate all and merge.
        merged = {}
        for shard in sorted(oroot.glob("model-*-of-*.safetensors")):
            sd = load_file(shard)
            for k, v in sd.items():
                if k.startswith("vision_backbone.featurizer."):
                    new_k = k[len("vision_backbone.featurizer."):]
                    merged[new_k] = v
        if not merged:
            raise RuntimeError(f"no vision_backbone.featurizer.* keys in {openvla_dir}")
        # The OpenVLA 'featurizer' is the FIRST encoder of the dinosiglip
        # fused backbone — by config inspection, the first listed
        # timm_model_ids is DINOv2, second is SigLIP. So 'featurizer' is
        # actually DINOv2! For SigLIP we need 'fused_featurizer'.
        # → Re-extract with the correct prefix.
        merged = {}
        for shard in sorted(oroot.glob("model-*-of-*.safetensors")):
            sd = load_file(shard)
            for k, v in sd.items():
                if k.startswith("vision_backbone.fused_featurizer."):
                    new_k = k[len("vision_backbone.fused_featurizer."):]
                    merged[new_k] = v
        if not merged:
            # Fallback: use featurizer (DINOv2). Still useful pretrained
            # features even if not SigLIP.
            print("[encoder] WARNING: no fused_featurizer.* keys found, "
                  "falling back to featurizer.* (likely DINOv2)")
            for shard in sorted(oroot.glob("model-*-of-*.safetensors")):
                sd = load_file(shard)
                for k, v in sd.items():
                    if k.startswith("vision_backbone.featurizer."):
                        new_k = k[len("vision_backbone.featurizer."):]
                        merged[new_k] = v
        # Load with strict=False (timm vs OpenVLA may have small diffs)
        missing, unexpected = self.model.load_state_dict(merged, strict=False)
        print(f"[encoder] state_dict load: {len(merged)} keys, "
              f"missing={len(missing)}, unexpected={len(unexpected)}")
        if missing[:3]:
            print(f"[encoder]   first missing: {missing[:3]}")
        if unexpected[:3]:
            print(f"[encoder]   first unexpected: {unexpected[:3]}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args: x — (B, 3, 224, 224) in [-1, 1] (SigLIP normalisation).
        Returns: (B, 1152) pooled features (timm head with num_classes=0
                 returns the pre-head pooled feature directly).
        """
        with torch.no_grad():
            return self.model(x)


def _build_encoder(args, device) -> tuple[nn.Module, int]:
    """Construct encoder per args.

    Returns:
        encoder: nn.Module
        image_size: int (224 for mock; 224 for SigLIP-so400m via timm)
    """
    if args.siglip_path:
        encoder = SiglipEncoderWrapper(args.siglip_path).to(device)
        return encoder, encoder.image_size
    else:
        encoder = MockEncoder(out_dim=args.proj_in_dim).to(device)
        for p in encoder.parameters():
            p.requires_grad = False
        return encoder, 224


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    print(f"[pretrain] device={device}, smoke={args.smoke}, rlds_root={args.rlds_root}")

    # ----- Encoder first (so we know required image size) -----
    encoder, encoder_image_size = _build_encoder(args, device)
    print(f"[pretrain] encoder image size: {encoder_image_size}")

    # ----- Data -----
    if args.rlds_root:
        from pretrain.dataset_rlds import LiberoRLDSPretrainDataset
        dataset = LiberoRLDSPretrainDataset(
            rlds_root=args.rlds_root,
            suites=tuple(args.suites),
            delta_steps=args.delta_steps,
            max_per_episode=args.max_per_episode,
            max_episodes_per_suite=args.max_episodes_per_suite,
            target_size=encoder_image_size,
        )
    else:
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
