"""Single-device Spirit LoRA SFT on prepared LIBERO demonstrations.

Uses Spirit's actual compute_loss and the same adapter/checkpoint layout as
post_training.eval_libero. Run with python -m spirit_adapter.train_libero.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time

import torch
from torch.utils.data import DataLoader

from post_training import PostTrainConfig
from post_training.adapters.spirit import SpiritAdapter
from .libero_dataset import LiberoSpiritDataset
from .train_lora import set_norm_stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base_ckpt", required=True)
    parser.add_argument("--backbone", default="", help="Qwen config/tokenizer/processor assets; no Qwen weights are loaded")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--episodes", nargs="+", help="Training episode IDs; keep held-out episodes separate")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_steps", type=int, default=5000)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--lora_r", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=64)
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="bfloat16")
    parser.add_argument("--device", default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--log_steps", type=int, default=10)
    args = parser.parse_args()
    if min(args.max_steps, args.batch_size, args.save_steps, args.log_steps) < 1:
        raise ValueError("Training steps, batch size, and logging/saving intervals must be positive")
    torch.manual_seed(args.seed)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    dataset = LiberoSpiritDataset(args.dataset, episodes=args.episodes)
    cfg = PostTrainConfig(
        name="spirit_libero_sft", base="spirit", algorithm="sft",
        base_ckpt_path=args.base_ckpt, base_dtype=args.dtype,
        spirit_backbone_path=args.backbone, spirit_action_hz=dataset.action_hz,
        spirit_device=args.device,
        use_lora=True, lora_r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.0,
        max_train_steps=args.max_steps, batch_size=args.batch_size,
        learning_rate=args.lr, output_dir=args.output_dir,
    )
    adapter = SpiritAdapter(cfg)
    if dataset.horizon != adapter.model.config.n_action_steps:
        raise ValueError("Dataset and Spirit action horizons differ")
    set_norm_stats(adapter.model, dataset.normalization_stats(), adapter.device)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    parameters = list(adapter.trainable_parameters())
    optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=1e-4)
    (out / "training_meta.json").write_text(json.dumps({
        "config": asdict(cfg), "dataset": str(Path(args.dataset).resolve()),
        "training_episodes": sorted(dataset.states), "max_steps": args.max_steps,
        "seed": args.seed,
    }, indent=2))
    iterator = iter(loader)
    adapter.model.train()
    adapter.model.qwen.eval()
    started = time.monotonic()
    with (out / "train.jsonl").open("w") as log:
        for step in range(args.max_steps):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                batch = next(iterator)
            batch = adapter._prepare_batch(batch)
            optimizer.zero_grad(set_to_none=True)
            with adapter._autocast():
                loss, _ = adapter.model(batch)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite Spirit training loss")
            loss.backward()
            gradient = torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
            optimizer.step()
            record = {"step": step + 1, "loss": loss.item(), "grad_norm": gradient.item(),
                      "elapsed_s": time.monotonic() - started}
            log.write(json.dumps(record) + "\n")
            log.flush()
            if step == 0 or (step + 1) % args.log_steps == 0:
                print(record, flush=True)
            if (step + 1) % args.save_steps == 0:
                adapter.save(str(out / f"adapter-step{step+1}.pt"))
    checkpoint = out / "adapter.pt"
    adapter.save(str(checkpoint))
    print(f"Saved adapter and normalization statistics: {checkpoint}")


if __name__ == "__main__":
    main()
