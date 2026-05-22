"""Reward-shaping evaluation of the multi-view InfoNCE proj_head ckpt.

Tests whether the projection head's 128-d embeddings can serve as a
chunk-level reward shaping signal for VLA post-training.

Hypothesis: for a successful trajectory of length T, the cosine
similarity between embedding(frame_t) and embedding(frame_T) (the goal
state) should grow approximately monotonically as t → T. If true, the
proj_head provides a free dense reward signal that can be reused
downstream (DPO reward shaping, RL critic warm-start, etc.).

Outputs:
  reward_shaping_results.json   — per-suite quantitative metrics
  reward_curves.png             — visual: 4 suites × 5 sample trajectories

Usage:
  python reward_shaping_eval.py \
    --rlds_root /path/to/modified_libero_rlds \
    --siglip_path /path/to/openvla-7b-finetuned-libero-spatial \
    --proj_ckpt models/pretrain_rlds_siglip_day8/proj_head.pt \
    --output_dir /tmp/reward_shaping_eval

Compute: 5 trajectories × 4 suites × ~150 frames = 3000 frame forward
+ cosine sim matrix → ~5 min on 1×H20.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--rlds_root", required=True)
    p.add_argument("--siglip_path", required=True)
    p.add_argument("--proj_ckpt", required=True)
    p.add_argument("--n_trajectories_per_suite", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--goal_window", type=int, default=10,
                   help="Last K frames averaged as the goal embedding (default 10)")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "code"))
    from pretrain.model import MultiViewProjectionHead
    from pretrain.train import SiglipEncoderWrapper

    # --- Load encoder + proj head ---
    print(f"[reward] loading SigLIP from {args.siglip_path}", flush=True)
    encoder = SiglipEncoderWrapper(args.siglip_path).to(device).eval()
    print(f"[reward] loading proj_head from {args.proj_ckpt}", flush=True)
    proj = MultiViewProjectionHead(in_dim=1152, hidden=512, out_dim=128).to(device).eval()
    ckpt = torch.load(args.proj_ckpt, map_location=device, weights_only=False)
    proj.load_state_dict(ckpt["proj_head"])

    # --- Load whole trajectories ---
    print(f"[reward] loading RLDS trajectories from {args.rlds_root}", flush=True)
    import tensorflow as tf
    tf.config.set_visible_devices([], "GPU")
    import tensorflow_datasets as tfds

    suites = ["spatial", "object", "goal", "10"]
    suite_to_label = {"spatial": "Spatial", "object": "Object", "goal": "Goal", "10": "Long10"}

    # Per-suite per-trajectory reward curves (cosine sim to goal)
    results = {
        "config": vars(args),
        "per_suite": {},
        "global": {},
    }
    all_curves = {}  # for plotting

    for suite in suites:
        suite_dir = Path(args.rlds_root) / f"libero_{suite}_no_noops" / "1.0.0"
        if not suite_dir.exists():
            print(f"[reward] WARN: {suite_dir} missing, skipping", flush=True)
            continue
        print(f"[reward] === suite {suite} ===", flush=True)
        ds = tfds.builder_from_directory(str(suite_dir)).as_dataset(split="train")
        ds = ds.take(args.n_trajectories_per_suite).prefetch(tf.data.AUTOTUNE)

        suite_curves = []
        suite_metrics = {
            "n_trajectories": 0,
            "monotonicity_ratios": [],   # per-trajectory: % of (r_{t+1} > r_t) steps
            "spread_start_to_goal": [],  # r_T - r_0
            "early_late_separation": [], # mean r in last 20% - mean r in first 20%
            "goal_window_consistency": [], # std of r_T-K..T_T (low = goal stable)
        }

        for ep in tfds.as_numpy(ds):
            steps = ep["steps"]
            agent_frames = steps["observation"]["image"]  # (T, 256, 256, 3) uint8
            T = agent_frames.shape[0]
            if T < args.goal_window + 5:
                continue

            # Encode all frames
            embeds = []
            for start in range(0, T, args.batch_size):
                end = min(start + args.batch_size, T)
                batch = []
                for j in range(start, end):
                    img_uint8 = agent_frames[j]
                    if img_uint8.shape[0] != 224:
                        import cv2
                        img_uint8 = cv2.resize(img_uint8, (224, 224), interpolation=cv2.INTER_LANCZOS4)
                    t = torch.from_numpy(img_uint8).float() / 255.0
                    t = t * 2.0 - 1.0
                    batch.append(t.permute(2, 0, 1))
                x = torch.stack(batch).to(device)
                with torch.no_grad():
                    feat = encoder(x)
                    z = proj(feat)
                embeds.append(z.cpu())
            z_traj = torch.cat(embeds, dim=0)  # (T, 128)
            assert z_traj.shape[0] == T

            # Goal embedding = mean of last K frames
            K = args.goal_window
            z_goal = z_traj[-K:].mean(dim=0, keepdim=True)
            z_goal = z_goal / z_goal.norm(dim=-1, keepdim=True).clamp(min=1e-8)

            # Reward curve: cos sim to goal
            r_curve = (z_traj @ z_goal.T).squeeze(-1).numpy()  # (T,)

            # Metrics
            mono_ratio = float((r_curve[1:] > r_curve[:-1]).mean())
            spread = float(r_curve[-1] - r_curve[0])
            n_seg = max(T // 5, 1)
            early = r_curve[:n_seg].mean()
            late = r_curve[-n_seg:].mean()
            sep = float(late - early)
            goal_cons = float(np.std(r_curve[-K:]))

            suite_metrics["n_trajectories"] += 1
            suite_metrics["monotonicity_ratios"].append(mono_ratio)
            suite_metrics["spread_start_to_goal"].append(spread)
            suite_metrics["early_late_separation"].append(sep)
            suite_metrics["goal_window_consistency"].append(goal_cons)
            suite_curves.append(r_curve.tolist())

        # Aggregate
        m = suite_metrics
        results["per_suite"][suite_to_label[suite]] = {
            "n_trajectories": m["n_trajectories"],
            "mean_monotonicity_ratio": float(np.mean(m["monotonicity_ratios"])) if m["monotonicity_ratios"] else 0.0,
            "mean_start_to_goal_spread": float(np.mean(m["spread_start_to_goal"])) if m["spread_start_to_goal"] else 0.0,
            "mean_early_late_separation": float(np.mean(m["early_late_separation"])) if m["early_late_separation"] else 0.0,
            "mean_goal_window_consistency": float(np.mean(m["goal_window_consistency"])) if m["goal_window_consistency"] else 0.0,
        }
        all_curves[suite_to_label[suite]] = suite_curves
        print(f"  monotonicity {results['per_suite'][suite_to_label[suite]]['mean_monotonicity_ratio']:.3f}, "
              f"spread {results['per_suite'][suite_to_label[suite]]['mean_start_to_goal_spread']:.3f}, "
              f"sep {results['per_suite'][suite_to_label[suite]]['mean_early_late_separation']:.3f}", flush=True)

    # Global aggregate
    all_mono = [v["mean_monotonicity_ratio"] for v in results["per_suite"].values()]
    all_spread = [v["mean_start_to_goal_spread"] for v in results["per_suite"].values()]
    all_sep = [v["mean_early_late_separation"] for v in results["per_suite"].values()]
    results["global"] = {
        "mean_monotonicity_ratio": float(np.mean(all_mono)) if all_mono else 0.0,
        "mean_start_to_goal_spread": float(np.mean(all_spread)) if all_spread else 0.0,
        "mean_early_late_separation": float(np.mean(all_sep)) if all_sep else 0.0,
    }

    print()
    print("=== GLOBAL ===")
    print(json.dumps(results["global"], indent=2))

    with open(out_dir / "reward_shaping_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"[reward] saved {out_dir / 'reward_shaping_results.json'}", flush=True)

    # --- Plot ---
    print("[reward] plotting curves...", flush=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    axes = axes.flatten()
    for ax, (suite_label, curves) in zip(axes, all_curves.items()):
        for i, c in enumerate(curves):
            ax.plot(c, alpha=0.7, lw=1.5, label=f"traj {i+1} (T={len(c)})")
        ax.axhline(y=1.0, color="gray", ls="--", alpha=0.5, lw=0.8)
        ax.set_title(f"{suite_label}  (n={len(curves)})", fontsize=11)
        ax.set_xlabel("trajectory step t")
        ax.set_ylabel("cos(z_t, z_goal)")
        ax.set_ylim(-0.1, 1.05)
        ax.grid(True, alpha=0.3)
        if len(curves) <= 6:
            ax.legend(fontsize=7, loc="lower right")

    fig.suptitle("Multi-view InfoNCE proj_head as chunk-level reward signal\n"
                 f"global monotonicity ratio = {results['global']['mean_monotonicity_ratio']:.2f}, "
                 f"start→goal spread = {results['global']['mean_start_to_goal_spread']:.2f}",
                 fontsize=13)
    plt.tight_layout()
    plt.savefig(out_dir / "reward_curves.png", dpi=130, bbox_inches="tight")
    print(f"[reward] saved {out_dir / 'reward_curves.png'}", flush=True)


if __name__ == "__main__":
    main()
