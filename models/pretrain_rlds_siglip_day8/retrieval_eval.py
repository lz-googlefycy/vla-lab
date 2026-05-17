"""k-NN retrieval evaluation of pretrained projection head.

For each query frame in LIBERO RLDS, compute the proj_head embedding,
find top-k nearest neighbors among ALL embeddings, and check whether
the retrieved frames belong to the **same task** (= same language
instruction) and / or **same episode**.

This is the standard self-supervised eval protocol (Wang et al. R3M;
Gupta et al. SiamMAE; Bardes et al. RILSS).

Output:
  retrieval_results.json
    - recall@1 / @5 / @10 for "same task" hit
    - recall@1 / @5 / @10 for "same episode" hit
    - recall@1 / @5 / @10 for "same task & |Δt|<=10" (temporal-aware)
    - per-suite breakdown
  retrieval_examples.png
    - 6 query frames + their top-3 retrievals (visual sanity check)

Usage:
  python retrieval_eval.py \\
    --rlds_root /path/to/modified_libero_rlds \\
    --siglip_path /path/to/openvla-7b-finetuned-libero-spatial \\
    --proj_ckpt /path/to/proj_head.pt \\
    --output_dir /tmp/retrieval_eval

Compute: 6000 frames × 1 H20 GPU forward + 6000² cosine sim → ~5 min.
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

os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--rlds_root", required=True)
    p.add_argument("--siglip_path", required=True)
    p.add_argument("--proj_ckpt", required=True)
    p.add_argument("--max_episodes_per_suite", type=int, default=50)
    p.add_argument("--max_per_episode", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--n_examples_to_plot", type=int, default=6)
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    # --- Imports (inject code/ to PYTHONPATH for direct script run) ---
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "code"))
    from pretrain.model import MultiViewProjectionHead
    from pretrain.train import SiglipEncoderWrapper
    from pretrain.dataset_rlds import LiberoRLDSPretrainDataset

    # --- Build dataset ---
    print(f"[retrieval] loading dataset from {args.rlds_root}")
    dataset = LiberoRLDSPretrainDataset(
        rlds_root=args.rlds_root,
        suites=("spatial", "object", "goal", "10"),
        delta_steps=5,
        max_per_episode=args.max_per_episode,
        max_episodes_per_suite=args.max_episodes_per_suite,
        target_size=224,
    )
    N = len(dataset)
    print(f"[retrieval] {N} samples across {len(dataset.episodes)} episodes")

    # --- Build encoder + proj head ---
    print(f"[retrieval] loading SigLIP from {args.siglip_path}")
    encoder = SiglipEncoderWrapper(args.siglip_path).to(device).eval()
    print(f"[retrieval] loading proj_head from {args.proj_ckpt}")
    proj = MultiViewProjectionHead(in_dim=1152, hidden=512, out_dim=128).to(device).eval()
    ckpt = torch.load(args.proj_ckpt, map_location=device, weights_only=False)
    proj.load_state_dict(ckpt["proj_head"])

    # --- Compute embeddings for ALL agent_t frames ---
    # We use only agent view at time t to build the retrieval index
    # (so retrieval is "agent view → agent view"). Wrist and Δ-frames
    # could be queried similarly.
    print(f"[retrieval] computing embeddings for {N} frames (batch={args.batch_size})...")
    t0 = time.time()
    embs = torch.zeros(N, 128, dtype=torch.float32)
    meta = []  # per-sample: {ep_idx, t, suite, instr}
    with torch.no_grad():
        for start in range(0, N, args.batch_size):
            end = min(start + args.batch_size, N)
            batch_imgs = []
            for idx in range(start, end):
                d = dataset[idx]
                batch_imgs.append(d["agent_t"])
                ep = dataset.episodes[dataset.samples[idx]["ep_idx"]]
                meta.append({
                    "ep_idx": dataset.samples[idx]["ep_idx"],
                    "t": dataset.samples[idx]["t"],
                    "suite": ep["suite"],
                    "instr": ep["instr"],
                })
            x = torch.stack(batch_imgs, dim=0).to(device)
            feat = encoder(x)         # (B, 1152)
            z = proj(feat)             # (B, 128) L2-normed
            embs[start:end] = z.cpu()
            if start % (args.batch_size * 10) == 0:
                print(f"  {end}/{N} ({(end/N*100):.0f}%)  t={time.time()-t0:.0f}s")
    print(f"[retrieval] embeddings done in {time.time()-t0:.0f}s, "
          f"shape={tuple(embs.shape)}, dtype={embs.dtype}")

    # --- Compute pairwise cosine sim (already L2-normalized → just dot) ---
    print("[retrieval] computing pairwise cosine sim matrix...")
    sim = embs @ embs.T  # (N, N), float32
    # Mask self-similarity
    sim.fill_diagonal_(-1e9)

    # --- Compute recall@k ---
    K = [1, 5, 10]
    metrics = {
        "n_total": N,
        "n_episodes": len(dataset.episodes),
        "config": vars(args),
        "global": {},
        "per_suite": {},
    }

    # Build "task_id" per sample (= hashed instr) and ep_id
    task_ids = np.array([hash(m["instr"]) % 100000 for m in meta])
    ep_ids = np.array([m["ep_idx"] for m in meta])
    times = np.array([m["t"] for m in meta])
    suites_arr = np.array([m["suite"] for m in meta])

    topk_max = max(K)
    topk_idx = sim.topk(topk_max, dim=1).indices.numpy()  # (N, topk_max)

    # Compute recall metrics
    same_task_hit = np.zeros((N, topk_max), dtype=bool)
    same_ep_hit = np.zeros((N, topk_max), dtype=bool)
    temporal_hit = np.zeros((N, topk_max), dtype=bool)  # same task AND |Δt|<=10

    for i in range(N):
        for j in range(topk_max):
            r = topk_idx[i, j]
            same_task_hit[i, j] = (task_ids[r] == task_ids[i])
            same_ep_hit[i, j] = (ep_ids[r] == ep_ids[i])
            temporal_hit[i, j] = same_task_hit[i, j] and abs(int(times[r]) - int(times[i])) <= 10

    for k in K:
        metrics["global"][f"recall@{k}_same_task"] = float(np.any(same_task_hit[:, :k], axis=1).mean())
        metrics["global"][f"recall@{k}_same_episode"] = float(np.any(same_ep_hit[:, :k], axis=1).mean())
        metrics["global"][f"recall@{k}_temporal"] = float(np.any(temporal_hit[:, :k], axis=1).mean())

    # Per-suite breakdown
    for suite in ("spatial", "object", "goal", "10"):
        mask = suites_arr == suite
        if mask.sum() == 0:
            continue
        m = {"n_samples": int(mask.sum())}
        for k in K:
            m[f"recall@{k}_same_task"] = float(np.any(same_task_hit[mask, :k], axis=1).mean())
            m[f"recall@{k}_same_episode"] = float(np.any(same_ep_hit[mask, :k], axis=1).mean())
            m[f"recall@{k}_temporal"] = float(np.any(temporal_hit[mask, :k], axis=1).mean())
        metrics["per_suite"][suite] = m

    # Random baseline (sanity check)
    # Random recall@k_same_task = (k * (n_per_task - 1)) / (N - 1)
    # We compute it empirically by random shuffle:
    rng = np.random.default_rng(42)
    rand_idx = np.stack([rng.permutation(N)[:topk_max] for _ in range(N)])
    rand_same_task = np.zeros((N, topk_max), dtype=bool)
    for i in range(N):
        for j in range(topk_max):
            rand_same_task[i, j] = (task_ids[rand_idx[i, j]] == task_ids[i])
    metrics["random_baseline"] = {
        f"recall@{k}_same_task": float(np.any(rand_same_task[:, :k], axis=1).mean())
        for k in K
    }

    print("\n=== RESULTS ===")
    print(json.dumps(metrics, indent=2))
    with open(out_dir / "retrieval_results.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # --- Qualitative examples figure ---
    print("\n[retrieval] generating qualitative examples figure...")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_examples = args.n_examples_to_plot
    fig, axes = plt.subplots(n_examples, 4, figsize=(13, 2.2 * n_examples))

    # Pick diverse queries: 1-2 from each suite
    rng = np.random.default_rng(0)
    suites_unique = np.unique(suites_arr)
    queries = []
    for suite in suites_unique:
        candidates = np.where(suites_arr == suite)[0]
        if len(candidates) > 0:
            q = rng.choice(candidates)
            queries.append(q)
    queries = queries[:n_examples]
    if len(queries) < n_examples:
        # pad with random
        queries.extend(rng.choice(N, size=n_examples - len(queries), replace=False))

    for row, q_idx in enumerate(queries):
        # Query
        q_data = dataset[q_idx]
        q_img = (q_data["agent_t"].permute(1, 2, 0).numpy() + 1) / 2  # to [0, 1]
        ax = axes[row, 0] if n_examples > 1 else axes[0]
        ax.imshow(np.clip(q_img, 0, 1))
        ax.set_title(f"QUERY [{meta[q_idx]['suite']}]\nt={meta[q_idx]['t']}\n{meta[q_idx]['instr'][:50]}", fontsize=8)
        ax.axis("off")

        # Top-3 retrievals
        for col in range(1, 4):
            r_idx = topk_idx[q_idx, col - 1]
            r_data = dataset[int(r_idx)]
            r_img = (r_data["agent_t"].permute(1, 2, 0).numpy() + 1) / 2
            ax = axes[row, col] if n_examples > 1 else axes[col]
            ax.imshow(np.clip(r_img, 0, 1))
            same_t = task_ids[r_idx] == task_ids[q_idx]
            same_e = ep_ids[r_idx] == ep_ids[q_idx]
            mark = ("✓task" if same_t else "✗") + ("·ep" if same_e else "")
            ax.set_title(f"top-{col} sim={sim[q_idx, r_idx]:.2f} {mark}\n[{meta[r_idx]['suite']}] t={meta[r_idx]['t']}\n{meta[r_idx]['instr'][:50]}",
                         fontsize=7,
                         color=("green" if same_t else "red"))
            ax.axis("off")

    plt.suptitle(f"Multi-view + Temporal Pretrained proj_head — k-NN Retrieval Examples\n"
                 f"global recall@1 same-task = {metrics['global']['recall@1_same_task']:.2%}, "
                 f"recall@5 = {metrics['global']['recall@5_same_task']:.2%}, "
                 f"random@1 = {metrics['random_baseline']['recall@1_same_task']:.2%}",
                 fontsize=11)
    plt.tight_layout()
    plt.savefig(out_dir / "retrieval_examples.png", dpi=130, bbox_inches="tight")
    print(f"[retrieval] saved figure to {out_dir / 'retrieval_examples.png'}")


if __name__ == "__main__":
    main()
