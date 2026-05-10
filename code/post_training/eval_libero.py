"""LIBERO 4-suite eval harness for v1.5 paper.

Computes the success rate per (suite, task, seed) for any VLABase
adapter. This is the **single source of truth** for paper §4.2 main
results table:

    | Model              | Spatial | Object | Goal | Long10 | Avg |
    | OpenVLA SFT (ref)  | 78.0    | 60.0   | 77.0 | 53.0   | 67.0 |
    | OpenVLA + DPO      | ?       | ?      | ?    | ?      | ?   |
    | OpenVLA + GRPO     | ?       | ?      | ?    | ?      | ?   |
    | Spirit SFT         | -       | -      | -    | -      | -   |
    | Spirit + DPO       | ?       | ?      | ?    | ?      | ?   |
    | Spirit + GRPO      | ?       | ?      | ?    | ?      | ?   |
    | π0.5 SFT (ref)     | 98.8    | 98.2   | 98.0 | 92.4   | 96.85
    | π0.5 + DPO         | ?       | ?      | ?    | ?      | ?   |
    | π0.5 + GRPO        | ?       | ?      | ?    | ?      | ?   |

Protocol (matches openpi LIBERO eval for direct comparability):
  - 10 tasks per suite
  - N_TRIALS_PER_TASK trials per task (default 50, paper-grade)
  - 3 seeds per (task, trial) — total = N × 10 × 3 per suite
  - Episode length cap: per-suite max_steps (220/280/300/520)
  - Success determined by env.check_success() at any point during episode

Output:
  - per-trial JSONL log
  - aggregated per-suite / per-task / overall summary JSON
  - optional rollout MP4 videos (for paper figures + spot-checking)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

# Settings shared with rollout.py
_MAX_STEPS = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_90": 400,
}

ALL_4_SUITES = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]


def patch_torch_load_for_libero() -> None:
    """LIBERO ships init_states as torch pickles incompatible with
    torch>=2.6 default weights_only=True."""
    import torch as _t
    if getattr(_t.load, "_libero_patched", False):
        return
    orig = _t.load
    def patched(f, *a, **kw):
        if "weights_only" not in kw:
            kw["weights_only"] = False
        return orig(f, *a, **kw)
    patched._libero_patched = True
    _t.load = patched


def get_libero_env(task, resolution: int = 256):
    """Match OpenVLA official eval: render at 256, resize to 224 with
    tf.image.lanczos3 + JPEG round-trip via _resize_image_rlds()."""
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    bddl = os.path.join(get_libero_path("bddl_files"),
                        task.problem_folder, task.bddl_file)
    env = OffScreenRenderEnv(
        bddl_file_name=bddl,
        camera_heights=resolution, camera_widths=resolution,
    )
    return env, task.language


def _resize_image_rlds(img_uint8: np.ndarray, target_size: int) -> np.ndarray:
    """RLDS-style preprocessing approximating OpenVLA's TF pipeline.

    See rollout.py:_resize_image_rlds for the full incident analysis —
    we use PIL JPEG round-trip + cv2.INTER_LANCZOS4 instead of tf
    because TF + libero env crash together in our docker setup.
    """
    import io
    import cv2
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(img_uint8).save(buf, format="JPEG", quality=95)
    img_after_jpeg = np.array(Image.open(buf))
    return cv2.resize(
        img_after_jpeg, (target_size, target_size),
        interpolation=cv2.INTER_LANCZOS4,
    )


def get_libero_dummy_action() -> np.ndarray:
    return np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0], dtype=np.float32)


def run_one_trial(
    adapter,
    env,
    instruction: str,
    initial_state,
    max_steps: int,
    num_steps_wait: int = 10,
    capture_video: bool = False,
) -> dict:
    """Run one trial, return outcome dict."""
    env.reset()
    obs = env.set_init_state(initial_state)

    # Settling phase
    for _ in range(num_steps_wait):
        obs, _, _, _ = env.step(get_libero_dummy_action().tolist())

    success = False
    n_steps = 0
    frames: list[np.ndarray] = []
    t0 = time.time()

    for t in range(max_steps):
        img_arr = obs.get("agentview_image")
        if img_arr is None:
            for k in ("image", "rgb_static"):
                if k in obs:
                    img_arr = obs[k]
                    break
        if img_arr is None:
            return {"success": False, "n_steps": 0, "elapsed_s": 0,
                    "error": "no image in obs"}

        # LIBERO renders agentview rotated 180° relative to OpenVLA's
        # training distribution (per openvla/experiments/.../libero_utils.py
        # get_libero_image: `img = img[::-1, ::-1]`).
        img = img_arr[::-1, ::-1]
        if img.dtype != np.uint8:
            img = (img * 255).astype(np.uint8)
        if capture_video:
            frames.append(img.copy())
        # OpenVLA RLDS resize (cv2 lanczos4 substitute for tf.lanczos3).
        if img.shape[0] != 224:
            img = _resize_image_rlds(img, 224)
        img_uint8 = img.copy()
        img_t = torch.from_numpy(img_uint8).permute(2, 0, 1).float() / 255.0

        # Pass BOTH float tensor (back-compat) and raw uint8 ndarray
        # (preferred by OpenVLA adapter — avoids float→uint8 round-trip
        # quantisation that OOD's the visual encoder).
        batch = {
            "instruction": [instruction],
            "image": img_t.unsqueeze(0),
            "image_uint8": [img_uint8],
        }
        with torch.no_grad():
            chunk = adapter.select_action(batch)
        action = chunk.squeeze(0).squeeze(0).cpu().numpy()

        # Gripper handling: OpenVLA → invert sign for LIBERO (see rollout.py).
        # Other bases: simple binarisation.
        if hasattr(adapter, "cfg") and getattr(adapter.cfg, "base", "") == "openvla":
            action[-1] = 1.0 if action[-1] > 0.5 else -1.0
            action[-1] = -action[-1]
        else:
            action[-1] = 1.0 if action[-1] > 0.5 else -1.0

        try:
            obs, _, done, _ = env.step(action.tolist())
        except Exception as e:
            return {"success": False, "n_steps": n_steps, "elapsed_s": time.time()-t0,
                    "error": f"env.step: {e}"}
        n_steps = t + 1

        if done or (hasattr(env, "check_success") and env.check_success()):
            success = True
            break

    return {
        "success": success,
        "n_steps": n_steps,
        "elapsed_s": time.time() - t0,
        "frames": frames if capture_video else None,
    }


def evaluate_suite(
    adapter,
    suite_name: str,
    n_tasks: int,
    n_trials_per_task: int,
    seeds: list[int],
    out_dir: Path,
    capture_videos: int = 0,
    print_progress: bool = True,
) -> dict:
    """Eval one LIBERO suite. Returns nested dict: {task_id: per-trial list}."""
    from libero.libero import benchmark
    bench = benchmark.get_benchmark_dict()
    task_suite = bench[suite_name]()
    n_tasks = min(n_tasks, task_suite.n_tasks)
    max_steps = _MAX_STEPS.get(suite_name, 220)

    suite_results = {}
    suite_jsonl = out_dir / f"{suite_name}.jsonl"
    log_f = open(suite_jsonl, "a")

    for task_id in range(n_tasks):
        task = task_suite.get_task(task_id)
        env, task_description = get_libero_env(task, resolution=224)
        init_states = task_suite.get_task_init_states(task_id)

        if print_progress:
            print(f"  [{suite_name}] task {task_id+1}/{n_tasks}: {task_description[:50]}")

        task_trials = []
        for trial_idx in range(n_trials_per_task):
            init_idx = trial_idx % len(init_states)
            for seed_idx, seed in enumerate(seeds):
                np.random.seed(seed)
                torch.manual_seed(seed)
                env.seed(seed)

                want_video = capture_videos > 0
                outcome = run_one_trial(
                    adapter, env, task_description,
                    init_states[init_idx], max_steps,
                    capture_video=want_video,
                )
                if want_video and outcome.get("frames"):
                    cap_path = out_dir / "videos" / suite_name / \
                               f"task{task_id:02d}_trial{trial_idx:02d}_seed{seed}.mp4"
                    cap_path.parent.mkdir(parents=True, exist_ok=True)
                    _save_video(outcome["frames"], cap_path)
                    capture_videos -= 1

                entry = {
                    "suite": suite_name,
                    "task_id": task_id,
                    "task": task_description,
                    "trial": trial_idx,
                    "seed": seed,
                    "init_idx": init_idx,
                    "success": outcome["success"],
                    "n_steps": outcome["n_steps"],
                    "elapsed_s": outcome.get("elapsed_s", 0),
                }
                if "error" in outcome:
                    entry["error"] = outcome["error"]
                log_f.write(json.dumps(entry) + "\n")
                log_f.flush()
                task_trials.append(entry)

                if print_progress and (trial_idx * len(seeds) + seed_idx + 1) % 10 == 0:
                    n_done = trial_idx * len(seeds) + seed_idx + 1
                    n_succ = sum(1 for t in task_trials if t["success"])
                    print(f"    [{n_done}/{n_trials_per_task * len(seeds)}] "
                          f"success {n_succ}/{n_done} ({100*n_succ/n_done:.0f}%)")

        suite_results[task_id] = task_trials
        env.close()

    log_f.close()
    return suite_results


def _save_video(frames: list[np.ndarray], path: Path):
    """Save a list of (H, W, 3) uint8 frames as mp4."""
    try:
        import imageio
        imageio.mimwrite(str(path), frames, fps=20, codec="libx264")
    except Exception as e:
        print(f"[warn] video save failed: {e}")


def aggregate(per_suite_results: dict) -> dict:
    """Build the paper §4.2 summary table."""
    summary = {"per_suite": {}, "overall_avg": None}
    suite_avgs = []
    for suite, suite_data in per_suite_results.items():
        all_trials = []
        per_task = {}
        for task_id, trials in suite_data.items():
            n = len(trials)
            n_succ = sum(1 for t in trials if t["success"])
            per_task[str(task_id)] = {
                "task": trials[0]["task"] if trials else "",
                "success": n_succ,
                "trials": n,
                "rate": (n_succ / n) if n else 0.0,
            }
            all_trials.extend(trials)
        n_total = len(all_trials)
        n_succ_total = sum(1 for t in all_trials if t["success"])
        suite_rate = (n_succ_total / n_total) if n_total else 0.0
        summary["per_suite"][suite] = {
            "rate": suite_rate,
            "success": n_succ_total,
            "trials": n_total,
            "per_task": per_task,
        }
        suite_avgs.append(suite_rate)
    summary["overall_avg"] = float(np.mean(suite_avgs)) if suite_avgs else 0.0
    return summary


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base", choices=["openvla", "spirit", "pi05"], required=True)
    p.add_argument("--base_ckpt", required=True)
    p.add_argument("--lora_ckpt", default="",
                   help="Path to LoRA-trained adapter checkpoint (optional)")
    p.add_argument("--suites", nargs="+", default=ALL_4_SUITES,
                   help="LIBERO suites to evaluate")
    p.add_argument("--n_tasks", type=int, default=10)
    p.add_argument("--n_trials_per_task", type=int, default=50)
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 1337, 2026])
    p.add_argument("--output_dir", required=True)
    p.add_argument("--capture_videos", type=int, default=0,
                   help="Save MP4s for the first N successful trials per suite")
    return p.parse_args()


def main():
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    patch_torch_load_for_libero()

    # Build adapter
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from post_training import PostTrainConfig
    from post_training.train_dpo import build_adapter

    cfg = PostTrainConfig(
        name=f"{args.base}_eval",
        base=args.base,
        algorithm="sft",
        base_ckpt_path=args.base_ckpt,
        libero_suite="spatial",
        use_lora=bool(args.lora_ckpt),
    )
    adapter = build_adapter(cfg)

    if args.lora_ckpt:
        adapter.load(args.lora_ckpt)
        print(f"[eval] loaded LoRA: {args.lora_ckpt}")

    # Evaluate each suite
    per_suite = {}
    t_start = time.time()
    for suite in args.suites:
        print(f"\n=== suite: {suite} ===")
        results = evaluate_suite(
            adapter, suite, args.n_tasks, args.n_trials_per_task,
            args.seeds, out, capture_videos=args.capture_videos,
        )
        per_suite[suite] = results

    # Aggregate + print
    summary = aggregate(per_suite)
    summary["meta"] = {
        "base": args.base,
        "base_ckpt": args.base_ckpt,
        "lora_ckpt": args.lora_ckpt,
        "suites": args.suites,
        "n_trials_per_task": args.n_trials_per_task,
        "seeds": args.seeds,
        "elapsed_s": time.time() - t_start,
    }

    summary_path = out / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n=== Summary (saved → {summary_path}) ===")
    for suite, data in summary["per_suite"].items():
        print(f"  {suite:20s}  {data['success']:4d}/{data['trials']:4d}  "
              f"= {100*data['rate']:5.1f}%")
    print(f"  {'-'*45}")
    print(f"  {'overall avg':20s}  {100*summary['overall_avg']:25.1f}%")


if __name__ == "__main__":
    main()
