"""
Phase A demo — "Spirit v1.5 sees a LIBERO scene, predicts an action chunk"

This is the simplest possible 'cross-embodiment reality check': we feed Spirit
a real robot-scene image (LIBERO agentview RGB), and see what 14-DoF ALOHA-style
action chunk it produces. Since Spirit was trained on ARX5/aloha/Franka/UR5
real-world data, and we're feeding it LIBERO (Franka-in-sim), the result
is an interesting "out-of-distribution" demo.

Outputs per instruction:
    <out>/<seed>_<instruction_id>_input.png    agentview frame
    <out>/<seed>_<instruction_id>_chunk.png    60-step action trajectory plot
    <out>/<seed>_<instruction_id>_state.json   raw 14→12 action
    <out>/summary.md                           markdown summary table

Usage (in spirit-sim-v1.0 image):
    python phase_a_libero_scene_demo.py \\
        --spirit_ckpt /workspace/models/Spirit-v1.5-patched \\
        --libero_task_suite libero_spatial \\
        --num_tasks 3 \\
        --num_seeds 2 \\
        --out_dir /workspace/output/phase_a_libero
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from xlerobot_adapter import SpiritLeRobotPolicy, unpad_action_14_to_12  # noqa: E402


def try_get_libero_scene_frame(task_suite: str, task_id: int, seed: int):
    """Use LIBERO's OffScreenRenderEnv to produce a single RGB frame + task string."""
    os.environ.setdefault("MUJOCO_GL", "osmesa")
    try:
        from libero.libero import benchmark, get_libero_path
        from libero.libero.envs import OffScreenRenderEnv
    except ImportError as e:
        print(f"[warn] libero not available: {e}")
        return None, None, None

    bd = benchmark.get_benchmark_dict()
    if task_suite not in bd:
        print(f"[warn] unknown suite {task_suite}")
        return None, None, None
    suite = bd[task_suite]()
    task = suite.get_task(task_id)
    bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)

    env = OffScreenRenderEnv(
        bddl_file_name=bddl, camera_heights=240, camera_widths=320
    )
    env.seed(seed)
    env.reset()
    init_states = suite.get_task_init_states(task_id)
    env.set_init_state(init_states[seed % len(init_states)])

    # Step once to stabilize
    for _ in range(5):
        obs, _, _, _ = env.step(np.zeros(7))

    frame = obs.get("agentview_image")
    if frame is None:
        frame = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
    else:
        # LIBERO's agentview is 240×320 uint8 HWC; often flipped
        frame = np.asarray(frame, dtype=np.uint8)
        if frame.shape[0] == 240 and frame.shape[1] == 320:
            pass
        else:
            # Resize if weird
            from PIL import Image
            frame = np.array(Image.fromarray(frame).resize((320, 240)))

    # LIBERO frames are often upside-down
    frame = np.flipud(frame).copy()

    # Fake 12-dim state (zeros — Spirit's normalizer will clip)
    state = np.zeros(12, dtype=np.float32)

    env.close()
    return frame, state, task.language


def plot_action_chunk(chunk_14: np.ndarray, title: str, out_path: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    DoF_NAMES = [
        "L.waist", "L.shoulder", "L.elbow", "L.forearm_roll",
        "L.wrist_angle", "L.wrist_rotate", "L.gripper",
        "R.waist", "R.shoulder", "R.elbow", "R.forearm_roll",
        "R.wrist_angle", "R.wrist_rotate", "R.gripper",
    ]
    fig, axes = plt.subplots(7, 2, figsize=(12, 14), sharex=True)
    for dof in range(14):
        ax = axes[dof % 7, dof // 7]
        ax.plot(chunk_14[:, dof], linewidth=1.3)
        ax.axhline(0, color="gray", linewidth=0.5, alpha=0.5)
        ax.set_title(f"DoF {dof}: {DoF_NAMES[dof]}", fontsize=9)
        ax.grid(True, alpha=0.2)
        if dof % 7 == 6:
            ax.set_xlabel("step (0–59)")
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def save_input_panel(frame: np.ndarray, instruction: str, out_path: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    ax.imshow(frame)
    ax.set_title(f'Spirit input\n"{instruction}"', fontsize=10)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spirit_ckpt", required=True)
    ap.add_argument("--libero_task_suite", default="libero_spatial")
    ap.add_argument("--num_tasks", type=int, default=3)
    ap.add_argument("--num_seeds", type=int, default=2)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--custom_instructions", nargs="*", default=None,
                    help="If set, use these instructions instead of libero tasks")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("Loading Spirit policy ...")
    t0 = time.time()
    policy = SpiritLeRobotPolicy(
        spirit_ckpt_path=args.spirit_ckpt,
        tile_cam_high=True,  # no wrist cams on LIBERO
        device="cuda",
    )
    policy._chunk_horizon = 1  # force re-infer every call
    print(f"  loaded in {time.time()-t0:.1f}s")

    summary_rows = []

    if args.custom_instructions:
        task_gen = [(0, i, args.custom_instructions[i]) for i in range(len(args.custom_instructions))]
    else:
        task_gen = [(seed, task_id, None)
                    for task_id in range(args.num_tasks)
                    for seed in range(args.num_seeds)]

    for (seed, task_id, custom_instr) in task_gen:
        tag = f"s{seed}_t{task_id}"
        print(f"\n=== {tag} ===")
        if custom_instr:
            # No env, just use dummy image
            frame = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
            state = np.zeros(12, dtype=np.float32)
            instruction = custom_instr
            print(f"  (custom) instruction: {instruction}")
        else:
            frame, state, instruction = try_get_libero_scene_frame(
                args.libero_task_suite, task_id, seed
            )
            if frame is None:
                continue
            print(f"  LIBERO task {task_id}, seed {seed}")
            print(f'  instruction: "{instruction}"')

        obs = {
            "state": state,
            "image_head": frame,
            "task": instruction,
        }

        policy.reset()
        t0 = time.time()
        _ = policy.select_action(obs)
        inf_ms = (time.time() - t0) * 1000
        chunk = policy._cached_chunk[0].detach().float().cpu().numpy()  # (60, 14)

        print(f"  inference: {inf_ms:.0f} ms  | chunk shape: {chunk.shape}")
        print(f"  chunk range: [{chunk.min():.3f}, {chunk.max():.3f}]")

        # Save input image
        inp = out / f"{tag}_input.png"
        save_input_panel(frame, instruction, str(inp))
        print(f"  input -> {inp.name}")

        # Save chunk plot
        plot_path = out / f"{tag}_chunk.png"
        plot_action_chunk(chunk, f"[{tag}] {instruction}  ({inf_ms:.0f} ms)", str(plot_path))
        print(f"  chunk -> {plot_path.name}")

        # Save state JSON (just first step for brevity)
        step0_14 = chunk[0].tolist()
        step0_12 = unpad_action_14_to_12(chunk[0]).tolist()
        state_json = out / f"{tag}_state.json"
        with open(state_json, "w") as f:
            json.dump({
                "tag": tag,
                "instruction": instruction,
                "inference_ms": inf_ms,
                "action_14_step0": step0_14,
                "action_12_step0_for_xlerobot": step0_12,
                "chunk_range": [float(chunk.min()), float(chunk.max())],
                "chunk_shape": list(chunk.shape),
            }, f, indent=2)

        summary_rows.append({
            "tag": tag,
            "instruction": instruction,
            "inference_ms": f"{inf_ms:.0f}",
            "chunk_min": f"{chunk.min():.2f}",
            "chunk_max": f"{chunk.max():.2f}",
            "input": f"{tag}_input.png",
            "chunk": f"{tag}_chunk.png",
        })

    # Write markdown summary
    md = out / "summary.md"
    with open(md, "w") as f:
        f.write("# Phase A — Spirit v1.5 on LIBERO scenes (zero-shot)\n\n")
        f.write("Feeding Spirit the LIBERO agentview image + language instruction, observing\n")
        f.write("what 14-DoF ALOHA-style action chunk it predicts. Spirit was NOT trained\n")
        f.write("on LIBERO — this is cross-embodiment, zero-shot.\n\n")
        f.write("Adapter: `tile_cam_high=True` (copies agentview to both wrist cam slots).\n\n")
        f.write("| tag | instruction | inf (ms) | chunk min | chunk max | input | chunk |\n")
        f.write("|---|---|---:|---:|---:|---|---|\n")
        for r in summary_rows:
            f.write(f"| {r['tag']} | {r['instruction'][:60]} | {r['inference_ms']} | "
                    f"{r['chunk_min']} | {r['chunk_max']} | "
                    f"![]({r['input']}) | ![]({r['chunk']}) |\n")
    print(f"\nSummary: {md}")
    print(f"\n✅ Done. {len(summary_rows)} runs in {out}")


if __name__ == "__main__":
    main()
