"""
Phase A rollout — Spirit v1.5 drives the XLeRobot MJCF model in pure Mujoco.

Why Mujoco instead of Maniskill: SAPIEN 3.x requires Vulkan, which doesn't work
in our k8s pod. Mujoco + osmesa works fine (already used by LIBERO).

This script loads XLeRobot's own MJCF (code/XLeRobot/simulation/mujoco/xlerobot.xml),
renders frames via osmesa, feeds them to Spirit, unwraps 14→12 actions, applies
to the first 12 actuators, and saves an MP4 of the resulting motion.

Because Spirit was trained on ALOHA/ARX5/UR5/Franka with wrist cameras, and
XLeRobot has no wrist cameras and different kinematics, this is a *cross-
embodiment zero-shot* demo — expected to produce interesting-but-not-successful
motion. That's fine: the video itself is the blog #2 narrative anchor.

Usage:
    export MUJOCO_GL=osmesa
    python phase_a_mujoco_rollout.py \\
        --spirit_ckpt /workspace/models/Spirit-v1.5-patched \\
        --mjcf /workspace/XLeRobot/simulation/mujoco/xlerobot.xml \\
        --instruction "pick up the red cube and put it on the blue plate" \\
        --num_steps 180 \\
        --out_video /workspace/output/phase_a_mujoco.mp4
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

# Ensure osmesa early
os.environ.setdefault("MUJOCO_GL", "osmesa")
os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from xlerobot_adapter import SpiritLeRobotPolicy  # noqa: E402


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spirit_ckpt", required=True)
    ap.add_argument("--mjcf", required=True, help="Path to xlerobot.xml")
    ap.add_argument("--instruction", default="pick up the red cube and place it on the blue plate")
    ap.add_argument("--num_steps", type=int, default=180, help="mujoco sim steps (1 step ~ 2ms)")
    ap.add_argument("--control_every", type=int, default=20,
                    help="ask Spirit for new action every N sim steps (20×2ms = 40ms)")
    ap.add_argument("--out_video", default="./phase_a_mujoco.mp4")
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--render_w", type=int, default=320)
    ap.add_argument("--render_h", type=int, default=240)
    ap.add_argument("--action_scale", type=float, default=0.05,
                    help="multiply Spirit's normalized action by this before applying to ctrl")
    return ap.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("Phase A — Spirit v1.5 × XLeRobot Mujoco (offscreen)")
    print("=" * 60)
    print(f"  Spirit ckpt:   {args.spirit_ckpt}")
    print(f"  MJCF:          {args.mjcf}")
    print(f"  Instruction:   {args.instruction}")
    print(f"  Sim steps:     {args.num_steps}")
    print(f"  Control every: {args.control_every} sim steps")
    print(f"  Out video:     {args.out_video}")

    import mujoco
    import imageio

    # --- Load Mujoco model + set up offscreen renderer ---
    print("\n[1/4] Loading Mujoco model...")
    t0 = time.time()
    model = mujoco.MjModel.from_xml_path(args.mjcf)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=args.render_h, width=args.render_w)
    mujoco.mj_forward(model, data)
    print(f"  loaded model in {time.time()-t0:.1f}s")
    print(f"  nq={model.nq}, nv={model.nv}, nu={model.nu}")
    print(f"  actuator count: {model.nu}")
    print(f"  joint names (first 10): {[mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(min(10, model.njnt))]}")

    # --- Load Spirit ---
    print(f"\n[2/4] Loading Spirit policy...")
    t0 = time.time()
    policy = SpiritLeRobotPolicy(
        spirit_ckpt_path=args.spirit_ckpt,
        tile_cam_high=True,
        device="cuda",
    )
    policy._chunk_horizon = args.control_every
    print(f"  loaded in {time.time()-t0:.1f}s")

    # --- Find camera (if scene has one) or use free camera ---
    cam_id = -1
    for i in range(model.ncam):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, i)
        if name is None:
            continue
        print(f"  camera[{i}] = {name}")
        if cam_id == -1:
            cam_id = i  # take first camera by default

    camera = cam_id if cam_id >= 0 else None
    if camera is None:
        print("  no camera defined in MJCF — using default free camera view")

    # --- Rollout loop ---
    print(f"\n[3/4] Running {args.num_steps} sim steps...")
    frames = []
    infer_times = []
    state_log = []
    action_log = []
    inferences = 0
    t0 = time.time()

    current_action_12 = np.zeros(12, dtype=np.float32)

    for step in range(args.num_steps):
        # --- Render current frame ---
        renderer.update_scene(data, camera=camera if camera is not None else -1)
        frame = renderer.render()  # (H, W, 3) uint8

        # --- Every control_every steps, ask Spirit for new action ---
        if step % args.control_every == 0:
            # Build obs: state = first 12 qpos (approximately the arm joints)
            state_12 = data.qpos[:12].astype(np.float32) if model.nq >= 12 else np.zeros(12, dtype=np.float32)
            obs = {
                "state": state_12,
                "image_head": frame,
                "task": args.instruction,
            }
            policy.reset()  # force fresh inference
            t_inf = time.time()
            current_action_12 = policy.select_action(obs)  # (12,) float64
            infer_times.append((time.time() - t_inf) * 1000)
            inferences += 1
            state_log.append(state_12.tolist())
            action_log.append(current_action_12.tolist())

        # --- Apply action to first min(12, nu) actuators ---
        n_apply = min(12, model.nu)
        data.ctrl[:n_apply] = np.clip(
            current_action_12[:n_apply] * args.action_scale, -1.0, 1.0
        )

        # Step sim
        mujoco.mj_step(model, data)

        # Save frame
        frames.append(frame)

    wall = time.time() - t0
    print(f"  rollout done in {wall:.1f}s  ({args.num_steps / wall:.1f} sim steps/s)")
    print(f"  total inferences: {inferences}")
    if infer_times:
        print(f"  inference mean:  {np.mean(infer_times):.0f} ms")
        print(f"  inference min:   {np.min(infer_times):.0f} ms")
        print(f"  inference max:   {np.max(infer_times):.0f} ms")

    # --- Save video ---
    print(f"\n[4/4] Saving video...")
    Path(args.out_video).parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(args.out_video, frames, fps=args.fps, codec="libx264", quality=8,
                     macro_block_size=1)
    print(f"  saved {args.out_video}  ({len(frames)} frames @ {args.fps} fps)")

    # --- Save log JSON ---
    import json
    log_path = str(Path(args.out_video).with_suffix(".json"))
    with open(log_path, "w") as f:
        json.dump({
            "instruction": args.instruction,
            "num_steps": args.num_steps,
            "control_every": args.control_every,
            "inferences": inferences,
            "inference_times_ms": infer_times,
            "inference_stats_ms": {
                "mean": float(np.mean(infer_times)) if infer_times else 0,
                "min": float(np.min(infer_times)) if infer_times else 0,
                "max": float(np.max(infer_times)) if infer_times else 0,
            },
            "sim_steps_per_sec": args.num_steps / wall,
            "first_action_12": action_log[0] if action_log else None,
            "final_state_12": state_log[-1] if state_log else None,
        }, f, indent=2)
    print(f"  saved {log_path}")

    print(f"\n✅ Done.  Inspect: {args.out_video}")


if __name__ == "__main__":
    main()
