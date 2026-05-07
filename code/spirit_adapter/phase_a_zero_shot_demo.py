"""
Phase A — zero-shot Spirit v1.5 on XLeRobot Mujoco sim.

This is the 'failure video' script: expected to NOT work well (12-DoF hardware
vs Spirit's 14-DoF training), but it produces the first interpretable motion
video and anchors blog #2's narrative ("here's why we need to fine-tune").

Usage (on dev machine inside spirit-v1.0-cu128-py310 image):

    python phase_a_zero_shot_demo.py \
        --spirit_ckpt /workspace/models/Spirit-v1.5 \
        --qwen_ckpt /workspace/models/Qwen3-VL-4B-Instruct \
        --instruction "pick up the red cube" \
        --num_steps 300 \
        --out_video /workspace/output/phase_a_zero_shot.mp4

Produces:
- phase_a_zero_shot.mp4: side-by-side (head cam | rendered scene) video
- phase_a_metrics.json: step-level state/action logs for analysis
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

from xlerobot_adapter import (  # noqa: E402
    XLeRobotSpiritAdapter,
    SpiritLeRobotPolicy,
    XLE_ACTION_DIM,
)


def parse_args():
    ap = argparse.ArgumentParser(description="Phase A Spirit zero-shot on XLeRobot sim")
    ap.add_argument("--spirit_ckpt", required=True, type=str,
                    help="Path to Spirit v1.5 ckpt directory")
    ap.add_argument("--qwen_ckpt", required=False, type=str, default=None,
                    help="(Optional) local Qwen3-VL backbone; HF_HOME usually covers this")
    ap.add_argument("--instruction", type=str, default="pick up the red cube and place it on the blue plate",
                    help="Language instruction to condition the policy on")
    ap.add_argument("--num_steps", type=int, default=300,
                    help="Number of sim steps to roll out")
    ap.add_argument("--control_hz", type=float, default=10.0,
                    help="Control frequency. Spirit chunks are 60 steps long; chunk_horizon in the"
                         " SpiritLeRobotPolicy will decide how many of those to use before re-inferring.")
    ap.add_argument("--out_video", type=str, default="./phase_a_zero_shot.mp4")
    ap.add_argument("--env_name", type=str, default="PushCube-v1",
                    help="Maniskill env to rollout in; PushCube-v1 is the simplest.")
    ap.add_argument("--robot_uids", type=str, default="xlerobot_single")
    ap.add_argument("--tile_cam_high", action="store_true", default=True,
                    help="Phase A only: copy head cam to both wrist cam slots.")
    ap.add_argument("--dry_run", action="store_true",
                    help="Skip sim setup; test adapter + policy only on dummy obs.")
    return ap.parse_args()


def make_env(env_name: str, robot_uids: str):
    """Create Maniskill env with XLeRobot agent. Gracefully exit if not installed."""
    try:
        import gymnasium as gym
        from mani_skill.envs.sapien_env import BaseEnv  # noqa
        # Import XLeRobot agent registration (side-effect)
        from agents.xlerobot import xlerobot  # noqa: F401
    except ImportError as e:
        print(f"[warn] Maniskill+XLeRobot agent not installed: {e}")
        print("      Run 'pip install mani_skill' and set PYTHONPATH=/workspace/XLeRobot/simulation/Maniskill")
        return None

    env_kwargs = dict(
        obs_mode="rgb+state",
        control_mode="pd_joint_delta_pos",
        render_mode="rgb_array",
        robot_uids=robot_uids,
        num_envs=1,
        sim_backend="auto",
    )
    env = gym.make(env_name, **env_kwargs)
    return env


def extract_xle_obs_from_maniskill(obs_dict) -> dict:
    """
    Pull out XLeRobot-compatible observation from Maniskill env step.

    Expected keys (varies by env version):
    - obs_dict['agent']['qpos']  (joint positions, including arms)
    - obs_dict['sensor_data']['head_camera']['rgb']  (or similar)
    - obs_dict['sensor_data']['wrist_camera_left']['rgb']  (if available)

    We keep this flexible because XLeRobot Maniskill sim is under
    development upstream.
    """
    # Fallback: if env returns tensor dict, convert to numpy
    def _to_np(x):
        if hasattr(x, "cpu"):
            return x.cpu().numpy()
        return np.asarray(x)

    # State: try 'agent.qpos' -> arm joints only
    try:
        qpos = _to_np(obs_dict["agent"]["qpos"])
        if qpos.ndim == 2:
            qpos = qpos[0]
        # XLeRobot qpos layout includes base + 2 arms; take last 12 (2 arms × 6)
        state = qpos[-XLE_ACTION_DIM:].astype(np.float32)
    except (KeyError, IndexError) as e:
        print(f"[warn] failed to extract state: {e}, returning zeros")
        state = np.zeros(XLE_ACTION_DIM, dtype=np.float32)

    # Head cam: try several keys
    head = None
    sensor = obs_dict.get("sensor_data", {})
    for key in ["head_camera", "base_camera", "front_camera", "camera"]:
        if key in sensor:
            cam = sensor[key]
            if isinstance(cam, dict) and "rgb" in cam:
                head = _to_np(cam["rgb"])
                if head.ndim == 4:
                    head = head[0]
                if head.dtype != np.uint8:
                    head = (head * 255).astype(np.uint8) if head.max() <= 1.0 else head.astype(np.uint8)
                break
    if head is None:
        print("[warn] no head camera found, using noise")
        head = np.random.randint(0, 255, size=(240, 320, 3), dtype=np.uint8)

    return {
        "state": state,
        "image_head": head,
    }


def main():
    args = parse_args()

    print("=" * 60)
    print("Phase A: Spirit v1.5 zero-shot on XLeRobot")
    print("=" * 60)
    print(f"  Spirit ckpt: {args.spirit_ckpt}")
    print(f"  Instruction: {args.instruction}")
    print(f"  Num steps:   {args.num_steps}")
    print(f"  Out video:   {args.out_video}")
    print(f"  Env:         {args.env_name} ({args.robot_uids})")
    print(f"  Dry run:     {args.dry_run}")
    print("=" * 60)

    # 1. Load policy
    print("\n[1/3] Loading Spirit v1.5 policy...")
    t0 = time.time()
    policy = SpiritLeRobotPolicy(
        spirit_ckpt_path=args.spirit_ckpt,
        tile_cam_high=args.tile_cam_high,
        device="cuda",
    )
    print(f"   loaded in {time.time()-t0:.1f}s")

    # 2. Create env (or fake obs for dry run)
    print("\n[2/3] Creating env...")
    if args.dry_run:
        obs_xle = {
            "state": np.zeros(XLE_ACTION_DIM, dtype=np.float32),
            "image_head": np.random.randint(0, 255, size=(480, 640, 3), dtype=np.uint8),
            "task": args.instruction,
        }
        env = None
    else:
        env = make_env(args.env_name, args.robot_uids)
        if env is None:
            print("[error] failed to create env; falling back to dry-run")
            obs_xle = {
                "state": np.zeros(XLE_ACTION_DIM, dtype=np.float32),
                "image_head": np.random.randint(0, 255, size=(480, 640, 3), dtype=np.uint8),
                "task": args.instruction,
            }

    # 3. Rollout loop
    print("\n[3/3] Rolling out...")
    frames = []
    log = {"step": [], "state": [], "action": [], "inference_ms": []}

    try:
        if env is not None:
            obs_raw, _ = env.reset(seed=42)
            obs_xle = extract_xle_obs_from_maniskill(obs_raw)
            obs_xle["task"] = args.instruction

        for step in range(args.num_steps):
            t_inf = time.time()
            action_12 = policy.select_action(obs_xle)
            inference_ms = (time.time() - t_inf) * 1000

            log["step"].append(step)
            log["state"].append(obs_xle["state"].tolist())
            log["action"].append(action_12.tolist())
            log["inference_ms"].append(inference_ms)

            if env is not None:
                # Apply action to sim. Maniskill may expect different format; adapt here.
                obs_raw, reward, terminated, truncated, info = env.step(action_12)
                obs_xle = extract_xle_obs_from_maniskill(obs_raw)
                obs_xle["task"] = args.instruction
                # Render frame
                frame = env.render()
                if frame is not None:
                    frame = frame.cpu().numpy() if hasattr(frame, "cpu") else np.asarray(frame)
                    if frame.ndim == 4:
                        frame = frame[0]
                    frames.append(frame)
                if terminated or truncated:
                    print(f"   early termination at step {step}")
                    break
            else:
                # dry-run: just keep going
                if step % 50 == 0:
                    print(f"   step {step}: action mean={action_12.mean():.3f}, inference {inference_ms:.0f} ms")

        # Save video
        if frames:
            try:
                import imageio
                imageio.mimsave(args.out_video, frames, fps=int(args.control_hz))
                print(f"\n✅ Saved video: {args.out_video} ({len(frames)} frames)")
            except ImportError:
                print("[warn] imageio not installed; video not saved")

        # Save metrics
        metrics_path = Path(args.out_video).with_suffix(".json")
        with open(metrics_path, "w") as f:
            json.dump(log, f)
        print(f"✅ Saved log: {metrics_path}")

    finally:
        if env is not None:
            env.close()

    print("\nDone. Expect: Spirit outputs small-magnitude actions that partly")
    print("move the arms but don't accomplish the task — this is the anchor")
    print("for blog #2 'cross-embodiment zero-shot doesn't work; fine-tune it'.")


if __name__ == "__main__":
    main()
