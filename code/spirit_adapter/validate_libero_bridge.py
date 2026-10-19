"""Replay demonstrations through Spirit target-pose conversion, without a VLA.

This checks the adapter/control interface. Its success count is a controller
round-trip result and must not be reported as Spirit model success rate.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
from pathlib import Path

import h5py
import numpy as np
from scipy.spatial.transform import Rotation

from .libero_bridge import eef_pose_in_base, encode_targets, to_osc_action
from .libero_dataset import reset_demo


def validate(source, episodes=2, chunk_steps=10, prepared=None):
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    results = []
    with h5py.File(source, "r") as file, (h5py.File(prepared, "r") if prepared else nullcontext()) as labels:
        if labels is not None and labels.attrs.get("spirit_libero_schema") != 2:
            raise ValueError("Prepared controller-goal targets (schema 2) are required")
        data = file["data"]
        relative = Path(data.attrs["bddl_file_name"])
        bddl = Path(get_libero_path("bddl_files")) / relative.parent.name / relative.name
        hz = json.loads(data.attrs["env_args"])["env_kwargs"]["control_freq"]
        env = OffScreenRenderEnv(bddl_file_name=str(bddl), control_freq=hz, use_camera_obs=False)
        try:
            for name in sorted(data, key=lambda key: int(key.split("_")[-1]))[:episodes]:
                demo = data[name]
                actions = demo["actions"][:]
                if not np.all(np.isin(actions[:, 6], [-1, 1])):
                    raise ValueError("This round-trip check requires binary demonstration gripper commands")
                obs = reset_demo(env, demo)
                targets, direct_states, direct_positions = [], [], []
                error = 0.0
                direct_success = False
                for step, command in enumerate(actions):
                    if step % chunk_steps == 0:
                        origin = eef_pose_in_base(obs, env)
                    pos, rot, base_rot, *_ = eef_pose_in_base(obs, env)
                    physical = env.robots[0].controller.scale_action(command[:6])
                    target = np.zeros(7, dtype=np.float64)
                    target[:3] = pos + base_rot.inv().apply(physical[:3]) - origin[0]
                    delta_rot_base = base_rot.inv() * Rotation.from_rotvec(physical[3:]) * base_rot
                    target[3:6] = (delta_rot_base * rot * origin[1].inv()).as_rotvec()
                    target[6] = 0.08 if command[6] < 0 else 0.0
                    recovered = to_osc_action(target, origin, obs, env)
                    error = max(error, float(np.max(np.abs(recovered[:6] - np.clip(command[:6], -1, 1)))))
                    targets.append(target)
                    obs, _, _, _ = env.step(command)
                    direct_states.append(env.get_sim_state())
                    direct_positions.append(np.asarray(obs["robot0_eef_pos"]).copy())
                    direct_success |= bool(env.check_success())

                obs = reset_demo(env, demo)
                bridge_success = False
                state_error = 0.0
                position_error = 0.0
                for step, target in enumerate(targets):
                    if step % chunk_steps == 0:
                        origin = eef_pose_in_base(obs, env)
                        if labels is not None:
                            encoded, _, _ = encode_targets(
                                labels[name]["state"][:], step, chunk_steps,
                                target_poses=labels[name]["target_pose"][:],
                            )
                    if labels is not None:
                        target = encoded[step % chunk_steps]
                    command = to_osc_action(target, origin, obs, env)
                    obs, _, _, _ = env.step(command)
                    state_error = max(state_error, float(np.max(np.abs(env.get_sim_state() - direct_states[step]))))
                    position_error = max(position_error, float(np.linalg.norm(obs["robot0_eef_pos"] - direct_positions[step])))
                    bridge_success |= bool(env.check_success())
                results.append({
                    "episode": name, "steps": len(actions), "direct_success": direct_success,
                    "bridge_success": bridge_success, "arm_command_max_abs_error": error,
                    "sim_state_max_abs_difference": state_error,
                    "eef_position_max_difference_m": position_error,
                })
                print(results[-1], flush=True)
        finally:
            env.close()
    return {"validation_type": "prepared_target_replay" if prepared else "demonstration_controller_round_trip", "chunk_steps": chunk_steps,
            "source": Path(source).name, "episodes": results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--chunk_steps", type=int, default=10)
    parser.add_argument("--output", required=True)
    parser.add_argument("--prepared", help="Replay targets from the actual converted training file")
    args = parser.parse_args()
    result = validate(args.source, args.episodes, args.chunk_steps, args.prepared)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))
    if not result["episodes"] or any(not item["direct_success"] or not item["bridge_success"]
                                     or item["arm_command_max_abs_error"] > 1e-5 for item in result["episodes"]):
        raise SystemExit("Demonstration controller round trip failed; inspect the saved report")


if __name__ == "__main__":
    main()
