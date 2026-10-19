"""Prepare LIBERO demonstrations and read them as Spirit flow-matching samples.

Run with ``python -m spirit_adapter.libero_dataset --help``. The third camera
is rendered from each recorded MuJoCo state; no camera or robot state is
fabricated. Dependencies on LIBERO are needed for preparation only.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from .libero_bridge import CAMERAS, IMAGE_SIZE, controller_goal, encode_targets, render_images, state_from_observation


def reset_demo(env, original):
    """Restore recorded geometry, relocating legacy LIBERO asset paths."""
    from libero.libero import get_libero_path
    from libero.libero.utils.utils import postprocess_model_xml

    xml = postprocess_model_xml(original.attrs["model_file"], {})
    root = ET.fromstring(xml)
    for asset in root.findall("./asset/*"):
        old = asset.get("file")
        if not old or Path(old).exists():
            continue
        if "/assets/" in old:
            relocated = Path(get_libero_path("assets")) / old.rsplit("/assets/", 1)[1]
            if not relocated.is_file():
                raise FileNotFoundError(relocated)
            asset.set("file", str(relocated))
    env.reset()
    env.reset_from_xml_string(ET.tostring(root, encoding="unicode"))
    env.sim.reset()
    return env.set_init_state(original["states"][0])


def prepare_file(source, destination, max_episodes=None):
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    destination = Path(destination)
    if max_episodes is not None and max_episodes < 1:
        raise ValueError("max_episodes must be positive")
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".partial")
    if partial.exists():
        raise FileExistsError(f"Unfinished conversion exists: {partial}")
    with h5py.File(source, "r") as raw, h5py.File(partial, "x") as prepared:
        data = raw["data"]
        relative = Path(data.attrs["bddl_file_name"])
        bddl = Path(get_libero_path("bddl_files")) / relative.parent.name / relative.name
        problem = json.loads(data.attrs["problem_info"])
        source_env = json.loads(data.attrs["env_args"])["env_kwargs"]
        if source_env["robots"] != ["Panda"] or source_env["controller_configs"]["type"] != "OSC_POSE":
            raise ValueError("Only Panda / OSC_POSE demonstrations are supported")
        hz = float(source_env["control_freq"])
        env = OffScreenRenderEnv(
            bddl_file_name=str(bddl), use_camera_obs=False,
            control_freq=hz, camera_heights=IMAGE_SIZE[0], camera_widths=IMAGE_SIZE[1],
        )
        prepared.attrs.update({
            "spirit_libero_schema": 2, "action_hz": hz,
            "source_file": Path(source).name, "instruction": problem["language_instruction"],
            "camera_convention": "upright_rgb", "camera_mapping": json.dumps(CAMERAS),
            "state_frame": "robot_base", "gripper_units": "meters",
            "gripper_target": "commanded_opening_m",
            "action_targets": "osc_controller_goals",
        })
        try:
            source_controller = source_env["controller_configs"]
            controller = env.robots[0].controller
            for field in ("input_min", "input_max", "output_min", "output_max"):
                if not np.allclose(source_controller[field], getattr(controller, field)):
                    raise ValueError(f"Demonstration controller {field} differs from the LIBERO environment")
            if not source_controller.get("control_delta", True) or source_controller.get("impedance_mode", "fixed") != "fixed":
                raise ValueError("Demonstrations must use fixed-impedance delta OSC_POSE")
            if source_controller.get("position_limits") is not None or source_controller.get("orientation_limits") is not None:
                raise ValueError("Demonstrations with additional OSC goal limits are unsupported")
            episodes = sorted(data, key=lambda name: int(name.split("_")[-1]))
            if max_episodes is not None:
                episodes = episodes[:max_episodes]
            for name in episodes:
                original = data[name]
                states = original["states"]
                if len(states) != len(original["actions"]):
                    raise ValueError(f"Observation/action counts differ in {name}")
                if len(states) < 1:
                    continue
                initial_obs = reset_demo(env, original)
                episode = prepared.create_group(name)
                opening = 0.08 if state_from_observation(initial_obs, env)[6] >= 0.04 else 0.0
                positions = episode.create_dataset("state", (len(states), 14), dtype="float32")
                targets = episode.create_dataset("target_pose", (len(states), 14), dtype="float32")
                images = {key: episode.create_dataset(
                    key.rsplit(".", 1)[-1], (len(states), *IMAGE_SIZE, 3), dtype="uint8",
                    chunks=(1, *IMAGE_SIZE, 3), compression="gzip", compression_opts=1,
                ) for key in CAMERAS}
                for index, sim_state in enumerate(states):
                    obs = env.set_init_state(sim_state)
                    positions[index] = state_from_observation(obs, env)
                    command = original["actions"][index]
                    if command[6] != 0:
                        opening = 0.08 if command[6] < 0 else 0.0
                    targets[index] = controller_goal(obs, env, command, opening)
                    for key, rgb in render_images(env).items():
                        images[key][index] = rgb
                print(f"Prepared {name}: {len(states)} frames", flush=True)
        finally:
            env.close()
    partial.replace(destination)
    return destination


class LiberoSpiritDataset(Dataset):
    """Only prepared files are accepted, to keep train/eval conventions explicit."""

    def __init__(self, path, horizon=60, episodes=None):
        if horizon < 1:
            raise ValueError("Action horizon must be positive")
        self.path = str(path)
        self.horizon = horizon
        self.index = []
        with h5py.File(self.path, "r") as source:
            if source.attrs.get("spirit_libero_schema") != 2:
                raise ValueError("Re-run spirit_adapter.libero_dataset to prepare OSC-goal targets (schema 2)")
            if source.attrs.get("camera_convention") != "upright_rgb":
                raise ValueError("Prepared Spirit camera inputs must be upright RGB")
            if episodes is not None and not set(episodes).issubset(source):
                raise ValueError(f"Unknown episode IDs: {set(episodes) - set(source)}")
            self.action_hz = float(source.attrs["action_hz"])
            self.instruction = str(source.attrs["instruction"])
            self.states = {
                key: group["state"][:]
                for key, group in source.items() if episodes is None or key in episodes
            }
            self.target_poses = {key: source[key]["target_pose"][:] for key in self.states}
        for key, states in self.states.items():
            self.index.extend((key, frame) for frame in range(len(states)))
        if not self.index:
            raise ValueError("No training frames with future targets")

    def __len__(self):
        return len(self.index)

    def get_lowdim_item(self, index):
        key, frame = self.index[index]
        states = self.states[key]
        action, mask, padding = encode_targets(states, frame, self.horizon, target_poses=self.target_poses[key])
        return {
            "observation.state": torch.from_numpy(states[frame:frame+1].copy()),
            "action": torch.from_numpy(action), "action_mask": torch.from_numpy(mask),
            "action_is_pad": torch.from_numpy(padding),
        }

    def __getitem__(self, index):
        key, frame = self.index[index]
        result = self.get_lowdim_item(index)
        with h5py.File(self.path, "r") as source:
            for camera in CAMERAS:
                rgb = source[key][camera.rsplit(".", 1)[-1]][frame]
                result[camera] = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
        result.update(task=self.instruction, robot_type="Franka")
        return result

    def normalization_stats(self):
        states, actions = [], []
        for index in range(len(self)):
            sample = self.get_lowdim_item(index)
            states.append(sample["observation.state"][0, :7])
            valid = ~sample["action_is_pad"]
            actions.append(sample["action"][valid, :7])
        result = {}
        for key, values in (("state", torch.stack(states)), ("action", torch.cat(actions))):
            low, high = torch.full((14,), -1.0), torch.ones(14)
            low[:7], high[:7] = values.amin(0), values.amax(0)
            constant = (high - low) < 1e-6
            low[constant] -= 1.0
            high[constant] += 1.0
            result[key + "_min"], result[key + "_max"] = low, high
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="One official LIBERO task HDF5")
    parser.add_argument("--output", required=True, help="Prepared Spirit HDF5")
    parser.add_argument("--max_episodes", type=int)
    args = parser.parse_args()
    print(prepare_file(args.source, args.output, args.max_episodes))


if __name__ == "__main__":
    main()
