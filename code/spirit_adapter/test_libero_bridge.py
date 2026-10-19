"""Small contract regressions; no checkpoints, downloads, or GPU needed."""
from types import SimpleNamespace
from pathlib import Path
import tempfile
import unittest

import numpy as np
from scipy.spatial.transform import Rotation
import torch
from torch import nn

from post_training.adapters.spirit import SpiritAdapter
from .libero_bridge import CAMERAS, eef_pose_in_base, encode_targets, render_images, resample_chunk, to_osc_action


class BridgeTests(unittest.TestCase):
    def test_base_frame_rotation_and_live_pose(self):
        base = Rotation.from_euler("xyz", [0.2, -0.3, 0.8])
        controller = SimpleNamespace(
            control_dim=6, use_delta=True, impedance_mode="fixed",
            input_min=-np.ones(6), input_max=np.ones(6),
            output_min=-np.array([0.03]*3 + [0.2]*3),
            output_max=np.array([0.03]*3 + [0.2]*3),
        )
        robot = SimpleNamespace(base_pos=np.array([0.2, 0.3, 0.4]), base_ori=base.as_quat(), controller=controller)
        env = SimpleNamespace(robots=[robot])
        rot = Rotation.from_euler("xyz", [0.1, -0.2, 0.3])
        obs = {"robot0_eef_pos": robot.base_pos + base.apply([0.4, 0.1, 0.3]),
               "robot0_eef_quat": (base * rot).as_quat(), "robot0_gripper_qpos": [0.04, -0.04]}
        origin = eef_pose_in_base(obs, env)
        np.testing.assert_allclose(origin[0], [0.4, 0.1, 0.3])
        target = np.array([0.01, -0.01, 0.02, 0.01, 0.02, -0.01, 0.08])
        # The robot has moved since the observation used for the chunk.
        obs["robot0_eef_pos"] += base.apply([0.005, 0, -0.005])
        actual = to_osc_action(target, origin, obs, env)
        expected_xyz = base.apply([0.005, -0.01, 0.025]) / 0.03
        expected_rot = (base * Rotation.from_rotvec(target[3:6]) * base.inv()).as_rotvec() / 0.2
        np.testing.assert_allclose(actual[:6], np.r_[expected_xyz, expected_rot], atol=1e-12)
        self.assertEqual(actual[6], -1)
        target[6] = 0
        self.assertEqual(to_osc_action(target, origin, obs, env)[6], 1)

    def test_future_targets_padding_and_gripper_command(self):
        states = np.zeros((4, 14), np.float32)
        states[:, 0] = [0, 0.01, 0.03, 0.06]
        states[:, 6] = 0.07  # measured opening differs from desired command
        states[:, 3:6] = Rotation.from_euler("zyx", [[0.1, 0.2, 0], [0.2, 0.3, 0.1],
                                                                   [0.3, 0.4, 0.2], [0.4, 0.5, 0.3]]).as_rotvec()
        action, mask, padding = encode_targets(states, 1, 4, np.array([0.08, 0, 0]))
        np.testing.assert_allclose(action[:, 0], [0.02, 0.05, 0.05, 0.05], atol=1e-7)
        np.testing.assert_array_equal(action[:, 6], [0, 0, 0, 0])
        np.testing.assert_array_equal(padding, [False, False, True, True])
        self.assertEqual(mask.sum(), 14)
        recovered = Rotation.from_rotvec(action[0, 3:6]) * Rotation.from_rotvec(states[1, 3:6])
        np.testing.assert_allclose(recovered.as_matrix(), Rotation.from_rotvec(states[2, 3:6]).as_matrix(), atol=1e-7)

    def test_resampling_uses_future_times_and_origin_anchor(self):
        chunk = np.zeros((3, 7), np.float64)
        chunk[:, 0] = np.arange(1, 4) / 15
        chunk[:, 5] = np.arange(1, 4) / 15
        chunk[:, 6] = 0.08
        converted = resample_chunk(chunk, np.zeros(3), Rotation.identity(), source_hz=15, target_hz=20)
        np.testing.assert_allclose(converted[:, 0], [0.05, 0.1, 0.15, 0.2], atol=1e-12)
        np.testing.assert_allclose(converted[:, 5], converted[:, 0], atol=1e-12)
        with self.assertRaises(ValueError):
            resample_chunk(chunk, np.zeros(3), Rotation.identity(), source_hz=0, target_hz=20)

    def test_controller_goal_is_aligned_with_current_command(self):
        states = np.zeros((2, 14), dtype=np.float32)
        states[1, 0] = 0.01
        goals = states.copy()
        goals[:, 0] = [0.05, 0.08]
        goals[:, 6] = [0.08, 0]
        action, mask, padding = encode_targets(states, 0, 3, target_poses=goals)
        np.testing.assert_allclose(action[:, 0], [0.05, 0.08, 0.08])
        np.testing.assert_array_equal(padding, [False, False, True])
        last, last_mask, last_padding = encode_targets(states, 1, 3, target_poses=goals)
        self.assertAlmostEqual(float(last[0, 0]), 0.07, places=6)
        self.assertEqual(last_mask.sum(), 7)
        np.testing.assert_array_equal(last_padding, [False, True, True])

    def test_dataset_stats_exclude_held_out_episodes(self):
        import h5py
        from .libero_dataset import LiberoSpiritDataset
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "prepared.hdf5"
            with h5py.File(path, "w") as file:
                file.attrs.update(spirit_libero_schema=2, action_hz=20,
                                  instruction="test", camera_convention="upright_rgb")
                for name, offset in (("demo_0", 0), ("demo_1", 50)):
                    group = file.create_group(name)
                    states = np.zeros((2, 14), np.float32)
                    states[:, 0] = np.array([0, 1]) + offset
                    group.create_dataset("state", data=states)
                    goals = states.copy()
                    goals[:, 0] += 0.5
                    group.create_dataset("target_pose", data=goals)
            data = LiberoSpiritDataset(path, episodes=["demo_0"])
            self.assertEqual(len(data), 2)
            self.assertEqual(data.normalization_stats()["state_max"][0], 1)
            self.assertEqual(data.normalization_stats()["action_max"][0], 1.5)
            self.assertEqual(data.get_lowdim_item(1)["action_mask"].sum(), 7)
            with self.assertRaises(ValueError): LiberoSpiritDataset(path, episodes=["demo_99"])

    def test_all_camera_roles_are_flipped_vertically(self):
        calls = []
        def render(camera_name, width, height):
            calls.append(camera_name)
            return np.repeat(np.arange(height, dtype=np.uint8)[:, None, None], width*3, axis=1).reshape(height, width, 3)
        images = render_images(SimpleNamespace(sim=SimpleNamespace(render=render)))
        self.assertEqual(set(calls), set(CAMERAS.values()))
        for image in images.values():
            self.assertEqual(image[0, 0, 0], 239)
            self.assertEqual(image[-1, 0, 0], 0)
            self.assertTrue(image.flags.c_contiguous)

    def test_velocity_score_has_real_gradient(self):
        class Policy(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.tensor(0.5))
                self.config = SimpleNamespace(n_action_steps=2, max_action_dim=32)
                self.normalize_inputs = nn.Identity()
                self.normalize_targets = nn.Identity()
                self.proj_vlm_output = nn.Identity()
            def prepare_state(self, batch): return batch["observation.state"]
            def _encode_vision(self, batch): return torch.ones(1, 2, 3)
            def _denoise_step(self, state, encoded, noisy, time): return self.weight * noisy
            def select_action(self, *args): raise AssertionError("Velocity scoring must not call the sampler")
        adapter = SpiritAdapter.__new__(SpiritAdapter)
        adapter.device, adapter.model = torch.device("cpu"), Policy()
        adapter._sampling_module = SimpleNamespace(
            sample_time=lambda size, device: torch.full((size,), 0.5),
            sample_noise=lambda shape, device: torch.ones(shape),
        )
        batch = {key: torch.zeros(1, 3, 2, 2) for key in CAMERAS}
        batch.update({"observation.state": torch.zeros(1, 1, 14), "task": ["test"], "robot_type": ["Franka"]})
        score = adapter.policy_logp(batch, torch.zeros(1, 2, 14))
        torch.testing.assert_close(score, torch.tensor([-0.5625]))
        score.sum().backward()
        torch.testing.assert_close(adapter.model.weight.grad, torch.tensor(0.75))
        adapter.reference_state_dict = {"weight": torch.tensor(0.5)}
        with torch.no_grad(): adapter.model.weight.fill_(0.7)
        adapter.model.weight.grad = None
        current, reference = adapter.policy_logp_with_ref(batch, torch.zeros(1, 2, 14))
        torch.testing.assert_close(reference, torch.tensor([-0.5625]))
        current.sum().backward()
        torch.testing.assert_close(adapter.model.weight.grad, torch.tensor(0.65))
        torch.testing.assert_close(adapter.model.weight, torch.tensor(0.7))

    def test_checkpoint_keeps_normalization(self):
        model = nn.Module()
        model.action_out_proj = nn.Linear(2, 2)
        model.normalize_inputs = nn.Module()
        model.normalize_inputs.register_buffer("minimum", torch.tensor([-0.31, 0.24]))
        model.config = SimpleNamespace(n_action_steps=60)
        adapter = SpiritAdapter.__new__(SpiritAdapter)
        adapter.model = model
        adapter.cfg = SimpleNamespace(lora_r=4, lora_alpha=8, spirit_action_hz=20)
        with tempfile.TemporaryDirectory() as folder:
            path = str(Path(folder) / "adapter.pt")
            adapter.save(path)
            model.normalize_inputs.minimum.zero_()
            adapter.load(path)
            torch.testing.assert_close(model.normalize_inputs.minimum, torch.tensor([-0.31, 0.24]))
            incomplete = torch.load(path, weights_only=True)
            incomplete["state_dict"].pop("normalize_inputs.minimum")
            broken = str(Path(folder) / "incomplete.pt")
            torch.save(incomplete, broken)
            with self.assertRaises(ValueError): adapter.load(broken)
            adapter.cfg.lora_r = 8
            with self.assertRaises(ValueError): adapter.load(path)

    def test_float32_normalization_does_not_disable_neural_autocast(self):
        model = nn.Module()
        model.normalize_inputs = nn.Linear(2, 2).float()
        model.qwen = nn.Linear(2, 2).bfloat16()
        adapter = SpiritAdapter.__new__(SpiritAdapter)
        adapter.model, adapter.device = model, torch.device("cpu")
        with adapter._autocast():
            output = model.qwen(torch.ones(1, 2, dtype=torch.float32))
        self.assertEqual(output.dtype, torch.bfloat16)

    def test_timeout_is_not_a_success(self):
        from post_training.eval_libero import run_one_trial
        obs = {"agentview_image": np.zeros((224, 224, 3), np.uint8)}
        env = SimpleNamespace(reset=lambda: obs, set_init_state=lambda state: obs,
                              step=lambda action: (obs, 0, True, {}), check_success=lambda: False)
        adapter = SimpleNamespace(cfg=SimpleNamespace(base="openvla"),
                                  select_action=lambda batch: torch.zeros(1, 1, 7))
        result = run_one_trial(adapter, env, "test", None, 2, num_steps_wait=0)
        self.assertFalse(result["success"])
        self.assertEqual(result["n_steps"], 1)

    def test_demo_reset_preserves_recorded_scene(self):
        from post_training.eval_libero import run_one_trial
        obs = {"agentview_image": np.zeros((224, 224, 3), np.uint8)}
        resets = []
        def forbidden_reset(*args):
            raise AssertionError("A benchmark reset would discard the recorded scene")
        def restore_demo():
            resets.append(True)
            return obs
        env = SimpleNamespace(reset=forbidden_reset, set_init_state=forbidden_reset,
                              step=lambda action: (obs, 0, True, {}), check_success=lambda: True)
        adapter = SimpleNamespace(cfg=SimpleNamespace(base="openvla"),
                                  select_action=lambda batch: torch.zeros(1, 1, 7))
        result = run_one_trial(adapter, env, "test", None, 2, num_steps_wait=0,
                               reset_callback=restore_demo)
        self.assertEqual(resets, [True])
        self.assertTrue(result["success"])
        self.assertEqual(result["n_steps"], 1)


if __name__ == "__main__":
    unittest.main()
