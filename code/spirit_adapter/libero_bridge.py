"""Franka pose and camera bridge for Spirit v1.5 and LIBERO / robosuite 1.4.

Spirit predicts physical target poses relative to the observation at chunk
start. OSC_POSE consumes scaled deltas relative to the current live pose.
The functions here are shared by evaluation and demonstration conversion.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


CAMERAS = {
    "observation.images.cam_high": "agentview",
    "observation.images.cam_left_wrist": "robot0_eye_in_hand",
    "observation.images.cam_right_wrist": "sideview",
}
IMAGE_SIZE = (240, 320)
GRIPPER_OPEN_M = 0.08


def eef_pose_in_base(obs, env):
    robot = env.robots[0]
    base_pos = np.asarray(robot.base_pos, dtype=np.float64)
    base_rot = Rotation.from_quat(robot.base_ori)  # robosuite exposes xyzw
    world_pos = np.asarray(obs["robot0_eef_pos"], dtype=np.float64)
    world_rot = Rotation.from_quat(obs["robot0_eef_quat"])
    return (
        base_rot.inv().apply(world_pos - base_pos),
        base_rot.inv() * world_rot,
        base_rot,
        world_pos,
        world_rot,
    )


def state_from_observation(obs, env):
    """Return [xyz, rotvec, width_m, zeros(7)] in the robot base frame."""
    pos, rot, *_ = eef_pose_in_base(obs, env)
    fingers = np.asarray(obs["robot0_gripper_qpos"], dtype=np.float64)
    if fingers.shape != (2,):
        raise ValueError("Spirit LIBERO bridge requires the two-finger Panda gripper")
    state = np.zeros(14, dtype=np.float32)
    state[:3] = pos
    state[3:6] = rot.as_rotvec()
    state[6] = np.abs(fingers).sum()
    if not np.isfinite(state).all():
        raise ValueError("Non-finite Franka state")
    return state


def render_images(env):
    """Render all three camera roles as upright RGB, independent of obs macros.

    MuJoCo / robosuite 1.4 read_pixels returns OpenGL bottom-up rows. Spirit's
    PIL preprocessing expects top-down rows. This is a vertical flip, with
    no horizontal reflection or OpenVLA-specific 180-degree rotation.
    """
    height, width = IMAGE_SIZE
    return {
        key: np.ascontiguousarray(env.sim.render(
            camera_name=name, width=width, height=height,
        )[::-1])
        for key, name in CAMERAS.items()
    }


def prepare_batch(obs, env, instruction):
    import torch

    result = {
        key: torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0) / 255.0
        for key, img in render_images(env).items()
    }
    result["observation.state"] = torch.from_numpy(state_from_observation(obs, env))[None, None]
    result["task"] = [instruction]
    result["robot_type"] = ["Franka"]
    return result


def controller_goal(obs, env, command, gripper_opening=None):
    """Record the actual OSC setpoint, before controller tracking dynamics.

    LIBERO's action is a command to the servo, rather than the measured EE
    pose one tick later. SFT must preserve that distinction.
    """
    command = np.asarray(command)
    if command.shape != (7,) or not np.isfinite(command).all():
        raise ValueError("Expected a finite seven-dimensional OSC command")
    physical = env.robots[0].controller.scale_action(command[:6])
    state = state_from_observation(obs, env)
    _, current_rot, base_rot, *_ = eef_pose_in_base(obs, env)
    goal = state.copy()
    goal[:3] += base_rot.inv().apply(physical[:3])
    goal[3:6] = (base_rot.inv() * Rotation.from_rotvec(physical[3:]) * base_rot * current_rot).as_rotvec()
    if gripper_opening is None:
        gripper_opening = GRIPPER_OPEN_M if command[6] < 0 else 0.0 if command[6] > 0 else state[6]
    goal[6] = gripper_opening
    return goal


def encode_targets(states, index, horizon, gripper_targets=None, target_poses=None):
    """Encode future measured poses using Spirit's chunk-origin convention.

    states[t] and cameras at t are observations BEFORE an action. LIBERO uses
    explicit controller target_poses[t] for command t. Without explicit goals,
    this can also encode measured future states in Spirit's original format.
    Episode padding and unused arm dimensions are excluded from the loss.
    """
    states = np.asarray(states)
    limit = len(states) if target_poses is not None else len(states) - 1
    if states.ndim != 2 or states.shape[1] != 14 or not 0 <= index < limit:
        raise ValueError("Need an (N,14) episode and a frame with a target")
    if horizon < 1:
        raise ValueError("horizon must be positive")
    if target_poses is not None and gripper_targets is not None:
        raise ValueError("Explicit goal poses already include gripper targets")
    future = index + (0 if target_poses is not None else 1) + np.arange(horizon)
    valid = future < len(states)
    poses = states if target_poses is None else np.asarray(target_poses)
    if poses.shape != states.shape:
        raise ValueError("Controller targets and observations must have matching (N,14) shapes")
    target = poses[np.minimum(future, len(states) - 1)]
    action = np.zeros((horizon, 14), dtype=np.float32)
    action[:, :3] = target[:, :3] - states[index, :3]
    action[:, 3:6] = (
        Rotation.from_rotvec(target[:, 3:6]) * Rotation.from_rotvec(states[index, 3:6]).inv()
    ).as_rotvec()
    action[:, 6] = target[:, 6]
    if gripper_targets is not None:
        # A target at state[t+1] corresponds to the command executed at t.
        action[:, 6] = gripper_targets[np.minimum(future - 1, len(states) - 2)]
    mask = np.zeros_like(action, dtype=bool)
    mask[:, :7] = valid[:, None]
    return action, mask, ~valid


def control_freq(env):
    wrapped = getattr(env, "env", env)
    return float(wrapped.control_freq)


def resample_chunk(chunk, origin_pos, origin_rot, *, source_hz, target_hz, origin_width=0.08):
    """Interpolate target poses at explicit action rates, including t=0 anchor."""
    chunk = np.asarray(chunk)
    if chunk.ndim != 2 or chunk.shape[1] < 7 or not len(chunk):
        raise ValueError("Spirit chunk must have shape (T, >=7)")
    if source_hz <= 0 or target_hz <= 0:
        raise ValueError("Action frequencies must be positive")
    if np.isclose(source_hz, target_hz):
        return chunk[:, :7].copy()
    source_t = np.arange(len(chunk) + 1) / source_hz
    target_t = np.arange(1, int(np.floor(len(chunk) * target_hz / source_hz + 1e-8)) + 1) / target_hz
    anchored = np.concatenate([np.zeros((1, 7), dtype=chunk.dtype), chunk[:, :7]])
    anchored[0, 6] = origin_width
    result = np.empty((len(target_t), 7), dtype=chunk.dtype)
    for axis in (0, 1, 2, 6):
        result[:, axis] = np.interp(target_t, source_t, anchored[:, axis])
    rotations = Rotation.from_rotvec(anchored[:, 3:6]) * origin_rot
    result[:, 3:6] = (Slerp(source_t, rotations)(target_t) * origin_rot.inv()).as_rotvec()
    return result


def to_osc_action(target, origin_pose, obs, env):
    """Convert one chunk-origin target to the active OSC_POSE input range.

    Target gripper width is absolute in meters. LIBERO training targets
    encode -1/open as 0.08 m and +1/closed as 0 m. The midpoint recovers the
    binary command without using statistics from other Spirit embodiments.
    """
    target = np.asarray(target, dtype=np.float64)
    if target.shape[0] < 7 or not np.isfinite(target[:7]).all():
        raise ValueError("Spirit target must contain seven finite values")
    origin_pos, origin_rot, base_rot, *_ = origin_pose
    current_pos, _, _, _, current_world_rot = eef_pose_in_base(obs, env)
    target_world_rot = base_rot * Rotation.from_rotvec(target[3:6]) * origin_rot
    physical = np.concatenate([
        base_rot.apply(origin_pos + target[:3] - current_pos),
        (target_world_rot * current_world_rot.inv()).as_rotvec(),
    ])
    controller = env.robots[0].controller
    if not getattr(controller, "use_delta", False) or getattr(controller, "control_dim", None) != 6:
        raise ValueError("Spirit bridge requires a 6D delta OSC_POSE controller")
    if getattr(controller, "impedance_mode", "fixed") != "fixed":
        raise ValueError("Variable-impedance OSC_POSE is unsupported")
    low, high = np.asarray(controller.input_min), np.asarray(controller.input_max)
    out_low, out_high = np.asarray(controller.output_min), np.asarray(controller.output_max)
    scale = (out_high - out_low) / (high - low)
    command = (physical - (out_high + out_low) / 2) / scale + (high + low) / 2
    grip = -1.0 if target[6] >= GRIPPER_OPEN_M / 2 else 1.0
    return np.concatenate([np.clip(command, low, high), [grip]])
