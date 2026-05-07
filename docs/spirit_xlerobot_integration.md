# Spirit v1.5 × XLeRobot 整合设计文档

> 创建：2026-05-08
> 目标：把千寻智能 Spirit v1.5 VLA 跑到 \$660 XLeRobot（SO-100 双臂 + 移动底座）上
> 最终交付：真机 demo 视频 + 博客 #2（Week 4）

---

## 0. TL;DR

**核心挑战**：Spirit 是为 **ALOHA 双臂（14-DoF）** 设计的，XLeRobot 用的是 **SO-100 双臂（12-DoF）** — 硬件不同构。

**解决路径**：
- Phase A（Week 3）：**Zero-shot 推理尝试** — 把 SO-100 state 零 padding 到 14 维喂给 Spirit，看"半工作"什么样。**大概率失败**，但 demo 视频有价值（"失败也是一种数据"）。
- Phase B（Week 4-5）：**采 50 条 XLeRobot teleop 数据 → Spirit LoRA fine-tune**（14→12 action projection 作为唯一被训的头）。**这才是真正的博客 #2 内容**。
- Phase C（Week 6-8）：**多任务 fine-tune + RoboChallenge 结构化评测**（如能申请上账号）。

---

## 1. 硬件配置对比

| 维度 | Spirit 训练硬件 (ALOHA) | XLeRobot (SO-100) | 兼容性 |
|:---|:---|:---|:---:|
| 臂数 | 2 | 2 | ✅ |
| 单臂 DoF | 7（6+gripper，带肘部 elbow_flex + shoulder_roll） | 6（shoulder_pan/lift/elbow_flex/wrist_flex/wrist_roll + gripper） | ⚠️ 少一个 shoulder_roll |
| 总 action dim | 14 = 7×2 | 12 = 6×2 | ⚠️ 需要 projection |
| 总 state dim | 14 = 7×2 proprioception | 12 = 6×2 | ⚠️ 需要 padding or projection |
| 头部相机 | `cam_high` 320×240 | 头部 1-2 RGB（RasberryPi 或 stereo） | ✅ |
| 左腕相机 | `cam_left_wrist` 320×240 | 无原生腕部相机 | ❌ **最大 gap** |
| 右腕相机 | `cam_right_wrist` 320×240 | 无原生腕部相机 | ❌ |
| 底盘 | 无（ALOHA 是固定桌面） | 2-wheel 差速 | — 暂不管（Spirit 也没有底盘控制）|

**关键 gap**：
- ❌ **腕部相机缺失**。Spirit 强依赖双腕相机（dexterity 重要信号源）。可选补救：
  - 在 SO-100 夹爪附近用扎带贴一个便宜 USB 摄像头（~$20 each）
  - 或用 `cam_high` 复制 3 份作为 3 个摄像头（肯定效果差，但能跑通 pipeline）
- ⚠️ **Action 维度不对齐**（12 vs 14）。Spirit 的 state/action 有 `max_state_dim=32, max_action_dim=32` 的 padding 机制，理论上可以零填充到 32，但 **normalization stats 要重新训**。

---

## 2. Spirit v1.5 API 剖析（复用本地 clone）

代码位置：`/home/ubuntu/spirit-v1.5/`，已 vendored 到 `docker-spirit/spirit-v1.5/`。

### 2.1 输入格式（来自 `config.json`）

```python
batch = {
    "observation.images.cam_high":         # [B, 3, 240, 320] float RGB in [0,1]
    "observation.images.cam_left_wrist":   # [B, 3, 240, 320]
    "observation.images.cam_right_wrist":  # [B, 3, 240, 320]
    "observation.state":                   # [B, 14] 关节角度（弧度）
    "task":                                # List[str] 指令，如 ["put the cup on the plate"]
}
```

### 2.2 推理入口（来自 `model/modeling_spirit_vla.py:814`）

```python
from model import SpiritVLAPolicy
policy = SpiritVLAPolicy.from_pretrained("Spirit-v1.5/")
policy = policy.to("cuda").eval()

with torch.no_grad():
    actions = policy.select_action(batch)  # [B, 60, 14] — 60-step chunk × 14-dim
```

### 2.3 关键内部参数

- **Backbone**: Qwen3-VL-4B-Instruct（需要单独下载）
- **DiT head**: 16 层 × 1536 hidden, 32 heads
- **Diffusion**: 10 去噪步
- **Action chunk**: 60 步
- **Normalization**: MinMax，stats 从 config 读（如果 ckpt 无 stats → 必须用 `compute_norm_stats` 从数据算）

---

## 3. 整合设计：三个层次

### 3.1 Adapter 层（最薄封装）

```
                       XLeRobot sensor/actuator space
                                     │
                      ┌──────────────┴──────────────┐
                      ▼                             ▲
       XLeRobotAdapter.wrap(obs)         XLeRobotAdapter.unwrap(actions)
                      │                             │
                      ▼                             ▲
                Spirit v1.5 expected space (14-DoF × ALOHA naming)
                              │                    │
                              ▼                    ▲
                          policy.select_action(batch)
```

**必要的 wrap/unwrap 逻辑**：

1. **State wrap (12 → 14)**:
   - XLeRobot 关节顺序: `[L_shoulder_pan, L_shoulder_lift, L_elbow_flex, L_wrist_flex, L_wrist_roll, L_gripper, R_...]`
   - Spirit ALOHA 顺序: `[L_waist, L_shoulder, L_elbow, L_forearm_roll, L_wrist_angle, L_wrist_rotate, L_gripper, R_...]`
   - 最简映射: **零填充 shoulder_roll**（第 4 维设 0），其他维度按语义对应
   - `state_14 = [L_pan, L_lift, 0, L_elbow, L_wrist_flex, L_wrist_roll, L_gripper, R_pan, R_lift, 0, R_elbow, R_wrist_flex, R_wrist_roll, R_gripper]`

2. **Image wrap**（Phase A，真机没有腕部相机时）:
   - `cam_high` = SO-100 头部相机，resize 到 320×240
   - `cam_left_wrist` / `cam_right_wrist` = 空白黑图或 `cam_high` 复制
   - ⚠️ 这会严重降低 Spirit 的性能，仅用于跑通 pipeline

3. **Action unwrap (14 → 12)**:
   - 丢弃 `shoulder_roll` 那一维（索引 2 和 9）
   - 其他维按语义对应到 SO-100 关节

### 3.2 推理循环（受控频率 5-10 Hz）

```python
# pseudo-code
loop:
    obs_xle = xle_robot.read()       # 12-dim state + head RGB
    obs_spirit = adapter.wrap(obs_xle)
    action_chunk = policy.select_action(obs_spirit)  # [1, 60, 14]
    action_14 = action_chunk[0, 0]   # 取第一步，或做 temporal ensemble
    action_12 = adapter.unwrap(action_14)
    xle_robot.command(action_12)
    time.sleep(1/5)  # 5 Hz, 每 12 步重新推理
```

### 3.3 LeRobot 集成（Phase B，fine-tune）

LeRobot 原生支持数据采集 + teleoperation + policy rollout 标准化。我们的集成点：

```python
from lerobot.common.policies.factory import make_policy
# 自定义一个 SpiritPolicy wrapper 实现 LeRobot 的 Policy 协议:
# - select_action(observation) -> action
# - 兼容 LeRobot's observation dict key convention

class SpiritLeRobotPolicy:
    def __init__(self, spirit_ckpt):
        self.inner = SpiritVLAPolicy.from_pretrained(spirit_ckpt)
        self.adapter = XLeRobotAdapter()
    def select_action(self, obs):
        spirit_obs = self.adapter.wrap(obs)
        action = self.inner.select_action(spirit_obs)
        return self.adapter.unwrap(action)
```

---

## 4. 实施路线（3 周）

### Week 3 (5/13-5/19): Phase A — Zero-shot 跑通

**目标**：**仿真中** Spirit v1.5 → SO-100 action 输出，即使完全不成功也拿到推理视频

- [ ] Day 1-2: 构建 `spirit-v1.0-cu128-py310` 镜像（已在建中）
- [ ] Day 3: Spirit base ckpt + Qwen3-VL-4B 下载完成（已启动）
- [ ] Day 4: `XLeRobotAdapter` v0 实现 + 单元测试
- [ ] Day 5: 在 XLeRobot **Mujoco 仿真**里跑 Spirit zero-shot（不上真机，避免损坏）
- [ ] Day 6-7: 录制**仿真失败视频**（作为"the gap"证据），写博客 #2 草稿

**验收**：仿真视频显示 Spirit 的 action chunk 被解码到 SO-100，即使 robot 随便乱动。博客指出"跨 embodiment zero-shot 是不work 的 — 这是我接下来要 fine-tune 的动机"。

### Week 4 (5/20-5/26): Phase B-1 — XLeRobot 数据采集 + 小规模 fine-tune

**目标**：采 50-100 条双臂示教 → LoRA fine-tune Spirit → 在仿真里验证 SR 提升

- [ ] Day 1-3: **用 XLeRobot 真机采数据**（你操作硬件）
  - 任务集：pick-place 3 个任务（cup / block / cloth），每任务 10-15 demos
  - 数据格式：LeRobot 标准 parquet（observation.state, observation.images.*, action）
  - 后处理：XLeRobot 12-dim → padded to Spirit 14-dim ALOHA 顺序
- [ ] Day 4-5: Spirit LoRA fine-tune 脚本
  - 基于 Spirit 官方 `train.py` 改，只训 DiT action head + state/action projection 层
  - 冻结 Qwen3-VL backbone（节省显存，50 条数据也不够训 backbone）
  - 开发机单卡 H20 144GB，BS 4-8
- [ ] Day 6-7: Spirit + XLeRobot (fine-tuned) 在仿真评测，对比 zero-shot baseline

**验收**：
- LoRA checkpoint 上 HuggingFace
- 对比视频：zero-shot vs fine-tuned
- 博客 #2 内容充实

### Week 5 (5/27-6/2): Phase B-2 — 真机验证

**目标**：在**真实 XLeRobot 上**跑 fine-tuned Spirit

- [ ] Day 1-2: 本机 + XLeRobot 通过 USB 串口/Wifi 连接验证
- [ ] Day 3-5: 真机 rollout 录制（每任务 10-20 次，统计 SR）
- [ ] Day 6-7: 后期剪辑成 **\$660 机器人 Spirit v1.5 demo video**（主 demo，2-3 min）

**验收**：
- B站视频 #1 发布：「\$660 机器人跑通千寻智能 Spirit v1.5」
- 知乎博客 #2 发布：技术深度版
- Twitter 英文短帖 @ SpiritAITeam

---

## 5. 关键风险与应对

| 风险 | 概率 | 影响 | 应对 |
|:---|:---:|:---:|:---|
| Spirit zero-shot 完全不 work | 高 | 中 | 按计划，作为 "motivation video"，不妨 |
| LoRA fine-tune 50 条数据不够 | 中 | 高 | fallback 到 **full fine-tune 小规模**或 **跟 OpenVLA 官方对比 baseline** |
| XLeRobot 硬件/软件 bug | 中 | 中 | 先在 Mujoco 仿真全跑通，真机只做最终验证 |
| 腕部相机缺失 → zero-shot 显著降低 | 高 | 中 | Phase B fine-tune 时学习只用 cam_high 的投影，或者花 $60 加装双 USB 腕部相机 |
| Qwen3-VL-4B 显存不够 | 低 | 高 | 本机 3090-24GB 4-bit 量化可能勉强；开发机 H20-144GB 肯定够 |
| 推理太慢（DiT 10 步 + 60 action chunk） | 中 | 中 | 减少 num_steps 到 5，用 action chunking 节省 inference frequency |

---

## 6. 成果物交付清单

### 必达（Week 5 末）
- [ ] `SpiritLeRobotPolicy` 类（LeRobot 原生兼容）
- [ ] Spirit v1.5 → XLeRobot LoRA checkpoint（HF: `lz-googlefycy/spirit-v1.5-xlerobot-lora`）
- [ ] 真机 demo 视频（mp4 + gif）
- [ ] 博客 #2（知乎 + Twitter）
- [ ] vla-lab / openvla-libero 仓库对应 commit + update

### 期望（Week 6-8）
- [ ] 多任务（>5 tasks）fine-tune
- [ ] 与 OpenVLA 同任务真机对比（横向对比博客 #5 材料）
- [ ] RoboChallenge 账号申请（如能拿到）

---

## 7. 参考实现（代码草稿）

```python
# code/spirit_adapter/xlerobot_adapter.py
import numpy as np
import torch
from typing import Dict

SPIRIT_ACTION_DIM = 14   # ALOHA 7+7
XLE_ACTION_DIM = 12      # SO-100 6+6
# Mapping: XLeRobot joint index -> Spirit ALOHA slot
# ALOHA slot: [waist, shoulder, elbow, forearm_roll, wrist_angle, wrist_rotate, gripper]
# SO-100:     [pan,   lift,    elbow_flex, (no forearm_roll), wrist_flex, wrist_roll, gripper]
# => slot 3 (forearm_roll) has no counterpart → zero-pad
XLE_TO_SPIRIT_ARM = [0, 1, 2, None, 3, 4, 5]  # length 7

def pad_arm_state(xle_arm_6: np.ndarray) -> np.ndarray:
    """12 SO-100 joints → Spirit 14-dim padded."""
    assert xle_arm_6.shape == (6,)
    spirit_arm = np.zeros(7, dtype=xle_arm_6.dtype)
    for spirit_idx, xle_idx in enumerate(XLE_TO_SPIRIT_ARM):
        if xle_idx is not None:
            spirit_arm[spirit_idx] = xle_arm_6[xle_idx]
    return spirit_arm

def unpad_arm_action(spirit_arm_7: np.ndarray) -> np.ndarray:
    """Drop the forearm_roll slot, return 6-dim."""
    assert spirit_arm_7.shape == (7,)
    return np.array([spirit_arm_7[i] for i in [0, 1, 2, 4, 5, 6]])

class XLeRobotSpiritAdapter:
    def __init__(self, tile_cam_high: bool = True):
        self.tile_cam_high = tile_cam_high

    def wrap_observation(self, obs_xle: Dict) -> Dict:
        """obs_xle: {'state': (12,), 'image_head': (H,W,3) uint8, 'task': str}"""
        state_left = pad_arm_state(obs_xle["state"][:6])
        state_right = pad_arm_state(obs_xle["state"][6:])
        spirit_state = np.concatenate([state_left, state_right])  # (14,)

        head_tensor = torch.from_numpy(obs_xle["image_head"]).permute(2, 0, 1).float() / 255.0
        head_tensor = torch.nn.functional.interpolate(
            head_tensor.unsqueeze(0), size=(240, 320), mode="bilinear"
        ).squeeze(0)

        # Phase A: no wrist cams → tile cam_high (will perform poorly but runs)
        if self.tile_cam_high:
            left_wrist = head_tensor.clone()
            right_wrist = head_tensor.clone()
        else:
            # Phase B: use real wrist cameras if available
            raise NotImplementedError("wrist cameras not implemented yet")

        return {
            "observation.state": torch.from_numpy(spirit_state).unsqueeze(0).float().cuda(),
            "observation.images.cam_high": head_tensor.unsqueeze(0).cuda(),
            "observation.images.cam_left_wrist": left_wrist.unsqueeze(0).cuda(),
            "observation.images.cam_right_wrist": right_wrist.unsqueeze(0).cuda(),
            "task": [obs_xle["task"]],
        }

    def unwrap_action(self, spirit_action: torch.Tensor, step_idx: int = 0) -> np.ndarray:
        """spirit_action: [1, 60, 14] → [12] for one control step."""
        a14 = spirit_action[0, step_idx].cpu().numpy()
        a_left = unpad_arm_action(a14[:7])
        a_right = unpad_arm_action(a14[7:])
        return np.concatenate([a_left, a_right])
```

---

## 8. 下一步执行顺序

1. ✅ `/home/ubuntu/ro_planning/docker-spirit/Dockerfile` — Spirit 镜像（正在 build）
2. ✅ `/home/ubuntu/openvla_assets/spirit_ckpts/Spirit-v1.5/` — Spirit base ckpt（下载中）
3. ✅ `/home/ubuntu/openvla_assets/spirit_ckpts/Qwen3-VL-4B-Instruct/` — VLM backbone（下载中）
4. 🔜 **Spirit smoke test**：镜像 + ckpt 就绪后，第一个里程碑
5. 🔜 `XLeRobotSpiritAdapter` 完整实现 + 单元测试
6. 🔜 仿真 rollout 测试（用 XLeRobot/simulation 里的 Mujoco 环境）
7. 🔜 数据采集 + LoRA fine-tune
8. 🔜 真机 rollout + 视频 + 博客发布
