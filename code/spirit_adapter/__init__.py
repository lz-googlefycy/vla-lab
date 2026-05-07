"""XLeRobot ↔ Spirit v1.5 adapter package.

Bridges the 12-DoF XLeRobot (SO-100 dual-arm) observation/action space to
Spirit v1.5's 14-DoF ALOHA expected format.

Two wrap modes:
- Phase A (tile_cam_high=True): use head cam 3× as fake wrist cams; for
  zero-shot smoke tests. Expected to perform poorly.
- Phase B (tile_cam_high=False, real_wrist_cams=True): for post-fine-tune
  deployments with actual wrist USB cameras mounted.
"""

from .xlerobot_adapter import (
    XLeRobotSpiritAdapter,
    SpiritLeRobotPolicy,
    pad_arm_state,
    unpad_arm_action,
    XLE_TO_SPIRIT_ARM,
    SPIRIT_ACTION_DIM,
    XLE_ACTION_DIM,
)

__all__ = [
    "XLeRobotSpiritAdapter",
    "SpiritLeRobotPolicy",
    "pad_arm_state",
    "unpad_arm_action",
    "XLE_TO_SPIRIT_ARM",
    "SPIRIT_ACTION_DIM",
    "XLE_ACTION_DIM",
]
