"""VLA post-training package.

Implements DPO and GRPO for three open VLA bases (OpenVLA, Spirit v1.5,
π0.5) on the LIBERO benchmark. See `docs/plan_v1.5.md` for the full
research plan.

Public surface
--------------
- `VLABase`: protocol that every base adapter must implement.
- `DPOLoss`, `GRPOLoss`: loss functions, ~200 / ~300 LOC each, no
  dependency on TRL or other frameworks.
- `LiberoRewardBuilder`: builds chunk-level reward signal from a LIBERO
  env rollout, used for DPO pair generation and GRPO advantage.
- `PostTrainConfig`: dataclass-style config (mirrors openpi style).

Sub-packages
------------
- `adapters/`: per-base wrappers (openvla, spirit, pi05).
- `eval/`: LIBERO 4-suite evaluation harness.
"""

from .interface import VLABase, PostTrainConfig

__all__ = ["VLABase", "PostTrainConfig"]
