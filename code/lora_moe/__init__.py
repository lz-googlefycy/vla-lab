"""
LoRA-MoE for OpenVLA — 次优先方向（Phase 4 之后）

当前是占位骨架。实装顺序（如果决定做）：
  1. skill_router.py    — Top-2 路由 + Load Balance
  2. lora_moe_module.py — 8 expert × LoRA r=16
  3. inject.py          — 把 LoRA-MoE 挂到 OpenVLA 的 q_proj / v_proj
  4. skill_labeler.py   — Gemini / GPT-4o 标 LIBERO instruction
"""

__version__ = "0.0.0-stub"
