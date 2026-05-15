"""Projection head + InfoNCE loss for multi-view + temporal pretraining.

Design (see docs/work_log/day5_multiview_design.md):
  z = proj_head(siglip_pool(image))     # (B, 128)
  L_mva = InfoNCE(z_agent_t, z_wrist_t)
  L_tc  = InfoNCE(z_agent_t, z_agent_{t+5})
  L = 0.5 * L_mva + 0.5 * L_tc
"""
from __future__ import annotations
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiViewProjectionHead(nn.Module):
    """Maps SigLIP per-token features → 128-dim L2-normalized embedding.

    Architecture:
        SigLIP output: (B, n_tokens=257, D=1152)  → CLS pool → (B, 1152)
        Linear(1152, 512) → GELU → Linear(512, 128) → L2 normalize → (B, 128)

    Why CLS pool (not mean pool):
        SigLIP/CLIP pretraining uses CLS token for global representation.
        Mean pool dilutes the signal; CLS is what was actually contrastively
        trained at upstream.
    """

    def __init__(self, in_dim: int = 1152, hidden: int = 512, out_dim: int = 128):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args: x — (B, D) image-level feature (already CLS-pooled)
        Returns: (B, out_dim) L2-normalized embedding
        """
        h = F.gelu(self.fc1(x))
        z = self.fc2(h)
        z = F.normalize(z, p=2, dim=-1)
        return z


def info_nce_loss(z_a: torch.Tensor, z_b: torch.Tensor, tau: float = 0.07) -> torch.Tensor:
    """Symmetric InfoNCE loss between two L2-normalized embedding batches.

    Args:
        z_a, z_b: (B, D) L2-normalized; row i in z_a matches row i in z_b.
        tau: temperature.

    Returns:
        scalar loss = mean of two-direction InfoNCE.
    """
    B = z_a.shape[0]
    logits = z_a @ z_b.T / tau                     # (B, B)
    labels = torch.arange(B, device=z_a.device)
    loss_ab = F.cross_entropy(logits, labels)
    loss_ba = F.cross_entropy(logits.T, labels)
    return 0.5 * (loss_ab + loss_ba)


def multiview_temporal_loss(
    z_agent_t: torch.Tensor,
    z_wrist_t: torch.Tensor,
    z_agent_t_delta: torch.Tensor,
    tau: float = 0.07,
    w_mva: float = 0.5,
    w_tc: float = 0.5,
) -> dict:
    """Compute combined multi-view + temporal InfoNCE loss.

    Args:
        z_agent_t:        (B, D) agent-view at time t
        z_wrist_t:        (B, D) wrist-view at time t
        z_agent_t_delta:  (B, D) agent-view at time t+Δ
        tau: temperature
        w_mva, w_tc: loss weights (default 0.5, 0.5)

    Returns:
        dict with keys: total_loss, mva_loss, tc_loss
    """
    L_mva = info_nce_loss(z_agent_t, z_wrist_t, tau)
    L_tc = info_nce_loss(z_agent_t, z_agent_t_delta, tau)
    total = w_mva * L_mva + w_tc * L_tc
    return {
        "total_loss": total,
        "mva_loss": L_mva.detach(),
        "tc_loss": L_tc.detach(),
    }
