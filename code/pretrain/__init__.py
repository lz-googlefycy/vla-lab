"""Multi-view spatiotemporal alignment self-supervised pretraining.

Adds robot-specific representation learning on top of frozen SigLIP-vit:
- Multi-view InfoNCE: (agent_view_t, wrist_view_t) positive pair
- Temporal consistency InfoNCE: (agent_view_t, agent_view_{t+Δ}) positive

Goal: warm up vision encoder for downstream LIBERO-Long fine-tuning.
"""
from .model import MultiViewProjectionHead, multiview_temporal_loss
from .dataset import LiberoPretrainDataset

__all__ = [
    "MultiViewProjectionHead",
    "multiview_temporal_loss",
    "LiberoPretrainDataset",
]
