"""
SkillRouter — 占位骨架

类比 TrajFlow 的 BehaviorRouter，路由 LIBERO 任务到 8 个 skill expert：
  pick / place / push / pull / slide / rotate / insert / open

血泪教训（来自 TrajFlow V3）：
  - 必须加 Load Balance Loss（否则 dead expert）
  - Router 监督要强：CE + KL，weight ≥ 10×
  - Top-2 routing 比 Top-1 / Top-4 在我们规模下更好
  - LoRA B 矩阵必须 zero-init（起点等价 vanilla LoRA）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SkillRouter(nn.Module):
    """Route token embeddings to top-k LoRA skill experts."""

    def __init__(self, d_model: int, num_skills: int = 8, top_k: int = 2,
                 noise_std: float = 0.1):
        super().__init__()
        self.num_skills = num_skills
        self.top_k = top_k
        self.noise_std = noise_std

        # Gate: token embedding -> skill logits
        self.gate = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, num_skills),
        )

    def forward(self, x: torch.Tensor, skill_labels: torch.Tensor = None):
        """
        Args:
            x: [B, T, d_model] or [N_tokens, d_model]
            skill_labels: [N_tokens, num_skills] one-hot or soft labels (optional)

        Returns:
            topk_weights: [N_tokens, top_k]
            topk_indices: [N_tokens, top_k]
            lb_loss: scalar (load-balance auxiliary loss)
            sup_loss: scalar (router supervision loss)
        """
        if x.dim() == 3:
            B, T, D = x.shape
            x_flat = x.view(B * T, D)
        else:
            x_flat = x

        logits = self.gate(x_flat)

        # Add noise during training to avoid early collapse
        if self.training and self.noise_std > 0:
            logits = logits + torch.randn_like(logits) * self.noise_std

        probs = F.softmax(logits, dim=-1)
        topk_logits, topk_indices = torch.topk(logits, self.top_k, dim=-1)
        topk_weights = F.softmax(topk_logits, dim=-1)

        # === Load Balance Loss (from Switch Transformer / Mixtral) ===
        one_hot = F.one_hot(topk_indices, self.num_skills).float()  # [N, top_k, S]
        expert_load = one_hot.sum(dim=1).mean(dim=0)  # [S]
        avg_probs = probs.mean(dim=0)  # [S]
        lb_loss = self.num_skills * (avg_probs * expert_load).sum()

        # === Supervised loss (CE + KL) ===
        if skill_labels is not None:
            target = skill_labels.argmax(dim=-1)
            ce_loss = F.cross_entropy(logits, target, reduction="mean")
            log_probs = F.log_softmax(logits, dim=-1)
            soft_target = skill_labels / skill_labels.sum(-1, keepdim=True).clamp(min=1e-8)
            kl_loss = F.kl_div(log_probs, soft_target, reduction="batchmean")
            sup_loss = ce_loss + 0.5 * kl_loss
        else:
            sup_loss = torch.tensor(0.0, device=x.device)

        return topk_weights, topk_indices, lb_loss, sup_loss


if __name__ == "__main__":
    # Quick sanity
    router = SkillRouter(d_model=4096, num_skills=8, top_k=2)
    x = torch.randn(32, 4096)
    labels = F.one_hot(torch.randint(0, 8, (32,)), 8).float()
    w, idx, lb, sup = router(x, labels)
    print(f"weights {w.shape}  idx {idx.shape}  lb {lb.item():.4f}  sup {sup.item():.4f}")
