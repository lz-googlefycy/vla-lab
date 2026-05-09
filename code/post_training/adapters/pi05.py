"""π0.5 adapter for v1.5 post-training pipeline.

π0.5 (Physical Intelligence, 2025.09) is the third VLA base in v1.5
paper. Built on flow matching like Spirit but with knowledge insulation
(KI) and Gemma backbone.

Status & decisions
------------------
1. **Inference path**: openpi's PyTorch implementation supports
   inference + finetuning on LIBERO. We wrap that.

2. **LoRA**: openpi PyTorch path explicitly does NOT support LoRA
   (per README). For v1.5 paper, two options:

   (a) Full-finetune in bfloat16 — supported, but ~70 GB VRAM, needs
       H20 144GB
   (b) LoRA via JAX path — JAX-native, harder to integrate but works

   v1.5 default = (a) full-finetune via PyTorch (matches our other
   bases' bf16 path) but on H20 only. Annotated in PostTrainConfig.

3. **Surrogate logp**: π0.5 is also flow-matching, same surrogate as
   Spirit (mean -MSE between predicted and target velocity).

4. **Action representation**: π0.5 default LIBERO config uses
   action_horizon=10, action_dim depends on robot — for LIBERO it's
   7 (matches OpenVLA). NOT 60-step like Spirit.

Reference
---------
- openpi/src/openpi/policies/policy_config.py:create_trained_policy
- openpi/src/openpi/policies/policy.py:Policy.infer
- openpi/src/openpi/training/config.py:pi05_libero TrainConfig
- v1.5 plan §4.1: π0.5 path decision
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Iterator

import numpy as np
import torch

from ..interface import VLABase, PostTrainConfig

# Allow openpi src to be loaded from common locations
_OPENPI_CANDIDATES = [
    os.environ.get("OPENPI_SRC"),
    "/workspace/openpi",
    str(Path.home() / "openpi"),
]
for _p in _OPENPI_CANDIDATES:
    if _p and Path(_p).exists():
        src = str(Path(_p) / "src")
        if Path(src).exists() and src not in sys.path:
            sys.path.insert(0, src)


# π0.5 LIBERO action defaults
PI05_ACTION_HORIZON = 10
PI05_ACTION_DIM = 7
PI05_T_EVAL = 4    # samples for surrogate logp estimation


class Pi05Adapter(VLABase):
    """π0.5 wrapper conforming to VLABase protocol.

    Critical caveat: this class is the v1.5 paper's most "research"
    cell — π0.5 internals (knowledge insulation, JAX/PyTorch model
    duality) require careful handling. The TODO list:

    - [ ] Wire openpi.policies.policy_config.create_trained_policy
          inside __init__ (currently lazy / stub)
    - [ ] Map our PostTrainConfig.libero_suite → openpi
          TrainConfig name (pi05_libero or pi05_libero_low_mem)
    - [ ] Hook the underlying torch model for trainable_parameters()
          — openpi's Policy wraps the model; we need the raw nn.Module
          to apply LoRA / list trainable params
    - [ ] Validate surrogate logp on flow-matching head — π0.5's head
          might have slightly different signature than Spirit's

    Until these are resolved, π0.5 cells in the paper run only:
    - eval (using the pre-trained pi05_libero ckpt as baseline)
    - We can't yet train DPO/GRPO on π0.5 — that's an open research
      task for after Week 2.
    """

    def __init__(self, cfg: PostTrainConfig):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.reference_state_dict: dict[str, torch.Tensor] | None = None

        # Lazy import — openpi has heavy JAX deps
        try:
            from openpi.training import config as _config
            from openpi.policies import policy_config
        except ImportError as e:
            raise ImportError(
                f"openpi import failed: {e}. Make sure openpi is on PYTHONPATH "
                "(set OPENPI_SRC env or clone to ~/openpi or /workspace/openpi)."
            ) from e

        # Resolve openpi train config
        train_config_name = self._resolve_openpi_config()
        train_config = _config.get_config(train_config_name)

        # Build openpi Policy (handles ckpt loading + transforms + norm stats)
        self.openpi_policy = policy_config.create_trained_policy(
            train_config,
            cfg.base_ckpt_path,
            pytorch_device=str(self.device),
        )
        # Hold reference to the underlying torch model for param iteration
        self.model = getattr(self.openpi_policy, "_model", None)
        if self.model is None:
            # try alternate attribute names
            for attr in ("model", "_model_pytorch", "_pytorch_model"):
                m = getattr(self.openpi_policy, attr, None)
                if m is not None and isinstance(m, torch.nn.Module):
                    self.model = m
                    break
        if self.model is None or not isinstance(self.model, torch.nn.Module):
            raise RuntimeError(
                "Could not locate the underlying torch.nn.Module inside "
                "openpi's Policy. The Policy class may have changed; "
                "inspect Policy attributes and update Pi05Adapter."
            )

        # LoRA (TODO — openpi PyTorch doesn't officially support; will
        # require manual injection like our other adapters)
        if cfg.use_lora:
            print("[pi05-adapter] WARNING: LoRA support for π0.5 is "
                  "experimental. See class docstring TODO list.")
            self._inject_lora_experimental()

    def _resolve_openpi_config(self) -> str:
        """Map our PostTrainConfig.libero_suite → openpi config name."""
        suite = self.cfg.libero_suite
        if suite in ("spatial", "object", "goal", "long10", "all4"):
            return "pi05_libero"
        raise ValueError(f"unknown LIBERO suite: {suite}")

    def _inject_lora_experimental(self) -> None:
        """Manual LoRA injection for π0.5 — experimental, may need
        per-version tuning of target_modules."""
        sys.path.insert(0, str(Path(__file__).parent.parent.parent))
        from spirit_adapter.train_lora import _LoRALinear, _replace_module_by_path

        for p in self.model.parameters():
            p.requires_grad = False

        # PaLI-Gemma uses different attention names than Llama
        # (k_proj/v_proj are present but module hierarchy differs).
        # Default targets:
        targets = ["q_proj", "k_proj", "v_proj", "o_proj"]

        replacements = []
        for name, m in self.model.named_modules():
            if not isinstance(m, torch.nn.Linear):
                continue
            for t in targets:
                if name.endswith("." + t) or name == t:
                    replacements.append((name, m))
                    break
        for name, orig in replacements:
            new_m = _LoRALinear(orig, r=self.cfg.lora_r,
                                alpha=self.cfg.lora_alpha,
                                dropout=self.cfg.lora_dropout)
            _replace_module_by_path(self.model, name, new_m)
        n_trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in self.model.parameters())
        print(
            f"[pi05-adapter] LoRA injected: {len(replacements)} Linears, "
            f"trainable {n_trainable/1e6:.1f}M / {n_total/1e6:.0f}M "
            f"({100*n_trainable/n_total:.2f}%)"
        )

    # --------------- VLABase: training-time interface --------------- #

    def policy_logp(
        self, batch: dict, chunk: torch.Tensor
    ) -> torch.Tensor:
        """Surrogate flow-matching logp for π0.5.

        Same approach as Spirit adapter: average over T_EVAL random
        timesteps of -||v_θ(x_t, t) - v_target||^2.

        TODO: hook the actual π0.5 velocity field forward — the
        openpi Policy.infer doesn't directly expose it. Currently
        falls back to comparing against select_action output as a
        proxy (which is suboptimal but functional).
        """
        B = chunk.shape[0]
        device = self.device
        dtype = next(self.model.parameters()).dtype

        scores = []
        for _ in range(PI05_T_EVAL):
            t = torch.rand(B, device=device).to(dtype) * 0.999 + 0.001
            noise = torch.randn_like(chunk).to(device).to(dtype)
            chunk_d = chunk.to(device).to(dtype)
            t_expand = t.view(-1, 1, 1)
            x_t = (1.0 - t_expand) * noise + t_expand * chunk_d
            v_target = chunk_d - noise
            # Proxy: predict the chunk via openpi Policy and use the
            # difference from x_t as a velocity estimate. NOT the
            # actual flow-matching velocity field.
            v_pred = self._proxy_velocity(batch, x_t, t)
            mse = ((v_pred - v_target) ** 2).mean(dim=(-2, -1))
            scores.append(-mse)
        return torch.stack(scores, dim=0).mean(dim=0).float()

    def _proxy_velocity(
        self, batch: dict, x_t: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """Placeholder velocity field — returns Pi05's predicted action
        chunk. Not the actual v_θ but lets the pipeline run end-to-end."""
        B = x_t.shape[0]
        # Build a per-sample obs dict for openpi.Policy.infer
        actions_list = []
        for b in range(B):
            obs = self._batch_to_openpi_obs(batch, b)
            out = self.openpi_policy.infer(obs)
            actions_list.append(torch.from_numpy(np.asarray(out["actions"])))
        return torch.stack(actions_list).to(x_t.device).to(x_t.dtype)

    def _batch_to_openpi_obs(self, batch: dict, b_idx: int) -> dict:
        """Convert v1.5-style batch[b_idx] into openpi observation dict.

        openpi LIBERO config expects:
            observation/exterior_image_1_left:  (H, W, 3) uint8
            observation/wrist_image_left:        (H, W, 3) uint8 — for LIBERO
                                                  pi0_libero uses both
            state:                                 (8,) — per LIBERO data spec
            prompt:                                str
        """
        if "image" in batch:
            img = batch["image"][b_idx]
            if isinstance(img, torch.Tensor):
                img_np = (img.permute(1, 2, 0).cpu().clamp(0, 1).numpy()
                          * 255).astype(np.uint8)
            else:
                img_np = np.asarray(img)
        else:
            img_np = np.zeros((224, 224, 3), dtype=np.uint8)

        # We don't have wrist image in our v1.5 batch — duplicate exterior
        # (paper §3 acknowledges this limitation; π0.5 is most affected since
        # it's trained with wrist images).
        instruction = batch["instruction"][b_idx] if "instruction" in batch else ""

        return {
            "observation/exterior_image_1_left": img_np,
            "observation/wrist_image_left": img_np,
            "observation/joint_position": np.zeros(7, dtype=np.float32),
            "observation/gripper_position": np.zeros(1, dtype=np.float32),
            "prompt": instruction,
        }

    def policy_logp_with_ref(
        self, batch: dict, chunk: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        logp_cur = self.policy_logp(batch, chunk)
        if self.reference_state_dict is None:
            return logp_cur, logp_cur.detach()
        # Same swap-load-restore pattern as Spirit/OpenVLA
        cur_sd = {n: p.detach().clone() for n, p in self.model.named_parameters()
                  if p.requires_grad}
        with torch.no_grad():
            for n, p in self.model.named_parameters():
                if p.requires_grad and n in self.reference_state_dict:
                    p.data.copy_(self.reference_state_dict[n].to(p.device).to(p.dtype))
        logp_ref = self.policy_logp(batch, chunk)
        with torch.no_grad():
            for n, p in self.model.named_parameters():
                if p.requires_grad and n in cur_sd:
                    p.data.copy_(cur_sd[n])
        return logp_cur, logp_ref.detach()

    @torch.no_grad()
    def policy_sample(
        self, batch: dict, n_samples: int, deterministic: bool = False
    ) -> torch.Tensor:
        B = batch["image"].shape[0] if "image" in batch else 1
        chunks = []
        for k in range(n_samples):
            sample = []
            for b in range(B):
                obs = self._batch_to_openpi_obs(batch, b)
                # K different noise seeds via openpi's noise param
                noise = np.random.randn(PI05_ACTION_HORIZON,
                                        PI05_ACTION_DIM).astype(np.float32)
                out = self.openpi_policy.infer(obs, noise=noise)
                sample.append(torch.from_numpy(np.asarray(out["actions"])))
            chunks.append(torch.stack(sample))
        return torch.stack(chunks, dim=1).float()    # (B, K, T, A)

    @torch.no_grad()
    def select_action(self, batch: dict) -> torch.Tensor:
        B = batch["image"].shape[0] if "image" in batch else 1
        out_list = []
        for b in range(B):
            obs = self._batch_to_openpi_obs(batch, b)
            out = self.openpi_policy.infer(obs)
            out_list.append(torch.from_numpy(np.asarray(out["actions"])))
        return torch.stack(out_list).float()    # (B, T, A)

    # --------------- weight management --------------- #

    def trainable_parameters(self) -> Iterator[torch.nn.Parameter]:
        for p in self.model.parameters():
            if p.requires_grad:
                yield p

    def freeze_reference(self) -> None:
        self.reference_state_dict = {
            n: p.detach().clone().cpu()
            for n, p in self.model.named_parameters()
            if p.requires_grad
        }
        print(f"[pi05-adapter] frozen reference: "
              f"{len(self.reference_state_dict)} param tensors")

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {n: p.detach().cpu() for n, p in self.model.named_parameters()
             if p.requires_grad},
            path,
        )

    def load(self, path: str) -> None:
        sd = torch.load(path, map_location="cpu")
        own = self.model.state_dict()
        for n, v in sd.items():
            if n in own:
                own[n].data.copy_(v.to(own[n].device).to(own[n].dtype))
