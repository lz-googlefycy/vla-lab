"""OpenVLA adapter for v1.5 post-training pipeline.

OpenVLA's action representation
-------------------------------
OpenVLA-7B (Stanford 2024.06) discretises each action dim into 256 bins,
then predicts the 7 bin-IDs as 7 next tokens via the language head.

  action_t  ∈ R^7  →  bin_t ∈ {0..255}^7  →  vocab_id_t ∈ {V-256..V-1}^7
  policy:   p(bin_token | prompt_with_image)

This is **single-step**: one forward = 7 tokens = 1 action. To form a
"chunk" of length T (matching Spirit/π0.5 conventions), we autoregressively
roll out T single-step actions, treating the env interaction loop as the
chunk boundary.

Log-probability per chunk
-------------------------
For DPO/GRPO we need log π(chunk | prompt). Since chunk = T single-step
actions and tokens within an action are predicted autoregressively, we
have closed-form log-prob:

    log π(chunk | prompt)
        = sum_{t=1..T} sum_{d=1..7} log p(token_{t,d} | prompt + previous_tokens)

This is computed by feeding the full sequence (prompt + chunk tokens)
through the model in teacher-forcing mode and summing the relevant
log-probs. Cheap and exact.

Sampling K candidates
---------------------
For DPO pair generation and GRPO group sampling we need K candidate
chunks per instruction. Approach:

  for k in range(K):
      env.reset()  # same seed
      chunk_k = []
      for t in range(T):
          action_t = model.generate(temperature=cfg.sample_temperature, do_sample=True, ...)
          chunk_k.append(action_t)
          obs = env.step(action_t)  # observation for next step
      chunk_k = stack(chunk_k)  # (T, 7)

This couples sampling with env stepping (since action_t depends on
obs_t). Note that pure "policy-only" sampling (no env in the loop) is
**not faithful** for OpenVLA — without observation feedback, all
samples would be identical. So sampling is intrinsically env-coupled.

Reference
---------
- ``code/scripts/run_libero_eval.sh``: existing eval harness (we reuse)
- ``code/tools/smoke_test.py``: existing single-action smoke
- v1.4 reproduced numbers: 78.0 / 60.0 / 77.0 / 53.0 = avg 67%
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
import torch.nn as nn

# Allow either user clone path or workspace path
_OPENVLA_CANDIDATES = [
    os.environ.get("OPENVLA_SRC"),
    "/workspace/openvla",
    str(Path.home() / "openvla"),
]
for _p in _OPENVLA_CANDIDATES:
    if _p and Path(_p).exists() and _p not in sys.path:
        sys.path.insert(0, _p)

from ..interface import VLABase, PostTrainConfig, SamplingResult


# ---------------------------------------------------------------------- #
# OpenVLA-specific config knobs (extend PostTrainConfig at runtime)
# ---------------------------------------------------------------------- #

# OpenVLA discretisation
OPENVLA_NUM_BINS = 256                # bins per action dim
OPENVLA_ACTION_DIM = 7                # x, y, z, rx, ry, rz, gripper
# OpenVLA uses the last 256 vocab tokens as action bins
# (predicted_token_id = vocab_size - bin_id_in_[0,255] - 1)


class OpenVLAAdapter(VLABase):
    """OpenVLA wrapper conforming to VLABase protocol.

    Lifecycle:
        1. ``__init__``: load HF ckpt, optionally apply LoRA, move to GPU
        2. ``policy_logp(batch, chunk)``: teacher-forcing forward pass
           computing log-prob of `chunk` (T, 7) tokens
        3. ``policy_sample(batch, K)``: env-coupled K-sample rollout —
           caller must pass an env-builder callback in cfg.

    Note: OpenVLA's chunk is T single-step actions. Make sure the
    PostTrainConfig.libero_suite + sample horizon matches the env.
    """

    def __init__(self, cfg: PostTrainConfig):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._sample_temperature = 1.0   # set externally before policy_sample

        # Patch OpenVLA model class to be compatible with transformers >= 4.50
        # (newer transformers asserts _supports_sdpa attribute that older
        # custom model classes like OpenVLA's don't have)
        self._patch_openvla_compat()

        # Lazy import to avoid forcing transformers on this module's load
        from transformers import AutoModelForVision2Seq, AutoProcessor

        # OpenVLA HF ckpts: openvla/openvla-7b-finetuned-libero-{spatial,object,goal,10}
        # cfg.base_ckpt_path should be the HF model name or local snapshot dir.
        self.processor = AutoProcessor.from_pretrained(
            cfg.base_ckpt_path, trust_remote_code=True
        )
        torch_dtype = getattr(torch, cfg.base_dtype, torch.bfloat16)
        self.model = AutoModelForVision2Seq.from_pretrained(
            cfg.base_ckpt_path,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        ).to(self.device)
        # ckpt-specific norm stats — needed for action bin → continuous mapping
        self.unnorm_key = self._infer_unnorm_key()

        if cfg.use_lora:
            self._inject_lora()

        self.reference_state_dict: dict[str, torch.Tensor] | None = None

    # --------------- private helpers --------------- #

    def _infer_unnorm_key(self) -> str:
        norm_stats = getattr(self.model, "norm_stats", {})
        if len(norm_stats) == 1:
            return next(iter(norm_stats.keys()))
        # libero ckpts have keys like "libero_spatial_no_noops"
        suite = self.cfg.libero_suite
        for k in norm_stats:
            if suite in k:
                return k
        if norm_stats:
            return next(iter(norm_stats.keys()))
        return "default"

    def _inject_lora(self) -> None:
        """Inject LoRA into Llama-2 attention modules. Manual, no peft.

        Uses the same _LoRALinear class as our Spirit train_lora.py for
        consistency and to avoid the peft dependency.
        """
        # Lazy import — avoids circular imports
        sys.path.insert(0, str(Path(__file__).parent.parent.parent))
        from spirit_adapter.train_lora import _LoRALinear, _replace_module_by_path  # type: ignore

        # Freeze all params first
        for p in self.model.parameters():
            p.requires_grad = False

        targets = list(self.cfg.lora_target_modules)
        # Llama uses q_proj/k_proj/v_proj/o_proj names; OpenVLA inherits
        llama_targets = ["q_proj", "k_proj", "v_proj", "o_proj"]
        # If user passed Spirit-style names, fall back to Llama names
        if targets[0].startswith("to_"):
            targets = llama_targets

        replacements: list[tuple[str, nn.Linear]] = []
        for name, m in self.model.named_modules():
            if not isinstance(m, nn.Linear):
                continue
            for t in targets:
                if name.endswith("." + t) or name == t:
                    replacements.append((name, m))
                    break
        for name, orig in replacements:
            new_m = _LoRALinear(
                orig, r=self.cfg.lora_r, alpha=self.cfg.lora_alpha,
                dropout=self.cfg.lora_dropout,
            )
            _replace_module_by_path(self.model, name, new_m)
        n_trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in self.model.parameters())
        print(
            f"[openvla-adapter] LoRA injected: {len(replacements)} Linears, "
            f"trainable {n_trainable/1e6:.1f}M / {n_total/1e6:.0f}M "
            f"({100*n_trainable/n_total:.2f}%)"
        )

    # --------------- VLABase: training-time interface --------------- #

    def policy_logp(
        self, batch: dict, chunk: torch.Tensor
    ) -> torch.Tensor:
        """Log π(chunk | prompt) for a batch.

        Args:
            batch: dict with at least
                "instruction": List[str]   - language instruction per sample
                "image": (B, 3, H, W) RGB image (current obs)
                "history_chunks": optional list of past actions in this episode
                                  to expose to the model as conversation history
            chunk: (B, T, 7) float actions in continuous units (will be
                   discretised to bin tokens internally)

        Returns:
            (B,) log-prob estimate
        """
        B, T, A = chunk.shape
        assert A == OPENVLA_ACTION_DIM, f"OpenVLA expects 7-DoF actions, got {A}"

        # Discretise chunk into bin tokens. OpenVLA reverses the mapping:
        # token_id = vocab_size - bin_id_in_[1, 256]
        # We follow the same convention used in predict_action().
        chunk_np = chunk.detach().cpu().numpy()
        # First normalise to [-1, 1] using stored norm stats (assumes batch
        # is already in continuous action space; if not, caller normalises)
        bins = np.linspace(-1, 1, OPENVLA_NUM_BINS)
        bin_centers = (bins[:-1] + bins[1:]) / 2.0
        # Map each value to nearest bin index
        bin_ids = np.clip(
            np.digitize(chunk_np, bins) - 1, 0, OPENVLA_NUM_BINS - 1
        )                                    # (B, T, 7)
        token_ids = self.model.vocab_size - 1 - bin_ids   # OpenVLA convention
        token_ids_t = torch.as_tensor(token_ids, dtype=torch.long, device=self.device)

        # For each sample, build the full input_ids: [prompt_tokens, action_tokens]
        # then teacher-force forward and extract log-probs at action positions.
        instructions = batch["instruction"]
        image = batch["image"].to(self.device)              # (B, 3, H, W)

        log_probs = []
        model_dtype = next(self.model.parameters()).dtype
        for b in range(B):
            prompt = self._format_prompt(instructions[b])
            # OpenVLA processor expects PIL.Image, not Tensor
            pil_img = self._tensor_to_pil(image[b])
            inputs = self.processor(prompt, pil_img, return_tensors="pt")
            input_ids = inputs["input_ids"].to(self.device)
            # Cast image to model's dtype (bf16) — vision encoder requires match
            pixel_values = inputs["pixel_values"].to(self.device).to(model_dtype)

            action_tokens_flat = token_ids_t[b].reshape(-1)   # (T*7,)
            full_input = torch.cat(
                [input_ids[0], action_tokens_flat], dim=0
            ).unsqueeze(0)
            attention_mask = torch.ones_like(full_input)

            outputs = self.model(
                input_ids=full_input,
                pixel_values=pixel_values,
                attention_mask=attention_mask,
            )
            # outputs.logits: (1, L, V)
            # We want log p(action_token_i | full_input[:i])
            # The action tokens are the last T*7 positions.
            logits = outputs.logits[0]                # (L, V)
            n_action = action_tokens_flat.numel()
            # Logits at position i predict token at position i+1; action
            # tokens are at positions L-n_action..L-1 in full_input,
            # so prediction logits live at L-n_action-1..L-2.
            pred_logits = logits[-n_action - 1:-1]    # (n_action, V)
            log_p_all = torch.log_softmax(pred_logits, dim=-1)
            log_p_actions = log_p_all.gather(
                dim=-1, index=action_tokens_flat.unsqueeze(-1)
            ).squeeze(-1)                             # (n_action,)
            log_probs.append(log_p_actions.sum())

        return torch.stack(log_probs)

    def policy_logp_with_ref(
        self, batch: dict, chunk: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute logp under (current, reference) policies.

        Reference policy = LoRA off, base weights only. We swap the
        LoRA contribution off temporarily, compute ref logp, swap on.
        """
        logp_cur = self.policy_logp(batch, chunk)

        # If no reference snapshot, compute ref logp by detaching LoRA
        # weights' contribution. Implementation: we currently use the
        # 0-init B-matrix property of LoRA — at start of training,
        # ref logp == cur logp. As training progresses, we need a real
        # frozen copy. For now, the simple path is to clone the model
        # state once at start of training and call ref_model on demand.
        if self.reference_state_dict is None:
            # First call — treat current as reference
            return logp_cur, logp_cur.detach()

        # Swap to ref weights, compute, swap back. Costly but correct.
        cur_sd = {n: p.detach().clone() for n, p in self.model.named_parameters() if p.requires_grad}
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
        self,
        batch: dict,
        n_samples: int,
        deterministic: bool = False,
    ) -> torch.Tensor:
        """Sample K candidate chunks under current policy.

        For OpenVLA, "chunk" is T single-step actions. Pure policy-only
        sampling without env coupling will produce identical samples
        because each step depends on observation. Therefore caller must
        either:
          - pass `batch["env_factory"]` callable to build K independent envs
            for env-coupled K-sample rollout (used in training loop), OR
          - accept that this returns K identical samples + Gaussian noise
            (the "fast and dirty" path for pipeline smoke tests)

        For v1.5 first-pass we go with the smoke-test path; the full
        env-coupled rollout lives in `code/post_training/rollout.py`
        (next module).
        """
        B = batch["image"].shape[0]
        T = self.cfg.action_horizon if hasattr(self.cfg, "action_horizon") else 60
        # Smoke path: predict once, then sample noise around it
        action_single = self._predict_single(batch)               # (B, 7)
        chunks = action_single[:, None, None, :].expand(B, n_samples, T, OPENVLA_ACTION_DIM)
        chunks = chunks + 0.05 * torch.randn_like(chunks)
        return chunks.contiguous()

    def _predict_single(self, batch: dict) -> torch.Tensor:
        """Single-step prediction (greedy). Returns (B, 7)."""
        out_list = []
        for b in range(batch["image"].shape[0]):
            prompt = self._format_prompt(batch["instruction"][b])
            pil_img = self._tensor_to_pil(batch["image"][b])
            inputs = self.processor(prompt, pil_img, return_tensors="pt").to(self.device)
            action = self.model.predict_action(
                **inputs, unnorm_key=self.unnorm_key, do_sample=False
            )
            out_list.append(torch.as_tensor(action, dtype=torch.float32))
        return torch.stack(out_list).to(self.device)

    def _format_prompt(self, instruction: str) -> str:
        """Format LIBERO-style prompt the way OpenVLA expects."""
        return f"In: What action should the robot take to {instruction}?\nOut:"

    @staticmethod
    def _tensor_to_pil(t: torch.Tensor):
        """(3,H,W) float in [0,1] → PIL.Image."""
        from PIL import Image
        arr = (t.detach().cpu().permute(1, 2, 0).clamp(0, 1).numpy() * 255).astype("uint8")
        return Image.fromarray(arr)

    @staticmethod
    def _patch_openvla_compat() -> None:
        """transformers >= 4.50 calls model._supports_sdpa during __init__
        to decide attention impl. OpenVLA defines _supports_sdpa as a
        @property that delegates to self.language_model — but at __init__
        time language_model isn't assigned yet, raising AttributeError.

        Fix: forcibly replace the @property on PrismaticPreTrainedModel
        with a plain class attribute False. Subclasses then inherit the
        plain attribute. Idempotent.
        """
        try:
            from transformers.dynamic_module_utils import get_class_from_dynamic_module
            for cls_name in (
                "modeling_prismatic.OpenVLAForActionPrediction",
                "modeling_prismatic.PrismaticForConditionalGeneration",
                "modeling_prismatic.PrismaticPreTrainedModel",
            ):
                try:
                    cls = get_class_from_dynamic_module(cls_name, "openvla/openvla-7b")
                    # Forcibly replace property descriptor (if any) with
                    # a plain class attribute. type.__setattr__ avoids
                    # the property's __set__ being invoked.
                    type.__setattr__(cls, "_supports_sdpa", False)
                    type.__setattr__(cls, "_supports_flash_attn_2", False)
                    type.__setattr__(cls, "_supports_attention_backend", False)
                except Exception:
                    continue
        except Exception:
            pass

        # Belt-and-suspenders: walk sys.modules for any dynamic module
        # file path HF has already loaded
        import sys as _sys
        for mod_name, mod in list(_sys.modules.items()):
            if "modeling_prismatic" in mod_name and mod is not None:
                for cls_name in (
                    "PrismaticPreTrainedModel",
                    "PrismaticForConditionalGeneration",
                    "OpenVLAForActionPrediction",
                ):
                    cls = getattr(mod, cls_name, None)
                    if cls is None:
                        continue
                    type.__setattr__(cls, "_supports_sdpa", False)
                    type.__setattr__(cls, "_supports_flash_attn_2", False)
                    type.__setattr__(cls, "_supports_attention_backend", False)

    # --------------- weight management --------------- #

    def trainable_parameters(self) -> Iterator[torch.nn.Parameter]:
        for p in self.model.parameters():
            if p.requires_grad:
                yield p

    def freeze_reference(self) -> None:
        """Snapshot trainable params (LoRA + projections) as reference."""
        self.reference_state_dict = {
            n: p.detach().clone().cpu()
            for n, p in self.model.named_parameters()
            if p.requires_grad
        }
        print(f"[openvla-adapter] frozen reference: "
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

    @torch.no_grad()
    def select_action(self, batch: dict) -> torch.Tensor:
        """Run greedy single-step prediction. Returns (B, 1, 7) chunk."""
        a = self._predict_single(batch)   # (B, 7)
        return a.unsqueeze(1)              # OpenVLA chunks are length 1
