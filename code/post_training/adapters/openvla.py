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
                "instruction": List[str] - language instruction per sample
                "image_uint8": List[np.ndarray] (preferred) - raw uint8 (H, W, 3)
                "image": (B, 3, H, W) float tensor in [0, 1] (back-compat)
            chunk: (B, T, 7) float actions in CONTINUOUS unnormalized
                   units (i.e. real EE delta — e.g. [-0.5, 0.5] meters).
                   Internally normalized via stored q01/q99 stats then
                   discretized into 256 bins matching OpenVLA's
                   training-time tokenization.

        Returns:
            (B,) log-prob estimate. Sum over T*7 action tokens.

        Implementation notes
        --------------------
        - Uses teacher-forcing forward (NOT model.generate) so it's
          unaffected by the transformers >= 4.50 generate degeneration
          bug that broke select_action.
        - Inserts the magic token 29871 to match training-time tokenization
          (same insertion as predict_action and our _manual_greedy_predict_action).
        - Restricts logits to action-token range when computing log_softmax
          so the probabilities are over the action vocabulary only,
          giving more meaningful logp values across (chosen, rejected)
          pairs in DPO.
        """
        B, T, A = chunk.shape
        assert A == OPENVLA_ACTION_DIM, f"OpenVLA expects 7-DoF actions, got {A}"

        from PIL import Image

        # 1. Continuous → bin token via stored q01/q99 (REVERSE of predict_action's
        #    bin → continuous decode, lines 521-535).
        stats = self.model.get_action_stats(self.unnorm_key)
        action_low = np.array(stats["q01"])         # (7,)
        action_high = np.array(stats["q99"])        # (7,)
        # Normalize to [-1, 1] (forward of predict_action's
        # 0.5 * (norm + 1) * (high - low) + low)
        chunk_np = chunk.detach().cpu().numpy()     # (B, T, 7)
        # avoid div-by-zero on dims where q99==q01
        denom = np.where(action_high - action_low > 1e-6,
                         action_high - action_low, 1.0)
        normalized = 2 * (chunk_np - action_low) / denom - 1
        normalized = np.clip(normalized, -1.0, 1.0)
        # Map to bin index in [0, 255]
        bin_ids = np.clip(
            np.digitize(normalized, self.model.bins) - 1,
            0, OPENVLA_NUM_BINS - 1,
        )
        # Token id: predict_action L522 says
        #   discretized_actions = vocab_size - predicted_action_token_ids
        # so reverse:
        #   token_id = vocab_size - bin_id - 1   (bin_id goes 0..255)
        # but predict_action then clips with `discretized - 1`. Account for the
        # +1 offset by using bin_id+1 here, matching the inverse map exactly.
        token_ids_np = self.model.vocab_size - (bin_ids + 1)
        token_ids_t = torch.as_tensor(token_ids_np, dtype=torch.long, device=self.device)

        instructions = batch["instruction"]
        if "image_uint8" in batch:
            uint8_imgs = batch["image_uint8"]
        else:
            uint8_imgs = None

        log_probs = []
        for b in range(B):
            prompt = self._format_prompt(instructions[b])
            if uint8_imgs is not None:
                pil_img = Image.fromarray(uint8_imgs[b]).convert("RGB")
                pil_img = self._center_crop_and_resize(pil_img, crop_scale=0.9)
            else:
                pil_img = self._tensor_to_pil(batch["image"][b])

            inputs = self.processor(prompt, pil_img, return_tensors="pt").to(
                self.device, dtype=torch.bfloat16
            )
            input_ids = inputs["input_ids"]   # (1, L_prompt)
            pixel_values = inputs["pixel_values"]

            # Insert magic 29871 token (matches predict_action L512-515 and
            # our _manual_greedy_predict_action)
            if not torch.all(input_ids[:, -1] == 29871):
                magic = torch.tensor([[29871]], dtype=input_ids.dtype, device=self.device)
                input_ids = torch.cat([input_ids, magic], dim=1)

            # Append action tokens for teacher-forcing
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
            logits = outputs.logits[0]              # (L, V)
            n_action = action_tokens_flat.numel()
            # Logits at position i predict token i+1; action tokens are
            # the last n_action positions of full_input, so the relevant
            # prediction logits live at positions [L-n_action-1 .. L-2].
            pred_logits = logits[-n_action - 1: -1]   # (n_action, V)

            # Restrict to action-token range before softmax — same trick
            # as in _manual_greedy_predict_action. Otherwise the softmax
            # is dominated by non-action vocabulary positions and the
            # action-token probabilities become tiny / unstable.
            n_bins = self.model.bin_centers.shape[0] + 1   # 256
            allowed_lo = self.model.vocab_size - n_bins
            mask = torch.full_like(pred_logits, float("-inf"))
            mask[:, allowed_lo:] = 0.0
            masked_logits = pred_logits + mask

            log_p_all = torch.log_softmax(masked_logits, dim=-1)
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
        """Single-step prediction. Returns (B, 7).

        Bypasses `model.predict_action` (which calls `model.generate()`,
        which DEGENERATES on OpenVLA under transformers >= 4.50, always
        returning bin 127 for every action dim). Instead we run a manual
        greedy forward pass: 7 autoregressive steps, argmax at each step.

        Image preprocessing matches OpenVLA's official `get_vla_action`
        as closely as possible without TF (TF clashes with libero env
        in the same process):
        - Image arrives as raw uint8 via `batch["image_uint8"]`
          (skipping the float→uint8 round-trip which OOD's the encoder)
        - Center crop scale=0.9 via cv2.warpAffine (~2.6 / 255 vs
          tf.image.crop_and_resize)
        - processor + .to(bf16) cast

        Stochastic sampling (for K-candidate rollout): set
        ``self._sample_do_sample = True`` and tune temperature/top_p
        via ``self._sample_temperature`` / ``self._sample_top_p``.
        """
        from PIL import Image

        do_sample = getattr(self, "_sample_do_sample", False)
        temperature = float(getattr(self, "_sample_temperature", 1.0))
        top_p = float(getattr(self, "_sample_top_p", 0.95))

        # Prefer uint8 path; fall back to tensor for back-compat
        if "image_uint8" in batch:
            B = len(batch["image_uint8"])
            uint8_imgs = batch["image_uint8"]
        else:
            B = batch["image"].shape[0]
            uint8_imgs = None

        out_list = []
        for b in range(B):
            prompt = self._format_prompt(batch["instruction"][b])

            if uint8_imgs is not None:
                pil_img = Image.fromarray(uint8_imgs[b]).convert("RGB")
                pil_img = self._center_crop_and_resize(pil_img, crop_scale=0.9)
            else:
                pil_img = self._tensor_to_pil(batch["image"][b])

            inputs = self.processor(prompt, pil_img, return_tensors="pt").to(
                self.device, dtype=torch.bfloat16
            )

            action = self._manual_greedy_predict_action(
                inputs, do_sample=do_sample, temperature=temperature, top_p=top_p,
            )
            out_list.append(torch.as_tensor(action, dtype=torch.float32))
        return torch.stack(out_list).to(self.device)

    def _manual_greedy_predict_action(
        self, inputs: dict, do_sample: bool = False,
        temperature: float = 1.0, top_p: float = 0.95,
    ) -> "np.ndarray":
        """Manual replacement for model.predict_action — works around
        the transformers >= 4.50 model.generate() degeneration bug for
        OpenVLA where every action dim collapses to bin 127.

        Reproduces `OpenVLAForActionPrediction.predict_action` step by
        step, but uses 7 manual forward + argmax (or sampled) calls
        instead of model.generate().

        Returns (action_dim,) numpy float32 array, already unnormalized
        via the model's stored norm_stats.
        """
        import numpy as np

        # Magic token insertion (matches predict_action line 512-515)
        input_ids = inputs["input_ids"]
        if not torch.all(input_ids[:, -1] == 29871):
            magic = torch.tensor([[29871]], dtype=input_ids.dtype, device=input_ids.device)
            input_ids = torch.cat([input_ids, magic], dim=1)

        action_dim = self.model.get_action_dim(self.unnorm_key)
        token_ids: list[int] = []

        cur = input_ids
        for step in range(action_dim):
            attn = torch.ones_like(cur)
            with torch.no_grad():
                out = self.model(
                    input_ids=cur,
                    attention_mask=attn,
                    pixel_values=inputs["pixel_values"],
                )
            logits = out.logits[0, -1, :]   # (vocab,)

            # Restrict sampling to action-token range
            # OpenVLA action tokens occupy [vocab_size - n_bins, vocab_size - 1]
            n_bins = self.model.bin_centers.shape[0] + 1   # 256
            allowed_lo = self.model.vocab_size - n_bins
            allowed_hi = self.model.vocab_size
            mask = torch.full_like(logits, float("-inf"))
            mask[allowed_lo:allowed_hi] = 0.0
            logits = logits + mask

            if do_sample:
                logits = logits / max(temperature, 1e-6)
                probs = torch.softmax(logits, dim=-1)
                # top-p (nucleus) filtering
                if 0.0 < top_p < 1.0:
                    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
                    cum = torch.cumsum(sorted_probs, dim=-1)
                    cutoff = (cum > top_p).nonzero()
                    if cutoff.numel() > 0:
                        cutoff_idx = cutoff[0].item()
                        sorted_probs[cutoff_idx + 1:] = 0.0
                    probs = torch.zeros_like(probs)
                    probs.scatter_(0, sorted_idx, sorted_probs)
                    probs = probs / probs.sum().clamp(min=1e-9)
                next_id = torch.multinomial(probs, 1)
            else:
                next_id = logits.argmax().unsqueeze(0)

            token_ids.append(int(next_id.item()))
            cur = torch.cat([cur, next_id.unsqueeze(0)], dim=1)

        # Decode bin → continuous action (matches predict_action lines 521-535)
        predicted = np.array(token_ids)
        discretized = self.model.vocab_size - predicted
        discretized = np.clip(
            discretized - 1, a_min=0, a_max=self.model.bin_centers.shape[0] - 1,
        )
        normalized = self.model.bin_centers[discretized]

        stats = self.model.get_action_stats(self.unnorm_key)
        mask_arr = stats.get("mask", np.ones_like(stats["q01"], dtype=bool))
        action_high = np.array(stats["q99"])
        action_low = np.array(stats["q01"])
        actions = np.where(
            mask_arr,
            0.5 * (normalized + 1) * (action_high - action_low) + action_low,
            normalized,
        )
        return actions.astype(np.float32)

    @staticmethod
    def _center_crop_and_resize(img: "Image.Image", crop_scale: float = 0.9) -> "Image.Image":
        """Center crop at scale=crop_scale (area), resize back to original
        size. Matches `tf.image.crop_and_resize` (which OpenVLA's official
        eval uses) numerically — uses cv2.warpAffine with continuous
        sub-pixel coordinates instead of PIL's integer-pixel crop.

        Pixel diff vs tf.image.crop_and_resize: ~2.6 / 255 (vs PIL crop's
        ~39 / 255, which is enough to OOD the visual encoder).
        """
        import cv2
        import numpy as np
        from PIL import Image

        arr = np.array(img)
        H, W = arr.shape[:2]
        # Per OpenVLA comment: new_h = orig_h * sqrt(crop_scale)
        scale_factor = crop_scale ** 0.5
        # tf.image.crop_and_resize works in normalized box coords [0,1]
        # and outputs (224,224) via bilinear sampling. Equivalent affine:
        # zoom by 1/scale around image center.
        zoom = 1.0 / scale_factor
        cx, cy = W / 2.0, H / 2.0
        M = np.float32([
            [zoom, 0,    cx * (1 - zoom)],
            [0,    zoom, cy * (1 - zoom)],
        ])
        out = cv2.warpAffine(arr, M, (W, H), flags=cv2.INTER_LINEAR)
        return Image.fromarray(out)

    def _format_prompt(self, instruction: str) -> str:
        """Format LIBERO-style prompt the way OpenVLA expects.

        Matches openvla/experiments/robot/openvla_utils.py:163 — the
        instruction is lowercased before insertion. Critical for
        finetuned LIBERO ckpts to stay in distribution.
        """
        return f"In: What action should the robot take to {instruction.lower()}?\nOut:"

    @classmethod
    def _tensor_to_pil(cls, t: torch.Tensor):
        """(3,H,W) float in [0,1] → PIL.Image, with mandatory center crop
        for OpenVLA finetuned LIBERO ckpts.

        Legacy path used when the batch only contains a float tensor
        (no `image_uint8`). The float→uint8 conversion here introduces
        ~1/255 quantisation drift vs OpenVLA's official PIL pipeline,
        which is enough to OOD the visual encoder. Whenever possible,
        callers should pass uint8 images via `batch["image_uint8"]`
        and skip this function.
        """
        from PIL import Image
        import numpy as np
        arr = (t.detach().cpu().permute(1, 2, 0).clamp(0, 1).numpy() * 255).astype("uint8")
        pil = Image.fromarray(arr)
        return cls._center_crop_and_resize(pil, crop_scale=0.9)

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
