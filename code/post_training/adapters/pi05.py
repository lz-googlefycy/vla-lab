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
PI05_ACTION_DIM = 7         # LIBERO robot action dim
PI05_MODEL_ACTION_DIM = 32  # pi0.5 internal "universal action space"; we pad
                            # 7-dim LIBERO chunks to 32 with zeros for forward,
                            # and slice [:, :, :7] when consuming sample_actions
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

        # We bypass openpi.policies.policy_config.create_trained_policy because
        # it pulls in lerobot.common.* which is removed in lerobot >= 0.4. Our
        # PyTorch model + safetensors weights load just fine without it.
        try:
            from openpi.models import pi0_config
            from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
        except ImportError as e:
            raise ImportError(
                f"openpi import failed: {e}. Make sure openpi is on PYTHONPATH "
                "(set OPENPI_SRC env or place under "
                "/e2e-data/users/liuzhi7/vla_workspace/openpi)."
            ) from e

        # Build PI0Pytorch with pi05 LIBERO defaults
        # (pi05_libero TrainConfig uses these — see openpi/training/config.py)
        pt_cfg = pi0_config.Pi0Config(
            pi05=True, action_horizon=10, discrete_state_input=False,
        )
        self.model = PI0Pytorch(pt_cfg)

        # Load weights from converted safetensors if base_ckpt_path is a
        # directory containing model.safetensors. Random init otherwise.
        from pathlib import Path as _Path
        ckpt_dir = _Path(cfg.base_ckpt_path)
        weight_file = ckpt_dir / "model.safetensors"
        if weight_file.exists():
            import safetensors.torch
            safetensors.torch.load_model(self.model, str(weight_file), device="cpu")
            print(f"[pi05-adapter] loaded weights from {weight_file}")
        else:
            print(f"[pi05-adapter] WARNING: {weight_file} missing; "
                  "model is RANDOM-INITIALISED (smoke-test only)")

        self.model = self.model.to(self.device)
        # pi0.5 uses bf16 for paligemma + expert (mirrors openpi convention)
        self.model.paligemma_with_expert.to_bfloat16_for_selected_params("bfloat16")
        # Stash a no-op handle so legacy code paths that read self.openpi_policy
        # don't crash (no longer used for anything).
        self.openpi_policy = None

        # Load norm_stats so we can unnormalize action chunks back into raw
        # LIBERO robot commands. Without this, env.step() gets normalised
        # [-1, 1] values which cause 0% success.
        # norm_stats search order:
        #   1. {ckpt_dir}/assets/physical-intelligence/libero/norm_stats.json
        #   2. {ckpt_dir}/../pi05_libero_jax/assets/.../norm_stats.json (sibling)
        #   3. /e2e-data/users/liuzhi7/vla_workspace/pi05_libero_jax/.../norm_stats.json
        self._action_norm = self._load_norm_stats(ckpt_dir)
        if self._action_norm is None:
            print("[pi05-adapter] WARNING: norm_stats not found — actions will "
                  "stay in normalised [-1, 1] (env.step will likely fail).")
        else:
            print("[pi05-adapter] loaded action norm_stats "
                  f"(q01={self._action_norm['q01'][:3].tolist()}..., "
                  f"q99={self._action_norm['q99'][:3].tolist()}...)")

        # LoRA (experimental for pi0.5 — same scheme as openvla / spirit)
        if cfg.use_lora:
            print("[pi05-adapter] LoRA injection (experimental)")
            self._inject_lora_experimental()

    def _resolve_openpi_config(self) -> str:
        """Map our PostTrainConfig.libero_suite → openpi config name."""
        suite = self.cfg.libero_suite
        if suite in ("spatial", "object", "goal", "long10", "all4"):
            return "pi05_libero"
        raise ValueError(f"unknown LIBERO suite: {suite}")

    @staticmethod
    def _load_norm_stats(ckpt_dir):
        """Locate + load openpi norm_stats.json (action q01/q99 for unnorm)."""
        from pathlib import Path as _P
        candidates = [
            ckpt_dir / "assets/physical-intelligence/libero/norm_stats.json",
            ckpt_dir.parent / "pi05_libero_jax/assets/physical-intelligence/libero/norm_stats.json",
            _P("/e2e-data/users/liuzhi7/vla_workspace/pi05_libero_jax/assets/physical-intelligence/libero/norm_stats.json"),
            _P.home() / "pi_assets/pi05_libero/assets/physical-intelligence/libero/norm_stats.json",
        ]
        for path in candidates:
            if path.exists():
                import json
                with open(path) as f:
                    raw = json.load(f)
                a = raw["norm_stats"]["actions"]
                return {
                    "q01": np.asarray(a["q01"], dtype=np.float32),  # (7,)
                    "q99": np.asarray(a["q99"], dtype=np.float32),  # (7,)
                    "mean": np.asarray(a["mean"], dtype=np.float32),
                    "std": np.asarray(a["std"], dtype=np.float32),
                }
        return None

    def _unnormalize_actions(self, actions_norm):
        """Convert normalised actions [-1,1] -> raw robot commands.

        openpi uses Quantile normalisation: x_norm = (x - q01) * 2/(q99-q01) - 1
        Inverse: x_raw = (x_norm + 1) * (q99 - q01) / 2 + q01

        Note: openpi excludes the gripper (last dim) from normalisation; we
        match that convention. Gripper stays in [-1, +1] / {-1, +1}.
        """
        if self._action_norm is None:
            return actions_norm
        # actions_norm shape: (..., 7); torch tensor on any device
        if isinstance(actions_norm, torch.Tensor):
            q01 = torch.from_numpy(self._action_norm["q01"]).to(actions_norm.device)
            q99 = torch.from_numpy(self._action_norm["q99"]).to(actions_norm.device)
            scale = (q99 - q01) / 2
            offset = (q99 + q01) / 2
            out = actions_norm.clone()
            # First 6 dims: position+rotation, normalised → unnormalize
            out[..., :6] = actions_norm[..., :6] * scale[:6] + offset[:6]
            # Gripper (dim 6): pass-through (LIBERO env binarises {-1,+1})
            return out
        else:
            arr = np.asarray(actions_norm)
            q01 = self._action_norm["q01"]
            q99 = self._action_norm["q99"]
            scale = (q99 - q01) / 2
            offset = (q99 + q01) / 2
            out = arr.copy()
            out[..., :6] = arr[..., :6] * scale[:6] + offset[:6]
            return out

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

        Direct hook into PI0Pytorch.forward(observation, actions) which
        returns per-element MSE between predicted velocity v_t and target
        velocity u_t = noise - actions. The mean MSE over (T_action × A)
        averaged over PI05_T_EVAL noise samples is our surrogate −logp.

        We negate so that "higher logp = lower MSE = better fit" (DPO/GRPO
        compatible). The averaging over noise/time samples reduces variance.

        NOTE: PI0Pytorch.forward samples both noise and time internally
        each call. We rely on monte-carlo averaging across PI05_T_EVAL calls.
        """
        from openpi.models.pi0_config import Pi0Config  # noqa: F401  ensure path

        # Build openpi Observation tensor batch
        observation = self._batch_to_openpi_observation(batch, chunk.shape[0])
        # actions: (B, T, A=7) — chunk is in normalised [-1, 1] action space.
        # pi0.5's model.forward expects action_dim=32 (universal action space);
        # we pad with zeros for unused dims (LIBERO uses 7-dim slice).
        dtype = next(self.model.parameters()).dtype
        chunk_dev = chunk.to(self.device).to(dtype)              # (B, T, 7)
        B, T, _ = chunk_dev.shape
        if PI05_MODEL_ACTION_DIM > PI05_ACTION_DIM:
            pad = torch.zeros(B, T, PI05_MODEL_ACTION_DIM - PI05_ACTION_DIM,
                              dtype=dtype, device=self.device)
            actions = torch.cat([chunk_dev, pad], dim=-1)        # (B, T, 32)
        else:
            actions = chunk_dev

        scores = []
        for _ in range(PI05_T_EVAL):
            # PI0Pytorch.forward returns per-element MSE (B, T, 32)
            # We only score the first 7 dims (LIBERO action dims) — the
            # padded dims contribute meaningless MSE that would dilute
            # the signal.
            mse = self.model(observation, actions)               # (B, T, 32)
            mse = mse[..., :PI05_ACTION_DIM]                      # (B, T, 7)
            score = -mse.mean(dim=(-2, -1))                       # (B,)
            scores.append(score)
        return torch.stack(scores, dim=0).mean(dim=0).float()

    def _batch_to_openpi_obs(self, batch: dict, b_idx: int) -> dict:
        """Convert v1.5-style batch[b_idx] into openpi inference dict.

        openpi LIBERO LiberoInputs transform expects:
            observation/state:        (8,) — robot proprioception (joints+gripper)
            observation/image:        (H, W, 3) uint8 — base camera
            observation/wrist_image:  (H, W, 3) uint8 — wrist camera
            prompt:                   str

        rollout.py / DPO pair generation must populate these in batch under
        keys: image_uint8 (base, list of (H,W,3) uint8), wrist_uint8 (list,
        same shape), state (B, 8 tensor), instruction (list[str]).
        """
        # Base camera
        if "image_uint8" in batch:
            base = np.asarray(batch["image_uint8"][b_idx])
        elif "image" in batch:
            t = batch["image"][b_idx]
            if isinstance(t, torch.Tensor):
                base = (t.permute(1, 2, 0).cpu().clamp(0, 1).numpy() * 255).astype(np.uint8)
            else:
                base = np.asarray(t)
        else:
            base = np.zeros((224, 224, 3), dtype=np.uint8)

        # Wrist camera (zeros if missing — π0.5 may degrade but won't crash)
        if "wrist_uint8" in batch:
            wrist = np.asarray(batch["wrist_uint8"][b_idx])
        else:
            wrist = np.zeros_like(base)

        # State (8-dim for LIBERO: 7 joint + 1 gripper)
        if "state" in batch:
            s = batch["state"][b_idx]
            state = s.cpu().numpy() if isinstance(s, torch.Tensor) else np.asarray(s)
        else:
            state = np.zeros(8, dtype=np.float32)

        instruction = batch["instruction"][b_idx] if "instruction" in batch else ""

        return {
            "observation/state": state.astype(np.float32),
            "observation/image": base,
            "observation/wrist_image": wrist,
            "prompt": instruction,
        }

    def _batch_to_openpi_observation(self, batch: dict, B: int):
        """Build a batched openpi Observation suitable for PI0Pytorch.forward.

        For training-time logp we need to bypass the openpi Policy's transform
        chain (which is inference-only and does normalisation/RDS-style aug).
        Instead build an Observation directly from the per-sample dicts.

        This is the single most fragile bit of the adapter — kept simple.
        """
        from openpi.models import model as _model

        per_sample = [self._batch_to_openpi_obs(batch, b) for b in range(B)]

        # Stack tensors / arrays
        states = np.stack([d["observation/state"] for d in per_sample]).astype(np.float32)
        bases = np.stack([d["observation/image"] for d in per_sample])
        wrists = np.stack([d["observation/wrist_image"] for d in per_sample])
        prompts = [d["prompt"] for d in per_sample]

        # Tokenize prompt via the model's PaliGemma tokenizer
        tokenized, mask = self._tokenize_prompts_batch(prompts)

        # Convert image: uint8 [0,255] → float [-1, 1]
        # (openpi.preprocess_observation_pytorch expects this normalized form;
        # see preprocessing_pytorch.py line 142 `image = image * 2.0 - 1.0`)
        device = self.device

        def _normalize_img(arr: np.ndarray) -> torch.Tensor:
            t = torch.from_numpy(arr).to(device).float()
            t = t / 255.0           # [0, 1]
            t = t * 2.0 - 1.0       # [-1, 1]
            return t

        data = {
            "image": {
                "base_0_rgb": _normalize_img(bases),
                "left_wrist_0_rgb": _normalize_img(wrists),
                "right_wrist_0_rgb": torch.zeros_like(_normalize_img(bases)),
            },
            "image_mask": {
                "base_0_rgb": torch.ones(B, dtype=torch.bool, device=device),
                "left_wrist_0_rgb": torch.ones(B, dtype=torch.bool, device=device),
                "right_wrist_0_rgb": torch.zeros(B, dtype=torch.bool, device=device),
            },
            "state": torch.from_numpy(states).to(device),
            "tokenized_prompt": torch.from_numpy(tokenized).to(device).long(),
            "tokenized_prompt_mask": torch.from_numpy(mask).to(device).bool(),
        }
        return _model.Observation.from_dict(data)

    def _tokenize_prompts_batch(self, prompts: list[str]) -> tuple[np.ndarray, np.ndarray]:
        """Tokenize a list of prompts using PaliGemma tokenizer (cached).

        On dev pod / cloudml the network can't reach gs://big_vision/ which
        is what openpi.models.tokenizer.PaligemmaTokenizer hits by default.
        We bypass openpi's tokenizer wrapper and load sentencepiece directly
        from a known local path. The tokenizer model file (~4.2MB) lives at:
          - $PALIGEMMA_TOKENIZER_PATH (env)
          - /e2e-data/users/liuzhi7/vla_workspace/paligemma_tokenizer.model (dev pod)
          - ~/pi_assets/tokenizer/paligemma_tokenizer.model (local machine)
        """
        if not hasattr(self, "_paligemma_sp"):
            import sentencepiece
            cands = [
                os.environ.get("PALIGEMMA_TOKENIZER_PATH"),
                "/e2e-data/users/liuzhi7/vla_workspace/paligemma_tokenizer.model",
                str(Path.home() / "pi_assets/tokenizer/paligemma_tokenizer.model"),
            ]
            tok_path = next((c for c in cands if c and Path(c).exists()), None)
            if tok_path is None:
                raise FileNotFoundError(
                    "PaliGemma tokenizer model not found. Set $PALIGEMMA_TOKENIZER_PATH "
                    f"or place the file at one of: {cands}"
                )
            with open(tok_path, "rb") as f:
                self._paligemma_sp = sentencepiece.SentencePieceProcessor(model_proto=f.read())
            print(f"[pi05-adapter] loaded tokenizer from {tok_path}")

        # openpi's tokenize() prepends "...\n" but for surrogate-logp scoring
        # we want a stable encoding. Use the same prompt template:
        #   prompt + "\n"   then encode with BOS.
        tokens_list, masks_list = [], []
        max_len = 200  # openpi default
        for p in prompts:
            text = p.lower() + "\n"
            tokens = self._paligemma_sp.encode(text, add_bos=True, add_eos=True)
            t = np.asarray(tokens, dtype=np.int64)
            if t.shape[0] > max_len:
                t = t[:max_len]
            mask = np.zeros(max_len, dtype=bool)
            mask[: t.shape[0]] = True
            padded = np.zeros(max_len, dtype=np.int64)
            padded[: t.shape[0]] = t
            tokens_list.append(padded)
            masks_list.append(mask)
        return np.stack(tokens_list), np.stack(masks_list)

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
        """K stochastic action chunks per batch element via model.sample_actions.

        Returns: (B, K, T, 7) — unnormalised LIBERO action commands.
        """
        B = self._infer_batch_size(batch)
        observation = self._batch_to_openpi_observation(batch, B)

        chunks_per_k = []
        for _ in range(n_samples):
            actions_32 = self.model.sample_actions(self.device, observation, num_steps=10)
            actions_7 = actions_32[..., :PI05_ACTION_DIM].float().cpu()
            actions_7 = self._unnormalize_actions(actions_7)
            chunks_per_k.append(actions_7)
        # (K, B, T, 7) -> (B, K, T, 7)
        return torch.stack(chunks_per_k, dim=0).permute(1, 0, 2, 3).contiguous()

    @torch.no_grad()
    def select_action(self, batch: dict) -> torch.Tensor:
        """Single deterministic action chunk per batch element.

        Returns: (B, T, 7) — unnormalised LIBERO action commands ready for env.step.
        """
        B = self._infer_batch_size(batch)
        observation = self._batch_to_openpi_observation(batch, B)
        actions_32 = self.model.sample_actions(self.device, observation, num_steps=10)
        actions_7 = actions_32[..., :PI05_ACTION_DIM].float().cpu()
        return self._unnormalize_actions(actions_7)

    @staticmethod
    def _infer_batch_size(batch: dict) -> int:
        if "image" in batch and isinstance(batch["image"], torch.Tensor):
            return batch["image"].shape[0]
        if "image_uint8" in batch:
            return len(batch["image_uint8"])
        if "instruction" in batch:
            return len(batch["instruction"])
        return 1

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
