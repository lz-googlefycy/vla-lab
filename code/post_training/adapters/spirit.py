"""Spirit v1.5 adapter for v1.5 post-training pipeline.

Spirit's action representation
------------------------------
Spirit v1.5 (千寻智能, 2026.01) is a flow-matching VLA:
  - Backbone: Qwen3-VL-4B-Instruct
  - Action head: BaseDiT (16 layers, 1536 hidden) → 60-step chunk × 14 dim
  - Loss: standard flow-matching MSE on velocity field v_θ(x_t, t)
          with x_t = action + t·(noise - action)

So Spirit is fundamentally different from OpenVLA:
  - chunk-level (not token-level)
  - continuous (not discrete)
  - flow-matching (no closed-form log-prob; need ODE solver or surrogate)

Log-probability for DPO
-----------------------
Flow matching defines an implicit density via the probability flow ODE.
Computing exact log p(chunk) requires:
  - Integrating div(v_θ) along an ODE from t=0 to t=1
  - Adding initial Gaussian log-prob
  → expensive (~10× cost of one forward), and gradient through ODE is
     fragile

We use a **flow-matching surrogate** instead:

    s(chunk) = -E_t[||v_θ(x_t, t) - (noise - chunk)||^2]

This is the negative MSE between the model's predicted velocity field
and the ground-truth velocity at random t. It's NOT a true log-prob
but it IS:
  - cheap (one forward per sample evaluation)
  - an approximation with no guarantee of true density ordering
  - self-consistent across (chosen, rejected) pairs

This provides a differentiable training surrogate. Its ranking quality
must be validated on the task; it is not an exact policy likelihood.

Reference choice:
  T_eval = 4 sample t values per evaluation (averaged), each from
  Beta(1.5, 1.0) like Spirit's training distribution. This gives a
  stable gradient signal while keeping cost low.

Sampling K candidates
---------------------
Use Spirit's flow-matching sampler with different noise seeds. Standard
Spirit uses 10 ODE steps; for sampling we keep that. K candidates =
K different noise initializations + same instruction.

Reference
---------
- ``code/spirit_adapter/xlerobot_adapter.py``: existing inference adapter
  (the existing XLeRobot prototype uses a different state representation)
- v1.4 troubleshooting.md: 6 dtype workarounds we already know
- ``Spirit-v1.5/model/modeling_spirit_vla.py``:
    - ``SpiritVLAPolicy.forward(batch)`` — flow-matching loss training
    - ``SpiritVLAPolicy.select_action(batch)`` — denoising inference
"""

from __future__ import annotations

import json
import os
import sys
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Iterator

import torch
import torch.nn as nn

from ..interface import VLABase, PostTrainConfig

# Spirit src locations (env var override → workspace → home dir)
_SPIRIT_CANDIDATES = [
    os.environ.get("SPIRIT_SRC"),
    "/workspace/spirit-v1.5",
    str(Path.home() / "spirit-v1.5"),
]
for _p in _SPIRIT_CANDIDATES:
    if _p and Path(_p).exists() and _p not in sys.path:
        sys.path.insert(0, _p)


SPIRIT_ACTION_HORIZON = 60     # chunk length
SPIRIT_ACTION_DIM = 14         # zero-padded 7D EE pose × 2 arms


@contextmanager
def _local_qwen_skeleton(SpiritVLAPolicy, local_backbone: str):
    """Build Spirit's Qwen module from local config without loading base weights.

    The released Spirit checkpoint contains the complete Qwen state dict. The
    upstream constructor nevertheless calls ``Qwen...from_pretrained`` before
    loading that checkpoint. Here we use the local AWQ directory only for its
    config/tokenizer/processor, instantiate a float32 Qwen skeleton,
    then let Spirit's normal ``load_state_dict`` populate every model weight.
    """
    from transformers import AutoConfig
    from transformers.modeling_utils import no_init_weights

    import model.modeling_spirit_vla as spirit_module

    qwen_class = spirit_module.Qwen3VLForConditionalGeneration
    had_local_loader = "from_pretrained" in vars(qwen_class)
    original_loader_descriptor = vars(qwen_class).get("from_pretrained")
    original_policy_init = SpiritVLAPolicy.__init__
    original_policy_load_state = SpiritVLAPolicy.load_state_dict

    def local_qwen_loader(cls, pretrained_model_name_or_path, *args, **kwargs):
        hf_config = AutoConfig.from_pretrained(local_backbone, trust_remote_code=True)
        # The local directory is an AWQ export, while Spirit's checkpoint is
        # floating point. Quantization metadata describes those AWQ files and
        # must not configure the freshly constructed model skeleton.
        if "quantization_config" in getattr(hf_config, "__dict__", {}):
            del hf_config.__dict__["quantization_config"]
        requested_attention = kwargs.get("attn_implementation")
        if requested_attention:
            hf_config._attn_implementation = requested_attention
            hf_config._attn_implementation_internal = requested_attention
        qwen = cls(hf_config)
        # HF normally ties embeddings during initialization; no_init_weights
        # skips that hook, so retain the configured parameter sharing here.
        qwen.tie_weights()
        return qwen

    def local_policy_init(self, config):
        # AutoTokenizer and AutoProcessor in Spirit's constructor resolve
        # these assets locally too.
        config.backbone = local_backbone
        return original_policy_init(self, config)

    def report_policy_state_load(self, state_dict, *args, **kwargs):
        result = original_policy_load_state(self, state_dict, *args, **kwargs)
        print(
            f"[Spirit] checkpoint key audit: missing={len(result.missing_keys)}, "
            f"unexpected={len(result.unexpected_keys)}; "
            f"missing sample={result.missing_keys[:5]}, "
            f"unexpected sample={result.unexpected_keys[:5]}"
        )
        return result

    qwen_class.from_pretrained = classmethod(local_qwen_loader)
    SpiritVLAPolicy.__init__ = local_policy_init
    SpiritVLAPolicy.load_state_dict = report_policy_state_load
    try:
        # A strict full-checkpoint load populates these real CPU tensors.
        # Avoid initializing billions of temporary random weights first.
        with no_init_weights():
            yield
    finally:
        if had_local_loader:
            qwen_class.from_pretrained = original_loader_descriptor
        else:
            delattr(qwen_class, "from_pretrained")
        SpiritVLAPolicy.__init__ = original_policy_init
        SpiritVLAPolicy.load_state_dict = original_policy_load_state


class SpiritAdapter(VLABase):
    """Spirit v1.5 wrapper conforming to VLABase protocol.

    Loads the complete Spirit checkpoint with config-only Qwen construction.
    LIBERO camera/pose conversion lives in spirit_adapter.libero_bridge.
    The training score evaluates Spirit's actual DiT velocity field.

    Lifecycle:
        1. ``__init__``: strictly load the checkpoint and optionally apply
           DiT LoRA. Call freeze_reference() explicitly for preference scoring.
        2. ``policy_logp(batch, chunk)``: flow-matching surrogate score
           using T_eval random timesteps.
        3. ``policy_sample(batch, K)``: Spirit's denoising sampler with
           K different noise seeds.
    """

    # Number of (t, noise) samples averaged to estimate the
    # flow-matching surrogate log-prob. 1 = noisy, 4 = stable.
    T_EVAL = 4

    def __init__(self, cfg: PostTrainConfig):
        self.cfg = cfg
        if cfg.spirit_action_hz <= 0 or cfg.spirit_replan_steps < 1:
            raise ValueError("Spirit action rate and replan interval must be positive")
        if cfg.use_dora:
            raise ValueError("Spirit LIBERO currently supports LoRA; DoRA requires separate validation")
        if cfg.use_lora and (cfg.lora_r < 1 or cfg.lora_alpha <= 0):
            raise ValueError("LoRA rank and alpha must be positive")
        self.device = torch.device(cfg.spirit_device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.reference_state_dict: dict[str, torch.Tensor] | None = None

        # Lazy import to avoid forcing Spirit src on this module load
        from model import SpiritVLAPolicy
        from utils import sampling

        self._sampling_module = sampling

        # `train=True` makes SpiritVLAPolicy.from_pretrained load on CPU.
        # Do not rewrite the checkpoint config: it may be read-only and is a
        # shared artifact used by other evaluation processes.
        config_path = Path(cfg.base_ckpt_path) / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"No config.json at {config_path}")
        with config_path.open() as f:
            checkpoint_config = json.load(f)
        local_backbone = (
            cfg.spirit_backbone_path or os.environ.get("SPIRIT_QWEN_LOCAL")
            or checkpoint_config["backbone"]
        )
        print(f"[Spirit] Constructing Qwen skeleton from local config: {local_backbone}")
        print(f"[Spirit] Loading all model weights from Spirit checkpoint: {cfg.base_ckpt_path}")
        with _local_qwen_skeleton(SpiritVLAPolicy, local_backbone):
            self.model = SpiritVLAPolicy.from_pretrained(
                ckpt_path=str(cfg.base_ckpt_path), strict=True, train=True
            )

        torch_dtype = getattr(torch, cfg.base_dtype, torch.bfloat16)
        self.model = self.model.to(device=self.device, dtype=torch_dtype)
        # Keep min/max arithmetic in float32 when neural weights use bf16.
        self.model.normalize_inputs.float()
        self.model.normalize_targets.float()
        self.model.unnormalize_outputs.float()

        if cfg.use_lora:
            self._inject_lora()

    def _autocast(self):
        neural_model = getattr(self.model, "qwen", self.model)
        dtype = next(neural_model.parameters()).dtype
        if dtype in (torch.float16, torch.bfloat16):
            return torch.autocast(self.device.type, dtype=dtype)
        return nullcontext()

    def _inject_lora(self) -> None:
        """Inject LoRA on DiT attention (same approach as Spirit train_lora.py)."""
        sys.path.insert(0, str(Path(__file__).parent.parent.parent))
        from spirit_adapter.train_lora import _LoRALinear, _replace_module_by_path

        for p in self.model.parameters():
            p.requires_grad = False

        targets = list(self.cfg.lora_target_modules)
        replacements: list[tuple[str, nn.Linear]] = []
        for name, m in self.model.named_modules():
            if not name.startswith("dit.") or not isinstance(m, nn.Linear):
                continue
            for t in targets:
                if name.endswith("." + t) or name == t:
                    replacements.append((name, m))
                    break
        if not replacements:
            raise ValueError("No Spirit DiT attention modules matched the LoRA targets")
        for name, orig in replacements:
            new_m = _LoRALinear(
                orig, r=self.cfg.lora_r, alpha=self.cfg.lora_alpha,
                dropout=self.cfg.lora_dropout,
            )
            _replace_module_by_path(self.model, name, new_m)

        if self.cfg.also_train_proj:
            for name, m in self.model.named_modules():
                if any(name.endswith(s) for s in
                       ["state_proj", "action_in_proj", "action_out_proj"]):
                    for p in m.parameters():
                        p.requires_grad = True

        n_trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in self.model.parameters())
        print(
            f"[spirit-adapter] LoRA injected: {len(replacements)} Linears, "
            f"trainable {n_trainable/1e6:.1f}M / {n_total/1e6:.0f}M "
            f"({100*n_trainable/n_total:.2f}%)"
        )

    # --------------- VLABase: training-time interface --------------- #

    def policy_logp(
        self, batch: dict, chunk: torch.Tensor
    ) -> torch.Tensor:
        """Flow-matching surrogate score for `chunk`.

        For each of T_EVAL random t values:
            x_t = chunk + t·(noise - chunk)
            v_target = noise - chunk
            v_pred = model.dit_velocity_field(x_t, t, batch)
            score_t = -||v_pred - v_target||^2

        Returns mean across T_EVAL samples, shape (B,).
        """
        if chunk.ndim != 3 or chunk.shape[1:] != (self.model.config.n_action_steps, 14):
            raise ValueError("Spirit scores physical (B, n_action_steps, 14) target chunks; OSC command pairs need conversion first")

        spirit_batch = self._prepare_batch(batch)
        normalized_batch = self.model.normalize_inputs(spirit_batch)
        state = self.model.prepare_state(normalized_batch)
        with self._autocast():
            vlm_last_embed = self.model.proj_vlm_output(self.model._encode_vision(normalized_batch))

        raw_actions = chunk.to(self.device)
        normalized_actions = self.model.normalize_targets({"action": raw_actions})["action"]
        normalized_actions = torch.nn.functional.pad(
            normalized_actions,
            (0, self.model.config.max_action_dim - normalized_actions.shape[-1]),
        )

        batch_size = chunk.shape[0]
        scores = []
        for _ in range(self.T_EVAL):
            t = self._sampling_module.sample_time(batch_size, device=self.device).float()
            noise = self._sampling_module.sample_noise(
                normalized_actions.shape, device=self.device,
            ).to(normalized_actions.dtype)
            t_expand = t.view(-1, 1, 1).to(normalized_actions.dtype)
            # Match Spirit's training convention: x_t = a + t(noise-a),
            # v_target = noise-a.
            x_t = normalized_actions + t_expand * (noise - normalized_actions)
            v_target = noise - normalized_actions
            v_pred = self._velocity_field(
                spirit_batch, x_t, t, state=state, vlm_last_embed=vlm_last_embed,
            )
            # Franka uses the first seven dimensions; the remaining padded
            # dimensions are not part of the LIBERO action probability.
            mse_per_sample = ((v_pred[..., :7] - v_target[..., :7]) ** 2).mean(dim=(-2, -1))
            scores.append(-mse_per_sample)
        return torch.stack(scores, dim=0).mean(dim=0).float()

    def _velocity_field(
        self, batch: dict, x_t: torch.Tensor, t: torch.Tensor,
        *, state: torch.Tensor | None = None, vlm_last_embed: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward through Spirit DiT to compute v(x_t, t).

        Simplified — actual Spirit model.forward() does train-time loss
        computation with sampling internally. We reproduce its forward
        velocity prediction by calling the DiT directly.
        """
        if state is None or vlm_last_embed is None:
            normalized_batch = self.model.normalize_inputs(batch)
            state = self.model.prepare_state(normalized_batch)
            with self._autocast():
                vlm_last_embed = self.model.proj_vlm_output(self.model._encode_vision(normalized_batch))
        with self._autocast():
            return self.model._denoise_step(state, vlm_last_embed, x_t, t)

    def select_action_for_libero(self, batch: dict) -> torch.Tensor:
        """Return Spirit's chunk as physical deltas from the chunk-start pose.

        Spirit's own runner turns every delta into a target pose relative to
        the state captured before inference. The LIBERO harness then converts
        each target pose into a delta from the live pose before env.step().
        """
        spirit_batch = self._prepare_batch(batch)
        with self._autocast():
            actions = self.model.select_action(spirit_batch).float()[..., :7].clone()
        if os.environ.get("SPIRIT_DEBUG_ACTIONS") and not getattr(self, "_debug_action_logged", False):
            state = spirit_batch["observation.state"].detach().float().cpu()
            print(f"[Spirit debug] state={state[0, 0, :7].tolist()}")
            print(f"[Spirit debug] physical target chunk first 3={actions[0, :3].cpu().tolist()}")
            self._debug_action_logged = True
        return actions

    def _prepare_batch(self, batch: dict) -> dict:
        required = (
            "observation.state", "observation.images.cam_high",
            "observation.images.cam_left_wrist", "observation.images.cam_right_wrist",
            "task", "robot_type",
        )
        missing = [key for key in required if key not in batch]
        if missing:
            raise ValueError(f"Spirit needs all three cameras and an EE state; missing {missing}. Use spirit_adapter.libero_bridge.prepare_batch.")
        return {key: value.to(self.device) if isinstance(value, torch.Tensor) else value
                for key, value in batch.items()}

    def policy_logp_with_ref(
        self, batch: dict, chunk: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cpu_rng = torch.get_rng_state()
        devices = [self.device.index if self.device.index is not None else torch.cuda.current_device()] if self.device.type == "cuda" else []
        cuda_rng = torch.cuda.get_rng_state(devices[0]) if devices else None
        logp_cur = self.policy_logp(batch, chunk)
        if self.reference_state_dict is None:
            raise RuntimeError("Call freeze_reference() before scoring Spirit DPO pairs")

        # Functional parameter substitution preserves the current policy's
        # backward graph; mutating parameter.data here corrupts its gradients.
        adapter = self
        class ScoreModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.policy = adapter.model
            def forward(self, batch, chunk):
                return adapter.policy_logp(batch, chunk)

        current = dict(self.model.named_parameters())
        reference = {"policy." + name: value.to(current[name].device, current[name].dtype)
                     for name, value in self.reference_state_dict.items()}
        with torch.random.fork_rng(devices=devices), torch.no_grad():
            torch.set_rng_state(cpu_rng)
            if devices:
                torch.cuda.set_rng_state(cuda_rng, devices[0])
            logp_ref = torch.func.functional_call(ScoreModule(), reference, (batch, chunk))
        return logp_cur, logp_ref

    @torch.no_grad()
    def policy_sample(
        self,
        batch: dict,
        n_samples: int,
        deterministic: bool = False,
    ) -> torch.Tensor:
        """Sample K candidate chunks via Spirit's denoising sampler with
        different noise seeds.

        Spirit's select_action() uses internal noise sampling. To get K
        different chunks for the same prompt, we call select_action()
        K times — each invocation re-samples noise.
        """
        spirit_batch = self._prepare_batch(batch)
        if n_samples < 1:
            raise ValueError("n_samples must be positive")
        batch_size = spirit_batch["observation.state"].shape[0]
        generator = torch.Generator(device=self.device).manual_seed(0) if deterministic else None
        chunks = []
        for _ in range(n_samples):
            noise = None
            if generator is not None:
                noise = torch.randn(
                    (batch_size, self.model.config.n_action_steps, self.model.config.max_action_dim),
                    generator=generator, device=self.device, dtype=torch.float32,
                )
            with self._autocast():
                out = self.model.select_action(spirit_batch, noise=noise)
            chunks.append(out)
        # stack along K dim
        return torch.stack(chunks, dim=1).float()   # (B, K, T, A)

    @torch.no_grad()
    def select_action(self, batch: dict) -> torch.Tensor:
        spirit_batch = self._prepare_batch(batch)
        with self._autocast():
            out = self.model.select_action(spirit_batch)
        return out.float()

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
        print(f"[spirit-adapter] frozen reference: "
              f"{len(self.reference_state_dict)} param tensors")

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        trainable = {name for name, param in self.model.named_parameters() if param.requires_grad}
        normalization = ("normalize_inputs.", "normalize_targets.", "unnormalize_outputs.")
        tensors = {name: value.detach().cpu() for name, value in self.model.state_dict().items()
                   if name in trainable or name.startswith(normalization)}
        torch.save({
            "format": "spirit-libero-v1", "state_dict": tensors,
            "metadata": {
                "lora_r": self.cfg.lora_r, "lora_alpha": self.cfg.lora_alpha,
                "action_hz": self.cfg.spirit_action_hz,
                "action_steps": self.model.config.n_action_steps,
                "camera_convention": "upright_rgb", "action_space": "base_frame_chunk_targets_m_rad",
                "target_semantics": "osc_controller_goals_v2",
            },
        }, path)

    def load(self, path: str) -> None:
        saved = torch.load(path, map_location="cpu", weights_only=True)
        if saved.get("format") == "spirit-libero-v1":
            meta = saved["metadata"]
            if meta["action_hz"] <= 0 or meta["camera_convention"] != "upright_rgb":
                raise ValueError("Invalid Spirit action rate or camera convention")
            if meta["lora_r"] != self.cfg.lora_r or meta["lora_alpha"] != self.cfg.lora_alpha:
                raise ValueError("LoRA configuration differs from the saved Spirit adapter")
            if meta["action_steps"] != self.model.config.n_action_steps:
                raise ValueError("Spirit action horizon differs from the saved adapter")
            self.cfg.spirit_action_hz = float(meta["action_hz"])
            sd = saved["state_dict"]
            required = {name for name, parameter in self.model.named_parameters() if parameter.requires_grad}
            required.update(name for name in self.model.state_dict()
                            if name.startswith(("normalize_inputs.", "normalize_targets.", "unnormalize_outputs.")))
            missing = required - sd.keys()
            if missing:
                raise ValueError(f"Incomplete Spirit adapter checkpoint: {sorted(missing)[:8]}")
        else:
            sd = saved  # legacy trainable-parameters-only checkpoint
        own = self.model.state_dict()
        invalid = [name for name, value in sd.items() if name not in own or own[name].shape != value.shape]
        if invalid:
            raise ValueError(f"Incompatible Spirit checkpoint keys: {invalid[:8]}")
        if not sd:
            raise ValueError("Empty Spirit adapter checkpoint")
        self.model.load_state_dict(sd, strict=False)
