# 3. Method

## 3.1 Cross-Paradigm VLA Interface

We define a minimal protocol that exposes the necessary primitives for
preference alignment across architecturally heterogeneous Vision-Language-Action
backbones. The interface deliberately abstracts away whether the
underlying policy emits discrete action tokens autoregressively (OpenVLA)
or continuous action chunks via flow-matching (π0.5).

```python
class VLABase(Protocol):
    def policy_logp(self, batch, chunk) -> Tensor:           # (B,)
    def policy_logp_with_ref(self, batch, chunk) -> tuple    # (cur, ref)
    def policy_sample(self, batch, K) -> Tensor              # (B, K, T, A)
    def encode_obs(self, obs) -> dict                        # processed inputs
    def sample_actions(self, obs, num_steps) -> Tensor       # (T, A) for env
```

The five-method contract is implemented per backbone:

- **OpenVLA adapter** (`code/post_training/adapters/openvla.py`):
  `policy_logp` is the closed-form sum of token log-probabilities under
  teacher forcing, since OpenVLA discretises each action dimension into
  256 bins and predicts them autoregressively.
- **π0.5 adapter** (`code/post_training/adapters/pi05.py`):
  `policy_logp` does not have a closed form — π0.5 emits 10-step action
  chunks via flow-matching ODE integration. We define a surrogate
  log-probability (§3.2) using the flow-matching MSE loss.

Both adapters share the same training (`code/post_training/train_dpo.py`)
and evaluation (`code/post_training/eval_libero.py`) entry points. The
backbone choice is a CLI flag (`--base {openvla,pi05}`).

## 3.2 Surrogate Log-Probability for Flow-Matching VLAs

DPO requires both `log p_θ(chunk | obs)` (current policy) and
`log p_ref(chunk | obs)` (frozen reference). For autoregressive VLAs
this is the standard token-level sum. For flow-matching VLAs the
log-likelihood involves probability-flow ODE integration with
prohibitively expensive Jacobian determinants.

We adopt a **surrogate** based on the conditional flow-matching loss
itself. Given a chunk `x_1` and noise `x_0`, the flow is
`x_t = (1-t) x_0 + t x_1` with target velocity
`v_target = x_1 - x_0`. The model predicts `v_θ(x_t, t, obs)`. We define

$$
\log \tilde{p}_\theta(x_1 \mid \text{obs})
\;=\;
-\frac{1}{T_{\text{eval}}} \sum_{t \sim \text{stratified}([0,1])}
\| v_\theta(x_t, t, \text{obs}) - v_{\text{target}} \|^2.
$$

Two design choices:

1. **Negative MSE as logp-surrogate**: the variational lower bound for
   diffusion models (\cite{kingma2021variational}) connects MSE to log-likelihood
   up to a known factor; in flow-matching the analogous bound exists
   with prefactor `||x_1 - x_0||^2 σ²(t)`. We absorb the prefactor into
   the DPO temperature β (Eq. 2 below) and use raw MSE.
2. **Stratified t-sampling, T_eval=4**: importance sampling over the
   curriculum [0, 1] reduces variance vs. uniform Monte Carlo. We use
   `t ∈ {0.125, 0.375, 0.625, 0.875}` plus one stochastic perturbation
   per step.

The surrogate is **deterministic given (obs, x_1, x_0 seed)**, which
the DPO objective requires for the reference forward to be reproducible.

## 3.3 PEFT Layer: DoRA

Low-Rank Adaptation (LoRA, \cite{hu2021lora}) decomposes the weight update
as `ΔW = α/r · B A` with `B ∈ R^{out × r}, A ∈ R^{r × in}`. DoRA
(\cite{liu2024dora}) further decomposes the adapted weight into
**magnitude** and **direction** components:

$$
W_{\text{eff}} = m \odot \frac{W_0 + \frac{\alpha}{r} B A}{\| W_0 + \frac{\alpha}{r} B A \|_{\text{col}}},
$$

where `m ∈ R^{out}` is a learnable per-output-channel magnitude vector,
`W_0` is the frozen pretrained weight, and `‖·‖_col` is the column-wise
L2 norm.

**Why DoRA on VLA**: LoRA simultaneously perturbs both the magnitude
and direction of `W`, which is well-suited to large, semantically
distant transfers (e.g. base LM → instruction-following). VLA fine-tuning
on LIBERO is a **narrow-distribution adaptation** — the demonstration
data shares scene priors with the SFT base. Decoupling magnitude from
direction allows DoRA to preserve direction (which encodes the
pretrained vision-language-grounding) while letting magnitude scale
freely per output channel — essentially a finer-grained gain control
on the existing skill.

Implementation (`code/spirit_adapter/train_lora.py:_DoRALinear`)
materialises `W_eff` once per forward, then applies the standard linear
op `F.linear(x, W_eff, b)`. Memory cost: an additional
`out × in` materialised tensor per layer (vs LoRA which only stores
`B, A`). For OpenVLA's 128 LoRA-target Linears at hidden 4096:

| | LoRA-r32 | DoRA-r32 |
|---|---|---|
| Trainable params | 33.55M (0.44%) | 34.08M (0.45%) |
| Peak GPU mem (eval) | 17.93 GB | 26.17 GB |
| Initial cur≡ref diff | 0.0 | 0.0 (after dropout fix*) |

*Bug record: an early implementation placed dropout on the `BA` input,
which caused stochastic divergence between cur and ref forwards in
DPO training. Fixed by following the LoRA convention of dropout on the
LoRA path only.

## 3.4 DPO Loss

We use the standard DPO objective (\cite{rafailov2023direct}) with the
surrogate logp:

$$
\mathcal{L}_{\text{DPO}}(\theta) =
-\mathbb{E}_{(\text{obs}, c^+, c^-)}
\log \sigma\!\left(\beta \cdot \big[
(\log \tilde p_\theta(c^+) - \log \tilde p_{\text{ref}}(c^+))
- (\log \tilde p_\theta(c^-) - \log \tilde p_{\text{ref}}(c^-))
\big]\right),
$$

where `c^+` is the chosen chunk (200-step ground-truth trajectory) and
`c^-` is the rejected chunk (SFT model rollout perturbed with
σ-ramp action noise). The reference model is the frozen SFT base.

In our experiments β = 0.1, learning rate 5e-5, batch size 1 with
gradient accumulation 1, max steps 500, warmup 100.

## 3.5 Multi-View + Temporal Contrastive Pretraining

We additionally explore **representation-level pretraining** as an
orthogonal axis to PEFT-based adaptation. The idea: while DoRA + DPO
shifts the policy distribution toward preferred actions, a frozen
SigLIP-so400m vision tower (used by both OpenVLA and π0.5) may benefit
from a small projection head that further aligns its features with
robot-specific multi-view + temporal structure.

**Architecture** (`code/pretrain/model.py`):

```
image (224×224×3, [-1, 1])
  ↓ SigLIP-so400m (timm vit_so400m_patch14_siglip_224, FROZEN)
1152-d feature (CLS pooled)
  ↓ MultiViewProjectionHead:  Linear(1152, 512) → GELU → Linear(512, 128)
128-d embedding
  ↓ L2 normalize
```

The SigLIP encoder weights are extracted bit-identically from
OpenVLA-7B's `vision_backbone.fused_featurizer.*` (342 keys, 0 missing,
0 unexpected) — so any feature this projection learns is directly
re-usable as input to OpenVLA's downstream LLM head.

**Dual-stream InfoNCE objective**:

$$
\mathcal{L} = w_{\text{mva}} \cdot \mathcal{L}_{\text{mva}}
+ w_{\text{tc}} \cdot \mathcal{L}_{\text{tc}},
$$

where `L_mva` is the symmetric InfoNCE between agent-view and wrist-view
embeddings at the same timestep (multi-view alignment), and `L_tc` is
the InfoNCE between agent-view embeddings at time `t` and `t+Δ` for
Δ=5 steps (temporal coherence). We use `τ = 0.07`, `w_mva = w_tc = 0.5`,
and batch size B = 128.

**Data**: OpenVLA team's published `modified_libero_rlds` (TFRecord
RLDS format), reading from
`code/pretrain/dataset_rlds.py:LiberoRLDSPretrainDataset`. We sample
50 episodes per LIBERO suite × 30 anchor times per episode × 4 suites
= 6000 (agent_t, wrist_t, agent_{t+5}) tuples.

**Training**: 10 epochs (≈ 470 steps) with AdamW, peak lr 3e-4 cosine
decay to 0, weight decay 1e-4. Wall-clock ≈ 30 min on a single H20-3e.
Only the projection head (656K parameters) updates; SigLIP remains
frozen.
