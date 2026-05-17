# 2. Background and Related Work

## 2.1 VLA Architectural Paradigms

We focus on two open VLA paradigms with publicly released LIBERO
checkpoints.

**Autoregressive (token-AR) VLAs**. Discretise each action dimension
into a vocabulary (typically 256 bins per dimension) and predict the
joint action vector autoregressively under a language model head.
RT-2 (Brohan et al. 2023) introduced this template; OpenVLA
(Kim et al. 2024) built a 7B-parameter open variant on Llama-2 with a
fused DINO-SigLIP vision encoder, releasing per-suite LIBERO-finetuned
checkpoints which we use directly. OpenVLA-OFT (Kim et al. 2024b)
investigates SFT data mixtures but does not study post-training.

**Flow-matching VLAs**. Emit continuous-action chunks via probability-
flow ODEs (Lipman et al. 2023). π0 (Black et al. 2024) introduced this
direction, π0.5 (openpi 2025) refined it with knowledge insulation
between the vision-language and action expert sub-networks. Spirit v1.5
(2026, RoboChallenge winner) uses a similar flow-matching architecture
on Qwen3-VL-4B. Both lack a closed-form per-chunk log-probability,
which has so far precluded DPO-style preference alignment.

**Mixed paradigms.** RDT (Liu et al. 2024) employs flow matching atop
a discrete-token diffusion. We exclude it from this study because
its public LIBERO checkpoints had not been released at the sprint
deadline.

## 2.2 Preference Alignment

**DPO** (Rafailov et al. 2023) replaces the explicit reward model of
RLHF with an implicit one defined by the policy ratio against a
reference. Its closed-form objective requires evaluating
`log p_θ(chunk | obs)` and `log p_ref(chunk | obs)`. For
autoregressive policies this is straightforward; for continuous-
action flow-matching policies it is an open problem.

**GRPO** (Shao et al. 2024) and its DeepSeek-R1 variant (DeepSeek
2024) sidestep the reference model by estimating advantage from a
*group* of K rollouts and using group-relative normalisation. GRPO
is appealing for VLAs (no reference model storage cost) but requires
online rollouts, which is expensive in robotics simulation
(~30 s/trial × K samples × 200 steps). We include a single-suite
GRPO ablation in App C and find it under-performs DPO at our compute
budget; we focus the main study on DPO.

**Other preference variants** — IPO (Azar et al. 2023), KTO
(Ethayarajh et al. 2024), ORPO (Hong et al. 2024) — are not
investigated here.

## 2.3 Parameter-Efficient Fine-Tuning

**LoRA** (Hu et al. 2021) injects a rank-r residual `α/r · BA` into
each Linear layer. **DoRA** (Liu et al. 2024) decomposes the LoRA-
adapted weight into magnitude × direction, training the magnitude as a
free per-output-channel scalar. DoRA was originally evaluated on LLM
instruction-following (GLUE, MT-Bench) where it gives modest
+1–3 point gains. To our knowledge **no prior work has evaluated DoRA
on VLA fine-tuning**; we report the first such study in §4.3.

## 2.4 VLA Inference Acceleration

**VLA-Cache** (Wang et al. 2025, NeurIPS workshop) caches static
visual tokens' KV across env timesteps, achieving ~1.7× speedup on
OpenVLA. Their target architecture is autoregressive, where the
vision tokens dominate per-call compute. We test (§4.5) whether the
strategy transfers to π0.5; our latency anatomy shows the prefix
forward is only 21% of per-call cost on flow-matching VLAs, capping
the strategy's potential.

**Speculative decoding** (Leviathan et al. 2023) and **Medusa**
(Cai et al. 2024) target autoregressive token generation; not
applicable to flow-matching action chunks.

**Consistency model distillation** (Salimans & Ho 2022; Lu et al.
2024) reduces N-step ODE integration to 1–4 steps, directly attacking
the dominant cost in flow-matching VLAs. We do not implement it in
this work but identify it as the productive direction (§5.4).

## 2.5 Self-Supervised Pretraining for Robotics

**R3M** (Nair et al. 2022) pretrains a visual encoder on ego-centric
human video with time-contrastive and language-alignment losses.
**SiamMAE** (Gupta et al. 2023) uses siamese masked autoencoding for
time-correspondence in video. **DeCUR** (Mokady et al. 2023) decouples
contrastive learning across modalities for vision-language models.
**MV-MWM** (Seo et al. 2023) does multi-view world-modelling for
robot manipulation. Our multi-view + temporal pretraining (§3.5)
combines these threads, instantiated specifically as a small
projection head on top of a frozen SigLIP-so400m so the resulting
features stack cleanly atop OpenVLA's existing vision encoder.
