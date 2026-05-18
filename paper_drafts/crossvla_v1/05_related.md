# 2. Background and Related Work

## 2.1 Vision-Language-Action Models

The VLA paradigm has rapidly diversified since RT-2 \cite{brohan2023rt2}.
Two open-source families with public LIBERO-finetuned checkpoints are
the focus of this paper.

**Autoregressive (token-AR) VLAs.** Discretise each action dimension
into a vocabulary (typically 256 bins per DoF) and predict the joint
action vector autoregressively under a language model head. **OpenVLA**
\cite{kim2024openvla} is the canonical 7B-parameter open variant on
Llama-2 with a fused DINO-SigLIP vision encoder, releasing per-suite
LIBERO-finetuned checkpoints. OpenVLA-OFT \cite{kim2025openvlaoft}
investigates SFT data mixtures and decoding for speed and success rate.
Smaller variants such as TinyVLA \cite{wen2024tinyvla} and lightweight
extensions such as RoboMamba \cite{robomamba2024} and ReVLA
\cite{revla2024} target deployment efficiency. Spatial reasoning
extensions include SpatialVLA \cite{spatialvla2025} and CoT-VLA
\cite{cotvla2025}; the latter is closely related to embodied
chain-of-thought \cite{ecot2024}. ChatVLA \cite{chatvla2025} unifies
multimodal understanding with robot control. Surveys
\cite{vla2025survey, vla2026embodied} provide broader context.

**Flow-matching VLAs.** Emit *continuous* action chunks via probability-
flow ODE integration. **π0** \cite{black2024pi0} introduced this
direction with a 10-step ODE action expert; **π0.5** \cite{openpi2025pi05}
refined it with knowledge insulation. Diffusion Policy
\cite{chi2023diffusion} is a precursor in non-VLA visuomotor settings;
RDT-1B \cite{liu2024rdt} extends diffusion to bimanual manipulation.
Both flow-matching and diffusion models lack a closed-form per-chunk
log-probability, which has so far precluded standard DPO-style
preference alignment.

**Concurrent work on VLA preference alignment.** GRAPE
\cite{grape2024} concurrently develops preference alignment for robot
policies, focusing on autoregressive VLAs and online preference
collection. We differ in (i) cross-paradigm scope (we cover both
autoregressive and flow-matching), (ii) offline pair-based DPO with a
flow-matching surrogate logp (§3.2), and (iii) PEFT-layer ablation
(LoRA vs DoRA, §4.3). Failure-prediction extensions such as FPC-VLA
\cite{fpcvla2026} take an orthogonal direction.

## 2.2 Preference Alignment

**DPO** \cite{rafailov2023direct} replaces the explicit reward model of
RLHF with an implicit reward defined by the policy ratio against a
reference. Its closed-form loss requires evaluating
`log p_θ(chunk | obs)` and `log p_ref(chunk | obs)`. For autoregressive
policies this is straightforward; for continuous-action flow-matching
policies it is an open problem that we address in §3.2.

**Variants.** IPO \cite{azar2023ipo}, KTO \cite{ethayarajh2024kto},
ORPO \cite{hong2024orpo}, and SimPO \cite{meng2024simpo} explore
different reward parameterisations (no reference model, prospect-
theoretic loss, single-stage objectives, length-normalised rewards).
We use vanilla DPO for clarity; extending to these variants is
straightforward in our pipeline.

**GRPO** \cite{shao2024deepseekmath, deepseek2025r1} sidesteps the
reference model via group-relative advantage. We include a single-suite
GRPO ablation (App C) but find it under-performs DPO at our compute
budget; cross-paradigm GRPO is left to future work.

**Multi-objective preference** \cite{multidpo2024} is orthogonal to
this work; we use a single scalar reward (LIBERO success rate).

## 2.3 Parameter-Efficient Fine-Tuning

**LoRA** \cite{hu2021lora} injects a rank-r residual `α/r · BA` into
each Linear layer. **DoRA** \cite{liu2024dora} additionally decomposes
the LoRA-adapted weight into magnitude × direction, training the
magnitude as a free per-output-channel scalar. DoRA was originally
evaluated on LLM instruction-following (GLUE, MT-Bench) where it gives
modest +1–3 point gains. To our knowledge **no prior work has
evaluated DoRA on VLA fine-tuning**; we report the first such study in
§4.3, with substantially larger per-suite gains than the LLM-domain
literature reports. Variants such as QA-LoRA \cite{xu2024qalora} and
CLIP-DoRA \cite{clipdora2025} address quantisation and vision-language
adaptation respectively, but not VLA action prediction.

## 2.4 VLA Inference Acceleration

**VLA-Cache** \cite{wang2025vlacache} caches static visual tokens'
KV across env timesteps, achieving a reported ~1.7× speedup on
OpenVLA. Their target architecture is autoregressive, where the
vision tokens dominate per-call compute. We test (§4.5) whether the
strategy transfers to π0.5; our latency anatomy shows the prefix
forward is only 21% of per-call cost on flow-matching VLAs, capping
the strategy's potential. KVSharer \cite{kvsharer2024} is a
complementary approach for general LLM inference but does not target
the action-expert denoise loop.

**Speculative decoding** \cite{leviathan2023speculative} targets
autoregressive token generation; not directly applicable to flow-
matching action chunks. Adaptive test-time compute approaches such as
VLA-ATTC \cite{vlaattc2026} and Sentinel-VLA \cite{sentinelvla2026}
trade compute against task progress signals; these are orthogonal to
our static-cache study.

**Consistency model distillation** \cite{salimans2022progressive,
song2023consistency, lu2024simpleconsistency} reduces N-step ODE
integration to 1–4 steps, directly attacking the dominant cost in
flow-matching VLAs. Concurrent to our work, **SnapFlow**
\cite{snapflow2026} applies progressive self-distillation to obtain
**one-step action generation for flow-matching VLAs** — empirically
validating the prescription we identify in §5.4. MoFlow
\cite{moflow2025} explores a similar one-step direction in trajectory
forecasting.

## 2.5 Self-Supervised Pretraining for Robot Representations

**R3M** \cite{nair2022r3m} pretrains a visual encoder on ego-centric
human video with time-contrastive and language-alignment losses.
**SiamMAE** \cite{gupta2023siammae} uses siamese masked autoencoding
for time-correspondence in video. **MV-MWM** \cite{seo2023multiview}
multi-view world-models for robot manipulation. Ag2Manip
\cite{ag2manip2024} learns agent-agnostic visual + action
representations. ReBot \cite{rebot2025} synthesises real-to-sim-to-real
video for scaling robot learning. Our multi-view + temporal pretraining
(§3.5) draws on these, instantiated as a small projection head on top
of a frozen SigLIP-so400m \cite{zhai2023siglip} so the resulting
features stack cleanly atop OpenVLA's existing vision encoder.
PaLI-3 \cite{chen2023pali3} explores smaller VLMs that share the
SigLIP encoder family.

## 2.6 Benchmarks

We evaluate on **LIBERO** \cite{liu2023libero}, the de-facto VLA
manipulation benchmark with four suites (Spatial, Object, Goal,
Long10) covering 130 unique tasks with paired demonstrations. CALVIN
\cite{mees2022calvin} is a complementary long-horizon benchmark we
do not use here; cross-benchmark generalisation is future work
(§5.4).
