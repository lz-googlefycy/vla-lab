# 1. Introduction

Vision-Language-Action (VLA) models — large multimodal policies that
map (image, language instruction, proprioceptive state) → robot action
— have established a small set of architectural lineages. **OpenVLA**
\cite{kim2024openvla} treats actions as 7 × 256-bin discretised tokens
and predicts them autoregressively atop a Llama-2 7B language model
with a fused DINO-SigLIP vision tower. **π0** \cite{black2024pi0} and
**π0.5** \cite{openpi2025pi05} instead emit *continuous* 7-DoF action
chunks via flow-matching, with a PaliGemma vision-language encoder and
a 10-step ordinary differential equation (ODE) action expert. RT-2
\cite{brohan2023rt2}, RDT \cite{liu2024rdt}, OpenVLA-OFT
\cite{kim2025openvlaoft}, TinyVLA \cite{wen2024tinyvla}, SpatialVLA
\cite{spatialvla2025}, RoboMamba \cite{robomamba2024}, and CoT-VLA
\cite{cotvla2025} span the design space between these poles.

While supervised fine-tuning (SFT) on demonstrations yields VLA
policies that solve held-out manipulation tasks at the 80–98% range on
LIBERO \cite{liu2023libero}, **post-training** — the analogue of
RLHF/DPO/GRPO in language models — remains nascent for VLAs. The
concurrent work GRAPE \cite{grape2024} addresses preference alignment
on autoregressive VLAs but does not consider flow-matching backbones,
which lack a clean DPO formulation because their chunk-level
log-probability requires probability-flow ODE integration with
prohibitive Jacobian costs.

This work asks: **does post-training generalise across architectural
paradigms in VLA?** Specifically:

1. Can DPO be made to *work* on flow-matching VLAs via a tractable
   surrogate log-probability?
2. Does the choice of **parameter-efficient adapter** (LoRA vs DoRA)
   interact with the backbone architecture or the task family?
3. Do **inference-time acceleration** techniques developed for
   autoregressive VLAs (specifically VLA-Cache style prefix-K/V
   reuse \cite{wang2025vlacache}) transfer to flow-matching VLAs?

We answer all three empirically on LIBERO 4-suite, with public OpenVLA
and π0.5 checkpoints. Headline findings:

- DPO with our **negative-MSE flow-matching surrogate** (§3.2) trains
  stably on π0.5 with the same protocol as on OpenVLA.
- **DoRA** \cite{liu2024dora}, originally proposed for LLM
  instruction-tuning, generalises to VLA DPO and **outperforms LoRA**
  on the harder LIBERO suites: +14 pp on Object, +10 pp on Long10
  (single-seed; LoRA multi-seed mean = DoRA single-seed on Long10).
  This is substantially larger than the +1–3 pp gains DoRA reports in
  the LLM domain.
- **VLA-Cache style prefix caching does not transfer to π0.5.** A
  latency anatomy reveals that 78.6% of per-call cost is the
  flow-matching denoise loop — *not* the prefix forward that
  VLA-Cache targets — leaving a hard 21% acceleration ceiling. Both
  chunk-level (§4.5.2) and token-level (§4.5.3) cache strategies
  degrade success rate while producing little or no speedup.
  **Concurrent work** SnapFlow \cite{snapflow2026} arrives at the same
  prescription via progressive self-distillation of the denoise loop.

**Contributions.**
1. **Surrogate flow-matching log-probability** for DPO on continuous-
   action VLAs (§3.2), validated to be drop-in compatible with the
   standard DPO objective.
2. **First reported DoRA-on-VLA empirical study**, with a per-suite
   breakdown identifying when magnitude/direction decoupling helps
   (narrow-distribution adaptation: Object, Long10) and when it
   doesn't (large direction shifts: Spatial).
3. **Negative result on prefix-K/V caching for flow-matching VLAs**,
   accompanied by a structural latency anatomy and a positive
   prescription (denoise-loop-targeting acceleration is the
   productive direction; concurrently validated by \cite{snapflow2026}).
4. **Multi-view + temporal contrastive pretraining framework**
   (§3.5) with a publicly released projection-head checkpoint
   achieving 99.5% k-NN recall@1 for same-task retrieval on 6000
   LIBERO frames.
5. **Open implementation**: all four contributions are reproducible
   from a single repository (https://github.com/lz-googlefycy/vla-lab),
   including ckpts, training logs, and end-to-end runnable scripts.

The remainder of the paper is organised as follows. §2 reviews
relevant VLA backbones, RLHF methods, and PEFT techniques. §3 details
our cross-paradigm interface, the surrogate logp, our DoRA + DPO
pipeline, and the multi-view pretraining framework. §4 reports
LIBERO 4-suite results. §5 discusses implications for near-term VLA
post-training research. §6 acknowledges limitations.
