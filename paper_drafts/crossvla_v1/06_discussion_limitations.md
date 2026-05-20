# 5. Discussion

## 5.1 What Generalises Across VLA Paradigms

DPO, with the right log-probability surrogate, **does** transfer from
autoregressive to flow-matching VLAs. The training dynamics are
qualitatively similar: chosen-vs-rejected reward margin grows
monotonically over 500 training steps; the BCE-style loss decreases
without instability; the resulting policy preserves SFT-saturated
suite performance (no catastrophic forgetting).

**This is the load-bearing positive result of the paper**: a single
post-training pipeline for two architectural lineages of VLA, requiring
only a method-level change (closed-form vs surrogate logp) but
no protocol-level changes.

## 5.2 What Doesn't Generalise: Inference Caching

KV-cache strategies designed for autoregressive VLAs (VLA-Cache and
its near-relatives) do **not** transfer to flow-matching π0.5 (§4.5).
The negative result is structural, not implementation-dependent:
flow-matching shifts the bulk of inference cost from the prefix
forward (where caching helps) into the denoise loop (where it
doesn't, because each denoise step depends on the previous step's
state).

This produces a clear research direction (§5.4): for flow-matching VLA
inference acceleration, **target the denoise loop**, e.g. via
consistency-model distillation (10 → 1–4 denoise steps).

## 5.3 PEFT-Architecture Interaction

DoRA's gains over LoRA are **not uniform across LIBERO suites**
(§4.3.1). The pattern correlates with whether the suite demands a
genuine direction shift (Spatial: 0 pp gain) or a magnitude refinement
on existing skills (Object +14, Long10 +10). This is consistent with
DoRA's design hypothesis (decoupled magnitude/direction) and suggests
**suite-level diagnosis** as an inexpensive heuristic for adapter
choice in production VLA fine-tuning.

A more cautious framing: at single-seed, our measurements are
confounded by per-seed variance. The Spatial 0-pp result could
genuinely indicate no advantage, or it could indicate an adversarial
seed; we discuss this in §6.

## 5.4 Implications for Future Work

1. **Denoise-loop-targeting acceleration** is the productive direction
   for flow-matching VLA inference. Consistency model distillation
   (\cite{salimans2022progressive}) is a natural fit but has not, to our
   knowledge, been applied to VLAs.
2. **DoRA on Spirit v1.5 / π0.5** is unstudied. Our work covers OpenVLA
   only; extending the DoRA+DPO combination to flow-matching backbones
   is straightforward given our pipeline.
3. **Multi-view pretraining as VLA encoder init.** Our `proj_head.pt`
   achieves 99.5% k-NN retrieval but has not been evaluated as a
   downstream initialisation for VLA fine-tunes. The natural next
   experiment is to swap the proj-head's output back into OpenVLA's
   vision-feature pipeline before SFT/DPO, then measure downstream
   success rate.
4. **GRPO on flow-matching VLAs.** Our DPO-vs-GRPO ablation is
   single-suite (App C). A full cross-paradigm × cross-algorithm grid
   on richer compute (3+ seeds, additional simulators like SimplerEnv
   or CALVIN) is the natural full-paper extension.

# 6. Limitations

We list known limitations honestly so reviewers can calibrate the
strength of our claims.

1. **Asymmetric multiseed coverage on the LoRA side.** All four DoRA
   cells are 3-seed pooled (seeds 42, 1337, 2026; 150 trials per
   suite). On the LoRA side, multiseed (seeds 1337 + 2026) is
   available for Object/Goal/Long10 only. **LoRA Spatial single-seed
   only** (s=42 = 78%), so the DoRA-vs-LoRA Spatial cell in §4.3 is
   reported with the asymmetry footnoted (direct s=42 tie at 78%; pool
   comparison shows DoRA 74.7%). This was deferred due to compute
   budget and is the most defendable cell to expand for camera-ready.

2. **No Spirit v1.5 results.** We had originally planned to include
   Spirit v1.5 as a third backbone (Qwen3-VL flow-matching, distinct
   from π0.5's PaliGemma), but its container build path conflicted
   with our cloudml runtime and we deferred it. The current paper
   therefore covers two paradigms (autoregressive, flow-matching) on
   one representative each.

3. **GRPO is single-suite.** The reward function design for VLA GRPO
   was under-explored at our compute budget; our single-suite Spatial
   GRPO result (App C, +2 pp over SFT) is reported but does not bear
   the weight of a comparable claim to DPO.

4. **π0.5 + DPO incomplete on Goal/Long10.** Dev pod tunnel
   disconnects mid-sprint capped our π0.5+DPO validation to Spatial
   and Object, both already SFT-saturated. We acknowledge this leaves
   the cross-paradigm DPO claim weaker on the harder suites.

5. **Pretrained projection head not evaluated downstream.** §4.6
   reports k-NN retrieval as an intrinsic representation quality
   measure, but we have not tested whether this projection improves
   downstream OpenVLA success rate (§5.4 lists this as future work).

6. **Sim only.** All evaluations are in LIBERO simulation. We argue
   sim-first is appropriate for this paper given the comparative
   nature of the study (cross-paradigm under controlled conditions),
   but real-robot validation remains future work.

7. **Decoding settings.** Our OpenVLA SFT baseline is below the OpenVLA
   paper number on Spatial/Object/Goal, attributable to using default
   temperature 1.0 vs the paper's calibrated decoding. This affects
   absolute numbers but not the **comparative** DoRA-vs-LoRA finding
   which holds the decoding fixed.
