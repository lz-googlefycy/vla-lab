# 7. Conclusion

We present **CrossVLA**, an empirical study of cross-paradigm
post-training for Vision-Language-Action models. Our work makes four
contributions:

1. A **surrogate flow-matching log-probability** estimator (§3.2) that
   makes DPO directly applicable to flow-matching VLAs (e.g. π0.5).
   The surrogate is the negative MSE of the conditional flow-matching
   loss, deterministic given (obs, x_1, x_0 seed), and stable in
   practice across 500-step DPO training runs.

2. **First DoRA-on-VLA empirical study**. We find DoRA outperforms
   LoRA at PEFT for VLA DPO on the harder LIBERO suites: +14 pp on
   Object, +10 pp on Long10 at single-seed seed=42. The win pattern is
   consistent with DoRA's magnitude/direction decoupling helping in
   narrow-distribution adaptation but not in suites that require new
   visual grounding (Spatial: 0 pp).

3. A **structural negative result on prefix-K/V caching** for flow-
   matching VLAs. Our latency anatomy shows the denoise loop dominates
   78.6% of `sample_actions` cost on π0.5, capping any prefix-targeted
   cache strategy at a 21% acceleration ceiling. We document failures
   of both chunk-level and token-level cache strategies and identify
   denoise-loop-targeting acceleration (consistency distillation) as
   the productive future direction.

4. A **multi-view + temporal contrastive pretraining framework** with
   a publicly released 656K-parameter `proj_head.pt` checkpoint that
   achieves 99.5% k-NN retrieval recall@1 on 6000 LIBERO frames
   (36× over random), available as a downstream initialisation.

The four contributions are not independently surprising — they are
each modest extensions of established techniques to the under-studied
flow-matching VLA paradigm. Their value is in the **comparative
crossing**: under one shared protocol, with one open code repository
and ckpt set, we map out which post-training techniques transfer
cleanly across architectural paradigms and which need paradigm-specific
adaptation.

We hope this work primes a more systematic cross-paradigm post-training
literature for VLAs as more open backbones (Spirit, RDT, future π
versions) become available.

**All code, training logs, ckpts, and reproduction scripts** are open
at https://github.com/lz-googlefycy/vla-lab.
