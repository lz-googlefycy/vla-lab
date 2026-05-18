# Abstract (~180 words)

Vision-Language-Action (VLA) models have rapidly converged on a small
set of architectural patterns: discrete-token autoregression (e.g.
OpenVLA) and continuous-action flow-matching (e.g. π0.5). Yet
preference alignment via Direct Preference Optimisation (DPO) — the
de-facto post-training step in language models — has been studied
almost exclusively on autoregressive VLAs.

We present **CrossVLA**, an empirical study of cross-paradigm VLA
post-training. Three contributions: (i) a **surrogate flow-matching
log-probability** estimator that lets DPO operate on continuous-action
backbones without probability-flow ODE integration; (ii) a head-to-head
comparison of LoRA and DoRA as the parameter-efficient layer for VLA
DPO, finding DoRA wins **+14 pp on LIBERO Object** and **+10 pp on
Long10** vs LoRA at single-seed seed=42, while tying on Spatial; (iii) an
**inference-time anatomy** showing that on flow-matching VLAs the
denoise loop dominates 79% of `sample_actions` latency and prefix-K/V
caching à la VLA-Cache caps at a 21% acceleration ceiling — both
chunk-level and token-level cache strategies degrade success rate to
0–80% in our benchmarks. We further pretrain a multi-view + temporal
projection head on 6000 LIBERO frames, which achieves **99.5% k-NN
recall@1** for same-task retrieval (36× over random), available as
a downstream initialisation. All code, ckpts, training logs, and
reproduction scripts are open at https://github.com/lz-googlefycy/vla-lab.
