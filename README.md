<div align="center">

# vla-lab

**Independent VLA Research — DoRA × DPO × Multi-View Pretraining on OpenVLA / π0.5**

*Personal project by Liu Zhi (刘志, [ORCID 0009-0006-4808-9202](https://orcid.org/0009-0006-4808-9202)) — transitioning from autonomous-driving motion planning into embodied AI / Vision-Language-Action models.*

[![Status](https://img.shields.io/badge/status-active-brightgreen)]()
[![arxiv](https://img.shields.io/badge/arXiv-2605.21854-b31b1b.svg)](https://arxiv.org/abs/2605.21854)
[![Paper](https://img.shields.io/badge/paper-arXiv:2605.21854-orange)](paper_drafts/crossvla_v1/00_full_draft.pdf)
[![ORCID](https://img.shields.io/badge/ORCID-0009--0006--4808--9202-A6CE39?logo=orcid)](https://orcid.org/0009-0006-4808-9202)
[![License](https://img.shields.io/badge/license-MIT-blue)]()
[![Updated](https://img.shields.io/badge/updated-May%202026-informational)]()

<br/>

https://github.com/lz-googlefycy/vla-lab/assets/openvla_libero_4suite_demo.mp4

<sub><b>Click ▶ to play</b> the full 4-suite demo (40 clips, ~4 min, 5 MB). Reproduction scripts: <a href="https://github.com/lz-googlefycy/openvla-libero">openvla-libero</a>.</sub>

</div>

---

## ⭐ TL;DR (May 2026 sprint)

- 📄 **Workshop paper draft** — [CrossVLA: Cross-Paradigm Post-Training and Inference Optimization for VLA](https://arxiv.org/abs/2605.21854) ([PDF](https://arxiv.org/pdf/2605.21854)) (5 pages, 45 OpenAlex-verified citations). arxiv preprint: 2605.21854.
- 🎯 **DoRA + DPO 4-suite multiseed** (600 trials × 3 seeds): **mean +10.4pp over OpenVLA SFT**. Per-suite: Object **+20pp**, Long10 **+11pp**, Goal **+8pp**, Spatial **+3pp**. See [§4.3](paper_drafts/crossvla_v1/04_experiments.md) and the 12 raw json files in [`assets/paper_v1.5_eval/`](assets/paper_v1.5_eval/).
- 🔬 **Multi-view InfoNCE pretrain ckpt** (SigLIP-so400m frozen + 656K projection head, trained on 6000 LIBERO frames in 30 min on 1×H20). k-NN retrieval **recall@1 same-task = 99.5%** (36× over random). Live demo: [`models/pretrain_rlds_siglip_day8/`](models/pretrain_rlds_siglip_day8/).
- ⚡ **KV-Cache inference anatomy** — both chunk-level and token-level prefix caching strategies fail on flow-matching VLAs. Latency anatomy reveals denoise loop dominates **78.6% of per-call cost** vs prefix forward at **21.4%**, capping VLA-Cache style strategies. Concurrent SnapFlow (arXiv:2604.05656) confirms denoise-loop distillation as the productive direction. See [§4.5](paper_drafts/crossvla_v1/04_experiments.md).
- 🧪 **OpenVLA + LIBERO base reproduction** (5/9): SFT 4-suite re-eval + adapter codebase + Docker image. Foundation for the May sprint above.

---

## 🎬 LIBERO 4-suite — Click any preview to play

<div align="center">

<table>
<tr>
<td align="center">
<b>Spatial</b><br/>
<i>same object, different positions</i><br/>
<a href="https://github.com/lz-googlefycy/vla-lab/releases/download/v0.1-demos/openvla_libero_spatial_demo.mp4" title="Click to play full Spatial demo MP4"><img src="assets/demos/preview_spatial.gif" width="320" alt="Spatial demo — click to play full MP4"/></a>
</td>
<td align="center">
<b>Object</b><br/>
<i>same layout, different objects</i><br/>
<a href="https://github.com/lz-googlefycy/vla-lab/releases/download/v0.1-demos/openvla_libero_4suite_demo.mp4" title="Click to play full 4-suite demo MP4"><img src="assets/demos/preview_object.gif" width="320" alt="Object demo — click to play full MP4"/></a>
</td>
</tr>
<tr>
<td align="center">
<b>Goal</b><br/>
<i>different task goals</i><br/>
<a href="https://github.com/lz-googlefycy/vla-lab/releases/download/v0.1-demos/openvla_libero_4suite_demo.mp4" title="Click to play full 4-suite demo MP4"><img src="assets/demos/preview_goal.gif" width="320" alt="Goal demo — click to play full MP4"/></a>
</td>
<td align="center">
<b>Long-horizon (10)</b><br/>
<i>multi-step composed tasks</i><br/>
<a href="https://github.com/lz-googlefycy/vla-lab/releases/download/v0.1-demos/openvla_libero_4suite_demo.mp4" title="Click to play full 4-suite demo MP4"><img src="assets/demos/preview_long.gif" width="320" alt="Long-horizon demo — click to play full MP4"/></a>
</td>
</tr>
</table>

<sub>Demos rendered from the May 9 OpenVLA-7B base SFT eval; quantitative results below come from the May sprint with updated DoRA / DPO / multiseed protocol. Full <a href="https://github.com/lz-googlefycy/vla-lab/releases/download/v0.1-demos/openvla_libero_4suite_demo.mp4">4-suite MP4 (40 clips, 5 MB)</a> on <a href="https://github.com/lz-googlefycy/vla-lab/releases/tag/v0.1-demos">Release v0.1-demos</a>.</sub>

</div>

---

## 🎯 LIBERO 4-suite — DoRA × DPO Multiseed Results

Full multiseed grid (3 seeds × 50 trials per cell = **150 trials per suite**, 600 trials total for the DoRA cells). All numbers reproducible from raw json files in [`assets/paper_v1.5_eval/`](assets/paper_v1.5_eval/).

| Suite | OpenVLA SFT (ours) | + LoRA + DPO (s=42) | + LoRA + DPO multiseed¹ | **+ DoRA + DPO (3-seed pool)** | Δ vs SFT |
|:---|---:|---:|---:|---:|---:|
| Spatial | 72% | 78% | n/a¹ | **74.7%** (112/150) | +2.7pp |
| **Object** | 56% | 62% | 75% | **76.0%** (114/150) ⭐ | **+20.0pp** |
| Goal | 70% | 76% | 77% | **78.0%** (117/150) | +8.0pp |
| Long-horizon | 53% | 54% | 64% | **64.0%** (96/150) | +11.0pp |
| **Mean** | **62.75** | **67.50** | — | **73.2** | **+10.4pp** |

¹ **LoRA Spatial multiseed not available** — at the time of the LoRA experiment we only ran seed=42 on the Spatial cell (other suites Object/Goal/Long10 had seed 1337+2026 pooled = "multiseed"). Spatial's missing LoRA-multiseed cell is the asymmetry footnoted in [paper §4.3 footnote 4](paper_drafts/crossvla_v1/04_experiments.md): direct seed-42 comparison shows DoRA 78% = LoRA 78% (tie); DoRA 3-seed pool 74.7% therefore averages over its own seed variance against a single LoRA point estimate. Expanding LoRA Spatial to multiseed is left to camera-ready.

**Key findings**:
- DoRA Object exhibits **zero seed variance**: each of seeds 42, 1337, 2026 yields exactly **38/50 = 76.0%**. This rules out lucky-seed concerns for the headline +20 pp gain over SFT.
- DoRA **mean +10.4 pp over OpenVLA SFT across 4 suites** (600 trials), with the largest gains on suites where SFT under-performs most (Object +20, Long-horizon +11).
- Reproduce the +10.4 pp number end-to-end:

  ```bash
  python -c "
  import json, os
  base = 'assets/paper_v1.5_eval'
  sft = {'spatial':72,'object':56,'goal':70,'10':53}
  total = 0
  for suite, sftv in sft.items():
      s, t = 0, 0
      for seed in [42, 1337, 2026]:
          f = f'{base}/openvla_dpo_libero_{suite}_dora_seed{seed}.json'
          if not os.path.exists(f) and suite == '10':
              f = f'{base}/openvla_dpo_libero_long10_dora_seed{seed}.json'
          d = json.load(open(f))
          v = list(d['per_suite'].values())[0]
          s += v['success']; t += v['trials']
      pct = s/t*100; delta = pct - sftv; total += delta
      print(f'{suite}: DoRA {s}/{t}={pct:.1f}% vs SFT {sftv}% = +{delta:.1f}pp')
  print(f'Mean: +{total/4:.1f}pp')
  "
  ```

---

## 🔬 Multi-View Spatiotemporal InfoNCE Pretraining

A 656K-parameter projection head trained on top of frozen SigLIP-so400m vision features with **dual-stream InfoNCE**: multi-view alignment (agent ↔ wrist) + temporal coherence (agent_t ↔ agent_{t+5}).

**Convergence** (1×H20-3e, 30 min wall clock):
- Total InfoNCE loss: **2.42 → 0.37** (random baseline log(B=128)=4.85, **92.5% recovery**)
- L_mva (multi-view): 3.53 → 0.51 (85.6% reduction)
- L_tc (temporal): 1.31 → 0.22 (83.0% reduction)

**Quantitative validation — k-NN retrieval over all 6000 LIBERO frames**:

| Recall@k | same-task hit | same-episode | random@1 |
|:---|---:|---:|---:|
| @1 | **99.5%** | 91.4% | 2.75% |
| @5 | 99.9% | 97.7% | — |
| @10 | 99.95% | 99.0% | — |

**99.5% / 2.75% = 36× over random**. Per-suite recall@1: spatial 99.3%, **object 100.0%**, goal 99.3%, long10 99.5%.

**Reproduce in 5 minutes** (1×H20):
```bash
python models/pretrain_rlds_siglip_day8/retrieval_eval.py \
    --rlds_root /path/to/modified_libero_rlds \
    --siglip_path /path/to/openvla-7b-finetuned-libero-spatial \
    --proj_ckpt models/pretrain_rlds_siglip_day8/proj_head.pt \
    --output_dir /tmp/retrieval_repro
```

Live notebook: [`models/pretrain_rlds_siglip_day8/demo.ipynb`](models/pretrain_rlds_siglip_day8/demo.ipynb).

Full doc: [`docs/teaching/multiview_infonce_pretrain_for_liu.md`](docs/teaching/multiview_infonce_pretrain_for_liu.md) (9 sections, includes 4 interview Q&A).

---

## ⚡ KV-Cache Inference Anatomy on π0.5

We instrumented `model.sample_actions()` on π0.5 (LIBERO Spatial, dev pod H20 96 GB):

| Stage | Time | % of total |
|:---|---:|---:|
| Image preprocess + tokenize | ~5 ms | 1.8% |
| `embed_prefix` + paligemma prefix forward | ~60 ms | **21.4%** ← VLA-Cache target |
| Denoise loop × 10 (action expert per step) | ~220 ms | **78.6%** ← dominates |
| **Total** | **~280 ms** | 100% |

We tested two cache strategies; **both fail on π0.5** (full data in [§4.5](paper_drafts/crossvla_v1/04_experiments.md)):

| Strategy | Hit rate | Success rate | Speedup |
|:---|---:|---:|---:|
| Baseline (no cache) | — | 100% (50/50) | 1.0× |
| Chunk-level cache (sim ≥ 0.95) | 82.1% | 80% (40/50) | 0.7× (slower) |
| Token-level prefix cache (sim ≥ 0.92) | 86% | 0% (0/1 task) | 1.4× |

**Conclusion**: VLA-Cache prefix caching does not transfer to flow-matching VLAs because the denoise loop dominates 78.6% of inference cost. **Concurrent work** [SnapFlow (arXiv:2604.05656)](https://arxiv.org/abs/2604.05656) confirms our prescription via progressive self-distillation of the denoise loop.

Code: [`code/inference/kv_cache.py`](code/inference/kv_cache.py) + [`code/inference/prefix_kv_cache.py`](code/inference/prefix_kv_cache.py).

---

## 📂 Repository Structure

```
vla-lab/
├── README.md                                 # this file
├── paper_drafts/crossvla_v1/                 # 📄 workshop paper v1.4 (5 pages)
│   ├── 00_full_draft.{md,pdf}                # stitched full draft
│   ├── 01_abstract.md … 07_conclusion.md     # per-section sources
│   ├── references.bib                        # 45 OpenAlex-verified citations
│   ├── figures/                              # 6 vector figures
│   └── oa_literature_*.json                  # raw OpenAlex search audit trail
├── code/
│   ├── post_training/                        # DPO/GRPO + LoRA/DoRA + interface
│   │   ├── interface.py                      # VLABase 5-method protocol
│   │   ├── train_dpo.py                      # DPO training (chunk-level)
│   │   ├── train_grpo.py                     # GRPO (single-suite App C)
│   │   ├── eval_libero.py                    # supports --use_dora, --use_prefix_cache
│   │   ├── adapters/
│   │   │   ├── openvla.py                    # autoregressive adapter
│   │   │   └── pi05.py                       # flow-matching adapter + surrogate logp
│   │   └── dpo_loss.py
│   ├── spirit_adapter/
│   │   └── train_lora.py                     # _LoRALinear + _DoRALinear (DoRA paper-faithful)
│   ├── inference/
│   │   ├── kv_cache.py                       # chunk-level cache (Day 4 fail)
│   │   └── prefix_kv_cache.py                # token-level prefix cache (Day 7 fail)
│   ├── pretrain/                             # multi-view InfoNCE pretraining
│   │   ├── model.py                          # MultiViewProjectionHead, info_nce, multiview_temporal_loss
│   │   ├── dataset_rlds.py                   # TFDS-backed LIBERO RLDS reader
│   │   └── train.py                          # AdamW + cosine LR + SiglipEncoderWrapper (timm)
│   └── scripts/                              # reproduction shells
├── models/
│   └── pretrain_rlds_siglip_day8/
│       ├── proj_head.pt                      # 656K-param ckpt (2.6 MB, force-added)
│       ├── README.md                         # full model card
│       ├── demo.ipynb                        # 5-cell load → forward → cos-sim heatmap
│       ├── retrieval_eval.py                 # standalone reproducer
│       ├── retrieval_results.json            # quantitative recall@1=99.5%
│       └── retrieval_examples.png            # 6 query × top-3 visualisation
├── assets/
│   ├── demos/                                # LIBERO mp4 + GIF previews
│   ├── paper_v1.5_eval/                      # 30+ raw json/jsonl from May sprint
│   │   ├── openvla_dpo_libero_*_dora_seed{42,1337,2026}.json    # DoRA 12 cells
│   │   ├── openvla_dpo_libero_*_5x10_seed42.json                # LoRA single
│   │   ├── openvla_dpo_libero_*_5x10_multiseed.json             # LoRA multiseed
│   │   ├── pi05_sft_libero_*.json                               # π0.5 SFT (paper-aligned)
│   │   ├── kvcache_*.json + day7/prefix_cache_threshold*.json   # KV-Cache anatomy
│   │   └── pretrain_rlds_siglip_day8/train_log.json             # convergence trace
│   ├── post_training_v0/                     # earlier prototype results
│   └── spirit/                               # Spirit v1.5 prototype assets
├── docs/
│   ├── teaching/                             # 9 in-depth teaching docs (interview-grade)
│   │   ├── multiview_infonce_pretrain_for_liu.md
│   │   ├── day1_dora_for_liu.md
│   │   ├── day3_kvcache_for_liu.md
│   │   ├── day7_kvcache_pivot_for_liu.md
│   │   └── …
│   ├── work_log/                             # day-by-day sprint logs
│   ├── env_setup.md                          # image + dependencies
│   ├── results_libero_official_ckpt.md       # base SFT 4-suite numbers
│   ├── pi_series_analysis.md                 # 7 VLA paper critical reviews (63 KB)
│   └── insights.md
├── docker/                                   # openvla-v1.0-cu118-py310
└── notebooks/ tests/
```

---

## 🚀 Quick Start

### Prerequisites

- NVIDIA GPU ≥ 24 GB (RTX 3090 works for 4-bit inference; H100/H20 for full-bf16 + multiseed eval)
- Docker + nvidia-container-runtime
- ~30 GB for model + ~10 GB for LIBERO RLDS dataset

### 1. Build or pull the image

```bash
cd docker && docker build -t openvla-v1.0-cu118-py310 .
```

### 2. Download model + dataset

```bash
HF_ENDPOINT=https://hf-mirror.com huggingface-cli download openvla/openvla-7b \
  --local-dir <path>/models/openvla-7b

HF_ENDPOINT=https://hf-mirror.com huggingface-cli download \
  --repo-type dataset openvla/modified_libero_rlds \
  --local-dir <path>/datasets/modified_libero_rlds

for s in spatial object goal 10; do
  huggingface-cli download openvla/openvla-7b-finetuned-libero-$s \
    --local-dir <path>/models/openvla-7b-finetuned-libero-$s
done
```

### 3. Smoke test

```bash
docker run --rm --gpus all -v <path>/models:/workspace/models \
  openvla-v1.0-cu118-py310 \
  python /workspace/openvla/../smoke_test.py
# 4-bit load 4.4 GB GPU, predict_action outputs (7,) float64 at ~3 Hz
```

### 4. Reproduce DoRA + DPO LIBERO 4-suite (full 600-trial sprint)

```bash
# Train DoRA + DPO per suite (~20 min × 4 = 80 min)
for suite in spatial object goal 10; do
  python -m post_training.train_dpo \
    --base openvla --base_ckpt <path>/models/openvla-7b-finetuned-libero-$suite \
    --suite $suite --pairs_file output/h20_rollout_$suite/${suite}_pairs.pt \
    --output_dir output/h20_dpo_${suite}_dora \
    --use_dora --lora_r 32 --lora_alpha 64 --max_steps 500 --beta 0.1
done

# 3-seed eval (~50 min × 12 cells; ⚠️ requires USE_TF=0 + MUJOCO_GL=egl, see App D)
for suite in spatial object goal 10; do
  for seed in 42 1337 2026; do
    USE_TF=0 TRANSFORMERS_NO_TF=1 MUJOCO_GL=egl \
    python -m post_training.eval_libero \
      --base openvla --base_ckpt <path>/models/openvla-7b-finetuned-libero-$suite \
      --lora_ckpt output/h20_dpo_${suite}_dora/checkpoint-500.pt \
      --use_dora --lora_r 32 --lora_alpha 64 \
      --suites libero_$suite --seeds $seed \
      --output_dir output/h20_dpo_${suite}_dora_eval_seed${seed}
  done
done
```

### 5. Run multi-view InfoNCE pretraining (30 min on 1×H20)

```bash
python -m code.pretrain.train \
    --rlds_root <path>/datasets/modified_libero_rlds \
    --siglip_path <path>/models/openvla-7b-finetuned-libero-spatial \
    --suites spatial object goal 10 --max_episodes_per_suite 50 --max_per_episode 30 \
    --epochs 10 --batch_size 128 --lr 3e-4 --temperature 0.07 \
    --output_dir output/pretrain_rlds_siglip
# Expected: total InfoNCE loss 4.85 → ~0.37 in 470 steps
```

### 6. Reproduce k-NN retrieval recall@1 = 99.5% (5 min on 1×H20)

```bash
python models/pretrain_rlds_siglip_day8/retrieval_eval.py \
    --rlds_root <path>/datasets/modified_libero_rlds \
    --siglip_path <path>/models/openvla-7b-finetuned-libero-spatial \
    --proj_ckpt models/pretrain_rlds_siglip_day8/proj_head.pt \
    --output_dir /tmp/retrieval_repro
```

### 7. Read the paper

```bash
# PDF
xdg-open paper_drafts/crossvla_v1/00_full_draft.pdf

# Or markdown
less paper_drafts/crossvla_v1/00_full_draft.md
```

---

## 📚 VLA Paper Critical Reviews

Applied a rigorous `paper_analysis` SOP (Phase 1–5, skeptical-by-default) to every major VLA paper. Full Chinese reports in [`docs/`](./docs/):

| Paper | Year | Score | Key Finding |
|:---|---:|:---:|:---|
| [RT-1](./docs/) | 2022 | 6/10 | Engineering existence proof, not a new method |
| [RT-2](./docs/) | 2023 | 5/10 | Paradigm-opening, but "emergent" is over-marketed |
| [Octo](./docs/) | 2024 | 7/10 | Clean open baseline (not strictly VLA) |
| [OpenVLA](./docs/) | 2024 | 8/10 | Current open-source VLA standard |
| [π0](./docs/pi_series_analysis.md) | 2024 | — | Flow matching = packaging of Transfusion+DP+ACT |
| [π0.5](./docs/pi_series_analysis.md) | 2025 | — | "Open-world" is 3 similar-distribution homes |
| [π*0.6](./docs/pi_series_analysis.md) | 2025 | — | RECAP = clever advantage-conditioning trick |

Key cross-paper insight: **all ablations tell the same story — pre-training + data multiplicity contribute far more than architecture novelty.**

---

## 🗺️ Roadmap

| Phase | Focus | Status |
|:---:|:---|:---|
| 0 | Infrastructure (image, data, dev environment) | ✅ done |
| 1 | LIBERO base reproduction on official ckpts | ✅ done |
| 2 | DoRA × DPO multiseed sprint (Apr–May 2026) | ✅ done — see [§4.3](paper_drafts/crossvla_v1/04_experiments.md) |
| 3 | Multi-view InfoNCE pretraining + retrieval eval | ✅ done — see [`models/pretrain_rlds_siglip_day8/`](models/pretrain_rlds_siglip_day8/) |
| 4 | KV-Cache inference anatomy + structural negative | ✅ done — see [§4.5](paper_drafts/crossvla_v1/04_experiments.md) |
| 5 | Workshop paper v1.4 + arxiv preprint | 🚧 active |
| 6 | Downstream pretrain ablation (proj_head → OpenVLA init → DPO) | planned |
| 7 | Real-robot demo on XLeRobot / SO-100 | planned |

---

## 🧭 Philosophy

1. **Skeptical-by-default paper reading.** Every ablation is audited, every "emergent" claim is challenged. The field over-markets.
2. **Reproducibility first.** If I can't run it end-to-end on public data + a single commodity GPU, the contribution value is discounted.
3. **Open everything.** Code, docs, eval logs, demo videos, ckpts (when small enough), mistakes. The field needs more honest post-mortems, not more hype posts.
4. **No institutional affiliation**, no hype vocabulary. Just experiments.

---

## 🤝 Community

- Found a bug or reproduction mismatch? Open an issue.
- Want to collaborate on XLeRobot / SO-100 / LeRobot? Ping me.
- Job-hunting in embodied AI? Let's compare notes.

**Contact**: open an issue, or email via Git commit history.

---

## 📜 License

Code: MIT. Written content (docs / paper drafts): CC BY 4.0.

---

## 🙏 Acknowledgments

- [Stanford + Berkeley + TRI OpenVLA team](https://openvla.github.io/) — for the only truly-open 7B VLA
- [Physical Intelligence](https://physicalintelligence.company/) — for `openpi` source that teaches flow-matching on action chunks
- [Spirit AI / 千寻智能](https://www.spirit-ai.com/) — for open-sourcing `spirit-v1.5` and their RoboChallenge wrapper code
- [HuggingFace LeRobot team](https://github.com/huggingface/lerobot) — for XLeRobot + SO-100 open hardware stack
- [LIBERO team](https://libero-project.github.io/) — for the benchmark
- [DoRA authors](https://arxiv.org/abs/2402.09353) (Liu et al.) — for the magnitude/direction decoupling that turns out to matter on VLAs too
- [SnapFlow team](https://arxiv.org/abs/2604.05656) — for empirically validating, concurrently, our denoise-loop-targeting prescription
