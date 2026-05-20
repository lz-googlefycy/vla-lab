# arxiv 投稿指南 — CrossVLA workshop draft v1.4

> 给 Zhi Liu (account `Zhi_Liu`, default `cs.AI`) 的手动上传步骤。

## 已准备好的 bundle

**位置**: `paper_drafts/crossvla_arxiv_v1.tar.gz` (1.45 MB, gzipped tarball)

**包内容**:
```
crossvla_arxiv_v1/
├── crossvla.tex          ← 主 LaTeX 文件 (NeurIPS 2024 [preprint] template)
├── crossvla.pdf          ← 编译好的 PDF (14 pages, 940 KB)
├── crossvla.bbl          ← 编译产物 (arxiv 接受;省去 bibtex 一轮)
├── references.bib        ← BibTeX (45 真实 OpenAlex citations)
├── neurips_2024.sty      ← Style file (NeurIPS 官方,已包入)
└── figures/              ← 6 张 vector PDF figure
    ├── fig_dora_vs_lora_4suite.pdf       (主表配图)
    ├── fig_kvcache_anatomy_bench.pdf
    ├── fig_pretrain_loss_curve.pdf
    ├── fig_retrieval_examples.pdf
    ├── fig_dora_forward_diagram.pdf      (currently unused)
    └── fig_dora_vs_lora_geometry.pdf     (currently unused)
```

## 上传前 sanity check

```bash
cd ~/ro_planning/paper_drafts/crossvla_arxiv_v1
# 重新本地编译验证
pdflatex -interaction=nonstopmode crossvla.tex   # 第 1 轮
bibtex crossvla
pdflatex -interaction=nonstopmode crossvla.tex   # 第 2 轮
pdflatex -interaction=nonstopmode crossvla.tex   # 第 3 轮
ls -la crossvla.pdf   # 应是 14 pages, ~940 KB
```

期望结果:
- `Output written on crossvla.pdf (14 pages, ~940000 bytes)`
- 0 undefined citations / 0 missing references

## arxiv 上传步骤

### 1. 登录 arxiv.org

用户名 `Zhi_Liu`，邮箱 `2022201433@tju.edu.cn`。

点击页面右上 **`START NEW SUBMISSION`** (你账号已显示)。

### 2. Stage 1 — Agreement

接受 arxiv 的 license 条款。

### 3. Stage 2 — Author Information

预填的应该是你账号信息。确认:
- **First Name**: Zhi
- **Last Name**: Liu
- **Affiliation**: Tianjin University
- 接着点 next

### 4. Stage 3 — Categories

**Primary**: `cs.AI` (你账号 default,不需要 endorsement)

**Cross-list**: 这两个 cross-list 通常不需要 endorsement,可以试加:
- `cs.LG` (Machine Learning)
- `cs.RO` (Robotics) ← 如果它仍然提示需要 endorsement,**去掉它,只保留 cs.AI + cs.LG 即可**

### 5. Stage 4 — Metadata

**Title**:
```
CrossVLA: Cross-Paradigm Post-Training and Inference Optimization for Vision-Language-Action Models
```

**Authors**:
```
Zhi Liu (Tianjin University)
```

**Abstract** (从 paper 复制,~ 200 words):
```
Vision-Language-Action (VLA) models have rapidly converged on a small set of architectural patterns: discrete-token autoregression (e.g. OpenVLA) and continuous-action flow-matching (e.g. pi-0.5). Yet preference alignment via Direct Preference Optimisation (DPO)---the de-facto post-training step in language models---has been studied almost exclusively on autoregressive VLAs. We present CrossVLA, an empirical study of cross-paradigm VLA post-training. Three contributions: (i) a surrogate flow-matching log-probability estimator that lets DPO operate on continuous-action backbones without probability-flow ODE integration; (ii) a head-to-head comparison of LoRA and DoRA as the parameter-efficient layer for VLA DPO, finding DoRA improves over OpenVLA SFT by a mean +10.4 pp across LIBERO 4-suite (600 trials, 3 seeds)---per-suite +20.0 Object, +11.0 Long-horizon, +8.0 Goal, +2.7 Spatial---with zero seed variance on Object (38/50 on each of 3 seeds); (iii) an inference-time anatomy showing the denoise loop dominates 78.6% of sample_actions latency and prefix-K/V caching a la VLA-Cache caps at a 21% acceleration ceiling---both chunk-level and token-level cache strategies degrade success rate to 0-80% in our benchmarks. We further pretrain a multi-view + temporal projection head on 6000 LIBERO frames, achieving 99.5% k-NN recall@1 for same-task retrieval (36x over random), available as a downstream initialisation. All code, ckpts, training logs, and reproduction scripts are open at https://github.com/lz-googlefycy/vla-lab.
```

**Comments** (这个写在 metadata 顶部):
```
Workshop draft, 14 pages, 6 figures. Code, ckpts, data: https://github.com/lz-googlefycy/vla-lab
```

**MSC Class** / **ACM Class**: 留空 (cs.AI 默认)

**Journal Reference**: 留空 (preprint)

**Report Number**: 留空

### 6. Stage 5 — Files (上传 bundle)

你有两种方式:

**方式 A: 上传 tar.gz** (推荐,arxiv 会自动 untar):
```
点击 "Add Files"
选择 paper_drafts/crossvla_arxiv_v1.tar.gz
等待 arxiv 解压并扫描 (~ 30 sec)
```

**方式 B: 单独上传** (备用):
解压 tar.gz 后逐个上传 .tex / .bbl / .sty / 6 个 .pdf figure。

注意:
- arxiv **要求上传 LaTeX source**, 不接受纯 PDF (会被 reject)
- 我们的 bundle 包含 .tex/.bbl 因此 OK
- 不要上传 .aux/.log/.out 这些中间文件 (我已经清理掉了)

### 7. Stage 6 — Process

arxiv 自动跑 `pdflatex` + `bibtex` + `pdflatex × 2`,生成最终 PDF。

如果失败,常见原因:
- **Missing reference**: 检查 .bib 是否正确上传
- **\bibliographystyle{plainnat}** 报错: 已经包含 `crossvla.bbl` 跳过 bibtex 阶段
- **Style file not found**: `neurips_2024.sty` 已经在 bundle 里

如果报错,告诉我具体 error message,我修。

### 8. Stage 7 — Preview

**仔细审查** arxiv 编译出的 PDF:
- [ ] 标题对吗?
- [ ] Author 是 "Zhi Liu, Tianjin University" 吗?
- [ ] Abstract 完整 (~200 词)?
- [ ] 14 pages,所有 figure 显示正常?
- [ ] References 显示了 (~45 entries)?

如果 OK → 点 **Submit**。

如果有问题 → 点 **Cancel**,在本地修 .tex,重新打包上传。

### 9. Stage 8 — Confirmation

提交后 arxiv 会:
1. 立即给你一个 **submission ID** (例 `submit/12345678`)
2. ~ 几小时内通过 moderator review (cs.AI / cs.LG 通常 < 6h)
3. 给你正式 arxiv ID (例 `arXiv:2606.01234`)
4. 发邮件通知

**arxiv ID 拿到后立刻告诉我**,我会:
- 简历 Bullet 1 加 arxiv link
- README 加 arxiv badge
- vla-lab + ro_planning 都 push 一版含 arxiv link 的 commit
- 飞书 wiki 加 paper milestone

## 投稿失败的应对

### 错误: cs.RO 需要 endorsement

```
回到 Stage 3, 去掉 cs.RO cross-list
保留 cs.AI primary + cs.LG cross-list 即可
```

### 错误: TeX 编译失败

最可能:某个包缺失。看 arxiv 给的 error log,找到 `! LaTeX Error: File 'xxxxx.sty' not found`。

通常 arxiv 服务器有完整 texlive,不应该缺。如果真缺:
```
本地 tlmgr install <package>
重打包 bundle
重新上传
```

### 错误: PDF 看起来不对

回到本地, edit `crossvla.tex`, 重跑 4 步编译 (`pdflatex × 2 + bibtex + pdflatex`), 重打包 bundle。

## 备份保留

提交完成后,**保留** `crossvla_arxiv_v1.tar.gz` 在仓库,用于:
- 之后做 v2 时 base
- 如果 arxiv 有 issue 可以重新上传

## 当前 paper 状态

- **Title**: CrossVLA: Cross-Paradigm Post-Training and Inference Optimization for Vision-Language-Action Models
- **Author**: Zhi Liu (Tianjin University)
- **Pages**: 14
- **Figures**: 6 (4 used in body + 2 archived for v2)
- **Citations**: 45 (OpenAlex-verified)
- **Bundle size**: 1.45 MB compressed

## 投稿后 我做什么

等你给我 arxiv ID 后:
1. 简历 P21 末尾加 `arXiv:XXXX.YYYYY`
2. 简历 PDF 重生
3. README 顶部加 `[![arxiv](https://img.shields.io/badge/arXiv-XXXX.YYYYY-b31b1b.svg)](https://arxiv.org/abs/XXXX.YYYYY)`
4. paper 最后一页加 published-on-arxiv note
5. push 双仓 + 飞书更新

---

**有任何 stage 出问题,贴 error message 给我,5 分钟内修。**
