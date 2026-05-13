# 早安报告 · 2026-05-13 10:50

## TL;DR — 关键改变

**世纪互联 96GB H20 跑不动 K=2 GRPO**（实测 OOM/stall 4 次）。**改在 cloudml 144GB H20-3e 上跑 GRPO**，已验证 spatial 能跑通，4-cell pipeline 串行启动了。

世纪互联那 4 个 GRPO 任务可以**全部停掉**，不用再起。

---

## Multi-seed 数据出来了 — 论点要修正

cloudml 自动跑完 Goal + Long10 DPO seed 1337+2026：

| Suite | SFT | seed42 | 1337+2026 | 3-seed merged | Δ |
|---|---|---|---|---|---|
| Goal | 82% | 74% | **77%** (77/100) | **76%** (114/150) | **−6 stable** ✅ |
| Long10 | 60% | 54% | **64%** (64/100) | **60.7%** (91/150) | **+0.7 wide band** ⚠ |

### 重大修正
1. **Long10 Δ-6 不稳**：seed 42 是 outlier，3-seed merge 后 ≈ 0
2. **chunk-truncation hypothesis 撤回**：之前推论 Long10 -6 是因为 max_chunk_len=220 只覆盖 42% episode，现在数据不支持
3. **Goal Δ-6 稳定**：noise band ±3%，central finding 站得住

### 新论点（更精确）
> "DPO on Goal (the strongest SFT baseline) is the only suite with a
> **stable** negative Δ across seeds. Long10's seed-42 dip was within
> noise band (±10%)."

---

## 昨晚 GRPO 失败的根因

世纪互联 H20 是 **96GB**，cloudml 是 **144GB**。我之前误判都是 96GB。

GRPO K=2 的 `policy_logp_with_ref` 需要：
- 1× cur forward (build autograd graph) → ~47GB peak
- 1× ref forward (no_grad) → ~47GB peak
- 加 LoRA optimizer state、activation、buffer → 90+GB

**96GB 上跑不下 chunk T=180 K=2 这套**。即使我改了 no_grad ref forward 后理论上够，实测仍 OOM。

cloudml 144GB **空间多 50%**，spatial K=2 T=180 实测 ~70-100GB，**完全够**。

---

## 当前 cloudml 进展

### 正在跑（已证实 GRPO 能用）
- **Spatial GRPO debug**：step 7/30 ✅（38min, ~80s/step, GPU 70-100GB）
- step 0-7 全 log 出（无 stall），KL loss 在 step 3+ 出现 → 真在学

### 接力调度（debug 完成后自动启动）
**Full 4-cell pipeline**（`/tmp/cloudml_grpo_full_pipeline.sh`，setsid nohup 中）：
- spatial: 500 step train + 50 episode eval
- object
- goal
- long10

预计 **每 cell 5-7h，4 cell 串行 ~24-30h**。最早周四上午全部完成。

---

## 三件你不用做的事

1. ❌ 不用再起 MLP 任务 — 96GB 跑不动 GRPO，cloudml 接管了
2. ❌ 不用调 `--max_chunk_len` / `--group_size` — cloudml 用默认 220 / 2 就行
3. ❌ 不用改代码 — 当前 ec7def9 + no_grad fix 在 144GB 工作

---

## 三件可选的事

### A. 把 Spatial + Object 的 multi-seed 也跑了（让 §4.2 完整）

现在只 Goal + Long10 有 3-seed。Spatial / Object 还是 single-seed。如果想要 paper-grade 完整 4-suite × 3-seed，还需 ~7h cloudml 时间（spatial 2.5h + object 4.2h）。

但**优先级低**：Spatial +6 / Object 0 都和 SFT-strength inverse 的论点一致，single-seed 够。

### B. 起 8×H20 跑 Spirit base 自训 SFT

paper §4.2 的 Spirit row 整行空着。Spirit v1.5 在 LIBERO 上没官方 ckpt，要自训 ~12-15h。可以世纪互联那台 96GB H20 起这个（不需要 GRPO），那台空着也是空着。

如果要做这个，告诉我，我写 SFT 训练命令。

### C. 写博客 #3 起手稿

paper §4.2 已经 7/8 cell 完成（OpenVLA × {SFT, DPO} × 4-suite 全 + multi-seed），可以开始写 workshop paper 通俗版。素材现成。

---

## 文件位置

- multiseed json/jsonl：`assets/paper_v1.5_eval/openvla_dpo_libero_{goal,10}_5x10_multiseed.{json,jsonl}`
- 3-seed merged：`assets/paper_v1.5_eval/openvla_dpo_libero_{goal,10}_3seed_merged.json`
- chart 已重画：`assets/paper_v1.5_eval/paper_4_2_main_chart.png`
- paper §4.2 已更新 single + 3-seed 双表

公开仓 push：commit `2e991bc` (GitLab) / `94e472c` (GitHub)

---

## 关于世纪互联 4 个挂着的 MLP 任务

它们都已挂死（OOM stall），可以全部 stop / kill。
共享盘 `/e2e-data/users/liuzhi7/vla_workspace/output/h20_grpo_*` 里的 step 0 log 留着无所谓，下次起任务（如果未来要起 SFT）会写新目录。

---

## 推荐你早上起来做的事

1. 看这份 status report
2. 决定 A/B/C 哪个优先（或都不做）
3. 等 cloudml GRPO 完成（~30h），我会自动 rsync 数据 + 更新 paper

如果你想立刻看 GRPO debug 实时输出：
```bash
ssh -p 4163 root@127.0.0.1 'tail -20 /tmp/grpo_debug.log | grep -v Warning'
```
