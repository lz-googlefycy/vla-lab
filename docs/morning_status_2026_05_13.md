# 早安报告 · 2026-05-13

## TL;DR

**4 个 GRPO MLP 任务需要重启一次**（代码 fix 后的第二轮）。

命令完全没变，挂上最新代码即可。

---

## 昨晚发生了什么

### 23:00-01:00 你起的 4 个 MLP GRPO 任务

1. **Spatial** — step 0 后 OOM（commit 前 `adapters/openvla.py` bug）
2. **Object** — step 0 完成 (187s, mean_reward=0.875)
3. **Goal** — step 0 完成 (163s, mean_reward=0.877)
4. **Long10** — KeyError: 'libero_long10'（`train_grpo.py` 里 'long10' 没映射到 'libero_10'）

### 01:00-01:30 我修了第一轮 bug

- `train_grpo.py`: 加 `long10 → libero_10` mapping（commit `3e3f9ca`）
- `adapters/openvla.py`: 加 gradient checkpointing（commit `c2529d1`）

你重启 4 个任务。

### 01:30-02:10 第二轮观察：**全部 4 个 stall**

- 4 个任务 step 0 都完成后**40-60 分钟不动**
- log 文件停在 step 0，文件 touched 时间 ~3600s 没更新
- MLP 任务没 crash 也没进展 — 典型死锁征兆

**诊断**：gradient checkpointing 在 OpenVLA 这种自定义 HF 模型（vision encoder + project + Llama concat）上，`enable_input_require_grads()` 没法让 requires_grad 从 frozen vision 传到 Llama layer → checkpoint recompute 路径里 autograd graph 构建失败或退化为无限 recompute。

### 02:10 我做的最终 fix（commit `ec7def9`）

**回滚 gradient checkpointing**，改为**让 reference forward 跑在 `torch.no_grad()` 里**：
- Current policy forward: 正常建 autograd graph（为 backward 准备）
- Reference forward: `with torch.no_grad():`，不建第二张 graph
- 峰值显存 ≈ 1 次 forward（~50GB on OpenVLA-7B, K=2, T=180）
- 完全避免"2 forward 叠加" 的显存问题

这是标准 DPO/GRPO memory pattern（DPO 已经在 cloudml 跑通，所以我们知道这个 pattern OK）。

---

## 你需要做的

**重启 4 个 GRPO 任务**（命令不变，挂最新代码 `ec7def9` 上的 dev pod 共享盘）。

**清理 + 启动步骤**：
1. MLP 控制台停掉 4 个还活着的旧任务（他们 stall 状态）
2. 删除旧 output 目录（可选，但建议干净）：
   ```bash
   ssh -p 4321 root@<dev_pod_ip> rm -rf /e2e-data/users/liuzhi7/vla_workspace/output/h20_grpo_*
   ```
3. 用**原来的命令**启动 4 个新任务（见 `docs/MLP_TRAIN_CMDS.md` 或之前的 chat）

**启动后日志应该显示**：
```
[openvla-adapter] LoRA injected: 128 Linears, trainable 16.8M / 7558M
[openvla-adapter] frozen reference: 256 param tensors
...
[train] starting 500 steps, K=2
{"step": 0, ...}       <- ~3 min 后
{"step": 10, ...}      <- ~15-20 min 后  这次应该会出！
```

**step 10 出了就说明 fix 生效，训练正常进行**。

---

## 同时 cloudml 昨晚的进度

- **Goal DPO multi-seed (seed 1337+2026)**: 过去 3+ 小时跑到 task 9/10, ~72% 成功率（和 seed 42 的 74% 一致 → noise band 稳）
- **Long10 DPO multi-seed**: Goal 完成后自动接力（预计 ~09:00-10:00 AM 完成）

我明天 (你醒后) 会自动 rsync multi-seed 数据回本机，合进 paper §4.2。

---

## 三代 commits 清单

| commit | 说明 |
|---|---|
| `3e3f9ca` | 第一轮 fix: long10 → libero_10 key mapping |
| `c2529d1` | 第二轮 fix（错的）: gradient checkpointing |
| `ec7def9` | 第三轮 fix（对的）: 回滚 + no_grad for ref forward |

Dev pod 共享盘 `git log --oneline -1` 应该显示 `ec7def9`。

---

## 如果还是卡 / OOM

降 `--max_chunk_len 180` → `150` 或 `120`（进一步减少单个 forward 的显存）。

如果 `--max_chunk_len 120` 还不行，降 `--group_size 2` → `1`（但 K=1 的 GRPO 退化为 advantage=0，没意义，只是让它先跑通）。

再不行来找我。
