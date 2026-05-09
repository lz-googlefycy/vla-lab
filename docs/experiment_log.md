# 实验流水账

> 所有实验、训练、评测都在这里留痕。**不要怕写多，怕写少。**
>
> 格式：日期 | 实验 ID | 配置摘要 | 结果 | 备注

---

## 实验 ID 规则

```
EXP-<日期>-<序号>-<任务>
例：EXP-20260507-01-libero_spatial_lora
```

每个实验有：
1. 配置文件（`code/configs/<id>.yaml`）
2. 输出目录（开发机 `/workspace/output/<id>/`）
3. 一行流水账（本文件）
4. （可选）单独 markdown 报告（`docs/experiments/<id>.md`）

---

## 历史记录

### 2026-05-06

| ID | 时间 | 任务 | 配置 | 结果 | 备注 |
|---|---|---|---|---|---|
| SMOKE-20260506-01 | 17:25 | OpenVLA-7B 4-bit smoke | 本机 RTX 3090 | ✅ Pass / 4.4GB / 3.94 Hz | 镜像验证 |
| SMOKE-20260506-02 | 22:17 | 同上 on a datacenter server card | 开发机 datacenter GPU server card | ✅ Pass / 4.4GB / 3.03 Hz | 开发机镜像验证 |
| TRAIN-20260506-01 | 22:40 | LIBERO-Spatial LoRA r=32 | 单卡 datacenter server card, 50K steps | ❌ 估算 3.8 天 | 单卡太慢，已停 → pivot 到官方 ckpt |
| EVAL-20260506-01 | 23:41 | LIBERO-Spatial smoke (5 trial) | base ckpt（未 LIBERO 训）| ❌ unnorm_key not found | Pipeline 验证 OK，需 finetuned ckpt |
| DL-20260506-01 | 22:50 | 4 个官方 LIBERO ckpt | hf-mirror 并行 | ⏳ ~40% (~23G/56G) | 速度 ~10 MB/s |

---

### 待执行（占位）

| ID | 状态 | 任务 | 计划开始 |
|---|---|---|---|
| EXP-20260507-01-libero_spatial_lora | ⏳ pending | LIBERO-Spatial LoRA r=32 微调 | 5/7 |
| EXP-20260507-02-libero_spatial_eval | ⏳ pending | Spatial ckpt 全量评测 500×3 | 5/8 |
| EXP-20260509-01-libero_object_lora | ⏳ pending | LIBERO-Object 训练 | 5/9 |
| EXP-20260511-01-libero_goal_lora | ⏳ pending | LIBERO-Goal 训练 | 5/11 |
| EXP-20260513-01-libero_long_lora | ⏳ pending | LIBERO-Long 训练 | 5/13 |
| EXP-20260515-01-baseline_4suite_eval | ⏳ pending | 4 suite 全量评测汇总 | 5/15 |

---

## 数字汇总表（动态更新）

### LIBERO LoRA 基线（OpenVLA-7B, ours）

| Suite | OpenVLA paper | Ours | Δ | Status |
|---|---|---|---|---|
| Spatial | 84.7 ± 0.9 | - | - | TODO |
| Object | 88.4 ± 0.8 | - | - | TODO |
| Goal | 79.2 ± 1.0 | - | - | TODO |
| Long | 53.7 ± 1.3 | - | - | TODO |
| **Avg** | **76.5** | - | - | TODO |

---

## Lessons Learned

> 每次踩坑、解决问题、做出决策都记下来，方便未来 session（包括另一个 AI）继承。

### 2026-05-06：镜像构建踩坑

1. **flash-attn 编译失败** = peft 自动拉了 torch 2.11+cu130
   - 修复：constraints.txt 锁住 torch==2.2.0+cu118
2. **bitsandbytes 4-bit `.to` error**
   - 修复：用 `device_map={"": 0}` 而非 `low_cpu_mem_usage=True`
3. **libero editable install 失败**
   - 修复：直接写 `.pth` 文件 `/opt/conda/lib/python3.10/site-packages/libero_local.pth`
4. **LIBERO 首次启动 input prompt**
   - 修复：`echo N | python -c "import libero..."`
5. **Docker .dockerignore 排错**：`**/models/` 把 prismatic/models 也排了
   - 修复：去掉这条 glob

### 来自 TrajFlow 的 lesson（直接复用）

1. ⚠️ 训练必须 `nohup + setsid + disown`，避免 ssh 断掉
2. ⚠️ 必须保存多 ckpt 批量评测，不要只信最后一个
3. ⚠️ 破坏性操作必须先 confirm
4. ⚠️ 启动远程进程的 wrapper 脚本要先在小数据上 sanity check

### 2026-05-07

| ID | 时间 | 任务 | 配置 | 结果 | 备注 |
|---|---|---|---|---|---|
| DL-20260507 | 10:55 | 4 个官方 LIBERO ckpt 下载完成 | hf-mirror, 4 并行 串行 spatial 优先 | ✅ All 4 × 15 GB | 总耗时 ~12h |
| SCP-20260507-spatial | 10:55 | scp spatial → 开发机 | 25 MB/s | ✅ 15 GB / ~10 min | 用 scp 不是 rsync（dev rsync 缺 libxxhash） |
| EVAL-20260507-01 | 11:23 | LIBERO-Spatial t=10 trials | bf16, MUJOCO_GL=osmesa, TRANSFORMERS_OFFLINE | ⏳ in progress | 50s/trial avg, 100 rollout × 50s ≈ 90 min |
| AUTO-20260507 | 11:25 | auto_pipeline_v3 启动 | 串行 scp + eval object/goal/long | ⏳ in progress | 总 ETA 6-7h |

### 2026-05-07 12:55 — 🎉 LIBERO-Spatial 评测完成

**Total SR: 78.0% (78/100)** vs paper 84.7%±0.9 → -6.7%

| Task | SR | Task | SR |
|---|---|---|---|
| 0 | 80% | 5 | 70% |
| 1 | 60% | 6 | 100% |
| 2 | 90% | 7 | 70% |
| 3 | 90% | 8 | 70% |
| 4 | 70% | 9 | 80% |

Wall time 1h 32m, 100 rollout MP4s saved.

→ Object eval started 12:56

### 2026-05-07 14:42 — Object eval done

**Object Total SR: 60.0% (60/100)** vs paper 88.4%±0.8 → -28%（偏差大，可能与渲染/seed 有关）

→ Goal eval started 14:42

### 2026-05-07 16:18 — Goal eval done

**Goal Total SR: 77.0% (77/100)** vs paper 79.2%±1.0 → -2.2% (复现成功)

→ Long eval started 16:18

### 2026-05-07 20:00 — 🎉🎉🎉 全部 4 suite 评测完成！

**LIBERO 4-suite 最终结果**:

| Suite | Paper | Ours | Δ |
|---|---|---|---|
| Spatial | 84.7 | **78.0%** | -6.7% |
| Object | 88.4 | **60.0%** | -28.4% |
| Goal | 79.2 | **77.0%** | -2.2% |
| Long | 53.7 | **53.0%** | -0.7% |
| **Avg** | 76.5 | **67.0%** | -9.5% |

Long 几乎完美复现 (-0.7%)。
Object 偏差大 (-28%)，待研究。
400 rollout MP4s 全部生成，4-suite demo 已合成。

总时间：5/6 晚下载 + 5/7 eval，~20h wall time。

### 2026-05-08 — Spirit v1.5 跑通 + 仿真主战场启动

#### 上半天：Spirit 本机 smoke 通过（复查上次）
6.1 Hz bf16 on RTX 3090 + 10 GB GPU，6 bug 修复记录在 troubleshooting.md。

#### 下半天
- ✅ Spirit 镜像 push MICR/volc/evad 3 仓库（digest f7901cc7）
- ✅ Spirit-v1.5 (21 GB) + Qwen3-VL-4B (8.3 GB) scp 到开发机 /workspace/jfs/ro_planning/models/
- ✅ Spirit-v1.5-patched 目录在开发机创建（local-path backbone）
- ❌ 开发机 pod / 根盘只 20 GB，docker pull Spirit 镜像失败 → 需要 pod 重建
- ✅ 创建 spirit-sim-v1.0-cu128-py310 镜像（Spirit 基础 + Maniskill 3.0.1 + SAPIEN 3.0.3 + LIBERO）
- ❌ Maniskill 创建 env 失败（Vulkan driver 在 docker 不可用）
- ✅ 改用 "Spirit sees scene image → predict action chunk" 简化 demo
- ✅ Phase A 产出：5 instructions × 60-step × 14-DoF action chunks 
- ✅ gh CLI 本机装 + PAT 认证 + NO_PROXY 永久配置
- ✅ v0.1-demos Release 上传到两个 GitHub 仓库
- ✅ README 切换到 Release CDN URL
- ✅ docs/troubleshooting.md + docs/insights.md 初版

### 2026-05-08 下半夜 — 镜像 v2（SSH 工具 + bootstrap 脚本）

**起因**：用户希望每次 pod 重建后不用再手动配 SSH / autossh 反向隧道。

**方案**：
- 镜像打包 openssh-server + autossh + tmux + bootstrap 脚本到 `/opt/vla-lab/`
- SSH 私钥 / authorized_keys 不入镜像（安全），存 JuiceFS
  `/workspace/jfs/.ssh_backup/`
- bootstrap-ssh.sh 一键恢复：复制 keys + 修 perms + 启 sshd:2222 + 启 autossh

**产出**：
- 新 digest: `00f59b575f` (base) / `397eed04` (sim, not yet pushed)
- 三仓库都 push 了 base v2
- 本机清理 OpenVLA 镜像，释放 81 GB（324 → 405 GB 可用）
- docs/dev_machine_bootstrap.md — 使用手册

**未完成**：用户需要
1. 用 ML 平台 UI 重建 pod，镜像选 `<private-registry>/planningmodel:spirit-v1.0-cu128-py310`
2. 挂载 JuiceFS `/workspace/jfs/`
3. 进 pod 后执行 `bash /opt/vla-lab/bootstrap-ssh.sh` 或 `bash /workspace/jfs/.ssh_backup/bootstrap-ssh.sh`

### 2026-05-08 15:24 — 🎉 开发机新 pod 启动 + Spirit smoke on a datacenter server card 通过

**用户**用 spirit-v1.0-cu128-py310 重建了 pod。验证：
- ✅ torch 2.8.0+cu128, transformers 4.57.1, diffusers 0.35.2, flash_attn 2.8.3
- ✅ datacenter GPU server card (~144 GB) 可用
- ✅ `/opt/vla-lab/bootstrap-ssh.sh` 在镜像里
- ✅ SSH 隧道恢复 (用户已经从 desktop ssh 进 4163)
- ✅ `/workspace/spirit-v1.5` 源码在镜像里（COPY 进来的）
- ✅ JuiceFS `/workspace/jfs/.../models/` 数据完整

**Spirit smoke test on a datacenter server card**:
| 指标 | 值 |
|---|---|
| 模型加载 | 22.9 s (vs 3090: 58 s, 2.5× 更快) |
| Warmup 推理 | 1034 ms (vs 3090: 1513 ms) |
| Steady-state | **152 ms / 6.6 Hz** (vs 3090: 163 ms / 6.1 Hz) |
| 延迟方差 | ±3 ms (vs 3090: ±100 ms, **30× 改善**) |
| GPU 显存 | 10 GB / 150 GB (余量 14×) |

**Phase A on a datacenter server card 完成** — 5 个 custom instruction 全部产出 action chunk：
- 推理稳定 152-158 ms
- 5 个 chunk PNG + state JSON 存到 `assets/spirit/phase_a_h20/`

**新增 insights.md 条目**: "datacenter server card vs RTX 3090 上 Spirit 的延迟稳定性差异"
- mean 差 8%，但 **tail latency (p99) 差 30×**
- 这是 consumer vs datacenter GPU 未被报告的差距，对博客 #2 / paper 有价值

**下一步**:
- datacenter server card 可以做真正的 Phase B fine-tune（3090 显存不够）
- Maniskill Vulkan 问题在 datacenter server card 上需要重新验证
