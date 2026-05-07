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
| SMOKE-20260506-02 | 22:17 | 同上 on H20 | 开发机 H20-3e | ✅ Pass / 4.4GB / 3.03 Hz | 开发机镜像验证 |
| TRAIN-20260506-01 | 22:40 | LIBERO-Spatial LoRA r=32 | 单卡 H20, 50K steps | ❌ 估算 3.8 天 | 单卡太慢，已停 → pivot 到官方 ckpt |
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
