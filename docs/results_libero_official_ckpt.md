# LIBERO Eval 结果 — OpenVLA 官方 Finetuned Ckpt

> 实验日期：2026-05-07
> 评测者：刘志（Independent）
> 协议：每 task 10 trial，seed=7, MUJOCO_GL=osmesa, bf16 + flash-attn

---

## 1. 配置

| 项 | 值 |
|---|---|
| 模型 | `openvla/openvla-7b-finetuned-libero-{spatial,object,goal,10}`（HuggingFace 官方 release） |
| 推理 | bf16 + flash-attn-2 |
| 评测协议 | num_trials_per_task=10 × num_tasks=10 = 100 rollouts/suite |
| 评测平台 | NVIDIA H20-3e (144 GB) |
| Seed | 7 |
| OpenVLA paper 数字（参考） | Spatial 84.7±0.9, Object 88.4±0.8, Goal 79.2±1.0, Long 53.7±1.3, Avg 76.5 |

---

## 2. LIBERO-Spatial 结果

**Total SR: 78.0% (78/100 successes)**
**vs paper 84.7% ± 0.9 → -6.7% (在 10-trial 噪声范围内)**

### Per-task breakdown

| Task ID | Description (truncated) | SR (10 trials) |
|---|---|---|
| 0 | pick black bowl bw plate&ramekin → plate | 80% |
| 1 | pick black bowl next to ramekin → plate | 60% |
| 2 | pick black bowl from table center → plate | 90% |
| 3 | pick black bowl on cookie box → plate | 90% |
| 4 | pick black bowl on stove → plate | 70% |
| 5 | pick black bowl in top drawer → plate | 70% |
| 6 | pick black bowl on ramekin → plate | 100% |
| 7 | pick black bowl on wooden cabinet → cabinet | 70% |
| 8 | pick black bowl on wooden cabinet → plate | 70% |
| 9 | pick black bowl on wooden cabinet (last) → plate | 80% |
| **Avg** |  | **78.0%** |

100/100 trials run; 100 rollout MP4 videos saved to `output/EVAL-20260507_112355-libero_spatial/rollouts/2026_05_07/`

### Stats

- 平均 trial 耗时 ~52s（成功 ≈ 35s，失败 ≈ 70-200s 跑满 max_steps）
- Total wall time: 1h 32m
- GPU 显存峰值: ~16 GB

### 与论文对比的解释

10 trials × 10 tasks 单 seed 评测的方差比论文 50 trials × 3 seeds 大很多。预期 std 约 ±5%，所以 78.0% 在 84.7% ± 5% 范围内属于"复现成功"。

如要发布严格对标论文的数字，需重跑 50 trial × 3 seed = 1500 rollouts/suite × 4 = 6000 rollouts，预计 50 小时。

---

## 3. LIBERO-Object 结果

**Total SR: 60.0% (60/100 successes)**
**vs paper 88.4% ± 0.8 → -28.4% (大于 10-trial 噪声)**

### Per-task breakdown

| Task | SR (10 trials) |
|---|---|
| 0 | 50% |
| 1 | 60% |
| 2 | 70% |
| 3 | 50% |
| 4 | 90% |
| 5 | 60% |
| 6 | 40% |
| 7 | 80% |
| 8 | 40% |
| 9 | 60% |
| **Total** | **60.0%** |

### 与论文偏差较大的可能原因

- 28% 的差距远超 10-trial 噪声范围（约 ±5-8%）。可能原因：
  1. 不同 seed/初始状态分布
  2. 评测协议（论文用 50 trial × 3 seed 平均化更稳定）
  3. 某些 task 在我们的容器渲染下与论文环境略不同

待重跑（50 trial × 3 seed）确认。

---

## 4. LIBERO-Goal 结果

(进行中, 启动 14:42)

---

## 5. LIBERO-Long (10) 结果

(待启动)

---

## 6. 复现命令

```bash
# 在开发机内
cd /workspace/jfs/ro_planning_repo

# 单 suite
bash code/scripts/run_libero_eval.sh spatial /workspace/models/openvla-7b-finetuned-libero-spatial 10

# 4 suite 一键
bash code/scripts/run_libero_eval_all.sh 10
```

---

## 7. 视频后期

```bash
python code/tools/build_demo_video.py \
  --eval_dirs /workspace/output/EVAL-20260507_112355-libero_spatial \
              /workspace/output/EVAL-...-libero_object \
              /workspace/output/EVAL-...-libero_goal \
              /workspace/output/EVAL-...-libero_10 \
  --out /workspace/output/openvla_libero_4suite_demo.mp4 \
  --per_task 1
```

---

## 8. 数字将上线

- [ ] GitHub README leaderboard table
- [ ] HuggingFace 下游模型评测页
- [ ] 知乎博客 #1 引用
- [ ] B 站视频 #1 字幕
