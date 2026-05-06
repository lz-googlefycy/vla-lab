# LIBERO 冲榜 + 演示视频实战计划

> 创建：2026-05-06
> 目标：3-4 周内做出**可上排行榜**的成绩 + **可投放**的演示视频

---

## 1. LIBERO 简介

LIBERO 是 NeurIPS 2023 提出的 lifelong robot learning benchmark，**4 个 suite**：
| Suite | 任务数 | 难点 |
|---|---|---|
| LIBERO-Spatial | 10 | 同物体不同空间位置 |
| LIBERO-Object | 10 | 同布局不同物体 |
| LIBERO-Goal | 10 | 不同任务目标 |
| LIBERO-Long (LIBERO-10) | 10 | 长程多步任务 |

**评测协议**：每任务 50 次 rollout × 3 seed = 1500 trial / suite，平均成功率（SR%）。

---

## 2. 当前 SOTA（OpenVLA 论文 v2 报告）

| Method | Spatial | Object | Goal | Long | Avg |
|---|---|---|---|---|---|
| Diffusion Policy | 78.3 ± 1.1 | **92.5** ± 0.7 | 68.3 ± 1.2 | 50.5 ± 1.3 | 72.4 |
| Octo fine-tuned | 78.9 ± 1.0 | 85.7 ± 0.9 | **84.6** ± 0.9 | 51.1 ± 1.3 | 75.1 |
| **OpenVLA fine-tuned** | **84.7** ± 0.9 | 88.4 ± 0.8 | 79.2 ± 1.0 | **53.7** ± 1.3 | **76.5** |

**我们的复现目标**：≥ 76.5% 平均（OpenVLA 水平），冲击 78%+ 上排行榜。

---

## 3. 三周里程碑

### Week 1（5/6-5/12）：第一个 suite 跑通

#### 任务清单
- [ ] 在开发机准备 LIBERO 训练脚本（基于 `vla-scripts/finetune.py`）
- [ ] LIBERO-Spatial LoRA r=32 微调（24h）
  - 配置参考 OpenVLA appendix E
  - BS：本地 batch 16，grad_accum=4 → effective 64
  - LR：5e-4（LoRA 标准）
  - Steps：50K（参照论文）
  - 保存 ckpt 间隔：5K
- [ ] LIBERO-Spatial 评测（500 rollout × 3 seed）
- [ ] **第一个数字**：Spatial SR 落库

**验收**：Spatial SR ≥ 80%（论文 84.7 的 95% 即可）

### Week 2（5/13-5/19）：4 suite 全跑

- [ ] LIBERO-Object 训练 + 评测
- [ ] LIBERO-Goal 训练 + 评测
- [ ] LIBERO-Long 训练 + 评测
- [ ] **4 suite 全部数字落库**

**验收**：平均 SR ≥ 75%

### Week 3（5/20-5/26）：超越 + 视频

- [ ] **超参数搜索**：LR / steps / batch / seed
- [ ] **数据增强尝试**：image augment / domain randomization
- [ ] **多 ckpt 集成**：top-k checkpoint averaging
- [ ] **录制演示视频**（异步进行）
- [ ] **排行榜提交**

**验收**：
- 数字层面：≥ 1 个 suite 超过 OpenVLA
- 视频：4 suite 主 demo 完成

---

## 4. 训练配置详细

### 4.1 LoRA 微调参数（参考 OpenVLA appendix E）

```bash
# Spatial / Object / Goal: 50K steps
# Long: 100K steps

torchrun --standalone --nnodes 1 --nproc-per-node 1 \
    vla-scripts/finetune.py \
    --vla_path /workspace/models/openvla-7b \
    --data_root_dir /workspace/datasets/modified_libero_rlds \
    --dataset_name libero_spatial_no_noops \
    --run_root_dir /workspace/output/libero_spatial \
    --use_l1_regression True \
    --use_diffusion False \
    --use_film False \
    --num_images_in_input 1 \
    --use_proprio False \
    --batch_size 16 \
    --learning_rate 5e-4 \
    --num_steps_before_decay 100000 \
    --max_steps 50000 \
    --use_val_set False \
    --save_freq 5000 \
    --save_latest_checkpoint_only False \
    --image_aug False \
    --lora_rank 32 \
    --wandb_project openvla-libero \
    --wandb_entity liuzhi7
```

### 4.2 评测命令

```bash
python experiments/robot/libero/run_libero_eval.py \
    --pretrained_checkpoint /workspace/output/libero_spatial/<ckpt> \
    --task_suite_name libero_spatial \
    --num_trials_per_task 50 \
    --num_episodes_per_task_in_a_row 50 \
    --use_l1_regression True \
    --num_images_in_input 1 \
    --use_proprio False \
    --center_crop True
```

### 4.3 资源消耗

| 阶段 | 时长 | GPU 显存 |
|---|---|---|
| LoRA train (50K steps) | ~24h on H20 | ~50 GB |
| Eval (500 rollout) | ~2h | ~30 GB |
| 4-bit eval | ~1h | ~10 GB |

---

## 5. 演示视频规划

### 5.1 视频内容

**主 demo**（1-2 分钟）：
- 开头：LIBERO 简介 + OpenVLA 介绍（5s）
- 4 suite 各 1 任务展示（每 task 10s × 4 = 40s）
- 整体成功率结果（10s）
- 致谢 + 链接（5s）

**详细 demo**（每 suite 30 秒，4 个）：
- 5 个任务连续展示
- 标注 instruction、SR、视角

### 5.2 录制脚本设计

需要自定义 `run_libero_eval_with_video.py`：
- 在 `run_libero_eval` 基础上 hook
- 用 robosuite 渲染器获取 RGB 帧（256x256 或更大）
- 保存为 MP4（用 imageio[ffmpeg]）
- 视频文件存到 `/workspace/output/videos/`

### 5.3 后期工具

| 工具 | 用途 |
|---|---|
| ffmpeg | 拼接 / 转码 / 字幕 |
| moviepy | Python 内剪辑 |
| OBS | 屏幕录制（备用） |

---

## 6. 排行榜提交流程

LIBERO 官方 GitHub 接受 PR，格式：
- 模型名 + 作者 + affiliation
- 4 suite SR + std
- 复现脚本 / 论文链接
- demo 视频链接

提交地址：https://github.com/Lifelong-Robot-Learning/LIBERO （或他们最新维护的页面）

---

## 7. 实验流水（实时维护 `experiment_log.md`）

每跑完一组实验，往 `docs/experiment_log.md` 追加一行，避免遗失。

---

## 8. 风险预案

| 风险 | 应对 |
|---|---|
| 复现达不到 84.7% | 优先用 OpenVLA 官方 LIBERO ckpt（HF: `openvla/openvla-7b-finetuned-libero-spatial` 等）直接评测做基线 |
| 训练崩 / 断 | nohup + setsid + 心跳监控（参考 TrajFlow 经验） |
| 数据格式问题 | LIBERO RLDS `_no_noops` 已是 OpenVLA 适配版，无需额外处理 |
| 评测 50 trial × 10 task × 3 seed = 1500 太慢 | 先跑 50 trial × 1 seed 看趋势，最后跑全量 |
| 排行榜不接受小米 affiliation | 用个人邮箱投，affiliation 写 Personal |

---

## 9. 立即下一步

1. **本机推 GitLab**（当前任务）
2. **开发机启动 LIBERO-Spatial 训练**
3. **训练时同步写视频录制脚本**
