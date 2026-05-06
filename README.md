# ro_planning — OpenVLA Post-Training & LIBERO Leaderboard

> 作者：刘志 (liuzhi7 (Independent))
> 仓库：[<private-gitlab>/ro_planning](https://<private-gitlab>/ro_planning)
> 创建：2026-05-06

---

## 🎯 项目目标（按优先级）

1. **LIBERO 公开排行榜成绩** — 复现并尝试超越 OpenVLA 论文 SR
2. **演示视频** — 漂亮的 LIBERO 任务执行 demo（10 任务 / 4 suite）
3. **后训练/VLA 实战** — OpenVLA 7B LoRA 微调，掌握工具链
4. **CoRL 2026 论文**（次优先）— 可选 LoRA-MoE 或其它创新点

> ⚠️ **当前优先级修正（2026-05-06）**：MoE 架构不再是首要目标，先冲榜+录视频。

---

## 📂 目录结构

```
ro_planning/
├── README.md                    # 本文件
├── docs/                        # 计划 / 实验记录 / 设计文档
│   ├── plan_v1.2.md            # 当前主计划
│   ├── env_setup.md            # 镜像 + 数据 + 开发机使用
│   ├── libero_leaderboard_plan.md  # 冲榜计划
│   ├── experiment_log.md       # 实验流水账
│   └── handover.md             # 给下一个 session 的交接信息
├── docker/                      # 镜像构建
│   ├── Dockerfile
│   └── .dockerignore
├── code/                        # 代码
│   ├── openvla/                # OpenVLA 仓库 vendored
│   ├── lora_moe/               # 我们的 MoE 代码（暂保留，次优先）
│   ├── scripts/                # 训练 / 评测脚本
│   ├── configs/                # 训练配置
│   └── tools/                  # smoke / 监控 / 视频生成
├── notebooks/                   # 探索性分析
├── tests/                       # 单元测试
└── assets/
    ├── videos/                 # 演示视频（>10MB 不入 git）
    └── figures/                # 论文用图
```

---

## 🚀 快速开始

### 拉镜像（开发机）
```bash
docker pull <registry>/planningmodel:openvla-v1.0-cu118-py310
```

### 在开发机数据布局（已就绪）
```
/workspace/jfs/   (JuiceFS, 744T 可用)
├── models/openvla-7b/                        # 15 GB OpenVLA-7B 权重
├── datasets/modified_libero_rlds/            # 9.6 GB LIBERO RLDS
├── output/                                   # 训练 ckpt
└── hf_cache/                                 # HuggingFace cache
```

### 容器内挂载约定
```
/workspace/datasets  -> /workspace/jfs/datasets
/workspace/models    -> /workspace/jfs/models
/workspace/output    -> /workspace/jfs/output
~/.cache/huggingface -> /workspace/jfs/hf_cache
```

---

## 📅 路线图（详细见 `docs/plan_v1.2.md`）

| Phase | 时间 | 目标 |
|-------|------|------|
| 0. 基础设施 | Week 1-2 | ✅ 镜像/数据/环境就位（已完成 80%） |
| 1. **LIBERO 冲榜** | Week 2-4 | 4 suite LoRA 微调 → 排行榜提交 |
| 2. **演示视频** | Week 4-5 | 录制 4 suite × 多任务 demo |
| 3. 后训练实战 | Week 5-8 | LoRA / 4-bit / 量化部署 |
| 4. 论文方向 | Week 8+ | 视情况投 CoRL 2026 |

---

## 🔗 重要链接

- 飞书 wiki（VLA 研究）：<notes-wiki>
- Octo & OpenVLA 批判分析：<notes-wiki>
- RT-1/RT-2 批判分析：<notes-wiki>
- LIBERO 排行榜：https://libero-project.github.io/
- OpenVLA 官方：https://github.com/openvla/openvla
- OpenVLA 论文：https://arxiv.org/abs/2406.09246

---

## 📌 当前状态（2026-05-06）

✅ 已完成：
- OpenVLA 镜像构建 + 推送 3 仓库
- 模型 + 数据传输到开发机 JuiceFS
- 开发机 smoke test 通过（H20-3e，4.4 GB / 3.03 Hz）

⏳ 进行中：
- LIBERO-Spatial LoRA 基线启动

🔜 下一步：
- LIBERO 4 suite 串行训练
- 演示视频录制脚本
