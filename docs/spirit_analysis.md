# Spirit v1.5 仓库分析

> 创建：2026-05-06
> 仓库：https://github.com/Spirit-AI-Team/spirit-v1.5
> 模型：https://huggingface.co/Spirit-AI-robotics/Spirit-v1.5
> Stars：562 / Forks：29（截至记录时）
> License：MIT
> 关键开源时间：2026-01 初版（推理 + base ckpt）；2026-04 fine-tune 代码

---

## 1. 一图看清架构

```
Spirit v1.5 = Qwen3-VL backbone + DiT (Diffusion Transformer) head + policy API
                                 ↑                ↑
                         视觉-语言理解        Action 生成
```

**和 OpenVLA / RT-2 对比**：

| 维度 | RT-2 (Google, 闭源) | OpenVLA (开源) | **Spirit v1.5** |
|---|---|---|---|
| Backbone | PaLI-X 5B/55B | Llama-2 7B + DINOv2+SigLIP | **Qwen3-VL**（最新） |
| Action 表示 | 离散 256-bin token | 离散 256-bin token | **DiT 连续 + diffusion** |
| 训练数据 | 自家 130k + WebLI | 970k OXE | RoboChallenge Table30 dirty data |
| 推理 | 1-3 Hz multi-TPU | 6 Hz A100 / 3 Hz H20 | （需测） |
| 开源 | ❌ | ✅ | **✅ MIT** |
| 真机评测 | EDR（关停） | LIBERO 仿真 | **RoboChallenge 真机** |

**Spirit 的关键差异**：
1. **Qwen3-VL 国产 backbone**（不依赖 Llama / PaLI）
2. **DiT diffusion action head**（区别于 RT-2/OpenVLA 离散 token）— 借鉴 π0
3. **真机评测原生**（RoboChallenge 真机集群 ARX5/UR5/Franka/ALOHA）
4. **"Dirty data is the enemy of clean data"** 哲学（论文标题反讽，强调真实多样性）

## 2. 仓库目录速览

```
spirit-v1.5/
├── model/modeling_spirit_vla.py     # 核心架构: Qwen3-VL + DiT head + policy API
├── dataset/                          # data + transforms
├── utils/                            # checkpoint, distributed, normalization, sampling, vlm_utils
├── robochallenge/                    # RoboChallenge runtime wrapper
│   ├── run_robochallenge.py
│   ├── runner/{executor, task_info}
│   ├── robot/{interface_client, job_worker}  # HTTP client + 任务轮询
│   └── utils/
├── scripts/{run_robochallenge.sh, run_finetune.sh}
├── train.py                          # FSDP 多卡训练入口
├── requirements-base.txt             # 推理
├── requirements-train.txt            # 训练增量
└── pyproject.toml                    # uv 兼容
```

仅 9 个 commits、98% Python，结构清晰。

## 3. 关键依赖（**与我们 OpenVLA 镜像不兼容**）

| 包 | OpenVLA 镜像 | Spirit v1.5 | 冲突 |
|---|---|---|---|
| torch | 2.2.0+cu118 | **2.8.0** | ❌ |
| transformers | 4.40.1 | **4.57.1** | ❌ |
| numpy | 1.26.4 | **2.2.6** | ❌ |
| flash-attn | 2.5.5 | **2.8.3** | ❌ |
| 新增 | — | **diffusers==0.35.2** | 需装 |

**结论**：**必须另起镜像或环境**，不能复用 OpenVLA v1.0 镜像。

## 4. 模型与数据

| 资源 | 链接 | 大小估计 |
|---|---|---|
| Spirit v1.5 base | https://huggingface.co/Spirit-AI-robotics/Spirit-v1.5 | 待确认（Qwen3-VL 7B 级） |
| Spirit v1.5 fine-tuned (move_objects_into_box) | https://huggingface.co/Spirit-AI-robotics/Spirit-v1.5-for-RoboChallenge-move-objects-into-box | 同上 |
| 训练数据集示例 | RoboChallenge/task_table30_move_objects_into_box | ? |

## 5. 推理流程（RoboChallenge 模式）

```
本地启动 run_robochallenge.sh
  ↓
环境变量: TASK_NAME / ROBOCHALLENGE_JOB_ID / USER_TOKEN / CKPT_PATH / USED_CHUNK_SIZE
  ↓
HTTP 客户端 (robot/interface_client.py) 连接 RoboChallenge 平台
  ↓
job_worker.py 轮询任务
  ↓
RoboChallengeExecutor 加载 ckpt + 推理 + 上报 action chunk
  ↓
RoboChallenge 真机集群执行 + 评分
```

**关键洞察**：Spirit v1.5 的推理**不是本地仿真**，而是**远程真机集群**执行。需要 USER_TOKEN（账号注册）。

## 6. Fine-tune 配置（官方推荐）

| 参数 | 默认值 |
|---|---|
| 硬件 | **8× A100 80GB** |
| BATCH_SIZE | 32 / GPU |
| MAX_TRAIN_STEPS | 40000 |
| SAVE_STEPS | 2500 |
| 框架 | **PyTorch FSDP** |
| 数据加载 worker | 32 |

**我们的硬件能力评估**：
- 单 H20-3e 144 GB > A100 80GB ✅
- BS 32 应该能跑（Spirit 7B + diffusers, 显存可能比 OpenVLA-7B 高）
- 单卡训 → BS 减到 8/16，steps 翻倍到 80K-160K
- **40K steps × 8 GPU 估算 → 单卡 H20 ~ 5-7 天**

## 7. 我们的复现路线（4 步）

### Step 1（Week 2-3）：推理 smoke test
- [ ] 拉 Spirit v1.5 仓库 + base ckpt
- [ ] 在我们开发机（H20）跑通推理（不上 RoboChallenge，用本地数据）
- [ ] 验证 action shape / dtype / latency
- **产出**：博客 #2 的工程素材

### Step 2（Week 3）：仿真验证
- [ ] 用 LIBERO 数据 / SimplerEnv 让 Spirit 跑评测
- [ ] 对比 Spirit vs OpenVLA 的 SR
- **产出**：横向对比第一份数字

### Step 3（Week 4-5）：XLeRobot 真机
- [ ] Spirit v1.5 → XLeRobot SO-100 推理（推理 base ckpt 直接看效果）
- [ ] 录视频
- **产出**：B 站视频 #1（千寻必看）

### Step 4（Week 5-7）：XLeRobot fine-tune
- [ ] XLeRobot 采 50-100 demo
- [ ] 用 Spirit fine-tune 代码 + LoRA（适应单卡 H20）
- [ ] 评测 fine-tune 前后
- **产出**：HuggingFace release + 博客 #3

### Step 5（Week 8）：RoboChallenge 提交（如可能）
- [ ] 注册账号 / 申请 token
- [ ] 提交 base + fine-tuned ckpt
- [ ] 看 leaderboard 排名
- **产出**：leaderboard 数字

## 8. 关键风险

| 风险 | 应对 |
|---|---|
| Spirit 镜像 build 失败（torch 2.8.0+cu12.x，我们是 cu118） | 另起镜像 spirit-v1.0-cu124-py310 |
| Qwen3-VL backbone 显存超支 | LoRA + 4-bit 量化 |
| RoboChallenge USER_TOKEN 不开放给个人 | 改用 LIBERO/SimplerEnv 评测 + 真机 demo 视频 |
| fine-tune 数据格式（专有？） | 看 RoboChallenge/task_table30_move_objects_into_box 数据集格式，转 LeRobot |
| 单卡训不下 40K steps | 用 LoRA / freeze backbone + DiT 头训 |

## 9. 联系方式（千寻官方）

仓库 README 提供：
- **guojunliang AT spirit-ai.com**
- **miaotianrun AT spirit-ai.com**

> ⚠️ 不要随便 cold email，要等我们有内容再联系（"我用你们 Spirit 跑通了 SO-100，请教一下"）。

## 10. 影响力杠杆要点

每出一个 Spirit 相关产出时：
1. **GitHub** 上 fork Spirit-AI-Team/spirit-v1.5（变成自己的 spirit-v1.5-xlerobot）
2. **HuggingFace** 上传 LoRA / fine-tune ckpt（命名 `liuzhi7/spirit-v1.5-so100-*`）
3. **博客**强调 "Spirit AI" 关键词（SEO + 千寻员工搜得到）
4. **B 站标题**："$660 机器人跑通千寻 Spirit v1.5"（双热点）
5. **可以礼貌 email**：写完整篇博客后发 email + 链接（"想请教 X 问题"），不要伸手党
