# 项目主计划 v1.4（FINAL）— 影响力工厂

> 创建：2026-05-06
> 触发：用户重新校准目标 — 转行机器人 + 找千寻智能 offer + 开源影响力
> 上一版：v1.3（OpenVLA 官方 ckpt 直接评测冲榜）

---

## 0. TL;DR

**双线并行：A 线 Spirit v1.5 直击千寻 + B 线 OpenVLA 横向对比**，用 **XLeRobot 真机 + LIBERO 仿真**双向产出，**每周至少一条公开内容**，4 个月做出能让千寻 HR 主动联系的作品集。

---

## 1. 战略重定位

### 1.1 真实目标

| 表层目标 | 真实目标 |
|---|---|
| LIBERO 冲榜 + 视频 | 建立机器人圈可见度（GitHub star / Twitter / 知乎） |
| OpenVLA 微调 | 掌握"千寻在用"的技术栈（VLA / 后训练 / 部署） |
| 论文（CoRL） | 简历亮点 + 面试谈资 |
| MoE 创新 | （次要，可放后） |

**核心 KPI**：3-6 个月后，千寻面试官主动说："我看到你 GitHub / 知乎 / B 站的 XX 项目了"。

### 1.2 千寻智能（Spirit AI）情报

| 维度 | 信息 |
|---|---|
| 创始人 | **韩峰涛**（前珞石 CTO，工业机器人）+ **高阳**（清华 + Berkeley，VLA / EfficientZero / ViLa / CoPa） |
| 团队来源 | UC Berkeley / CMU / 清华 / 北大 / 字节 / **小米** / 腾讯 |
| 旗舰产品 | **Spirit v1.5**（VLA 大模型）+ **Moz1** 人形机器人 |
| 评测 | RoboChallenge **Table30** 登顶（2026.01），超 π0.5 |
| 技术特色 | "dirty data" 多样性训练、零样本泛化 |
| 工业落地 | 宁德时代电池产线 |
| 开源 | Spirit v1.5 已在 [Spirit-AI-Team/spirit-v1.5](https://github.com/Spirit-AI-Team/spirit-v1.5) 开源 |

### 1.3 XLeRobot 硬件优势

- **$660 / 4h 组装**：SO-100 双臂 + 移动底座 + 头部相机
- **HuggingFace LeRobot 生态原生支持**
- **已组装 + 校准 OK**（用户确认）
- **能立即跑示教**

### 1.4 目标岗位定位

**VLA 算法工程师**（高阳那条线：模型 / 训练 / 评测）
- 关键技能：VLA 训练 + 微调 + 评测 + 数据 + 真机部署
- 加分项：开源贡献 + 公开内容输出 + 真机经验

---

## 2. 三大产出资产

| 资产 | 说明 | 评估时机 |
|---|---|---|
| **GitHub `ro_planning`** | 主仓库，所有代码 + 文档 | 持续 commit |
| **HuggingFace `liuzhi7`** | 至少 3 个 model + 1 个数据集 | Month 4 末 |
| **作品集 README** | 个人主页 / GitHub Pages，集成所有视频博客链接 | Month 4 末 |
| **博客矩阵** | 知乎 5+ 篇 / Twitter 30+ 条 / B 站 5+ 视频 | 持续 |

---

## 3. 三条主路径（双线并行 + 数据集贡献）

### 🥇 路径 A：Spirit v1.5 + XLeRobot（千寻最爱）

| 步骤 | 产出 | 影响力杠杆 |
|---|---|---|
| 1. Clone Spirit v1.5 跑通推理 | 博客：「Spirit v1.5 跑通需要踩的坑」 | 千寻员工会读 |
| 2. Spirit v1.5 接 XLeRobot 真机 | **B 站视频：Spirit 在 SO-100 上做家务** | 千寻一定会看到 |
| 3. XLeRobot 采 50-200 demo → fine-tune Spirit | 博客 + HF checkpoint | 简历核心成果 |
| 4. RoboChallenge Table30 提交（如规则允许） | leaderboard 排名 | 学术 + 工业可见 |

**面试金句**：
> "我把贵司的 Spirit v1.5 跑通了真机，并且在我自己的 XLeRobot 上做了 fine-tune，发现的问题是 X，你们看怎么样？"

### 🥈 路径 B：OpenVLA 横向对比

| 步骤 | 产出 |
|---|---|
| 1. LIBERO 4 suite 官方 ckpt 评测 + 视频 | 博客 #1（即时） |
| 2. OpenVLA → SO-100 LoRA 跨形态迁移 | 博客 + HF checkpoint |
| 3. OpenVLA-OFT 复现（25× 加速） | 博客 |
| 4. **横向对比：OpenVLA / Spirit / SmolVLA / π0 在 SO-100** | 爆款博客 + 视频 |

### 🥉 路径 C：XLeRobot 数据贡献（社区影响力）

| 步骤 | 产出 |
|---|---|
| 1. 用 XLeRobot 采 100-200 条 demo | HF 数据集 release |
| 2. LeRobot 仓库提 PR / docs 加教程 | LeRobot 贡献者 |
| 3. "$660 机器人 24h 干家务"主题视频 | B 站 / YouTube 头部题材 |

---

## 4. 16 周时间线

### Week 1-2：基础 + 第一桶金

| 时段 | A 线 | B 线 | 交付物 |
|---|---|---|---|
| Week 1 ✅ | — | OpenVLA 镜像 + 数据 + 开发机 | 镜像 v1.0 |
| Week 2 上半 | Clone Spirit v1.5，摸结构 | LIBERO ckpt 下完 + rsync | GitLab 仓库 push |
| Week 2 下半 | Spirit v1.5 推理 smoke | LIBERO-Spatial 评测 + 视频 | **博客 #1：批判性论文分析**（已有素材秒出） |

### Week 3-4：双线启动

| 时段 | A 线 | B 线 | 交付物 |
|---|---|---|---|
| Week 3 | Spirit v1.5 接 XLeRobot 推理（仿真） | LIBERO 4 suite 评测完 | **B 站视频 #1：4 suite OpenVLA demo** |
| Week 4 | XLeRobot 真机示教数据采集 50 条 | OpenVLA 4-bit 推理优化 | **博客 #2：「Spirit v1.5 在 SO-100 跑通」**（千寻必读） |

### Week 5-8：核心战役

| 时段 | A 线 | B 线 | 交付物 |
|---|---|---|---|
| Week 5 | Spirit + XLeRobot LoRA fine-tune（首版） | OpenVLA → SO-100 LoRA 迁移 | **HF release #1：Spirit-XLeRobot-LoRA** |
| Week 6 | Spirit fine-tune 评测 + 视频 | OpenVLA → SO-100 评测 + 视频 | **B 站视频 #2：Spirit 微调前后对比** |
| Week 7 | Spirit 改进尝试（数据增强 / OFT） | OpenVLA-OFT 复现 | **博客 #3：「Spirit fine-tune 经验总结」** |
| Week 8 | RoboChallenge 提交准备 | OFT 加速博客 | **博客 #4：「OpenVLA-OFT 25× 加速复现」** |

### Week 9-12：横向对比 + 数据贡献

| 时段 | A+B 线 | C 线 | 交付物 |
|---|---|---|---|
| Week 9-10 | Spirit / OpenVLA / SmolVLA / π0 在 SO-100 同任务对比 | 数据采集 200 条 | **博客 #5（爆款）：「4 个 VLA 模型 $660 机器人对比」** |
| Week 11 | 横向对比视频后期 | LeRobot 数据集 PR | **HF 数据集 release** + **B 站视频 #3** |
| Week 12 | XLeRobot 真机长时段 demo（家务任务） | LeRobot 教程 PR | **YouTube 主 demo（3-5 min）** |

### Week 13-16：作品集 + 投递

| 时段 | 任务 | 交付物 |
|---|---|---|
| Week 13 | 整合所有产出，做个人作品集（GitHub Pages） | personal site online |
| Week 14 | 简历 + cover letter + 自我介绍视频 | resume v1 |
| Week 15 | 投千寻 + 5-10 家同级公司 | 投递完成 |
| Week 16 | 面试准备 + 持续内容输出（保持热度） | 第一轮面试 |

---

## 5. Week 2 末（5/12）必交付

1. ✅ GitLab `ro_planning` push（已完成）
2. ⏳ LIBERO 4 ckpt 评测完 + 数字落库
3. ⏳ B 站视频 #1：LIBERO 4 suite OpenVLA demo
4. ⏳ **博客 #1（知乎）：「VLA 论文批判分析：RT-1 到 OpenVLA 我打了什么分」**
5. ⏳ Spirit v1.5 仓库分析文档（docs/spirit_analysis.md）
6. ⏳ **GitHub public mirror 开**（B 选项决策）

---

## 6. 影响力放大矩阵

| 渠道 | 内容 | 频率 |
|---|---|---|
| GitHub `liuzhi7/ro_planning` | 代码 + 文档 + checkpoint | 持续 commit |
| HuggingFace `liuzhi7` | LoRA + 数据集 | 每月 1-2 个 |
| **知乎** | 长文，技术深度（具身智能话题） | 2 周 1 篇 |
| **小红书 / B 站** | 真机视频 | 每周 1-2 个 |
| Twitter | 英文短帖 + 视频 | 每周 1-2 条，@svlevine @physical_int |
| 微信视频号 | 内推朋友看到 | 每个里程碑 |

### 关键人物 @ 列表

**千寻**：
- @SpiritAITeam（如有）
- 韩峰涛 / 高阳 / 解浚源（如有 Twitter）
- GitHub `Spirit-AI-Team/spirit-v1.5` issue + PR

**学术圈**：
- @svlevine（OpenVLA / Octo 第一作者圈）
- @physical_int（π0 团队）
- @chelseafinn @moo_jin_kim
- LeRobot Discord / OpenVLA Discord

### 公开身份

- 真名"刘志" + Independent / Personal Project
- **不写小米 affiliation**（合规风险 + 转行不靠小米品牌）
- GitHub bio: `Liu Zhi · Robotics & Embodied AI · Independent`

---

## 7. 风险与应对

| 风险 | 应对 |
|---|---|
| Spirit v1.5 仓库依赖与开发机镜像不兼容 | 先尝试现有镜像；不行另起子镜像 |
| RoboChallenge 真机评测需寄硬件 / 申请名额 | 先看规则；不行用其它公开 leaderboard |
| XLeRobot 数据采集 200 条耗时 | 周末批量采，分散压力 |
| 千寻在你产出之前已招满 | 内容公开化，转其他公司直接复用 |
| OpenVLA 镜像跑 Spirit 不通 | Spirit 应该是 HF transformers 路线，兼容 |
| 面试官不重视开源影响力 | 至少有：复现报告 + 真机视频 + 数据集 三件套 |

---

## 8. 成功画像（4 个月后）

```
GitHub: liuzhi7
  ├── ro_planning ⭐ 核心仓库
  ├── spirit-v1.5-xlerobot ⭐ Spirit 真机适配 fork
  └── 30+ stars

HuggingFace: liuzhi7
  ├── liuzhi7/spirit-v1.5-xlerobot-lora
  ├── liuzhi7/openvla-7b-so100-lora
  └── liuzhi7/xlerobot-household-50ep (dataset)

知乎: 5 篇深度博客
  - 「VLA 论文批判分析」(Week 2)
  - 「Spirit v1.5 在 SO-100 跑通」(Week 4)
  - 「Spirit fine-tune 经验」(Week 7)
  - 「OpenVLA-OFT 25× 加速复现」(Week 8)
  - 「4 个 VLA 模型在 $660 机器人对比」(Week 10)
B 站: 5+ 真机演示视频
Twitter: 30+ 英文短帖

简历底部:
  - GitHub stars: 50+
  - HuggingFace 模型/数据集 downloads: 1000+
  - "Reproduced and improved Spirit v1.5 on a $660 SO-100 robot"
  - "Cross-embodiment LoRA transfer for OpenVLA"
```

---

## 9. 核心原则（避坑清单）

1. ❌ **不要从零训 LIBERO 50K steps**：3.8 天/suite × 4 = 影响力为 0
2. ❌ **不要追求 CoRL 投稿在 6 个月内出**：deadline + 接收概率风险
3. ❌ **不要憋 1 个月才发第一条内容**：影响力靠频率
4. ❌ **不要先做 MoE 这种纯学术创新**：千寻面试官未必感冒
5. ✅ **每周必须有可见产出**：commit / blog / video / tweet 任一种
6. ✅ **每条公开内容 @ 千寻员工**（如有 Twitter 账号）
7. ✅ **HuggingFace 个人页要有几个 release**
8. ✅ **简历底部"GitHub: ..." "HF: ..."**

---

## 10. 计划版本控制

- v1.0：MoE-OpenVLA 主路线
- v1.1：改 LoRA-MoE，保留 Llama 主干
- v1.2：调整优先级 — 冲榜+视频 > 后训练 > MoE
- v1.3：用 OpenVLA 官方 LIBERO ckpt 直接评测
- **v1.4 (FINAL)：影响力工厂 — 双线并行 A+B 路径，4 个月千寻 offer**
