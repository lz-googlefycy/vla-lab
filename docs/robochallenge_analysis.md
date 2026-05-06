# RoboChallenge Table30 调研

> 创建：2026-05-06
> 官网：https://robochallenge.ai （或 robochallenge.cn）
> HuggingFace org：https://huggingface.co/RoboChallenge
> 主办：Dexmal（原力灵机）+ HuggingFace

---

## 1. 是什么

**RoboChallenge Table30** 是一个**真机 VLA 评测平台**：
- 30 个标准化桌面操作任务（grasp / 工具使用 / 双臂协作等）
- 平台拥有 **20+ 真实机器人集群**（ARX5 / UR5 / Franka / ALOHA）
- **集中式、在线、真机原生**：你提交模型，他们在真机上自动评测
- 评分 = Success Rate + Progress Score

**和 LIBERO 区别**：
- LIBERO = 仿真，自评，论文 self-report
- RoboChallenge = 真机，第三方评测，**有 leaderboard**

---

## 2. 排行榜（截至 2026.01）

Spirit v1.5 排第 1，超 π0.5。
- 区分两类：**Task-Specific**（单任务）和 **Generalist**（多任务）
- Spirit 在 **Table30 V2** 上登顶

---

## 3. 提交流程

### 3.1 注册账号

平台需要 USER_TOKEN（API key），只有注册后才能拿到。

### 3.2 模型格式

- 必须适配支持的硬件（ARX5/UR5/Franka/ALOHA）
- 数据格式：转 LeRobot 格式（HF 标准）
- 数据集：Table30 已开源在 HF（`RoboChallenge/task_table30_*`）

### 3.3 提交模式

**关键限制**：**不通过提交视频**，必须**在线提交模型**让平台跑真机评测。
- 模型按平台 API 协议接入（参考 Spirit 仓库的 `robochallenge/` 目录）
- 平台调度真机集群运行

### 3.4 评测时间

需要"预约测试时间"，可能排队（大热门）。

---

## 4. 我们能不能上 Table30 leaderboard？

### 关键问题
- 注册是否开放给个人？还是 inviting only？
- USER_TOKEN 怎么拿？（CVPR workshop 等专门赛季可能有名额）
- 评测费用？（运行真机有成本）

### 行动项
- [ ] 访问 https://robochallenge.ai 看 sign-up
- [ ] 加入他们的 community（WeChat / Discord / Slack）
- [ ] 看 HF org 是否有报名链接
- [ ] 评估 token 申请通过率

---

## 5. 备选评测平台（如果 RoboChallenge 不可访问）

### 5.1 SimplerEnv（Google + UC Berkeley，开源仿真）
- 跨域评测：训自己的，仿真测
- OpenVLA / Octo 都用这个

### 5.2 LIBERO（已有）
- 仿真，4 suite
- 论文标杆

### 5.3 LeRobot Eval（HuggingFace）
- 适配自己的硬件
- XLeRobot / SO-100 原生支持

### 5.4 我们自定义（XLeRobot 真机视频）
- 不上任何 leaderboard，只录视频
- 但有真机的可信度

---

## 6. 战略建议

### 优先级排序

1. **Week 4-5**：先用 XLeRobot 真机做 Spirit 推理 + 录视频（无需 RoboChallenge）
2. **Week 6+**：如 RoboChallenge 注册开放，提交 base ckpt 看排名（不一定要超 Spirit，参与本身已是亮点）
3. **Week 8+**：如 token 拿到，提交 fine-tuned ckpt，**目标进 leaderboard top 10**

### 不要把 RoboChallenge 当唯一目标

- 它是**锦上添花**而不是必需
- 我们的核心是 Spirit + XLeRobot + 真机视频，**这条路径不依赖 RoboChallenge**
- 如果 token 拿不到，照样能出博客/视频/简历

---

## 7. 链接清单

- 官网（中文）：https://robochallenge.cn/home
- 官网（英文）：https://www.robochallenge.ai
- HuggingFace org：https://huggingface.co/RoboChallenge
- Spirit v1.5 官方 wrapper：https://github.com/Spirit-AI-Team/spirit-v1.5/tree/main/robochallenge
- 数据集（move_objects_into_box）：https://huggingface.co/datasets/RoboChallenge/task_table30_move_objects_into_box

---

## 8. TODO

- [ ] 注册账号（待用户决策）
- [ ] 加入 Discord / WeChat 群
- [ ] 评估提交可行性
- [ ] 写一篇博客分析 RoboChallenge（可作为 Week 7+ 内容）
