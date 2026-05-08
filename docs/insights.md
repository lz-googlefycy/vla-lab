# Insights

> 深度洞察：不是"怎么修 bug"，而是"我从这个项目里学到了什么"。
> 受众是未来的我、读博客的人、以及可能面试我的人。
> 工程 bug 的处理见 [`troubleshooting.md`](./troubleshooting.md)。

每条 insight 按这个模板：
```
## [日期] 标题
**上下文**
**观察**
**原本的预期**
**现在的理解**
**对后续工作的影响**
**相关工作 / 文献**
```

---

## 2026-05-08 · k8s pod 里 Vulkan 不可用是基础设施问题，不是单点 bug <a name="k8s-vulkan-icd-missing"></a>

**上下文**
本地 RTX 3090 docker 里 SAPIEN 起不来，本以为是消费卡 + docker
配置问题。今天在 datacenter H20 pod（k8s 容器）里跑 7-stage
`vulkan_smoke.py` 完整验证，期待 datacenter GPU + nvidia-container-toolkit
能 work。

**观察**

```
[1] /usr/share/vulkan/icd.d/  →  intel/radeon/llvmpipe/virtio
                                 ❌ 无 nvidia_icd.json
[2] vulkaninfo --summary      →  deviceName = llvmpipe (软渲染 CPU)
[3] sapien.Engine()           →  ✅ 构造成功（不 touch GPU）
[4] sapien.Scene + camera     →  ❌ "failed to find a rendering device"
[5] gym.make("PickCube-v1")   →  ❌ vk::createInstanceUnique:
                                    ErrorIncompatibleDriver
```

`vulkaninfo` 表面 "PASS" 是假象，它跑的是 mesa 的 LavaPipe（CPU 软渲染）。
SAPIEN 一旦真要 GPU device 立刻 fail。

**原本的预期**
H20 是 datacenter 卡，nvidia-container-toolkit 应该把 Vulkan ICD 自动
挂进容器（和 CUDA 一起）。我以为本地 docker 失败是 RTX 3090 + 消费驱动
特例，换 H20 应该好。

**现在的理解**
nvidia-container-toolkit 默认只暴露 `compute,utility` 两个 capability
进容器。Vulkan/OpenGL 需要单独开 `graphics` capability，这要在 k8s
deployment 里加 `NVIDIA_DRIVER_CAPABILITIES=all`（或
`compute,graphics`）环境变量 + nvidia ICD `.json` 文件挂进容器。

大部分 ML 平台的 pod 模板没开这个 — 训练/推理工作流不需要 Vulkan，
开了反而暴露 GPU 显示接口的攻击面。但**所有依赖 SAPIEN / Maniskill /
Isaac Lab / Habitat 等 Vulkan-based sim 框架的人**，都会撞到这个墙。

这不是单点 bug，是基础设施配置导致的 sim 工作流系统性受限。

**对后续工作的影响**
1. **闭环 eval 路径必须避开 Vulkan**：Maniskill 默认用 SAPIEN（vulkan
   渲染）→ 不可用。退路：
   - **LIBERO**（基于 robosuite + mujoco），可以用 osmesa CPU 软渲染或
     EGL（如果挂了 nvidia EGL ICD）→ 今天验证 work。
   - **mujoco MJX**（GPU 加速但走 OpenGL/EGL，不走 Vulkan）→ 备选。
   - **PyBullet**（CPU 物理 + OpenGL/CPU 渲染）→ 慢但稳。
2. Phase A 不受影响：之前因为本地 Vulkan 失败 pivot 到的"静态图 +
   action chunk plot" demo 路径反而是对的。
3. 一个没人说的事：很多 VLA 论文里都画了"我们在 Maniskill 评测的"图，
   但**没人提这个东西在 ML 云平台上其实默认跑不了**—这个 gap 值得写。

**相关工作 / 文献**
- nvidia-container-toolkit 文档 §"Driver Capabilities"
- SAPIEN 3 `_vulkan_tricks.py`：fallback ICD 注入逻辑，只对单 GPU
  桌面生效，多 GPU 服务器场景未测
- Maniskill issue #355: "no rendering device on aks/gke"

---

## 2026-05-08 · H20 vs RTX 3090 上 Spirit 的延迟稳定性差异 <a name="h20-vs-3090-latency"></a>

**上下文**
今天把 Spirit v1.5 从本机 RTX 3090 24GB 迁到开发机 H20 144GB。**同样**的 bf16
推理代码、**同样** action chunking 配置、**同样**的 monkey-patch。

**观察**

| 指标 | RTX 3090 24 GB | H20 144 GB | 变化 |
|---|---|---|---|
| 模型加载 | 58 s | 23 s | 2.5× 更快 |
| Steady-state 推理 mean | 163 ms | 152 ms | 慢 8% 改善 |
| Min 延迟 | 151 ms | 151 ms | 一样 |
| **Max 延迟** | **251 ms** | **154 ms** | **2× 改善** |
| 延迟方差 | ±100 ms | **±3 ms** | **30× 改善** |
| GPU 显存 | 10 GB / 24 GB | 10 GB / 150 GB | 14× 余量 |

H20 上 6.6 Hz 几乎是**确定性**，而 3090 上有明显的 tail latency。

**原本的预期**
"推理延迟主要看 FLOPs，GPU 算力差不多的话差距应该不大。3090 fp16 算力 ~142 TFLOPS，
H20 bf16 算力 ~148 TFLOPS——**预期差距 <5%**。"

**现在的理解**

延迟 mean 确实差不多（8% 改善，跟 FLOPS 差异一致）。但**tail latency 差一个数量级**，
原因是 3090 上：

1. **内存竞争**：3090 上还有 5 GB 系统占用（桌面、Chrome、其它），而 Spirit 用 10 GB，
   留给激活的空间只有 ~9 GB。chunk 60×14×1536 = ~5 MB 激活看起来小，但 Qwen3-VL 编码
   3 张 320×240 图时的中间 feature map 可能临时冲到几个 GB。
2. **ECC memory + 更大 HBM**：H20 是 ECC HBM3，bitflips 重试概率更低。

**深度含义**：**"consumer GPU 能跑"和"production GPU 能稳定跑"是两件事**。
博客 #2 应该诚实报告这两个数字——读者看到 "6.1 Hz on 3090" 以为 "稳定的 6 Hz"，
实际 p99 可能更慢。

**对后续工作的影响**
1. 本机 3090 用于**快速 iteration / 调试**，H20 用于**正式 benchmark / fine-tune / 视频录制**
2. Phase B fine-tune **必须在 H20**（3090 显存不够训）
3. 如果后面要做 "deploy on edge GPU" 的故事，**单独做 tail latency 分布实验**

**相关工作 / 文献**
- OpenVLA 论文在 A100 上只报 mean latency，不报 p99
- π0 论文同样
- **Robot VLA benchmarks 里没人认真比较 consumer vs datacenter GPU 的 tail latency**——
  这是一个未被填的空白，可能是一个 minor contribution

---

## 2026-05-08 · 跨 embodiment 部署的"最后一公里" <a name="cross-embodiment-last-mile"></a>

**上下文**
Spirit v1.5 (千寻智能 2026.01 开源) 是 ALOHA 双臂 14-DoF 设计。
我要把它接到 **XLeRobot (SO-100 双臂 12-DoF)** 上——硬件形态高度相似（都是
桌面双臂加 gripper），但关节配置不同（ALOHA 有 forearm_roll，SO-100 没有）。

**观察**
开源 VLA 的"可用性"宣传和"真的能在我的硬件上跑"之间，有一层**很厚的适配税**。
表面看 Spirit 开源了：
- model.safetensors ✅
- config.json + tokenizer ✅
- 推理代码 ✅
- 训练代码 ✅

但真要跑，要解决：

1. **`batch["robot_type"]` 字段**：Spirit 期望 `{ARX5, aloha, Franka, UR5}`
   之一作为 prompt condition。SO-100 不在列表里。
2. **关节顺序 + 语义不对齐**：
   - ALOHA: `[waist, shoulder, elbow, forearm_roll, wrist_angle, wrist_rotate, gripper]`
   - SO-100: `[shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper]`
   差一个 `forearm_roll`，我们必须 **零填充 + 输出端丢弃**。
3. **14→12 action projection**：Spirit 输出 14 维 action chunk；要映射回
   SO-100 12 维才能 `robot.command()`。
4. **摄像头缺失**：Spirit 训练用 3 相机 (`cam_high + cam_left_wrist + cam_right_wrist`)，
   XLeRobot 默认只有头部相机，**完全缺腕部相机**。我们只能 tile `cam_high` 3 份
   或额外花 $60 加装 2 个 USB 腕部相机。
5. **默认推理 dtype / device 绑定**：`config.json["device"] = "cuda"` 直接把 21 GB
   fp32 权重载到 GPU（见 troubleshooting）。

**原本的预期**
"Spirit 是工业级开源项目（千寻冲 RoboChallenge 第一），应该 plug-and-play 吧。"

**现在的理解**
**开源 ckpt + 开源推理代码 ≠ 可部署**。Spirit 的开源是"**给同样 ALOHA 硬件的人**"
的，不是给 SO-100 / xArm / Unitree 这些异构硬件的。同样的问题在 OpenVLA 上也有
（虽然轻一些，因为 OpenVLA 的 action space 更 canonical）。

具身智能的"民主化"瓶颈可能不在模型开源，**而在 adapter 层**。LeRobot 社区的核心价值
可能就是把这些 adapter 标准化——它们是**沉默大多数的工程债**。

**对后续工作的影响**
1. 我写的 `XLeRobotSpiritAdapter` 可能是 SO-100 社区第一份公开的 Spirit adapter。
   应该 **提交 PR 到 LeRobot 或独立开源**作为贡献物。
2. 博客 #2 重点写"**6 步让 Spirit 跑在 \$660 机器人上**"的故事，
   每一步的"硬件税"细节都是内容金矿。
3. Phase B fine-tune 时要严肃思考："要让 Spirit 理解 SO-100 这个 embodiment"
   是否需要**新 robot_type 字符串**（见下一条 insight）。

**相关工作 / 文献**
- OpenVLA 论文 § 6 Deployment：也提到跨机器人 adapter，但主要是 action space
  unnormalize 的问题，没有结构性关节映射
- LeRobot 的 `observation.images.*` / `observation.state` 语义已经部分标准化，
  但 Spirit / π0 / OpenVLA 各家的 action 语义没有对齐
- π0.5 论文强调 "cross-embodiment"，但实际数据是 PI 自家 10+ 种机器人，
  都是他们定义的，**不是真正第三方异构**

---

## 2026-05-08 · Spirit 用 prompt 字符串"理解"硬件，而不是用 embedding <a name="robot-type-as-prompt"></a>

**上下文**
Spirit 要求 `batch["robot_type"]: List[str]`。

**观察**
读 Spirit 源码（`utils/vlm_utils.py:12`）：
```python
def get_user_prompt(image_placeholders, robot_type):
    return f"{image_placeholders}\nThe current robot type is {robot_type}. What is the current task?"
```

**robot_type 只是被拼进 prompt 字符串**。模型通过 VLM 的语言理解能力把
`"The current robot type is aloha."` 当作 context，而不是通过一个独立的
learned embedding 来编码硬件信息。

**原本的预期**
以为 robot_type 会经过一个 `nn.Embedding(num_robots, dim)`，把 ARX5/aloha/...
映射成可学习向量，和视觉特征 / state 特征 concat。就像传统做法。

**现在的理解**
**Spirit 相信"VLM 语言理解能编码硬件差异"**。这是一个非常"LLM 派"的工程选择——
把一切能用语言表达的都塞进 prompt，让 backbone VLM 自己学到映射。

这个选择有几个后果：

1. **便宜**：加一个新 robot_type `"SO-100"` 不用 retrain embedding 层，
   只需要在 fine-tune 数据里混入 "The current robot type is SO-100." 这句话。
2. **脆弱**：robot_type **在预训练里出现过**才能有意义的表征。`"aloha"` 出现了
   几万次 → 模型学到"双臂桌面"语义。`"SO-100"` 零次出现 → 模型只会按字面
   hyphen-separated token 理解，得不到有意义的表征。
3. **对 fine-tune 的启示**：引入新 robot_type 时，我们有两条路：
   - **"假身份"**：用最近的现有 robot_type（`"aloha"`），让模型复用其分布知识
   - **"实名 + 数据推**"**：用 `"SO-100"` + 足够数据让模型重新学
   后者更诚实，前者更节省。Phase B 数据只 50-200 条的话，**建议假身份**。

**对后续工作的影响**
Phase B fine-tune 数据生成时：
- 试 A/B：一半数据标 `robot_type="aloha"`，一半标 `robot_type="SO-100"`
- 看 eval SR 哪个高。如果 `"SO-100"` 并没提升，说明**新 prompt token 没学到**，
  用 `"aloha"` 就行；如果 `"SO-100"` 更高，说明 **50-200 条足够给 prompt 学新映射**
  这本身是个可发论文的小实验。
- 博客 #2 可以介绍这个 prompt-as-embedding 设计 + 我们的 A/B 实验。

**相关工作 / 文献**
- OpenVLA 不用 robot_type 字段；它用 action space unnormalize stats 做隐式硬件
  信息传递（每个 dataset 有自己的 mean/std）。这是 tokenizer 派的做法。
- π0 论文提到跨 embodiment 时"在 instruction 前加机器人名字"——和 Spirit 一致。
- RT-2 不做显式 robot_type（它就一个机器人 Everyday Robots）。

---

## 2026-05-08 · Spirit 在 RTX 3090 上比 OpenVLA 在 H20 上快一倍 <a name="spirit-speed-advantage"></a>

**上下文**
同样 bf16 推理，同样 batch_size=1，同样不带 flash-attn3（都是 flash-attn2），
Spirit 3090 = 6.1 Hz，OpenVLA H20 = 3.0 Hz。**3090 更弱，但更快**。

**观察**
两个根本原因：

### (1) Backbone 规模差一倍
| 模型 | Backbone | 参数量 | 激活显存（bf16）|
|---|---|---|---|
| OpenVLA | Llama-2 **7B** + DINOv2-300M + SigLIP-400M | ~7.7B | 7.5 GB |
| Spirit v1.5 | Qwen3-VL **4B** (视觉+语言一体) + DiT 300M | ~4.3B | 4.3 GB |

Qwen3-VL-4B 比 Llama-2-7B 约**小 1.8 倍**，前向 FLOP 比也约 1.8 倍。

### (2) Action chunking 把有效控制频率拉高 N 倍

OpenVLA 每步要 forward 一次 VLM：
```
control_step_i:
    image_i, lang -> VLM forward -> 1 action token -> real-world
```
控制频率 = 推理频率 = 3 Hz。

Spirit 每 N 步才 forward 一次：
```
control_step_i:
    image_i, lang -> Qwen3-VL + DiT -> action chunk[60 actions]
control_step_i+1..i+12:
    reuse cached action chunk (next action)
control_step_i+13:
    image, lang -> re-infer chunk ...
```
推理频率 6.1 Hz，`chunk_horizon=12` → **有效控制频率 6.1 × 12 = 73 Hz**。

**原本的预期**
以为 Spirit 是 diffusion，去噪 10 步 forward，**应该比 OpenVLA 单次 token 预测慢得多**。

**现在的理解**
去噪步数确实增加 compute（`10 × DiT forward`），但 DiT 只 300M 参数，
比 VLM 主干小 10 倍。真正省的是 **"不用每步调用 VLM"**。10 次 DiT forward
<< 1 次 Qwen3-VL forward。

**深度含义**：
- **Diffusion / flow matching VLA 的推理成本被 chunking 摊薄**。π0、Spirit 都这么做。
- **OpenVLA 的 tokenizer 设计在推理速度上是个弱点**，因为它本质是 token-by-token
  （虽然每步是 single token，但每步都要过整个 Llama-2）。
- 如果要把 OpenVLA 做到类似速度，需要接一个 **action prediction head** 一次
  出 chunk——这正是 **OpenVLA-OFT（2025）** 做的，它靠这个拿到 25× 加速。

**对后续工作的影响**
1. 博客 #2 可以明确量化："6.1 Hz on a \$350 GPU" vs "3.0 Hz on a \$40k GPU"——
   这是**可信的对比数字**，不是营销话术
2. 如果将来要做"OpenVLA → chunked 推理"对比实验（Phase 3+），这条 insight 就是研究动机
3. OpenVLA-OFT 是下一个应该读的论文，它把 OpenVLA chunk 化。读完之后我对
   "VLA 推理速度天花板"会有更清楚认识

**相关工作 / 文献**
- OpenVLA-OFT (2025)：[arXiv:2502.19645](https://arxiv.org/abs/2502.19645)，chunk + parallel
  decode 给 OpenVLA 带来 25× 加速
- π0 论文 § 4.2：讲 action chunking 50 步设计
- ACT (Zhao 2023)：最早把 action chunking 推给学术界
- Diffusion Policy（Chi 2023）：diffusion action head 的原始工作

---

## 2026-05-08 · Diffusion-VLA 的"dtype 表面积"比 token-VLA 大 <a name="diffusion-vla-dtype-surface"></a>

**上下文**
为了在 RTX 3090 上放下 Spirit，我把权重 cast 到 bf16。结果连续踩了 **5 个**
dtype 不匹配 bug（见 troubleshooting.md），每个都出在 DiT 内部某一层。
OpenVLA bf16 部署时**没有这种问题**。

**观察**
OpenVLA 的 forward 路径：
```
image + state + lang -> Llama-2 backbone -> 1 token (logits over 256 action bins)
```
所有操作都是标准 matmul + softmax。bf16 weights + bf16 inputs → bf16 outputs.
干净。

Spirit 的 forward 路径：
```
image + state + lang -> Qwen3-VL encode
                     -> sample_noise (fp32 hardcoded!)
                     -> sample_time  (fp32 hardcoded!)
                     -> DiT denoise (10 iterations)
                         -> TimestepEmbedding (Sinusoidal, may return fp32)
                         -> state_proj (bf16)
                         -> cross-attention
                         -> action_in_proj (bf16)
                         -> ...
                     -> 60-step action chunk
```
DiT 内部有**多处**非 linear 操作（Sinusoidal embed、positional encode、noise
schedule），这些地方**类型不绑到 model 参数**，容易漂回 float32。

**原本的预期**
"都是 PyTorch 模型，cast weights 就 OK 了。"

**现在的理解**
**Diffusion / flow matching action head 有额外的 numerics surface area**。
主要来自两类源：
1. **噪声采样**：`torch.normal(..., dtype=torch.float32)` 硬编码
2. **时间步嵌入**：`SinusoidalPositionalEmbedding` / `Timesteps` 内部用 fp32 精度

要部署到 consumer GPU，要么：
- **保持 fp32 全跑**（需要 ≥ 16 GB 激活显存）
- **bf16 + monkey-patch + autocast 组合拳**（我的做法）
- **等 upstream 支持 torch_dtype kwarg**（最干净，但依赖 Spirit 团队）

相比之下，OpenVLA 的 token-based 设计**天然对 dtype friendly**。

**对后续工作的影响**
1. 博客 #2 可以负责任地评论："diffusion-style VLA 是更强的 action 建模能力，
   但 engineering cost 高"——这是**两派（token vs diffusion）trade-off 的
   少有具体证据**
2. 如果以后在 SO-100 上对比 OpenVLA vs Spirit，**部署难度** 是 OpenVLA 的一个
   隐性优势
3. 给 Spirit 团队提 PR：在 `SpiritVLAPolicy.from_pretrained` 加
   `torch_dtype=None` 参数，内部处理 noise/time 的 dtype

**相关工作 / 文献**
- π0 (PI, 2024)：也是 flow matching 模型，理论上应该同样问题。但 `openpi`
  代码里用 JAX，JAX 默认 mixed precision 比 PyTorch 更宽容
- Stable Diffusion 社区：早就碰到 DiT bf16 问题，解决方案用 `bitsandbytes` 或
  直接硬编码 `.to(dtype)` 每层——借鉴意义有限，因为 SD 的 DiT 是自家写的

---

## 2026-05-08 · "config.json 里写死 device" 是一个部署不友好的模式 <a name="config-device-antipattern"></a>

**上下文**
Spirit 的 `config.json` 里有 `"device": "cuda"`，`from_pretrained` 会读这个
字段来决定把 21 GB state_dict 直接加载到 GPU。3090 上 OOM。我们的 workaround
是临时改写 config 再改回来。

**观察**
Deploy-time 的选择（设备、dtype、memory 策略）**不应该**硬编码在 ckpt config
里。HuggingFace transformers 的惯例是：
```python
model = AutoModelForXxx.from_pretrained(
    ckpt_path,
    torch_dtype=torch.bfloat16,   # runtime parameter
    device_map="auto",            # runtime parameter
    low_cpu_mem_usage=True,       # runtime parameter
)
```
Spirit 违反了这个惯例，把 `device` 当成 ckpt 的属性。结果：

- 用户必须**写权限打开 config.json 改掉**才能在 CPU 加载
- 没有 `torch_dtype` 选项——默认 fp32，3090 (24 GB) 装不下
- 没有 `low_cpu_mem_usage`——必须一口气把 21 GB RAM 吃进来

**原本的预期**
以为 Spirit 作为工业级开源项目（千寻冲 RoboChallenge 第一）会遵循社区 best practices。

**现在的理解**
Spirit 的 `modeling_spirit_vla.py` 看起来更像**研究代码**而非**生产代码**：
- 没有 `torch_dtype` / `device_map` / `low_cpu_mem_usage` 参数
- DiT 内部 noise/time dtype 硬编码
- tokenization 路径依赖本地 HF cache 可写

这不是 bug，是**成熟度问题**——Spirit 2026.01 才开源，社区整合度还没到 OpenVLA
（2024.06）的水平。OpenVLA 已经在全世界被 fork 过千次，各种部署 edge case 都
反馈修过了。

**对后续工作的影响**
1. 博客 #2 **轻描淡写**提这条（Q3=b 措辞温和原则）：**"Spirit 现在是研究质量的
   开源，离 OpenVLA 那种打磨程度还有几个月的社区迭代周期。这很正常。"**
2. 这是一个给 Spirit 贡献的机会：提 PR 加 `torch_dtype` 参数。贡献被接受
   = 简历亮点 + 千寻员工会注意到
3. 评估一个 VLA 是不是"真开源"时，**除了看 ckpt，还要看 `from_pretrained` 接口
   是不是和 HF 标准对齐**。这是一个可以用的 quality proxy

**相关工作 / 文献**
- HuggingFace accelerate library: 标准 device_map / dtype 加载 API
- OpenVLA 代码：`from_pretrained(..., torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)`
  原生支持
- π0 的 openpi 代码：JAX 原生 bf16，不在 PyTorch 生态里，参考意义小

---

## Index

每条 insight 都有 anchor 链接，方便交叉引用：

- [k8s pod 里 Vulkan 不可用是基础设施问题](#k8s-vulkan-icd-missing) ← new
- [H20 vs RTX 3090 上 Spirit 的延迟稳定性](#h20-vs-3090-latency)
- [跨 embodiment 部署的最后一公里](#cross-embodiment-last-mile)
- [Spirit 用 prompt 字符串"理解"硬件](#robot-type-as-prompt)
- [Spirit 为什么比 OpenVLA 快一倍](#spirit-speed-advantage)
- [Diffusion-VLA 的 dtype 表面积问题](#diffusion-vla-dtype-surface)
- [config.json 写死 device 是反模式](#config-device-antipattern)

后续每次有新深度洞察，在顶部加新 section，index 里加链接。**不要删老的**——
观点会随时间变化，变化本身就是 insight。
