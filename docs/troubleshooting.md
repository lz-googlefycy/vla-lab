# Troubleshooting

> 纯事实记录：遇到什么问题、根因是什么、怎么修的。
> 查询为主，不带思考。思考见 [`insights.md`](./insights.md)。

---

## 2026-05-08 — Spirit v1.5 × RTX 3090 consumer-GPU inference

复现 Spirit v1.5 在 **RTX 3090 24 GB** 上跑起来时，踩到 6 个 bug。最终 smoke test 通过：
`163 ms / 6.1 Hz steady-state, 10 GB GPU`。下面按出现顺序记录。

镜像：`spirit-v1.0-cu128-py310` (torch 2.8.0, transformers 4.57.1, diffusers 0.35.2)
代码：`code/spirit_adapter/xlerobot_adapter.py`

---

### [docker-mount-ro] 只读挂载导致 HF cache 不能写

**现象**：
```
OSError: [Errno 30] Read-only file system: '/workspace/models/hf_cache'
```

**触发条件**：
`docker run -v <host>:/workspace/models:ro` + Spirit 内部尝试下载 Qwen tokenizer。

**根因**：
Spirit 的 `SpiritVLAConfig.backbone = "Qwen/Qwen3-VL-4B-Instruct"`（HF 名），
`AutoTokenizer.from_pretrained(self.config.backbone, ...)` 仍然尝试去 HF 抓取，
即使我们本地已经有完整 Qwen3-VL-4B。

**修复**：
用 `rw` 挂载 + 在 `config.json` 里把 `backbone` 字段改为**绝对路径**
(`/workspace/models/Qwen3-VL-4B-Instruct`)。然后加 `TRANSFORMERS_OFFLINE=1` +
`HF_HUB_OFFLINE=1` 两个 env var 强制 offline 模式。

**影响**：
打包 "patched" 目录结构：`Spirit-v1.5-patched/` 里用 hardlink 到原始
`model.safetensors`，用新的 `config.json`（只修 backbone 一字段）。

---

### [docker-symlink] 容器内 symlink 指向宿主机绝对路径失效

**现象**：
```
FileNotFoundError: model.safetensors not found in /workspace/models/Spirit-v1.5-patched
```
（即使宿主机上这个文件存在）

**触发条件**：
用 `ln -sf /home/ubuntu/openvla_assets/spirit_ckpts/Spirit-v1.5/model.safetensors
Spirit-v1.5-patched/model.safetensors` 后 docker mount。

**根因**：
symlink 里存的是**宿主机绝对路径**，容器内 `/home/ubuntu/...` 不存在。

**修复**：
改用 **hardlink**。同一个 filesystem（`/home/ubuntu/openvla_assets/` 整个在同一
NVMe 分区）可以 hardlink，不占双份空间（inode 共享）。

```bash
ln /home/ubuntu/openvla_assets/spirit_ckpts/Spirit-v1.5/model.safetensors \
   /home/ubuntu/openvla_assets/spirit_ckpts/Spirit-v1.5-patched/model.safetensors
```

**影响**：
Docker 挂载方案里 hardlink > symlink。已加入 troubleshooting 心得。

---

### [cuda-oom-fp32] 21 GB fp32 权重直接 .to("cuda") 在 24 GB GPU 上 OOM

**现象**：
```
torch.OutOfMemoryError: CUDA out of memory.
Tried to allocate 96.00 MiB. GPU 0 has a total capacity of 23.55 GiB
of which 90.19 MiB is free.
```

**触发条件**：
RTX 3090 (24 GB) + Spirit `from_pretrained(..., config.device="cuda")` 默认加载到 GPU。
Spirit v1.5 单文件 `model.safetensors` = 21.6 GB（**fp32 全量权重**），直接 load
到 GPU ≈ 22 GB，加上 PyTorch 内部开销 → 超过 24 GB。

**根因**：
`Spirit/model/modeling_spirit_vla.py:856` 读 `config.device` 决定加载目标：
```python
if config.device.startswith("cuda") and torch.cuda.is_available():
    load_device = config.device
state_dict = safe_load_file(weight_path, device=load_device)
```
没有 `torch_dtype` 参数，也没有 `low_cpu_mem_usage` 逻辑。

**修复**：
在 `SpiritLeRobotPolicy.__init__` 里**临时改写** `config.json` 的
`device` 字段为 `"cpu"`，加载后再 cast dtype + 手动 `.to("cuda")`。用完恢复原
config 字符串。

```python
with open(config_path, "r") as f: orig = f.read()
cfg = json.loads(orig); cfg["device"] = "cpu"
with open(config_path, "w") as f: json.dump(cfg, f, indent=2)
try:
    self.inner = SpiritVLAPolicy.from_pretrained(ckpt_path, train=False)
finally:
    with open(config_path, "w") as f: f.write(orig)
self.inner = self.inner.to(torch.bfloat16).to("cuda").eval()
```

**影响**：
需要 docker `rw` 挂载（因为要临时改 config.json）。H20 144 GB 上不需要这个 workaround。

---

### [spirit-robot-type] `batch["robot_type"]` 缺失

**现象**：
```
KeyError: 'robot_type'
  File "modeling_spirit_vla.py", line 709, in preprocess_rb_batch
    robot_type = batch["robot_type"][i]
```

**触发条件**：
Spirit `select_action(batch)` 预处理阶段读 `batch["robot_type"]`（List[str]），
用来拼接 prompt `"The current robot type is {robot_type}."`。

**根因**：
Spirit 训练数据里 robot_type 是必填字段（取自 `{ARX5, aloha, Franka, UR5}`
这 4 个 RoboChallenge 硬件枚举）。上游 `robochallenge/` wrapper 自动填，但
我们自写的 adapter 漏了。

**修复**：
`XLeRobotSpiritAdapter.__init__(robot_type="aloha", ...)` + `wrap_observation`
输出里加 `"robot_type": [self.robot_type]`。XLeRobot（双臂 SO-100）最接近 ALOHA 语义。

**影响**：
这是 Phase A zero-shot 的"**假身份**"——Spirit 根本没见过 SO-100 训练数据，
我们用 `"aloha"` 蒙混。Phase B fine-tune 时**可能需要加入新 robot_type `"SO-100"`**
作为 prompt condition。

---

### [dtype-state-bf16] state 投影层 dtype 不匹配

**现象**：
```
RuntimeError: mat1 and mat2 must have the same dtype, but got Float and BFloat16
  File "modeling_spirit_vla.py", line 589, in _embed_suffix
    state_emb = self.state_proj(state)
```

**触发条件**：
模型 `.to(torch.bfloat16)` 后，但 adapter 的 state tensor 仍 float32。

**根因**：
`torch.from_numpy(np_array).float()` 默认 float32；模型权重已 cast 到 bf16；
`F.linear(input_fp32, weight_bf16)` 直接报错。

**修复**：
`XLeRobotSpiritAdapter.wrap_observation` 里 state tensor cast 到 `torch_dtype`：
```python
state_t = torch.from_numpy(spirit_state).unsqueeze(0).to(device).to(self.torch_dtype)
```

**影响**：
**图片不能 cast bf16**，见下一个 bug。

---

### [dtype-image-bf16] 图片 bf16 化后 numpy cast 失败

**现象**：
```
TypeError: Got unsupported ScalarType BFloat16
  File "modeling_spirit_vla.py", line 697, in preprocess_rb_batch
    img_np = (img_torch.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
```

**触发条件**：
刚把 state 改成 bf16 后，顺手把 image tensor 也改成 bf16。

**根因**：
Spirit 内部 `preprocess_rb_batch` 先把 image tensor `.cpu().numpy()`，
但 **numpy 不支持 bf16 scalar type**（numpy < 2.1 需要 float/uint8/etc，
不支持 bfloat16）。

**修复**：
**图片保持 float32**，只有 state 走 bf16 路径。在 adapter 里显式分开：
```python
state_t = ...to(self.torch_dtype)    # bf16 OK (goes into self.state_proj)
head    = ...to(self.device)         # keep fp32, Spirit will uint8 later
```

**影响**：
VLA 部署的一个非显然 split：scalar inputs (state) 可以 cast 权重对齐，
tensor inputs (images) 往往因为 pre-processing 走 numpy 而必须保留 float32。

---

### [dtype-noise-bf16] Spirit 硬编码 float32 noise，和 bf16 权重冲突

**现象**：
```
RuntimeError: mat1 and mat2 must have the same dtype, but got Float and BFloat16
  File "modeling_spirit_vla.py", line 597, in _embed_suffix
    action_emb = self.action_in_proj(noisy_actions)
```

**触发条件**：
上一条 bug 修了，模型走到 DiT denoise step，发现 `x_t`（noise + action）还是 float32。

**根因**：
Spirit `utils/sampling.py`：
```python
def sample_noise(shape, device):
    return torch.normal(..., dtype=torch.float32, device=device)  # <-- hardcoded
```
`_sample_actions_unified` 内部调 `sample_noise(...)` 生成 noise tensor，
dtype 硬编码 float32，不跟随模型 cast。

**修复 A**（tried first）：**在 `SpiritLeRobotPolicy.select_action` 预生成 noise**：
```python
noise = torch.randn(noise_shape, dtype=torch.bfloat16, device=device)
self.inner.select_action(batch, noise=noise)  # Spirit 支持外部 noise
```

**修复 B**（需要并加的）：**monkey-patch `sample_noise` / `sample_time`**：
因为 Spirit 内部还有其他地方也调这俩函数（timestep embedding 等）。
在 init 时替换：
```python
import utils.sampling as S
S.sample_noise = lambda shape, dev: torch.randn(shape, dtype=torch.bfloat16, device=dev)
S.sample_time  = lambda bs, dev: _orig(bs, dev).to(torch.bfloat16)
# Also patch the already-bound reference in modeling_spirit_vla:
import model.modeling_spirit_vla as M
M.sample_noise = S.sample_noise
M.sample_time  = S.sample_time
```

**影响**：
python 的 `from utils import sample_noise` 会**在 import 时绑定**到 `modeling_spirit_vla` 的
命名空间。patch sampling 模块**不够**，还得 patch `modeling_spirit_vla` 里的引用。
这是一个 python 初学者坑。

---

### [dtype-autocast] DiT 内部 tensor 仍漂回 float32

**现象**：
```
RuntimeError: mat1 and mat2 must have the same dtype, but got Float and BFloat16
  File "modeling_spirit_vla.py", line 624, in _denoise_step
    v_t = self.action_out_proj(suffix_out)
```

**触发条件**：
前 3 个 dtype bug 都修了，但 DiT 内部 forward pass 走到 `action_out_proj` 还是 mismatch。

**根因**：
DiT 内部有些中间层（可能是 `SinusoidalPositionalEmbedding` / `TimestepEmbedding`
或某个 LayerNorm）**隐式返回 float32**。追踪每一层太费时。

**修复**：
用 `torch.autocast(dtype=torch.bfloat16)` 包整个 `select_action`:
```python
with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
    actions = self.inner.select_action(batch, noise=noise)
```
autocast 强制在 cuda 上所有 matmul/linear 用 bf16，不管 input 是啥。

**影响**：
有效，但"锤子解法"——没真正搞清楚是哪一层漂 fp32。性能影响微乎其微。

---

### [numpy-bf16-cast] action tensor bf16 → numpy 失败

**现象**：
```
TypeError: Got unsupported ScalarType BFloat16
  File "xlerobot_adapter.py", line 206, in unwrap_action
    a = spirit_action[0, step_idx].detach().cpu().numpy()
```

**触发条件**：
上面 5 个修复都到位，模型跑完 `select_action` 返回 `torch.Tensor (1, 60, 14, bf16)`，
我直接 `.cpu().numpy()`。

**根因**：
numpy 旧版本（< 2.1）不支持 bfloat16 scalar type。

**修复**：
先 `.float()` cast 到 fp32 再 numpy：
```python
a = spirit_action[0, step_idx].detach().float().cpu().numpy()
```

**影响**：
所有 bf16/fp16 model output 回 numpy 都要这个 pattern。

---

---

## 2026-05-08 — gh CLI 认证失败 (proxy blocks api.github.com)

**现象**：
```
error validating token: Get "https://api.github.com/": EOF
```

**触发条件**：
在本机已经有 SSH key 到 github.com 能正常 `git push`，但 `gh auth login --with-token`
失败。

**根因**：
本机 shell 默认环境变量有 `ALL_PROXY=socks5h://127.0.0.1:20989` +
`HTTPS_PROXY=http://127.0.0.1:41211`。gh 是 HTTP API 客户端，走这些代理，
代理出口**被 block 到 api.github.com**（但不 block `github.com:22` SSH）。

用 `curl https://api.github.com/` 实测：
- **default**：SSL_ERROR_SYSCALL (5s timeout)
- **no proxy** (unset all)：HTTP 200 in 222 ms ✅
- **socks5 20989**：timeout
- **http 41211**：EOF

**修复**：
永久加到 `~/.bashrc`:
```bash
export NO_PROXY="${NO_PROXY:+$NO_PROXY,}github.com,api.github.com,raw.githubusercontent.com,codeload.github.com,objects.githubusercontent.com"
export no_proxy="${no_proxy:+$no_proxy,}github.com,api.github.com,..."

# Helper function to unset all proxies before a command
gh-direct() {
    env -u ALL_PROXY -u all_proxy -u HTTPS_PROXY -u HTTP_PROXY \
        -u https_proxy -u http_proxy -u CLASH_SOCKS_PROXY -u CLASH_HTTP_PROXY \
        "$@"
}

# Alias: 'gh' always goes direct
alias gh='gh-direct /home/ubuntu/mambaforge/bin/gh'
```

下次登录：
```bash
unset ALL_PROXY HTTPS_PROXY HTTP_PROXY all_proxy https_proxy http_proxy
echo 'ghp_xxx' | /home/ubuntu/mambaforge/bin/gh auth login --with-token
gh auth status  # ✓ Logged in to github.com as lz-googlefycy
```

**影响**：
- 本机 gh 现在全功能可用
- 开发机**不能**访问 github.com（连 anaconda.com 也 timeout），所以 gh 只在本机跑
- 所有 GitHub release asset 上传只能从本机做

---

## 性能结果（6 个 bug 全修之后）

| 指标 | RTX 3090 24GB |
|---|---|
| 模型加载 | 57 s |
| GPU 显存 | 10.02 GB |
| Warmup 推理 | 1513 ms |
| **Steady-state 推理** | **163 ms ≈ 6.1 Hz** |
| Action chunk | 60 × 14 |
| 有效控制频率（chunk horizon 12） | 73 Hz |

对比 OpenVLA-7B 在 H20 的 3 Hz，Spirit 在更弱 GPU 上快一倍。深度分析见
[insights.md # spirit-speed-advantage](./insights.md#spirit-speed-advantage)。

---

## 历史记录

此文档由 `ro_planning/docs/troubleshooting.md` 同步到 `vla-lab/docs/troubleshooting.md`。
每次踩新坑，在这里按**日期 + 类别 + 短标题**加条目，不要合并不同主题。

---

## 2026-05-08 — SAPIEN / Maniskill Vulkan unavailable in container

**现象**：
```
[svulkan2] [error] Your GPU driver does not support Vulkan.
RuntimeError: vk::createInstanceUnique: ErrorIncompatibleDriver
```

**触发条件**：
`spirit-sim-v1.0-cu128-py310` 镜像内 `gym.make("PushCube-v1", robot_uids="xlerobot")`
失败。SAPIEN 3.x 渲染器走 Vulkan，docker 里默认没 Vulkan ICD。

**根因**：
SAPIEN 切换到 Vulkan-only 渲染（3.x 起），不再支持 OpenGL/osmesa 回退。要在
容器里启用 Vulkan 需要：
1. 宿主机装 `nvidia-container-toolkit-vulkan` (或 `libnvidia-gl-xxx` 被正确挂载)
2. 容器内装 `libvulkan1 + mesa-vulkan-drivers`（已装）
3. `docker run --gpus all` 实际传入 NVIDIA ICD（有时需要 `--privileged` 或特殊 mount）

我们 docker 环境似乎**已装依赖但 ICD 加载失败**，warning 里提到 `Failed to find glvnd ICD file. ... NVIDIA driver ... incorrect or partial installation`。

**修复**（暂）：
**不用 Maniskill 跑 Phase A**。改用 "Spirit sees scene image + language → predict action chunk"
简化 demo。博客 #2 的核心素材（action chunk plots + 推理速度）不依赖仿真 env。

**未来修复路径**：
1. 换用 Maniskill `sim_backend='cpu'` — 但这会非常慢
2. 换用 XLeRobot 的 `simulation/mujoco/`（纯 mujoco，无 Vulkan 依赖）
3. 在 H20 datacenter pod 验证是否只是消费 GPU + 桌面 docker 的问题
4. 使用 `--runtime=nvidia` 而不是 `--gpus`，或在 host 装 `nvidia-docker2`

**影响**：
Phase A 里"Spirit 在仿真机器人里执行 rollout"这种视觉视频**本周不会有**。
改用"Spirit 看静态图 + 预测动作轨迹"作为 blog #2 主图。真机视频等硬件接入。

**2026-05-08 后续验证**（H20 datacenter pod 上跑 vulkan_smoke.py 7-stage check）：

| Check | Result | Note |
|---|---|---|
| 1. libvulkan.so | ✅ | apt 装的 mesa loader |
| 1b. nvidia ICD .json | ❌ | `/usr/share/vulkan/icd.d/` 只有 intel/radeon/llvmpipe/virtio |
| 2. vulkaninfo --summary | ✅* | 假阳性：deviceName = `llvmpipe` (LLVM 软渲染 CPU) |
| 3. sapien.Engine() | ✅ | 不 touch GPU，只构造对象 |
| 4. SAPIEN scene + camera | ❌ | "failed to find a rendering device" |
| 5. gym.make PickCube cpu | ❌ | `vk::createInstanceUnique: ErrorIncompatibleDriver` |
| 6. gym.make rgb sensors | ❌ | 同上 |
| 7. env.step | ❌ | 同上 |

**结论**：H20 pod 的 nvidia driver 是 570.86.10，但 k8s
nvidia-container-toolkit 默认只暴露 `compute,utility` capability，
没挂 `graphics` capability + nvidia ICD。**不是消费卡 vs datacenter
卡的问题，是 k8s pod 配置**。本机 docker 也是同样原因。

**根因升级**：要让 SAPIEN work，需要在 pod 模板里加：
```
env:
  - name: NVIDIA_DRIVER_CAPABILITIES
    value: "compute,utility,graphics"
```
+ host 上正确挂载 `/usr/share/glvnd/egl_vendor.d/10_nvidia.json` 和
`/usr/share/vulkan/icd.d/nvidia_icd.json`。这通常是 ML 平台层面的配
置变更，单个 pod 改不动。

**永久绕开方案**：闭环 eval 用 LIBERO（mujoco/osmesa），不用 Maniskill/SAPIEN。

---

## 2026-05-08 — 开发机 pod 根盘 20 GB 不够装 Spirit 镜像

**现象**：
```
write /var/lib/docker/tmp/GetImageBlob532615915: no space left on device
```

**触发条件**：
`ssh dev "docker pull <registry>/spirit-v1.0-cu128-py310"` 在开发机 pod 里失败。

**根因**：
开发机是 k8s pod，root overlay 只 20 GB，`/var/lib/docker` 在 root。Spirit 镜像 20 GB
（加上已在 pod 里用的其他镜像），拉取时超容量。

JuiceFS (`/ad-alg/...`) 5 PB 可用但不能放 `/var/lib/docker`（docker 要 local 块设备）。

**修复**：
**不能在 pod 内 docker pull 镜像**。解决方案：
1. **(推荐) 请用户重建 pod**，指定 Spirit 镜像作为 pod image，这样所有 Spirit 依赖
   都在 pod 里，不占 docker 空间
2. 备选：在 pod 里创建新 conda env + pip install（但外网不通，行不通）

**影响**：
当前开发机 pod 用的是 `openvla-v1.0-cu118-py310` 镜像（torch 2.2 + transformers 4.40）。
要跑 Spirit（需要 torch 2.8 + transformers 4.57），**必须等用户用新镜像重建 pod**。
暂时 Spirit 只能在本机 RTX 3090 跑。

