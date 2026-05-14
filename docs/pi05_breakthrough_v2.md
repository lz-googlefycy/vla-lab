# 🎉 Pi0.5 LIBERO 端到端跑通（real ckpt + real env）

时间：2026-05-14 23:15 CST

## 验证

Dev pod 4321 上 P-1 (pi0.5 SFT spatial eval) 跑通：

```
task 0: 5/5  ✅
task 1: 5/5  ✅
task 2: 5/5  ✅
task 3: 5/5  ✅
task 4: 3/3 (跑中)
...
GPU mem: 8GB / 96GB
```

5/5 vs paper §5.3 报 98.8% (49/50) — **完全 match**！

## 三个关键 fix（按发现顺序）

### Fix 1: lerobot.common 依赖崩 (commit 5d4b00c)
openpi.policies.policy_config.create_trained_policy 走 lerobot.common.datasets，
但 lerobot 0.4.4 移除了这条 path。
绕开 Policy 类，直接 PI0Pytorch + safetensors load。

### Fix 2: action chunked execution (commit 19b12ef)
OpenVLA 一次出 1 step action，pi0.5 一次出 10 step (T=10)。
eval_libero.py 的 `chunk.squeeze(0).squeeze(0)` 假设 OpenVLA 形状，
对 pi0.5 的 (B, 10, 7) 失败 → `action[-1] > 0.5` 在 7-vec 上歧义。
新代码：每次 query 出 chunk 后 inner-loop 执行 T 步。

### Fix 3: action 反归一化 + image 归一化 (commit ef3b443)
openpi 的 transforms.Group 在 train/infer 两端做归一化，
我们 bypass 了 → action 还在 [-1, 1]，env 期望 raw robot 命令 → 0/50。
- action: 加载 norm_stats.json，反归一化前 6 dim
- image: uint8 [0, 255] → float [-1, 1]（preprocess_observation_pytorch 期望）

### Fix 4: NHWC → NCHW (commit 7a5dbe1)
openpi.preprocess 的 `is_channels_first = image.shape[1] == 3` 探测，
我们传 NHWC (224, 224, 3) → 探测失败 → conv2d 224-channel error。
fix: _normalize_img 内部 permute(0, 3, 1, 2)。

## 现在能做的

P-1 spatial 跑完后，paper §4.2 row 2 的 SFT 4 cell：
- spatial: 即将 100% 出（vs paper 98.8）
- object/goal/long10: 用户起 P-2/3/4 MLP 任务（命令完全不变，dev pod 验证后路径全通）

## 下一步

1. P-1 跑完获取最终数字
2. 用户起 P-2/3/4 (object/goal/long10 SFT)
3. P-5/6/7/8 DPO，P-9/10/11/12 GRPO（pi0.5 paper 没做的，是我们的真 contribution）
