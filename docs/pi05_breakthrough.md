# 🎉 Pi0.5 突破：本机 + dev pod conversion 成功

时间：2026-05-14 13:30 CST

## 全 pipeline 跑通

1. ✅ JAX ckpt 12GB 下载（GCS）
2. ✅ JAX → PyTorch 转换（dev pod 上跑通，本机 OOM 后改 dev pod）
3. ✅ PyTorch ckpt 6.8GB 加载（dev pod 96GB H20）
4. ✅ Forward + sample_actions 输出真实 LIBERO action

## 关键 fix

- pip 装 `jax==0.5.3 / jaxlib==0.5.3 / flax==0.10.2 / orbax==0.11.13` 三件套精确 pin
- 本机 conda env `openpi-convert` 装好 wheels → scp 到 dev pod
- dev pod 装上 wheel 后 conversion 直接成功（221GB RAM 充足，本机 62GB OOM）
- numpy 1.26.4 pin（augmax / chex 会偷偷升 numpy 2.x 破坏兼容）

## ckpt 位置

`/e2e-data/users/liuzhi7/vla_workspace/models/pi05_libero_pytorch/`
- model.safetensors: 6.8 GB
- config.json: 149 B

## 下一步

1. 跑 1-task LIBERO eval (real env)
2. 跑 GRPO 3-step smoke
3. 输出 12 个 MLP 单卡命令清单（pi0.5 SFT eval × 4 + DPO × 4 + GRPO × 4）
