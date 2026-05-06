# 策略 v1.3 — 用 OpenVLA 官方 LIBERO ckpt 直接评测冲榜

> 创建：2026-05-06
> 触发：发现训练 50K steps 单卡 H20 要 3.8 天，4 suite 累计 15+ 天，时间不够

---

## 1. 关键发现

### LIBERO "排行榜" 真相
- **LIBERO 没有官方在线排行榜**（不像 Waymo / nuScenes）
- 所有 SR 数字 = 论文里 self-report
- "上榜" = 在论文/HuggingFace/GitHub 公开 SR 数字 + 视频，让社区认可

### OpenVLA 已发布 4 个官方 LIBERO ckpt
HuggingFace 直接可下载：
```
openvla/openvla-7b-finetuned-libero-spatial
openvla/openvla-7b-finetuned-libero-object
openvla/openvla-7b-finetuned-libero-goal
openvla/openvla-7b-finetuned-libero-10
```
单个 ~14 GB（fully merged，非 LoRA）。

### OpenVLA 论文报告数字
| Suite | Paper SR | Source |
|---|---|---|
| Spatial | 84.7 ± 0.9 | OpenVLA paper Table 6 |
| Object | 88.4 ± 0.8 | 同上 |
| Goal   | 79.2 ± 1.0 | 同上 |
| Long (10) | 53.7 ± 1.3 | 同上 |
| **Avg** | **76.5 ± 0.6** | |

---

## 2. 调整后的 18 周路线

### Phase 0：基础设施（已完成 90%）

### Phase 1：用官方 ckpt 评测复现 + 录视频（Week 2，4-5 天）⭐ P0
- [x] 停止训练 → 释放 GPU
- [ ] **下载 4 个官方 LIBERO ckpt**（本机 hf-mirror，并行下，约 1-2h）
- [ ] **rsync 推到开发机**（24-56 GB）
- [ ] **跑 4 suite 评测**（每 suite 500 rollout，约 2h；4 suite 共 8h）
- [ ] **同步录制演示视频**（评测过程中保存渲染帧）
- [ ] 数字落 `experiment_log.md`，与论文 SR 对比

**验收**：
- 4 suite SR 数字（应 ≈ 论文水平）
- 4 suite × 5 任务 demo 视频
- 漂亮的主 demo（1-2 min 剪辑）

### Phase 2：自训 vs 官方 ckpt 对比（Week 3-4）⭐ P1

**目标**：决定要不要"自训超越官方"
- [ ] 短训（10K steps，~18h）单 suite，看 SR 是否接近 50K 论文水平
  - 若 10K ≈ 论文水平 → 4 suite × 18h = 3 天可完成
  - 若 10K 差 5+ 个百分点 → 接受官方 ckpt 数字，时间投到视频/论文
- [ ] 大概率：**直接基于官方 ckpt 做创新**（4-bit / FAST / OFT 思路），不重训

### Phase 3：演示视频精修（Week 4-5）⭐ P0
- [ ] 拍 4 suite × 多任务 demo（每 suite 30s × 4 = 2min）
- [ ] 主 demo 剪辑（1-2 min）
- [ ] 上传 HuggingFace / 飞书 / Twitter / 内部展示

### Phase 4：选择创新方向（Week 5+）⭐ P1
按学术价值排序选一：
1. **Quantization deployment 论文**：4-bit / 8-bit OpenVLA 在 LIBERO 上 SR 几乎不掉，加速量化
2. **OpenVLA-OFT 复现 + 改进**：继承官方 ckpt + OFT 框架，做 25× 推理加速
3. **FAST tokenizer 整合**：替换 OpenVLA 256-bin 为 FAST，看 LIBERO 表现
4. **LoRA-MoE**（最初方向，已降优先级）

---

## 3. 现在马上做的事

### 顺序
1. **下载 4 个 ckpt**（本机后台并行，~1h）
2. **修 eval 脚本**（OpenVLA 的 `run_libero_eval.py` 参数）
3. **rsync 推到开发机**（每个 ckpt 14 GB → 总约 56 GB）
4. **跑首个 suite 评测**（Spatial，~2h）
5. 边跑边写视频录制脚本

---

## 4. ckpt 下载脚本

```bash
# 本机
mkdir -p ~/openvla_assets/finetuned_libero/

unset ALL_PROXY all_proxy

for SUITE in spatial object goal 10; do
  HF_HOME=~/openvla_assets/hf_cache \
  HF_ENDPOINT=https://hf-mirror.com \
  HF_HUB_ENABLE_HF_TRANSFER=1 \
  ~/mambaforge/envs/openvla-tools/bin/hf download \
    openvla/openvla-7b-finetuned-libero-$SUITE \
    --local-dir ~/openvla_assets/finetuned_libero/openvla-7b-finetuned-libero-$SUITE &
done
wait
```

约 56 GB，hf-mirror 30-50 MB/s → 约 30-60 min。
