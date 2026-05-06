# 项目主计划 v1.2

> 创建：2026-05-06
> 上一版：v1.1（飞书 wiki，LoRA-MoE 路线）
> 本版关键变化：**回归核心目标，调整优先级 — 冲榜+视频 > 后训练 > MoE**

---

## 1. 优先级（用户明确）

```
P0  LIBERO 排行榜成绩
P0  演示视频（漂亮、可投放）
P1  后训练 / VLA 工具链熟练
P2  论文（CoRL 2026 可选目标）
P3  LoRA-MoE 创新（不再首要）
```

---

## 2. 18 周路线图

### Phase 0：基础设施（Week 1-2，已 80%）

- ✅ OpenVLA 镜像 v1.0 (32.3 GB) + 推送 micr/volc/vnet 三仓库
- ✅ 模型 + 数据上传 JuiceFS（24 GB total）
- ✅ 开发机 H20-3e 环境验证
- ✅ Smoke test pass（4.4 GB GPU / 3.03 Hz）
- ✅ Git 仓库初始化（<private-gitlab>/ro_planning）
- ⏳ 仓库初始 push

### Phase 1：LIBERO 冲榜（Week 2-4）⭐ P0

#### Week 2：基线复现
- [ ] LIBERO-Spatial LoRA r=32 微调（24h on 1×H20）
  - 论文目标 SR：84.7% ± 0.9
  - 配置参考 OpenVLA `vla-scripts/finetune.py`
- [ ] LIBERO-Spatial 评测 → 数字落库
- [ ] 同步启动 LIBERO-Object（如有 GPU）

#### Week 3：4 suite 全跑
- [ ] LIBERO-Object（论文 88.4 ± 0.8）
- [ ] LIBERO-Goal（论文 79.2 ± 1.0）
- [ ] LIBERO-Long（论文 53.7 ± 1.3）
- [ ] 4 suite 平均 SR ≥ 76.5%（论文水平）
- [ ] 评测脚本规范化（`code/scripts/eval_libero.sh`）

#### Week 4：超越尝试
- [ ] **超参数搜索**：LR / batch / steps / LoRA r
- [ ] **数据增强**：LIBERO 图像扰动 / 多视角
- [ ] **更长训练**：超过论文的 50K-100K steps
- [ ] **集成**：多 ckpt 平均（参考 OpenVLA-OFT 思路）
- [ ] **排行榜提交**

### Phase 2：演示视频（Week 4-5）⭐ P0

#### Week 4-5：视频录制 + 后期
- [ ] 写录制脚本（每个任务录 5-10 个成功 episode）
- [ ] 4 suite × 5 任务 × 3 角度 = 60 段视频
- [ ] 后期：标注 instruction、添加成功率字幕、剪辑成 1-2 分钟主 demo
- [ ] 输出：YouTube / 抖音 / 小米内部展示版

### Phase 3：后训练 / VLA 工具链（Week 5-8）⭐ P1

- [ ] **4-bit 推理优化**：bitsandbytes nf4 + double-quant 实测速度
- [ ] **LoRA 部署**：PEFT 加载 + 合并主干
- [ ] **vLLM / TGI 部署**：尝试 OpenVLA 7B 高吞吐推理
- [ ] **量化模型上传 HuggingFace**：方便复用
- [ ] **OpenVLA-OFT 学习**：25-50× 快推理 + 高频控制（2025 OpenVLA 后续工作）
- [ ] **FAST tokenizer 学习**（2025）：动作 chunk 压缩

### Phase 4：可选论文方向（Week 8+）⭐ P2

任选其一：

#### 方向 A：LoRA-MoE-OpenVLA（保留原方案）
- 8 expert × LoRA r=16
- SkillRouter（Gemini 标 skill）
- LIBERO 全 suite +3% SR

#### 方向 B：LIBERO 跨域泛化
- 训 LIBERO-Spatial 测 Object/Goal/Long zero-shot
- 找 OpenVLA 在跨任务的弱点 → 改进

#### 方向 C：高频控制 / OpenVLA-OFT 重现 + 改进
- 复现 OpenVLA-OFT 25× 加速 + bimanual
- 在自己机器人或仿真上验证

#### 方向 D：FAST + LoRA 组合
- FAST tokenizer + 我们的 LoRA 微调
- 看是否在 LIBERO 上有提升

---

## 3. 资源 & 风险

### 算力（H20-3e × 1，144 GB）
| 任务 | 时长 |
|---|---|
| LIBERO 单 suite LoRA 微调 | ~24h |
| LIBERO 单 suite 评测（500 rollout） | ~2h |
| 4 suite 全部 | 串行 ~5 天 / 并行（如有多卡）~1 天 |
| 视频录制 | ~1 天 |

### 风险
| 风险 | 应对 |
|---|---|
| 复现达不到论文 SR | 用 OpenVLA 已发布的 LIBERO ckpt 直接评测做 baseline |
| 排行榜接受流程不清楚 | Week 2 先确认 LIBERO 提交格式 |
| 视频录制耗时 | 早做录制脚本框架，异步录制不阻塞训练 |
| 开发机断连 | 用 nohup + setsid + 监控脚本（参考 TrajFlow 经验） |

---

## 4. 立即下一步（本周）

按时间顺序：

| 顺序 | 事项 | 时长 |
|---|---|---|
| 1 | 仓库 push 到 GitLab | 30 min |
| 2 | LIBERO-Spatial LoRA 训练启动 | 30 min 启动 + 24h 训练 |
| 3 | 等训练时：写视频录制脚本框架 | 1-2 h |
| 4 | 训完评测 → 数字记到 experiment_log.md | 2 h |
| 5 | LIBERO-Object 启动 | 重复 |

---

## 5. 计划版本控制

- v1.0 (5/6) 初版，MoE-OpenVLA 主路线
- v1.1 (5/6) 改 LoRA-MoE，保留 Llama 主干
- **v1.2 (5/6) 调整优先级：冲榜+视频 > 后训练 > MoE**
