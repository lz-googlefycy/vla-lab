
---

## 📌 Day 7 sprint 教学子节点（2026-05-17）

> **KV-Cache 三连撞墙、π0.5 inference 真相、DoRA Object 大胜利**
>
> - 教学全文（含 8 道面试官 Q&A、3 张图、仓库代码索引）：[GitHub: docs/teaching/day7_kvcache_pivot_for_liu.md](https://github.com/lz-googlefycy/vla-lab/blob/main/docs/teaching/day7_kvcache_pivot_for_liu.md)
> - 工作日志：[GitHub: docs/work_log/day7_kvcache_pivot_dora_object_win.md](https://github.com/lz-googlefycy/vla-lab/blob/main/docs/work_log/day7_kvcache_pivot_dora_object_win.md)
> - 简历最新版改动 commit：[ro_planning@e3d08ca](https://git.n.xiaomi.com/liuzhi7/ro_planning/-/commit/e3d08ca)
>
> **核心数字**：
> - DoRA-r32 + DPO Object：**76% (vs LoRA 62%, +14pp)** ⭐
> - DoRA Goal: 78% (+2pp), Spatial: 78% (持平)
> - π0.5 sample_actions latency: prefix 21% / denoise 79% → KV-Cache paper 假设在 π0.5 不直接成立
> - 多视角预训练 synthetic 收敛: log(B)=5.55 → 0.005 (恢复 99.9% 信号)
>
> 简历对应：sub-bullet 3 已从 "latency ↓2.4×" 改为 research-grade 表述；sub-bullet 1 加入 Object DPO 62→76 (+14pp)、显存↓78% vs full FT。
