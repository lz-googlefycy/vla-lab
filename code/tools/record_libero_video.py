"""
LIBERO 演示视频录制脚本（占位框架，5/7 实装）

目标：基于 OpenVLA `experiments/robot/libero/run_libero_eval.py` 改造，
在每个 rollout 中保存渲染帧序列为 MP4。

输出：
  - <out_dir>/<task_name>_seed<X>_succ<Y>.mp4 — 每个 rollout 一个视频
  - <out_dir>/index.json — 视频 metadata（任务/SR/seed等）

用法:
  python record_libero_video.py \
      --pretrained_checkpoint <ckpt> \
      --task_suite_name libero_spatial \
      --num_trials_per_task 5 \
      --out_dir /workspace/output/videos/spatial

TODO（5/7 实装）:
  1. 加载 OpenVLA 的 run_libero_eval.py 中的 rollout loop
  2. 在每步 step 后捕获 RGB（robosuite render('rgb_array')）
  3. 累积帧 + imageio[ffmpeg].mimsave
  4. 标注 instruction、SR、step 计数
  5. 后期：用 moviepy 拼成 4 suite 主 demo
"""
import argparse
import json
import os
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pretrained_checkpoint", required=True)
    p.add_argument("--task_suite_name", default="libero_spatial")
    p.add_argument("--num_trials_per_task", type=int, default=5)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--video_fps", type=int, default=20)
    p.add_argument("--save_only_success", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"  LIBERO Video Recorder (PLACEHOLDER)")
    print("=" * 60)
    print(f"  Checkpoint   : {args.pretrained_checkpoint}")
    print(f"  Suite        : {args.task_suite_name}")
    print(f"  Trials/task  : {args.num_trials_per_task}")
    print(f"  Out dir      : {out_dir}")
    print("=" * 60)
    print()
    print("[TODO] Implement actual rollout loop. For now this just"
          " writes a stub index.json so the pipeline can be wired up.")

    index_path = out_dir / "index.json"
    with open(index_path, "w") as f:
        json.dump({
            "checkpoint": args.pretrained_checkpoint,
            "suite": args.task_suite_name,
            "trials_per_task": args.num_trials_per_task,
            "videos": [],  # 5/7 实装后填充
            "status": "stub",
        }, f, indent=2)
    print(f"Stub index written to {index_path}")


if __name__ == "__main__":
    main()
