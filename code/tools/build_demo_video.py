"""
build_demo_video.py — 把 LIBERO eval 自动产出的 MP4 整理成 demo 视频

OpenVLA 的 run_libero_eval.py 内置 `save_rollout_video`，每个 episode 输出一个 MP4：
  rollouts/<DATE>/<DATETIME>--episode=N--success=True/False--task=<task_name>.mp4

这个脚本：
1. 扫一个或多个 EVAL-* 目录下的 rollouts/
2. 按 (suite, task, success=True) 筛选
3. 每个 task 选 top-K 个成功 rollout（默认 1）
4. 拼接成 1 个 MP4，每段加字幕（task 描述 + suite 名 + 序号 + 总 SR%）
5. 输出 final demo MP4

依赖：imageio[ffmpeg] (镜像里已装) + 可选 PIL 加字幕

用法:
  python build_demo_video.py \
    --eval_dirs /workspace/output/EVAL-2026...-libero_spatial \
                /workspace/output/EVAL-2026...-libero_object ... \
    --out /workspace/output/demo_4suite.mp4 \
    --per_task 1
"""
import argparse
import json
import re
from pathlib import Path

import imageio
import numpy as np


SUITE_RE = re.compile(r"libero_(spatial|object|goal|10|long)", re.I)
EP_RE = re.compile(r"episode=(\d+)--success=(True|False)--task=([^.]+)")


def parse_mp4_filename(fname: str):
    """fname: '2026-05-07_10-15-22--episode=3--success=True--task=pick_up_..._50.mp4'"""
    m = EP_RE.search(fname)
    if m is None:
        return None
    ep = int(m.group(1))
    succ = m.group(2) == "True"
    task = m.group(3)[:80]  # OpenVLA truncates at 50
    return ep, succ, task


def parse_suite_from_path(p: Path):
    """Look for libero_<suite> in the parent path."""
    for part in p.parts:
        m = SUITE_RE.search(part)
        if m:
            s = m.group(1).lower()
            if s == "10":
                s = "long"
            return s
    return "unknown"


def collect_videos(eval_dirs, per_task=1, only_success=True):
    """
    Returns list of dicts:
      [{suite, task, episode, success, path}, ...]
    Sorted by (suite_order, task, episode).
    """
    suite_order = {"spatial": 0, "object": 1, "goal": 2, "long": 3, "unknown": 9}
    rollouts = []
    for d in eval_dirs:
        d = Path(d)
        if not d.exists():
            print(f"[warn] eval dir does not exist: {d}")
            continue
        for mp4 in d.rglob("*.mp4"):
            parsed = parse_mp4_filename(mp4.name)
            if parsed is None:
                continue
            ep, succ, task = parsed
            if only_success and not succ:
                continue
            rollouts.append({
                "suite": parse_suite_from_path(mp4),
                "task": task,
                "episode": ep,
                "success": succ,
                "path": str(mp4),
            })

    # Group by (suite, task), keep top-K per group sorted by episode
    grouped = {}
    for r in rollouts:
        key = (r["suite"], r["task"])
        grouped.setdefault(key, []).append(r)

    selected = []
    for key, items in grouped.items():
        items.sort(key=lambda x: x["episode"])
        selected.extend(items[:per_task])

    selected.sort(key=lambda x: (suite_order.get(x["suite"], 9), x["task"], x["episode"]))
    return selected


def annotate_frame(frame, text_lines, position="top"):
    """Add text overlay to a frame. Pure numpy (PIL optional for nicer font)."""
    try:
        from PIL import Image, ImageDraw, ImageFont
        img = Image.fromarray(frame)
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
        except Exception:
            font = ImageFont.load_default()
        h, w = frame.shape[:2]
        y = 8 if position == "top" else h - 24 * len(text_lines) - 8
        for line in text_lines:
            draw.rectangle([(4, y - 2), (w - 4, y + 22)], fill=(0, 0, 0, 200))
            draw.text((8, y), line, fill=(255, 255, 255), font=font)
            y += 24
        return np.array(img)
    except ImportError:
        return frame  # No PIL, return unchanged


def build_demo(selected, out_path, fps=30, max_frames_per_clip=300,
               include_title_card=True):
    """Build demo. All frames resized to a common (H, W) determined by 1st clip."""
    # Determine target frame size from the first valid clip
    target_h, target_w = None, None
    for r in selected:
        try:
            reader = imageio.get_reader(r["path"])
            first = reader.get_next_data()
            target_h, target_w = first.shape[:2]
            reader.close()
            break
        except Exception:
            continue
    if target_h is None:
        print("[error] no clip readable to set target size")
        return
    print(f"[info] target frame size: {target_w}x{target_h}")
    print(f"[info] writing demo to {out_path} (fps={fps})")

    writer = imageio.get_writer(out_path, fps=fps, codec="libx264",
                                 quality=8, macro_block_size=1)

    def resize_frame(frame, h, w):
        """Resize via PIL if available; otherwise skip mismatched frames."""
        if frame.shape[:2] == (h, w):
            return frame
        try:
            from PIL import Image
            img = Image.fromarray(frame).resize((w, h), Image.BILINEAR)
            return np.array(img)
        except Exception:
            return None

    if include_title_card:
        title_frame = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        title_frame = annotate_frame(title_frame, [
            "OpenVLA-7B on LIBERO",
            f"Clips: {len(selected)}",
            "github.com/liuzhi7/ro_planning",
        ], position="top")
        for _ in range(int(2.0 * fps)):  # 2-second title
            writer.append_data(title_frame)

    for i, r in enumerate(selected):
        path = r["path"]
        try:
            reader = imageio.get_reader(path)
        except Exception as e:
            print(f"  [skip] {path}: {e}")
            continue

        frames = []
        for f in reader:
            frames.append(f)
            if len(frames) >= max_frames_per_clip:
                break
        reader.close()

        if not frames:
            continue

        text_lines = [
            f"[{i+1}/{len(selected)}] {r['suite']}: {r['task'][:60]}",
            f"episode {r['episode']} {'✓' if r['success'] else '✗'}",
        ]

        for f in frames:
            f_resized = resize_frame(f, target_h, target_w)
            if f_resized is None:
                continue
            annotated = annotate_frame(f_resized, text_lines, position="top")
            writer.append_data(annotated)

        blank = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        for _ in range(int(0.3 * fps)):
            writer.append_data(blank)

        print(f"  [{i+1}/{len(selected)}] {r['suite']}/{r['task'][:40]} ({len(frames)} frames)")

    writer.close()
    print(f"[done] {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_dirs", nargs="+", required=True,
                    help="One or more EVAL-* directories")
    ap.add_argument("--out", required=True, help="Output MP4 path")
    ap.add_argument("--per_task", type=int, default=1)
    ap.add_argument("--include_failures", action="store_true")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--max_frames_per_clip", type=int, default=300)
    args = ap.parse_args()

    selected = collect_videos(
        args.eval_dirs,
        per_task=args.per_task,
        only_success=not args.include_failures,
    )

    if not selected:
        print("[error] no clips found")
        return 1

    print(f"[info] selected {len(selected)} clips")
    for r in selected:
        print(f"  - {r['suite']}/{r['task'][:50]} ep={r['episode']} ✓={r['success']}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    build_demo(selected, args.out, fps=args.fps,
               max_frames_per_clip=args.max_frames_per_clip)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
