#!/usr/bin/env python3
"""Aggregate all v1.5 paper §4.2 eval results into a single Markdown table.

Looks for summary.json files under both:
  - assets/paper_v1.5_eval/  (synced from cloudml + H20)
  - output/eval_*_full/      (in-progress local evals)

Prints the v1.5 main results table + per-suite per-task breakdown.

Usage: python scripts/aggregate_paper_results.py
"""

import json
from pathlib import Path

ROOT = Path(__file__).parent.parent
ASSET_DIR = ROOT / "assets/paper_v1.5_eval"

# Reference numbers
PAPER = {"libero_spatial": 84.7, "libero_object": 88.4,
         "libero_goal": 79.2, "libero_10": 53.7}
V14_REPRO = {"libero_spatial": 78.0, "libero_object": 60.0,
             "libero_goal": 77.0, "libero_10": 53.0}

SUITES = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]
SUITE_LABELS = {"libero_spatial": "Spatial", "libero_object": "Object",
                "libero_goal": "Goal", "libero_10": "Long10"}

# Find summary.json files
sft_data = {}
dpo_data = {}

for suite in SUITES:
    suite_short = suite.replace("libero_", "")
    # SFT
    for path in [
        ASSET_DIR / f"openvla_sft_{suite}_5x10_seed42.json",
        ROOT / f"output/eval_{suite_short}_full/summary.json",
    ]:
        if path.exists():
            try:
                d = json.load(open(path))
                rate = d["per_suite"][suite]["rate"]
                sft_data[suite] = rate * 100
                break
            except (KeyError, json.JSONDecodeError):
                pass

    # DPO
    for path in [
        ASSET_DIR / f"openvla_dpo_{suite}_5x10_seed42.json",
    ]:
        if path.exists():
            try:
                d = json.load(open(path))
                rate = d["per_suite"][suite]["rate"]
                dpo_data[suite] = rate * 100
                break
            except (KeyError, json.JSONDecodeError):
                pass

# Print main table
print("# v1.5 Paper §4.2 — Current Results\n")
print(f"Generated: $(date)\n" if False else "")  # placeholder
print("| Suite   | SFT (paper) | SFT (v1.4) | SFT (v1.5) | + DPO    | Δ DPO |")
print("|---------|------------:|-----------:|-----------:|---------:|------:|")

avg_sft_v15 = []
avg_dpo = []
for suite in SUITES:
    sft = sft_data.get(suite, None)
    dpo = dpo_data.get(suite, None)
    sft_str = f"{sft:.0f}%" if sft is not None else "—"
    dpo_str = f"{dpo:.0f}%" if dpo is not None else "TBD"
    if sft is not None and dpo is not None:
        delta = dpo - sft
        delta_str = f"{'+' if delta > 0 else ''}{delta:.0f}"
    else:
        delta_str = "TBD"
    print(f"| {SUITE_LABELS[suite]:7} | "
          f"{PAPER[suite]:6.1f}     | "
          f"{V14_REPRO[suite]:5.1f}     | "
          f"**{sft_str:>5}**    | "
          f"**{dpo_str:>5}** | "
          f"**{delta_str:>5}** |")
    if sft is not None:
        avg_sft_v15.append(sft)
    if dpo is not None:
        avg_dpo.append(dpo)

# Avg row
if avg_sft_v15:
    avg_sft_str = f"{sum(avg_sft_v15)/len(avg_sft_v15):.1f}%"
    avg_dpo_str = f"{sum(avg_dpo)/len(avg_dpo):.1f}%" if avg_dpo else "TBD"
    print(f"| **Avg** |             |            | **{avg_sft_str}** | **{avg_dpo_str}** |       |")

print()
print(f"v1.5 SFT cells with data: {len(avg_sft_v15)}/4")
print(f"v1.5 + DPO cells with data: {len(avg_dpo)}/4")

# Per-task breakdown for cells with both SFT and DPO
print("\n## Per-task breakdowns where DPO has data\n")
for suite in SUITES:
    if suite not in dpo_data or suite not in sft_data:
        continue

    sft_path = ASSET_DIR / f"openvla_sft_{suite}_5x10_seed42.json"
    dpo_path = ASSET_DIR / f"openvla_dpo_{suite}_5x10_seed42.json"
    if not (sft_path.exists() and dpo_path.exists()):
        continue

    sft_per_task = json.load(open(sft_path))["per_suite"][suite]["per_task"]
    dpo_per_task = json.load(open(dpo_path))["per_suite"][suite]["per_task"]

    print(f"### {SUITE_LABELS[suite]}\n")
    print("| Task | DPO | SFT | Δ |")
    print("|------|-----|-----|---|")
    for tid in sorted(sft_per_task.keys(), key=int):
        sft_succ = sft_per_task[tid]["success"]
        dpo_succ = dpo_per_task.get(tid, {}).get("success", 0)
        n = sft_per_task[tid]["trials"]
        delta = dpo_succ - sft_succ
        sign = "+" if delta > 0 else ""
        print(f"| {tid} | {dpo_succ}/{n} | {sft_succ}/{n} | {sign}{delta} |")
    print()
