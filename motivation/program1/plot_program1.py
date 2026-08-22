#!/usr/bin/env python3
"""Render the one-page motivation figure as dependency-free vector SVG."""

import html
import json
import sys
from pathlib import Path


if len(sys.argv) != 2:
    raise SystemExit("usage: plot_program1.py <run-root>")
root = Path(sys.argv[1]).resolve()
d = json.loads((root / "summary.json").read_text(encoding="utf-8"))

width, height = 1320, 350
panels = []


def esc(value):
    return html.escape(str(value))


def panel(index, title, labels, values, colors, unit, maximum=None):
    x0 = 20 + index * 325
    y0 = 25
    w = 300
    h = 300
    max_value = maximum or max(max(values), 1)
    pieces = [
        f'<g transform="translate({x0},{y0})">',
        f'<text class="title" x="0" y="18">{esc(title)}</text>',
    ]
    for row, (label, value, color) in enumerate(zip(labels, values, colors)):
        y = 65 + row * 85
        bar_width = 210 * max(0.0, value) / max_value
        pieces += [
            f'<text class="label" x="0" y="{y}">{esc(label)}</text>',
            f'<rect class="track" x="0" y="{y + 12}" width="210" height="28"/>',
            f'<rect x="0" y="{y + 12}" width="{bar_width:.2f}" height="28" fill="{color}"/>',
            f'<text class="value" x="220" y="{y + 33}">{value:.2f}{esc(unit)}</text>',
        ]
    pieces.append('</g>')
    panels.append("\n".join(pieces))


def stack_panel(index, title, labels, values, colors):
    x0 = 20 + index * 325
    total = max(sum(values), 1)
    pieces = [
        f'<g transform="translate({x0},25)">',
        f'<text class="title" x="0" y="18">{esc(title)}</text>',
    ]
    left = 0.0
    for label, value, color in zip(labels, values, colors):
        segment = 285 * value / total
        pieces.append(
            f'<rect x="{left:.2f}" y="52" width="{segment:.2f}" '
            f'height="34" fill="{color}"/>')
        left += segment
    for row, (label, value, color) in enumerate(zip(labels, values, colors)):
        y = 116 + row * 31
        pieces += [
            f'<rect x="0" y="{y-13}" width="13" height="13" fill="{color}"/>',
            f'<text class="label" x="20" y="{y}">{esc(label)}</text>',
            f'<text class="value" x="230" y="{y}">{100*value/total:.1f}%</text>',
        ]
    pieces.append('</g>')
    panels.append("\n".join(pieces))


stack = d.get("coupled_stack", {})
stack_panel(
    0, "(a) Coupled critical path",
    ["Stage1 local search", "Global continuation", "Remote reverse",
     "Final prune", "Write + metadata"],
    [stack.get("stage1_local_search_ns", 0),
     stack.get("global_continuation_ns", 0) +
       stack.get("final_candidate_snapshot_ns", 0),
     stack.get("remote_reverse_ns", 0),
     stack.get("final_prune_ns", 0),
     stack.get("allocate_write_ns", 0) +
       stack.get("local_reverse_ns", 0) +
       stack.get("metadata_and_other_ns", 0)],
    ["#4daf4a", "#377eb8", "#e41a1c", "#984ea3", "#999999"])

qps_max = 1.12 * max(d["baseline_insert_qps"], d["solution_insert_qps"], 1)
panel(1, f"(b) Update throughput ({d['insert_speedup']:.2f}x)",
      ["Coupled", "Two-stage"],
      [d["baseline_insert_qps"], d["solution_insert_qps"]],
      ["#7570b3", "#1b9e77"], " ops/s", qps_max)

panel(2, "(c) Temporary query quality",
      ["Stage1-only", "Stage2 final"],
      [100 * d["stage1_only_self_hit_rate"],
       100 * d["finalized_self_hit_rate"]],
      ["#e6ab02", "#1b9e77"], "%", 100)

locality_max = 1.12 * max(
    100 * d["cross_edge_ratio_stage1_home"],
    100 * d["cross_edge_ratio_final_home"], 1)
panel(3, f"(d) Locality ({100*d['cross_edge_reduction_ratio']:.1f}% fewer)",
      ["Before move", "After move"],
      [100 * d["cross_edge_ratio_stage1_home"],
       100 * d["cross_edge_ratio_final_home"]],
      ["#e7298a", "#66a61e"], "%", locality_max)

svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<style>
  text {{ font-family: Arial, "Noto Sans", sans-serif; fill: #222; }}
  .title {{ font-size: 17px; font-weight: 700; }}
  .label {{ font-size: 14px; }}
  .value {{ font-size: 13px; font-weight: 700; }}
  .track {{ fill: #f0f0f0; }}
</style>
<rect width="100%" height="100%" fill="white"/>
{''.join(panels)}
</svg>
'''
output = root / "program1_motivation.svg"
output.write_text(svg, encoding="utf-8")
print(output)
