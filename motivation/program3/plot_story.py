#!/usr/bin/env python3
"""Render the Program 3 story figures as dependency-free academic SVG."""

import html
import json
import sys
from pathlib import Path


if len(sys.argv) != 2:
    raise SystemExit("usage: plot_story.py <run-root>")
root = Path(sys.argv[1]).resolve()
d = json.loads((root / "story_summary.json").read_text(encoding="utf-8"))


def esc(value):
    return html.escape(str(value))


def svg(name, body, width, height=365):
    text = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<style>
  text {{ font-family: Arial, "Noto Sans", sans-serif; fill: #25313b; }}
  .title {{ font-size: 17px; font-weight: 700; }}
  .label {{ font-size: 11px; fill: #374151; }}
  .tick {{ font-size: 10px; fill: #4b5563; }}
  .axis-title {{ font-size: 11px; fill: #374151; }}
  .value {{ font-size: 11px; font-weight: 700; }}
  .inside {{ font-size: 11px; font-weight: 700; fill: #25313b; text-anchor: middle; }}
  .axis {{ stroke: #555; stroke-width: 1.1; }}
  .grid {{ stroke: #e5e7eb; stroke-width: 1; }}
</style>
<rect width="100%" height="100%" fill="white"/>
{body}
</svg>'''
    output = root / name
    output.write_text(text, encoding="utf-8")
    print(output)


def bar_panel(x0, width, title, ylabel, labels, values, colors,
              maximum, ticks, value_format):
    plot_left, plot_right = x0 + 67, x0 + width - 14
    plot_top, plot_bottom = 66, 291
    plot_height = plot_bottom - plot_top
    spacing = (plot_right - plot_left) / len(values)
    bar_width = 54
    pieces = [
        f'<text class="title" x="{x0 + width/2:.1f}" y="34" text-anchor="middle">{esc(title)}</text>',
    ]
    for tick in ticks:
        y = plot_bottom - plot_height * tick / maximum
        pieces += [
            f'<line class="grid" x1="{plot_left}" y1="{y:.2f}" x2="{plot_right}" y2="{y:.2f}"/>',
            f'<text class="tick" x="{plot_left - 7}" y="{y + 4:.2f}" text-anchor="end">{esc(value_format(tick, True))}</text>',
        ]
    pieces += [
        f'<line class="axis" x1="{plot_left}" y1="{plot_top}" x2="{plot_left}" y2="{plot_bottom}"/>',
        f'<line class="axis" x1="{plot_left}" y1="{plot_bottom}" x2="{plot_right}" y2="{plot_bottom}"/>',
        f'<text class="axis-title" x="{x0 + 17}" y="{(plot_top + plot_bottom)/2:.1f}" text-anchor="middle" transform="rotate(-90 {x0 + 17} {(plot_top + plot_bottom)/2:.1f})">{esc(ylabel)}</text>',
    ]
    for index, (label, value, color) in enumerate(zip(labels, values, colors)):
        center = plot_left + spacing * (index + 0.5)
        height = plot_height * value / maximum
        y = plot_bottom - height
        pieces += [
            f'<rect x="{center - bar_width/2:.1f}" y="{y:.2f}" width="{bar_width}" height="{height:.2f}" fill="{color}" stroke="white" stroke-width="1"/>',
            f'<text class="value" x="{center:.1f}" y="{max(plot_top + 12, y - 7):.2f}" text-anchor="middle">{esc(value_format(value, False))}</text>',
            f'<text class="label" x="{center:.1f}" y="310" text-anchor="middle">{esc(label)}</text>',
        ]
    return "".join(pieces)


items = d["motivation"]
width_labels = [f"C={item['commit_width']}" for item in items]
motivation = bar_panel(
    10, 530, "(a) Merge input grows with expansion batch",
    "Candidate slots per round", width_labels,
    [item["candidate_slots_per_round"] for item in items],
    ["#b8c4d8"] * len(items), 700, [0, 100, 200, 300, 400, 500, 600, 700],
    lambda value, tick: f"{value:.0f}")
motivation += bar_panel(
    555, 530, "(b) Full Merge time grows with expansion batch",
    "Full Merge time per round (μs)", width_labels,
    [item["full_merge_pipeline_us_per_round"] for item in items],
    ["#efc08d"] * len(items), 25, [0, 5, 10, 15, 20, 25],
    lambda value, tick: f"{value:g}" if tick else f"{value:.2f}")
svg("program3_story_motivation.svg", motivation, 1100)


# Baseline query-time composition. Distance, RDMA wait, and Merge are
# mutually exclusive intervals; Other closes the stack to GPU residence time.
late = d["performance"]["late"]
total_us = late["gpu_query_us"]
segments = [
    ("Other", late["gpu_other_us"], "#d9dde3"),
    ("Distance", late["gpu_distance_score_us"], "#c9b8d8"),
    ("RDMA", late["rdma_wait_us_per_query"], "#9ecae1"),
    ("Merge", late["gpu_merge_us"], "#efc08d"),
]
plot_left, plot_right = 75, 250
plot_top, plot_bottom = 67, 292
bar_x, bar_width = 160, 62
stack = [
    '<text class="title" x="270" y="32" text-anchor="middle">Query-time breakdown before decoupling</text>',
]
for tick in [0, 20, 40, 60, 80, 100]:
    y = plot_bottom - (plot_bottom - plot_top) * tick / 100
    stack += [
        f'<line class="grid" x1="{plot_left}" y1="{y:.2f}" x2="{plot_right}" y2="{y:.2f}"/>',
        f'<text class="tick" x="68" y="{y + 4:.2f}" text-anchor="end">{tick}</text>',
    ]
stack += [
    f'<line class="axis" x1="{plot_left}" y1="{plot_top}" x2="{plot_left}" y2="{plot_bottom}"/>',
    f'<line class="axis" x1="{plot_left}" y1="{plot_bottom}" x2="{plot_right}" y2="{plot_bottom}"/>',
    '<text class="axis-title" x="20" y="179.5" text-anchor="middle" transform="rotate(-90 20 179.5)">GPU query time (%)</text>',
    f'<text class="label" x="{bar_x + bar_width/2}" y="311" text-anchor="middle">Late-Issue, C=16</text>',
]
bottom = plot_bottom
for name, value, color in segments:
    ratio = value / total_us
    height = (plot_bottom - plot_top) * ratio
    y = bottom - height
    center_y = y + height / 2
    stack += [
        f'<rect x="{bar_x}" y="{y:.2f}" width="{bar_width}" height="{height:.2f}" fill="{color}" stroke="white" stroke-width="1"/>',
        f'<text class="inside" x="{bar_x + bar_width/2}" y="{center_y + 4:.2f}">{esc(name)}</text>',
        f'<text class="value" x="{bar_x + bar_width + 10}" y="{center_y + 4:.2f}">{value:.1f} μs ({100*ratio:.1f}%)</text>',
    ]
    bottom = y
svg("program3_query_time_breakdown.svg", "".join(stack), 540, 335)


early = d["performance"]["early"]
effectiveness = bar_panel(
    80, 700, "Query throughput before and after decoupling",
    "Query throughput (QPS)", ["Late-Issue", "Exact-Early-Issue"],
    [late["qps"], early["qps"]], ["#b8c4d8", "#8fc9b5"],
    75000, [0, 15000, 30000, 45000, 60000, 75000],
    lambda value, tick: (f"{value/1000:.0f}k" if tick and value else
                         ("0" if tick else f"{value:,.0f}")))
svg("program3_story_effectiveness.svg", effectiveness, 860)
