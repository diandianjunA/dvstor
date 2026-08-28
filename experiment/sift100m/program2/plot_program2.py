#!/usr/bin/env python3
"""Render two dependency-free SVG figures for the dynamic Program 2 run."""

import html
import json
import sys
from pathlib import Path


if len(sys.argv) != 2:
    raise SystemExit("usage: plot_program2.py <run-root>")
root = Path(sys.argv[1]).resolve()
d = json.loads((root / "summary.json").read_text(encoding="utf-8"))


def esc(value):
    return html.escape(str(value))


def write_svg(name, body, width=1050, height=365):
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<style>
text {{ font-family: Arial, "Noto Sans", sans-serif; fill: #222; }}
.title {{ font-size: 18px; font-weight: 700; }}
.label {{ font-size: 12px; fill: #374151; }}
.small {{ font-size: 11px; fill: #555; }}
.value {{ font-size: 12px; font-weight: 700; fill: #25313b; }}
.tick {{ font-size: 10px; fill: #4b5563; }}
.axis-title {{ font-size: 11px; fill: #374151; }}
.axis {{ stroke: #555; stroke-width: 1.1; }}
.grid {{ stroke: #e5e7eb; stroke-width: 1; }}
</style><rect width="100%" height="100%" fill="white"/>{body}</svg>'''
    (root / name).write_text(svg, encoding="utf-8")
    print(root / name)


def vertical_bars(x0, title, ylabel, labels, values, colors,
                  maximum, ticks, value_format, width=460):
    plot_left = x0 + 62
    plot_right = x0 + width - 12
    plot_top, plot_bottom = 70, 295
    plot_height = plot_bottom - plot_top
    spacing = (plot_right - plot_left) / len(values)
    bar_width = 56
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
        f'<text class="axis-title" x="{x0 + 16}" y="{(plot_top + plot_bottom)/2:.1f}" text-anchor="middle" transform="rotate(-90 {x0 + 16} {(plot_top + plot_bottom)/2:.1f})">{esc(ylabel)}</text>',
    ]
    for index, (label, value, color) in enumerate(zip(labels, values, colors)):
        center = plot_left + spacing * (index + 0.5)
        height = plot_height * value / maximum
        y = plot_bottom - height
        pieces += [
            f'<rect x="{center - bar_width/2:.1f}" y="{y:.1f}" width="{bar_width}" height="{height:.1f}" fill="{color}" stroke="white" stroke-width="1"/>',
            f'<text class="value" x="{center:.1f}" y="{max(plot_top + 12, y - 7):.1f}" text-anchor="middle">{esc(value_format(value, False))}</text>',
            f'<text class="label" x="{center:.1f}" y="316" text-anchor="middle">{esc(label)}</text>',
        ]
    return "".join(pieces)


hist = d["dynamic_degree_histogram"]
total = max(sum(hist), 1)
points, cumulative = [], 0
for extent_class, count in enumerate(hist):
    cumulative += count
    x = 65 + min(extent_class * 8, 104) / 104 * 390
    y = 295 - cumulative / total * 225
    points.append(f"{x:.1f},{y:.1f}")

oracle_b = d["oracle"]["average_bytes_per_committed_dynamic_parent"]
motivation = f'''
<text class="title" x="242" y="34" text-anchor="middle">(a) Distribution of valid neighbor counts</text>
<line class="grid" x1="65" y1="295" x2="455" y2="295"/><line class="grid" x1="65" y1="182.5" x2="455" y2="182.5"/><line class="grid" x1="65" y1="70" x2="455" y2="70"/>
<line class="axis" x1="65" y1="295" x2="455" y2="295"/><line class="axis" x1="65" y1="295" x2="65" y2="70"/>
<polyline points="{' '.join(points)}" fill="none" stroke="#7faed0" stroke-width="3"/>
<text class="axis-title" x="260" y="326" text-anchor="middle">Number of valid neighbors</text>
<text class="axis-title" x="18" y="182.5" text-anchor="middle" transform="rotate(-90 18 182.5)">Cumulative query accesses (%)</text>
<text class="tick" x="58" y="299" text-anchor="end">0</text><text class="tick" x="58" y="186.5" text-anchor="end">50</text><text class="tick" x="58" y="74" text-anchor="end">100</text>
<text class="tick" x="65" y="311" text-anchor="middle">0</text><text class="tick" x="140" y="311" text-anchor="middle">20</text><text class="tick" x="215" y="311" text-anchor="middle">40</text><text class="tick" x="290" y="311" text-anchor="middle">60</text><text class="tick" x="365" y="311" text-anchor="middle">80</text><text class="tick" x="440" y="311" text-anchor="middle">100</text>
<text class="value" x="91" y="102">Average = {d['average_dynamic_degree']:.2f} neighbors</text>
<text class="value" x="91" y="123">50% of accesses ≤ {d['dynamic_degree_p50_upper_bound']} neighbors</text>
<text class="value" x="91" y="144">95% of accesses ≤ {d['dynamic_degree_p95_upper_bound']} neighbors</text>
'''
motivation += vertical_bars(
    540, "(b) Traditional reads vs ideal lower bound", "Bytes per node",
    ["Fixed", "Header→N", "Oracle"], [832, oracle_b, oracle_b],
    ["#b8c4d8", "#efc08d", "#d9dde3"], 900,
    [0, 200, 400, 600, 800],
    lambda value, tick: f"{value:g}" if tick else f"{value:.0f} B")
write_svg("program2_motivation.svg", motivation)


cases = d["cases"]
fixed, header, live = cases["fixed"], cases["header"], cases["live"]
effect = vertical_bars(
    20, "(a) Dynamic mixed-workload query throughput", "Query throughput (QPS)",
    ["Fixed", "Header→N", "ClassExtent"],
    [fixed["query_qps"], header["query_qps"], live["query_qps"]],
    ["#b8c4d8", "#efc08d", "#8fc9b5"], 80000,
    [0, 20000, 40000, 60000, 80000],
    lambda value, tick: f"{value/1000:.0f}k" if tick and value else
      ("0" if tick else f"{value:,.0f}"), width=470)
effect += vertical_bars(
    555, "(b) Dynamic bytes per committed parent", "Bytes per parent",
    ["Oracle", "Fixed", "Header→N", "ClassExtent"],
    [
        oracle_b,
        fixed["dynamic_bytes_per_committed_parent"],
        header["dynamic_bytes_per_committed_parent"],
        live["dynamic_bytes_per_committed_parent"],
    ],
    ["#d9dde3", "#b8c4d8", "#efc08d", "#8fc9b5"], 900,
    [0, 200, 400, 600, 800],
    lambda value, tick: f"{value:g}" if tick else f"{value:.0f} B",
    width=470)
write_svg("program2_effectiveness.svg", effect)
