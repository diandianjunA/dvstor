#!/usr/bin/env python3
"""Render two dependency-free SVG figures for Program 2."""

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


def write_svg(name, body, width=1000, height=390):
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<style>
text {{ font-family: Arial, "Noto Sans", sans-serif; fill: #222; }}
.title {{ font-size: 18px; font-weight: 700; }}
.label {{ font-size: 14px; }}
.value {{ font-size: 14px; font-weight: 700; }}
.axis {{ stroke: #555; stroke-width: 1; }}
.grid {{ stroke: #ddd; stroke-width: 1; }}
</style>
<rect width="100%" height="100%" fill="white"/>
{body}
</svg>'''
    path = root / name
    path.write_text(svg, encoding="utf-8")
    print(path)


hist = d["degree_histogram"]
total = max(sum(hist), 1)
points = []
cumulative = 0
for extent_class, count in enumerate(hist):
    cumulative += count
    x = 70 + min(extent_class * 8, 104) / 104 * 370
    y = 310 - cumulative / total * 235
    points.append(f"{x:.1f},{y:.1f}")

max_qps = max(int(key) for key in d["transport_probe"])
probe = d["transport_probe"][str(max_qps)]
methods = [
    ("Fixed 832 B", "fixed_full", "#7570b3"),
    ("Header + body", "dependent_header_body", "#d95f02"),
    ("Hinted one-read", "hinted_one_read", "#1b9e77"),
]
values = [probe[key]["logical_reads_per_s_median"] for _, key, _ in methods]
maximum = max(values)
bars = []
for index, ((label, _, color), value) in enumerate(zip(methods, values)):
    y = 105 + index * 78
    width = 330 * value / maximum
    bars.append(
        f'<text class="label" x="535" y="{y}">{esc(label)}</text>'
        f'<rect x="535" y="{y + 12}" width="330" height="25" fill="#eee"/>'
        f'<rect x="535" y="{y + 12}" width="{width:.1f}" height="25" fill="{color}"/>'
        f'<text class="value" x="875" y="{y + 31}">{value / 1e6:.2f} Mread/s</text>'
    )

motivation = f'''
<text class="title" x="40" y="32">(a) Query-weighted live degree CDF</text>
<line class="axis" x1="70" y1="310" x2="450" y2="310"/>
<line class="axis" x1="70" y1="310" x2="70" y2="65"/>
<line class="grid" x1="70" y1="192.5" x2="450" y2="192.5"/>
<line class="grid" x1="70" y1="75" x2="450" y2="75"/>
<polyline points="{' '.join(points)}" fill="none" stroke="#377eb8" stroke-width="4"/>
<text class="label" x="245" y="345">Live neighbors</text>
<text class="label" x="18" y="315">0%</text><text class="label" x="12" y="198">50%</text><text class="label" x="5" y="80">100%</text>
<text class="value" x="90" y="110">Mean degree: {d['average_expanded_parent_degree']:.2f}</text>
<text class="value" x="90" y="135">Required: {d['average_required_prefix_bytes']:.1f} B</text>
<text class="value" x="90" y="160">Fixed-read waste: {100*d['average_fixed_read_waste_ratio']:.1f}%</text>
<text class="title" x="515" y="32">(b) RDMA protocol at {max_qps} active QPs</text>
{''.join(bars)}
'''
write_svg("program2_motivation.svg", motivation)


fixed = d["fixed"]
live = d["live"]


def paired_bars(x0, title, labels, values, colors, unit):
    maximum = max(values) * 1.12 or 1
    pieces = [f'<text class="title" x="{x0}" y="32">{esc(title)}</text>']
    for index, (label, value, color) in enumerate(zip(labels, values, colors)):
        x = x0 + 45 + index * 165
        height = 230 * value / maximum
        y = 310 - height
        pieces += [
            f'<rect x="{x}" y="{y:.1f}" width="95" height="{height:.1f}" fill="{color}"/>',
            f'<text class="value" x="{x + 47}" y="{y - 8:.1f}" text-anchor="middle">{value:.2f}{esc(unit)}</text>',
            f'<text class="label" x="{x + 47}" y="338" text-anchor="middle">{esc(label)}</text>',
        ]
    return ''.join(pieces)


effect = paired_bars(
    35, f"(a) Query throughput (+{100*d['qps_improvement_ratio']:.1f}%)",
    ["Fixed", "LiveExtent"], [fixed["query_qps"], live["query_qps"]],
    ["#7570b3", "#1b9e77"], "")
effect += paired_bars(
    520, f"(b) P99 latency (-{100*d['p99_reduction_ratio']:.1f}%)",
    ["Fixed", "LiveExtent"], [fixed["p99_latency_ms"], live["p99_latency_ms"]],
    ["#7570b3", "#1b9e77"], " ms")
effect += (
    f'<text class="label" x="500" y="375" text-anchor="middle">'
    f'Graph bytes/query: -{100*d["graph_bytes_reduction_ratio"]:.1f}% | '
    f'Physical WQE/query: {100*d["physical_wqe_change_ratio"]:+.2f}% | '
    f'Recall equal: {esc(d["recall_equal"])}</text>'
)
write_svg("program2_effectiveness.svg", effect)
