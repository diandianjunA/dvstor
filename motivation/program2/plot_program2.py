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


def write_svg(name, body, width=1050, height=420):
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<style>
text {{ font-family: Arial, "Noto Sans", sans-serif; fill: #222; }}
.title {{ font-size: 18px; font-weight: 700; }}
.label {{ font-size: 14px; }}
.small {{ font-size: 12px; fill: #555; }}
.value {{ font-size: 14px; font-weight: 700; }}
.axis {{ stroke: #555; stroke-width: 1; }}
.grid {{ stroke: #ddd; stroke-width: 1; }}
</style><rect width="100%" height="100%" fill="white"/>{body}</svg>'''
    (root / name).write_text(svg, encoding="utf-8")
    print(root / name)


def vertical_bars(x0, title, labels, values, colors, unit, width=460):
    maximum = max(values) * 1.15 or 1
    spacing = width / len(values)
    pieces = [f'<text class="title" x="{x0}" y="34">{esc(title)}</text>']
    for index, (label, value, color) in enumerate(zip(labels, values, colors)):
        x = x0 + 30 + index * spacing
        height = 235 * value / maximum
        y = 325 - height
        pieces += [
            f'<rect x="{x:.1f}" y="{y:.1f}" width="92" height="{height:.1f}" fill="{color}"/>',
            f'<text class="value" x="{x + 46:.1f}" y="{y - 8:.1f}" text-anchor="middle">{value:.1f}{esc(unit)}</text>',
            f'<text class="label" x="{x + 46:.1f}" y="351" text-anchor="middle">{esc(label)}</text>',
        ]
    return "".join(pieces)


hist = d["dynamic_degree_histogram"]
total = max(sum(hist), 1)
points, cumulative = [], 0
for extent_class, count in enumerate(hist):
    cumulative += count
    x = 65 + min(extent_class * 8, 104) / 104 * 390
    y = 325 - cumulative / total * 240
    points.append(f"{x:.1f},{y:.1f}")

oracle_b = d["oracle"]["average_bytes_per_committed_dynamic_parent"]
motivation = f'''
<text class="title" x="30" y="34">(a) Dynamic-node live-degree CDF</text>
<line class="axis" x1="65" y1="325" x2="455" y2="325"/><line class="axis" x1="65" y1="325" x2="65" y2="75"/>
<line class="grid" x1="65" y1="205" x2="455" y2="205"/><line class="grid" x1="65" y1="85" x2="455" y2="85"/>
<polyline points="{' '.join(points)}" fill="none" stroke="#377eb8" stroke-width="4"/>
<text class="label" x="225" y="357">Live neighbors (8-neighbor buckets)</text>
<text class="small" x="22" y="330">0%</text><text class="small" x="15" y="210">50%</text><text class="small" x="8" y="90">100%</text>
<text class="value" x="92" y="120">Mean = {d['average_dynamic_degree']:.2f}</text>
<text class="value" x="92" y="143">P50 ≤ {d['dynamic_degree_p50_upper_bound']}</text>
<text class="value" x="92" y="166">P95 ≤ {d['dynamic_degree_p95_upper_bound']}</text>
'''
motivation += vertical_bars(
    540, "(b) Traditional reads vs ideal lower bound",
    ["Fixed", "Header→N", "Oracle"], [832, oracle_b, oracle_b],
    ["#7570b3", "#d95f02", "#999999"], " B")
motivation += '''
<text class="small" x="610" y="383">RDMA reads per node: Fixed 1 · Header→N 2 serial · Oracle 1</text>
<text class="small" x="610" y="401">Oracle assumes free exact-length knowledge; it is not deployable.</text>
'''
write_svg("program2_motivation.svg", motivation)


cases = d["cases"]
fixed, header, live = cases["fixed"], cases["header"], cases["live"]
effect = vertical_bars(
    20, "(a) Dynamic mixed-workload query throughput",
    ["Fixed", "Header→N", "ClassExtent"],
    [fixed["query_qps"], header["query_qps"], live["query_qps"]],
    ["#7570b3", "#d95f02", "#1b9e77"], "", width=470)
effect += vertical_bars(
    555, "(b) Dynamic bytes per committed parent",
    ["Oracle", "Fixed", "Header→N", "ClassExtent"],
    [
        oracle_b,
        fixed["dynamic_bytes_per_committed_parent"],
        header["dynamic_bytes_per_committed_parent"],
        live["dynamic_bytes_per_committed_parent"],
    ],
    ["#999999", "#7570b3", "#d95f02", "#1b9e77"], " B", width=470)
effect += f'''
<text class="small" x="525" y="383" text-anchor="middle">fixed update target: {d['target_write_qps']:.0f} op/s · attained: {fixed['write_rate_attainment_ratio']:.3f}/{header['write_rate_attainment_ratio']:.3f}/{live['write_rate_attainment_ratio']:.3f}</text>
<text class="small" x="525" y="402" text-anchor="middle">ClassExtent fallback: {100*live['fallback_ratio']:.3f}% · dynamic access share: {100*live['dynamic_expanded_parent_ratio']:.3f}% · Oracle has no QPS bar</text>
'''
write_svg("program2_effectiveness.svg", effect)
