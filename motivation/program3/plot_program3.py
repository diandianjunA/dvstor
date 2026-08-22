#!/usr/bin/env python3
"""Create two dependency-free SVG figures for Program 3."""

import html
import json
import sys
from pathlib import Path


if len(sys.argv) != 2:
    raise SystemExit("usage: plot_program3.py <run-root>")
root = Path(sys.argv[1]).resolve()
d = json.loads((root / "summary.json").read_text(encoding="utf-8"))


def esc(value):
    return html.escape(str(value))


def write_svg(name, body, width=1050, height=430):
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<style>text {{ font-family: Arial, "Noto Sans", sans-serif; fill:#222 }} .title {{font-size:19px;font-weight:700}} .label {{font-size:14px}} .small {{font-size:12px;fill:#555}} .value {{font-size:14px;font-weight:700}} .axis {{stroke:#555}} .grid {{stroke:#ddd}}</style>
<rect width="100%" height="100%" fill="white"/>{body}</svg>'''
    (root / name).write_text(svg, encoding="utf-8")
    print(root / name)


preferred = "query" if "query" in d["cases"] else next(iter(d["cases"]))
early = d["cases"][preferred]["early"]
values = [
    early["prefix_to_publish_us_per_certificate"],
    early["issue_to_publish_us_per_certificate"],
    early["rdma_completion_latency_us"],
]
labels = ["Prefix→Beam publish", "RDMA issue→Beam publish", "RDMA completion latency"]
colors = ["#7570b3", "#1b9e77", "#d95f02"]
maximum = max(values) * 1.18 or 1
bars = []
for i, (label, value, color) in enumerate(zip(labels, values, colors)):
    y = 105 + i * 82
    width = 700 * value / maximum
    bars += [
        f'<text class="label" x="25" y="{y+23}">{esc(label)}</text>',
        f'<rect x="245" y="{y}" width="{width:.1f}" height="34" fill="{color}"/>',
        f'<text class="value" x="{255+width:.1f}" y="{y+23}">{value:.3f} μs</text>',
    ]
motivation = f'''
<text class="title" x="25" y="38">Software barrier leaves a safe RDMA-overlap window ({esc(preferred)})</text>
<text class="small" x="25" y="65">Exact next-C parents are known at Prefix ready; traditional execution waits for full Beam publication.</text>
{''.join(bars)}
<text class="small" x="25" y="390">Measured per exact certificate; issue width = commit width, so the window contains mandatory reads only.</text>
'''
write_svg("program3_motivation.svg", motivation)


workloads = list(d["cases"])
panel = ['<text class="title" x="25" y="38">Exact-core early issue: strict Persistent A/B</text>']
group_width = 900 / max(len(workloads), 1)
for wi, workload in enumerate(workloads):
    x0 = 85 + wi * group_width
    late = d["cases"][workload]["late"]
    early = d["cases"][workload]["early"]
    ratio = d["paired"][workload]["qps_geomean_ratio"]
    heights = [210, 210 * ratio]
    for i, (name, height, color, value) in enumerate([
        ("Late", heights[0], "#7570b3", 1.0),
        ("Exact Early", heights[1], "#1b9e77", ratio),
    ]):
        x = x0 + i * 125
        y = 315 - height
        panel += [
            f'<rect x="{x}" y="{y:.1f}" width="90" height="{height:.1f}" fill="{color}"/>',
            f'<text class="value" x="{x+45}" y="{y-8:.1f}" text-anchor="middle">{value:.3f}×</text>',
            f'<text class="label" x="{x+45}" y="340" text-anchor="middle">{name}</text>',
        ]
    panel += [
        f'<text class="title" x="{x0+107}" y="382" text-anchor="middle">{esc(workload)}</text>',
        f'<text class="small" x="{x0+107}" y="405" text-anchor="middle">P99 {100*(d["paired"][workload]["p99_geomean_ratio"]-1):+.2f}% · ROB hit {100*early["critical_rob_hit_ratio"]:.1f}%</text>',
    ]
panel += ['<line class="axis" x1="55" y1="315" x2="1000" y2="315"/>']
write_svg("program3_effectiveness.svg", ''.join(panel))
