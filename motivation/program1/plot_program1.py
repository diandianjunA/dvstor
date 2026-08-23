#!/usr/bin/env python3
"""Render Program 1 motivation figures as dependency-free SVG files."""

import html
import json
import sys
from pathlib import Path


if len(sys.argv) != 2:
    raise SystemExit("usage: plot_program1.py <run-root>")
root = Path(sys.argv[1]).resolve()


def read_first(pattern):
    paths = sorted(root.glob(pattern))
    if not paths:
        raise SystemExit(f"no input matched: {root / pattern}")
    return json.loads(paths[0].read_text(encoding="utf-8"))


def load_data():
    baseline = read_first("baseline/**/*.json")
    critical = baseline["coupled_insert_critical_path"]
    if "rdma_wait_ns" not in critical:
        raise SystemExit(
            "baseline predates exact RDMA timing; rerun the baseline case")
    baseline_qps = baseline["throughput"]["effective_insert_ops_per_sec"]
    data = {
        "baseline_insert_qps": baseline_qps,
        "critical_path_total_ns": critical["total_ns"],
        "critical_path_rdma_ns": critical["rdma_wait_ns"],
    }
    solution_paths = sorted(root.glob("solution/**/*.json"))
    quality_path = root / "quality" / "quality.json"
    if solution_paths and quality_path.exists():
        solution = json.loads(solution_paths[0].read_text(encoding="utf-8"))
        quality = json.loads(quality_path.read_text(encoding="utf-8"))
        stage2 = solution["stage2"]
        final_edges = max(stage2["stage2_final_edges"], 1)
        solution_qps = solution["throughput"]["effective_insert_ops_per_sec"]
        data.update({
            "solution_insert_qps": solution_qps,
            "insert_speedup": solution_qps / baseline_qps,
            "stage1_only_self_hit_rate": quality["stage1_only_self_recall"]["hit_rate"],
            "finalized_self_hit_rate": quality["finalized_self_recall"]["hit_rate"],
            "cross_edge_ratio_stage1_home": stage2["stage2_cross_edges_stage1_home"] / final_edges,
            "cross_edge_ratio_final_home": stage2["stage2_cross_edges_final_home"] / final_edges,
            "cross_edge_reduction_ratio": stage2["cross_edge_reduction_ratio"],
        })
    return data


d = load_data()


def esc(value):
    return html.escape(str(value))


def rdma_cpu_chart(width, height, title, panel_label=""):
    total_ns = d["critical_path_total_ns"]
    rdma_ns = d["critical_path_rdma_ns"]
    cpu_ns = max(total_ns - rdma_ns, 0)
    rdma_ratio = rdma_ns / max(total_ns, 1)
    cpu_ratio = cpu_ns / max(total_ns, 1)
    plot_left, plot_right = 67, 260
    bar_x, bar_y, bar_w, bar_h = 132, 66, 64, 220
    rdma_h = bar_h * rdma_ratio
    cpu_h = bar_h - rdma_h
    rdma_y = bar_y
    cpu_y = bar_y + rdma_h
    heading = f"{panel_label} {title}".strip()
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<style>
  text {{ font-family: Arial, "Noto Sans", sans-serif; fill: #222; }}
  .title {{ font-size: 17px; font-weight: 700; }}
  .subtitle {{ font-size: 13px; fill: #555; }}
  .axis {{ font-size: 11px; fill: #444; }}
  .axis-title {{ font-size: 12px; fill: #333; }}
  .inside {{ font-size: 12px; font-weight: 700; fill: #25313b; text-anchor: middle; }}
  .annotation {{ font-size: 11px; font-weight: 700; fill: #25313b; }}
  .note {{ font-size: 11px; fill: #666; text-anchor: middle; }}
  .grid {{ stroke: #e5e7eb; stroke-width: 1; }}
  .axis-line {{ stroke: #555; stroke-width: 1.1; }}
</style>
<rect width="100%" height="100%" fill="white"/>
<text class="title" x="{width/2:.0f}" y="24" text-anchor="middle">{esc(heading)}</text>
<text class="subtitle" x="{width/2:.0f}" y="45" text-anchor="middle">Total: {total_ns/1e9:.2f} ms per insertion</text>
<line class="grid" x1="{plot_left}" y1="66" x2="{plot_right}" y2="66"/>
<line class="grid" x1="{plot_left}" y1="121" x2="{plot_right}" y2="121"/>
<line class="grid" x1="{plot_left}" y1="176" x2="{plot_right}" y2="176"/>
<line class="grid" x1="{plot_left}" y1="231" x2="{plot_right}" y2="231"/>
<line class="grid" x1="{plot_left}" y1="286" x2="{plot_right}" y2="286"/>
<line class="axis-line" x1="{plot_left}" y1="66" x2="{plot_left}" y2="286"/>
<line class="axis-line" x1="{plot_left}" y1="286" x2="{plot_right}" y2="286"/>
<text class="axis" x="60" y="70" text-anchor="end">100</text>
<text class="axis" x="60" y="125" text-anchor="end">75</text>
<text class="axis" x="60" y="180" text-anchor="end">50</text>
<text class="axis" x="60" y="235" text-anchor="end">25</text>
<text class="axis" x="60" y="290" text-anchor="end">0</text>
<text class="axis-title" x="17" y="176" text-anchor="middle" transform="rotate(-90 17 176)">Insertion time (%)</text>
<rect x="{bar_x}" y="{rdma_y:.2f}" width="{bar_w}" height="{rdma_h:.2f}" fill="#9ecae1" stroke="white" stroke-width="1"/>
<rect x="{bar_x}" y="{cpu_y:.2f}" width="{bar_w}" height="{cpu_h:.2f}" fill="#d9dde3" stroke="white" stroke-width="1"/>
<text class="inside" x="{bar_x + bar_w/2}" y="{rdma_y + rdma_h/2 + 4:.2f}">RDMA</text>
<text class="annotation" x="{bar_x + bar_w + 7}" y="{rdma_y + rdma_h/2 + 4:.2f}">{rdma_ns/1e9:.2f} ms ({100*rdma_ratio:.1f}%)</text>
<text class="inside" x="{bar_x + bar_w/2}" y="{cpu_y + cpu_h/2 + 4:.2f}">CPU</text>
<text class="annotation" x="{bar_x + bar_w + 7}" y="{cpu_y + cpu_h/2 + 4:.2f}">{cpu_ns/1e9:.2f} ms ({100*cpu_ratio:.1f}%)</text>
<text class="axis" x="{bar_x + bar_w/2}" y="301" text-anchor="middle">Insertion</text>
<text class="note" x="{width/2:.0f}" y="326">RDMA = submission/backpressure-to-completion wait</text>
</svg>'''


# Standalone paper-ready version of Figure 1.
standalone = root / "program1_figure1_rdma_cpu.svg"
standalone.write_text(rdma_cpu_chart(330, 340, "Insertion-time breakdown"), encoding="utf-8")

if "solution_insert_qps" not in d:
    print(standalone)
    raise SystemExit(0)


width, height = 1320, 350
panels = []


def panel(index, title, labels, values, colors, unit, maximum=None):
    x0 = 20 + index * 325
    max_value = maximum or max(max(values), 1)
    pieces = [
        f'<g transform="translate({x0},25)">',
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


def first_panel():
    total_ns = d["critical_path_total_ns"]
    rdma_ns = d["critical_path_rdma_ns"]
    cpu_ns = total_ns - rdma_ns
    rdma_ratio = rdma_ns / total_ns
    cpu_ratio = cpu_ns / total_ns
    plot_left, plot_right = 55, 250
    bar_x, bar_y, bar_w, bar_h = 119, 67, 62, 205
    rdma_h = bar_h * rdma_ratio
    cpu_h = bar_h - rdma_h
    pieces = [
        '<g transform="translate(20,25)">',
        '<text class="title" x="0" y="18">(a) Insertion-time breakdown</text>',
        f'<text class="subtitle" x="145" y="42" text-anchor="middle">{total_ns/1e9:.2f} ms total</text>',
        f'<line class="grid" x1="{plot_left}" y1="67" x2="{plot_right}" y2="67"/>',
        f'<line class="grid" x1="{plot_left}" y1="118.25" x2="{plot_right}" y2="118.25"/>',
        f'<line class="grid" x1="{plot_left}" y1="169.5" x2="{plot_right}" y2="169.5"/>',
        f'<line class="grid" x1="{plot_left}" y1="220.75" x2="{plot_right}" y2="220.75"/>',
        f'<line class="grid" x1="{plot_left}" y1="272" x2="{plot_right}" y2="272"/>',
        f'<line class="axis-line" x1="{plot_left}" y1="67" x2="{plot_left}" y2="272"/>',
        f'<line class="axis-line" x1="{plot_left}" y1="272" x2="{plot_right}" y2="272"/>',
        '<text class="axis" x="49" y="71" text-anchor="end">100</text>',
        '<text class="axis" x="49" y="122.25" text-anchor="end">75</text>',
        '<text class="axis" x="49" y="173.5" text-anchor="end">50</text>',
        '<text class="axis" x="49" y="224.75" text-anchor="end">25</text>',
        '<text class="axis" x="49" y="276" text-anchor="end">0</text>',
        '<text class="axis-title" x="9" y="169.5" text-anchor="middle" transform="rotate(-90 9 169.5)">Insertion time (%)</text>',
        f'<rect x="{bar_x}" y="{bar_y}" width="{bar_w}" height="{rdma_h:.2f}" fill="#9ecae1" stroke="white" stroke-width="1"/>',
        f'<rect x="{bar_x}" y="{bar_y + rdma_h:.2f}" width="{bar_w}" height="{cpu_h:.2f}" fill="#d9dde3" stroke="white" stroke-width="1"/>',
        f'<text class="inside" x="{bar_x + bar_w/2}" y="{bar_y + rdma_h/2 + 4:.2f}">RDMA</text>',
        f'<text class="annotation" x="{bar_x + bar_w + 7}" y="{bar_y + rdma_h/2 + 4:.2f}">{rdma_ns/1e9:.2f} ms ({100*rdma_ratio:.1f}%)</text>',
        f'<text class="inside" x="{bar_x + bar_w/2}" y="{bar_y + rdma_h + cpu_h/2 + 4:.2f}">CPU</text>',
        f'<text class="annotation" x="{bar_x + bar_w + 7}" y="{bar_y + rdma_h + cpu_h/2 + 4:.2f}">{cpu_ns/1e9:.2f} ms ({100*cpu_ratio:.1f}%)</text>',
        f'<text class="axis" x="{bar_x + bar_w/2}" y="288" text-anchor="middle">Insertion</text>',
        '</g>',
    ]
    panels.append("\n".join(pieces))


first_panel()
qps_max = 1.12 * max(d["baseline_insert_qps"], d["solution_insert_qps"], 1)
panel(1, f"(b) Update throughput ({d['insert_speedup']:.2f}x)",
      ["Coupled", "Two-stage"],
      [d["baseline_insert_qps"], d["solution_insert_qps"]],
      ["#7570b3", "#1b9e77"], " ops/s", qps_max)
panel(2, "(c) Temporary query quality",
      ["Stage1-only", "Stage2 final"],
      [100 * d["stage1_only_self_hit_rate"], 100 * d["finalized_self_hit_rate"]],
      ["#e6ab02", "#1b9e77"], "%", 100)
locality_max = 1.12 * max(100 * d["cross_edge_ratio_stage1_home"],
                          100 * d["cross_edge_ratio_final_home"], 1)
panel(3, f"(d) Locality ({100*d['cross_edge_reduction_ratio']:.1f}% fewer)",
      ["Before move", "After move"],
      [100 * d["cross_edge_ratio_stage1_home"], 100 * d["cross_edge_ratio_final_home"]],
      ["#e7298a", "#66a61e"], "%", locality_max)

svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<style>
  text {{ font-family: Arial, "Noto Sans", sans-serif; fill: #222; }}
  .title {{ font-size: 17px; font-weight: 700; }}
  .subtitle {{ font-size: 13px; fill: #555; }}
  .axis {{ font-size: 11px; fill: #444; }}
  .axis-title {{ font-size: 12px; fill: #333; }}
  .label {{ font-size: 14px; }}
  .value {{ font-size: 13px; font-weight: 700; }}
  .inside {{ font-size: 11px; font-weight: 700; fill: #25313b; text-anchor: middle; }}
  .annotation {{ font-size: 10px; font-weight: 700; fill: #25313b; }}
  .track {{ fill: #f0f0f0; }}
  .grid {{ stroke: #e5e7eb; stroke-width: 1; }}
  .axis-line {{ stroke: #555; stroke-width: 1.1; }}
</style>
<rect width="100%" height="100%" fill="white"/>
{''.join(panels)}
</svg>
'''
combined = root / "program1_motivation.svg"
combined.write_text(svg, encoding="utf-8")
print(standalone)
print(combined)
