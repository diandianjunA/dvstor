#!/usr/bin/env python3
import html
import json
import math
import sys
from pathlib import Path


if len(sys.argv) != 2:
    raise SystemExit("usage: plot_ablation.py <run-root>")

root = Path(sys.argv[1]).resolve()
summary_path = root / "summary.json"
if not summary_path.is_file():
    raise SystemExit(f"missing summary: {summary_path}")
with summary_path.open(encoding="utf-8") as stream:
    summary = json.load(stream)

rows = summary["rows"]
labels = ["Baseline", "+ P1", "+ P2", "+ P3 (Full)"]
reference = summary.get("reference")

canvas_width = 920
canvas_height = 350
plot_top = 46
plot_bottom = 286
panel_width = 330
panel_lefts = [80, 535]
bar_width = 42
bar_centers = [48, 126, 204, 282]


def nice_max(values):
    largest = max(values) if values else 1.0
    if largest <= 0:
        return 1.0
    raw_step = largest * 1.15 / 4
    magnitude = 10 ** math.floor(math.log10(raw_step))
    normalized = raw_step / magnitude
    factor = 1 if normalized <= 1 else 2 if normalized <= 2 else 5
    step = factor * magnitude
    return step * math.ceil(largest * 1.12 / step)


elements = [
    f'<svg xmlns="http://www.w3.org/2000/svg" width="{canvas_width}" '
    f'height="{canvas_height}" viewBox="0 0 {canvas_width} {canvas_height}">',
    '<rect width="100%" height="100%" fill="white"/>',
    '<style>text{font-family:DejaVu Sans,Arial,sans-serif;fill:#27323A}'
    '.tick{font-size:11px}.value{font-size:11px}.title{font-size:13px}'
    '.axis-label{font-size:12px}</style>',
]


def draw_panel(panel_index, metric, title, ylabel, fill, reference_metric):
    left = panel_lefts[panel_index]
    values = [float(row[metric]) for row in rows]
    ref_value = (float(reference.get(reference_metric, 0))
                 if reference else 0.0)
    upper = nice_max(values + ([ref_value] if ref_value else []))
    plot_height = plot_bottom - plot_top

    def y_coord(value):
        return plot_bottom - value / upper * plot_height

    elements.append(
        f'<text class="title" x="{left + panel_width / 2:.1f}" y="20" '
        f'text-anchor="middle">{html.escape(title)}</text>')
    for tick in range(5):
        value = upper * tick / 4
        y = y_coord(value)
        elements.append(
            f'<line x1="{left}" y1="{y:.2f}" x2="{left + panel_width}" '
            f'y2="{y:.2f}" stroke="#D8DDE1" stroke-width="0.8"/>')
        elements.append(
            f'<text class="tick" x="{left - 8}" y="{y + 4:.2f}" '
            f'text-anchor="end">{value:,.0f}</text>')
    elements.append(
        f'<line x1="{left}" y1="{plot_top}" x2="{left}" y2="{plot_bottom}" '
        'stroke="#34424B" stroke-width="1"/>')
    elements.append(
        f'<line x1="{left}" y1="{plot_bottom}" x2="{left + panel_width}" '
        f'y2="{plot_bottom}" stroke="#34424B" stroke-width="1"/>')

    if ref_value:
        ref_y = y_coord(ref_value)
        elements.append(
            f'<line x1="{left}" y1="{ref_y:.2f}" x2="{left + panel_width}" '
            f'y2="{ref_y:.2f}" stroke="#7E8A92" stroke-width="1.2" '
            'stroke-dasharray="5 4"/>')
        elements.append(
            f'<text class="tick" x="{left + panel_width - 2}" '
            f'y="{ref_y - 5:.2f}" text-anchor="end">Published full</text>')

    for center, label, value in zip(bar_centers, labels, values):
        x = left + center - bar_width / 2
        y = y_coord(value)
        height = plot_bottom - y
        elements.append(
            f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_width}" '
            f'height="{height:.2f}" fill="{fill}" stroke="#65727A" '
            'stroke-width="0.8"/>')
        elements.append(
            f'<text class="value" x="{left + center:.2f}" y="{max(plot_top + 11, y - 6):.2f}" '
            f'text-anchor="middle">{value:,.0f}</text>')
        elements.append(
            f'<text class="tick" x="{left + center:.2f}" y="{plot_bottom + 19}" '
            f'text-anchor="middle">{html.escape(label)}</text>')

    label_x = left - 58
    label_y = (plot_top + plot_bottom) / 2
    elements.append(
        f'<text class="axis-label" x="{label_x}" y="{label_y}" '
        f'text-anchor="middle" transform="rotate(-90 {label_x} {label_y})">'
        f'{html.escape(ylabel)}</text>')


draw_panel(0, "query_qps", "(a) Query", "Query throughput (QPS)",
           "#A9CFE5", "query_qps")
draw_panel(1, "write_qps", "(b) Insert", "Insert throughput (QPS)",
           "#F3C59D", "write_qps")
elements.append("</svg>")

output = root / "ablation_performance.svg"
output.write_text("\n".join(elements) + "\n", encoding="utf-8")
print(f"figure: {output}")
