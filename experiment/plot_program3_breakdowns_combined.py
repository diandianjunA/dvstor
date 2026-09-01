#!/usr/bin/env python3
"""Plot Program 3 breakdowns for SIFT, DEEP, and SPACEV as academic SVGs."""

import argparse
import html
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RUNS = {
    "SIFT100M": REPO_ROOT / "experiment/sift100m/program3/results/program3_20260831_191030",
    "DEEP100M": REPO_ROOT / "experiment/deep100m/program3/results/program3_20260831_200109",
    "SPACEV100M": REPO_ROOT / "experiment/spacev100m/program3/results/program3_20260831_193611",
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    for dataset, default in DEFAULT_RUNS.items():
        parser.add_argument(
            f"--{dataset.lower()}", type=Path, default=default,
            help=f"{dataset} Program 3 run directory (default: {default})")
    parser.add_argument(
        "--output-dir", type=Path,
        default=REPO_ROOT / "experiment/paper_figures")
    return parser.parse_args()


def load_data(dataset, run_root):
    source = run_root.resolve() / "summary.json"
    if not source.is_file():
        raise SystemExit(f"missing {dataset} summary: {source}")
    report = json.loads(source.read_text(encoding="utf-8"))
    late = report["performance"]["late"]
    return {"dataset": dataset, "late": late, "source": source}


def esc(value):
    return html.escape(str(value))


def render_stacked(rows, title, ylabel, segment_fn, legend_columns):
    width, height = 650, 390
    plot_left, plot_right = 76, 625
    plot_top, plot_bottom = 91, 323
    plot_height = plot_bottom - plot_top
    centers = [190, 350, 510]
    bar_width = 72

    pieces = [f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<style>
  text {{ font-family: Arial, sans-serif; fill: #25313b; }}
  .title {{ font-size: 15px; font-weight: 700; }}
  .tick {{ font-size: 10px; fill: #4b5563; }}
  .axis-title {{ font-size: 12px; fill: #374151; }}
  .dataset {{ font-size: 12px; font-weight: 700; }}
  .inside {{ font-size: 10px; font-weight: 700; text-anchor: middle; }}
  .total {{ font-size: 10px; fill: #4b5563; text-anchor: middle; }}
  .legend {{ font-size: 10px; fill: #30343b; }}
  .axis {{ stroke: #555; stroke-width: 1.15; }}
  .grid {{ stroke: #e1e5ea; stroke-width: 1; }}
</style>
<rect width="100%" height="100%" fill="white"/>
<text class="title" x="{width / 2:.1f}" y="24" text-anchor="middle">{esc(title)}</text>''']

    first_segments, _ = segment_fn(rows[0]["late"])
    column_width = 180 if legend_columns == 3 else 230
    legend_width = legend_columns * column_width
    legend_x = (width - legend_width) / 2
    for index, (name, _, color) in enumerate(first_segments):
        column = index % legend_columns
        row = index // legend_columns
        x = legend_x + column * column_width
        y = 51 + row * 20
        pieces.extend([
            f'<rect x="{x:.1f}" y="{y - 10}" width="12" height="12" fill="{color}"/>',
            f'<text class="legend" x="{x + 18:.1f}" y="{y}">{esc(name)}</text>',
        ])

    for tick in [0, 20, 40, 60, 80, 100]:
        y = plot_bottom - plot_height * tick / 100
        pieces.extend([
            f'<line class="grid" x1="{plot_left}" y1="{y:.2f}" x2="{plot_right}" y2="{y:.2f}"/>',
            f'<text class="tick" x="{plot_left - 8}" y="{y + 4:.2f}" text-anchor="end">{tick}</text>',
        ])
    pieces.extend([
        f'<line class="axis" x1="{plot_left}" y1="{plot_top}" x2="{plot_left}" y2="{plot_bottom}"/>',
        f'<line class="axis" x1="{plot_left}" y1="{plot_bottom}" x2="{plot_right}" y2="{plot_bottom}"/>',
        f'<text class="axis-title" x="0" y="0" text-anchor="middle" transform="translate(22 {(plot_top + plot_bottom) / 2:.1f}) rotate(-90)">{esc(ylabel)}</text>',
    ])

    for center, row in zip(centers, rows):
        segments, total_us = segment_fn(row["late"])
        bottom = plot_bottom
        for name, value, color in segments:
            ratio = value / total_us
            segment_height = plot_height * ratio
            y = bottom - segment_height
            pieces.extend([
                f'<rect x="{center - bar_width / 2:.1f}" y="{y:.2f}" width="{bar_width}" height="{segment_height:.2f}" fill="{color}" stroke="white" stroke-width="1"/>',
                f'<text class="inside" x="{center}" y="{y + segment_height / 2 + 3.5:.2f}">{100 * ratio:.1f}%</text>',
            ])
            bottom = y
        pieces.extend([
            f'<text class="total" x="{center}" y="{plot_top - 8}">{total_us / 1000:.2f} ms</text>',
            f'<text class="dataset" x="{center}" y="345" text-anchor="middle">{esc(row["dataset"])}</text>',
        ])
    pieces.append('</svg>')
    return "\n".join(pieces)


def query_segments(late):
    return ([
        ("Candidate maintenance", late["gpu_candidate_maintenance_us"], "#efc08d"),
        ("RDMA", late["gpu_rdma_us"], "#9ecae1"),
        ("Distance computation", late["gpu_distance_us"], "#c9b8d8"),
        ("Others", late["gpu_other_us"], "#d9dde3"),
    ], late["gpu_query_us"])


def maintenance_segments(late):
    breakdown = late["candidate_maintenance_breakdown"]
    return ([
        ("Candidate validation", breakdown["Candidate Validation"], "#b8c4d8"),
        ("Candidate select", breakdown["Candidate Select"], "#f5d8b5"),
        ("Merge", breakdown["Merge"], "#efb77e"),
    ], late["gpu_candidate_maintenance_us"])


def main():
    args = parse_args()
    rows = [
        load_data("SIFT100M", args.sift100m),
        load_data("DEEP100M", args.deep100m),
        load_data("SPACEV100M", args.spacev100m),
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = [
        (
            args.output_dir / "program3_query_time_breakdown_three_datasets.svg",
            render_stacked(
                rows, "GPU query-time breakdown before decoupling",
                "GPU query time (%)", query_segments, 2),
        ),
        (
            args.output_dir / "program3_candidate_maintenance_breakdown_three_datasets.svg",
            render_stacked(
                rows, "Candidate-maintenance breakdown",
                "Candidate maintenance time (%)", maintenance_segments, 3),
        ),
    ]
    for output, content in outputs:
        output.write_text(content, encoding="utf-8")
        print(output.resolve())
    for row in rows:
        print(f'{row["dataset"]}: source={row["source"]}')


if __name__ == "__main__":
    main()
