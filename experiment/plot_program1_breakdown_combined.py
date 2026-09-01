#!/usr/bin/env python3
"""Plot the Program 1 RDMA/CPU insertion-time breakdown for all datasets."""

import argparse
import html
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RUNS = {
    "SIFT100M": REPO_ROOT / "experiment/sift100m/program1/results/program1_20260831_190010",
    "DEEP100M": REPO_ROOT / "experiment/deep100m/program1/results/program1_20260831_194733",
    "SPACEV100M": REPO_ROOT / "experiment/spacev100m/program1/results/program1_20260831_192505",
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    for dataset, default in DEFAULT_RUNS.items():
        parser.add_argument(
            f"--{dataset.lower()}", type=Path, default=default,
            help=f"{dataset} Program 1 run directory (default: {default})")
    parser.add_argument(
        "--output", type=Path,
        default=REPO_ROOT / "experiment/paper_figures/program1_rdma_cpu_three_datasets.svg")
    return parser.parse_args()


def load_breakdown(dataset, run_root):
    paths = sorted(run_root.resolve().glob("baseline/**/*.json"))
    if len(paths) != 1:
        raise SystemExit(
            f"expected exactly one baseline JSON for {dataset} in {run_root}, found {len(paths)}")
    report = json.loads(paths[0].read_text(encoding="utf-8"))
    critical = report["coupled_insert_critical_path"]
    total_ns = critical["total_ns"]
    rdma_ns = critical["rdma_wait_ns"]
    if total_ns <= 0 or not 0 <= rdma_ns <= total_ns:
        raise SystemExit(f"invalid timing values in {paths[0]}")
    return {
        "dataset": dataset,
        "source": paths[0],
        # Match plot_program1.py and the three existing standalone figures.
        # Those figures intentionally display the raw ns counters divided by 1e9.
        "total_ms": total_ns / 1e9,
        "rdma_ms": rdma_ns / 1e9,
        "cpu_ms": (total_ns - rdma_ns) / 1e9,
        "rdma_pct": 100.0 * rdma_ns / total_ns,
        "cpu_pct": 100.0 * (total_ns - rdma_ns) / total_ns,
    }


def esc(value):
    return html.escape(str(value))


def render(rows):
    width, height = 650, 405
    plot_left, plot_right = 72, 625
    plot_top, plot_bottom = 58, 330
    plot_height = plot_bottom - plot_top
    centers = [165, 330, 495]
    bar_width = 92
    pieces = [f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<style>
  text {{ font-family: Arial, sans-serif; fill: #222; }}
  .tick {{ font-size: 13px; fill: #444; }}
  .axis-title {{ font-size: 14px; fill: #333; }}
  .dataset {{ font-size: 14px; font-weight: 700; }}
  .inside {{ font-size: 12px; font-weight: 700; fill: #25313b; text-anchor: middle; }}
  .value {{ font-size: 10px; font-weight: 700; fill: #25313b; text-anchor: middle; }}
  .legend {{ font-size: 13px; fill: #333; }}
  .grid {{ stroke: #dfe3e8; stroke-width: 1; }}
  .axis {{ stroke: #555; stroke-width: 1.2; }}
</style>
<rect width="100%" height="100%" fill="white"/>''']

    for tick in [0, 25, 50, 75, 100]:
        y = plot_bottom - plot_height * tick / 100
        pieces.extend([
            f'<line class="grid" x1="{plot_left}" y1="{y:.2f}" x2="{plot_right}" y2="{y:.2f}"/>',
            f'<text class="tick" x="68" y="{y + 4:.2f}" text-anchor="end">{tick}</text>',
        ])
    pieces.extend([
        f'<line class="axis" x1="{plot_left}" y1="{plot_top}" x2="{plot_left}" y2="{plot_bottom}"/>',
        f'<line class="axis" x1="{plot_left}" y1="{plot_bottom}" x2="{plot_right}" y2="{plot_bottom}"/>',
        f'<text class="axis-title" x="0" y="0" text-anchor="middle" transform="translate(22 {(plot_top + plot_bottom) / 2:.1f}) rotate(-90)">Insertion time (%)</text>',
    ])

    for center, row in zip(centers, rows):
        rdma_height = plot_height * row["rdma_pct"] / 100
        cpu_height = plot_height - rdma_height
        rdma_y = plot_top
        cpu_y = plot_top + rdma_height
        pieces.extend([
            f'<rect x="{center - bar_width / 2:.1f}" y="{rdma_y:.2f}" width="{bar_width}" height="{rdma_height:.2f}" fill="#9ecae1" stroke="white"/>',
            f'<rect x="{center - bar_width / 2:.1f}" y="{cpu_y:.2f}" width="{bar_width}" height="{cpu_height:.2f}" fill="#d9dde3" stroke="white"/>',
            f'<text class="inside" x="{center}" y="{rdma_y + rdma_height / 2 - 13:.2f}">RDMA</text>',
            f'<text class="value" x="{center}" y="{rdma_y + rdma_height / 2 + 2:.2f}">{row["rdma_ms"]:.2f} ms</text>',
            f'<text class="value" x="{center}" y="{rdma_y + rdma_height / 2 + 15:.2f}">({row["rdma_pct"]:.1f}%)</text>',
            f'<text class="inside" x="{center}" y="{cpu_y + cpu_height / 2 - 12:.2f}">CPU</text>',
            f'<text class="value" x="{center}" y="{cpu_y + cpu_height / 2 + 3:.2f}">{row["cpu_ms"]:.2f} ms</text>',
            f'<text class="value" x="{center}" y="{cpu_y + cpu_height / 2 + 16:.2f}">({row["cpu_pct"]:.1f}%)</text>',
            f'<text class="dataset" x="{center}" y="354" text-anchor="middle">{esc(row["dataset"])}</text>',
        ])

    pieces.extend([
        '<g transform="translate(232,386)">',
        '<rect x="0" y="-12" width="16" height="16" fill="#9ecae1"/>',
        '<text class="legend" x="23" y="1">RDMA wait</text>',
        '<rect x="116" y="-12" width="16" height="16" fill="#d9dde3"/>',
        '<text class="legend" x="139" y="1">CPU</text>',
        '</g>',
        '</svg>',
    ])
    return "\n".join(pieces)


def main():
    args = parse_args()
    rows = [
        load_breakdown("SIFT100M", args.sift100m),
        load_breakdown("DEEP100M", args.deep100m),
        load_breakdown("SPACEV100M", args.spacev100m),
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render(rows), encoding="utf-8")
    print(args.output.resolve())
    for row in rows:
        print(
            f'{row["dataset"]}: total={row["total_ms"]:.2f} ms, '
            f'RDMA={row["rdma_ms"]:.2f} ms ({row["rdma_pct"]:.1f}%), '
            f'CPU={row["cpu_ms"]:.2f} ms ({row["cpu_pct"]:.1f}%)')


if __name__ == "__main__":
    main()
