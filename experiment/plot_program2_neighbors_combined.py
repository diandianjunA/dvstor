#!/usr/bin/env python3
"""Plot Program 2 valid-neighbor ECDFs for SIFT, DEEP, and SPACEV."""

import argparse
import html
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RUNS = {
    "SIFT100M": REPO_ROOT / "experiment/sift100m/program2/results/program2_20260831_190527",
    "DEEP100M": REPO_ROOT / "experiment/deep100m/program2/results/program2_20260831_195403",
    "SPACEV100M": REPO_ROOT / "experiment/spacev100m/program2/results/program2_20260831_193042",
}
STYLES = {
    "SIFT100M": ("#0072B2", ""),
    "DEEP100M": ("#D55E00", "9 4"),
    "SPACEV100M": ("#009E73", "2 3"),
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    for dataset, default in DEFAULT_RUNS.items():
        parser.add_argument(
            f"--{dataset.lower()}", type=Path, default=default,
            help=f"{dataset} Program 2 run directory (default: {default})")
    parser.add_argument(
        "--output", type=Path,
        default=REPO_ROOT / "experiment/paper_figures/program2_neighbor_distribution_three_datasets.svg")
    return parser.parse_args()


def load_data(dataset, run_root):
    source = run_root.resolve() / "summary.json"
    if not source.is_file():
        raise SystemExit(f"missing {dataset} summary: {source}")
    report = json.loads(source.read_text(encoding="utf-8"))
    histogram = report["dynamic_degree_histogram"]
    quantum = report["dynamic_degree_histogram_quantum"]
    total = sum(histogram)
    if total <= 0:
        raise SystemExit(f"empty degree histogram: {source}")
    cumulative = 0
    points = []
    for extent_class, count in enumerate(histogram):
        cumulative += count
        points.append((extent_class * quantum, 100.0 * cumulative / total))
    return {
        "dataset": dataset,
        "mean": report["average_dynamic_degree"],
        "points": points,
        "source": source,
    }


def esc(value):
    return html.escape(str(value))


def smooth_monotone(points, samples_per_interval=12):
    """Sample a monotone cubic Hermite curve without overshooting the CDF."""
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    slopes = [(ys[i + 1] - ys[i]) / (xs[i + 1] - xs[i])
              for i in range(len(points) - 1)]
    tangents = [slopes[0]]
    for left_slope, right_slope in zip(slopes, slopes[1:]):
        if left_slope <= 0 or right_slope <= 0:
            tangents.append(0.0)
        else:
            tangents.append(2.0 * left_slope * right_slope /
                            (left_slope + right_slope))
    tangents.append(slopes[-1])

    sampled = []
    for i in range(len(points) - 1):
        x0, x1 = xs[i], xs[i + 1]
        y0, y1 = ys[i], ys[i + 1]
        width = x1 - x0
        for sample in range(samples_per_interval):
            t = sample / samples_per_interval
            h00 = 2 * t**3 - 3 * t**2 + 1
            h10 = t**3 - 2 * t**2 + t
            h01 = -2 * t**3 + 3 * t**2
            h11 = t**3 - t**2
            sampled.append((
                x0 + width * t,
                h00 * y0 + h10 * width * tangents[i] +
                h01 * y1 + h11 * width * tangents[i + 1],
            ))
    sampled.append(points[-1])
    return sampled


def render(rows):
    width, height = 650, 360
    left, right, top, bottom = 68, 625, 54, 292
    plot_width, plot_height = right - left, bottom - top
    x_max = 104.0

    def sx(value):
        return left + plot_width * value / x_max

    def sy(value):
        return bottom - plot_height * value / 100.0

    pieces = [f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<style>
  text {{ font-family: Arial, sans-serif; fill: #202124; }}
  .panel-title {{ font-size: 14px; font-weight: 700; fill: #202124; }}
  .tick {{ font-size: 10px; fill: #4b5563; }}
  .axis-title {{ font-size: 12px; fill: #30343b; }}
  .legend {{ font-size: 12px; fill: #202124; }}
  .grid {{ stroke: #e1e5ea; stroke-width: 1; }}
  .axis {{ stroke: #4b4f56; stroke-width: 1.15; }}
</style>
<rect width="100%" height="100%" fill="white"/>''']

    pieces.append(
        f'<text class="panel-title" x="{(left + right) / 2:.1f}" y="25" text-anchor="middle">Distribution of valid neighbor counts</text>')

    for tick in [0, 25, 50, 75, 100]:
        y = sy(tick)
        pieces.extend([
            f'<line class="grid" x1="{left}" y1="{y:.2f}" x2="{right}" y2="{y:.2f}"/>',
            f'<text class="tick" x="{left - 9}" y="{y + 4:.2f}" text-anchor="end">{tick}</text>',
        ])
    for tick in [0, 20, 40, 60, 80, 100]:
        x = sx(tick)
        pieces.extend([
            f'<line class="grid" x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{bottom}"/>',
            f'<text class="tick" x="{x:.2f}" y="309" text-anchor="middle">{tick}</text>',
        ])
    pieces.extend([
        f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{bottom}"/>',
        f'<line class="axis" x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}"/>',
        f'<text class="axis-title" x="{(left + right) / 2:.1f}" y="338" text-anchor="middle">Number of valid neighbors</text>',
        f'<text class="axis-title" x="0" y="0" text-anchor="middle" transform="translate(22 {(top + bottom) / 2:.1f}) rotate(-90)">Cumulative query accesses (%)</text>',
    ])

    # Smooth monotone CDFs retain the measured bin values without overshoot.
    for row in rows:
        color, dash = STYLES[row["dataset"]]
        coords = [(sx(min(degree, x_max)), sy(cumulative))
                  for degree, cumulative in smooth_monotone(row["points"])]
        path = " ".join(
            ("M" if index == 0 else "L") + f" {x:.2f} {y:.2f}"
            for index, (x, y) in enumerate(coords))
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        pieces.append(
            f'<path d="{path}" fill="none" stroke="{color}" stroke-width="2.6"{dash_attr} stroke-linejoin="round"/>')

    legend_x, legend_y, legend_w, legend_h = 86, 68, 190, 75
    pieces.append(
        f'<rect x="{legend_x}" y="{legend_y}" width="{legend_w}" height="{legend_h}" rx="3" fill="white" fill-opacity="0.94" stroke="#cfd5dc"/>')
    for index, row in enumerate(rows):
        color, dash = STYLES[row["dataset"]]
        y = legend_y + 20 + index * 21
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        pieces.extend([
            f'<line x1="{legend_x + 12}" y1="{y - 4}" x2="{legend_x + 42}" y2="{y - 4}" stroke="{color}" stroke-width="2.6"{dash_attr}/>',
            f'<text class="legend" x="{legend_x + 50}" y="{y}">{esc(row["dataset"])}  (mean {row["mean"]:.2f})</text>',
        ])
    pieces.append('</svg>')
    return "\n".join(pieces)


def main():
    args = parse_args()
    rows = [
        load_data("SIFT100M", args.sift100m),
        load_data("DEEP100M", args.deep100m),
        load_data("SPACEV100M", args.spacev100m),
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render(rows), encoding="utf-8")
    print(args.output.resolve())
    for row in rows:
        print(f'{row["dataset"]}: mean={row["mean"]:.2f}, source={row["source"]}')


if __name__ == "__main__":
    main()
