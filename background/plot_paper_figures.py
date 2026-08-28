#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
为论文背景实验绘制两张“更直观”的图。

这一版重点修改了动态更新图：
- 不再使用吞吐损失率等派生概念；
- 直接展示“绝对查询吞吐（KQPS）”；
- 使用 2x2 small multiples（每个系统一个小面板）；
- x 轴从左到右依次为 10:0 -> 1:9，使“写压力增加时，吞吐下降”一眼可见。

运行：
    python plot_paper_figures.py \
        --memory-csv "内存占用变化.csv" \
        --rw-csv "动态更新读写比例变化.csv" \
        --output-dir "paper_figures"
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": [
                "Times New Roman",
                "Times",
                "Nimbus Roman",
                "Liberation Serif",
                "DejaVu Serif",
            ],
            "font.size": 9,
            "axes.labelsize": 10,
            "axes.linewidth": 0.8,
            "legend.fontsize": 8.3,
            "xtick.labelsize": 8.2,
            "ytick.labelsize": 8.2,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.major.size": 3.5,
            "ytick.major.size": 3.5,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "lines.linewidth": 2.0,
            "lines.markersize": 5.2,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def read_csv_robust(path: Path) -> pd.DataFrame:
    errors = []
    for encoding in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError as exc:
            errors.append(f"{encoding}: {exc}")
    raise UnicodeError(f"无法识别 CSV 编码：{path}\n" + "\n".join(errors))


def prettify_memory_series(name: str) -> str:
    normalized = re.sub(r"[\s_\-]+", "", str(name)).lower()
    if "indexfile" in normalized:
        return "Index file"
    if "indexmemory" in normalized:
        return "In-memory index"
    return str(name)


def style_axis(ax: plt.Axes, *, x_grid: bool = False) -> None:
    ax.tick_params(which="both", top=True, right=True)
    ax.set_axisbelow(True)
    ax.grid(
        True,
        which="major",
        axis="both" if x_grid else "y",
        linestyle="--",
        linewidth=0.55,
        alpha=0.30,
    )
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)


def save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.035)
    fig.savefig(output_dir / f"{stem}.svg", bbox_inches="tight", pad_inches=0.035)
    fig.savefig(output_dir / f"{stem}.png", dpi=600, bbox_inches="tight", pad_inches=0.035)


def draw_break_marks(ax_top: plt.Axes, ax_bottom: plt.Axes) -> None:
    diagonal = 0.010
    neutral = ax_top.spines["left"].get_edgecolor()
    kwargs_top = dict(transform=ax_top.transAxes, clip_on=False, linewidth=0.9, color=neutral)
    kwargs_bottom = dict(transform=ax_bottom.transAxes, clip_on=False, linewidth=0.9, color=neutral)

    ax_top.plot((-diagonal, +diagonal), (-diagonal, +diagonal), **kwargs_top)
    ax_top.plot((1 - diagonal, 1 + diagonal), (-diagonal, +diagonal), **kwargs_top)
    ax_bottom.plot((-diagonal, +diagonal), (1 - diagonal, 1 + diagonal), **kwargs_bottom)
    ax_bottom.plot((1 - diagonal, 1 + diagonal), (1 - diagonal, 1 + diagonal), **kwargs_bottom)


def plot_memory_footprint(csv_path: Path, output_dir: Path) -> None:
    df = read_csv_robust(csv_path)
    series_col = df.columns[0]
    dataset_labels = [str(column).strip() for column in df.columns[1:]]
    x = np.arange(len(dataset_labels), dtype=float)

    series = []
    for _, row in df.iterrows():
        values = pd.to_numeric(row.iloc[1:], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"内存占用数据包含非数值项：{row[series_col]}")
        series.append((prettify_memory_series(row[series_col]), values))

    fig, (ax_top, ax_bottom) = plt.subplots(
        2,
        1,
        sharex=True,
        figsize=(6.9, 4.55),
        gridspec_kw={"height_ratios": [1.0, 1.65], "hspace": 0.055},
    )
    fig.subplots_adjust(left=0.105, right=0.985, bottom=0.13, top=0.965)

    width = 0.34
    bar_edge = ax_top.spines["left"].get_edgecolor()
    top_bars = []
    bottom_bars = []

    for idx, (label, values) in enumerate(series):
        offset = (idx - (len(series) - 1) / 2.0) * width

        bars_top = ax_top.bar(
            x + offset,
            values,
            width=width,
            label=label,
            edgecolor=bar_edge,
            linewidth=0.45,
            zorder=3,
        )
        bars_bottom = ax_bottom.bar(
            x + offset,
            values,
            width=width,
            label=label,
            edgecolor=bar_edge,
            linewidth=0.45,
            zorder=3,
        )
        top_bars.append((bars_top, values))
        bottom_bars.append((bars_bottom, values))

    ax_bottom.set_ylim(0, 55)
    ax_top.set_ylim(345, 500)

    ax_top.spines["bottom"].set_visible(False)
    ax_bottom.spines["top"].set_visible(False)
    ax_top.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
    ax_bottom.tick_params(axis="x", top=False)
    draw_break_marks(ax_top, ax_bottom)

    ax_bottom.set_xticks(x)
    ax_bottom.set_xticklabels(dataset_labels)
    ax_bottom.set_xlabel("Dataset size (number of vectors)")
    fig.supylabel("HNSW index footprint (GB)", x=0.018, fontsize=10)

    style_axis(ax_top)
    style_axis(ax_bottom)

    ax_top.legend(loc="upper left", ncol=2, columnspacing=1.2, handlelength=1.8)

    index_100m = dataset_labels.index("100M") if "100M" in dataset_labels else None
    index_1b = dataset_labels.index("1B") if "1B" in dataset_labels else len(x) - 1

    for bars, values in bottom_bars:
        if index_100m is not None:
            bar = bars[index_100m]
            ax_bottom.annotate(
                f"{values[index_100m]:.1f}",
                xy=(bar.get_x() + bar.get_width() / 2, values[index_100m]),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=7.8,
            )

    for bars, values in top_bars:
        bar = bars[index_1b]
        ax_top.annotate(
            f"{values[index_1b]:.1f} GB",
            xy=(bar.get_x() + bar.get_width() / 2, values[index_1b]),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8.0,
        )

    growth_factors = [values[index_1b] / values[0] for _, values in series if values[0] > 0]
    mean_growth = np.mean(growth_factors)
    one_b_values = [values[index_1b] for _, values in series]
    ax_top.text(
        x[index_1b] - 2.45,
        432,
        (
            f""
        ),
        ha="left",
        va="center",
        fontsize=8.5,
    )

    save_figure(fig, output_dir, "fig_memory_footprint_emphasized")
    plt.close(fig)


def ratio_label(read_fraction: float) -> str:
    read_part = int(round(read_fraction * 10))
    return f"{read_part}:{10 - read_part}"


def format_kqps(v: float) -> str:
    return f"{v/1000.0:.1f}K"


def plot_query_throughput_panels(csv_path: Path, output_dir: Path) -> None:
    df = read_csv_robust(csv_path)
    system_col = df.columns[0]
    ratio_columns = list(df.columns[1:])

    try:
        read_fractions = np.asarray([float(str(col).strip()) for col in ratio_columns], dtype=float)
    except ValueError as exc:
        raise ValueError("读写比例列名应为 0.1、0.2、...、1。") from exc

    order = np.argsort(read_fractions)[::-1]  # 10:0 -> 1:9
    read_fractions = read_fractions[order]
    ratio_columns = [ratio_columns[idx] for idx in order]
    x = np.arange(len(ratio_columns), dtype=float)
    xlabels = [ratio_label(v) for v in read_fractions]

    systems = []
    for _, row in df.iterrows():
        values = pd.to_numeric(row[ratio_columns], errors="coerce").to_numpy(dtype=float)
        systems.append((str(row[system_col]), values))

    fig, axes = plt.subplots(2, 2, figsize=(7.15, 4.85), sharex=True)
    fig.subplots_adjust(left=0.10, right=0.985, bottom=0.12, top=0.87, wspace=0.24, hspace=0.30)
    axes = axes.ravel()

    global_edge = axes[0].spines["left"].get_edgecolor()

    for ax, (system_name, qps) in zip(axes, systems):
        qps_k = qps / 1000.0
        valid = np.isfinite(qps_k)

        line = ax.plot(
            x[valid],
            qps_k[valid],
            marker="o",
            zorder=4,
            label=system_name,
        )[0]

        ax.fill_between(
            x[valid],
            qps_k[valid],
            np.zeros(valid.sum()),
            alpha=0.18,
            zorder=2,
            color=line.get_color(),
        )

        ax.bar(
            x[valid],
            qps_k[valid],
            width=0.62,
            alpha=0.10,
            edgecolor=global_edge,
            linewidth=0.35,
            color=line.get_color(),
            zorder=1,
        )

        style_axis(ax, x_grid=False)
        ax.set_title(system_name, fontsize=9.3, pad=3)

        invalid = np.flatnonzero(~valid)
        if invalid.size > 0:
            start = invalid.min() - 0.5
            end = invalid.max() + 0.5
            ax.axvspan(start, end, alpha=0.10, zorder=0)
            ax.text(
                (start + end) / 2,
                0.83,
                "unsupported",
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="center",
                fontsize=7.8,
            )

        if valid.any():
            valid_idx = np.flatnonzero(valid)
            first = valid_idx[0]
            last = valid_idx[-1]

            ax.annotate(
                format_kqps(qps[first]),
                xy=(x[first], qps_k[first]),
                xytext=(0, 5),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=7.7,
                color=line.get_color(),
            )
            ax.annotate(
                format_kqps(qps[last]),
                xy=(x[last], qps_k[last]),
                xytext=(0, 5),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=7.7,
                color=line.get_color(),
            )

            if valid_idx.size >= 2:
                summary = f""
                ax.text(
                    0.03,
                    0.94,
                    summary,
                    transform=ax.transAxes,
                    ha="left",
                    va="top",
                    fontsize=7.8,
                )

        local_valid = qps_k[valid]
        if local_valid.size > 0:
            ymax = float(local_valid.max())
            ax.set_ylim(0.0, ymax * 1.24 + 0.02)

    for idx, ax in enumerate(axes):
        ax.set_xticks(x)
        ax.set_xticklabels(xlabels, rotation=0)
        if idx < 2:
            ax.tick_params(labelbottom=False)

    fig.supylabel("Query throughput (KQPS)", x=0.018, fontsize=10)
    fig.supxlabel("Read:write ratio", y=0.04, fontsize=10)
    fig.text(0.50, 0.915, "Background write pressure increases \u2192", ha="center", va="center", fontsize=9.0)

    save_figure(fig, output_dir, "fig_query_throughput_panels")
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="绘制更直观的论文背景实验图。")
    parser.add_argument("--memory-csv", type=Path, default=Path("内存占用变化.csv"))
    parser.add_argument("--rw-csv", type=Path, default=Path("动态更新读写比例变化.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("paper_figures"))
    return parser


def main() -> None:
    args = build_parser().parse_args()

    if not args.memory_csv.is_file():
        raise FileNotFoundError(f"找不到内存占用 CSV：{args.memory_csv}")
    if not args.rw_csv.is_file():
        raise FileNotFoundError(f"找不到读写比例 CSV：{args.rw_csv}")

    configure_matplotlib()
    plot_memory_footprint(args.memory_csv, args.output_dir)
    plot_query_throughput_panels(args.rw_csv, args.output_dir)

    print(f"New figures saved to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
