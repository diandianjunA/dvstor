import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# 1. Global style
# ============================================================
plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42

plt.rcParams["font.size"] = 10
plt.rcParams["axes.linewidth"] = 0.9


# ============================================================
# 2. Utilities
# ============================================================
def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def ns_to_ms_per_op(ns, count):
    return ns / count / 1e6


# ============================================================
# 3. Plot
# ============================================================
def plot_breakdown(json_path: Path, output_prefix: str):
    data = load_json(json_path)

    insert = data["insert_breakdown"]
    query = data["query_breakdown"]

    insert_count = insert["count"]
    query_count = query["count"]

    insert_gpu = insert["sub_breakdown"]["gpu_ns"]
    query_gpu = query["sub_breakdown"]["gpu_ns"]

    # ------------------------------------------------------------
    # Split Insert GPU time
    # ------------------------------------------------------------
    insert_gpu_effective_ns = (
        insert_gpu["gpu_insert_distance_ns"]
        + insert_gpu["gpu_insert_quantize_ns"]
    )

    insert_gpu_graph_ns = (
        insert_gpu["gpu_insert_overflow_distance_ns"]
        + insert_gpu["gpu_insert_prune_ns"]
        + insert_gpu["gpu_insert_overflow_prune_ns"]
    )

    insert_gpu_other_ns = (
        insert["breakdown"]["gpu_ns"]
        - insert_gpu_effective_ns
        - insert_gpu_graph_ns
    )

    # ------------------------------------------------------------
    # Split Query GPU time
    # ------------------------------------------------------------
    query_gpu_effective_ns = (
        query_gpu["gpu_query_distance_ns"]
        + query_gpu["gpu_query_rerank_ns"]
    )

    query_gpu_graph_ns = 0

    query_gpu_other_ns = (
        query["breakdown"]["gpu_ns"]
        - query_gpu_effective_ns
        - query_gpu_graph_ns
    )

    labels = ["Insert", "Query"]

    components = {
        "CPU control": [
            ns_to_ms_per_op(insert["breakdown"]["cpu_ns"], insert_count),
            ns_to_ms_per_op(query["breakdown"]["cpu_ns"], query_count),
        ],
        "RDMA access": [
            ns_to_ms_per_op(insert["breakdown"]["rdma_ns"], insert_count),
            ns_to_ms_per_op(query["breakdown"]["rdma_ns"], query_count),
        ],
        "CPU-GPU transfer": [
            ns_to_ms_per_op(insert["breakdown"]["transfer_ns"], insert_count),
            ns_to_ms_per_op(query["breakdown"]["transfer_ns"], query_count),
        ],
        "Effective GPU compute": [
            ns_to_ms_per_op(insert_gpu_effective_ns, insert_count),
            ns_to_ms_per_op(query_gpu_effective_ns, query_count),
        ],
        "GPU graph-maint. kernels": [
            ns_to_ms_per_op(insert_gpu_graph_ns, insert_count),
            ns_to_ms_per_op(query_gpu_graph_ns, query_count),
        ],
        "Other GPU overhead": [
            ns_to_ms_per_op(max(insert_gpu_other_ns, 0), insert_count),
            ns_to_ms_per_op(max(query_gpu_other_ns, 0), query_count),
        ],
    }

    # Low-saturation academic colors
    colors = {
        "CPU control": "#A9C9DD",
        "RDMA access": "#E8B77A",
        "CPU-GPU transfer": "#C9C9C9",
        "Effective GPU compute": "#9FC59D",
        "GPU graph-maint. kernels": "#D98E8E",
        "Other GPU overhead": "#B9A8C9",
    }

    insert_total_ms = insert["latency"]["mean_end_to_end_ns"] / 1e6
    query_total_ms = query["latency"]["mean_end_to_end_ns"] / 1e6

    insert_graph_ratio = insert_gpu_graph_ns / insert["breakdown"]["gpu_ns"] * 100
    insert_effective_ratio = insert_gpu_effective_ns / insert["breakdown"]["gpu_ns"] * 100

    # ============================================================
    # 4. Figure layout
    # ============================================================
    fig, ax = plt.subplots(figsize=(6.8, 4.8), dpi=300)

    # Narrower x range and thinner bars
    x = np.array([0.00, 0.42])
    width = 0.14

    bottom = np.zeros(2)

    for name, values in components.items():
        values = np.array(values)

        bars = ax.bar(
            x,
            values,
            width=width,
            bottom=bottom,
            label=name,
            color=colors[name],
            edgecolor="white",
            linewidth=0.7,
        )

        # Only annotate large Insert segments.
        # Query is too short, so segment labels are omitted to avoid overlap.
        for i, bar in enumerate(bars):
            value = values[i]
            total = insert_total_ms if i == 0 else query_total_ms

            if i == 0 and value / total >= 0.08:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bottom[i] + value / 2,
                    f"{value:.1f}",
                    ha="center",
                    va="center",
                    fontsize=8.2,
                    color="#333333",
                )

        bottom += values

    # Total latency labels
    total_offset = insert_total_ms * 0.025

    ax.text(
        x[0],
        bottom[0] + total_offset,
        f"{insert_total_ms:.1f} ms",
        ha="center",
        va="bottom",
        fontsize=10.5,
        fontweight="bold",
        color="#222222",
    )

    ax.text(
        x[1],
        bottom[1] + total_offset,
        f"{query_total_ms:.1f} ms",
        ha="center",
        va="bottom",
        fontsize=10.5,
        fontweight="bold",
        color="#222222",
    )

    # ============================================================
    # 5. Key annotations
    # ============================================================
    insert_cpu_ms = components["CPU control"][0]
    insert_rdma_ms = components["RDMA access"][0]
    insert_transfer_ms = components["CPU-GPU transfer"][0]
    insert_effective_ms = components["Effective GPU compute"][0]
    insert_graph_ms = components["GPU graph-maint. kernels"][0]

    # Graph-maintenance annotation
    ax.annotate(
        f"{insert_graph_ratio:.1f}% of Insert GPU time\nis graph-maintenance kernels",
        xy=(
            x[0],
            insert_cpu_ms
            + insert_rdma_ms
            + insert_transfer_ms
            + insert_effective_ms
            + insert_graph_ms * 0.62,
        ),
        xytext=(0.20, insert_total_ms * 0.73),
        arrowprops=dict(
            arrowstyle="->",
            lw=1.0,
            color="#8F3A3A",
            shrinkA=0,
            shrinkB=3,
        ),
        fontsize=8.9,
        color="#8F3A3A",
        ha="left",
        va="center",
    )

    # Effective GPU compute annotation
    ax.annotate(
        f"Only {insert_effective_ratio:.1f}%\neffective GPU compute",
        xy=(
            x[0],
            insert_cpu_ms
            + insert_rdma_ms
            + insert_transfer_ms
            + insert_effective_ms * 0.5,
        ),
        xytext=(0.20, insert_total_ms * 0.46),
        arrowprops=dict(
            arrowstyle="->",
            lw=1.0,
            color="#3E7048",
            shrinkA=0,
            shrinkB=3,
        ),
        fontsize=8.9,
        color="#3E7048",
        ha="left",
        va="center",
    )

    # ============================================================
    # 6. Axes and legend
    # ============================================================
    ax.set_title(
        "Insert vs. Query Breakdown",
        fontsize=13,
        fontweight="bold",
        pad=28,
    )

    ax.set_ylabel("Average time per operation (ms)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11, fontweight="bold")

    # Narrow coordinate system
    ax.set_xlim(-0.16, 0.68)
    ax.set_ylim(0, insert_total_ms * 1.18)

    ax.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.35)
    ax.set_axisbelow(True)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.legend(
        frameon=False,
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.10),
        fontsize=8.2,
        columnspacing=0.9,
        handlelength=1.1,
        handletextpad=0.4,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.94])

    png_path = f"{output_prefix}.png"
    pdf_path = f"{output_prefix}.pdf"

    plt.savefig(png_path, bbox_inches="tight")
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.show()

    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")


# ============================================================
# 7. Entry
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--json",
        type=str,
        default="../breakdown/gpucache-16g.json",
        help="Path to breakdown JSON file.",
    )

    parser.add_argument(
        "--out",
        type=str,
        default="insert_query_breakdown_stacked_bar",
        help="Output file prefix.",
    )

    args = parser.parse_args()

    plot_breakdown(Path(args.json), args.out)