import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullFormatter

# =========================
# Data
# =========================
read_ratio = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

data = {
    "HNSW":         [63.95, 140.28, 138.81, 336.30, 545.05, 690.38, 936.22, 1313.64, 2135.83, 3478.39],
    "FreshDiskANN": [82.00, 158.36, 206.07, 259.96, 285.76, 308.51, 317.76, 332.12, 344.41, 367.57],
    "OdinANN":      [13.52, 59.40, 175.67, 232.92, 313.47, 389.52, 611.66, 1045.93, 3152.70, 3709.98],
    "shine":        [18.98, 39.27, 59.91, 86.75, 140.82, 156.49, 292.85, 350.22, 497.27, 262.13],
    "shine_gpu":    [0.73, 1.42, 15.18, 22.58, 33.63, 49.83, 59.73, 115.75, 180.07, 190.52],
}

# =========================
# Style
# =========================
plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
plt.rcParams["axes.unicode_minus"] = False

colors = {
    "Vamana": "#1f77b4",
    "HNSW": "#ff7f0e",
    "FreshDiskANN": "#2ca02c",
    "OdinANN": "#d62728",
    "shine": "#9467bd",
    "shine_gpu": "#8c564b",
}

markers = {
    "Vamana": "o",
    "HNSW": "s",
    "FreshDiskANN": "^",
    "OdinANN": "D",
    "shine": "P",
    "shine_gpu": "X",
}

line_styles = {
    "Vamana": "-",
    "HNSW": "-",
    "FreshDiskANN": "-",
    "OdinANN": "-",
    "shine": "--",
    "shine_gpu": "--",
}

# =========================
# Plot
# =========================
fig, ax = plt.subplots(figsize=(10.8, 5.8), dpi=300)

for name, values in data.items():
    values = np.array(values, dtype=float)
    ax.plot(
        read_ratio,
        values,
        label=name,
        color=colors[name],
        marker=markers[name],
        linestyle=line_styles[name],
        linewidth=2.2,
        markersize=6,
        markerfacecolor="white",
        markeredgewidth=1.3
    )

# =========================
# Axes
# =========================
ax.set_yscale("log")
ax.set_ylim(0.5, 6000)
ax.set_xlim(0.08, 1.02)

ax.set_xlabel("Read Ratio", fontsize=13, fontweight="bold")
ax.set_ylabel("Throughput (ops/s, log scale)", fontsize=13, fontweight="bold")

ax.set_title(
    "Mixed Read/Write Throughput under Varying Read Ratios\n"
    "(1024-D, 50M Vectors, 16 Threads)",
    fontsize=15,
    fontweight="bold",
    pad=14
)

ax.set_xticks(read_ratio)
ax.set_xticklabels([f"{x:.1f}" for x in read_ratio], fontsize=11)

ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=10))
ax.yaxis.set_minor_formatter(NullFormatter())
ax.tick_params(axis="y", labelsize=11)

# 网格和边框
ax.grid(True, which="major", linestyle="--", linewidth=0.8, alpha=0.30)
ax.grid(True, which="minor", linestyle=":", linewidth=0.5, alpha=0.18)
ax.set_axisbelow(True)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# =========================
# Legend on the far right
# =========================
legend = ax.legend(
    loc="center left",
    bbox_to_anchor=(1.02, 0.5),
    frameon=False,
    fontsize=11.5,
    ncol=1,
    handlelength=2.6,
    borderaxespad=0.0
)

# 可选：让图例线条更清晰
for line in legend.get_lines():
    line.set_linewidth(2.4)

# =========================
# Footnote
# =========================
fig.text(
    0.42,
    0.02,
    "Higher read ratio indicates a more read-dominant workload.",
    ha="center",
    fontsize=9.5,
    color="gray"
)

# 为右侧图例预留空间
plt.tight_layout(rect=[0, 0.04, 0.82, 1])

# Save
plt.savefig("mixed_workload_throughput_log_legend_right.png", dpi=300, bbox_inches="tight")
plt.savefig("mixed_workload_throughput_log_legend_right.pdf", bbox_inches="tight")
plt.savefig("mixed_workload_throughput_log_legend_right.svg", bbox_inches="tight")

plt.show()