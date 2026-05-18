import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# =========================
# Data
# =========================
scales = ["1M", "5M", "10M", "50M"]
methods = ["Vamana", "HNSW", "FreshDiskANN", "OdinANN", "Shine"]

data = {
    "Vamana":      [4.09, 20.75, 41.49, 219.91],
    "HNSW":         [4.19, 21.20, 42.99, 221.17],
    "FreshDiskANN": [4.27, 8.54, 9.03, 20.66],
    "OdinANN":      [6.96, np.nan, np.nan, 12],
    "Shine":        [10.00, 10.00, 50.00, 50.00],
}

colors = {
    "Vamana": "#4E79A7",
    "HNSW": "#F28E2B",
    "FreshDiskANN": "#59A14F",
    "OdinANN": "#E15759",
    "Shine": "#9C755F",
}

# =========================
# Figure settings
# =========================
plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False

x = np.arange(len(scales))
bar_width = 0.14
offsets = (np.arange(len(methods)) - (len(methods) - 1) / 2) * bar_width

lower_ylim = 65
upper_min = 205
upper_max = 232

fig, (ax_top, ax_bottom) = plt.subplots(
    2, 1,
    sharex=True,
    figsize=(11, 6.4),
    gridspec_kw={
        "height_ratios": [1.05, 2.35],
        "hspace": 0.04
    },
    dpi=150
)

# =========================
# Draw bars
# =========================
for i, method in enumerate(methods):
    values = np.array(data[method], dtype=float)
    xpos = x + offsets[i]

    for j, value in enumerate(values):
        if np.isnan(value):
            continue

        # Low range bars
        if value <= lower_ylim:
            ax_bottom.bar(
                xpos[j],
                value,
                width=bar_width * 0.9,
                color=colors[method],
                edgecolor="white",
                linewidth=1.0,
                zorder=3
            )
        else:
            # Draw lower truncated part
            ax_bottom.bar(
                xpos[j],
                lower_ylim,
                width=bar_width * 0.9,
                color=colors[method],
                edgecolor="white",
                linewidth=1.0,
                alpha=0.95,
                zorder=3
            )

            # Draw upper visible part
            ax_top.bar(
                xpos[j],
                value - upper_min,
                bottom=upper_min,
                width=bar_width * 0.9,
                color=colors[method],
                edgecolor="white",
                linewidth=1.0,
                alpha=0.95,
                zorder=3
            )

# =========================
# Value labels
# =========================
for i, method in enumerate(methods):
    values = np.array(data[method], dtype=float)
    xpos = x + offsets[i]

    for j, value in enumerate(values):
        if np.isnan(value):
            continue

        label = f"{value:.2f}".rstrip("0").rstrip(".")

        if value > upper_min:
            ax_top.text(
                xpos[j],
                value + 1.0,
                label,
                ha="center",
                va="bottom",
                fontsize=9,
                color=colors[method],
                fontweight="bold"
            )
        else:
            # 小数值标签竖着放，避免 1M 处互相挤压
            if value < 12:
                ax_bottom.text(
                    xpos[j],
                    value + 1.0,
                    label,
                    ha="center",
                    va="bottom",
                    fontsize=8.5,
                    color=colors[method],
                    fontweight="bold",
                    rotation=90
                )
            else:
                ax_bottom.text(
                    xpos[j],
                    value + 1.2,
                    label,
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    color=colors[method],
                    fontweight="bold"
                )

# =========================
# Broken axis style
# =========================
ax_bottom.set_ylim(0, lower_ylim)
ax_top.set_ylim(upper_min, upper_max)

ax_top.spines["bottom"].set_visible(False)
ax_bottom.spines["top"].set_visible(False)
ax_top.tick_params(axis="x", bottom=False, labelbottom=False)
ax_bottom.tick_params(axis="x", top=False)

# 断轴斜线
d = 0.008
kwargs = dict(color="black", clip_on=False, linewidth=1.2)

ax_top.plot((-d, +d), (-d, +d), transform=ax_top.transAxes, **kwargs)
ax_top.plot((1 - d, 1 + d), (-d, +d), transform=ax_top.transAxes, **kwargs)

ax_bottom.plot((-d, +d), (1 - d, 1 + d), transform=ax_bottom.transAxes, **kwargs)
ax_bottom.plot((1 - d, 1 + d), (1 - d, 1 + d), transform=ax_bottom.transAxes, **kwargs)

# =========================
# Axes, title, legend
# =========================
ax_bottom.set_xticks(x)
ax_bottom.set_xticklabels(scales, fontsize=12)

ax_bottom.set_xlabel("Dataset Scale", fontsize=13, fontweight="bold")

fig.text(
    0.04,
    0.5,
    "Memory Usage / Storage Cost (GB)",
    va="center",
    rotation="vertical",
    fontsize=13,
    fontweight="bold"
)

fig.suptitle(
    "Memory Usage of Vector Indexes at Different Dataset Scales",
    fontsize=17,
    fontweight="bold",
    y=0.98
)

legend_handles = [
    Patch(facecolor=colors[m], edgecolor="white", label=m)
    for m in methods
]

fig.legend(
    handles=legend_handles,
    loc="upper center",
    bbox_to_anchor=(0.5, 0.92),
    ncol=5,
    frameon=False,
    fontsize=11
)

# =========================
# Grid and style
# =========================
for ax in [ax_top, ax_bottom]:
    ax.grid(axis="y", linestyle="--", linewidth=0.8, alpha=0.28, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", labelsize=11)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

ax_top.set_yticks([210, 220, 230])
ax_bottom.set_yticks(np.arange(0, 70, 10))

# 脚注放在图外，不挡数据
fig.text(
    0.5,
    0.015,
    "Note: Blank bars indicate unavailable measurements.",
    ha="center",
    va="bottom",
    fontsize=9,
    color="gray"
)

plt.tight_layout(rect=[0.06, 0.05, 1, 0.90])

plt.savefig("vector_index_memory_grouped_bar_clean.png", dpi=300, bbox_inches="tight")
plt.savefig("vector_index_memory_grouped_bar_clean.pdf", bbox_inches="tight")
plt.show()