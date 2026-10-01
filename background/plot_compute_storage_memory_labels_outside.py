#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
import io
import copy
import xml.etree.ElementTree as ET

import fitz  # pip install pymupdf
from pathlib import Path
from typing import Dict, List

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter, MaxNLocator

COLORS = {
    'Milvus-HNSW': '#4C78A8',
    'Milvus-DiskANN': '#F58518',
    'SHINE': '#54A24B',
    'Baseline': '#B279A2',
}
HATCHES = {
    'Milvus-HNSW': '',
    'Milvus-DiskANN': '//',
    'SHINE': '..',
    'Baseline': '\\\\',
}
ORDER = ['Milvus-HNSW', 'Milvus-DiskANN', 'SHINE', 'Baseline']


def try_read_text(path: Path) -> str:
    for enc in ('utf-8-sig', 'gb18030', 'utf-8'):
        try:
            return path.read_text(encoding=enc)
        except UnicodeDecodeError:
            continue
    raise ValueError(f'Cannot decode file: {path}')


def is_blank(row: List[str]) -> bool:
    return not row or not any(str(v).strip() for v in row)


def parse_value(text: str) -> float:
    s = str(text).strip().replace(',', '')
    if not s:
        raise ValueError('Empty numeric field encountered.')
    return float(s)


def parse_csv(path: Path) -> Dict[str, Dict[str, Dict[str, float]]]:
    rows = list(csv.reader(io.StringIO(try_read_text(path))))
    data: Dict[str, Dict[str, Dict[str, float]]] = {}
    i = 0
    while i < len(rows):
        row = rows[i]
        if is_blank(row):
            i += 1
            continue
        first = str(row[0]).strip()
        if len(row) >= 3 and ('compute' in str(row[1]).lower()) and ('storage' in str(row[2]).lower()):
            dataset = first
            data[dataset] = {}
            i += 1
            while i < len(rows) and not is_blank(rows[i]):
                r = rows[i]
                if len(r) < 3 or not str(r[0]).strip():
                    break
                system = str(r[0]).strip()
                data[dataset][system] = {
                    'compute': parse_value(r[1]),
                    'storage': parse_value(r[2]),
                }
                i += 1
            continue
        i += 1
    if not data:
        raise ValueError('No dataset blocks found in CSV.')
    return data


def apply_style(font_size: float = 9.6):
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif', 'Liberation Serif'],
        'font.size': font_size,
        'axes.labelsize': font_size + 0.8,
        'axes.titlesize': font_size + 1.2,
        'legend.fontsize': font_size + 0.2,
        'xtick.labelsize': font_size + 0.2,
        'ytick.labelsize': font_size + 0.2,
        'axes.linewidth': 0.9,
        'axes.edgecolor': '#333333',
        'grid.linewidth': 0.6,
        'grid.alpha': 0.22,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'svg.fonttype': 'none',
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'savefig.facecolor': 'white',
        'hatch.linewidth': 0.9,
    })


def format_tick(v: float, _pos: float) -> str:
    if v >= 1000:
        return f'{v/1000:.1f}k' if abs(v % 1000) > 1e-9 else f'{int(round(v/1000))}k'
    if v >= 100:
        return f'{v:.0f}'
    if v >= 10:
        return f'{v:.0f}'
    if v >= 1:
        return f'{v:.1f}' if abs(v-round(v)) > 1e-9 else f'{int(round(v))}'
    return f'{v:.2f}'


def format_label(v: float) -> str:
    if v >= 100:
        return f'{v:.1f}'
    if v >= 10:
        return f'{v:.1f}'
    return f'{v:.2f}'


def plot_memory(data: Dict[str, Dict[str, Dict[str, float]]], output_prefix: Path,
                width: float = 8.1, height: float = 3.65, font_size: float = 9.6,
                compute_log: bool = True, annotate: bool = True, dpi: int = 600):
    apply_style(font_size)

    datasets = list(data.keys())
    systems = [s for s in ORDER if any(s in data[d] for d in datasets)]
    if not systems:
        systems = []
        for d in datasets:
            for s in data[d]:
                if s not in systems:
                    systems.append(s)

    group_gap = 1.15
    x = np.arange(len(datasets), dtype=float) * group_gap
    n_sys = len(systems)
    bar_width = 0.22 if n_sys <= 2 else 0.68 / n_sys
    offsets = (np.arange(n_sys) - (n_sys - 1) / 2.0) * (bar_width * 1.10)

    # Render one chart per figure; compose the two vector panels at export.
    # These positions reproduce the previous equal-size panel layout.
    panel_width = (0.985 - 0.085) / (2 + 0.19)
    panel_lefts = [0.085, 0.085 + panel_width * 1.19]
    figures = []

    panel_specs = [
        ('compute', 'Compute node memory (GB)', '(a)'),
        ('storage', 'Storage node memory (GB)', '(b)'),
    ]

    legend_handles = []

    for col, (metric, title, panel_tag) in enumerate(panel_specs):
        fig = plt.figure(figsize=(width, height))
        ax = fig.add_axes([panel_lefts[col], 0.20, panel_width, 0.60])
        figures.append(fig)
        all_vals = []
        for idx, system in enumerate(systems):
            vals = [data[ds][system][metric] for ds in datasets]
            xpos = x + offsets[idx]
            bars = ax.bar(
                xpos,
                vals,
                width=bar_width,
                color=COLORS.get(system, '#999999'),
                edgecolor='#333333',
                linewidth=0.8,
                hatch=HATCHES.get(system, ''),
                zorder=3,
                label=system,
            )
            if len(legend_handles) < len(systems):
                legend_handles.append(bars[0])
            all_vals.extend(vals)
            if annotate:
                for b, v in zip(bars, vals):
                    if metric == 'compute' and compute_log:
                        y = v * 1.10
                    else:
                        y = v + max(all_vals) * 0.025
                    ax.text(
                        b.get_x() + b.get_width() / 2,
                        y,
                        format_label(v),
                        ha='center', va='bottom',
                        fontsize=font_size - 0.8,
                    )

        # Panel labels belong to the title, outside the plotting rectangle.
        ax.set_title(f'{panel_tag} {title}', pad=8)
        ax.set_xticks(x)
        ax.set_xticklabels(datasets)
        ax.grid(axis='y', linestyle='--', color='#BFBFBF')
        ax.set_axisbelow(True)
        ax.tick_params(direction='out', length=3.2, width=0.85)
        for spine in ax.spines.values():
            spine.set_linewidth(0.9)
        ax.yaxis.set_major_formatter(FuncFormatter(format_tick))
        ax.margins(x=0.05)

        if metric == 'compute' and compute_log:
            ax.set_yscale('log')
            positive_min = min(v for v in all_vals if v > 0)
            positive_max = max(all_vals)
            ax.set_ylim(max(0.35, positive_min / 1.8), positive_max * 1.75)
            ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
            ax.yaxis.set_minor_formatter(NullFormatter())
        else:
            ymax = max(all_vals)
            ax.set_ylim(0, ymax * 1.25)
            ax.yaxis.set_major_locator(MaxNLocator(nbins=6, steps=[1, 2, 2.5, 5, 10]))

    figures[0].legend(
        legend_handles,
        systems,
        loc='upper center',
        ncol=len(systems),
        frameon=False,
        bbox_to_anchor=(0.5, 0.98),
        handlelength=1.8,
        columnspacing=1.6,
        handletextpad=0.6,
    )

    output_prefix = Path(output_prefix)
    if output_prefix.suffix.lower() in {'.pdf', '.svg', '.png'}:
        output_prefix = output_prefix.with_suffix('')
    return export_panels(figures, output_prefix, dpi)



def export_panels(figures, output_prefix: Path, dpi: int):
    """Combine independent charts, preserving vector PDF and editable SVG text."""
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    pdf = Path(str(output_prefix) + '.pdf')
    svg = Path(str(output_prefix) + '.svg')
    png = Path(str(output_prefix) + '.png')
    preview = Path(str(output_prefix) + '_preview.png')

    # A common crop preserves exact panel alignment and equal dimensions.
    extents = []
    for fig in figures:
        fig.canvas.draw()
        extents.append(fig.get_tightbbox(fig.canvas.get_renderer()))
    common_bbox = Bbox.union(extents).padded(0.10)
    pdf_layers, svg_layers = [], []
    for fig in figures:
        for fmt, layers in [('pdf', pdf_layers), ('svg', svg_layers)]:
            stream = io.BytesIO()
            fig.savefig(stream, format=fmt, bbox_inches=common_bbox,
                        pad_inches=0, transparent=True)
            layers.append(stream.getvalue())
        plt.close(fig)

    with fitz.open() as result:
        page = result.new_page(width=common_bbox.width * 72,
                               height=common_bbox.height * 72)
        for layer in pdf_layers:
            with fitz.open(stream=layer, filetype='pdf') as source:
                page.show_pdf_page(page.rect, source, 0, keep_proportion=False)
        result.set_metadata({
            'title': 'Compute and storage memory usage',
            'subject': 'Panel labels outside the plot frames; original measurements',
            'creator': 'plot_compute_storage_memory_labels_outside.py',
        })
        result.save(str(pdf), garbage=4, deflate=True)
        page.get_pixmap(dpi=dpi, alpha=False).save(str(png))
        page.get_pixmap(dpi=220, alpha=False).save(str(preview))

    # Prefix SVG element identifiers so each panel keeps its own hatch/clip IDs.
    ns = 'http://www.w3.org/2000/svg'
    ET.register_namespace('', ns)
    ET.register_namespace('xlink', 'http://www.w3.org/1999/xlink')
    root_out = None
    for i, layer in enumerate(svg_layers):
        root = ET.fromstring(layer)
        id_map = {node.attrib['id']: f'panel{i}_' + node.attrib['id']
                  for node in root.iter() if 'id' in node.attrib}
        for node in root.iter():
            for key, value in list(node.attrib.items()):
                if key == 'id':
                    value = id_map[value]
                else:
                    for old, new in id_map.items():
                        value = value.replace('url(#' + old + ')', 'url(#' + new + ')')
                        if value == '#' + old:
                            value = '#' + new
                node.set(key, value)
        if root_out is None:
            root_out = ET.Element(root.tag, dict(root.attrib))
            ET.SubElement(root_out, '{' + ns + '}rect', {
                'x': '0', 'y': '0', 'width': '100%', 'height': '100%',
                'fill': 'white',
            })
        for child in root:
            root_out.append(copy.deepcopy(child))
    ET.ElementTree(root_out).write(svg, encoding='utf-8', xml_declaration=True)
    return [pdf, svg, png, preview]


def main():
    parser = argparse.ArgumentParser(description='Plot compute/storage memory usage as grouped bar charts.')
    parser.add_argument('csv_path', type=Path)
    parser.add_argument('--output', type=Path, default=Path('compute_storage_memory_labels_outside'))
    parser.add_argument('--width', type=float, default=8.1)
    parser.add_argument('--height', type=float, default=3.65)
    parser.add_argument('--font-size', type=float, default=9.6)
    parser.add_argument('--linear-compute', action='store_true')
    parser.add_argument('--hide-annotations', action='store_true')
    parser.add_argument('--dpi', type=int, default=600)
    args = parser.parse_args()

    outputs = plot_memory(
        parse_csv(args.csv_path),
        args.output,
        width=args.width,
        height=args.height,
        font_size=args.font_size,
        compute_log=(not args.linear_compute),
        annotate=(not args.hide_annotations),
        dpi=args.dpi,
    )
    print('Saved:')
    for p in outputs:
        print(f'  {p}')


if __name__ == '__main__':
    main()
