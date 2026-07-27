#!/usr/bin/env python3
"""Summarize the one-shot live-extent byte roofline across concurrency.

The output is deliberately a *bandwidth-only upper bound*, not a performance
prediction.  It holds each run's observed RDMA byte rate fixed, replaces only
the fixed-record graph payload by the measured extent payload ratio, and gives
the QPS that the same byte rate could carry if bytes were the sole bottleneck.
It does not claim that the NIC is saturated or that compute, WQE, validation,
queueing, and tail-latency costs disappear.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys
from typing import Any


CONCURRENCY_PATTERN = re.compile(r"(?:^|/)concurrency_(\d+)(?:/|$)")


def ratio(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator else None


def pct(value: float | None) -> str:
    return "n/a" if value is None else f"{100.0 * value:.2f}%"


def number(value: float | None, digits: int = 3) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def load_json(path: pathlib.Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as source:
        value = json.load(source)
    if not isinstance(value, dict):
        raise ValueError(f"{path}: top-level JSON must be an object")
    return value


def load_extent(path: pathlib.Path) -> dict[str, float]:
    report = load_json(path)
    model = report.get("extent_8")
    assumptions = report.get("layout_assumptions")
    if not isinstance(model, dict) or not isinstance(assumptions, dict):
        raise ValueError(f"{path}: not a live-extent oracle report")
    payload_ratio = float(model.get("payload_ratio", 0) or 0)
    record_bytes = float(assumptions.get("fixed_record_bytes", 0) or 0)
    extent_edges = float(assumptions.get("extent_edges", 0) or 0)
    if not 0 < payload_ratio <= 1:
        raise ValueError(f"{path}: invalid extent payload ratio")
    if record_bytes <= 0 or extent_edges != 8:
        raise ValueError(f"{path}: invalid fixed-record/extent assumptions")
    return {
        "payload_ratio": payload_ratio,
        "record_bytes": record_bytes,
        "bytes_per_parent": float(model.get("bytes_per_parent", 0) or 0),
    }


def infer_concurrency(path: pathlib.Path) -> int:
    match = CONCURRENCY_PATTERN.search(path.as_posix())
    if match is None:
        raise ValueError(
            f"{path}: cannot infer concurrency from a concurrency_N directory")
    return int(match.group(1))


def summarize_run(
    path: pathlib.Path,
    extent_payload_ratio: float,
    fixed_record_bytes: float,
) -> dict[str, Any]:
    report = load_json(path)
    throughput = report.get("throughput", {})
    gpu = report.get("gpu_persistent", {})
    query = report.get("query_breakdown", {})
    qps = float(
        throughput.get(
            "effective_query_ops_per_sec",
            throughput.get("query_ops_per_sec", 0))
        or 0)
    query_count = float(
        query.get("count", 0)
        or gpu.get("queries_completed", 0)
        or 0)
    total_bytes = float(gpu.get("rdma_read_bytes", 0) or 0)
    graph_reads = float(gpu.get("graph_page_requests", 0) or 0)
    if qps <= 0 or query_count <= 0 or total_bytes <= 0:
        raise ValueError(f"{path}: missing positive QPS/query/RDMA counters")
    total_bytes_per_query = total_bytes / query_count
    graph_reads_per_query = graph_reads / query_count
    graph_bytes_per_query = graph_reads_per_query * fixed_record_bytes
    if graph_bytes_per_query > total_bytes_per_query * (1.0 + 1e-9):
        raise ValueError(
            f"{path}: fixed graph bytes exceed total RDMA bytes")
    non_graph_bytes_per_query = max(
        0.0, total_bytes_per_query - graph_bytes_per_query)
    extent_graph_bytes_per_query = (
        graph_bytes_per_query * extent_payload_ratio)
    extent_total_bytes_per_query = (
        non_graph_bytes_per_query + extent_graph_bytes_per_query)
    observed_bytes_per_second = qps * total_bytes_per_query
    extent_bytes_per_second_at_current_qps = (
        qps * extent_total_bytes_per_query)
    bandwidth_only_speedup = (
        total_bytes_per_query / extent_total_bytes_per_query)
    return {
        "concurrency": infer_concurrency(path),
        "source": str(path),
        "qps": qps,
        "query_count": query_count,
        "graph_reads_per_query": graph_reads_per_query,
        "current_total_bytes_per_query": total_bytes_per_query,
        "current_graph_bytes_per_query": graph_bytes_per_query,
        "current_non_graph_bytes_per_query": non_graph_bytes_per_query,
        "current_graph_byte_fraction": ratio(
            graph_bytes_per_query, total_bytes_per_query),
        "current_total_GB_per_s": observed_bytes_per_second / 1e9,
        "current_graph_GB_per_s": qps * graph_bytes_per_query / 1e9,
        "extent_graph_bytes_per_query": extent_graph_bytes_per_query,
        "extent_total_bytes_per_query": extent_total_bytes_per_query,
        "extent_total_byte_reduction": (
            1.0 - extent_total_bytes_per_query / total_bytes_per_query),
        "extent_GB_per_s_at_current_qps":
            extent_bytes_per_second_at_current_qps / 1e9,
        # This counterfactual keeps the run's observed bytes/s constant.  It is
        # an upper bound on a byte-proportional model, not an estimate of the
        # hardware's available bandwidth or the implementation's final QPS.
        "bandwidth_only_speedup_upper_bound": bandwidth_only_speedup,
        "bandwidth_only_qps_upper_bound_at_observed_byte_rate":
            qps * bandwidth_only_speedup,
    }


def summarize(
    extent: dict[str, float],
    report_paths: list[pathlib.Path],
) -> dict[str, Any]:
    if not report_paths:
        raise ValueError("at least one stable-run report is required")
    runs = [
        summarize_run(
            path, extent["payload_ratio"], extent["record_bytes"])
        for path in report_paths
    ]
    runs.sort(key=lambda row: row["concurrency"])
    concurrencies = [row["concurrency"] for row in runs]
    if len(set(concurrencies)) != len(concurrencies):
        raise ValueError("stable-run reports contain duplicate concurrencies")
    return {
        "schema": 1,
        "extent": extent,
        "runs": runs,
        "semantics": {
            "GB": "decimal 10^9 bytes",
            "one_shot_extent_wqes": (
                "unchanged from the fixed-record graph READ count"),
            "upper_bound": (
                "For each run, hold its observed RDMA bytes/s constant and "
                "assume all throughput cost is proportional to total RDMA "
                "bytes. This is deliberately optimistic and is not a "
                "performance prediction or a claim of NIC saturation."),
            "omitted_costs": [
                "extent-class lookup or tagged-handle decoding",
                "version/checksum layout changes",
                "GPU decode and validation changes",
                "WQE/doorbell/CQE limits",
                "RDMA latency and queueing",
                "all non-RDMA GPU work",
            ],
        },
    }


def markdown(report: dict[str, Any]) -> str:
    extent = report["extent"]
    lines = [
        "# Live-extent concurrency byte roofline",
        "",
        f"Measured one-shot 8-edge extent payload ratio: "
        f"**{pct(extent['payload_ratio'])}** of the fixed "
        f"{int(extent['record_bytes'])}-byte graph record.",
        "",
        "| concurrency | current QPS | current total GB/s | graph byte share | "
        "extent bytes/query | byte reduction | bandwidth-only QPS upper bound | "
        "upper-bound gain |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["runs"]:
        lines.append(
            f"| {row['concurrency']} | {number(row['qps'], 2)} | "
            f"{number(row['current_total_GB_per_s'], 3)} | "
            f"{pct(row['current_graph_byte_fraction'])} | "
            f"{number(row['extent_total_bytes_per_query'], 1)} | "
            f"{pct(row['extent_total_byte_reduction'])} | "
            f"{number(row['bandwidth_only_qps_upper_bound_at_observed_byte_rate'], 2)} | "
            f"{pct(row['bandwidth_only_speedup_upper_bound'] - 1.0)} |")
    lines += [
        "",
        "The last two columns are a strict byte-proportional roofline: each "
        "run's observed RDMA byte rate is held fixed and every other cost is "
        "assigned zero marginal penalty. They are **not performance "
        "predictions**, do not establish NIC saturation, and must not be "
        "reported as expected QPS.",
        "",
    ]
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--live-extent-json", required=True, type=pathlib.Path)
    parser.add_argument(
        "--output-json", type=pathlib.Path)
    parser.add_argument(
        "--output-markdown", type=pathlib.Path)
    parser.add_argument(
        "stable_run_json", nargs="+", type=pathlib.Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = summarize(
        load_extent(args.live_extent_json), args.stable_run_json)
    rendered = markdown(report)
    print(rendered)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8")
    if args.output_markdown:
        args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
        args.output_markdown.write_text(rendered, encoding="utf-8")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2)
