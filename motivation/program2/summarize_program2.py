#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import statistics
import sys
from pathlib import Path


RECORD_BYTES = 832
EXTENT_QUANTUM = 8
POINTER_BYTES = 8
HEADER_BYTES = 16


def nested(root, *keys, default=0):
    value = root
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value


def manifest(root: Path) -> dict[str, Path]:
    with (root / "manifest.tsv").open(encoding="utf-8", newline="") as stream:
        rows = csv.DictReader(stream, delimiter="\t")
        return {row["case"]: Path(row["report"]) for row in rows}


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def required_bytes(extent_class: int) -> int:
    return min(
        RECORD_BYTES,
        HEADER_BYTES + extent_class * EXTENT_QUANTUM * POINTER_BYTES,
    )


def histogram_percentile(histogram: list[int], fraction: float) -> int:
    total = sum(histogram)
    if not total:
        return 0
    target = total * fraction
    cumulative = 0
    for extent_class, count in enumerate(histogram):
        cumulative += count
        if cumulative >= target:
            return extent_class
    return len(histogram) - 1


def parse_probe(path: Path) -> dict:
    rows = []
    header = None
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("LIVE_EXTENT_RDMA_HEADER,"):
            header = next(csv.reader([line]))[1:]
        elif line.startswith("LIVE_EXTENT_RDMA_CSV,"):
            if header is None:
                raise RuntimeError("probe CSV data appeared before its header")
            values = next(csv.reader([line]))[1:]
            row = dict(zip(header, values, strict=True))
            first = int(row["stage1_B"])
            second = int(row["stage2_B"])
            if first == 832 and second == 0:
                method = "fixed_full"
            elif first == 400 and second == 0:
                method = "hinted_one_read"
            elif first == 16 and second == 384:
                method = "dependent_header_body"
            else:
                continue
            elapsed_s = float(row["elapsed_ms"]) / 1000.0
            stages = int(row["stages"])
            read_wqes = int(row["read_WQEs"])
            row["method"] = method
            row["active_QPs"] = int(row["active_QPs"])
            row["logical_reads_per_s"] = read_wqes / stages / elapsed_s
            for key in (
                "batch_latency_mean_us",
                "batch_latency_p50_us",
                "batch_latency_p95_us",
                "batch_latency_p99_us",
            ):
                row[key] = float(row[key])
            rows.append(row)
    if not rows:
        raise RuntimeError(f"no recognized probe rows in {path}")

    output = {}
    for active_qps in sorted({row["active_QPs"] for row in rows}):
        point = {}
        for method in (
            "fixed_full",
            "dependent_header_body",
            "hinted_one_read",
        ):
            samples = [
                row for row in rows
                if row["active_QPs"] == active_qps and row["method"] == method
            ]
            if not samples:
                continue
            point[method] = {
                "samples": len(samples),
                "logical_reads_per_s_median": statistics.median(
                    row["logical_reads_per_s"] for row in samples
                ),
                "batch_latency_p50_us_median": statistics.median(
                    row["batch_latency_p50_us"] for row in samples
                ),
                "batch_latency_p99_us_median": statistics.median(
                    row["batch_latency_p99_us"] for row in samples
                ),
            }
        output[str(active_qps)] = point
    return output


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: summarize_program2.py <run-root>")
    root = Path(sys.argv[1]).resolve()
    reports = manifest(root)
    missing = {"fixed", "header", "live", "probe"} - reports.keys()
    if missing:
        raise SystemExit(f"missing cases in manifest: {', '.join(sorted(missing))}")

    fixed = load_json(reports["fixed"])
    header = load_json(reports["header"])
    live = load_json(reports["live"])
    fixed_gpu = fixed.get("gpu_persistent", {})
    header_gpu = header.get("gpu_persistent", {})
    live_gpu = live.get("gpu_persistent", {})
    for case_name, report, expected_policy in (
        ("fixed", fixed, "fixed"),
        ("header", header, "header-neighbor"),
        ("live", live, "live-extent"),
    ):
        observed_policy = nested(
            report, "meta", "gpu_query_graph_read_policy", default=""
        )
        if observed_policy != expected_policy:
            raise SystemExit(
                f"{case_name} report policy is {observed_policy!r}, "
                f"expected {expected_policy!r}"
            )
    histogram = [int(value) for value in live_gpu.get("expanded_degree_histogram", [])]
    parents = int(live_gpu.get("expanded_parent_count", 0))
    if not histogram or sum(histogram) != parents or parents == 0:
        raise SystemExit(
            "invalid expanded-degree histogram; rebuild both compute and storage binaries"
        )

    average_degree = float(live_gpu.get("average_expanded_parent_degree", 0))
    average_required_bytes = sum(
        count * required_bytes(extent_class)
        for extent_class, count in enumerate(histogram)
    ) / parents
    p50_class = histogram_percentile(histogram, 0.50)
    p95_class = histogram_percentile(histogram, 0.95)

    fixed_qps = float(nested(fixed, "throughput", "query_ops_per_sec"))
    header_qps = float(nested(header, "throughput", "query_ops_per_sec"))
    live_qps = float(nested(live, "throughput", "query_ops_per_sec"))
    fixed_p99_ms = float(
        nested(fixed, "query_breakdown", "latency", "p99_end_to_end_ns")
    ) / 1e6
    live_p99_ms = float(
        nested(live, "query_breakdown", "latency", "p99_end_to_end_ns")
    ) / 1e6
    header_p99_ms = float(
        nested(header, "query_breakdown", "latency", "p99_end_to_end_ns")
    ) / 1e6
    fixed_graph_bpq = float(
        fixed_gpu.get("average_graph_read_bytes_per_query", 0)
    )
    live_graph_bpq = float(
        live_gpu.get("average_graph_read_bytes_per_query", 0)
    )
    fixed_queries = int(fixed_gpu.get("queries_completed", 0))
    header_queries = int(header_gpu.get("queries_completed", 0))
    live_queries = int(live_gpu.get("queries_completed", 0))
    fixed_wqes_per_query = (
        (int(fixed_gpu.get("graph_live_extent_reads", 0)) +
         int(fixed_gpu.get("graph_full_record_reads", 0))) / fixed_queries
        if fixed_queries else 0
    )
    live_wqes_per_query = (
        (int(live_gpu.get("graph_live_extent_reads", 0)) +
         int(live_gpu.get("graph_full_record_reads", 0))) / live_queries
        if live_queries else 0
    )
    header_wqes_per_query = (
        (int(header_gpu.get("graph_live_extent_reads", 0)) +
         int(header_gpu.get("graph_full_record_reads", 0))) / header_queries
        if header_queries else 0
    )

    summary = {
        "record_bytes": RECORD_BYTES,
        "expanded_parent_samples": parents,
        "average_expanded_parent_degree": average_degree,
        "average_required_prefix_bytes": average_required_bytes,
        "average_fixed_read_wasted_bytes": RECORD_BYTES - average_required_bytes,
        "average_fixed_read_waste_ratio": 1.0 - average_required_bytes / RECORD_BYTES,
        "required_prefix_p50_bytes": required_bytes(p50_class),
        "required_prefix_p95_bytes": required_bytes(p95_class),
        "degree_histogram_quantum": EXTENT_QUANTUM,
        "degree_histogram": histogram,
        "transport_probe": parse_probe(reports["probe"]),
        "fixed": {
            "query_qps": fixed_qps,
            "p99_latency_ms": fixed_p99_ms,
            "graph_bytes_per_query": fixed_graph_bpq,
            "physical_graph_wqes_per_query": fixed_wqes_per_query,
            "gpu_rdma_wait_us_per_query": float(
                fixed_gpu.get("average_gpu_rdma_wait_us", 0)
            ),
            "recall_at_10": float(nested(fixed, "recall", "recall")),
        },
        "header_neighbor": {
            "query_qps": header_qps,
            "p99_latency_ms": header_p99_ms,
            "graph_bytes_per_query": float(
                header_gpu.get("average_graph_read_bytes_per_query", 0)
            ),
            "physical_graph_wqes_per_query": header_wqes_per_query,
            "gpu_rdma_wait_us_per_query": float(
                header_gpu.get("average_gpu_rdma_wait_us", 0)
            ),
            "recall_at_10": float(nested(header, "recall", "recall")),
        },
        "live": {
            "query_qps": live_qps,
            "p99_latency_ms": live_p99_ms,
            "graph_bytes_per_query": live_graph_bpq,
            "physical_graph_wqes_per_query": live_wqes_per_query,
            "gpu_rdma_wait_us_per_query": float(
                live_gpu.get("average_gpu_rdma_wait_us", 0)
            ),
            "recall_at_10": float(nested(live, "recall", "recall")),
            "fallback_reads": int(live_gpu.get("graph_extent_fallback_reads", 0)),
        },
        "qps_improvement_ratio": live_qps / fixed_qps - 1 if fixed_qps else 0,
        "live_vs_header_qps_improvement_ratio": (
            live_qps / header_qps - 1 if header_qps else 0
        ),
        "p99_reduction_ratio": 1 - live_p99_ms / fixed_p99_ms if fixed_p99_ms else 0,
        "graph_bytes_reduction_ratio": (
            1 - live_graph_bpq / fixed_graph_bpq if fixed_graph_bpq else 0
        ),
        "physical_wqe_change_ratio": (
            live_wqes_per_query / fixed_wqes_per_query - 1
            if fixed_wqes_per_query else 0
        ),
        "recall_equal": abs(
            float(nested(fixed, "recall", "recall")) -
            float(nested(live, "recall", "recall"))
        ) < 1e-12,
        "all_recall_equal": max(
            float(nested(fixed, "recall", "recall")),
            float(nested(header, "recall", "recall")),
            float(nested(live, "recall", "recall")),
        ) - min(
            float(nested(fixed, "recall", "recall")),
            float(nested(header, "recall", "recall")),
            float(nested(live, "recall", "recall")),
        ) < 1e-12,
    }

    (root / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    with (root / "summary.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["metric", "value"])
        for key, value in summary.items():
            if not isinstance(value, (dict, list)):
                writer.writerow([key, value])
    with (root / "degree_histogram.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow(["extent_class", "degree_upper_bound", "required_bytes", "count"])
        for extent_class, count in enumerate(histogram):
            writer.writerow([
                extent_class,
                extent_class * EXTENT_QUANTUM,
                required_bytes(extent_class),
                count,
            ])

    print(json.dumps(summary, indent=2))
    print(f"wrote {root / 'summary.json'}")


if __name__ == "__main__":
    main()
