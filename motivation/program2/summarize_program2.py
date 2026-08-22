#!/usr/bin/env python3
"""Summarize Program 2's dynamic mixed-workload experiment.

Oracle is reconstructed after a dynamic version has been fetched and committed:
16 header bytes + 8 bytes per true live neighbor, one read per committed parent.
It is a byte/WQE lower bound only and deliberately has no end-to-end QPS.
"""

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


def read_manifest(root: Path) -> dict[str, Path]:
    with (root / "manifest.tsv").open(encoding="utf-8", newline="") as stream:
        return {
            row["case"]: Path(row["report"])
            for row in csv.DictReader(stream, delimiter="\t")
        }


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def class_bytes(extent_class: int) -> int:
    return min(
        RECORD_BYTES,
        HEADER_BYTES + extent_class * EXTENT_QUANTUM * POINTER_BYTES,
    )


def histogram_percentile(histogram: list[int], fraction: float) -> int:
    target = sum(histogram) * fraction
    cumulative = 0
    for extent_class, count in enumerate(histogram):
        cumulative += count
        if cumulative >= target:
            return extent_class
    return max(0, len(histogram) - 1)


def parse_probe(path: Path) -> dict:
    rows: list[dict] = []
    header = None
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("LIVE_EXTENT_RDMA_HEADER,"):
            header = next(csv.reader([line]))[1:]
        elif line.startswith("LIVE_EXTENT_RDMA_CSV,") and header is not None:
            values = next(csv.reader([line]))[1:]
            row = dict(zip(header, values, strict=True))
            first, second = int(row["stage1_B"]), int(row["stage2_B"])
            if (first, second) == (832, 0):
                method = "fixed_full"
            elif first == 16 and second > 0:
                method = "dependent_header_body"
            elif second == 0:
                method = "one_read_payload"
            else:
                continue
            elapsed_s = float(row["elapsed_ms"]) / 1000.0
            row.update(
                method=method,
                active_QPs=int(row["active_QPs"]),
                logical_reads_per_s=(
                    int(row["read_WQEs"]) / int(row["stages"]) / elapsed_s
                ),
                batch_latency_p99_us=float(row["batch_latency_p99_us"]),
            )
            rows.append(row)
    output = {}
    for active_qps in sorted({row["active_QPs"] for row in rows}):
        output[str(active_qps)] = {}
        for method in {row["method"] for row in rows}:
            samples = [
                row for row in rows
                if row["active_QPs"] == active_qps and row["method"] == method
            ]
            if samples:
                output[str(active_qps)][method] = {
                    "samples": len(samples),
                    "logical_reads_per_s_median": statistics.median(
                        row["logical_reads_per_s"] for row in samples
                    ),
                    "batch_latency_p99_us_median": statistics.median(
                        row["batch_latency_p99_us"] for row in samples
                    ),
                }
    return output


def case_summary(report: dict) -> dict:
    gpu = report.get("gpu_persistent", {})
    throughput = report.get("throughput", {})
    dynamic_parents = int(gpu.get("dynamic_expanded_parent_count", 0))
    dynamic_attempts = (
        int(gpu.get("dynamic_graph_short_reads", 0))
        + int(gpu.get("dynamic_graph_full_reads", 0))
    )
    return {
        "query_qps": float(throughput.get("query_ops_per_sec", 0)),
        "write_qps": float(throughput.get("write_ops_per_sec", 0)),
        "query_rate_attainment_ratio": float(
            throughput.get("query_rate_attainment_ratio", 0)
        ),
        "write_rate_attainment_ratio": float(
            throughput.get("write_rate_attainment_ratio", 0)
        ),
        "completed_queries": int(throughput.get("query_ops", 0)),
        "completed_writes": int(throughput.get("write_ops", 0)),
        "p99_latency_ms": float(
            nested(report, "query_breakdown", "latency", "p99_end_to_end_ns")
        ) / 1e6,
        "recall_at_10": float(nested(report, "recall", "recall")),
        "dynamic_expanded_parents": dynamic_parents,
        "dynamic_expanded_parent_ratio": float(
            gpu.get("dynamic_expanded_parent_ratio", 0)
        ),
        "average_dynamic_degree": float(
            gpu.get("average_dynamic_expanded_parent_degree", 0)
        ),
        "dynamic_physical_reads": dynamic_attempts,
        "dynamic_read_bytes": int(gpu.get("dynamic_graph_read_bytes", 0)),
        "dynamic_bytes_per_committed_parent": (
            float(gpu.get("dynamic_graph_read_bytes", 0)) / dynamic_parents
            if dynamic_parents else 0
        ),
        "dynamic_wqes_per_committed_parent": (
            dynamic_attempts / dynamic_parents if dynamic_parents else 0
        ),
        "fallback_reads": int(gpu.get("dynamic_graph_fallback_reads", 0)),
        "fallback_ratio": float(gpu.get("dynamic_graph_fallback_ratio", 0)),
        "hint_promotions": int(gpu.get("dynamic_graph_hint_promotions", 0)),
        "hint_demotions": int(gpu.get("dynamic_graph_hint_demotions", 0)),
        "gpu_rdma_wait_us_per_query": float(
            gpu.get("average_gpu_rdma_wait_us", 0)
        ),
    }


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: summarize_program2.py <run-root>")
    root = Path(sys.argv[1]).resolve()
    reports = read_manifest(root)
    missing = {"fixed", "header", "live"} - reports.keys()
    if missing:
        raise SystemExit(f"missing dynamic cases: {', '.join(sorted(missing))}")

    loaded = {name: load_json(reports[name]) for name in ("fixed", "header", "live")}
    expected = {"fixed": "fixed", "header": "header-neighbor", "live": "live-extent"}
    contracts = set()
    for name, report in loaded.items():
        meta = report.get("meta", {})
        if meta.get("workload") != "mixed" or meta.get("mixed_dispatch_policy") != "rate_limited":
            raise SystemExit(f"{name} is not a rate-limited mixed run")
        if meta.get("gpu_query_graph_read_policy") != expected[name]:
            raise SystemExit(f"{name} has the wrong graph-read policy")
        contracts.add((
            float(meta.get("target_query_qps", 0)),
            float(meta.get("target_write_qps", 0)),
            int(meta.get("warmup_seconds", 0)),
            int(meta.get("measure_seconds", 0)),
            int(meta.get("insert_start_id", 0)),
            float(meta.get("write_insert_ratio", 0)),
            float(meta.get("write_upsert_ratio", 0)),
            float(meta.get("write_delete_ratio", 0)),
        ))
    if len(contracts) != 1:
        raise SystemExit(
            "cases used different rates, durations, insertion IDs, or write mix"
        )
    contract = next(iter(contracts))

    live_gpu = loaded["live"].get("gpu_persistent", {})
    histogram = [
        int(value)
        for value in live_gpu.get("dynamic_expanded_degree_histogram", [])
    ]
    dynamic_parents = int(live_gpu.get("dynamic_expanded_parent_count", 0))
    dynamic_neighbor_sum = int(
        live_gpu.get("dynamic_expanded_neighbor_count_sum", 0)
    )
    if not histogram or sum(histogram) != dynamic_parents or not dynamic_parents:
        raise SystemExit(
            "missing dynamic degree trace; rebuild the updated compute binary and rerun"
        )

    oracle_total_bytes = (
        HEADER_BYTES * dynamic_parents + POINTER_BYTES * dynamic_neighbor_sum
    )
    oracle_average_bytes = oracle_total_bytes / dynamic_parents
    class_average_bytes = sum(
        count * class_bytes(extent_class)
        for extent_class, count in enumerate(histogram)
    ) / dynamic_parents
    p50_class = histogram_percentile(histogram, 0.50)
    p95_class = histogram_percentile(histogram, 0.95)
    cases = {name: case_summary(report) for name, report in loaded.items()}
    fixed, header, live = cases["fixed"], cases["header"], cases["live"]

    summary = {
        "experiment_kind": "dynamic_rate_limited_mixed",
        "target_query_qps": contract[0],
        "target_write_qps": contract[1],
        "warmup_seconds": contract[2],
        "measure_seconds": contract[3],
        "insert_start_id": contract[4],
        "record_bytes": RECORD_BYTES,
        "dynamic_degree_histogram_quantum": EXTENT_QUANTUM,
        "dynamic_degree_histogram": histogram,
        "dynamic_expanded_parent_samples": dynamic_parents,
        "average_dynamic_degree": dynamic_neighbor_sum / dynamic_parents,
        "dynamic_degree_p50_upper_bound": p50_class * EXTENT_QUANTUM,
        "dynamic_degree_p95_upper_bound": p95_class * EXTENT_QUANTUM,
        "oracle": {
            "definition": "post-observed exact length; zero metadata cost; one read; not deployable",
            "performance_measured": False,
            "total_required_bytes": oracle_total_bytes,
            "average_bytes_per_committed_dynamic_parent": oracle_average_bytes,
            "wqes_per_committed_dynamic_parent": 1.0,
        },
        "theoretical_class": {
            "average_bytes_per_committed_dynamic_parent": class_average_bytes,
            "rounding_over_oracle_ratio": class_average_bytes / oracle_average_bytes - 1,
        },
        "fixed_theoretical_waste_over_oracle_ratio": (
            RECORD_BYTES / oracle_average_bytes - 1
        ),
        "cases": cases,
        "live_vs_fixed_qps_improvement_ratio": (
            live["query_qps"] / fixed["query_qps"] - 1 if fixed["query_qps"] else 0
        ),
        "live_vs_header_qps_improvement_ratio": (
            live["query_qps"] / header["query_qps"] - 1 if header["query_qps"] else 0
        ),
        "live_actual_bytes_over_oracle_ratio": (
            live["dynamic_bytes_per_committed_parent"] / oracle_average_bytes - 1
        ),
        "transport_probe": (
            parse_probe(reports["probe"]) if "probe" in reports else None
        ),
    }

    (root / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    with (root / "summary.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow([
            "case", "query_qps", "write_qps", "p99_ms", "recall_at_10",
            "dynamic_share", "dynamic_degree", "dynamic_bytes_per_parent",
            "dynamic_wqes_per_parent", "fallback_ratio", "write_attainment",
        ])
        for name in ("fixed", "header", "live"):
            case = cases[name]
            writer.writerow([
                name, case["query_qps"], case["write_qps"], case["p99_latency_ms"],
                case["recall_at_10"], case["dynamic_expanded_parent_ratio"],
                case["average_dynamic_degree"],
                case["dynamic_bytes_per_committed_parent"],
                case["dynamic_wqes_per_committed_parent"], case["fallback_ratio"],
                case["write_rate_attainment_ratio"],
            ])
        writer.writerow([
            "oracle", "", "", "", "", "", summary["average_dynamic_degree"],
            oracle_average_bytes, 1.0, "", "",
        ])
    with (root / "dynamic_degree_histogram.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow(["extent_class", "degree_upper_bound", "class_bytes", "count"])
        for extent_class, count in enumerate(histogram):
            writer.writerow([
                extent_class, extent_class * EXTENT_QUANTUM,
                class_bytes(extent_class), count,
            ])

    print(json.dumps(summary, indent=2))
    print(f"wrote {root / 'summary.json'}")


if __name__ == "__main__":
    main()
