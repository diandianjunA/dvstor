#!/usr/bin/env python3
"""Validate and summarize M1.1 AB/BA reports using only the Python stdlib."""

from __future__ import annotations

import csv
import json
import math
import statistics
import sys
from pathlib import Path


EXPECTED_MODE = {"coupled_one_stage": "coupled", "two_stage": "decoupled"}


def fail(message: str) -> None:
    raise SystemExit(f"summarize_m1_1: {message}")


def nested(root: dict, *path: str, default=0):
    value = root
    for part in path:
        if not isinstance(value, dict) or part not in value:
            return default
        value = value[part]
    return value


def discover_report(report_dir: Path) -> Path:
    reports = sorted(report_dir.rglob("sift100m_*.json"))
    if len(reports) != 1:
        fail(f"expected exactly one JSON report under {report_dir}, found {len(reports)}")
    return reports[0]


def load_manifest(run_root: Path) -> list[dict[str, str]]:
    path = run_root / "manifest.tsv"
    if not path.is_file():
        fail(f"missing {path}")
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    if not rows:
        fail("manifest contains no completed cases")
    return rows


def parse_row(entry: dict[str, str]) -> dict[str, object]:
    case_name = entry["case"]
    expected = EXPECTED_MODE.get(case_name)
    if expected is None:
        fail(f"unknown case {case_name!r}")
    report_path = discover_report(Path(entry["report_dir"]))
    with report_path.open(encoding="utf-8") as stream:
        report = json.load(stream)

    resolved = nested(report, "meta", "system_variant", "resolved_modes", default={})
    if resolved.get("storage_owner_update_completion_mode") != expected:
        fail(f"{report_path}: expected update mode {expected}, got {resolved}")
    if resolved.get("gpu_dynamic_graph_access_mode") != "adaptive":
        fail(f"{report_path}: graph access mode was not held at adaptive")
    if resolved.get("gpu_rdma_search_progression_mode") != "decoupled":
        fail(f"{report_path}: search progression mode was not held at decoupled")

    insert = report.get("insert_breakdown", {})
    count = int(insert.get("count", 0))
    if count <= 0:
        fail(f"{report_path}: no measured inserts")
    if not insert.get("fine_grained_breakdown_observed", False):
        fail(f"{report_path}: fine-grained breakdown is missing")

    latency = insert.get("latency", {})
    breakdown = insert.get("breakdown", {})
    cpu_sub = nested(insert, "sub_breakdown", "cpu_ns", default={})
    rdma_sub = nested(insert, "sub_breakdown", "rdma_ns", default={})
    service_ns = float(latency.get("service_ns", 0))
    rdma_ns = float(breakdown.get("rdma_ns", 0))
    query_count = int(nested(report, "query_breakdown", "count", default=0))

    def per_insert_ms(values: dict, key: str) -> float:
        return float(values.get(key, 0)) / count / 1e6

    return {
        "repeat": int(entry["repeat"]),
        "scenario": entry["scenario"],
        "order": int(entry["order"]),
        "case": case_name,
        "update_mode": expected,
        "report": str(report_path),
        "insert_count": count,
        "query_count": query_count,
        "insert_qps": float(nested(report, "throughput", "insert_ops_per_sec")),
        "query_qps": float(nested(report, "throughput", "query_ops_per_sec")),
        "insert_mean_ms": float(latency.get("mean_end_to_end_ns", 0)) / 1e6,
        "insert_p50_ms": float(latency.get("p50_end_to_end_ns", 0)) / 1e6,
        "insert_p99_ms": float(latency.get("p99_end_to_end_ns", 0)) / 1e6,
        "insert_p999_ms": float(latency.get("p999_end_to_end_ns", 0)) / 1e6,
        "query_p99_ms": float(nested(report, "query_breakdown", "latency", "p99_end_to_end_ns")) / 1e6,
        "rdma_share": rdma_ns / service_ns if service_ns else 0.0,
        "rdma_ms_per_insert": rdma_ns / count / 1e6,
        "cpu_ms_per_insert": float(breakdown.get("cpu_ns", 0)) / count / 1e6,
        "search_neighbor_rdma_ms": per_insert_ms(rdma_sub, "rdma_storage_owner_search_neighbor_read_ns"),
        "search_snapshot_rdma_ms": per_insert_ms(rdma_sub, "rdma_storage_owner_search_snapshot_read_ns"),
        "prune_snapshot_rdma_ms": per_insert_ms(rdma_sub, "rdma_storage_owner_prune_snapshot_read_ns"),
        "remote_reverse_cpu_ms": per_insert_ms(cpu_sub, "cpu_storage_owner_remote_reverse_ns"),
        "queue_wait_cpu_ms": per_insert_ms(cpu_sub, "cpu_storage_owner_queue_wait_ns"),
        "recall": float(nested(report, "recall", "recall")),
        "stage2_drain_s": float(nested(report, "storage_owner_runtime", "maintenance_drain_seconds")),
        "owner_rpc_batches": int(nested(report, "storage_owner_runtime", "completed_batches")),
        "owner_rpc_items": int(nested(report, "storage_owner_runtime", "completed_items")),
        "owner_rpc_wall_us": float(nested(report, "storage_owner_runtime", "average_completed_rpc_wall_us")),
    }


def mean_ci(values: list[float]) -> tuple[float, float]:
    mean = statistics.fmean(values)
    if len(values) < 2:
        return mean, 0.0
    # Normal CI is deliberately descriptive; retain raw paired rows for inference.
    return mean, 1.96 * statistics.stdev(values) / math.sqrt(len(values))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: list[dict[str, object]]) -> None:
    metrics = ["insert_p50_ms", "insert_p99_ms", "insert_p999_ms", "insert_qps",
               "rdma_share", "rdma_ms_per_insert", "query_qps", "query_p99_ms"]
    lines = ["# M1.1 summary", "", "Mean ± approximate 95% CI across repetitions.", "",
             "| scenario | case | n | metric | mean | 95% CI half-width |",
             "| --- | --- | ---: | --- | ---: | ---: |"]
    for scenario in sorted({str(row["scenario"]) for row in rows}):
        for case_name in EXPECTED_MODE:
            group = [row for row in rows if row["scenario"] == scenario and row["case"] == case_name]
            if not group:
                continue
            for metric in metrics:
                values = [float(row[metric]) for row in group]
                mean, ci = mean_ci(values)
                lines.append(f"| {scenario} | {case_name} | {len(values)} | {metric} | {mean:.6g} | {ci:.6g} |")

    lines += ["", "## Paired difference (two-stage - coupled)", "",
              "| scenario | metric | pairs | mean difference | 95% CI half-width |",
              "| --- | --- | ---: | ---: | ---: |"]
    for scenario in sorted({str(row["scenario"]) for row in rows}):
        for metric in metrics:
            differences = []
            for repeat in sorted({int(row["repeat"]) for row in rows if row["scenario"] == scenario}):
                pair = {str(row["case"]): float(row[metric]) for row in rows
                        if row["scenario"] == scenario and row["repeat"] == repeat}
                if set(pair) == set(EXPECTED_MODE):
                    differences.append(pair["two_stage"] - pair["coupled_one_stage"])
            if differences:
                mean, ci = mean_ci(differences)
                lines.append(f"| {scenario} | {metric} | {len(differences)} | {mean:.6g} | {ci:.6g} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    if len(sys.argv) != 2:
        fail("usage: summarize_m1_1.py <run-root>")
    run_root = Path(sys.argv[1]).resolve()
    rows = [parse_row(entry) for entry in load_manifest(run_root)]
    write_csv(run_root / "summary.csv", rows)
    write_markdown(run_root / "summary.md", rows)
    print(f"validated {len(rows)} reports; wrote {run_root / 'summary.csv'} and summary.md")


if __name__ == "__main__":
    main()
