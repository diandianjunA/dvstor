#!/usr/bin/env python3
"""Collect all batch-barrier runs without filtering unfavorable results."""

import argparse
import csv
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path


RUN_PATTERN = re.compile(
    r"(performance|trace)/depth_(\d+)/concurrency_(\d+)/repeat_(\d+)")


def mean(values):
    filtered = [value for value in values if value is not None]
    return statistics.fmean(filtered) if filtered else None


def format_number(value, digits=2):
    return "n/a" if value is None else f"{value:.{digits}f}"


def format_percent(value):
    return "n/a" if value is None else f"{value * 100.0:.1f}%"


def identify_run(path, root):
    match = RUN_PATTERN.search(str(path.relative_to(root)))
    if match is None:
        return None
    return {
        "phase": match.group(1),
        "depth": int(match.group(2)),
        "concurrency": int(match.group(3)),
        "repetition": int(match.group(4)),
    }


def integrity_clean(integrity):
    return (
        integrity.get("metadata_schema", 0) >= 2
        and integrity.get("route_attempt_present", False)
        and integrity.get("wait_phase_start_present", False)
        and integrity.get("incomplete_round_attempt_groups", 1) == 0
        and integrity.get("invalid_timestamp_events", 1) == 0
        and integrity.get("duplicate_target_shard_events", 1) == 0
        and integrity.get("inconsistent_process_start_groups", 1) == 0
        and integrity.get("inconsistent_wait_start_groups", 1) == 0
        and integrity.get("missing_query_record_groups", 1) == 0
        and integrity.get("trace_overflow_queries", 1) == 0
        and integrity.get("failed_queries", 1) == 0
        and integrity.get("query_event_count_mismatches", 1) == 0
        and integrity.get("query_graph_round_count_mismatches", 1) == 0
        and integrity.get("query_graph_batch_count_mismatches", 1) == 0
        and integrity.get("query_graph_read_count_mismatches", 1) == 0
    )


def read_trace_runs(root):
    rows = []
    for path in sorted(root.glob(
            "trace/depth_*/concurrency_*/repeat_*/rdma_trace.summary.json")):
        identity = identify_run(path, root)
        if identity is None:
            continue
        summary = json.loads(path.read_text(encoding="utf-8"))
        primary = summary["aggregate"]["primary_complete"]
        aggregate = summary["aggregate"]
        integrity = summary["integrity"]
        rows.append({
            **identity,
            "summary_path": str(path.resolve()),
            "integrity_clean": integrity_clean(integrity),
            "traced_queries": integrity["query_records"],
            "primary_attempts": primary["round_attempts"],
            "multi_shard_attempts": primary["multi_shard_round_attempts"],
            "multi_shard_round_fraction":
                primary.get("multi_shard_round_fraction"),
            "strict_wait_spread_p50_us": (
                None if primary.get("strict_wait_spread_p50_ns") is None
                else primary["strict_wait_spread_p50_ns"] / 1000.0),
            "strict_wait_spread_p90_us": (
                None if primary.get("strict_wait_spread_p90_ns") is None
                else primary["strict_wait_spread_p90_ns"] / 1000.0),
            "completion_spread_p50_us": (
                None if primary.get("completion_spread_p50_ns") is None
                else primary["completion_spread_p50_ns"] / 1000.0),
            "completion_spread_p90_us": (
                None if primary.get("completion_spread_p90_ns") is None
                else primary["completion_spread_p90_ns"] / 1000.0),
            "normalized_strict_wait_barrier_waste":
                primary.get("normalized_strict_wait_barrier_waste"),
            "ready_before_tail_parent_fraction":
                primary.get("ready_before_tail_parent_fraction"),
            "ready_tile_10us_round_fraction":
                primary.get(
                    "round_fraction_with_ready_parent_tile_10us_before_tail"),
            "overlap_upper_bound_over_rdma_wait_p50":
                aggregate.get(
                    "query_strict_wait_overlap_upper_bound_over_rdma_wait_p50"),
            "overlap_upper_bound_over_gpu_time_p50":
                aggregate.get(
                    "query_strict_wait_overlap_upper_bound_over_gpu_time_p50"),
            "overflow_queries": integrity["trace_overflow_queries"],
            "failed_queries": integrity["failed_queries"],
            "invalid_events": integrity["invalid_timestamp_events"],
            "incomplete_groups":
                integrity["incomplete_round_attempt_groups"],
        })
    return rows


def read_benchmark_runs(root, phase):
    by_identity = {}
    pattern = f"{phase}/depth_*/concurrency_*/repeat_*/*/*.json"
    for path in sorted(root.glob(pattern)):
        identity = identify_run(path, root)
        if identity is None:
            continue
        report = json.loads(path.read_text(encoding="utf-8"))
        if "throughput" not in report or "query_breakdown" not in report:
            continue
        throughput = report["throughput"]
        latency = report["query_breakdown"].get("latency", {})
        gpu = report.get("gpu_persistent", {})
        recall = report.get("recall", {})
        row = {
            **identity,
            "report_path": str(path.resolve()),
            "qps": throughput.get("effective_query_ops_per_sec"),
            "mean_latency_us": (
                None if latency.get("mean_end_to_end_ns") is None
                else latency["mean_end_to_end_ns"] / 1000.0),
            "p50_latency_us": (
                None if latency.get("p50_end_to_end_ns") is None
                else latency["p50_end_to_end_ns"] / 1000.0),
            "p95_latency_us": (
                None if latency.get("p95_end_to_end_ns") is None
                else latency["p95_end_to_end_ns"] / 1000.0),
            "p99_latency_us": (
                None if latency.get("p99_end_to_end_ns") is None
                else latency["p99_end_to_end_ns"] / 1000.0),
            "p999_latency_us": (
                None if latency.get("p999_end_to_end_ns") is None
                else latency["p999_end_to_end_ns"] / 1000.0),
            "recall_at_k": recall.get("recall"),
            "average_gpu_query_us": gpu.get("average_gpu_query_us"),
            "average_gpu_rdma_wait_us":
                gpu.get("average_gpu_rdma_wait_us"),
            "average_gpu_beam_merge_us":
                gpu.get("average_gpu_beam_merge_us"),
            "average_graph_rounds_per_query":
                gpu.get("average_graph_rounds_per_query"),
            "average_graph_shard_batches_per_query":
                gpu.get("average_graph_shard_batches_per_query"),
            "average_selected_batch": gpu.get("average_selected_batch"),
        }
        key = (
            identity["depth"],
            identity["concurrency"],
            identity["repetition"],
        )
        # A rerun may leave an older timestamped report in the same repeat
        # directory. Keep the latest complete report rather than double count.
        previous = by_identity.get(key)
        if previous is None or path.stat().st_mtime_ns > previous[0]:
            by_identity[key] = (path.stat().st_mtime_ns, row)
    return [entry[1] for _, entry in sorted(by_identity.items())]


def write_csv(path, rows):
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def grouped_trace_rows(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["depth"], row["concurrency"])].append(row)
    result = []
    numeric_fields = [
        "multi_shard_round_fraction",
        "strict_wait_spread_p50_us",
        "strict_wait_spread_p90_us",
        "completion_spread_p50_us",
        "completion_spread_p90_us",
        "normalized_strict_wait_barrier_waste",
        "ready_before_tail_parent_fraction",
        "ready_tile_10us_round_fraction",
        "overlap_upper_bound_over_rdma_wait_p50",
        "overlap_upper_bound_over_gpu_time_p50",
    ]
    for (depth, concurrency), group in sorted(groups.items()):
        record = {
            "depth": depth,
            "concurrency": concurrency,
            "repetitions": len(group),
            "all_integrity_clean": all(
                row["integrity_clean"] for row in group),
            "traced_queries": sum(row["traced_queries"] for row in group),
            "primary_attempts": sum(row["primary_attempts"] for row in group),
        }
        for field in numeric_fields:
            record[field] = mean([row[field] for row in group])
        result.append(record)
    return result


def screening_result(row):
    # These thresholds are declared before inspecting a run.  They are a
    # conservative prototype screen, not a statistical significance test.
    checks = {
        "integrity": row["all_integrity_clean"],
        "multi_shard_coverage":
            (row["multi_shard_round_fraction"] or 0) >= 0.25,
        "strict_wait_spread":
            (row["strict_wait_spread_p50_us"] or 0) >= 5.0,
        "parent_weighted_waste":
            (row["normalized_strict_wait_barrier_waste"] or 0) >= 0.20,
        "ready_compute_tile":
            (row["ready_tile_10us_round_fraction"] or 0) >= 0.50,
        "end_to_end_headroom":
            (row["overlap_upper_bound_over_gpu_time_p50"] or 0) >= 0.10,
    }
    if not checks["integrity"]:
        verdict = "invalid"
    elif all(checks.values()):
        verdict = "supports prototype"
    else:
        verdict = "insufficient/negative"
    return verdict, checks


def write_report(root, trace_groups, performance_rows, trace_runtime_rows):
    lines = [
        "# Batch barrier motivation matrix",
        "",
        "This report includes every discovered run. Trace QPS is never used as "
        "a performance result; performance rows were collected with tracing "
        "fully off.",
        "",
        "## Mechanism trace",
        "",
        "| depth | concurrency | reps | queries | multi-shard rounds | strict "
        "spread P50/P90 (us) | strict parent waste | ready tile +10us | "
        "upper bound / GPU time | screen |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in trace_groups:
        verdict, _ = screening_result(row)
        strict_waste = format_percent(
            row["normalized_strict_wait_barrier_waste"])
        overlap_gpu = format_percent(
            row["overlap_upper_bound_over_gpu_time_p50"])
        lines.append(
            f"| {row['depth']} | {row['concurrency']} | "
            f"{row['repetitions']} | {row['traced_queries']} | "
            f"{format_percent(row['multi_shard_round_fraction'])} | "
            f"{format_number(row['strict_wait_spread_p50_us'])}/"
            f"{format_number(row['strict_wait_spread_p90_us'])} | "
            f"{strict_waste} | "
            f"{format_percent(row['ready_tile_10us_round_fraction'])} | "
            f"{overlap_gpu} | "
            f"{verdict} |")

    lines.extend([
        "",
        "The screen is preregistered and deliberately conservative: clean "
        "integrity, >=25% multi-shard primary attempts, >=5 us strict P50 "
        "spread, >=20% parent-weighted strict waste, a natural parent tile "
        "ready >=10 us early in >=50% of eligible attempts, and a per-query "
        "strict overlap upper bound >=10% of GPU residence. Failing the screen "
        "is retained as negative evidence.",
        "",
        "## Trace-off performance controls",
        "",
        "| depth | concurrency | repeat | QPS | mean/P99 latency (us) | "
        "Recall | GPU RDMA wait (us/query) | rounds/query |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in performance_rows:
        lines.append(
            f"| {row['depth']} | {row['concurrency']} | "
            f"{row['repetition']} | {format_number(row['qps'])} | "
            f"{format_number(row['mean_latency_us'])}/"
            f"{format_number(row['p99_latency_us'])} | "
            f"{format_number(row['recall_at_k'], 4)} | "
            f"{format_number(row['average_gpu_rdma_wait_us'])} | "
            f"{format_number(row['average_graph_rounds_per_query'])} |")

    performance_by_key = {
        (row["depth"], row["concurrency"], row["repetition"]): row
        for row in performance_rows}
    overhead_rows = []
    for trace_row in trace_runtime_rows:
        key = (
            trace_row["depth"],
            trace_row["concurrency"],
            trace_row["repetition"],
        )
        control = performance_by_key.get(key)
        if control is None or not control.get("qps"):
            continue
        overhead_rows.append((
            trace_row,
            control,
            trace_row["qps"] / control["qps"] - 1.0,
        ))
    lines.extend([
        "",
        "## Sampled-trace overhead sanity",
        "",
        "| depth | concurrency | repeat | trace-off QPS | sampled QPS | delta |",
        "|---:|---:|---:|---:|---:|---:|",
    ])
    for trace_row, control, delta in overhead_rows:
        lines.append(
            f"| {trace_row['depth']} | {trace_row['concurrency']} | "
            f"{trace_row['repetition']} | {format_number(control['qps'])} | "
            f"{format_number(trace_row['qps'])} | {format_percent(delta)} |")

    lines.extend([
        "",
        "## Scope limit",
        "",
        "A completion event is a query shard descriptor observed at its owner "
        "submission-group completion boundary. The experiment cannot see "
        "parent/WQE completion within a shard or within a shared final-CQE "
        "group. Therefore it can support only a shard-batch-granularity "
        "execute-ready/commit-in-order design. If most rounds contain one "
        "observable shard batch, the result is inconclusive at this interface "
        "rather than evidence that parent-level dispersion is absent.",
        "",
    ])
    (root / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    trace_rows = read_trace_runs(args.root)
    performance_rows = read_benchmark_runs(args.root, "performance")
    trace_runtime_rows = read_benchmark_runs(args.root, "trace")
    trace_groups = grouped_trace_rows(trace_rows)
    write_csv(args.root / "trace_runs.csv", trace_rows)
    write_csv(args.root / "trace_matrix.csv", trace_groups)
    write_csv(args.root / "performance_runs.csv", performance_rows)
    write_csv(args.root / "trace_runtime_runs.csv", trace_runtime_rows)
    write_report(
        args.root, trace_groups, performance_rows, trace_runtime_rows)
    print(args.root / "REPORT.md")


if __name__ == "__main__":
    main()
