#!/usr/bin/env python3
"""Summarize shard-batch GPU RDMA JSONL traces without inventing per-WQE data."""

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


def percentile(values, quantile):
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def safe_ratio(numerator, denominator):
    return 0.0 if denominator <= 0 else numerator / denominator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    queries = {}
    batches = defaultdict(list)
    metadata = []
    with args.trace.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            kind = record.get("type")
            if kind == "metadata":
                metadata.append(record)
            elif kind == "query":
                queries[record["request_id"]] = record
            elif kind == "shard_batch":
                key = (
                    record["request_id"],
                    record["search_round"],
                    record["snapshot_attempt"],
                )
                batches[key].append(record)
            else:
                raise ValueError(f"{args.trace}:{line_number}: unknown type {kind!r}")

    round_rows = []
    query_accumulators = defaultdict(
        lambda: {
            "unused_ready_ns_requests": 0,
            "read_window_ns_requests": 0,
            "parent_count": 0,
            "rounds": 0,
            "round_average_unused_ready_ns": [],
            "round_max_unused_ready_ns": [],
        }
    )
    for (request_id, search_round, attempt), events in sorted(batches.items()):
        valid = [
            event for event in events
            if event["completion_timestamp_ns"] >= event["issue_timestamp_ns"] > 0
            and event["batch_process_start_timestamp_ns"]
            >= event["completion_timestamp_ns"]
        ]
        if not valid:
            continue
        process_start = max(
            event["batch_process_start_timestamp_ns"] for event in valid)
        batch_start = min(event["issue_timestamp_ns"] for event in valid)
        completions = [event["completion_timestamp_ns"] for event in valid]
        completion_offsets = [value - batch_start for value in completions]
        parent_count = sum(event["parent_count"] for event in valid)
        weighted_unused = sum(
            event["parent_count"] *
            (process_start - event["completion_timestamp_ns"])
            for event in valid
        )
        request_wastes = [
            process_start - event["completion_timestamp_ns"]
            for event in valid
            for _ in range(event["parent_count"])
        ]
        read_window = max(0, process_start - batch_start)
        denominator = parent_count * read_window
        row = {
            "request_id": request_id,
            "status": query.get("status", 0),
            "search_round": search_round,
            "snapshot_attempt": attempt,
            "shard_batch_count": len(valid),
            "parent_count": parent_count,
            "batch_start_ns": batch_start,
            "batch_process_start_ns": process_start,
            "completion_min_offset_ns": min(completion_offsets),
            "completion_median_offset_ns": percentile(completion_offsets, 0.50),
            "completion_p90_offset_ns": percentile(completion_offsets, 0.90),
            "completion_max_offset_ns": max(completion_offsets),
            "max_minus_median_ns":
                max(completion_offsets) - percentile(completion_offsets, 0.50),
            "max_minus_min_ns":
                max(completion_offsets) - min(completion_offsets),
            "average_unused_ready_ns_per_parent":
                statistics.fmean(request_wastes) if request_wastes else 0.0,
            "max_unused_ready_ns_per_shard_batch":
                max(request_wastes) if request_wastes else 0,
            "unused_ready_ns_requests": weighted_unused,
            "read_window_ns_requests": denominator,
            "normalized_barrier_waste": safe_ratio(weighted_unused, denominator),
        }
        round_rows.append(row)
        accumulator = query_accumulators[request_id]
        accumulator["unused_ready_ns_requests"] += weighted_unused
        accumulator["read_window_ns_requests"] += denominator
        accumulator["parent_count"] += parent_count
        accumulator["rounds"] += 1
        accumulator["round_average_unused_ready_ns"].append(
            row["average_unused_ready_ns_per_parent"])
        accumulator["round_max_unused_ready_ns"].append(
            row["max_unused_ready_ns_per_shard_batch"])

    query_rows = []
    for request_id, accumulator in sorted(query_accumulators.items()):
        query = queries.get(request_id, {})
        gpu_clock_khz = query.get("gpu_clock_khz", 0)
        query_ns = safe_ratio(query.get("gpu_cycles", 0) * 1_000_000,
                              gpu_clock_khz)
        unused = accumulator["unused_ready_ns_requests"]
        # This is request-time (ns*parents), not wall time. Dividing by query
        # wall time is intentionally labelled with that dimensional caveat.
        query_rows.append({
            "request_id": request_id,
            "round_attempts": accumulator["rounds"],
            "cumulative_unused_ready_ns_requests": unused,
            "mean_round_average_unused_ready_ns_per_parent":
                statistics.fmean(
                    accumulator["round_average_unused_ready_ns"]),
            "max_round_unused_ready_ns_per_shard_batch":
                max(accumulator["round_max_unused_ready_ns"]),
            "normalized_barrier_waste":
                safe_ratio(unused, accumulator["read_window_ns_requests"]),
            "query_gpu_time_ns": query_ns,
            "mean_parent_unused_ready_ns": safe_ratio(
                unused, accumulator["parent_count"]),
            "mean_parent_unused_over_query_gpu_time": safe_ratio(
                safe_ratio(unused, accumulator["parent_count"]), query_ns),
            "unused_ready_ns_requests_over_query_gpu_ns":
                safe_ratio(unused, query_ns),
            "trace_overflow": query.get("overflow", 0),
        })

    summary = {
        "schema": 1,
        "input": str(args.trace.resolve()),
        "measurement_granularity": "shard_batch",
        "warning": (
            "Completion distributions are across shard batches, not parents or "
            "WQEs. Request-weighted waste assigns a shard batch's observable "
            "completion to all parents in that shard batch."
        ),
        "metadata": metadata,
        "counts": {
            "queries": len(query_rows),
            "round_attempts": len(round_rows),
            "trace_overflow_queries":
                sum(row["trace_overflow"] != 0 for row in query_rows),
            "failed_queries": sum(row["status"] != 0 for row in query_rows),
        },
        "aggregate": {
            "median_max_minus_median_ns": percentile(
                [row["max_minus_median_ns"] for row in round_rows], 0.50),
            "p90_max_minus_median_ns": percentile(
                [row["max_minus_median_ns"] for row in round_rows], 0.90),
            "median_max_minus_min_ns": percentile(
                [row["max_minus_min_ns"] for row in round_rows], 0.50),
            "p90_max_minus_min_ns": percentile(
                [row["max_minus_min_ns"] for row in round_rows], 0.90),
            "total_unused_ready_ns_requests": sum(
                row["unused_ready_ns_requests"] for row in round_rows),
            "total_read_window_ns_requests": sum(
                row["read_window_ns_requests"] for row in round_rows),
            "normalized_barrier_waste": safe_ratio(
                sum(row["unused_ready_ns_requests"] for row in round_rows),
                sum(row["read_window_ns_requests"] for row in round_rows)),
        },
        "queries": query_rows,
        "round_attempts": round_rows,
    }
    destination = args.output or args.trace.with_suffix(".summary.json")
    with destination.open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2)
        stream.write("\n")
    print(destination)


if __name__ == "__main__":
    main()
