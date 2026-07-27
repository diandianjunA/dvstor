#!/usr/bin/env python3
"""Analyze observable GPU graph-read completion barriers.

The transport exposes an owner submission-group completion boundary for each
query shard descriptor.  It does not expose per-parent, per-WQE, or
NIC-internal completion.  All metric names below preserve that distinction.
"""

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


READY_LEAD_THRESHOLDS_NS = (5_000, 10_000, 20_000, 50_000)


def percentile(values, quantile):
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def weighted_percentile(value_weights, quantile):
    filtered = sorted(
        (value, weight) for value, weight in value_weights if weight > 0)
    if not filtered:
        return None
    total_weight = sum(weight for _, weight in filtered)
    threshold = quantile * total_weight
    cumulative = 0
    for value, weight in filtered:
        cumulative += weight
        if cumulative >= threshold:
            return float(value)
    return float(filtered[-1][0])


def safe_ratio(numerator, denominator):
    return None if denominator <= 0 else numerator / denominator


def cycles_to_ns(cycles, gpu_clock_khz):
    return safe_ratio(cycles * 1_000_000, gpu_clock_khz)


def _event_key(record):
    # schema=1 did not have route_attempt.  Treat it as route attempt zero,
    # while marking the summary as lacking collision-proof route attribution.
    return (
        record["request_id"],
        record.get("route_attempt", 0),
        record["search_round"],
        record["snapshot_attempt"],
    )


def load_trace(trace_path):
    queries = {}
    batches = defaultdict(list)
    metadata = []
    input_counts = defaultdict(int)
    with trace_path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"{trace_path}:{line_number}: malformed JSON: {error}"
                ) from error
            kind = record.get("type")
            input_counts[kind or "missing_type"] += 1
            if kind == "metadata":
                metadata.append(record)
            elif kind == "query":
                request_id = record["request_id"]
                if request_id in queries:
                    raise ValueError(
                        f"{trace_path}:{line_number}: duplicate query "
                        f"{request_id}")
                queries[request_id] = record
            elif kind == "shard_batch":
                batches[_event_key(record)].append(record)
            else:
                raise ValueError(
                    f"{trace_path}:{line_number}: unknown type {kind!r}")
    return queries, batches, metadata, dict(input_counts)


def _valid_timestamp_event(event):
    issue = event.get("issue_timestamp_ns", 0)
    wait_start = event.get("wait_phase_start_timestamp_ns", 0)
    completion = event.get("completion_timestamp_ns", 0)
    process_start = event.get("batch_process_start_timestamp_ns", 0)
    parent_count = event.get("parent_count", 0)
    return (
        issue > 0
        and completion >= issue
        and process_start >= completion
        and parent_count > 0
        and (wait_start == 0 or issue <= wait_start <= process_start)
    )


def build_round_row(key, events, query):
    request_id, route_attempt, search_round, snapshot_attempt = key
    invalid_events = [
        event for event in events if not _valid_timestamp_event(event)]
    valid = [
        event for event in events if _valid_timestamp_event(event)]
    unique_shards = {event.get("target_shard") for event in valid}
    duplicate_target_shards = len(valid) - len(unique_shards)
    process_starts = {
        event["batch_process_start_timestamp_ns"] for event in valid}
    wait_starts = {
        event.get("wait_phase_start_timestamp_ns", 0) for event in valid
        if event.get("wait_phase_start_timestamp_ns", 0) > 0}
    row = {
        "request_id": request_id,
        "query_status": query.get("status"),
        "route_attempt": route_attempt,
        "search_round": search_round,
        "snapshot_attempt": snapshot_attempt,
        "primary_snapshot_attempt": snapshot_attempt == 0,
        "event_count": len(events),
        "valid_event_count": len(valid),
        "invalid_event_count": len(invalid_events),
        "duplicate_target_shard_count": duplicate_target_shards,
        "process_start_consistent": len(process_starts) <= 1,
        "wait_start_consistent": len(wait_starts) <= 1,
        "complete": (
            len(valid) == len(events)
            and duplicate_target_shards == 0
            and len(process_starts) == 1
            and len(wait_starts) <= 1
        ),
    }
    if not valid:
        row["eligible"] = False
        row["ineligible_reason"] = "no_valid_events"
        return row

    batch_start = min(event["issue_timestamp_ns"] for event in valid)
    last_issue = max(event["issue_timestamp_ns"] for event in valid)
    # schema=1 fallback: after all issue calls is the earliest point at which
    # the query CTA could begin consuming a ready descriptor.
    wait_start = (
        next(iter(wait_starts)) if len(wait_starts) == 1 else last_issue)
    process_start = max(process_starts)
    completions = [event["completion_timestamp_ns"] for event in valid]
    completion_min = min(completions)
    completion_max = max(completions)
    completion_sorted = sorted(completions)
    completion_second_max = (
        completion_sorted[-2] if len(completion_sorted) >= 2
        else completion_sorted[-1])
    parent_count = sum(event["parent_count"] for event in valid)
    # schema<=2 used one uniform bytes_per_parent value. Live-extent batches
    # carry exact per-shard payload sums because requests in one owner
    # descriptor can have different lengths.
    bytes_total = sum(
        event.get(
            "payload_bytes",
            event["parent_count"] * event.get("bytes_per_parent", 0))
        for event in valid)

    completion_offsets = [
        event["completion_timestamp_ns"] - batch_start for event in valid]
    observable_latencies = [
        event["completion_timestamp_ns"] - event["issue_timestamp_ns"]
        for event in valid]
    weighted_completion_offsets = [
        (event["completion_timestamp_ns"] - batch_start,
         event["parent_count"])
        for event in valid]

    # Two request-time areas are deliberately separated:
    # 1. straggler barrier: only the time until the last observable completion.
    # 2. ready until process: also includes the common post-completion handoff.
    straggler_barrier_area = sum(
        event["parent_count"]
        * (completion_max - event["completion_timestamp_ns"])
        for event in valid)
    strict_wait_barrier_area = sum(
        event["parent_count"]
        * max(0, completion_max - max(
            event["completion_timestamp_ns"], wait_start))
        for event in valid)
    ready_until_process_area = sum(
        event["parent_count"]
        * (process_start - event["completion_timestamp_ns"])
        for event in valid)
    read_window = max(0, completion_max - batch_start)
    strict_wait_window = max(0, completion_max - wait_start)
    consume_window = max(0, process_start - batch_start)

    tail_parent_count = sum(
        event["parent_count"] for event in valid
        if event["completion_timestamp_ns"] == completion_max)
    ready_before_tail_parent_count = parent_count - tail_parent_count
    ready_at_wait_start_parent_count = sum(
        event["parent_count"] for event in valid
        if event["completion_timestamp_ns"] <= wait_start)

    row.update({
        "eligible": True,
        "ineligible_reason": None,
        "shard_batch_count": len(valid),
        "multi_shard_batch": len(valid) >= 2,
        "unique_completion_boundary_count": len(set(completions)),
        "parent_count": parent_count,
        "bytes_total": bytes_total,
        "batch_start_ns": batch_start,
        "last_issue_ns": last_issue,
        "wait_phase_start_ns": wait_start,
        "batch_process_start_ns": process_start,
        "issue_span_ns": last_issue - batch_start,
        "completion_min_ns": completion_min,
        "completion_max_ns": completion_max,
        "completion_min_offset_ns": min(completion_offsets),
        "completion_median_offset_ns": percentile(
            completion_offsets, 0.50),
        "completion_p90_offset_ns": percentile(completion_offsets, 0.90),
        "completion_max_offset_ns": max(completion_offsets),
        "parent_weighted_completion_p50_offset_ns": weighted_percentile(
            weighted_completion_offsets, 0.50),
        "parent_weighted_completion_p90_offset_ns": weighted_percentile(
            weighted_completion_offsets, 0.90),
        "observable_latency_p50_ns": percentile(
            observable_latencies, 0.50),
        "observable_latency_p90_ns": percentile(
            observable_latencies, 0.90),
        "observable_latency_max_ns": max(observable_latencies),
        "completion_max_minus_median_ns":
            max(completion_offsets)
            - percentile(completion_offsets, 0.50),
        "completion_max_minus_min_ns": completion_max - completion_min,
        "lone_straggler_tail_ns":
            completion_max - completion_second_max,
        "strict_wait_completion_spread_ns":
            max(0, completion_max - min(
                max(completion, wait_start) for completion in completions)),
        "post_completion_handoff_ns": process_start - completion_max,
        "ready_before_tail_parent_count": ready_before_tail_parent_count,
        "ready_before_tail_parent_fraction": safe_ratio(
            ready_before_tail_parent_count, parent_count),
        "tail_parent_count": tail_parent_count,
        "tail_parent_fraction": safe_ratio(tail_parent_count, parent_count),
        "ready_at_wait_start_parent_count":
            ready_at_wait_start_parent_count,
        "ready_at_wait_start_parent_fraction": safe_ratio(
            ready_at_wait_start_parent_count, parent_count),
        "straggler_barrier_parent_ns": straggler_barrier_area,
        "strict_wait_barrier_parent_ns": strict_wait_barrier_area,
        "ready_until_process_parent_ns": ready_until_process_area,
        "normalized_straggler_barrier_waste": safe_ratio(
            straggler_barrier_area, parent_count * read_window),
        "normalized_strict_wait_barrier_waste": safe_ratio(
            strict_wait_barrier_area, parent_count * strict_wait_window),
        "normalized_ready_until_process": safe_ratio(
            ready_until_process_area, parent_count * consume_window),
        "mean_straggler_barrier_ns_per_parent": safe_ratio(
            straggler_barrier_area, parent_count),
        "mean_ready_until_process_ns_per_parent": safe_ratio(
            ready_until_process_area, parent_count),
        "maximum_ready_until_process_ns":
            process_start - completion_min,
    })
    for threshold in READY_LEAD_THRESHOLDS_NS:
        ready = sum(
            event["parent_count"] for event in valid
            if completion_max - event["completion_timestamp_ns"] >= threshold)
        row[f"parent_count_ready_{threshold // 1000}us_before_tail"] = ready
        row[f"parent_fraction_ready_{threshold // 1000}us_before_tail"] = (
            safe_ratio(ready, parent_count))
    return row


def aggregate_rounds(rows, natural_parent_tile=1):
    if not rows:
        return {
            "round_attempts": 0,
            "multi_shard_round_attempts": 0,
        }
    multi = [row for row in rows if row["multi_shard_batch"]]
    total_parents = sum(row["parent_count"] for row in rows)
    total_straggler_area = sum(
        row["straggler_barrier_parent_ns"] for row in rows)
    total_strict_area = sum(
        row["strict_wait_barrier_parent_ns"] for row in rows)
    total_consume_area = sum(
        row["ready_until_process_parent_ns"] for row in rows)
    straggler_denominator = sum(
        row["parent_count"]
        * max(0, row["completion_max_ns"] - row["batch_start_ns"])
        for row in rows)
    strict_denominator = sum(
        row["parent_count"]
        * max(0, row["completion_max_ns"] - row["wait_phase_start_ns"])
        for row in rows)
    consume_denominator = sum(
        row["parent_count"]
        * max(0, row["batch_process_start_ns"] - row["batch_start_ns"])
        for row in rows)

    result = {
        "round_attempts": len(rows),
        "single_shard_round_attempts": len(rows) - len(multi),
        "multi_shard_round_attempts": len(multi),
        "multi_shard_round_fraction": safe_ratio(len(multi), len(rows)),
        "rounds_with_distinct_completion_boundaries": sum(
            row["unique_completion_boundary_count"] >= 2 for row in rows),
        "rounds_with_nonzero_completion_spread": sum(
            row["completion_max_minus_min_ns"] > 0 for row in multi),
        "rounds_with_ready_parent_tile_before_tail": sum(
            row["ready_before_tail_parent_count"] >= natural_parent_tile
            for row in multi),
        "round_fraction_with_ready_parent_tile_before_tail": safe_ratio(
            sum(
                row["ready_before_tail_parent_count"] >= natural_parent_tile
                for row in multi),
            len(multi)),
        "rounds_with_ready_parent_tile_10us_before_tail": sum(
            row["parent_count_ready_10us_before_tail"]
                >= natural_parent_tile
            for row in multi),
        "round_fraction_with_ready_parent_tile_10us_before_tail": safe_ratio(
            sum(
                row["parent_count_ready_10us_before_tail"]
                    >= natural_parent_tile
                for row in multi),
            len(multi)),
        "parent_count": total_parents,
        "ready_before_tail_parent_fraction": safe_ratio(
            sum(row["ready_before_tail_parent_count"] for row in rows),
            total_parents),
        "tail_parent_fraction": safe_ratio(
            sum(row["tail_parent_count"] for row in rows), total_parents),
        "ready_at_wait_start_parent_fraction": safe_ratio(
            sum(row["ready_at_wait_start_parent_count"] for row in rows),
            total_parents),
        "total_straggler_barrier_parent_ns": total_straggler_area,
        "total_strict_wait_barrier_parent_ns": total_strict_area,
        "total_ready_until_process_parent_ns": total_consume_area,
        "normalized_straggler_barrier_waste": safe_ratio(
            total_straggler_area, straggler_denominator),
        "normalized_strict_wait_barrier_waste": safe_ratio(
            total_strict_area, strict_denominator),
        "normalized_ready_until_process": safe_ratio(
            total_consume_area, consume_denominator),
        # This is a sum of sequential per-attempt opportunity windows, not a
        # measured end-to-end latency saving.
        "total_observable_overlap_window_upper_bound_ns": sum(
            row["completion_max_minus_min_ns"] for row in multi),
        "total_strict_wait_overlap_window_upper_bound_ns": sum(
            row["strict_wait_completion_spread_ns"] for row in multi),
        "completion_spread_p50_ns": percentile(
            [row["completion_max_minus_min_ns"] for row in multi], 0.50),
        "completion_spread_p90_ns": percentile(
            [row["completion_max_minus_min_ns"] for row in multi], 0.90),
        "completion_spread_p95_ns": percentile(
            [row["completion_max_minus_min_ns"] for row in multi], 0.95),
        "completion_max_minus_median_p50_ns": percentile(
            [row["completion_max_minus_median_ns"] for row in multi], 0.50),
        "completion_max_minus_median_p90_ns": percentile(
            [row["completion_max_minus_median_ns"] for row in multi], 0.90),
        "strict_wait_spread_p50_ns": percentile(
            [row["strict_wait_completion_spread_ns"] for row in multi], 0.50),
        "strict_wait_spread_p90_ns": percentile(
            [row["strict_wait_completion_spread_ns"] for row in multi], 0.90),
        "lone_straggler_tail_p50_ns": percentile(
            [row["lone_straggler_tail_ns"] for row in multi], 0.50),
        "lone_straggler_tail_p90_ns": percentile(
            [row["lone_straggler_tail_ns"] for row in multi], 0.90),
        "post_completion_handoff_p50_ns": percentile(
            [row["post_completion_handoff_ns"] for row in rows], 0.50),
        "post_completion_handoff_p90_ns": percentile(
            [row["post_completion_handoff_ns"] for row in rows], 0.90),
        "issue_span_p50_ns": percentile(
            [row["issue_span_ns"] for row in multi], 0.50),
        "issue_span_p90_ns": percentile(
            [row["issue_span_ns"] for row in multi], 0.90),
    }
    for threshold in READY_LEAD_THRESHOLDS_NS:
        key = f"parent_fraction_ready_{threshold // 1000}us_before_tail"
        ready = 0
        for row in rows:
            fraction = row[key]
            if fraction is not None:
                ready += fraction * row["parent_count"]
        result[key] = safe_ratio(ready, total_parents)
    return result


def build_query_rows(queries, round_rows):
    by_query = defaultdict(list)
    for row in round_rows:
        if row.get("eligible"):
            by_query[row["request_id"]].append(row)

    query_rows = []
    # Include every emitted query record, even if all its events are invalid.
    for request_id, query in sorted(queries.items()):
        rows = by_query.get(request_id, [])
        primary = [
            row for row in rows if row["primary_snapshot_attempt"]]
        primary_multi = [
            row for row in primary if row["multi_shard_batch"]]
        final_route_attempt = max(
            (row["route_attempt"] for row in rows), default=0)
        final_route = [
            row for row in rows
            if row["route_attempt"] == final_route_attempt]
        final_route_primary = [
            row for row in final_route if row["primary_snapshot_attempt"]]
        final_route_retry = [
            row for row in final_route if not row["primary_snapshot_attempt"]]
        gpu_clock_khz = query.get("gpu_clock_khz", 0)
        query_gpu_ns = cycles_to_ns(
            query.get("gpu_cycles", 0), gpu_clock_khz)
        rdma_wait_ns = cycles_to_ns(
            query.get("rdma_wait_cycles", 0), gpu_clock_khz)
        phase_ns = {
            "beam_selection_ns": cycles_to_ns(
                query.get("beam_selection_cycles", 0), gpu_clock_khz),
            "rdma_issue_ns": cycles_to_ns(
                query.get("rdma_issue_cycles", 0), gpu_clock_khz),
            "graph_validation_ns": cycles_to_ns(
                query.get("graph_validation_cycles", 0), gpu_clock_khz),
            "neighbor_decode_ns": cycles_to_ns(
                query.get("neighbor_decode_cycles", 0), gpu_clock_khz),
            "pq_score_ns": cycles_to_ns(
                query.get("pq_score_cycles", 0), gpu_clock_khz),
            "visited_ns": cycles_to_ns(
                query.get("visited_cycles", 0), gpu_clock_khz),
            "beam_merge_ns": cycles_to_ns(
                query.get("beam_merge_cycles", 0), gpu_clock_khz),
            "exact_ns": cycles_to_ns(
                query.get("exact_cycles", 0), gpu_clock_khz),
        }
        overlap_upper = sum(
            row["strict_wait_completion_spread_ns"]
            for row in primary_multi)
        observable_upper = sum(
            row["completion_max_minus_min_ns"] for row in primary_multi)
        query_rows.append({
            "request_id": request_id,
            "status": query.get("status"),
            "trace_overflow": query.get("overflow", 0),
            "emitted_event_count": query.get("event_count", 0),
            "analyzed_event_count": sum(
                row["valid_event_count"] for row in rows),
            "final_route_attempt": final_route_attempt,
            "query_graph_rounds": query.get("graph_rounds"),
            "trace_final_route_primary_rounds": len(final_route_primary),
            "query_graph_batches": query.get("graph_batches"),
            "trace_final_route_shard_batches": sum(
                row["valid_event_count"] for row in final_route),
            "query_graph_reads": query.get("graph_reads"),
            "trace_final_route_primary_parents": sum(
                row["parent_count"] for row in final_route_primary),
            "query_graph_read_retries": query.get("graph_read_retries"),
            "trace_final_route_retry_parents": sum(
                row["parent_count"] for row in final_route_retry),
            "eligible_round_attempts": len(rows),
            "primary_round_attempts": len(primary),
            "primary_multi_shard_round_attempts": len(primary_multi),
            "query_gpu_time_ns": query_gpu_ns,
            "rdma_wait_ns": rdma_wait_ns,
            **phase_ns,
            "primary_observable_overlap_window_upper_bound_ns":
                observable_upper,
            "primary_strict_wait_overlap_window_upper_bound_ns":
                overlap_upper,
            "strict_wait_overlap_upper_bound_over_rdma_wait": safe_ratio(
                overlap_upper, rdma_wait_ns or 0),
            "strict_wait_overlap_upper_bound_over_query_gpu_time": safe_ratio(
                overlap_upper, query_gpu_ns or 0),
            "primary_straggler_barrier_parent_ns": sum(
                row["straggler_barrier_parent_ns"] for row in primary),
            "primary_ready_before_tail_parent_fraction": safe_ratio(
                sum(row["ready_before_tail_parent_count"] for row in primary),
                sum(row["parent_count"] for row in primary)),
        })
    return query_rows


def analyze_trace(trace_path, include_round_details=True):
    queries, batches, metadata, input_counts = load_trace(trace_path)
    round_rows = []
    for key, events in sorted(batches.items()):
        round_rows.append(
            build_round_row(key, events, queries.get(key[0], {})))

    eligible = [row for row in round_rows if row.get("eligible")]
    complete = [row for row in eligible if row["complete"]]
    primary_complete = [
        row for row in complete if row["primary_snapshot_attempt"]]
    retry_complete = [
        row for row in complete if not row["primary_snapshot_attempt"]]
    query_rows = build_query_rows(queries, complete)
    metadata_schema = max(
        (record.get("schema", 0) for record in metadata), default=0)
    natural_parent_tile = max(
        (record.get("natural_parent_tile", 1) for record in metadata),
        default=1)
    all_events = sum(len(events) for events in batches.values())
    timestamp_values = sorted({
        event.get(field, 0)
        for events in batches.values()
        for event in events
        for field in (
            "issue_timestamp_ns",
            "wait_phase_start_timestamp_ns",
            "completion_timestamp_ns",
            "batch_process_start_timestamp_ns",
        )
        if event.get(field, 0) > 0
    })
    observed_timestamp_quantum_ns = 0
    if len(timestamp_values) >= 2:
        for prior, current in zip(
                timestamp_values, timestamp_values[1:]):
            observed_timestamp_quantum_ns = math.gcd(
                observed_timestamp_quantum_ns, current - prior)

    summary = {
        "schema": 2,
        "input": str(trace_path.resolve()),
        "measurement_granularity":
            "shard_batch_owner_completion_boundary",
        "headline_scope":
            "complete primary snapshot attempts only; multi-shard CDFs "
            "exclude single-shard attempts",
        "interpretation": {
            "completion_spread": (
                "Observable wall-time opportunity between the first and last "
                "owner completion boundaries. It is an upper bound on a "
                "barrier-removal opportunity, not measured recoverable query "
                "latency."),
            "strict_wait_spread": (
                "Completion spread clamped to the common post-issue wait "
                "start, so serialized issue/enqueue time is not credited as "
                "an out-of-order execution opportunity."),
            "parent_ns": (
                "Parent-weighted ready area has units parent*ns and measures "
                "ready work supply. It must not be called wall-clock time."),
            "completion_limit": (
                "One owner submission may contain up to multiple descriptors "
                "and uses its final CQE as their software-visible boundary. "
                "Within-group and within-shard physical completion dispersion "
                "is unobservable and may be understated."),
        },
        "metadata": metadata,
        "input_counts": input_counts,
        "integrity": {
            "metadata_schema": metadata_schema,
            "observed_timestamp_quantum_ns":
                observed_timestamp_quantum_ns,
            "route_attempt_present": all(
                "route_attempt" in event
                for events in batches.values() for event in events),
            "wait_phase_start_present": all(
                event.get("wait_phase_start_timestamp_ns", 0) > 0
                for events in batches.values() for event in events),
            "query_records": len(queries),
            "shard_batch_events": all_events,
            "round_attempt_groups": len(round_rows),
            "complete_round_attempt_groups": len(complete),
            "incomplete_round_attempt_groups": sum(
                not row.get("complete", False) for row in round_rows),
            "invalid_timestamp_events": sum(
                row["invalid_event_count"] for row in round_rows),
            "duplicate_target_shard_events": sum(
                row["duplicate_target_shard_count"] for row in round_rows),
            "inconsistent_process_start_groups": sum(
                not row["process_start_consistent"] for row in round_rows),
            "inconsistent_wait_start_groups": sum(
                not row["wait_start_consistent"] for row in round_rows),
            "missing_query_record_groups": sum(
                row["request_id"] not in queries for row in round_rows),
            "trace_overflow_queries": sum(
                row["trace_overflow"] != 0 for row in query_rows),
            "failed_queries": sum(
                row["status"] not in (None, 0) for row in query_rows),
            "query_event_count_mismatches": sum(
                row["emitted_event_count"] != row["analyzed_event_count"]
                for row in query_rows
                if row["trace_overflow"] == 0),
            "query_graph_round_count_mismatches": sum(
                row["query_graph_rounds"] is not None
                and row["query_graph_rounds"]
                    != row["trace_final_route_primary_rounds"]
                for row in query_rows),
            "query_graph_batch_count_mismatches": sum(
                row["query_graph_batches"] is not None
                and row["query_graph_batches"]
                    != row["trace_final_route_shard_batches"]
                for row in query_rows),
            "query_graph_read_count_mismatches": sum(
                (
                    row["query_graph_reads"] is not None
                    and row["query_graph_reads"]
                        != row["trace_final_route_primary_parents"]
                )
                or (
                    row["query_graph_read_retries"] is not None
                    and row["query_graph_read_retries"]
                        != row["trace_final_route_retry_parents"]
                )
                for row in query_rows),
            "queries_with_route_retry": sum(
                row["final_route_attempt"] != 0 for row in query_rows),
        },
        "aggregate": {
            "primary_complete": aggregate_rounds(
                primary_complete, natural_parent_tile),
            "retry_complete": aggregate_rounds(
                retry_complete, natural_parent_tile),
            "all_complete": aggregate_rounds(
                complete, natural_parent_tile),
            "query_strict_wait_overlap_upper_bound_over_rdma_wait_p50":
                percentile([
                    row["strict_wait_overlap_upper_bound_over_rdma_wait"]
                    for row in query_rows
                    if row[
                        "strict_wait_overlap_upper_bound_over_rdma_wait"
                    ] is not None
                ], 0.50),
            "query_strict_wait_overlap_upper_bound_over_rdma_wait_p90":
                percentile([
                    row["strict_wait_overlap_upper_bound_over_rdma_wait"]
                    for row in query_rows
                    if row[
                        "strict_wait_overlap_upper_bound_over_rdma_wait"
                    ] is not None
                ], 0.90),
            "query_strict_wait_overlap_upper_bound_over_gpu_time_p50":
                percentile([
                    row[
                        "strict_wait_overlap_upper_bound_over_query_gpu_time"
                    ]
                    for row in query_rows
                    if row[
                        "strict_wait_overlap_upper_bound_over_query_gpu_time"
                    ] is not None
                ], 0.50),
            "query_strict_wait_overlap_upper_bound_over_gpu_time_p90":
                percentile([
                    row[
                        "strict_wait_overlap_upper_bound_over_query_gpu_time"
                    ]
                    for row in query_rows
                    if row[
                        "strict_wait_overlap_upper_bound_over_query_gpu_time"
                    ] is not None
                ], 0.90),
            "query_gpu_time_p50_ns": percentile([
                row["query_gpu_time_ns"] for row in query_rows
                if row["query_gpu_time_ns"] is not None
            ], 0.50),
            "query_rdma_wait_p50_ns": percentile([
                row["rdma_wait_ns"] for row in query_rows
                if row["rdma_wait_ns"] is not None
            ], 0.50),
            "query_graph_validation_p50_ns": percentile([
                row["graph_validation_ns"] for row in query_rows
                if row["graph_validation_ns"] is not None
            ], 0.50),
            "query_neighbor_decode_p50_ns": percentile([
                row["neighbor_decode_ns"] for row in query_rows
                if row["neighbor_decode_ns"] is not None
            ], 0.50),
            "query_pq_score_p50_ns": percentile([
                row["pq_score_ns"] for row in query_rows
                if row["pq_score_ns"] is not None
            ], 0.50),
            "query_visited_p50_ns": percentile([
                row["visited_ns"] for row in query_rows
                if row["visited_ns"] is not None
            ], 0.50),
            "query_beam_merge_p50_ns": percentile([
                row["beam_merge_ns"] for row in query_rows
                if row["beam_merge_ns"] is not None
            ], 0.50),
        },
        "queries": query_rows,
        "round_attempts": round_rows if include_round_details else [],
    }
    return summary


def _format_us(value):
    return "n/a" if value is None else f"{value / 1000.0:.2f}"


def _format_pct(value):
    return "n/a" if value is None else f"{value * 100.0:.1f}%"


def write_markdown(summary, destination):
    integrity = summary["integrity"]
    primary = summary["aggregate"]["primary_complete"]
    aggregate = summary["aggregate"]
    integrity_ok = all((
        integrity["metadata_schema"] >= 2,
        integrity["incomplete_round_attempt_groups"] == 0,
        integrity["invalid_timestamp_events"] == 0,
        integrity["duplicate_target_shard_events"] == 0,
        integrity["inconsistent_process_start_groups"] == 0,
        integrity["inconsistent_wait_start_groups"] == 0,
        integrity["missing_query_record_groups"] == 0,
        integrity["trace_overflow_queries"] == 0,
        integrity["failed_queries"] == 0,
        integrity["query_event_count_mismatches"] == 0,
        integrity["query_graph_round_count_mismatches"] == 0,
        integrity["query_graph_batch_count_mismatches"] == 0,
        integrity["query_graph_read_count_mismatches"] == 0,
    ))
    ready_10us = primary.get("parent_fraction_ready_10us_before_tail")
    overlap_over_wait = aggregate.get(
        "query_strict_wait_overlap_upper_bound_over_rdma_wait_p50")
    overlap_over_gpu = aggregate.get(
        "query_strict_wait_overlap_upper_bound_over_gpu_time_p50")
    lines = [
        "# GPU graph-read batch barrier motivation",
        "",
        f"- Trace: `{summary['input']}`",
        f"- Integrity suitable for a headline result: "
        f"**{'yes' if integrity_ok else 'no'}**",
        f"- Traced queries: {integrity['query_records']}",
        f"- Complete primary attempts: {primary['round_attempts']}",
        f"- Multi-shard primary attempts: "
        f"{primary['multi_shard_round_attempts']} "
        f"({_format_pct(primary.get('multi_shard_round_fraction'))})",
        "",
        "## Observable barrier evidence",
        "",
        "| Metric | Result |",
        "|---|---:|",
        f"| Completion spread P50 (eligible multi-shard) | "
        f"{_format_us(primary.get('completion_spread_p50_ns'))} us |",
        f"| Completion spread P90 (eligible multi-shard) | "
        f"{_format_us(primary.get('completion_spread_p90_ns'))} us |",
        f"| Strict post-issue wait spread P50 | "
        f"{_format_us(primary.get('strict_wait_spread_p50_ns'))} us |",
        f"| Strict post-issue wait spread P90 | "
        f"{_format_us(primary.get('strict_wait_spread_p90_ns'))} us |",
        f"| Parent-weighted strict barrier waste | "
        f"{_format_pct(primary.get('normalized_strict_wait_barrier_waste'))} |",
        f"| Parents observable before tail completion | "
        f"{_format_pct(primary.get('ready_before_tail_parent_fraction'))} |",
        f"| Parents ready >=10 us before tail | "
        f"{_format_pct(ready_10us)} |",
        f"| Query overlap upper bound / RDMA wait P50 | "
        f"{_format_pct(overlap_over_wait)} |",
        f"| Query overlap upper bound / GPU time P50 | "
        f"{_format_pct(overlap_over_gpu)} |",
        "",
        "The overlap values are upper bounds, not measured speedups. "
        "Parent-weighted waste has units parent·time. A single-shard attempt "
        "is unobservable at finer granularity and is excluded from spread "
        "percentiles rather than counted as zero.",
        "",
        "## Integrity counters",
        "",
        "```json",
        json.dumps(integrity, indent=2),
        "```",
        "",
        "## Decision rule",
        "",
        "Proceed to a shard-batch out-of-order execution prototype only if "
        "the trace is clean, multi-shard coverage is substantial, strict "
        "wait spread is material relative to query RDMA wait, and at least "
        "one natural parent tile is commonly ready before the tail. Otherwise "
        "this trace is negative or inconclusive evidence; it must not be "
        "presented as proof of a parent-level barrier.",
        "",
    ]
    destination.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--markdown-output", type=Path)
    parser.add_argument(
        "--include-round-details",
        action="store_true",
        help=(
            "embed every derived round row in the summary; the raw JSONL "
            "always remains available and this can make summaries much larger"
        ))
    args = parser.parse_args()

    summary = analyze_trace(
        args.trace, include_round_details=args.include_round_details)
    destination = args.output or args.trace.with_suffix(".summary.json")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2)
        stream.write("\n")
    markdown_destination = (
        args.markdown_output
        or destination.with_suffix(".md"))
    write_markdown(summary, markdown_destination)
    print(destination)
    print(markdown_destination)


if __name__ == "__main__":
    main()
