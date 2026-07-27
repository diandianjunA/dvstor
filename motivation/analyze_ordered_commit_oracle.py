#!/usr/bin/env python3
"""Estimate execute-ready / commit-in-order opportunity from an RDMA trace.

This is a deliberately conservative motivation oracle, not a performance
prediction.  The trace supplies software-visible release times.  Per-query
validation/decode/PQ/visited wall time is distributed over graph parents to
model the service that could move before the current batch barrier.

The model compares, independently for every search-round snapshot attempt:

  baseline: all completions -> all movable work
  oracle:   each completion -> its movable work, commit after all work

Beam merge, parent selection, expansion accounting, and termination stay
after the commit barrier and are therefore excluded from both sides.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional


MOVABLE_CYCLE_FIELDS = (
    "graph_validation_cycles",
    "neighbor_decode_cycles",
    "pq_score_cycles",
    "visited_cycles",
)
DEFAULT_OVERHEADS_US = (0.0, 1.0, 2.0, 5.0, 10.0)
READY_LEADS_US = (5, 10, 20, 50)


def percentile(values: Iterable[float], quantile: float) -> Optional[float]:
    ordered = sorted(values)
    if not ordered:
        return None
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    fraction = position - lower
    return (
        ordered[lower] * (1.0 - fraction)
        + ordered[upper] * fraction
    )


def safe_ratio(numerator: float, denominator: float) -> Optional[float]:
    return None if denominator <= 0 else numerator / denominator


def cycles_to_ns(cycles: float, gpu_clock_khz: float) -> Optional[float]:
    if gpu_clock_khz <= 0:
        return None
    return cycles * 1_000_000.0 / gpu_clock_khz


def parse_overheads(text: str) -> tuple[float, ...]:
    values = tuple(float(part) for part in text.split(",") if part.strip())
    if not values or any(value < 0 for value in values):
        raise argparse.ArgumentTypeError(
            "task overheads must be a non-empty comma-separated list of "
            "non-negative microsecond values")
    return values


@dataclass(frozen=True)
class WorkUnit:
    release_ns: int
    parent_count: int
    completion_id: tuple


def event_group_key(event: dict) -> tuple:
    return (
        int(event["request_id"]),
        int(event.get("route_attempt", 0)),
        int(event["search_round"]),
        int(event.get("snapshot_attempt", 0)),
    )


def event_completion_id(event: dict) -> tuple:
    """Return an identifier for work sharing one observable release boundary."""
    event_type = event.get("type")
    if event_type == "parent_read":
        return (
            event.get("target_shard"),
            event.get("parent_ordinal"),
        )
    if event_type == "parent_tile":
        return (
            event.get("target_shard"),
            event.get("tile_ordinal"),
        )
    return (event.get("target_shard"),)


def load_trace(path: Path):
    metadata = []
    queries = {}
    grouped_events = defaultdict(list)
    input_counts = defaultdict(int)
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"{path}:{line_number}: malformed JSON: {error}"
                ) from error
            kind = record.get("type")
            input_counts[kind or "missing_type"] += 1
            if kind == "metadata":
                metadata.append(record)
            elif kind == "query":
                request_id = int(record["request_id"])
                if request_id in queries:
                    raise ValueError(
                        f"{path}:{line_number}: duplicate query "
                        f"{request_id}")
                queries[request_id] = record
            elif kind in ("shard_batch", "parent_tile", "parent_read"):
                grouped_events[event_group_key(record)].append(record)
            else:
                raise ValueError(
                    f"{path}:{line_number}: unsupported record type "
                    f"{kind!r}")
    return metadata, queries, grouped_events, dict(input_counts)


def valid_event(event: dict) -> bool:
    issue_ns = int(event.get("issue_timestamp_ns", 0))
    wait_ns = int(event.get("wait_phase_start_timestamp_ns", 0))
    completion_ns = int(event.get("completion_timestamp_ns", 0))
    process_ns = int(event.get("batch_process_start_timestamp_ns", 0))
    parents = int(event.get("parent_count", 0))
    return (
        issue_ns > 0
        and wait_ns >= issue_ns
        and completion_ns >= issue_ns
        and process_ns >= completion_ns
        and process_ns >= wait_ns
        and parents > 0
    )


def collapse_work_units(events: list[dict]) -> list[WorkUnit]:
    """Collapse duplicate event rows that name the same observable unit.

    A future per-parent trace may emit one row per parent.  A shard-batch trace
    emits one row per shard descriptor.  This helper retains either without
    pretending that a shard-batch timestamp is a parent timestamp.
    """
    units = []
    for event in events:
        release_ns = max(
            int(event["completion_timestamp_ns"]),
            int(event["wait_phase_start_timestamp_ns"]),
        )
        units.append(WorkUnit(
            release_ns=release_ns,
            parent_count=int(event["parent_count"]),
            completion_id=event_completion_id(event),
        ))
    return units


def split_into_tasks(units: list[WorkUnit], natural_tile: int,
                     granularity: str) -> list[WorkUnit]:
    if granularity == "completion":
        return units
    tasks = []
    for unit in units:
        remaining = unit.parent_count
        tile_ordinal = 0
        while remaining > 0:
            parents = min(natural_tile, remaining)
            tasks.append(WorkUnit(
                release_ns=unit.release_ns,
                parent_count=parents,
                completion_id=unit.completion_id + (tile_ordinal,),
            ))
            remaining -= parents
            tile_ordinal += 1
    return tasks


def serial_release_schedule(tasks: list[WorkUnit], service_per_parent_ns: float,
                            overhead_per_task_ns: float) -> float:
    """Return oracle movable-work finish time for one query compute CTA."""
    now_ns = float(min(task.release_ns for task in tasks))
    for task in sorted(
            tasks, key=lambda item: (item.release_ns, item.completion_id)):
        now_ns = max(now_ns, float(task.release_ns))
        now_ns += (
            task.parent_count * service_per_parent_ns
            + overhead_per_task_ns
        )
    return now_ns


def build_round_model(
        key: tuple,
        events: list[dict],
        service_per_parent_ns: float,
        natural_tile: int,
        task_granularity: str,
        overheads_us: tuple[float, ...]) -> dict:
    invalid = [event for event in events if not valid_event(event)]
    valid = [event for event in events if valid_event(event)]
    row = {
        "request_id": key[0],
        "route_attempt": key[1],
        "search_round": key[2],
        "snapshot_attempt": key[3],
        "primary_snapshot_attempt": key[3] == 0,
        "event_count": len(events),
        "invalid_event_count": len(invalid),
        "eligible": bool(valid) and not invalid,
    }
    if not row["eligible"]:
        return row

    wait_starts = {
        int(event["wait_phase_start_timestamp_ns"]) for event in valid}
    process_starts = {
        int(event["batch_process_start_timestamp_ns"]) for event in valid}
    if len(wait_starts) != 1 or len(process_starts) != 1:
        row["eligible"] = False
        row["inconsistent_round_boundary"] = True
        return row

    units = collapse_work_units(valid)
    tasks = split_into_tasks(units, natural_tile, task_granularity)
    release_min = min(unit.release_ns for unit in units)
    release_max = max(unit.release_ns for unit in units)
    wait_start = next(iter(wait_starts))
    process_start = next(iter(process_starts))
    parent_count = sum(unit.parent_count for unit in units)
    service_total_ns = parent_count * service_per_parent_ns
    # The strict baseline excludes common completion-to-process handoff.  It
    # therefore cannot make the opportunity look larger than completion
    # dispersion alone permits.
    strict_baseline_finish_ns = release_max + service_total_ns
    observed_baseline_finish_ns = process_start + service_total_ns
    savings = {}
    observed_savings = {}
    for overhead_us in overheads_us:
        finish = serial_release_schedule(
            tasks, service_per_parent_ns, overhead_us * 1000.0)
        key_name = f"{overhead_us:g}us"
        savings[key_name] = max(
            0.0, strict_baseline_finish_ns - finish)
        observed_savings[key_name] = max(
            0.0, observed_baseline_finish_ns - finish)

    row.update({
        "inconsistent_round_boundary": False,
        "completion_granularity_event_types":
            sorted({event.get("type") for event in valid}),
        "observable_completion_count": len(units),
        "distinct_release_count": len({
            unit.release_ns for unit in units}),
        "parent_count": parent_count,
        "task_count": len(tasks),
        "wait_phase_start_ns": wait_start,
        "process_start_ns": process_start,
        "release_min_ns": release_min,
        "release_max_ns": release_max,
        "strict_completion_spread_ns": release_max - release_min,
        "completion_to_process_handoff_ns": process_start - release_max,
        "modeled_movable_service_ns": service_total_ns,
        "strict_saved_ns_by_task_overhead": savings,
        "observed_saved_ns_by_task_overhead": observed_savings,
    })
    for lead_us in READY_LEADS_US:
        threshold = lead_us * 1000
        ready_parents = sum(
            unit.parent_count for unit in units
            if release_max - unit.release_ns >= threshold)
        row[f"parents_ready_{lead_us}us_before_tail"] = ready_parents
        row[f"has_tile_ready_{lead_us}us_before_tail"] = (
            ready_parents >= natural_tile)
    return row


def query_service_per_parent_ns(query: dict) -> Optional[float]:
    clock_khz = float(query.get("gpu_clock_khz", 0))
    reads = int(query.get("graph_reads", 0))
    cycles = sum(float(query.get(field, 0)) for field in MOVABLE_CYCLE_FIELDS)
    service_ns = cycles_to_ns(cycles, clock_khz)
    if reads <= 0 or service_ns is None:
        return None
    return service_ns / reads


def aggregate_query_rows(
        queries: dict[int, dict],
        round_rows: list[dict],
        overheads_us: tuple[float, ...]) -> list[dict]:
    by_query = defaultdict(list)
    for row in round_rows:
        if row.get("eligible") and row["primary_snapshot_attempt"]:
            by_query[row["request_id"]].append(row)

    result = []
    for request_id, query in sorted(queries.items()):
        clock_khz = float(query.get("gpu_clock_khz", 0))
        gpu_ns = cycles_to_ns(float(query.get("gpu_cycles", 0)), clock_khz)
        wait_ns = cycles_to_ns(
            float(query.get("rdma_wait_cycles", 0)), clock_khz)
        rows = by_query.get(request_id, [])
        record = {
            "request_id": request_id,
            "status": query.get("status"),
            "overflow": int(query.get("overflow", 0)),
            "modeled_round_count": len(rows),
            "query_graph_rounds": int(query.get("graph_rounds", 0)),
            "query_graph_reads": int(query.get("graph_reads", 0)),
            "query_gpu_time_ns": gpu_ns,
            "query_rdma_wait_ns": wait_ns,
            "modeled_service_per_parent_ns":
                query_service_per_parent_ns(query),
            "strict_saved_ns_by_task_overhead": {},
            "strict_saved_over_gpu_by_task_overhead": {},
            "strict_saved_over_rdma_wait_by_task_overhead": {},
        }
        for overhead_us in overheads_us:
            key_name = f"{overhead_us:g}us"
            saved_ns = sum(
                row["strict_saved_ns_by_task_overhead"][key_name]
                for row in rows)
            record["strict_saved_ns_by_task_overhead"][key_name] = saved_ns
            record["strict_saved_over_gpu_by_task_overhead"][key_name] = (
                safe_ratio(saved_ns, gpu_ns or 0))
            record[
                "strict_saved_over_rdma_wait_by_task_overhead"
            ][key_name] = safe_ratio(saved_ns, wait_ns or 0)
        result.append(record)
    return result


def integrity_summary(metadata, queries, grouped_events, round_rows):
    traced_events = sum(len(events) for events in grouped_events.values())
    emitted_events = sum(
        int(query.get("event_count", 0)) for query in queries.values())
    return {
        "query_records": len(queries),
        "trace_events": traced_events,
        "query_emitted_event_count": emitted_events,
        "query_event_count_matches": traced_events == emitted_events,
        "failed_queries": sum(
            query.get("status") not in (None, 0)
            for query in queries.values()),
        "overflow_queries": sum(
            int(query.get("overflow", 0)) != 0
            for query in queries.values()),
        "invalid_event_count": sum(
            row.get("invalid_event_count", 0) for row in round_rows),
        "inconsistent_round_groups": sum(
            row.get("inconsistent_round_boundary", False)
            for row in round_rows),
        "metadata_schema": max(
            (int(record.get("schema", 0)) for record in metadata),
            default=0),
    }


def aggregate_summary(round_rows, query_rows, overheads_us, natural_tile):
    primary = [
        row for row in round_rows
        if row.get("eligible") and row["primary_snapshot_attempt"]]
    multi_release = [
        row for row in primary if row["distinct_release_count"] >= 2]
    result = {
        "primary_rounds": len(primary),
        "multi_release_rounds": len(multi_release),
        "multi_release_round_fraction": safe_ratio(
            len(multi_release), len(primary)),
        "natural_parent_tile": natural_tile,
        "strict_completion_spread_p50_ns": percentile(
            (row["strict_completion_spread_ns"] for row in multi_release),
            0.50),
        "strict_completion_spread_p90_ns": percentile(
            (row["strict_completion_spread_ns"] for row in multi_release),
            0.90),
        "strict_completion_spread_p99_ns": percentile(
            (row["strict_completion_spread_ns"] for row in multi_release),
            0.99),
        "completion_to_process_handoff_p50_ns": percentile(
            (row["completion_to_process_handoff_ns"] for row in primary),
            0.50),
        "modeled_service_per_parent_p50_ns": percentile(
            (row["modeled_service_per_parent_ns"] for row in query_rows
             if row["modeled_service_per_parent_ns"] is not None),
            0.50),
        "query_strict_saved_ns_by_task_overhead": {},
        "query_strict_saved_over_gpu_by_task_overhead": {},
        "query_strict_saved_over_rdma_wait_by_task_overhead": {},
    }
    for lead_us in READY_LEADS_US:
        result[
            f"round_fraction_with_tile_ready_{lead_us}us_before_tail"
        ] = safe_ratio(
            sum(
                row[f"has_tile_ready_{lead_us}us_before_tail"]
                for row in primary),
            len(primary))
    for overhead_us in overheads_us:
        key_name = f"{overhead_us:g}us"
        saved_values = [
            row["strict_saved_ns_by_task_overhead"][key_name]
            for row in query_rows]
        gpu_ratios = [
            row["strict_saved_over_gpu_by_task_overhead"][key_name]
            for row in query_rows
            if row["strict_saved_over_gpu_by_task_overhead"][key_name]
                is not None]
        wait_ratios = [
            row["strict_saved_over_rdma_wait_by_task_overhead"][key_name]
            for row in query_rows
            if row["strict_saved_over_rdma_wait_by_task_overhead"][key_name]
                is not None]
        result["query_strict_saved_ns_by_task_overhead"][key_name] = {
            "mean": statistics.mean(saved_values) if saved_values else None,
            "p50": percentile(saved_values, 0.50),
            "p90": percentile(saved_values, 0.90),
        }
        result[
            "query_strict_saved_over_gpu_by_task_overhead"
        ][key_name] = {
            "mean": statistics.mean(gpu_ratios) if gpu_ratios else None,
            "p50": percentile(gpu_ratios, 0.50),
            "p90": percentile(gpu_ratios, 0.90),
        }
        result[
            "query_strict_saved_over_rdma_wait_by_task_overhead"
        ][key_name] = {
            "mean": statistics.mean(wait_ratios) if wait_ratios else None,
            "p50": percentile(wait_ratios, 0.50),
            "p90": percentile(wait_ratios, 0.90),
        }
    return result


def screening(summary: dict) -> dict:
    integrity = summary["integrity"]
    aggregate = summary["aggregate"]
    zero_key = "0us"
    zero_gpu = aggregate[
        "query_strict_saved_over_gpu_by_task_overhead"
    ].get(zero_key, {}).get("p50")
    p50_spread = aggregate.get("strict_completion_spread_p50_ns")
    p90_spread = aggregate.get("strict_completion_spread_p90_ns")
    checks = {
        "integrity_clean": (
            integrity["query_event_count_matches"]
            and integrity["failed_queries"] == 0
            and integrity["overflow_queries"] == 0
            and integrity["invalid_event_count"] == 0
            and integrity["inconsistent_round_groups"] == 0
        ),
        "multi_release_coverage_ge_25pct": (
            aggregate.get("multi_release_round_fraction") or 0) >= 0.25,
        "dispersion_ge_10us_p50_or_25us_p90": (
            (p50_spread or 0) >= 10_000
            or (p90_spread or 0) >= 25_000
        ),
        "zero_overhead_oracle_ge_8pct_gpu_time": (
            zero_gpu or 0) >= 0.08,
    }
    completion_granularity = summary["measurement"].get(
        "completion_granularity", "unknown")
    parent_granular = (
        "parent" in completion_granularity
        or "tile" in completion_granularity)
    if not checks["integrity_clean"]:
        verdict = "invalid trace"
    elif all(checks.values()):
        verdict = "supports ordered-commit prototype"
    elif not parent_granular:
        verdict = (
            "negative at observed granularity; parent/tile result remains "
            "unmeasured")
    else:
        verdict = "does not support ordered-commit prototype"
    return {
        "verdict": verdict,
        "checks": checks,
        "parent_or_tile_completion_observed": parent_granular,
    }


def analyze(path: Path, overheads_us: tuple[float, ...],
            task_granularity: str, include_rounds: bool) -> dict:
    metadata, queries, grouped_events, input_counts = load_trace(path)
    metadata_head = metadata[-1] if metadata else {}
    natural_tile = max(1, int(metadata_head.get("natural_parent_tile", 1)))
    service_by_query = {
        request_id: query_service_per_parent_ns(query)
        for request_id, query in queries.items()}
    round_rows = []
    for key, events in sorted(grouped_events.items()):
        service = service_by_query.get(key[0])
        if service is None:
            round_rows.append({
                "request_id": key[0],
                "route_attempt": key[1],
                "search_round": key[2],
                "snapshot_attempt": key[3],
                "eligible": False,
                "missing_service_calibration": True,
                "invalid_event_count": 0,
            })
            continue
        round_rows.append(build_round_model(
            key, events, service, natural_tile, task_granularity,
            overheads_us))
    query_rows = aggregate_query_rows(queries, round_rows, overheads_us)
    summary = {
        "schema": 1,
        "input": str(path.resolve()),
        "measurement": {
            "completion_granularity": metadata_head.get(
                "completion_granularity", "unknown"),
            "task_granularity": task_granularity,
            "natural_parent_tile": natural_tile,
            "timestamp_clock": metadata_head.get(
                "timestamp_clock", "unknown"),
            "service_model": (
                "Per-query validation+neighbor_decode+PQ+visited wall time "
                "is distributed linearly over graph parents. This is an "
                "oracle sensitivity model, not measured per-task service."),
            "baseline": (
                "all observable releases, then modeled movable work"),
            "oracle": (
                "one query compute CTA executes modeled work at each "
                "observable release; Beam/search state commits only after "
                "all work"),
            "excluded_from_movable_work": (
                "beam merge, parent selection, expansion budget, "
                "termination, exact rerank"),
        },
        "metadata": metadata,
        "input_counts": input_counts,
        "task_overheads_us": list(overheads_us),
        "integrity": integrity_summary(
            metadata, queries, grouped_events, round_rows),
        "aggregate": aggregate_summary(
            round_rows, query_rows, overheads_us, natural_tile),
        "queries": query_rows,
        "rounds": round_rows if include_rounds else [],
    }
    summary["screening"] = screening(summary)
    return summary


def format_us(value):
    return "n/a" if value is None else f"{value / 1000.0:.2f}"


def format_percent(value):
    return "n/a" if value is None else f"{value * 100.0:.2f}%"


def write_markdown(summary: dict, destination: Path):
    aggregate = summary["aggregate"]
    measurement = summary["measurement"]
    screening_result = summary["screening"]
    ready_tile_10us = aggregate[
        "round_fraction_with_tile_ready_10us_before_tail"]
    handoff_p50_ns = aggregate[
        "completion_to_process_handoff_p50_ns"]
    lines = [
        "# Execute-ready / commit-in-order motivation oracle",
        "",
        f"- Input: `{summary['input']}`",
        f"- Observable completion granularity: "
        f"`{measurement['completion_granularity']}`",
        f"- Modeled task granularity: "
        f"`{measurement['task_granularity']}`",
        f"- Verdict: **{screening_result['verdict']}**",
        "",
        "## Directly observed release dispersion",
        "",
        "| Metric | Result |",
        "|---|---:|",
        f"| Primary rounds | {aggregate['primary_rounds']} |",
        f"| Rounds with >=2 release boundaries | "
        f"{format_percent(aggregate['multi_release_round_fraction'])} |",
        f"| Strict spread P50/P90/P99 | "
        f"{format_us(aggregate['strict_completion_spread_p50_ns'])} / "
        f"{format_us(aggregate['strict_completion_spread_p90_ns'])} / "
        f"{format_us(aggregate['strict_completion_spread_p99_ns'])} us |",
        f"| Natural tile ready >=10 us before tail | "
        f"{format_percent(ready_tile_10us)} |",
        f"| Completion-to-process handoff P50 | "
        f"{format_us(handoff_p50_ns)} us |",
        "",
        "## Release-time oracle",
        "",
        "The oracle moves only validation, decode, PQ scoring, and visited "
        "work. It leaves the authoritative Beam and all search decisions "
        "behind the epoch commit barrier.",
        "",
        "| Queue/state overhead per task | Saved / GPU time P50 | "
        "Saved / RDMA wait P50 | Saved/query P50 |",
        "|---:|---:|---:|---:|",
    ]
    for overhead_us in summary["task_overheads_us"]:
        key = f"{overhead_us:g}us"
        gpu = aggregate[
            "query_strict_saved_over_gpu_by_task_overhead"][key]["p50"]
        wait = aggregate[
            "query_strict_saved_over_rdma_wait_by_task_overhead"][key]["p50"]
        saved = aggregate[
            "query_strict_saved_ns_by_task_overhead"][key]["p50"]
        lines.append(
            f"| {overhead_us:g} us | {format_percent(gpu)} | "
            f"{format_percent(wait)} | {format_us(saved)} us |")
    lines.extend([
        "",
        "## Preregistered screen",
        "",
    ])
    for name, passed in screening_result["checks"].items():
        lines.append(f"- [{'x' if passed else ' '}] `{name}`")
    lines.extend([
        "",
        "## Interpretation limit",
        "",
        measurement["service_model"],
        "",
        "A shard-batch completion timestamp cannot establish parent-level "
        "dispersion inside that shard. A negative shard-granularity verdict "
        "therefore stops a shard-only reorder design, but does not silently "
        "stand in for a parent/tile-signaled experiment. Conversely, "
        "parent-weighted ready area is not wall-clock query time and is never "
        "reported as speedup.",
        "",
    ])
    destination.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", type=Path)
    parser.add_argument(
        "--task-overheads-us",
        type=parse_overheads,
        default=DEFAULT_OVERHEADS_US,
        help="comma-separated queue/state overhead sensitivity values")
    parser.add_argument(
        "--task-granularity",
        choices=("completion", "tile"),
        default="tile")
    parser.add_argument(
        "--include-round-details",
        action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    summary = analyze(
        args.trace, args.task_overheads_us, args.task_granularity,
        args.include_round_details)
    output = args.output or args.trace.with_suffix(
        ".ordered_commit_oracle.summary.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    markdown = output.with_suffix(".md")
    write_markdown(summary, markdown)
    print(output)
    print(markdown)


if __name__ == "__main__":
    main()
