#!/usr/bin/env python3
"""Analyze the query-selective adjacency-transfer motivation trace.

The trace contains two deliberately different upper bounds:

* perfect: exact ADC over every edge in the record, including visited rejects;
* interval/suffix: a query-independent parent/neighbor PQ geometry synopsis.

Only the second is implementable.  Post-visited and final-Beam counters are
reported as a usefulness funnel, never as bytes that can be skipped before the
neighbor IDs have arrived.
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
from typing import Any


GROUP_SIZES = (4, 8, 16, 32)
PREFIX_SIZES = (8, 16, 32, 48, 64)
GROUP_ARRAYS = (
    "total_groups",
    "certificate_needed_groups",
    "post_visited_needed_groups",
    "final_beam_needed_groups",
    "certificate_needed_runs",
    "certificate_first_group_needed_parents",
    "interval_needed_groups",
    "interval_needed_runs",
    "interval_first_group_needed_parents",
)
PREFIX_ARRAYS = (
    "parents_with_tail",
    "perfect_tail_needed_parents",
    "suffix_interval_tail_needed_parents",
    "post_visited_tail_needed_parents",
    "final_beam_tail_needed_parents",
    "total_tail_edges",
    "perfect_tail_needed_edges",
    "suffix_interval_tail_needed_edges",
)


def ratio(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator else None


def pct(value: float | None) -> str:
    return "n/a" if value is None else f"{100.0 * value:.2f}%"


def number(value: float | None, digits: int = 3) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def load_jsonl(path: pathlib.Path) -> tuple[dict[str, Any], list[dict[str, Any]],
                                             list[dict[str, Any]]]:
    metadata: dict[str, Any] = {}
    queries: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as source:
        for line_number, raw in enumerate(source, 1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"{path}:{line_number}: invalid JSON: {error}") from error
            kind = row.get("type")
            if kind == "adjacency_oracle_metadata":
                metadata.update(row)
            elif kind == "query":
                queries.append(row)
            elif kind == "adjacency_oracle":
                events.append(row)
    if not events:
        raise ValueError(f"{path}: no adjacency_oracle events")
    return metadata, queries, events


def load_baseline(path: pathlib.Path | None) -> dict[str, float]:
    if path is None:
        return {}
    with path.open("r", encoding="utf-8") as source:
        source_json = json.load(source)
    gpu = source_json.get("gpu_persistent", {})
    query = source_json.get("query_breakdown", {})
    throughput = source_json.get("throughput", {})
    count = float(query.get("count", 0) or 0)

    def per_query(counter: str) -> float:
        value = float(gpu.get(counter, 0) or 0)
        return value / count if count else 0.0

    return {
        "qps": float(
            throughput.get(
                "effective_query_ops_per_sec",
                throughput.get("query_ops_per_sec", 0)) or 0),
        "query_count": count,
        "graph_reads_per_query": per_query("graph_page_requests"),
        "exact_reads_per_query": per_query("exact_vector_reads"),
        "rdma_ops_per_query": per_query("rdma_read_ops"),
        "rdma_bytes_per_query": per_query("rdma_read_bytes"),
        "query_us": float(gpu.get("average_gpu_query_us", 0) or 0),
        "rdma_issue_us": float(gpu.get("average_gpu_rdma_issue_us", 0) or 0),
        "rdma_wait_us": float(gpu.get("average_gpu_rdma_wait_us", 0) or 0),
        "validation_us": float(
            gpu.get("average_gpu_graph_validation_us", 0) or 0),
        "decode_us": float(
            gpu.get("average_gpu_neighbor_decode_us", 0) or 0),
        "pq_us": float(gpu.get("average_gpu_pq_score_us", 0) or 0),
        "visited_us": float(gpu.get("average_gpu_visited_us", 0) or 0),
    }


def sum_array(events: list[dict[str, Any]], name: str, count: int) -> list[int]:
    output = [0] * count
    for event in events:
        values = event.get(name)
        if not isinstance(values, list) or len(values) != count:
            raise ValueError(
                f"event {event.get('request_id')} has invalid {name}")
        for index, value in enumerate(values):
            output[index] += int(value)
    return output


def aggregate(
    metadata: dict[str, Any],
    queries: list[dict[str, Any]],
    events: list[dict[str, Any]],
    baseline: dict[str, float],
    base_header_bytes: int,
    interval_bytes_per_group: int,
    suffix_bytes_per_parent: int,
) -> dict[str, Any]:
    record_bytes = int(metadata.get("graph_entry_bytes", 832))
    pointer_bytes = int(metadata.get("remote_ptr_bytes", 8))
    parents = sum(int(event["parent_count"]) for event in events)
    edges = sum(int(event["edge_count"]) for event in events)
    invalid = sum(int(event["invalid_decoded_count"]) for event in events)
    dynamic = sum(int(event["dynamic_edge_count"]) for event in events)
    visited = sum(int(event["visited_survivor_count"]) for event in events)
    finite = sum(int(event["finite_scored_count"]) for event in events)
    entered = sum(int(event["new_candidates_in_beam"]) for event in events)
    beam_not_full_events = sum(
        int(event["beam_count_before"]) < int(event["beam_capacity"])
        for event in events)
    violations = sum(
        int(event["interval_lb_violation_count"]) for event in events)
    margins = [
        float(event["minimum_interval_safety_margin"])
        for event in events
        if event.get("minimum_interval_safety_margin") is not None
    ]
    overflows = sum(int(query.get("adjacency_oracle_overflow", 0))
                    for query in queries)
    arrays = {
        name: sum_array(
            events, name,
            len(GROUP_SIZES) if name in GROUP_ARRAYS else len(PREFIX_SIZES))
        for name in GROUP_ARRAYS + PREFIX_ARRAYS
    }
    baseline_sample_bytes = parents * record_bytes
    packed_full_bytes_without_synopsis = (
        parents * base_header_bytes + edges * pointer_bytes)
    padding_only_payload_savings = (
        1.0 - packed_full_bytes_without_synopsis / baseline_sample_bytes
        if baseline_sample_bytes else None)

    group_results: list[dict[str, Any]] = []
    for lane, group_size in enumerate(GROUP_SIZES):
        total = arrays["total_groups"][lane]
        perfect_needed = arrays["certificate_needed_groups"][lane]
        interval_needed = arrays["interval_needed_groups"][lane]
        perfect_saved_groups = total - perfect_needed
        interval_saved_groups = total - interval_needed
        interval_header_bytes = (
            parents * base_header_bytes +
            total * interval_bytes_per_group)

        def group_model(needed: int, runs: int, first_needed: int) -> dict[str, Any]:
            strict_bytes = (
                interval_header_bytes +
                needed * group_size * pointer_bytes)
            certificate_saved_bytes = (
                total - needed) * group_size * pointer_bytes
            fused_bytes = (
                interval_header_bytes +
                parents * group_size * pointer_bytes +
                (needed - first_needed) * group_size * pointer_bytes)
            return {
                "strict_payload_bytes": strict_bytes,
                "strict_payload_ratio": ratio(
                    strict_bytes, baseline_sample_bytes),
                "strict_payload_savings": (
                    1.0 - strict_bytes / baseline_sample_bytes
                    if baseline_sample_bytes else None),
                "certificate_incremental_bytes_saved":
                    certificate_saved_bytes,
                "certificate_incremental_savings_vs_current": ratio(
                    certificate_saved_bytes, baseline_sample_bytes),
                "coalesced_wqes_per_parent": ratio(parents + runs, parents),
                "group_wqes_per_parent": ratio(parents + needed, parents),
                "fused_first_payload_bytes": fused_bytes,
                "fused_first_payload_savings": (
                    1.0 - fused_bytes / baseline_sample_bytes
                    if baseline_sample_bytes else None),
                "fused_first_wqes_per_parent": ratio(
                    parents + runs - first_needed, parents),
            }

        perfect_model = group_model(
            perfect_needed,
            arrays["certificate_needed_runs"][lane],
            arrays["certificate_first_group_needed_parents"][lane])
        interval_model = group_model(
            interval_needed,
            arrays["interval_needed_runs"][lane],
            arrays["interval_first_group_needed_parents"][lane])
        group_results.append({
            "group_size": group_size,
            "total_groups": total,
            "perfect_needed_groups": perfect_needed,
            "interval_needed_groups": interval_needed,
            "post_visited_needed_groups":
                arrays["post_visited_needed_groups"][lane],
            "final_beam_needed_groups":
                arrays["final_beam_needed_groups"][lane],
            "perfect_group_skip_rate": ratio(perfect_saved_groups, total),
            "interval_group_skip_rate": ratio(interval_saved_groups, total),
            "interval_retention_of_perfect_skip": ratio(
                interval_saved_groups, perfect_saved_groups),
            "metadata_bytes": interval_header_bytes,
            "metadata_to_baseline_ratio": ratio(
                interval_header_bytes, baseline_sample_bytes),
            "perfect_model": perfect_model,
            "interval_model": interval_model,
        })

    prefix_results: list[dict[str, Any]] = []
    for lane, prefix_size in enumerate(PREFIX_SIZES):
        total_tail_edges = arrays["total_tail_edges"][lane]
        prefix_edges = edges - total_tail_edges
        parents_with_tail = arrays["parents_with_tail"][lane]
        perfect_tail_parents = arrays["perfect_tail_needed_parents"][lane]
        suffix_tail_parents = arrays[
            "suffix_interval_tail_needed_parents"][lane]
        perfect_tail_edges = arrays["perfect_tail_needed_edges"][lane]
        suffix_tail_edges = arrays[
            "suffix_interval_tail_needed_edges"][lane]
        header_bytes = parents * (base_header_bytes + suffix_bytes_per_parent)

        def prefix_model(
            tail_parents: int, tail_edges: int, event_field: str
        ) -> dict[str, Any]:
            transferred_edges = prefix_edges + tail_edges
            payload_bytes = header_bytes + transferred_edges * pointer_bytes
            skipped_live_edges = total_tail_edges - tail_edges
            always_transfer_live_bytes = header_bytes + edges * pointer_bytes
            return {
                "tail_parent_count": tail_parents,
                "tail_parent_rate": ratio(tail_parents, parents),
                "tail_rate_among_parents_with_tail": ratio(
                    tail_parents, parents_with_tail),
                "tail_free_parent_rate": (
                    1.0 - tail_parents / parents if parents else None),
                "transferred_edges": transferred_edges,
                "edge_transfer_ratio": ratio(transferred_edges, edges),
                "skipped_live_edges": skipped_live_edges,
                "live_edge_skip_rate": ratio(skipped_live_edges, edges),
                "tail_edge_skip_rate": ratio(
                    skipped_live_edges, total_tail_edges),
                "payload_bytes": payload_bytes,
                "payload_ratio": ratio(payload_bytes, baseline_sample_bytes),
                "payload_savings": (
                    1.0 - payload_bytes / baseline_sample_bytes
                    if baseline_sample_bytes else None),
                "certificate_incremental_bytes_saved":
                    skipped_live_edges * pointer_bytes,
                "certificate_incremental_savings_vs_current": ratio(
                    skipped_live_edges * pointer_bytes,
                    baseline_sample_bytes),
                "certificate_incremental_savings_vs_packed": ratio(
                    skipped_live_edges * pointer_bytes,
                    always_transfer_live_bytes),
                "wqes_per_parent": ratio(parents + tail_parents, parents),
                "second_stage_chunk_rate": ratio(
                    sum(
                        int(event[event_field][lane]) > 0
                        for event in events),
                    len(events)),
            }

        perfect_model = prefix_model(
            perfect_tail_parents, perfect_tail_edges,
            "perfect_tail_needed_parents")
        suffix_model = prefix_model(
            suffix_tail_parents, suffix_tail_edges,
            "suffix_interval_tail_needed_parents")
        perfect_saved_edges = total_tail_edges - perfect_tail_edges
        suffix_saved_edges = total_tail_edges - suffix_tail_edges
        prefix_results.append({
            "prefix_size": prefix_size,
            "parents_with_tail": parents_with_tail,
            "total_tail_edges": total_tail_edges,
            "header_bytes": header_bytes,
            "metadata_to_baseline_ratio": ratio(
                parents * suffix_bytes_per_parent, baseline_sample_bytes),
            "perfect_model": perfect_model,
            "suffix_model": suffix_model,
            "suffix_retention_of_perfect_skip": ratio(
                suffix_saved_edges, perfect_saved_edges),
            "post_visited_tail_needed_parents":
                arrays["post_visited_tail_needed_parents"][lane],
            "final_beam_tail_needed_parents":
                arrays["final_beam_tail_needed_parents"][lane],
        })

    # Add a transport and query-time roofline when a normal, trace-off baseline
    # report is supplied.  It is intentionally an upper bound: the extra
    # dependent tail RTT and checksum redesign cost are not subtracted.
    if baseline:
        graph_reads = baseline.get("graph_reads_per_query", 0)
        exact_reads = baseline.get("exact_reads_per_query", 0)
        current_ops = baseline.get("rdma_ops_per_query", 0)
        current_bytes = baseline.get("rdma_bytes_per_query", 0)
        graph_bytes = graph_reads * record_bytes
        non_graph_bytes = max(0.0, current_bytes - graph_bytes)
        phase_scalable = (
            baseline.get("rdma_issue_us", 0) +
            baseline.get("rdma_wait_us", 0) +
            baseline.get("validation_us", 0))
        edge_scalable = (
            baseline.get("decode_us", 0) +
            baseline.get("pq_us", 0) +
            baseline.get("visited_us", 0))
        for result in prefix_results:
            model = result["suffix_model"]
            payload_ratio = model["payload_ratio"] or 1.0
            edge_ratio = model["edge_transfer_ratio"] or 1.0
            wqe_ratio = model["wqes_per_parent"] or 1.0
            proposed_ops = exact_reads + graph_reads * wqe_ratio
            proposed_bytes = non_graph_bytes + graph_bytes * payload_ratio
            observed_wqe_rate = baseline.get("qps", 0) * current_ops
            required_wqe_rate_20pct = (
                baseline.get("qps", 0) * 1.2 * proposed_ops)
            payload_only_saved_us = (
                phase_scalable * (1.0 - payload_ratio) +
                edge_scalable * (1.0 - edge_ratio))
            certificate_byte_fraction = (
                model["certificate_incremental_savings_vs_current"] or 0.0)
            certificate_edge_fraction = (
                model["live_edge_skip_rate"] or 0.0)
            certificate_only_saved_us = (
                phase_scalable * certificate_byte_fraction +
                edge_scalable * certificate_edge_fraction)
            model["baseline_total_wqes_per_query"] = current_ops
            model["proposed_total_wqes_per_query"] = proposed_ops
            model["wqe_count_ratio"] = ratio(proposed_ops, current_ops)
            model["baseline_total_bytes_per_query"] = current_bytes
            model["proposed_total_bytes_per_query"] = proposed_bytes
            model["total_byte_ratio"] = ratio(proposed_bytes, current_bytes)
            model["observed_wqe_rate"] = observed_wqe_rate
            model["required_wqe_rate_for_20pct_qps"] = required_wqe_rate_20pct
            model["required_wqe_rate_multiplier_for_20pct_qps"] = ratio(
                required_wqe_rate_20pct, observed_wqe_rate)
            model["payload_only_query_time_upper_bound_saved_us"] = (
                payload_only_saved_us)
            model["payload_only_query_time_upper_bound_gain"] = ratio(
                payload_only_saved_us, baseline.get("query_us", 0))
            model["certificate_only_query_time_upper_bound_saved_us"] = (
                certificate_only_saved_us)
            model["certificate_only_query_time_upper_bound_gain"] = ratio(
                certificate_only_saved_us, baseline.get("query_us", 0))

    # Apply the fail-fast gates to the best implementable suffix choice.
    viable: list[dict[str, Any]] = []
    for result in prefix_results:
        perfect = result["perfect_model"]
        suffix = result["suffix_model"]
        gates = {
            "perfect_live_edge_skip_ge_70pct":
                (perfect["live_edge_skip_rate"] or -1) >= 0.70,
            "suffix_retains_ge_80pct_perfect_skip":
                (result["suffix_retention_of_perfect_skip"] or -1) >= 0.80,
            "tail_free_parents_ge_92pct":
                (suffix["tail_free_parent_rate"] or -1) >= 0.92,
            "metadata_le_10pct":
                (result["metadata_to_baseline_ratio"] or math.inf) <= 0.10,
            "zero_geometric_lb_violations": violations == 0,
        }
        if "payload_only_query_time_upper_bound_gain" in suffix:
            gates["certificate_only_query_gain_ge_20pct"] = (
                (suffix["certificate_only_query_time_upper_bound_gain"] or -1)
                >= 0.20)
            gates["20pct_qps_wqe_rate_le_1p10x_observed"] = (
                (suffix[
                    "required_wqe_rate_multiplier_for_20pct_qps"] or math.inf)
                <= 1.10)
        result["gates"] = gates
        result["passes_all_available_gates"] = all(gates.values())
        if result["passes_all_available_gates"]:
            viable.append(result)

    best = max(
        prefix_results,
        key=lambda result: (
            result["suffix_model"].get(
                "certificate_only_query_time_upper_bound_gain",
                result["suffix_model"]["live_edge_skip_rate"] or -math.inf),
            -(result["suffix_model"]["wqes_per_parent"] or math.inf)),
    )
    return {
        "schema": 1,
        "trace": {
            "query_records": len(queries),
            "events": len(events),
            "parents": parents,
            "edges": edges,
            "record_bytes": record_bytes,
            "pointer_bytes": pointer_bytes,
            "beam_not_full_event_count": beam_not_full_events,
            "beam_not_full_event_rate": ratio(
                beam_not_full_events, len(events)),
            "overflow_count": overflows,
        },
        "edge_funnel": {
            "decoded_edges": edges,
            "invalid_decoded_edges": invalid,
            "dynamic_edges": dynamic,
            "visited_survivors": visited,
            "finite_scored": finite,
            "entered_beam": entered,
            "visited_survival_rate": ratio(visited, edges),
            "finite_score_rate": ratio(finite, edges),
            "beam_entry_rate": ratio(entered, edges),
        },
        "fixed_record_padding": {
            "packed_full_bytes_without_synopsis":
                packed_full_bytes_without_synopsis,
            "padding_only_payload_savings": padding_only_payload_savings,
            "note": (
                "This is a record-layout opportunity, not a query-dependent "
                "certificate benefit."),
        },
        "certificate_safety": {
            "interval_lb_violation_count": violations,
            "minimum_interval_safety_margin": min(margins) if margins else None,
        },
        "layout_assumptions": {
            "base_header_bytes": base_header_bytes,
            "interval_bytes_per_group": interval_bytes_per_group,
            "suffix_bytes_per_parent": suffix_bytes_per_parent,
            "checksum_and_version_redesign_bytes_included": False,
            "second_stage_rtt_cost_included": False,
        },
        "group_results": group_results,
        "prefix_results": prefix_results,
        "baseline": baseline,
        "decision": {
            "verdict": (
                "PASS_TO_LAYOUT_PROTOTYPE" if viable
                else "STOP_NO_LARGE_BENEFIT_EVIDENCE"),
            "viable_prefix_sizes": [
                result["prefix_size"] for result in viable],
            "best_observed_prefix_size": best["prefix_size"],
            "failed_gates_for_best": [
                name for name, passed in best["gates"].items() if not passed],
            "note": (
                "PASS only authorizes a versioned partial-record layout "
                "prototype and a size-dependent two-stage RDMA microbenchmark; "
                "it is not an end-to-end performance claim."),
        },
    }


def markdown(report: dict[str, Any]) -> str:
    trace = report["trace"]
    funnel = report["edge_funnel"]
    padding = report["fixed_record_padding"]
    safety = report["certificate_safety"]
    decision = report["decision"]
    lines = [
        "# Query-selective adjacency transfer motivation",
        "",
        f"Verdict: **{decision['verdict']}**",
        "",
        "This is a fail-fast observation test. The perfect oracle includes every "
        "edge (including visited rejects); post-visited/final-Beam counters are "
        "diagnostic and are not credited as pre-transfer savings.",
        "",
        "## Sample",
        "",
        f"- Queries with trace records: {trace['query_records']}",
        f"- Score chunks: {trace['events']}",
        f"- Parents / edges: {trace['parents']} / {trace['edges']}",
        f"- Beam-not-full chunks: {trace['beam_not_full_event_count']} "
        f"({pct(trace['beam_not_full_event_rate'])})",
        f"- Trace overflow: {trace['overflow_count']}",
        "",
        "## Edge usefulness funnel",
        "",
        "| decoded | invalid | dynamic | visited survivors | finite scored | entered Beam |",
        "|---:|---:|---:|---:|---:|---:|",
        f"| {funnel['decoded_edges']} | {funnel['invalid_decoded_edges']} | "
        f"{funnel['dynamic_edges']} | {funnel['visited_survivors']} "
        f"({pct(funnel['visited_survival_rate'])}) | "
        f"{funnel['finite_scored']} ({pct(funnel['finite_score_rate'])}) | "
        f"{funnel['entered_beam']} ({pct(funnel['beam_entry_rate'])}) |",
        "",
        "The current fixed 832-byte record also contains substantial unused "
        "degree capacity. Packing only the live edges would save "
        f"**{pct(padding['padding_only_payload_savings'])}** of sampled graph "
        "payload before any query-dependent certificate. This is reported "
        "separately as a layout effect.",
        "",
        "## Prefix + certified suffix (recommended one-sided layout)",
        "",
        "| prefix | perfect live-edge skip | suffix live-edge skip | retain perfect | "
        "tail-free parents | WQE/parent | chunks needing stage 2 | certificate-only query gain upper bound |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for result in report["prefix_results"]:
        perfect = result["perfect_model"]
        suffix = result["suffix_model"]
        lines.append(
            f"| {result['prefix_size']} | {pct(perfect['live_edge_skip_rate'])} | "
            f"{pct(suffix['live_edge_skip_rate'])} | "
            f"{pct(result['suffix_retention_of_perfect_skip'])} | "
            f"{pct(suffix['tail_free_parent_rate'])} | "
            f"{number(suffix['wqes_per_parent'])} | "
            f"{pct(suffix['second_stage_chunk_rate'])} | "
            f"{pct(suffix.get('certificate_only_query_time_upper_bound_gain'))} |")
    lines += [
        "",
        "## Arbitrary groups (diagnostic; requires remote runs)",
        "",
        "| group | perfect skip | interval skip | retain perfect | interval certificate bytes saved | "
        "coalesced WQE/parent |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for result in report["group_results"]:
        model = result["interval_model"]
        lines.append(
            f"| {result['group_size']} | "
            f"{pct(result['perfect_group_skip_rate'])} | "
            f"{pct(result['interval_group_skip_rate'])} | "
            f"{pct(result['interval_retention_of_perfect_skip'])} | "
            f"{pct(model['certificate_incremental_savings_vs_current'])} | "
            f"{number(model['coalesced_wqes_per_parent'])} |")
    lines += [
        "",
        "## Safety and gate",
        "",
        f"- Geometric lower-bound violations: "
        f"{safety['interval_lb_violation_count']}",
        f"- Minimum measured safety margin: "
        f"{number(safety['minimum_interval_safety_margin'], 7)}",
        f"- Best observed prefix: {decision['best_observed_prefix_size']}",
        f"- Failed gates: "
        f"{', '.join(decision['failed_gates_for_best']) or 'none'}",
        "",
        "The model does not charge the second dependent RDMA RTT, the new "
        "version/checksum layout, or certificate evaluation. Therefore any "
        "reported query-time gain is an optimistic upper bound.",
        "",
    ]
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", type=pathlib.Path)
    parser.add_argument("--baseline-json", type=pathlib.Path)
    parser.add_argument("--output-json", type=pathlib.Path)
    parser.add_argument("--output-markdown", type=pathlib.Path)
    parser.add_argument("--base-header-bytes", type=int, default=16)
    parser.add_argument("--interval-bytes-per-group", type=int, default=8)
    parser.add_argument("--suffix-bytes-per-parent", type=int, default=4)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    metadata, queries, events = load_jsonl(args.trace)
    report = aggregate(
        metadata, queries, events, load_baseline(args.baseline_json),
        args.base_header_bytes, args.interval_bytes_per_group,
        args.suffix_bytes_per_parent)
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
