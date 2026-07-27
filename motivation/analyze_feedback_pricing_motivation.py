#!/usr/bin/env python3
"""Analyze sampled Beam-turnover traces from the fixed-C16 baseline.

This is deliberately an observation/oracle analysis, not a performance model.
It asks two falsifiable questions:

1. Does old/new Beam turnover predict whether a candidate is subsequently
   selected and whether a selected parent contributes a child to the Beam?
2. Even with perfect knowledge of the last immediately productive parent, is
   there enough removable suffix work to pay for another Beam feedback round?
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


WIDTHS = (1, 4, 8, 12, 16)


def safe_ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (
        position - lower
    )


def load_trace(path: Path) -> tuple[dict[int, dict[str, Any]],
                                    dict[int, list[dict[str, Any]]]]:
    queries: dict[int, dict[str, Any]] = {}
    events: dict[int, list[dict[str, Any]]] = defaultdict(list)
    with path.open("r", encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            record = json.loads(line)
            request_id = int(record.get("request_id", -1))
            if record.get("type") == "query":
                queries[request_id] = record
            elif (record.get("type") == "adjacency_oracle" and
                  int(record.get("schema", 0)) >= 2):
                events[request_id].append(record)
    for query_events in events.values():
        query_events.sort(
            key=lambda event: (
                int(event["search_round"]), int(event.get("chunk_begin", 0))
            )
        )
    return queries, events


def turnover_bin(value: float) -> str:
    if value < 0.25:
        return "[0,.25)"
    if value < 0.50:
        return "[.25,.50)"
    if value < 0.75:
        return "[.50,.75)"
    return "[.75,1]"


def choose_oracle_width(parent_count: int, productive_mask: int) -> int:
    if parent_count <= 0:
        return 0
    last_productive = 0
    for rank in range(parent_count):
        if productive_mask & (1 << rank):
            last_productive = rank + 1
    required = max(1, last_productive)
    for width in WIDTHS:
        if width >= required:
            return min(width, parent_count)
    return parent_count


def analyze(path: Path, baseline_path: Path | None = None) -> dict[str, Any]:
    queries, grouped = load_trace(path)
    outcome_counts: dict[tuple[str, str, str], list[int]] = defaultdict(
        lambda: [0, 0]
    )
    productive_counts: dict[tuple[str, str], list[int]] = defaultdict(
        lambda: [0, 0]
    )
    rank_productive: dict[int, list[int]] = defaultdict(lambda: [0, 0])
    round_rows: list[dict[str, Any]] = []
    query_rows: list[dict[str, Any]] = []

    for request_id, query_events in grouped.items():
        query = queries.get(request_id, {})
        selected_future: list[set[int]] = [
            set(map(int, event.get("selected_handles", [])))
            for event in query_events
        ]
        total_baseline_cycles = 0.0
        total_oracle_cycles = 0.0
        total_saved_parents = 0
        total_parents = 0
        for index, event in enumerate(query_events):
            frontier = list(map(int, event.get("frontier_handles", [])))
            frontier_count = min(
                int(event.get("frontier_count", len(frontier))), len(frontier)
            )
            frontier = frontier[:frontier_count]
            new_mask = int(event.get("frontier_new_mask", 0))
            top = min(16, frontier_count)
            top_new = sum(bool(new_mask & (1 << rank)) for rank in range(top))
            turnover = safe_ratio(top_new, top)
            bucket = turnover_bin(turnover)

            later_selected = set().union(*selected_future[index + 1:]) \
                if index + 1 < len(selected_future) else set()
            next_selected = selected_future[index + 1] \
                if index + 1 < len(selected_future) else set()
            for rank, handle in enumerate(frontier):
                origin = "new" if new_mask & (1 << rank) else "old"
                key = (origin, bucket, "0-15" if rank < 16 else "16-31")
                outcome_counts[key][1] += 1
                if handle in later_selected:
                    outcome_counts[key][0] += 1

            selected = list(map(int, event.get("selected_handles", [])))
            parent_count = min(
                int(event.get("parent_count", len(selected))), len(selected)
            )
            productive_mask = int(event.get("selected_productive_mask", 0))
            previous_origin: dict[int, str] = {}
            previous_turnover = "startup"
            if index:
                previous = query_events[index - 1]
                previous_frontier = list(
                    map(int, previous.get("frontier_handles", []))
                )
                previous_new_mask = int(previous.get("frontier_new_mask", 0))
                previous_top = min(16, len(previous_frontier))
                previous_turnover = turnover_bin(safe_ratio(
                    sum(bool(previous_new_mask & (1 << rank))
                        for rank in range(previous_top)),
                    previous_top,
                ))
                for rank, handle in enumerate(previous_frontier):
                    previous_origin[handle] = (
                        "new" if previous_new_mask & (1 << rank) else "old"
                    )
            for rank, handle in enumerate(selected[:parent_count]):
                productive = bool(productive_mask & (1 << rank))
                origin = previous_origin.get(handle, "startup")
                productive_counts[(origin, previous_turnover)][1] += 1
                rank_productive[rank][1] += 1
                if productive:
                    productive_counts[(origin, previous_turnover)][0] += 1
                    rank_productive[rank][0] += 1

            graph_cycles = float(event.get("round_graph_cycles", 0))
            score_cycles = float(event.get("round_score_cycles", 0))
            beam_cycles = float(event.get("round_beam_cycles", 0))
            baseline_cycles = graph_cycles + score_cycles + beam_cycles
            oracle_width = choose_oracle_width(parent_count, productive_mask)
            kept_fraction = safe_ratio(oracle_width, parent_count)
            # Optimistic suffix oracle: graph/score work scales with kept
            # parents.  Conservatively charge one full observed merge for each
            # extra feedback round needed to cover the same parent prefix.
            extra_feedback_rounds = (
                max(0, math.ceil(parent_count / oracle_width) - 1)
                if oracle_width else 0
            )
            oracle_cycles = (
                (graph_cycles + score_cycles) * kept_fraction +
                beam_cycles * (1 + extra_feedback_rounds)
            )
            total_baseline_cycles += baseline_cycles
            total_oracle_cycles += oracle_cycles
            total_saved_parents += parent_count - oracle_width
            total_parents += parent_count
            round_rows.append({
                "request_id": request_id,
                "round": int(event["search_round"]),
                "parent_count": parent_count,
                "productive_count": int(
                    (productive_mask & ((1 << parent_count) - 1)).bit_count()
                ),
                "oracle_width": oracle_width,
                "turnover_top16": turnover,
                "next_selected_from_frontier": sum(
                    handle in next_selected for handle in frontier
                ),
                "baseline_cycles": baseline_cycles,
                "oracle_cycles": oracle_cycles,
            })

        gpu_cycles = float(query.get("gpu_cycles", 0))
        estimated_query_cycles = (
            gpu_cycles - total_baseline_cycles + total_oracle_cycles
            if gpu_cycles else total_oracle_cycles
        )
        query_rows.append({
            "request_id": request_id,
            "rounds": len(query_events),
            "parents": total_parents,
            "saved_parents": total_saved_parents,
            "baseline_modeled_cycles": total_baseline_cycles,
            "oracle_modeled_cycles": total_oracle_cycles,
            "gpu_cycles": gpu_cycles,
            "estimated_query_cycles": estimated_query_cycles,
        })

    def rate_rows(source: dict[tuple[str, ...], list[int]]) -> list[dict[str, Any]]:
        return [
            {
                "origin": key[0],
                "turnover_bin": key[1],
                **({"rank_band": key[2]} if len(key) > 2 else {}),
                "positive": counts[0],
                "total": counts[1],
                "rate": safe_ratio(counts[0], counts[1]),
            }
            for key, counts in sorted(source.items())
        ]

    total_modeled = sum(row["baseline_modeled_cycles"] for row in query_rows)
    oracle_modeled = sum(row["oracle_modeled_cycles"] for row in query_rows)
    parent_total = sum(row["parents"] for row in query_rows)
    saved_total = sum(row["saved_parents"] for row in query_rows)
    turnovers = [row["turnover_top16"] for row in round_rows]
    modeled_reduction = safe_ratio(
        total_modeled - oracle_modeled, total_modeled
    )
    production_projection: dict[str, Any] = {}
    if baseline_path is not None:
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
        gpu = baseline["gpu_persistent"]
        controlled_us = (
            float(gpu["average_gpu_graph_us"]) +
            float(gpu["average_gpu_score_us"]) +
            float(gpu["average_gpu_beam_us"])
        )
        total_us = float(gpu["average_gpu_query_us"])
        projected_reduction = modeled_reduction * safe_ratio(
            controlled_us, total_us
        )
        baseline_qps = float(
            baseline["throughput"]["effective_query_ops_per_sec"]
        )
        production_projection = {
            "baseline_json": str(baseline_path),
            "baseline_qps": baseline_qps,
            "baseline_gpu_query_us": total_us,
            "controlled_gpu_us": controlled_us,
            "controlled_gpu_fraction": safe_ratio(controlled_us, total_us),
            "projected_end_to_end_reduction": projected_reduction,
            "projected_speedup": (
                1.0 / (1.0 - projected_reduction)
                if projected_reduction < 1.0 else math.inf
            ),
            "projected_qps": (
                baseline_qps / (1.0 - projected_reduction)
                if projected_reduction < 1.0 else math.inf
            ),
        }

    return {
        "trace": str(path),
        "sampled_queries": len(query_rows),
        "sampled_rounds": len(round_rows),
        "turnover_top16": {
            "mean": statistics.fmean(turnovers) if turnovers else 0.0,
            "p50": percentile(turnovers, 0.50),
            "p90": percentile(turnovers, 0.90),
        },
        "frontier_eventual_selection": rate_rows(outcome_counts),
        "selected_parent_productivity": rate_rows(productive_counts),
        "productivity_by_selected_rank": [
            {
                "rank": rank,
                "productive": counts[0],
                "total": counts[1],
                "rate": safe_ratio(counts[0], counts[1]),
            }
            for rank, counts in sorted(rank_productive.items())
        ],
        "productive_suffix_oracle": {
            "parents": parent_total,
            "removable_suffix_parents": saved_total,
            "removable_parent_fraction": safe_ratio(saved_total, parent_total),
            "baseline_modeled_cycles": total_modeled,
            "oracle_modeled_cycles_with_merge_toll": oracle_modeled,
            "modeled_round_work_reduction": modeled_reduction,
            "note": (
                "Optimistic upper bound: omitted parents are known not to "
                "change the immediate Beam; it does not replay changed visited "
                "or future traversal state, and it does not charge extra RDMA "
                "wait for smaller batches."
            ),
        },
        "production_projection": production_projection,
        "rounds": round_rows,
    }


def markdown(result: dict[str, Any]) -> str:
    oracle = result["productive_suffix_oracle"]
    lines = [
        "# Feedback-Priced Expansion Motivation",
        "",
        f"- sampled queries: {result['sampled_queries']}",
        f"- sampled rounds: {result['sampled_rounds']}",
        "- top-16 Beam turnover mean/P50/P90: "
        f"{result['turnover_top16']['mean']:.3f} / "
        f"{result['turnover_top16']['p50']:.3f} / "
        f"{result['turnover_top16']['p90']:.3f}",
        "",
        "## Candidate fate by origin and turnover",
        "",
        "| origin | turnover | rank | eventually selected | samples |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in result["frontier_eventual_selection"]:
        lines.append(
            f"| {row['origin']} | {row['turnover_bin']} | "
            f"{row['rank_band']} | "
            f"{row['rate']:.3%} | {row['total']} |"
        )
    lines += [
        "",
        "## Selected-parent immediate productivity",
        "",
        "| prior origin | prior turnover | productive | samples |",
        "|---|---:|---:|---:|",
    ]
    for row in result["selected_parent_productivity"]:
        lines.append(
            f"| {row['origin']} | {row['turnover_bin']} | "
            f"{row['rate']:.3%} | {row['total']} |"
        )
    lines += [
        "",
        "## Productive-suffix oracle",
        "",
        f"- removable selected-parent suffix: "
        f"{oracle['removable_parent_fraction']:.3%}",
        f"- modeled round-work reduction after charging extra merge toll: "
        f"{oracle['modeled_round_work_reduction']:.3%}",
        f"> {oracle['note']}",
    ]
    projection = result["production_projection"]
    if projection:
        lines += [
            "",
            "## Projection on the uninstrumented C16 baseline",
            "",
            f"- controlled GPU fraction: "
            f"{projection['controlled_gpu_fraction']:.3%}",
            f"- projected end-to-end change: "
            f"{projection['projected_end_to_end_reduction']:.3%}",
            f"- projected QPS: {projection['projected_qps']:.1f} "
            f"(baseline {projection['baseline_qps']:.1f})",
        ]
    lines += [
        "",
        "Gate: if this deliberately optimistic production projection is below "
        "15%, adaptive batching does not have enough headroom to justify "
        "another controller.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", type=Path)
    parser.add_argument("--baseline-json", type=Path)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-markdown", type=Path)
    args = parser.parse_args()
    result = analyze(args.trace, args.baseline_json)
    rendered = markdown(result)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(result, indent=2) + "\n", encoding="utf-8"
        )
    if args.output_markdown:
        args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
        args.output_markdown.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
