#!/usr/bin/env python3
"""Estimate the byte opportunity of live-size adjacency extents.

This analyzer is intentionally independent of the query-selective adjacency
certificate analyzer.  It credits only record-layout facts that are known once
an adjacency record is addressed:

* ``ideal_live_prefix`` stores the record header and exactly the live handles;
* ``extent_8`` rounds each record's live handles to an eight-handle class;
* ``continuation`` shows the dependent-read cost when the extent class is not
  available before the first READ.

The trace already contains the sufficient statistics.  ``edge_count`` is the
number of live handles, ``total_groups[1]`` is the exact number of eight-edge
groups, and the prefix arrays report how many parents and edges remain after
each measured prefix.  No post-visited or final-Beam information is used.
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
EXTENT_EDGES = 8


def ratio(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator else None


def pct(value: float | None) -> str:
    return "n/a" if value is None else f"{100.0 * value:.2f}%"


def number(value: float | None, digits: int = 3) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def load_jsonl(
    path: pathlib.Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
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
        report = json.load(source)
    gpu = report.get("gpu_persistent", {})
    query = report.get("query_breakdown", {})
    count = float(
        query.get("count", 0)
        or gpu.get("queries_completed", 0)
        or 0)

    def per_query(counter: str) -> float:
        value = float(gpu.get(counter, 0) or 0)
        return value / count if count else 0.0

    return {
        "query_count": count,
        "graph_reads_per_query": per_query("graph_page_requests"),
        "rdma_ops_per_query": per_query("rdma_read_ops"),
        "rdma_bytes_per_query": per_query("rdma_read_bytes"),
    }


def require_array(
    event: dict[str, Any], name: str, expected: int,
) -> list[int]:
    values = event.get(name)
    if not isinstance(values, list) or len(values) != expected:
        raise ValueError(
            f"event request={event.get('request_id')} "
            f"round={event.get('search_round')} has invalid {name}")
    return [int(value) for value in values]


def layout_model(payload_bytes: int, fixed_bytes: int) -> dict[str, Any]:
    return {
        "payload_bytes": payload_bytes,
        "payload_ratio": ratio(payload_bytes, fixed_bytes),
        "payload_savings": (
            1.0 - payload_bytes / fixed_bytes if fixed_bytes else None),
    }


def aggregate(
    metadata: dict[str, Any],
    queries: list[dict[str, Any]],
    events: list[dict[str, Any]],
    baseline: dict[str, float] | None = None,
    header_bytes: int = 16,
    extent_edges: int = EXTENT_EDGES,
) -> dict[str, Any]:
    if header_bytes < 0:
        raise ValueError("header_bytes must be non-negative")
    if extent_edges <= 0:
        raise ValueError("extent_edges must be positive")
    if extent_edges not in GROUP_SIZES:
        raise ValueError(
            f"extent_edges={extent_edges} is absent from trace group sizes")

    baseline = baseline or {}
    record_bytes = int(metadata.get("graph_entry_bytes", 832))
    pointer_bytes = int(metadata.get("remote_ptr_bytes", 8))
    if record_bytes <= 0 or pointer_bytes <= 0:
        raise ValueError("record and RemotePtr sizes must be positive")

    extent_lane = GROUP_SIZES.index(extent_edges)
    parents = 0
    edges = 0
    extent_count = 0
    tail_parents = [0] * len(PREFIX_SIZES)
    tail_edges = [0] * len(PREFIX_SIZES)
    chunks_with_tail = [0] * len(PREFIX_SIZES)

    for event in events:
        event_parents = int(event.get("parent_count", -1))
        event_edges = int(event.get("edge_count", -1))
        if event_parents < 0 or event_edges < 0:
            raise ValueError(
                f"event request={event.get('request_id')} has negative counts")
        groups = require_array(event, "total_groups", len(GROUP_SIZES))
        parent_tails = require_array(
            event, "parents_with_tail", len(PREFIX_SIZES))
        edge_tails = require_array(
            event, "total_tail_edges", len(PREFIX_SIZES))
        if any(value < 0 for value in groups + parent_tails + edge_tails):
            raise ValueError(
                f"event request={event.get('request_id')} has negative arrays")
        if any(value > event_parents for value in parent_tails):
            raise ValueError(
                f"event request={event.get('request_id')} has too many "
                "parents_with_tail")
        if any(value > event_edges for value in edge_tails):
            raise ValueError(
                f"event request={event.get('request_id')} has too many "
                "total_tail_edges")
        if any(
            parent_tails[index] < parent_tails[index + 1]
            or edge_tails[index] < edge_tails[index + 1]
            for index in range(len(PREFIX_SIZES) - 1)
        ):
            raise ValueError(
                f"event request={event.get('request_id')} has non-monotone "
                "prefix tails")

        parents += event_parents
        edges += event_edges
        extent_count += groups[extent_lane]
        for lane in range(len(PREFIX_SIZES)):
            tail_parents[lane] += parent_tails[lane]
            tail_edges[lane] += edge_tails[lane]
            chunks_with_tail[lane] += int(parent_tails[lane] > 0)

    fixed_bytes = parents * record_bytes
    live_bytes = parents * header_bytes + edges * pointer_bytes
    extent_bytes = (
        parents * header_bytes
        + extent_count * extent_edges * pointer_bytes)
    fixed = layout_model(fixed_bytes, fixed_bytes)
    ideal = layout_model(live_bytes, fixed_bytes)
    extent = layout_model(extent_bytes, fixed_bytes)
    ideal["bytes_per_parent"] = ratio(live_bytes, parents)
    extent.update({
        "bytes_per_parent": ratio(extent_bytes, parents),
        "extent_count": extent_count,
        "extents_per_parent": ratio(extent_count, parents),
        "rounding_bytes": extent_bytes - live_bytes,
        "rounding_overhead_vs_ideal": ratio(
            extent_bytes - live_bytes, live_bytes),
        # A one-shot extent requires its length class to be available before
        # posting the READ (for example in a tagged handle or local metadata).
        "one_shot_wqes_per_parent": 1.0 if parents else None,
    })
    fixed["bytes_per_parent"] = float(record_bytes) if parents else None

    continuation_results: list[dict[str, Any]] = []
    for lane, prefix in enumerate(PREFIX_SIZES):
        count = tail_parents[lane]
        live_tail = tail_edges[lane]
        # Every tail is positive.  If it is split into extent_edges-sized
        # dependent READs, these are safe aggregate bounds on
        # sum(ceil(tail_i / extent_edges)) without inventing per-parent degree.
        extent_tail_lower = (
            max(count, math.ceil(live_tail / extent_edges))
            if count else 0)
        extent_tail_upper = (
            count + (live_tail - count) // extent_edges
            if count else 0)
        initial_bytes = parents * (
            header_bytes + prefix * pointer_bytes)
        contiguous_bytes = initial_bytes + live_tail * pointer_bytes
        extent_bytes_lower = (
            initial_bytes
            + extent_tail_lower * extent_edges * pointer_bytes)
        extent_bytes_upper = (
            initial_bytes
            + extent_tail_upper * extent_edges * pointer_bytes)
        continuation_results.append({
            "prefix_edges": prefix,
            "parents_with_continuation": count,
            "continuation_parent_rate": ratio(count, parents),
            "chunks_with_continuation": chunks_with_tail[lane],
            "chunk_continuation_rate": ratio(
                chunks_with_tail[lane], len(events)),
            "live_tail_edges": live_tail,
            # Best WQE case after an untagged first READ: one contiguous,
            # exactly-sized tail READ for each parent that has a tail.
            "contiguous_tail_wqes": parents + count,
            "contiguous_tail_wqes_per_parent": ratio(
                parents + count, parents),
            "contiguous_tail_payload_bytes": contiguous_bytes,
            "contiguous_tail_payload_ratio": ratio(
                contiguous_bytes, fixed_bytes),
            # If the tail itself is represented as dependent fixed extents,
            # aggregate statistics do not reveal each tail length.  Report
            # bounds rather than fabricating an exact per-parent result.
            "extent_tail_wqes_lower": extent_tail_lower,
            "extent_tail_wqes_upper": extent_tail_upper,
            "extent_chain_wqes_per_parent_lower": ratio(
                parents + extent_tail_lower, parents),
            "extent_chain_wqes_per_parent_upper": ratio(
                parents + extent_tail_upper, parents),
            "extent_chain_payload_bytes_lower": extent_bytes_lower,
            "extent_chain_payload_bytes_upper": extent_bytes_upper,
            "extent_chain_payload_ratio_lower": ratio(
                extent_bytes_lower, fixed_bytes),
            "extent_chain_payload_ratio_upper": ratio(
                extent_bytes_upper, fixed_bytes),
        })

    projection: dict[str, Any] = {}
    if baseline:
        graph_reads = baseline.get("graph_reads_per_query", 0.0)
        current_total = baseline.get("rdma_bytes_per_query", 0.0)
        current_graph = graph_reads * record_bytes
        non_graph = max(0.0, current_total - current_graph)
        for name, model in (
            ("ideal_live_prefix", ideal),
            ("extent_8", extent),
        ):
            payload_ratio = float(model["payload_ratio"] or 0.0)
            proposed_graph = current_graph * payload_ratio
            proposed_total = non_graph + proposed_graph
            projection[name] = {
                "current_total_rdma_bytes_per_query": current_total,
                "current_graph_bytes_per_query": current_graph,
                "non_graph_bytes_per_query": non_graph,
                "proposed_graph_bytes_per_query": proposed_graph,
                "proposed_total_rdma_bytes_per_query": proposed_total,
                "total_rdma_byte_reduction": (
                    1.0 - proposed_total / current_total
                    if current_total else None),
            }

    overflows = sum(
        int(query.get("adjacency_oracle_overflow", 0) or 0)
        for query in queries)
    return {
        "schema": 1,
        "trace": {
            "query_records": len(queries),
            "score_chunks": len(events),
            "parents": parents,
            "live_edges": edges,
            "average_live_edges_per_parent": ratio(edges, parents),
            "overflow_count": overflows,
        },
        "layout_assumptions": {
            "fixed_record_bytes": record_bytes,
            "header_bytes": header_bytes,
            "remote_ptr_bytes": pointer_bytes,
            "extent_edges": extent_edges,
            "extent_bytes": extent_edges * pointer_bytes,
            "one_shot_extent_requires_pre_read_length_class": True,
            "version_checksum_redesign_cost_included": False,
        },
        "fixed_record": fixed,
        "ideal_live_prefix": ideal,
        "extent_8": extent,
        "continuation": continuation_results,
        "baseline": baseline,
        "total_rdma_projection": projection,
        "interpretation": {
            "layout_not_query_selection": (
                "Savings come only from eliminating unused fixed-record "
                "capacity; no edge is skipped based on the query."),
            "continuation_bounds": (
                "The lower WQE model reads one contiguous live tail. The "
                "extent-chain bounds use only aggregate tail counts and never "
                "pretend that per-parent degrees were traced."),
        },
    }


def markdown(report: dict[str, Any]) -> str:
    trace = report["trace"]
    fixed = report["fixed_record"]
    ideal = report["ideal_live_prefix"]
    extent = report["extent_8"]
    lines = [
        "# Live-extent adjacency byte oracle",
        "",
        "This oracle credits only live-degree packing. It does not use visited "
        "outcomes, Beam membership, or a query-dependent edge certificate.",
        "",
        "## Sample",
        "",
        f"- Queries / score chunks: {trace['query_records']} / "
        f"{trace['score_chunks']}",
        f"- Parents / live edges: {trace['parents']} / {trace['live_edges']}",
        f"- Average live degree: "
        f"{number(trace['average_live_edges_per_parent'], 2)}",
        f"- Trace overflow: {trace['overflow_count']}",
        "",
        "## Byte opportunity",
        "",
        "| layout | sample bytes | bytes/parent | reduction vs fixed |",
        "|---|---:|---:|---:|",
        f"| fixed record | {fixed['payload_bytes']} | "
        f"{number(fixed['bytes_per_parent'], 2)} | 0.00% |",
        f"| ideal live prefix | {ideal['payload_bytes']} | "
        f"{number(ideal['bytes_per_parent'], 2)} | "
        f"{pct(ideal['payload_savings'])} |",
        f"| 8-edge extent class | {extent['payload_bytes']} | "
        f"{number(extent['bytes_per_parent'], 2)} | "
        f"{pct(extent['payload_savings'])} |",
        "",
        f"The 8-edge class needs {number(extent['extents_per_parent'])} "
        "extents/parent and adds "
        f"{pct(extent['rounding_overhead_vs_ideal'])} rounding bytes over "
        "the impossible byte-exact layout.",
        "",
        "## Untagged-length continuation cost",
        "",
        "| first prefix | parents needing tail | one contiguous tail WQE/parent | "
        "8-edge chain WQE/parent lower–upper | contiguous payload/fixed | "
        "extent payload/fixed lower–upper |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["continuation"]:
        lines.append(
            f"| {row['prefix_edges']} | "
            f"{row['parents_with_continuation']} "
            f"({pct(row['continuation_parent_rate'])}) | "
            f"{number(row['contiguous_tail_wqes_per_parent'])} | "
            f"{number(row['extent_chain_wqes_per_parent_lower'])}–"
            f"{number(row['extent_chain_wqes_per_parent_upper'])} | "
            f"{pct(row['contiguous_tail_payload_ratio'])} | "
            f"{pct(row['extent_chain_payload_ratio_lower'])}–"
            f"{pct(row['extent_chain_payload_ratio_upper'])} |")

    projection = report["total_rdma_projection"]
    if projection:
        lines += [
            "",
            "## Projected total RDMA bytes",
            "",
            "| layout | proposed graph bytes/query | proposed total bytes/query | "
            "total RDMA byte reduction |",
            "|---|---:|---:|---:|",
        ]
        for label, key in (
            ("ideal live prefix", "ideal_live_prefix"),
            ("8-edge extent class", "extent_8"),
        ):
            row = projection[key]
            lines.append(
                f"| {label} | "
                f"{number(row['proposed_graph_bytes_per_query'], 1)} | "
                f"{number(row['proposed_total_rdma_bytes_per_query'], 1)} | "
                f"{pct(row['total_rdma_byte_reduction'])} |")

    lines += [
        "",
        "A one-shot extent keeps one graph READ per parent only if the extent "
        "class is known before posting the RDMA READ. Otherwise the "
        "continuation table exposes the dependent-WQE cost. Version/checksum "
        "layout changes and their GPU cost are deliberately not modeled.",
        "",
    ]
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", type=pathlib.Path)
    parser.add_argument("--baseline-json", type=pathlib.Path)
    parser.add_argument("--output-json", type=pathlib.Path)
    parser.add_argument("--output-markdown", type=pathlib.Path)
    parser.add_argument("--header-bytes", type=int, default=16)
    parser.add_argument("--extent-edges", type=int, default=EXTENT_EDGES)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    metadata, queries, events = load_jsonl(args.trace)
    report = aggregate(
        metadata, queries, events, load_baseline(args.baseline_json),
        args.header_bytes, args.extent_edges)
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
