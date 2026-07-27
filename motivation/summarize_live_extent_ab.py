#!/usr/bin/env python3
"""Strictly summarize paired fixed/live-extent end-to-end reports.

The runner writes one report below each

  POLICY/concurrency_N/repeat_R/**/sift100m_*.json

directory.  This analyzer deliberately fails on missing, duplicate, malformed,
failed, or non-comparable cases.  In particular, it never picks the newest
report from a partially rerun directory.
"""

import argparse
import json
import math
import re
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path


POLICIES = ("fixed", "live-extent")
REPORT_PATTERN = "sift100m_*.json"
CASE_COMPONENT = re.compile(r"concurrency_([1-9][0-9]*)")
REPEAT_COMPONENT = re.compile(r"repeat_([1-9][0-9]*)")

# The graph-read policy is the sole controlled variable.  Runtime counters and
# consumed query-row counts are intentionally absent: they are outcomes.
PAIR_FIELDS = (
    ("client_threads", ("meta", "client_threads")),
    ("search", ("meta", "search")),
    ("workload", ("meta", "workload")),
    ("run_mode", ("meta", "run_mode")),
    ("recall_only", ("meta", "recall_only")),
    ("time_completion_policy", ("meta", "time_completion_policy")),
    ("time_issue_policy", ("meta", "time_issue_policy")),
    ("mixed_dispatch_policy", ("meta", "mixed_dispatch_policy")),
    ("read_ratio", ("meta", "read_ratio")),
    ("target_query_qps", ("meta", "target_query_qps")),
    ("target_write_qps", ("meta", "target_write_qps")),
    ("write_insert_ratio", ("meta", "write_insert_ratio")),
    ("write_upsert_ratio", ("meta", "write_upsert_ratio")),
    ("write_delete_ratio", ("meta", "write_delete_ratio")),
    ("index_prefix", ("meta", "index_prefix")),
    ("dim", ("meta", "dim")),
    ("vector_bytes", ("meta", "vector_bytes")),
    ("vector_component_size", ("meta", "vector_component_size")),
    ("vector_data_type", ("meta", "vector_data_type")),
    ("navigation_quantizer", ("meta", "navigation_quantizer")),
    ("candidate_vector_rdma_bytes",
     ("meta", "candidate_vector_rdma_bytes")),
    ("entry_seed_policy", ("meta", "entry_seed_policy")),
    ("entry_seed_capacity", ("meta", "entry_seed_capacity")),
    ("entry_seed_shards", ("meta", "entry_seed_shards")),
    ("gpu_query_slots", ("meta", "gpu_query_slots")),
    ("gpu_rdma_qps", ("meta", "gpu_rdma_qps")),
    ("gpu_graph_physical_record_bytes",
     ("meta", "gpu_graph_physical_record_bytes")),
    ("gpu_graph_entry_capacity",
     ("meta", "gpu_graph_entry_capacity")),
    ("gpu_graph_extent_quantum_edges",
     ("meta", "gpu_graph_extent_quantum_edges")),
    ("gpu_graph_extent_sidecar_format",
     ("meta", "gpu_graph_extent_sidecar_format")),
    ("prefetch_depth", ("meta", "gpu_graph_prefetch_depth")),
    ("expansion_policy", ("meta", "gpu_query_expansion_policy")),
    ("beam_merge_policy", ("meta", "gpu_query_beam_merge_policy")),
    ("beam_width", ("meta", "traversal_beam_width")),
    ("max_expansions", ("meta", "max_expansions")),
    ("rerank_width", ("meta", "final_rerank_width")),
    ("warmup_seconds", ("meta", "warmup_seconds")),
    ("measure_seconds", ("meta", "measure_seconds")),
    ("warmup_ops", ("meta", "warmup_ops")),
    ("measure_ops", ("meta", "measure_ops")),
    ("recall_mode_meta", ("meta", "recall_mode")),
    ("recall_base_id_limit_meta", ("meta", "recall_base_id_limit")),
    ("performance_query_source",
     ("meta", "performance_query", "canonical_source")),
    ("performance_query_type",
     ("meta", "performance_query", "data_type")),
    ("performance_query_rows",
     ("meta", "performance_query", "rows")),
    ("performance_query_vector_bytes",
     ("meta", "performance_query", "vector_bytes")),
    ("performance_query_reuse",
     ("meta", "performance_query", "row_reuse_policy")),
    ("recall_query_source", ("meta", "recall_query", "source")),
    ("recall_query_type", ("meta", "recall_query", "data_type")),
    ("recall_query_rows", ("meta", "recall_query", "rows")),
    ("recall_query_vector_bytes",
     ("meta", "recall_query", "vector_bytes")),
    ("recall_k", ("recall", "k")),
    ("recall_mode", ("recall", "mode")),
    ("recall_queries", ("recall", "queries")),
    ("recall_query_file", ("recall", "query_file")),
    ("groundtruth_file", ("recall", "groundtruth_file")),
    ("recall_base_id_limit", ("recall", "base_id_limit")),
    ("recall_result_width", ("recall", "search_result_width")),
)


@dataclass(frozen=True)
class MetricSpec:
    name: str
    path: tuple | None = None
    scale: float = 1.0


METRIC_SPECS = (
    MetricSpec("query_qps", ("throughput", "query_ops_per_sec")),
    MetricSpec(
        "latency_mean_us",
        ("query_breakdown", "latency", "mean_end_to_end_ns"),
        0.001,
    ),
    MetricSpec(
        "latency_p50_us",
        ("query_breakdown", "latency", "p50_end_to_end_ns"),
        0.001,
    ),
    MetricSpec(
        "latency_p95_us",
        ("query_breakdown", "latency", "p95_end_to_end_ns"),
        0.001,
    ),
    MetricSpec(
        "latency_p99_us",
        ("query_breakdown", "latency", "p99_end_to_end_ns"),
        0.001,
    ),
    MetricSpec(
        "latency_p999_us",
        ("query_breakdown", "latency", "p999_end_to_end_ns"),
        0.001,
    ),
    MetricSpec("recall_at_k", ("recall", "recall")),
    MetricSpec(
        "gpu_rdma_wait_us",
        ("gpu_persistent", "average_gpu_rdma_wait_us"),
    ),
    MetricSpec(
        "gpu_query_us",
        ("gpu_persistent", "average_gpu_query_us"),
    ),
    MetricSpec("logical_graph_reads_per_query"),
    MetricSpec("graph_rounds_per_query"),
    MetricSpec("actual_graph_bytes_per_query"),
    MetricSpec("total_rdma_bytes_per_query"),
    MetricSpec("short_graph_reads_per_query"),
    MetricSpec("full_graph_reads_per_query"),
    MetricSpec("fallback_graph_reads_per_query"),
)

METRIC_NAMES = tuple(spec.name for spec in METRIC_SPECS)
LATENCY_METRICS = (
    "latency_mean_us",
    "latency_p50_us",
    "latency_p95_us",
    "latency_p99_us",
    "latency_p999_us",
)


class ReportError(ValueError):
    """A report set is incomplete, invalid, or not a controlled A/B."""


@dataclass(frozen=True)
class Report:
    path: Path
    policy: str
    concurrency: int
    repeat: int
    pair_values: dict
    metrics: dict


def _field_name(parts):
    return ".".join(parts)


def _require_field(root, parts, report_path):
    value = root
    for part in parts:
        if not isinstance(value, dict) or part not in value:
            raise ReportError(
                f"{report_path}: missing JSON field {_field_name(parts)}")
        value = value[part]
    return value


def _require_number(root, parts, report_path, *, positive=False):
    value = _require_field(root, parts, report_path)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ReportError(
            f"{report_path}: {_field_name(parts)} is not numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ReportError(
            f"{report_path}: {_field_name(parts)} is not finite")
    if number < 0.0 or (positive and number == 0.0):
        qualifier = "positive" if positive else "nonnegative"
        raise ReportError(
            f"{report_path}: {_field_name(parts)} is not {qualifier}")
    return number


def _require_scalar(root, parts, report_path):
    value = _require_field(root, parts, report_path)
    if isinstance(value, (dict, list)) or value is None:
        raise ReportError(
            f"{report_path}: {_field_name(parts)} is not a scalar")
    return value


def _require_equal(root, parts, expected, report_path):
    value = _require_field(root, parts, report_path)
    if value != expected:
        raise ReportError(
            f"{report_path}: {_field_name(parts)}={value!r}, "
            f"expected {expected!r}")


def _require_close(reported, derived, label, report_path):
    if not math.isclose(reported, derived, rel_tol=1e-9, abs_tol=1e-12):
        raise ReportError(
            f"{report_path}: {label}={reported} does not match "
            f"the derived value {derived}")


def _validate_recall_section(root, section, report_path):
    prefix = (section,)
    _require_equal(
        root, prefix + ("result_set_complete",), True, report_path)
    _require_equal(
        root,
        prefix + ("queries_with_insufficient_base_results",),
        0,
        report_path,
    )
    queries = _require_number(
        root, prefix + ("queries",), report_path, positive=True)
    recall = _require_number(root, prefix + ("recall",), report_path)
    if recall > 1.0:
        raise ReportError(
            f"{report_path}: {section}.recall={recall} is above 1")
    return queries, recall


def load_report(path, policy, concurrency, repeat):
    try:
        with path.open("r", encoding="utf-8") as stream:
            root = json.load(stream)
    except (OSError, json.JSONDecodeError) as error:
        raise ReportError(f"{path}: cannot read JSON: {error}") from error

    _require_equal(
        root,
        ("meta", "gpu_query_graph_read_policy"),
        policy,
        path,
    )
    _require_equal(root, ("meta", "client_threads"), concurrency, path)
    _require_equal(
        root, ("meta", "gpu_query_expansion_policy"), "fixed", path)
    _require_equal(
        root, ("meta", "gpu_graph_prefetch_depth"), 16, path)
    _require_equal(
        root, ("meta", "gpu_query_beam_merge_policy"), "stable-run", path)
    _require_equal(root, ("meta", "workload"), "query", path)
    _require_equal(root, ("meta", "recall_only"), False, path)
    _require_equal(
        root, ("meta", "fine_grained_breakdown_enabled"), True, path)

    pair_values = {
        name: _require_scalar(root, parts, path)
        for name, parts in PAIR_FIELDS
    }

    completed = _require_number(
        root, ("gpu_persistent", "queries_completed"), path, positive=True)
    submitted = _require_number(
        root, ("gpu_persistent", "queries_submitted"), path, positive=True)
    query_count = _require_number(
        root, ("query_breakdown", "count"), path, positive=True)
    query_ops = _require_number(
        root, ("throughput", "query_ops"), path, positive=True)
    if submitted != completed or query_count != completed or query_ops != completed:
        raise ReportError(
            f"{path}: query accounting mismatch: submitted={submitted}, "
            f"completed={completed}, breakdown={query_count}, "
            f"throughput={query_ops}")

    _require_equal(
        root, ("gpu_persistent", "direct_path_failures"), 0, path)
    _require_equal(
        root, ("gpu_persistent", "centroid_route_query_timeouts"), 0, path)
    _require_equal(root, ("stage2", "failures"), 0, path)

    _, recall = _validate_recall_section(root, "recall", path)
    _, post_recall = _validate_recall_section(
        root, "static_gt_post_recall", path)
    _require_equal(root, ("recall", "phase"), "before_performance", path)
    _require_equal(
        root,
        ("static_gt_post_recall", "phase"),
        "after_performance",
        path,
    )
    for field in (
        "k",
        "mode",
        "queries",
        "query_file",
        "groundtruth_file",
        "base_id_limit",
        "search_result_width",
    ):
        before = _require_field(root, ("recall", field), path)
        after = _require_field(root, ("static_gt_post_recall", field), path)
        if before != after:
            raise ReportError(
                f"{path}: Recall protocol changed within the run: "
                f"recall.{field}={before!r}, "
                f"static_gt_post_recall.{field}={after!r}")
    if not math.isclose(recall, post_recall, rel_tol=0.0, abs_tol=1e-12):
        raise ReportError(
            f"{path}: Recall changed within the query-only run: "
            f"before={recall}, after={post_recall}")

    metrics = {}
    for spec in METRIC_SPECS:
        if spec.path is not None:
            metrics[spec.name] = (
                _require_number(root, spec.path, path) * spec.scale)

    for name in ("query_qps",) + LATENCY_METRICS:
        if metrics[name] <= 0.0:
            raise ReportError(f"{path}: {name} must be positive")

    graph_reads = _require_number(
        root, ("gpu_persistent", "graph_page_requests"), path)
    graph_rounds = _require_number(
        root, ("gpu_persistent", "graph_dependency_rounds"), path)
    graph_bytes = _require_number(
        root, ("gpu_persistent", "graph_read_bytes"), path)
    total_rdma_bytes = _require_number(
        root, ("gpu_persistent", "rdma_read_bytes"), path)
    short_reads = _require_number(
        root, ("gpu_persistent", "graph_live_extent_reads"), path)
    full_reads = _require_number(
        root, ("gpu_persistent", "graph_full_record_reads"), path)
    fallback_reads = _require_number(
        root, ("gpu_persistent", "graph_extent_fallback_reads"), path)

    if total_rdma_bytes < graph_bytes:
        raise ReportError(
            f"{path}: rdma_read_bytes={total_rdma_bytes} is below "
            f"graph_read_bytes={graph_bytes}")
    if policy == "fixed" and (short_reads != 0.0 or fallback_reads != 0.0):
        raise ReportError(
            f"{path}: fixed policy reported short/fallback graph reads: "
            f"short={short_reads}, fallback={fallback_reads}")

    metrics.update({
        "logical_graph_reads_per_query": graph_reads / completed,
        "graph_rounds_per_query": graph_rounds / completed,
        "actual_graph_bytes_per_query": graph_bytes / completed,
        "total_rdma_bytes_per_query": total_rdma_bytes / completed,
        "short_graph_reads_per_query": short_reads / completed,
        "full_graph_reads_per_query": full_reads / completed,
        "fallback_graph_reads_per_query": fallback_reads / completed,
    })

    reported_rounds = _require_number(
        root,
        ("gpu_persistent", "average_graph_rounds_per_query"),
        path,
    )
    reported_graph_bytes = _require_number(
        root,
        ("gpu_persistent", "average_graph_read_bytes_per_query"),
        path,
    )
    _require_close(
        reported_rounds,
        metrics["graph_rounds_per_query"],
        "average_graph_rounds_per_query",
        path,
    )
    _require_close(
        reported_graph_bytes,
        metrics["actual_graph_bytes_per_query"],
        "average_graph_read_bytes_per_query",
        path,
    )

    return Report(
        path=path.resolve(),
        policy=policy,
        concurrency=concurrency,
        repeat=repeat,
        pair_values=pair_values,
        metrics=metrics,
    )


def _parse_case_path(path, policy_root):
    relative = path.relative_to(policy_root)
    parts = relative.parts
    if len(parts) < 3:
        raise ReportError(
            f"{path}: expected concurrency_N/repeat_R/**/{REPORT_PATTERN}")
    concurrency_match = CASE_COMPONENT.fullmatch(parts[0])
    repeat_match = REPEAT_COMPONENT.fullmatch(parts[1])
    if concurrency_match is None or repeat_match is None:
        raise ReportError(
            f"{path}: expected concurrency_N/repeat_R as the first "
            f"two components below the policy directory")
    return int(concurrency_match.group(1)), int(repeat_match.group(1))


def discover_pairs(root, pattern=REPORT_PATTERN):
    root = Path(root)
    by_policy = {policy: {} for policy in POLICIES}
    errors = []
    for policy in POLICIES:
        policy_root = root / policy
        if not policy_root.is_dir():
            errors.append(f"missing policy directory: {policy_root}")
            continue
        paths = sorted(policy_root.rglob(pattern))
        if not paths:
            errors.append(
                f"no reports matching {pattern!r} below {policy_root}")
            continue
        for path in paths:
            try:
                concurrency, repeat = _parse_case_path(path, policy_root)
                report = load_report(
                    path, policy, concurrency, repeat)
            except ReportError as error:
                errors.append(str(error))
                continue
            case = (concurrency, repeat)
            if case in by_policy[policy]:
                errors.append(
                    f"duplicate reports for policy={policy}, "
                    f"concurrency={concurrency}, repeat={repeat}: "
                    f"{by_policy[policy][case].path} and {path}")
                continue
            by_policy[policy][case] = report

    fixed_cases = set(by_policy["fixed"])
    live_cases = set(by_policy["live-extent"])
    if fixed_cases != live_cases:
        for case in sorted(fixed_cases | live_cases):
            present = [
                policy for policy in POLICIES
                if case in by_policy[policy]
            ]
            if len(present) != len(POLICIES):
                missing = sorted(set(POLICIES) - set(present))
                errors.append(
                    f"unpaired case concurrency={case[0]}, "
                    f"repeat={case[1]}: present={present}, missing={missing}")

    pairs = []
    for case in sorted(fixed_cases & live_cases):
        fixed = by_policy["fixed"][case]
        live = by_policy["live-extent"][case]
        mismatches = [
            (name, fixed.pair_values[name], live.pair_values[name])
            for name, _ in PAIR_FIELDS
            if fixed.pair_values[name] != live.pair_values[name]
        ]
        if mismatches:
            details = ", ".join(
                f"{name}: fixed={left!r}, live-extent={right!r}"
                for name, left, right in mismatches
            )
            errors.append(
                f"non-comparable pair concurrency={case[0]}, "
                f"repeat={case[1]}: {details}")
            continue
        pairs.append((fixed, live))

    if errors:
        raise ReportError("\n".join(errors))
    if not pairs:
        raise ReportError(f"no paired Live-Extent reports below {root}")
    return pairs


def _ratio(numerator, denominator):
    if denominator == 0.0:
        return None
    return numerator / denominator


def _paired_comparison(fixed, live):
    ratios = {
        name: _ratio(live.metrics[name], fixed.metrics[name])
        for name in METRIC_NAMES
    }
    deltas = {
        name: live.metrics[name] - fixed.metrics[name]
        for name in METRIC_NAMES
    }
    changes = {
        name: None if ratios[name] is None else ratios[name] - 1.0
        for name in METRIC_NAMES
    }
    graph_ratio = ratios["actual_graph_bytes_per_query"]
    total_ratio = ratios["total_rdma_bytes_per_query"]
    if graph_ratio is None or total_ratio is None:
        raise ReportError(
            f"cannot calculate byte reduction for concurrency="
            f"{fixed.concurrency}, repeat={fixed.repeat}: fixed bytes are zero")
    return {
        "ratio_live_over_fixed": ratios,
        "delta_live_minus_fixed": deltas,
        "change_fraction": changes,
        "graph_bytes_reduction_fraction": 1.0 - graph_ratio,
        "total_rdma_bytes_reduction_fraction": 1.0 - total_ratio,
        "qps_ratio_live_over_fixed": ratios["query_qps"],
        "latency_ratio_live_over_fixed": {
            name: ratios[name] for name in LATENCY_METRICS
        },
    }


def _median(values):
    return statistics.median(values)


def _median_optional(values):
    present = [value for value in values if value is not None]
    return None if not present else _median(present)


def build_summary(root, pairs):
    grouped = {}
    for fixed, live in pairs:
        grouped.setdefault(fixed.concurrency, []).append((fixed, live))

    cases = {}
    for concurrency, concurrency_pairs in sorted(grouped.items()):
        concurrency_pairs.sort(key=lambda pair: pair[0].repeat)
        repeats = []
        for fixed, live in concurrency_pairs:
            repeats.append({
                "repeat": fixed.repeat,
                "reports": {
                    "fixed": str(fixed.path),
                    "live-extent": str(live.path),
                },
                "metrics": {
                    "fixed": fixed.metrics,
                    "live-extent": live.metrics,
                },
                "paired": _paired_comparison(fixed, live),
            })

        policy_medians = {}
        for policy in POLICIES:
            policy_medians[policy] = {
                name: _median([
                    repeat["metrics"][policy][name]
                    for repeat in repeats
                ])
                for name in METRIC_NAMES
            }
        paired_medians = {
            "ratio_live_over_fixed": {
                name: _median_optional([
                    repeat["paired"]["ratio_live_over_fixed"][name]
                    for repeat in repeats
                ])
                for name in METRIC_NAMES
            },
            "delta_live_minus_fixed": {
                name: _median([
                    repeat["paired"]["delta_live_minus_fixed"][name]
                    for repeat in repeats
                ])
                for name in METRIC_NAMES
            },
            "change_fraction": {
                name: _median_optional([
                    repeat["paired"]["change_fraction"][name]
                    for repeat in repeats
                ])
                for name in METRIC_NAMES
            },
            "graph_bytes_reduction_fraction": _median([
                repeat["paired"]["graph_bytes_reduction_fraction"]
                for repeat in repeats
            ]),
            "total_rdma_bytes_reduction_fraction": _median([
                repeat["paired"]["total_rdma_bytes_reduction_fraction"]
                for repeat in repeats
            ]),
            "qps_ratio_live_over_fixed": _median([
                repeat["paired"]["qps_ratio_live_over_fixed"]
                for repeat in repeats
            ]),
            "latency_ratio_live_over_fixed": {
                name: _median([
                    repeat["paired"]["latency_ratio_live_over_fixed"][name]
                    for repeat in repeats
                ])
                for name in LATENCY_METRICS
            },
        }
        cases[f"concurrency_{concurrency}"] = {
            "concurrency": concurrency,
            "repeat_count": len(repeats),
            "policy_medians": policy_medians,
            "paired_medians": paired_medians,
            "repeats": repeats,
        }

    return {
        "schema_version": 1,
        "result_root": str(Path(root).resolve()),
        "controlled_variable": "gpu_query_graph_read_policy",
        "policies": list(POLICIES),
        "fixed_search_contract": {
            "gpu_query_expansion_policy": "fixed",
            "gpu_graph_prefetch_depth": 16,
            "gpu_query_beam_merge_policy": "stable-run",
            "workload": "query",
        },
        "pair_count": len(pairs),
        "cases": cases,
    }


def _format_number(value):
    if value is None:
        return "n/a"
    magnitude = abs(value)
    if magnitude != 0.0 and (magnitude >= 1e7 or magnitude < 1e-4):
        return f"{value:.6e}"
    return f"{value:.6f}"


def render_markdown(summary):
    lines = [
        "# Live-Extent end-to-end A/B summary",
        "",
        "All comparisons are paired by `(concurrency, repeat)`. "
        "Ratios are `live-extent / fixed`; byte reductions are "
        "`1 - live-extent / fixed`. Missing, duplicate, failed, or "
        "non-comparable reports make the analyzer fail instead of being "
        "discarded.",
        "",
        f"Paired reports: **{summary['pair_count']}**",
        "",
    ]
    for case_name, case in summary["cases"].items():
        lines.extend([
            f"## {case_name}",
            "",
            f"Repeats: **{case['repeat_count']}**",
            "",
            "### Per-policy medians",
            "",
            "| metric | fixed | live-extent |",
            "|---|---:|---:|",
        ])
        for name in METRIC_NAMES:
            lines.append(
                f"| {name} | "
                f"{_format_number(case['policy_medians']['fixed'][name])} | "
                f"{_format_number(case['policy_medians']['live-extent'][name])} |")

        paired = case["paired_medians"]
        lines.extend([
            "",
            "### Paired median headline",
            "",
            "| comparison | median |",
            "|---|---:|",
            f"| QPS ratio | "
            f"{_format_number(paired['qps_ratio_live_over_fixed'])} |",
            f"| mean latency ratio | "
            f"{_format_number(paired['latency_ratio_live_over_fixed']['latency_mean_us'])} |",
            f"| P95 latency ratio | "
            f"{_format_number(paired['latency_ratio_live_over_fixed']['latency_p95_us'])} |",
            f"| P99 latency ratio | "
            f"{_format_number(paired['latency_ratio_live_over_fixed']['latency_p99_us'])} |",
            f"| P999 latency ratio | "
            f"{_format_number(paired['latency_ratio_live_over_fixed']['latency_p999_us'])} |",
            f"| graph-byte reduction | "
            f"{_format_number(paired['graph_bytes_reduction_fraction'])} |",
            f"| total-RDMA-byte reduction | "
            f"{_format_number(paired['total_rdma_bytes_reduction_fraction'])} |",
            "",
            "### Every paired repeat",
            "",
            "| repeat | metric | fixed | live-extent | live/fixed | "
            "live-fixed |",
            "|---:|---|---:|---:|---:|---:|",
        ])
        for repeat in case["repeats"]:
            for name in METRIC_NAMES:
                lines.append(
                    f"| {repeat['repeat']} | {name} | "
                    f"{_format_number(repeat['metrics']['fixed'][name])} | "
                    f"{_format_number(repeat['metrics']['live-extent'][name])} | "
                    f"{_format_number(repeat['paired']['ratio_live_over_fixed'][name])} | "
                    f"{_format_number(repeat['paired']['delta_live_minus_fixed'][name])} |")
            lines.extend([
                f"| {repeat['repeat']} | graph_bytes_reduction_fraction | "
                f"n/a | n/a | "
                f"{_format_number(repeat['paired']['graph_bytes_reduction_fraction'])} | "
                "n/a |",
                f"| {repeat['repeat']} | total_rdma_bytes_reduction_fraction | "
                f"n/a | n/a | "
                f"{_format_number(repeat['paired']['total_rdma_bytes_reduction_fraction'])} | "
                "n/a |",
            ])
        lines.append("")
    return "\n".join(lines)


def _parse_args(argv=None):
    default_root = (
        Path(__file__).resolve().parent / "results" / "live_extent_ab")
    parser = argparse.ArgumentParser(
        description=(
            "Strictly pair fixed/live-extent end-to-end reports and emit "
            "JSON plus Markdown summaries."))
    parser.add_argument(
        "root",
        type=Path,
        nargs="?",
        default=default_root,
        help=f"run_live_extent_ab.sh RESULT_ROOT (default: {default_root})",
    )
    parser.add_argument(
        "--pattern",
        default=REPORT_PATTERN,
        help=f"report filename glob (default: {REPORT_PATTERN})",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        help="output JSON (default: ROOT/live_extent_ab_summary.json)",
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        help="output Markdown (default: ROOT/live_extent_ab_summary.md)",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    json_output = (
        args.json_output
        if args.json_output is not None
        else args.root / "live_extent_ab_summary.json"
    )
    markdown_output = (
        args.markdown_output
        if args.markdown_output is not None
        else args.root / "live_extent_ab_summary.md"
    )
    try:
        pairs = discover_pairs(args.root, args.pattern)
        summary = build_summary(args.root, pairs)
        markdown = render_markdown(summary)
        json_output.parent.mkdir(parents=True, exist_ok=True)
        markdown_output.parent.mkdir(parents=True, exist_ok=True)
        json_output.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        markdown_output.write_text(markdown + "\n", encoding="utf-8")
    except (OSError, ReportError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(f"JSON summary: {json_output}")
    print(f"Markdown summary: {markdown_output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
