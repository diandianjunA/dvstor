#!/usr/bin/env python3
"""Compare paired legacy and stable-run beam-merge benchmark reports.

The analyzer is deliberately read-only: it discovers JSON reports below the
two policy directories and writes the comparison only to stdout/stderr.
"""

import argparse
import json
import math
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


POLICIES = ("legacy", "stable-run")

# Every field here must be identical for two reports to be considered an A/B
# pair.  Policy is checked separately because it is the controlled variable.
PAIR_FIELDS = (
    ("client_threads", ("meta", "client_threads")),
    ("prefetch_depth", ("meta", "gpu_graph_prefetch_depth")),
    ("expansion_policy", ("meta", "gpu_query_expansion_policy")),
    ("beam_width", ("meta", "traversal_beam_width")),
    ("rerank_width", ("meta", "final_rerank_width")),
    ("max_expansions", ("meta", "max_expansions")),
    ("search", ("meta", "search")),
    ("workload", ("meta", "workload")),
    ("run_mode", ("meta", "run_mode")),
    ("warmup_seconds", ("meta", "warmup_seconds")),
    ("measure_seconds", ("meta", "measure_seconds")),
    ("warmup_ops", ("meta", "warmup_ops")),
    ("measure_ops", ("meta", "measure_ops")),
    ("recall_k", ("recall", "k")),
    ("recall_mode", ("recall", "mode")),
    ("recall_queries", ("recall", "queries")),
    ("recall_query_file", ("recall", "query_file")),
    ("groundtruth_file", ("recall", "groundtruth_file")),
)


@dataclass(frozen=True)
class MetricSpec:
    name: str
    path: tuple
    scale: float = 1.0
    decimals: int = 3


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
    MetricSpec("recall", ("recall", "recall"), decimals=9),
    MetricSpec(
        "beam_merge_us",
        ("gpu_persistent", "average_gpu_beam_merge_us"),
    ),
    MetricSpec(
        "beam_merge_prepare_us",
        ("gpu_persistent", "average_gpu_beam_merge_prepare_us"),
    ),
    MetricSpec(
        "beam_merge_sort_us",
        ("gpu_persistent", "average_gpu_beam_merge_sort_us"),
    ),
    MetricSpec(
        "beam_merge_materialize_us",
        ("gpu_persistent", "average_gpu_beam_merge_materialize_us"),
    ),
)


class ReportError(ValueError):
    """A report is malformed or is not comparable."""


@dataclass
class Report:
    path: Path
    policy: str
    pair_key: tuple
    metrics: dict


def field_name(parts):
    return ".".join(parts)


def require_field(root, parts, report_path):
    value = root
    for part in parts:
        if not isinstance(value, dict) or part not in value:
            raise ReportError(
                f"{report_path}: missing JSON field {field_name(parts)}")
        value = value[part]
    return value


def require_number(root, parts, report_path, *, nonnegative=True):
    value = require_field(root, parts, report_path)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ReportError(
            f"{report_path}: {field_name(parts)} is not numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ReportError(
            f"{report_path}: {field_name(parts)} is not finite")
    if nonnegative and number < 0.0:
        raise ReportError(
            f"{report_path}: {field_name(parts)} is negative")
    return number


def require_pair_value(root, parts, report_path):
    value = require_field(root, parts, report_path)
    if isinstance(value, (dict, list)) or value is None:
        raise ReportError(
            f"{report_path}: {field_name(parts)} is not a scalar")
    try:
        hash(value)
    except TypeError as error:
        raise ReportError(
            f"{report_path}: {field_name(parts)} is not comparable") from error
    return value


def load_report(path, expected_policy):
    try:
        with path.open("r", encoding="utf-8") as stream:
            root = json.load(stream)
    except (OSError, json.JSONDecodeError) as error:
        raise ReportError(f"{path}: cannot read JSON: {error}") from error

    policy_path = ("meta", "gpu_query_beam_merge_policy")
    actual_policy = require_field(root, policy_path, path)
    if actual_policy != expected_policy:
        raise ReportError(
            f"{path}: {field_name(policy_path)}={actual_policy!r}, "
            f"but report is under {expected_policy!r}")

    key = tuple(
        require_pair_value(root, parts, path)
        for _, parts in PAIR_FIELDS
    )
    expansion_policy = key[2]
    if expansion_policy != "fixed":
        raise ReportError(
            f"{path}: beam-merge A/B requires fixed expansion policy, "
            f"got {expansion_policy!r}")

    metrics = {}
    for spec in METRIC_SPECS:
        metrics[spec.name] = (
            require_number(root, spec.path, path) * spec.scale)

    completed = require_number(
        root, ("gpu_persistent", "queries_completed"), path)
    if completed == 0.0:
        raise ReportError(f"{path}: gpu_persistent.queries_completed is zero")
    graph_reads = require_number(
        root, ("gpu_persistent", "graph_page_requests"), path)
    graph_rounds = require_number(
        root, ("gpu_persistent", "graph_dependency_rounds"), path)
    reported_rounds = require_number(
        root, ("gpu_persistent", "average_graph_rounds_per_query"), path)
    derived_rounds = graph_rounds / completed
    if not math.isclose(
            reported_rounds, derived_rounds, rel_tol=1e-9, abs_tol=1e-12):
        raise ReportError(
            f"{path}: average_graph_rounds_per_query={reported_rounds} "
            f"does not match graph_dependency_rounds/queries_completed="
            f"{derived_rounds}")

    metrics["graph_reads_per_query"] = graph_reads / completed
    metrics["graph_rounds_per_query"] = reported_rounds
    recall = metrics["recall"]
    if recall > 1.0:
        raise ReportError(f"{path}: recall.recall={recall} is above 1")

    return Report(path, expected_policy, key, metrics)


def describe_key(key):
    values = dict(
        (name, value) for (name, _), value in zip(PAIR_FIELDS, key))
    run_extent = (
        f"seconds={values['warmup_seconds']}+{values['measure_seconds']}"
        if values["run_mode"] == "time"
        else f"ops={values['warmup_ops']}+{values['measure_ops']}"
    )
    return (
        f"concurrency={values['client_threads']} "
        f"run_mode={values['run_mode']} {run_extent} "
        f"prefetch_depth={values['prefetch_depth']} "
        f"beam={values['beam_width']} "
        f"rerank={values['rerank_width']} "
        f"max_expansions={values['max_expansions']}")


def describe_key_full(key):
    return ", ".join(
        f"{name}={value!r}"
        for (name, _), value in zip(PAIR_FIELDS, key)
    )


def discover_pairs(root, pattern):
    grouped = {
        policy: defaultdict(list)
        for policy in POLICIES
    }
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
                report = load_report(path, policy)
            except ReportError as error:
                errors.append(str(error))
                continue
            grouped[policy][report.pair_key].append(report)

    if errors:
        raise ReportError("\n".join(errors))

    pairs = []
    all_keys = sorted(
        set(grouped["legacy"]) | set(grouped["stable-run"]),
        key=lambda key: (int(key[0]), repr(key[1:])),
    )
    for key in all_keys:
        legacy = sorted(
            grouped["legacy"].get(key, []),
            key=lambda report: str(report.path),
        )
        stable = sorted(
            grouped["stable-run"].get(key, []),
            key=lambda report: str(report.path),
        )
        if len(legacy) != len(stable):
            errors.append(
                f"unpaired reports for {describe_key_full(key)}: "
                f"legacy={len(legacy)}, stable-run={len(stable)}")
            continue
        pairs.extend(zip(legacy, stable))

    if errors:
        raise ReportError("\n".join(errors))
    if not pairs:
        raise ReportError(f"no legacy/stable-run pairs found below {root}")
    return pairs


def relative_difference(left, right):
    denominator = max(abs(left), abs(right))
    if denominator == 0.0:
        return 0.0
    return abs(right - left) / denominator


def metric_delta_percent(legacy, stable):
    if legacy == 0.0:
        return None
    return (stable - legacy) / abs(legacy) * 100.0


def format_metric(value, decimals):
    return f"{value:.{decimals}f}"


def print_table(legacy, stable):
    rows = []
    metrics_by_name = {spec.name: spec for spec in METRIC_SPECS}
    ordered_names = (
        "query_qps",
        "latency_mean_us",
        "latency_p50_us",
        "latency_p95_us",
        "latency_p99_us",
        "latency_p999_us",
        "recall",
        "graph_reads_per_query",
        "graph_rounds_per_query",
        "beam_merge_us",
        "beam_merge_prepare_us",
        "beam_merge_sort_us",
        "beam_merge_materialize_us",
    )
    for name in ordered_names:
        spec = metrics_by_name.get(name)
        decimals = spec.decimals if spec else 6
        legacy_value = legacy.metrics[name]
        stable_value = stable.metrics[name]
        delta = stable_value - legacy_value
        delta_percent = metric_delta_percent(legacy_value, stable_value)
        rows.append((
            name,
            format_metric(legacy_value, decimals),
            format_metric(stable_value, decimals),
            format_metric(delta, decimals),
            "n/a" if delta_percent is None else f"{delta_percent:+.3f}%",
        ))

    headers = ("metric", "legacy", "stable-run", "delta", "delta_%")
    widths = [
        max(len(headers[index]), *(len(row[index]) for row in rows))
        for index in range(len(headers))
    ]
    print("  ".join(
        value.ljust(widths[index])
        for index, value in enumerate(headers)
    ))
    print("  ".join("-" * width for width in widths))
    for row in rows:
        print("  ".join(
            value.ljust(widths[index]) if index == 0
            else value.rjust(widths[index])
            for index, value in enumerate(row)
        ))


def validate_pair(legacy, stable, graph_read_rel_tol, round_rel_tol,
                  recall_abs_tol):
    graph_read_difference = relative_difference(
        legacy.metrics["graph_reads_per_query"],
        stable.metrics["graph_reads_per_query"],
    )
    round_difference = relative_difference(
        legacy.metrics["graph_rounds_per_query"],
        stable.metrics["graph_rounds_per_query"],
    )
    recall_difference = abs(
        stable.metrics["recall"] - legacy.metrics["recall"])
    checks = (
        (
            "graph_reads_per_query",
            graph_read_difference <= graph_read_rel_tol,
            f"relative_diff={graph_read_difference:.6%} "
            f"limit={graph_read_rel_tol:.6%}",
        ),
        (
            "graph_rounds_per_query",
            round_difference <= round_rel_tol,
            f"relative_diff={round_difference:.6%} "
            f"limit={round_rel_tol:.6%}",
        ),
        (
            "recall",
            recall_difference <= recall_abs_tol,
            f"absolute_diff={recall_difference:.12g} "
            f"limit={recall_abs_tol:.12g}",
        ),
    )
    for name, passed, detail in checks:
        print(f"  {'PASS' if passed else 'FAIL'} {name}: {detail}")
    return all(passed for _, passed, _ in checks)


def nonnegative_float(value):
    try:
        number = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            f"{value!r} is not a number") from error
    if not math.isfinite(number) or number < 0.0:
        raise argparse.ArgumentTypeError(
            f"{value!r} must be a finite nonnegative number")
    return number


def parse_args():
    default_root = (
        Path(__file__).resolve().parent / "results" / "beam_merge")
    parser = argparse.ArgumentParser(
        description=(
            "Read and compare paired legacy/stable-run beam-merge reports. "
            "Delta is stable-run minus legacy."))
    parser.add_argument(
        "root",
        type=Path,
        nargs="?",
        default=default_root,
        help=f"A/B result root (default: {default_root})",
    )
    parser.add_argument(
        "--pattern",
        default="sift100m_*.json",
        help="report filename glob used below each policy directory",
    )
    parser.add_argument(
        "--graph-read-rel-tol",
        type=nonnegative_float,
        default=0.01,
        help="maximum symmetric relative graph-reads difference (default: 0.01)",
    )
    parser.add_argument(
        "--round-rel-tol",
        type=nonnegative_float,
        default=0.01,
        help="maximum symmetric relative graph-rounds difference (default: 0.01)",
    )
    parser.add_argument(
        "--recall-abs-tol",
        type=nonnegative_float,
        default=1e-9,
        help="maximum absolute Recall difference (default: 1e-9)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    try:
        pairs = discover_pairs(args.root, args.pattern)
    except ReportError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    print(f"Beam-merge A/B pairs: {len(pairs)}")
    print("Delta and delta_% are stable-run minus legacy.")
    all_consistent = True
    key_occurrences = defaultdict(int)
    key_totals = defaultdict(int)
    for legacy, _ in pairs:
        key_totals[legacy.pair_key] += 1

    for legacy, stable in pairs:
        key = legacy.pair_key
        key_occurrences[key] += 1
        repetition = ""
        if key_totals[key] > 1:
            repetition = (
                f" repeat={key_occurrences[key]}/{key_totals[key]}")
        print(f"\n[{describe_key(key)}{repetition}]")
        print(f"legacy:     {legacy.path.resolve()}")
        print(f"stable-run: {stable.path.resolve()}")
        print_table(legacy, stable)
        print("consistency checks:")
        pair_consistent = validate_pair(
            legacy,
            stable,
            args.graph_read_rel_tol,
            args.round_rel_tol,
            args.recall_abs_tol,
        )
        all_consistent = all_consistent and pair_consistent

    print(
        f"\nOverall consistency: "
        f"{'PASS' if all_consistent else 'FAIL'}")
    return 0 if all_consistent else 1


if __name__ == "__main__":
    sys.exit(main())
