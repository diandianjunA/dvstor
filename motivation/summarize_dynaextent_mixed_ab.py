#!/usr/bin/env python3
"""Strictly summarize the fixed/static-only/DynaExtent mixed triplet.

The experiment has one registered workload contract and one controlled policy
dimension.  It requires a successful per-case reset log and rejects missing,
duplicate, mislabelled, non-comparable, reset-log-unbound, or incomplete
Latin-square reports.  DynaExtent ``short`` and ``full`` counters are physical
snapshot attempts; their sum is deliberately never presented as a logical
graph-read count, and headline traffic counters are normalized per query.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from motivation import summarize_live_extent_mixed_ab as mixed
except ModuleNotFoundError:  # Direct execution from the motivation directory.
    import summarize_live_extent_mixed_ab as mixed


REPORT_PATTERN = mixed.REPORT_PATTERN
RESET_LOG_NAME = "before_case_reset.log"
RESET_CERTIFICATE_KEY = "dynaextent_reset"
SNAPSHOT_ID_PATTERN = re.compile(
    r"[A-Za-z0-9][A-Za-z0-9._:+/@-]{7,255}")
RESET_DIGEST_PATTERN = re.compile(r"[0-9a-f]{64}")
LATIN_SQUARE_SIZE = 3


@dataclass(frozen=True)
class ModeSpec:
    name: str
    graph_policy: str
    dynamic_enabled: bool
    dynamic_source: str


MODE_SPECS = {
    "fixed": ModeSpec(
        name="fixed",
        graph_policy="fixed",
        dynamic_enabled=False,
        dynamic_source="full_physical_record",
    ),
    "static-only": ModeSpec(
        name="static-only",
        graph_policy="live-extent",
        dynamic_enabled=False,
        dynamic_source="full_physical_record",
    ),
    "dynaextent": ModeSpec(
        name="dynaextent",
        graph_policy="live-extent",
        dynamic_enabled=True,
        dynamic_source="incarnation_tagged_live_extent",
    ),
}
MODES = tuple(MODE_SPECS)

# The runner explicitly pins 336 clients.  Do not silently accept the older
# analyzer's auto-derived-client metadata even though it derives the same
# numerical value.
CONTRACT = mixed.ExperimentContract(
    name="dynaextent-rate-limited-explicit-336",
    concurrency=336,
    mixed_mode="rate_limited",
    read_ratio=0.5,
    target_query_qps=40_000.0,
    target_write_qps=1_000.0,
    time_issue_policy="shared_two_stream_pacer_until_deadline",
    driver_source="explicit",
    driver_required_threads=336,
    driver_derivation=(
        "sum(active_bounded_path_capacities);shared_rate_pacer"),
)

DYNA_RAW_FIELDS = (
    "dynamic_graph_short_reads",
    "dynamic_graph_full_reads",
    "dynamic_graph_read_bytes",
    "dynamic_graph_fallback_reads",
    "dynamic_graph_hint_promotions",
    "dynamic_graph_hint_demotions",
)
DYNA_PHYSICAL_DERIVED_FIELDS = (
    "dynamic_graph_snapshot_attempts",
    "dynamic_graph_nonfallback_full_attempts",
    "dynamic_graph_short_physical_ratio",
    "dynamic_graph_fallback_ratio",
    "average_dynamic_graph_read_bytes_per_physical_read",
    "average_dynamic_graph_read_bytes_per_query",
)
DYNA_PER_QUERY_FIELDS = (
    "dynamic_graph_short_reads_per_query",
    "dynamic_graph_full_reads_per_query",
    "dynamic_graph_fallback_reads_per_query",
    "dynamic_graph_hint_promotions_per_query",
    "dynamic_graph_hint_demotions_per_query",
    "dynamic_graph_snapshot_attempts_per_query",
)
METRIC_NAMES = (
    *mixed.METRIC_NAMES,
    *DYNA_RAW_FIELDS,
    *DYNA_PHYSICAL_DERIVED_FIELDS,
    *DYNA_PER_QUERY_FIELDS,
)

CERTIFICATE_PRIORITY = {
    "initial_recall": 0,
    "warmup_completed_writes": 1,
    "warmup_completed_inserts": 2,
    "warmup_completed_upserts": 3,
    "warmup_completed_deletes": 4,
    "measure_completed_writes": 5,
    "measure_completed_inserts": 6,
    "measure_completed_upserts": 7,
    "measure_completed_deletes": 8,
}


@dataclass(frozen=True)
class ModeReport:
    path: Path
    reset_log: Path
    mode: str
    concurrency: int
    repeat: int
    snapshot_id: str
    latin_position: int
    latin_cycle: int
    pair_values: dict[str, Any]
    metrics: dict[str, float | int]


ReportError = mixed.ReportError


def _read_json(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as stream:
            root = json.load(stream)
    except (OSError, json.JSONDecodeError) as error:
        raise ReportError(f"{path}: cannot read JSON: {error}") from error
    if not isinstance(root, dict):
        raise ReportError(f"{path}: JSON root is not an object")
    return root


def _validate_mode_mapping(
        root: dict[str, Any], spec: ModeSpec, path: Path) -> None:
    mixed._require_equal(
        root, ("meta", "gpu_query_graph_read_policy"),
        spec.graph_policy, path)
    dynamic = mixed._require_field(
        root, ("meta", "gpu_dynamic_graph_extent"), path)
    if not isinstance(dynamic, bool) or dynamic is not spec.dynamic_enabled:
        raise ReportError(
            f"{path}: directory policy {spec.name!r} requires "
            f"meta.gpu_dynamic_graph_extent={spec.dynamic_enabled!r}, "
            f"found {dynamic!r}")
    mixed._require_equal(
        root, ("meta", "gpu_dynamic_graph_extent_source"),
        spec.dynamic_source, path)


def _expected_latin_position(mode: str, repeat: int) -> int:
    policy_index = MODES.index(mode)
    offset = (repeat - 1) % LATIN_SQUARE_SIZE
    return (policy_index - offset) % LATIN_SQUARE_SIZE + 1


def _certificate_integer(
        certificate: dict[str, Any], field: str, expected: int,
        path: Path) -> int:
    value = certificate.get(field)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ReportError(
            f"{path}: {RESET_CERTIFICATE_KEY}.{field} is not an integer")
    if value != expected:
        raise ReportError(
            f"{path}: {RESET_CERTIFICATE_KEY}.{field}={value}, "
            f"expected {expected}")
    return value


def _validate_reset_log(
        root: dict[str, Any], report_path: Path, policy_root: Path,
        spec: ModeSpec, concurrency: int,
        repeat: int) -> tuple[Path, str, int, int]:
    reset_log = (
        policy_root / f"concurrency_{concurrency}" /
        f"repeat_{repeat}" / RESET_LOG_NAME)
    try:
        payload = reset_log.read_bytes()
        text = payload.decode("utf-8")
    except (OSError, UnicodeDecodeError) as error:
        raise ReportError(
            f"missing readable before-case reset log: {reset_log}: "
            f"{error}") from error
    certificate = root.get(RESET_CERTIFICATE_KEY)
    if not isinstance(certificate, dict):
        raise ReportError(
            f"{report_path}: missing JSON object {RESET_CERTIFICATE_KEY}")
    _certificate_integer(certificate, "schema_version", 1, report_path)
    if certificate.get("policy") != spec.name:
        raise ReportError(
            f"{report_path}: {RESET_CERTIFICATE_KEY}.policy="
            f"{certificate.get('policy')!r}, expected {spec.name!r}")
    _certificate_integer(
        certificate, "concurrency", concurrency, report_path)
    _certificate_integer(certificate, "repetition", repeat, report_path)
    latin_position = _expected_latin_position(spec.name, repeat)
    latin_cycle = (repeat - 1) // LATIN_SQUARE_SIZE + 1
    _certificate_integer(
        certificate, "latin_position", latin_position, report_path)
    _certificate_integer(
        certificate, "latin_cycle", latin_cycle, report_path)

    snapshot_id = certificate.get("snapshot_id")
    if (not isinstance(snapshot_id, str) or
            SNAPSHOT_ID_PATTERN.fullmatch(snapshot_id) is None):
        raise ReportError(
            f"{report_path}: {RESET_CERTIFICATE_KEY}.snapshot_id is not a "
            "valid immutable snapshot identifier")
    reported_digest = certificate.get("reset_log_sha256")
    if (not isinstance(reported_digest, str) or
            RESET_DIGEST_PATTERN.fullmatch(reported_digest) is None):
        raise ReportError(
            f"{report_path}: {RESET_CERTIFICATE_KEY}.reset_log_sha256 is "
            "not a lowercase SHA-256 digest")
    observed_digest = hashlib.sha256(payload).hexdigest()
    if reported_digest != observed_digest:
        raise ReportError(
            f"{report_path}: reset log digest mismatch: report binds "
            f"{reported_digest}, observed {observed_digest} for {reset_log}")

    required_lines = (
        f"policy={spec.name}",
        f"concurrency={concurrency}",
        f"repetition={repeat}",
        f"latin_position={latin_position}",
        f"latin_cycle={latin_cycle}",
        "exit_status=0",
    )
    lines = set(text.splitlines())
    missing = [line for line in required_lines if line not in lines]
    if missing:
        raise ReportError(
            f"{reset_log}: incomplete/unsuccessful reset certificate; "
            f"missing {missing}")
    snapshot_lines = [
        line.removeprefix("snapshot_id=")
        for line in text.splitlines()
        if line.startswith("snapshot_id=")
    ]
    if snapshot_lines != [snapshot_id]:
        raise ReportError(
            f"{report_path}: report snapshot_id={snapshot_id!r} is not "
            f"bound to exactly one matching line in {reset_log}: "
            f"observed {snapshot_lines!r}")
    return reset_log.resolve(), snapshot_id, latin_position, latin_cycle


def _validate_dynamic_telemetry(
        root: dict[str, Any], spec: ModeSpec, path: Path,
        query_count: int) -> dict[str, float | int]:
    prefix = ("gpu_persistent",)
    raw = {
        field: mixed._require_integer(root, prefix + (field,), path)
        for field in DYNA_RAW_FIELDS
    }
    short_reads = raw["dynamic_graph_short_reads"]
    full_reads = raw["dynamic_graph_full_reads"]
    read_bytes = raw["dynamic_graph_read_bytes"]
    fallback_reads = raw["dynamic_graph_fallback_reads"]
    promotions = raw["dynamic_graph_hint_promotions"]
    demotions = raw["dynamic_graph_hint_demotions"]
    attempts = short_reads + full_reads
    nonfallback_full = max(full_reads - fallback_reads, 0)

    if attempts == 0:
        raise ReportError(
            f"{path}: the measured workload exercised no dynamic graph "
            "snapshot attempts")
    if read_bytes == 0:
        raise ReportError(
            f"{path}: dynamic graph attempts reported zero transferred bytes")
    if fallback_reads > short_reads or fallback_reads > full_reads:
        raise ReportError(
            f"{path}: dynamic fallback reads={fallback_reads} exceed "
            f"short/full physical attempts={short_reads}/{full_reads}")
    if promotions > fallback_reads:
        raise ReportError(
            f"{path}: dynamic hint promotions={promotions} exceed "
            f"fallback repairs={fallback_reads}")
    if demotions > attempts:
        raise ReportError(
            f"{path}: dynamic hint demotions={demotions} exceed "
            f"physical attempts={attempts}")

    global_limits = {
        "dynamic_graph_short_reads": "graph_live_extent_reads",
        "dynamic_graph_full_reads": "graph_full_record_reads",
        "dynamic_graph_read_bytes": "graph_read_bytes",
        "dynamic_graph_fallback_reads": "graph_extent_fallback_reads",
        "dynamic_graph_hint_promotions": "graph_extent_hint_promotions",
    }
    for dynamic_field, aggregate_field in global_limits.items():
        aggregate = mixed._require_integer(
            root, prefix + (aggregate_field,), path)
        if raw[dynamic_field] > aggregate:
            raise ReportError(
                f"{path}: {dynamic_field}={raw[dynamic_field]} exceeds "
                f"aggregate {aggregate_field}={aggregate}")

    record_bytes = mixed._require_integer(
        root, ("meta", "gpu_graph_physical_record_bytes"), path,
        positive=True)
    if read_bytes < full_reads * record_bytes:
        raise ReportError(
            f"{path}: dynamic graph bytes={read_bytes} are below "
            f"full-attempt minimum={full_reads * record_bytes}")
    if read_bytes > attempts * record_bytes:
        raise ReportError(
            f"{path}: dynamic graph bytes={read_bytes} exceed "
            f"physical-attempt maximum={attempts * record_bytes}")

    if spec.dynamic_enabled:
        if short_reads == 0:
            raise ReportError(
                f"{path}: DynaExtent is enabled but no dynamic short "
                "physical attempt was observed")
    elif any((short_reads, fallback_reads, promotions, demotions)):
        raise ReportError(
            f"{path}: {spec.name} disables DynaExtent but reported "
            f"short/fallback/promotions/demotions="
            f"{short_reads}/{fallback_reads}/{promotions}/{demotions}")
    elif full_reads == 0:
        raise ReportError(
            f"{path}: {spec.name} did not exercise dynamic full-record reads")

    expected: dict[str, float | int] = {
        "dynamic_graph_snapshot_attempts": attempts,
        "dynamic_graph_nonfallback_full_attempts": nonfallback_full,
        "dynamic_graph_short_physical_ratio": short_reads / attempts,
        "dynamic_graph_fallback_ratio": (
            fallback_reads / short_reads if short_reads else 0.0),
        "average_dynamic_graph_read_bytes_per_physical_read":
            read_bytes / attempts,
        "average_dynamic_graph_read_bytes_per_query":
            read_bytes / query_count,
    }
    derived: dict[str, float | int] = {}
    for field in DYNA_PHYSICAL_DERIVED_FIELDS:
        if field in (
                "dynamic_graph_snapshot_attempts",
                "dynamic_graph_nonfallback_full_attempts"):
            reported: float | int = mixed._require_integer(
                root, prefix + (field,), path)
            if reported != expected[field]:
                raise ReportError(
                    f"{path}: gpu_persistent.{field}={reported} does not "
                    f"match derived physical value {expected[field]}")
        else:
            reported = mixed._require_number(
                root, prefix + (field,), path)
            mixed._require_close(
                float(reported), float(expected[field]),
                f"gpu_persistent.{field}", path)
        derived[field] = reported
    per_query: dict[str, float] = {
        "dynamic_graph_short_reads_per_query": short_reads / query_count,
        "dynamic_graph_full_reads_per_query": full_reads / query_count,
        "dynamic_graph_fallback_reads_per_query":
            fallback_reads / query_count,
        "dynamic_graph_hint_promotions_per_query": promotions / query_count,
        "dynamic_graph_hint_demotions_per_query": demotions / query_count,
        "dynamic_graph_snapshot_attempts_per_query": attempts / query_count,
    }
    return {**raw, **derived, **per_query}


def load_report(
        path: Path, mode: str, concurrency: int, repeat: int,
        policy_root: Path) -> ModeReport:
    if mode not in MODE_SPECS:
        raise ReportError(f"unsupported DynaExtent directory policy: {mode}")
    spec = MODE_SPECS[mode]
    root = _read_json(path)
    _validate_mode_mapping(root, spec, path)
    base_report = mixed.load_report(
        path, spec.graph_policy, concurrency, repeat, CONTRACT)
    query_count = mixed._require_integer(
        root, ("gpu_persistent", "queries_completed"), path,
        positive=True)
    metrics: dict[str, float | int] = dict(base_report.metrics)
    metrics.update(_validate_dynamic_telemetry(
        root, spec, path, query_count))
    missing = set(METRIC_NAMES) - set(metrics)
    if missing:
        raise AssertionError(
            f"internal analyzer error: missing metrics {sorted(missing)}")
    (reset_log, snapshot_id, latin_position,
     latin_cycle) = _validate_reset_log(
        root, path, policy_root, spec, concurrency, repeat)
    return ModeReport(
        path=base_report.path,
        reset_log=reset_log,
        mode=mode,
        concurrency=concurrency,
        repeat=repeat,
        snapshot_id=snapshot_id,
        latin_position=latin_position,
        latin_cycle=latin_cycle,
        pair_values=base_report.pair_values,
        metrics=metrics,
    )


def _format_observed(
        reports: tuple[ModeReport, ModeReport, ModeReport],
        key: str) -> str:
    return ", ".join(
        f"{report.mode}={report.pair_values.get(key, '<missing>')!r}"
        for report in reports)


def discover_triplets(
        root: Path, pattern: str = REPORT_PATTERN
        ) -> list[tuple[ModeReport, ModeReport, ModeReport]]:
    root = Path(root)
    if not root.is_dir():
        raise ReportError(f"result root is not a directory: {root}")
    errors: list[str] = []
    by_mode: dict[str, dict[tuple[int, int], ModeReport]] = {
        mode: {} for mode in MODES
    }

    for child in root.iterdir():
        if (child.is_dir() and child.name not in MODE_SPECS and
                any(child.rglob(pattern))):
            errors.append(
                f"unexpected report-bearing policy directory: {child}; "
                f"accepted policies are {list(MODES)}")

    for mode in MODES:
        policy_root = root / mode
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
                concurrency, repeat = mixed._parse_case_path(
                    path, policy_root)
                report = load_report(
                    path, mode, concurrency, repeat, policy_root)
            except ReportError as error:
                errors.append(str(error))
                continue
            case = (concurrency, repeat)
            previous = by_mode[mode].get(case)
            if previous is not None:
                errors.append(
                    f"duplicate reports for policy={mode}, "
                    f"concurrency={concurrency}, repeat={repeat}: "
                    f"{previous.path} and {path}")
                continue
            by_mode[mode][case] = report

    all_cases: set[tuple[int, int]] = set()
    for reports in by_mode.values():
        all_cases.update(reports)
    for case in sorted(all_cases):
        present = [mode for mode in MODES if case in by_mode[mode]]
        if len(present) != len(MODES):
            errors.append(
                f"incomplete triplet concurrency={case[0]}, "
                f"repeat={case[1]}: present={present}, "
                f"missing={sorted(set(MODES) - set(present))}")

    common_cases = set.intersection(
        *(set(by_mode[mode]) for mode in MODES))
    triplets: list[tuple[ModeReport, ModeReport, ModeReport]] = []
    for case in sorted(common_cases):
        reports = tuple(by_mode[mode][case] for mode in MODES)
        snapshot_ids = {report.snapshot_id for report in reports}
        if len(snapshot_ids) != 1:
            errors.append(
                f"non-comparable triplet concurrency={case[0]}, "
                f"repeat={case[1]}: reset snapshot_id mismatch: "
                + ", ".join(
                    f"{report.mode}={report.snapshot_id!r}"
                    for report in reports))
            continue
        keys: set[str] = set()
        for report in reports:
            keys.update(report.pair_values)
        mismatches = [
            key for key in sorted(
                keys,
                key=lambda name: (
                    CERTIFICATE_PRIORITY.get(name, 100), name))
            if len({
                repr(report.pair_values.get(key, "<missing>"))
                for report in reports
            }) != 1
        ]
        if mismatches:
            details = "; ".join(
                f"{key}: {_format_observed(reports, key)}"
                for key in mismatches[:20])
            if len(mismatches) > 20:
                details += f"; ... {len(mismatches) - 20} more"
            errors.append(
                f"non-comparable triplet concurrency={case[0]}, "
                f"repeat={case[1]}: {details}")
            continue
        triplets.append(reports)

    complete_repeats = sorted(report_group[0].repeat
                              for report_group in triplets)
    if complete_repeats:
        expected_repeats = list(range(1, complete_repeats[-1] + 1))
        if (complete_repeats != expected_repeats or
                len(complete_repeats) % LATIN_SQUARE_SIZE != 0):
            errors.append(
                "complete DynaExtent analysis requires consecutive "
                "repetitions starting at 1 and a whole 3x3 Latin-square "
                f"cycle; observed repeats={complete_repeats}")

    if errors:
        raise ReportError("\n".join(errors))
    if not triplets:
        raise ReportError(f"no complete DynaExtent triplets below {root}")
    return triplets


def _ratio(numerator: float | int,
           denominator: float | int) -> float | None:
    return None if denominator == 0 else numerator / denominator


def _comparison(
        numerator: ModeReport, denominator: ModeReport) -> dict[str, Any]:
    ratios = {
        name: _ratio(
            numerator.metrics[name], denominator.metrics[name])
        for name in METRIC_NAMES
    }
    deltas = {
        name: numerator.metrics[name] - denominator.metrics[name]
        for name in METRIC_NAMES
    }
    return {
        "numerator": numerator.mode,
        "denominator": denominator.mode,
        "ratio": ratios,
        "delta_numerator_minus_denominator": deltas,
        "change_fraction": {
            name: None if ratios[name] is None else ratios[name] - 1.0
            for name in METRIC_NAMES
        },
    }


def _median_optional(values: list[float | None]) -> float | None:
    present = [value for value in values if value is not None]
    return None if not present else statistics.median(present)


def build_summary(
        root: Path,
        triplets: list[tuple[ModeReport, ModeReport, ModeReport]],
        ) -> dict[str, Any]:
    repeats: list[dict[str, Any]] = []
    for fixed, static, dyna in sorted(
            triplets, key=lambda group: group[0].repeat):
        repeats.append({
            "repeat": fixed.repeat,
            "latin_cycle": fixed.latin_cycle,
            "snapshot_id": fixed.snapshot_id,
            "latin_positions": {
                report.mode: report.latin_position
                for report in (fixed, static, dyna)
            },
            "reports": {
                report.mode: str(report.path)
                for report in (fixed, static, dyna)
            },
            "reset_logs": {
                report.mode: str(report.reset_log)
                for report in (fixed, static, dyna)
            },
            "metrics": {
                report.mode: report.metrics
                for report in (fixed, static, dyna)
            },
            "comparisons": {
                "static_over_fixed": _comparison(static, fixed),
                "dyna_over_static": _comparison(dyna, static),
                "dyna_over_fixed": _comparison(dyna, fixed),
            },
        })

    policy_medians = {
        mode: {
            name: statistics.median([
                repeat["metrics"][mode][name] for repeat in repeats
            ])
            for name in METRIC_NAMES
        }
        for mode in MODES
    }
    paired_medians: dict[str, dict[str, dict[str, float | None]]] = {}
    for comparison_name in (
            "static_over_fixed", "dyna_over_static", "dyna_over_fixed"):
        paired_medians[comparison_name] = {
            "ratio": {
                name: _median_optional([
                    repeat["comparisons"][comparison_name]["ratio"][name]
                    for repeat in repeats
                ])
                for name in METRIC_NAMES
            },
            "delta_numerator_minus_denominator": {
                name: statistics.median([
                    repeat["comparisons"][comparison_name]
                    ["delta_numerator_minus_denominator"][name]
                    for repeat in repeats
                ])
                for name in METRIC_NAMES
            },
            "change_fraction": {
                name: _median_optional([
                    repeat["comparisons"][comparison_name]
                    ["change_fraction"][name]
                    for repeat in repeats
                ])
                for name in METRIC_NAMES
            },
        }

    return {
        "schema_version": 1,
        "result_root": str(Path(root).resolve()),
        "controlled_variable": "dynaextent_policy",
        "policies": {
            mode: {
                "gpu_query_graph_read_policy": spec.graph_policy,
                "gpu_dynamic_graph_extent": spec.dynamic_enabled,
                "gpu_dynamic_graph_extent_source": spec.dynamic_source,
            }
            for mode, spec in MODE_SPECS.items()
        },
        "experiment_contract": {
            "name": CONTRACT.name,
            "workload": "mixed",
            "mixed_dispatch_policy": CONTRACT.mixed_mode,
            "client_threads": CONTRACT.concurrency,
            "client_threads_source": CONTRACT.driver_source,
            "target_query_qps": CONTRACT.target_query_qps,
            "target_write_qps": CONTRACT.target_write_qps,
            "gpu_graph_prefetch_depth": 16,
            "gpu_query_beam_merge_policy": "stable-run",
            "traversal_beam_width": 128,
            "max_expansions": 384,
            "final_rerank_width": 128,
            "warmup_seconds": 30,
            "measure_seconds": 120,
            "recall_queries": 1000,
            "latin_square_size": LATIN_SQUARE_SIZE,
            "repetitions": len(repeats),
            "complete_latin_cycles": len(repeats) // LATIN_SQUARE_SIZE,
        },
        "comparability_guarantee": {
            "runner_requires": (
                "a trusted reset hook that emits one immutable snapshot_id "
                "or content digest per case"),
            "reports_verify": [
                "reset-log SHA-256 is embedded in and matches the report",
                "same snapshot_id within each three-policy repetition",
                "complete consecutive 3x3 Latin-square cycles",
                "same configured input source",
                "same insert ID range",
                "same initial Recall certificate",
                "same completed warmup and measured update counts",
            ],
            "not_claimed": (
                "independent verification that a trusted hook's snapshot_id "
                "describes storage contents, deterministic per-operation "
                "commit order among concurrent writers, or a mutation-order "
                "hash"),
        },
        "dynamic_telemetry_semantics": {
            "raw_fields": list(DYNA_RAW_FIELDS),
            "derived_physical_fields": list(
                DYNA_PHYSICAL_DERIVED_FIELDS),
            "per_query_fields": list(DYNA_PER_QUERY_FIELDS),
            "headline_uses_raw_totals": False,
            "snapshot_attempts_are_logical_reads": False,
        },
        "triplet_count": len(repeats),
        "policy_medians": policy_medians,
        "paired_medians": paired_medians,
        "repeats": repeats,
    }


def _format_number(value: float | int | None) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, int):
        return str(value)
    magnitude = abs(value)
    if magnitude and (magnitude >= 1e8 or magnitude < 1e-4):
        return f"{value:.6e}"
    return f"{value:.6f}"


HEADLINE_METRICS = (
    "query_qps",
    "write_qps",
    "durable_write_qps",
    "query_latency_mean_us",
    "query_latency_p99_us",
    "write_latency_mean_us",
    "write_latency_p99_us",
    "recall_before",
    "recall_after",
    "graph_bytes_per_query",
    "total_rdma_bytes_per_query",
    *DYNA_PER_QUERY_FIELDS,
    "dynamic_graph_short_physical_ratio",
    "dynamic_graph_fallback_ratio",
    "average_dynamic_graph_read_bytes_per_physical_read",
    "average_dynamic_graph_read_bytes_per_query",
)


def render_markdown(summary: dict[str, Any]) -> str:
    contract = summary["experiment_contract"]
    lines = [
        "# DynaExtent dynamic mixed-workload triplet",
        "",
        "The sole policy dimension is `fixed`, `static-only`, or "
        "`dynaextent`. Every report is bound to its successful reset log by "
        "SHA-256, and all three policies in a repetition carry the same "
        "trusted-hook snapshot ID. Reports additionally verify the same "
        "input/ID range, initial Recall, and completed update counts.",
        "",
        "The snapshot ID is a trusted reset-hook certificate, not an "
        "independent hash of live storage. Concurrent writers are not claimed "
        "to commit in an identical per-operation order, and no mutation-order "
        "hash is claimed.",
        "",
        "DynaExtent short/full values count physical snapshot attempts. "
        "`short + full` is not a dynamic logical-read count. Headline dynamic "
        "traffic uses per-query values or physical ratios rather than raw "
        "totals, so a query-attainment difference is not mistaken for a "
        "per-query mechanism difference.",
        "",
        f"Contract: **{contract['name']}**, clients="
        f"{contract['client_threads']} ({contract['client_threads_source']}), "
        f"query target={contract['target_query_qps']:.0f}/s, "
        f"write target={contract['target_write_qps']:.0f}/s",
        "",
        "Because this is a fixed offered-rate experiment, query QPS measures "
        "target attainment rather than maximum system capacity; latency and "
        "per-query transport are the primary mechanism outcomes.",
        "",
        f"Complete triplets: **{summary['triplet_count']}**",
        f"Complete 3x3 Latin cycles: "
        f"**{contract['complete_latin_cycles']}**",
        "",
        "## Policy mapping",
        "",
        "| directory | graph policy | dynamic extent | source |",
        "|---|---|---:|---|",
    ]
    for mode in MODES:
        policy = summary["policies"][mode]
        lines.append(
            f"| {mode} | {policy['gpu_query_graph_read_policy']} | "
            f"{str(policy['gpu_dynamic_graph_extent']).lower()} | "
            f"{policy['gpu_dynamic_graph_extent_source']} |")
    lines.extend([
        "",
        "## Headline policy medians",
        "",
        "| metric | fixed | static-only | dynaextent |",
        "|---|---:|---:|---:|",
    ])
    for name in HEADLINE_METRICS:
        lines.append(
            f"| {name} | "
            f"{_format_number(summary['policy_medians']['fixed'][name])} | "
            f"{_format_number(summary['policy_medians']['static-only'][name])} | "
            f"{_format_number(summary['policy_medians']['dynaextent'][name])} |")
    lines.extend([
        "",
        "## Paired median ratios",
        "",
        "| metric | static/fixed | Dyna/static | Dyna/fixed |",
        "|---|---:|---:|---:|",
    ])
    for name in HEADLINE_METRICS:
        lines.append(
            f"| {name} | "
            f"{_format_number(summary['paired_medians']['static_over_fixed']['ratio'][name])} | "
            f"{_format_number(summary['paired_medians']['dyna_over_static']['ratio'][name])} | "
            f"{_format_number(summary['paired_medians']['dyna_over_fixed']['ratio'][name])} |")
    lines.extend([
        "",
        "## All policy medians",
        "",
        "| metric | fixed | static-only | dynaextent |",
        "|---|---:|---:|---:|",
    ])
    for name in METRIC_NAMES:
        lines.append(
            f"| {name} | "
            f"{_format_number(summary['policy_medians']['fixed'][name])} | "
            f"{_format_number(summary['policy_medians']['static-only'][name])} | "
            f"{_format_number(summary['policy_medians']['dynaextent'][name])} |")
    lines.append("")
    return "\n".join(lines)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    default_root = (
        Path(__file__).resolve().parent / "results" /
        "dynaextent_mixed_ab")
    parser = argparse.ArgumentParser(
        description=(
            "Strictly validate and summarize fixed/static-only/DynaExtent "
            "dynamic mixed-workload triplets."))
    parser.add_argument(
        "root", type=Path, nargs="?", default=default_root,
        help=f"triplet result root (default: {default_root})")
    parser.add_argument(
        "--pattern", default=REPORT_PATTERN,
        help=f"report filename glob (default: {REPORT_PATTERN})")
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--markdown-output", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        triplets = discover_triplets(args.root, args.pattern)
        summary = build_summary(args.root, triplets)
        json_path = args.json_output or (
            args.root / "dynaextent_mixed_ab_summary.json")
        markdown_path = args.markdown_output or (
            args.root / "dynaextent_mixed_ab_summary.md")
        json_path.parent.mkdir(parents=True, exist_ok=True)
        markdown_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8")
        markdown_path.write_text(
            render_markdown(summary), encoding="utf-8")
    except ReportError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(f"json: {json_path}")
    print(f"markdown: {markdown_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
