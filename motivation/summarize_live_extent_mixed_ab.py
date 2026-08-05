#!/usr/bin/env python3
"""Strictly compare paired fixed/live-extent mixed-update reports.

This analyzer is deliberately separate from ``summarize_live_extent_ab.py``:
query-only Recall is expected to remain bit-for-bit stable, whereas a mixed
run may legitimately change post-workload Recall after applying updates.

The default contract is the pre-registered, open-loop causal experiment:

  * mixed/rate_limited, 40K query/s + 1K write/s
  * auto-derived 336 clients (256 query slots + 5 * 16 write RPCs)
  * fixed C16, stable-run, beam 128, max-expansions 384, rerank 128
  * 30 second warmup, 120 second measurement, 1000 Recall queries

``--contract fixed-threads`` retains the older 512-client, 50/50 closed-loop
experiment as an explicitly selected secondary analysis.  Reports are never
silently discarded: missing, duplicate, failed, under-attained, or
non-comparable pairs make the command fail.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import re
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


POLICIES = ("fixed", "live-extent")
REPORT_PATTERN = "sift100m_*.json"
CASE_COMPONENT = re.compile(r"concurrency_([1-9][0-9]*)")
REPEAT_COMPONENT = re.compile(r"repeat_([1-9][0-9]*)")


@dataclass(frozen=True)
class ExperimentContract:
    name: str
    concurrency: int
    mixed_mode: str
    read_ratio: float
    target_query_qps: float
    target_write_qps: float
    time_issue_policy: str
    driver_source: str
    driver_required_threads: int
    driver_derivation: str
    fixed_read_threads: int | None = None
    fixed_write_threads: int | None = None


CONTRACTS = {
    "rate-limited": ExperimentContract(
        name="rate-limited",
        concurrency=336,
        mixed_mode="rate_limited",
        read_ratio=0.5,
        target_query_qps=40_000.0,
        target_write_qps=1_000.0,
        time_issue_policy="shared_two_stream_pacer_until_deadline",
        driver_source="auto",
        driver_required_threads=336,
        driver_derivation=(
            "sum(active_bounded_path_capacities);shared_rate_pacer"),
    ),
    "fixed-threads": ExperimentContract(
        name="fixed-threads",
        concurrency=512,
        mixed_mode="fixed_threads",
        read_ratio=0.5,
        target_query_qps=0.0,
        target_write_qps=0.0,
        time_issue_policy="fixed_read_write_threads_until_deadline",
        driver_source="auto",
        driver_required_threads=512,
        driver_derivation=(
            "max(ceil(gpu_query_slots/read_ratio),"
            "ceil(shards_x_storage_rpc_depth/write_ratio))"),
        fixed_read_threads=256,
        fixed_write_threads=256,
    ),
}

# Runtime outcomes in ``meta`` must not be compared as configuration.  All
# other metadata is retained, so a newly added control silently becomes part
# of the "policy is the sole variable" audit.
META_OUTCOME_FIELDS = frozenset(("warmup_mixed", "measure_mixed"))
PERFORMANCE_QUERY_OUTCOME_FIELDS = frozenset((
    "warmup_rows_consumed",
    "measure_rows_consumed",
    "total_rows_consumed",
    "remaining_rows",
    "row_reuse_count",
))

GPU_PHASE_FIELDS = (
    "average_gpu_query_us",
    "average_gpu_prepare_us",
    "average_gpu_beam_selection_us",
    "average_gpu_rdma_issue_us",
    "average_gpu_rdma_wait_us",
    "average_gpu_graph_validation_us",
    "average_gpu_neighbor_decode_us",
    "average_gpu_pq_score_us",
    "average_gpu_visited_us",
    "average_gpu_beam_merge_us",
    "average_gpu_exact_us",
    "average_gpu_other_us",
)

LATENCY_METRICS = (
    "query_latency_mean_us",
    "query_latency_p50_us",
    "query_latency_p95_us",
    "query_latency_p99_us",
    "query_latency_p999_us",
    "write_latency_mean_us",
    "write_latency_p50_us",
    "write_latency_p95_us",
    "write_latency_p99_us",
    "write_latency_p999_us",
)

METRIC_NAMES = (
    "query_ops",
    "write_ops",
    "query_qps",
    "effective_query_qps",
    "query_rate_attainment_ratio",
    "write_qps",
    "effective_write_qps",
    "durable_write_qps",
    "write_rate_attainment_ratio",
    "total_qps",
    "durable_total_qps",
    *LATENCY_METRICS,
    "recall_before",
    "recall_after",
    "recall_change",
    *(field.removeprefix("average_") for field in GPU_PHASE_FIELDS),
    "logical_graph_reads_per_query",
    "graph_rounds_per_query",
    "graph_bytes_per_query",
    "graph_bytes_per_logical_parent",
    "physical_graph_wqes_per_query",
    "physical_graph_wqes_per_logical_parent",
    "total_rdma_bytes_per_query",
    "rdma_wqes_per_query",
    "short_graph_reads_per_query",
    "full_graph_reads_per_query",
    "fallback_graph_reads_per_query",
    "graph_extent_fallback_ratio",
    "underhint_graph_reads_per_query",
    "extent_hint_promotions_per_query",
    "extent_underhint_ratio",
    "extent_hint_promotion_rate",
    "graph_read_retries_per_query",
    "stage2_remaining",
    "stage2_max_backlog",
    "stage2_backlog_slope_per_sec",
    "stage2_p99_delay_upper_ms",
    "stage2_p99_delay_samples",
    "stage2_completion_outstanding",
    "stage2_max_completion_outstanding_per_shard",
    "stage2_finalized_live_delta",
    "stage2_pressure_yields",
    "stage2_peer_reverse_retry_attempts",
    "stage2_failures",
    "storage_late_rpc_completions",
    "maintenance_drain_seconds",
)


class ReportError(ValueError):
    """A report or pair violates the registered mixed-update contract."""


@dataclass(frozen=True)
class Report:
    path: Path
    policy: str
    concurrency: int
    repeat: int
    pair_values: dict[str, Any]
    metrics: dict[str, float]


def _field_name(parts: tuple[str, ...]) -> str:
    return ".".join(parts)


def _require_field(
        root: dict[str, Any], parts: tuple[str, ...], path: Path) -> Any:
    value: Any = root
    for part in parts:
        if not isinstance(value, dict) or part not in value:
            raise ReportError(
                f"{path}: missing JSON field {_field_name(parts)}")
        value = value[part]
    return value


def _require_equal(
        root: dict[str, Any], parts: tuple[str, ...], expected: Any,
        path: Path) -> None:
    value = _require_field(root, parts, path)
    if value != expected:
        raise ReportError(
            f"{path}: {_field_name(parts)}={value!r}, "
            f"expected {expected!r}")


def _require_number(
        root: dict[str, Any], parts: tuple[str, ...], path: Path, *,
        positive: bool = False, allow_negative: bool = False) -> float:
    value = _require_field(root, parts, path)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ReportError(
            f"{path}: {_field_name(parts)} is not numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ReportError(
            f"{path}: {_field_name(parts)} is not finite")
    if ((not allow_negative and number < 0.0) or
            (positive and number == 0.0)):
        qualifier = "positive" if positive else "nonnegative"
        raise ReportError(
            f"{path}: {_field_name(parts)} is not {qualifier}")
    return number


def _require_integer(
        root: dict[str, Any], parts: tuple[str, ...], path: Path, *,
        positive: bool = False) -> int:
    value = _require_field(root, parts, path)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ReportError(
            f"{path}: {_field_name(parts)} is not an integer")
    if value < 0 or (positive and value == 0):
        qualifier = "positive" if positive else "nonnegative"
        raise ReportError(
            f"{path}: {_field_name(parts)} is not {qualifier}")
    return value


def _optional_number_or_zero(
        root: dict[str, Any], parts: tuple[str, ...], path: Path) -> float:
    """Read backward-compatible telemetry while rejecting malformed values."""
    value: Any = root
    for part in parts[:-1]:
        if not isinstance(value, dict) or part not in value:
            raise ReportError(
                f"{path}: missing JSON object {_field_name(parts[:-1])}")
        value = value[part]
    if not isinstance(value, dict):
        raise ReportError(
            f"{path}: {_field_name(parts[:-1])} is not an object")
    if parts[-1] not in value:
        return 0.0
    return _require_number(root, parts, path)


def _require_close(
        reported: float, derived: float, label: str, path: Path) -> None:
    if not math.isclose(reported, derived, rel_tol=1e-9, abs_tol=1e-9):
        raise ReportError(
            f"{path}: {label}={reported} does not match "
            f"derived value {derived}")


def _flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    if not isinstance(value, dict):
        return {prefix: value}
    flattened: dict[str, Any] = {}
    for key in sorted(value):
        child = f"{prefix}.{key}" if prefix else key
        flattened.update(_flatten(value[key], child))
    return flattened


def _controlled_meta(root: dict[str, Any], path: Path) -> dict[str, Any]:
    meta = copy.deepcopy(_require_field(root, ("meta",), path))
    if not isinstance(meta, dict):
        raise ReportError(f"{path}: meta is not an object")
    meta.pop("gpu_query_graph_read_policy", None)
    # Derived components of the graph-read policy, not independent controls.
    # fixed reports disabled/full while live-extent may enable DynaExtent.
    meta.pop("gpu_dynamic_graph_extent", None)
    meta.pop("gpu_dynamic_graph_extent_source", None)
    # Historical reports may contain the removed expansion-policy field.
    # It is neither required nor an A/B pairing dimension.
    meta.pop("gpu_query_expansion_policy", None)
    for field in META_OUTCOME_FIELDS:
        meta.pop(field, None)
    performance_query = meta.get("performance_query")
    if not isinstance(performance_query, dict):
        raise ReportError(f"{path}: meta.performance_query is not an object")
    for field in PERFORMANCE_QUERY_OUTCOME_FIELDS:
        performance_query.pop(field, None)
    return meta


def _validate_contract(
        root: dict[str, Any], policy: str, contract: ExperimentContract,
        path: Path) -> None:
    expected = {
        ("meta", "gpu_query_graph_read_policy"): policy,
        ("meta", "client_threads"): contract.concurrency,
        ("meta", "workload"): "mixed",
        ("meta", "mixed_dispatch_policy"): contract.mixed_mode,
        ("meta", "read_ratio"): contract.read_ratio,
        ("meta", "target_query_qps"): contract.target_query_qps,
        ("meta", "target_write_qps"): contract.target_write_qps,
        ("meta", "time_issue_policy"): contract.time_issue_policy,
        ("meta", "time_completion_policy"): "drain",
        ("meta", "run_mode"): "time",
        ("meta", "recall_only"): False,
        ("meta", "fine_grained_breakdown_enabled"): True,
        ("meta", "gpu_graph_prefetch_depth"): 16,
        ("meta", "gpu_query_beam_merge_policy"): "stable-run",
        ("meta", "traversal_beam_width"): 128,
        ("meta", "max_expansions"): 384,
        ("meta", "final_rerank_width"): 128,
        ("meta", "warmup_seconds"): 30,
        ("meta", "measure_seconds"): 120,
        ("meta", "write_insert_ratio"): 1.0,
        ("meta", "write_upsert_ratio"): 0.0,
        ("meta", "write_delete_ratio"): 0.0,
        ("meta", "normalized_write_mix", "insert"): 1.0,
        ("meta", "normalized_write_mix", "upsert"): 0.0,
        ("meta", "normalized_write_mix", "delete"): 0.0,
        ("meta", "performance_query", "row_reuse_policy"):
            "single_pass_no_reuse",
        ("meta", "performance_query", "row_reuse_count"): 0,
        ("meta", "benchmark_driver_concurrency", "semantics"):
            "closed_loop_synchronous_no_drop",
        ("meta", "benchmark_driver_concurrency", "client_threads_source"):
            contract.driver_source,
        ("meta", "benchmark_driver_concurrency", "selected_client_threads"):
            contract.concurrency,
        ("meta", "benchmark_driver_concurrency", "auto_required_threads"):
            contract.driver_required_threads,
        ("meta", "benchmark_driver_concurrency", "auto_cap_applied"): False,
        ("meta", "benchmark_driver_concurrency",
         "gpu_query_slot_capacity"): 256,
        ("meta", "benchmark_driver_concurrency",
         "storage_rpc_inflight_capacity"): 80,
        ("meta", "benchmark_driver_concurrency", "storage_shards"): 5,
        ("meta", "benchmark_driver_concurrency",
         "storage_rpc_depth_per_shard"): 16,
        ("meta", "benchmark_driver_concurrency", "derivation"):
            contract.driver_derivation,
    }
    for parts, value in expected.items():
        _require_equal(root, parts, value, path)

    if contract.fixed_read_threads is None:
        if "mixed_fixed_threads" in _require_field(root, ("meta",), path):
            raise ReportError(
                f"{path}: rate-limited report unexpectedly contains "
                "meta.mixed_fixed_threads")
    else:
        _require_equal(
            root, ("meta", "mixed_fixed_threads", "read_threads"),
            contract.fixed_read_threads, path)
        _require_equal(
            root, ("meta", "mixed_fixed_threads", "write_threads"),
            contract.fixed_write_threads, path)


MIXED_COUNTERS = (
    "issued_reads",
    "issued_writes",
    "issued_inserts",
    "issued_upserts",
    "issued_deletes",
    "completed_reads",
    "completed_writes",
    "completed_inserts",
    "completed_upserts",
    "completed_deletes",
    "scheduled_reads",
    "scheduled_writes",
)


def _validate_mixed_phase(
        root: dict[str, Any], name: str, seconds: int,
        contract: ExperimentContract, path: Path) -> dict[str, int]:
    prefix = ("meta", name)
    counters = {
        field: _require_integer(root, prefix + (field,), path)
        for field in MIXED_COUNTERS
    }
    if counters["issued_reads"] != counters["completed_reads"]:
        raise ReportError(
            f"{path}: {name} read accounting mismatch: "
            f"issued={counters['issued_reads']}, "
            f"completed={counters['completed_reads']}")
    if counters["issued_writes"] != counters["completed_writes"]:
        raise ReportError(
            f"{path}: {name} write accounting mismatch: "
            f"issued={counters['issued_writes']}, "
            f"completed={counters['completed_writes']}")
    for prefix_name in ("issued", "completed"):
        total = sum(
            counters[f"{prefix_name}_{kind}"]
            for kind in ("inserts", "upserts", "deletes"))
        if counters[f"{prefix_name}_writes"] != total:
            raise ReportError(
                f"{path}: {name} {prefix_name}_writes="
                f"{counters[f'{prefix_name}_writes']} but subtype sum={total}")
    if counters["completed_reads"] == 0 or counters["completed_writes"] == 0:
        raise ReportError(
            f"{path}: {name} must complete both reads and writes")

    if contract.mixed_mode == "rate_limited":
        expected_reads = math.ceil(contract.target_query_qps * seconds)
        expected_writes = math.ceil(contract.target_write_qps * seconds)
        if (counters["scheduled_reads"], counters["scheduled_writes"]) != (
                expected_reads, expected_writes):
            raise ReportError(
                f"{path}: {name} scheduled read/write count "
                f"{counters['scheduled_reads']}/"
                f"{counters['scheduled_writes']} does not match "
                f"{expected_reads}/{expected_writes}")
        if (counters["completed_reads"] > expected_reads or
                counters["completed_writes"] > expected_writes):
            raise ReportError(
                f"{path}: {name} completed more operations than scheduled: "
                f"completed={counters['completed_reads']}/"
                f"{counters['completed_writes']}, "
                f"scheduled={expected_reads}/{expected_writes}")
    elif counters["scheduled_reads"] != 0 or counters["scheduled_writes"] != 0:
        raise ReportError(
            f"{path}: {name} fixed-thread run has scheduled counters")
    return counters


RECALL_PROTOCOL_FIELDS = (
    "k",
    "mode",
    "queries",
    "query_file",
    "groundtruth_file",
    "base_id_limit",
    "search_result_width",
)


def _validate_recall(
        root: dict[str, Any], section: str, phase: str,
        path: Path) -> tuple[dict[str, Any], float]:
    prefix = (section,)
    _require_equal(root, prefix + ("phase",), phase, path)
    _require_equal(root, prefix + ("result_set_complete",), True, path)
    _require_equal(
        root, prefix + ("queries_with_insufficient_base_results",), 0, path)
    _require_equal(root, prefix + ("queries",), 1000, path)
    recall = _require_number(root, prefix + ("recall",), path)
    if recall > 1.0:
        raise ReportError(f"{path}: {section}.recall={recall} is above 1")
    protocol = {
        field: _require_field(root, prefix + (field,), path)
        for field in RECALL_PROTOCOL_FIELDS
    }
    return protocol, recall


def _validate_throughput(
        root: dict[str, Any], measure: dict[str, int],
        contract: ExperimentContract, path: Path) -> dict[str, float]:
    throughput = ("throughput",)
    query_ops = _require_integer(root, throughput + ("query_ops",), path)
    write_ops = _require_integer(root, throughput + ("write_ops",), path)
    insert_ops = _require_integer(root, throughput + ("insert_ops",), path)
    upsert_ops = _require_integer(root, throughput + ("upsert_ops",), path)
    delete_ops = _require_integer(root, throughput + ("delete_ops",), path)
    total_ops = _require_integer(root, throughput + ("total_ops",), path)
    expected = (
        measure["completed_reads"],
        measure["completed_writes"],
        measure["completed_inserts"],
        measure["completed_upserts"],
        measure["completed_deletes"],
    )
    if (query_ops, write_ops, insert_ops, upsert_ops, delete_ops) != expected:
        raise ReportError(
            f"{path}: throughput operation accounting does not match "
            "meta.measure_mixed")
    if total_ops != query_ops + write_ops:
        raise ReportError(
            f"{path}: throughput.total_ops={total_ops}, expected "
            f"{query_ops + write_ops}")

    duration = _require_number(
        root, throughput + ("duration_seconds",), path, positive=True)
    query_duration = _require_number(
        root, throughput + ("query_duration_seconds",), path, positive=True)
    write_duration = _require_number(
        root, throughput + ("write_duration_seconds",), path, positive=True)
    durable_duration = _require_number(
        root, throughput + ("durable_effective_measure_seconds",), path,
        positive=True)
    reported = {
        "query_ops": float(query_ops),
        "write_ops": float(write_ops),
        "query_qps": _require_number(
            root, throughput + ("query_ops_per_sec",), path, positive=True),
        "effective_query_qps": _require_number(
            root, throughput + ("effective_query_ops_per_sec",), path,
            positive=True),
        "query_rate_attainment_ratio": _require_number(
            root, throughput + ("query_rate_attainment_ratio",), path),
        "write_qps": _require_number(
            root, throughput + ("write_ops_per_sec",), path, positive=True),
        "effective_write_qps": _require_number(
            root, throughput + ("effective_write_ops_per_sec",), path,
            positive=True),
        "durable_write_qps": _require_number(
            root, throughput + ("durable_write_ops_per_sec",), path,
            positive=True),
        "write_rate_attainment_ratio": _require_number(
            root, throughput + ("write_rate_attainment_ratio",), path),
        "total_qps": _require_number(
            root, throughput + ("total_ops_per_sec",), path, positive=True),
        "durable_total_qps": _require_number(
            root, throughput + ("durable_total_ops_per_sec",), path,
            positive=True),
    }
    nominal_duration = (
        120.0 if contract.mixed_mode == "rate_limited" else query_duration)
    _require_close(
        reported["query_qps"], query_ops / nominal_duration,
        "throughput.query_ops_per_sec", path)
    nominal_write_duration = (
        120.0 if contract.mixed_mode == "rate_limited" else write_duration)
    _require_close(
        reported["write_qps"], write_ops / nominal_write_duration,
        "throughput.write_ops_per_sec", path)
    _require_close(
        reported["effective_query_qps"], query_ops / query_duration,
        "throughput.effective_query_ops_per_sec", path)
    _require_close(
        reported["effective_write_qps"], write_ops / write_duration,
        "throughput.effective_write_ops_per_sec", path)
    _require_close(
        reported["durable_write_qps"], write_ops / durable_duration,
        "throughput.durable_write_ops_per_sec", path)
    _require_close(
        reported["total_qps"], total_ops / duration,
        "throughput.total_ops_per_sec", path)
    _require_close(
        reported["durable_total_qps"], total_ops / durable_duration,
        "throughput.durable_total_ops_per_sec", path)
    scheduled_reads = measure["scheduled_reads"]
    scheduled_writes = measure["scheduled_writes"]
    _require_close(
        reported["query_rate_attainment_ratio"],
        query_ops / scheduled_reads if scheduled_reads else 1.0,
        "throughput.query_rate_attainment_ratio", path)
    _require_close(
        reported["write_rate_attainment_ratio"],
        write_ops / scheduled_writes if scheduled_writes else 1.0,
        "throughput.write_rate_attainment_ratio", path)
    return reported


def _latency_metrics(
        root: dict[str, Any], section: str, prefix: str,
        expected_count: int, path: Path) -> dict[str, float]:
    _require_equal(root, (section, "count"), expected_count, path)
    _require_equal(root, (section, "operation"),
                   "query" if prefix == "query" else "insert", path)
    result = {}
    fields = (
        ("mean", "mean_end_to_end_ns"),
        ("p50", "p50_end_to_end_ns"),
        ("p95", "p95_end_to_end_ns"),
        ("p99", "p99_end_to_end_ns"),
        ("p999", "p999_end_to_end_ns"),
    )
    for label, field in fields:
        value = _require_number(
            root, (section, "latency", field), path, positive=True)
        result[f"{prefix}_latency_{label}_us"] = value / 1000.0
    return result


def _validate_stage2(
        root: dict[str, Any], path: Path) -> tuple[dict[str, Any],
                                                  dict[str, float]]:
    required_true = (
        "backlog_slope_available",
        "p99_stage2_delay_available",
        "failure_delta_available",
        "execution_counter_delta_available",
        "completion_window_available",
        "locality_delta_available",
        "peer_reverse_retry_delta_available",
        "search_budget_delta_available",
    )
    for field in required_true:
        _require_equal(root, ("stage2", field), True, path)
    _require_equal(root, ("stage2", "failures"), 0, path)
    unreadable = _require_field(root, ("stage2", "unreadable_logs"), path)
    if unreadable != []:
        raise ReportError(
            f"{path}: stage2.unreadable_logs is not empty: {unreadable!r}")
    requested = _require_integer(
        root, ("stage2", "requested_logs"), path, positive=True)
    observed_logs = _require_integer(
        root, ("stage2", "logs_with_observations"), path, positive=True)
    if requested != observed_logs:
        raise ReportError(
            f"{path}: Stage2 telemetry is incomplete: requested_logs="
            f"{requested}, logs_with_observations={observed_logs}")
    samples = _require_number(
        root, ("stage2", "p99_stage2_delay_samples"), path, positive=True)
    finalized = _require_number(
        root, ("stage2", "stage2_finalized_live_delta"), path, positive=True)
    max_completion_outstanding = _require_number(
        root,
        ("stage2", "max_completion_outstanding_per_shard"),
        path,
        positive=True,
    )

    pair_contract = {
        "source": _require_field(root, ("stage2", "source"), path),
        "requested_logs": requested,
        "observation_period_seconds_assumed": _require_number(
            root, ("stage2", "observation_period_seconds_assumed"), path,
            positive=True),
        "admission_window": _require_number(
            root, ("stage2", "admission_window"), path, positive=True),
    }
    metrics = {
        "stage2_remaining": _require_number(
            root, ("stage2", "remaining"), path),
        "stage2_max_backlog": _require_number(
            root, ("stage2", "max_backlog_observed"), path),
        "stage2_backlog_slope_per_sec": _require_number(
            root, ("stage2", "backlog_slope_per_sec"), path,
            allow_negative=True),
        "stage2_p99_delay_upper_ms": _require_number(
            root, ("stage2", "p99_stage2_delay_upper_ms"), path),
        "stage2_p99_delay_samples": samples,
        "stage2_completion_outstanding": _require_number(
            root, ("stage2", "completion_outstanding"), path),
        "stage2_max_completion_outstanding_per_shard":
            max_completion_outstanding,
        "stage2_finalized_live_delta": finalized,
        "stage2_pressure_yields": _require_number(
            root, ("stage2", "pressure_yields"), path),
        "stage2_peer_reverse_retry_attempts": _require_number(
            root, ("stage2", "peer_reverse_retry_attempts"), path),
        "stage2_failures": 0.0,
    }
    return pair_contract, metrics


def load_report(
        path: Path, policy: str, concurrency: int, repeat: int,
        contract: ExperimentContract) -> Report:
    try:
        with path.open("r", encoding="utf-8") as stream:
            root = json.load(stream)
    except (OSError, json.JSONDecodeError) as error:
        raise ReportError(f"{path}: cannot read JSON: {error}") from error
    if not isinstance(root, dict):
        raise ReportError(f"{path}: JSON root is not an object")

    _validate_contract(root, policy, contract, path)
    if concurrency != contract.concurrency:
        raise ReportError(
            f"{path}: case concurrency={concurrency}, expected "
            f"{contract.concurrency} for contract {contract.name}")

    warmup = _validate_mixed_phase(
        root, "warmup_mixed", 30, contract, path)
    measure = _validate_mixed_phase(
        root, "measure_mixed", 120, contract, path)

    performance = ("meta", "performance_query")
    _require_equal(
        root, performance + ("warmup_rows_consumed",),
        warmup["completed_reads"], path)
    _require_equal(
        root, performance + ("measure_rows_consumed",),
        measure["completed_reads"], path)
    total_rows = warmup["completed_reads"] + measure["completed_reads"]
    _require_equal(
        root, performance + ("total_rows_consumed",), total_rows, path)
    rows = _require_integer(root, performance + ("rows",), path, positive=True)
    remaining = _require_integer(
        root, performance + ("remaining_rows",), path)
    if total_rows + remaining != rows:
        raise ReportError(
            f"{path}: performance query row accounting mismatch")
    # This field is a capacity certificate computed before execution.  It is
    # the number of rows the pacer may schedule, not the number that happened
    # to be claimed before the two phase deadlines.
    expected_required = (
        warmup["scheduled_reads"] + measure["scheduled_reads"]
        if contract.mixed_mode == "rate_limited" else 0)
    _require_equal(
        root, performance + ("rate_limited_required_rows",),
        expected_required, path)

    query_count = measure["completed_reads"]
    write_count = measure["completed_writes"]
    completed = _require_integer(
        root, ("gpu_persistent", "queries_completed"), path, positive=True)
    submitted = _require_integer(
        root, ("gpu_persistent", "queries_submitted"), path, positive=True)
    if completed != query_count or submitted != query_count:
        raise ReportError(
            f"{path}: GPU query accounting mismatch: "
            f"submitted={submitted}, completed={completed}, "
            f"mixed_reads={query_count}")

    _require_equal(
        root, ("gpu_persistent", "direct_path_failures"), 0, path)
    _require_equal(
        root, ("gpu_persistent", "centroid_route_query_timeouts"), 0, path)
    _require_equal(
        root, ("storage_owner_runtime", "late_rpc_completions"), 0, path)
    for noun in ("batches", "items"):
        submitted_value = _require_integer(
            root, ("storage_owner_runtime", f"submitted_{noun}"), path,
            positive=True)
        completed_value = _require_integer(
            root, ("storage_owner_runtime", f"completed_{noun}"), path,
            positive=True)
        if submitted_value != completed_value:
            raise ReportError(
                f"{path}: storage-owner {noun} accounting mismatch: "
                f"submitted={submitted_value}, completed={completed_value}")
    targets = _require_field(
        root, ("storage_owner_runtime", "maintenance_target_sequences"), path)
    durable = _require_field(
        root, ("storage_owner_runtime", "maintenance_durable_sequences"), path)
    if not isinstance(targets, list) or not targets or targets != durable:
        raise ReportError(
            f"{path}: maintenance durable watermarks do not match targets")

    before_protocol, before_recall = _validate_recall(
        root, "recall", "before_performance", path)
    after_protocol, after_recall = _validate_recall(
        root, "static_gt_post_recall", "after_performance", path)
    if before_protocol != after_protocol:
        raise ReportError(
            f"{path}: Recall protocol changed within the mixed run")

    metrics = _validate_throughput(root, measure, contract, path)
    metrics.update(_latency_metrics(
        root, "query_breakdown", "query", query_count, path))
    metrics.update(_latency_metrics(
        root, "insert_breakdown", "write", write_count, path))
    metrics.update({
        "recall_before": before_recall,
        "recall_after": after_recall,
        "recall_change": after_recall - before_recall,
    })
    for field in GPU_PHASE_FIELDS:
        metrics[field.removeprefix("average_")] = _require_number(
            root, ("gpu_persistent", field), path)

    graph_reads = _require_number(
        root, ("gpu_persistent", "graph_page_requests"), path, positive=True)
    graph_rounds = _require_number(
        root, ("gpu_persistent", "graph_dependency_rounds"), path,
        positive=True)
    graph_bytes = _require_number(
        root, ("gpu_persistent", "graph_read_bytes"), path, positive=True)
    rdma_bytes = _require_number(
        root, ("gpu_persistent", "rdma_read_bytes"), path, positive=True)
    rdma_wqes = _require_number(
        root, ("gpu_persistent", "rdma_read_ops"), path, positive=True)
    short_reads = _require_number(
        root, ("gpu_persistent", "graph_live_extent_reads"), path)
    full_reads = _require_number(
        root, ("gpu_persistent", "graph_full_record_reads"), path)
    fallback_reads = _require_number(
        root, ("gpu_persistent", "graph_extent_fallback_reads"), path)
    underhint_reads = _optional_number_or_zero(
        root, ("gpu_persistent", "graph_extent_underhint_reads"), path)
    hint_promotions = _optional_number_or_zero(
        root, ("gpu_persistent", "graph_extent_hint_promotions"), path)
    retries = _require_number(
        root, ("gpu_persistent", "graph_read_retries"), path)
    fallback_ratio = _require_number(
        root, ("gpu_persistent", "graph_extent_fallback_ratio"), path)
    derived_fallback_ratio = (
        fallback_reads / short_reads if short_reads else 0.0)
    _require_close(
        fallback_ratio, derived_fallback_ratio,
        "gpu_persistent.graph_extent_fallback_ratio", path)
    if rdma_bytes < graph_bytes:
        raise ReportError(
            f"{path}: total RDMA bytes are below graph bytes")
    if policy == "fixed" and (
            underhint_reads != 0 or hint_promotions != 0):
        raise ReportError(
            f"{path}: fixed policy reported live-extent adaptation: "
            f"underhint={underhint_reads}, promotions={hint_promotions}")
    physical_graph_wqes = short_reads + full_reads
    if physical_graph_wqes != graph_reads + retries:
        raise ReportError(
            f"{path}: physical graph WQEs={physical_graph_wqes}, "
            f"logical graph reads={graph_reads}, retries={retries}")
    if fallback_reads > retries:
        raise ReportError(
            f"{path}: fallback reads={fallback_reads} exceed all graph "
            f"read retries={retries}")
    if underhint_reads > fallback_reads:
        raise ReportError(
            f"{path}: underhint reads={underhint_reads} exceed fallback "
            f"reads={fallback_reads}")
    if hint_promotions > underhint_reads:
        raise ReportError(
            f"{path}: hint promotions={hint_promotions} exceed underhint "
            f"reads={underhint_reads}")
    if policy == "fixed":
        if (short_reads != 0 or fallback_reads != 0 or
                underhint_reads != 0 or hint_promotions != 0):
            raise ReportError(
                f"{path}: fixed policy reported live-extent work: "
                f"short={short_reads}, fallback={fallback_reads}, "
                f"underhint={underhint_reads}, "
                f"promotions={hint_promotions}")
        # ``graph_page_requests`` is logical parent work.  A failed snapshot
        # validation retries the same parent with another physical full-record
        # read, and both physical reads are intentionally counted.
        if full_reads != graph_reads + retries:
            raise ReportError(
                f"{path}: fixed full reads={full_reads}, "
                f"logical graph reads={graph_reads}, retries={retries}")
    elif short_reads == 0:
        raise ReportError(
            f"{path}: live-extent policy reported zero short reads")

    metrics.update({
        "logical_graph_reads_per_query": graph_reads / query_count,
        "graph_rounds_per_query": graph_rounds / query_count,
        "graph_bytes_per_query": graph_bytes / query_count,
        "graph_bytes_per_logical_parent": graph_bytes / graph_reads,
        "physical_graph_wqes_per_query":
            physical_graph_wqes / query_count,
        "physical_graph_wqes_per_logical_parent":
            physical_graph_wqes / graph_reads,
        "total_rdma_bytes_per_query": rdma_bytes / query_count,
        "rdma_wqes_per_query": rdma_wqes / query_count,
        "short_graph_reads_per_query": short_reads / query_count,
        "full_graph_reads_per_query": full_reads / query_count,
        "fallback_graph_reads_per_query": fallback_reads / query_count,
        "graph_extent_fallback_ratio": fallback_ratio,
        "underhint_graph_reads_per_query":
            underhint_reads / query_count,
        "extent_hint_promotions_per_query":
            hint_promotions / query_count,
        "extent_underhint_ratio":
            underhint_reads / short_reads if short_reads else 0.0,
        "extent_hint_promotion_rate":
            hint_promotions / underhint_reads if underhint_reads else 0.0,
        "graph_read_retries_per_query": retries / query_count,
    })
    for field, derived in (
        ("average_graph_rounds_per_query",
         metrics["graph_rounds_per_query"]),
        ("average_graph_read_bytes_per_query",
         metrics["graph_bytes_per_query"]),
        ("average_graph_read_bytes_per_logical_parent",
         metrics["graph_bytes_per_logical_parent"]),
    ):
        reported = _require_number(
            root, ("gpu_persistent", field), path)
        _require_close(reported, derived, f"gpu_persistent.{field}", path)

    stage2_pair, stage2_metrics = _validate_stage2(root, path)
    metrics.update(stage2_metrics)
    metrics["storage_late_rpc_completions"] = 0.0
    metrics["maintenance_drain_seconds"] = _require_number(
        root, ("storage_owner_runtime", "maintenance_drain_seconds"), path)

    missing_metrics = set(METRIC_NAMES) - set(metrics)
    if missing_metrics:
        raise AssertionError(
            f"internal analyzer error: missing metrics {sorted(missing_metrics)}")
    pair_values = _flatten(_controlled_meta(root, path), "meta")
    pair_values.update(
        _flatten(stage2_pair, "stage2_observation_contract"))
    pair_values.update(
        _flatten(before_protocol, "recall_protocol"))
    # The pre-workload Recall is an initial-state certificate.  Post-workload
    # Recall remains an outcome and is intentionally not a pair control.
    pair_values["initial_recall"] = before_recall
    # Rate limiting controls offered work, but claims scheduled immediately
    # before the deadline can legitimately miss that deadline.  Do not erase
    # this performance outcome.  For a causal update A/B, however, both sides
    # must apply exactly the same warmup and measured mutations.
    pair_values.update({
        "warmup_completed_writes": warmup["completed_writes"],
        "warmup_completed_inserts": warmup["completed_inserts"],
        "warmup_completed_upserts": warmup["completed_upserts"],
        "warmup_completed_deletes": warmup["completed_deletes"],
        "measure_completed_writes": measure["completed_writes"],
        "measure_completed_inserts": measure["completed_inserts"],
        "measure_completed_upserts": measure["completed_upserts"],
        "measure_completed_deletes": measure["completed_deletes"],
    })

    return Report(
        path=path.resolve(),
        policy=policy,
        concurrency=concurrency,
        repeat=repeat,
        pair_values=pair_values,
        metrics=metrics,
    )


def _parse_case_path(path: Path, policy_root: Path) -> tuple[int, int]:
    parts = path.relative_to(policy_root).parts
    if len(parts) < 3:
        raise ReportError(
            f"{path}: expected concurrency_N/repeat_R/**/{REPORT_PATTERN}")
    concurrency_match = CASE_COMPONENT.fullmatch(parts[0])
    repeat_match = REPEAT_COMPONENT.fullmatch(parts[1])
    if concurrency_match is None or repeat_match is None:
        raise ReportError(
            f"{path}: expected concurrency_N/repeat_R as first two "
            "components below the policy directory")
    return int(concurrency_match.group(1)), int(repeat_match.group(1))


def discover_pairs(
        root: Path, contract: ExperimentContract,
        pattern: str = REPORT_PATTERN) -> list[tuple[Report, Report]]:
    root = Path(root)
    by_policy: dict[str, dict[tuple[int, int], Report]] = {
        policy: {} for policy in POLICIES
    }
    errors: list[str] = []
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
                    path, policy, concurrency, repeat, contract)
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
    for case in sorted(fixed_cases | live_cases):
        present = [policy for policy in POLICIES
                   if case in by_policy[policy]]
        if len(present) != len(POLICIES):
            errors.append(
                f"unpaired case concurrency={case[0]}, repeat={case[1]}: "
                f"present={present}, "
                f"missing={sorted(set(POLICIES) - set(present))}")

    pairs = []
    for case in sorted(fixed_cases & live_cases):
        fixed = by_policy["fixed"][case]
        live = by_policy["live-extent"][case]
        keys = set(fixed.pair_values) | set(live.pair_values)
        diagnostic_priority = {
            "warmup_completed_writes": 0,
            "warmup_completed_inserts": 1,
            "warmup_completed_upserts": 2,
            "warmup_completed_deletes": 3,
            "measure_completed_writes": 4,
            "measure_completed_inserts": 5,
            "measure_completed_upserts": 6,
            "measure_completed_deletes": 7,
        }
        mismatches = [
            (key, fixed.pair_values.get(key, "<missing>"),
             live.pair_values.get(key, "<missing>"))
            for key in sorted(
                keys,
                key=lambda name: (
                    diagnostic_priority.get(name, 100), name))
            if fixed.pair_values.get(key, "<missing>") !=
               live.pair_values.get(key, "<missing>")
        ]
        if mismatches:
            details = "; ".join(
                f"{name}: fixed={left!r}, live-extent={right!r}"
                for name, left, right in mismatches[:20])
            if len(mismatches) > 20:
                details += f"; ... {len(mismatches) - 20} more"
            errors.append(
                f"non-comparable pair concurrency={case[0]}, "
                f"repeat={case[1]}: {details}")
            continue
        pairs.append((fixed, live))

    if errors:
        raise ReportError("\n".join(errors))
    if not pairs:
        raise ReportError(f"no paired mixed Live-Extent reports below {root}")
    return pairs


def _ratio(numerator: float, denominator: float) -> float | None:
    return None if denominator == 0.0 else numerator / denominator


def _paired_comparison(fixed: Report, live: Report) -> dict[str, Any]:
    ratios = {
        name: _ratio(live.metrics[name], fixed.metrics[name])
        for name in METRIC_NAMES
    }
    deltas = {
        name: live.metrics[name] - fixed.metrics[name]
        for name in METRIC_NAMES
    }
    return {
        "ratio_live_over_fixed": ratios,
        "delta_live_minus_fixed": deltas,
        "change_fraction": {
            name: None if ratios[name] is None else ratios[name] - 1.0
            for name in METRIC_NAMES
        },
        "graph_bytes_reduction_fraction":
            1.0 - ratios["graph_bytes_per_query"],
        "total_rdma_bytes_reduction_fraction":
            1.0 - ratios["total_rdma_bytes_per_query"],
    }


def _median_optional(values: list[float | None]) -> float | None:
    present = [value for value in values if value is not None]
    return None if not present else statistics.median(present)


def build_summary(
        root: Path, pairs: list[tuple[Report, Report]],
        contract: ExperimentContract) -> dict[str, Any]:
    repeats = []
    for fixed, live in sorted(pairs, key=lambda pair: pair[0].repeat):
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
    policy_medians = {
        policy: {
            name: statistics.median([
                repeat["metrics"][policy][name] for repeat in repeats])
            for name in METRIC_NAMES
        }
        for policy in POLICIES
    }
    paired_medians = {
        kind: {
            name: _median_optional([
                repeat["paired"][kind][name] for repeat in repeats])
            for name in METRIC_NAMES
        }
        for kind in ("ratio_live_over_fixed", "change_fraction")
    }
    paired_medians["delta_live_minus_fixed"] = {
        name: statistics.median([
            repeat["paired"]["delta_live_minus_fixed"][name]
            for repeat in repeats])
        for name in METRIC_NAMES
    }
    paired_medians["graph_bytes_reduction_fraction"] = statistics.median([
        repeat["paired"]["graph_bytes_reduction_fraction"]
        for repeat in repeats])
    paired_medians["total_rdma_bytes_reduction_fraction"] = statistics.median([
        repeat["paired"]["total_rdma_bytes_reduction_fraction"]
        for repeat in repeats])
    return {
        "schema_version": 1,
        "result_root": str(Path(root).resolve()),
        "controlled_variable": "gpu_query_graph_read_policy",
        "policies": list(POLICIES),
        "experiment_contract": {
            "name": contract.name,
            "workload": "mixed",
            "mixed_dispatch_policy": contract.mixed_mode,
            "client_threads": contract.concurrency,
            "read_ratio": contract.read_ratio,
            "target_query_qps": contract.target_query_qps,
            "target_write_qps": contract.target_write_qps,
            "gpu_graph_prefetch_depth": 16,
            "gpu_query_beam_merge_policy": "stable-run",
            "traversal_beam_width": 128,
            "max_expansions": 384,
            "final_rerank_width": 128,
            "warmup_seconds": 30,
            "measure_seconds": 120,
            "recall_queries": 1000,
        },
        "pair_count": len(repeats),
        "policy_medians": policy_medians,
        "paired_medians": paired_medians,
        "repeats": repeats,
    }


def _format_number(value: float | None) -> str:
    if value is None:
        return "n/a"
    magnitude = abs(value)
    if magnitude and (magnitude >= 1e8 or magnitude < 1e-4):
        return f"{value:.6e}"
    return f"{value:.6f}"


def render_markdown(summary: dict[str, Any]) -> str:
    contract = summary["experiment_contract"]
    lines = [
        "# Live-Extent mixed-update A/B summary",
        "",
        "The graph-read policy is the sole algorithmic variable. Missing, "
        "duplicate, failed, under-attained, or non-comparable reports make "
        "the analyzer fail. Ratios are `live-extent / fixed`.",
        "",
        f"Contract: **{contract['name']}**, "
        f"clients={contract['client_threads']}, "
        f"query target={contract['target_query_qps']:.0f}/s, "
        f"write target={contract['target_write_qps']:.0f}/s",
        "",
        f"Paired repeats: **{summary['pair_count']}**",
        "",
        "## Headline paired medians",
        "",
        "| metric | fixed | live-extent | live/fixed | live-fixed |",
        "|---|---:|---:|---:|---:|",
    ]
    headline = (
        "query_qps", "write_qps", "durable_write_qps",
        "query_latency_mean_us", "query_latency_p99_us",
        "query_latency_p999_us", "write_latency_mean_us",
        "write_latency_p99_us", "gpu_query_us", "gpu_rdma_wait_us",
        "gpu_graph_validation_us", "graph_bytes_per_query",
        "physical_graph_wqes_per_query", "total_rdma_bytes_per_query",
        "rdma_wqes_per_query",
        "graph_extent_fallback_ratio", "extent_underhint_ratio",
        "extent_hint_promotion_rate", "stage2_max_backlog",
        "stage2_backlog_slope_per_sec", "stage2_p99_delay_upper_ms",
        "stage2_failures", "recall_before", "recall_after",
    )
    for name in headline:
        lines.append(
            f"| {name} | "
            f"{_format_number(summary['policy_medians']['fixed'][name])} | "
            f"{_format_number(summary['policy_medians']['live-extent'][name])} | "
            f"{_format_number(summary['paired_medians']['ratio_live_over_fixed'][name])} | "
            f"{_format_number(summary['paired_medians']['delta_live_minus_fixed'][name])} |")
    lines.extend([
        f"| graph_bytes_reduction_fraction | n/a | n/a | "
        f"{_format_number(summary['paired_medians']['graph_bytes_reduction_fraction'])} | n/a |",
        f"| total_rdma_bytes_reduction_fraction | n/a | n/a | "
        f"{_format_number(summary['paired_medians']['total_rdma_bytes_reduction_fraction'])} | n/a |",
        "",
        "## All metrics (policy medians)",
        "",
        "| metric | fixed | live-extent |",
        "|---|---:|---:|",
    ])
    for name in METRIC_NAMES:
        lines.append(
            f"| {name} | "
            f"{_format_number(summary['policy_medians']['fixed'][name])} | "
            f"{_format_number(summary['policy_medians']['live-extent'][name])} |")
    lines.extend([
        "",
        "## Every paired repeat",
        "",
        "| repeat | metric | fixed | live-extent | live/fixed | live-fixed |",
        "|---:|---|---:|---:|---:|---:|",
    ])
    for repeat in summary["repeats"]:
        for name in METRIC_NAMES:
            lines.append(
                f"| {repeat['repeat']} | {name} | "
                f"{_format_number(repeat['metrics']['fixed'][name])} | "
                f"{_format_number(repeat['metrics']['live-extent'][name])} | "
                f"{_format_number(repeat['paired']['ratio_live_over_fixed'][name])} | "
                f"{_format_number(repeat['paired']['delta_live_minus_fixed'][name])} |")
    lines.append("")
    return "\n".join(lines)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    default_root = (
        Path(__file__).resolve().parent / "results" /
        "live_extent_mixed_ab")
    parser = argparse.ArgumentParser(
        description=(
            "Strictly pair fixed/live-extent mixed-update reports and emit "
            "JSON plus Markdown summaries."))
    parser.add_argument(
        "root", type=Path, nargs="?", default=default_root,
        help=f"paired result root (default: {default_root})")
    parser.add_argument(
        "--contract", choices=tuple(CONTRACTS), default="rate-limited",
        help="registered workload contract (default: rate-limited)")
    parser.add_argument(
        "--pattern", default=REPORT_PATTERN,
        help=f"report filename glob (default: {REPORT_PATTERN})")
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--markdown-output", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    contract = CONTRACTS[args.contract]
    try:
        pairs = discover_pairs(args.root, contract, args.pattern)
        summary = build_summary(args.root, pairs, contract)
        json_path = args.json_output or (
            args.root / "live_extent_mixed_ab_summary.json")
        markdown_path = args.markdown_output or (
            args.root / "live_extent_mixed_ab_summary.md")
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
