#!/usr/bin/env python3
"""Analyze the dedicated live-extent GPUNetIO transport probe.

This analyzer deliberately does *not* project the microbenchmark into query
QPS.  It answers two narrower transport questions:

1. With the same number of one-sided READ WQEs, does a 400/448-byte one-shot
   transfer differ from the fixed 832-byte record transfer?
2. What WQE and batch-latency penalty is paid when 400/448 bytes require a
   dependent 16-byte header READ first?

Repeated cases are paired by repeat number before ratios are calculated, so
the forward/reverse sweep order controls remain paired.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import pathlib
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Iterable


REQUIRED_COLUMNS = (
    "repeat",
    "order",
    "stage1_B",
    "stage2_B",
    "payload_B",
    "stages",
    "active_QPs",
    "batch_reads",
    "working_set_B",
    "measured_batches_per_QP",
    "read_WQEs",
    "dump_WQEs",
    "transport_WQEs",
    "CQEs",
    "elapsed_ms",
    "read_WQE_per_s",
    "application_payload_GB_per_s",
    "batch_latency_mean_us",
    "batch_latency_p50_us",
    "batch_latency_p95_us",
    "batch_latency_p99_us",
)

INTEGER_COLUMNS = (
    "repeat",
    "stage1_B",
    "stage2_B",
    "payload_B",
    "stages",
    "active_QPs",
    "batch_reads",
    "working_set_B",
    "measured_batches_per_QP",
    "read_WQEs",
    "dump_WQEs",
    "transport_WQEs",
    "CQEs",
)

FLOAT_COLUMNS = (
    "elapsed_ms",
    "read_WQE_per_s",
    "application_payload_GB_per_s",
    "batch_latency_mean_us",
    "batch_latency_p50_us",
    "batch_latency_p95_us",
    "batch_latency_p99_us",
)

GROUP_METRICS = (
    "elapsed_ms",
    "read_WQEs",
    "dump_WQEs",
    "transport_WQEs",
    "CQEs",
    "read_WQE_per_s",
    "transport_WQE_per_s",
    "logical_batches_per_s",
    "application_payload_GB_per_s",
    "batch_latency_mean_us",
    "batch_latency_p50_us",
    "batch_latency_p95_us",
    "batch_latency_p99_us",
)

ONE_SHOT_COMPARISON_METRICS = (
    "read_WQE_per_s",
    "application_payload_GB_per_s",
    "batch_latency_p50_us",
    "batch_latency_p99_us",
)

TWO_STAGE_COMPARISON_METRICS = (
    "read_WQEs",
    "transport_WQEs",
    "read_WQE_per_s",
    "logical_batches_per_s",
    "application_payload_GB_per_s",
    "batch_latency_p50_us",
    "batch_latency_p99_us",
)


class ProbeAnalysisError(ValueError):
    """The CSV is malformed or lacks a required comparable case."""


@dataclass(frozen=True, order=True)
class GroupKey:
    active_qps: int
    stages: int
    stage1_bytes: int
    stage2_bytes: int
    payload_bytes: int


@dataclass(frozen=True)
class ProbeRow:
    source_line: int
    repeat: int
    order: str
    stage1_bytes: int
    stage2_bytes: int
    payload_bytes: int
    stages: int
    active_qps: int
    batch_reads: int
    working_set_bytes: int
    measured_batches_per_qp: int
    read_wqes: int
    dump_wqes: int
    transport_wqes: int
    cqes: int
    elapsed_ms: float
    read_wqe_per_s: float
    application_payload_gb_per_s: float
    batch_latency_mean_us: float
    batch_latency_p50_us: float
    batch_latency_p95_us: float
    batch_latency_p99_us: float

    @property
    def key(self) -> GroupKey:
        return GroupKey(
            self.active_qps,
            self.stages,
            self.stage1_bytes,
            self.stage2_bytes,
            self.payload_bytes,
        )

    def metric(self, name: str) -> float:
        if name == "read_WQEs":
            return float(self.read_wqes)
        if name == "dump_WQEs":
            return float(self.dump_wqes)
        if name == "transport_WQEs":
            return float(self.transport_wqes)
        if name == "CQEs":
            return float(self.cqes)
        if name == "elapsed_ms":
            return self.elapsed_ms
        if name == "read_WQE_per_s":
            return self.read_wqe_per_s
        if name == "transport_WQE_per_s":
            return self.transport_wqes / (self.elapsed_ms / 1000.0)
        if name == "logical_batches_per_s":
            logical_batches = self.active_qps * self.measured_batches_per_qp
            return logical_batches / (self.elapsed_ms / 1000.0)
        if name == "application_payload_GB_per_s":
            return self.application_payload_gb_per_s
        if name == "batch_latency_mean_us":
            return self.batch_latency_mean_us
        if name == "batch_latency_p50_us":
            return self.batch_latency_p50_us
        if name == "batch_latency_p95_us":
            return self.batch_latency_p95_us
        if name == "batch_latency_p99_us":
            return self.batch_latency_p99_us
        raise KeyError(name)


def _parse_int(value: str | None, field: str, path: pathlib.Path,
               line_number: int) -> int:
    try:
        if value is None or value.strip() == "":
            raise ValueError
        return int(value)
    except ValueError as error:
        raise ProbeAnalysisError(
            f"{path}:{line_number}: {field} is not an integer") from error


def _parse_float(value: str | None, field: str, path: pathlib.Path,
                 line_number: int) -> float:
    try:
        if value is None or value.strip() == "":
            raise ValueError
        result = float(value)
    except ValueError as error:
        raise ProbeAnalysisError(
            f"{path}:{line_number}: {field} is not numeric") from error
    if not math.isfinite(result):
        raise ProbeAnalysisError(
            f"{path}:{line_number}: {field} is not finite")
    return result


def _validate_row(row: ProbeRow, path: pathlib.Path) -> None:
    prefix = f"{path}:{row.source_line}"
    if row.repeat <= 0:
        raise ProbeAnalysisError(f"{prefix}: repeat must be positive")
    if row.order not in ("forward", "reverse"):
        raise ProbeAnalysisError(
            f"{prefix}: order must be forward or reverse")
    if row.stage1_bytes <= 0 or row.stage2_bytes < 0:
        raise ProbeAnalysisError(f"{prefix}: invalid stage byte count")
    expected_stages = 1 if row.stage2_bytes == 0 else 2
    if row.stages != expected_stages:
        raise ProbeAnalysisError(
            f"{prefix}: stages={row.stages}, expected {expected_stages}")
    if row.payload_bytes != row.stage1_bytes + row.stage2_bytes:
        raise ProbeAnalysisError(
            f"{prefix}: payload_B does not equal stage1_B + stage2_B")
    if (
        row.active_qps <= 0
        or row.batch_reads <= 0
        or row.working_set_bytes <= 0
        or row.measured_batches_per_qp <= 0
    ):
        raise ProbeAnalysisError(
            f"{prefix}: QP, batch, working-set, and iteration counts "
            "must be positive")
    if row.dump_wqes < 0:
        raise ProbeAnalysisError(f"{prefix}: dump_WQEs is negative")
    expected_reads = (
        row.active_qps
        * row.measured_batches_per_qp
        * row.batch_reads
        * row.stages
    )
    if row.read_wqes != expected_reads:
        raise ProbeAnalysisError(
            f"{prefix}: read_WQEs={row.read_wqes}, expected "
            f"{expected_reads}")
    if row.transport_wqes != row.read_wqes + row.dump_wqes:
        raise ProbeAnalysisError(
            f"{prefix}: transport_WQEs does not equal READ + dump WQEs")
    expected_cqes = (
        row.active_qps * row.measured_batches_per_qp * row.stages)
    if row.cqes != expected_cqes:
        raise ProbeAnalysisError(
            f"{prefix}: CQEs={row.cqes}, expected {expected_cqes}")
    positive_metrics = (
        row.elapsed_ms,
        row.read_wqe_per_s,
        row.application_payload_gb_per_s,
        row.batch_latency_mean_us,
        row.batch_latency_p50_us,
        row.batch_latency_p95_us,
        row.batch_latency_p99_us,
    )
    if any(value <= 0.0 for value in positive_metrics):
        raise ProbeAnalysisError(
            f"{prefix}: elapsed, rates, and latencies must be positive")
    if not (
        row.batch_latency_p50_us
        <= row.batch_latency_p95_us
        <= row.batch_latency_p99_us
    ):
        raise ProbeAnalysisError(
            f"{prefix}: batch latency percentiles are not monotone")


def load_csv(path: pathlib.Path) -> list[ProbeRow]:
    try:
        source = path.open("r", encoding="utf-8", newline="")
    except OSError as error:
        raise ProbeAnalysisError(f"{path}: cannot open CSV: {error}") from error
    with source:
        reader = csv.DictReader(source)
        if reader.fieldnames is None:
            raise ProbeAnalysisError(f"{path}: CSV has no header")
        missing = [
            column for column in REQUIRED_COLUMNS
            if column not in reader.fieldnames
        ]
        if missing:
            raise ProbeAnalysisError(
                f"{path}: missing CSV columns: {', '.join(missing)}")
        rows: list[ProbeRow] = []
        for line_number, fields in enumerate(reader, 2):
            integers = {
                field: _parse_int(fields.get(field), field, path, line_number)
                for field in INTEGER_COLUMNS
            }
            floats = {
                field: _parse_float(
                    fields.get(field), field, path, line_number)
                for field in FLOAT_COLUMNS
            }
            row = ProbeRow(
                source_line=line_number,
                repeat=integers["repeat"],
                order=(fields.get("order") or "").strip(),
                stage1_bytes=integers["stage1_B"],
                stage2_bytes=integers["stage2_B"],
                payload_bytes=integers["payload_B"],
                stages=integers["stages"],
                active_qps=integers["active_QPs"],
                batch_reads=integers["batch_reads"],
                working_set_bytes=integers["working_set_B"],
                measured_batches_per_qp=integers[
                    "measured_batches_per_QP"],
                read_wqes=integers["read_WQEs"],
                dump_wqes=integers["dump_WQEs"],
                transport_wqes=integers["transport_WQEs"],
                cqes=integers["CQEs"],
                elapsed_ms=floats["elapsed_ms"],
                read_wqe_per_s=floats["read_WQE_per_s"],
                application_payload_gb_per_s=floats[
                    "application_payload_GB_per_s"],
                batch_latency_mean_us=floats["batch_latency_mean_us"],
                batch_latency_p50_us=floats["batch_latency_p50_us"],
                batch_latency_p95_us=floats["batch_latency_p95_us"],
                batch_latency_p99_us=floats["batch_latency_p99_us"],
            )
            _validate_row(row, path)
            rows.append(row)
    if not rows:
        raise ProbeAnalysisError(f"{path}: CSV has no data rows")
    return rows


def summarize(values: Iterable[float]) -> dict[str, float | int | None]:
    samples = [float(value) for value in values]
    if not samples:
        raise ProbeAnalysisError("cannot summarize an empty sample")
    if any(not math.isfinite(value) for value in samples):
        raise ProbeAnalysisError("cannot summarize non-finite samples")
    ordered = sorted(samples)
    if len(ordered) == 1:
        q1 = q3 = ordered[0]
        sample_stdev = 0.0
    else:
        q1, _, q3 = statistics.quantiles(
            ordered, n=4, method="inclusive")
        sample_stdev = statistics.stdev(ordered)
    sample_mean = statistics.mean(ordered)
    return {
        "count": len(ordered),
        "median": statistics.median(ordered),
        "q1": q1,
        "q3": q3,
        "iqr": q3 - q1,
        "mean": sample_mean,
        "sample_stdev": sample_stdev,
        "cv": (
            sample_stdev / abs(sample_mean)
            if sample_mean != 0.0 else None
        ),
        "minimum": ordered[0],
        "maximum": ordered[-1],
    }


def _group_rows(
    rows: Iterable[ProbeRow], expected_repeats: int | None,
) -> dict[GroupKey, list[ProbeRow]]:
    grouped: dict[GroupKey, list[ProbeRow]] = defaultdict(list)
    seen: set[tuple[GroupKey, int]] = set()
    for row in rows:
        identity = (row.key, row.repeat)
        if identity in seen:
            raise ProbeAnalysisError(
                "duplicate row for "
                f"QP={row.active_qps}, stages={row.stages}, "
                f"{row.stage1_bytes}+{row.stage2_bytes}B, "
                f"repeat={row.repeat}")
        seen.add(identity)
        grouped[row.key].append(row)

    expected_repeat_ids = (
        set(range(1, expected_repeats + 1))
        if expected_repeats is not None else None
    )
    for key, group in grouped.items():
        repeat_ids = {row.repeat for row in group}
        if expected_repeat_ids is not None and repeat_ids != expected_repeat_ids:
            raise ProbeAnalysisError(
                f"QP={key.active_qps}, stages={key.stages}, "
                f"{key.stage1_bytes}+{key.stage2_bytes}B has repeats "
                f"{sorted(repeat_ids)}, expected "
                f"{sorted(expected_repeat_ids)}")
        invariants = {
            (
                row.batch_reads,
                row.working_set_bytes,
                row.measured_batches_per_qp,
            )
            for row in group
        }
        if len(invariants) != 1:
            raise ProbeAnalysisError(
                f"QP={key.active_qps}, stages={key.stages}, "
                f"{key.stage1_bytes}+{key.stage2_bytes}B changes batch, "
                "working-set, or measured-iteration configuration")
        group.sort(key=lambda row: row.repeat)
    return dict(grouped)


def _group_record(key: GroupKey,
                  rows: list[ProbeRow]) -> dict[str, Any]:
    exemplar = rows[0]
    return {
        "active_QPs": key.active_qps,
        "stages": key.stages,
        "stage1_B": key.stage1_bytes,
        "stage2_B": key.stage2_bytes,
        "payload_B": key.payload_bytes,
        "pattern": (
            f"one-shot {key.stage1_bytes}B"
            if key.stages == 1
            else f"dependent {key.stage1_bytes}+{key.stage2_bytes}B"
        ),
        "repeat_count": len(rows),
        "repeats": [row.repeat for row in rows],
        "orders": [row.order for row in rows],
        "batch_reads": exemplar.batch_reads,
        "working_set_B": exemplar.working_set_bytes,
        "measured_batches_per_QP": exemplar.measured_batches_per_qp,
        "metrics": {
            metric: summarize(row.metric(metric) for row in rows)
            for metric in GROUP_METRICS
        },
    }


def _assert_comparable(
    candidate: list[ProbeRow], reference: list[ProbeRow],
    description: str,
) -> list[tuple[ProbeRow, ProbeRow]]:
    candidate_by_repeat = {row.repeat: row for row in candidate}
    reference_by_repeat = {row.repeat: row for row in reference}
    if candidate_by_repeat.keys() != reference_by_repeat.keys():
        raise ProbeAnalysisError(
            f"{description}: repeat sets do not match")
    pairs = [
        (candidate_by_repeat[repeat], reference_by_repeat[repeat])
        for repeat in sorted(candidate_by_repeat)
    ]
    for candidate_row, reference_row in pairs:
        if candidate_row.order != reference_row.order:
            raise ProbeAnalysisError(
                f"{description}: repeat={candidate_row.repeat} has "
                "different sweep order")
        if (
            candidate_row.active_qps != reference_row.active_qps
            or candidate_row.batch_reads != reference_row.batch_reads
            or candidate_row.working_set_bytes
                != reference_row.working_set_bytes
            or candidate_row.measured_batches_per_qp
                != reference_row.measured_batches_per_qp
        ):
            raise ProbeAnalysisError(
                f"{description}: transport configuration differs in "
                f"repeat={candidate_row.repeat}")
    return pairs


def _paired_metric(
    pairs: list[tuple[ProbeRow, ProbeRow]], metric: str,
) -> dict[str, Any]:
    candidate = [left.metric(metric) for left, _ in pairs]
    reference = [right.metric(metric) for _, right in pairs]
    if any(value == 0.0 for value in reference):
        raise ProbeAnalysisError(
            f"cannot form paired ratio for zero-valued {metric}")
    ratios = [
        left / right for left, right in zip(candidate, reference)]
    deltas = [
        left - right for left, right in zip(candidate, reference)]
    changes = [ratio - 1.0 for ratio in ratios]
    return {
        "candidate": summarize(candidate),
        "reference": summarize(reference),
        "paired_ratio": summarize(ratios),
        "paired_delta": summarize(deltas),
        "paired_change_fraction": summarize(changes),
    }


def _find_group(
    grouped: dict[GroupKey, list[ProbeRow]], key: GroupKey,
    description: str,
) -> list[ProbeRow]:
    try:
        return grouped[key]
    except KeyError as error:
        raise ProbeAnalysisError(
            f"missing {description}: QP={key.active_qps}, stages={key.stages}, "
            f"stage1={key.stage1_bytes}B, stage2={key.stage2_bytes}B") from error


def _core_comparisons(
    grouped: dict[GroupKey, list[ProbeRow]],
    require_core_cases: bool,
) -> dict[str, list[dict[str, Any]]]:
    active_qps = sorted({key.active_qps for key in grouped})
    one_shot_vs_fixed: list[dict[str, Any]] = []
    two_stage_vs_one_shot: list[dict[str, Any]] = []

    for qps in active_qps:
        fixed_key = GroupKey(qps, 1, 832, 0, 832)
        fixed = grouped.get(fixed_key)
        if fixed is None:
            if require_core_cases:
                _find_group(grouped, fixed_key, "832B one-shot reference")
            continue
        for payload in (400, 448):
            candidate_key = GroupKey(qps, 1, payload, 0, payload)
            candidate = grouped.get(candidate_key)
            if candidate is None:
                if require_core_cases:
                    _find_group(
                        grouped, candidate_key,
                        f"{payload}B one-shot candidate")
                continue
            description = (
                f"QP={qps} one-shot {payload}B versus one-shot 832B")
            pairs = _assert_comparable(candidate, fixed, description)
            one_shot_vs_fixed.append({
                "active_QPs": qps,
                "candidate_payload_B": payload,
                "reference_payload_B": 832,
                "application_payload_reduction_fraction":
                    1.0 - payload / 832.0,
                "read_WQE_count_ratio": _paired_metric(
                    pairs, "read_WQEs"),
                "metrics": {
                    metric: _paired_metric(pairs, metric)
                    for metric in ONE_SHOT_COMPARISON_METRICS
                },
            })

        for body in (400, 448):
            dependent_key = GroupKey(qps, 2, 16, body, 16 + body)
            one_shot_key = GroupKey(qps, 1, body, 0, body)
            dependent = grouped.get(dependent_key)
            one_shot = grouped.get(one_shot_key)
            if dependent is None or one_shot is None:
                if require_core_cases:
                    _find_group(
                        grouped, dependent_key,
                        f"dependent 16+{body}B case")
                    _find_group(
                        grouped, one_shot_key,
                        f"{body}B one-shot reference")
                continue
            description = (
                f"QP={qps} dependent 16+{body}B versus one-shot {body}B")
            pairs = _assert_comparable(dependent, one_shot, description)
            two_stage_vs_one_shot.append({
                "active_QPs": qps,
                "header_B": 16,
                "body_B": body,
                "dependent_payload_B": 16 + body,
                "one_shot_payload_B": body,
                "payload_bytes_ratio": (16 + body) / body,
                "metrics": {
                    metric: _paired_metric(pairs, metric)
                    for metric in TWO_STAGE_COMPARISON_METRICS
                },
            })

    return {
        "one_shot_400_448_vs_832": one_shot_vs_fixed,
        "dependent_16_plus_body_vs_corresponding_one_shot":
            two_stage_vs_one_shot,
    }


def analyze(
    rows: Iterable[ProbeRow],
    expected_repeats: int | None = 3,
    require_core_cases: bool = True,
) -> dict[str, Any]:
    grouped = _group_rows(rows, expected_repeats)
    repeat_ids = sorted({row.repeat for group in grouped.values()
                         for row in group})
    orders_by_repeat: dict[int, set[str]] = defaultdict(set)
    for group in grouped.values():
        for row in group:
            orders_by_repeat[row.repeat].add(row.order)
    inconsistent_orders = {
        repeat: sorted(orders)
        for repeat, orders in orders_by_repeat.items()
        if len(orders) != 1
    }
    if inconsistent_orders:
        raise ProbeAnalysisError(
            f"sweep order is inconsistent within repeats: "
            f"{inconsistent_orders}")

    return {
        "scope": {
            "classification": (
                "transport-only GPU-initiated one-sided RDMA READ "
                "microbenchmark"),
            "measures": [
                "dedicated-probe READ WQE throughput",
                "requested application-payload throughput",
                "logical batch latency at one final CQE per stage",
                "dependent header-then-body READ penalty",
            ],
            "does_not_measure": [
                "end-to-end dvstor query QPS or latency",
                "Beam merge, graph decode, PQ scoring, or visited work",
                "persistent-query descriptor-ring and owner-warp contention",
                "storage-node CPU request processing",
                "NIC wire bytes or wire-rate throughput",
                "the cost or correctness of a changed storage index format",
            ],
            "interpretation": (
                "The result can establish a transport opportunity or rule it "
                "out. It cannot by itself establish end-to-end system speedup."
            ),
            "sweep_order_limit": (
                "Forward/reverse repeats counterbalance order within the "
                "one-shot list, but every dependent two-stage case runs after "
                "all one-shot cases. Dependent-versus-one-shot pairs may "
                "therefore retain temporal drift and are not a randomized "
                "causal A/B."
            ),
        },
        "statistics": {
            "expected_repeats": expected_repeats,
            "observed_repeat_ids": repeat_ids,
            "center": "median",
            "iqr": (
                "Q3-Q1 using inclusive quartiles "
                "(statistics.quantiles(method='inclusive'))"),
            "cv": "sample standard deviation divided by absolute mean",
            "comparison_ratios": (
                "paired by repeat before median/IQR/CV aggregation"),
        },
        "groups": [
            _group_record(key, grouped[key])
            for key in sorted(grouped)
        ],
        "comparisons": _core_comparisons(grouped, require_core_cases),
    }


def _format_number(value: float | None, digits: int = 3) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def _format_percent(value: float | None, digits: int = 1) -> str:
    return "n/a" if value is None else f"{100.0 * value:.{digits}f}%"


def _summary_cell(summary: dict[str, Any], digits: int = 3) -> str:
    return (
        f"{_format_number(summary['median'], digits)} / "
        f"{_format_number(summary['iqr'], digits)} / "
        f"{_format_percent(summary['cv'])}"
    )


def _ratio_cell(metric: dict[str, Any]) -> str:
    ratio = metric["paired_ratio"]
    return (
        f"{_format_number(ratio['median'], 3)}x "
        f"(IQR {_format_number(ratio['iqr'], 3)}, "
        f"CV {_format_percent(ratio['cv'])})"
    )


def _latency_penalty_cell(metric: dict[str, Any]) -> str:
    delta = metric["paired_delta"]
    return (
        f"{_format_number(delta['median'], 3)} us; "
        f"{_ratio_cell(metric)}"
    )


def markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Live-extent RDMA transport probe",
        "",
        "**Scope: transport-only.** This is a dedicated GPU-initiated, "
        "one-sided RDMA READ microbenchmark. It does not measure query QPS, "
        "query latency, Beam/decode/PQ/visited work, storage-side format cost, "
        "or NIC wire bandwidth. `application_payload_GB_per_s` is requested "
        "application bytes divided by probe wall time.",
        "",
        "All aggregate cells below are `median / IQR / CV` over repeats. "
        "IQR uses inclusive quartiles; CV uses sample standard deviation. "
        "Comparison ratios are paired by repeat before aggregation.",
        "",
        "## Repeated transport cases",
        "",
        "| active QPs | transfer | reps | READ WQE/s | requested-payload "
        "GB/s | "
        "batch P50 us | batch P99 us |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for group in report["groups"]:
        metrics = group["metrics"]
        lines.append(
            f"| {group['active_QPs']} | {group['pattern']} | "
            f"{group['repeat_count']} | "
            f"{_summary_cell(metrics['read_WQE_per_s'], 0)} | "
            f"{_summary_cell(metrics['application_payload_GB_per_s'])} | "
            f"{_summary_cell(metrics['batch_latency_p50_us'])} | "
            f"{_summary_cell(metrics['batch_latency_p99_us'])} |"
        )

    lines.extend([
        "",
        "## One-shot 400/448B versus fixed 832B",
        "",
        "Each paired case posts the same number of READ WQEs. A WQE/s ratio "
        "above 1 means the shorter payload completed more READs per second; "
        "latency ratios below 1 are better. Application GB/s is shown "
        "separately because its numerator changes with payload size.",
        "",
        "| active QPs | payload | payload reduction | READ WQE count | "
        "READ WQE/s ratio | batch P50 ratio | batch P99 ratio | "
        "application GB/s ratio |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in report["comparisons"]["one_shot_400_448_vs_832"]:
        metrics = row["metrics"]
        lines.append(
            f"| {row['active_QPs']} | {row['candidate_payload_B']}B | "
            f"{_format_percent(row['application_payload_reduction_fraction'])} "
            f"| {_ratio_cell(row['read_WQE_count_ratio'])} | "
            f"{_ratio_cell(metrics['read_WQE_per_s'])} | "
            f"{_ratio_cell(metrics['batch_latency_p50_us'])} | "
            f"{_ratio_cell(metrics['batch_latency_p99_us'])} | "
            f"{_ratio_cell(metrics['application_payload_GB_per_s'])} |"
        )

    lines.extend([
        "",
        "## Dependent 16B header + body versus corresponding one-shot body",
        "",
        "The dependent case waits for the 16B header stage before issuing the "
        "body stage. It carries 16 additional application bytes versus the "
        "listed one-shot reference, so this is deliberately reported as a "
        "measured transport penalty rather than a byte-identical comparison.",
        "All two-stage cases execute after the one-shot list even in reverse "
        "repeats; this comparison is paired but not fully order-counterbalanced "
        "and can retain temporal drift.",
        "",
        "| active QPs | dependent/reference | READ WQE count penalty | "
        "READ WQE/s ratio | logical batch/s ratio | batch P50 penalty | "
        "batch P99 penalty | application GB/s ratio |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ])
    for row in report["comparisons"][
            "dependent_16_plus_body_vs_corresponding_one_shot"]:
        metrics = row["metrics"]
        lines.append(
            f"| {row['active_QPs']} | 16+{row['body_B']}B / "
            f"{row['one_shot_payload_B']}B | "
            f"{_ratio_cell(metrics['read_WQEs'])} | "
            f"{_ratio_cell(metrics['read_WQE_per_s'])} | "
            f"{_ratio_cell(metrics['logical_batches_per_s'])} | "
            f"{_latency_penalty_cell(metrics['batch_latency_p50_us'])} | "
            f"{_latency_penalty_cell(metrics['batch_latency_p99_us'])} | "
            f"{_ratio_cell(metrics['application_payload_GB_per_s'])} |"
        )

    lines.extend([
        "",
        "## Interpretation boundary",
        "",
        "A shorter one-shot READ improving this table establishes only that "
        "fixed 832B records leave transport headroom in this isolated access "
        "pattern. A dependent two-stage penalty shows how much of that "
        "headroom is lost when the live length is unavailable before the "
        "first READ. Neither result proves that changing the index layout "
        "improves dvstor end to end; that requires a query-path A/B with "
        "identical Recall and graph-read semantics.",
        "",
    ])
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate the live-extent transport-only RDMA probe")
    parser.add_argument("csv", type=pathlib.Path)
    parser.add_argument("--output-json", type=pathlib.Path)
    parser.add_argument("--output-markdown", type=pathlib.Path)
    parser.add_argument(
        "--expected-repeats",
        type=int,
        default=3,
        help="require exactly repeats 1..N in every case (default: 3)")
    parser.add_argument(
        "--allow-incomplete-core",
        action="store_true",
        help="aggregate available rows without requiring all core cases")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.expected_repeats <= 0:
        raise ProbeAnalysisError("--expected-repeats must be positive")
    report = analyze(
        load_csv(args.csv),
        expected_repeats=args.expected_repeats,
        require_core_cases=not args.allow_incomplete_core,
    )
    rendered = markdown(report)
    print(rendered)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    if args.output_markdown:
        args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
        args.output_markdown.write_text(rendered, encoding="utf-8")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ProbeAnalysisError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2)
