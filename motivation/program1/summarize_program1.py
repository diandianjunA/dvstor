#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path


def get(root, *keys, default=0):
    value = root
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value


def load_manifest(root: Path):
    with (root / "manifest.tsv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    reports = {}
    for row in rows:
        path = Path(row["report"])
        with path.open(encoding="utf-8") as stream:
            reports[row["case"]] = json.load(stream)
    return reports


def main():
    if len(sys.argv) != 2:
        raise SystemExit("usage: summarize_program1.py <run-root>")
    root = Path(sys.argv[1]).resolve()
    reports = load_manifest(root)
    baseline = reports.get("baseline", {})
    solution = reports.get("solution", {})
    quality = reports.get("quality", {})

    base_qps = float(get(baseline, "throughput", "insert_ops_per_sec"))
    solution_qps = float(get(solution, "throughput", "insert_ops_per_sec"))
    critical = baseline.get("coupled_insert_critical_path", {})
    stage2 = solution.get("stage2", {})
    final_edges = float(stage2.get("stage2_final_edges", 0))
    cross_before = float(stage2.get("stage2_cross_edges_stage1_home", 0))
    cross_after = float(stage2.get("stage2_cross_edges_final_home", 0))
    stage1_hit = float(get(quality, "stage1_only_self_recall", "hit_rate"))
    final_hit = float(get(quality, "finalized_self_recall", "hit_rate"))

    summary = {
        "rdma_remote_dependency_ratio": float(critical.get("remote_dependency_ratio", 0)),
        "rdma_avg_remote_dependency_us": float(critical.get("avg_remote_dependency_us", 0)),
        "deferred_stage2_ratio": float(critical.get("deferred_stage2_ratio", 0)),
        "coupled_stack": critical.get("stack", {}),
        "baseline_insert_qps": base_qps,
        "solution_insert_qps": solution_qps,
        "insert_speedup": solution_qps / base_qps if base_qps else 0,
        "stage1_only_self_hit_rate": stage1_hit,
        "finalized_self_hit_rate": final_hit,
        "temporary_self_hit_drop": final_hit - stage1_hit,
        "stage1_final_result_overlap_at_k": float(
            get(quality, "self_recall_delta", "stage1_final_result_overlap_at_k")
        ),
        "stage1_window_valid": bool(get(quality, "stage1_only_window", "valid", default=False)),
        "avg_stage2_delay_ms": float(stage2.get("avg_stage2_delay_ms", 0)),
        "p99_stage2_delay_upper_ms": float(stage2.get("p99_stage2_delay_upper_ms", 0)),
        "cross_edge_ratio_stage1_home": cross_before / final_edges if final_edges else 0,
        "cross_edge_ratio_final_home": cross_after / final_edges if final_edges else 0,
        "cross_edge_reduction_ratio": float(stage2.get("cross_edge_reduction_ratio", 0)),
        "stage2_failures": int(stage2.get("failures", 0)),
        "stage2_telemetry_valid": bool(
            stage2.get("stage2_finalized_live_delta", 0)
            and stage2.get("stage2_final_edges", 0)
            and stage2.get("p99_stage2_delay_samples", 0)
        ),
    }
    (root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    with (root / "summary.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["metric", "value"])
        writer.writerows(summary.items())
    print(json.dumps(summary, indent=2))
    print(f"wrote {root / 'summary.json'} and summary.csv")


if __name__ == "__main__":
    main()
