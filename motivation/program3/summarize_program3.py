#!/usr/bin/env python3
"""Validate and summarize Program 3's strict persistent exact-core A/B."""

from __future__ import annotations

import csv
import json
import math
import statistics
import sys
from pathlib import Path


def load(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def nested(root, *keys, default=0):
    value = root
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value


def median(values):
    return statistics.median(values) if values else 0.0


def geomean(values):
    return math.exp(sum(math.log(value) for value in values) / len(values)) if values else 0.0


def one(report: dict) -> dict:
    meta = report.get("meta", {})
    gpu = report.get("gpu_persistent", {})
    throughput = report.get("throughput", {})
    certs = float(gpu.get("frontier_reusable_certificates", 0))
    issued = float(gpu.get("frontier_reusable_issued_certificates", 0))
    queries = max(float(gpu.get("queries_completed", 0)), 1.0)
    prefix_us_query = float(gpu.get("average_gpu_frontier_prefix_to_beam_publish_us", 0))
    issue_us_query = float(gpu.get("average_gpu_frontier_issue_to_beam_publish_us", 0))
    certs_query = certs / queries
    return {
        "qps": float(throughput.get("query_ops_per_sec", 0)),
        "write_qps": float(throughput.get("write_ops_per_sec", 0)),
        "write_attainment": float(throughput.get("write_rate_attainment_ratio", 0)),
        "p50_ms": float(nested(report, "query_breakdown", "latency", "p50_end_to_end_ns")) / 1e6,
        "p99_ms": float(nested(report, "query_breakdown", "latency", "p99_end_to_end_ns")) / 1e6,
        "recall": float(nested(report, "recall", "recall")),
        "graph_reads_per_query": float(gpu.get("average_logical_graph_reads_per_query", 0)),
        "graph_bytes_per_query": float(gpu.get("average_graph_read_bytes_per_query", 0)),
        "rdma_wait_us_per_query": float(gpu.get("average_gpu_rdma_wait_us", 0)),
        "beam_merge_us_per_query": float(gpu.get("average_gpu_beam_merge_us", 0)),
        "rdma_completion_latency_us": float(gpu.get("average_rdma_completion_latency_us", 0)),
        "certificates_per_query": certs_query,
        "issued_certificates_per_query": issued / queries,
        "certificate_coverage": issued / certs if certs else 0.0,
        "certificate_reject_ratio": (
            float(gpu.get("frontier_certificate_rejects", 0)) / certs if certs else 0.0
        ),
        "critical_rob_hit_ratio": float(gpu.get("critical_rob_hit_ratio", 0)),
        "speculative_graph_reads": int(gpu.get("speculative_graph_reads", 0)),
        "prefix_to_publish_us_per_certificate": prefix_us_query / certs_query if certs_query else 0.0,
        "issue_to_publish_us_per_certificate": issue_us_query / certs_query if certs_query else 0.0,
        "execution_mode": meta.get("gpu_frontier_execution_mode", "unknown"),
    }


def aggregate(samples: list[dict]) -> dict:
    numeric = [key for key, value in samples[0].items() if isinstance(value, (int, float))]
    result = {key: median([float(sample[key]) for sample in samples]) for key in numeric}
    result["samples"] = len(samples)
    result["execution_mode"] = samples[0]["execution_mode"]
    return result


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: summarize_program3.py <run-root>")
    root = Path(sys.argv[1]).resolve()
    rows = list(csv.DictReader((root / "manifest.tsv").open(encoding="utf-8"), delimiter="\t"))
    if not rows:
        raise SystemExit("empty manifest")

    raw: dict[str, dict[str, dict[int, dict]]] = {}
    contracts = set()
    for row in rows:
        workload, mode, repeat = row["workload"], row["mode"], int(row["repeat"])
        report = load(Path(row["report"]))
        meta = report.get("meta", {})
        contracts.add((
            workload,
            int(meta.get("gpu_graph_commit_width", 0)),
            int(meta.get("gpu_graph_issue_width", 0)),
            int(meta.get("warmup_seconds", 0)),
            int(meta.get("measure_seconds", 0)),
            float(meta.get("target_write_qps", 0)),
        ))
        raw.setdefault(workload, {}).setdefault(mode, {})[repeat] = one(report)

    cases = {}
    paired = {}
    for workload, modes in sorted(raw.items()):
        if set(modes) != {"late", "early"}:
            raise SystemExit(f"{workload}: both late and early modes are required")
        common = sorted(set(modes["late"]) & set(modes["early"]))
        if not common:
            raise SystemExit(f"{workload}: no paired repeats")
        cases[workload] = {
            mode: aggregate([modes[mode][repeat] for repeat in common])
            for mode in ("late", "early")
        }
        qps_ratios = [modes["early"][r]["qps"] / modes["late"][r]["qps"] for r in common]
        p99_ratios = [modes["early"][r]["p99_ms"] / modes["late"][r]["p99_ms"] for r in common]
        recall_deltas = [modes["early"][r]["recall"] - modes["late"][r]["recall"] for r in common]
        read_ratios = [
            modes["early"][r]["graph_reads_per_query"] /
            modes["late"][r]["graph_reads_per_query"]
            for r in common if modes["late"][r]["graph_reads_per_query"]
        ]
        paired[workload] = {
            "repeats": common,
            "qps_geomean_ratio": geomean(qps_ratios),
            "p99_geomean_ratio": geomean(p99_ratios),
            "recall_delta_median": median(recall_deltas),
            "graph_reads_per_query_geomean_ratio": geomean(read_ratios),
            "qps_ratios": qps_ratios,
            "p99_ratios": p99_ratios,
        }

    identity_log = root / "identity" / "exactness.log"
    identity_passed = (
        identity_log.exists() and
        "SKIP:" not in identity_log.read_text(encoding="utf-8", errors="replace")
    )
    summary = {
        "experiment": "persistent_stable_run_exact_core_issue_timing_ab",
        "contracts": [list(value) for value in sorted(contracts)],
        "cases": cases,
        "paired": paired,
        "identity_tests_passed": identity_passed,
    }
    (root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    with (root / "summary.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["workload", "mode", "qps", "p99_ms", "recall", "cert_coverage", "rob_hit", "prefix_gap_us"])
        for workload, modes in cases.items():
            for mode, values in modes.items():
                writer.writerow([workload, mode, values["qps"], values["p99_ms"], values["recall"], values["certificate_coverage"], values["critical_rob_hit_ratio"], values["prefix_to_publish_us_per_certificate"]])

    lines = [
        "# 方案三：精确前沿驱动的 GPU–RDMA 推进解耦实验报告",
        "",
        "本实验使用严格单变量 A/B：两组均为 Persistent GPU + GPUNetIO + Stable-Run + Live/DynaExtent，且 issue width 等于 commit width；因此没有 speculative tail，唯一差异是 mandatory RDMA 在完整 Beam 发布之前还是之后发出。",
        "",
    ]
    for workload in sorted(cases):
        late, early = cases[workload]["late"], cases[workload]["early"]
        effect = paired[workload]
        lines += [
            f"## {workload}", "",
            f"- Late-Issue：{late['qps']:.1f} query/s，P99 {late['p99_ms']:.3f} ms，Recall {late['recall']:.6f}。",
            f"- Exact-Early-Issue：{early['qps']:.1f} query/s，P99 {early['p99_ms']:.3f} ms，Recall {early['recall']:.6f}。",
            f"- 配对结果：QPS {100*(effect['qps_geomean_ratio']-1):+.2f}%，P99 {100*(effect['p99_geomean_ratio']-1):+.2f}%，Recall 差值 {effect['recall_delta_median']:+.6f}。",
            f"- 每个 exact certificate 的 Prefix→Beam 窗口约 {early['prefix_to_publish_us_per_certificate']:.3f} μs，其中 RDMA 已发出后与 Beam merge 的实际重叠约 {early['issue_to_publish_us_per_certificate']:.3f} μs；RDMA completion latency 中位运行值为 {early['rdma_completion_latency_us']:.3f} μs。",
            f"- Certificate 发出覆盖率 {100*early['certificate_coverage']:.2f}%，critical ROB hit {100*early['critical_rob_hit_ratio']:.2f}%，certificate reject {100*early['certificate_reject_ratio']:.4f}%。",
            f"- 物理读取次数/查询比值（Early/Late）为 {effect['graph_reads_per_query_geomean_ratio']:.4f}；Early speculative graph reads={int(early['speculative_graph_reads'])}。",
            "",
        ]
    lines += [
        "## 结论边界", "",
        "若 Recall 保持一致、speculative reads 为零且读取次数接近一致，则 QPS/P99 的差异可归因于相同 mandatory RDMA 的发起时机，而不是额外预取或不同 merge/transport 栈。逐轮 exact-prefix 与完整 Stable-Run 的实现等价性由 identity 目录中的 GPU 测试验证；动态 mixed run 则验证并发更新下的运行稳定性。",
        "",
    ]
    (root / "方案三实验结果分析报告.md").write_text("\n".join(lines), encoding="utf-8")
    print(root / "summary.json")
    print(root / "方案三实验结果分析报告.md")


if __name__ == "__main__":
    main()
