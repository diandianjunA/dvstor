#!/usr/bin/env python3
import csv
import json
import math
import os
import sys
from pathlib import Path


if len(sys.argv) not in (2, 3):
    raise SystemExit("usage: summarize_ablation.py <run-root> [reference-report]")

root = Path(sys.argv[1]).resolve()
reference_path = Path(sys.argv[2]).resolve() if len(sys.argv) == 3 else None
manifest_path = root / "manifest.tsv"
if not manifest_path.is_file():
    raise SystemExit(f"missing manifest: {manifest_path}")

expected_order = ["baseline", "program1", "program3", "full"]
expected_modes = {
    "baseline": ("coupled", "fixed", "manual"),
    "program1": ("decoupled", "fixed", "manual"),
    "program3": ("decoupled", "fixed", "decoupled"),
    "full": ("decoupled", "adaptive", "decoupled"),
}
display_names = {
    "baseline": "Baseline",
    "program1": "+ 方案一",
    "program3": "+ 方案三",
    "full": "+ 方案二（Full）",
}


def finite_number(value, default=0.0):
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def latency_ms(report, section, percentile):
    value = report.get(section, {}).get("latency", {}).get(
        f"{percentile}_end_to_end_ns", 0)
    return finite_number(value) / 1e6


def recall_value(report, key):
    return finite_number(report.get(key, {}).get("recall", 0.0))


with manifest_path.open(encoding="utf-8", newline="") as stream:
    manifest_rows = list(csv.DictReader(stream, delimiter="\t"))
manifest_by_case = {row["case"]: row for row in manifest_rows}
missing = [name for name in expected_order if name not in manifest_by_case]
if missing:
    raise SystemExit("cannot summarize; missing cases: " + ", ".join(missing))

rows = []
warnings = []
common_contract = None
for case_name in expected_order:
    manifest = manifest_by_case[case_name]
    report_path = Path(manifest["report"])
    if not report_path.is_file():
        raise SystemExit(f"missing report for {case_name}: {report_path}")
    with report_path.open(encoding="utf-8") as stream:
        report = json.load(stream)

    meta = report.get("meta", {})
    system = meta.get("system_variant", {})
    modes = system.get("resolved_modes", {})
    actual_modes = (
        modes.get("storage_owner_update_completion_mode"),
        modes.get("gpu_dynamic_graph_access_mode"),
        modes.get("gpu_rdma_search_progression_mode"),
    )
    if actual_modes != expected_modes[case_name]:
        raise SystemExit(
            f"{case_name}: modes {actual_modes!r}, expected "
            f"{expected_modes[case_name]!r}")
    if actual_modes[2] == "manual":
        if (int(meta.get("gpu_graph_commit_width", 0)) != 16 or
                int(meta.get("gpu_graph_issue_width", 0)) != 16 or
                meta.get("gpu_query_beam_merge_policy") != "stable-run" or
                bool(meta.get("gpu_exact_frontier_early_issue", False))):
            raise SystemExit(
                f"{case_name}: invalid persistent-GPU late-issue baseline")

    index = system.get("index", {})
    concurrency = meta.get("benchmark_driver_concurrency", {})
    contract = {
        "index_prefix": index.get("prefix"),
        "schema_version": index.get("schema_version"),
        "build_fingerprint": index.get("build_fingerprint"),
        "warmup_seconds": meta.get("warmup_seconds"),
        "measure_seconds": meta.get("measure_seconds"),
        "read_ratio": meta.get("read_ratio"),
        "mixed_dispatch_policy": meta.get("mixed_dispatch_policy"),
        "client_threads": meta.get("client_threads"),
        "write_mix": (
            meta.get("write_insert_ratio"),
            meta.get("write_upsert_ratio"),
            meta.get("write_delete_ratio"),
        ),
        "query_slots": concurrency.get("gpu_query_slot_capacity"),
        "storage_rpc_inflight": concurrency.get(
            "storage_rpc_inflight_capacity"),
    }
    if common_contract is None:
        common_contract = contract
    elif contract != common_contract:
        differing = [key for key in contract if contract[key] != common_contract[key]]
        raise SystemExit(
            f"{case_name}: fairness contract differs in: " + ", ".join(differing))

    throughput = report.get("throughput", {})
    stage2 = report.get("stage2", {})
    row = {
        "case": case_name,
        "display_name": display_names[case_name],
        "code": manifest["code"],
        "profile": system.get("profile_name", manifest["profile"]),
        "update_mode": actual_modes[0],
        "access_mode": actual_modes[1],
        "progression_mode": actual_modes[2],
        "total_qps": finite_number(throughput.get("total_ops_per_sec")),
        "query_qps": finite_number(throughput.get("query_ops_per_sec")),
        "write_qps": finite_number(throughput.get("write_ops_per_sec")),
        "durable_total_qps": finite_number(
            throughput.get("durable_total_ops_per_sec")),
        "durable_write_qps": finite_number(
            throughput.get("durable_write_ops_per_sec")),
        "client_drain_seconds": finite_number(
            throughput.get("client_drain_seconds")),
        "maintenance_drain_seconds": finite_number(
            throughput.get("maintenance_drain_seconds")),
        "query_p99_ms": latency_ms(report, "query_breakdown", "p99"),
        "write_p99_ms": latency_ms(report, "insert_breakdown", "p99"),
        "recall_before": recall_value(report, "recall"),
        "recall_after": recall_value(report, "static_gt_post_recall"),
        "stage2_remaining": int(stage2.get("remaining", 0) or 0),
        "stage2_failures": int(stage2.get("failures", 0) or 0),
        "report": str(report_path),
    }
    if case_name != "baseline" and row["stage2_failures"]:
        warnings.append(
            f"{display_names[case_name]} reported "
            f"{row['stage2_failures']} hard Stage2 failures.")
    if row["recall_after"] == 0:
        warnings.append(f"{display_names[case_name]} has no post-run recall value.")
    rows.append(row)

baseline = rows[0]
for row in rows:
    row["total_speedup_vs_baseline"] = (
        row["total_qps"] / baseline["total_qps"]
        if baseline["total_qps"] else 0.0)
    row["query_speedup_vs_baseline"] = (
        row["query_qps"] / baseline["query_qps"]
        if baseline["query_qps"] else 0.0)
    row["write_speedup_vs_baseline"] = (
        row["write_qps"] / baseline["write_qps"]
        if baseline["write_qps"] else 0.0)

reference = None
if reference_path is not None and reference_path.is_file():
    with reference_path.open(encoding="utf-8") as stream:
        reference_report = json.load(stream)
    ref_tp = reference_report.get("throughput", {})
    reference = {
        "path": str(reference_path),
        "total_qps": finite_number(ref_tp.get("total_ops_per_sec")),
        "query_qps": finite_number(ref_tp.get("query_ops_per_sec")),
        "write_qps": finite_number(ref_tp.get("write_ops_per_sec")),
        "durable_total_qps": finite_number(
            ref_tp.get("durable_total_ops_per_sec")),
        "durable_write_qps": finite_number(
            ref_tp.get("durable_write_ops_per_sec")),
    }
    full = rows[-1]
    for metric in ("total_qps", "query_qps", "write_qps"):
        wanted = reference[metric]
        reference[f"full_{metric}_relative_difference"] = (
            (full[metric] - wanted) / wanted if wanted else 0.0)

summary = {
    "run_root": str(root),
    "fairness_contract": common_contract,
    "rows": rows,
    "reference": reference,
    "warnings": warnings,
}
with (root / "summary.json").open("w", encoding="utf-8") as stream:
    json.dump(summary, stream, ensure_ascii=False, indent=2)
    stream.write("\n")

csv_fields = [
    "case", "display_name", "code", "update_mode", "access_mode",
    "progression_mode", "total_qps", "query_qps", "write_qps",
    "durable_total_qps", "durable_write_qps", "query_p99_ms",
    "write_p99_ms", "recall_before", "recall_after",
    "client_drain_seconds", "maintenance_drain_seconds",
    "stage2_remaining", "stage2_failures", "total_speedup_vs_baseline",
    "query_speedup_vs_baseline", "write_speedup_vs_baseline", "report",
]
with (root / "summary.csv").open("w", encoding="utf-8", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=csv_fields, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)


def fmt(value, digits=1):
    return f"{value:,.{digits}f}"


lines = [
    "# 三项方案累积消融实验报告",
    "",
    "## 实验设计",
    "",
    "本实验从三个机制均关闭的 GPU-centric Baseline 出发，依次开启方案一、方案三和方案二。"
    "每个配置只运行一次；四组实验均从同一份静态索引重启，使用相同数据、硬件资源、"
    "查询参数、Persistent GPU/GPUNetIO 查询底座、客户端并发和 50% 读线程/50% 写线程"
    "的闭环混合负载。写操作均为 fresh "
    "insert，因此严格 coupled Baseline 与完整系统具有相同的有效操作语义。",
    "",
    "| 配置 | 编码 | 更新完成 | 动态邻接读取 | 查询推进 |",
    "|---|---:|---|---|---|",
]
for row in rows:
    lines.append(
        f"| {row['display_name']} | {row['code']} | {row['update_mode']} | "
        f"{row['access_mode']} | {row['progression_mode']} |")

lines += [
    "",
    "编码 `100 → 101 → 111` 表示在上一配置上依次只增加方案一、方案三和方案二。"
    "方案三关闭时仍使用 Persistent GPU + GPUNetIO，只关闭 early/ahead-of-commit "
    "progression，避免把 CPU→GPU 架构迁移错误计入方案三。`111` 直接使用正式 "
    "`04_gpu_persistent_gpunetio` profile，因此它就是当前全功能系统，而不是另行调参的配置。",
    "",
    "## 性能结果",
    "",
    "| 配置 | Query QPS | Insert QPS | Total QPS | Total/基线 | Query p99 (ms) | Insert p99 (ms) |",
    "|---|---:|---:|---:|---:|---:|---:|",
]
for row in rows:
    lines.append(
        f"| {row['display_name']} | {fmt(row['query_qps'])} | "
        f"{fmt(row['write_qps'])} | {fmt(row['total_qps'])} | "
        f"{row['total_speedup_vs_baseline']:.3f}× | "
        f"{fmt(row['query_p99_ms'], 3)} | {fmt(row['write_p99_ms'], 3)} |")

lines += [
    "",
    "图见 [ablation_performance.svg](ablation_performance.svg)。方案一带来的增量主要应"
    "体现在插入完成吞吐和前台写延迟；方案三在相同 GPU-centric 查询引擎内把下一轮"
    "RDMA 从完整 Merge 之后提前到 exact prefix 就绪之后；最后开启方案二，减少动态"
    "邻接表的 RDMA 读取字节。由于这是累积消融，每一行相对上一行的变化才是对应方案"
    "的边际收益。",
    "",
    "## 完整性与质量",
    "",
    "| 配置 | 测前 Recall@10 | 测后 Recall@10 | Stage2 drain (s) | Durable Insert QPS | Stage2 failures |",
    "|---|---:|---:|---:|---:|---:|",
]
for row in rows:
    lines.append(
        f"| {row['display_name']} | {row['recall_before']:.4f} | "
        f"{row['recall_after']:.4f} | {fmt(row['maintenance_drain_seconds'], 3)} | "
        f"{fmt(row['durable_write_qps'])} | {row['stage2_failures']} |")

lines += [
    "",
    "前台 Insert QPS 衡量 API 完成性能；对启用两阶段更新的配置，还必须结合 Stage2 "
    "drain 与 Durable Insert QPS 判断后台维护是否真正完成。测后 Recall@10 用于排除"
    "以索引质量换吞吐的情况。",
]

if reference is not None:
    full = rows[-1]
    lines += [
        "",
        "## 与正式主实验核对",
        "",
        f"参考报告：`{reference['path']}`。两者使用相同的 full profile 和负载参数。",
        "",
        "| 指标 | 本次 Full | 正式报告 | 相对差异 |",
        "|---|---:|---:|---:|",
    ]
    for label, metric in (
            ("Query QPS", "query_qps"),
            ("Insert QPS", "write_qps"),
            ("Total QPS", "total_qps")):
        delta = reference[f"full_{metric}_relative_difference"]
        lines.append(
            f"| {label} | {fmt(full[metric])} | {fmt(reference[metric])} | "
            f"{delta:+.1%} |")
    lines += [
        "",
        "短时运行波动、GPU/CPU 占用和 Stage2 drain 会造成一定差异。若 Full 的 Query "
        "或 Insert QPS 与参考值偏差明显，应先检查存储节点是否全部重启、GPU 是否被其他"
        "任务占用，以及本次报告中的实际并发和三个 resolved modes，而不要直接把偏差"
        "解释为消融收益。",
    ]

lines += [
    "",
    "## 公平性校验",
    "",
    f"- 索引：`{common_contract['index_prefix']}`，schema "
    f"{common_contract['schema_version']}，fingerprint "
    f"{common_contract['build_fingerprint']}。",
    f"- 负载：warmup {common_contract['warmup_seconds']} s，measure "
    f"{common_contract['measure_seconds']} s，read ratio "
    f"{common_contract['read_ratio']}，策略 "
    f"`{common_contract['mixed_dispatch_policy']}`。",
    f"- 并发：{common_contract['client_threads']} 个客户端线程；GPU query slots="
    f"{common_contract['query_slots']}，storage RPC inflight="
    f"{common_contract['storage_rpc_inflight']}。",
    "- 汇总脚本已逐项验证四份 JSON 的模式组合和上述公共实验契约；任一项不一致会"
    "直接拒绝生成报告。",
]
if warnings:
    lines += ["", "## 需要注意", ""]
    lines.extend(f"- {warning}" for warning in warnings)

with (root / "消融实验分析报告.md").open("w", encoding="utf-8") as stream:
    stream.write("\n".join(lines) + "\n")

print(f"summary: {root / 'summary.csv'}")
print(f"report: {root / '消融实验分析报告.md'}")
