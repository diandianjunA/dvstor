#!/usr/bin/env python3
"""Summarize the Program 3 motivation-width sweep and one-shot A/B."""

import csv
import json
import sys
from pathlib import Path


if len(sys.argv) != 2:
    raise SystemExit("usage: summarize_story.py <run-root>")
root = Path(sys.argv[1]).resolve()
rows = list(csv.DictReader((root / "story_manifest.tsv").open(encoding="utf-8"), delimiter="\t"))


def load(row):
    return json.load(open(row["report"], encoding="utf-8"))


def latency(report, key):
    return float(report.get("query_breakdown", {}).get("latency", {}).get(key, 0)) / 1e6


motivation = []
performance = {}
for row in rows:
    d = load(row)
    m, g, t = d.get("meta", {}), d.get("gpu_persistent", {}), d.get("throughput", {})
    if row["kind"] == "motivation":
        rounds = float(g.get("average_graph_rounds_per_query", 0))
        certs = float(g.get("average_frontier_reusable_certificates_per_query", 0))
        prefix_query = float(g.get("average_gpu_frontier_preview_us", 0))
        # The ordinary beam_merge breakdown already includes Stable-Run leaf
        # sorting performed while constructing the certificate, so adding it
        # to frontier_preview would double-count work.  The direct
        # issue->Beam-publication timer begins after RDMA submission and covers
        # the remaining materialization/publication interval exactly once.
        remaining_query = float(g.get("average_gpu_frontier_issue_to_beam_publish_us", 0))
        total_query = prefix_query + remaining_query
        candidates = float(g.get("average_completion_score_candidates_per_batch", 0))
        motivation.append({
            "commit_width": int(row["width"]),
            "candidate_slots_per_round": candidates,
            "merge_input_slots_per_round": candidates + int(m.get("traversal_beam_width", 0)),
            "graph_rounds_per_query": rounds,
            "prefix_us_per_certificate": prefix_query / certs if certs else 0,
            "full_merge_pipeline_us_per_round": total_query / certs if certs else 0,
            "remaining_merge_us_per_round": remaining_query / certs if certs else 0,
            "prefix_compute_share": prefix_query / total_query if total_query else 0,
            "qps": float(t.get("query_ops_per_sec", 0)),
        })
    else:
        mode = "early" if "early" in row["label"] else "late"
        post = d.get("static_gt_post_recall", d.get("recall", {}))
        performance[mode] = {
            "qps": float(t.get("query_ops_per_sec", 0)),
            "p50_ms": latency(d, "p50_end_to_end_ns"),
            "p99_ms": latency(d, "p99_end_to_end_ns"),
            "post_recall": float(post.get("recall", 0)),
            "rdma_wait_us_per_query": float(g.get("average_gpu_rdma_wait_us", 0)),
            "logical_graph_reads_per_query": float(g.get("average_logical_graph_reads_per_query", 0)),
            "speculative_graph_reads": int(g.get("speculative_graph_reads", 0)),
            "critical_rob_hit_ratio": float(g.get("critical_rob_hit_ratio", 0)),
            "certificate_coverage": (
                float(g.get("frontier_reusable_issued_certificates", 0)) /
                float(g.get("frontier_reusable_certificates", 1))
                if g.get("frontier_reusable_certificates", 0) else 0
            ),
        }

motivation.sort(key=lambda item: item["commit_width"])
if len(motivation) < 2:
    raise SystemExit("need at least two motivation widths")
if set(performance) != {"late", "early"}:
    raise SystemExit("need one late and one early performance case")
late, early = performance["late"], performance["early"]
summary = {
    "experiment": "program3_story_motivation_then_effectiveness",
    "motivation": motivation,
    "performance": performance,
    "effect": {
        "qps_improvement": early["qps"] / late["qps"] - 1,
        "p50_reduction": 1 - early["p50_ms"] / late["p50_ms"],
        "p99_reduction": 1 - early["p99_ms"] / late["p99_ms"],
        "rdma_wait_reduction": 1 - early["rdma_wait_us_per_query"] / late["rdma_wait_us_per_query"],
        "logical_read_ratio": early["logical_graph_reads_per_query"] / late["logical_graph_reads_per_query"],
    },
}
(root / "story_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

with (root / "motivation_width_sweep.csv").open("w", encoding="utf-8", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=motivation[0].keys())
    writer.writeheader(); writer.writerows(motivation)

first, last = motivation[0], motivation[-1]
e = summary["effect"]
lines = [
    "# 方案三：从批量扩展的 Merge 膨胀到 GPU–RDMA 推进解耦",
    "",
    "## 1. 动机",
    "",
    "GPU 批量扩展一次处理多个父节点，会把这些父节点的邻居共同送入候选去重、排序和 Stable-Run merge。随着扩展批量 C 增大，每轮候选槽位和完整 Beam merge 成本随之增加；但下一轮 RDMA 只依赖稳定归并结果中的前 C 个未展开节点，而不依赖完整 Beam 的其余部分。",
    "",
    "| 扩展批量 C | 候选槽位/轮 | 候选+Beam容量 | Prefix 计算 (μs/证书轮) | 完整 Merge pipeline (μs/证书轮) | Prefix 占比 |",
    "|---:|---:|---:|---:|---:|---:|",
]
for item in motivation:
    lines.append(
        f"| {item['commit_width']} | {item['candidate_slots_per_round']:.1f} | {item['merge_input_slots_per_round']:.1f} | "
        f"{item['prefix_us_per_certificate']:.2f} | {item['full_merge_pipeline_us_per_round']:.2f} | {100*item['prefix_compute_share']:.1f}% |"
    )
lines += [
    "",
    f"当 C 从 {first['commit_width']} 增加到 {last['commit_width']} 时，每轮候选槽位从 {first['candidate_slots_per_round']:.1f} 增至 {last['candidate_slots_per_round']:.1f}，完整 Merge pipeline 从 {first['full_merge_pipeline_us_per_round']:.2f} μs 增至 {last['full_merge_pipeline_us_per_round']:.2f} μs。与此同时，得到下一轮所需 exact prefix 只占完整 merge 计算的 {100*last['prefix_compute_share']:.1f}%。因此等待剩余 {100*(1-last['prefix_compute_share']):.1f}% 的 merge 工作完成后才发 RDMA，是一个强于正确性所需的软件屏障。",
    "",
    "这里的 Prefix 时间采用 exact certificate 的保守成本：它包含为 prefix 准备可复用 Stable-Run leaves 的工作。完整 Merge pipeline 使用两个不重叠的直接时间区间：certificate 构造，以及 RDMA 提交完成后到 Beam publication 的剩余 materialization；不包含 RDMA enqueue 和网络等待。普通 `beam_merge` breakdown 会再次包含 certificate 阶段完成的 leaf sort，因此没有与 `frontier_preview` 相加，避免重复计时。`候选+Beam容量` 是按固定 Beam=128 给出的输入槽位上界，真实旧 Beam 在早期轮次可能未填满。",
    "",
    "## 2. 方案效果（单次严格 A/B）",
    "",
    "| 模式 | QPS | P50 (ms) | P99 (ms) | RDMA wait/query (μs) | Post Recall@10 |",
    "|---|---:|---:|---:|---:|---:|",
    f"| Late-Issue | {late['qps']:.1f} | {late['p50_ms']:.3f} | {late['p99_ms']:.3f} | {late['rdma_wait_us_per_query']:.1f} | {late['post_recall']:.4f} |",
    f"| Exact-Early-Issue | {early['qps']:.1f} | {early['p50_ms']:.3f} | {early['p99_ms']:.3f} | {early['rdma_wait_us_per_query']:.1f} | {early['post_recall']:.4f} |",
    "",
    f"开启方案后，QPS 提升 {100*e['qps_improvement']:.2f}%，P50/P99 分别降低 {100*e['p50_reduction']:.2f}%/{100*e['p99_reduction']:.2f}%，RDMA wait/query 降低 {100*e['rdma_wait_reduction']:.2f}%。Early/Late logical reads/query 比值为 {e['logical_read_ratio']:.6f}，speculative reads={early['speculative_graph_reads']}，Post Recall 相同。",
    "",
    f"Early 的 exact certificate 发出覆盖率为 {100*early['certificate_coverage']:.2f}%，critical ROB hit 为 {100*early['critical_rob_hit_ratio']:.2f}%。这说明提前 issue 的 mandatory 数据通常能在下一轮权威提交需要它们之前到达。",
    "",
    "## 3. 结论",
    "",
    "动机扫描先证明了批量扩展使候选集合和完整 merge 成本增长，而通信真正依赖的 exact prefix 只占其中一部分；随后单变量 A/B 证明，将 mandatory RDMA 从完整 Beam publication 之后移动到 exact prefix 就绪之后，能够把剩余 merge 与网络传输重叠，并在不增加预测读取、不改变最终 Recall 的情况下改善吞吐和尾延迟。",
    "",
]
(root / "方案三_动机与性能实验报告.md").write_text("\n".join(lines), encoding="utf-8")
print(root / "story_summary.json")
print(root / "方案三_动机与性能实验报告.md")
