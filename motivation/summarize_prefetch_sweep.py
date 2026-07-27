#!/usr/bin/env python3
"""Create one CSV row per unfiltered prefetch-depth benchmark report."""

import argparse
import csv
import json
from pathlib import Path


FIELDS = [
    "report", "prefetch_depth", "client_threads", "query_qps",
    "mean_latency_ns", "p50_latency_ns", "p95_latency_ns",
    "p99_latency_ns", "p999_latency_ns", "recall_at_10",
    "average_gpu_query_us", "average_gpu_beam_selection_us",
    "average_gpu_rdma_issue_us", "average_gpu_rdma_wait_us",
    "average_gpu_graph_validation_us", "average_gpu_neighbor_decode_us",
    "average_gpu_pq_score_us", "average_gpu_visited_us",
    "average_gpu_beam_merge_us", "average_gpu_exact_us",
    "average_gpu_other_us", "graph_reads_per_query",
    "graph_retry_reads_per_query", "rdma_ops_per_query",
    "rdma_bytes_per_query", "average_parent_batch_size",
    "average_graph_shard_batches_per_query", "average_graph_rounds_per_query",
    "dynamic_pq_reads_per_query", "exact_reads_per_query",
]


def ratio(value, count):
    return 0.0 if not count else value / count


def report_row(path):
    with path.open("r", encoding="utf-8") as stream:
        root = json.load(stream)
    meta = root.get("meta", {})
    throughput = root.get("throughput", {})
    latency = root.get("query_breakdown", {}).get("latency", {})
    recall = root.get("recall", {})
    gpu = root.get("gpu_persistent", {})
    completed = gpu.get("queries_completed", 0)
    return {
        "report": str(path.resolve()),
        "prefetch_depth": meta.get("gpu_graph_prefetch_depth"),
        "client_threads": meta.get("client_threads"),
        "query_qps": throughput.get("query_ops_per_sec"),
        "mean_latency_ns": latency.get("mean_end_to_end_ns"),
        "p50_latency_ns": latency.get("p50_end_to_end_ns"),
        "p95_latency_ns": latency.get("p95_end_to_end_ns"),
        "p99_latency_ns": latency.get("p99_end_to_end_ns"),
        "p999_latency_ns": latency.get("p999_end_to_end_ns"),
        "recall_at_10": recall.get("recall") if recall.get("k") == 10 else None,
        "average_gpu_query_us": gpu.get("average_gpu_query_us"),
        "average_gpu_beam_selection_us":
            gpu.get("average_gpu_beam_selection_us"),
        "average_gpu_rdma_issue_us": gpu.get("average_gpu_rdma_issue_us"),
        "average_gpu_rdma_wait_us": gpu.get("average_gpu_rdma_wait_us"),
        "average_gpu_graph_validation_us":
            gpu.get("average_gpu_graph_validation_us"),
        "average_gpu_neighbor_decode_us":
            gpu.get("average_gpu_neighbor_decode_us"),
        "average_gpu_pq_score_us": gpu.get("average_gpu_pq_score_us"),
        "average_gpu_visited_us": gpu.get("average_gpu_visited_us"),
        "average_gpu_beam_merge_us": gpu.get("average_gpu_beam_merge_us"),
        "average_gpu_exact_us": gpu.get("average_gpu_exact_us"),
        "average_gpu_other_us": gpu.get("average_gpu_other_us"),
        "graph_reads_per_query": ratio(
            gpu.get("graph_page_requests", 0), completed),
        "graph_retry_reads_per_query": ratio(
            gpu.get("graph_read_retries", 0), completed),
        "rdma_ops_per_query": ratio(gpu.get("rdma_read_ops", 0), completed),
        "rdma_bytes_per_query": ratio(gpu.get("rdma_read_bytes", 0), completed),
        "average_parent_batch_size": gpu.get("average_parent_batch_size"),
        "average_graph_shard_batches_per_query":
            gpu.get("average_graph_shard_batches_per_query"),
        "average_graph_rounds_per_query":
            gpu.get("average_graph_rounds_per_query"),
        "dynamic_pq_reads_per_query": ratio(
            gpu.get("dynamic_code_reads", 0), completed),
        "exact_reads_per_query": ratio(
            gpu.get("exact_vector_reads", 0), completed),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "root", type=Path, nargs="?",
        default=Path(__file__).resolve().parent / "results" / "sweep")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    paths = sorted(args.root.rglob("sift100m_*.json"))
    rows = [report_row(path) for path in paths]
    destination = args.output or args.root / "prefetch_sweep.csv"
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"{destination} ({len(rows)} reports)")


if __name__ == "__main__":
    main()
