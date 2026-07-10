#!/usr/bin/env python3
"""Compare a DVSTOR GPU-persistent report with DVSTOR or OdinANN JSON."""

import argparse
import json
from pathlib import Path


def nested(document, *paths, default=0.0):
    for path in paths:
        value = document
        try:
            for key in path.split("."):
                value = value[key]
            return float(value)
        except (KeyError, TypeError, ValueError):
            continue
    return float(default)


def metrics(document):
    query_p99_us = nested(document, "read_p99_us")
    if query_p99_us == 0:
        query_p99_us = nested(document, "query_breakdown.latency.p99_end_to_end_ns") / 1000.0
    write_p99_us = nested(document, "write_p99_us")
    if write_p99_us == 0:
        write_p99_us = nested(document, "insert_breakdown.latency.p99_end_to_end_ns") / 1000.0
    return {
        "query_qps": nested(document, "throughput.query_ops_per_sec", "read_qps"),
        "write_qps": nested(document, "throughput.write_ops_per_sec", "write_qps"),
        "total_qps": nested(document, "throughput.total_ops_per_sec"),
        "recall": nested(document, "static_gt_post_recall.recall", "recall.recall", "post_recall"),
        "query_p99_us": query_p99_us,
        "write_p99_us": write_p99_us,
    }


def ratio(candidate, baseline):
    return candidate / baseline if baseline > 0 else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--min-query-speedup", type=float, default=1.0)
    parser.add_argument("--max-recall-loss", type=float, default=0.01)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    baseline = metrics(json.loads(args.baseline.read_text(encoding="utf-8")))
    candidate_document = json.loads(args.candidate.read_text(encoding="utf-8"))
    candidate = metrics(candidate_document)
    result = {
        "baseline": baseline,
        "candidate": candidate,
        "query_speedup": ratio(candidate["query_qps"], baseline["query_qps"]),
        "write_speedup": ratio(candidate["write_qps"], baseline["write_qps"]),
        "query_p99_ratio": ratio(candidate["query_p99_us"], baseline["query_p99_us"]),
        "write_p99_ratio": ratio(candidate["write_p99_us"], baseline["write_p99_us"]),
        "recall_delta": candidate["recall"] - baseline["recall"],
        "gpu_persistent": candidate_document.get("gpu_persistent", {}),
    }
    result["passed"] = (
        result["query_speedup"] >= args.min_query_speedup
        and result["recall_delta"] >= -args.max_recall_loss
    )

    print(f"query QPS : {baseline['query_qps']:.2f} -> {candidate['query_qps']:.2f} "
          f"({result['query_speedup']:.3f}x)")
    print(f"write QPS : {baseline['write_qps']:.2f} -> {candidate['write_qps']:.2f} "
          f"({result['write_speedup']:.3f}x)")
    print(f"recall    : {baseline['recall']:.6f} -> {candidate['recall']:.6f} "
          f"({result['recall_delta']:+.6f})")
    print(f"query p99 : {baseline['query_p99_us']:.2f} -> {candidate['query_p99_us']:.2f} us")
    print(f"result    : {'PASS' if result['passed'] else 'FAIL'}")
    if args.output:
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    raise SystemExit(0 if result["passed"] else 2)


if __name__ == "__main__":
    main()
