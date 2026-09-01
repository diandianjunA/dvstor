#!/usr/bin/env python3
import argparse
import csv
import json
import math
import statistics
from pathlib import Path


T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
       6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
       11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131,
       16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
       21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060,
       26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042}


def number(value, default=math.nan):
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def latency_us(report, section, key):
    return number(report.get(section, {}).get("latency", {}).get(key)) / 1000


def mean(values):
    values = [value for value in values if math.isfinite(value)]
    return statistics.fmean(values) if values else math.nan


def ci95(values):
    values = [value for value in values if math.isfinite(value)]
    if len(values) < 2:
        return math.nan
    return T95.get(len(values) - 1, 1.96) * statistics.stdev(values) / math.sqrt(len(values))


def recall(report, key):
    return number(report.get(key, {}).get("recall"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("result_root", type=Path)
    args = parser.parse_args()

    rows = []
    for metadata_path in sorted(args.result_root.glob("runs/*/run_metadata.json")):
        case_dir = metadata_path.parent
        report_path = case_dir / "report.json"
        if not (case_dir / "DONE").is_file() or not report_path.is_file():
            continue
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        report = json.loads(report_path.read_text(encoding="utf-8"))
        throughput = report.get("throughput", {})
        stage2 = report.get("stage2", {})
        query_ops = number(throughput.get("query_ops"), 0.0)
        update_ops = number(throughput.get("write_ops"), 0.0)
        rows.append({
            "dataset": metadata["dataset"],
            "profile": metadata["profile"],
            "run_id": case_dir.name,
            "repeat": metadata["repeat"],
            "query_threads": metadata["query_threads"],
            "update_threads": metadata["update_threads"],
            "configured_read_ratio": metadata["configured_read_ratio"],
            "actual_query_operation_ratio": query_ops / (query_ops + update_ops)
                if query_ops + update_ops else math.nan,
            "query_qps": number(throughput.get("query_ops_per_sec")),
            "normalized_query_qps": math.nan,
            "update_qps": number(throughput.get("write_ops_per_sec")),
            "durable_update_qps": number(throughput.get("durable_write_ops_per_sec")),
            "total_qps": number(throughput.get("total_ops_per_sec")),
            "query_mean_us": latency_us(report, "query_breakdown", "mean_end_to_end_ns"),
            "query_p95_us": latency_us(report, "query_breakdown", "p95_end_to_end_ns"),
            "query_p99_us": latency_us(report, "query_breakdown", "p99_end_to_end_ns"),
            "update_mean_us": latency_us(report, "insert_breakdown", "mean_end_to_end_ns"),
            "update_p99_us": latency_us(report, "insert_breakdown", "p99_end_to_end_ns"),
            "recall_before": recall(report, "recall"),
            "recall_after": recall(report, "static_gt_post_recall"),
            "client_drain_seconds": number(throughput.get("client_drain_seconds")),
            "maintenance_drain_seconds": number(throughput.get("maintenance_drain_seconds")),
            "stage2_remaining": number(stage2.get("remaining")),
            "stage2_failures": number(stage2.get("failures")),
            "report": str(report_path.resolve()),
        })
    if not rows:
        raise SystemExit(f"no completed runs found under {args.result_root / 'runs'}")

    baselines = {}
    for row in rows:
        if row["update_threads"] == 0:
            key = (row["dataset"], row["profile"], row["query_threads"])
            baselines.setdefault(key, []).append(row["query_qps"])
    for row in rows:
        key = (row["dataset"], row["profile"], row["query_threads"])
        baseline = mean(baselines.get(key, []))
        if math.isfinite(baseline) and baseline > 0:
            row["normalized_query_qps"] = row["query_qps"] / baseline

    args.result_root.mkdir(parents=True, exist_ok=True)
    raw_path = args.result_root / "raw_results.csv"
    with raw_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    groups = {}
    for row in rows:
        key = (row["dataset"], row["profile"], row["query_threads"], row["update_threads"])
        groups.setdefault(key, []).append(row)
    metrics = [
        "query_qps", "normalized_query_qps", "update_qps", "durable_update_qps",
        "total_qps", "query_mean_us", "query_p95_us", "query_p99_us",
        "update_mean_us", "update_p99_us", "actual_query_operation_ratio",
        "recall_before", "recall_after", "client_drain_seconds",
        "maintenance_drain_seconds", "stage2_remaining", "stage2_failures",
    ]
    fields = ["dataset", "profile", "query_threads", "update_threads", "runs",
              "configured_read_ratio"]
    for metric in metrics:
        fields.extend((f"{metric}_mean", f"{metric}_ci95"))
    summary_path = args.result_root / "summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for key, group in sorted(groups.items()):
            output = dict(zip(("dataset", "profile", "query_threads", "update_threads"), key))
            output["runs"] = len(group)
            output["configured_read_ratio"] = group[0]["configured_read_ratio"]
            for metric in metrics:
                values = [row[metric] for row in group]
                output[f"{metric}_mean"] = mean(values)
                output[f"{metric}_ci95"] = ci95(values)
            writer.writerow(output)
    print(f"[mixed-test] raw results: {raw_path}")
    print(f"[mixed-test] summary: {summary_path}")


if __name__ == "__main__":
    main()
