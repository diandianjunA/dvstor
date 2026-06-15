#!/usr/bin/env python3

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
REPORT_ROOT = ROOT / "reports"


def read_manifest(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key] = value
    return values


def ns_to_ms(value: int | float) -> float:
    return float(value) / 1_000_000.0


def per_query(value: int | float, count: int) -> float:
    return float(value) / count if count else 0.0


rows: list[dict[str, object]] = []
for report_path in sorted(REPORT_ROOT.glob("*/*/report.json")):
    report = json.loads(report_path.read_text(encoding="utf-8"))
    manifest = read_manifest(report_path.parent / "case.env")
    query = report.get("query_breakdown", {})
    latency = query.get("latency", {})
    breakdown = query.get("breakdown", {})
    counters = query.get("counters", {})
    count = int(query.get("count", 0))
    service_ns = float(latency.get("service_ns", 0))

    def share(category: str) -> float:
        return 100.0 * float(breakdown.get(category, 0)) / service_ns if service_ns else 0.0

    rows.append({
        "experiment": manifest.get("experiment", report_path.parents[1].name),
        "label": manifest.get("label", report_path.parent.name),
        "gpudirect": manifest.get("gpudirect_rdma", ""),
        "coroutines": manifest.get("coroutines", ""),
        "clients": manifest.get("client_threads", ""),
        "expansion_batch": manifest.get("expansion_batch", ""),
        "queries": count,
        "qps": report.get("throughput", {}).get("query_ops_per_sec", 0),
        "mean_ms": ns_to_ms(latency.get("mean_end_to_end_ns", 0)),
        "p95_ms": ns_to_ms(latency.get("p95_end_to_end_ns", 0)),
        "p99_ms": ns_to_ms(latency.get("p99_end_to_end_ns", 0)),
        "cpu_pct": share("cpu_ns"),
        "rdma_pct": share("rdma_ns"),
        "gpu_pct": share("gpu_ns"),
        "transfer_pct": share("transfer_ns"),
        "rdma_bytes_per_query": per_query(counters.get("rdma_read_bytes", 0), count),
        "rdma_ops_per_query": per_query(counters.get("rdma_read_ops", 0), count),
        "neighbor_reads_per_query": per_query(counters.get("neighbor_rdma_read_ops", 0), count),
        "vector_reads_per_query": per_query(counters.get("vector_rdma_read_ops", 0), count),
        "vector_reads_per_batch": counters.get("vector_rdma_reads_per_batch", 0),
        "visited_lists_per_query": per_query(counters.get("visited_neighborlists", 0), count),
        "host_staging_bytes_per_query": per_query(counters.get("query_host_staging_fallback_bytes", 0), count),
        "gpu_direct_bytes_per_query": per_query(counters.get("query_rdma_to_staging_bytes", 0), count),
        "recall_at_10": report.get("recall", {}).get("recall", ""),
        "report": str(report_path.relative_to(ROOT)),
    })

if not rows:
    raise SystemExit(f"no reports found under {REPORT_ROOT}")

csv_path = REPORT_ROOT / "summary.csv"
with csv_path.open("w", newline="", encoding="utf-8") as output:
    writer = csv.DictWriter(output, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)

markdown_path = REPORT_ROOT / "summary.md"
columns = [
    "experiment", "label", "qps", "mean_ms", "p95_ms",
    "rdma_pct", "gpu_pct", "transfer_pct", "rdma_ops_per_query",
    "host_staging_bytes_per_query", "gpu_direct_bytes_per_query",
]
with markdown_path.open("w", encoding="utf-8") as output:
    output.write("| " + " | ".join(columns) + " |\n")
    output.write("| " + " | ".join(["---"] * len(columns)) + " |\n")
    for row in rows:
        values = []
        for column in columns:
            value = row[column]
            if isinstance(value, float):
                values.append(f"{value:.3f}")
            else:
                values.append(str(value))
        output.write("| " + " | ".join(values) + " |\n")

print(f"csv: {csv_path}")
print(f"markdown: {markdown_path}")

