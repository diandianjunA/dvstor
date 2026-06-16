#!/usr/bin/env python3
import json
import sys
from pathlib import Path


DEFAULT_PROFILES = [
    "00_baseline",
    "01_rabitq_gpu_pipeline",
    "02_rabitq_gpu_pipeline_aldi",
    "03_rabitq_gpu_pipeline_aldi_rdma",
]


def latest_report(report_dir: Path, profile: str) -> Path | None:
    files = sorted((report_dir / profile).glob(f"sift100m_{profile}_*.json"))
    return files[-1] if files else None


def ns_to_ms(value: float | int | None) -> float:
    return 0.0 if value is None else float(value) / 1_000_000.0


def get_counter(section: dict, name: str) -> float:
    return float(section.get("counters", {}).get(name, 0.0))


def pct(num: float, den: float) -> float:
    return 0.0 if den == 0 else 100.0 * num / den


def row_for(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    throughput = data.get("throughput", {})
    recall = data.get("recall", {})
    query = data.get("query_breakdown", {})
    insert = data.get("insert_breakdown", {})
    qlat = query.get("latency", {})
    ilat = insert.get("latency", {})

    q_vector_bytes = get_counter(query, "vector_rdma_bytes")
    q_l0 = get_counter(query, "rabitq_l0_candidates")
    q_l1 = get_counter(query, "rabitq_l1_candidates")
    q_exact_reads = get_counter(query, "rabitq_exact_vector_reads")
    a_audits = get_counter(insert, "storage_owner_anchor_audits")
    a_failures = get_counter(insert, "storage_owner_anchor_audit_failures")

    return {
        "profile": path.parent.name,
        "total_qps": throughput.get("total_ops_per_sec", 0.0),
        "query_qps": throughput.get("query_ops_per_sec", 0.0),
        "write_qps": throughput.get("write_ops_per_sec", 0.0),
        "recall": recall.get("recall", 0.0),
        "query_p50_ms": ns_to_ms(qlat.get("p50_end_to_end_ns")),
        "query_p95_ms": ns_to_ms(qlat.get("p95_end_to_end_ns")),
        "insert_p50_ms": ns_to_ms(ilat.get("p50_end_to_end_ns")),
        "insert_p95_ms": ns_to_ms(ilat.get("p95_end_to_end_ns")),
        "vector_rdma_gb": q_vector_bytes / (1024.0 ** 3),
        "rabitq_keep_pct": pct(q_l1, q_l0),
        "rabitq_drop_pct": pct(q_l0 - q_l1, q_l0),
        "rabitq_exact_reads": q_exact_reads,
        "rdma_nodes_per_batch": float(query.get("counters", {}).get(
            "vector_rdma_mean_active_nodes_per_batch", 0.0)),
        "rdma_qps_per_batch": float(query.get("counters", {}).get(
            "vector_rdma_mean_active_qps_per_batch", 0.0)),
        "rdma_reads_per_cqe": float(query.get("counters", {}).get(
            "vector_rdma_reads_per_cqe", 0.0)),
        "aldi_audits": a_audits,
        "aldi_fail_pct": pct(a_failures, a_audits),
        "report": str(path),
    }


def print_table(rows: list[dict]) -> None:
    columns = [
        ("profile", "profile"),
        ("query_qps", "query_qps"),
        ("write_qps", "write_qps"),
        ("recall", "recall"),
        ("query_p50_ms", "q_p50_ms"),
        ("insert_p50_ms", "ins_p50_ms"),
        ("vector_rdma_gb", "vec_rdma_gb"),
        ("rabitq_drop_pct", "rabitq_drop_%"),
        ("rdma_nodes_per_batch", "rdma_nodes/b"),
        ("rdma_qps_per_batch", "rdma_qps/b"),
        ("rdma_reads_per_cqe", "rdma_reads/cqe"),
        ("aldi_fail_pct", "aldi_fail_%"),
    ]
    widths = []
    for key, title in columns:
        width = len(title)
        for row in rows:
            value = row[key]
            if isinstance(value, float):
                text = f"{value:.3f}"
            else:
                text = str(value)
            width = max(width, len(text))
        widths.append(width)

    header = "  ".join(title.ljust(widths[i]) for i, (_, title) in enumerate(columns))
    print(header)
    print("  ".join("-" * width for width in widths))
    for row in rows:
        cells = []
        for i, (key, _) in enumerate(columns):
            value = row[key]
            if isinstance(value, float):
                text = f"{value:.3f}"
                cells.append(text.rjust(widths[i]))
            else:
                cells.append(str(value).ljust(widths[i]))
        print("  ".join(cells))

    print()
    for row in rows:
        print(f"{row['profile']}: {row['report']}")


def main() -> int:
    script_dir = Path(__file__).resolve().parent
    report_dir = Path(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].startswith("/") else script_dir / "reports"
    profiles = sys.argv[1:] if report_dir == script_dir / "reports" else sys.argv[2:]
    if not profiles:
        profiles = DEFAULT_PROFILES

    rows = []
    missing = []
    for profile in profiles:
        path = latest_report(report_dir, profile)
        if path is None:
            missing.append(profile)
            continue
        rows.append(row_for(path))

    if rows:
        print_table(rows)
    if missing:
        print("missing reports: " + ", ".join(missing), file=sys.stderr)
    return 1 if missing and not rows else 0


if __name__ == "__main__":
    raise SystemExit(main())
