#!/usr/bin/env python3

import argparse
import csv
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path


IDENTITY_PATTERN = re.compile(
    r"depth_(?P<depth>\d+)/concurrency_(?P<concurrency>\d+)/"
    r"repeat_(?P<repetition>\d+)")


def mean(values):
    available = [value for value in values if value is not None]
    return statistics.mean(available) if available else None


def format_number(value, digits=2):
    return "n/a" if value is None else f"{value:.{digits}f}"


def format_percent(value):
    return "n/a" if value is None else f"{value * 100.0:.2f}%"


def load_rows(root):
    rows = []
    for path in sorted(root.glob(
            "trace/**/rdma_trace.ordered_commit_oracle.summary.json")):
        match = IDENTITY_PATTERN.search(path.as_posix())
        if match is None:
            continue
        summary = json.loads(path.read_text(encoding="utf-8"))
        aggregate = summary["aggregate"]
        zero_gpu = aggregate[
            "query_strict_saved_over_gpu_by_task_overhead"
        ].get("0us", {}).get("p50")
        two_gpu = aggregate[
            "query_strict_saved_over_gpu_by_task_overhead"
        ].get("2us", {}).get("p50")
        rows.append({
            "depth": int(match.group("depth")),
            "concurrency": int(match.group("concurrency")),
            "repetition": int(match.group("repetition")),
            "completion_granularity":
                summary["measurement"]["completion_granularity"],
            "queries": summary["integrity"]["query_records"],
            "primary_rounds": aggregate["primary_rounds"],
            "multi_release_round_fraction":
                aggregate["multi_release_round_fraction"],
            "spread_p50_us": (
                None
                if aggregate["strict_completion_spread_p50_ns"] is None
                else aggregate["strict_completion_spread_p50_ns"] / 1000.0),
            "spread_p90_us": (
                None
                if aggregate["strict_completion_spread_p90_ns"] is None
                else aggregate["strict_completion_spread_p90_ns"] / 1000.0),
            "ready_tile_10us_round_fraction": aggregate[
                "round_fraction_with_tile_ready_10us_before_tail"],
            "zero_overhead_saved_over_gpu_p50": zero_gpu,
            "two_us_overhead_saved_over_gpu_p50": two_gpu,
            "verdict": summary["screening"]["verdict"],
            "integrity_clean": summary["screening"]["checks"][
                "integrity_clean"],
            "summary_path": str(path.resolve()),
        })
    return rows


def group_rows(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["depth"], row["concurrency"])].append(row)
    result = []
    fields = (
        "multi_release_round_fraction",
        "spread_p50_us",
        "spread_p90_us",
        "ready_tile_10us_round_fraction",
        "zero_overhead_saved_over_gpu_p50",
        "two_us_overhead_saved_over_gpu_p50",
    )
    for (depth, concurrency), group in sorted(groups.items()):
        record = {
            "depth": depth,
            "concurrency": concurrency,
            "repetitions": len(group),
            "queries": sum(row["queries"] for row in group),
            "primary_rounds": sum(row["primary_rounds"] for row in group),
            "completion_granularity": ",".join(sorted({
                row["completion_granularity"] for row in group})),
            "all_integrity_clean": all(
                row["integrity_clean"] for row in group),
            "all_support_prototype": all(
                row["verdict"] == "supports ordered-commit prototype"
                for row in group),
        }
        for field in fields:
            record[field] = mean(row[field] for row in group)
        result.append(record)
    return result


def write_csv(path, rows):
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_report(path, grouped):
    lines = [
        "# Ordered-commit motivation matrix",
        "",
        "The zero-overhead column is an oracle upper bound. It already assumes "
        "that movable validation/decode/PQ/visited work can execute at each "
        "observable release without queueing, state-transfer, or scheduling "
        "cost. It is not a projected speedup.",
        "",
        "| depth | concurrency | reps | queries | release coverage | strict "
        "spread P50/P90 | tile ready +10us | zero-overhead oracle / GPU P50 | "
        "2us/tile oracle / GPU P50 | verdict |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in grouped:
        if not row["all_integrity_clean"]:
            verdict = "invalid"
        elif row["all_support_prototype"]:
            verdict = "supports prototype"
        elif "parent" not in row["completion_granularity"] and (
                "tile" not in row["completion_granularity"]):
            verdict = "negative at shard granularity"
        else:
            verdict = "does not support prototype"
        zero_oracle = row["zero_overhead_saved_over_gpu_p50"]
        two_us_oracle = row["two_us_overhead_saved_over_gpu_p50"]
        lines.append(
            f"| {row['depth']} | {row['concurrency']} | "
            f"{row['repetitions']} | {row['queries']} | "
            f"{format_percent(row['multi_release_round_fraction'])} | "
            f"{format_number(row['spread_p50_us'])}/"
            f"{format_number(row['spread_p90_us'])} us | "
            f"{format_percent(row['ready_tile_10us_round_fraction'])} | "
            f"{format_percent(zero_oracle)} | "
            f"{format_percent(two_us_oracle)} | "
            f"{verdict} |")
    lines.extend([
        "",
        "## Decision rule",
        "",
        "Proceed only with clean integrity, >=25% rounds carrying multiple "
        "release boundaries, strict spread P50 >=10 us or P90 >=25 us, and "
        "a zero-overhead release-time oracle >=8% of query GPU residence at "
        "P50. A shard-level failure stops a shard-only design. It does not "
        "fabricate a conclusion about unobserved parent completion.",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    rows = load_rows(args.root)
    grouped = group_rows(rows)
    write_csv(args.root / "ordered_commit_oracle_runs.csv", rows)
    write_csv(args.root / "ordered_commit_oracle_matrix.csv", grouped)
    write_report(args.root / "ORDERED_COMMIT_REPORT.md", grouped)
    print(args.root / "ORDERED_COMMIT_REPORT.md")


if __name__ == "__main__":
    main()
