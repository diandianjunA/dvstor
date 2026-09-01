#!/usr/bin/env python3
import argparse
import json
import math
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--query-threads", type=int, required=True)
    parser.add_argument("--update-threads", type=int, required=True)
    parser.add_argument("--warmup-seconds", type=int, required=True)
    parser.add_argument("--measure-seconds", type=int, required=True)
    args = parser.parse_args()

    with args.report.open(encoding="utf-8") as stream:
        report = json.load(stream)
    meta = report.get("meta", {})
    system = meta.get("system_variant", {})
    errors = []
    expected_workload = "mixed" if args.update_threads else "query"
    if system.get("profile_name") != args.profile:
        errors.append(f"profile={system.get('profile_name')!r}, expected {args.profile!r}")
    expected_modes = {
        "04_gpu_persistent_gpunetio_baseline": ("coupled", "fixed", "coupled"),
        "04_gpu_persistent_gpunetio": ("decoupled", "adaptive", "decoupled"),
    }.get(args.profile)
    if expected_modes:
        modes = system.get("resolved_modes", {})
        actual_modes = (
            modes.get("storage_owner_update_completion_mode"),
            modes.get("gpu_dynamic_graph_access_mode"),
            modes.get("gpu_rdma_search_progression_mode"),
        )
        if actual_modes != expected_modes:
            errors.append(f"resolved modes={actual_modes!r}, expected {expected_modes!r}")
    if meta.get("workload") != expected_workload:
        errors.append(f"workload={meta.get('workload')!r}, expected {expected_workload!r}")
    if int(meta.get("client_threads", -1)) != args.query_threads + args.update_threads:
        errors.append("client thread count differs from the requested total")
    if int(meta.get("warmup_seconds", -1)) != args.warmup_seconds:
        errors.append("warmup duration differs from the requested setting")
    if int(meta.get("measure_seconds", -1)) != args.measure_seconds:
        errors.append("measurement duration differs from the requested setting")
    if args.update_threads:
        if meta.get("mixed_dispatch_policy") != "fixed_threads":
            errors.append("mixed dispatch policy is not fixed_threads")
        split = meta.get("mixed_fixed_threads", {})
        if int(split.get("read_threads", -1)) != args.query_threads:
            errors.append(f"actual query threads are {split.get('read_threads')!r}")
        if int(split.get("write_threads", -1)) != args.update_threads:
            errors.append(f"actual update threads are {split.get('write_threads')!r}")
        expected_ratio = args.query_threads / (args.query_threads + args.update_threads)
        if not math.isclose(float(meta.get("read_ratio", -1)), expected_ratio,
                            rel_tol=1e-12, abs_tol=1e-12):
            errors.append("read ratio does not encode the requested fixed split")
        if any(float(meta.get(key, -1)) != value for key, value in (
                ("write_insert_ratio", 1.0),
                ("write_upsert_ratio", 0.0),
                ("write_delete_ratio", 0.0))):
            errors.append("update mix is not append-only 1/0/0")
    concurrency = meta.get("benchmark_driver_concurrency", {})
    if concurrency.get("client_threads_source") != "explicit":
        errors.append("benchmark runner did not use explicit client concurrency")
    if errors:
        raise SystemExit("invalid mixed-test report:\n  - " + "\n  - ".join(errors))

    throughput = report.get("throughput", {})
    print(
        "[mixed-test] valid report "
        f"query_qps={float(throughput.get('query_ops_per_sec', 0)):.1f} "
        f"update_qps={float(throughput.get('write_ops_per_sec', 0)):.1f}"
    )


if __name__ == "__main__":
    main()
