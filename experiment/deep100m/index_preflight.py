#!/usr/bin/env python3
"""Independent DEEP100M index preflight and graph-stage validator."""

import argparse
import json
import math
import os
from pathlib import Path
import struct
import sys

DATASET_NAME = "DEEP100M"
EXPECTED_DTYPE = "float32"
EXPECTED_SUFFIX = ".fbin"
COMPONENT_BYTES = 4
NODE_BYTES = 416
GRAPH_RECORD_BYTES = 832
IDMAP_BYTES = 24
EXTENT_BYTES = 1
MAX_PQ_SUBQUANTIZERS = 32
GIB = 1 << 30


def fail(message: str) -> None:
    raise ValueError(message)


def positive(name: str, value: int) -> int:
    if value <= 0:
        fail(f"{name} must be > 0: {value}")
    return value


def exact_dataset(path: Path, expected_dim: int, max_vectors: int) -> tuple[int, int]:
    if not path.is_file():
        fail(f"base dataset is missing: {path}")
    if path.suffix != EXPECTED_SUFFIX:
        fail(f"{DATASET_NAME} base must use {EXPECTED_SUFFIX}: {path}")
    with path.open("rb") as stream:
        header = stream.read(8)
    if len(header) != 8:
        fail(f"dataset header is truncated: {path}")
    vectors, dim = struct.unpack("<II", header)
    if vectors == 0 or dim == 0:
        fail(f"dataset header has zero vectors/dimension: {vectors} x {dim}")
    if dim != expected_dim:
        fail(f"dataset dim is {dim}, expected {expected_dim}")
    component_bytes = COMPONENT_BYTES
    expected_bytes = 8 + vectors * dim * component_bytes
    actual_bytes = path.stat().st_size
    if actual_bytes != expected_bytes:
        fail(
            f"dataset size/header mismatch: bytes={actual_bytes}, "
            f"expected={expected_bytes}"
        )
    if max_vectors > vectors:
        fail(f"MAX_VECTORS={max_vectors} exceeds dataset vectors={vectors}")
    return max_vectors, dim


def available_memory_bytes() -> int:
    fields = {}
    with open("/proc/meminfo", "r", encoding="utf-8") as stream:
        for line in stream:
            key, raw = line.split(":", 1)
            fields[key] = int(raw.strip().split()[0]) * 1024
    return fields.get("MemAvailable", 0)


def cgroup_memory_available() -> int | None:
    candidates = (
        (Path("/sys/fs/cgroup/memory.max"), Path("/sys/fs/cgroup/memory.current")),
        (
            Path("/sys/fs/cgroup/memory/memory.limit_in_bytes"),
            Path("/sys/fs/cgroup/memory/memory.usage_in_bytes"),
        ),
    )
    for limit_path, used_path in candidates:
        try:
            raw_limit = limit_path.read_text(encoding="utf-8").strip()
            if raw_limit == "max":
                return None
            limit = int(raw_limit)
            used = int(used_path.read_text(encoding="utf-8").strip())
            # Ignore the conventional v1 "unlimited" sentinel.
            if limit >= (1 << 60):
                return None
            return max(0, limit - used)
        except (OSError, ValueError):
            continue
    return None


def validate_numeric(args: argparse.Namespace, vectors: int, dim: int) -> None:
    for name in (
        "shards",
        "degree",
        "beam",
        "partition_max_degree",
        "pq_subquantizers",
        "pq_train_samples",
        "build_threads",
        "pq_threads",
    ):
        positive(name.upper(), getattr(args, name))
    if args.shards > 64:
        fail(f"SHARDS exceeds tagged-pointer capacity 64: {args.shards}")
    if vectors > 0xFFFFFFFF:
        fail(f"vector count exceeds uint32 id capacity: {vectors}")
    if vectors <= args.degree:
        fail(f"dataset must contain at least R+1 vectors: N={vectors}, R={args.degree}")
    if args.beam == 0:
        fail("BUILD_BEAM must be > 0")
    if args.partition_max_degree > args.degree:
        fail(
            "PARTITION_MAX_DEGREE must be <= R: "
            f"{args.partition_max_degree} > {args.degree}"
        )
    if args.partition not in {"balanced", "bfs", "metis"}:
        fail(f"unsupported PARTITION_STRATEGY: {args.partition}")
    if not math.isfinite(args.alpha) or args.alpha < 1.0:
        fail(f"ALPHA must be finite and >= 1.0: {args.alpha}")
    if not math.isfinite(args.imbalance) or args.imbalance < 1.0:
        fail(
            "PARTITION_IMBALANCE must be finite and >= 1.0: "
            f"{args.imbalance}"
        )
    if args.pq_subquantizers > MAX_PQ_SUBQUANTIZERS:
        fail(
            f"PQ_SUBQUANTIZERS exceeds runtime maximum "
            f"{MAX_PQ_SUBQUANTIZERS}"
        )
    if dim % args.pq_subquantizers:
        fail(
            f"DIM={dim} is not divisible by "
            f"PQ_SUBQUANTIZERS={args.pq_subquantizers}"
        )
    if args.pq_train_samples < 256:
        fail("PQ_TRAIN_SAMPLES must be >= 256")
    for name, value in (
        ("BUILD_THREADS", args.build_threads),
        ("PQ_THREADS", args.pq_threads),
    ):
        if value > 32:
            fail(f"{name} exceeds the 32-thread safety limit: {value}")


def resource_check(args: argparse.Namespace, vectors: int, dim: int) -> None:
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    if not os.access(output_dir, os.W_OK | os.X_OK):
        fail(f"index output directory is not writable: {output_dir}")

    estimate = vectors * (
        NODE_BYTES
        + GRAPH_RECORD_BYTES
        + IDMAP_BYTES
        + args.pq_subquantizers
        + EXTENT_BYTES
    )
    required_disk = math.ceil(estimate * 1.10)
    stat = os.statvfs(output_dir)
    available_disk = stat.f_bavail * stat.f_frsize
    if available_disk < required_disk:
        fail(
            f"insufficient free disk for a fresh {DATASET_NAME} index: "
            f"available={available_disk / GIB:.1f} GiB, "
            f"required={required_disk / GIB:.1f} GiB"
        )

    # Includes raw vectors, graph arrays, edge extraction/sort and METIS headroom.
    peak_memory = max(
        vectors * (dim * COMPONENT_BYTES + args.degree * 4 + 1),
        int(vectors * 1560),
    )
    required_memory = math.ceil(peak_memory * 1.20)
    host_available = available_memory_bytes()
    if host_available and host_available < required_memory:
        fail(
            f"insufficient MemAvailable for the conservative build estimate: "
            f"available={host_available / GIB:.1f} GiB, "
            f"required={required_memory / GIB:.1f} GiB"
        )
    cgroup_available = cgroup_memory_available()
    if cgroup_available is not None and cgroup_available < required_memory:
        fail(
            f"cgroup memory headroom is too small: "
            f"available={cgroup_available / GIB:.1f} GiB, "
            f"required={required_memory / GIB:.1f} GiB"
        )
    print(
        f"[preflight] resources: estimated_final={estimate / GIB:.1f} GiB "
        f"disk_required_with_margin={required_disk / GIB:.1f} GiB "
        f"disk_available={available_disk / GIB:.1f} GiB "
        f"memory_required_with_margin={required_memory / GIB:.1f} GiB "
        f"memory_available={host_available / GIB:.1f} GiB"
    )


def command_preflight(args: argparse.Namespace) -> None:
    vectors, dim = exact_dataset(args.data, args.dim, args.max_vectors)
    if args.dtype != EXPECTED_DTYPE:
        fail(f"{DATASET_NAME} dtype must be {EXPECTED_DTYPE}: {args.dtype}")
    validate_numeric(args, vectors, dim)
    if args.check_resources:
        resource_check(args, vectors, dim)
    print(
        f"[preflight] {DATASET_NAME}: vectors={vectors} dim={dim} "
        f"dtype={args.dtype} R={args.degree} beam={args.beam} "
        f"partition={args.partition} pmd={args.partition_max_degree} "
        f"shards={args.shards} pq={args.pq_subquantizers}"
    )


def read_metadata(path: Path) -> dict:
    try:
        with path.open("r", encoding="utf-8") as stream:
            value = json.load(stream)
    except (OSError, json.JSONDecodeError) as error:
        fail(f"cannot read metadata {path}: {error}")
    if not isinstance(value, dict):
        fail(f"metadata root is not an object: {path}")
    return value


def command_schema(args: argparse.Namespace) -> None:
    metadata = read_metadata(args.metadata)
    value = metadata.get("schema_version")
    if not isinstance(value, int):
        fail(f"metadata schema_version is invalid: {value!r}")
    print(value)


def command_graph(args: argparse.Namespace) -> None:
    metadata = read_metadata(args.metadata)
    expected = {
        "schema_version": 15,
        "output_prefix": str(args.prefix),
        "distance": "l2",
        "num_vectors": args.max_vectors,
        "dim": args.dim,
        "R": args.degree,
        "beam_width_construction": args.beam,
        "num_memory_nodes": args.shards,
        "node_layout": "plain",
        "storage_format": "vamana_tagged_v2",
        "remote_ptr_format": "tagged_inc24_shard6_off34x16_v1",
        "vector_data_type": EXPECTED_DTYPE,
        "partition_strategy": args.partition,
        "partition_max_degree": args.partition_max_degree,
        "idmap_format": "owner_sharded_v2_bound",
        "centroid_state_format": "physical_shard_centroid_v2_bound",
    }
    errors = [
        f"{key}: metadata={metadata.get(key)!r}, expected={wanted!r}"
        for key, wanted in expected.items()
        if metadata.get(key) != wanted
    ]
    fingerprints = metadata.get("shard_build_fingerprints")
    counts = metadata.get("hot_graph_entry_counts")
    if (
        not isinstance(fingerprints, list)
        or len(fingerprints) != args.shards
        or any(not isinstance(value, int) or value == 0 for value in fingerprints)
    ):
        errors.append("shard_build_fingerprints is invalid")
        fingerprints = [None] * args.shards
    if (
        not isinstance(counts, list)
        or len(counts) != args.shards
        or any(not isinstance(value, int) or value <= 0 for value in counts)
        or sum(counts) != args.max_vectors
    ):
        errors.append("hot_graph_entry_counts is invalid")

    for shard in range(1, args.shards + 1):
        dat = Path(f"{args.prefix}_node{shard}_of{args.shards}.dat")
        idmap = Path(f"{args.prefix}_node{shard}_of{args.shards}.idmap")
        centroid = Path(f"{args.prefix}_node{shard}_of{args.shards}.centroid")
        for sidecar in (idmap, centroid):
            if not sidecar.is_file() or sidecar.stat().st_size == 0:
                errors.append(f"missing graph-stage artifact: {sidecar}")
        try:
            actual_size = dat.stat().st_size
            with dat.open("rb") as stream:
                header = stream.read(16)
            if len(header) != 16:
                errors.append(f"shard header is truncated: {dat}")
                continue
            declared_size, fingerprint = struct.unpack("<QQ", header)
            if declared_size != actual_size:
                errors.append(
                    f"shard declared size mismatch: {dat}: "
                    f"declared={declared_size}, actual={actual_size}"
                )
            expected_fingerprint = fingerprints[shard - 1]
            if expected_fingerprint is not None and fingerprint != expected_fingerprint:
                errors.append(f"shard fingerprint mismatch: {dat}")
        except OSError as error:
            errors.append(f"cannot inspect shard {dat}: {error}")

    if errors:
        print(f"invalid resumable graph stage: {args.metadata}", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        raise SystemExit(1)
    print(
        f"[validate] schema-15 graph stage is complete and resumable: "
        f"{args.prefix}"
    )


def add_contract_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--prefix", required=True, type=Path)
    parser.add_argument("--max-vectors", required=True, type=int)
    parser.add_argument("--dim", required=True, type=int)
    parser.add_argument("--degree", required=True, type=int)
    parser.add_argument("--beam", required=True, type=int)
    parser.add_argument("--shards", required=True, type=int)
    parser.add_argument("--partition", required=True)
    parser.add_argument("--partition-max-degree", required=True, type=int)


def main() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)

    preflight = commands.add_parser("preflight")
    preflight.add_argument("--data", required=True, type=Path)
    preflight.add_argument("--output-dir", required=True, type=Path)
    preflight.add_argument("--max-vectors", required=True, type=int)
    preflight.add_argument("--dim", required=True, type=int)
    preflight.add_argument("--dtype", required=True)
    preflight.add_argument("--degree", required=True, type=int)
    preflight.add_argument("--beam", required=True, type=int)
    preflight.add_argument("--shards", required=True, type=int)
    preflight.add_argument("--partition", required=True)
    preflight.add_argument("--partition-max-degree", required=True, type=int)
    preflight.add_argument("--imbalance", required=True, type=float)
    preflight.add_argument("--alpha", required=True, type=float)
    preflight.add_argument("--pq-subquantizers", required=True, type=int)
    preflight.add_argument("--pq-train-samples", required=True, type=int)
    preflight.add_argument("--build-threads", required=True, type=int)
    preflight.add_argument("--pq-threads", required=True, type=int)
    preflight.add_argument("--check-resources", action="store_true")
    preflight.set_defaults(run=command_preflight)

    schema = commands.add_parser("schema")
    schema.add_argument("--metadata", required=True, type=Path)
    schema.set_defaults(run=command_schema)

    graph = commands.add_parser("graph")
    graph.add_argument("--metadata", required=True, type=Path)
    add_contract_arguments(graph)
    graph.set_defaults(run=command_graph)

    args = parser.parse_args()
    try:
        args.run(args)
    except (ValueError, KeyError, TypeError, OverflowError) as error:
        print(f"{DATASET_NAME} index preflight failed: {error}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
