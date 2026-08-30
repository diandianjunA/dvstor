#!/usr/bin/env python3
"""Split the native SPACEV query/GT files into disjoint benchmark inputs."""

import argparse
import os
import pathlib
import struct

HEADER = struct.Struct("<II")
CHUNK_BYTES = 8 << 20


def header(path: pathlib.Path) -> tuple[int, int]:
    with path.open("rb") as stream:
        raw = stream.read(HEADER.size)
    if len(raw) != HEADER.size:
        raise ValueError(f"short header: {path}")
    return HEADER.unpack(raw)


def check_size(path: pathlib.Path, expected: int) -> None:
    actual = path.stat().st_size
    if actual != expected:
        raise ValueError(f"unexpected size for {path}: got {actual}, expected {expected}")


def copy_bytes(source, target, offset: int, count: int) -> None:
    source.seek(offset)
    remaining = count
    while remaining:
        data = source.read(min(CHUNK_BYTES, remaining))
        if not data:
            raise ValueError("source ended while copying")
        target.write(data)
        remaining -= len(data)


def write_rows(source_path: pathlib.Path, target_path: pathlib.Path,
               start: int, count: int, dim: int) -> None:
    temporary = target_path.with_suffix(target_path.suffix + f".tmp.{os.getpid()}")
    with source_path.open("rb") as source, temporary.open("wb") as target:
        target.write(HEADER.pack(count, dim))
        copy_bytes(source, target, HEADER.size + start * dim, count * dim)
    os.replace(temporary, target_path)


def write_groundtruth(source_path: pathlib.Path, target_path: pathlib.Path,
                      topk: int, count: int) -> None:
    """Write the benchmark's header-plus-ID-only groundtruth format."""
    row_bytes = topk * 4
    ids_offset = HEADER.size
    temporary = target_path.with_suffix(target_path.suffix + f".tmp.{os.getpid()}")
    with source_path.open("rb") as source, temporary.open("wb") as target:
        target.write(HEADER.pack(count, topk))
        copy_bytes(source, target, ids_offset, count * row_bytes)
    os.replace(temporary, target_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=pathlib.Path, required=True)
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    parser.add_argument("--recall-rows", type=int, default=10000)
    parser.add_argument("--performance-rows", type=int, default=10000)
    args = parser.parse_args()

    base = args.dataset_dir / "spacev100m_base.i8bin"
    queries = args.dataset_dir / "query.i8bin"
    truth = args.dataset_dir / "msspacev-gt-100M"
    base_rows, base_dim = header(base)
    query_rows, query_dim = header(queries)
    gt_rows, gt_topk = header(truth)
    if (base_rows, base_dim) != (100000000, 100):
        raise ValueError(f"expected SPACEV100M 100000000x100 base, got {base_rows}x{base_dim}")
    if query_dim != base_dim or gt_rows != query_rows:
        raise ValueError("base/query/ground-truth headers disagree")
    if args.recall_rows <= 0 or args.performance_rows <= 0:
        raise ValueError("split sizes must be positive")
    insert_start = args.recall_rows + args.performance_rows
    if insert_start >= query_rows:
        raise ValueError("query file has no rows left for inserts")

    check_size(base, HEADER.size + base_rows * base_dim)
    check_size(queries, HEADER.size + query_rows * query_dim)
    check_size(truth, HEADER.size + 2 * gt_rows * gt_topk * 4)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_rows(queries, args.output_dir / "recall_query.i8bin",
               0, args.recall_rows, query_dim)
    write_groundtruth(truth, args.output_dir / "recall_groundtruth.bin",
                      gt_topk, args.recall_rows)
    write_rows(queries, args.output_dir / "performance_query.i8bin",
               args.recall_rows, args.performance_rows, query_dim)
    write_rows(queries, args.output_dir / "insert.i8bin",
               insert_start, query_rows - insert_start, query_dim)
    print(f"prepared recall={args.recall_rows}, performance={args.performance_rows}, "
          f"insert={query_rows - insert_start}, dim={query_dim}, gt_topk={gt_topk}")


if __name__ == "__main__":
    main()
