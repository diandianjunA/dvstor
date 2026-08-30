#!/usr/bin/env python3
"""Split native Deep100M float queries/GT into disjoint benchmark inputs."""

import argparse
import os
import pathlib
import struct

HEADER = struct.Struct("<II")
CHUNK_BYTES = 8 << 20
COMPONENT_BYTES = 4


def read_header(path: pathlib.Path) -> tuple[int, int]:
    with path.open("rb") as stream:
        raw = stream.read(HEADER.size)
    if len(raw) != HEADER.size:
        raise ValueError(f"short header: {path}")
    return HEADER.unpack(raw)


def require_size(path: pathlib.Path, expected: int) -> None:
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


def write_vectors(source_path: pathlib.Path, target_path: pathlib.Path,
                  start: int, count: int, dim: int) -> None:
    row_bytes = dim * COMPONENT_BYTES
    temporary = target_path.with_suffix(target_path.suffix + f".tmp.{os.getpid()}")
    with source_path.open("rb") as source, temporary.open("wb") as target:
        target.write(HEADER.pack(count, dim))
        copy_bytes(source, target, HEADER.size + start * row_bytes,
                   count * row_bytes)
    os.replace(temporary, target_path)


def write_groundtruth(source_path: pathlib.Path, target_path: pathlib.Path,
                      total_rows: int, topk: int, count: int) -> None:
    row_bytes = topk * 4
    ids_offset = HEADER.size
    distances_offset = ids_offset + total_rows * row_bytes
    temporary = target_path.with_suffix(target_path.suffix + f".tmp.{os.getpid()}")
    with source_path.open("rb") as source, temporary.open("wb") as target:
        target.write(HEADER.pack(count, topk))
        copy_bytes(source, target, ids_offset, count * row_bytes)
        copy_bytes(source, target, distances_offset, count * row_bytes)
    os.replace(temporary, target_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=pathlib.Path, required=True)
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    parser.add_argument("--recall-rows", type=int, default=3334)
    parser.add_argument("--performance-rows", type=int, default=3333)
    args = parser.parse_args()

    base = args.dataset_dir / "100M.fbin"
    queries = args.dataset_dir / "queries.fbin"
    truth = args.dataset_dir / "100M_gt.bin"
    base_rows, base_dim = read_header(base)
    query_rows, query_dim = read_header(queries)
    gt_rows, gt_topk = read_header(truth)
    if (base_rows, base_dim) != (100000000, 96):
        raise ValueError(f"expected Deep100M 100000000x96 base, got {base_rows}x{base_dim}")
    if (query_rows, query_dim) != (10000, 96) or gt_rows != query_rows:
        raise ValueError("base/query/ground-truth headers disagree")
    insert_start = args.recall_rows + args.performance_rows
    if args.recall_rows <= 0 or args.performance_rows <= 0 or insert_start >= query_rows:
        raise ValueError("invalid recall/performance/insert split")

    require_size(base, HEADER.size + base_rows * base_dim * COMPONENT_BYTES)
    require_size(queries, HEADER.size + query_rows * query_dim * COMPONENT_BYTES)
    require_size(truth, HEADER.size + 2 * gt_rows * gt_topk * 4)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_vectors(queries, args.output_dir / "recall_query.fbin",
                  0, args.recall_rows, query_dim)
    write_groundtruth(truth, args.output_dir / "recall_groundtruth.bin",
                      gt_rows, gt_topk, args.recall_rows)
    write_vectors(queries, args.output_dir / "performance_query.fbin",
                  args.recall_rows, args.performance_rows, query_dim)
    write_vectors(queries, args.output_dir / "insert.fbin",
                  insert_start, query_rows - insert_start, query_dim)
    print(f"prepared recall={args.recall_rows}, performance={args.performance_rows}, "
          f"insert={query_rows - insert_start}, dim={query_dim}, gt_topk={gt_topk}")


if __name__ == "__main__":
    main()
