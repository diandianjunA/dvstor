#!/usr/bin/env python3
"""Extract disjoint query/insert ranges from a bvecs dataset into u8bin files."""

import argparse
import os
import struct
from pathlib import Path

try:
    import numpy as np
except ImportError as exc:
    raise SystemExit(
        "prepare_sift_benchmark_data.py requires numpy for practical SIFT1B extraction"
    ) from exc


def infer_bvecs(path: Path) -> tuple[int, int]:
    with path.open("rb") as stream:
        dim_bytes = stream.read(4)
    if len(dim_bytes) != 4:
        raise RuntimeError(f"empty bvecs file: {path}")

    dim = struct.unpack("<i", dim_bytes)[0]
    if dim <= 0:
        raise RuntimeError(f"invalid bvecs dim {dim}: {path}")

    size = path.stat().st_size
    record_bytes = 4 + dim
    if size % record_bytes != 0:
        raise RuntimeError(
            f"{path} size {size} is not divisible by bvecs record size {record_bytes}"
        )
    return dim, size // record_bytes


def validate_range(label: str, start: int, end: int, available_rows: int) -> None:
    if start < 0 or end <= start:
        raise RuntimeError(
            f"invalid {label} range [{start}, {end}); expected 0 <= start < end"
        )
    if end > available_rows:
        raise RuntimeError(
            f"{label} range [{start}, {end}) exceeds source row count {available_rows}"
        )


def existing_u8bin_matches(path: Path, rows: int, dim: int) -> bool:
    if not path.exists():
        return False
    expected_size = 8 + rows * dim
    if path.stat().st_size != expected_size:
        return False
    with path.open("rb") as stream:
        header = stream.read(8)
    return header == struct.pack("<II", rows, dim)


def extract_bvecs_range(
    source: Path,
    data: np.memmap,
    destination: Path,
    start: int,
    end: int,
    dim: int,
    chunk_rows: int,
    overwrite: bool,
) -> None:
    output_rows = end - start
    if existing_u8bin_matches(destination, output_rows, dim) and not overwrite:
        print(
            f"[benchmark-data] exists: {destination} "
            f"source_rows=[{start},{end}) rows={output_rows} dim={dim}"
        )
        return
    if destination.exists() and not overwrite:
        raise RuntimeError(
            f"existing output has an unexpected header or size: {destination}; "
            "pass --overwrite to replace it atomically"
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    print(
        f"[benchmark-data] extracting {source} rows [{start},{end}) -> "
        f"{destination} rows={output_rows} dim={dim}"
    )
    try:
        with temporary.open("wb") as output:
            output.write(struct.pack("<II", output_rows, dim))
            done = start
            while done < end:
                chunk_end = min(done + chunk_rows, end)
                dims = data["dim"][done:chunk_end]
                if not np.all(dims == dim):
                    bad = done + int(np.nonzero(dims != dim)[0][0])
                    raise RuntimeError(f"inconsistent bvecs dim at source row {bad}")
                vectors = np.ascontiguousarray(data["vec"][done:chunk_end])
                output.write(vectors.tobytes(order="C"))
                done = chunk_end
                print(
                    f"[benchmark-data]   source rows {done}/{end} "
                    f"({done - start}/{output_rows})",
                    flush=True,
                )
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise

    if not existing_u8bin_matches(destination, output_rows, dim):
        raise RuntimeError(f"generated output failed validation: {destination}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Extract non-overlapping performance-query and insert ranges from "
            "SIFT1B bigann_base.bvecs. All ranges are half-open [start,end)."
        )
    )
    parser.add_argument(
        "--source", default="/data/xjs/datasets/sift1b/bigann_base.bvecs"
    )
    parser.add_argument("--query-output", required=True)
    parser.add_argument("--query-start", type=int, default=100_000_000)
    parser.add_argument("--query-end", type=int, default=103_000_000)
    parser.add_argument("--insert-output", required=True)
    parser.add_argument("--insert-start", type=int, default=103_000_000)
    parser.add_argument("--insert-end", type=int, default=105_000_000)
    parser.add_argument("--chunk-rows", type=int, default=1_000_000)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.chunk_rows <= 0:
        raise RuntimeError("--chunk-rows must be > 0")

    source = Path(args.source)
    query_output = Path(args.query_output)
    insert_output = Path(args.insert_output)
    if query_output.resolve() == insert_output.resolve():
        raise RuntimeError("query and insert outputs must be different files")

    dim, source_rows = infer_bvecs(source)
    validate_range("query", args.query_start, args.query_end, source_rows)
    validate_range("insert", args.insert_start, args.insert_end, source_rows)
    if max(args.query_start, args.insert_start) < min(args.query_end, args.insert_end):
        raise RuntimeError(
            "query and insert source ranges overlap: "
            f"[{args.query_start}, {args.query_end}) vs "
            f"[{args.insert_start}, {args.insert_end})"
        )

    dtype = np.dtype([("dim", "<i4"), ("vec", "u1", (dim,))])
    data = np.memmap(source, dtype=dtype, mode="r", shape=(source_rows,))
    extract_bvecs_range(
        source,
        data,
        query_output,
        args.query_start,
        args.query_end,
        dim,
        args.chunk_rows,
        args.overwrite,
    )
    extract_bvecs_range(
        source,
        data,
        insert_output,
        args.insert_start,
        args.insert_end,
        dim,
        args.chunk_rows,
        args.overwrite,
    )


if __name__ == "__main__":
    main()
