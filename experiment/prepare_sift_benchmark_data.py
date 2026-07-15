#!/usr/bin/env python3
"""Extract query/insert ranges from a bvecs dataset into u8bin files."""

import argparse
import os
import struct
import sys
from pathlib import Path

def require_numpy():
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError(
            "numpy is required only when benchmark data must be generated"
        ) from exc
    return np


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


def validate_range(
    label: str, start: int, end: int, available_rows: int | None = None
) -> None:
    if start < 0 or end <= start:
        raise RuntimeError(
            f"invalid {label} range [{start}, {end}); expected 0 <= start < end"
        )
    if available_rows is not None and end > available_rows:
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
    data,
    numpy,
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
                if not numpy.all(dims == dim):
                    bad = done + int(numpy.nonzero(dims != dim)[0][0])
                    raise RuntimeError(f"inconsistent bvecs dim at source row {bad}")
                vectors = numpy.ascontiguousarray(data["vec"][done:chunk_end])
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
            "Extract performance-query and insert ranges from "
            "SIFT1B bigann_base.bvecs. All ranges are half-open [start,end)."
        )
    )
    parser.add_argument(
        "--source", default="/data/xjs/datasets/sift1b/bigann_base.bvecs"
    )
    parser.add_argument("--query-output", required=True)
    parser.add_argument("--query-start", type=int, default=100_000_000)
    parser.add_argument("--query-end", type=int, default=105_000_000)
    parser.add_argument("--insert-output", required=True)
    parser.add_argument("--insert-start", type=int, default=105_000_000)
    parser.add_argument("--insert-end", type=int, default=107_000_000)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--chunk-rows", type=int, default=1_000_000)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.chunk_rows <= 0:
        raise RuntimeError("--chunk-rows must be > 0")
    if args.dim <= 0:
        raise RuntimeError("--dim must be > 0")

    source = Path(args.source)
    query_output = Path(args.query_output)
    insert_output = Path(args.insert_output)
    if query_output.resolve() == insert_output.resolve():
        raise RuntimeError("query and insert outputs must be different files")

    validate_range("query", args.query_start, args.query_end)
    validate_range("insert", args.insert_start, args.insert_end)
    if max(args.query_start, args.insert_start) < min(
        args.query_end, args.insert_end
    ):
        raise RuntimeError(
            "query and insert source ranges must not overlap: "
            f"query=[{args.query_start},{args.query_end}) "
            f"insert=[{args.insert_start},{args.insert_end})"
        )
    outputs = (
        ("query", query_output, args.query_start, args.query_end),
        ("insert", insert_output, args.insert_start, args.insert_end),
    )
    ready = {
        label: existing_u8bin_matches(path, end - start, args.dim)
        for label, path, start, end in outputs
    }
    if not args.overwrite:
        invalid = [
            str(path)
            for label, path, _, _ in outputs
            if path.exists() and not ready[label]
        ]
        if invalid:
            raise RuntimeError(
                "existing output has an unexpected header or size: "
                + ", ".join(invalid)
            )
        if all(ready.values()):
            for label, path, start, end in outputs:
                print(
                    f"[benchmark-data] exists: {path} "
                    f"source_rows=[{start},{end}) rows={end - start} dim={args.dim}"
                )
            print("[benchmark-data] all outputs validated; source bvecs is not required")
            return

    if not source.is_file():
        pending = [
            label
            for label, _, _, _ in outputs
            if args.overwrite or not ready[label]
        ]
        pending_text = ", ".join(pending)
        raise RuntimeError(
            f"source bvecs is required to generate {pending_text} data but was not found: "
            f"{source}. Copy the pre-generated u8bin files to the configured output paths "
            "or provide --source on a data-preparation node"
        )

    dim, source_rows = infer_bvecs(source)
    if dim != args.dim:
        raise RuntimeError(f"source dim {dim} does not match expected dim {args.dim}")
    validate_range("query", args.query_start, args.query_end, source_rows)
    validate_range("insert", args.insert_start, args.insert_end, source_rows)

    np = require_numpy()
    dtype = np.dtype([("dim", "<i4"), ("vec", "u1", (dim,))])
    data = np.memmap(source, dtype=dtype, mode="r", shape=(source_rows,))
    extract_bvecs_range(
        source,
        data,
        np,
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
        np,
        insert_output,
        args.insert_start,
        args.insert_end,
        dim,
        args.chunk_rows,
        args.overwrite,
    )


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as error:
        print(f"prepare_sift_benchmark_data.py: error: {error}", file=sys.stderr)
        raise SystemExit(1) from None
