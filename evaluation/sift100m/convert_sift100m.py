#!/usr/bin/env python3
import argparse
import os
import struct
from pathlib import Path

try:
    import numpy as np
except ImportError as exc:
    raise SystemExit("convert_sift100m.py requires numpy for practical SIFT100M conversion") from exc


def infer_bvecs(path: Path):
    with path.open('rb') as f:
        dim_bytes = f.read(4)
    if len(dim_bytes) != 4:
        raise RuntimeError(f"empty bvecs file: {path}")
    dim = struct.unpack('<i', dim_bytes)[0]
    if dim <= 0:
        raise RuntimeError(f"invalid bvecs dim {dim}: {path}")
    size = path.stat().st_size
    record = 4 + dim
    if size % record != 0:
        raise RuntimeError(f"{path} size {size} is not divisible by bvecs record size {record}")
    return dim, size // record


def infer_ivecs(path: Path):
    with path.open('rb') as f:
        dim_bytes = f.read(4)
    if len(dim_bytes) != 4:
        raise RuntimeError(f"empty ivecs file: {path}")
    topk = struct.unpack('<i', dim_bytes)[0]
    if topk <= 0:
        raise RuntimeError(f"invalid ivecs topk {topk}: {path}")
    size = path.stat().st_size
    record = 4 + topk * 4
    if size % record != 0:
        raise RuntimeError(f"{path} size {size} is not divisible by ivecs record size {record}")
    return topk, size // record


def convert_bvecs_to_u8bin(src: Path, dst: Path, max_rows: int, chunk_rows: int):
    dim, rows = infer_bvecs(src)
    out_rows = rows if max_rows <= 0 else min(rows, max_rows)
    if dst.exists():
        expected = 8 + out_rows * dim
        if dst.stat().st_size == expected:
            print(f"[convert] exists: {dst} rows={out_rows} dim={dim}")
            return
        raise RuntimeError(f"existing output has unexpected size: {dst}")

    print(f"[convert] bvecs -> u8bin: {src} -> {dst} rows={out_rows}/{rows} dim={dim}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    dtype = np.dtype([('dim', '<i4'), ('vec', 'u1', (dim,))])
    data = np.memmap(src, dtype=dtype, mode='r', shape=(rows,))
    with dst.open('wb') as out:
        out.write(struct.pack('<II', out_rows, dim))
        done = 0
        while done < out_rows:
            end = min(done + chunk_rows, out_rows)
            dims = data['dim'][done:end]
            if not np.all(dims == dim):
                bad = done + int(np.nonzero(dims != dim)[0][0])
                raise RuntimeError(f"inconsistent bvecs dim at row {bad}")
            out.write(np.ascontiguousarray(data['vec'][done:end]).tobytes(order='C'))
            done = end
            print(f"[convert]   rows {done}/{out_rows}", flush=True)


def convert_ivecs_to_bin(src: Path, dst: Path, max_rows: int, topk_limit: int, chunk_rows: int):
    topk, rows = infer_ivecs(src)
    out_rows = rows if max_rows <= 0 else min(rows, max_rows)
    out_topk = topk if topk_limit <= 0 else min(topk, topk_limit)
    if dst.exists():
        expected = 8 + out_rows * out_topk * 4
        if dst.stat().st_size == expected:
            print(f"[convert] exists: {dst} rows={out_rows} topk={out_topk}")
            return
        raise RuntimeError(f"existing output has unexpected size: {dst}")

    print(f"[convert] ivecs -> bin: {src} -> {dst} rows={out_rows}/{rows} topk={out_topk}/{topk}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    dtype = np.dtype([('dim', '<i4'), ('ids', '<i4', (topk,))])
    data = np.memmap(src, dtype=dtype, mode='r', shape=(rows,))
    with dst.open('wb') as out:
        out.write(struct.pack('<II', out_rows, out_topk))
        done = 0
        while done < out_rows:
            end = min(done + chunk_rows, out_rows)
            dims = data['dim'][done:end]
            if not np.all(dims == topk):
                bad = done + int(np.nonzero(dims != topk)[0][0])
                raise RuntimeError(f"inconsistent ivecs topk at row {bad}")
            ids = np.ascontiguousarray(data['ids'][done:end, :out_topk].astype('<u4', copy=False))
            out.write(ids.tobytes(order='C'))
            done = end
            print(f"[convert]   rows {done}/{out_rows}", flush=True)


def main():
    parser = argparse.ArgumentParser(description='Convert SIFT100M bvecs/ivecs files to dvstor .u8bin/.bin files.')
    parser.add_argument('--dataset-dir', default='/data/xjs/datasets/sift1b')
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--base-src')
    parser.add_argument('--query-src')
    parser.add_argument('--groundtruth-src')
    parser.add_argument('--groundtruth-label', default='100M')
    parser.add_argument('--max-base', type=int, default=1_000_000_000)
    parser.add_argument('--max-query', type=int, default=10_000)
    parser.add_argument('--topk', type=int, default=1000)
    parser.add_argument('--chunk-rows', type=int, default=1_000_000)
    args = parser.parse_args()

    dataset = Path(args.dataset_dir)
    out = Path(args.out_dir)
    base_src = Path(args.base_src) if args.base_src else dataset / 'bigann_base.bvecs'
    query_src = Path(args.query_src) if args.query_src else dataset / 'bigann_query.bvecs'
    gt_src = Path(args.groundtruth_src) if args.groundtruth_src else dataset / 'gnd' / f'idx_{args.groundtruth_label}.ivecs'

    base_suffix = '' if args.max_base in (0, 1_000_000_000) else f'_{args.max_base}'
    query_suffix = '' if args.max_query in (0, 10_000) else f'_{args.max_query}'
    convert_bvecs_to_u8bin(base_src, out / f'base{base_suffix}.u8bin', args.max_base, args.chunk_rows)
    convert_bvecs_to_u8bin(query_src, out / f'query{query_suffix}.u8bin', args.max_query, args.chunk_rows)
    convert_ivecs_to_bin(gt_src, out / f'groundtruth_{args.groundtruth_label}.bin', args.max_query, args.topk, args.chunk_rows)


if __name__ == '__main__':
    main()
