#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import numpy as np


def read_u8bin(path: Path) -> np.memmap:
    header = np.fromfile(path, dtype=np.uint32, count=2)
    if header.size != 2:
        raise ValueError(f"invalid u8bin header: {path}")
    count, dim = map(int, header)
    expected = 8 + count * dim
    if path.stat().st_size != expected:
        raise ValueError(f"u8bin size mismatch: {path}")
    return np.memmap(path, dtype=np.uint8, mode="r", offset=8, shape=(count, dim))


def rotation_signs(code_bits: int, round_index: int) -> np.ndarray:
    values = (np.arange(code_bits, dtype=np.uint32) + np.uint32(0x9E3779B9) +
              np.uint32((round_index * 0x85EBCA6B) & 0xFFFFFFFF))
    values ^= values >> np.uint32(16)
    values *= np.uint32(0x7FEB352D)
    values ^= values >> np.uint32(15)
    return np.where((values & np.uint32(1)) != 0, 1.0, -1.0).astype(np.float32)


def rotate(vectors: np.ndarray, centroid: np.ndarray, code_bits: int,
           sign_rounds: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    rows, dim = vectors.shape
    rotated = np.zeros((rows, code_bits), dtype=np.float32)
    rotated[:, :dim] = vectors.astype(np.float32) - centroid
    norm2 = np.einsum("ij,ij->i", rotated[:, :dim], rotated[:, :dim])
    scale = np.float32(1.0 / np.sqrt(code_bits))
    for signs in sign_rounds:
        rotated *= signs
        width = 1
        while width < code_bits:
            groups = rotated.reshape(rows, -1, width * 2)
            lhs = groups[:, :, :width].copy()
            rhs = groups[:, :, width:].copy()
            groups[:, :, :width] = lhs + rhs
            groups[:, :, width:] = lhs - rhs
            width *= 2
        rotated *= scale
    return rotated, norm2


def top_indices(values: np.ndarray, count: int) -> np.ndarray:
    count = min(count, values.size)
    if count == values.size:
        return np.argsort(values)
    selected = np.argpartition(values, count - 1)[:count]
    return selected[np.argsort(values[selected])]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit DVSTOR RaBitQ candidate coverage on sampled vectors.")
    parser.add_argument("--metadata", type=Path, required=True,
                        help="Index .meta.json containing rabitq_centroid.")
    parser.add_argument("--vectors", type=Path, required=True, help="Base vectors in u8bin format.")
    parser.add_argument("--queries", type=Path, required=True, help="Queries in u8bin format.")
    parser.add_argument("--sample-vectors", type=int, default=100000)
    parser.add_argument("--sample-queries", type=int, default=100)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--gate-widths", type=int, nargs="+", default=[32, 48, 64])
    parser.add_argument("--rotation-rounds", type=int, default=1, choices=range(1, 5),
                        help="1 matches the runtime; 2-4 are stronger-rotation what-if audits.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    metadata = json.loads(args.metadata.read_text())
    vectors = read_u8bin(args.vectors)
    queries = read_u8bin(args.queries)
    if vectors.shape[1] != queries.shape[1]:
        raise ValueError("vector/query dimensions differ")
    dim = vectors.shape[1]
    centroid = np.asarray(metadata["rabitq_centroid"], dtype=np.float32)
    if centroid.shape != (dim,):
        raise ValueError("metadata centroid dimension mismatch")
    code_bits = max(8, 1 << (dim - 1).bit_length())
    if metadata.get("rabitq_code_bits", code_bits) != code_bits:
        raise ValueError("metadata code width differs from runtime layout")

    generator = np.random.default_rng(args.seed)
    vector_count = min(args.sample_vectors, vectors.shape[0])
    query_count = min(args.sample_queries, queries.shape[0])
    vector_ids = generator.choice(vectors.shape[0], size=vector_count, replace=False)
    query_ids = generator.choice(queries.shape[0], size=query_count, replace=False)
    sampled_vectors = np.asarray(vectors[vector_ids])
    sampled_queries = np.asarray(queries[query_ids])

    sign_rounds = [rotation_signs(code_bits, round_index)
                   for round_index in range(args.rotation_rounds)]
    rotated_vectors, vector_norm2 = rotate(sampled_vectors, centroid, code_bits, sign_rounds)
    binary_codes = np.where(rotated_vectors > 0, 1.0, -1.0).astype(np.float32)
    vector_norms = np.sqrt(vector_norm2)
    correction = np.ones(vector_count, dtype=np.float32)
    nonzero = vector_norms > np.float32(1e-15)
    correction[nonzero] = np.maximum(
        np.einsum("ij,ij->i", binary_codes[nonzero], rotated_vectors[nonzero]) /
        (vector_norms[nonzero] * np.float32(np.sqrt(code_bits))),
        np.float32(1e-15))

    recalls = {width: [] for width in args.gate_widths}
    quantized_topk_recall = []
    relative_errors = []
    sampled_vectors_f32 = sampled_vectors.astype(np.float32)
    rotated_queries, query_norm2 = rotate(sampled_queries, centroid, code_bits, sign_rounds)
    for query_index in range(query_count):
        query = sampled_queries[query_index].astype(np.float32)
        difference = sampled_vectors_f32 - query
        exact = np.einsum("ij,ij->i", difference, difference)
        signed_dot = binary_codes @ rotated_queries[query_index]
        inner = (vector_norms * signed_dot /
                 (np.float32(np.sqrt(code_bits)) *
                  np.maximum(correction, np.float32(1e-6))))
        approximate = np.maximum(query_norm2[query_index] + vector_norm2 - 2.0 * inner, 0.0)
        ground_truth = set(top_indices(exact, args.k).tolist())
        quantized = set(top_indices(approximate, args.k).tolist())
        quantized_topk_recall.append(len(ground_truth & quantized) / args.k)
        for width in args.gate_widths:
            candidates = set(top_indices(approximate, width).tolist())
            recalls[width].append(len(ground_truth & candidates) / args.k)
        scale = np.maximum(exact, 1.0)
        relative_errors.append(float(np.median(np.abs(approximate - exact) / scale)))

    report = {
        "vectors": str(args.vectors),
        "queries": str(args.queries),
        "sample_vectors": vector_count,
        "sample_queries": query_count,
        "dim": dim,
        "code_bits": code_bits,
        "k": args.k,
        "rotation": f"fixed_sign_hadamard_{args.rotation_rounds}_rounds",
        "quantized_topk_recall_mean": float(np.mean(quantized_topk_recall)),
        "median_relative_distance_error_mean": float(np.mean(relative_errors)),
        "gate_candidate_recall": {
            str(width): {
                "mean": float(np.mean(values)),
                "p05": float(np.percentile(values, 5)),
                "min": float(np.min(values)),
            }
            for width, values in recalls.items()
        },
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
