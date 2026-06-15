# SIFT100M Evaluation Scripts

This directory contains a self-contained SIFT100M test harness using the SIFT1B source files under `/data/xjs/datasets/sift1b`, limited to the first 100M base vectors and `gnd/idx_100M.ivecs`.
It uses the exact-only, dtype-aware index layout: SIFT base/query vectors are converted to `.u8bin`, and recall uses raw `uint8` query input through `search_raw()`.

## Files

- `convert_sift100m.py`: converts `bigann_base.bvecs`, `bigann_query.bvecs`, and `gnd/idx_100M.ivecs` to dvstor `.u8bin` / `.bin`.
- `build_sift100m_index.sh`: builds a 5-shard index. Default partition strategy is `bfs`; set `PARTITION_STRATEGY=metis` or `balanced` to change it.
- `start_memory_node_1.sh` ... `start_memory_node_5.sh`: per-node launchers.
- `start_all_memory_nodes.sh`: starts all five memory nodes for a profile.
- `stop_memory_nodes.sh`: stops locally started memory nodes.
- `configs/*.ini`: compute-side benchmark configs for baseline and optimization profiles.
- `run_breakdown.sh`: mixed/query/insert breakdown benchmark driver.
- `run_recall.sh`: recall-only driver using the benchmark tool's recall phase.

## Default Flow

```bash
./evaluation/sift100m/build_sift100m_index.sh
./evaluation/sift100m/start_all_memory_nodes.sh baseline
./evaluation/sift100m/run_recall.sh baseline
./evaluation/sift100m/run_breakdown.sh baseline
./evaluation/sift100m/stop_memory_nodes.sh
```

Optimization profiles:

```bash
./evaluation/sift100m/start_all_memory_nodes.sh gpudirect_rdma
./evaluation/sift100m/run_breakdown.sh gpudirect_rdma

./evaluation/sift100m/start_all_memory_nodes.sh gpudirect_rdma_storage_owner
./evaluation/sift100m/run_breakdown.sh gpudirect_rdma_storage_owner
```

## Anchor-Local Direct Insert (ALDI)

The ALDI profile keeps the existing GPUDirect/RaBitQ query path and changes only
storage-owner update routing and candidate discovery. Existing indexes do not
need to be rebuilt. The existing shard files or original base dataset must be
available while generating the bounded anchor sidecar once:

```bash
./build/vamana_anchor_sidecar_builder \
  --index-prefix /data/xjs/index/dvstor_sift100m/index/sift100m_R48_bw200_balanced \
  --anchors-per-shard 4096
```

Then restart the memory nodes and run the dedicated profile:

```bash
./evaluation/sift100m/start_all_memory_nodes.sh gpudirect_rdma_storage_owner_rabitq_aldi
./evaluation/sift100m/run_breakdown.sh gpudirect_rdma_storage_owner_rabitq_aldi
```

To compare ALDI with different shard placements, build both partitioned indexes:

```bash
PARTITION_STRATEGY=bfs ./evaluation/sift100m/build_sift100m_index.sh
PARTITION_STRATEGY=metis ./evaluation/sift100m/build_sift100m_index.sh
```

The corresponding profiles select
`sift100m_R${R}_bw${BUILD_BEAM}_bfs` and
`sift100m_R${R}_bw${BUILD_BEAM}_metis` respectively:

```bash
./evaluation/sift100m/start_all_memory_nodes.sh gpudirect_rdma_storage_owner_rabitq_aldi_bfs
./evaluation/sift100m/run_breakdown.sh gpudirect_rdma_storage_owner_rabitq_aldi_bfs
./evaluation/sift100m/stop_memory_nodes.sh

./evaluation/sift100m/start_all_memory_nodes.sh gpudirect_rdma_storage_owner_rabitq_aldi_metis
./evaluation/sift100m/run_breakdown.sh gpudirect_rdma_storage_owner_rabitq_aldi_metis
./evaluation/sift100m/stop_memory_nodes.sh
```

Use `BFS_INDEX_PREFIX=/path/to/index` or
`METIS_INDEX_PREFIX=/path/to/index` to test custom repartitioner outputs.

If the sidecar is absent or incompatible, the ALDI profile fails at startup
instead of silently falling back. The baseline
`gpudirect_rdma_storage_owner_rabitq` profile remains unchanged and uses the
default exact storage-owner update path.

## Common Overrides

```bash
HOSTS="mn1 mn2 mn3 mn4 mn5" BASE_PORT=1234 IB_DEVICE=mlx5_0 ./evaluation/sift100m/start_all_memory_nodes.sh baseline
HOSTS="mn1 mn2 mn3 mn4 mn5" ./evaluation/sift100m/run_breakdown.sh baseline
MAX_VECTORS=100000000 MAX_QUERIES=10000 GROUNDTRUTH_LABEL=100M ./evaluation/sift100m/build_sift100m_index.sh
PARTITION_STRATEGY=metis ./evaluation/sift100m/build_sift100m_index.sh
WORKLOAD=query WARMUP_SECONDS=10 MEASURE_SECONDS=60 ./evaluation/sift100m/run_breakdown.sh gpudirect_rdma
```

When `MAX_VECTORS` is not the 100M, use the matching SIFT ground truth label, for example `GROUNDTRUTH_LABEL=100M` for `MAX_VECTORS=100000000`.


## Build Performance Notes

The default SIFT100M build uses the new raw-dtype offline GPU path with `GPU_MEMORY_GB=18`. For SIFT100M `uint8`, the builder keeps the raw base vectors resident on one GPU and sends candidate IDs to distance kernels, avoiding full float expansion and per-batch float candidate staging. Set `NO_GPU=1` to force the CPU typed path.

The old deferred reverse-update mode has been removed. Reverse edges are maintained with bounded immediate updates, so the builder no longer allocates the former `MAX_VECTORS * R * 4` deferred edge buffer. For quick pipeline tests, use `MAX_VECTORS=1000000 GROUNDTRUTH_LABEL=1M`; full SIFT100M exact Vamana construction is still a long offline job.

## Memory Defaults

The scripts now estimate memory-node memory from the index shape:

```text
node_bytes ~= 16 + dim * component_size + R * 8
mn_memory ~= ceil(MAX_VECTORS / SHARDS * node_bytes * 1.2) + 4GB
```

For full SIFT100M with `dim=128`, `R=64`, `uint8`, and 5 shards, this is about 152GB per memory node, not 200GB. Override with `MN_MEMORY_GB=...` if your partition or insert workload needs more slack.

Compute-node memory defaults to 16GB.

Ground truth conversion defaults to top-10, which is enough for recall@10 and avoids carrying the full top-1000 unless explicitly requested. Set `GROUNDTRUTH_TOPK=100` or `GROUNDTRUTH_TOPK=1000` if you need broader recall metrics. Recall scripts default to 1000 query vectors for quick testing; set `RECALL_QUERIES=10000` for the standard full query set.
