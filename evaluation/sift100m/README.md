# SIFT100M DVSTOR Test

This directory contains a 5-storage-node SIFT100M setup.

## Build BFS-sharded index

```bash
./evaluation/sift100m/build_bfs_index.sh
```

Defaults:

- data: `/data/xjs/datasets/sift100m/learn.100M.u8bin`
- query: `/data/xjs/datasets/sift100m/query.public.10K.u8bin`
- output prefix: `/data/xjs/index/dvstor/sift100m/sift100m`
- shards: 5
- partition strategy: `bfs`
- graph quality defaults: `R=48`, `BEAM_WIDTH=320`
- build threads: `32` by default; override with `BUILD_THREADS=<n>` after calibrating throughput
- offline reverse updates: `OFFLINE_REVERSE_MODE=immediate` by default
- post-build brute-force sanity check: skipped by default with `SKIP_SANITY_CHECK=true`
- offline build GPU: `BUILD_USE_GPU=false` by default because the current builder uses small synchronized GPU batches
- memory: `MN_MEMORY=32` GB per storage node and `CN_MEMORY=24` GB on the compute node by default
- caches: `NEIGHBOR_CACHE_MB=2048`, `GPU_RABITQ_CACHE_MB=8192`, `STORAGE_OWNER_CACHE_MB=1024`
- neighbor cache invalidation: `NEIGHBOR_CACHE_INVALIDATION_MS=100`, `NEIGHBOR_CACHE_INVALIDATION_INSERTS=1024`

## Start storage nodes

```bash
./evaluation/sift100m/start_all_memory_nodes.sh start
./evaluation/sift100m/start_all_memory_nodes.sh status
./evaluation/sift100m/start_all_memory_nodes.sh stop
```

Use `HOST=<ip>` when the compute node should connect through a non-local address.

## Run mixed read/write benchmark

```bash
./evaluation/sift100m/run_mixed_benchmark.sh
```

Defaults are `READ_RATIO=0.5`, `MIXED_MODE=probability`, `CLIENT_THREADS=16`, warmup `30s`, measure `120s`.

`MIXED_MODE=probability` samples `READ_RATIO` for every operation in every client thread. `MIXED_MODE=fixed_threads` splits client threads by `READ_RATIO`; for example `READ_RATIO=0.75 CLIENT_THREADS=16` assigns 12 query threads and 4 insert threads.

Mixed inserts currently use deterministic synthetic vectors derived from the insert id, not vectors read from SIFT100M. Queries use `QUERY_FILE`, which defaults to `/data/xjs/datasets/sift100m/query.public.10K.u8bin`.

Enable a before-performance recall check with:

```bash
ENABLE_RECALL=true RECALL_QUERIES=1000 ./evaluation/sift100m/run_mixed_benchmark.sh
```

The recall check runs before warmup/measure, uses `GROUNDTRUTH_FILE` from `sift100m_common.sh`, and reports `recall@K` in the JSON report and terminal summary. Set `MIN_RECALL=<value>` to fail the run when recall is below a threshold.
