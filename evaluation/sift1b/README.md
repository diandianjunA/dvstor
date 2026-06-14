# SIFT1B Evaluation Scripts

This directory contains a self-contained SIFT1B test harness for `/data/xjs/datasets/sift1b`.
It uses the exact-only, dtype-aware index layout: SIFT base/query vectors are converted to `.u8bin`, and recall uses raw `uint8` query input through `search_raw()`.

## Files

- `convert_sift1b.py`: converts `bigann_base.bvecs`, `bigann_query.bvecs`, and `gnd/idx_*.ivecs` to dvstor `.u8bin` / `.bin`.
- `build_sift1b_index.sh`: builds a 5-shard index. Default partition strategy is `bfs`; set `PARTITION_STRATEGY=metis` or `balanced` to change it.
- `start_memory_node_1.sh` ... `start_memory_node_5.sh`: per-node launchers.
- `start_all_memory_nodes.sh`: starts all five memory nodes for a profile.
- `stop_memory_nodes.sh`: stops locally started memory nodes.
- `configs/*.ini`: compute-side benchmark configs for baseline and optimization profiles.
- `run_breakdown.sh`: mixed/query/insert breakdown benchmark driver.
- `run_recall.sh`: recall-only driver using the benchmark tool's recall phase.

## Default Flow

```bash
./evaluation/sift1b/build_sift1b_index.sh
./evaluation/sift1b/start_all_memory_nodes.sh baseline
./evaluation/sift1b/run_recall.sh baseline
./evaluation/sift1b/run_breakdown.sh baseline
./evaluation/sift1b/stop_memory_nodes.sh
```

Optimization profiles:

```bash
./evaluation/sift1b/start_all_memory_nodes.sh gpudirect_rdma
./evaluation/sift1b/run_breakdown.sh gpudirect_rdma

./evaluation/sift1b/start_all_memory_nodes.sh gpudirect_rdma_storage_owner
./evaluation/sift1b/run_breakdown.sh gpudirect_rdma_storage_owner
```

QIR Direct Insert profiles:

```bash
./evaluation/sift1b/run_breakdown.sh gpudirect_rdma_storage_owner_rabitq
./evaluation/sift1b/run_breakdown.sh gpudirect_rdma_storage_owner_rabitq_qir_search_only
./evaluation/sift1b/run_breakdown.sh gpudirect_rdma_storage_owner_rabitq_qir_prune
./evaluation/sift1b/run_breakdown.sh gpudirect_rdma_storage_owner_rabitq_qir
```

These profiles isolate quantized search, quantized RobustPrune, and bounded
asynchronous repair. The full profile also enables exact shadow audits and emits
QIR qcode, exact-read, repair-queue, and audit counters in breakdown JSON.

## Common Overrides

```bash
HOSTS="mn1 mn2 mn3 mn4 mn5" BASE_PORT=1234 IB_DEVICE=mlx5_0 ./evaluation/sift1b/start_all_memory_nodes.sh baseline
HOSTS="mn1 mn2 mn3 mn4 mn5" ./evaluation/sift1b/run_breakdown.sh baseline
MAX_VECTORS=100000000 MAX_QUERIES=10000 GROUNDTRUTH_LABEL=100M ./evaluation/sift1b/build_sift1b_index.sh
PARTITION_STRATEGY=metis ./evaluation/sift1b/build_sift1b_index.sh
WORKLOAD=query WARMUP_SECONDS=10 MEASURE_SECONDS=60 ./evaluation/sift1b/run_breakdown.sh gpudirect_rdma
```

When `MAX_VECTORS` is not the full 1B, use the matching SIFT ground truth label, for example `GROUNDTRUTH_LABEL=100M` for `MAX_VECTORS=100000000`.


## Memory Defaults

The scripts now estimate memory-node memory from the index shape:

```text
node_bytes ~= 16 + dim * component_size + R * 8
mn_memory ~= ceil(MAX_VECTORS / SHARDS * node_bytes * 1.2) + 4GB
```

For full SIFT1B with `dim=128`, `R=64`, `uint8`, and 5 shards, this is about 152GB per memory node, not 200GB. Override with `MN_MEMORY_GB=...` if your partition or insert workload needs more slack.

Compute-node memory defaults to 16GB.

Ground truth conversion defaults to top-100, which is enough for recall@10/recall@100 and avoids carrying the full top-1000 unless explicitly requested. Set `GROUNDTRUTH_TOPK=1000` if you need full SIFT1B ground truth rows. Recall scripts default to 1000 query vectors for quick testing; set `RECALL_QUERIES=10000` for the standard full query set.
