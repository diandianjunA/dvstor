# Contribution Ablation Experiments

This directory reorganizes the SIFT100M evaluation into the revised three paper
contributions. The current split keeps RaBitQ with the GPU query pipeline and
isolates adaptive RDMA scheduling as the third contribution.

| Profile | Enabled mechanisms | Purpose |
| --- | --- | --- |
| `00_baseline` | storage-owner exact search/update substrate | baseline, no paper contribution |
| `01_rabitq_expension_aware` | Contribution 1 | RaBitQ-aware expansion and GPU query pipeline |
| `02_rabitq_expension_aware_two_stage_aldi` | Contribution 1 + 2 | add local-stitch two-stage ALDI and METIS/locality-oriented placement |
| `03_rabitq_expension_aware_two_stage_aldi_rdma` | Contribution 1 + 2 + 3 | add adaptive multi-QP RDMA scheduling |
| `04_gpu_persistent_gpunetio` | GPU-centric redesign | persistent GPU scheduling, tiered graph cache, GPUNetIO, and MVCC delta |

Contribution 1 includes the compute-side RaBitQ proxy gate because the current
measurements show it behaves as part of the query execution pipeline: it trades
CPU gate work for fewer full-vector RDMA reads. RaBitQ now has a single
CPU-gate execution path in these experiments.

Contribution 3 is the adaptive RDMA scheduler. It should be evaluated after
Contribution 2 because ALDI + METIS increases edge locality, which can reduce
the number of active memory nodes per vector batch. The adaptive scheduler then
recovers parallelism through multiple QPs per hot memory node and chained READ
WRs.

## Prerequisites

The scripts include the SIFT100M harness locally. The default index build
already writes RaBitQ and anchor sidecars:

```bash
./experiment/build_sift100m_index.sh
```

The GPU-persistent profile needs its GPU image and graph-page sidecars:

```bash
./experiment/build_sift100m_index.sh 04_gpu_persistent_gpunetio
```

If an older index is missing the ALDI anchor sidecar, generate it once for the
actual index prefix being tested. Profiles `02_*` and `03_*` use the METIS
prefix by default:

```bash
./build/vamana_anchor_sidecar_builder \
  --index-prefix /data/xjs/index/dvstor_sift100m/index/sift100m_R48_bw200_metis \
  --anchors-per-shard 4096
```

## Run One Profile

```bash
./experiment/start_all_memory_nodes.sh 00_baseline
./experiment/run_breakdown.sh 00_baseline
./experiment/stop_memory_nodes.sh
```

The same pattern works for the other profiles.

```bash
./experiment/start_all_memory_nodes.sh 04_gpu_persistent_gpunetio
./experiment/run_breakdown.sh 04_gpu_persistent_gpunetio
./experiment/stop_memory_nodes.sh
```

## Run The Full Ablation

```bash
./experiment/run_ablation.sh
```

Reports are written to `experiment/reports/<profile>/`. The generated
`service_*.ini` file in each report directory is the exact runtime config used
for the run.

## Short Query-Only Runs

To isolate Contribution 1 and Contribution 3 on query performance:

```bash
WORKLOAD=query WARMUP_SECONDS=10 MEASURE_SECONDS=60 ./experiment/run_breakdown.sh 01_rabitq_expension_aware
WORKLOAD=query WARMUP_SECONDS=10 MEASURE_SECONDS=60 ./experiment/run_breakdown.sh 02_rabitq_expension_aware_two_stage_aldi
WORKLOAD=query WARMUP_SECONDS=10 MEASURE_SECONDS=60 ./experiment/run_breakdown.sh 03_rabitq_expension_aware_two_stage_aldi_rdma
```

## Summarize Latest Reports

```bash
./experiment/summarize_reports.py
```

The summary prints query/write throughput, recall, p50 latencies, vector RDMA
traffic, RaBitQ drop ratio, RDMA active-node/QP indicators, and ALDI repair
progress.
