# Contribution Ablation Experiments

This directory reorganizes the SIFT100M evaluation into the three paper
contributions:

| Profile | Enabled mechanisms | Purpose |
| --- | --- | --- |
| `00_baseline` | exact disaggregated search/update | baseline |
| `01_gpu_rdma_pipeline` | Contribution 1 | GPU-RDMA batched remote graph traversal |
| `02_gpu_rdma_pipeline_aldi` | Contribution 1 + 2 | add communication-bounded ALDI updates |
| `03_gpu_rdma_pipeline_aldi_vcpi` | Contribution 1 + 2 + 3 | add VCPI/RaBitQ remote-vector access reduction |

VCPI means "versioned compressed proxy index". In the current implementation it
maps to the compute-side RaBitQ proxy path. The default profile uses
`RABITQ_MODE=exact_safe` because that is the conservative paper-safe setting.
Use `RABITQ_MODE=cpu_gate` when you want to reproduce the faster heuristic
reports under `evaluation/sift100m/reports`.

## Prerequisites

The scripts reuse the SIFT100M harness under `evaluation/sift100m`. The default
index build already writes RaBitQ and anchor sidecars:

```bash
./evaluation/sift100m/build_sift100m_index.sh
```

If an older index is missing the ALDI anchor sidecar, generate it once:

```bash
./build/vamana_anchor_sidecar_builder \
  --index-prefix /data/xjs/index/dvstor_sift100m/index/sift100m_R48_bw200_balanced \
  --anchors-per-shard 4096
```

## Run One Profile

```bash
./experiment/start_all_memory_nodes.sh 00_baseline
./experiment/run_breakdown.sh 00_baseline
./experiment/stop_memory_nodes.sh
```

The same pattern works for the other profiles.

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
WORKLOAD=query WARMUP_SECONDS=10 MEASURE_SECONDS=60 ./experiment/run_breakdown.sh 01_gpu_rdma_pipeline
WORKLOAD=query WARMUP_SECONDS=10 MEASURE_SECONDS=60 ./experiment/run_breakdown.sh 03_gpu_rdma_pipeline_aldi_vcpi
```

## Summarize Latest Reports

```bash
./experiment/summarize_reports.py
```

The summary prints query/write throughput, recall, p50 latencies, vector RDMA
traffic, VCPI filtering counters, and ALDI audit failure rate.

