# GPU-Direct Pipeline Motivation Experiments

This suite characterizes why disaggregated GPU graph search needs a GPU-direct
data path and coroutine-driven heterogeneous pipeline. It deliberately keeps
RaBitQ and dynamic-update optimizations disabled.

## Experimental Questions

1. `stage-breakdown`: Where does a naive single-query execution spend time?
2. `concurrency-sweep`: How much independent cross-query overlap is available?
3. `data-path`: What CPU/PCIe staging work is introduced by the host path?
4. `expansion-sweep`: How fine-grained is one query, and how much intra-query
   batching opportunity exists?
5. `timeline`: Optional Nsight Systems timeline for the serialized baseline.

These are motivation and opportunity-characterization experiments. The final
paper evaluation should separately report the complete system and an ablation
such as `naive -> GPUDirect -> coroutines -> K-way expansion -> prefetch`.

## Baseline Configuration

The strict baseline uses:

```text
workload=query
query workers=1
query coroutines=1
client threads=1
GPUDirect=false
RaBitQ=false
query batch size=1
expansion batch=1
```

All cases use the same SIFT100M exact Vamana index and search parameters. The
runtime INI and effective case settings are saved with every report.

## Running

Build or select the same balanced SIFT100M index used by the main evaluation,
then start the memory nodes once:

```bash
./motivation/gpu-direct-pipeline/start_memory_nodes.sh
```

Run individual experiments:

```bash
./motivation/gpu-direct-pipeline/run_stage_breakdown.sh
./motivation/gpu-direct-pipeline/run_concurrency_sweep.sh
./motivation/gpu-direct-pipeline/run_data_path.sh
./motivation/gpu-direct-pipeline/run_expansion_sweep.sh
```

Or run the complete suite:

```bash
./motivation/gpu-direct-pipeline/run_all.sh
```

Stop the memory nodes afterward:

```bash
./motivation/gpu-direct-pipeline/stop_memory_nodes.sh
```

Important cluster overrides are inherited from `evaluation/sift100m`, for
example:

```bash
HOSTS="mn1 mn2 mn3 mn4 mn5" \
IB_DEVICE=mlx5_0 GPU_DEVICE=0 \
INDEX_PREFIX=/data/index/sift100m_R48_bw200_balanced \
  ./motivation/gpu-direct-pipeline/start_memory_nodes.sh
```

Pass the same overrides to every run command. Short smoke tests can use:

```bash
WARMUP_SECONDS=2 MEASURE_SECONDS=5 RECALL_QUERIES=10 \
CONCURRENCY_VALUES="1 2" EXPANSION_VALUES="1 2" \
  ./motivation/gpu-direct-pipeline/run_all.sh
```

Set `ENABLE_DEVICE_SAMPLING=1` to save `nvidia-smi dmon` output beside each
report. Run the optional CUDA/CPU timeline with:

```bash
./motivation/gpu-direct-pipeline/run_timeline.sh
```

## Outputs

Results are written below `motivation/gpu-direct-pipeline/reports/`. Each case
contains:

- `service.ini`: exact runtime service configuration.
- `case.env`: effective independent variables.
- `report.json` and `report.txt`: benchmark output.
- `resource_usage.txt`: process CPU utilization and memory statistics from
  `/usr/bin/time -v`, when available.
- `gpu_dmon.txt`: optional device-utilization samples.
- `timeline.nsys-rep`: optional Nsight Systems trace.

Generate aggregate tables with:

```bash
./motivation/gpu-direct-pipeline/summarize.py
```

The summary includes QPS, latency, CPU/RDMA/GPU/transfer shares, per-query RDMA
operations and bytes, and Host-staging versus GPU-direct bytes.

## Paper Figures

- **Motivation Figure 1:** stacked stage breakdown from `stage-breakdown`.
- **Motivation Figure 2:** QPS and p95 latency versus resident queries from
  `concurrency-sweep`; add GPU utilization when sampling is enabled.
- **Motivation Figure 3:** Host-staging bytes, transfer share, and latency from
  `data-path`.
- **Motivation Figure 4:** RDMA operations/query and latency versus expansion
  batch from `expansion-sweep`.

Do not use total RDMA bytes across a timed run for cross-case conclusions.
Higher-throughput cases complete more queries. Use the per-query fields emitted
by `summarize.py`.
