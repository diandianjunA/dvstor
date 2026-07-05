# DVSTOR: Disaggregated GPU Vamana Index

Implementation of a GPU-accelerated Vamana index for memory disaggregation.
This repository is used as a storage-compute disaggregated GPU vector-search baseline.

## Project Structure

```
dvstor/
├── src/                    # Core runtime source code
│   ├── common/             # Shared config, types, distance functions, utilities
│   ├── gpu/                # CUDA kernels and GPU resource management
│   ├── http/               # Service request types and worker schedulers
│   ├── io/                 # In-memory vector batch containers
│   ├── memory_node/        # Storage node: index storage, insert, peer RDMA/RPC
│   ├── rdma/               # RDMA operation wrappers (read/write/atomics)
│   ├── router/             # Multi-compute-node adaptive query routing
│   ├── service/            # Compute service orchestration (ComputeService)
│   └── vamana/             # Vamana graph index core implementation
├── tools/                  # Standalone binaries and scripts
│   ├── vamana_offline/     # Offline Vamana graph builder library
│   ├── breakdown_benchmark/# Performance breakdown benchmark tool
│   ├── vamana_offline_builder.cc   # Offline index builder
│   ├── vamana_metis_repartitioner.cc # METIS graph partitioning tool
│   └── run_recall_test.sh  # Recall evaluation helper
├── experiment/             # SIFT100M experiment harness, profiles, and reports
├── rdma-library/           # RDMA low-level library (QP management, memory registration)
├── thirdparty/             # Bundled third-party header libraries (nlohmann/json)
└── motivation/             # Historical motivation experiment outputs
```

## Setup

### C++ Libraries and Unix Packages

The following C++ libraries and Unix packages are required to compile the code.
Note that `ibverbs` (the RDMA library) is Linux-only. 
The code also compiles without InfiniBand network cards.

* [ibverbs](https://github.com/linux-rdma/rdma-core/tree/master)
* [boost](https://www.boost.org/doc/libs/1_83_0/doc/html/program_options.html) (to support `boost::program_options` for CLI parsing)
* pthreads (for multithreading)
* [oneTBB](https://github.com/oneapi-src/oneTBB) (for concurrent data structures)
* a C++ compiler that supports C++20 (we have used `g++-12`)
* cmake
* numactl
* vmtouch (to map index files into main memory)
* axel (a download accelerator for the datasets)

For instance, to install the requirements on Debian, run the following command:
```
apt-get -y install g++ libboost-all-dev libibverbs1 libibverbs-dev numactl cmake libtbb-dev git python3-venv vmtouch axel
```

For METIS-based graph partitioning support (optional), also install:
```
apt-get -y install libmetis-dev
```

### Cluster Nodes Configuration

Adjust the IP addresses of the cluster nodes accordingly in `rdma-library/library/utils.cc`.

### Compilation

After cloning the repository and installing the requirements, the code must be compiled on all cluster nodes:
```
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make
```

This produces these main binaries:
- `build/dvstor`: online compute node service
- `build/dvstor_memory_node`: memory-node-only service
- `build/vamana_offline_builder`: offline Vamana builder that exports DVSTOR shard files plus RaBitQ artifacts
- `build/vamana_metis_repartitioner`: repartition an existing offline-built index using METIS graph partitioning
- `build/dvstor_breakdown_benchmark`: performance breakdown benchmark tool for throughput/latency profiling

Storage nodes may not have GPUs. For storage-node-only deployment, build just the memory-node binary:
```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DDVSTOR_STORAGE_NODE_ONLY=ON
cmake --build build -j
```

Additional CMake options:
- `-DDVSTOR_METIS_PARTITION=ON`: require METIS partitioning support (default: AUTO)
- `-DDVSTOR_METIS_PARTITION=OFF`: disable METIS partitioning support
- `-DDVSTOR_USE_NATIVE_ARCH=ON`: compile with `-march=native` for host-specific optimizations
- `-DDVSTOR_BUILD_EXECUTABLES=OFF`: skip building standalone executables

## Data Preparation

### SIFT100M Dataset

Experiment scripts for SIFT100M are provided under `experiment/`:

```bash
# Convert SIFT data to DVSTOR format (adjust paths inside the script first)
python3 experiment/convert_sift100m.py

# Build the offline index
./experiment/build_sift100m_index.sh

# Quick test with reduced scale:
MAX_VECTORS=1000000 GROUNDTRUTH_LABEL=1M ./experiment/build_sift100m_index.sh
```

See `experiment/README.md` for experiment profiles, memory defaults, and configuration options.

### Custom Dataset

For custom datasets, use `vamana_offline_builder` directly (see "Offline Build And Online Load" below), or adapt the evaluation scripts to your data paths.

## Run the Experiments

The experiment harness under `experiment/` supports the ablation profiles in
`experiment/profiles/`. Each run writes the generated service config and reports
under `experiment/reports/<profile>/`.

### Quick Start: SIFT100M

```bash
# 1. Build the index
./experiment/build_sift100m_index.sh

# 2. Start memory nodes with a profile
./experiment/start_all_memory_nodes.sh 00_baseline

# 3. Run recall evaluation
./experiment/run_recall.sh 00_baseline

# 4. Run breakdown benchmark (throughput/latency profiling)
./experiment/run_breakdown.sh 00_baseline

# 5. Stop memory nodes
./experiment/stop_memory_nodes.sh
```

### Optimization Profiles

```bash
# RaBitQ CPU gate + GPU exact beam
./experiment/start_all_memory_nodes.sh 01_rabitq_expension_aware
./experiment/run_breakdown.sh 01_rabitq_expension_aware

# RaBitQ + ALDI + locality-oriented placement
./experiment/start_all_memory_nodes.sh 02_rabitq_expension_aware_aldi
./experiment/run_breakdown.sh 02_rabitq_expension_aware_aldi

# RaBitQ + ALDI + adaptive RDMA scheduling
./experiment/start_all_memory_nodes.sh 03_rabitq_expension_aware_aldi_rdma
./experiment/run_breakdown.sh 03_rabitq_expension_aware_aldi_rdma
```

### Common Overrides

```bash
# Specify cluster hosts
HOSTS="mn1 mn2 mn3 mn4 mn5" BASE_PORT=1234 IB_DEVICE=mlx5_0 \
  ./experiment/start_all_memory_nodes.sh 00_baseline

# Different partition strategy
PARTITION_STRATEGY=metis ./experiment/build_sift100m_index.sh

# Query-only workload
WORKLOAD=query WARMUP_SECONDS=10 MEASURE_SECONDS=60 \
  ./experiment/run_breakdown.sh 01_rabitq_expension_aware
```

### Breakdown Benchmark Tool

The `dvstor_breakdown_benchmark` binary provides detailed performance breakdowns for throughput and latency across query and insert workloads. Run it standalone:

```bash
./build/dvstor_breakdown_benchmark \
  --service-config ./experiment/reports/00_baseline/service_<timestamp>.ini \
  --workload mixed \
  --client-threads 16 \
  --read-ratio 0.5 \
  --warmup-seconds 30 \
  --measure-seconds 60
```

Results are written next to the selected `--output` path as JSON and TXT files with per-category
CPU, GPU, RDMA, and transfer timing breakdowns.

## Offline Build And Online Load

The project supports building a Vamana graph offline and exporting it into DVSTOR's native
memory-node shard format. The offline build also emits RaBitQ search artifacts used by the online
GPU query path.

### Build an Offline Index

```bash
./build/vamana_offline_builder \
  --data-path /path/to/dataset-or-dir \
  --memory-nodes 2 \
  --partition-strategy bfs \
  --threads 32 \
  --R 32 \
  --beam-width-construction 128 \
  --alpha 1.2 \
  --rabitq-bits 4 \
  --output-prefix /path/to/index/dvstor_index
```

The offline builder has only one beam-width knob, and it is the construction/search width used
while building the Vamana graph. It is separate from the online service's `beam-width` and
`beam-width-construction` settings used during query and dynamic insert.

This writes files like:
```text
/path/to/index/dvstor_index_node1_of2.dat
/path/to/index/dvstor_index_node2_of2.dat
/path/to/index/dvstor_index.meta.json
/path/to/index/dvstor_index.rotation.bin
```

Offline shard placement supports `--partition-strategy balanced` (default), `bfs`, and `metis`.
The `bfs` strategy starts from the graph medoid, orders nodes by BFS traversal, and writes contiguous
balanced BFS ranges into shard files so graph-near nodes are more likely to stay on the same memory node.
The `metis` strategy additionally requires METIS support at CMake configure time.

### METIS Graph Repartitioning

The `vamana_metis_repartitioner` tool repartitions an existing offline-built index using METIS
graph partitioning without rebuilding the Vamana graph:

```bash
./build/vamana_metis_repartitioner \
  --input-prefix /path/to/index/dvstor_index \
  --output-prefix /path/to/index/dvstor_metis_index \
  --num-partitions 5
```

This reads the existing shard files and meta JSON, builds a METIS graph from the Vamana adjacency
structure, computes a new balanced partition, and writes repartitioned shard files with a new
`.meta.json`. The repartitioned index can be used directly in the online cluster without rebuilding.

### Online Load

Start each memory node with its local shard:
```bash
./build/dvstor_memory_node --index-file /path/to/index/dvstor_index_node1_of2.dat \
  --port 1234 --mn-memory 152
```

Or let the compute node trigger startup loading on all memory nodes:
```bash
./build/dvstor --servers mn1:1234 mn2:1235 \
  --load-index --index-prefix /path/to/index/dvstor_index \
  --dim 128 --threads 16 --coroutines 16 --k 10 --ef-search 32
```

In both cases, the online cluster reuses the offline-built graph directly instead of rebuilding it through RDMA.
When `--use-rabitq` is enabled on the compute side, the online service loads
the RaBitQ RFQ5 sidecar for the index prefix and uses a CPU gate before exact
GPU distance evaluation.

### Search Modes

- Exact search: GPU exact distance search over remotely fetched full vectors.
  This is the default path and remains the correctness reference.
- RaBitQ CPU gate: enable with `--use-rabitq`. The gate ranks cached RFQ5
  candidates locally, then only exact distances enter the beam.

`--servers` can now be specified either as plain node names such as `cluster3` or as explicit `host:port`
endpoints such as `127.0.0.1:1235`. This allows running multiple memory nodes on the same machine as long as each
instance uses a distinct port.

### Multi-Node Local Example

Example: five memory nodes on one host with online load:

```bash
# Build with 5 shards
./build/vamana_offline_builder \
  --data-path /path/to/dataset-or-dir \
  --memory-nodes 5 \
  --threads 32 \
  --output-prefix /tmp/dvstor_index

# Start memory nodes on different ports
for i in $(seq 1 5); do
  port=$((1233 + i))
  ./build/dvstor_memory_node --port $port \
    --index-file /tmp/dvstor_index_node${i}_of5.dat \
    --mn-memory 10 &
done

# Start compute node
./build/dvstor \
  --servers 127.0.0.1:1234 127.0.0.1:1235 127.0.0.1:1236 127.0.0.1:1237 127.0.0.1:1238 \
  --load-index \
  --index-prefix /tmp/dvstor_index \
  --dim 128 \
  --threads 16 \
  --coroutines 16 \
  --k 10 \
  --ef-search 32
```

## Recall Testing

The `tools/run_recall_test.sh` script builds an offline index and evaluates recall against
ground truth:

```bash
./tools/run_recall_test.sh \
  --data-path /path/to/dataset \
  --output-prefix /path/to/index/output \
  --query-path /path/to/queries.fbin \
  --groundtruth-path /path/to/groundtruth.bin \
  --R 32 \
  --alpha 1.2 \
  --threads 32
```

Environment variable overrides are also supported:
```bash
DATA_PATH=/path/to/dataset OUTPUT_PREFIX=/path/to/index ./tools/run_recall_test.sh
```

## License

MIT License — see [LICENSE](LICENSE).
