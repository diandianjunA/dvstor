# SHINE GPU: Disaggregated GPU Vamana Index

Implementation of a GPU-accelerated Vamana index for memory disaggregation.
This repository is used as a storage-compute disaggregated GPU vector-search baseline.

## Setup

### C++ Libraries and Unix Packages

The following C++ libraries and Unix packages are required to compile the code.
Note that `ibverbs` (the RDMA library) is Linux-only. 
The code also compiles without InfiniBand network cards.

* [ibverbs](https://github.com/linux-rdma/rdma-core/tree/master)
* [boost](https://www.boost.org/doc/libs/1_83_0/doc/html/program_options.html) (to support `boost::program_options` for
  CLI parsing)
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

### Cluster Nodes Configuration

Adjust the IP addresses of the cluster nodes accordingly in `rdma-library/library/utils.cc`:
https://frosch.cosy.sbg.ac.at/mwidmoser/shine-hnsw-index/-/blob/main/rdma-library/library/utils.cc?ref_type=heads#L14-L23

### Compilation

After cloning the repository and installing the requirements, the code must be compiled on all cluster nodes:
```
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make
```

This produces these main binaries:
- `build/shine`: online memory/compute node service
- `build/vamana_offline_builder`: offline Vamana builder that exports SHINE GPU shard files plus RaBitQ artifacts

## Download the Data

First, install all Python requirements:
```
cd scripts
python3 -m pip install -r requirements.txt
```

Then, run the following script to download the data. 
This may take a while, we recommend to run the script within a `tmux` session.
Also make sure that `axel` (a download accelerator) is installed.
```
cd data
bash download.sh
```

Finally, create the queries (adjust the `DATASET_PATH` in `create_queries.py`):
```
python3 create_queries.py
```

Now all the data is available in `data/datasets`, move them to a location where all cluster nodes have access to (e.g., to an NFS).
Then, adjust the path in `config.py`.

## Run the Experiments

* TODO

### Configuration-Driven Performance Test

Use `run_breakdown_test.sh` to benchmark throughput and latency directly from your config file.
This keeps optimization control fully decoupled from the script: enable/disable any current or
future optimization only in `service-config` (e.g., `gpudirect-rdma`, `gpu-cache`, etc.).

```bash
./scripts/run_breakdown_test.sh \
  --service-config ./test/config/local_same_host_5mn.ini \
  --workload mixed \
  --client-threads 16 \
  --read-ratio 0.5 \
  --warmup-seconds 30 \
  --measure-seconds 60
```

The script writes JSON/TXT reports under `reports/breakdown/` and prints key metrics
(throughput, query/insert mean/p95/p99 latency, mixed read/write completions).

## Offline Build And Online Load

The project supports building a Vamana graph offline and exporting it into SHINE GPU's native
memory-node shard format. The offline build also emits RaBitQ search artifacts used by the online
GPU query path.

Build an offline index:
```bash
./build/vamana_offline_builder \
  --data-path /path/to/dataset-or-dir \
  --memory-nodes 2 \
  --threads 32 \
  --R 32 \
  --beam-width-construction 128 \
  --alpha 1.2 \
  --rabitq-bits 4 \
  --output-prefix /path/to/index/shine_index
```

The offline builder has only one beam-width knob, and it is the construction/search width used
while building the Vamana graph. It is separate from the online service's `beam-width` and
`beam-width-construction` settings used during query and dynamic insert.

This writes files like:
```text
/path/to/index/shine_index_node1_of2.dat
/path/to/index/shine_index_node2_of2.dat
/path/to/index/shine_index.meta.json
/path/to/index/shine_index.rotation.bin
```

Then start each memory node with its local shard:
```bash
./scripts/start_memory_node.sh --index-file /path/to/index/shine_index_node1_of2.dat
```

Or let the compute-node initiator trigger startup loading on all memory nodes:
```bash
./scripts/start_compute_node.sh --load-index --index-prefix /path/to/index/shine_index
```

In both cases, the online cluster reuses the offline-built graph directly instead of rebuilding it through RDMA.
When `--search-mode rabitq_gpu` is enabled on the compute side, the online service loads
`<index_prefix>.rotation.bin` and uploads the rotation matrix, rotated centroid, and `t_const`
to each GPU worker.

### Search Modes

- `exact_gpu`: GPU exact distance search over remotely fetched full vectors. Useful as an ablation and correctness reference.
- `rabitq_gpu`: GPU RaBitQ search with final exact rerank. This is the intended paper baseline mode when evaluating
  the disaggregated GPU index without cache or GPU-direct transport optimizations.

`--servers` can now be specified either as plain node names such as `cluster3` or as explicit `host:port`
endpoints such as `127.0.0.1:1235`. This allows running multiple memory nodes on the same machine as long as each
instance uses a distinct port.

Example: five memory nodes on one host with online load:

```bash
./scripts/build_offline_index.sh \
  --data-path /path/to/dataset-or-dir \
  --memory-nodes 5 \
  --threads 32 \
  --output-prefix /tmp/shine_index

./scripts/start_memory_node.sh --port 1234 --index-file /tmp/shine_index_node1_of5.dat
./scripts/start_memory_node.sh --port 1235 --index-file /tmp/shine_index_node2_of5.dat
./scripts/start_memory_node.sh --port 1236 --index-file /tmp/shine_index_node3_of5.dat
./scripts/start_memory_node.sh --port 1237 --index-file /tmp/shine_index_node4_of5.dat
./scripts/start_memory_node.sh --port 1238 --index-file /tmp/shine_index_node5_of5.dat

./scripts/start_compute_node.sh \
  --servers 127.0.0.1:1234 127.0.0.1:1235 127.0.0.1:1236 127.0.0.1:1237 127.0.0.1:1238 \
  --load-index \
  --index-prefix /tmp/shine_index \
  --dim 128 \
  --threads 16 \
  --coroutines 16 \
  --k 10 \
  --ef-search 32
```
