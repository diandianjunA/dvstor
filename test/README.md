# DvstorIndex Smoke Test

This directory contains a small in-repository test for the in-process `DvstorIndex` wrapper.

Build:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j --target DvstorIndexSmokeTest
```

Run on a single machine after a memory node is up:

```bash
./scripts/start_memory_node.sh -f --mn-memory 10
./build/test/DvstorIndexSmokeTest ./test/config/local_single_cn.ini
```

The test process itself acts as the compute node. It inserts a small synthetic dataset, checks
that self-query succeeds, stores the index, reloads it, and checks the query again.

It then runs a concurrent insertion stress test. You can override its default concurrency:

```bash
./build/test/DvstorIndexSmokeTest ./test/config/local_single_cn.ini /tmp/dvstor_test_idx 8 64
```

The last two arguments are:

- `concurrent_threads`
- `vectors_per_thread`

Offline layout integration smoke test:

```bash
./test/run_offline_layout_smoke.sh
```

This generates a tiny `.fbin` dataset, builds a `rabitq_search_block` index with
`vamana_offline_builder`, starts one local memory node, loads the offline shard,
then checks query recall before and after a small storage-owner update batch.
Set `GPU_DEVICE`, `PORT`, or `KEEP_WORK_DIR=1` in the environment when needed.
