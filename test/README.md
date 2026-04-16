# ShineIndex Smoke Test

This directory contains a small in-repository test for the in-process `ShineIndex` wrapper.

Build:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j --target ShineIndexSmokeTest
```

Run on a single machine after a memory node is up:

```bash
./scripts/start_memory_node.sh -f --mn-memory 10
./build/test/ShineIndexSmokeTest ./test/config/local_single_cn.ini
```

The test process itself acts as the compute node. It inserts a small synthetic dataset, checks
that self-query succeeds, stores the index, reloads it, and checks the query again.

It then runs a concurrent insertion stress test. You can override its default concurrency:

```bash
./build/test/ShineIndexSmokeTest ./test/config/local_single_cn.ini /tmp/shine_test_idx 8 64
```

The last two arguments are:

- `concurrent_threads`
- `vectors_per_thread`
