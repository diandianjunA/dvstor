# SIFT100M Optimization Profiles

Use these profiles to compare optimization gains one by one. Start memory nodes with the same profile used for the benchmark, because the final profile switches insert execution to `storage_owner`.

```bash
# 1. Original baseline: no GPUDirect RDMA, no GPU RaBitQ cache, compute-side inserts.
./evaluation/sift100m/start_profile_memory_nodes.sh baseline restart
./evaluation/sift100m/run_profile.sh baseline

# 2. GPUDirect RDMA only.
./evaluation/sift100m/start_profile_memory_nodes.sh gpudirect_rdma restart
./evaluation/sift100m/run_profile.sh gpudirect_rdma

# 3. GPUDirect RDMA + GPU cache slot_clock mode.
./evaluation/sift100m/start_profile_memory_nodes.sh gpudirect_slot_clock restart
./evaluation/sift100m/run_profile.sh gpudirect_slot_clock

# 4. GPUDirect RDMA + GPU cache slot_clock mode + storage_owner mode.
./evaluation/sift100m/start_profile_memory_nodes.sh gpudirect_slot_clock_storage_owner restart
./evaluation/sift100m/run_profile.sh gpudirect_slot_clock_storage_owner

```

Each run writes a generated compute config to `evaluation/sift100m/generated_sift100m_<profile>.ini` and reports under `evaluation/sift100m/reports/<profile>/`.

The profile files are plain shell env files, so common benchmark knobs still work:

```bash
READ_RATIO=1.0 CLIENT_THREADS=32 MEASURE_SECONDS=120 ./evaluation/sift100m/run_profile.sh gpudirect_slot_clock_storage_owner
```

Profile summary:

| Profile | GPUDirect RDMA | Neighbor cache | GPU RaBitQ cache | Insert mode | Workers |
| --- | --- | --- | --- | --- | --- |
| `baseline` | off | 2048 MB | 0 MB | `compute` | 8 insert / 8 query |
| `gpudirect_rdma` | on | 2048 MB | 0 MB | `compute` | 8 insert / 8 query |
| `gpudirect_slot_clock` | on | 2048 MB | 8192 MB | `compute` | 8 insert / 8 query |
| `gpudirect_slot_clock_storage_owner` | on | 2048 MB | 8192 MB | `storage_owner` | 0 insert / 16 query |

