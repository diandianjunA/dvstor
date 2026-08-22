# Program 2: Live/DynaExtent motivation

This runner produces the three requested pieces of evidence with two query
runs and one short transport probe:

1. query-weighted live-neighbor degree/required-prefix distribution;
2. one-shot 832 B, dependent 16+384 B, and hinted one-shot 400 B RDMA;
3. end-to-end Fixed, dependent Header→Neighbor, and LiveExtent query
   performance.

The `header-neighbor` baseline performs a 16-byte header RDMA followed by an
exact-size neighbor-body RDMA. The assembled prefix is accepted only after the
same checksum/incarnation validation used by the other policies; a concurrent
mutation restarts the two-stage snapshot.

The fixed, header-neighbor, and live cases use the same full-system profile and
only change the graph-read policy. Storage must be restarted for every prompt
so the startup contract and remote session are fresh.

Compute node:

```bash
cd /home/xjs/experiment/dvstor
./motivation/program2/run_program2.sh
```

Storage node, execute the command printed by the compute runner at each of the
four prompts:

```bash
./motivation/program2/start_storage_case.sh fixed
./motivation/program2/start_storage_case.sh header
./motivation/program2/start_storage_case.sh live
./motivation/program2/start_storage_case.sh probe
```

Short smoke settings can be supplied on the compute node:

```bash
WARMUP_SECONDS=2 MEASURE_SECONDS=5 RECALL_QUERIES=100 \
  ./motivation/program2/run_program2.sh
```

If a case fails, reuse its printed result root:

```bash
RUN_ROOT=/path/to/program2_TIMESTAMP ./motivation/program2/run_program2.sh live
RUN_ROOT=/path/to/program2_TIMESTAMP ./motivation/program2/run_program2.sh probe
RUN_ROOT=/path/to/program2_TIMESTAMP ./motivation/program2/run_program2.sh summarize
```

Outputs include `summary.json`, `summary.csv`, `degree_histogram.csv`,
`program2_motivation.svg`, and `program2_effectiveness.svg`.
