# Repository Guidelines

## Project Structure & Module Organization
`src/` contains the runtime code for the disaggregated GPU Vamana service. Key areas are `src/common/` for shared config and utilities, `src/gpu/` for CUDA kernels and GPU helpers, `src/rdma/` for RDMA operations, `src/service/` for compute-service orchestration, and `src/vamana/` for index logic. `tools/` holds standalone binaries such as `vamana_offline_builder`. `scripts/` contains launcher helpers for memory and compute nodes. `test/` contains the in-repo smoke test, headers under `test/include/`, source under `test/src/`, and sample INI configs under `test/config/`. `rdma-library/` and `thirdparty/` are bundled dependencies; avoid broad edits there unless the change is dependency-specific.

## Build, Test, and Development Commands
Configure once with `cmake -S . -B build -DCMAKE_BUILD_TYPE=Release`; this also exports `compile_commands.json` for editor tooling. Build the main binaries with `cmake --build build -j`. Build only the smoke test with `cmake --build build -j --target ShineIndexSmokeTest`. Run the offline builder directly with `./build/vamana_offline_builder --data-path /path/to/data --output-prefix /tmp/shine_index ...`. Start a local memory node with `./scripts/start_memory_node.sh -f --mn-memory 10`, then run the smoke test with `./build/test/ShineIndexSmokeTest ./test/config/local_single_cn.ini`.

## Coding Style & Naming Conventions
Follow the existing C++20/CUDA style: 2-space indentation in `.cc` and `.hh` files, braces on the same line, and small anonymous namespaces for local helpers. Use `PascalCase` for types (`ComputeService`), `snake_case` for functions and variables (`wait_for_shutdown_signal`), and lower-case file names with `.cc`, `.cu`, `.hh`, and `.cuh` suffixes. No formatter or linter config is checked in, so keep changes consistent with surrounding code and preserve `set -euo pipefail` in shell scripts.

## Testing Guidelines
This repository currently relies on executable smoke tests rather than a unit-test framework. Add new test cases under `test/src/` and register new binaries in `test/CMakeLists.txt`. Prefer config-driven tests that reuse files in `test/config/`. There is no enforced coverage threshold, but every PR should run the relevant build target and at least one realistic execution path for the touched area.

## Commit & Pull Request Guidelines
Recent history uses short subject-only commits such as `debug` and `change worker config`; keep commits concise, imperative, and scoped, but use clearer summaries than the existing baseline. For PRs, include the subsystem touched, the exact commands run, any required cluster or GPU assumptions, and sample output when behavior changes affect launch, indexing, or recall workflows.
