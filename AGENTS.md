# Repository Guidelines

## Project Structure & Module Organization
`src/` contains the runtime code for the disaggregated GPU Vamana service. Key areas include `src/common/` for shared config and utilities, `src/gpu/` and `src/gpu/kernels/` for CUDA support, `src/rdma/` for RDMA operations, `src/service/` for orchestration, and `src/vamana/` for index logic. Support code also appears in `src/cache/`, `src/http/`, `src/io/`, and `src/router/`. Use `tools/` for standalone binaries, `scripts/` for cluster launch helpers, and `test/` for the smoke test, headers, and sample INI configs. Avoid broad edits under `rdma-library/` and `thirdparty/` unless the dependency itself is the target.

## Build, Test, and Development Commands
Configure once with `cmake -S . -B build -DCMAKE_BUILD_TYPE=Release` to generate build files and `compile_commands.json`. Build the project with `cmake --build build -j`. Build only the smoke test with `cmake --build build -j --target DvstorIndexSmokeTest`. Run the offline builder with `./build/vamana_offline_builder --data-path /path/to/data --output-prefix /tmp/dvstor_index ...`. For local validation, start a memory node with `./scripts/start_memory_node.sh -f --mn-memory 10`, then run `./build/test/DvstorIndexSmokeTest ./test/config/local_single_cn.ini`.
For storage nodes without CUDA/GPU/DOCA GPUNetIO, configure with `cmake -S . -B build-storage -DCMAKE_BUILD_TYPE=Release -DDVSTOR_STORAGE_NODE_ONLY=ON` and build `cmake --build build-storage -j`; this produces only `build-storage/dvstor_memory_node`.

## Coding Style & Naming Conventions
Follow the surrounding C++20/CUDA style: 2-space indentation, same-line braces, and compact anonymous namespaces for local helpers. Use `PascalCase` for types such as `ComputeService`, `snake_case` for functions and variables such as `wait_for_shutdown_signal`, and lower-case filenames with `.cc`, `.hh`, `.cu`, and `.cuh` suffixes. No formatter or linter is checked in, so keep changes consistent with nearby code. Preserve `set -euo pipefail` in shell scripts.

## Testing Guidelines
This repository uses executable smoke tests rather than a unit-test framework. Add new test sources under `test/src/`, shared headers under `test/include/`, and new configs under `test/config/`; register new binaries in `test/CMakeLists.txt`. Prefer config-driven scenarios that exercise realistic node layouts. Every change should build the touched target and run at least one relevant execution path.

## Commit & Pull Request Guidelines
The original history is not available in this workspace, but existing guidance points to short, imperative commit subjects such as `change worker config`; keep that style, while making summaries clearer and scoped. Pull requests should name the subsystem changed, list exact build and test commands run, call out GPU or cluster assumptions, and include sample output when behavior changes affect launch, indexing, or recall workflows.
