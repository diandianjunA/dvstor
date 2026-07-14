cmake_minimum_required(VERSION 3.18)

option(DVSTOR_STORAGE_NODE_ONLY
    "Build storage-host runtime and CPU offline tools without CUDA/GPU dependencies" OFF)
if (DVSTOR_STORAGE_NODE_ONLY)
    project(dvstor CXX)
else ()
    project(dvstor CXX CUDA)
endif ()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_FLAGS_DEBUG "-g -fno-omit-frame-pointer -fno-inline -fsanitize=address")

if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 11)
    message(FATAL_ERROR
        "DVSTOR requires GCC 11 or newer for standard C++20 coroutines and atomic_ref. "
        "Select a newer compiler with -DCMAKE_CXX_COMPILER=/path/to/g++.")
endif ()

# Use RPATH (not RUNPATH) so that custom METIS/GKlib paths take priority
# over system library paths at runtime.
set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,--disable-new-dtags")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--disable-new-dtags")

if (NOT DVSTOR_STORAGE_NODE_ONLY)
    # CUDA standard
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)

    # Auto-detect GPU architecture if not specified
    if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES 80 86 89 90)
    endif ()
endif ()

# set default build type to Release
if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif ()

option(DVSTOR_BUILD_EXECUTABLES "Build standalone DVSTOR executables and tools" ON)
option(DVSTOR_BUILD_TESTS "Build local DVSTOR smoke tests" ON)
option(DVSTOR_BUILD_OFFLINE_TOOLS
    "Build CPU-only index construction, partitioning, conversion, and PQ tools" ON)
option(DVSTOR_USE_NATIVE_ARCH "Compile C++ sources with -march=native" OFF)
if (DVSTOR_STORAGE_NODE_ONLY)
    set(Boost_NO_BOOST_CMAKE ON CACHE BOOL
        "Use CMake's FindBoost module on CPU-only storage hosts to avoid Conda RPATH injection")
endif ()
if (NOT DVSTOR_STORAGE_NODE_ONLY)
    set(DVSTOR_DOCA_ROOT "/opt/mellanox/doca" CACHE PATH "DOCA SDK installation root")
endif ()
set(DVSTOR_METIS_PARTITION "AUTO" CACHE STRING "METIS graph partitioning support: AUTO, ON, or OFF")
set_property(CACHE DVSTOR_METIS_PARTITION PROPERTY STRINGS AUTO ON OFF)
string(TOUPPER "${DVSTOR_METIS_PARTITION}" DVSTOR_METIS_PARTITION_MODE)
if (NOT DVSTOR_METIS_PARTITION_MODE STREQUAL "AUTO" AND
    NOT DVSTOR_METIS_PARTITION_MODE STREQUAL "ON" AND
    NOT DVSTOR_METIS_PARTITION_MODE STREQUAL "OFF")
    message(FATAL_ERROR "DVSTOR_METIS_PARTITION must be AUTO, ON, or OFF")
endif ()

# CXX compiler flags (not applied to CUDA files)
add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-Wall>)
add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-Wextra>)
add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-Wpedantic>)
add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-Wfatal-errors>)
if (DVSTOR_USE_NATIVE_ARCH)
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-march=native>)
endif ()
add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-ffast-math>)
add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-mavx2>)
add_compile_options(-DNOHUGEPAGES)
#add_compile_options(-g -fno-omit-frame-pointer -fno-inline)  # for profiling

set(DVSTOR_SOURCE_ROOT "${CMAKE_CURRENT_SOURCE_DIR}")
set(DVSTOR_CONDA_LIB_DIR "/home/xjs/anaconda3/lib" CACHE PATH
    "Optional local libstdc++ search path used on the development machine")
set(DVSTOR_METIS_ROOT "" CACHE PATH "Optional METIS installation root")
set(DVSTOR_FAISS_BLAS_VENDOR "Generic" CACHE STRING
    "BLAS vendor for CPU Faiss offline tools; Generic avoids nested MKL/OpenMP runtimes")

list(APPEND CMAKE_MODULE_PATH
    "${DVSTOR_SOURCE_ROOT}"
    "${DVSTOR_SOURCE_ROOT}/cmake")

include(DvstorDependencies)
include(DvstorTargetHelpers)
if (DVSTOR_BUILD_OFFLINE_TOOLS)
    find_package(FaissCPU REQUIRED)
endif ()

include_directories(${DVSTOR_SOURCE_ROOT}/src)
include_directories(${DVSTOR_SOURCE_ROOT}/thirdparty)
add_subdirectory(rdma-library)
add_subdirectory(thirdparty)

set(DVSTOR_MEMORY_NODE_SOURCES
    src/gpu_search/index_format.cc
    src/gpu_search/pq_index.cc
    src/memory_node/memory_node.cc
    src/memory_node/peer_rdma.cc
    src/memory_node/peer_rpc.cc
    src/memory_node/storage_owner_anchor.cc
    src/memory_node/storage_owner_index.cc
    src/memory_node/storage_owner_maintenance.cc
    src/memory_node/storage_owner_runtime.cc
    src/vamana/anchor_index.cc
)

set(DVSTOR_OFFLINE_BUILDER_SOURCES
    tools/vamana_offline/anchor_builder.cc
    tools/vamana_offline/config.cc
    tools/vamana_offline/dataset_io.cc
    tools/vamana_offline/graph.cc
    tools/vamana_offline/partitioning.cc
    tools/vamana_offline/progress.cc
    tools/vamana_offline/recall_check.cc
    tools/vamana_offline/shard_writer.cc
)

set(DVSTOR_PQ_INDEXER_SOURCES
    src/gpu_search/index_format.cc
    src/gpu_search/pq_index.cc
    src/vamana/anchor_index.cc
    tools/vamana_offline/pq_indexer.cc
)

set(DVSTOR_BREAKDOWN_SUPPORT_SOURCES
    tools/breakdown_benchmark/dataset.cc
    tools/breakdown_benchmark/report.cc
)

set(DVSTOR_BREAKDOWN_BENCHMARK_SOURCES
    tools/breakdown_benchmark/args.cc
    tools/breakdown_benchmark/progress.cc
    tools/breakdown_benchmark/workload.cc
)

function(dvstor_add_offline_tools)
    add_executable(vamana_offline_builder
        tools/vamana_offline_builder.cc
        ${DVSTOR_OFFLINE_BUILDER_SOURCES}
    )
    dvstor_target_tool_includes(vamana_offline_builder)
    target_link_libraries(vamana_offline_builder rdma_library)
    target_link_metis(vamana_offline_builder)

    add_executable(vamana_anchor_sidecar_builder
        tools/vamana_anchor_sidecar_builder.cc
        src/vamana/anchor_index.cc
    )
    dvstor_target_tool_includes(vamana_anchor_sidecar_builder)
    target_link_libraries(vamana_anchor_sidecar_builder rdma_library)

    add_executable(vamana_pq_indexer
        tools/vamana_pq_indexer.cc
        ${DVSTOR_PQ_INDEXER_SOURCES}
    )
    dvstor_target_tool_includes(vamana_pq_indexer)
    target_link_libraries(vamana_pq_indexer rdma_library FaissCPU::FaissCPU)

    add_executable(vamana_legacy_index_converter
        tools/vamana_legacy_index_converter.cc
        tools/legacy_index/migrator.cc
    )
    dvstor_target_tool_includes(vamana_legacy_index_converter)
    target_link_libraries(vamana_legacy_index_converter rdma_library)

    add_executable(generate_sift101m_recall_data
        tools/generate_sift101m_recall_data.cc
    )
    dvstor_target_tool_includes(generate_sift101m_recall_data)
endfunction()

if (DVSTOR_STORAGE_NODE_ONLY)
    message(STATUS "Storage-node build: CUDA and DOCA are disabled; CPU offline tools=${DVSTOR_BUILD_OFFLINE_TOOLS}")
    add_executable(dvstor_memory_node
        src/memory_node_main.cc
        ${DVSTOR_MEMORY_NODE_SOURCES}
        src/common/core_assignment.cc
        src/common/timing.cc
        src/service/index_metadata.cc
    )
    dvstor_target_common_includes(dvstor_memory_node)
    target_link_libraries(dvstor_memory_node rdma_library)
    if (DVSTOR_BUILD_OFFLINE_TOOLS)
        dvstor_add_offline_tools()
    endif ()
else ()
    find_path(DVSTOR_DOCA_INCLUDE_DIR doca_gpunetio.h
        HINTS "${DVSTOR_DOCA_ROOT}/include")
    find_library(DVSTOR_DOCA_GPUNETIO_LIBRARY doca_gpunetio
        HINTS "${DVSTOR_DOCA_ROOT}/lib/x86_64-linux-gnu")
    find_library(DVSTOR_DOCA_VERBS_LIBRARY doca_verbs
        HINTS "${DVSTOR_DOCA_ROOT}/lib/x86_64-linux-gnu")
    find_library(DVSTOR_DOCA_COMMON_LIBRARY doca_common
        HINTS "${DVSTOR_DOCA_ROOT}/lib/x86_64-linux-gnu")
    find_library(DVSTOR_DOCA_RDMA_LIBRARY doca_rdma
        HINTS "${DVSTOR_DOCA_ROOT}/lib/x86_64-linux-gnu")
    find_library(DVSTOR_MLX5_LIBRARY mlx5)
    if (NOT DVSTOR_DOCA_INCLUDE_DIR OR
        NOT DVSTOR_DOCA_GPUNETIO_LIBRARY OR
        NOT DVSTOR_DOCA_VERBS_LIBRARY OR
        NOT DVSTOR_DOCA_COMMON_LIBRARY OR
        NOT DVSTOR_DOCA_RDMA_LIBRARY OR
        NOT DVSTOR_MLX5_LIBRARY)
        message(FATAL_ERROR
            "The compute-node runtime requires a complete DOCA GPUNetIO installation")
    endif ()
    set(DVSTOR_HAVE_GPUNETIO ON)
    list(FIND CMAKE_CUDA_ARCHITECTURES 52 DVSTOR_CUDA_SM52_INDEX)
    if (NOT DVSTOR_CUDA_SM52_INDEX EQUAL -1)
        message(WARNING "GPUNetIO requires sm_60+; replacing cached sm_52 with sm_80")
        set(CMAKE_CUDA_ARCHITECTURES 80 CACHE STRING "CUDA architectures" FORCE)
    endif ()

    # GPU-centric query engine kernels. Keep this list explicit so retired
    # CPU-driven CUDA paths cannot silently re-enter the runtime.
    set(DVSTOR_GPU_SOURCES
        src/gpu_search/persistent_kernel.cu
    )
    if (DVSTOR_HAVE_GPUNETIO)
        list(APPEND DVSTOR_GPU_SOURCES src/gpu/gpunetio_probe.cu)
    endif ()
    if (DVSTOR_GPU_SOURCES)
        add_library(dvstor_gpu_kernels STATIC ${DVSTOR_GPU_SOURCES})
        dvstor_target_common_includes(dvstor_gpu_kernels PUBLIC)
        target_link_libraries(dvstor_gpu_kernels
            CUDA::cudart
            rdma_library
        )
        if (DVSTOR_HAVE_GPUNETIO)
            target_include_directories(dvstor_gpu_kernels PUBLIC ${DVSTOR_DOCA_INCLUDE_DIR})
            target_compile_definitions(dvstor_gpu_kernels PUBLIC DVSTOR_HAVE_GPUNETIO=1)
        endif ()
        set_target_properties(dvstor_gpu_kernels PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON
            CUDA_RESOLVE_DEVICE_SYMBOLS ON
            POSITION_INDEPENDENT_CODE ON
        )
        target_compile_options(dvstor_gpu_kernels PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
            $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda>
            $<$<COMPILE_LANGUAGE:CUDA>:-O3>
        )
    endif ()

    # Compute-node runtime. This is intentionally explicit: the dev branch has
    # one query engine, with CPU threads limited to admission and update RPCs.
    set(DVSTOR_RUNTIME_SOURCES
        src/common/core_assignment.cc
        src/common/timing.cc
        src/gpu_search/delta_index.cc
        src/gpu_search/navigation_bootstrapper.cc
        src/gpu_search/index_format.cc
        src/gpu_search/persistent_engine.cc
        src/gpu_search/pq_index.cc
        src/gpu_search/types.cc
        src/service/compute_service.cc
        src/service/index_metadata.cc
        src/vamana/anchor_index.cc
    )
    if (DVSTOR_HAVE_GPUNETIO)
        list(APPEND DVSTOR_RUNTIME_SOURCES src/gpu/gpunetio_transport.cc)
        set_source_files_properties(src/gpu/gpunetio_transport.cc PROPERTIES
            COMPILE_OPTIONS "-Wno-deprecated-declarations;-Wno-pedantic")
    endif ()

    add_library(dvstor_runtime STATIC ${DVSTOR_RUNTIME_SOURCES})
    dvstor_target_common_includes(dvstor_runtime PUBLIC)
    target_link_libraries(dvstor_runtime
        rdma_library
        CUDA::cudart
    )
    if (DVSTOR_HAVE_GPUNETIO)
        target_include_directories(dvstor_runtime PUBLIC ${DVSTOR_DOCA_INCLUDE_DIR})
        target_compile_definitions(dvstor_runtime PUBLIC DVSTOR_HAVE_GPUNETIO=1)
        target_link_libraries(dvstor_runtime
            ${DVSTOR_DOCA_GPUNETIO_LIBRARY}
            ${DVSTOR_DOCA_VERBS_LIBRARY}
            ${DVSTOR_DOCA_COMMON_LIBRARY}
            ${DVSTOR_DOCA_RDMA_LIBRARY}
            ${DVSTOR_MLX5_LIBRARY}
        )
    endif ()
    if (DVSTOR_GPU_SOURCES)
        target_link_libraries(dvstor_runtime dvstor_gpu_kernels)
    endif ()
    target_compile_options(dvstor_runtime PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-fcoroutines>)

    if (DVSTOR_BUILD_EXECUTABLES OR DVSTOR_BUILD_TESTS)
        add_library(dvstor_breakdown_support STATIC ${DVSTOR_BREAKDOWN_SUPPORT_SOURCES})
        dvstor_target_tool_includes(dvstor_breakdown_support)
        target_link_libraries(dvstor_breakdown_support PUBLIC dvstor_runtime)
    endif ()

    add_executable(dvstor_memory_node
        src/memory_node_main.cc
        ${DVSTOR_MEMORY_NODE_SOURCES}
        src/common/core_assignment.cc
        src/common/timing.cc
        src/service/index_metadata.cc
    )
    dvstor_target_common_includes(dvstor_memory_node)
    target_link_libraries(dvstor_memory_node rdma_library)

    if (DVSTOR_BUILD_EXECUTABLES)
        add_executable(dvstor_compute_node src/main.cc)
        target_link_libraries(dvstor_compute_node dvstor_runtime)

        if (DVSTOR_HAVE_GPUNETIO)
            add_executable(dvstor_gpunetio_probe tools/gpunetio_probe.cc)
            dvstor_target_common_includes(dvstor_gpunetio_probe)
            target_include_directories(dvstor_gpunetio_probe PRIVATE ${DVSTOR_DOCA_INCLUDE_DIR})
            target_compile_definitions(dvstor_gpunetio_probe PRIVATE ALLOW_EXPERIMENTAL_API=1)
            target_compile_options(dvstor_gpunetio_probe PRIVATE
                -Wno-deprecated-declarations
                -Wno-pedantic
            )
            target_link_libraries(dvstor_gpunetio_probe
                rdma_library
                CUDA::cudart
                ${DVSTOR_DOCA_GPUNETIO_LIBRARY}
                ${DVSTOR_DOCA_VERBS_LIBRARY}
                ${DVSTOR_DOCA_COMMON_LIBRARY}
                ${DVSTOR_MLX5_LIBRARY}
            )

            add_executable(dvstor_gpunetio_loopback_probe tools/gpunetio_loopback_probe.cc)
            dvstor_target_common_includes(dvstor_gpunetio_loopback_probe)
            target_link_libraries(dvstor_gpunetio_loopback_probe dvstor_runtime)
        endif ()

        add_executable(dvstor_breakdown_benchmark
            tools/dvstor_breakdown_benchmark.cc
            ${DVSTOR_BREAKDOWN_BENCHMARK_SOURCES}
        )
        dvstor_target_tool_includes(dvstor_breakdown_benchmark)
        target_link_libraries(dvstor_breakdown_benchmark dvstor_breakdown_support)

        add_executable(dvstor_sift101m_long_insert_recall
            tools/dvstor_sift101m_long_insert_recall.cc
            tools/breakdown_benchmark/args.cc
            tools/breakdown_benchmark/progress.cc
        )
        dvstor_target_tool_includes(dvstor_sift101m_long_insert_recall)
        target_link_libraries(dvstor_sift101m_long_insert_recall dvstor_runtime)
    endif ()

    if (DVSTOR_BUILD_OFFLINE_TOOLS)
        dvstor_add_offline_tools()
    endif ()

    if (DVSTOR_BUILD_TESTS AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test/CMakeLists.txt")
        enable_testing()
        add_subdirectory(test)
    endif ()

    # status prints
    get_target_property(compile_options dvstor_runtime COMPILE_OPTIONS)
    message(STATUS "Compile options: ${compile_options}")
    message(STATUS "CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
    message(STATUS "DOCA GPUNetIO backend: ${DVSTOR_HAVE_GPUNETIO}")
endif ()
