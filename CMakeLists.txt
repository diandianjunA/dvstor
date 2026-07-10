cmake_minimum_required(VERSION 3.18)

option(DVSTOR_STORAGE_NODE_ONLY "Build storage-node-only binary without CUDA/GPU dependencies" OFF)
if (DVSTOR_STORAGE_NODE_ONLY)
    project(rdma-vamana CXX)
else ()
    project(rdma-vamana CXX CUDA)
endif ()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_FLAGS_DEBUG "-g -fno-omit-frame-pointer -fno-inline -fsanitize=address")

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
option(DVSTOR_USE_NATIVE_ARCH "Compile C++ sources with -march=native" OFF)
option(DVSTOR_ENABLE_GPUNETIO "Build the DOCA GPUNetIO query backend when the SDK is available" ON)
set(DVSTOR_DOCA_ROOT "/opt/mellanox/doca" CACHE PATH "DOCA SDK installation root")
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

list(APPEND CMAKE_MODULE_PATH
    "${DVSTOR_SOURCE_ROOT}"
    "${DVSTOR_SOURCE_ROOT}/cmake")

include(DvstorDependencies)
include(DvstorTargetHelpers)

include_directories(${DVSTOR_SOURCE_ROOT}/src)
include_directories(${DVSTOR_SOURCE_ROOT}/thirdparty)
add_subdirectory(rdma-library)
add_subdirectory(thirdparty)

set(DVSTOR_MEMORY_NODE_SOURCES
    src/gpu_search/index_format.cc
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
    src/gpu_search/index_format.cc
    tools/vamana_offline/anchor_builder.cc
    tools/vamana_offline/config.cc
    tools/vamana_offline/dataset_io.cc
    tools/vamana_offline/gpu_tiered_writer.cc
    tools/vamana_offline/graph.cc
    tools/vamana_offline/partitioning.cc
    tools/vamana_offline/progress.cc
    tools/vamana_offline/recall_check.cc
    tools/vamana_offline/shard_writer.cc
)

set(DVSTOR_GPU_SIDECAR_CONVERTER_SOURCES
    src/gpu_search/index_format.cc
    tools/vamana_offline/gpu_sidecar_converter.cc
    tools/vamana_offline/progress.cc
)

set(DVSTOR_BREAKDOWN_BENCHMARK_SOURCES
    tools/breakdown_benchmark/args.cc
    tools/breakdown_benchmark/progress.cc
    tools/breakdown_benchmark/workload.cc
)

if (DVSTOR_STORAGE_NODE_ONLY)
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
        add_executable(vamana_offline_builder
            tools/vamana_offline_builder.cc
            ${DVSTOR_OFFLINE_BUILDER_SOURCES}
        )
        dvstor_target_tool_includes(vamana_offline_builder)
        target_link_libraries(vamana_offline_builder
            rdma_library
        )
        target_link_metis(vamana_offline_builder)

        add_vamana_metis_repartitioner_target()
        add_vamana_bfs_repartitioner_target()

        add_executable(vamana_rabitq_sidecar_converter
            tools/vamana_rabitq_sidecar_converter.cc
        )
        dvstor_target_common_includes(vamana_rabitq_sidecar_converter)
        target_link_libraries(vamana_rabitq_sidecar_converter rdma_library)

        add_executable(vamana_gpu_sidecar_converter
            tools/vamana_gpu_sidecar_converter.cc
            ${DVSTOR_GPU_SIDECAR_CONVERTER_SOURCES}
        )
        dvstor_target_tool_includes(vamana_gpu_sidecar_converter)
        target_link_libraries(vamana_gpu_sidecar_converter rdma_library)

        add_executable(generate_sift101m_recall_data
            tools/generate_sift101m_recall_data.cc
        )
        dvstor_target_tool_includes(generate_sift101m_recall_data)
    endif ()
else ()
    set(DVSTOR_HAVE_GPUNETIO OFF)
    if (DVSTOR_ENABLE_GPUNETIO)
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
        if (DVSTOR_DOCA_INCLUDE_DIR AND
            DVSTOR_DOCA_GPUNETIO_LIBRARY AND
            DVSTOR_DOCA_VERBS_LIBRARY AND
            DVSTOR_DOCA_COMMON_LIBRARY AND
            DVSTOR_DOCA_RDMA_LIBRARY AND
            DVSTOR_MLX5_LIBRARY)
            set(DVSTOR_HAVE_GPUNETIO ON)
            list(FIND CMAKE_CUDA_ARCHITECTURES 52 DVSTOR_CUDA_SM52_INDEX)
            if (NOT DVSTOR_CUDA_SM52_INDEX EQUAL -1)
                message(WARNING "GPUNetIO requires sm_60+; replacing cached sm_52 with sm_80")
                set(CMAKE_CUDA_ARCHITECTURES 80 CACHE STRING "CUDA architectures" FORCE)
            endif ()
        else ()
            message(WARNING "DOCA GPUNetIO SDK is incomplete; building verbs_proxy/local backends only")
        endif ()
    endif ()

    # GPU kernels library (compiled by nvcc)
    file(GLOB_RECURSE DVSTOR_GPU_SOURCES CONFIGURE_DEPENDS
        src/gpu/*.cu
        src/gpu_search/*.cu)
    if (NOT DVSTOR_HAVE_GPUNETIO)
        list(REMOVE_ITEM DVSTOR_GPU_SOURCES
            "${CMAKE_CURRENT_SOURCE_DIR}/src/gpu/gpunetio_query_launcher.cu")
    endif ()
    if (DVSTOR_GPU_SOURCES)
        add_library(dvstor_gpu_kernels STATIC ${DVSTOR_GPU_SOURCES})
        dvstor_target_common_includes(dvstor_gpu_kernels PUBLIC)
        target_link_libraries(dvstor_gpu_kernels
            CUDA::cudart
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

    # Main runtime library (compiled by host compiler, C++20 with coroutines)
    file(GLOB_RECURSE DVSTOR_RUNTIME_SOURCES CONFIGURE_DEPENDS src/*.cc)
    list(REMOVE_ITEM DVSTOR_RUNTIME_SOURCES
        "${CMAKE_CURRENT_SOURCE_DIR}/src/main.cc"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/memory_node_main.cc"
    )
    if (NOT DVSTOR_HAVE_GPUNETIO)
        list(REMOVE_ITEM DVSTOR_RUNTIME_SOURCES
            "${CMAKE_CURRENT_SOURCE_DIR}/src/gpu/gpunetio_query_engine.cc")
    endif ()
    if (DVSTOR_HAVE_GPUNETIO)
        set_source_files_properties(src/gpu/gpunetio_query_engine.cc PROPERTIES
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
        add_executable(dvstor src/main.cc)
        target_link_libraries(dvstor dvstor_runtime)

        add_executable(vamana_offline_builder
            tools/vamana_offline_builder.cc
            ${DVSTOR_OFFLINE_BUILDER_SOURCES}
        )
        dvstor_target_tool_includes(vamana_offline_builder)
        target_link_libraries(vamana_offline_builder
            rdma_library
        )
        target_link_metis(vamana_offline_builder)

        add_executable(vamana_anchor_sidecar_builder
            tools/vamana_anchor_sidecar_builder.cc
            src/vamana/anchor_index.cc
        )
        dvstor_target_tool_includes(vamana_anchor_sidecar_builder)
        target_link_libraries(vamana_anchor_sidecar_builder rdma_library)

        add_vamana_metis_repartitioner_target()
        add_vamana_bfs_repartitioner_target()

        add_executable(vamana_rabitq_sidecar_converter
            tools/vamana_rabitq_sidecar_converter.cc
        )
        dvstor_target_common_includes(vamana_rabitq_sidecar_converter)
        target_link_libraries(vamana_rabitq_sidecar_converter rdma_library)

        add_executable(vamana_gpu_sidecar_converter
            tools/vamana_gpu_sidecar_converter.cc
            ${DVSTOR_GPU_SIDECAR_CONVERTER_SOURCES}
        )
        dvstor_target_tool_includes(vamana_gpu_sidecar_converter)
        target_link_libraries(vamana_gpu_sidecar_converter rdma_library)

        add_executable(generate_sift101m_recall_data
            tools/generate_sift101m_recall_data.cc
        )
        dvstor_target_tool_includes(generate_sift101m_recall_data)

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
        endif ()

        add_executable(dvstor_breakdown_benchmark
            tools/dvstor_breakdown_benchmark.cc
            ${DVSTOR_BREAKDOWN_BENCHMARK_SOURCES}
        )
        dvstor_target_tool_includes(dvstor_breakdown_benchmark)
        target_link_libraries(dvstor_breakdown_benchmark dvstor_runtime)

        add_executable(dvstor_sift101m_long_insert_recall
            tools/dvstor_sift101m_long_insert_recall.cc
            tools/breakdown_benchmark/args.cc
            tools/breakdown_benchmark/progress.cc
        )
        dvstor_target_tool_includes(dvstor_sift101m_long_insert_recall)
        target_link_libraries(dvstor_sift101m_long_insert_recall dvstor_runtime)
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
