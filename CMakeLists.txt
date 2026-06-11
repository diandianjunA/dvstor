cmake_minimum_required(VERSION 3.18)

option(DVSTOR_STORAGE_NODE_ONLY "Build storage-node-only binary without CUDA/GPU dependencies" OFF)
if (DVSTOR_STORAGE_NODE_ONLY)
    project(rdma-vamana CXX)
else ()
    project(rdma-vamana CXX CUDA)
endif ()

include(CTest)

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

# prefer the Anaconda stdc++/gcc build so we satisfy CXXABI_1.3.15 dependencies from libtbb
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -L/home/xjs/anaconda3/lib -Wl,-rpath,/home/xjs/anaconda3/lib")

option(DVSTOR_BUILD_EXECUTABLES "Build standalone DVSTOR executables and tools" ON)
option(DVSTOR_BUILD_TESTS "Build local DVSTOR smoke tests" ON)
option(DVSTOR_USE_NATIVE_ARCH "Compile C++ sources with -march=native" OFF)
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

# Find dependencies
if (NOT DVSTOR_STORAGE_NODE_ONLY)
    find_package(CUDAToolkit 11.8 REQUIRED)
endif ()
if (NOT DVSTOR_METIS_PARTITION_MODE STREQUAL "OFF")
    find_path(METIS_INCLUDE_DIR metis.h)
    find_library(METIS_LIBRARY NAMES metis)
    find_library(GKLIB_LIBRARY NAMES GKlib)
    if (METIS_INCLUDE_DIR AND METIS_LIBRARY)
        set(DVSTOR_HAVE_METIS ON)
        if (GKLIB_LIBRARY)
            message(STATUS "METIS partitioning: enabled (${METIS_LIBRARY}, GKlib: ${GKLIB_LIBRARY})")
        else ()
            message(STATUS "METIS partitioning: enabled (${METIS_LIBRARY})")
        endif ()
    elseif (DVSTOR_METIS_PARTITION_MODE STREQUAL "ON")
        message(FATAL_ERROR "METIS partitioning requested but metis.h/libmetis were not found. Install libmetis-dev or set -DDVSTOR_METIS_PARTITION=OFF.")
    else ()
        set(DVSTOR_HAVE_METIS OFF)
        message(STATUS "METIS partitioning: disabled (metis.h/libmetis not found)")
    endif ()
else ()
    set(DVSTOR_HAVE_METIS OFF)
    message(STATUS "METIS partitioning: disabled")
endif ()

set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/thirdparty)
add_subdirectory(rdma-library)
add_subdirectory(thirdparty)

set(DVSTOR_MEMORY_NODE_SOURCES
    src/memory_node/memory_node.cc
    src/memory_node/peer_rdma.cc
    src/memory_node/peer_rpc.cc
    src/memory_node/storage_owner_handoff.cc
    src/memory_node/storage_owner_index.cc
    src/memory_node/storage_owner_runtime.cc
)

set(DVSTOR_OFFLINE_BUILDER_SOURCES
    tools/vamana_offline/config.cc
    tools/vamana_offline/dataset_io.cc
    tools/vamana_offline/graph.cc
    tools/vamana_offline/partitioning.cc
    tools/vamana_offline/progress.cc
    tools/vamana_offline/shard_writer.cc
)

set(DVSTOR_BREAKDOWN_BENCHMARK_SOURCES
    tools/breakdown_benchmark/args.cc
    tools/breakdown_benchmark/progress.cc
    tools/breakdown_benchmark/workload.cc
)

function(add_vamana_metis_repartitioner_target)
    add_executable(vamana_metis_repartitioner
        tools/vamana_metis_repartitioner.cc
        tools/vamana_offline/partitioning.cc
    )
    target_include_directories(vamana_metis_repartitioner PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/rdma-library
        ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty
    )
    target_link_libraries(vamana_metis_repartitioner rdma_library)
    target_link_metis(vamana_metis_repartitioner)
endfunction()

function(add_vamana_bfs_repartitioner_target)
    add_executable(vamana_bfs_repartitioner
        tools/vamana_bfs_repartitioner.cc
        tools/vamana_offline/partitioning.cc
    )
    target_include_directories(vamana_bfs_repartitioner PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/rdma-library
        ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty
    )
    target_link_libraries(vamana_bfs_repartitioner rdma_library)
endfunction()

function(target_link_metis target_name)
    if (DVSTOR_HAVE_METIS)
        target_include_directories(${target_name} PRIVATE ${METIS_INCLUDE_DIR})
        target_link_libraries(${target_name} ${METIS_LIBRARY})
        if (GKLIB_LIBRARY)
            target_link_libraries(${target_name} ${GKLIB_LIBRARY})
        endif ()
        target_compile_definitions(${target_name} PRIVATE DVSTOR_HAVE_METIS=1)
    else ()
        target_compile_definitions(${target_name} PRIVATE DVSTOR_HAVE_METIS=0)
    endif ()
endfunction()

if (DVSTOR_STORAGE_NODE_ONLY)
    add_executable(dvstor_memory_node
        src/memory_node_main.cc
        ${DVSTOR_MEMORY_NODE_SOURCES}
        src/common/core_assignment.cc
        src/common/timing.cc
        src/service/index_metadata.cc
    )
    target_include_directories(dvstor_memory_node PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/rdma-library
        ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty
    )
    target_link_libraries(dvstor_memory_node rdma_library)

    if (DVSTOR_BUILD_EXECUTABLES)
        add_vamana_metis_repartitioner_target()
        add_vamana_bfs_repartitioner_target()
    endif ()
else ()
    # GPU kernels library (compiled by nvcc)
    file(GLOB_RECURSE DVSTOR_GPU_SOURCES CONFIGURE_DEPENDS src/gpu/*.cu)
    if (DVSTOR_GPU_SOURCES)
        add_library(dvstor_gpu_kernels STATIC ${DVSTOR_GPU_SOURCES})
        target_include_directories(dvstor_gpu_kernels PUBLIC
            ${CMAKE_CURRENT_SOURCE_DIR}/src
            ${CMAKE_CURRENT_SOURCE_DIR}/rdma-library
            ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty
        )
        target_link_libraries(dvstor_gpu_kernels
            CUDA::cudart
        )
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

    add_library(dvstor_runtime STATIC ${DVSTOR_RUNTIME_SOURCES})
    target_include_directories(dvstor_runtime PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/rdma-library
        ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty
    )
    target_link_libraries(dvstor_runtime
        rdma_library
        CUDA::cudart
    )
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
    target_include_directories(dvstor_memory_node PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/rdma-library
        ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty
    )
    target_link_libraries(dvstor_memory_node rdma_library)

    if (DVSTOR_BUILD_EXECUTABLES)
        add_executable(dvstor src/main.cc)
        target_link_libraries(dvstor dvstor_runtime)

        add_executable(vamana_offline_builder
            tools/vamana_offline_builder.cc
            ${DVSTOR_OFFLINE_BUILDER_SOURCES}
        )
        target_include_directories(vamana_offline_builder PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}
            ${CMAKE_CURRENT_SOURCE_DIR}/src
            ${CMAKE_CURRENT_SOURCE_DIR}/rdma-library
            ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty
        )
        target_link_libraries(vamana_offline_builder
            rdma_library
        )
        target_link_metis(vamana_offline_builder)

        add_vamana_metis_repartitioner_target()
        add_vamana_bfs_repartitioner_target()

        add_executable(dvstor_breakdown_benchmark
            tools/dvstor_breakdown_benchmark.cc
            ${DVSTOR_BREAKDOWN_BENCHMARK_SOURCES}
        )
        target_include_directories(dvstor_breakdown_benchmark PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}
            ${CMAKE_CURRENT_SOURCE_DIR}/src
            ${CMAKE_CURRENT_SOURCE_DIR}/rdma-library
            ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty
        )
        target_link_libraries(dvstor_breakdown_benchmark dvstor_runtime)
    endif ()

    if (DVSTOR_BUILD_TESTS AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test/CMakeLists.txt")
        add_subdirectory(test)
    endif ()

    # status prints
    get_target_property(compile_options dvstor_runtime COMPILE_OPTIONS)
    message(STATUS "Compile options: ${compile_options}")
    message(STATUS "CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
endif ()
