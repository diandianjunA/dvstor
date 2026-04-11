cmake_minimum_required(VERSION 3.18)
project(dvstor C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_FLAGS_DEBUG "-g -fno-omit-frame-pointer -fno-inline -fsanitize=address")

# set default build type to Release
if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif ()

option(DVSTOR_STORAGE_NODE_ONLY "Build only the storage/memory node artifact without CUDA, GPU, or DOCA GPUNetIO" OFF)
option(DVSTOR_BUILD_EXECUTABLES "Build standalone DVSTOR executables and tools" ON)
option(DVSTOR_BUILD_TESTS "Build local DVSTOR smoke tests" ON)

if (DVSTOR_STORAGE_NODE_ONLY)
    set(DVSTOR_BUILD_EXECUTABLES OFF CACHE BOOL "Build standalone DVSTOR executables and tools" FORCE)
    set(DVSTOR_BUILD_TESTS OFF CACHE BOOL "Build local DVSTOR smoke tests" FORCE)
else ()
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    # GPUNetIO device-side code requires sm_60+ atomics. Keep the project on modern datacenter targets.
    set(CMAKE_CUDA_ARCHITECTURES 80 86 89 90 CACHE STRING "CUDA architectures" FORCE)
endif ()

function(dvstor_ignore_prefix prefix)
    if (prefix AND IS_DIRECTORY "${prefix}")
        list(APPEND CMAKE_IGNORE_PREFIX_PATH "${prefix}")
        list(APPEND CMAKE_IGNORE_PATH
            "${prefix}/bin"
            "${prefix}/include"
            "${prefix}/lib"
            "${prefix}/lib/cmake"
            "${prefix}/lib64"
        )
        set(CMAKE_IGNORE_PREFIX_PATH "${CMAKE_IGNORE_PREFIX_PATH}" PARENT_SCOPE)
        set(CMAKE_IGNORE_PATH "${CMAKE_IGNORE_PATH}" PARENT_SCOPE)
    endif ()
endfunction()

# Keep host builds independent of an activated Conda environment. Mixing Conda
# Boost/TBB with the system compiler frequently selects an incompatible C++ ABI.
dvstor_ignore_prefix("$ENV{CONDA_PREFIX}")
dvstor_ignore_prefix("/home/xjs/anaconda3")
dvstor_ignore_prefix("/home/lym/anaconda3")
unset(Boost_DIR CACHE)
unset(Boost_INCLUDE_DIR CACHE)
unset(Boost_LIBRARY_DIR_RELEASE CACHE)
unset(Boost_PROGRAM_OPTIONS_LIBRARY_RELEASE CACHE)
unset(boost_headers_DIR CACHE)
unset(boost_program_options_DIR CACHE)
unset(TBB_DIR CACHE)

if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    execute_process(
        COMMAND ${CMAKE_CXX_COMPILER} -print-file-name=libstdc++.so
        OUTPUT_VARIABLE DVSTOR_LIBSTDCXX
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    get_filename_component(DVSTOR_LIBSTDCXX_REALPATH "${DVSTOR_LIBSTDCXX}" REALPATH)
    get_filename_component(DVSTOR_LIBSTDCXX_DIR "${DVSTOR_LIBSTDCXX}" DIRECTORY)
    get_filename_component(DVSTOR_LIBSTDCXX_RUNTIME_DIR "${DVSTOR_LIBSTDCXX_REALPATH}" DIRECTORY)
    if (DVSTOR_LIBSTDCXX_DIR AND IS_DIRECTORY "${DVSTOR_LIBSTDCXX_DIR}")
        link_directories(BEFORE "${DVSTOR_LIBSTDCXX_DIR}")
        add_link_options(
            "SHELL:-L${DVSTOR_LIBSTDCXX_DIR}"
            "SHELL:-Wl,--disable-new-dtags"
            "SHELL:-Wl,-rpath,${DVSTOR_LIBSTDCXX_DIR}"
        )
    endif ()
    if (DVSTOR_LIBSTDCXX_RUNTIME_DIR AND IS_DIRECTORY "${DVSTOR_LIBSTDCXX_RUNTIME_DIR}")
        add_link_options("SHELL:-Wl,-rpath,${DVSTOR_LIBSTDCXX_RUNTIME_DIR}")
    endif ()
endif ()

# CXX compiler flags (not applied to CUDA files)
add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-Wall>)
add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-Wextra>)
add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-Wpedantic>)
add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-Wfatal-errors>)
add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-march=native>)
add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-ffast-math>)
add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-mavx2>)
add_compile_options(-DNOHUGEPAGES)
#add_compile_options(-g -fno-omit-frame-pointer -fno-inline)  # for profiling

# Find dependencies
if (NOT DVSTOR_STORAGE_NODE_ONLY)
    find_package(CUDAToolkit 11.8 REQUIRED)
    find_package(Eigen3 3.3 REQUIRED NO_MODULE)

    set(DOCA_INCLUDE_DIR "/opt/mellanox/doca/include")
    set(DOCA_LIBRARY_DIR "/opt/mellanox/doca/lib/x86_64-linux-gnu")
endif ()

set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/thirdparty)
add_subdirectory(rdma-library)
add_subdirectory(thirdparty)

add_executable(dvstor_memory_node
    src/memory_node_main.cc
    src/common/core_assignment.cc
    src/common/timing.cc
)
target_include_directories(dvstor_memory_node PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/rdma-library
    ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty
)
target_link_libraries(dvstor_memory_node PRIVATE rdma_library)
set_target_properties(dvstor_memory_node PROPERTIES LINKER_LANGUAGE CXX)

if (NOT DVSTOR_STORAGE_NODE_ONLY)

# GPU kernels library (compiled by nvcc)
file(GLOB_RECURSE DVSTOR_GPU_SOURCES CONFIGURE_DEPENDS src/gpu/*.cu)
if (DVSTOR_GPU_SOURCES)
    add_library(dvstor_gpu_kernels STATIC ${DVSTOR_GPU_SOURCES})
    target_include_directories(dvstor_gpu_kernels PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty
        ${DOCA_INCLUDE_DIR}
    )
    target_link_libraries(dvstor_gpu_kernels
        CUDA::cudart
        CUDA::cublas
        Eigen3::Eigen
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
list(REMOVE_ITEM DVSTOR_RUNTIME_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/main.cc")
list(REMOVE_ITEM DVSTOR_RUNTIME_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/memory_node_main.cc")

add_library(dvstor_runtime STATIC ${DVSTOR_RUNTIME_SOURCES})
target_include_directories(dvstor_runtime PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/rdma-library
    ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty
    ${DOCA_INCLUDE_DIR}
)
target_link_directories(dvstor_runtime PUBLIC ${DOCA_LIBRARY_DIR})
target_link_libraries(dvstor_runtime
    rdma_library
    CUDA::cudart
    Eigen3::Eigen
    doca_gpunetio
    doca_verbs
    doca_common
    doca_rdma
    mlx5
)
if (DVSTOR_GPU_SOURCES)
    target_link_libraries(dvstor_runtime dvstor_gpu_kernels)
endif ()
target_compile_options(dvstor_runtime PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-fcoroutines>)
set_target_properties(dvstor_runtime PROPERTIES LINKER_LANGUAGE CXX)

if (DVSTOR_BUILD_EXECUTABLES)
    add_executable(dvstor src/main.cc)
    target_link_libraries(dvstor dvstor_runtime)
    set_target_properties(dvstor PROPERTIES LINKER_LANGUAGE CXX)

    add_executable(vamana_offline_builder tools/vamana_offline_builder.cc)
    target_include_directories(vamana_offline_builder PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/rdma-library
        ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty
    )
    target_link_libraries(vamana_offline_builder rdma_library Eigen3::Eigen dvstor_gpu_kernels CUDA::cudart CUDA::cublas)

    add_executable(dvstor_breakdown_benchmark tools/dvstor_breakdown_benchmark.cc)
    target_include_directories(dvstor_breakdown_benchmark PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/rdma-library
        ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty
    )
    target_link_libraries(dvstor_breakdown_benchmark dvstor_runtime)
endif ()

if (DVSTOR_BUILD_TESTS)
    add_subdirectory(test)
endif ()

# status prints
get_target_property(compile_options dvstor_runtime COMPILE_OPTIONS)
message(STATUS "Compile options: ${compile_options}")
message(STATUS "CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
else ()
    message(STATUS "Storage-node-only build: CUDA, GPU kernels, compute service, and DOCA GPUNetIO are disabled")
endif ()
