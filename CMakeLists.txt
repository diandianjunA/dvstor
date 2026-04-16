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
endif ()

set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/thirdparty)
add_subdirectory(rdma-library)
add_subdirectory(thirdparty)

if (DVSTOR_STORAGE_NODE_ONLY)
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
    target_link_libraries(dvstor_memory_node rdma_library)
else ()
    # GPU kernels library (compiled by nvcc)
    file(GLOB_RECURSE DVSTOR_GPU_SOURCES CONFIGURE_DEPENDS src/gpu/*.cu)
    if (DVSTOR_GPU_SOURCES)
        add_library(dvstor_gpu_kernels STATIC ${DVSTOR_GPU_SOURCES})
        target_include_directories(dvstor_gpu_kernels PUBLIC
            ${CMAKE_CURRENT_SOURCE_DIR}/src
            ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty
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
        Eigen3::Eigen
    )
    if (DVSTOR_GPU_SOURCES)
        target_link_libraries(dvstor_runtime dvstor_gpu_kernels)
    endif ()
    target_compile_options(dvstor_runtime PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-fcoroutines>)

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
    target_link_libraries(dvstor_memory_node rdma_library)

    if (DVSTOR_BUILD_EXECUTABLES)
        add_executable(dvstor src/main.cc)
        target_link_libraries(dvstor dvstor_runtime)

        add_executable(vamana_offline_builder tools/vamana_offline_builder.cc)
        target_include_directories(vamana_offline_builder PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/src
            ${CMAKE_CURRENT_SOURCE_DIR}/rdma-library
            ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty
        )
        target_link_libraries(vamana_offline_builder
            rdma_library
            Eigen3::Eigen
            dvstor_gpu_kernels
            CUDA::cudart
            CUDA::cublas
        )

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
endif ()
