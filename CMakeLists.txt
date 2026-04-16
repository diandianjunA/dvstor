cmake_minimum_required(VERSION 3.18)
project(rdma-vamana CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_FLAGS_DEBUG "-g -fno-omit-frame-pointer -fno-inline -fsanitize=address")

# CUDA standard
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Auto-detect GPU architecture if not specified
if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 80 86 89 90)
endif ()

# set default build type to Release
if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif ()

# prefer the Anaconda stdc++/gcc build so we satisfy CXXABI_1.3.15 dependencies from libtbb
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -L/home/xjs/anaconda3/lib -Wl,-rpath,/home/xjs/anaconda3/lib")

option(SHINE_BUILD_EXECUTABLES "Build standalone SHINE executables and tools" ON)
option(SHINE_BUILD_TESTS "Build local SHINE smoke tests" ON)

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

# Find CUDA and dependencies
find_package(CUDAToolkit 11.8 REQUIRED)
find_package(Eigen3 3.3 REQUIRED NO_MODULE)

set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/thirdparty)
add_subdirectory(rdma-library)
add_subdirectory(thirdparty)

# GPU kernels library (compiled by nvcc)
file(GLOB_RECURSE SHINE_GPU_SOURCES CONFIGURE_DEPENDS src/gpu/*.cu)
if (SHINE_GPU_SOURCES)
    add_library(shine_gpu_kernels STATIC ${SHINE_GPU_SOURCES})
    target_include_directories(shine_gpu_kernels PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty
    )
    target_link_libraries(shine_gpu_kernels
        CUDA::cudart
        CUDA::cublas
        Eigen3::Eigen
    )
    set_target_properties(shine_gpu_kernels PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
        POSITION_INDEPENDENT_CODE ON
    )
    target_compile_options(shine_gpu_kernels PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda>
        $<$<COMPILE_LANGUAGE:CUDA>:-O3>
    )
endif ()

# Main runtime library (compiled by host compiler, C++20 with coroutines)
file(GLOB_RECURSE SHINE_RUNTIME_SOURCES CONFIGURE_DEPENDS src/*.cc)
list(REMOVE_ITEM SHINE_RUNTIME_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/main.cc")

add_library(shine_runtime STATIC ${SHINE_RUNTIME_SOURCES})
target_include_directories(shine_runtime PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/rdma-library
    ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty
)
target_link_libraries(shine_runtime
    rdma_library
    CUDA::cudart
    Eigen3::Eigen
)
if (SHINE_GPU_SOURCES)
    target_link_libraries(shine_runtime shine_gpu_kernels)
endif ()
target_compile_options(shine_runtime PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-fcoroutines>)

if (SHINE_BUILD_EXECUTABLES)
    add_executable(shine src/main.cc)
    target_link_libraries(shine shine_runtime)

    add_executable(vamana_offline_builder tools/vamana_offline_builder.cc)
    target_include_directories(vamana_offline_builder PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/rdma-library
        ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty
    )
    target_link_libraries(vamana_offline_builder rdma_library Eigen3::Eigen shine_gpu_kernels CUDA::cudart CUDA::cublas)

    add_executable(shine_breakdown_benchmark tools/shine_breakdown_benchmark.cc)
    target_include_directories(shine_breakdown_benchmark PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/rdma-library
        ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty
    )
    target_link_libraries(shine_breakdown_benchmark shine_runtime)
endif ()

if (SHINE_BUILD_TESTS)
    add_subdirectory(test)
endif ()

# status prints
get_target_property(compile_options shine_runtime COMPILE_OPTIONS)
message(STATUS "Compile options: ${compile_options}")
message(STATUS "CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
