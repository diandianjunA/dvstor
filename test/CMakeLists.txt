add_executable(DvstorIndexSmokeTest
    src/main.cpp
    src/dvstor_index.cpp
)

target_include_directories(DvstorIndexSmokeTest PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../src
)

target_link_libraries(DvstorIndexSmokeTest PRIVATE
    dvstor_runtime
)

add_executable(GpuRdmaProbe
    src/gpu_rdma_probe.c
)

target_include_directories(GpuRdmaProbe PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../src
    /opt/mellanox/doca/include
)

target_compile_definitions(GpuRdmaProbe PRIVATE
    ALLOW_EXPERIMENTAL_API
)

target_link_directories(GpuRdmaProbe PRIVATE
    /opt/mellanox/doca/lib/x86_64-linux-gnu
)

target_link_libraries(GpuRdmaProbe PRIVATE
    CUDA::cudart
    ibverbs
    mlx5
    doca_common
    doca_gpunetio
    doca_rdma
    doca_verbs
)
