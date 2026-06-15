add_executable(rabitq_encoding_test
    rabitq_encoding_test.cc
)
target_link_libraries(rabitq_encoding_test PRIVATE dvstor_gpu_kernels rdma_library CUDA::cudart)
target_include_directories(rabitq_encoding_test PRIVATE
    ${PROJECT_SOURCE_DIR}/src
    ${PROJECT_SOURCE_DIR}/rdma-library
    ${PROJECT_SOURCE_DIR}/thirdparty
)
add_test(NAME rabitq_encoding_test
         COMMAND rabitq_encoding_test)

add_executable(rabitq_cache_test
    rabitq_cache_test.cc
)
target_link_libraries(rabitq_cache_test PRIVATE rdma_library)
target_include_directories(rabitq_cache_test PRIVATE
    ${PROJECT_SOURCE_DIR}/src
    ${PROJECT_SOURCE_DIR}/rdma-library
    ${PROJECT_SOURCE_DIR}/thirdparty
)
add_test(NAME rabitq_cache_test
         COMMAND rabitq_cache_test)

add_executable(storage_layout_resolver_test
    storage_layout_resolver_test.cc
)
target_link_libraries(storage_layout_resolver_test PRIVATE rdma_library)
target_include_directories(storage_layout_resolver_test PRIVATE
    ${PROJECT_SOURCE_DIR}/src
    ${PROJECT_SOURCE_DIR}/rdma-library
    ${PROJECT_SOURCE_DIR}/thirdparty
)
add_test(NAME storage_layout_resolver_test
         COMMAND storage_layout_resolver_test)

add_executable(anchor_update_protocol_test
    anchor_update_protocol_test.cc
    ${PROJECT_SOURCE_DIR}/src/vamana/anchor_index.cc
)
target_link_libraries(anchor_update_protocol_test PRIVATE rdma_library)
target_include_directories(anchor_update_protocol_test PRIVATE
    ${PROJECT_SOURCE_DIR}/src
    ${PROJECT_SOURCE_DIR}/rdma-library
    ${PROJECT_SOURCE_DIR}/thirdparty
)
add_test(NAME anchor_update_protocol_test
         COMMAND anchor_update_protocol_test)

add_executable(vector_batch_planner_test
    vector_batch_planner_test.cc
)
target_link_libraries(vector_batch_planner_test PRIVATE rdma_library)
target_include_directories(vector_batch_planner_test PRIVATE
    ${PROJECT_SOURCE_DIR}/src
    ${PROJECT_SOURCE_DIR}/rdma-library
    ${PROJECT_SOURCE_DIR}/thirdparty
)
add_test(NAME vector_batch_planner_test
         COMMAND vector_batch_planner_test)
