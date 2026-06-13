add_executable(storage_owner_handoff_protocol_test
    storage_owner_handoff_protocol_test.cc
)
target_include_directories(storage_owner_handoff_protocol_test PRIVATE
    ${PROJECT_SOURCE_DIR}/src
    ${PROJECT_SOURCE_DIR}/rdma-library
    ${PROJECT_SOURCE_DIR}/thirdparty
)
add_test(NAME storage_owner_handoff_protocol_test
         COMMAND storage_owner_handoff_protocol_test)

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
