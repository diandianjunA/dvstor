add_executable(breakdown_aggregate_test breakdown_aggregate_test.cc)
target_link_libraries(breakdown_aggregate_test PRIVATE dvstor_runtime)
target_compile_options(breakdown_aggregate_test PRIVATE -UNDEBUG)
add_test(NAME breakdown_aggregate_test COMMAND breakdown_aggregate_test)

add_executable(breakdown_benchmark_support_test breakdown_benchmark_support_test.cc)
dvstor_target_tool_includes(breakdown_benchmark_support_test)
target_link_libraries(breakdown_benchmark_support_test PRIVATE dvstor_breakdown_support)
target_compile_options(breakdown_benchmark_support_test PRIVATE -UNDEBUG)
add_test(NAME breakdown_benchmark_support_test COMMAND breakdown_benchmark_support_test)

add_executable(breakdown_benchmark_control_test
    breakdown_benchmark_control_test.cc
    ${CMAKE_SOURCE_DIR}/tools/breakdown_benchmark/args.cc
    ${CMAKE_SOURCE_DIR}/tools/breakdown_benchmark/progress.cc)
dvstor_target_tool_includes(breakdown_benchmark_control_test)
target_link_libraries(breakdown_benchmark_control_test PRIVATE dvstor_breakdown_support)
target_compile_options(breakdown_benchmark_control_test PRIVATE -UNDEBUG)
add_test(NAME breakdown_benchmark_control_test COMMAND breakdown_benchmark_control_test)

add_executable(common_atomic_utils_test common_atomic_utils_test.cc)
target_link_libraries(common_atomic_utils_test PRIVATE dvstor_runtime)
target_compile_options(common_atomic_utils_test PRIVATE -UNDEBUG)
add_test(NAME common_atomic_utils_test COMMAND common_atomic_utils_test)

add_executable(bounded_update_primitives_test bounded_update_primitives_test.cc)
target_link_libraries(bounded_update_primitives_test PRIVATE dvstor_runtime)
target_compile_options(bounded_update_primitives_test PRIVATE -UNDEBUG)
add_test(NAME bounded_update_primitives_test COMMAND bounded_update_primitives_test)

add_executable(base_owner_map_test
    base_owner_map_test.cc
    ${CMAKE_SOURCE_DIR}/src/service/base_owner_map.cc)
dvstor_target_common_includes(base_owner_map_test)
target_compile_options(base_owner_map_test PRIVATE -UNDEBUG)
add_test(NAME base_owner_map_test COMMAND base_owner_map_test)

if (TARGET dvstor_gpu_kernels)
    add_executable(gpu_persistent_dispatch_test gpu_persistent_dispatch_test.cc)
    target_link_libraries(gpu_persistent_dispatch_test PRIVATE dvstor_runtime)
    add_test(NAME gpu_persistent_dispatch_test COMMAND gpu_persistent_dispatch_test)
    set_tests_properties(gpu_persistent_dispatch_test PROPERTIES
        TIMEOUT 15
        SKIP_REGULAR_EXPRESSION "^SKIP:")
    add_test(NAME gpu_persistent_dispatch_pq32_test
        COMMAND gpu_persistent_dispatch_test 1024 10 8 32)
    set_tests_properties(gpu_persistent_dispatch_pq32_test PROPERTIES
        TIMEOUT 15
        SKIP_REGULAR_EXPRESSION "^SKIP:")
endif ()

add_executable(gpu_navigation_layout_test gpu_navigation_layout_test.cc)
target_link_libraries(gpu_navigation_layout_test PRIVATE dvstor_runtime)
target_compile_options(gpu_navigation_layout_test PRIVATE -UNDEBUG)
add_test(NAME gpu_navigation_layout_test COMMAND gpu_navigation_layout_test)

add_executable(gpu_pq_index_test gpu_pq_index_test.cc)
target_link_libraries(gpu_pq_index_test PRIVATE dvstor_runtime)
target_compile_options(gpu_pq_index_test PRIVATE -UNDEBUG)
add_test(NAME gpu_pq_index_test COMMAND gpu_pq_index_test)

add_executable(gpu_delta_index_test gpu_delta_index_test.cc)
target_link_libraries(gpu_delta_index_test PRIVATE dvstor_runtime)
target_compile_options(gpu_delta_index_test PRIVATE -UNDEBUG)
add_test(NAME gpu_delta_index_test COMMAND gpu_delta_index_test)

add_executable(storage_reclaim_test storage_reclaim_test.cc)
target_link_libraries(storage_reclaim_test PRIVATE dvstor_runtime)
target_compile_options(storage_reclaim_test PRIVATE -UNDEBUG)
add_test(NAME storage_reclaim_test COMMAND storage_reclaim_test)

add_executable(storage_owner_protocol_test storage_owner_protocol_test.cc)
target_link_libraries(storage_owner_protocol_test PRIVATE dvstor_runtime)
target_compile_options(storage_owner_protocol_test PRIVATE -UNDEBUG)
add_test(NAME storage_owner_protocol_test COMMAND storage_owner_protocol_test)

add_executable(storage_owner_stage2_tracker_test
    storage_owner_stage2_tracker_test.cc)
target_compile_options(storage_owner_stage2_tracker_test PRIVATE -UNDEBUG)
add_test(NAME storage_owner_stage2_tracker_test
    COMMAND storage_owner_stage2_tracker_test)

add_executable(storage_owner_reverse_outbox_test
    storage_owner_reverse_outbox_test.cc)
dvstor_target_common_includes(storage_owner_reverse_outbox_test)
target_compile_options(storage_owner_reverse_outbox_test PRIVATE -UNDEBUG)
add_test(NAME storage_owner_reverse_outbox_test
    COMMAND storage_owner_reverse_outbox_test)

add_executable(peer_async_response_registry_test
    peer_async_response_registry_test.cc)
dvstor_target_common_includes(peer_async_response_registry_test)
target_compile_options(peer_async_response_registry_test PRIVATE -UNDEBUG)
add_test(NAME peer_async_response_registry_test
    COMMAND peer_async_response_registry_test)

add_executable(gpu_memory_budget_test gpu_memory_budget_test.cc)
target_link_libraries(gpu_memory_budget_test PRIVATE dvstor_runtime)
target_compile_options(gpu_memory_budget_test PRIVATE -UNDEBUG)
add_test(NAME gpu_memory_budget_test COMMAND gpu_memory_budget_test)

add_executable(legacy_index_migrator_test
    legacy_index_migrator_test.cc
    ${CMAKE_SOURCE_DIR}/tools/legacy_index/migrator.cc
)
dvstor_target_tool_includes(legacy_index_migrator_test)
target_link_libraries(legacy_index_migrator_test PRIVATE dvstor_runtime)
target_compile_options(legacy_index_migrator_test PRIVATE -UNDEBUG)
add_test(NAME legacy_index_migrator_test COMMAND legacy_index_migrator_test)
