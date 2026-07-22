add_executable(breakdown_aggregate_test breakdown_aggregate_test.cc)
target_link_libraries(breakdown_aggregate_test PRIVATE dvstor_runtime)
target_compile_options(breakdown_aggregate_test PRIVATE -UNDEBUG)
add_test(NAME breakdown_aggregate_test COMMAND breakdown_aggregate_test)

add_test(NAME mn_memory_estimator_test
         COMMAND bash ${CMAKE_SOURCE_DIR}/test/mn_memory_estimator_test.sh
                      ${CMAKE_SOURCE_DIR})

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

add_executable(vector_dtype_test vector_dtype_test.cc)
dvstor_target_common_includes(vector_dtype_test)
target_compile_options(vector_dtype_test PRIVATE -UNDEBUG)
add_test(NAME vector_dtype_test COMMAND vector_dtype_test)

add_executable(storage_owner_wire_size_test storage_owner_wire_size_test.cc)
dvstor_target_common_includes(storage_owner_wire_size_test)
target_compile_options(storage_owner_wire_size_test PRIVATE -UNDEBUG)
add_test(NAME storage_owner_wire_size_test COMMAND storage_owner_wire_size_test)

add_executable(storage_owner_graph_pointer_validation_test
    storage_owner_graph_pointer_validation_test.cc)
dvstor_target_common_includes(storage_owner_graph_pointer_validation_test)
target_compile_options(storage_owner_graph_pointer_validation_test PRIVATE -UNDEBUG)
add_test(NAME storage_owner_graph_pointer_validation_test
         COMMAND storage_owner_graph_pointer_validation_test)

add_executable(core_partition_test core_partition_test.cc)
dvstor_target_common_includes(core_partition_test)
target_compile_options(core_partition_test PRIVATE -UNDEBUG)
add_test(NAME core_partition_test COMMAND core_partition_test)

add_executable(peer_rdma_credit_policy_test peer_rdma_credit_policy_test.cc)
dvstor_target_common_includes(peer_rdma_credit_policy_test)
target_compile_options(peer_rdma_credit_policy_test PRIVATE -UNDEBUG)
add_test(NAME peer_rdma_credit_policy_test COMMAND peer_rdma_credit_policy_test)

add_executable(stage1_control_fanout_policy_test
    stage1_control_fanout_policy_test.cc)
dvstor_target_common_includes(stage1_control_fanout_policy_test)
target_compile_options(stage1_control_fanout_policy_test PRIVATE -UNDEBUG)
add_test(NAME stage1_control_fanout_policy_test
         COMMAND stage1_control_fanout_policy_test)

add_executable(stage2_score_round_robin_test
    stage2_score_round_robin_test.cc)
dvstor_target_common_includes(stage2_score_round_robin_test)
target_compile_options(stage2_score_round_robin_test PRIVATE -UNDEBUG)
add_test(NAME stage2_score_round_robin_test
         COMMAND stage2_score_round_robin_test)

add_executable(bounded_update_primitives_test bounded_update_primitives_test.cc)
target_link_libraries(bounded_update_primitives_test PRIVATE dvstor_runtime)
target_compile_options(bounded_update_primitives_test PRIVATE -UNDEBUG)
add_test(NAME bounded_update_primitives_test COMMAND bounded_update_primitives_test)

add_executable(compute_update_hot_path_test compute_update_hot_path_test.cc)
target_link_libraries(compute_update_hot_path_test PRIVATE dvstor_runtime)
target_compile_options(compute_update_hot_path_test PRIVATE -UNDEBUG)
add_test(NAME compute_update_hot_path_test COMMAND compute_update_hot_path_test)

add_executable(authority_directory_policy_test
    authority_directory_policy_test.cc)
dvstor_target_common_includes(authority_directory_policy_test)
target_compile_options(authority_directory_policy_test PRIVATE -UNDEBUG)
add_test(NAME authority_directory_policy_test
         COMMAND authority_directory_policy_test)

add_executable(idmap_v2_test idmap_v2_test.cc)
dvstor_target_common_includes(idmap_v2_test)
target_compile_options(idmap_v2_test PRIVATE -UNDEBUG)
add_test(NAME idmap_v2_test COMMAND idmap_v2_test)

if (TARGET FaissCPU::FaissCPU)
    add_executable(legacy_index_converter_test
        legacy_index_converter_test.cc
        ${CMAKE_SOURCE_DIR}/tools/vamana_offline/legacy_index_converter.cc
        ${CMAKE_SOURCE_DIR}/tools/vamana_offline/pq_indexer.cc
        ${CMAKE_SOURCE_DIR}/src/gpu_search/index_format.cc
        ${CMAKE_SOURCE_DIR}/src/gpu_search/pq_index.cc)
    dvstor_target_tool_includes(legacy_index_converter_test)
    target_link_libraries(legacy_index_converter_test PRIVATE
        rdma_library FaissCPU::FaissCPU)
    target_compile_options(legacy_index_converter_test PRIVATE -UNDEBUG)
    add_test(NAME legacy_index_converter_test
             COMMAND legacy_index_converter_test)
endif ()

add_executable(dynamic_allocation_receipt_policy_test
    dynamic_allocation_receipt_policy_test.cc)
dvstor_target_common_includes(dynamic_allocation_receipt_policy_test)
target_compile_options(dynamic_allocation_receipt_policy_test PRIVATE -UNDEBUG)
add_test(NAME dynamic_allocation_receipt_policy_test
         COMMAND dynamic_allocation_receipt_policy_test)

add_executable(centroid_router_test
    centroid_router_test.cc
    ${CMAKE_SOURCE_DIR}/src/vamana/centroid_router.cc)
dvstor_target_common_includes(centroid_router_test)
target_link_libraries(centroid_router_test PRIVATE rdma_library)
target_compile_options(centroid_router_test PRIVATE -UNDEBUG)
add_test(NAME centroid_router_test COMMAND centroid_router_test)

add_executable(centroid_home_selector_test centroid_home_selector_test.cc)
dvstor_target_common_includes(centroid_home_selector_test)
target_compile_options(centroid_home_selector_test PRIVATE -UNDEBUG)
add_test(NAME centroid_home_selector_test COMMAND centroid_home_selector_test)

add_executable(centroid_route_poll_policy_test
    centroid_route_poll_policy_test.cc)
dvstor_target_common_includes(centroid_route_poll_policy_test)
target_compile_options(centroid_route_poll_policy_test PRIVATE -UNDEBUG)
add_test(NAME centroid_route_poll_policy_test
         COMMAND centroid_route_poll_policy_test)

add_executable(centroid_membership_lifecycle_test
    centroid_membership_lifecycle_test.cc)
dvstor_target_common_includes(centroid_membership_lifecycle_test)
target_compile_options(centroid_membership_lifecycle_test PRIVATE -UNDEBUG)
add_test(NAME centroid_membership_lifecycle_test
         COMMAND centroid_membership_lifecycle_test)

add_executable(centroid_seed_policy_test centroid_seed_policy_test.cc)
dvstor_target_common_includes(centroid_seed_policy_test)
target_compile_options(centroid_seed_policy_test PRIVATE -UNDEBUG)
add_test(NAME centroid_seed_policy_test COMMAND centroid_seed_policy_test)

add_executable(centroid_route_ranking_test centroid_route_ranking_test.cc)
dvstor_target_common_includes(centroid_route_ranking_test)
target_compile_options(centroid_route_ranking_test PRIVATE -UNDEBUG)
add_test(NAME centroid_route_ranking_test COMMAND centroid_route_ranking_test)

add_executable(storage_owner_reconcile_reverse_test
    storage_owner_reconcile_reverse_test.cc)
dvstor_target_common_includes(storage_owner_reconcile_reverse_test)
target_link_libraries(storage_owner_reconcile_reverse_test PRIVATE rdma_library)
target_compile_options(storage_owner_reconcile_reverse_test PRIVATE -UNDEBUG)
add_test(NAME storage_owner_reconcile_reverse_test
         COMMAND storage_owner_reconcile_reverse_test)

add_executable(stage1_reachability_policy_test
    stage1_reachability_policy_test.cc)
dvstor_target_common_includes(stage1_reachability_policy_test)
target_compile_options(stage1_reachability_policy_test PRIVATE -UNDEBUG)
add_test(NAME stage1_reachability_policy_test
         COMMAND stage1_reachability_policy_test)

add_executable(configuration_update_protocol_test configuration_update_protocol_test.cc)
dvstor_target_common_includes(configuration_update_protocol_test)
target_link_libraries(configuration_update_protocol_test PRIVATE rdma_library)
target_compile_options(configuration_update_protocol_test PRIVATE -UNDEBUG)
add_test(NAME configuration_update_protocol_test COMMAND configuration_update_protocol_test)

add_executable(partition_local_search_test partition_local_search_test.cc)
dvstor_target_common_includes(partition_local_search_test)
target_compile_options(partition_local_search_test PRIVATE -UNDEBUG)
add_test(NAME partition_local_search_test COMMAND partition_local_search_test)

add_executable(persistent_grid_plan_test persistent_grid_plan_test.cc)
dvstor_target_common_includes(persistent_grid_plan_test)
target_compile_options(persistent_grid_plan_test PRIVATE -UNDEBUG)
add_test(NAME persistent_grid_plan_test COMMAND persistent_grid_plan_test)

add_executable(persistent_owner_watchdog_test
    persistent_owner_watchdog_test.cc)
dvstor_target_common_includes(persistent_owner_watchdog_test)
target_compile_options(persistent_owner_watchdog_test PRIVATE -UNDEBUG)
add_test(NAME persistent_owner_watchdog_test
         COMMAND persistent_owner_watchdog_test)

add_executable(maintenance_fence_test maintenance_fence_test.cc)
dvstor_target_common_includes(maintenance_fence_test)
target_compile_options(maintenance_fence_test PRIVATE -UNDEBUG)
add_test(NAME maintenance_fence_test COMMAND maintenance_fence_test)

add_executable(maintenance_telemetry_test maintenance_telemetry_test.cc)
dvstor_target_common_includes(maintenance_telemetry_test)
target_compile_options(maintenance_telemetry_test PRIVATE -UNDEBUG)
add_test(NAME maintenance_telemetry_test COMMAND maintenance_telemetry_test)

if (TARGET dvstor_gpu_kernels)
    add_executable(gpu_centroid_route_ranking_test
        gpu_centroid_route_ranking_test.cu)
    dvstor_target_common_includes(gpu_centroid_route_ranking_test)
    target_link_libraries(gpu_centroid_route_ranking_test PRIVATE CUDA::cudart)
    target_compile_options(gpu_centroid_route_ranking_test PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda>)
    add_test(NAME gpu_centroid_route_ranking_test
        COMMAND gpu_centroid_route_ranking_test)
    set_tests_properties(gpu_centroid_route_ranking_test PROPERTIES
        TIMEOUT 15
        SKIP_REGULAR_EXPRESSION "^SKIP:")

    add_executable(gpu_compact_beam_merge_test
        gpu_compact_beam_merge_test.cu)
    dvstor_target_common_includes(gpu_compact_beam_merge_test)
    target_link_libraries(gpu_compact_beam_merge_test PRIVATE CUDA::cudart)
    target_compile_options(gpu_compact_beam_merge_test PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda>)
    add_test(NAME gpu_compact_beam_merge_test
        COMMAND gpu_compact_beam_merge_test)
    set_tests_properties(gpu_compact_beam_merge_test PROPERTIES
        TIMEOUT 15
        SKIP_REGULAR_EXPRESSION "^SKIP:")

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
    add_test(NAME gpu_persistent_dispatch_256_test
        COMMAND gpu_persistent_dispatch_test 128 10 8 32 256)
    set_tests_properties(gpu_persistent_dispatch_256_test PROPERTIES
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

add_executable(storage_reclaim_test storage_reclaim_test.cc)
target_link_libraries(storage_reclaim_test PRIVATE dvstor_runtime)
target_compile_options(storage_reclaim_test PRIVATE -UNDEBUG)
add_test(NAME storage_reclaim_test COMMAND storage_reclaim_test)

find_package(Threads REQUIRED)
add_executable(incarnation_lock_test incarnation_lock_test.cc)
dvstor_target_common_includes(incarnation_lock_test)
target_link_libraries(incarnation_lock_test PRIVATE Threads::Threads)
target_compile_options(incarnation_lock_test PRIVATE -UNDEBUG)
add_test(NAME incarnation_lock_test COMMAND incarnation_lock_test)

add_executable(stage2_remote_publication_test
    stage2_remote_publication_test.cc)
dvstor_target_common_includes(stage2_remote_publication_test)
target_compile_options(stage2_remote_publication_test PRIVATE -UNDEBUG)
add_test(NAME stage2_remote_publication_test
         COMMAND stage2_remote_publication_test)

add_executable(storage_owner_protocol_test storage_owner_protocol_test.cc)
target_link_libraries(storage_owner_protocol_test PRIVATE dvstor_runtime)
target_compile_options(storage_owner_protocol_test PRIVATE -UNDEBUG)
add_test(NAME storage_owner_protocol_test COMMAND storage_owner_protocol_test)

add_executable(explicit_release_replay_test
    explicit_release_replay_test.cc)
dvstor_target_common_includes(explicit_release_replay_test)
target_compile_options(explicit_release_replay_test PRIVATE -UNDEBUG)
add_test(NAME explicit_release_replay_test
         COMMAND explicit_release_replay_test)

add_executable(storage_owner_stage2_tracker_test
    storage_owner_stage2_tracker_test.cc)
target_compile_options(storage_owner_stage2_tracker_test PRIVATE -UNDEBUG)
add_test(NAME storage_owner_stage2_tracker_test
    COMMAND storage_owner_stage2_tracker_test)

add_executable(stage2_reconcile_batch_state_test
    stage2_reconcile_batch_state_test.cc)
dvstor_target_common_includes(stage2_reconcile_batch_state_test)
target_compile_options(stage2_reconcile_batch_state_test PRIVATE -UNDEBUG)
add_test(NAME stage2_reconcile_batch_state_test
    COMMAND stage2_reconcile_batch_state_test)

add_executable(stage2_batch_policy_test stage2_batch_policy_test.cc)
dvstor_target_common_includes(stage2_batch_policy_test)
target_compile_options(stage2_batch_policy_test PRIVATE -UNDEBUG)
add_test(NAME stage2_batch_policy_test COMMAND stage2_batch_policy_test)

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
