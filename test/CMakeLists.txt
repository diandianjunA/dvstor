include(DvstorTestHelpers)
include(cpu_tests.cmake)

dvstor_add_test(breakdown_aggregate_test)
target_link_libraries(breakdown_aggregate_test PRIVATE dvstor_runtime)

add_test(NAME mn_memory_estimator_test
         COMMAND bash ${CMAKE_SOURCE_DIR}/test/mn_memory_estimator_test.sh
                      ${CMAKE_SOURCE_DIR})

add_test(NAME experiment_profile_contract_test
         COMMAND bash
                 ${CMAKE_SOURCE_DIR}/test/experiment_profile_contract_test.sh)

add_test(NAME spacev_data_preparation_test
         COMMAND python3
                 ${CMAKE_SOURCE_DIR}/test/spacev_data_preparation_test.py)

add_test(NAME deep_data_preparation_test
         COMMAND python3
                 ${CMAKE_SOURCE_DIR}/test/deep_data_preparation_test.py)

dvstor_add_test(breakdown_benchmark_support_test)
dvstor_target_tool_includes(breakdown_benchmark_support_test)
target_link_libraries(breakdown_benchmark_support_test PRIVATE dvstor_breakdown_support)

dvstor_add_test(breakdown_benchmark_control_test
    breakdown_benchmark_control_test.cc
    ${CMAKE_SOURCE_DIR}/tools/breakdown_benchmark/args.cc
    ${CMAKE_SOURCE_DIR}/tools/breakdown_benchmark/progress.cc)
dvstor_target_tool_includes(breakdown_benchmark_control_test)
target_link_libraries(breakdown_benchmark_control_test PRIVATE dvstor_breakdown_support)

dvstor_add_test(common_atomic_utils_test)
target_link_libraries(common_atomic_utils_test PRIVATE dvstor_runtime)

dvstor_add_test(adaptive_frontier_policy_test)
dvstor_target_common_includes(adaptive_frontier_policy_test)
target_link_libraries(adaptive_frontier_policy_test PRIVATE rdma_library)

dvstor_add_test(storage_owner_wire_size_test)
dvstor_target_common_includes(storage_owner_wire_size_test)

dvstor_add_test(storage_owner_graph_pointer_validation_test)
dvstor_target_common_includes(storage_owner_graph_pointer_validation_test)

dvstor_add_test(core_partition_test)
dvstor_target_common_includes(core_partition_test)

dvstor_add_test(peer_rdma_credit_policy_test)
dvstor_target_common_includes(peer_rdma_credit_policy_test)

dvstor_add_test(stage1_control_fanout_policy_test)
dvstor_target_common_includes(stage1_control_fanout_policy_test)

dvstor_add_test(stage2_score_round_robin_test)
dvstor_target_common_includes(stage2_score_round_robin_test)

dvstor_add_test(bounded_update_primitives_test)
target_link_libraries(bounded_update_primitives_test PRIVATE dvstor_runtime)

dvstor_add_test(compute_update_hot_path_test)
target_link_libraries(compute_update_hot_path_test PRIVATE dvstor_runtime)

dvstor_add_test(authority_directory_policy_test)
dvstor_target_common_includes(authority_directory_policy_test)

dvstor_add_test(idmap_v2_test)
dvstor_target_common_includes(idmap_v2_test)

if (TARGET FaissCPU::FaissCPU)
    dvstor_add_test(legacy_index_converter_test
        legacy_index_converter_test.cc
        ${CMAKE_SOURCE_DIR}/tools/vamana_offline/legacy_index_converter.cc
        ${CMAKE_SOURCE_DIR}/tools/vamana_offline/pq_indexer.cc
        ${CMAKE_SOURCE_DIR}/src/gpu_search/index_format.cc
        ${CMAKE_SOURCE_DIR}/src/gpu_search/pq_index.cc)
    dvstor_target_tool_includes(legacy_index_converter_test)
    target_link_libraries(legacy_index_converter_test PRIVATE
        rdma_library FaissCPU::FaissCPU)
endif ()

dvstor_add_test(dynamic_allocation_receipt_policy_test)
dvstor_target_common_includes(dynamic_allocation_receipt_policy_test)

dvstor_add_test(dynamic_allocation_policy_test)
dvstor_target_common_includes(dynamic_allocation_policy_test)

dvstor_add_test(centroid_router_test
    centroid_router_test.cc
    ${CMAKE_SOURCE_DIR}/src/vamana/centroid_router.cc)
dvstor_target_common_includes(centroid_router_test)
target_link_libraries(centroid_router_test PRIVATE rdma_library)

dvstor_add_test(centroid_home_selector_test)
dvstor_target_common_includes(centroid_home_selector_test)

dvstor_add_test(centroid_route_poll_policy_test)
dvstor_target_common_includes(centroid_route_poll_policy_test)

dvstor_add_test(centroid_membership_lifecycle_test)
dvstor_target_common_includes(centroid_membership_lifecycle_test)

dvstor_add_test(centroid_seed_policy_test)
dvstor_target_common_includes(centroid_seed_policy_test)

dvstor_add_test(centroid_route_ranking_test)
dvstor_target_common_includes(centroid_route_ranking_test)

dvstor_add_test(storage_owner_reconcile_reverse_test)
dvstor_target_common_includes(storage_owner_reconcile_reverse_test)
target_link_libraries(storage_owner_reconcile_reverse_test PRIVATE rdma_library)

dvstor_add_test(stage1_reachability_policy_test)
dvstor_target_common_includes(stage1_reachability_policy_test)

dvstor_add_test(partition_local_search_test)
dvstor_target_common_includes(partition_local_search_test)

dvstor_add_test(persistent_grid_plan_test)
dvstor_target_common_includes(persistent_grid_plan_test)

dvstor_add_test(persistent_owner_watchdog_test)
dvstor_target_common_includes(persistent_owner_watchdog_test)

dvstor_add_test(direct_owner_train_layout_test)
dvstor_target_common_includes(direct_owner_train_layout_test)

dvstor_add_test(frontier_staging_test)
dvstor_target_common_includes(frontier_staging_test)

dvstor_add_test(maintenance_fence_test)
dvstor_target_common_includes(maintenance_fence_test)

dvstor_add_test(maintenance_telemetry_test)
dvstor_target_common_includes(maintenance_telemetry_test)

if (TARGET dvstor_gpu_kernels)
    add_executable(host_distance_kernel_test host_distance_kernel_test.cu)
    target_link_libraries(host_distance_kernel_test
        PRIVATE dvstor_gpu_kernels CUDA::cudart)
    add_test(NAME host_distance_kernel_test COMMAND host_distance_kernel_test)
    set_tests_properties(host_distance_kernel_test PROPERTIES
        TIMEOUT 15
        RESOURCE_LOCK dvstor_gpu
        SKIP_REGULAR_EXPRESSION "^SKIP:")

    dvstor_add_cuda_test(gpu_centroid_route_ranking_test 15)
    dvstor_add_cuda_test(gpu_compact_beam_merge_test 15)
    dvstor_add_cuda_test(gpu_pq_packed_scoring_test 15)
    dvstor_add_cuda_test(gpu_beam_merge_equivalence_test 15)
    dvstor_add_cuda_test(gpu_stable_run_merge_microbench 30)
    dvstor_add_cuda_test(gpu_stable_frontier_preview_test 30)
    dvstor_add_cuda_test(gpu_partition_bounded_certificate_test 30)
    dvstor_add_cuda_test(gpu_dominance_envelope_certificate_test 60)
    dvstor_add_cuda_test(gpu_fused_prefix_materialization_test 30)
    dvstor_add_cuda_test(gpu_frontier_rob_test 30)

    add_executable(gpu_persistent_dispatch_test gpu_persistent_dispatch_test.cc)
    target_link_libraries(gpu_persistent_dispatch_test PRIVATE dvstor_runtime)
    add_test(NAME gpu_persistent_dispatch_test COMMAND gpu_persistent_dispatch_test)
    add_test(NAME gpu_persistent_dispatch_pq32_test
        COMMAND gpu_persistent_dispatch_test 1024 10 8 32)
    add_test(NAME gpu_persistent_dispatch_pq20_test
        COMMAND gpu_persistent_dispatch_test 128 10 8 20)
    add_test(NAME gpu_persistent_dispatch_pq25_test
        COMMAND gpu_persistent_dispatch_test 128 10 8 25)
    add_test(NAME gpu_persistent_dispatch_256_test
        COMMAND gpu_persistent_dispatch_test 128 10 8 32 256)
    # Each test selects device 0. Persistent kernels can prevent another
    # process from making progress, so serialize GPU tests under ctest -j.
    set_tests_properties(
        gpu_persistent_dispatch_test
        gpu_persistent_dispatch_pq32_test
        gpu_persistent_dispatch_pq20_test
        gpu_persistent_dispatch_pq25_test
        gpu_persistent_dispatch_256_test PROPERTIES
        TIMEOUT 15
        RESOURCE_LOCK dvstor_gpu
        SKIP_REGULAR_EXPRESSION "^SKIP:")
endif ()

dvstor_add_test(gpu_navigation_layout_test)
target_link_libraries(gpu_navigation_layout_test PRIVATE dvstor_runtime)

dvstor_add_test(host_orchestrated_policy_test)
target_link_libraries(host_orchestrated_policy_test PRIVATE dvstor_runtime)

dvstor_add_test(gpu_pq_index_test)
target_link_libraries(gpu_pq_index_test PRIVATE dvstor_runtime)

dvstor_add_test(graph_extent_indexer_test
    graph_extent_indexer_test.cc
    ${CMAKE_SOURCE_DIR}/tools/vamana_offline/graph_extent_indexer.cc)
dvstor_target_tool_includes(graph_extent_indexer_test)
target_link_libraries(graph_extent_indexer_test PRIVATE dvstor_runtime)

dvstor_add_test(storage_reclaim_test)
target_link_libraries(storage_reclaim_test PRIVATE dvstor_runtime)

find_package(Threads REQUIRED)
dvstor_add_test(incarnation_lock_test)
dvstor_target_common_includes(incarnation_lock_test)
target_link_libraries(incarnation_lock_test PRIVATE Threads::Threads)

dvstor_add_test(stage2_remote_publication_test)
dvstor_target_common_includes(stage2_remote_publication_test)

dvstor_add_test(storage_owner_protocol_test)
target_link_libraries(storage_owner_protocol_test PRIVATE dvstor_runtime)

dvstor_add_test(storage_startup_protocol_test)
dvstor_target_common_includes(storage_startup_protocol_test)

dvstor_add_test(explicit_release_replay_test)
dvstor_target_common_includes(explicit_release_replay_test)

dvstor_add_test(storage_owner_stage2_tracker_test)

dvstor_add_test(stage2_reconcile_batch_state_test)
dvstor_target_common_includes(stage2_reconcile_batch_state_test)

dvstor_add_test(stage2_batch_policy_test)
dvstor_target_common_includes(stage2_batch_policy_test)

dvstor_add_test(stage2_budget_backlog_policy_test)
dvstor_target_common_includes(stage2_budget_backlog_policy_test)

dvstor_add_test(storage_owner_flow_control_policy_test)
dvstor_target_common_includes(storage_owner_flow_control_policy_test)

dvstor_add_test(storage_owner_reverse_outbox_test)
dvstor_target_common_includes(storage_owner_reverse_outbox_test)

dvstor_add_test(storage_owner_home_rpc_outbox_test)
dvstor_target_common_includes(storage_owner_home_rpc_outbox_test)

dvstor_add_test(storage_owner_ready_context_queue_test)
dvstor_target_common_includes(storage_owner_ready_context_queue_test)
target_link_libraries(storage_owner_ready_context_queue_test PRIVATE Threads::Threads)

dvstor_add_test(peer_async_response_registry_test)
dvstor_target_common_includes(peer_async_response_registry_test)

dvstor_add_test(gpu_memory_budget_test)
target_link_libraries(gpu_memory_budget_test PRIVATE dvstor_runtime)
