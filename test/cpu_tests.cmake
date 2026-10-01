dvstor_add_test(vector_dtype_test)
dvstor_target_common_includes(vector_dtype_test)

dvstor_add_test(index_metadata_test
    ${DVSTOR_SOURCE_ROOT}/test/index_metadata_test.cc
    ${DVSTOR_SOURCE_ROOT}/src/service/index_metadata.cc
    ${DVSTOR_SOURCE_ROOT}/src/gpu_search/index_format.cc)
dvstor_target_common_includes(index_metadata_test)
set_tests_properties(index_metadata_test PROPERTIES TIMEOUT 10)

dvstor_add_test(rdma_contract_test)
dvstor_target_common_includes(rdma_contract_test)
target_link_libraries(rdma_contract_test PRIVATE rdma_library)
set_tests_properties(rdma_contract_test PROPERTIES TIMEOUT 10)

dvstor_add_test(configuration_update_protocol_test)
dvstor_target_common_includes(configuration_update_protocol_test)
target_link_libraries(configuration_update_protocol_test PRIVATE rdma_library)
if (DVSTOR_STORAGE_NODE_ONLY)
    set_tests_properties(configuration_update_protocol_test PROPERTIES TIMEOUT 10)
endif ()

dvstor_add_test(offline_local_id_set_test)
dvstor_target_tool_includes(offline_local_id_set_test)
set_tests_properties(offline_local_id_set_test PROPERTIES TIMEOUT 10)

dvstor_add_test(shard_publication_test
    ${DVSTOR_SOURCE_ROOT}/test/shard_publication_test.cc
    ${DVSTOR_SOURCE_ROOT}/tools/vamana_offline/dataset_io.cc
    ${DVSTOR_SOURCE_ROOT}/tools/vamana_offline/graph.cc
    ${DVSTOR_SOURCE_ROOT}/tools/vamana_offline/partitioning.cc
    ${DVSTOR_SOURCE_ROOT}/tools/vamana_offline/progress.cc
    ${DVSTOR_SOURCE_ROOT}/tools/vamana_offline/shard_writer.cc)
dvstor_target_tool_includes(shard_publication_test)
target_link_libraries(shard_publication_test PRIVATE rdma_library)
set_tests_properties(shard_publication_test PROPERTIES TIMEOUT 20)

dvstor_add_test(vamana_offline_graph_test
    ${DVSTOR_SOURCE_ROOT}/test/vamana_offline_graph_test.cc
    ${DVSTOR_SOURCE_ROOT}/tools/vamana_offline/dataset_io.cc
    ${DVSTOR_SOURCE_ROOT}/tools/vamana_offline/graph.cc
    ${DVSTOR_SOURCE_ROOT}/tools/vamana_offline/progress.cc)
dvstor_target_tool_includes(vamana_offline_graph_test)
target_link_libraries(vamana_offline_graph_test PRIVATE rdma_library)
set_tests_properties(vamana_offline_graph_test PROPERTIES TIMEOUT 10)

if (TARGET FaissCPU::FaissCPU)
    dvstor_add_test(metis_repartitioner_test
        ${DVSTOR_SOURCE_ROOT}/test/metis_repartitioner_test.cc
        ${DVSTOR_METIS_REPARTITIONER_SOURCES})
    dvstor_target_tool_includes(metis_repartitioner_test)
    target_link_libraries(metis_repartitioner_test
        rdma_library FaissCPU::FaissCPU)
    target_link_metis(metis_repartitioner_test)
    set_tests_properties(metis_repartitioner_test PROPERTIES
        TIMEOUT 180
        SKIP_RETURN_CODE 77)
endif ()
