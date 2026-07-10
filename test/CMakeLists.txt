add_executable(gpu_search_ring_test gpu_search_ring_test.cc)
target_link_libraries(gpu_search_ring_test PRIVATE dvstor_runtime)
target_compile_options(gpu_search_ring_test PRIVATE -UNDEBUG)
add_test(NAME gpu_search_ring_test COMMAND gpu_search_ring_test)

if (TARGET dvstor_gpu_kernels)
    add_executable(gpu_persistent_dispatch_test gpu_persistent_dispatch_test.cc)
    target_link_libraries(gpu_persistent_dispatch_test PRIVATE dvstor_runtime)
    add_test(NAME gpu_persistent_dispatch_test COMMAND gpu_persistent_dispatch_test)
    set_tests_properties(gpu_persistent_dispatch_test PROPERTIES
        TIMEOUT 15
        SKIP_REGULAR_EXPRESSION "^SKIP:")
endif ()

add_executable(gpu_tiered_index_format_test gpu_tiered_index_format_test.cc)
target_link_libraries(gpu_tiered_index_format_test PRIVATE dvstor_runtime)
target_compile_options(gpu_tiered_index_format_test PRIVATE -UNDEBUG)
add_test(NAME gpu_tiered_index_format_test COMMAND gpu_tiered_index_format_test)

add_executable(gpu_rabitq_layout_test gpu_rabitq_layout_test.cc)
target_link_libraries(gpu_rabitq_layout_test PRIVATE dvstor_runtime)
target_compile_options(gpu_rabitq_layout_test PRIVATE -UNDEBUG)
add_test(NAME gpu_rabitq_layout_test COMMAND gpu_rabitq_layout_test)

add_executable(gpu_delta_index_test gpu_delta_index_test.cc)
target_link_libraries(gpu_delta_index_test PRIVATE dvstor_runtime)
target_compile_options(gpu_delta_index_test PRIVATE -UNDEBUG)
add_test(NAME gpu_delta_index_test COMMAND gpu_delta_index_test)

add_executable(gpu_memory_budget_test gpu_memory_budget_test.cc)
target_link_libraries(gpu_memory_budget_test PRIVATE dvstor_runtime)
target_compile_options(gpu_memory_budget_test PRIVATE -UNDEBUG)
add_test(NAME gpu_memory_budget_test COMMAND gpu_memory_budget_test)

add_executable(gpu_sidecar_converter_test
    gpu_sidecar_converter_test.cc
    ${CMAKE_SOURCE_DIR}/tools/vamana_offline/gpu_sidecar_converter.cc
    ${CMAKE_SOURCE_DIR}/tools/vamana_offline/progress.cc
)
dvstor_target_tool_includes(gpu_sidecar_converter_test)
target_link_libraries(gpu_sidecar_converter_test PRIVATE dvstor_runtime)
target_compile_options(gpu_sidecar_converter_test PRIVATE -UNDEBUG)
add_test(NAME gpu_sidecar_converter_test COMMAND gpu_sidecar_converter_test)
