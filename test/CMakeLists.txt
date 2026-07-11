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
