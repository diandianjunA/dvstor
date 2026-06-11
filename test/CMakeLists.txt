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
