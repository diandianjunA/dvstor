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
target_include_directories(rabitq_encoding_test PRIVATE
    ${PROJECT_SOURCE_DIR}/src
    ${PROJECT_SOURCE_DIR}/rdma-library
    ${PROJECT_SOURCE_DIR}/thirdparty
)
add_test(NAME rabitq_encoding_test
         COMMAND rabitq_encoding_test)
