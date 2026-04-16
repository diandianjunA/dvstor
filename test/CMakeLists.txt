add_executable(DvstorIndexSmokeTest
    src/main.cpp
    src/dvstor_index.cpp
)

target_include_directories(DvstorIndexSmokeTest PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../src
)

target_link_libraries(DvstorIndexSmokeTest PRIVATE
    dvstor_runtime
)
