add_executable(ShineIndexSmokeTest
    src/main.cpp
    src/shine_index.cpp
)

target_include_directories(ShineIndexSmokeTest PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../src
)

target_link_libraries(ShineIndexSmokeTest PRIVATE
    shine_runtime
)
