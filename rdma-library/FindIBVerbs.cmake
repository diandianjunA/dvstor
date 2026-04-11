# FindIBVerbs.cmake
# CMake module to find libibverbs

# Keep legacy rdma-core installs under /usr/local out of the build.
foreach(_cache_var
    IBVerbs_INCLUDE_DIR
    IBVerbs_LIBRARY
    IBVERBS_INCLUDE_DIR
    IBVERBS_LIBRARY
    NL_ROUTE_LIBRARY
    NL_3_LIBRARY
)
    if(DEFINED ${_cache_var} AND "${${_cache_var}}" MATCHES "^/usr/local/")
        unset(${_cache_var} CACHE)
        unset(${_cache_var})
    endif()
endforeach()

set(_ibverbs_include_paths
    /usr/include
)
set(_ibverbs_library_paths
    /usr/lib/${CMAKE_LIBRARY_ARCHITECTURE}
    /lib/${CMAKE_LIBRARY_ARCHITECTURE}
    /usr/lib64
    /lib64
    /usr/lib
    /lib
)

find_path(IBVerbs_INCLUDE_DIR
    NAMES infiniband/verbs.h
    PATHS ${_ibverbs_include_paths}
    NO_DEFAULT_PATH
)

find_library(IBVerbs_LIBRARY
    NAMES ibverbs
    PATHS ${_ibverbs_library_paths}
    NO_DEFAULT_PATH
)

find_library(NL_ROUTE_LIBRARY
    NAMES nl-route-3
    PATHS ${_ibverbs_library_paths}
    NO_DEFAULT_PATH
)

find_library(NL_3_LIBRARY
    NAMES nl-3
    PATHS ${_ibverbs_library_paths}
    NO_DEFAULT_PATH
)

foreach(_resolved_var
    IBVerbs_INCLUDE_DIR
    IBVerbs_LIBRARY
    NL_ROUTE_LIBRARY
    NL_3_LIBRARY
)
    if(DEFINED ${_resolved_var} AND "${${_resolved_var}}" MATCHES "^/usr/local/")
        message(FATAL_ERROR "Refusing to use legacy RDMA artifact from /usr/local: ${_resolved_var}=${${_resolved_var}}")
    endif()
endforeach()

if(IBVerbs_LIBRARY MATCHES "\\.a$")
    message(FATAL_ERROR "Expected shared libibverbs, found static archive: ${IBVerbs_LIBRARY}")
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(IBVerbs
    REQUIRED_VARS
        IBVerbs_LIBRARY
        IBVerbs_INCLUDE_DIR
)

if(IBVerbs_FOUND)
    set(IBVERBS_LIBRARY ${IBVerbs_LIBRARY})
    set(IBVERBS_INCLUDE_DIR ${IBVerbs_INCLUDE_DIR})
    set(NL_ROUTE_LIBRARY ${NL_ROUTE_LIBRARY})
    set(NL_3_LIBRARY ${NL_3_LIBRARY})
endif()
