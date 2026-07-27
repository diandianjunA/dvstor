include_guard(GLOBAL)

include(CheckCXXSourceCompiles)
include(CMakePushCheckState)

function(dvstor_check_metis_link output_variable)
    cmake_push_check_state(RESET)
    set(CMAKE_REQUIRED_INCLUDES "${METIS_INCLUDE_DIR}")
    set(CMAKE_REQUIRED_LIBRARIES "${METIS_LIBRARY}")
    if (GKLIB_LIBRARY)
        list(APPEND CMAKE_REQUIRED_LIBRARIES "${GKLIB_LIBRARY}")
    endif ()
    unset(DVSTOR_METIS_LINK_COMPATIBLE CACHE)
    check_cxx_source_compiles(
        "#include <metis.h>
         int main() {
             idx_t options[METIS_NOPTIONS];
             return METIS_SetDefaultOptions(options) == METIS_OK ? 0 : 1;
         }"
        DVSTOR_METIS_LINK_COMPATIBLE)
    set(${output_variable} "${DVSTOR_METIS_LINK_COMPATIBLE}" PARENT_SCOPE)
    cmake_pop_check_state()
endfunction()

if (NOT DVSTOR_STORAGE_NODE_ONLY)
    find_package(CUDAToolkit 11.8 REQUIRED)

    if (DVSTOR_CONDA_LIB_DIR AND EXISTS "${DVSTOR_CONDA_LIB_DIR}")
        set(CMAKE_EXE_LINKER_FLAGS
            "${CMAKE_EXE_LINKER_FLAGS} -L${DVSTOR_CONDA_LIB_DIR} -Wl,-rpath,${DVSTOR_CONDA_LIB_DIR}")
    endif ()

endif ()

if (DVSTOR_BUILD_OFFLINE_TOOLS)
    set(_DVSTOR_METIS_HINTS)
    if (DVSTOR_METIS_ROOT)
        list(APPEND _DVSTOR_METIS_HINTS "${DVSTOR_METIS_ROOT}")
    endif ()
    if (EXISTS "${DVSTOR_SOURCE_ROOT}/thirdparty/metis64")
        list(APPEND _DVSTOR_METIS_HINTS "${DVSTOR_SOURCE_ROOT}/thirdparty/metis64")
    endif ()

    if (NOT DVSTOR_METIS_PARTITION_MODE STREQUAL "OFF")
        find_path(METIS_INCLUDE_DIR metis.h
            HINTS ${_DVSTOR_METIS_HINTS}
            PATH_SUFFIXES include)
        find_library(METIS_LIBRARY NAMES metis
            HINTS ${_DVSTOR_METIS_HINTS}
            PATH_SUFFIXES lib lib64)
        find_library(GKLIB_LIBRARY NAMES GKlib
            HINTS ${_DVSTOR_METIS_HINTS}
            PATH_SUFFIXES lib lib64)

        if (METIS_INCLUDE_DIR AND METIS_LIBRARY)
            dvstor_check_metis_link(_DVSTOR_METIS_LINKS)
            string(FIND "${METIS_LIBRARY}"
                "${DVSTOR_SOURCE_ROOT}/thirdparty/metis64/"
                _DVSTOR_BUNDLED_METIS_INDEX)
            if (NOT _DVSTOR_METIS_LINKS AND _DVSTOR_BUNDLED_METIS_INDEX EQUAL 0)
                set(_DVSTOR_REJECTED_METIS "${METIS_LIBRARY};${GKLIB_LIBRARY}")
                foreach (_DVSTOR_METIS_VARIABLE
                    METIS_INCLUDE_DIR METIS_LIBRARY GKLIB_LIBRARY)
                    unset(${_DVSTOR_METIS_VARIABLE})
                    unset(${_DVSTOR_METIS_VARIABLE} CACHE)
                endforeach ()
                set(_DVSTOR_SYSTEM_LIBRARY_SUFFIXES lib lib64)
                if (CMAKE_LIBRARY_ARCHITECTURE)
                    list(APPEND _DVSTOR_SYSTEM_LIBRARY_SUFFIXES
                        "lib/${CMAKE_LIBRARY_ARCHITECTURE}")
                endif ()
                find_path(METIS_INCLUDE_DIR metis.h
                    PATHS /usr /usr/local
                    PATH_SUFFIXES include
                    NO_DEFAULT_PATH)
                find_library(METIS_LIBRARY NAMES metis
                    PATHS /usr /usr/local
                    PATH_SUFFIXES ${_DVSTOR_SYSTEM_LIBRARY_SUFFIXES}
                    NO_DEFAULT_PATH)
                find_library(GKLIB_LIBRARY NAMES GKlib
                    PATHS /usr /usr/local
                    PATH_SUFFIXES ${_DVSTOR_SYSTEM_LIBRARY_SUFFIXES}
                    NO_DEFAULT_PATH)
                if (METIS_INCLUDE_DIR AND METIS_LIBRARY)
                    dvstor_check_metis_link(_DVSTOR_METIS_LINKS)
                    if (_DVSTOR_METIS_LINKS)
                        message(STATUS
                            "Bundled METIS is not link-compatible; using system METIS instead")
                    endif ()
                endif ()
                unset(_DVSTOR_SYSTEM_LIBRARY_SUFFIXES)
                unset(_DVSTOR_METIS_VARIABLE)
            endif ()
            if (_DVSTOR_METIS_LINKS)
                set(DVSTOR_HAVE_METIS ON)
                if (GKLIB_LIBRARY)
                    message(STATUS "METIS partitioning: enabled (${METIS_LIBRARY}, GKlib: ${GKLIB_LIBRARY})")
                else ()
                    message(STATUS "METIS partitioning: enabled (${METIS_LIBRARY})")
                endif ()
            elseif (DVSTOR_METIS_PARTITION_MODE STREQUAL "ON")
                message(FATAL_ERROR
                    "METIS partitioning was requested, but the discovered libraries cannot be linked: "
                    "${_DVSTOR_REJECTED_METIS};${METIS_LIBRARY};${GKLIB_LIBRARY}. "
                    "Rebuild/install METIS for this host, set "
                    "DVSTOR_METIS_ROOT to a compatible CPU installation, or use "
                    "-DDVSTOR_METIS_PARTITION=OFF.")
            else ()
                set(DVSTOR_HAVE_METIS OFF)
                message(WARNING
                    "METIS partitioning: disabled because the discovered libraries cannot be linked. "
                    "This commonly means the bundled METIS/GKlib was built against a newer glibc. "
                    "The balanced and bfs partition strategies remain available.")
            endif ()
            unset(_DVSTOR_BUNDLED_METIS_INDEX)
            unset(_DVSTOR_REJECTED_METIS)
            unset(_DVSTOR_METIS_LINKS)
        elseif (DVSTOR_METIS_PARTITION_MODE STREQUAL "ON")
            message(FATAL_ERROR
                "METIS partitioning requested but metis.h/libmetis were not found. "
                "Install libmetis-dev, set DVSTOR_METIS_ROOT, or set -DDVSTOR_METIS_PARTITION=OFF.")
        else ()
            set(DVSTOR_HAVE_METIS OFF)
            message(STATUS "METIS partitioning: disabled (metis.h/libmetis not found)")
        endif ()
    else ()
        set(DVSTOR_HAVE_METIS OFF)
        message(STATUS "METIS partitioning: disabled")
    endif ()

    unset(_DVSTOR_METIS_HINTS)
else ()
    set(DVSTOR_HAVE_METIS OFF)
endif ()
