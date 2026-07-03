include_guard(GLOBAL)

if (NOT DVSTOR_STORAGE_NODE_ONLY)
    find_package(CUDAToolkit 11.8 REQUIRED)
endif ()

if (DVSTOR_CONDA_LIB_DIR AND EXISTS "${DVSTOR_CONDA_LIB_DIR}")
    set(CMAKE_EXE_LINKER_FLAGS
        "${CMAKE_EXE_LINKER_FLAGS} -L${DVSTOR_CONDA_LIB_DIR} -Wl,-rpath,${DVSTOR_CONDA_LIB_DIR}")
endif ()

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
        set(DVSTOR_HAVE_METIS ON)
        if (GKLIB_LIBRARY)
            message(STATUS "METIS partitioning: enabled (${METIS_LIBRARY}, GKlib: ${GKLIB_LIBRARY})")
        else ()
            message(STATUS "METIS partitioning: enabled (${METIS_LIBRARY})")
        endif ()
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
