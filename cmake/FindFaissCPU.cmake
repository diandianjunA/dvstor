find_path(FaissCPU_INCLUDE_DIR
    NAMES faiss/VectorTransform.h
    HINTS /usr/local/include /usr/include)

find_library(FaissCPU_LIBRARY
    NAMES faiss
    HINTS /usr/local/lib /usr/local/lib64 /usr/lib /usr/lib64
    PATH_SUFFIXES x86_64-linux-gnu)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(FaissCPU
    REQUIRED_VARS FaissCPU_INCLUDE_DIR FaissCPU_LIBRARY)

if (FaissCPU_FOUND AND NOT TARGET FaissCPU::FaissCPU)
    find_package(Threads REQUIRED)
    set(_FaissCPU_saved_bla_vendor "${BLA_VENDOR}")
    set(BLA_VENDOR "${DVSTOR_FAISS_BLAS_VENDOR}")
    find_package(BLAS REQUIRED)
    find_package(LAPACK REQUIRED)
    set(BLA_VENDOR "${_FaissCPU_saved_bla_vendor}")
    unset(_FaissCPU_saved_bla_vendor)
    find_package(OpenMP REQUIRED COMPONENTS CXX)
    message(STATUS
        "FaissCPU BLAS vendor=${DVSTOR_FAISS_BLAS_VENDOR}; libraries=${LAPACK_LIBRARIES};${BLAS_LIBRARIES}")

    add_library(FaissCPU::FaissCPU UNKNOWN IMPORTED)
    set_target_properties(FaissCPU::FaissCPU PROPERTIES
        IMPORTED_LOCATION "${FaissCPU_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${FaissCPU_INCLUDE_DIR}"
        INTERFACE_LINK_LIBRARIES
            "OpenMP::OpenMP_CXX;${LAPACK_LIBRARIES};${BLAS_LIBRARIES};Threads::Threads;${CMAKE_DL_LIBS};m"
    )
endif ()

mark_as_advanced(FaissCPU_INCLUDE_DIR FaissCPU_LIBRARY)
