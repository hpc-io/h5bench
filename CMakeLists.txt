cmake_minimum_required(VERSION 3.10 FATAL_ERROR)

project(
    h5bench
    VERSION 1.3.0
)

include(ExternalProject)
include(CTest)

set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/CMake" ${CMAKE_MODULE_PATH})

find_package(Python REQUIRED)
find_package(MPI REQUIRED)

include_directories(SYSTEM ${MPI_INCLUDE_PATH})

enable_testing()

# Options and Variants ########################################################
#

set(CMAKE_BUILD_TYPE Debug)

option(H5BENCH_ALL              "Enable all benchmarks"             OFF)
option(H5BENCH_METADATA         "Enable Metadata benchmark"         OFF)
option(H5BENCH_EXERCISER        "Enable Exerciser benchmark"        OFF)
option(H5BENCH_AMREX            "Enable AMReX benchmark"            OFF)
option(H5BENCH_OPENPMD          "Enable OpenPMD benchmark"          OFF)
option(H5BENCH_E3SM             "Enable E3SM benchmark"             OFF)

message(STATUS "h5bench baseline: ON")

if(H5BENCH_ALL)
    message(STATUS "h5bench ENABLE ALL benchmarks")

    set(H5BENCH_METADATA ON)
    set(H5BENCH_EXERCISER ON)
    set(H5BENCH_AMREX ON)
    set(H5BENCH_OPENPMD ON)
    set(H5BENCH_E3SM ON)
endif()

message(STATUS "h5bench METADATA: ${H5BENCH_METADATA}")
message(STATUS "h5bench EXERCISER: ${H5BENCH_EXERCISER}")
message(STATUS "h5bench AMREX: ${H5BENCH_AMREX}")
message(STATUS "h5bench OPENPMD: ${H5BENCH_OPENPMD}")
message(STATUS "h5bench E3SM: ${H5BENCH_E3SM}")

if(H5BENCH_E3SM)
    set(PNETCDF_HOME $ENV{PNETCDF_HOME})
    set(CMAKE_PREFIX_PATH ${PNETCDF_HOME})

    find_package(PNetCDF REQUIRED)
    include_directories(${PNETCDF_INCLUDES})
endif()

# HDF5 Dependency #############################################################
#

set(HDF5_HOME $ENV{HDF5_HOME})
set(CMAKE_PREFIX_PATH ${HDF5_HOME})

find_package(HDF5 REQUIRED)

message(STATUS "Using HDF5 version: ${HDF5_VERSION}")

include_directories(${HDF5_HOME}/include)
link_directories(${HDF5_HOME}/lib)

# VOL ASYNC Dependency ########################################################
#

set(ASYNC_HOME $ENV{ASYNC_HOME})
option(WITH_ASYNC_VOL "Enable HDF5 VOL ASYNC connector" OFF)

if(WITH_ASYNC_VOL)
    if(${HDF5_VERSION} VERSION_GREATER_EQUAL "1.13.0")
        add_definitions(-DUSE_ASYNC_VOL)
        include_directories(${ASYNC_HOME}/include)
        link_directories(${ASYNC_HOME}/lib)
    else()
        message(SEND_ERROR "VOL ASYNC requires HDF5 1.13.0 or newer.")
    endif()
endif()

message(STATUS "HDF5 VOL ASYNC: ${WITH_ASYNC_VOL}")

# h5bench Utility #############################################################
#

set(h5bench_util_src
    commons/h5bench_util.c
    commons/h5bench_util.h
)

add_library(h5bench_util ${h5bench_util_src})

if(WITH_ASYNC_VOL)
    target_link_libraries(h5bench_util asynchdf5 h5async)
endif()

# h5bench WRITE ###############################################################
#

set(h5bench_write_src h5bench_patterns/h5bench_write.c)

add_executable(h5bench_write ${h5bench_write_src})
target_link_libraries(h5bench_write h5bench_util hdf5 z ${CMAKE_DL_LIBS} MPI::MPI_C) 

# h5bench WRITE UNLIMITED #####################################################
#

set(h5bench_write_unlimited_src h5bench_patterns/h5bench_write_unlimited.c)

add_executable(h5bench_write_unlimited ${h5bench_write_unlimited_src})
target_link_libraries(h5bench_write_unlimited h5bench_util hdf5 z ${CMAKE_DL_LIBS} MPI::MPI_C) 

# h5bench OVERWRITE ###########################################################
#

set(h5bench_overwrite_src h5bench_patterns/h5bench_overwrite.c)

add_executable(h5bench_overwrite ${h5bench_overwrite_src})
target_link_libraries(h5bench_overwrite h5bench_util hdf5 z ${CMAKE_DL_LIBS} MPI::MPI_C)

# h5bench APPEND ##############################################################
#

set(h5bench_append_src h5bench_patterns/h5bench_append.c)

add_executable(h5bench_append ${h5bench_append_src})
target_link_libraries(h5bench_append h5bench_util hdf5 z ${CMAKE_DL_LIBS} MPI::MPI_C)

# h5bench READ ################################################################
#

set(h5bench_read_src h5bench_patterns/h5bench_read.c)

add_executable(h5bench_read ${h5bench_read_src})
target_link_libraries(h5bench_read h5bench_util hdf5 z ${CMAKE_DL_LIBS} MPI::MPI_C)


# Exerciser ###################################################################
#

if(H5BENCH_EXERCISER)
    set(exerciser_src exerciser/h5bench_exerciser.c)

    add_executable(h5bench_exerciser ${exerciser_src})

    target_link_libraries(h5bench_exerciser hdf5 z m ${CMAKE_DL_LIBS} MPI::MPI_C)
endif()

# IOTEST ######################################################################
#

if(H5BENCH_EXERCISER)
    set(meta_stress_src
        metadata_stress/hdf5_iotest.c
        metadata_stress/configuration.c
        metadata_stress/configuration.h
        metadata_stress/dataset.c
        metadata_stress/dataset.h
        metadata_stress/ini.c
        metadata_stress/ini.h
    )

    add_executable(h5bench_hdf5_iotest ${meta_stress_src})

    target_link_libraries(h5bench_hdf5_iotest h5bench_util hdf5 z m ${CMAKE_DL_LIBS} MPI::MPI_C)
endif()

# AMReX #######################################################################
#
# https://github.com/AMReX-Codes/amrex/tree/development/Tests/HDF5Benchmark

if(H5BENCH_AMREX)
    set(AMReX_HDF5 YES)
    set(AMReX_PARTICLES YES)
    set(AMReX_MPI_THREAD_MULTIPLE YES)

    add_subdirectory(amrex)

    set(amrex_src amrex/Tests/HDF5Benchmark/main.cpp)

    add_executable(h5bench_amrex_sync ${amrex_src})
    target_link_libraries(h5bench_amrex_sync hdf5 z m amrex pthread ${CMAKE_DL_LIBS} MPI::MPI_C)

    if(WITH_ASYNC_VOL)
        set(AMReX_HDF5_ASYNC YES)
        
        add_executable(h5bench_amrex_async ${amrex_src})
        target_link_libraries(h5bench_amrex_async hdf5 z m amrex pthread asynchdf5 h5async ${CMAKE_DL_LIBS} MPI::MPI_C)
    endif()
endif()

# E3SM ########################################################################
#
# https://github.com/Parallel-NetCDF/E3SM-IO

if(H5BENCH_E3SM)
    ExternalProject_Add(h5bench_e3sm
        SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/e3sm
        CONFIGURE_COMMAND autoreconf -i COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/e3sm/configure --prefix=${CMAKE_BINARY_DIR} --with-pnetcdf=${PNETCDF_HOME} --with-hdf5=${HDF5_HOME} CFLAGS=-fno-var-tracking-assignments CXXFLAGS=-fno-var-tracking-assignments
        BUILD_COMMAND make
        INSTALL_COMMAND ${CMAKE_COMMAND} -E copy src/e3sm_io ${CMAKE_BINARY_DIR}/h5bench_e3sm 
        BUILD_IN_SOURCE 1
        LOG_CONFIGURE 1
        LOG_BUILD 1
    )
endif()

# OpenPMD #####################################################################
#
# https://github.com/openPMD/openPMD-api

if(H5BENCH_OPENPMD)
    set(openPMD_USE_MPI ON)
    set(openPMD_USE_HDF5 ON)
    set(openPMD_USE_ADIOS1 OFF)
    set(openPMD_USE_ADIOS2 OFF)
    set(openPMD_USE_JSON OFF)
    set(openPMD_USE_PYTHON OFF)
    set(openPMD_INSTALL OFF)
    set(openPMD_BUILD_TESTING OFF)
    set(openPMD_BUILD_EXAMPLES OFF)
    set(openPMD_BUILD_CLI_TOOLS OFF)

    add_subdirectory(openpmd)

    add_executable(h5bench_openpmd_write openpmd/examples/8a_benchmark_write_parallel.cpp)
    target_link_libraries(h5bench_openpmd_write openPMD hdf5 MPI::MPI_C)

    set_target_properties(h5bench_openpmd_write PROPERTIES
        CXX_EXTENSIONS OFF
        CXX_STANDARD_REQUIRED ON
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
    )

    add_executable(h5bench_openpmd_read openpmd/examples/8b_benchmark_read_parallel.cpp)
    target_link_libraries(h5bench_openpmd_read openPMD hdf5 MPI::MPI_C)

    set_target_properties(h5bench_openpmd_read PROPERTIES
        CXX_EXTENSIONS OFF
        CXX_STANDARD_REQUIRED ON
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
    )
endif()

configure_file(${CMAKE_SOURCE_DIR}/src/h5bench.py ${CMAKE_BINARY_DIR}/h5bench COPYONLY)

# Install binaries ############################################################

install(
    FILES
    src/h5bench.py RENAME h5bench
    PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ
    DESTINATION bin
)

install(
    FILES
    src/version.py
    PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ
    DESTINATION bin
)

if(H5BENCH_EXERCISER)
    install(
        TARGETS
        h5bench_exerciser 
        DESTINATION bin
    )
endif()

if(H5BENCH_METADATA)
    install(
        TARGETS
        h5bench_hdf5_iotest
        DESTINATION bin
    )
endif()

if(H5BENCH_OPENPMD)
    install(
        TARGETS
        h5bench_openpmd_write
        h5bench_openpmd_read
        DESTINATION bin
    )
endif()

if(H5BENCH_AMREX)
    install(
        TARGETS
        h5bench_amrex_sync
        DESTINATION bin
    )

    if(WITH_ASYNC_VOL)
        install( 
            TARGETS
            h5bench_amrex_async 
            DESTINATION bin
        )
    endif()
endif()

# Testing #####################################################################

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/tests)
