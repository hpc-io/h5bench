cmake_minimum_required(VERSION 3.13 FATAL_ERROR)

set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)

project(
    h5bench
    VERSION 1.6.0
)

include(ExternalProject)
include(CTest)

set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/CMake" ${CMAKE_MODULE_PATH})

# Keep every h5bench_* executable at the top of the build tree. The Python
# driver's is_available() check falls back to PWD (shutil.which path=PATH + ':.'),
# and some CI environments (notably NERSC) install to a non-PATH prefix and rely
# on running `./h5bench` from the build dir. Before the per-subdir CMakeLists
# refactor every add_executable() was at root scope, so binaries naturally
# landed at ${CMAKE_BINARY_DIR}; this restores that layout explicitly.
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

find_package(Python REQUIRED)
find_package(MPI REQUIRED)

enable_testing()

# Options and Variants ########################################################
#

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS Debug Release RelWithDebInfo MinSizeRel)
endif()

option(H5BENCH_ALL              "Enable all benchmarks"             OFF)
option(H5BENCH_METADATA         "Enable Metadata benchmark"         OFF)
option(H5BENCH_EXERCISER        "Enable Exerciser benchmark"        OFF)
option(H5BENCH_AMREX            "Enable AMReX benchmark"            OFF)
option(H5BENCH_OPENPMD          "Enable OpenPMD benchmark"          OFF)
option(H5BENCH_E3SM             "Enable E3SM benchmark"             OFF)
option(H5BENCH_MACSIO           "Enable MACSio benchmark"           OFF)

message(STATUS "h5bench baseline: ON")

if(H5BENCH_ALL)
    message(STATUS "h5bench ENABLE ALL benchmarks")

    set(H5BENCH_METADATA ON)
    set(H5BENCH_EXERCISER ON)
    set(H5BENCH_AMREX ON)
    set(H5BENCH_OPENPMD ON)
    set(H5BENCH_E3SM ON)
    set(H5BENCH_MACSIO ON)
endif()

message(STATUS "h5bench METADATA: ${H5BENCH_METADATA}")
message(STATUS "h5bench EXERCISER: ${H5BENCH_EXERCISER}")
message(STATUS "h5bench AMREX: ${H5BENCH_AMREX}")
message(STATUS "h5bench OPENPMD: ${H5BENCH_OPENPMD}")
message(STATUS "h5bench E3SM: ${H5BENCH_E3SM}")
message(STATUS "h5bench MACSIO: ${H5BENCH_MACSIO}")

# HDF5 Dependency #############################################################
#

if(DEFINED ENV{HDF5_DIR})
    message(STATUS "Using HDF5_DIR: $ENV{HDF5_DIR}")

    set(HDF5_HOME $ENV{HDF5_DIR})
    set(CMAKE_PREFIX_PATH $ENV{HDF5_DIR})
endif()

if(DEFINED ENV{HDF5_HOME})
    message(STATUS "Using HDF5_HOME: $ENV{HDF5_HOME}")

    set(HDF5_HOME $ENV{HDF5_HOME})
    set(CMAKE_PREFIX_PATH $ENV{HDF5_HOME})
endif()

find_package(HDF5 REQUIRED)

message(STATUS "Using HDF5 version: ${HDF5_VERSION}")

# Check if HDF5 has subfiling VFD
set(HAVE_SUBFILING_VFD 0)
find_file(H5FDsubfiling_H H5FDsubfiling.h PATHS ${HDF5_INCLUDE_DIRS})
if(EXISTS ${H5FDsubfiling_H})
    include(CheckSymbolExists)
    set(CMAKE_REQUIRED_INCLUDES ${CMAKE_REQUIRED_INCLUDES} ${HDF5_INCLUDE_DIRS})
    check_symbol_exists(H5_HAVE_SUBFILING_VFD hdf5.h HAVE_SUBFILING_VFD)
    unset(CMAKE_REQUIRED_INCLUDES)
endif()
message(STATUS "   Detected HDF5 subfiling support: ${HAVE_SUBFILING_VFD}")

# ZLIB Dependency #############################################################
#

find_package(ZLIB REQUIRED)

# VOL ASYNC Dependency ########################################################
#

set(ASYNC_HOME $ENV{ASYNC_HOME})
option(WITH_ASYNC_VOL "Enable HDF5 VOL ASYNC connector" OFF)

if(WITH_ASYNC_VOL)
    if(${HDF5_VERSION} VERSION_GREATER_EQUAL "1.13.0")
        # USE_ASYNC_VOL define + include/link paths are applied to
        # h5bench_util (and thus every consumer) in commons/CMakeLists.txt.
    else()
        message(SEND_ERROR "VOL ASYNC requires HDF5 1.13.0 or newer.")
    endif()
endif()

message(STATUS "HDF5 VOL ASYNC: ${WITH_ASYNC_VOL}")

# h5bench Benchmarks ##########################################################
#

include(${CMAKE_CURRENT_SOURCE_DIR}/CMake/h5benchFunctions.cmake)

add_subdirectory(commons)
add_subdirectory(h5bench_patterns)

if(H5BENCH_EXERCISER)
    add_subdirectory(exerciser)
endif()

if(H5BENCH_METADATA)
    add_subdirectory(metadata_stress)
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
    target_include_directories(h5bench_amrex_sync PRIVATE ${HDF5_INCLUDE_DIRS})
    target_link_libraries(h5bench_amrex_sync
        PRIVATE
            ${HDF5_LIBRARIES}
            ZLIB::ZLIB
            m
            amrex
            pthread
            ${CMAKE_DL_LIBS}
            MPI::MPI_C
    )

    if(WITH_ASYNC_VOL)
        message(WARNING "h5bench AMReX with ASYNC-VOL: ensure ASYNC-VOL was compiled with -DENABLE_WRITE_MEMCPY=ON")

        set(AMReX_HDF5_ASYNC YES)

        add_executable(h5bench_amrex_async ${amrex_src})
        target_include_directories(h5bench_amrex_async PRIVATE ${HDF5_INCLUDE_DIRS} ${ASYNC_HOME}/include)
        target_link_directories(h5bench_amrex_async PRIVATE ${ASYNC_HOME}/lib)
        target_link_libraries(h5bench_amrex_async
            PRIVATE
                ${HDF5_LIBRARIES}
                ZLIB::ZLIB
                m
                amrex
                pthread
                asynchdf5
                h5async
                ${CMAKE_DL_LIBS}
                MPI::MPI_C
        )
    endif()
endif()

# E3SM ########################################################################
#
# https://github.com/Parallel-NetCDF/E3SM-IO

if(H5BENCH_E3SM)
    ExternalProject_Add(h5bench_e3sm
        SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/e3sm
        CONFIGURE_COMMAND autoreconf -i COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/e3sm/configure --prefix=${CMAKE_BINARY_DIR} --with-hdf5=${HDF5_HOME} CFLAGS=-fno-var-tracking-assignments CXXFLAGS=-fno-var-tracking-assignments
        BUILD_COMMAND make -j 1
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
    add_subdirectory(openpmd)

    set(openPMD_USE_MPI ON CACHE BOOL "" FORCE)
    set(openPMD_USE_HDF5 ON CACHE BOOL "" FORCE)
    set(openPMD_USE_ADIOS1 OFF CACHE BOOL "" FORCE)
    set(openPMD_USE_ADIOS2 OFF CACHE BOOL "" FORCE)
    set(openPMD_USE_JSON OFF CACHE BOOL "" FORCE)
    set(openPMD_USE_PYTHON OFF CACHE BOOL "" FORCE)
    set(openPMD_INSTALL ON CACHE BOOL "" FORCE)
    set(openPMD_BUILD_TESTING OFF CACHE BOOL "" FORCE)
    set(openPMD_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(openPMD_BUILD_CLI_TOOLS OFF CACHE BOOL "" FORCE)

    add_executable(h5bench_openpmd_write openpmd/examples/8a_benchmark_write_parallel.cpp)
    target_include_directories(h5bench_openpmd_write PRIVATE ${HDF5_INCLUDE_DIRS})
    target_link_libraries(h5bench_openpmd_write PRIVATE openPMD ${HDF5_LIBRARIES} MPI::MPI_C)

    set_target_properties(h5bench_openpmd_write PROPERTIES
        CXX_EXTENSIONS OFF
        CXX_STANDARD_REQUIRED ON
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
    )

    add_executable(h5bench_openpmd_read openpmd/examples/8b_benchmark_read_parallel.cpp)
    target_include_directories(h5bench_openpmd_read PRIVATE ${HDF5_INCLUDE_DIRS})
    target_link_libraries(h5bench_openpmd_read PRIVATE openPMD ${HDF5_LIBRARIES} MPI::MPI_C)

    set_target_properties(h5bench_openpmd_read PROPERTIES
        CXX_EXTENSIONS OFF
        CXX_STANDARD_REQUIRED ON
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
    )
endif()

# MACSIO ######################################################################
#
# https://github.com/LLNL/MACSio

if(H5BENCH_MACSIO)
    set(JSON_HOME ${CMAKE_BINARY_DIR}/json-cwx)

    ExternalProject_Add(json_cwx
        GIT_REPOSITORY https://github.com/LLNL/json-cwx
        SOURCE_DIR ${CMAKE_CURRENT_BINARY_DIR}/json-cwx
        BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/json-cwx/json-cwx
        CONFIGURE_COMMAND autoreconf -i COMMAND ./configure --prefix=${JSON_HOME}
        BUILD_COMMAND make
        BUILD_IN_SOURCE 0
        LOG_CONFIGURE 1
        LOG_BUILD 1
    )

    set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)
    set(CMAKE_POLICY_DEFAULT_CMP0114 NEW)

    set(BUILD_DOCS OFF)
    set(ENABLE_SILO_PLUGIN OFF)
    set(ENABLE_HDF5_PLUGIN ON)
    set(WITH_HDF5_PREFIX ${HDF5_HOME})

    set(CMAKE_PREFIX_PATH ${HDF5_HOME};${JSON_HOME})

    ExternalProject_Add(h5bench_macsio
        SOURCE_DIR ${CMAKE_SOURCE_DIR}/macsio
        CMAKE_ARGS
            -DBUILD_DOCS=OFF
            -DENABLE_SILO_PLUGIN=OFF
            -DENABLE_HDF5_PLUGIN=ON
            -DWITH_HDF5_PREFIX=${HDF5_HOME}
            -DWITH_JSON-CWX_PREFIX=${JSON_HOME}
            -DCMAKE_INSTALL_PREFIX:PATH=${CMAKE_CURRENT_BINARY_DIR}
            -DCMAKE_POLICY_VERSION_MINIMUM=3.5
        BUILD_COMMAND make
        COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_BINARY_DIR}/h5bench_macsio-prefix/src/h5bench_macsio-build/macsio/macsio ${CMAKE_CURRENT_BINARY_DIR}/h5bench_macsio
        LOG_CONFIGURE 1
        LOG_BUILD 1
    )

    add_dependencies(h5bench_macsio json_cwx)
endif()

# Python driver ###############################################################

configure_file(${CMAKE_SOURCE_DIR}/src/h5bench.py ${CMAKE_BINARY_DIR}/h5bench COPYONLY)
configure_file(${CMAKE_SOURCE_DIR}/src/h5bench_version.py ${CMAKE_BINARY_DIR}/h5bench_version.py COPYONLY)
configure_file(${CMAKE_SOURCE_DIR}/src/h5bench_configuration.py.in ${CMAKE_BINARY_DIR}/h5bench_configuration.py)

install(
    FILES
    src/h5bench.py RENAME h5bench
    PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ
    DESTINATION bin
)

install(
    FILES
    src/h5bench_version.py
    PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ
    DESTINATION bin
)

install(
    FILES
    ${CMAKE_BINARY_DIR}/h5bench_configuration.py
    PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ
    DESTINATION bin
)

# Optional-benchmark installs (core patterns, exerciser, metadata_stress
# install themselves from their own subdirectory CMakeLists.txt).

if(H5BENCH_OPENPMD)
    install(
        TARGETS
        openPMD
        LIBRARY
        DESTINATION lib
    )

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

if(H5BENCH_MACSIO)
    install(
        FILES
        ${CMAKE_BINARY_DIR}/h5bench_macsio
        PERMISSIONS WORLD_EXECUTE OWNER_WRITE OWNER_READ
        DESTINATION bin
    )
endif()

# Testing #####################################################################

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/tests)
