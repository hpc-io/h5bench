cmake_minimum_required(VERSION 3.10 FATAL_ERROR)

project(
	h5bench
    VERSION 1.2.0
)

include(ExternalProject)

set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/CMake" ${CMAKE_MODULE_PATH})

find_package(MPI REQUIRED)
include_directories(SYSTEM ${MPI_INCLUDE_PATH})

set(PNETCDF_HOME $ENV{PNETCDF_HOME})
set(CMAKE_PREFIX_PATH ${PNETCDF_HOME})
find_package(PNetCDF REQUIRED)
include_directories(${PNETCDF_INCLUDES})

# Dependency: HDF5
set(HDF5_HOME $ENV{HDF5_HOME})
set(CMAKE_PREFIX_PATH ${HDF5_HOME})
find_package(HDF5 REQUIRED)
message(STATUS "Using HDF5 version: ${HDF5_VERSION}")
include_directories(${HDF5_HOME}/include)
link_directories(${HDF5_HOME}/lib)

# Dependency: VOL Async
set(ASYNC_HOME $ENV{ASYNC_HOME})
option(WITH_ASYNC_VOL "" OFF)
if(WITH_ASYNC_VOL)
	if(${HDF5_VERSION} VERSION_GREATER_EQUAL "1.13.0")
		add_definitions(-DUSE_ASYNC_VOL)
		include_directories(${ASYNC_HOME})
		link_directories(${ASYNC_HOME})
		message(STATUS "VOL ASYNC: ENABLED")
	else()
		message(SEND_ERROR "VOL ASYNC requires HDF5 1.13.0 or newer.")
	endif()
endif()

# =========== Utilility libs ==============
set(h5bench_util_src
    commons/h5bench_util.c
    commons/h5bench_util.h
)
add_library(h5bench_util ${h5bench_util_src})
if(WITH_ASYNC_VOL)
	target_link_libraries(h5bench_util asynchdf5 h5async)
endif()
# =================================================================

# ===========  Build h5bench_patterns: h5bench_write and h5bench_read ===========
set(h5bench_write_src h5bench_patterns/h5bench_write.c)
set(h5bench_read_src h5bench_patterns/h5bench_read.c)

add_executable(h5bench_write ${h5bench_write_src})
target_link_libraries(h5bench_write h5bench_util hdf5 z MPI::MPI_C) 

add_executable(h5bench_read ${h5bench_read_src})
target_link_libraries(h5bench_read h5bench_util hdf5 z MPI::MPI_C)

# ===========  Build exerciser ===========
set(exerciser_src exerciser/h5bench_exerciser.c)
add_executable(h5bench_exerciser ${exerciser_src})
target_link_libraries(h5bench_exerciser hdf5 z m MPI::MPI_C)

# ===========  Build metadata stress test: hdf5_iotest  ===========
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
target_link_libraries(h5bench_hdf5_iotest h5bench_util hdf5 z m MPI::MPI_C)

# ===========  Build AMReX ===========
set(AMReX_HDF5 YES)
set(AMReX_PARTICLES YES)
set(AMReX_MPI_THREAD_MULTIPLE YES)
add_subdirectory(amrex)

set(amrex_src amrex/Tests/HDF5Benchmark/main.cpp)

add_executable(h5bench_amrex_sync ${amrex_src})
target_link_libraries(h5bench_amrex_sync hdf5 z m amrex MPI::MPI_C)

if(WITH_ASYNC_VOL)
	set(AMReX_HDF5_ASYNC YES)
	
	add_executable(h5bench_amrex_async ${amrex_src})
	target_link_libraries(h5bench_amrex_async hdf5 z m amrex asynchdf5 h5async MPI::MPI_C)
endif()
	
configure_file(${CMAKE_SOURCE_DIR}/h5bench ${CMAKE_BINARY_DIR}/h5bench COPYONLY)

# ===========  Build E3SM ===========

ExternalProject_Add(h5bench_e3sm
    SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/e3sm
    CONFIGURE_COMMAND autoreconf -i COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/e3sm/configure --prefix=${CMAKE_BINARY_DIR} --with-pnetcdf=${PNETCDF_HOME} --with-hdf5=${HDF5_HOME} CFLAGS=-fno-var-tracking-assignments CXXFLAGS=-fno-var-tracking-assignments
    BUILD_COMMAND make
    INSTALL_COMMAND ${CMAKE_COMMAND} -E copy src/e3sm_io ${CMAKE_BINARY_DIR}/h5bench_e3sm 
    BUILD_IN_SOURCE 1
    LOG_CONFIGURE 1
    LOG_BUILD 1
)

install(
	FILES
	h5bench
	PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ
	DESTINATION bin
)

install(
	TARGETS 
	h5bench_write 
	h5bench_read 
	h5bench_exerciser 
	h5bench_hdf5_iotest
	h5bench_amrex_sync
	#h5bench_vl_stream_hl
	DESTINATION bin
)

if(WITH_ASYNC_VOL)
	install(TARGETS 
		h5bench_amrex_async 
		DESTINATION bin
	)
endif()