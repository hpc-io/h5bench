cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
project(h5bench)

# cmake_policy(SET CMP0074 NEW)

# Dependency: MPI
find_package(MPI REQUIRED)
include_directories(SYSTEM /usr/local/include)#for MPI

# Dependency: HDF5
set(HDF5_HOME $ENV{HDF5_HOME})
set(CMAKE_PREFIX_PATH ${HDF5_HOME})
find_package(HDF5 REQUIRED)
message("Detected hdf5 version: ${HDF5_VERSION}")
include_directories(${HDF5_HOME}/include)
link_directories(${HDF5_HOME}/lib)

# Dependency: VOL Async
set(ASYNC_HOME $ENV{ASYNC_HOME})
option(WITH_ASYNC_VOL "" OFF)
if(WITH_ASYNC_VOL)
	add_definitions(-DUSE_ASYNC_VOL)
	include_directories(${ASYNC_HOME})
	link_directories(${ASYNC_HOME})
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

# ===========  Streaming (append) test: h5bench_vl_stream ===========
#message("============== Build without USE_DEV_VL ==============")
#add_executable(h5bench_vl_stream_hl streaming/h5bench_vl_stream.c)
#target_link_libraries(h5bench_vl_stream_hl h5bench_util hdf5 hdf5_hl z)


#set(HDF5_VL_HOME $ENV{HDF5_VL_HOME})
#include_directories(${HDF5_VL_HOME}/include)
#link_directories(${HDF5_VL_HOME}/lib)
#find_package(HDF5 1.13.0)
#	message("============== Build with USE_DEV_VL ==============")
#	set(HDF5_VL_HOME $ENV{HDF5_VL_HOME})
#	include_directories(${HDF5_VL_HOME}/include)
#	link_directories(${HDF5_VL_HOME}/lib)
#	message("hdf5 home path = ${HDF5_VL_HOME}")
#	add_executable(h5bench_vl_stream_lib streaming/h5bench_vl_stream.c)
#	target_compile_definitions(h5bench_vl_stream_lib PUBLIC -DDEV_VL)
#	target_link_libraries(h5bench_vl_stream_lib h5bench_util ${HDF5_VL_HOME}/lib/libhdf5.a z)
#	install(TARGET h5bench_vl_stream_lib DESTINATION bin)
#endif()

# ===========  Build AMReX ===========
set(AMReX_HDF5 YES)
set(AMReX_PARTICLES YES)
set(AMReX_MPI_THREAD_MULTIPLE YES)
add_subdirectory(amrex)

set(amrex_src amrex/Tests/HDF5Benchmark/main.cpp)
add_executable(h5bench_amrex ${amrex_src})

if(WITH_ASYNC_VOL)
	set(AMREX_USE_HDF5_ASYNC YES)
	target_link_libraries(h5bench_amrex hdf5 z m amrex asynchdf5 h5async MPI::MPI_C)
else()
	target_link_libraries(h5bench_amrex hdf5 z m amrex MPI::MPI_C)
endif()

configure_file(${CMAKE_SOURCE_DIR}/h5bench ${CMAKE_BINARY_DIR}/h5bench COPYONLY)

# =================================================================

# =========== Install binaries ===========
install(
	FILES
	h5bench
	PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ
	DESTINATION bin
)
install(TARGETS 
	h5bench_write 
	h5bench_read 
	h5bench_exerciser 
	h5bench_hdf5_iotest 
	#h5bench_vl_stream_hl
	DESTINATION bin
)

