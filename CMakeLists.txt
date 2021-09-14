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

# ===========  GPUIO test: h5bench_write_cuda ===========
set(METAMEM_CUDA_SUPPORT OFF)
set(METAMEM_HIP_SUPPORT OFF)

find_package(CUDAToolkit)
if(CUDAToolkit_FOUND)
  enable_language(CUDA)
  set(METAMEM_CUDA_SUPPORT ON)
endif(CUDAToolkit_FOUND)

message(STATUS "METAMEM_CUDA_SUPPORT=" ${METAMEM_CUDA_SUPPORT})

if(METAMEM_CUDA_SUPPORT)
  file (GLOB METAMEM_API_CUDA_SRCS
    ./gpuio/metamemory/include/*.h
    ./gpuio/metamemory/src/*.c
    ./gpuio/metamemory/src/cuda/*.c
  )

  add_library(cuda_metamemapi SHARED ${METAMEM_API_CUDA_SRCS})
  set_target_properties(cuda_metamemapi PROPERTIES COMPILE_DEFINITIONS METAMEM_USE_CUDA)
  target_include_directories(cuda_metamemapi PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
  target_include_directories(cuda_metamemapi PRIVATE ./gpuio/metamemory/include/)

  add_executable(h5bench_cuda_write gpuio/h5bench_write.c)
  target_include_directories(h5bench_cuda_write PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
  set_target_properties(h5bench_cuda_write PROPERTIES COMPILE_DEFINITIONS METAMEM_USE_CUDA)
  target_link_libraries(h5bench_cuda_write h5bench_util hdf5 z MPI::MPI_C CUDA::cudart cuda_metamemapi)
endif(METAMEM_CUDA_SUPPORT)


# Search for rocm in common locations
list(APPEND CMAKE_PREFIX_PATH /opt/rocm/hip /opt/rocm)

# Find hip
find_package(HIP)
if(HIP_VERSION)
  set(METAMEM_HIP_SUPPORT ON)
  set(CMAKE_CXX_COMPILER ${HIP_HIPCC_EXECUTABLE})
  set(CMAKE_CXX_LINKER   ${HIP_HIPCC_EXECUTABLE})
endif(HIP_VERSION)

message(STATUS "METAMEM_HIP_SUPPORT=" ${METAMEM_HIP_SUPPORT})

if(METAMEM_HIP_SUPPORT)
  file (GLOB METAMEM_API_HIP_SRCS
    ./gpuio/metamemory/include/*.h
    ./gpuio/metamemory/src/*.c
    ./gpuio/metamemory/src/hip/*.c
  )

  set(ORIGINAL_CMAKE_C_COMPILER ${CMAKE_C_COMPILER})
  set(CMAKE_C_COMPILER amdclang) # TODO: hardcoded c compiler for now..
  add_library(hip_metamemapi SHARED ${METAMEM_API_HIP_SRCS})
  set_target_properties(hip_metamemapi PROPERTIES COMPILE_DEFINITIONS METAMEM_USE_HIP)
  target_include_directories(hip_metamemapi PRIVATE ${HIP_ROOT_DIR}/include)
  target_include_directories(hip_metamemapi PRIVATE ./gpuio/metamemory/include/)
  target_link_libraries(hip_metamemapi hip::host)

  set(CMAKE_C_COMPILER ${ORIGINAL_CMAKE_C_COMPILER})
  add_executable(h5bench_hip_write gpuio/h5bench_write.c)
  set_target_properties(h5bench_hip_write PROPERTIES COMPILE_DEFINITIONS METAMEM_USE_HIP)
  target_link_libraries(h5bench_hip_write h5bench_util hdf5 z MPI::MPI_C hip::host hip_metamemapi)
endif(METAMEM_HIP_SUPPORT)
# =================================================================

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

# =================================================================

# =========== Install binaries ===========
install(TARGETS 
	h5bench_write 
	h5bench_read 
	h5bench_exerciser 
	h5bench_hdf5_iotest 
	#h5bench_vl_stream_hl
	DESTINATION bin)
