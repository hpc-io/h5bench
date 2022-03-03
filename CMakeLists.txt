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

set(CMAKE_BUILD_TYPE Debug)

# ===========  Build h5bench_patterns: h5bench_write and h5bench_read ===========
set(h5bench_write_src h5bench_patterns/h5bench_write.c)
set(h5bench_write_unlimited_src h5bench_patterns/h5bench_write_unlimited.c)
set(h5bench_read_src h5bench_patterns/h5bench_read.c)

add_executable(h5bench_write ${h5bench_write_src})
target_link_libraries(h5bench_write h5bench_util hdf5 z MPI::MPI_C) 

add_executable(h5bench_write_unlimited ${h5bench_write_unlimited_src})
target_link_libraries(h5bench_write_unlimited h5bench_util hdf5 z MPI::MPI_C) 

add_executable(h5bench_read ${h5bench_read_src})
target_link_libraries(h5bench_read h5bench_util hdf5 z MPI::MPI_C)

# ===========  Build h5bench_patterns: h5bench_overwrite ===========
set(h5bench_overwrite_src h5bench_patterns/h5bench_overwrite.c)

add_executable(h5bench_overwrite ${h5bench_overwrite_src})
target_link_libraries(h5bench_overwrite h5bench_util hdf5 z MPI::MPI_C)

# ===========  Build h5bench_patterns: h5bench_append ===========
set(h5bench_append_src h5bench_patterns/h5bench_append.c)

add_executable(h5bench_append ${h5bench_append_src})
target_link_libraries(h5bench_append h5bench_util hdf5 z MPI::MPI_C)

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
set(MEM_CUDA_SUPPORT OFF)
set(MEM_HIP_SUPPORT OFF)

find_package(CUDAToolkit)
if(CUDAToolkit_FOUND)
  enable_language(CUDA)
  set(MEM_CUDA_SUPPORT ON)
endif(CUDAToolkit_FOUND)

message(STATUS "MEM_CUDA_SUPPORT=" ${MEM_CUDA_SUPPORT})

if(MEM_CUDA_SUPPORT)
  file (GLOB MEM_API_CUDA_SRCS
    ./gpuio/mem/include/*.h
    ./gpuio/mem/src/*.c
    ./gpuio/mem/src/cuda/*.c
  )

  add_library(cuda_memapi SHARED ${MEM_API_CUDA_SRCS})
  set_target_properties(cuda_memapi PROPERTIES COMPILE_DEFINITIONS METAMEM_USE_CUDA)
  target_include_directories(cuda_memapi PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
  target_include_directories(cuda_memapi PRIVATE ./gpuio/mem/include/)
  target_link_libraries(cuda_memapi PRIVATE CUDA::cudart CUDA::cuda_driver)

  add_executable(h5bench_cuda_write gpuio/h5bench_gpu_write.c)
  target_include_directories(h5bench_cuda_write PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
  set_target_properties(h5bench_cuda_write PROPERTIES COMPILE_DEFINITIONS METAMEM_USE_CUDA)
  target_link_libraries(h5bench_cuda_write h5bench_util hdf5 z MPI::MPI_C CUDA::cudart CUDA::cuda_driver cuda_memapi)

  add_executable(h5bench_cuda_read gpuio/h5bench_gpu_read.c)
  target_include_directories(h5bench_cuda_read PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
  set_target_properties(h5bench_cuda_read PROPERTIES COMPILE_DEFINITIONS METAMEM_USE_CUDA)
  target_link_libraries(h5bench_cuda_read h5bench_util hdf5 z MPI::MPI_C CUDA::cudart CUDA::cuda_driver cuda_memapi)

  # BUILD ctests
  enable_testing()
  # 1 GPU, cuda, tested on summit
  add_test( h5bench_cuda_write_00 jsrun --nrs 1 -c1 -g1 ./h5bench_cuda_write ../gpuio/sample_config/sample_write_cc1d_fileperproc.cfg ./test.h5 )
  add_test( h5bench_cuda_read_01 jsrun --nrs 1 -c1 -g1 ./h5bench_cuda_read ../gpuio/sample_config/sample_read_cc1d_fileperproc.cfg ./test.h5 )

  # 6 GPU, cuda, tested on summit
  # add_test( h5bench_cuda_write_02 jsrun --nrs 6 -c1 -g1 ./h5bench_cuda_write ../gpuio/sample_config/sample_write_cc1d_fileperproc.cfg ./test.h5 )

endif(MEM_CUDA_SUPPORT)


# Search for rocm in common locations
list(APPEND CMAKE_PREFIX_PATH /opt/rocm/hip /opt/rocm)

# Find hip
find_package(HIP)
if(HIP_VERSION)
  set(MEM_HIP_SUPPORT ON)
  set(CMAKE_CXX_COMPILER ${HIP_HIPCC_EXECUTABLE})
  set(CMAKE_CXX_LINKER   ${HIP_HIPCC_EXECUTABLE})
endif(HIP_VERSION)

message(STATUS "MEM_HIP_SUPPORT=" ${MEM_HIP_SUPPORT})

if(MEM_HIP_SUPPORT)
  file (GLOB MEM_API_HIP_SRCS
    ./gpuio/mem/include/*.h
    ./gpuio/mem/src/*.c
    ./gpuio/mem/src/hip/*.c
  )

  set(ORIGINAL_CMAKE_C_COMPILER ${CMAKE_C_COMPILER})
  set(CMAKE_C_COMPILER amdclang) # TODO: hardcoded c compiler for now..
  add_library(hip_memapi SHARED ${MEM_API_HIP_SRCS})
  set_target_properties(hip_memapi PROPERTIES COMPILE_DEFINITIONS METAMEM_USE_HIP)
  target_include_directories(hip_memapi PRIVATE ${HIP_ROOT_DIR}/include)
  target_include_directories(hip_memapi PRIVATE ./gpuio/mem/include/)
  target_link_libraries(hip_memapi hip::host)

  set(CMAKE_C_COMPILER ${ORIGINAL_CMAKE_C_COMPILER})
  add_executable(h5bench_hip_write gpuio/h5bench_write.c)
  set_target_properties(h5bench_hip_write PROPERTIES COMPILE_DEFINITIONS METAMEM_USE_HIP)
  target_link_libraries(h5bench_hip_write h5bench_util hdf5 z MPI::MPI_C hip::host hip_memapi)
endif(MEM_HIP_SUPPORT)
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

# ===========  Build AMReX ===========
set(AMReX_HDF5 YES)
set(AMReX_PARTICLES YES)
set(AMReX_MPI_THREAD_MULTIPLE YES)
add_subdirectory(amrex)

set(amrex_src amrex/Tests/HDF5Benchmark/main.cpp)

add_executable(h5bench_amrex_sync ${amrex_src})
target_link_libraries(h5bench_amrex_sync hdf5 z m amrex pthread MPI::MPI_C)

if(WITH_ASYNC_VOL)
	set(AMReX_HDF5_ASYNC YES)
	
	add_executable(h5bench_amrex_async ${amrex_src})
	target_link_libraries(h5bench_amrex_async hdf5 z m amrex pthread asynchdf5 h5async MPI::MPI_C)
endif()
	
configure_file(${CMAKE_SOURCE_DIR}/h5bench ${CMAKE_BINARY_DIR}/h5bench COPYONLY)

# =================================================================

# ===========  Build OpenPMD ===========
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
	h5bench_write_unlimited
	h5bench_read 
	h5bench_overwrite
	h5bench_append
	h5bench_exerciser 
	h5bench_hdf5_iotest 
	h5bench_openpmd_write
	h5bench_openpmd_read
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
