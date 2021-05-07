cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
project(h5bench)

# Dependency: MPI
find_package(MPI REQUIRED)
include_directories(SYSTEM /usr/local/include)#for MPI
include_directories(${MPI_C_INCLUDE_PATH})

# Dependency: HDF5
set(HDF5_HOME $ENV{HDF5_HOME})
set(CMAKE_PREFIX_PATH ${HDF5_HOME})
find_package(HDF5 REQUIRED)
message("Detected hdf5 version: ${HDF5_VERSION}")
include_directories(${HDF5_HOME}/include)
include_directories(${HDF5_HOME}/../vol/include)
link_directories(${HDF5_HOME}/lib)
link_directories(${HDF5_HOME}/lib/plugin)

# Optional Dependency: CUDA
set(USE_GPU OFF)
find_package(CUDAToolkit)
if(CUDAToolkit_FOUND)
  enable_language(CUDA)
  set(HDF5_USE_GPU ON)
endif(CUDAToolkit_FOUND)
option(WITH_GPU "Compile Source Code with NVIDIA GPU" ${HDF5_USE_GPU})
message(STATUS "WITH_GPU=" ${WITH_GPU})

# Optional Dependency: HIP
# Search for rocm in common locations
if(WIN32)
  list(APPEND CMAKE_PREFIX_PATH "C:/hip")
  list(APPEND CMAKE_PREFIX_PATH "C:/Program Files/AMD HIP SDK/hip")
else()
  list(APPEND CMAKE_PREFIX_PATH /opt/rocm/hip /opt/rocm)
endif()

# Find HIP.
# The user may override AMDGPU_TARGETS defined in the HIP config file
# to select the AMDGPU archs to compile for.
# ex. set(AMDGPU_TARGETS "gfx803;gfx900;gfx906")
find_package(hip)
if(HIP_FOUND)
  set(HDF5_USE_HIP_GPU ON)
endif(HIP_FOUND)
# # Set compiler and linker.
# if(NOT WIN32)
#   set(CMAKE_CXX_COMPILER ${HIP_HIPCC_EXECUTABLE})
#   set(CMAKE_CXX_LINKER   ${HIP_HIPCC_EXECUTABLE})
# endif()

# # Preparing the executable.
# add_executable(test_openmp_helloworld ${CPP_SOURCES})
# 
# # Link Libraries - HIP Device and OpenMP.
# target_compile_options(test_openmp_helloworld PRIVATE ${OpenMP_CXX_FLAGS})
# target_link_libraries(test_openmp_helloworld PRIVATE hip::device ${OpenMP_CXX_FLAGS})
# 
# if(WIN32)
#   target_link_libraries(test_openmp_helloworld PRIVATE OpenMP::OpenMP_CXX)
# endif()
###################################################

#Dependency: Async
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

# ===========  Build basic_io: vpic and bdcats ===========
include_directories(${ASYNC_HOME})

if(WITH_GPU)
  ###########################################################3
  add_library(h5_memory SHARED
    gpu_io/h5_memory/h5_mem.c
    gpu_io/h5_memory/h5_mem.h

    # gpu_io/h5_memory/h5_xfer.c
    # gpu_io/h5_memory/h5_xfer.h
  )
  set_target_properties(h5_memory PROPERTIES COMPILE_DEFINITIONS HDF5_USE_CUDA)
  set_property(TARGET h5_memory PROPERTY CUDA_ARCHITECTURES 50 70 72)
  target_include_directories(h5_memory PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
  ###########################################################3

  ###########################################################3
  add_library(h5bench_gpu_kernels SHARED
    gpu_io/gpu_kernels/cuda_kernel.cu
    gpu_io/gpu_kernels/cuda_kernel.h
  )
  set_target_properties(h5bench_gpu_kernels PROPERTIES COMPILE_DEFINITIONS HDF5_USE_CUDA)
  set_property(TARGET h5bench_gpu_kernels PROPERTY CUDA_ARCHITECTURES 50 70 72)
  target_include_directories(h5bench_gpu_kernels PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
  ###########################################################3

  ###########################################################3
  set(vpic_gpu_src gpu_io/h5bench_vpicio_gpu.c)
  set(bdcats_gpu_src gpu_io/h5bench_bdcatsio_gpu.c)
  add_executable(h5bench_vpicio_cuda ${vpic_gpu_src})
  set_target_properties(h5bench_vpicio_cuda PROPERTIES COMPILE_DEFINITIONS HDF5_USE_CUDA)
  target_link_libraries(h5bench_vpicio_cuda PRIVATE h5_memory h5bench_gpu_kernels h5bench_util asynchdf5 h5async hdf5 z CUDA::cudart ${MPI_C_LIBRARIES} ${MPI_CXX_LINK_FLAGS})
  target_compile_options(h5bench_vpicio_cuda PRIVATE ${MPI_C_COMPILE_FLAGS})
  ###########################################################3
endif(WITH_GPU)

set(vpic_src basic_io/h5bench_vpicio.c)
set(bdcats_src basic_io/h5bench_bdcatsio.c)

#link_directories(${ASYNC_HOME})
# async libs: h5async asynchdf5
add_executable(h5bench_vpicio ${vpic_src})
target_link_libraries(h5bench_vpicio h5bench_util asynchdf5 h5async hdf5 z MPI::MPI_C)
#link_directories(${ASYNC_HOME}})

add_executable(h5bench_bdcatsio ${bdcats_src})
target_link_libraries(h5bench_bdcatsio h5bench_util hdf5 z MPI::MPI_C)

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

# =================================================================

# =========== Install binaries ===========
install(TARGETS 
	h5bench_vpicio 
	h5bench_bdcatsio 
	h5bench_exerciser 
	h5bench_hdf5_iotest 
	#h5bench_vl_stream_hl
	DESTINATION bin)

# TODO: add *_gpu to install bin
# if(WITH_GPU)
	# h5bench_vpicio_gpu
# endif(WITH_GPU)

