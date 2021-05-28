cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
project(h5bench)

#########################################
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wextra -Wshadow -Wpedantic")
set(CMAKE_C_FLAGS_DEBUG "-g -O0")
set(CMAKE_C_FLAGS_RELEASE "-O3")

set(CMAKE_BUILD_TYPE Release)
# set(CMAKE_BUILD_TYPE Debug)
# set(CMAKE_BUILD_TYPE RelWithDebInfo)
################################################

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

#Dependency: Async
set(ASYNC_HOME $ENV{ASYNC_HOME})
option(WITH_ASYNC_VOL "" OFF)
if(WITH_ASYNC_VOL)
  add_definitions(-DUSE_ASYNC_VOL)
endif()

##############################################
set(H5BENCH_GPUIO OFF)
set(H5BENCH_CUDA_SUPPORT OFF)
set(H5BENCH_HIP_SUPPORT OFF)
set(H5BENCH_OneAPI_SUPPORT OFF)

# Optional Dependency: CUDA
find_package(CUDAToolkit)
if(CUDAToolkit_FOUND)
  enable_language(CUDA)
  set(H5BENCH_GPUIO ON)
  set(H5BENCH_CUDA_SUPPORT ON)
endif(CUDAToolkit_FOUND)

# Optional Dependency: HIP
# Search for rocm in common locations
if(WIN32)
  list(APPEND CMAKE_PREFIX_PATH "C:/hip")
  list(APPEND CMAKE_PREFIX_PATH "C:/Program Files/AMD HIP SDK/hip")
else()
  list(APPEND CMAKE_PREFIX_PATH /opt/rocm/hip /opt/rocm)
endif()
find_package(HIP QUIET)
if(HIP_FOUND)
  message(STATUS "Found HIP: " ${HIP_VERSION})
  set(H5BENCH_GPUIO ON)
  set(H5BENCH_HIP_SUPPORT ON)
endif()

# TODO: Optional Dependency: OneAPI

option(WITH_GPUIO "Compile GPU IO Source Code" ${H5BENCH_GPUIO})
message(STATUS "WITH_GPUIO=" ${WITH_GPUIO})
message(STATUS "H5BENCH_CUDA_SUPPORT=" ${H5BENCH_CUDA_SUPPORT})
message(STATUS "H5BENCH_HIP_SUPPORT=" ${H5BENCH_HIP_SUPPORT})
message(STATUS "H5BENCH_OneAPI_SUPPORT=" ${H5BENCH_OneAPI_SUPPORT})

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

if(WITH_GPUIO)
  ###########################################################3
  add_library(h5deepmem SHARED
    gpu_io/h5deepmem/h5deepmem_globals.h

    gpu_io/h5deepmem/h5deepmem.c
    gpu_io/h5deepmem/h5deepmem.h

    gpu_io/h5deepmem/h5mem.c
    gpu_io/h5deepmem/h5mem.h

    gpu_io/h5deepmem/h5xfer.c
    gpu_io/h5deepmem/h5xfer.h

    gpu_io/h5deepmem/h5device.c
    gpu_io/h5deepmem/h5device.h
  )

  if(H5BENCH_CUDA_SUPPORT)
    set_target_properties(h5deepmem PROPERTIES COMPILE_DEFINITIONS HDF5_USE_CUDA)
    set_property(TARGET h5deepmem PROPERTY CUDA_ARCHITECTURES 50 70 72)
    target_include_directories(h5deepmem PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
  endif(H5BENCH_CUDA_SUPPORT)


  # if(H5BENCH_HIP_SUPPORT)
  #   set_target_properties(h5deepmem PROPERTIES COMPILE_DEFINITIONS HDF5_USE_HIP)
  #   target_link_libraries(h5deepmem PRIVATE hip_hcc)
  #   # set_property(TARGET h5deepmem PROPERTY CUDA_ARCHITECTURES 50 70 72)
  #   target_include_directories(h5deepmem PRIVATE ${ROCM_PATH}/include)
  # endif(H5BENCH_HIP_SUPPORT)

  set_target_properties(h5deepmem PROPERTIES INTERPROCEDURAL_OPTIMIZATION TRUE)
  ###########################################################3

  ###########################################################3
  add_library(h5bench_gpu_kernels SHARED
    gpu_io/gpu_kernels/cuda_kernel.cu
    gpu_io/gpu_kernels/cuda_kernel.h
  )

  if(H5BENCH_CUDA_SUPPORT)
    set_target_properties(h5bench_gpu_kernels PROPERTIES COMPILE_DEFINITIONS HDF5_USE_CUDA)
    set_property(TARGET h5bench_gpu_kernels PROPERTY CUDA_ARCHITECTURES 50 70 72)
    target_include_directories(h5bench_gpu_kernels PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
  endif(H5BENCH_CUDA_SUPPORT)

  set_target_properties(h5bench_gpu_kernels PROPERTIES INTERPROCEDURAL_OPTIMIZATION TRUE)
  ###########################################################3

  ###########################################################3
  set(vpicio_gpu_src gpu_io/h5bench_vpicio_gpu.c)
  set(bdcatsio_gpu_src gpu_io/h5bench_bdcatsio_gpu.c)

  add_executable(h5bench_vpicio_gpuio ${vpicio_gpu_src})
  set_target_properties(h5bench_vpicio_gpuio PROPERTIES COMPILE_DEFINITIONS HDF5_USE_CUDA)
  target_link_libraries(h5bench_vpicio_gpuio PRIVATE h5deepmem h5bench_gpu_kernels h5bench_util asynchdf5 h5async hdf5 z CUDA::cudart ${MPI_C_LIBRARIES} ${MPI_CXX_LINK_FLAGS})
  target_compile_options(h5bench_vpicio_gpuio PRIVATE ${MPI_C_COMPILE_FLAGS})
  set_target_properties(h5bench_vpicio_gpuio PROPERTIES INTERPROCEDURAL_OPTIMIZATION TRUE)

   add_executable(h5bench_bdcatsio_gpuio ${bdcatsio_gpu_src})
  set_target_properties(h5bench_bdcatsio_gpuio PROPERTIES COMPILE_DEFINITIONS HDF5_USE_CUDA)
  target_link_libraries(h5bench_bdcatsio_gpuio PRIVATE h5deepmem h5bench_gpu_kernels h5bench_util asynchdf5 h5async hdf5 z CUDA::cudart ${MPI_C_LIBRARIES} ${MPI_CXX_LINK_FLAGS})
  target_compile_options(h5bench_bdcatsio_gpuio PRIVATE ${MPI_C_COMPILE_FLAGS})
  set_target_properties(h5bench_bdcatsio_gpuio PROPERTIES INTERPROCEDURAL_OPTIMIZATION TRUE)
  ###########################################################3
endif(WITH_GPUIO)

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
# if(WITH_GPUIO)
  # h5bench_vpicio_gpu
  # endif(WITH_GPUIO)

