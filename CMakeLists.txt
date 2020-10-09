cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
project(h5bench)

# Dependency: MPI
find_package(MPI REQUIRED)
include_directories(SYSTEM /usr/local/include)#for MPI

# Dependecy: HDF5
find_package(HDF5 REQUIRED)
#set(HDF5_HOME "/Users/tonglin/nersc_dev_sync/hdf5_build/hdf5")
set(HDF5_HOME $ENV{HDF5_HOME})
include_directories(${HDF5_HOME}/include)
link_directories(${HDF5_HOME}/lib)

# =========== Utilility libs ==============
set(h5bench_util_src
    commons/h5bench_util.c
    commons/h5bench_util.h
    )
add_library(h5bench_util ${h5bench_util_src})
# =================================================================

# ===========  Build basic_io: vpic and bdcats ===========
set(vpic_src basic_io/h5bench_vpicio.c)
set(bdcats_src basic_io/h5bench_bdcatsio.c)
add_executable(h5bench_vpicio ${vpic_src})
target_link_libraries(h5bench_vpicio h5bench_util hdf5 z MPI::MPI_C)
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

# =================================================================

# =========== Install binaries ===========
install(TARGETS h5bench_vpicio h5bench_bdcatsio h5bench_exerciser h5bench_hdf5_iotest DESTINATION bin)
