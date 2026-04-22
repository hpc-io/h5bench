# h5bench_util: shared utility library used by every pattern binary and
# the metadata benchmark. Propagates HDF5 + MPI (and optionally VOL-ASYNC
# and subfiling) to consumers via PUBLIC properties so callers just link
# h5bench_util and get the rest for free.

add_library(h5bench_util
    h5bench_util.c
    h5bench_util.h
)

target_include_directories(h5bench_util
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        ${HDF5_INCLUDE_DIRS}
)

target_link_libraries(h5bench_util
    PUBLIC
        MPI::MPI_C
        ${HDF5_LIBRARIES}
    PRIVATE
        ${CMAKE_DL_LIBS}
)

if(HAVE_SUBFILING_VFD)
    target_compile_definitions(h5bench_util PUBLIC HAVE_SUBFILING)
endif()

if(WITH_ASYNC_VOL)
    target_compile_definitions(h5bench_util PUBLIC USE_ASYNC_VOL)
    target_include_directories(h5bench_util PUBLIC ${ASYNC_HOME}/include)
    target_link_directories(h5bench_util PUBLIC ${ASYNC_HOME}/lib)
    target_link_libraries(h5bench_util PUBLIC asynchdf5 h5async)
endif()
