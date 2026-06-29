# HDF5 Exerciser — stand-alone benchmark, does not use h5bench_util.

add_executable(h5bench_exerciser h5bench_exerciser.c)

target_include_directories(h5bench_exerciser
    PRIVATE
        ${HDF5_INCLUDE_DIRS}
)

target_link_libraries(h5bench_exerciser
    PRIVATE
        ${HDF5_LIBRARIES}
        MPI::MPI_C
        ZLIB::ZLIB
        ${CMAKE_DL_LIBS}
        m
)

install(TARGETS h5bench_exerciser DESTINATION bin)
