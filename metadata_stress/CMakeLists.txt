# Metadata stress test (hdf5_iotest).

add_executable(h5bench_hdf5_iotest
    hdf5_iotest.c
    configuration.c
    configuration.h
    dataset.c
    dataset.h
    ini.c
    ini.h
)

target_link_libraries(h5bench_hdf5_iotest
    PRIVATE
        h5bench_util
        ZLIB::ZLIB
        ${CMAKE_DL_LIBS}
        m
)

install(TARGETS h5bench_hdf5_iotest DESTINATION bin)
