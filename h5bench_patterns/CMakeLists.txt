# Core pattern benchmarks — all share the same link recipe via
# h5bench_add_pattern(), which is defined in cmake/h5benchFunctions.cmake
# and loaded by the top-level CMakeLists.txt.

h5bench_add_pattern(h5bench_write
    SOURCES h5bench_write.c
    INSTALL
)

h5bench_add_pattern(h5bench_write_unlimited
    SOURCES h5bench_write_unlimited.c
    INSTALL
)

h5bench_add_pattern(h5bench_write_var_normal_dist
    SOURCES h5bench_write_normal_dist.c
    LIBS    m
    INSTALL
)

h5bench_add_pattern(h5bench_overwrite
    SOURCES h5bench_overwrite.c
    INSTALL
)

h5bench_add_pattern(h5bench_append
    SOURCES h5bench_append.c
    INSTALL
)

h5bench_add_pattern(h5bench_read
    SOURCES h5bench_read.c
    INSTALL
)
