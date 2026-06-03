# Shared helpers for h5bench benchmark targets.
#
# h5bench_add_pattern(<name>
#     SOURCES  <file> [<file>...]
#     [LIBS    <extra-libs>...]    # extra private libs (e.g. m)
#     [INSTALL]                    # add install(TARGETS ...) in bin/
# )
#
# Creates an executable <name> that links against h5bench_util (which
# transitively provides HDF5, MPI, and any VOL-ASYNC glue), plus the
# compression and dl libs every pattern needs. Callers only have to
# declare what actually varies per benchmark.
function(h5bench_add_pattern name)
    cmake_parse_arguments(PARSE_ARGV 1 H5BP "INSTALL" "" "SOURCES;LIBS")

    if(NOT H5BP_SOURCES)
        message(FATAL_ERROR "h5bench_add_pattern(${name}): SOURCES is required")
    endif()

    add_executable(${name} ${H5BP_SOURCES})
    target_link_libraries(${name}
        PRIVATE
            h5bench_util
            ZLIB::ZLIB
            ${CMAKE_DL_LIBS}
            ${H5BP_LIBS}
    )

    if(H5BP_INSTALL)
        install(TARGETS ${name} DESTINATION bin)
    endif()
endfunction()
