# - Find PNetCDF
# Find the native PNetCDF includes and library
#
#  PNETCDF_INCLUDES    - where to find netcdf.h, etc
#  PNETCDF_LIBRARIES   - Link these libraries when using NetCDF
#  PNETCDF_FOUND       - True if PNetCDF was found
#
# Normal usage would be:
#  find_package (PNetCDF REQUIRED)
#  target_link_libraries (uses_pnetcdf ${PNETCDF_LIBRARIES})

if (PNETCDF_INCLUDES AND PNETCDF_LIBRARIES)
  # Already in cache, be silent
  set (PNETCDF_FIND_QUIETLY TRUE)
endif (PNETCDF_INCLUDES AND PNETCDF_LIBRARIES)

find_path (PNETCDF_INCLUDES pnetcdf.h
  HINTS "${PNETCDF_ROOT}/include" "$ENV{PNETCDF_ROOT}/include")

string(REGEX REPLACE "/include/?$" "/lib"
  PNETCDF_LIB_HINT ${PNETCDF_INCLUDES})

find_library (PNETCDF_LIBRARIES
  NAMES pnetcdf
  HINTS ${PNETCDF_LIB_HINT})

if ((NOT PNETCDF_LIBRARIES) OR (NOT PNETCDF_INCLUDES))
  message(STATUS "Trying to find PNetCDF using LD_LIBRARY_PATH (we're desperate)...")

  file(TO_CMAKE_PATH "$ENV{LD_LIBRARY_PATH}" LD_LIBRARY_PATH)

  find_library(PNETCDF_LIBRARIES
    NAMES pnetcdf
    HINTS ${LD_LIBRARY_PATH})

  if (PNETCDF_LIBRARIES)
    get_filename_component(PNETCDF_LIB_DIR ${PNETCDF_LIBRARIES} PATH)
    string(REGEX REPLACE "/lib/?$" "/include"
      PNETCDF_H_HINT ${PNETCDF_LIB_DIR})

    find_path (PNETCDF_INCLUDES pnetcdf.h
      HINTS ${PNETCDF_H_HINT}
      DOC "Path to pnetcdf.h")
  endif()
endif()

string(REGEX REPLACE "/include/?$" "" PNETCDF_HOME "${PNETCDF_INCLUDES}")

# handle the QUIETLY and REQUIRED arguments and set PNETCDF_FOUND to TRUE if
# all listed variables are TRUE
include (FindPackageHandleStandardArgs)
find_package_handle_standard_args (PNetCDF DEFAULT_MSG PNETCDF_LIBRARIES PNETCDF_INCLUDES)

mark_as_advanced (PNETCDF_HOME PNETCDF_LIBRARIES PNETCDF_INCLUDES)
