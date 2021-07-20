
#include "configuration.h"

#include <assert.h>
#include <stdlib.h>
#include <string.h>

/*
 *
 * Handle parameter conversion
 *
 */

int handler(void* user,
            const char* section,
            const char* name,
            const char* value)
{
  configuration* pconfig = (configuration*)user;

#define MATCH(s, n) strcmp(section, s) == 0 && strcmp(name, n) == 0

  if (strncmp(section, "DEFAULT", 7) == 0)
    {
      if (MATCH("DEFAULT", "version")) {
        pconfig->version = atoi(value);
      } else if (MATCH("DEFAULT", "steps")) {
        pconfig->steps = (unsigned int) atoi(value);
      } else if (MATCH("DEFAULT", "arrays")) {
        pconfig->arrays = (unsigned int) atoi(value);
      } else if (MATCH("DEFAULT", "rows")) {
        pconfig->rows = (unsigned long) atol(value);
      } else if (MATCH("DEFAULT", "columns")) {
        pconfig->cols = (unsigned long) atol(value);
      } else if (MATCH("DEFAULT", "process-rows")) {
        pconfig->proc_rows = (unsigned int) atoi(value);
      } else if (MATCH("DEFAULT", "process-columns")) {
        pconfig->proc_cols = (unsigned int) atoi(value);
      } else if (MATCH("DEFAULT", "scaling")) {
        strncpy(pconfig->scaling, value, 16);
      } else if (MATCH("DEFAULT", "dataset-rank")) {
        pconfig->rank = (unsigned int) atoi(value);
      } else if (MATCH("DEFAULT", "slowest-dimension")) {
        strncpy(pconfig->slowest_dimension, value, 16);
      } else if (MATCH("DEFAULT", "layout")) {
        strncpy(pconfig->layout, value, 16);
      } else if (MATCH("DEFAULT", "mpi-io")) {
        strncpy(pconfig->mpi_io, value, 16);
      } else if (MATCH("DEFAULT", "hdf5-file")) {
        strncpy(pconfig->hdf5_file, value, PATH_MAX);
      } else if (MATCH("DEFAULT", "csv-file")) {
        strncpy(pconfig->csv_file, value, PATH_MAX);
      } else {
        return 0;  /* unknown section/name, error */
      }

    }
  else
    {
      if (MATCH(section, "steps")) {
        pconfig->steps = (unsigned int) atoi(value);
      } else if (MATCH(section, "arrays")) {
        pconfig->arrays = (unsigned int) atoi(value);
      } else if (MATCH(section, "rows")) {
        pconfig->rows = (unsigned long) atol(value);
      } else if (MATCH(section, "columns")) {
        pconfig->cols = (unsigned long) atol(value);
      } else if (MATCH(section, "process-rows")) {
        pconfig->proc_rows = (unsigned int) atoi(value);
      } else if (MATCH(section, "process-columns")) {
        pconfig->proc_cols = (unsigned int) atoi(value);
      } else if (MATCH(section, "scaling")) {
        strncpy(pconfig->scaling, value, 16);
      } else if (MATCH(section, "dataset-rank")) {
        pconfig->rank = (unsigned int) atoi(value);
      } else if (MATCH(section, "slowest-dimension")) {
        strncpy(pconfig->slowest_dimension, value, 16);
      } else if (MATCH(section, "layout")) {
        strncpy(pconfig->layout, value, 16);
      } else if (MATCH(section, "mpi-io")) {
        strncpy(pconfig->mpi_io, value, 16);
      } else if (MATCH(section, "hdf5-file")) {
        strncpy(pconfig->hdf5_file, value, PATH_MAX);
      } else if (MATCH(section, "csv-file")) {
        strncpy(pconfig->csv_file, value, PATH_MAX);
      } else {
        return 0;  /* unknown name, error */
      }
    }

  return 1;
}

/*
 *
 * Check if the parameters have sensible values
 *
 */

int sanity_check(void* user)
{
  configuration* pconfig = (configuration*)user;

  assert(pconfig->version == 0);
  assert(pconfig->steps > 0);
  assert(pconfig->arrays > 0);
  assert(pconfig->rows > 1);
  assert(pconfig->proc_rows >= 1);
  assert(pconfig->cols > 1);
  assert(pconfig->proc_cols >= 1);
  assert(strncmp(pconfig->scaling, "weak", 16) == 0 ||
         strncmp(pconfig->scaling, "strong", 16) == 0);
  assert(pconfig->rank > 1 && pconfig->rank < 5);
  assert(strncmp(pconfig->slowest_dimension, "step", 16) == 0 ||
         strncmp(pconfig->slowest_dimension, "array", 16) == 0);
  assert(strncmp(pconfig->layout, "contiguous", 16) == 0 ||
         strncmp(pconfig->layout, "chunked", 16) == 0);
  assert(strncmp(pconfig->mpi_io, "independent", 16) == 0 ||
         strncmp(pconfig->mpi_io, "collective", 16) == 0);

  return 0;
}

/*
 *
 * Validate the configuration
 *
 */

int validate(void* user, const int size)
{
  configuration* pconfig = (configuration*)user;

  assert(pconfig->proc_rows*pconfig->proc_cols == (unsigned)size);

  if (strncmp(pconfig->scaling, "strong", 16) == 0) {
    assert(pconfig->rows%pconfig->proc_rows == 0);
    assert(pconfig->cols%pconfig->proc_cols == 0);
  }

  return 0;
}
