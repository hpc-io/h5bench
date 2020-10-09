#ifndef CONFIGURATION_H
#define CONFIGURATION_H

#include "ini.h"

#include <limits.h>

/* Configuration parameters */

typedef struct
{
  int version;
  unsigned int steps;
  unsigned int arrays;
  unsigned long rows;
  unsigned long cols;
  unsigned int proc_rows;
  unsigned int proc_cols;
  char scaling[16];
  unsigned int rank;
  char slowest_dimension[16];
  char layout[16];
  char mpi_io[16];
  char hdf5_file[PATH_MAX];
  char csv_file[PATH_MAX];
} configuration;

extern int handler(void* user,
                   const char* section,
                   const char* name,
                   const char* value);

extern int sanity_check(void* user);

extern int validate(void* user, const int size);

#endif
