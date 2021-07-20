#ifndef DATASET_H
#define DATASET_H

#include "configuration.h"

#include "hdf5.h"

extern hid_t create_dataset(const configuration* config,
                            hid_t file,
                            const char* name);

extern int create_selection(const configuration* config,
                            hid_t fspace,
                            const int proc_row,
                            const int proc_col,
                            const unsigned int step,
                            const unsigned int array);

#endif
