#include "dataset.h"

#include <assert.h>
#include <string.h>

/*
 *
 * Create an anonymous dataset for the current configuration
 * (this then needs to be linked into the file)
 *
 */

hid_t create_dataset(const configuration* config, hid_t file, const char* name)
{
  hid_t result = -1, fspace, dcpl, lcpl;
  unsigned int strong_scaling_flg, step_first_flg, chunked_flg;
  unsigned long total_rows, total_cols, my_rows, my_cols;
  hsize_t dims[H5S_MAX_RANK], max_dims[H5S_MAX_RANK], cdims[H5S_MAX_RANK];

  strong_scaling_flg = (strncmp(config->scaling, "strong", 16) == 0);
  total_rows = strong_scaling_flg ?
    config->rows : config->proc_rows*config->rows;
  total_cols = strong_scaling_flg ?
    config->cols : config->proc_cols*config->cols;
  my_rows = strong_scaling_flg ? config->rows/config->proc_rows : config->rows;
  my_cols = strong_scaling_flg ? config->cols/config->proc_cols : config->cols;

  step_first_flg = (strncmp(config->slowest_dimension, "step", 16) == 0);
  chunked_flg = (strncmp(config->layout, "chunked", 16) == 0);

  switch (config->rank)
    {
    case 2:
      max_dims[0] = dims[0] = (hsize_t)total_rows;
      max_dims[1] = dims[1] = (hsize_t)total_cols;
      if (chunked_flg)
        {
          cdims[0] = (hsize_t)my_rows;
          cdims[1] = (hsize_t)my_cols;
        }
      break;
    case 3:
      max_dims[0] = dims[0] = (hsize_t)
        (step_first_flg ? config->arrays : config->steps);
      max_dims[1] = dims[1] = (hsize_t)total_rows;
      max_dims[2] = dims[2] = (hsize_t)total_cols;
      if (chunked_flg)
        {
          if (!step_first_flg)
            max_dims[0] = H5S_UNLIMITED;
          cdims[0] = 1;
          cdims[1] = (hsize_t)my_rows;
          cdims[2] = (hsize_t)my_cols;
        }
      break;
    case 4:
      max_dims[0] = dims[0] = (hsize_t)
        (step_first_flg ? config->steps : config->arrays);
      max_dims[1] = dims[1] = (hsize_t)
        (step_first_flg ? config->arrays : config->steps);
      max_dims[2] = dims[2] = (hsize_t)total_rows;
      max_dims[3] = dims[3] = (hsize_t)total_cols;
      if (chunked_flg)
        {
          if (step_first_flg)
            {
              cdims[0] = 1;
              max_dims[0] = H5S_UNLIMITED;
              cdims[1] = (hsize_t)config->arrays;
            }
          else
            {
              cdims[0] = (hsize_t)config->arrays;
              max_dims[1] = H5S_UNLIMITED;
              cdims[1] = 1;
            }
          cdims[2] = (hsize_t)my_rows;
          cdims[3] = (hsize_t)my_cols;
        }
      break;
    default:
      break;
    }

  assert((fspace = H5Screate_simple(config->rank, dims, max_dims)) >= 0);

  assert((dcpl = H5Pcreate(H5P_DATASET_CREATE)) >= 0);
  if (chunked_flg)
  {
    assert(H5Pset_chunk(dcpl, config->rank, cdims) >= 0);
#ifdef NO_CHUNK_INIT
    assert(H5Pset_fill_time(dcpl, H5D_FILL_TIME_NEVER) >= 0);
#endif
  }

  assert((lcpl = H5Pcreate(H5P_LINK_CREATE)) >= 0);
  assert(H5Pset_create_intermediate_group(lcpl, 1) >= 0);

  assert((result = H5Dcreate(file, name, H5T_NATIVE_DOUBLE, fspace,
                             lcpl, dcpl, H5P_DEFAULT)) >= 0);
  assert(H5Pclose(lcpl) >= 0);
  assert(H5Pclose(dcpl) >= 0);
  assert(H5Sclose(fspace) >= 0);

  return result;
}

/*
 *
 * Create an in-file dataspace selection depending on the step and variable
 *
 */

int create_selection(const configuration* config,
                        hid_t fspace,
                        const int proc_row,
                        const int proc_col,
                        const unsigned int step,
                        const unsigned int array)
{
  hid_t result = 0;
  unsigned int strong_scaling_flg, step_first_flg;
  unsigned long my_rows, my_cols;
  hsize_t start[H5S_MAX_RANK], count[H5S_MAX_RANK], block[H5S_MAX_RANK];

  strong_scaling_flg = (strncmp(config->scaling, "strong", 16) == 0);
  my_rows = strong_scaling_flg ? config->rows/config->proc_rows : config->rows;
  my_cols = strong_scaling_flg ? config->cols/config->proc_cols : config->cols;

  step_first_flg = (strncmp(config->slowest_dimension, "step", 16) == 0);

  switch (config->rank)
    {
    case 2:
      start[0] = (hsize_t)proc_row*my_rows;
      start[1] = (hsize_t)proc_col*my_cols;
      count[0] = count[1] = 1;
      block[0] = (hsize_t)my_rows;
      block[1] = (hsize_t)my_cols;
      break;
    case 3:
      start[0] = (hsize_t)
        (step_first_flg ? array : step);
      start[1] = (hsize_t)proc_row*my_rows;
      start[2] = (hsize_t)proc_col*my_cols;
      count[0] = count[1] = count[2] = 1;
      block[0] = 1;
      block[1] = (hsize_t)my_rows;
      block[2] = (hsize_t)my_cols;
      break;
    case 4:
      start[0] = (hsize_t)
        (step_first_flg ? step : array);
      start[1] = (hsize_t)
        (step_first_flg ? array : step);
      start[2] = (hsize_t)proc_row*my_rows;
      start[3] = (hsize_t)proc_col*my_cols;
      count[0] = count[1] = count[2] = count[3] = 1;
      block[0] = block[1] = 1;
      block[2] = (hsize_t)my_rows;
      block[3] = (hsize_t)my_cols;
      break;
    default:
      break;
    }

  assert(H5Sselect_none(fspace) >= 0);
  assert(H5Sselect_hyperslab(fspace, H5S_SELECT_SET, start, NULL, count, block)
         >= 0);
  return result;
}
