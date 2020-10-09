#include "dataset.h"

#include "hdf5.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>


#define CONFIG_FILE "hdf5_iotest.ini"

int main(int argc, char* argv[])
{
  const char* ini = (argc > 1) ? argv[1] : CONFIG_FILE;

  configuration config;
  unsigned int strong_scaling_flg, coll_mpi_io_flg, step_first_flg;

  int size, rank, my_proc_row, my_proc_col;
  unsigned long my_rows, my_cols;
  unsigned int istep, iarray;
  double *wbuf, *rbuf;
  size_t i;

  hid_t mspace, dxpl, fapl, file, dset, fspace;
  char path[255];
  hsize_t fsize;

  double wall_time, create_time, write_phase, write_time, read_phase, read_time;
  double min_create_time, max_create_time;
  double min_write_phase, max_write_phase;
  double min_write_time, max_write_time;
  double min_read_phase, max_read_phase;
  double min_read_time, max_read_time;

  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  wall_time = -MPI_Wtime();

  read_time = write_time = create_time = 0.0;

  if (rank == 0) /* rank 0 reads and checks the config. file */
    {
      if (ini_parse(ini, handler, &config) < 0)
        {
          printf("Can't load '%s'\n", ini);
          return 1;
        }

      sanity_check(&config);
      validate(&config, size);
      strong_scaling_flg = (strncmp(config.scaling, "strong", 16) == 0);
      printf("Config loaded from '%s':\n\tsteps=%d, arrays=%d, rows=%ld, columns=%ld, scaling=%s\n",
             ini, config.steps, config.arrays, config.rows, config.cols,
             (strong_scaling_flg ? "strong" : "weak"));
      printf("\tproc-grid=%dx%d, slowest-dimension=%s, rank=%d\n",
             config.proc_rows, config.proc_cols, config.slowest_dimension,
             config.rank);
      printf("\tlayout=%s, mpi-io=%s\n", config.layout, config.mpi_io);
    }

  /* broadcast the input parameters */
  MPI_Bcast(&config, sizeof(configuration), MPI_BYTE, 0, MPI_COMM_WORLD);

  my_proc_row = rank / config.proc_cols;
  my_proc_col = rank % config.proc_cols;

  strong_scaling_flg = (strncmp(config.scaling, "strong", 16) == 0);
  my_rows = strong_scaling_flg ? config.rows/config.proc_rows : config.rows;
  my_cols = strong_scaling_flg ? config.cols/config.proc_cols : config.cols;

  /* allocate the write and read arrays */
  wbuf = (double*) malloc(my_rows*my_cols*sizeof(double));
  rbuf = (double*) calloc(my_rows*my_cols, sizeof(double));

  for (i = 0; i < (size_t)my_rows*my_cols; ++i)
    wbuf[i] = (double) (my_proc_row + my_proc_col);

  { /* create the in-memory dataspace */
    hsize_t dims[2];
    dims[0] = (hsize_t)my_rows;
    dims[1] = (hsize_t)my_cols;
    mspace = H5Screate_simple(2, dims, dims);
    assert(H5Sselect_all(mspace) >= 0);
  }

  assert((dxpl = H5Pcreate(H5P_DATASET_XFER)) >= 0);
  coll_mpi_io_flg = (strncmp(config.mpi_io, "collective", 16) == 0);
  if (coll_mpi_io_flg)
    assert(H5Pset_dxpl_mpio(dxpl, H5FD_MPIO_COLLECTIVE) >= 0);
  else
    assert(H5Pset_dxpl_mpio(dxpl, H5FD_MPIO_INDEPENDENT) >= 0);

  assert((fapl = H5Pcreate(H5P_FILE_ACCESS)) >= 0);
  assert(H5Pset_fapl_mpio(fapl, MPI_COMM_WORLD, MPI_INFO_NULL) >= 0);

  step_first_flg = (strncmp(config.slowest_dimension, "step", 16) == 0);

  MPI_Barrier(MPI_COMM_WORLD);

  write_phase = -MPI_Wtime();

  { /* WRITE phase */
    create_time -= MPI_Wtime();
    assert((file = H5Fcreate(config.hdf5_file, H5F_ACC_TRUNC, H5P_DEFAULT,
                             fapl)) >= 0);
    create_time += MPI_Wtime();

    switch (config.rank)
      {
      case 4:
        {
          /* one 4D array */
          create_time -= MPI_Wtime();
          assert((dset = create_dataset(&config, file, "dataset")) >= 0);
          create_time += MPI_Wtime();

          for (istep = 0; istep < config.steps; ++istep)
            {
              for (iarray = 0; iarray < config.arrays; ++iarray)
                {
                  assert((fspace = H5Dget_space(dset)) >= 0);
                  create_selection(&config, fspace, my_proc_row, my_proc_col,
                                   istep, iarray);

                  write_time -= MPI_Wtime();
                  assert(H5Dwrite(dset, H5T_NATIVE_DOUBLE, mspace, fspace,
                                  dxpl, wbuf) >= 0);
                  write_time += MPI_Wtime();

                  assert(H5Sclose(fspace) >= 0);
                }
            }

          assert(H5Dclose(dset) >= 0);
        }
        break;
      case 3:
        {
          if (step_first_flg) /* dataset per step */
            {
              for (istep = 0; istep < config.steps; ++istep)
                {
                  create_time -= MPI_Wtime();
                  sprintf(path, "step=%d", istep);
                  assert((dset = create_dataset(&config, file, path)) >= 0);
                  create_time += MPI_Wtime();

                  for (iarray = 0; iarray < config.arrays; ++iarray)
                    {
                      assert((fspace = H5Dget_space(dset)) >= 0);
                      create_selection(&config, fspace, my_proc_row,
                                       my_proc_col, istep, iarray);

                      write_time -= MPI_Wtime();
                      assert(H5Dwrite(dset, H5T_NATIVE_DOUBLE, mspace, fspace,
                                      dxpl, wbuf) >= 0);
                      write_time += MPI_Wtime();

                      assert(H5Sclose(fspace) >= 0);
                    }

                  assert(H5Dclose(dset) >= 0);
                }
            }
          else /* dataset per array */
            {
              for (istep = 0; istep < config.steps; ++istep)
                {
                  for (iarray = 0; iarray < config.arrays; ++iarray)
                    {
                      sprintf(path, "array=%d", iarray);
                      if (istep == 0)
                        {
                          create_time -= MPI_Wtime();
                          assert((dset = create_dataset(&config, file, path))
                                 >= 0);
                          create_time += MPI_Wtime();
                        }
                      else
                        {
                          create_time -= MPI_Wtime();
                          assert((dset = H5Dopen(file, path, H5P_DEFAULT)) >=
                                 0);
                          create_time += MPI_Wtime();
                        }

                      assert((fspace = H5Dget_space(dset)) >= 0);
                      create_selection(&config, fspace, my_proc_row,
                                       my_proc_col, istep, iarray);

                      write_time -= MPI_Wtime();
                      assert(H5Dwrite(dset, H5T_NATIVE_DOUBLE, mspace, fspace,
                                      dxpl, wbuf) >= 0);
                      write_time += MPI_Wtime();
                      assert(H5Sclose(fspace) >= 0);

                      assert(H5Dclose(dset) >= 0);
                    }
                }
            }
        }
        break;
      case 2:
        {
          hid_t lcpl = H5Pcreate(H5P_LINK_CREATE);
          assert(lcpl >= 0);
          assert(H5Pset_create_intermediate_group(lcpl, 1) >= 0);

          for (istep = 0; istep < config.steps; ++istep)
            {
              for (iarray = 0; iarray < config.arrays; ++iarray)
                {
                  create_time -= MPI_Wtime();
                  /* group per step or array of 2D datasets */
                  sprintf(path, (step_first_flg ?
                                 "step=%d/array=%d" : "array=%d/step=%d"),
                          (step_first_flg ? istep : iarray),
                          (step_first_flg ? iarray : istep));
                  assert((dset = create_dataset(&config, file, path)) >= 0);
                  create_time += MPI_Wtime();

                  assert((fspace = H5Dget_space(dset)) >= 0);
                  create_selection(&config, fspace, my_proc_row, my_proc_col,
                                   istep, iarray);

                  write_time -= MPI_Wtime();
                  assert(H5Dwrite(dset, H5T_NATIVE_DOUBLE, mspace, fspace,
                                  dxpl, wbuf) >= 0);
                  write_time += MPI_Wtime();
                  assert(H5Sclose(fspace) >= 0);
                  H5Dclose(dset);
                }
            }

          H5Pclose(lcpl);
        }
        break;
      default:
        break;
      }

    assert(H5Fclose(file) >= 0);
  }

  write_phase += MPI_Wtime();

  MPI_Barrier(MPI_COMM_WORLD);

  read_phase = -MPI_Wtime();

  { /* READ phase */
    assert((file = H5Fopen(config.hdf5_file, H5F_ACC_RDONLY, fapl)) >= 0);

    switch (config.rank)
      {
      case 4:
        {
          assert((dset = H5Dopen(file, "dataset", H5P_DEFAULT)) >= 0);

          for (istep = 0; istep < config.steps; ++istep)
            {
              for (iarray = 0; iarray < config.arrays; ++iarray)
                {
                  assert((fspace = H5Dget_space(dset)) >= 0);
                  create_selection(&config, fspace, my_proc_row, my_proc_col,
                                   istep, iarray);
                  read_time -= MPI_Wtime();
                  assert(H5Dread(dset, H5T_NATIVE_DOUBLE, mspace, fspace, dxpl,
                                 rbuf) >= 0);
                  read_time += MPI_Wtime();
                  assert(H5Sclose(fspace) >= 0);
                }
            }

          assert(H5Dclose(dset) >= 0);
        }
        break;
      case 3:
        {
          if (step_first_flg) /* dataset per step */
            {
              for (istep = 0; istep < config.steps; ++istep)
                {
                  sprintf(path, "step=%d", istep);
                  assert((dset = H5Dopen(file, path, H5P_DEFAULT)) >= 0);
                  assert((fspace = H5Dget_space(dset)) >= 0);

                  for (iarray = 0; iarray < config.arrays; ++iarray)
                    {
                      create_selection(&config, fspace, my_proc_row,
                                       my_proc_col, istep, iarray);

                      read_time -= MPI_Wtime();
                      assert(H5Dread(dset, H5T_NATIVE_DOUBLE, mspace, fspace,
                                     dxpl, rbuf) >= 0);
                      read_time += MPI_Wtime();

                    }

                  assert(H5Sclose(fspace) >= 0);
                  assert(H5Dclose(dset) >= 0);
                }
            }
          else /* dataset per array */
            {
              for (istep = 0; istep < config.steps; ++istep)
                {
                  for (iarray = 0; iarray < config.arrays; ++iarray)
                    {
                      sprintf(path, "array=%d", iarray);
                      assert((dset = H5Dopen(file, path, H5P_DEFAULT)) >= 0);
                      assert((fspace = H5Dget_space(dset)) >= 0);
                      create_selection(&config, fspace, my_proc_row,
                                       my_proc_col, istep, iarray);

                      read_time -= MPI_Wtime();
                      assert(H5Dread(dset, H5T_NATIVE_DOUBLE, mspace, fspace,
                                     dxpl, rbuf) >= 0);
                      read_time += MPI_Wtime();

                      assert(H5Sclose(fspace) >= 0);
                      assert(H5Dclose(dset) >= 0);
                    }
                }
            }
        }
        break;
      case 2:
        {
          for (istep = 0; istep < config.steps; ++istep)
            {
              for (iarray = 0; iarray < config.arrays; ++iarray)
                {
                  /* group per step or array */
                  sprintf(path, (step_first_flg ?
                                 "step=%d/array=%d" : "array=%d/step=%d"),
                          (step_first_flg ? istep : iarray),
                          (step_first_flg ? iarray : istep));

                  assert((dset = H5Dopen(file, path, H5P_DEFAULT)) >= 0);

                  assert((fspace = H5Dget_space(dset)) >= 0);
                  create_selection(&config, fspace, my_proc_row, my_proc_col,
                                   istep, iarray);

                  read_time -= MPI_Wtime();
                  assert(H5Dread(dset, H5T_NATIVE_DOUBLE, mspace, fspace, dxpl,
                                 rbuf) >= 0);
                  read_time += MPI_Wtime();

                  assert(H5Sclose(fspace) >= 0);
                  assert(H5Dclose(dset) >= 0);
                }
            }
        }
        break;
      default:
        break;
      }

    assert(H5Fclose(file) >= 0);
  }

  read_phase += MPI_Wtime();

  assert(H5Pclose(fapl) >= 0);
  assert(H5Pclose(dxpl) >= 0);
  assert(H5Sclose(mspace) >= 0);

  MPI_Barrier(MPI_COMM_WORLD);

  wall_time += MPI_Wtime();

  if (rank == 0)
  {
    assert((file =
            H5Fopen(config.hdf5_file, H5F_ACC_RDONLY, H5P_DEFAULT)) >= 0);
    assert(H5Fget_filesize(file, &fsize) >= 0);
    assert(H5Fclose(file) >= 0);
  }

  // TODO: compare the write and read buffers

  free(wbuf);
  free(rbuf);

  max_write_phase = min_write_phase = 0.0;
  max_create_time = min_create_time = 0.0;
  max_write_time = min_write_time = 0.0;
  max_read_phase = min_read_phase = 0.0;
  max_read_time = min_read_time = 0.0;

  MPI_Reduce(&write_phase, &min_write_phase, 1, MPI_DOUBLE, MPI_MIN, 0,
             MPI_COMM_WORLD);
  MPI_Reduce(&write_phase, &max_write_phase, 1, MPI_DOUBLE, MPI_MAX, 0,
             MPI_COMM_WORLD);
  MPI_Reduce(&create_time, &min_create_time, 1, MPI_DOUBLE, MPI_MIN, 0,
             MPI_COMM_WORLD);
  MPI_Reduce(&create_time, &max_create_time, 1, MPI_DOUBLE, MPI_MAX, 0,
             MPI_COMM_WORLD);
  MPI_Reduce(&write_time, &min_write_time, 1, MPI_DOUBLE, MPI_MIN, 0,
             MPI_COMM_WORLD);
  MPI_Reduce(&write_time, &max_write_time, 1, MPI_DOUBLE, MPI_MAX, 0,
             MPI_COMM_WORLD);
  MPI_Reduce(&read_phase, &min_read_phase, 1, MPI_DOUBLE, MPI_MIN, 0,
             MPI_COMM_WORLD);
  MPI_Reduce(&read_phase, &max_read_phase, 1, MPI_DOUBLE, MPI_MAX, 0,
             MPI_COMM_WORLD);
  MPI_Reduce(&read_time, &min_read_time, 1, MPI_DOUBLE, MPI_MIN, 0,
             MPI_COMM_WORLD);
  MPI_Reduce(&read_time, &max_read_time, 1, MPI_DOUBLE, MPI_MAX, 0,
             MPI_COMM_WORLD);

  if (rank == 0)
    {
      double  byte_count =
        (double)config.steps*config.arrays*my_rows*my_cols*sizeof(double);
      double min_write_rate = byte_count / (1024*1024*max_write_time);
      double max_write_rate = byte_count / (1024*1024*min_write_time);
      double min_read_rate = byte_count / (1024*1024*max_read_time);
      double max_read_rate = byte_count / (1024*1024*min_read_time);
      printf("\nWall clock [s]:\t\t%.2f\n", wall_time);
      printf("File size [B]:\t\t%.0f\n", (double)fsize);
      printf("---------------------------------------------\n");
      printf("Measurement:\t\t_MIN (over MPI ranks)\n");
      printf("\t\t\t^MAX (over MPI ranks)\n");
      printf("---------------------------------------------\n");
      printf("Write phase [s]:\t_%.2f\n\t\t\t^%.2f\n", min_write_phase,
             max_write_phase);
      printf("Create time [s]:\t_%.2f\n\t\t\t^%.2f\n", min_create_time,
             max_create_time);
      printf("Write time [s]:\t\t_%.2f\n\t\t\t^%.2f\n", min_write_time,
             max_write_time);
      printf("Write rate [MiB/s]:\t_%.2f\n\t\t\t^%.2f\n",
             min_write_rate, max_write_rate);
      printf("Read phase [s]:\t\t_%.2f\n\t\t\t^%.2f\n", min_read_phase,
             max_read_phase);
      printf("Read time [s]:\t\t_%.2f\n\t\t\t^%.2f\n", min_read_time,
             max_read_time);
      printf("Read rate [MiB/s]:\t_%.2f\n\t\t\t^%.2f\n",
             min_read_rate, max_read_rate);

      /* write results CSV file */
      FILE *fptr = fopen(config.csv_file, "w");
      assert(fptr != NULL);
      fprintf(fptr, "steps,arrays,rows,cols,scaling,proc-rows,proc-cols,slowdim,rank,layout,mpi-io,wall [s],fsize [B],write-phase-min [s],write-phase-max [s],creat-min [s],creat-max [s],write-min [s],write-max [s],write-rate-min [MiB/s],write-rate-max [MiB/s],read-phase-min [s],read-phase-max [s],read-min [s],read-max [s],read-rate-min [MiB/s],read-rate-max [MiB/s]\n");
      fprintf(fptr,
      "%d,%d,%ld,%ld,%s,%d,%d,%s,%d,%s,%s,%.2f,%.0f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f\n",
              config.steps, config.arrays, config.rows, config.cols,
              config.scaling, config.proc_rows, config.proc_cols,
              config.slowest_dimension, config.rank, config.layout,
              config.mpi_io, wall_time, (double)fsize, min_write_phase,
              max_write_phase, min_create_time, max_create_time,
              min_write_time, max_write_time, min_write_rate, max_write_rate,
              min_read_phase, max_read_phase, min_read_time, max_read_time,
              min_read_rate, max_read_rate);
      fclose(fptr);
    }

  MPI_Finalize();

  return 0;
}
