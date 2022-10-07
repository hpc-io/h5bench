/**
 *
 * Email questions to runzhouhan@lbl.gov
 * Scientific Data Management Research Group
 * Lawrence Berkeley National Laboratory
 *
 */

// Description: Overwrite every dataset in a given file
// Author:  Runzhou Han <runzhouhan@lbl.gov>
//      Lawrence Berkeley National Laboratory, Berkeley, CA
// Created: in 2021

#include <math.h>
#include <hdf5.h>
#include <stdlib.h>
#include <sys/time.h>
#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <assert.h>
#include "../commons/h5bench_util.h"
#include "../commons/async_adaptor.h"

// Global Variables and dimensions
long long  NUM_PARTICLES = 0, FILE_OFFSET = 0;
long long  TOTAL_PARTICLES = 0;
async_mode ASYNC_MODE;
int        NUM_RANKS, MY_RANK, NUM_TIMESTEPS;
hid_t      PARTICLE_COMPOUND_TYPE;
hid_t      PARTICLE_COMPOUND_TYPE_SEPARATES[8];

herr_t          ierr;
data_contig_md *BUF_STRUCT;
mem_monitor *   MEM_MONITOR;

void
print_data(int n)
{
    int i;
    for (i = 0; i < n; i++)
        printf("sample data: %f %f %f %d %f %f %f %f\n", BUF_STRUCT->x[i], BUF_STRUCT->y[i], BUF_STRUCT->z[i],
               BUF_STRUCT->id_1[i], BUF_STRUCT->id_2[i], BUF_STRUCT->px[i], BUF_STRUCT->py[i],
               BUF_STRUCT->pz[i]);
}

void
set_dspace_plist(hid_t *plist_id_out, int data_collective)
{
    *plist_id_out = H5Pcreate(H5P_DATASET_XFER);
    if (data_collective == 1)
        H5Pset_dxpl_mpio(*plist_id_out, H5FD_MPIO_COLLECTIVE);
    else
        H5Pset_dxpl_mpio(*plist_id_out, H5FD_MPIO_INDEPENDENT);
}

// Overwrite dataset
void
overwrite_h5_data(bench_params params, time_step *ts, hid_t loc, hid_t *dset_ids, hid_t filespace,
                  hid_t memspace, unsigned long *read_time, unsigned long *metadata_time)
{
    hid_t         dapl;
    unsigned long t1, t2, t3;

    hsize_t dims[3];
    hsize_t dims_memory[3];

    dapl = H5Pcreate(H5P_DATASET_ACCESS);

    int *  data_1D_INT, **data_2D_INT, ***data_3D_INT;
    float *data_1D_FLOAT, **data_2D_FLOAT, ***data_3D_FLOAT;

    if (params.num_dims == 1) {
        data_1D_INT   = malloc(params.dim_1 * sizeof(int));
        data_1D_FLOAT = malloc(params.dim_1 * sizeof(float));
    }

    if (params.num_dims == 2) {
        data_2D_INT   = malloc(params.dim_1 * sizeof(int *));
        data_2D_FLOAT = malloc(params.dim_1 * sizeof(float *));
        for (int i = 0; i < params.dim_1; i++) {
            data_2D_INT[i]   = malloc(params.dim_2 * sizeof(int));
            data_2D_FLOAT[i] = malloc(params.dim_2 * sizeof(float));
        }
    }

    if (params.num_dims == 3) {
        data_3D_INT   = malloc(params.dim_1 * sizeof(int **));
        data_3D_FLOAT = malloc(params.dim_1 * sizeof(float **));
        for (int i = 0; i < params.dim_1; i++) {
            data_3D_INT[i]   = malloc(params.dim_2 * sizeof(int *));
            data_3D_FLOAT[i] = malloc(params.dim_2 * sizeof(float *));
            for (int j = 0; j < params.dim_2; j++) {
                data_3D_INT[i][j]   = malloc(params.dim_3 * sizeof(int));
                data_3D_FLOAT[i][j] = malloc(params.dim_3 * sizeof(float));
            }
        }
    }

    switch (params.access_pattern.pattern_read) {
        case CONTIG_1D:
        case STRIDED_1D:
            for (long i = 0; i < params.dim_1; i++) {
                data_1D_INT[i]   = i;
                data_1D_FLOAT[i] = (float)i * 0.1;
            }
            break;

        case CONTIG_2D:
            for (long i = 0; i < params.dim_1; i++) {
                for (long j = 0; j < params.dim_2; j++) {
                    data_2D_INT[i][j]   = i + j;
                    data_2D_FLOAT[i][j] = i / (float)(j + 1) * 0.1;
                }
            }
            break;

        case CONTIG_3D:
            for (long i = 0; i < params.dim_1; i++) {
                for (long j = 0; j < params.dim_2; j++) {
                    for (long k = 0; k < params.dim_3; k++) {
                        data_3D_INT[i][j][k]   = i + j + k;
                        data_3D_FLOAT[i][j][k] = i / (float)(j + 1) / (float)(k + 1) * 0.1;
                    }
                }
            }
            break;
        default:
            printf("Unknown read pattern\n");
            break;
    }

    dims[0]        = params.dim_1;
    dims_memory[0] = params.dim_1;

    dims[1]        = params.dim_2;
    dims_memory[1] = params.dim_2;

    dims[2]        = params.dim_3;
    dims_memory[2] = params.dim_3;

    t1 = get_time_usec();

    dset_ids[0] = H5Dopen_async(loc, "x", dapl, ts->es_meta_create);
    dset_ids[1] = H5Dopen_async(loc, "y", dapl, ts->es_meta_create);
    dset_ids[2] = H5Dopen_async(loc, "z", dapl, ts->es_meta_create);
    dset_ids[3] = H5Dopen_async(loc, "id_1", dapl, ts->es_meta_create);
    dset_ids[4] = H5Dopen_async(loc, "id_2", dapl, ts->es_meta_create);
    dset_ids[5] = H5Dopen_async(loc, "px", dapl, ts->es_meta_create);
    dset_ids[6] = H5Dopen_async(loc, "py", dapl, ts->es_meta_create);
    dset_ids[7] = H5Dopen_async(loc, "pz", dapl, ts->es_meta_create);

    t2 = get_time_usec();

    set_dspace_plist(&dapl, params.data_coll);

    switch (params.access_pattern.pattern_read) {
        case CONTIG_1D:
        case STRIDED_1D:
            H5Dwrite(dset_ids[0], H5T_NATIVE_FLOAT, memspace, filespace, dapl, data_1D_FLOAT);
            H5Dwrite(dset_ids[1], H5T_NATIVE_FLOAT, memspace, filespace, dapl, data_1D_FLOAT);
            H5Dwrite(dset_ids[2], H5T_NATIVE_FLOAT, memspace, filespace, dapl, data_1D_FLOAT);
            H5Dwrite(dset_ids[3], H5T_NATIVE_INT, memspace, filespace, dapl, data_1D_INT);
            H5Dwrite(dset_ids[4], H5T_NATIVE_FLOAT, memspace, filespace, dapl, data_1D_FLOAT);
            H5Dwrite(dset_ids[5], H5T_NATIVE_FLOAT, memspace, filespace, dapl, data_1D_FLOAT);
            H5Dwrite(dset_ids[6], H5T_NATIVE_FLOAT, memspace, filespace, dapl, data_1D_FLOAT);
            H5Dwrite(dset_ids[7], H5T_NATIVE_FLOAT, memspace, filespace, dapl, data_1D_FLOAT);
            break;

        case CONTIG_2D:
            H5Dwrite(dset_ids[0], H5T_NATIVE_FLOAT, memspace, filespace, dapl, data_2D_FLOAT);
            H5Dwrite(dset_ids[1], H5T_NATIVE_FLOAT, memspace, filespace, dapl, data_2D_FLOAT);
            H5Dwrite(dset_ids[2], H5T_NATIVE_FLOAT, memspace, filespace, dapl, data_2D_FLOAT);
            H5Dwrite(dset_ids[3], H5T_NATIVE_INT, memspace, filespace, dapl, data_2D_INT);
            H5Dwrite(dset_ids[4], H5T_NATIVE_FLOAT, memspace, filespace, dapl, data_2D_FLOAT);
            H5Dwrite(dset_ids[5], H5T_NATIVE_FLOAT, memspace, filespace, dapl, data_2D_FLOAT);
            H5Dwrite(dset_ids[6], H5T_NATIVE_FLOAT, memspace, filespace, dapl, data_2D_FLOAT);
            H5Dwrite(dset_ids[7], H5T_NATIVE_FLOAT, memspace, filespace, dapl, data_2D_FLOAT);
            break;

        case CONTIG_3D:
            H5Dwrite(dset_ids[0], H5T_NATIVE_FLOAT, memspace, filespace, dapl, data_3D_FLOAT);
            H5Dwrite(dset_ids[1], H5T_NATIVE_FLOAT, memspace, filespace, dapl, data_3D_FLOAT);
            H5Dwrite(dset_ids[2], H5T_NATIVE_FLOAT, memspace, filespace, dapl, data_3D_FLOAT);
            H5Dwrite(dset_ids[3], H5T_NATIVE_INT, memspace, filespace, dapl, data_3D_INT);
            H5Dwrite(dset_ids[4], H5T_NATIVE_FLOAT, memspace, filespace, dapl, data_3D_FLOAT);
            H5Dwrite(dset_ids[5], H5T_NATIVE_FLOAT, memspace, filespace, dapl, data_3D_FLOAT);
            H5Dwrite(dset_ids[6], H5T_NATIVE_FLOAT, memspace, filespace, dapl, data_3D_FLOAT);
            H5Dwrite(dset_ids[7], H5T_NATIVE_FLOAT, memspace, filespace, dapl, data_3D_FLOAT);
            break;
        default:
            printf("Unknown read pattern\n");
            break;
    }

    t3 = get_time_usec();

    *read_time     = t3 - t2;
    *metadata_time = t2 - t1;

    if (MY_RANK == 0)
        printf("  Overwrite 8 variable completed\n");
    H5Pclose(dapl);

    if (params.num_dims == 1) {
        free(data_1D_INT);
        free(data_1D_FLOAT);
    }

    if (params.num_dims == 2) {
        for (int i = 0; i < params.dim_1; i++) {
            free(data_2D_INT[i]);
            free(data_2D_FLOAT[i]);
        }
        free(data_2D_INT);
        free(data_2D_FLOAT);
    }

    if (params.num_dims == 3) {
        for (int i = 0; i < params.dim_1; i++) {
            for (int j = 0; j < params.dim_2; j++) {
                free(data_3D_INT[i][j]);
                free(data_3D_FLOAT[i][j]);
            }
            free(data_3D_INT[i]);
            free(data_3D_FLOAT[i]);
        }
        free(data_3D_INT);
        free(data_3D_FLOAT);
    }
}

int
_set_dataspace_seq_read(unsigned long read_elem_cnt, hid_t *filespace_in, hid_t *memspace_out)
{
    hsize_t count[1] = {1};
    /* Overwrite dataset, set dim to NULL. */
    *memspace_out = H5Screate_simple(1, (hsize_t *)&read_elem_cnt, NULL);

    H5Sselect_hyperslab(*filespace_in, H5S_SELECT_SET, (hsize_t *)&FILE_OFFSET, NULL, count,
                        (hsize_t *)&read_elem_cnt);
    return read_elem_cnt;
}

// returns actual rounded read element count.
unsigned long
_set_dataspace_strided_read(unsigned long read_elem_cnt, bench_params params, hid_t *filespace_in,
                            hid_t *memspace_out)
{
    if (MY_RANK == 0) {
        printf("Stride parameters: STRIDE_SIZE = %lu, BLOCK_SIZE = %lu, BLOCK_CNT = %lu\n", params.stride,
               params.block_size, params.block_cnt);
    }

    if ((params.stride + params.block_size) * params.block_cnt > params.dim_1) {
        printf("\n\nInvalid hyperslab setting: (STRIDE_SIZE + BLOCK_SIZE) * BLOCK_CNT"
               "must be no greater than the number of available particles per rank(%lu).\n\n",
               params.chunk_dim_1);
        return 0;
    }

    unsigned long actual_elem_cnt = params.block_size * params.block_cnt;
    *memspace_out                 = H5Screate_simple(1, (hsize_t *)&actual_elem_cnt, NULL);

    H5Sselect_hyperslab(*filespace_in, H5S_SELECT_SET,
                        (hsize_t *)&FILE_OFFSET,        // start-offset
                        (hsize_t *)&params.stride,      // stride
                        (hsize_t *)&params.block_cnt,   // block cnt
                        (hsize_t *)&params.block_size); // block size

    return actual_elem_cnt;
}

// filespace should be read from the file first, then select the hyperslab.
unsigned long
_set_dataspace_seq_2D(hid_t *filespace_in_out, hid_t *memspace_out, unsigned long long dim_1,
                      unsigned long long dim_2)
{
    hsize_t mem_dims[2], file_dims[2];
    mem_dims[0] = (hsize_t)dim_1;
    mem_dims[1] = (hsize_t)dim_2;

    file_dims[0] = (hsize_t)dim_1 * NUM_RANKS; // total x length: dim_1 * world_size.
    file_dims[1] = (hsize_t)dim_2;             // always the same dim_2

    hsize_t count[2] = {1, 1};
    hsize_t file_starts[2], block[2];   // select start point and range in each dimension.
    file_starts[0] = dim_1 * (MY_RANK); // file offset for each rank
    file_starts[1] = 0;

    block[0]      = dim_1;
    block[1]      = dim_2;
    *memspace_out = H5Screate_simple(2, mem_dims, NULL);
    H5Sselect_hyperslab(*filespace_in_out, H5S_SELECT_SET, file_starts, NULL, count, block);
    return dim_1 * dim_2;
}

unsigned long
_set_dataspace_seq_3D(hid_t *filespace_in_out, hid_t *memspace_out, unsigned long long dim_1,
                      unsigned long long dim_2, unsigned long long dim_3)
{
    hsize_t mem_dims[3];
    hsize_t file_dims[3];
    mem_dims[0]  = (hsize_t)dim_1;
    mem_dims[1]  = (hsize_t)dim_2;
    mem_dims[2]  = (hsize_t)dim_3;
    file_dims[0] = (hsize_t)dim_1 * NUM_RANKS;
    file_dims[1] = (hsize_t)dim_2;
    file_dims[2] = (hsize_t)dim_3;

    hsize_t count[3] = {1, 1, 1};
    hsize_t file_starts[3], file_range[3]; // select start point and range in each dimension.
    file_starts[0] = dim_1 * (MY_RANK);
    file_starts[1] = 0;
    file_starts[2] = 0;
    file_range[0]  = dim_1;
    file_range[1]  = dim_2;
    file_range[2]  = dim_3;

    *memspace_out = H5Screate_simple(3, mem_dims, NULL);

    H5Sselect_hyperslab(*filespace_in_out, H5S_SELECT_SET, file_starts, NULL, count, file_range);
    return dim_1 * dim_2 * dim_3;
}

hid_t
get_filespace(hid_t file_id)
{
    char *grp_name  = "/Timestep_0";
    char *ds_name   = "px";
    hid_t gid       = H5Gopen2(file_id, grp_name, H5P_DEFAULT);
    hid_t dsid      = H5Dopen2(gid, ds_name, H5P_DEFAULT);
    hid_t filespace = H5Dget_space(dsid);
    H5Dclose(dsid);
    H5Gclose(gid);
    return filespace;
}

unsigned long
set_dataspace(bench_params params, unsigned long long try_read_elem_cnt, hid_t *filespace_in_out,
              hid_t *memspace_out)
{
    unsigned long actual_read_cnt = 0;
    switch (params.access_pattern.pattern_read) {
        case CONTIG_1D:

            _set_dataspace_seq_read(try_read_elem_cnt, filespace_in_out, memspace_out);
            actual_read_cnt = try_read_elem_cnt;
            break;

        case STRIDED_1D:
            actual_read_cnt =
                _set_dataspace_strided_read(try_read_elem_cnt, params, filespace_in_out, memspace_out);
            break;

        case CONTIG_2D:
            actual_read_cnt =
                _set_dataspace_seq_2D(filespace_in_out, memspace_out, params.dim_1, params.dim_2);
            break;

        case CONTIG_3D:
            actual_read_cnt = _set_dataspace_seq_3D(filespace_in_out, memspace_out, params.dim_1,
                                                    params.dim_2, params.dim_3);
            break;

        default:
            printf("Unknown read pattern\n");
            break;
    }
    return actual_read_cnt;
}

int
_run_benchmark_modify(hid_t file_id, hid_t fapl, hid_t gapl, hid_t filespace, bench_params params,
                      unsigned long *total_data_size_out, unsigned long *raw_read_time_out,
                      unsigned long *inner_metadata_time)
{
    *raw_read_time_out               = 0;
    *inner_metadata_time             = 0;
    int                nts           = params.cnt_time_step;
    unsigned long long read_elem_cnt = params.try_num_particles;
    hid_t              grp;
    char               grp_name[128];
    unsigned long      rt1 = 0, rt2 = 0;
    unsigned long      actual_read_cnt = 0;
    hid_t              memspace;
    actual_read_cnt = set_dataspace(params, read_elem_cnt, &filespace, &memspace);

    if (actual_read_cnt < 1)
        return -1;

    if (MY_RANK == 0)
        print_params(&params);

    MEM_MONITOR      = mem_monitor_new(nts, ASYNC_MODE, actual_read_cnt, params.io_mem_limit);
    unsigned long t1 = 0, t2 = 0, t3 = 0, t4 = 0;
    unsigned long meta_time1 = 0, meta_time2 = 0, meta_time3 = 0, meta_time4 = 0, meta_time5;
    unsigned long read_time_exp = 0, metadata_time_exp = 0;
    unsigned long read_time_imp = 0, metadata_time_imp = 0;
    int           dset_cnt = 8;
    for (int ts_index = 0; ts_index < nts; ts_index++) {
        meta_time1 = 0, meta_time2 = 0, meta_time3 = 0, meta_time4 = 0, meta_time5 = 0;
        sprintf(grp_name, "Timestep_%d", ts_index);
        time_step *ts = &(MEM_MONITOR->time_steps[ts_index]);
        MEM_MONITOR->mem_used += ts->mem_size;
        assert(ts);

        if (params.cnt_time_step_delay > 0) {
            if (ts_index > params.cnt_time_step_delay - 1) // delayed close on all ids of the previous ts
                ts_delayed_close(MEM_MONITOR, &meta_time1, dset_cnt);
        }
        mem_monitor_check_run(MEM_MONITOR, &meta_time2, &read_time_imp);

        t1         = get_time_usec();
        ts->grp_id = H5Gopen_async(file_id, grp_name, gapl, ts->es_meta_create);
        t2         = get_time_usec();
        meta_time3 = (t2 - t1);

        if (MY_RANK == 0)
            printf("Reading %s ... \n", grp_name);

        overwrite_h5_data(params, ts, ts->grp_id, ts->dset_ids, filespace, memspace, &read_time_exp,
                          &meta_time4);

        ts->status = TS_DELAY;

        if (params.cnt_time_step_delay == 0) {
            t3 = get_time_usec();
            for (int j = 0; j < dset_cnt; j++) {
                if (ts->dset_ids[j] != 0) {
                    H5Dclose_async(ts->dset_ids[j], ts->es_meta_close);
                }
            }
            H5Gclose_async(ts->grp_id, ts->es_meta_close);
            ts->status = TS_READY;
            t4         = get_time_usec();
            meta_time5 = (t4 - t3);
        }

        if (ts_index != nts - 1) { // no sleep after the last ts
            if (params.compute_time.time_num >= 0) {
                if (MY_RANK == 0)
                    printf("Computing... \n");
                async_sleep(ts->es_meta_close, params.compute_time);
            }
        }

        *raw_read_time_out += (read_time_exp + read_time_imp);
        *inner_metadata_time += (meta_time1 + meta_time2 + meta_time3 + meta_time4 + meta_time5);
    }

    mem_monitor_final_run(MEM_MONITOR, &metadata_time_imp, &read_time_imp);
    *raw_read_time_out += read_time_imp;
    *inner_metadata_time += metadata_time_imp;
    *total_data_size_out = nts * actual_read_cnt * (6 * sizeof(float) + 2 * sizeof(int));
    H5Sclose(memspace);
    H5Sclose(filespace);
    return 0;
}

void
set_pl(hid_t *fapl, hid_t *gapl)
{
    *fapl = H5Pcreate(H5P_FILE_ACCESS);
    H5Pset_fapl_mpio(*fapl, MPI_COMM_WORLD, MPI_INFO_NULL);
    H5Pset_all_coll_metadata_ops(*fapl, true);
    H5Pset_coll_metadata_write(*fapl, true);
    *gapl = H5Pcreate(H5P_GROUP_ACCESS);
    H5Pset_all_coll_metadata_ops(*gapl, true);
}

void
print_usage(char *name)
{
    printf("Usage: %s /path/to/file #timestep sleep_sec [# mega particles]\n", name);
}

int
main(int argc, char *argv[])
{
    int mpi_thread_lvl_provided = -1;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &mpi_thread_lvl_provided);
    assert(MPI_THREAD_MULTIPLE == mpi_thread_lvl_provided);
    MPI_Comm_rank(MPI_COMM_WORLD, &MY_RANK);
    MPI_Comm_size(MPI_COMM_WORLD, &NUM_RANKS);

    int sleep_time = 0;

    bench_params params;

    char *cfg_file_path = argv[1];
    char *file_name     = argv[2]; // data file to read

    if (MY_RANK == 0) {
        printf("Configuration file: %s\n", argv[1]);
        printf("Read data file: %s\n", argv[2]);
    }
    int do_write = 0;
    if (read_config(cfg_file_path, &params, do_write) < 0) {
        if (MY_RANK == 0)
            printf("Configuration file read failed. Please, check %s\n", cfg_file_path);
        return 0;
    }
    ASYNC_MODE    = params.asyncMode;
    NUM_TIMESTEPS = params.cnt_time_step;

    if (NUM_TIMESTEPS <= 0) {
        if (MY_RANK == 0)
            printf("Usage: ./%s /path/to/file #timestep [# mega particles]\n", argv[0]);
        return 0;
    }

    if (params.io_op != IO_OVERWRITE) {
        if (MY_RANK == 0)
            printf("Make sure the configuration file has IO_OPERATION=OVERWRITE defined\n");
        return 0;
    }

    hid_t fapl, gapl;
    set_pl(&fapl, &gapl);

    hsize_t dims[64] = {0};

    hid_t         file_id         = H5Fopen(file_name, H5F_ACC_RDWR, fapl);
    hid_t         filespace       = get_filespace(file_id);
    int           dims_cnt        = H5Sget_simple_extent_dims(filespace, dims, NULL);
    unsigned long total_particles = 1;
    if (dims_cnt > 0) {
        for (int i = 0; i < dims_cnt; i++) {
            if (MY_RANK == 0)
                printf("dims[%d] = %llu (total number for the file)\n", i, dims[i]);
            total_particles *= dims[i];
        }
    }
    else {
        if (MY_RANK == 0)
            printf("Failed to read dimensions. \n");
        return 0;
    }

    if (params.num_dims != dims_cnt) {
        printf("Number of dimensions to be overwriten (%d) is inconsist with original dimension: %d\n",
               params.num_dims, dims_cnt);
        goto error;
    }

    if (dims_cnt > 0) { // 1D
        if (params.dim_1 > dims[0] / NUM_RANKS) {
            if (MY_RANK == 0)
                printf("Failed: Required dimension(%lu) is greater than the allowed dimension per rank "
                       "(%llu).\n",
                       params.dim_1, dims[0] / NUM_RANKS);
            goto error;
        }
    }
    if (dims_cnt > 1) { // 2D
        if (params.dim_2 > dims[1]) {
            if (MY_RANK == 0)
                printf("Failed: Required dimension_2 (%lu) is greater than file dimension (%llu).\n",
                       params.dim_2, dims[1]);
            goto error;
        }
    }
    if (dims_cnt > 2) { // 3D
        if (params.dim_2 > dims[1]) {
            if (MY_RANK == 0)
                printf("Failed: Required dimension_3 (%lu) is greater than file dimension (%llu).\n",
                       params.dim_3, dims[2]);
            goto error;
        }
    }

    NUM_PARTICLES = total_particles / NUM_RANKS;

    unsigned long long read_elem_cnt = params.try_num_particles;

    if (read_elem_cnt > NUM_PARTICLES) {
        if (MY_RANK == 0)
            printf("read_elem_cnt_m <= num_particles must hold.\n");
        return 0;
    }

    MPI_Info info = MPI_INFO_NULL;
    if (MY_RANK == 0) {
        printf("Total particles in the file: %lu\n", total_particles);
        printf("Number of particles available per rank: %llu \n", NUM_PARTICLES);
    }

    if (params.num_particles > NUM_PARTICLES) {
        printf("Number of particle per rank exceeds maximum value: %llu \n", NUM_PARTICLES);
        return -1;
    }

    MPI_Barrier(MPI_COMM_WORLD);

    MPI_Allreduce(&NUM_PARTICLES, &TOTAL_PARTICLES, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
    MPI_Scan(&NUM_PARTICLES, &FILE_OFFSET, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
    FILE_OFFSET -= NUM_PARTICLES;
    BUF_STRUCT = prepare_contig_memory_multi_dim(params.dim_1, params.dim_2, params.dim_3);

    unsigned long t1 = get_time_usec();

    if (file_id < 0) {
        if (MY_RANK == 0)
            printf("Error with opening file [%s]!\n", file_name);
        goto done;
    }

    if (MY_RANK == 0)
        printf("Opened HDF5 file... [%s]\n", file_name);

    unsigned long raw_read_time, metadata_time, local_data_size;

    int ret = _run_benchmark_modify(file_id, fapl, gapl, filespace, params, &local_data_size, &raw_read_time,
                                    &metadata_time);

    if (ret < 0) {
        if (MY_RANK == 0)
            printf("_run_benchmark_read() failed.\n");

        goto error;
    }
    MPI_Barrier(MPI_COMM_WORLD);

    H5Pclose(fapl);
    H5Pclose(gapl);
    H5Fclose_async(file_id, 0);

    MPI_Barrier(MPI_COMM_WORLD);
    unsigned long t4 = get_time_usec();

    free_contig_memory(BUF_STRUCT);

    if (MY_RANK == 0) {
        human_readable value;
        char *         mode_str = NULL;

        if (has_vol_async) {
            mode_str = "ASYNC";
        }
        else {
            mode_str = "SYNC";
        }

        printf("\n=================== Performance Results ==================\n");

        printf("Total number of ranks: %d\n", NUM_RANKS);

        unsigned long long total_sleep_time_us =
            read_time_val(params.compute_time, TIME_US) * (params.cnt_time_step - 1);
        printf("Total emulated compute time: %.3lf s\n", total_sleep_time_us / (1000.0 * 1000.0));

        unsigned long total_size_bytes = NUM_RANKS * local_data_size;
        value                          = format_human_readable(total_size_bytes);
        printf("Total modify size: %.3lf %cB\n", value.value, value.unit);

        float rrt_s = (float)raw_read_time / (1000.0 * 1000.0);

        float raw_rate = total_size_bytes / rrt_s;
        printf("Raw modify time: %.3f s \n", rrt_s);

        float meta_time_s = (float)metadata_time / (1000.0 * 1000.0);
        printf("Metadata time: %.3f s\n", meta_time_s);

        float oct_s = (float)(t4 - t1) / (1000.0 * 1000.0);
        printf("Observed modify completion time: %.3f s\n", oct_s);

        value = format_human_readable(raw_rate);
        printf("%s Raw modify rate: %.3f %cB/s \n", mode_str, value.value, value.unit);

        float or_bs = (float)total_size_bytes / ((float)(t4 - t1 - total_sleep_time_us) / (1000.0 * 1000.0));
        value       = format_human_readable(or_bs);
        printf("%s Observed modify rate: %.3f %cB/s\n", mode_str, value.value, value.unit);

        printf("===========================================================\n");

        if (params.useCSV) {
            fprintf(params.csv_fs, "metric, value, unit\n");
            fprintf(params.csv_fs, "operation, %s, %s\n", "overwrite", "");
            fprintf(params.csv_fs, "ranks, %d, %s\n", NUM_RANKS, "");
            fprintf(params.csv_fs, "total compute time, %.3lf, %s\n", total_sleep_time_us / (1000.0 * 1000.0),
                    "seconds");
            value = format_human_readable(total_size_bytes);
            fprintf(params.csv_fs, "total size, %.3lf, %cB\n", value.value, value.unit);
            fprintf(params.csv_fs, "raw time, %.3f, %s\n", rrt_s, "seconds");
            value = format_human_readable(raw_rate);
            fprintf(params.csv_fs, "raw rate, %.3lf, %cB/s\n", value.value, value.unit);
            fprintf(params.csv_fs, "metadata time, %.3f, %s\n", meta_time_s, "seconds");
            value = format_human_readable(or_bs);
            fprintf(params.csv_fs, "observed rate, %.3f, %cB/s\n", value.value, value.unit);
            fprintf(params.csv_fs, "observed time, %.3f, %s\n", oct_s, "seconds");
            fclose(params.csv_fs);
        }
    }

error:
    H5E_BEGIN_TRY
    {
        H5Fclose(file_id);
        H5Pclose(fapl);
    }
    H5E_END_TRY;

done:
    H5close();
    MPI_Finalize();
    return 0;
}
