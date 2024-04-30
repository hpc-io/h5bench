/****** Copyright Notice ***
 *
 * PIOK - Parallel I/O Kernels - VPIC-IO, VORPAL-IO, and GCRM-IO, Copyright
 * (c) 2015, The Regents of the University of California, through Lawrence
 * Berkeley National Laboratory (subject to receipt of any required
 * approvals from the U.S. Dept. of Energy).  All rights reserved.
 *
 * If you have questions about your rights to use or distribute this
 * software, please contact Berkeley Lab's Innovation & Partnerships Office
 * at  IPO@lbl.gov.
 *
 * NOTICE.  This Software was developed under funding from the U.S.
 * Department of Energy and the U.S. Government consequently retains
 * certain rights. As such, the U.S. Government has been granted for itself
 * and others acting on its behalf a paid-up, nonexclusive, irrevocable,
 * worldwide license in the Software to reproduce, distribute copies to the
 * public, prepare derivative works, and perform publicly and display
 * publicly, and to permit other to do so.
 *
 ****************************/

/**
 *
 * Email questions to SByna@lbl.gov
 * Scientific Data Management Research Group
 * Lawrence Berkeley National Laboratory
 *
 */

// Description: This is a simple benchmark based on VPIC's I/O interface
//      Each process writes a specified number of particles into
//      a hdf5 output file using only HDF5 calls
// Author:  Suren Byna <SByna@lbl.gov>
//      Lawrence Berkeley National Laboratory, Berkeley, CA
// Created: in 2011
// Modified:    01/06/2014 --> Removed all H5Part calls and using HDF5 calls
//              02/19/2019 --> Add option to write multiple timesteps of data - Tang
//

// This file is modified by kaushik.v@anl.gov (ALCF)
// Currently, in h5bench_write, each process writes the same amount of local data.
// This program h5bench_write_var_normal_dist.c demonstrates a prototype for each process writing a varying
// size local data buffer which follows a normal distribution based on the given mean number of praticles
// provided from DIM1 and standard deviation STDEV_DIM1 in the config file.

#include <hdf5.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <unistd.h>
#include <assert.h>
#include <string.h>
#include <sys/time.h>
#include <time.h>
#include "../commons/h5bench_util.h"
#include "../commons/async_adaptor.h"
#ifdef HAVE_SUBFILING
#include "H5FDsubfiling.h"
#include "H5FDioc.h"
#endif
#define DIM_MAX 3

herr_t ierr;

typedef struct compress_info {
    int     USE_COMPRESS;
    hid_t   dcpl_id;
    hsize_t chunk_dims[DIM_MAX];
} compress_info;

// Global Variables and dimensions
async_mode    ASYNC_MODE;
compress_info COMPRESS_INFO; // Using parallel compressing: need to set chunk dimensions for dcpl.
long long     NUM_PARTICLES = 0, FILE_OFFSET; // 8  meg particles per process
long long     TOTAL_PARTICLES;
int           NUM_RANKS, MY_RANK, NUM_TIMESTEPS;
int           X_DIM = 64;
int           Y_DIM = 64;
int           Z_DIM = 64;
hid_t         ES_ID, ES_META_CREATE, ES_META_CLOSE, ES_DATA;

// Factors for filling data
const int X_RAND = 191;
const int Y_RAND = 1009;
const int Z_RAND = 3701;

hid_t PARTICLE_COMPOUND_TYPE;
hid_t PARTICLE_COMPOUND_TYPE_SEPARATES[8];

// Optimization globals
int           ALIGN                = 0;
unsigned long ALIGN_THRESHOLD      = 0;
unsigned long ALIGN_LEN            = 0; // 16777216
int           COLL_METADATA        = 0;
int           DEFER_METADATA_FLUSH = 1;
unsigned long STDEV_DIM_1          = 1;

long long *holder = NULL;

typedef struct Particle {
    float x, y, z;
    float px, py, pz;
    int   id_1;
    float id_2;
} particle;

int subfiling = 0;

double
normal_dist_particle_giver(double mu, double sigma)
{
    double        U1, U2, W, mult;
    static double X1, X2;
    static int    call = 0;

    if (call == 1) {
        call = !call;
        // printf("\n Kaushik A %lf call %d",(mu + sigma * (double) X1), call );

        return (mu + sigma * (double)X2);
    }

    do {
        U1 = -1 + ((double)rand() / RAND_MAX) * 2;
        U2 = -1 + ((double)rand() / RAND_MAX) * 2;
        W  = pow(U1, 2) + pow(U2, 2);
    } while (W >= 1 || W == 0);

    mult = sqrt((-2 * log(W)) / W);
    X1   = U1 * mult;
    X2   = U2 * mult;

    call = !call;
    //   printf("\n Kaushik B %lf call %d",(mu + sigma * (double) X1), call );
    return (mu + sigma * (double)X1);
}

void
timestep_es_id_set()
{
    ES_META_CREATE = es_id_set(ASYNC_MODE);
    ES_DATA        = es_id_set(ASYNC_MODE);
    ES_META_CLOSE  = es_id_set(ASYNC_MODE);
}

mem_monitor *MEM_MONITOR;

/**
 * Create a compound HDF5 type to represent the particle.
 * @return The compound HDF5 type.
 */
hid_t
make_compound_type()
{
    PARTICLE_COMPOUND_TYPE = H5Tcreate(H5T_COMPOUND, sizeof(particle));
    H5Tinsert(PARTICLE_COMPOUND_TYPE, "x", HOFFSET(particle, x), H5T_NATIVE_FLOAT);
    H5Tinsert(PARTICLE_COMPOUND_TYPE, "y", HOFFSET(particle, y), H5T_NATIVE_FLOAT);
    H5Tinsert(PARTICLE_COMPOUND_TYPE, "z", HOFFSET(particle, z), H5T_NATIVE_FLOAT);
    H5Tinsert(PARTICLE_COMPOUND_TYPE, "px", HOFFSET(particle, px), H5T_NATIVE_FLOAT);
    H5Tinsert(PARTICLE_COMPOUND_TYPE, "py", HOFFSET(particle, py), H5T_NATIVE_FLOAT);
    H5Tinsert(PARTICLE_COMPOUND_TYPE, "pz", HOFFSET(particle, pz), H5T_NATIVE_FLOAT);
    H5Tinsert(PARTICLE_COMPOUND_TYPE, "id_1", HOFFSET(particle, id_1), H5T_NATIVE_INT);
    H5Tinsert(PARTICLE_COMPOUND_TYPE, "id_2", HOFFSET(particle, id_2), H5T_NATIVE_FLOAT);
    return PARTICLE_COMPOUND_TYPE;
}

hid_t *
make_compound_type_separates()
{
    PARTICLE_COMPOUND_TYPE_SEPARATES[0] = H5Tcreate(H5T_COMPOUND, sizeof(float));
    H5Tinsert(PARTICLE_COMPOUND_TYPE_SEPARATES[0], "x", 0, H5T_NATIVE_FLOAT);

    PARTICLE_COMPOUND_TYPE_SEPARATES[1] = H5Tcreate(H5T_COMPOUND, sizeof(float));
    H5Tinsert(PARTICLE_COMPOUND_TYPE_SEPARATES[1], "y", 0, H5T_NATIVE_FLOAT);

    PARTICLE_COMPOUND_TYPE_SEPARATES[2] = H5Tcreate(H5T_COMPOUND, sizeof(float));
    H5Tinsert(PARTICLE_COMPOUND_TYPE_SEPARATES[2], "z", 0, H5T_NATIVE_FLOAT);

    PARTICLE_COMPOUND_TYPE_SEPARATES[3] = H5Tcreate(H5T_COMPOUND, sizeof(float));
    H5Tinsert(PARTICLE_COMPOUND_TYPE_SEPARATES[3], "px", 0, H5T_NATIVE_FLOAT);

    PARTICLE_COMPOUND_TYPE_SEPARATES[4] = H5Tcreate(H5T_COMPOUND, sizeof(float));
    H5Tinsert(PARTICLE_COMPOUND_TYPE_SEPARATES[4], "py", 0, H5T_NATIVE_FLOAT);

    PARTICLE_COMPOUND_TYPE_SEPARATES[5] = H5Tcreate(H5T_COMPOUND, sizeof(float));
    H5Tinsert(PARTICLE_COMPOUND_TYPE_SEPARATES[5], "pz", 0, H5T_NATIVE_FLOAT);

    PARTICLE_COMPOUND_TYPE_SEPARATES[6] = H5Tcreate(H5T_COMPOUND, sizeof(int));
    H5Tinsert(PARTICLE_COMPOUND_TYPE_SEPARATES[6], "id_1", 0, H5T_NATIVE_INT);

    PARTICLE_COMPOUND_TYPE_SEPARATES[7] = H5Tcreate(H5T_COMPOUND, sizeof(float));
    H5Tinsert(PARTICLE_COMPOUND_TYPE_SEPARATES[7], "id_2", 0, H5T_NATIVE_FLOAT);

    return PARTICLE_COMPOUND_TYPE_SEPARATES;
}

// returns prepared local data volume, used to calculate bandwidth
particle *
prepare_data_interleaved(long particle_cnt, unsigned long *data_size_out)
{
    particle *data_out = (particle *)malloc(particle_cnt * sizeof(particle));

    for (long i = 0; i < particle_cnt; i++) {
        data_out[i].id_1 = i;
        data_out[i].id_2 = (float)(2 * i);
        data_out[i].x    = uniform_random_number() * X_DIM;
        data_out[i].y    = uniform_random_number() * Y_DIM;
        data_out[i].z    = ((float)i / particle_cnt) * Z_DIM;
        data_out[i].px   = uniform_random_number() * X_DIM;
        data_out[i].py   = uniform_random_number() * Y_DIM;
        data_out[i].pz   = ((float)2 * i / particle_cnt) * Z_DIM;
    }
    *data_size_out = particle_cnt * sizeof(particle);
    return data_out;
}

data_contig_md *
prepare_data_contig_1D(unsigned long long particle_cnt, unsigned long *data_size_out)
{
    data_contig_md *data_out = (data_contig_md *)malloc(sizeof(data_contig_md));
    data_out->particle_cnt   = particle_cnt;

    data_out->x     = (float *)malloc(particle_cnt * sizeof(float));
    data_out->y     = (float *)malloc(particle_cnt * sizeof(float));
    data_out->z     = (float *)malloc(particle_cnt * sizeof(float));
    data_out->px    = (float *)malloc(particle_cnt * sizeof(float));
    data_out->py    = (float *)malloc(particle_cnt * sizeof(float));
    data_out->pz    = (float *)malloc(particle_cnt * sizeof(float));
    data_out->id_1  = (int *)malloc(particle_cnt * sizeof(int));
    data_out->id_2  = (float *)malloc(particle_cnt * sizeof(float));
    data_out->dim_1 = particle_cnt;
    data_out->dim_2 = 1;
    data_out->dim_3 = 1;

    for (long i = 0; i < particle_cnt; i++) {
        data_out->id_1[i] = i;
        data_out->id_2[i] = (float)(i * 2);
        data_out->x[i]    = uniform_random_number() * X_DIM;
        data_out->y[i]    = uniform_random_number() * Y_DIM;
        data_out->px[i]   = uniform_random_number() * X_DIM;
        data_out->py[i]   = uniform_random_number() * Y_DIM;
        data_out->z[i]    = ((float)data_out->id_1[i] / NUM_PARTICLES) * Z_DIM;
        data_out->pz[i]   = (data_out->id_2[i] / NUM_PARTICLES) * Z_DIM;
    }
    *data_size_out = particle_cnt * (7 * sizeof(float) + sizeof(int));

    return data_out;
}

data_contig_md *
prepare_data_contig_2D(unsigned long long particle_cnt, long dim_1, long dim_2, unsigned long *data_size_out)
{
    if (particle_cnt != dim_1 * dim_2) {
        if (MY_RANK == 0)
            printf("Invalid dimension definition: dim_1(%ld) * dim_2(%ld) = %ld, must equal num_particles "
                   "(%llu) per rank.\n",
                   dim_1, dim_2, dim_1 * dim_2, particle_cnt);
        return NULL;
    }
    assert(particle_cnt == dim_1 * dim_2);
    data_contig_md *data_out = (data_contig_md *)malloc(sizeof(data_contig_md));
    data_out->particle_cnt   = particle_cnt;
    data_out->dim_1          = dim_1;
    data_out->dim_2          = dim_2;
    data_out->dim_3          = 1;

    data_out->x    = (float *)malloc(particle_cnt * sizeof(float));
    data_out->y    = (float *)malloc(particle_cnt * sizeof(float));
    data_out->z    = (float *)malloc(particle_cnt * sizeof(float));
    data_out->px   = (float *)malloc(particle_cnt * sizeof(float));
    data_out->py   = (float *)malloc(particle_cnt * sizeof(float));
    data_out->pz   = (float *)malloc(particle_cnt * sizeof(float));
    data_out->id_1 = (int *)malloc(particle_cnt * sizeof(int));
    data_out->id_2 = (float *)malloc(particle_cnt * sizeof(float));

    long idx = 0;
    for (long i1 = 0; i1 < dim_1; i1++) {
        for (long i2 = 0; i2 < dim_2; i2++) {
            data_out->id_1[idx] = i1;
            data_out->id_2[idx] = (float)(i1 * 2);
            data_out->x[idx]    = uniform_random_number() * X_DIM;
            data_out->y[idx]    = uniform_random_number() * Y_DIM;
            data_out->px[idx]   = uniform_random_number() * X_DIM;
            data_out->py[idx]   = uniform_random_number() * Y_DIM;
            data_out->z[idx]    = ((float)data_out->id_1[idx] / NUM_PARTICLES) * Z_DIM;
            data_out->pz[idx]   = (data_out->id_2[idx] / NUM_PARTICLES) * Z_DIM;
            idx++;
        }
    }
    *data_size_out = particle_cnt * (7 * sizeof(float) + sizeof(int));

    return data_out;
}

data_contig_md *
prepare_data_contig_3D(unsigned long long particle_cnt, long dim_1, long dim_2, long dim_3,
                       unsigned long *data_size_out)
{
    if (particle_cnt != dim_1 * dim_2 * dim_3) {
        if (MY_RANK == 0)
            printf("Invalid dimension definition: dim_1(%ld) * dim_2(%ld) * dim_3(%ld) = %ld,"
                   " must equal num_particles (%llu) per rank.\n",
                   dim_1, dim_2, dim_3, dim_1 * dim_2 * dim_3, particle_cnt);
        return NULL;
    }

    assert(particle_cnt == dim_1 * dim_2 * dim_3);
    data_contig_md *data_out = (data_contig_md *)malloc(sizeof(data_contig_md));
    data_out->particle_cnt   = particle_cnt;
    data_out->dim_1          = dim_1;
    data_out->dim_2          = dim_2;
    data_out->dim_3          = dim_3;
    data_out->x              = (float *)malloc(particle_cnt * sizeof(float));
    data_out->y              = (float *)malloc(particle_cnt * sizeof(float));
    data_out->z              = (float *)malloc(particle_cnt * sizeof(float));
    data_out->px             = (float *)malloc(particle_cnt * sizeof(float));
    data_out->py             = (float *)malloc(particle_cnt * sizeof(float));
    data_out->pz             = (float *)malloc(particle_cnt * sizeof(float));
    data_out->id_1           = (int *)malloc(particle_cnt * sizeof(int));
    data_out->id_2           = (float *)malloc(particle_cnt * sizeof(float));
    long idx                 = 0;
    for (long i1 = 0; i1 < dim_1; i1++) {
        for (long i2 = 0; i2 < dim_2; i2++) {
            for (long i3 = 0; i3 < dim_3; i3++) {
                data_out->x[idx]    = uniform_random_number() * X_DIM;
                data_out->id_1[idx] = i1;
                data_out->id_2[idx] = (float)(i1 * 2);
                data_out->x[idx]    = uniform_random_number() * X_DIM;
                data_out->y[idx]    = uniform_random_number() * Y_DIM;
                data_out->px[idx]   = uniform_random_number() * X_DIM;
                data_out->py[idx]   = uniform_random_number() * Y_DIM;
                data_out->z[idx]    = ((float)data_out->id_1[idx] / NUM_PARTICLES) * Z_DIM;
                data_out->pz[idx]   = (data_out->id_2[idx] / NUM_PARTICLES) * Z_DIM;
                idx++;
            }
        }
    }
    *data_size_out = particle_cnt * (7 * sizeof(float) + sizeof(int));
    return data_out;
}

void
data_free(write_pattern mode, void *data)
{
    assert(data);
    switch (mode) {
        case CONTIG_CONTIG_1D:
        case CONTIG_COMPOUND_1D:
        case CONTIG_COMPOUND_2D:
        case CONTIG_CONTIG_2D:
        case CONTIG_CONTIG_3D:
            free(((data_contig_md *)data)->x);
            free(((data_contig_md *)data)->y);
            free(((data_contig_md *)data)->z);
            free(((data_contig_md *)data)->px);
            free(((data_contig_md *)data)->py);
            free(((data_contig_md *)data)->pz);
            free(((data_contig_md *)data)->id_1);
            free(((data_contig_md *)data)->id_2);
            free(((data_contig_md *)data));
            break;
        case COMPOUND_CONTIG_1D:
        case COMPOUND_CONTIG_2D:
        case COMPOUND_COMPOUND_1D:
        case COMPOUND_COMPOUND_2D:
            free(data);
            break;
        default:
            break;
    }
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

int
set_select_spaces_default(hid_t *filespace_out, hid_t *memspace_out)
{
    hsize_t count[1] = {1};
    *filespace_out   = H5Screate_simple(1, (hsize_t *)&TOTAL_PARTICLES, NULL);
    *memspace_out    = H5Screate_simple(1, (hsize_t *)&NUM_PARTICLES, NULL);
    H5Sselect_hyperslab(*filespace_out, H5S_SELECT_SET, (hsize_t *)&FILE_OFFSET, NULL, count,
                        (hsize_t *)&NUM_PARTICLES);
    //    printf("TOTAL_PARTICLES = %lld, NUM_PARTICLES = %lld \n", TOTAL_PARTICLES, NUM_PARTICLES);
    return 0;
}

unsigned long
set_select_spaces_strided(bench_params params, hid_t *filespace_out, hid_t *memspace_out)
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

    *memspace_out  = H5Screate_simple(1, (hsize_t *)&actual_elem_cnt, NULL);
    *filespace_out = H5Screate_simple(1, (hsize_t *)&TOTAL_PARTICLES, NULL);
    H5Sselect_hyperslab(*filespace_out, H5S_SELECT_SET, (hsize_t *)&FILE_OFFSET, // start-offset
                        (hsize_t *)&params.stride,                               // stride
                        (hsize_t *)&params.block_cnt,                            // block cnt
                        (hsize_t *)&params.block_size);                          // block size
    return actual_elem_cnt;
}

int
set_select_space_2D_array(hid_t *filespace_out, hid_t *memspace_out, unsigned long dim_1, unsigned long dim_2)
{ // dim_1 * dim_2 === NUM_PARTICLES
    hsize_t mem_dims[2], file_dims[2];
    mem_dims[0]  = (hsize_t)dim_1;
    mem_dims[1]  = (hsize_t)dim_2;
    file_dims[0] = (hsize_t)dim_1 * NUM_RANKS; // total x length: dim_1 * world_size.
    file_dims[1] = (hsize_t)dim_2;             // always the same dim_2

    hsize_t count[2] = {1, 1};
    hsize_t file_starts[2], block[2];   // select start point and range in each dimension.
    file_starts[0] = dim_1 * (MY_RANK); // file offset for each rank
    file_starts[1] = 0;
    block[0]       = dim_1;
    block[1]       = dim_2;

    *filespace_out = H5Screate_simple(2, file_dims, NULL);
    *memspace_out  = H5Screate_simple(2, mem_dims, NULL);
    if (MY_RANK == 0)
        printf("%lu * %lu 2D array, my x_start = %lu, y_start = %lu, x_blk = %lu, y_blk = %lu\n", dim_1,
               dim_2, file_starts[0], file_starts[1], block[0], block[1]);
    H5Sselect_hyperslab(*filespace_out, H5S_SELECT_SET, file_starts, NULL, count, block);
    return 0;
}

int
set_select_space_multi_3D_array(hid_t *filespace_out, hid_t *memspace_out, unsigned long dim_1,
                                unsigned long dim_2, unsigned long dim_3)
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

    *filespace_out = H5Screate_simple(3, file_dims, NULL);
    *memspace_out  = H5Screate_simple(3, mem_dims, NULL);

    H5Sselect_hyperslab(*filespace_out, H5S_SELECT_SET, file_starts, NULL, count, file_range);
    return 0;
}

/*
 *  write file: create m-D array as the dateset type, now linear-linear is 8 datasets of 1D array
 */
void
data_write_contig_contig_MD_array(time_step *ts, hid_t loc, hid_t *dset_ids, hid_t filespace, hid_t memspace,
                                  hid_t plist_id, data_contig_md *data_in, unsigned long *metadata_time,
                                  unsigned long *data_time)
{
    assert(data_in && data_in->x);
    hid_t dcpl;
    if (COMPRESS_INFO.USE_COMPRESS)
        dcpl = COMPRESS_INFO.dcpl_id;
    else
        dcpl = H5P_DEFAULT;

    if (MY_RANK == 0) {
        if (COMPRESS_INFO.USE_COMPRESS)
            printf("Parallel compressed: chunk_dim1 = %lu, chunk_dim2 = %lu\n", COMPRESS_INFO.chunk_dims[0],
                   COMPRESS_INFO.chunk_dims[1]);
    }

    unsigned t1 = get_time_usec();

    dset_ids[0] = H5Dcreate_async(loc, "x", H5T_NATIVE_FLOAT, filespace, H5P_DEFAULT, dcpl, H5P_DEFAULT,
                                  ts->es_meta_create);
    dset_ids[1] = H5Dcreate_async(loc, "y", H5T_NATIVE_FLOAT, filespace, H5P_DEFAULT, dcpl, H5P_DEFAULT,
                                  ts->es_meta_create);
    dset_ids[2] = H5Dcreate_async(loc, "z", H5T_NATIVE_FLOAT, filespace, H5P_DEFAULT, dcpl, H5P_DEFAULT,
                                  ts->es_meta_create);
    dset_ids[3] = H5Dcreate_async(loc, "px", H5T_NATIVE_FLOAT, filespace, H5P_DEFAULT, dcpl, H5P_DEFAULT,
                                  ts->es_meta_create);
    dset_ids[4] = H5Dcreate_async(loc, "py", H5T_NATIVE_FLOAT, filespace, H5P_DEFAULT, dcpl, H5P_DEFAULT,
                                  ts->es_meta_create);
    dset_ids[5] = H5Dcreate_async(loc, "pz", H5T_NATIVE_FLOAT, filespace, H5P_DEFAULT, dcpl, H5P_DEFAULT,
                                  ts->es_meta_create);
    dset_ids[6] = H5Dcreate_async(loc, "id_1", H5T_NATIVE_INT, filespace, H5P_DEFAULT, dcpl, H5P_DEFAULT,
                                  ts->es_meta_create);
    dset_ids[7] = H5Dcreate_async(loc, "id_2", H5T_NATIVE_FLOAT, filespace, H5P_DEFAULT, dcpl, H5P_DEFAULT,
                                  ts->es_meta_create);

    unsigned t2 = get_time_usec();

    ierr =
        H5Dwrite_async(dset_ids[0], H5T_NATIVE_FLOAT, memspace, filespace, plist_id, data_in->x, ts->es_data);
    ierr =
        H5Dwrite_async(dset_ids[1], H5T_NATIVE_FLOAT, memspace, filespace, plist_id, data_in->y, ts->es_data);
    ierr =
        H5Dwrite_async(dset_ids[2], H5T_NATIVE_FLOAT, memspace, filespace, plist_id, data_in->z, ts->es_data);
    ierr = H5Dwrite_async(dset_ids[3], H5T_NATIVE_FLOAT, memspace, filespace, plist_id, data_in->px,
                          ts->es_data);
    ierr = H5Dwrite_async(dset_ids[4], H5T_NATIVE_FLOAT, memspace, filespace, plist_id, data_in->py,
                          ts->es_data);
    ierr = H5Dwrite_async(dset_ids[5], H5T_NATIVE_FLOAT, memspace, filespace, plist_id, data_in->pz,
                          ts->es_data);
    ierr = H5Dwrite_async(dset_ids[6], H5T_NATIVE_INT, memspace, filespace, plist_id, data_in->id_1,
                          ts->es_data);
    ierr = H5Dwrite_async(dset_ids[7], H5T_NATIVE_FLOAT, memspace, filespace, plist_id, data_in->id_2,
                          ts->es_data);

    unsigned t3 = get_time_usec();

    *metadata_time = t2 - t1;
    *data_time     = t3 - t2;

    if (MY_RANK == 0)
        printf("    %s: Finished writing time step \n", __func__);
}

void
data_write_contig_to_interleaved(time_step *ts, hid_t loc, hid_t *dset_ids, hid_t filespace, hid_t memspace,
                                 hid_t plist_id, data_contig_md *data_in, unsigned long *metadata_time,
                                 unsigned long *data_time)
{
    assert(data_in && data_in->x);
    hid_t dcpl;
    if (COMPRESS_INFO.USE_COMPRESS)
        dcpl = COMPRESS_INFO.dcpl_id;
    else
        dcpl = H5P_DEFAULT;

    unsigned t1 = get_time_usec();
    dset_ids[0] = H5Dcreate_async(loc, "particles", PARTICLE_COMPOUND_TYPE, filespace, H5P_DEFAULT, dcpl,
                                  H5P_DEFAULT, ts->es_meta_create);

    unsigned t2 = get_time_usec();
    ierr = H5Dwrite_async(dset_ids[0], PARTICLE_COMPOUND_TYPE_SEPARATES[0], memspace, filespace, plist_id,
                          data_in->x, ts->es_data);
    ierr = H5Dwrite_async(dset_ids[0], PARTICLE_COMPOUND_TYPE_SEPARATES[1], memspace, filespace, plist_id,
                          data_in->y, ts->es_data);
    ierr = H5Dwrite_async(dset_ids[0], PARTICLE_COMPOUND_TYPE_SEPARATES[2], memspace, filespace, plist_id,
                          data_in->z, ts->es_data);
    ierr = H5Dwrite_async(dset_ids[0], PARTICLE_COMPOUND_TYPE_SEPARATES[3], memspace, filespace, plist_id,
                          data_in->px, ts->es_data);
    ierr = H5Dwrite_async(dset_ids[0], PARTICLE_COMPOUND_TYPE_SEPARATES[4], memspace, filespace, plist_id,
                          data_in->py, ts->es_data);
    ierr = H5Dwrite_async(dset_ids[0], PARTICLE_COMPOUND_TYPE_SEPARATES[5], memspace, filespace, plist_id,
                          data_in->pz, ts->es_data);
    ierr = H5Dwrite_async(dset_ids[0], PARTICLE_COMPOUND_TYPE_SEPARATES[6], memspace, filespace, plist_id,
                          data_in->id_1, ts->es_data);
    ierr = H5Dwrite_async(dset_ids[0], PARTICLE_COMPOUND_TYPE_SEPARATES[7], memspace, filespace, plist_id,
                          data_in->id_2, ts->es_data);

    unsigned t3    = get_time_usec();
    *metadata_time = t2 - t1;
    *data_time     = t3 - t2;
    if (MY_RANK == 0)
        printf("    %s: Finished writing time step \n", __func__);
}

void
data_write_interleaved_to_contig(time_step *ts, hid_t loc, hid_t *dset_ids, hid_t filespace, hid_t memspace,
                                 hid_t plist_id, particle *data_in, unsigned long *metadata_time,
                                 unsigned long *data_time)
{
    assert(data_in);
    hid_t dcpl;
    if (COMPRESS_INFO.USE_COMPRESS)
        dcpl = COMPRESS_INFO.dcpl_id;
    else
        dcpl = H5P_DEFAULT;

    unsigned t1 = get_time_usec();
    dset_ids[0] = H5Dcreate_async(loc, "x", PARTICLE_COMPOUND_TYPE_SEPARATES[0], filespace, H5P_DEFAULT, dcpl,
                                  H5P_DEFAULT, ts->es_meta_create);
    dset_ids[1] = H5Dcreate_async(loc, "y", PARTICLE_COMPOUND_TYPE_SEPARATES[1], filespace, H5P_DEFAULT, dcpl,
                                  H5P_DEFAULT, ts->es_meta_create);
    dset_ids[2] = H5Dcreate_async(loc, "z", PARTICLE_COMPOUND_TYPE_SEPARATES[2], filespace, H5P_DEFAULT, dcpl,
                                  H5P_DEFAULT, ts->es_meta_create);
    dset_ids[3] = H5Dcreate_async(loc, "px", PARTICLE_COMPOUND_TYPE_SEPARATES[3], filespace, H5P_DEFAULT,
                                  dcpl, H5P_DEFAULT, ts->es_meta_create);
    dset_ids[4] = H5Dcreate_async(loc, "py", PARTICLE_COMPOUND_TYPE_SEPARATES[4], filespace, H5P_DEFAULT,
                                  dcpl, H5P_DEFAULT, ts->es_meta_create);
    dset_ids[5] = H5Dcreate_async(loc, "pz", PARTICLE_COMPOUND_TYPE_SEPARATES[5], filespace, H5P_DEFAULT,
                                  dcpl, H5P_DEFAULT, ts->es_meta_create);
    dset_ids[6] = H5Dcreate_async(loc, "id_1", PARTICLE_COMPOUND_TYPE_SEPARATES[6], filespace, H5P_DEFAULT,
                                  dcpl, H5P_DEFAULT, ts->es_meta_create);
    dset_ids[7] = H5Dcreate_async(loc, "id_2", PARTICLE_COMPOUND_TYPE_SEPARATES[7], filespace, H5P_DEFAULT,
                                  dcpl, H5P_DEFAULT, ts->es_meta_create);

    unsigned t2 = get_time_usec();

    ierr = H5Dwrite_async(dset_ids[0], PARTICLE_COMPOUND_TYPE, memspace, filespace, plist_id, data_in,
                          ts->es_data);
    ierr = H5Dwrite_async(dset_ids[1], PARTICLE_COMPOUND_TYPE, memspace, filespace, plist_id, data_in,
                          ts->es_data);
    ierr = H5Dwrite_async(dset_ids[2], PARTICLE_COMPOUND_TYPE, memspace, filespace, plist_id, data_in,
                          ts->es_data);
    ierr = H5Dwrite_async(dset_ids[3], PARTICLE_COMPOUND_TYPE, memspace, filespace, plist_id, data_in,
                          ts->es_data);
    ierr = H5Dwrite_async(dset_ids[4], PARTICLE_COMPOUND_TYPE, memspace, filespace, plist_id, data_in,
                          ts->es_data);
    ierr = H5Dwrite_async(dset_ids[5], PARTICLE_COMPOUND_TYPE, memspace, filespace, plist_id, data_in,
                          ts->es_data);
    ierr = H5Dwrite_async(dset_ids[6], PARTICLE_COMPOUND_TYPE, memspace, filespace, plist_id, data_in,
                          ts->es_data);
    ierr = H5Dwrite_async(dset_ids[7], PARTICLE_COMPOUND_TYPE, memspace, filespace, plist_id, data_in,
                          ts->es_data);

    unsigned t3    = get_time_usec();
    *metadata_time = t2 - t1;
    *data_time     = t3 - t2;
    if (MY_RANK == 0)
        printf("    %s: Finished writing time step \n", __func__);
}

void
data_write_interleaved_to_interleaved(time_step *ts, hid_t loc, hid_t *dset_ids, hid_t filespace,
                                      hid_t memspace, hid_t plist_id, particle *data_in,
                                      unsigned long *metadata_time, unsigned long *data_time)
{
    assert(data_in);
    hid_t dcpl;
    if (COMPRESS_INFO.USE_COMPRESS)
        dcpl = COMPRESS_INFO.dcpl_id;
    else
        dcpl = H5P_DEFAULT;
    unsigned t1 = get_time_usec();
    dset_ids[0] = H5Dcreate_async(loc, "particles", PARTICLE_COMPOUND_TYPE, filespace, H5P_DEFAULT, dcpl,
                                  H5P_DEFAULT, ts->es_meta_create);

    unsigned t2 = get_time_usec();
    ierr        = H5Dwrite_async(dset_ids[0], PARTICLE_COMPOUND_TYPE, memspace, filespace, plist_id, data_in,
                          ts->es_data);

    // should write all things in data_in
    unsigned t3    = get_time_usec();
    *metadata_time = t2 - t1;
    *data_time     = t3 - t2;
    if (MY_RANK == 0)
        printf("    %s: Finished writing time step \n", __func__);
}

void *
_prepare_data(bench_params params, hid_t *filespace_out, hid_t *memspace_out,
              unsigned long *data_preparation_time, unsigned long *data_size)
{
    void *data = NULL;

    make_compound_type_separates();
    make_compound_type();
    hid_t filespace, memspace;
    *data_preparation_time = 0;

    //    unsigned long data_size;
    unsigned long long particle_cnt    = NUM_PARTICLES;
    unsigned long      actual_elem_cnt = 0; // only for set_select_spaces_strided()
    int                dset_cnt        = 0;
    unsigned long      t_prep_start    = get_time_usec();
    switch (params.access_pattern.pattern_write) {
        case CONTIG_CONTIG_1D:
            set_select_spaces_default(filespace_out, memspace_out);
            data     = (void *)prepare_data_contig_1D(particle_cnt, data_size);
            dset_cnt = 8;
            break;

        case CONTIG_CONTIG_2D:
            set_select_space_2D_array(filespace_out, memspace_out, params.dim_1, params.dim_2);
            data     = (void *)prepare_data_contig_2D(particle_cnt, params.dim_1, params.dim_2, data_size);
            dset_cnt = 8;
            break;

        case CONTIG_CONTIG_STRIDED_1D:
            actual_elem_cnt = set_select_spaces_strided(params, filespace_out, memspace_out);
            if (actual_elem_cnt < 1) {
                printf("Strided write setting error.\n");
                return NULL;
            }
            data     = (void *)prepare_data_contig_1D(actual_elem_cnt, data_size);
            dset_cnt = 8;
            break;

        case CONTIG_COMPOUND_1D:
            set_select_spaces_default(filespace_out, memspace_out);
            data     = (void *)prepare_data_contig_1D(particle_cnt, data_size);
            dset_cnt = 1;
            break;

        case CONTIG_COMPOUND_2D:
            set_select_space_2D_array(filespace_out, memspace_out, params.dim_1, params.dim_2);
            data     = (void *)prepare_data_contig_2D(particle_cnt, params.dim_1, params.dim_2, data_size);
            dset_cnt = 1;
            break;

        case COMPOUND_CONTIG_1D:
            set_select_spaces_default(filespace_out, memspace_out);
            data     = (void *)prepare_data_interleaved(particle_cnt, data_size);
            dset_cnt = 8;
            break;

        case COMPOUND_CONTIG_2D:
            set_select_space_2D_array(filespace_out, memspace_out, params.dim_1, params.dim_2);
            data     = (void *)prepare_data_interleaved(particle_cnt, data_size);
            dset_cnt = 8;
            break;

        case COMPOUND_COMPOUND_1D:
            set_select_spaces_default(filespace_out, memspace_out);
            data     = (void *)prepare_data_interleaved(particle_cnt, data_size);
            dset_cnt = 1;
            break;

        case COMPOUND_COMPOUND_2D:
            set_select_space_2D_array(filespace_out, memspace_out, params.dim_1, params.dim_2);
            data     = (void *)prepare_data_interleaved(particle_cnt, data_size);
            dset_cnt = 1;
            break;

        case CONTIG_CONTIG_3D:
            set_select_space_multi_3D_array(filespace_out, memspace_out, params.dim_1, params.dim_2,
                                            params.dim_3);
            data     = (void *)prepare_data_contig_3D(particle_cnt, params.dim_1, params.dim_2, params.dim_3,
                                                  data_size);
            dset_cnt = 8;
            break;
        default:
            assert(0 && "this mode is not yet implemented");
            break;
    }
    *data_preparation_time = get_time_usec() - t_prep_start;
    return data;
}

int
_run_benchmark_write(bench_params params, hid_t file_id, hid_t fapl, hid_t filespace, hid_t memspace,
                     void *data, unsigned long data_size, unsigned long *total_data_size_out,
                     unsigned long *data_time_total, unsigned long *metadata_time_total)
{
    unsigned long long data_preparation_time;

    write_pattern pattern      = params.access_pattern.pattern_write;
    int           timestep_cnt = params.cnt_time_step;
    *metadata_time_total       = 0;
    *data_time_total           = 0;
    char  grp_name[128];
    int   grp_cnt = 0, dset_cnt = 0;
    hid_t plist_id; //, filespace, memspace;

    if (params.file_per_proc) {
        plist_id = H5Pcreate(H5P_DATASET_XFER);
    }
    else {
        set_dspace_plist(&plist_id, params.data_coll);
    }

    if (!data) {
        if (MY_RANK == 0)
            printf("Failed to generate data for writing, "
                   "please check dimension settings in the config file.\n");
        return -1;
    }

    MEM_MONITOR = mem_monitor_new(timestep_cnt, ASYNC_MODE, data_size, params.io_mem_limit);

    if (!MEM_MONITOR) {
        printf("Invalid MEM_MONITOR returned: NULL\n");
        return -1;
    }

    timestep_es_id_set();

    unsigned long metadata_time_exp = 0, data_time_exp = 0, t0, t1, t2, t3, t4;
    unsigned long metadata_time_imp = 0, data_time_imp = 0;
    unsigned long meta_time1 = 0, meta_time2 = 0, meta_time3 = 0, meta_time4 = 0, meta_time5 = 0;
    for (int ts_index = 0; ts_index < timestep_cnt; ts_index++) {
        meta_time1 = 0, meta_time2 = 0, meta_time3 = 0, meta_time4 = 0, meta_time5 = 0;
        time_step *ts = &(MEM_MONITOR->time_steps[ts_index]);
        MEM_MONITOR->mem_used += ts->mem_size;
        //        print_mem_bound(MEM_MONITOR);
        sprintf(grp_name, "Timestep_%d", ts_index);
        assert(ts);

        if (params.cnt_time_step_delay > 0) {
            if (ts_index > params.cnt_time_step_delay - 1) // delayed close on all ids of the previous ts
                ts_delayed_close(MEM_MONITOR, &meta_time1, dset_cnt);
        }

        mem_monitor_check_run(MEM_MONITOR, &meta_time2, &data_time_imp);

        t0 = get_time_usec();
        ts->grp_id =
            H5Gcreate_async(file_id, grp_name, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT, ts->es_meta_create);

        t1         = get_time_usec();
        meta_time3 = (t1 - t0);

        if (MY_RANK == 0)
            printf("Writing %s ... \n", grp_name);

        switch (pattern) {
            case CONTIG_CONTIG_1D:
            case CONTIG_CONTIG_2D:
            case CONTIG_CONTIG_3D:
            case CONTIG_CONTIG_STRIDED_1D:
                data_write_contig_contig_MD_array(ts, ts->grp_id, ts->dset_ids, filespace, memspace, plist_id,
                                                  (data_contig_md *)data, &meta_time4, &data_time_exp);
                dset_cnt = 8;
                break;

            case CONTIG_COMPOUND_1D:
            case CONTIG_COMPOUND_2D:
                data_write_contig_to_interleaved(ts, ts->grp_id, ts->dset_ids, filespace, memspace, plist_id,
                                                 (data_contig_md *)data, &meta_time4, &data_time_exp);
                dset_cnt = 1;
                break;

            case COMPOUND_CONTIG_1D:
            case COMPOUND_CONTIG_2D:
                data_write_interleaved_to_contig(ts, ts->grp_id, ts->dset_ids, filespace, memspace, plist_id,
                                                 (particle *)data, &meta_time4, &data_time_exp);
                dset_cnt = 8;
                break;

            case COMPOUND_COMPOUND_1D:
            case COMPOUND_COMPOUND_2D:
                data_write_interleaved_to_interleaved(ts, ts->grp_id, ts->dset_ids, filespace, memspace,
                                                      plist_id, (particle *)data, &meta_time4,
                                                      &data_time_exp);
                dset_cnt = 1;
                break;

            default:
                break;
        }

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
            meta_time5 += (t4 - t3);
        }

        if (ts_index != timestep_cnt - 1) { // no sleep after the last ts
            if (params.compute_time.time_num >= 0) {
                if (MY_RANK == 0)
                    printf("Computing...\n");
                async_sleep(ts->es_data, params.compute_time);
            }
        }

        *metadata_time_total += (meta_time1 + meta_time2 + meta_time3 + meta_time4);
        *data_time_total += (data_time_exp + data_time_imp);
    } // end for timestep_cnt

    // all done, check if any timesteps undone

    mem_monitor_final_run(MEM_MONITOR, &metadata_time_imp, &data_time_imp);

    *metadata_time_total += metadata_time_imp;
    *data_time_total += data_time_imp;

    H5Tclose(PARTICLE_COMPOUND_TYPE);
    for (int i = 0; i < 8; i++)
        H5Tclose(PARTICLE_COMPOUND_TYPE_SEPARATES[i]);

    *total_data_size_out = timestep_cnt * data_size;

    data_free(pattern, data);
    H5Sclose(memspace);
    H5Sclose(filespace);
    H5Pclose(plist_id);
    return 0;
}

void
set_globals(const bench_params *params)
{
    NUM_PARTICLES = params->num_particles;
    NUM_TIMESTEPS = params->cnt_time_step;
    // Following variables only used to generate data
    X_DIM                       = X_RAND;
    Y_DIM                       = Y_RAND;
    Z_DIM                       = Z_RAND;
    COMPRESS_INFO.USE_COMPRESS  = params->useCompress;
    COMPRESS_INFO.chunk_dims[0] = params->chunk_dim_1;
    COMPRESS_INFO.chunk_dims[1] = params->chunk_dim_2;
    COMPRESS_INFO.chunk_dims[2] = params->chunk_dim_3;

    if (COMPRESS_INFO.USE_COMPRESS) { // set DCPL
        herr_t ret;
        COMPRESS_INFO.dcpl_id = H5Pcreate(H5P_DATASET_CREATE);
        assert(COMPRESS_INFO.dcpl_id > 0);

        /* Set chunked layout and chunk dimensions */
        ret = H5Pset_layout(COMPRESS_INFO.dcpl_id, H5D_CHUNKED);
        assert(ret >= 0);
        ret =
            H5Pset_chunk(COMPRESS_INFO.dcpl_id, params->num_dims, (const hsize_t *)COMPRESS_INFO.chunk_dims);
        assert(ret >= 0);
        ret = H5Pset_deflate(COMPRESS_INFO.dcpl_id, 9);
        assert(ret >= 0);
    }

    ASYNC_MODE = params->asyncMode;
}

hid_t
set_fapl()
{
    hid_t fapl = H5Pcreate(H5P_FILE_ACCESS);
    return fapl;
}

hid_t
set_metadata(hid_t fapl, int align, unsigned long threshold, unsigned long alignment_len, int meta_collective)
{
    hsize_t threshold_o, alignment_len_o;
    herr_t  ret;
    if (align == 1) {
        H5Pset_alignment(fapl, threshold, alignment_len);

        ret = H5Pget_alignment(fapl, &threshold_o, &alignment_len_o);
        if (ret < 0)
            if (MY_RANK == 0)
                printf("H5Pget_alignment failed \n");

        if (MY_RANK == 0) {
            printf("GPFS alignment settings: ON\n");
            printf("Value of alignment length :  %ld\n", alignment_len_o);
            printf("Value of alignment threshold :  %ld\n", threshold_o);
        }
    }
    if (meta_collective == 1) {
        if (MY_RANK == 0)
            printf("Collective Metadata operations: ON\n");
#if H5_VERSION_GE(1, 10, 0)
        H5Pset_all_coll_metadata_ops(fapl, 1);
        H5Pset_coll_metadata_write(fapl, 1);
#endif
    }
    else {
        if (MY_RANK == 0)
            printf("Collective Metadata operations: OFF\n");
    }

    // Defer metadata flush
    if (DEFER_METADATA_FLUSH) {
        H5AC_cache_config_t cache_config;
        cache_config.version = H5AC__CURR_CACHE_CONFIG_VERSION;
        H5Pget_mdc_config(fapl, &cache_config);
        cache_config.set_initial_size  = 1;
        cache_config.initial_size      = 16 * M_VAL;
        cache_config.evictions_enabled = 0;
        cache_config.incr_mode         = H5C_incr__off;
        cache_config.flash_incr_mode   = H5C_flash_incr__off;
        cache_config.decr_mode         = H5C_decr__off;
        H5Pset_mdc_config(fapl, &cache_config);
    }
    return fapl;
}

void
print_usage(char *name)
{
    if (MY_RANK == 0) {
        printf("=============== Usage: %s /path_to_config_file /path_to_output_data_file [CSV "
               "csv_file_path]=============== \n",
               name);
        printf("- CSV is optional.\n");
        printf("- Only CC/CI/IC/II/CC2D/CC3D is used to set benchmark mode in the config file, stands for "
               "CONTIG_CONTIG_1D, CONTIG_COMPOUND_1D, COMPOUND_CONTIG_1D, COMPOUND_COMPOUND_1D, 2D Array and "
               "3D Array\n");
        printf("- For 2D/3D benchmarks, make sure the dimensions are set correctly and matches the per rank "
               "particle number.\n");
        printf("- For example, when your PATTERN is CC3D, and PARTICLE_CNT_M is 1, setting DIM_1~3 to 64, "
               "64, and 256 is valid, because 64*64*256 = 1,048,576 (1M); and 10*20*30 is invalid. \n");
    }
}

int
main(int argc, char *argv[])
{
    int mpi_thread_lvl_provided = -1;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &mpi_thread_lvl_provided);
    assert(MPI_THREAD_MULTIPLE == mpi_thread_lvl_provided);
    MPI_Comm_rank(MPI_COMM_WORLD, &MY_RANK);
    MPI_Comm_size(MPI_COMM_WORLD, &NUM_RANKS);
    MPI_Comm           comm    = MPI_COMM_WORLD;
    MPI_Info           info    = MPI_INFO_NULL;
    char *             num_str = "1024 Ks";
    unsigned long long num     = 0;

    char buffer[200];

    int rand_seed_value = time(NULL);
    srand(rand_seed_value);

    if (MY_RANK == 0) {
        if (argc != 3) {
            print_usage(argv[0]);
            return 0;
        }
    }

    char *       output_file;
    bench_params params;

    char *cfg_file_path = argv[1];
    output_file         = argv[2];
    if (MY_RANK == 0) {
        printf("Configuration file: %s\n", argv[1]);
        printf("Output data file: %s\n", argv[2]);
    }
    int do_write = 1;
    if (read_config(cfg_file_path, &params, do_write) < 0) {
        if (MY_RANK == 0)
            printf("Configuration file read failed. Please, check %s\n", cfg_file_path);
        return 0;
    }

    if (params.io_op != IO_WRITE) {
        if (MY_RANK == 0)
            printf("Make sure the configuration file has IO_OPERATION=WRITE defined\n");
        return 0;
    }

    if (params.useCompress)
        params.data_coll = 1;

    if (params.subfiling)
        subfiling = 1;

#if H5_VERSION_GE(1, 13, 0)
    if (H5VLis_connector_registered_by_name("async")) {
        if (MY_RANK == 0) {
            printf("Using 'async' VOL connector\n");
        }
    }
#endif

    if (MY_RANK == 0) {
        print_params(&params);
    }

    set_globals(&params);

    NUM_TIMESTEPS = params.cnt_time_step;

    STDEV_DIM_1 = params.stdev_dim_1;

    holder = (long long *)malloc(NUM_RANKS * sizeof(long long));

    if (MY_RANK == 0) {
        printf("Start benchmark: h5bench_write\n");
        printf("Given mean particles : %llu \n", NUM_PARTICLES);
        printf("Given standard deviation : %ld \n", STDEV_DIM_1);
        printf("Total ranks %i \n", NUM_RANKS);


	if (params.useDataDist) {
	  // read data file listed in config file
	  char size_line[256] = "";

	  printf("Begin data dist processing\n");
	  printf("Read data file %s\n", params.data_dist_path);

	  FILE *file = fopen(params.data_dist_path, "r");

	  while (fgets(size_line, 256, file)) {

	    printf("Read line: %s\n", size_line);
	    char *tokens[2];
	    int index;
	    long long size;
	    char *tok = strtok(size_line, " ");
	    if (tok) {
	      index = atoi(tok);
	      tok       = strtok(NULL, " ");
	      if (tok) {
		/* don't compute scale if factor is 1, identity */
		if (params.data_dist_scale == 1.0)
		  holder[index] = strtoll(tok, NULL, 10);
		else
		  holder[index] = (long long) (params.data_dist_scale * strtoll(tok, NULL, 10));
	      }
	    } else {
	      return -1;
	    }
	  }
	}

	else {
	  for (int i = 0; i < NUM_RANKS; i++) {
	    holder[i] = (long long)normal_dist_particle_giver(NUM_PARTICLES, STDEV_DIM_1);
	  }
	}
    }

    MPI_Barrier(MPI_COMM_WORLD);

    MPI_Scatter(&holder[0], 1, MPI_LONG_LONG, &NUM_PARTICLES, 1, MPI_LONG_LONG, 0, MPI_COMM_WORLD);
    printf("\n Number of particles in rank %d = %lld \n", MY_RANK, NUM_PARTICLES);

    unsigned long total_write_size =
        NUM_RANKS * NUM_TIMESTEPS * NUM_PARTICLES * (7 * sizeof(float) + sizeof(int));
    hid_t         filespace = 0, memspace = 0;
    unsigned long data_size             = 0;
    unsigned long data_preparation_time = 0;

    MPI_Barrier(MPI_COMM_WORLD);

    MPI_Allreduce(&NUM_PARTICLES, &TOTAL_PARTICLES, 1, MPI_LONG_LONG, MPI_SUM, comm);
    MPI_Scan(&NUM_PARTICLES, &FILE_OFFSET, 1, MPI_LONG_LONG, MPI_SUM, comm);

    FILE_OFFSET -= NUM_PARTICLES;

    if (MY_RANK == 0)
        printf("Total number of particles: %lldM\n", TOTAL_PARTICLES / (M_VAL));

    hid_t fapl      = set_fapl();
    ALIGN           = params.align;
    ALIGN_THRESHOLD = params.align_threshold;
    ALIGN_LEN       = params.align_len;

    if (params.file_per_proc) {
    }
    else {
#ifdef HAVE_SUBFILING
        if (params.subfiling == 1)
            H5Pset_fapl_subfiling(fapl, NULL);
        else
#endif
            H5Pset_fapl_mpio(fapl, comm, info);
        set_metadata(fapl, ALIGN, ALIGN_THRESHOLD, ALIGN_LEN, params.meta_coll);
    }

    void *data = _prepare_data(params, &filespace, &memspace, &data_preparation_time, &data_size);

    unsigned long t1 = get_time_usec();

    hid_t file_id;

    unsigned long tfopen_start = get_time_usec();
    if (params.file_per_proc) {
        char mpi_rank_output_file_path[4096];
        sprintf(mpi_rank_output_file_path, "%s/rank_%d_%s", get_dir_from_path(output_file), MY_RANK,
                get_file_name_from_path(output_file));

        file_id = H5Fcreate_async(mpi_rank_output_file_path, H5F_ACC_TRUNC, H5P_DEFAULT, fapl, 0);
    }
    else {
        file_id = H5Fcreate_async(output_file, H5F_ACC_TRUNC, H5P_DEFAULT, fapl, 0);
    }
    unsigned long tfopen_end = get_time_usec();

    if (MY_RANK == 0)
        printf("Opened HDF5 file... \n");

    MPI_Barrier(MPI_COMM_WORLD);
    unsigned long t2 = get_time_usec(); // t2 - t1: metadata: creating/opening

    unsigned long raw_write_time, inner_metadata_time, local_data_size;
    int           stat = _run_benchmark_write(params, file_id, fapl, filespace, memspace, data, data_size,
                                    &local_data_size, &raw_write_time, &inner_metadata_time);

    if (stat < 0) {
        if (MY_RANK == 0)
            printf("\n==================== Benchmark Failed ====================\n");
        assert(0);
    }

    unsigned long t3 = get_time_usec(); // t3 - t2: writting data, including metadata

    H5Pclose(fapl);
    unsigned long tflush_start = get_time_usec();
    H5Fflush(file_id, H5F_SCOPE_LOCAL);
    MPI_Barrier(MPI_COMM_WORLD);
    unsigned long tflush_end = get_time_usec();

    unsigned long tfclose_start = get_time_usec();

    H5Fclose_async(file_id, 0);

    unsigned long tfclose_end = get_time_usec();
    MPI_Barrier(MPI_COMM_WORLD);
    unsigned long t4 = get_time_usec();

    long final_mean = 0, final_std = 0, std_1 = 0;
    final_mean = TOTAL_PARTICLES / NUM_RANKS;
    std_1      = pow(NUM_PARTICLES - final_mean, 2);
    MPI_Allreduce(&std_1, &final_std, 1, MPI_LONG_LONG, MPI_SUM, comm);

    //if (MY_RANK == 0) {
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
        printf("Total number of particles: %lldM\n", TOTAL_PARTICLES / (M_VAL));
        printf("Final mean particles: %ld \n", final_mean);
        printf("Final standard deviation: %f \n", sqrt(final_std / NUM_RANKS));
        unsigned long long total_sleep_time_us =
            read_time_val(params.compute_time, TIME_US) * (params.cnt_time_step - 1);
        printf("Total emulated compute time: %.3lf s\n", total_sleep_time_us / (1000.0 * 1000.0));

        //double total_size_bytes = NUM_RANKS * local_data_size;
	double total_size_bytes = local_data_size;
        value                   = format_human_readable(total_size_bytes);
        printf("Total write size: %.3lf %cB\n", value.value, value.unit);

        float rwt_s    = (float)raw_write_time / (1000.0 * 1000.0);
        float raw_rate = (float)total_size_bytes / rwt_s;
        printf("Raw write time: %.3f s\n", rwt_s);

        float meta_time_s = (float)inner_metadata_time / (1000.0 * 1000.0);
        printf("Metadata time: %.3f s\n", meta_time_s);

        float fcreate_time_s = (float)(tfopen_end - tfopen_start) / (1000.0 * 1000.0);
        printf("H5Fcreate() time: %.3f s\n", fcreate_time_s);

        float flush_time_s = (float)(tflush_end - tflush_start) / (1000.0 * 1000.0);
        printf("H5Fflush() time: %.3f s\n", flush_time_s);

        float fclose_time_s = (float)(tfclose_end - tfclose_start) / (1000.0 * 1000.0);
        printf("H5Fclose() time: %.3f s\n", fclose_time_s);

        float oct_s = (float)(t4 - t1) / (1000.0 * 1000.0);
        printf("Observed completion time: %.3f s\n", oct_s);

        value = format_human_readable(raw_rate);
        printf("%s Raw write rate: %.3f %cB/s \n", mode_str, value.value, value.unit);

        float or_bs = (float)total_size_bytes / ((float)(t4 - t1 - total_sleep_time_us) / (1000.0 * 1000.0));
        value       = format_human_readable(or_bs);
        printf("%s Observed write rate: %.3f %cB/s\n", mode_str, value.value, value.unit);

        printf("===========================================================\n");

        if (params.useCSV) {
            fprintf(params.csv_fs, "metric, value, unit\n");
            fprintf(params.csv_fs, "operation, %s, %s\n", "write_var_normal_dist", "");
            fprintf(params.csv_fs, "ranks, %d, %s\n", NUM_RANKS, "");
            fprintf(params.csv_fs, "Total number of particles, %lldM, %s\n", TOTAL_PARTICLES / (M_VAL), "");
            fprintf(params.csv_fs, "Final mean particles, %ld, %s\n", final_mean, "");
            fprintf(params.csv_fs, "Final standard deviation, %f, %s\n", sqrt(final_std / NUM_RANKS), "");
            fprintf(params.csv_fs, "collective data, %s, %s\n", params.data_coll == 1 ? "YES" : "NO", "");
            fprintf(params.csv_fs, "collective meta, %s, %s\n", params.meta_coll == 1 ? "YES" : "NO", "");
            fprintf(params.csv_fs, "subfiling, %s, %s\n", params.subfiling == 1 ? "YES" : "NO", "");
            fprintf(params.csv_fs, "total compute time, %.3lf, %s\n", total_sleep_time_us / (1000.0 * 1000.0),
                    "seconds");
            value = format_human_readable(total_size_bytes);
            fprintf(params.csv_fs, "total size, %.3lf, %cB\n", value.value, value.unit);
            fprintf(params.csv_fs, "raw time, %.3f, %s\n", rwt_s, "seconds");
            value = format_human_readable(raw_rate);
            fprintf(params.csv_fs, "raw rate, %.3lf, %cB/s\n", value.value, value.unit);
            fprintf(params.csv_fs, "metadata time, %.3f, %s\n", meta_time_s, "seconds");
            value = format_human_readable(or_bs);
            fprintf(params.csv_fs, "observed rate, %.3f, %cB/s\n", value.value, value.unit);
            fprintf(params.csv_fs, "observed time, %.3f, %s\n", oct_s, "seconds");
            fclose(params.csv_fs);
        }
	//}

    MPI_Finalize();
    return 0;
}
