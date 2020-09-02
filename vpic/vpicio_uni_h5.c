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
//		Each process writes a specified number of particles into
//		a hdf5 output file using only HDF5 calls
// Author:	Suren Byna <SByna@lbl.gov>
//		Lawrence Berkeley National Laboratory, Berkeley, CA
// Created:	in 2011
// Modified:	01/06/2014 --> Removed all H5Part calls and using HDF5 calls
//          	02/19/2019 --> Add option to write multiple timesteps of data - Tang
//


#include <hdf5.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <unistd.h>
#include <assert.h>
#include <string.h>
#include <sys/time.h>
#include "../commons/h5bench_util.h"

herr_t ierr;

// Global Variables and dimensions
long long NUM_PARTICLES = 0, FILE_OFFSET;	// 8  meg particles per process
long long TOTAL_PARTICLES;
int NUM_RANKS, MY_RANK, NUM_TIMESTEPS;
int X_DIM = 64;
int Y_DIM = 64;
int Z_DIM = 64;
hid_t PARTICLE_COMPOUND_TYPE;
hid_t PARTICLE_COMPOUND_TYPE_SEPARATES[8];
// Uniform random number
double uniform_random_number()
{
    //DEBUG_PRINT
    return (((double)rand())/((double)(RAND_MAX)));
}

// Initialize particle data
//void init_particles ()
//{
//    long i;
//    for (i=0; i<NUM_PARTICLES; i++)
//    {
//        id1[i] = i;
//        id2[i] = i*2;
//        x[i] = uniform_random_number()*x_dim;
//        y[i] = uniform_random_number()*y_dim;
//        z[i] = ((double)id1[i]/NUM_PARTICLES)*z_dim;
//        px[i] = uniform_random_number()*x_dim;
//        py[i] = uniform_random_number()*y_dim;
//        pz[i] = ((double)id2[i]/NUM_PARTICLES)*z_dim;
//    }
//}

// Create HDF5 file and write data
//void create_and_write_synthetic_h5_data(int rank, hid_t loc, hid_t *dset_ids, hid_t filespace, hid_t memspace, hid_t plist_id)
//{
//    int i;
//    // Note: printf statements are inserted basically
//    // to check the progress. Other than that they can be removed
//    dset_ids[0] = H5Dcreate(loc, "x", H5T_NATIVE_DOUBLE, filespace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
//    dset_ids[1] = H5Dcreate(loc, "y", H5T_NATIVE_DOUBLE, filespace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
//    dset_ids[2] = H5Dcreate(loc, "z", H5T_NATIVE_DOUBLE, filespace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
//    dset_ids[3] = H5Dcreate(loc, "id1", H5T_NATIVE_LONG, filespace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
//    dset_ids[4] = H5Dcreate(loc, "id2", H5T_NATIVE_LONG, filespace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
//    dset_ids[5] = H5Dcreate(loc, "px", H5T_NATIVE_DOUBLE, filespace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
//    dset_ids[6] = H5Dcreate(loc, "py", H5T_NATIVE_DOUBLE, filespace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
//    dset_ids[7] = H5Dcreate(loc, "pz", H5T_NATIVE_DOUBLE, filespace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
//
//    ierr = H5Dwrite(dset_ids[0], H5T_NATIVE_DOUBLE, memspace, filespace, plist_id, x);
//    /* if (rank == 0) printf ("Written variable 1 \n"); */
//
//    ierr = H5Dwrite(dset_ids[1], H5T_NATIVE_DOUBLE, memspace, filespace, plist_id, y);
//    /* if (rank == 0) printf ("Written variable 2 \n"); */
//
//    ierr = H5Dwrite(dset_ids[2], H5T_NATIVE_DOUBLE, memspace, filespace, plist_id, z);
//    /* if (rank == 0) printf ("Written variable 3 \n"); */
//
//    ierr = H5Dwrite(dset_ids[3], H5T_NATIVE_LONG, memspace, filespace, plist_id, id1);
//    /* if (rank == 0) printf ("Written variable 4 \n"); */
//
//    ierr = H5Dwrite(dset_ids[4], H5T_NATIVE_LONG, memspace, filespace, plist_id, id2);
//    /* if (rank == 0) printf ("Written variable 5 \n"); */
//
//    ierr = H5Dwrite(dset_ids[5], H5T_NATIVE_DOUBLE, memspace, filespace, plist_id, px);
//    /* if (rank == 0) printf ("Written variable 6 \n"); */
//
//    ierr = H5Dwrite(dset_ids[6], H5T_NATIVE_DOUBLE, memspace, filespace, plist_id, py);
//    /* if (rank == 0) printf ("Written variable 7 \n"); */
//
//    ierr = H5Dwrite(dset_ids[7], H5T_NATIVE_DOUBLE, memspace, filespace, plist_id, pz);
//    /* if (rank == 0) printf ("Written variable 8 \n"); */
//    if (rank == 0) printf ("  Finished written 8 variables \n");
//}

typedef enum Benchmark_mode{
    LINEAR_LINEAR,
    LINEAR_COMPOUND,
    COMPOUND_LINEAR,
    COMPOUND_COMPOUND,
    ARRAY_2D,
    ARRAY_3D
}bench_mode;

typedef struct Particle{
    double x, y, z;
    double px, py, pz;
    long id_1, id_2;
}particle;

typedef struct data_linear{
    long particle_cnt;
    double *x, *y, *z;
    double *px, *py, *pz;
    long *id_1, *id_2;
}data_linear;

typedef struct data_2d{
    long particle_cnt;
    long dim_1, dim_2;
    double **x, **y, **z;
    double **px, **py, **pz;
    long **id_1, **id_2;
}data_2d;

typedef struct data_3d{
    long particle_cnt;
    long dim_1, dim_2, dim_3;
    double ***x, ***y, ***z;
    double ***px, ***py, ***pz;
    long ***id_1, ***id_2;
}data_3d;

//need to close later.
hid_t make_compound_type(){
    PARTICLE_COMPOUND_TYPE = H5Tcreate(H5T_COMPOUND, sizeof(particle));
    H5Tinsert(PARTICLE_COMPOUND_TYPE, "x", HOFFSET(particle, x),   H5T_NATIVE_DOUBLE);
    H5Tinsert(PARTICLE_COMPOUND_TYPE, "y", HOFFSET(particle, y),   H5T_NATIVE_DOUBLE);
    H5Tinsert(PARTICLE_COMPOUND_TYPE, "z", HOFFSET(particle, z),   H5T_NATIVE_DOUBLE);
    H5Tinsert(PARTICLE_COMPOUND_TYPE, "px", HOFFSET(particle, px), H5T_NATIVE_DOUBLE);
    H5Tinsert(PARTICLE_COMPOUND_TYPE, "py", HOFFSET(particle, py), H5T_NATIVE_DOUBLE);
    H5Tinsert(PARTICLE_COMPOUND_TYPE, "pz", HOFFSET(particle, pz), H5T_NATIVE_DOUBLE);
    H5Tinsert(PARTICLE_COMPOUND_TYPE, "id_1", HOFFSET(particle, id_1), H5T_NATIVE_LONG);
    H5Tinsert(PARTICLE_COMPOUND_TYPE, "id_2", HOFFSET(particle, id_2), H5T_NATIVE_LONG);
    return PARTICLE_COMPOUND_TYPE;
}

hid_t* make_compound_type_separates(){
    PARTICLE_COMPOUND_TYPE_SEPARATES[0] = H5Tcreate(H5T_COMPOUND, sizeof(double));
    H5Tinsert(PARTICLE_COMPOUND_TYPE_SEPARATES[0], "x", 0, H5T_NATIVE_DOUBLE);

    PARTICLE_COMPOUND_TYPE_SEPARATES[1] = H5Tcreate(H5T_COMPOUND, sizeof(double));
    H5Tinsert(PARTICLE_COMPOUND_TYPE_SEPARATES[1], "y", 0, H5T_NATIVE_DOUBLE);

    PARTICLE_COMPOUND_TYPE_SEPARATES[2] = H5Tcreate(H5T_COMPOUND, sizeof(double));
    H5Tinsert(PARTICLE_COMPOUND_TYPE_SEPARATES[2], "z", 0, H5T_NATIVE_DOUBLE);

    PARTICLE_COMPOUND_TYPE_SEPARATES[3] = H5Tcreate(H5T_COMPOUND, sizeof(double));
    H5Tinsert(PARTICLE_COMPOUND_TYPE_SEPARATES[3], "px", 0, H5T_NATIVE_DOUBLE);

    PARTICLE_COMPOUND_TYPE_SEPARATES[4] = H5Tcreate(H5T_COMPOUND, sizeof(double));
    H5Tinsert(PARTICLE_COMPOUND_TYPE_SEPARATES[4], "py", 0, H5T_NATIVE_DOUBLE);

    PARTICLE_COMPOUND_TYPE_SEPARATES[5] = H5Tcreate(H5T_COMPOUND, sizeof(double));
    H5Tinsert(PARTICLE_COMPOUND_TYPE_SEPARATES[5], "pz", 0, H5T_NATIVE_DOUBLE);

    PARTICLE_COMPOUND_TYPE_SEPARATES[6] = H5Tcreate(H5T_COMPOUND, sizeof(long));
    H5Tinsert(PARTICLE_COMPOUND_TYPE_SEPARATES[6], "id_1", 0, H5T_NATIVE_LONG);

    PARTICLE_COMPOUND_TYPE_SEPARATES[7] = H5Tcreate(H5T_COMPOUND, sizeof(long));
    H5Tinsert(PARTICLE_COMPOUND_TYPE_SEPARATES[7], "id_2", 0, H5T_NATIVE_LONG);

    return PARTICLE_COMPOUND_TYPE_SEPARATES;
}

//returns prepared local data volume, used to calculate bandwidth
particle* prepare_data_compound(long particle_cnt, unsigned long *data_size_out) {
    particle *data_out = (particle*) malloc(particle_cnt * sizeof(particle));

    for (long i = 0; i < particle_cnt; i++) {
        data_out[i].id_1 = i;
        data_out[i].id_2 = 2 * i;
        data_out[i].x = uniform_random_number() * X_DIM;
        data_out[i].y = uniform_random_number() * Y_DIM;
        data_out[i].z = ((double) i / particle_cnt) * Z_DIM;
        data_out[i].px = uniform_random_number() * X_DIM;
        data_out[i].py = uniform_random_number() * Y_DIM;
        data_out[i].pz = ((double) 2 * i / particle_cnt) * Z_DIM;
    }
    //printf("sizeof(particle) = %lu, sizeof(double) = %lu, sizeof(long) = %lu\n", sizeof(particle), sizeof(double), sizeof(long));
    *data_size_out = particle_cnt * sizeof(particle);
    return data_out;
}

data_linear * prepare_data_linear(long particle_cnt, unsigned long * data_size_out) {

    data_linear *data_out = (data_linear*) malloc(sizeof(data_linear));

    data_out->particle_cnt = particle_cnt;

    data_out->x =  (double*) malloc(particle_cnt * sizeof(double));
    data_out->y =  (double*) malloc(particle_cnt * sizeof(double));
    data_out->z =  (double*) malloc(particle_cnt * sizeof(double));
    data_out->px = (double*) malloc(particle_cnt * sizeof(double));
    data_out->py = (double*) malloc(particle_cnt * sizeof(double));
    data_out->pz = (double*) malloc(particle_cnt * sizeof(double));
    data_out->id_1 = (long*) malloc(particle_cnt * sizeof(long));
    data_out->id_2 = (long*) malloc(particle_cnt * sizeof(long));

    for (long i = 0; i < particle_cnt; i++) {
        data_out->id_1[i] = i;
        data_out->id_2[i] = i * 2;
        data_out->x[i] =  uniform_random_number() * X_DIM;
        data_out->y[i] =  uniform_random_number() * Y_DIM;
        data_out->px[i] = uniform_random_number() * X_DIM;
        data_out->py[i] = uniform_random_number() * Y_DIM;
        data_out->z[i] =  ((double) data_out->id_1[i] / NUM_PARTICLES) * Z_DIM;
        data_out->pz[i] = ((double) data_out->id_2[i] / NUM_PARTICLES) * Z_DIM;
    }
    *data_size_out = particle_cnt * (6 * sizeof(double) + 2 * sizeof(long));

    return data_out;
}

data_2d* prepare_data_2D(long particle_cnt, long dim_1, long dim_2, unsigned long * data_size_out){
    assert(particle_cnt == dim_1 * dim_2);
    data_2d *data_out = (data_2d*) malloc(sizeof(data_2d));
    data_out->particle_cnt = particle_cnt;
    data_out->dim_1 = dim_1;
    data_out->dim_2 = dim_2;

    data_out->x =  (double**) malloc(dim_1 * sizeof(double*));
    data_out->y =  (double**) malloc(dim_1 * sizeof(double*));
    data_out->z =  (double**) malloc(dim_1 * sizeof(double*));
    data_out->px = (double**) malloc(dim_1 * sizeof(double*));
    data_out->py = (double**) malloc(dim_1 * sizeof(double*));
    data_out->pz = (double**) malloc(dim_1 * sizeof(double*));
    data_out->id_1 = (long**) malloc(dim_1 * sizeof(long*));
    data_out->id_2 = (long**) malloc(dim_1 * sizeof(long*));

    for(long i1 = 0; i1 < dim_1; i1++){
        data_out->x[i1] =  (double*) malloc(dim_2 * sizeof(double));
        data_out->y[i1] =  (double*) malloc(dim_2 * sizeof(double));
        data_out->z[i1] =  (double*) malloc(dim_2 * sizeof(double));
        data_out->px[i1] = (double*) malloc(dim_2 * sizeof(double));
        data_out->py[i1] = (double*) malloc(dim_2 * sizeof(double));
        data_out->pz[i1] = (double*) malloc(dim_2 * sizeof(double));
        data_out->id_1[i1] = (long*) malloc(dim_2 * sizeof(long));
        data_out->id_2[i1] = (long*) malloc(dim_2 * sizeof(long));
        for(long i2 = 0; i2 < dim_2; i2++){
            data_out->x[i1][i2] = uniform_random_number() * X_DIM;
            data_out->id_1[i1][i2] = i1;
            data_out->id_2[i1][i2] = i1 * 2;
            data_out->x[i1][i2] =  uniform_random_number() * X_DIM;
            data_out->y[i1][i2] =  uniform_random_number() * Y_DIM;
            data_out->px[i1][i2] = uniform_random_number() * X_DIM;
            data_out->py[i1][i2] = uniform_random_number() * Y_DIM;
            data_out->z[i1][i2] =  ((double) data_out->id_1[i1][i2] / NUM_PARTICLES) * Z_DIM;
            data_out->pz[i1][i2] = ((double) data_out->id_2[i1][i2] / NUM_PARTICLES) * Z_DIM;
        }
    }
    *data_size_out = dim_1 * dim_2 * (6 * sizeof(double) + 2 * sizeof(long));

    return data_out;
}

data_3d* prepare_data_3D(long particle_cnt, long dim_1, long dim_2, long dim_3, unsigned long * data_size_out){
    assert(particle_cnt == dim_1 * dim_2 * dim_3);
    data_3d *data_out = (data_3d*) malloc(sizeof(data_3d));
    data_out->particle_cnt = particle_cnt;
    data_out->dim_1 = dim_1;
    data_out->dim_2 = dim_2;
    data_out->dim_3 = dim_3;

    data_out->x =  (double***) malloc(dim_1 * sizeof(double**));
    data_out->y =  (double***) malloc(dim_1 * sizeof(double**));
    data_out->z =  (double***) malloc(dim_1 * sizeof(double**));
    data_out->px = (double***) malloc(dim_1 * sizeof(double**));
    data_out->py = (double***) malloc(dim_1 * sizeof(double**));
    data_out->pz = (double***) malloc(dim_1 * sizeof(double**));
    data_out->id_1 = (long***) malloc(dim_1 * sizeof(long**));
    data_out->id_2 = (long***) malloc(dim_1 * sizeof(long**));

    for(long i1 = 0; i1 < dim_1; i1++){
        data_out->x[i1] =  (double**) malloc(dim_2 * sizeof(double*));
        data_out->y[i1] =  (double**) malloc(dim_2 * sizeof(double*));
        data_out->z[i1] =  (double**) malloc(dim_2 * sizeof(double*));
        data_out->px[i1] = (double**) malloc(dim_2 * sizeof(double*));
        data_out->py[i1] = (double**) malloc(dim_2 * sizeof(double*));
        data_out->pz[i1] = (double**) malloc(dim_2 * sizeof(double*));
        data_out->id_1[i1] = (long**) malloc(dim_2 * sizeof(long*));
        data_out->id_2[i1] = (long**) malloc(dim_2 * sizeof(long*));

        for(long i2 = 0; i2 < dim_2; i2++){
            data_out->x[i1][i2] =  (double*) malloc(dim_3 * sizeof(double));
            data_out->y[i1][i2] =  (double*) malloc(dim_3 * sizeof(double));
            data_out->z[i1][i2] =  (double*) malloc(dim_3 * sizeof(double));
            data_out->px[i1][i2] = (double*) malloc(dim_3 * sizeof(double));
            data_out->py[i1][i2] = (double*) malloc(dim_3 * sizeof(double));
            data_out->pz[i1][i2] = (double*) malloc(dim_3 * sizeof(double));
            data_out->id_1[i1][i2] = (long*) malloc(dim_3 * sizeof(long));
            data_out->id_2[i1][i2] = (long*) malloc(dim_3 * sizeof(long));

            for(long i3 = 0; i3 < dim_3; i3++){
                data_out->x[i1][i2][i3] =  uniform_random_number() * X_DIM;
                data_out->y[i1][i2][i3] =  uniform_random_number() * Y_DIM;
                data_out->px[i1][i2][i3] = uniform_random_number() * X_DIM;
                data_out->py[i1][i2][i3] = uniform_random_number() * Y_DIM;
                data_out->id_1[i1][i2][i3] = i1;
                data_out->id_2[i1][i2][i3] = i1 * 2;
                data_out->z[i1][i2][i3] =  ((double) data_out->id_1[i1][i2][i3] / NUM_PARTICLES) * Z_DIM;
                data_out->pz[i1][i2][i3] = ((double) data_out->id_2[i1][i2][i3] / NUM_PARTICLES) * Z_DIM;
            }
        }
    }
    *data_size_out = dim_1 * dim_2 * dim_3 * (6 * sizeof(double) + 2 * sizeof(long));

    return data_out;
}

void data_free(bench_mode mode, void* data){
    assert(data);
    switch(mode){
        case LINEAR_LINEAR:
        case LINEAR_COMPOUND:
            free(((data_linear*)data)->x);
            free(((data_linear*)data)->y);
            free(((data_linear*)data)->z);
            free(((data_linear*)data)->px);
            free(((data_linear*)data)->py);
            free(((data_linear*)data)->pz);
            free(((data_linear*)data)->id_1);
            free(((data_linear*)data)->id_2);
            free(((data_linear*)data));
            break;

        case COMPOUND_LINEAR:

        case COMPOUND_COMPOUND:
            free(data);
            break;

        case ARRAY_2D: {
            data_2d* d2 = (data_2d*)data;
            assert(d2->particle_cnt = d2->dim_1 * d2->dim_2);
            for(long i1 = 0; i1 < d2->dim_1; i1++){
                free(d2->x[i1]);
                free(d2->y[i1]);
                free(d2->z[i1]);
                free(d2->px[i1]);
                free(d2->py[i1]);
                free(d2->pz[i1]);
                free(d2->id_1[i1]);
                free(d2->id_2[i1]);
            }
            free(d2->x);
            free(d2->y);
            free(d2->z);
            free(d2->px);
            free(d2->py);
            free(d2->pz);
            free(d2->id_1);
            free(d2->id_2);
            break;
        }

        case ARRAY_3D: {
            data_3d* d3 = (data_3d*)data;
            assert(d3->particle_cnt = d3->dim_1 * d3->dim_2 * d3->dim_3);
            for(long i1 = 0; i1 < d3->dim_1; i1++){
                for(long i2 = 0; i2 < d3->dim_2; i2++){
                    free(d3->x[i1][i2]);
                    free(d3->y[i1][i2]);
                    free(d3->z[i1][i2]);
                    free(d3->px[i1][i2]);
                    free(d3->py[i1][i2]);
                    free(d3->pz[i1][i2]);
                    free(d3->id_1[i1][i2]);
                    free(d3->id_2[i1][i2]);
                }

                free(d3->x[i1]);
                free(d3->y[i1]);
                free(d3->z[i1]);
                free(d3->px[i1]);
                free(d3->py[i1]);
                free(d3->pz[i1]);
                free(d3->id_1[i1]);
                free(d3->id_2[i1]);
            }

            free(d3->x);
            free(d3->y);
            free(d3->z);
            free(d3->px);
            free(d3->py);
            free(d3->pz);
            free(d3->id_1);
            free(d3->id_2);
            break;
        }

        default:
            break;
    }
}

int set_select_spaces_default(hid_t* filespace_out, hid_t* memspace_out, hid_t* plist_id_out){
    *filespace_out = H5Screate_simple(1, (hsize_t *) &TOTAL_PARTICLES, NULL);//= world_size * numparticles
    *memspace_out =  H5Screate_simple(1, (hsize_t *) &NUM_PARTICLES, NULL);
    *plist_id_out = H5Pcreate(H5P_DATASET_XFER);
    H5Pset_dxpl_mpio(*plist_id_out, H5FD_MPIO_COLLECTIVE);
    H5Sselect_hyperslab(*filespace_out, H5S_SELECT_SET, (hsize_t *) &FILE_OFFSET, NULL, (hsize_t *) &NUM_PARTICLES, NULL);
    return 0;
}

int set_select_space_multi_2D_array(hid_t* filespace_out, hid_t* memspace_out, hid_t* plist_id_out,
        unsigned long long dim_1, unsigned long long dim_2){//dim_1 * dim_2 === NUM_PARTICLES
    hsize_t mem_dims[2];
    hsize_t file_dims[2];
    mem_dims[0] = (hsize_t)dim_1;
    mem_dims[1] = (hsize_t)dim_2;
    file_dims[0] = (hsize_t)dim_1 * NUM_RANKS; //NUM_TIMESTEPS no use?
    file_dims[1] = (hsize_t)dim_2;
    hsize_t file_starts[2], count[2];//select start point and range in each dimension.
    file_starts[0] = dim_1 * (MY_RANK);//NUM_TIMESTEPS
    file_starts[1] = 0;
    count[0] = dim_1;
    count[1] = dim_2;
    DEBUG_PRINT
    *filespace_out = H5Screate_simple(2, file_dims, NULL); //(1, (hsize_t *) &TOTAL_PARTICLES, NULL);//= world_size * numparticles
    *memspace_out =  H5Screate_simple(2, mem_dims, NULL);
    printf("%llu * %llu 2D array, my x_start = %llu, y_start = %llu, x_cnt = %llu, y_cnt = %llu\n",
            dim_1, dim_2, file_starts[0], file_starts[1], count[0], count[1]);
    *plist_id_out = H5Pcreate(H5P_DATASET_XFER);
    H5Pset_dxpl_mpio(*plist_id_out, H5FD_MPIO_COLLECTIVE);
    H5Sselect_hyperslab(*filespace_out, H5S_SELECT_SET, file_starts, NULL, count, NULL);
    return 0;
}

int set_select_space_multi_3D_array(hid_t* filespace_out, hid_t* memspace_out, hid_t* plist_id_out,
        unsigned long long dim_1, unsigned long long dim_2, unsigned long long dim_3){
    hsize_t mem_dims[3];
    hsize_t file_dims[3];
    mem_dims[0] = (hsize_t)dim_1;
    mem_dims[1] = (hsize_t)dim_2;
    mem_dims[2] = (hsize_t)dim_3;
    file_dims[0] = (hsize_t)dim_1 * NUM_RANKS * NUM_TIMESTEPS;
    file_dims[1] = (hsize_t)dim_2;
    file_dims[2] = (hsize_t)dim_3;

    hsize_t file_starts[3], file_range[3];//select start point and range in each dimension.
    file_starts[0] = dim_1 * (MY_RANK);
    file_starts[1] = 0;
    file_starts[2] = 0;
    file_range[0] = dim_1;
    file_range[1] = dim_2;
    file_range[2] = dim_3;

    DEBUG_PRINT
    *filespace_out = H5Screate_simple(3, file_dims, NULL); //(1, (hsize_t *) &TOTAL_PARTICLES, NULL);//= world_size * numparticles
    *memspace_out =  H5Screate_simple(3, mem_dims, NULL);

    *plist_id_out = H5Pcreate(H5P_DATASET_XFER);
    H5Pset_dxpl_mpio(*plist_id_out, H5FD_MPIO_COLLECTIVE);
    H5Sselect_hyperslab(*filespace_out, H5S_SELECT_SET, file_starts, NULL, file_range, NULL);
    return 0;
}

void data_write_2D_array(hid_t loc, hid_t *dset_ids, hid_t filespace, hid_t memspace, hid_t plist_id,
        data_2d* data_in){
    //write file:
    //      - create 2D array as the dateset type, now linear-linear is 8 datasets of 1D array
    //      - need to have 8 datasets, each is a 2D array of double/long, 3D case is similar.
    assert(data_in && data_in->x);
    DEBUG_PRINT
    dset_ids[0] = H5Dcreate(loc, "x", H5T_NATIVE_DOUBLE, memspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    dset_ids[1] = H5Dcreate(loc, "y", H5T_NATIVE_DOUBLE, memspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    dset_ids[2] = H5Dcreate(loc, "z", H5T_NATIVE_DOUBLE, memspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    dset_ids[3] = H5Dcreate(loc, "px", H5T_NATIVE_DOUBLE, memspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    dset_ids[4] = H5Dcreate(loc, "py", H5T_NATIVE_DOUBLE, memspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    dset_ids[5] = H5Dcreate(loc, "pz", H5T_NATIVE_DOUBLE, memspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

    dset_ids[6] = H5Dcreate(loc, "id_1", H5T_NATIVE_LONG, memspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    dset_ids[7] = H5Dcreate(loc, "id_2", H5T_NATIVE_LONG, memspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    DEBUG_PRINT
    ierr = H5Dwrite(dset_ids[0], H5T_NATIVE_DOUBLE, memspace, filespace, plist_id, data_in->x);
    /* if (rank == 0) printf ("Written variable 1 \n"); */
    DEBUG_PRINT
    ierr = H5Dwrite(dset_ids[1], H5T_NATIVE_DOUBLE, memspace, filespace, plist_id, data_in->y);
    /* if (rank == 0) printf ("Written variable 2 \n"); */
    DEBUG_PRINT
    ierr = H5Dwrite(dset_ids[2], H5T_NATIVE_DOUBLE, memspace, filespace, plist_id, data_in->z);
    /* if (rank == 0) printf ("Written variable 3 \n"); */
    DEBUG_PRINT
    ierr = H5Dwrite(dset_ids[3], H5T_NATIVE_DOUBLE, memspace, filespace, plist_id, data_in->px);
    /* if (rank == 0) printf ("Written variable 4 \n"); */

    ierr = H5Dwrite(dset_ids[4], H5T_NATIVE_DOUBLE, memspace, filespace, plist_id, data_in->py);
    /* if (rank == 0) printf ("Written variable 5 \n"); */

    ierr = H5Dwrite(dset_ids[5], H5T_NATIVE_DOUBLE, memspace, filespace, plist_id, data_in->pz);
    /* if (rank == 0) printf ("Written variable 6 \n"); */

    ierr = H5Dwrite(dset_ids[6], H5T_NATIVE_LONG, memspace, filespace, plist_id, data_in->id_1);
    /* if (rank == 0) printf ("Written variable 7 \n"); */

    ierr = H5Dwrite(dset_ids[7], H5T_NATIVE_LONG, memspace, filespace, plist_id, data_in->id_2);
    /* if (rank == 0) printf ("Written variable 8 \n"); */

    if (MY_RANK == 0) printf ("    %s: Finished writing time step \n", __func__);
}

void data_write_3D_array(hid_t loc, hid_t *dset_ids, hid_t filespace, hid_t memspace, hid_t plist_id,
        data_3d* data_in){
    assert(data_in && data_in->x);
}

void data_write_linear_to_linear(hid_t loc, hid_t *dset_ids, hid_t filespace, hid_t memspace, hid_t plist_id,
        data_linear* data_in) {

    assert(data_in && data_in->x);

    dset_ids[0] = H5Dcreate(loc, "x", H5T_NATIVE_DOUBLE, filespace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    dset_ids[1] = H5Dcreate(loc, "y", H5T_NATIVE_DOUBLE, filespace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    dset_ids[2] = H5Dcreate(loc, "z", H5T_NATIVE_DOUBLE, filespace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    dset_ids[3] = H5Dcreate(loc, "px", H5T_NATIVE_DOUBLE, filespace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    dset_ids[4] = H5Dcreate(loc, "py", H5T_NATIVE_DOUBLE, filespace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    dset_ids[5] = H5Dcreate(loc, "pz", H5T_NATIVE_DOUBLE, filespace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

    dset_ids[6] = H5Dcreate(loc, "id_1", H5T_NATIVE_LONG, filespace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    dset_ids[7] = H5Dcreate(loc, "id_2", H5T_NATIVE_LONG, filespace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

    ierr = H5Dwrite(dset_ids[0], H5T_NATIVE_DOUBLE, memspace, filespace, plist_id, data_in->x);
    /* if (rank == 0) printf ("Written variable 1 \n"); */

    ierr = H5Dwrite(dset_ids[1], H5T_NATIVE_DOUBLE, memspace, filespace, plist_id, data_in->y);
    /* if (rank == 0) printf ("Written variable 2 \n"); */

    ierr = H5Dwrite(dset_ids[2], H5T_NATIVE_DOUBLE, memspace, filespace, plist_id, data_in->z);
    /* if (rank == 0) printf ("Written variable 3 \n"); */

    ierr = H5Dwrite(dset_ids[3], H5T_NATIVE_DOUBLE, memspace, filespace, plist_id, data_in->px);
    /* if (rank == 0) printf ("Written variable 4 \n"); */

    ierr = H5Dwrite(dset_ids[4], H5T_NATIVE_DOUBLE, memspace, filespace, plist_id, data_in->py);
    /* if (rank == 0) printf ("Written variable 5 \n"); */

    ierr = H5Dwrite(dset_ids[5], H5T_NATIVE_DOUBLE, memspace, filespace, plist_id, data_in->pz);
    /* if (rank == 0) printf ("Written variable 6 \n"); */

    ierr = H5Dwrite(dset_ids[6], H5T_NATIVE_LONG, memspace, filespace, plist_id, data_in->id_1);
    /* if (rank == 0) printf ("Written variable 7 \n"); */

    ierr = H5Dwrite(dset_ids[7], H5T_NATIVE_LONG, memspace, filespace, plist_id, data_in->id_2);
    /* if (rank == 0) printf ("Written variable 8 \n"); */

    if (MY_RANK == 0) printf ("    %s: Finished writing time step \n", __func__);
}

void data_write_linear_to_compound(hid_t loc, hid_t *dset_ids, hid_t filespace, hid_t memspace, hid_t plist_id,
        data_linear* data_in){
    dset_ids[0] = H5Dcreate(loc, "particles", PARTICLE_COMPOUND_TYPE, filespace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

    ierr = H5Dwrite(dset_ids[0], PARTICLE_COMPOUND_TYPE_SEPARATES[0], memspace, filespace, plist_id, data_in->x);
    ierr = H5Dwrite(dset_ids[0], PARTICLE_COMPOUND_TYPE_SEPARATES[1], memspace, filespace, plist_id, data_in->y);
    ierr = H5Dwrite(dset_ids[0], PARTICLE_COMPOUND_TYPE_SEPARATES[2], memspace, filespace, plist_id, data_in->z);
    ierr = H5Dwrite(dset_ids[0], PARTICLE_COMPOUND_TYPE_SEPARATES[3], memspace, filespace, plist_id, data_in->px);
    ierr = H5Dwrite(dset_ids[0], PARTICLE_COMPOUND_TYPE_SEPARATES[4], memspace, filespace, plist_id, data_in->py);
    ierr = H5Dwrite(dset_ids[0], PARTICLE_COMPOUND_TYPE_SEPARATES[5], memspace, filespace, plist_id, data_in->pz);
    ierr = H5Dwrite(dset_ids[0], PARTICLE_COMPOUND_TYPE_SEPARATES[6], memspace, filespace, plist_id, data_in->id_1);
    ierr = H5Dwrite(dset_ids[0], PARTICLE_COMPOUND_TYPE_SEPARATES[7], memspace, filespace, plist_id, data_in->id_2);

    if (MY_RANK == 0) printf ("    %s: Finished writing time step \n", __func__);
}

void data_write_comp_to_linear(hid_t loc, hid_t *dset_ids, hid_t filespace, hid_t memspace, hid_t plist_id,
        particle* data_in) {
    assert(data_in && data_in->x);

    dset_ids[0] = H5Dcreate(loc, "x",   PARTICLE_COMPOUND_TYPE_SEPARATES[0], filespace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    dset_ids[1] = H5Dcreate(loc, "y",   PARTICLE_COMPOUND_TYPE_SEPARATES[1], filespace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    dset_ids[2] = H5Dcreate(loc, "z",   PARTICLE_COMPOUND_TYPE_SEPARATES[2], filespace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    dset_ids[3] = H5Dcreate(loc, "px",  PARTICLE_COMPOUND_TYPE_SEPARATES[3], filespace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    dset_ids[4] = H5Dcreate(loc, "py",  PARTICLE_COMPOUND_TYPE_SEPARATES[4], filespace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    dset_ids[5] = H5Dcreate(loc, "pz",  PARTICLE_COMPOUND_TYPE_SEPARATES[5], filespace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    dset_ids[6] = H5Dcreate(loc, "id_1", PARTICLE_COMPOUND_TYPE_SEPARATES[6], filespace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    dset_ids[7] = H5Dcreate(loc, "id_2", PARTICLE_COMPOUND_TYPE_SEPARATES[7], filespace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);


    ierr = H5Dwrite(dset_ids[0], PARTICLE_COMPOUND_TYPE, memspace, filespace, plist_id, data_in);
    /* if (rank == 0) printf ("Written variable 1 \n"); */

    ierr = H5Dwrite(dset_ids[1], PARTICLE_COMPOUND_TYPE, memspace, filespace, plist_id, data_in);
    /* if (rank == 0) printf ("Written variable 2 \n"); */

    ierr = H5Dwrite(dset_ids[2], PARTICLE_COMPOUND_TYPE, memspace, filespace, plist_id, data_in);
    /* if (rank == 0) printf ("Written variable 3 \n"); */

    ierr = H5Dwrite(dset_ids[3], PARTICLE_COMPOUND_TYPE, memspace, filespace, plist_id, data_in);
    /* if (rank == 0) printf ("Written variable 6 \n"); */

    ierr = H5Dwrite(dset_ids[4], PARTICLE_COMPOUND_TYPE, memspace, filespace, plist_id, data_in);
    /* if (rank == 0) printf ("Written variable 7 \n"); */

    ierr = H5Dwrite(dset_ids[5], PARTICLE_COMPOUND_TYPE, memspace, filespace, plist_id, data_in);
    /* if (rank == 0) printf ("Written variable 8 \n"); */

    ierr = H5Dwrite(dset_ids[6], PARTICLE_COMPOUND_TYPE, memspace, filespace, plist_id, data_in);
    /* if (rank == 0) printf ("Written variable 4 \n"); */

    ierr = H5Dwrite(dset_ids[7], PARTICLE_COMPOUND_TYPE, memspace, filespace, plist_id, data_in);
    /* if (rank == 0) printf ("Written variable 5 \n"); */

    if (MY_RANK == 0) printf ("    %s: Finished writing time step \n", __func__);
}

void data_write_comp_to_comp(hid_t loc, hid_t *dset_ids, hid_t filespace, hid_t memspace, hid_t plist_id,
        particle* data_in) {
    assert(data_in && data_in->x);

    dset_ids[0] = H5Dcreate(loc, "particles", PARTICLE_COMPOUND_TYPE, filespace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    ierr = H5Dwrite(dset_ids[0], PARTICLE_COMPOUND_TYPE, memspace, filespace, plist_id, data_in);//should write all things in data_in

    if (MY_RANK == 0) printf ("    %s: Finished writing time step \n", __func__);
}

int _run_time_steps(bench_mode mode, long particle_cnt, int timestep_cnt, int sleep_time,
        hid_t file_id, hid_t plist_id, hid_t filespace, hid_t memspace,
        unsigned long* total_data_size_out, unsigned long* raw_write_time_out) {

    char grp_name[128];
    unsigned long  rt_start, rt_end;
    int grp_cnt = 0, dset_cnt = 0;
    hid_t **dset_ids = (hid_t**)calloc(timestep_cnt, sizeof(hid_t*));
    hid_t *grp_ids  = (hid_t*)calloc(timestep_cnt, sizeof(hid_t));
    *raw_write_time_out = 0;
    void* data = NULL;
    unsigned long data_size;

    make_compound_type_separates();
    make_compound_type();

    switch(mode){
        case LINEAR_LINEAR:
            data = (void*)prepare_data_linear(particle_cnt, &data_size);
            dset_cnt = 8;
            break;

        case LINEAR_COMPOUND:
            data = (void*)prepare_data_linear(particle_cnt, &data_size);
            dset_cnt = 1;
            break;

        case COMPOUND_LINEAR:
            data = (void*)prepare_data_compound(particle_cnt, &data_size);
            dset_cnt = 8;
            break;

        case COMPOUND_COMPOUND:
            data = (void*)prepare_data_compound(particle_cnt, &data_size);
            dset_cnt = 1;
            break;

        case ARRAY_2D:
            data = (void*)prepare_data_2D(particle_cnt, 64, particle_cnt/64, &data_size);
            dset_cnt = 8;
            break;
        case ARRAY_3D:
            data = (void*)prepare_data_3D(particle_cnt, 64, 64, particle_cnt/4096, &data_size);
            dset_cnt = 8;
            break;
        default:
            break;
    }

    for (int i = 0; i < timestep_cnt; i++) {
        sprintf(grp_name, "Timestep_%d", i);

        grp_ids[i] = H5Gcreate2(file_id, grp_name, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

        if (MY_RANK == 0)
            printf ("Writing %s ... \n", grp_name);

        dset_ids[i] = (hid_t*)calloc(8, sizeof(hid_t));

        rt_start = get_time_usec();

        switch(mode){
            case LINEAR_LINEAR:
                data_write_linear_to_linear(grp_ids[i], dset_ids[i], filespace, memspace, plist_id, (data_linear*)data);
                break;

            case LINEAR_COMPOUND:
                //assert(0 && "LINEAR_COMPOUND is not implemented yet ");
                data_write_linear_to_compound(grp_ids[i], dset_ids[i], filespace, memspace, plist_id, (data_linear*)data);
                break;

            case COMPOUND_LINEAR:
                data_write_comp_to_linear(grp_ids[i], dset_ids[i], filespace, memspace, plist_id, (particle*)data);
                break;

            case COMPOUND_COMPOUND:
                data_write_comp_to_comp(grp_ids[i], dset_ids[i], filespace, memspace, plist_id, (particle*)data);
                break;

            case ARRAY_2D:
                data_write_2D_array(grp_ids[i], dset_ids[i], filespace, memspace, plist_id, (data_2d*)data);
                break;

            case ARRAY_3D:
                data_write_3D_array(grp_ids[i], dset_ids[i], filespace, memspace, plist_id, (data_3d*)data);
                break;

            default:
                break;
        }
        rt_end = get_time_usec();

        *raw_write_time_out += (rt_end - rt_start);

        fflush(stdout);
        if (i != timestep_cnt - 1) {
            if (MY_RANK == 0) printf ("  sleep for %ds\n", sleep_time);
            if (sleep_time > 0) sleep(sleep_time);
        }

        for (int j = 0; j < dset_cnt; j++)
            H5Dclose(dset_ids[i][j]);
        H5Gclose(grp_ids[i]);

        MPI_Barrier (MPI_COMM_WORLD);

        free(dset_ids[i]);
    }

    H5Tclose(PARTICLE_COMPOUND_TYPE);
    for(int i = 0; i < 8; i++)
        H5Tclose(PARTICLE_COMPOUND_TYPE_SEPARATES[i]);

    *total_data_size_out = timestep_cnt * data_size;

    data_free(mode, data);
    return 0;
}

hid_t set_fapl(){
    hid_t fapl = H5Pcreate(H5P_FILE_ACCESS);
    return fapl;
}

hid_t set_metadata(hid_t fapl){
    // Alignmemt
    int alignment = 16777216;
    //H5Pset_alignment(fapl, alignment, alignment),

    // Collective metadata
    H5Pset_all_coll_metadata_ops(fapl, 1);
    H5Pset_coll_metadata_write(fapl, 1);

    // Defer metadata flush
    H5AC_cache_config_t cache_config;
    cache_config.version = H5AC__CURR_CACHE_CONFIG_VERSION;
    H5Pget_mdc_config(fapl, &cache_config);
    cache_config.set_initial_size = 1;
    cache_config.initial_size = 16 * 1024 * 1024;
    cache_config.evictions_enabled = 0;
    cache_config.incr_mode = H5C_incr__off;
    cache_config.flash_incr_mode = H5C_flash_incr__off;
    cache_config.decr_mode = H5C_decr__off;
    H5Pset_mdc_config (fapl, &cache_config);

    return fapl;
}

void print_usage(char *name) {
    printf("Usage: %s /path_to_file num_time_steps num_sleep_sec num_M_particles LL/LC/CL/CC/2D/3D \n", name);
    printf("LL/LC/CL/CC is used to set benchmark mode, stands for LINEAR_LINEAR, LINEAR_COMPOUND, COMPOUND_LINEAR, COMPOUND_COMPOUND, 2D Array and 3D Array");
}

int main(int argc, char* argv[]) {
    char *file_name = argv[1];

    MPI_Init(&argc, &argv);
    int my_rank, num_procs, nts, i, j, sleep_time = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    MY_RANK = my_rank;
    NUM_RANKS = num_procs;
    MPI_Comm comm = MPI_COMM_WORLD;
    MPI_Info info = MPI_INFO_NULL;

    if (argc != 6) {
        print_usage(argv[0]);
        return 0;
    }
    DEBUG_PRINT
    nts = atoi(argv[2]);
    NUM_TIMESTEPS = nts;
    if (nts <= 0) {
        print_usage(argv[0]);
        return 0;
    }

    sleep_time = atoi(argv[3]);
    if (sleep_time < 0) {
        print_usage(argv[0]);
        return 0;
    }

    NUM_PARTICLES = (atoi(argv[4])) * 1024 * 1024;

    char* mode_str = argv[5];
    bench_mode mode;
    if (strcmp(mode_str, "LL") == 0) {
        mode = LINEAR_LINEAR;
    } else if (strcmp(mode_str, "LC") == 0) {
        mode = LINEAR_COMPOUND;
    } else if (strcmp(mode_str, "CC") == 0) {
        mode = COMPOUND_COMPOUND;
    } else if (strcmp(mode_str, "CL") == 0) {
        mode = COMPOUND_LINEAR;
    } else if(strcmp(mode_str, "2D") == 0){
        mode = ARRAY_2D;
    } else if(strcmp(mode_str, "3D") == 0){
        mode = ARRAY_3D;
    } else {
        printf("Benchmark mode can only be one of these: LL, LC, CL, CC \n");
        return 0;
    }

    if (my_rank == 0) {
        printf("Number of paritcles: %lld \n", NUM_PARTICLES);
    }

    unsigned long total_write_size = num_procs * nts * NUM_PARTICLES * (6 * sizeof(double) + 2 * sizeof(long));

    //init_particles ();

    if (my_rank == 0)
        printf("Finished initializeing particles \n");

    MPI_Barrier(MPI_COMM_WORLD);

    unsigned long t0 = get_time_usec();
    MPI_Allreduce(&NUM_PARTICLES, &TOTAL_PARTICLES, 1, MPI_LONG_LONG, MPI_SUM, comm);
    MPI_Scan(&NUM_PARTICLES, &FILE_OFFSET, 1, MPI_LONG_LONG, MPI_SUM, comm);
    FILE_OFFSET -= NUM_PARTICLES;

    if (my_rank == 0)
        printf("Total particle number = %lldM\n", TOTAL_PARTICLES / (1024 * 1024));

    hid_t fapl = set_fapl();

    H5Pset_fapl_mpio(fapl, comm, info);

    set_metadata(fapl);

    unsigned long t1 = get_time_usec(); // t1 - t0: cost of settings
    hid_t file_id = H5Fcreate(file_name, H5F_ACC_TRUNC, H5P_DEFAULT, fapl);
    H5Pclose(fapl);

    if (my_rank == 0)
        printf("Opened HDF5 file... \n");

    hid_t filespace, memspace, plist_id;

    if(mode == ARRAY_2D){
        set_select_space_multi_2D_array(&filespace, &memspace, &plist_id, 64, NUM_PARTICLES/64);
    } else if(mode == ARRAY_3D){
        set_select_space_multi_3D_array(&filespace, &memspace, &plist_id, 64, 64, NUM_PARTICLES/4096);
    } else
        set_select_spaces_default(&filespace, &memspace, &plist_id);

    MPI_Barrier(MPI_COMM_WORLD);

    unsigned long t2 = get_time_usec(); // t2 - t1: metadata: creating/opening

    unsigned long raw_write_time, total_data_size;
    _run_time_steps(mode, NUM_PARTICLES, nts, sleep_time, file_id, plist_id, filespace, memspace,
            &total_data_size, &raw_write_time);

    unsigned long t3 = get_time_usec(); // t3 - t2: writting data, including metadata

    if (my_rank == 0) {
        printf("\nTiming results with %d ranks\n", num_procs);
        printf("Total running time = %lu ms\n", (t3 - t0) / 1000);
    }

    H5Sclose(memspace);
    H5Sclose(filespace);
    H5Pclose(plist_id);
    H5Fclose(file_id);

    MPI_Barrier(MPI_COMM_WORLD);
    unsigned long t4 = get_time_usec();

    if (my_rank == 0) {
        printf("\nTiming results\n");
        printf("Total sleep time %ds, total write size = %lu MB\n", sleep_time * (nts - 1), num_procs * total_data_size/(1024*1024));
        printf("RR: Raw write time = %lu ms, RR = %lu MB/sec \n", raw_write_time / 1000,
                total_data_size / raw_write_time);
        printf("Core metadata time = %lu ms\n",
                (t3 - t2 - raw_write_time - sleep_time * (nts - 1) * 1000 * 1000) / 1000);
        printf("Opening + closing time = %lu ms\n", (t1 - t0 + t4 - t3) / 1000);
        printf("OR (observed rate):  = %lu ms, OR = %lu MB/sec\n", (t4 - t1) / 1000 - (nts - 1) * 1000,
                total_data_size / (t4 - t1 - (nts - 1) * 1000 * 1000));
        printf("OCT(observed completion time) = %lu ms\n", (t4 - t0) / 1000);
        printf("\n");
    }

    MPI_Finalize();

    return 0;
}
