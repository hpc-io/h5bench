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
#include "../commons/async_adaptor.h"
#define DIM_MAX 3

herr_t ierr;
typedef struct compress_info{
    int USE_COMPRESS;
    hid_t dcpl_id;
    unsigned long chunk_dims[DIM_MAX];
}compress_info;
// Global Variables and dimensions

compress_info COMPRESS_INFO;  //Using parallel compressing: need to set chunk dimensions for dcpl.
long long NUM_PARTICLES = 0, FILE_OFFSET;	// 8  meg particles per process
long long TOTAL_PARTICLES;
int NUM_RANKS, MY_RANK, NUM_TIMESTEPS;
int X_DIM = 64;
int Y_DIM = 64;
int Z_DIM = 64;

//Factors for filling data.
const int X_RAND = 191;
const int Y_RAND = 1009;
const int Z_RAND = 3701;

hid_t PARTICLE_COMPOUND_TYPE;
hid_t PARTICLE_COMPOUND_TYPE_SEPARATES[8];

//Optimization globals
int ALIGN = 1;
unsigned long ALIGN_THRESHOLD = 16777216;
unsigned long ALIGN_LEN = 16777216;
int COLL_METADATA = 1;
int DEFER_METADATA_FLUSH = 1;

typedef struct Particle{
    float x, y, z;
    float px, py, pz;
    int id_1;
    float id_2;
}particle;

float uniform_random_number(){
    return (((float)rand())/((float)(RAND_MAX)));
}

hid_t make_compound_type(){
    PARTICLE_COMPOUND_TYPE = H5Tcreate(H5T_COMPOUND, sizeof(particle));
    H5Tinsert(PARTICLE_COMPOUND_TYPE, "x", HOFFSET(particle, x),   H5T_NATIVE_FLOAT);
    H5Tinsert(PARTICLE_COMPOUND_TYPE, "y", HOFFSET(particle, y),   H5T_NATIVE_FLOAT);
    H5Tinsert(PARTICLE_COMPOUND_TYPE, "z", HOFFSET(particle, z),   H5T_NATIVE_FLOAT);
    H5Tinsert(PARTICLE_COMPOUND_TYPE, "px", HOFFSET(particle, px), H5T_NATIVE_FLOAT);
    H5Tinsert(PARTICLE_COMPOUND_TYPE, "py", HOFFSET(particle, py), H5T_NATIVE_FLOAT);
    H5Tinsert(PARTICLE_COMPOUND_TYPE, "pz", HOFFSET(particle, pz), H5T_NATIVE_FLOAT);
    H5Tinsert(PARTICLE_COMPOUND_TYPE, "id_1", HOFFSET(particle, id_1), H5T_NATIVE_INT);
    H5Tinsert(PARTICLE_COMPOUND_TYPE, "id_2", HOFFSET(particle, id_2), H5T_NATIVE_FLOAT);
    return PARTICLE_COMPOUND_TYPE;
}

hid_t* make_compound_type_separates(){
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

//returns prepared local data volume, used to calculate bandwidth
particle* prepare_data_interleaved(long particle_cnt, unsigned long *data_size_out) {
    particle *data_out = (particle*) malloc(particle_cnt * sizeof(particle));

    for (long i = 0; i < particle_cnt; i++) {
        data_out[i].id_1 = i;
        data_out[i].id_2 = (float)(2 * i);
        data_out[i].x = uniform_random_number() * X_DIM;
        data_out[i].y = uniform_random_number() * Y_DIM;
        data_out[i].z = ((float) i / particle_cnt) * Z_DIM;
        data_out[i].px = uniform_random_number() * X_DIM;
        data_out[i].py = uniform_random_number() * Y_DIM;
        data_out[i].pz = ((float) 2 * i / particle_cnt) * Z_DIM;
    }
    *data_size_out = particle_cnt * sizeof(particle);
    return data_out;
}

data_contig_md * prepare_data_contig_1D(long particle_cnt, unsigned long * data_size_out) {

    data_contig_md *data_out = (data_contig_md*) malloc(sizeof(data_contig_md));

    data_out->particle_cnt = particle_cnt;

    data_out->x =  (float*) malloc(particle_cnt * sizeof(float));
    data_out->y =  (float*) malloc(particle_cnt * sizeof(float));
    data_out->z =  (float*) malloc(particle_cnt * sizeof(float));
    data_out->px = (float*) malloc(particle_cnt * sizeof(float));
    data_out->py = (float*) malloc(particle_cnt * sizeof(float));
    data_out->pz = (float*) malloc(particle_cnt * sizeof(float));
    data_out->id_1 = (int*) malloc(particle_cnt * sizeof(int));
    data_out->id_2 = (float*) malloc(particle_cnt * sizeof(float));

    for (long i = 0; i < particle_cnt; i++) {
        data_out->id_1[i] = i;
        data_out->id_2[i] = (float)(i * 2);
        data_out->x[i] =  uniform_random_number() * X_DIM;
        data_out->y[i] =  uniform_random_number() * Y_DIM;
        data_out->px[i] = uniform_random_number() * X_DIM;
        data_out->py[i] = uniform_random_number() * Y_DIM;
        data_out->z[i] =  ((float) data_out->id_1[i] / NUM_PARTICLES) * Z_DIM;
        data_out->pz[i] = ( data_out->id_2[i] / NUM_PARTICLES) * Z_DIM;
    }
    *data_size_out = particle_cnt * (7 * sizeof(float) +  sizeof(int));

    return data_out;
}

data_contig_md* prepare_data_contig_2D(long particle_cnt, long dim_1, long dim_2, unsigned long * data_size_out){
    if(particle_cnt != dim_1 * dim_2){
        if(MY_RANK == 0)
            printf("Dimension definition is invalid: dim_1(%ld) * dim_2(%ld) must equal num_particles (%ld) per rank.\n", dim_1, dim_2, particle_cnt);
        return NULL;
    }
    assert(particle_cnt == dim_1 * dim_2);
    data_contig_md *data_out = (data_contig_md*) malloc(sizeof(data_contig_md));
    data_out->particle_cnt = particle_cnt;
    data_out->dim_1 = dim_1;
    data_out->dim_2 = dim_2;

    data_out->x =  (float*) malloc(particle_cnt * sizeof(float));
    data_out->y =  (float*) malloc(particle_cnt * sizeof(float));
    data_out->z =  (float*) malloc(particle_cnt * sizeof(float));
    data_out->px = (float*) malloc(particle_cnt * sizeof(float));
    data_out->py = (float*) malloc(particle_cnt * sizeof(float));
    data_out->pz = (float*) malloc(particle_cnt * sizeof(float));
    data_out->id_1 = (int*) malloc(particle_cnt * sizeof(int));
    data_out->id_2 = (float*) malloc(particle_cnt * sizeof(float));

    long idx = 0;
    for(long i1 = 0; i1 < dim_1; i1++){
        for(long i2 = 0; i2 < dim_2; i2++){
            data_out->x[idx] = uniform_random_number() * X_DIM;
            data_out->id_1[idx] = i1;
            data_out->id_2[idx] = (float)(i1 * 2);
            data_out->x[idx] =  uniform_random_number() * X_DIM;
            data_out->y[idx] =  uniform_random_number() * Y_DIM;
            data_out->px[idx] = uniform_random_number() * X_DIM;
            data_out->py[idx] = uniform_random_number() * Y_DIM;
            data_out->z[idx] =  ((float) data_out->id_1[idx] / NUM_PARTICLES) * Z_DIM;
            data_out->pz[idx] = ( data_out->id_2[idx] / NUM_PARTICLES) * Z_DIM;
            idx++;
        }
    }
    *data_size_out = particle_cnt * (7 * sizeof(float) + sizeof(int));

    return data_out;
}

data_contig_md* prepare_data_contig_3D(long particle_cnt, long dim_1, long dim_2, long dim_3, unsigned long * data_size_out){
    if(particle_cnt != dim_1 * dim_2 * dim_3){
        if(MY_RANK == 0)
            printf("Dimension definition is invalid: dim_1(%ld) * dim_2(%ld) * dim_3(%ld) must equal num_particles (%ld) per rank.\n", dim_1, dim_2, dim_3, particle_cnt);
        return NULL;
    }

    assert(particle_cnt == dim_1 * dim_2 * dim_3);
    data_contig_md *data_out = (data_contig_md*) malloc(sizeof(data_contig_md));
    data_out->particle_cnt = particle_cnt;
    data_out->dim_1 = dim_1;
    data_out->dim_2 = dim_2;
    data_out->dim_3 = dim_3;
    data_out->x =  (float*) malloc(particle_cnt * sizeof(float));
    data_out->y =  (float*) malloc(particle_cnt * sizeof(float));
    data_out->z =  (float*) malloc(particle_cnt * sizeof(float));
    data_out->px = (float*) malloc(particle_cnt * sizeof(float));
    data_out->py = (float*) malloc(particle_cnt * sizeof(float));
    data_out->pz = (float*) malloc(particle_cnt * sizeof(float));
    data_out->id_1 = (int*) malloc(particle_cnt * sizeof(int));
    data_out->id_2 = (float*) malloc(particle_cnt * sizeof(float));
    long idx = 0;
    for(long i1 = 0; i1 < dim_1; i1++){
        for(long i2 = 0; i2 < dim_2; i2++){
            for(long i3 = 0; i3 < dim_3; i3++){
                data_out->x[idx] = uniform_random_number() * X_DIM;
                data_out->id_1[idx] = i1;
                data_out->id_2[idx] = (float)(i1 * 2);
                data_out->x[idx] =  uniform_random_number() * X_DIM;
                data_out->y[idx] =  uniform_random_number() * Y_DIM;
                data_out->px[idx] = uniform_random_number() * X_DIM;
                data_out->py[idx] = uniform_random_number() * Y_DIM;
                data_out->z[idx] =  ((float) data_out->id_1[idx] / NUM_PARTICLES) * Z_DIM;
                data_out->pz[idx] = (data_out->id_2[idx] / NUM_PARTICLES) * Z_DIM;
                idx++;
            }
        }
    }
    *data_size_out = particle_cnt * (7 * sizeof(float) + sizeof(int));
    return data_out;
}

void data_free(write_pattern mode, void* data){
    assert(data);
    switch(mode){
        case CONTIG_CONTIG_1D:
        case CONTIG_INTERLEAVED_1D:
        case CONTIG_INTERLEAVED_2D:
        case CONTIG_CONTIG_2D:
        case CONTIG_CONTIG_3D:
            free(((data_contig_md*)data)->x);
            free(((data_contig_md*)data)->y);
            free(((data_contig_md*)data)->z);
            free(((data_contig_md*)data)->px);
            free(((data_contig_md*)data)->py);
            free(((data_contig_md*)data)->pz);
            free(((data_contig_md*)data)->id_1);
            free(((data_contig_md*)data)->id_2);
            free(((data_contig_md*)data));
            break;
        case INTERLEAVED_CONTIG_1D:
        case INTERLEAVED_CONTIG_2D:
        case INTERLEAVED_INTERLEAVED_1D:
        case INTERLEAVED_INTERLEAVED_2D:
            free(data);
            break;
        default:
            break;
    }
}

void set_dspace_plist(hid_t* plist_id_out){
    *plist_id_out = H5Pcreate(H5P_DATASET_XFER);
    H5Pset_dxpl_mpio(*plist_id_out, H5FD_MPIO_COLLECTIVE);
}

int set_select_spaces_default(hid_t* filespace_out, hid_t* memspace_out){
    *filespace_out = H5Screate_simple(1, (hsize_t *) &TOTAL_PARTICLES, NULL);
    *memspace_out =  H5Screate_simple(1, (hsize_t *) &NUM_PARTICLES, NULL);
    H5Sselect_hyperslab(*filespace_out, H5S_SELECT_SET, (hsize_t *) &FILE_OFFSET, NULL, (hsize_t *) &NUM_PARTICLES, NULL);
    return 0;
}

int set_select_space_2D_array(hid_t* filespace_out, hid_t* memspace_out,
        unsigned long dim_1, unsigned long dim_2){//dim_1 * dim_2 === NUM_PARTICLES
    hsize_t mem_dims[2], file_dims[2];
    mem_dims[0] = (hsize_t)dim_1;
    mem_dims[1] = (hsize_t)dim_2;
    file_dims[0] = (hsize_t)dim_1 * NUM_RANKS; //total x length: dim_1 * world_size.
    file_dims[1] = (hsize_t)dim_2;//always the same dim_2

    hsize_t file_starts[2], count[2];//select start point and range in each dimension.
    file_starts[0] = dim_1 * (MY_RANK);//file offset for each rank
    file_starts[1] = 0;
    count[0] = dim_1;
    count[1] = dim_2;

    *filespace_out = H5Screate_simple(2, file_dims, NULL);
    *memspace_out =  H5Screate_simple(2, mem_dims, NULL);
    if(MY_RANK == 0) printf("%lu * %lu 2D array, my x_start = %llu, y_start = %llu, x_cnt = %llu, y_cnt = %llu\n",
            dim_1, dim_2, file_starts[0], file_starts[1], count[0], count[1]);
    H5Sselect_hyperslab(*filespace_out, H5S_SELECT_SET, file_starts, NULL, count, NULL);
    return 0;
}

int set_select_space_multi_3D_array(hid_t* filespace_out, hid_t* memspace_out,
        unsigned long dim_1, unsigned long dim_2, unsigned long dim_3){
    hsize_t mem_dims[3];
    hsize_t file_dims[3];
    mem_dims[0] = (hsize_t)dim_1;
    mem_dims[1] = (hsize_t)dim_2;
    mem_dims[2] = (hsize_t)dim_3;
    file_dims[0] = (hsize_t)dim_1 * NUM_RANKS;
    file_dims[1] = (hsize_t)dim_2;
    file_dims[2] = (hsize_t)dim_3;
    hsize_t file_starts[3], file_range[3];//select start point and range in each dimension.
    file_starts[0] = dim_1 * (MY_RANK);
    file_starts[1] = 0;
    file_starts[2] = 0;
    file_range[0] = dim_1;
    file_range[1] = dim_2;
    file_range[2] = dim_3;

    *filespace_out = H5Screate_simple(3, file_dims, NULL); //(1, (hsize_t *) &TOTAL_PARTICLES, NULL);//= world_size * numparticles
    *memspace_out =  H5Screate_simple(3, mem_dims, NULL);

    H5Sselect_hyperslab(*filespace_out, H5S_SELECT_SET, file_starts, NULL, file_range, NULL);
    return 0;
}

/*
 *  write file: create m-D array as the dateset type, now linear-linear is 8 datasets of 1D array
 */
void data_write_contig_contig_MD_array(hid_t loc, hid_t *dset_ids, hid_t filespace, hid_t memspace, hid_t plist_id,
        data_contig_md* data_in){
    assert(data_in && data_in->x);
    hid_t dcpl;
    if(COMPRESS_INFO.USE_COMPRESS)
        dcpl = COMPRESS_INFO.dcpl_id;
    else
        dcpl = H5P_DEFAULT;
    if(MY_RANK == 0){
        if(COMPRESS_INFO.USE_COMPRESS)
            printf("Parallel compressed: chunk_dim1 = %lu, chunk_dim2 = %lu\n", COMPRESS_INFO.chunk_dims[0], COMPRESS_INFO.chunk_dims[1]);
        else
            printf("compression not invoked.\n");
    }

    dset_ids[0] = H5Dcreate_async(loc, "x", H5T_NATIVE_FLOAT, filespace, H5P_DEFAULT, dcpl, H5P_DEFAULT, 0);
    dset_ids[1] = H5Dcreate_async(loc, "y", H5T_NATIVE_FLOAT, filespace, H5P_DEFAULT, dcpl, H5P_DEFAULT, 0);
    dset_ids[2] = H5Dcreate_async(loc, "z", H5T_NATIVE_FLOAT, filespace, H5P_DEFAULT, dcpl, H5P_DEFAULT, 0);
    dset_ids[3] = H5Dcreate_async(loc, "px", H5T_NATIVE_FLOAT, filespace, H5P_DEFAULT, dcpl, H5P_DEFAULT, 0);
    dset_ids[4] = H5Dcreate_async(loc, "py", H5T_NATIVE_FLOAT, filespace, H5P_DEFAULT, dcpl, H5P_DEFAULT, 0);
    dset_ids[5] = H5Dcreate_async(loc, "pz", H5T_NATIVE_FLOAT, filespace, H5P_DEFAULT, dcpl, H5P_DEFAULT, 0);
    dset_ids[6] = H5Dcreate_async(loc, "id_1", H5T_NATIVE_INT, filespace, H5P_DEFAULT, dcpl, H5P_DEFAULT, 0);
    dset_ids[7] = H5Dcreate_async(loc, "id_2", H5T_NATIVE_FLOAT, filespace, H5P_DEFAULT, dcpl, H5P_DEFAULT, 0);

    ierr = H5Dwrite_async(dset_ids[0], H5T_NATIVE_FLOAT, memspace, filespace, plist_id, data_in->x, 0);
    ierr = H5Dwrite_async(dset_ids[1], H5T_NATIVE_FLOAT, memspace, filespace, plist_id, data_in->y, 0);
    ierr = H5Dwrite_async(dset_ids[2], H5T_NATIVE_FLOAT, memspace, filespace, plist_id, data_in->z, 0);
    ierr = H5Dwrite_async(dset_ids[3], H5T_NATIVE_FLOAT, memspace, filespace, plist_id, data_in->px, 0);
    ierr = H5Dwrite_async(dset_ids[4], H5T_NATIVE_FLOAT, memspace, filespace, plist_id, data_in->py, 0);
    ierr = H5Dwrite_async(dset_ids[5], H5T_NATIVE_FLOAT, memspace, filespace, plist_id, data_in->pz, 0);
    ierr = H5Dwrite_async(dset_ids[6], H5T_NATIVE_INT, memspace, filespace, plist_id, data_in->id_1, 0);
    ierr = H5Dwrite_async(dset_ids[7], H5T_NATIVE_FLOAT, memspace, filespace, plist_id, data_in->id_2, 0);

    if (MY_RANK == 0) printf ("    %s: Finished writing time step \n", __func__);
}

void data_write_contig_to_interleaved(hid_t loc, hid_t *dset_ids, hid_t filespace, hid_t memspace, hid_t plist_id,
        data_contig_md* data_in){
    assert(data_in && data_in->x);
    hid_t dcpl;
    if(COMPRESS_INFO.USE_COMPRESS)
        dcpl = COMPRESS_INFO.dcpl_id;
    else
        dcpl = H5P_DEFAULT;

    dset_ids[0] = H5Dcreate_async(loc, "particles", PARTICLE_COMPOUND_TYPE, filespace, H5P_DEFAULT, dcpl, H5P_DEFAULT, 0);

    ierr = H5Dwrite_async(dset_ids[0], PARTICLE_COMPOUND_TYPE_SEPARATES[0], memspace, filespace, plist_id, data_in->x, 0);
    ierr = H5Dwrite_async(dset_ids[0], PARTICLE_COMPOUND_TYPE_SEPARATES[1], memspace, filespace, plist_id, data_in->y, 0);
    ierr = H5Dwrite_async(dset_ids[0], PARTICLE_COMPOUND_TYPE_SEPARATES[2], memspace, filespace, plist_id, data_in->z, 0);
    ierr = H5Dwrite_async(dset_ids[0], PARTICLE_COMPOUND_TYPE_SEPARATES[3], memspace, filespace, plist_id, data_in->px, 0);
    ierr = H5Dwrite_async(dset_ids[0], PARTICLE_COMPOUND_TYPE_SEPARATES[4], memspace, filespace, plist_id, data_in->py, 0);
    ierr = H5Dwrite_async(dset_ids[0], PARTICLE_COMPOUND_TYPE_SEPARATES[5], memspace, filespace, plist_id, data_in->pz, 0);
    ierr = H5Dwrite_async(dset_ids[0], PARTICLE_COMPOUND_TYPE_SEPARATES[6], memspace, filespace, plist_id, data_in->id_1, 0);
    ierr = H5Dwrite_async(dset_ids[0], PARTICLE_COMPOUND_TYPE_SEPARATES[7], memspace, filespace, plist_id, data_in->id_2, 0);

    if (MY_RANK == 0) printf ("    %s: Finished writing time step \n", __func__);
}

void data_write_interleaved_to_contig(hid_t loc, hid_t *dset_ids, hid_t filespace, hid_t memspace, hid_t plist_id,
        particle* data_in) {
    assert(data_in);
    hid_t dcpl;
    if(COMPRESS_INFO.USE_COMPRESS)
        dcpl = COMPRESS_INFO.dcpl_id;
    else
        dcpl = H5P_DEFAULT;

    dset_ids[0] = H5Dcreate_async(loc, "x",   PARTICLE_COMPOUND_TYPE_SEPARATES[0], filespace, H5P_DEFAULT, dcpl, H5P_DEFAULT, 0);
    dset_ids[1] = H5Dcreate_async(loc, "y",   PARTICLE_COMPOUND_TYPE_SEPARATES[1], filespace, H5P_DEFAULT, dcpl, H5P_DEFAULT, 0);
    dset_ids[2] = H5Dcreate_async(loc, "z",   PARTICLE_COMPOUND_TYPE_SEPARATES[2], filespace, H5P_DEFAULT, dcpl, H5P_DEFAULT, 0);
    dset_ids[3] = H5Dcreate_async(loc, "px",  PARTICLE_COMPOUND_TYPE_SEPARATES[3], filespace, H5P_DEFAULT, dcpl, H5P_DEFAULT, 0);
    dset_ids[4] = H5Dcreate_async(loc, "py",  PARTICLE_COMPOUND_TYPE_SEPARATES[4], filespace, H5P_DEFAULT, dcpl, H5P_DEFAULT, 0);
    dset_ids[5] = H5Dcreate_async(loc, "pz",  PARTICLE_COMPOUND_TYPE_SEPARATES[5], filespace, H5P_DEFAULT, dcpl, H5P_DEFAULT, 0);
    dset_ids[6] = H5Dcreate_async(loc, "id_1", PARTICLE_COMPOUND_TYPE_SEPARATES[6], filespace, H5P_DEFAULT, dcpl, H5P_DEFAULT, 0);
    dset_ids[7] = H5Dcreate_async(loc, "id_2", PARTICLE_COMPOUND_TYPE_SEPARATES[7], filespace, H5P_DEFAULT, dcpl, H5P_DEFAULT, 0);

    ierr = H5Dwrite_async(dset_ids[0], PARTICLE_COMPOUND_TYPE, memspace, filespace, plist_id, data_in, 0);
    ierr = H5Dwrite_async(dset_ids[1], PARTICLE_COMPOUND_TYPE, memspace, filespace, plist_id, data_in, 0);
    ierr = H5Dwrite_async(dset_ids[2], PARTICLE_COMPOUND_TYPE, memspace, filespace, plist_id, data_in, 0);
    ierr = H5Dwrite_async(dset_ids[3], PARTICLE_COMPOUND_TYPE, memspace, filespace, plist_id, data_in, 0);
    ierr = H5Dwrite_async(dset_ids[4], PARTICLE_COMPOUND_TYPE, memspace, filespace, plist_id, data_in, 0);
    ierr = H5Dwrite_async(dset_ids[5], PARTICLE_COMPOUND_TYPE, memspace, filespace, plist_id, data_in, 0);
    ierr = H5Dwrite_async(dset_ids[6], PARTICLE_COMPOUND_TYPE, memspace, filespace, plist_id, data_in, 0);
    ierr = H5Dwrite_async(dset_ids[7], PARTICLE_COMPOUND_TYPE, memspace, filespace, plist_id, data_in, 0);

    if (MY_RANK == 0) printf ("    %s: Finished writing time step \n", __func__);
}

void data_write_interleaved_to_interleaved(hid_t loc, hid_t *dset_ids, hid_t filespace, hid_t memspace, hid_t plist_id,
        particle* data_in) {
    assert(data_in);
    hid_t dcpl;
    if(COMPRESS_INFO.USE_COMPRESS)
        dcpl = COMPRESS_INFO.dcpl_id;
    else
        dcpl = H5P_DEFAULT;

    dset_ids[0] = H5Dcreate_async(loc, "particles", PARTICLE_COMPOUND_TYPE, filespace, H5P_DEFAULT, dcpl, H5P_DEFAULT, 0);
    ierr = H5Dwrite_async(dset_ids[0], PARTICLE_COMPOUND_TYPE, memspace, filespace, plist_id, data_in, 0);//should write all things in data_in

    if (MY_RANK == 0) printf ("    %s: Finished writing time step \n", __func__);
}

int _run_time_steps(bench_params params, hid_t file_id, unsigned long* total_data_size_out, unsigned long* raw_write_time_out) {
    write_pattern mode = params.access_pattern.pattern_write;
    long particle_cnt = params.cnt_particle_M * M_VAL;
    int timestep_cnt = params.cnt_time_step;
    int sleep_time = params.sleep_time;

    char grp_name[128];
    unsigned long  rt_start, rt_end;
    int grp_cnt = 0, dset_cnt = 0;
    hid_t **dset_ids = (hid_t**)calloc(timestep_cnt, sizeof(hid_t*));
    hid_t *grp_ids  = (hid_t*)calloc(timestep_cnt, sizeof(hid_t));
    *raw_write_time_out = 0;
    void* data = NULL;
    unsigned long data_size;
    hid_t plist_id, filespace, memspace;
    set_dspace_plist(&plist_id);

    make_compound_type_separates();
    make_compound_type();

    switch(mode){
        case CONTIG_CONTIG_1D:
            set_select_spaces_default(&filespace, &memspace);
            data = (void*)prepare_data_contig_1D(particle_cnt, &data_size);
            dset_cnt = 8;
            break;

        case CONTIG_CONTIG_2D:
            set_select_space_2D_array(&filespace, &memspace, params.dim_1, params.dim_2);
            data = (void*)prepare_data_contig_2D(particle_cnt, params.dim_1, params.dim_2, &data_size);
            dset_cnt = 8;
            break;


        case CONTIG_INTERLEAVED_1D:
            set_select_spaces_default(&filespace, &memspace);
            data = (void*)prepare_data_contig_1D(particle_cnt, &data_size);
            dset_cnt = 1;
            break;

        case CONTIG_INTERLEAVED_2D:
            set_select_space_2D_array(&filespace, &memspace, params.dim_1, params.dim_2);
            data = (void*)prepare_data_contig_2D(particle_cnt, params.dim_1, params.dim_2, &data_size);
            dset_cnt = 1;
            break;

        case INTERLEAVED_CONTIG_1D:
            set_select_spaces_default(&filespace, &memspace);
            data = (void*)prepare_data_interleaved(particle_cnt, &data_size);
            dset_cnt = 8;
            break;

        case INTERLEAVED_CONTIG_2D:
            set_select_space_2D_array(&filespace, &memspace, params.dim_1, params.dim_2);
            data = (void*)prepare_data_interleaved(particle_cnt, &data_size);
            dset_cnt = 8;
            break;

        case INTERLEAVED_INTERLEAVED_1D:
            set_select_spaces_default(&filespace, &memspace);
            data = (void*)prepare_data_interleaved(particle_cnt, &data_size);
            dset_cnt = 1;
            break;

        case INTERLEAVED_INTERLEAVED_2D:
            set_select_space_2D_array(&filespace, &memspace, params.dim_1, params.dim_2);
            data = (void*)prepare_data_interleaved(particle_cnt, &data_size);
            dset_cnt = 1;
            break;

        case CONTIG_CONTIG_3D:
            set_select_space_multi_3D_array(&filespace, &memspace, params.dim_1, params.dim_2, params.dim_3);
            data = (void*)prepare_data_contig_3D(particle_cnt, params.dim_1, params.dim_2, params.dim_3, &data_size);
            dset_cnt = 8;
            break;
        default:
            assert(0 && "this mode is not yet implemented");
            break;
    }

    if(!data){
        if (MY_RANK == 0)
            printf("Failed to generate data for writing, please check dimension settings in the config file.\n");
        return -1;
    }

    for (int i = 0; i < timestep_cnt; i++) {
        sprintf(grp_name, "Timestep_%d", i);
        MPI_Barrier (MPI_COMM_WORLD);
        grp_ids[i] = H5Gcreate_async(file_id, grp_name, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT, 0);

        if (MY_RANK == 0)
            printf ("Writing %s ... \n", grp_name);

        dset_ids[i] = (hid_t*)calloc(8, sizeof(hid_t));

        rt_start = get_time_usec();
        MPI_Barrier (MPI_COMM_WORLD);
        switch(mode){
            case CONTIG_CONTIG_1D:
            case CONTIG_CONTIG_2D:
            case CONTIG_CONTIG_3D:
                data_write_contig_contig_MD_array(grp_ids[i], dset_ids[i], filespace, memspace, plist_id, (data_contig_md*)data);
                break;

            case CONTIG_INTERLEAVED_1D:
            case CONTIG_INTERLEAVED_2D:
                data_write_contig_to_interleaved(grp_ids[i], dset_ids[i], filespace, memspace, plist_id, (data_contig_md*)data);
                break;

            case INTERLEAVED_CONTIG_1D:
            case INTERLEAVED_CONTIG_2D:
                data_write_interleaved_to_contig(grp_ids[i], dset_ids[i], filespace, memspace, plist_id, (particle*)data);
                break;

            case INTERLEAVED_INTERLEAVED_1D:
            case INTERLEAVED_INTERLEAVED_2D:
                data_write_interleaved_to_interleaved(grp_ids[i], dset_ids[i], filespace, memspace, plist_id, (particle*)data);
                break;

            default:
                break;
        }
        rt_end = get_time_usec();

        *raw_write_time_out += (rt_end - rt_start);
        MPI_Barrier (MPI_COMM_WORLD);

        if (i != timestep_cnt - 1) {
            if (sleep_time > 0) {
                if (MY_RANK == 0) printf ("  sleep for %ds\n", sleep_time);
                sleep(sleep_time);
            }
        }

        for (int j = 0; j < dset_cnt; j++)
            H5Dclose_async(dset_ids[i][j], 0);
        H5Gclose_async(grp_ids[i], 0);

        MPI_Barrier (MPI_COMM_WORLD);
        free(dset_ids[i]);
    }

    H5Tclose(PARTICLE_COMPOUND_TYPE);
    for(int i = 0; i < 8; i++)
        H5Tclose(PARTICLE_COMPOUND_TYPE_SEPARATES[i]);

    *total_data_size_out = timestep_cnt * data_size;

    data_free(mode, data);

    H5Sclose(memspace);
    H5Sclose(filespace);
    H5Pclose(plist_id);

    return 0;
}

void set_globals(bench_params params){
    NUM_PARTICLES = params.cnt_particle_M * 1024 *1024;
    NUM_TIMESTEPS = params.cnt_time_step;
    //following variables only used to generate data
    X_DIM = X_RAND;
    Y_DIM = Y_RAND;
    Z_DIM = Z_RAND;
    COMPRESS_INFO.USE_COMPRESS = params.useCompress;
    COMPRESS_INFO.chunk_dims[0] = params.chunk_dim_1;
    COMPRESS_INFO.chunk_dims[1] = params.chunk_dim_2;
    COMPRESS_INFO.chunk_dims[2] = params.chunk_dim_3;

    if(COMPRESS_INFO.USE_COMPRESS) {//set DCPL
        herr_t ret;
        COMPRESS_INFO.dcpl_id = H5Pcreate(H5P_DATASET_CREATE);
        assert(COMPRESS_INFO.dcpl_id > 0);

        /* Set chunked layout and chunk dimensions */
        ret = H5Pset_layout(COMPRESS_INFO.dcpl_id, H5D_CHUNKED);
        assert(ret >= 0);
        ret = H5Pset_chunk(COMPRESS_INFO.dcpl_id, params._dim_cnt, (const hsize_t*) COMPRESS_INFO.chunk_dims);
        assert(ret >= 0);
        ret = H5Pset_deflate(COMPRESS_INFO.dcpl_id, 9);
        assert(ret >= 0);
    }
}

hid_t set_fapl(){
    hid_t fapl = H5Pcreate(H5P_FILE_ACCESS);
    return fapl;
}

hid_t set_metadata(hid_t fapl, int align, unsigned long threshold, unsigned long alignment_len, int collective){
    if(align != 0)
        H5Pset_alignment(fapl, threshold, alignment_len);

    if(collective == 1){
        if(MY_RANK == 0)
            printf("Collective write: enabled.\n");
        H5Pset_all_coll_metadata_ops(fapl, 1);
        H5Pset_coll_metadata_write(fapl, 1);
    } else {
        if(MY_RANK == 0)
            printf("Collective write: disabled.\n");
    }

    // Defer metadata flush
    if(DEFER_METADATA_FLUSH){
        H5AC_cache_config_t cache_config;
        cache_config.version = H5AC__CURR_CACHE_CONFIG_VERSION;
        H5Pget_mdc_config(fapl, &cache_config);
        cache_config.set_initial_size = 1;
        cache_config.initial_size = 16 * M_VAL;
        cache_config.evictions_enabled = 0;
        cache_config.incr_mode = H5C_incr__off;
        cache_config.flash_incr_mode = H5C_flash_incr__off;
        cache_config.decr_mode = H5C_decr__off;
        H5Pset_mdc_config (fapl, &cache_config);
    }
    return fapl;
}

void print_usage(char *name) {
    if(MY_RANK == 0){
        printf("=============== Usage: %s /path_to_config_file /path_to_output_data_file [CSV csv_file_path]=============== \n", name);
        printf("- CSV is optional.\n");
        printf("- Only CC/CI/IC/II/CC2D/CC3D is used to set benchmark mode in the config file, stands for CONTIG_CONTIG_1D, CONTIG_INTERLEAVED_1D, INTERLEAVED_CONTIG_1D, INTERLEAVED_INTERLEAVED_1D, 2D Array and 3D Array\n");
        printf("- For 2D/3D benchmarks, make sure the dimensions are set correctly and matches the per rank particle number.\n");
        printf("- For example, when your PATTERN is CC3D, and PARTICLE_CNT_M is 1, setting DIM_1~3 to 64, 64, and 256 is valid, because 64*64*256 = 1,048,576 (1M); and 10*20*30 is invalid. \n");
    }
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &MY_RANK);
    MPI_Comm_size(MPI_COMM_WORLD, &NUM_RANKS);
    MPI_Comm comm = MPI_COMM_WORLD;
    MPI_Info info = MPI_INFO_NULL;

    int sleep_time = 0;
    if(MY_RANK == 0){
        if(argc != 3 && argc != 5){
            print_usage(argv[0]);
            return 0;
        }
    }

    char *output_file;
    bench_params params;

    char* cfg_file_path = argv[1];
    output_file = argv[2];
    if (MY_RANK == 0)
        printf("config file: %s, output data file: %s\n", argv[1], argv[2]);

    if(read_config(cfg_file_path, &params) < 0){
        if (MY_RANK == 0)
            printf("Config file read failed. check path: %s\n", cfg_file_path);
        return 0;
    }

    params.useCSV = 0;
    int arg_idx_csv = 3;
    if(argc ==5){
        if(MY_RANK == 0 && strcmp(argv[arg_idx_csv], "CSV") == 0) {
            char* csv_path = argv[arg_idx_csv + 1];
            if(csv_path){
                FILE* csv_fs = csv_init(csv_path);
                if(!csv_fs){
                    printf("Failed to create CSV file. \n");
                    return -1;
                }
                params.csv_fs = csv_fs;
                params.useCSV = 1;
            } else {
                printf("CSV option is enabled but file path is not specified.\n");
                return -1;
            }
        }
    }

    if(MY_RANK == 0)
        print_params(&params);

    set_globals(params);

    NUM_TIMESTEPS = params.cnt_time_step;

    if (MY_RANK == 0)
        printf("Start benchmark: VPIC %s, Number of paritcles per rank: %lld M\n", params.pattern_name, NUM_PARTICLES/(1024*1024));

    unsigned long total_write_size = NUM_RANKS * NUM_TIMESTEPS * NUM_PARTICLES * (7 * sizeof(float) + sizeof(int));

    MPI_Barrier(MPI_COMM_WORLD);

    unsigned long t0 = get_time_usec();
    MPI_Allreduce(&NUM_PARTICLES, &TOTAL_PARTICLES, 1, MPI_LONG_LONG, MPI_SUM, comm);
    MPI_Scan(&NUM_PARTICLES, &FILE_OFFSET, 1, MPI_LONG_LONG, MPI_SUM, comm);
    FILE_OFFSET -= NUM_PARTICLES;

    if (MY_RANK == 0)
        printf("Total particle number = %lldM\n", TOTAL_PARTICLES / (M_VAL));

    hid_t fapl = set_fapl();

    H5Pset_fapl_mpio(fapl, comm, info);

    int align = 0;
    set_metadata(fapl, ALIGN, ALIGN_THRESHOLD, ALIGN_LEN, params.collective);

    unsigned long t1 = get_time_usec(); // t1 - t0: cost of settings
    hid_t file_id = H5Fcreate_async(output_file, H5F_ACC_TRUNC, H5P_DEFAULT, fapl, 0);
    H5Pclose(fapl);

    if (MY_RANK == 0)
        printf("Opened HDF5 file... \n");

    MPI_Barrier(MPI_COMM_WORLD);
    unsigned long t2 = get_time_usec(); // t2 - t1: metadata: creating/opening

    unsigned long raw_write_time, total_data_size;
    int stat = _run_time_steps(params, file_id, &total_data_size, &raw_write_time);

    if(stat < 0){
        if (MY_RANK == 0)
            printf("=============== Benchmark failed. ===============\n");
        assert(0);
    }

    unsigned long t3 = get_time_usec(); // t3 - t2: writting data, including metadata

    if (MY_RANK == 0) {
        printf("\n Performance measured with %d ranks\n", NUM_RANKS);
        if(params.collective == 1)
            printf("CollectiveWrite: YES\n");
        else
            printf("CollectiveWrite: NO\n");
    }

    H5Fclose_async(file_id, 0);

    MPI_Barrier(MPI_COMM_WORLD);
    unsigned long t4 = get_time_usec();

    if (MY_RANK == 0) {
        printf("\n =================  Performance results  =================\n");
        int total_sleep_time = sleep_time * (NUM_TIMESTEPS - 1);
        unsigned long total_size_mb = NUM_RANKS * total_data_size/(1024*1024);
        printf("Total sleep time %ds, total write size = %lu MB\n", total_sleep_time, total_size_mb);

        float rwt_s = (float)raw_write_time / (1000*1000);
        unsigned long raw_rate_mbs = total_data_size / raw_write_time;
        printf("RR: Raw write time = %.3f sec, RR = %lu MB/sec \n", rwt_s, raw_rate_mbs);

        unsigned long meta_time_ms = (t3 - t2 - raw_write_time - sleep_time * (NUM_TIMESTEPS - 1) * 1000 * 1000) / 1000;
        printf("Core metadata time = %lu ms\n", meta_time_ms);

        unsigned long or_mbs = total_data_size / (t4 - t1 - (NUM_TIMESTEPS - 1) * 1000 * 1000);
        printf("OR (observed rate) = %lu MB/sec\n", or_mbs);

        float oct_s = (float)(t4 - t0) / (1000*1000);
        printf("OCT(observed completion time) = %.3f sec\n", oct_s);
        printf("\n");

        if(params.useCSV){
            fprintf(params.csv_fs, "NUM_RANKS, %d\n", NUM_RANKS);
            if(params.collective == 1) fprintf(params.csv_fs, "CollectiveWrite, YES\n");
            else fprintf(params.csv_fs, "CollectiveWrite, NO\n");
            fprintf(params.csv_fs, "Total_sleep_time, %d, sec\n", total_sleep_time);
            fprintf(params.csv_fs, "Total_write_size, %lu, MB\n", total_size_mb);
            fprintf(params.csv_fs, "Raw_write_time, %.3f, sec\n", rwt_s);
            fprintf(params.csv_fs, "Raw_write_rate, %lu, MB/sec\n", raw_rate_mbs);
            fprintf(params.csv_fs, "Core_metadata_time, %lu, ms\n", meta_time_ms);
            fprintf(params.csv_fs, "Observed_rate, %lu, MB/sec\n", or_mbs);
            fprintf(params.csv_fs, "Observed_completion_time, %.3f, sec\n", oct_s);
            fclose(params.csv_fs);
        }
    }

    MPI_Finalize();
    return 0;
}
