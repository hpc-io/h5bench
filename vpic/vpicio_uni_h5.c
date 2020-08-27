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

// A simple timer based on gettimeofday

#define DTYPE double

// HDF5 specific declerations
herr_t ierr;

// Global Variables and dimensions
long NUM_PARTICLES = 8388608;	// 8  meg particles per process
long long TOTAL_PARTICLES, FILE_OFFSET;
int X_DIM = 64;
int Y_DIM = 64;
int Z_DIM = 64;

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
    COMPOUND_COMPOUND
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

//need to close later.
hid_t make_compound_type(){
    hid_t tid = H5Tcreate(H5T_COMPOUND, sizeof(particle));
    H5Tinsert(tid, "x", HOFFSET(particle, x), H5T_NATIVE_DOUBLE);
    H5Tinsert(tid, "y", HOFFSET(particle, y), H5T_NATIVE_DOUBLE);
    H5Tinsert(tid, "z", HOFFSET(particle, z), H5T_NATIVE_DOUBLE);
    H5Tinsert(tid, "px", HOFFSET(particle, px), H5T_NATIVE_DOUBLE);
    H5Tinsert(tid, "py", HOFFSET(particle, py), H5T_NATIVE_DOUBLE);
    H5Tinsert(tid, "pz", HOFFSET(particle, pz), H5T_NATIVE_DOUBLE);
    H5Tinsert(tid, "id_1", HOFFSET(particle, id_1), H5T_NATIVE_LONG);
    H5Tinsert(tid, "id_2", HOFFSET(particle, id_2), H5T_NATIVE_LONG);
    return tid;
}

hid_t* make_compound_separates(){
    hid_t* tids = (hid_t*)malloc(8 * sizeof(hid_t));
    DEBUG_PRINT
    tids[0] = H5Tcreate(H5T_COMPOUND, sizeof(double));
    DEBUG_PRINT
    H5Tinsert(tids[0], "x", HOFFSET(particle, x), H5T_NATIVE_DOUBLE);
    DEBUG_PRINT
    tids[1] = H5Tcreate(H5T_COMPOUND, sizeof(double));
    DEBUG_PRINT
    H5Tinsert(tids[1], "y", HOFFSET(particle, y), H5T_NATIVE_DOUBLE);
    DEBUG_PRINT
    tids[2] = H5Tcreate(H5T_COMPOUND, sizeof(double));
    H5Tinsert(tids[2], "z", HOFFSET(particle, z), H5T_NATIVE_DOUBLE);
    tids[3] = H5Tcreate(H5T_COMPOUND, sizeof(double));
    H5Tinsert(tids[3], "px", HOFFSET(particle, px), H5T_NATIVE_DOUBLE);
    tids[4] = H5Tcreate(H5T_COMPOUND, sizeof(double));
    H5Tinsert(tids[4], "py", HOFFSET(particle, py), H5T_NATIVE_DOUBLE);
    tids[5] = H5Tcreate(H5T_COMPOUND, sizeof(double));
    H5Tinsert(tids[5], "pz", HOFFSET(particle, pz), H5T_NATIVE_DOUBLE);

    tids[6] = H5Tcreate(H5T_COMPOUND, sizeof(particle));
    H5Tinsert(tids[6], "id_1", HOFFSET(particle, id_1), H5T_NATIVE_LONG);
    tids[7] = H5Tcreate(H5T_COMPOUND, sizeof(particle));
    H5Tinsert(tids[7], "id_2", HOFFSET(particle, id_2), H5T_NATIVE_LONG);

    return tids;
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
    *data_size_out = particle_cnt * sizeof(particle);
    return data_out;
}

data_linear * prepare_data_linear(long particle_cnt, unsigned long * data_size_out) {

    data_linear *data_out = (data_linear*) malloc(sizeof(data_linear));

    data_out->particle_cnt = particle_cnt;

    data_out->x = (double*) malloc(particle_cnt * sizeof(double));
    data_out->y = (double*) malloc(particle_cnt * sizeof(double));
    data_out->z = (double*) malloc(particle_cnt * sizeof(double));
    data_out->px = (double*) malloc(particle_cnt * sizeof(double));
    data_out->py = (double*) malloc(particle_cnt * sizeof(double));
    data_out->pz = (double*) malloc(particle_cnt * sizeof(double));
    data_out->id_1 = (long*) malloc(particle_cnt * sizeof(long));
    data_out->id_2 = (long*) malloc(particle_cnt * sizeof(long));

    for (long i = 0; i < particle_cnt; i++) {
        data_out->id_1[i] = i;
        data_out->id_2[i] = i * 2;
        data_out->x[i] = uniform_random_number() * X_DIM;
        data_out->y[i] = uniform_random_number() * Y_DIM;
        data_out->z[i] = ((double) data_out->id_1[i] / NUM_PARTICLES) * Z_DIM;
        data_out->px[i] = uniform_random_number() * X_DIM;
        data_out->py[i] = uniform_random_number() * Y_DIM;
        data_out->pz[i] = ((double) data_out->id_2[i] / NUM_PARTICLES) * Z_DIM;
    }
    *data_size_out = particle_cnt * (6 * sizeof(double) + 2 * sizeof(long));

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
        default:
            break;
    }
}
void data_write_linear_to_linear(int rank, hid_t loc, hid_t *dset_ids, hid_t filespace, hid_t memspace, hid_t plist_id,
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

    if (rank == 0) printf ("  Finished written 8 variables \n");
}

void data_write_linear_to_compound(int rank, hid_t loc, hid_t *dset_ids, hid_t filespace, hid_t memspace, hid_t plist_id,
        data_linear* data_in){

}

void data_write_comp_to_linear(int rank, hid_t loc, hid_t *dset_ids, hid_t filespace, hid_t memspace, hid_t plist_id,
        particle* data_in) {
    assert(data_in && data_in->x);
    assert(0 && "data_write_comp_to_linear() implementation not finished yet, ");
    DEBUG_PRINT
    hid_t *tids = make_compound_separates();
    DEBUG_PRINT
    dset_ids[0] = H5Dcreate(loc, "x", tids[0], filespace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    DEBUG_PRINT
    dset_ids[1] = H5Dcreate(loc, "y", tids[1], filespace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    dset_ids[2] = H5Dcreate(loc, "z", tids[2], filespace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    dset_ids[3] = H5Dcreate(loc, "px", tids[3], filespace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    dset_ids[4] = H5Dcreate(loc, "py", tids[4], filespace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    dset_ids[5] = H5Dcreate(loc, "pz", tids[5], filespace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

    dset_ids[6] = H5Dcreate(loc, "id_1", tids[6], filespace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    dset_ids[7] = H5Dcreate(loc, "id_2", tids[7], filespace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    DEBUG_PRINT
    ierr = H5Dwrite(dset_ids[0], tids[0], memspace, filespace, plist_id, data_in);
    /* if (rank == 0) printf ("Written variable 1 \n"); */
    DEBUG_PRINT
    ierr = H5Dwrite(dset_ids[1], tids[1], memspace, filespace, plist_id, data_in);
    /* if (rank == 0) printf ("Written variable 2 \n"); */

    ierr = H5Dwrite(dset_ids[2], tids[2], memspace, filespace, plist_id, data_in);
    /* if (rank == 0) printf ("Written variable 3 \n"); */

    ierr = H5Dwrite(dset_ids[3], tids[3], memspace, filespace, plist_id, data_in);
    /* if (rank == 0) printf ("Written variable 6 \n"); */

    ierr = H5Dwrite(dset_ids[4], tids[4], memspace, filespace, plist_id, data_in);
    /* if (rank == 0) printf ("Written variable 7 \n"); */

    ierr = H5Dwrite(dset_ids[5], tids[5], memspace, filespace, plist_id, data_in);
    /* if (rank == 0) printf ("Written variable 8 \n"); */

    ierr = H5Dwrite(dset_ids[6], tids[6], memspace, filespace, plist_id, data_in);
    /* if (rank == 0) printf ("Written variable 4 \n"); */

    ierr = H5Dwrite(dset_ids[7], tids[7], memspace, filespace, plist_id, data_in);
    /* if (rank == 0) printf ("Written variable 5 \n"); */

    if (rank == 0) printf ("  Finished written 8 variables \n");
}

void data_write_comp_to_comp(int rank, hid_t loc, hid_t *dset_ids, hid_t filespace, hid_t memspace, hid_t plist_id,
        particle* data_in) {
    assert(data_in && data_in->x);

    hid_t particle_type = make_compound_type();
    dset_ids[0] = H5Dcreate(loc, "particles", particle_type, filespace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    ierr = H5Dwrite(dset_ids[0], particle_type, memspace, filespace, plist_id, data_in);//should write all things in data_in
    if (rank == 0) printf ("  Finished written 8 variables \n");
}



int _run_time_steps(bench_mode mode, long particle_cnt, int timestep_cnt, int my_rank, int sleep_time,
        hid_t file_id, hid_t plist_id, hid_t filespace, hid_t memspace,
        unsigned long* raw_write_time_out) {

    char grp_name[128];
    unsigned long  rt_start, rt_end;
    int grp_cnt = 0, dset_cnt = 0;
    hid_t **dset_ids = (hid_t**)calloc(timestep_cnt, sizeof(hid_t*));
    hid_t *grp_ids  = (hid_t*)calloc(timestep_cnt, sizeof(hid_t));
    *raw_write_time_out = 0;
    void* data = NULL;
    unsigned long data_size;

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
        default:
            break;
    }

    for (int i = 0; i < timestep_cnt; i++) {
        sprintf(grp_name, "Timestep_%d", i);

        grp_ids[i] = H5Gcreate2(file_id, grp_name, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

        if (my_rank == 0)
            printf ("Writing %s ... \n", grp_name);

        dset_ids[i] = (hid_t*)calloc(8, sizeof(hid_t));

        rt_start = get_time_usec();
        switch(mode){
            case LINEAR_LINEAR:
                data_write_linear_to_linear(my_rank, grp_ids[i], dset_ids[i], filespace, memspace, plist_id, (data_linear*)data);
                break;
            case LINEAR_COMPOUND:
                assert(0 && "LINEAR_COMPOUND is not implemented yet ");
                data_write_linear_to_compound(my_rank, grp_ids[i], dset_ids[i], filespace, memspace, plist_id, (data_linear*)data);
                break;
            case COMPOUND_LINEAR:
                data_write_comp_to_linear(my_rank, grp_ids[i], dset_ids[i], filespace, memspace, plist_id, (particle*)data);
                break;
            case COMPOUND_COMPOUND:
                data_write_comp_to_comp(my_rank, grp_ids[i], dset_ids[i], filespace, memspace, plist_id, (particle*)data);
                break;
            default:
                break;
        }
        rt_end = get_time_usec();

        *raw_write_time_out += (rt_end - rt_start);

        if (my_rank == 0)
            printf("create and write\n");

        fflush(stdout);
        if (i != timestep_cnt - 1) {
            if (my_rank == 0) printf ("  sleep for %ds\n", sleep_time);
            if (sleep_time > 0) sleep(sleep_time);
        }

        for (int j = 0; j < dset_cnt; j++)
            H5Dclose(dset_ids[i][j]);
        H5Gclose(grp_ids[i]);

        MPI_Barrier (MPI_COMM_WORLD);

        free(dset_ids[i]);

        if (my_rank == 0)
            printf("write and wait\n");
    }
    data_free(mode, data);
    return 0;
}

int set_select_spaces_default(hid_t* filespace_out, hid_t* memspace_out, hid_t* plist_id_out){
    *filespace_out = H5Screate_simple(1, (hsize_t *) &TOTAL_PARTICLES, NULL);//= world_size * numparticles
    *memspace_out =  H5Screate_simple(1, (hsize_t *) &NUM_PARTICLES, NULL);
    *plist_id_out = H5Pcreate(H5P_DATASET_XFER);
    H5Pset_dxpl_mpio(*plist_id_out, H5FD_MPIO_COLLECTIVE);
    H5Sselect_hyperslab(*filespace_out, H5S_SELECT_SET, (hsize_t *) &FILE_OFFSET, NULL, (hsize_t *) &NUM_PARTICLES, NULL);
    return 0;
}

hid_t set_fapl(){
    hid_t fapl = H5Pcreate(H5P_FILE_ACCESS);
    return fapl;
}

hid_t set_metadata(hid_t fapl){
    // Alignmemt
    int alignment = 16777216;
    H5Pset_alignment(fapl, alignment, alignment),

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
    printf("Usage: %s /path_to_file num_time_steps num_sleep_sec num_M_particles LL/LC/CL/CC \n", name);
    printf("LL/LC/CL/CC is used to set benchmark mode, stands for LINEAR_LINEAR, LINEAR_COMPOUND, COMPOUND_LINEAR, COMPOUND_COMPOUND");
}

int main(int argc, char* argv[]) {
    char *file_name = argv[1];

    MPI_Init(&argc, &argv);
    int my_rank, num_procs, nts, i, j, sleep_time = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    MPI_Comm comm = MPI_COMM_WORLD;
    MPI_Info info = MPI_INFO_NULL;

    if (argc != 6) {
        print_usage(argv[0]);
        return 0;
    }
    DEBUG_PRINT
    nts = atoi(argv[2]);
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
    } else {
        printf("Benchmark mode can only be one of these: LL, LC, CL, CC \n");
        return 0;
    }

    if (my_rank == 0) {
        printf("Number of paritcles: %ld \n", NUM_PARTICLES);
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
        printf("Total particle number = %lldM, total write size = %luMB\n", TOTAL_PARTICLES / (1024 * 1024),
                total_write_size / (1024 * 1024));

    hid_t fapl = set_fapl();

    H5Pset_fapl_mpio(fapl, comm, info);

    set_metadata(fapl);

    unsigned long t1 = get_time_usec(); // t1 - t0: cost of settings
    hid_t file_id = H5Fcreate(file_name, H5F_ACC_TRUNC, H5P_DEFAULT, fapl);
    H5Pclose(fapl);

    if (my_rank == 0)
        printf("Opened HDF5 file... \n");

    hid_t filespace, memspace, plist_id;
    set_select_spaces_default(&filespace, &memspace, &plist_id);

    MPI_Barrier(MPI_COMM_WORLD);

    unsigned long t2 = get_time_usec(); // t2 - t1: metadata: creating/opening

    unsigned long raw_write_time;
    _run_time_steps(mode, NUM_PARTICLES, nts, my_rank, sleep_time, file_id, plist_id, filespace, memspace,
            &raw_write_time);

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
        printf("Total sleep time %ds\n", sleep_time * (nts - 1));
        printf("RR: Raw write time = %lu ms, RR = %lu MB/sec \n", raw_write_time / 1000,
                total_write_size / raw_write_time);
        printf("Core metadata time = %lu ms\n",
                (t3 - t2 - raw_write_time - sleep_time * (nts - 1) * 1000 * 1000) / 1000);
        printf("Opening + closing time = %lu ms\n", (t1 - t0 + t4 - t3) / 1000);
        printf("OR (observed rate):  = %lu ms, OR = %lu MB/sec\n", (t4 - t1) / 1000 - (nts - 1) * 1000,
                total_write_size / (t4 - t1 - (nts - 1) * 1000 * 1000));
        printf("OCT(observed completion time) = %lu ms\n", (t4 - t0) / 1000);
        printf("\n");
    }

    MPI_Finalize();

    return 0;
}
