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
#include <sys/time.h>
#include "../commons/h5bench_util.h"

// A simple timer based on gettimeofday

#define DTYPE float

extern struct timeval start_time[5];
extern float elapse[5];
#define timer_on(id) gettimeofday (&start_time[id], NULL)
#define timer_off(id) 	\
		{	\
		     struct timeval result, now; \
		     gettimeofday (&now, NULL);  \
		     timeval_subtract(&result, &now, &start_time[id]);	\
		     elapse[id] += result.tv_sec+ (DTYPE) (result.tv_usec)/1000000.;	\
		}

#define timer_msg(id, msg) \
	printf("%f seconds elapsed in %s\n", (DTYPE)(elapse[id]), msg);  \

#define timer_reset(id) elapse[id] = 0

/* Subtract the `struct timeval' values X and Y,
   storing the result in RESULT.
   Return 1 if the difference is negative, otherwise 0.  */

int
timeval_subtract (struct timeval *result, struct timeval *x, struct timeval *y)
{
  /* Perform the carry for the later subtraction by updating y. */
  if (x->tv_usec < y->tv_usec) {
    int nsec = (y->tv_usec - x->tv_usec) / 1000000 + 1;
    y->tv_usec -= 1000000 * nsec;
    y->tv_sec += nsec;
  }
  if (x->tv_usec - y->tv_usec > 1000000) {
    int nsec = (y->tv_usec - x->tv_usec) / 1000000;
    y->tv_usec += 1000000 * nsec;
    y->tv_sec -= nsec;
  }

  /* Compute the time remaining to wait.
     tv_usec is certainly positive. */
  result->tv_sec = x->tv_sec - y->tv_sec;
  result->tv_usec = x->tv_usec - y->tv_usec;

  /* Return 1 if result is negative. */
  return x->tv_sec < y->tv_sec;
}


struct timeval start_time[5];
float elapse[5];

// HDF5 specific declerations
herr_t ierr;

// Variables and dimensions
long numparticles = 8388608;	// 8  meg particles per process
long long total_particles, offset;

float *x, *y, *z;
float *px, *py, *pz;
int *id1, *id2;
int x_dim = 64;
int y_dim = 64;
int z_dim = 64;

// Uniform random number
double uniform_random_number()
{
    return (((double)rand())/((double)(RAND_MAX)));
}

// Initialize particle data
void init_particles ()
{
    int i;
    for (i=0; i<numparticles; i++)
    {
        id1[i] = i;
        id2[i] = i*2;
        x[i] = uniform_random_number()*x_dim;
        y[i] = uniform_random_number()*y_dim;
        z[i] = ((double)id1[i]/numparticles)*z_dim;
        px[i] = uniform_random_number()*x_dim;
        py[i] = uniform_random_number()*y_dim;
        pz[i] = ((double)id2[i]/numparticles)*z_dim;
    }
}

// Create HDF5 file and write data
void create_and_write_synthetic_h5_data(int rank, hid_t loc, hid_t *dset_ids, hid_t filespace, hid_t memspace, hid_t plist_id)
{
    int i;
    // Note: printf statements are inserted basically
    // to check the progress. Other than that they can be removed
    dset_ids[0] = H5Dcreate(loc, "x", H5T_NATIVE_DOUBLE, filespace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    dset_ids[1] = H5Dcreate(loc, "y", H5T_NATIVE_DOUBLE, filespace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    dset_ids[2] = H5Dcreate(loc, "z", H5T_NATIVE_DOUBLE, filespace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    dset_ids[3] = H5Dcreate(loc, "id1", H5T_NATIVE_INT, filespace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    dset_ids[4] = H5Dcreate(loc, "id2", H5T_NATIVE_INT, filespace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    dset_ids[5] = H5Dcreate(loc, "px", H5T_NATIVE_DOUBLE, filespace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    dset_ids[6] = H5Dcreate(loc, "py", H5T_NATIVE_DOUBLE, filespace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    dset_ids[7] = H5Dcreate(loc, "pz", H5T_NATIVE_DOUBLE, filespace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

    ierr = H5Dwrite(dset_ids[0], H5T_NATIVE_FLOAT, memspace, filespace, plist_id, x);
    /* if (rank == 0) printf ("Written variable 1 \n"); */

    ierr = H5Dwrite(dset_ids[1], H5T_NATIVE_FLOAT, memspace, filespace, plist_id, y);
    /* if (rank == 0) printf ("Written variable 2 \n"); */

    ierr = H5Dwrite(dset_ids[2], H5T_NATIVE_FLOAT, memspace, filespace, plist_id, z);
    /* if (rank == 0) printf ("Written variable 3 \n"); */

    ierr = H5Dwrite(dset_ids[3], H5T_NATIVE_INT, memspace, filespace, plist_id, id1);
    /* if (rank == 0) printf ("Written variable 4 \n"); */

    ierr = H5Dwrite(dset_ids[4], H5T_NATIVE_INT, memspace, filespace, plist_id, id2);
    /* if (rank == 0) printf ("Written variable 5 \n"); */

    ierr = H5Dwrite(dset_ids[5], H5T_NATIVE_FLOAT, memspace, filespace, plist_id, px);
    /* if (rank == 0) printf ("Written variable 6 \n"); */

    ierr = H5Dwrite(dset_ids[6], H5T_NATIVE_FLOAT, memspace, filespace, plist_id, py);
    /* if (rank == 0) printf ("Written variable 7 \n"); */

    ierr = H5Dwrite(dset_ids[7], H5T_NATIVE_FLOAT, memspace, filespace, plist_id, pz);
    /* if (rank == 0) printf ("Written variable 8 \n"); */
    if (rank == 0) printf ("  Finished written 8 variables \n");

}


int set_space_1D_interleave(){
    return -1;
}

int run_time_steps(int timestep_cnt, int my_rank, int sleep_time,
        hid_t file_id, hid_t plist_id, hid_t filespace, hid_t memspace,
        unsigned long* raw_write_time_out) {

    char grp_name[128];
    unsigned long  rt_start, rt_end;
    hid_t **dset_ids = (hid_t**)calloc(timestep_cnt, sizeof(hid_t*));
    hid_t *grp_ids  = (hid_t*)calloc(timestep_cnt, sizeof(hid_t));
    *raw_write_time_out = 0;
    for (int i = 0; i < timestep_cnt; i++) {
        sprintf(grp_name, "Timestep_%d", i);
        grp_ids[i] = H5Gcreate2(file_id, grp_name, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

        if (my_rank == 0)
            printf ("Writing %s ... \n", grp_name);

        dset_ids[i] = (hid_t*)calloc(8, sizeof(hid_t));


        rt_start = get_time_usec();
        create_and_write_synthetic_h5_data(my_rank, grp_ids[i], dset_ids[i], filespace, memspace, plist_id);
        rt_end = get_time_usec();

        *raw_write_time_out += (rt_end - rt_start);

        if (my_rank == 0)
            printf("create and write\n");

        fflush(stdout);
        if (i != timestep_cnt - 1) {
            if (my_rank == 0) printf ("  sleep for %ds\n", sleep_time);
            if (sleep_time > 0) sleep(sleep_time);
        }

        for (int j = 0; j < 8; j++)
            H5Dclose(dset_ids[i][j]);
        H5Gclose(grp_ids[i]);

        MPI_Barrier (MPI_COMM_WORLD);

        if (my_rank == 0)
            printf("write and wait\n");
    }
    return -1;
}
int set_select_spaces(hid_t* filespace_out, hid_t* memspace_out, hid_t* plist_id_out){
    *filespace_out = H5Screate_simple(1, (hsize_t *) &total_particles, NULL);//= world_size * numparticles
    *memspace_out =  H5Screate_simple(1, (hsize_t *) &numparticles, NULL);

    *plist_id_out = H5Pcreate(H5P_DATASET_XFER);
    H5Pset_dxpl_mpio(*plist_id_out, H5FD_MPIO_COLLECTIVE);
    H5Sselect_hyperslab(*filespace_out, H5S_SELECT_SET, (hsize_t *) &offset, NULL, (hsize_t *) &numparticles, NULL);
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
void print_usage(char *name)
{
    printf("Usage: %s /path/to/file #timestep sleep_sec [# mega particles]\n", name);
}

int main (int argc, char* argv[])
{
    char *file_name = argv[1];

    MPI_Init(&argc,&argv);
    int my_rank, num_procs, nts, i, j, sleep_time = 0;
    MPI_Comm_rank (MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size (MPI_COMM_WORLD, &num_procs);

    MPI_Comm comm  = MPI_COMM_WORLD;
    MPI_Info info  = MPI_INFO_NULL;

    if (argc < 4) {
        print_usage(argv[0]);
        return 0;
    }

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

    if (argc == 5) {
        numparticles = (atoi (argv[4]))*1024*1024;
    }
    else {
        numparticles = 8*1024*1024;
    }

    if (my_rank == 0) {
        printf ("Number of paritcles: %ld \n", numparticles);
    }

    x=(float*)malloc(numparticles*sizeof(double));
    y=(float*)malloc(numparticles*sizeof(double));
    z=(float*)malloc(numparticles*sizeof(double));

    px=(float*)malloc(numparticles*sizeof(double));
    py=(float*)malloc(numparticles*sizeof(double));
    pz=(float*)malloc(numparticles*sizeof(double));

    id1=(int*)malloc(numparticles*sizeof(int));
    id2=(int*)malloc(numparticles*sizeof(int));

    unsigned long total_write_size = num_procs * nts * numparticles * (6* sizeof(double) + 2*sizeof(int));

    init_particles ();

    if (my_rank == 0)
        printf ("Finished initializeing particles \n");

    MPI_Barrier (MPI_COMM_WORLD);
    timer_on (0);
    unsigned long t0 = get_time_usec();
    MPI_Allreduce(&numparticles, &total_particles, 1, MPI_LONG_LONG, MPI_SUM, comm);
    MPI_Scan(&numparticles, &offset, 1, MPI_LONG_LONG, MPI_SUM, comm);
    offset -= numparticles;

    if(my_rank == 0)
        printf("Total particle number = %lu, total write size = %lu\n", total_particles, total_write_size);

    hid_t fapl = set_fapl();

    H5Pset_fapl_mpio(fapl, comm, info);

    set_metadata(fapl);

    unsigned long t1 = get_time_usec(); // t1 - t0: cost of settings
    hid_t file_id = H5Fcreate(file_name, H5F_ACC_TRUNC, H5P_DEFAULT, fapl);
    H5Pclose(fapl);

    if (my_rank == 0)
        printf ("Opened HDF5 file... \n");

    hid_t filespace, memspace, plist_id;
    set_select_spaces(&filespace, &memspace, &plist_id);

    MPI_Barrier (MPI_COMM_WORLD);    

    unsigned long t2 = get_time_usec(); // t2 - t1: metadata: creating/opening
    unsigned long raw_write_time;
    run_time_steps(nts, my_rank, sleep_time, file_id, plist_id, filespace, memspace, &raw_write_time);
    
    unsigned long t3 = get_time_usec();// t3 - t2: writting data, including metadata
    
    if (my_rank == 0) {
        printf ("\nTiming results with %d ranks\n", num_procs);
        timer_msg (1, "total running time");
        printf("Total running time = %lu ms\n", (t3 - t0)/1000);
    }

    H5Sclose(memspace);
    H5Sclose(filespace);
    H5Pclose(plist_id);
    H5Fclose(file_id);

    MPI_Barrier (MPI_COMM_WORLD);
    unsigned long t4 = get_time_usec();

    if (my_rank == 0) {
        printf ("\nTiming results\n");
        printf("Total sleep time %ds\n", sleep_time*(nts-1));
        printf("RR: Raw write time = %lu ms, RR = %lu MB/sec \n", raw_write_time/1000, total_write_size/raw_write_time);
        printf("Core metadata time = %lu ms\n", (t3 - t2 - raw_write_time - sleep_time*(nts-1)*1000*1000)/1000);
        printf("Opening + closing time = %lu ms\n", (t1 - t0 + t4 - t3)/1000);
        printf("OR (observed rate):  = %lu ms, OR = %lu MB/sec\n", (t4 - t1)/1000 - (nts-1)*1000, total_write_size/(t4 - t1 - (nts-1)*1000*1000));
        printf("OCT(observed completion time) = %lu ms\n", (t4-t0)/1000);
        printf ("\n");
    }

    free(x);
    free(y);
    free(z);
    free(px);
    free(py);
    free(pz);
    free(id1);
    free(id2);


    MPI_Finalize();

    return 0;
}
