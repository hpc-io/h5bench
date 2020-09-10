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
//      Each process reads a specified number of particles into
//      a hdf5 output file using only HDF5 calls
// Author:  Suren Byna <SByna@lbl.gov>
//      Lawrence Berkeley National Laboratory, Berkeley, CA
// Created: in 2011
// Modified:    01/06/2014 --> Removed all H5Part calls and using HDF5 calls
//              02/19/2019 --> Add option to read multiple timesteps of data - Tang

#include <math.h>
#include <hdf5.h>
#include <stdlib.h>
#include <sys/time.h>
#include <stdio.h>
#include <unistd.h>
#include "../commons/h5bench_util.h"

// Global Variables and dimensions
long long NUM_PARTICLES = 0, FILE_OFFSET;   // 8  meg particles per process
long long TOTAL_PARTICLES;
int NUM_RANKS, MY_RANK, NUM_TIMESTEPS;

hid_t PARTICLE_COMPOUND_TYPE;
hid_t PARTICLE_COMPOUND_TYPE_SEPARATES[8];

// HDF5 specific declerations
herr_t ierr;

// Variables and dimensions
long numparticles = 8388608;    // 8  meg particles per process
long long total_particles, offset;
data_contig_md* buf_struct;

void print_data(int n)
{
    int i;
    for (i = 0; i < n; i++)
        printf("sample data: %f %f %f %d %d %f %f %f\n",
            buf_struct->x[i], buf_struct->y[i], buf_struct->z[i],
            buf_struct->id_1[i], buf_struct->id_2[i],
            buf_struct->px[i], buf_struct->py[i], buf_struct->pz[i]);
}

// Create HDF5 file and read data
void read_h5_data(int rank, hid_t loc, hid_t filespace, hid_t memspace)
{
    hid_t dset_id, dapl;
    dapl = H5Pcreate(H5P_DATASET_ACCESS);
    H5Pset_all_coll_metadata_ops(dapl, true);
    DEBUG_PRINT
    dset_id = H5Dopen2(loc, "x", dapl);
    DEBUG_PRINT
    ierr = H5Dread(dset_id, H5T_NATIVE_FLOAT, memspace, filespace, H5P_DEFAULT, buf_struct->x);
    DEBUG_PRINT
    H5Dclose(dset_id);

    dset_id = H5Dopen2(loc, "y", dapl);
    ierr = H5Dread(dset_id, H5T_NATIVE_FLOAT, memspace, filespace, H5P_DEFAULT, buf_struct->y);
    H5Dclose(dset_id);

    dset_id = H5Dopen2(loc, "z", dapl);
    ierr = H5Dread(dset_id, H5T_NATIVE_FLOAT, memspace, filespace, H5P_DEFAULT, buf_struct->z);
    H5Dclose(dset_id);

    dset_id = H5Dopen2(loc, "id_1", dapl);
    ierr = H5Dread(dset_id, H5T_NATIVE_INT, memspace, filespace, H5P_DEFAULT, buf_struct->id_1);
    H5Dclose(dset_id);

    dset_id = H5Dopen2(loc, "id_2", dapl);
    ierr = H5Dread(dset_id, H5T_NATIVE_INT, memspace, filespace, H5P_DEFAULT, buf_struct->id_2);
    H5Dclose(dset_id);

    dset_id = H5Dopen2(loc, "px", dapl);
    ierr = H5Dread(dset_id, H5T_NATIVE_FLOAT, memspace, filespace, H5P_DEFAULT, buf_struct->px);
    H5Dclose(dset_id);

    dset_id = H5Dopen2(loc, "py", dapl);
    ierr = H5Dread(dset_id, H5T_NATIVE_FLOAT, memspace, filespace, H5P_DEFAULT, buf_struct->py);
    H5Dclose(dset_id);

    dset_id = H5Dopen2(loc, "pz", dapl);
    ierr = H5Dread(dset_id, H5T_NATIVE_FLOAT, memspace, filespace, H5P_DEFAULT, buf_struct->pz);
    H5Dclose(dset_id);

    if (rank == 0) printf ("  Read 8 variable completed\n");

    H5Pclose(dapl);
    print_data(3);
}

int set_dataspace(hid_t* filespace_out, hid_t* memspace_out){
    *filespace_out = H5Screate_simple(1, (hsize_t *) &total_particles, NULL);//replace with getspace.
    *memspace_out =  H5Screate_simple(1, (hsize_t *) &numparticles, NULL);
    DEBUG_PRINT
    H5Sselect_hyperslab(*filespace_out, H5S_SELECT_SET, (hsize_t *) &offset, NULL, (hsize_t *) &numparticles, NULL);
    return 0;
}

int _run_benchmark_read(hid_t file_id, hid_t fapl, hid_t gapl, int nts, int sleep_time){
    hid_t filespace, memspace;
    set_dataspace(&filespace, &memspace);
    hid_t grp;
    char grp_name[128];
    for (int i = 0; i < nts; i++) {
        DEBUG_PRINT
        sprintf(grp_name, "Timestep_%d", i);
        grp = H5Gopen(file_id, grp_name, gapl);
        DEBUG_PRINT
        if (MY_RANK == 0)
            printf ("Reading %s ... \n", grp_name);
        read_h5_data(MY_RANK, grp, filespace, memspace);
        DEBUG_PRINT
        if (i != 0) {
            if (MY_RANK == 0) printf ("  sleep for %ds\n", sleep_time);
            sleep(sleep_time);
        }
        H5Gclose(grp);
        MPI_Barrier(MPI_COMM_WORLD);
    }
    H5Sclose(memspace);
    H5Sclose(filespace);
    return -1;
}

void print_usage(char *name)
{
    printf("Usage: %s /path/to/file #timestep sleep_sec [# mega particles]\n", name);
}

int main (int argc, char* argv[])
{

    MPI_Init(&argc,&argv);
    if(argc < 3) {
        printf("Usage: ./%s /path/to/file #timestep [# mega particles]\n", argv[0]);
        return 0;
    }
    int my_rank, num_procs, nts, i, sleep_time;
    hid_t file_id;
    hid_t filespace, memspace;
    hid_t fapl, gapl;
    MPI_Comm_rank (MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size (MPI_COMM_WORLD, &num_procs);

    char *file_name = argv[1];

    nts = atoi(argv[2]);
    if (nts <= 0) {
        printf("Usage: ./%s /path/to/file #timestep [# mega particles]\n", argv[0]);
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

    MPI_Info info  = MPI_INFO_NULL;
    if (my_rank == 0) {
        printf ("Number of paritcles: %ld \n", numparticles);
    }
    MPI_Barrier (MPI_COMM_WORLD);
//    timer_on (0);

    MPI_Allreduce(&numparticles, &total_particles, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
    MPI_Scan(&numparticles, &offset, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
    offset -= numparticles;

DEBUG_PRINT
    /* Set up FAPL */
    if((fapl = H5Pcreate(H5P_FILE_ACCESS)) < 0)
        goto error;
    if(H5Pset_fapl_mpio(fapl, MPI_COMM_WORLD, MPI_INFO_NULL) < 0)
        goto error;
    H5Pset_all_coll_metadata_ops(fapl, true);
    H5Pset_coll_metadata_write(fapl, true);
DEBUG_PRINT
    if((gapl = H5Pcreate(H5P_GROUP_ACCESS)) < 0)
        goto error;
    if(H5Pset_all_coll_metadata_ops(gapl, true) < 0)
        goto error;
DEBUG_PRINT

    buf_struct = prepare_contig_memory(numparticles, 0, 0, 0);
    /* Open file */
    file_id = H5Fopen(file_name, H5F_ACC_RDONLY, fapl);
    if(file_id < 0) {
        printf("Error with opening file [%s]!\n", file_name);
        goto done;
    }
DEBUG_PRINT
    if (my_rank == 0)
        printf ("Opened HDF5 file ... [%s]\n", file_name);

    _run_benchmark_read(file_id, fapl, gapl, nts, sleep_time);


DEBUG_PRINT
    MPI_Barrier (MPI_COMM_WORLD);


    H5Pclose(fapl);
    H5Pclose(gapl);
    H5Fclose(file_id);

    MPI_Barrier (MPI_COMM_WORLD);
//    timer_off (0);
    if (my_rank == 0)
    {
        printf ("\nTiming results\n");
        printf("Total sleep time %ds\n", sleep_time*(nts-1));
//        timer_msg (1, "just reading data");
//        timer_msg (0, "opening, reading, closing file");
        printf ("\n");
    }

    free_contig_memory(buf_struct);

error:
    H5E_BEGIN_TRY {
        H5Fclose(file_id);
        H5Pclose(fapl);
    } H5E_END_TRY;

done:
    H5close();
    MPI_Finalize();

    return 0;
}
