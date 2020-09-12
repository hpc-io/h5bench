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
#include <string.h>
#include "../commons/h5bench_util.h"

// Global Variables and dimensions
long long NUM_PARTICLES = 0, FILE_OFFSET = 0;
long long TOTAL_PARTICLES = 0;

int NUM_RANKS, MY_RANK, NUM_TIMESTEPS;

hid_t PARTICLE_COMPOUND_TYPE;
hid_t PARTICLE_COMPOUND_TYPE_SEPARATES[8];

herr_t ierr;

data_contig_md* buf_struct;

void print_data(int n) {
    int i;
    for (i = 0; i < n; i++)
        printf("sample data: %f %f %f %d %d %f %f %f\n",
            buf_struct->x[i], buf_struct->y[i], buf_struct->z[i],
            buf_struct->id_1[i], buf_struct->id_2[i],
            buf_struct->px[i], buf_struct->py[i], buf_struct->pz[i]);
}

// Create HDF5 file and read data
void read_h5_data(int rank, hid_t loc, hid_t filespace, hid_t memspace) {
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

int _set_dataspace_seq_read(unsigned long read_elem_cnt, hid_t filespace_in, hid_t* memspace_out){
    *memspace_out =  H5Screate_simple(1, (hsize_t *) &read_elem_cnt, NULL);
    DEBUG_PRINT
    H5Sselect_hyperslab(filespace_in, H5S_SELECT_SET, (hsize_t *) &FILE_OFFSET, NULL,
            (hsize_t *) &read_elem_cnt, NULL);
    return 0;
}

//returns actual rounded read element count.
unsigned long _set_dataspace_random_read(unsigned long read_elem_cnt, hid_t filespace_in, hid_t* memspace_out){
    unsigned long stride = 37;//stride must be greater than block size??
    unsigned long block_size = 29; //in element
    unsigned long block_cnt = read_elem_cnt/block_size;
    unsigned long actual_elem_cnt = block_cnt * block_size;
    *memspace_out =  H5Screate_simple(1, (hsize_t *) &actual_elem_cnt, NULL);
    DEBUG_PRINT

    printf("read_elem_cnt = %lu, block_size = %lu, block_cnt = %lu\n", read_elem_cnt, block_size, block_cnt);

    H5Sselect_hyperslab(filespace_in,
            H5S_SELECT_SET,
            (hsize_t *) &FILE_OFFSET, //start-offset
            (hsize_t *) &stride, //stride
            (hsize_t *) &block_cnt, //block cnt
            (hsize_t*) &block_size); //block size

    return actual_elem_cnt;
}

//filespace should be read from the file.
unsigned long _set_dataspace_seq_2D(hid_t* memspace_out,
        unsigned long long dim_1, unsigned long long dim_2){
    return 0;
}

unsigned long _set_dataspace_seq_3D(hid_t* memspace_out,
        unsigned long long dim_1, unsigned long long dim_2, unsigned long long dim_3){
    return 0;
}

hid_t get_filespace(hid_t file_id){
    char* grp_name = "/Timestep_0";
    char* ds_name = "px";
    hid_t gid = H5Gopen(file_id, grp_name, H5P_DEFAULT);
    hid_t dsid = H5Dopen2(gid, ds_name, H5P_DEFAULT);
    hid_t filespace = H5Dget_space(dsid);
    H5Dclose(dsid);
    H5Gclose(gid);
    return filespace;
}

typedef enum read_pattern{
    CONTIG_1D,
    RANDOM_1D
}read_pattern;

unsigned long set_dataspace(read_pattern pattern, unsigned long read_elem_cnt, hid_t filespace_in, hid_t* memspace_out){
    unsigned long actual_read_cnt = 0;
    switch(pattern){
        case CONTIG_1D:
            _set_dataspace_seq_read(read_elem_cnt, filespace_in, memspace_out);
            actual_read_cnt = read_elem_cnt;
            break;
        case RANDOM_1D:
            actual_read_cnt = _set_dataspace_random_read(read_elem_cnt, filespace_in, memspace_out);
            break;

        default:
            break;
    }
    return actual_read_cnt;
}

int _run_benchmark_read(hid_t file_id, hid_t fapl, hid_t gapl, read_pattern pattern, int nts, int sleep_time, unsigned long read_elem_cnt,
         unsigned long* raw_read_time_out, unsigned long* total_data_size_out){
    *raw_read_time_out = 0;

    hid_t grp;
    char grp_name[128];
    unsigned long rt1 = 0, rt2 = 0;

    unsigned long actual_read_cnt = 0;
    hid_t filespace, memspace;

    filespace = get_filespace(file_id);

    actual_read_cnt = set_dataspace(pattern, read_elem_cnt, filespace, &memspace);
    for (int i = 0; i < nts; i++) {
        DEBUG_PRINT
        sprintf(grp_name, "Timestep_%d", i);
        grp = H5Gopen(file_id, grp_name, gapl);

        if (MY_RANK == 0) printf ("Reading %s ... \n", grp_name);

        rt1 = get_time_usec();
        read_h5_data(MY_RANK, grp, filespace, memspace);
        rt2 = get_time_usec();
        *raw_read_time_out += (rt2 - rt1);

        if (i != 0) {
            if (MY_RANK == 0) printf ("  sleep for %ds\n", sleep_time);
            sleep(sleep_time);
        }
        H5Gclose(grp);
        MPI_Barrier(MPI_COMM_WORLD);
    }
    *total_data_size_out = NUM_RANKS * NUM_TIMESTEPS * actual_read_cnt * (6 * sizeof(float) + 2 * sizeof(int));
    H5Sclose(memspace);
    H5Sclose(filespace);
    return -1;
}

void set_pl(hid_t* fapl, hid_t* gapl ){
    *fapl = H5Pcreate(H5P_FILE_ACCESS);
    H5Pset_fapl_mpio(*fapl, MPI_COMM_WORLD, MPI_INFO_NULL);
    H5Pset_all_coll_metadata_ops(*fapl, true);
    H5Pset_coll_metadata_write(*fapl, true);
    *gapl = H5Pcreate(H5P_GROUP_ACCESS);
    H5Pset_all_coll_metadata_ops(*gapl, true);
}

void print_usage(char *name){
    printf("Usage: %s /path/to/file #timestep sleep_sec [# mega particles]\n", name);
}

int main (int argc, char* argv[])
{

    MPI_Init(&argc,&argv);
    if(argc < 3) {
        printf("Usage: ./%s /path/to/file #timestep [# mega particles]\n", argv[0]);
        return 0;
    }
    int sleep_time;

    MPI_Comm_rank (MPI_COMM_WORLD, &MY_RANK);
    MPI_Comm_size (MPI_COMM_WORLD, &NUM_RANKS);

    char *file_name = argv[1];
    NUM_TIMESTEPS = atoi(argv[2]);

    if (NUM_TIMESTEPS <= 0) {
        printf("Usage: ./%s /path/to/file #timestep [# mega particles]\n", argv[0]);
        return 0;
    }

    sleep_time = atoi(argv[3]);
    if (sleep_time < 0) {
        print_usage(argv[0]);
        return 0;
    }
    int num_particles = atoi(argv[4]);
    NUM_PARTICLES = num_particles * 1024 * 1024;

    int read_elem_cnt_m = atoi(argv[5]);
    if(read_elem_cnt_m > num_particles){
        printf("read_elem_cnt_m <= num_particles must hold.\n");
        return 0;
    }

    unsigned long read_elem_cnt = read_elem_cnt_m * 1024 * 1024;

    read_pattern pattern;
    if (strcmp("S", argv[6]) == 0)
        pattern = CONTIG_1D;
    else {
        pattern = RANDOM_1D;
    }

    MPI_Info info  = MPI_INFO_NULL;
    if (MY_RANK == 0) {
        printf ("Number of paritcles: %lld \n", NUM_PARTICLES);
    }
    MPI_Barrier (MPI_COMM_WORLD);
//    timer_on (0);
    unsigned long t0 = get_time_usec();
    MPI_Allreduce(&NUM_PARTICLES, &TOTAL_PARTICLES, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
    MPI_Scan(&NUM_PARTICLES, &FILE_OFFSET, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
    FILE_OFFSET -= NUM_PARTICLES;


    hid_t fapl, gapl;
    set_pl(&fapl, &gapl);

    buf_struct = prepare_contig_memory(read_elem_cnt, 0, 0, 0);

    unsigned long t1 = get_time_usec();
    hid_t file_id = H5Fopen(file_name, H5F_ACC_RDONLY, fapl);
    if(file_id < 0) {
        printf("Error with opening file [%s]!\n", file_name);
        goto done;
    }

    if (MY_RANK == 0) printf ("Opened HDF5 file ... [%s]\n", file_name);


    unsigned long raw_read_time, total_data_size;
    unsigned long t2 = get_time_usec();
    _run_benchmark_read(file_id, fapl, gapl, pattern, NUM_TIMESTEPS, sleep_time, read_elem_cnt,
            &raw_read_time, &total_data_size);
    unsigned long t3 = get_time_usec();

DEBUG_PRINT
    MPI_Barrier (MPI_COMM_WORLD);

    H5Pclose(fapl);
    H5Pclose(gapl);
    H5Fclose(file_id);

    MPI_Barrier (MPI_COMM_WORLD);
    unsigned long t4 = get_time_usec();

    if (MY_RANK == 0) {
        printf("\n =================  Performance results  =================\n");
        printf("Total sleep time %ds, total read size = %lu MB\n", sleep_time * (NUM_TIMESTEPS - 1), NUM_RANKS * total_data_size/(1024*1024));
        printf("RR: Raw read time = %lu ms, RR = %lu MB/sec \n", raw_read_time / 1000, total_data_size / raw_read_time);
        printf("Core metadata time = %lu ms\n",
                (t3 - t2 - raw_read_time - sleep_time * (NUM_TIMESTEPS - 1) * 1000 * 1000) / 1000);
        printf("OR (observed rate):  = %lu ms, OR = %lu MB/sec\n", (t4 - t1) / 1000 - (NUM_TIMESTEPS - 1) * 1000,
                total_data_size / (t4 - t1 - (NUM_TIMESTEPS - 1) * 1000 * 1000));
        printf("OCT(observed completion time) = %lu ms\n", (t4 - t0) / 1000);
        printf("\n");
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
