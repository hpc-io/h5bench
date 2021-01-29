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
#include <assert.h>
#include "../commons/h5bench_util.h"
#include "../commons/async_adaptor.h"
// Global Variables and dimensions
long long NUM_PARTICLES = 0, FILE_OFFSET = 0;
long long TOTAL_PARTICLES = 0;

int NUM_RANKS, MY_RANK, NUM_TIMESTEPS;
hid_t ES_ID;
hid_t PARTICLE_COMPOUND_TYPE;
hid_t PARTICLE_COMPOUND_TYPE_SEPARATES[8];

herr_t ierr;

data_contig_md* BUF_STRUCT;

void print_data(int n) {
    int i;
    for (i = 0; i < n; i++)
        printf("sample data: %f %f %f %d %f %f %f %f\n",
            BUF_STRUCT->x[i], BUF_STRUCT->y[i], BUF_STRUCT->z[i],
            BUF_STRUCT->id_1[i], BUF_STRUCT->id_2[i],
            BUF_STRUCT->px[i], BUF_STRUCT->py[i], BUF_STRUCT->pz[i]);
}

// Create HDF5 file and read data
void read_h5_data(int rank, hid_t loc, hid_t filespace, hid_t memspace, unsigned long* read_time) {
    hid_t dset_id, dapl;
    unsigned long core_read_time = 0, start_read, end;

    dapl = H5Pcreate(H5P_DATASET_ACCESS);
    H5Pset_all_coll_metadata_ops(dapl, true);

    dset_id = H5Dopen_async(loc, "x", dapl, ES_ID);
    start_read = get_time_usec();
    ierr = H5Dread_async(dset_id, H5T_NATIVE_FLOAT, memspace, filespace, H5P_DEFAULT, BUF_STRUCT->x, ES_ID);
    core_read_time += (get_time_usec() - start_read);

    H5Dclose_async(dset_id, ES_ID);

    dset_id = H5Dopen_async(loc, "y", dapl, ES_ID);
    start_read = get_time_usec();
    ierr = H5Dread_async(dset_id, H5T_NATIVE_FLOAT, memspace, filespace, H5P_DEFAULT, BUF_STRUCT->y, ES_ID);
    core_read_time += (get_time_usec() - start_read);
    H5Dclose_async(dset_id, ES_ID);

    dset_id = H5Dopen_async(loc, "z", dapl, ES_ID);
    start_read = get_time_usec();
    ierr = H5Dread_async(dset_id, H5T_NATIVE_FLOAT, memspace, filespace, H5P_DEFAULT, BUF_STRUCT->z, ES_ID);
    core_read_time += (get_time_usec() - start_read);
    H5Dclose_async(dset_id, ES_ID);

    dset_id = H5Dopen_async(loc, "id_1", dapl, ES_ID);
    start_read = get_time_usec();
    ierr = H5Dread_async(dset_id, H5T_NATIVE_INT, memspace, filespace, H5P_DEFAULT, BUF_STRUCT->id_1, ES_ID);
    core_read_time += (get_time_usec() - start_read);
    H5Dclose_async(dset_id, ES_ID);

    dset_id = H5Dopen_async(loc, "id_2", dapl, ES_ID);
    start_read = get_time_usec();
    ierr = H5Dread_async(dset_id, H5T_NATIVE_INT, memspace, filespace, H5P_DEFAULT, BUF_STRUCT->id_2, ES_ID);
    core_read_time += (get_time_usec() - start_read);
    H5Dclose_async(dset_id, ES_ID);

    dset_id = H5Dopen_async(loc, "px", dapl, ES_ID);
    start_read = get_time_usec();
    ierr = H5Dread_async(dset_id, H5T_NATIVE_FLOAT, memspace, filespace, H5P_DEFAULT, BUF_STRUCT->px, ES_ID);
    core_read_time += (get_time_usec() - start_read);
    H5Dclose_async(dset_id, ES_ID);

    dset_id = H5Dopen_async(loc, "py", dapl, ES_ID);
    start_read = get_time_usec();
    ierr = H5Dread_async(dset_id, H5T_NATIVE_FLOAT, memspace, filespace, H5P_DEFAULT, BUF_STRUCT->py, ES_ID);
    core_read_time += (get_time_usec() - start_read);
    H5Dclose_async(dset_id, ES_ID);

    dset_id = H5Dopen_async(loc, "pz", dapl, ES_ID);
    start_read = get_time_usec();
    ierr = H5Dread_async(dset_id, H5T_NATIVE_FLOAT, memspace, filespace, H5P_DEFAULT, BUF_STRUCT->pz, ES_ID);
    core_read_time += (get_time_usec() - start_read);
    H5Dclose_async(dset_id, ES_ID);

    if (rank == 0) printf ("  Read 8 variable completed\n");

    *read_time = core_read_time;
    H5Pclose(dapl);
    //if(MY_RANK == 0) print_data(3); //print sample data
}

int _set_dataspace_seq_read(unsigned long read_elem_cnt, hid_t* filespace_in, hid_t* memspace_out){
    *memspace_out =  H5Screate_simple(1, (hsize_t *) &read_elem_cnt, NULL);

    H5Sselect_hyperslab(*filespace_in, H5S_SELECT_SET, (hsize_t *) &FILE_OFFSET, NULL,
            (hsize_t *) &read_elem_cnt, NULL);
    return read_elem_cnt;
}

//returns actual rounded read element count.
unsigned long _set_dataspace_strided_read(unsigned long read_elem_cnt, unsigned long stride, unsigned long block_size,
        hid_t* filespace_in, hid_t* memspace_out){
    unsigned long block_cnt = read_elem_cnt/(block_size + stride);
    unsigned long actual_elem_cnt = block_cnt * block_size;
    *memspace_out =  H5Screate_simple(1, (hsize_t *) &actual_elem_cnt, NULL);

    if(MY_RANK == 0)
        printf("Stride parameters: read_elem_cnt = %lu, actual_elem_cnt = %lu, block_size = %lu, block_cnt = %lu\n", read_elem_cnt, actual_elem_cnt, block_size, block_cnt);

    H5Sselect_hyperslab(*filespace_in,
            H5S_SELECT_SET,
            (hsize_t *) &FILE_OFFSET, //start-offset
            (hsize_t *) &stride, //stride
            (hsize_t *) &block_cnt, //block cnt
            (hsize_t*) &block_size); //block size

    return actual_elem_cnt;
}

//filespace should be read from the file first, then select the hyperslab.
unsigned long _set_dataspace_seq_2D(hid_t* filespace_in_out, hid_t* memspace_out,
        unsigned long long dim_1, unsigned long long dim_2){
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
    *memspace_out =  H5Screate_simple(2, mem_dims, NULL);
    H5Sselect_hyperslab(*filespace_in_out, H5S_SELECT_SET, file_starts, NULL, count, NULL);
    return dim_1 * dim_2;
}

unsigned long _set_dataspace_seq_3D(hid_t* filespace_in_out, hid_t* memspace_out,
        unsigned long long dim_1, unsigned long long dim_2, unsigned long long dim_3){
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


    *memspace_out =  H5Screate_simple(3, mem_dims, NULL);

    H5Sselect_hyperslab(*filespace_in_out, H5S_SELECT_SET, file_starts, NULL, file_range, NULL);
    return dim_1 * dim_2 * dim_3;
}

hid_t get_filespace(hid_t file_id){
    char* grp_name = "/Timestep_0";
    char* ds_name = "px";
    hid_t gid = H5Gopen_async(file_id, grp_name, H5P_DEFAULT, 0);
    hid_t dsid = H5Dopen2(gid, ds_name, H5P_DEFAULT);
    hid_t filespace = H5Dget_space(dsid);
    H5Dclose_async(dsid, 0);
    H5Gclose_async(gid, 0);
    return filespace;
}

unsigned long set_dataspace(bench_params params, unsigned long read_elem_cnt, hid_t* filespace_in_out, hid_t* memspace_out){
    unsigned long actual_read_cnt = 0;
    switch(params.access_pattern.pattern_read){
        case CONTIG_1D:
            _set_dataspace_seq_read(read_elem_cnt, filespace_in_out, memspace_out);
            actual_read_cnt = read_elem_cnt;
            break;

        case STRIDED_1D:
            actual_read_cnt = _set_dataspace_strided_read(read_elem_cnt, params.stride, params.block_size, filespace_in_out, memspace_out);
            break;

        case CONTIG_2D:
            actual_read_cnt = _set_dataspace_seq_2D(filespace_in_out, memspace_out, params.dim_1, params.dim_2);
            break;

        case CONTIG_3D:
            actual_read_cnt = _set_dataspace_seq_3D(filespace_in_out, memspace_out, params.dim_1, params.dim_2, params.dim_3);
            break;

        default:
            break;
    }
    return actual_read_cnt;
}

int _run_benchmark_read(hid_t file_id, hid_t fapl, hid_t gapl, hid_t filespace, bench_params params, unsigned long* raw_read_time_out, unsigned long* total_data_size_out){
    *raw_read_time_out = 0;
    int nts = params.cnt_time_step;
    int sleep_time = params.sleep_time;
    unsigned long read_elem_cnt = params.cnt_actual_particles_M * M_VAL;
    hid_t grp;
    char grp_name[128];
    unsigned long rt1 = 0, rt2 = 0;

    unsigned long actual_read_cnt = 0;
    hid_t memspace;

    actual_read_cnt = set_dataspace(params, read_elem_cnt, &filespace, &memspace);
    if(MY_RANK == 0)
        print_params(&params);
    ES_ID = H5EScreate();
    for (int i = 0; i < nts; i++) {
        sprintf(grp_name, "Timestep_%d", i);
        grp = H5Gopen_async(file_id, grp_name, gapl, ES_ID);
        if (MY_RANK == 0) printf ("Reading %s ... \n", grp_name);
        read_h5_data(MY_RANK, grp, filespace, memspace, raw_read_time_out);
        if (i != 0) {
            if (MY_RANK == 0) printf ("  sleep for %ds\n", sleep_time);
            sleep(sleep_time);
            size_t num_in_progress;
            H5ES_status_t op_failed;
            H5ESwait(ES_ID, H5ES_WAIT_FOREVER, &num_in_progress, &op_failed);
        }
        H5Gclose_async(grp, ES_ID);
        MPI_Barrier(MPI_COMM_WORLD);
    }
    H5ESclose(ES_ID);
    *total_data_size_out = NUM_TIMESTEPS * actual_read_cnt * (6 * sizeof(float) + 2 * sizeof(int));
    H5Sclose(memspace);
    H5Sclose(filespace);
    return 0;
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


int _fill_csv_args(bench_params* params, char* argv[], int arg_idx_csv){
    if(argv[arg_idx_csv]){//CSV
        if(MY_RANK == 0 && strcmp(argv[arg_idx_csv], "CSV") == 0) {
                char* csv_path = argv[arg_idx_csv + 1];
                char* metadata_list = NULL;
                if(argv[arg_idx_csv + 2]){
                    if(strcmp(argv[arg_idx_csv + 2], "META") == 0){
                        if(!argv[arg_idx_csv + 3]){
                            printf("META is requested but metadata lit file is not specified.\n");
                            return -1;
                        }
                        metadata_list = argv[arg_idx_csv + 3];
                    }
                }
                if(csv_path){
                    printf("csv_path = %s, metadata_list = %s\n", csv_path, metadata_list);
                    FILE* csv_fs = csv_init(csv_path, metadata_list);
                    if(!csv_fs){
                        printf("Failed to create CSV file. \n");
                        return -1;
                    }
                    params->csv_fs = csv_fs;
                    params->useCSV = 1;
                } else {
                    printf("CSV option is enabled but file path is not specified.\n");
                    return -1;
                }
            }
    }
    return 0;
}

bench_params* args_set_params(int argc, char* argv[]){
    bench_params* params = calloc(1, sizeof(bench_params));
    int arg_idx = 1;

    //file name
    params->data_file_path = strdup(argv[arg_idx++]);
    //time steps
    params->cnt_time_step = atoi(argv[arg_idx++]);
    //sleep time
    params->sleep_time = atoi(argv[arg_idx++]);
    //read pattern: SEQ/PART/STRIDED/2D/3D
    params->stride = 0;
    params->block_size = 0;

    params->dim_2 = 1;
    params->dim_3 = 1;
    params->useCSV = 0;
    if(strcmp(argv[arg_idx], "SEQ") == 0){//$file $nts $sleeptime $OP $to_read_particles
        params->pattern_name = strdup("1D sequential read");
        if(MY_RANK == 0) printf("Read benchmark pattern = %s\n", params->pattern_name);
        params->_dim_cnt = 1;
        params->access_pattern.pattern_read = CONTIG_1D;
        params->cnt_particle_M = atoi(argv[arg_idx + 1]);//to read particles per rank
        params->cnt_actual_particles_M = params->cnt_particle_M;
        params->dim_1 = params->cnt_actual_particles_M * M_VAL;
        if(_fill_csv_args(params, argv, arg_idx + 2) != 0)
            return NULL;

    } else if(strcmp(argv[arg_idx], "PART") == 0){//same with SEQ
        params->pattern_name = strdup("1D partial read");
        if(MY_RANK == 0) printf("Read benchmark pattern = %s\n", params->pattern_name);
        params->_dim_cnt = 1;
        params->access_pattern.pattern_read = CONTIG_1D;
        params->cnt_particle_M = atoi(argv[arg_idx + 1]);
        params->cnt_actual_particles_M = params->cnt_particle_M;
        params->dim_1 = params->cnt_actual_particles_M * M_VAL;

        if(_fill_csv_args(params, argv, arg_idx + 2) != 0)
            return NULL;

    } else if(strcmp(argv[arg_idx], "STRIDED") == 0){//$file $nts $sleeptime $OP $attempt_to_read_particles $stride $block_size
        params->_dim_cnt = 1;
        params->pattern_name = strdup("1D strided read");
        if(MY_RANK == 0) printf("Read benchmark pattern = %s\n", params->pattern_name);
        params->access_pattern.pattern_read = STRIDED_1D;
        params->cnt_particle_M = atoi(argv[arg_idx + 1]);
        params->cnt_actual_particles_M = params->cnt_particle_M;
        params->stride = atoi(argv[arg_idx + 2]);
        params->block_size = atoi(argv[arg_idx + 3]);
        params->dim_1 = params->cnt_actual_particles_M * M_VAL;

        if(_fill_csv_args(params, argv, arg_idx + 4) != 0)
            return NULL;

    } else if(strcmp(argv[arg_idx], "2D") == 0){//$file $nts $sleeptime $OP $dim_1 $dim_2
        params->access_pattern.pattern_read = CONTIG_2D;
        params->pattern_name = strdup("CONTIG_2D");
        if(MY_RANK == 0) printf("Read benchmark pattern = %s\n", params->pattern_name);
        params->_dim_cnt = 2;
        params->dim_1 = atoi(argv[arg_idx + 1]);
        params->dim_2 = atoi(argv[arg_idx + 2]);
        params->cnt_actual_particles_M = params->dim_1 * params->dim_2 / M_VAL;

        if(_fill_csv_args(params, argv, arg_idx + 3) != 0)
            return NULL;

    } else if(strcmp(argv[arg_idx], "3D") == 0) {//$file $nts $sleeptime $OP $dim_1 $dim_2 $dim_3
        params->access_pattern.pattern_read = CONTIG_3D;
        params->pattern_name = strdup("CONTIG_3D");
        if(MY_RANK == 0) printf("Read benchmark pattern = %s\n", params->pattern_name);
        params->_dim_cnt = 3;
        params->dim_1 = atoi(argv[arg_idx + 1]);
        params->dim_2 = atoi(argv[arg_idx + 2]);
        params->dim_3 = atoi(argv[arg_idx + 3]);
        params->cnt_actual_particles_M = params->dim_1 * params->dim_2 * params->dim_3 / M_VAL;

        if(_fill_csv_args(params, argv, arg_idx + 4) != 0)
            return NULL;

    } else {
        if(MY_RANK == 0) printf("Unsupported benchmark pattern: [%s]. Only SEQ/PART/STRIDED/2D/3D are supported.\n ", argv[arg_idx]);
        bench_params_free(params);
        return NULL;
    }

    return params;
}

csv_handle CSV_HDL;

int main (int argc, char* argv[]){
    MPI_Init(&argc,&argv);
    MPI_Comm_rank (MPI_COMM_WORLD, &MY_RANK);
    MPI_Comm_size (MPI_COMM_WORLD, &NUM_RANKS);

    //argv_print(argc, argv);

    int sleep_time;
    char *file_name = argv[1]; //strdup(params->data_file_path); //argv[1];
    bench_params* params = args_set_params(argc, argv);

    if(!params){
        if(MY_RANK == 0) printf("ERROR: Invalid parameters.\n");
        return -1;
    }

    NUM_TIMESTEPS = params->cnt_time_step;

    if (NUM_TIMESTEPS <= 0) {
        if(MY_RANK == 0) printf("Usage: ./%s /path/to/file #timestep [# mega particles]\n", argv[0]);
        return 0;
    }

    sleep_time = params->sleep_time;
    if (sleep_time < 0) {
        print_usage(argv[0]);
        return 0;
    }

    hid_t fapl, gapl;
    set_pl(&fapl, &gapl);

    hsize_t dims[64] = {0};
    hid_t file_id = H5Fopen_async(file_name, H5F_ACC_RDONLY, fapl, 0);
    hid_t filespace = get_filespace(file_id);
    int dims_cnt = H5Sget_simple_extent_dims(filespace, dims, NULL);
    unsigned long total_particles = 1;
    if(dims_cnt > 0){
        for(int i = 0; i < dims_cnt; i++){
            if(MY_RANK == 0) printf("dims[%d] = %llu (total number for the file)\n", i, dims[i]);
            total_particles *= dims[i];
        }
    } else {
        if(MY_RANK == 0) printf("Failed to read dimensions. \n");
        return 0;
    }

    if(dims_cnt > 0){//1D
        if(params->dim_1 > dims[0]/NUM_RANKS){
            if(MY_RANK == 0) printf("Failed: Required dimension(%lu) is greater than the allowed dimension per rank (%llu).\n", params->dim_1, dims[0]/NUM_RANKS);
            goto error;
        }
    }
    if(dims_cnt > 1){//2D
        if(params->dim_2 > dims[1]){
            if(MY_RANK == 0) printf("Failed: Required dimension_2(%lu) is greater than file dimension(%llu).\n", params->dim_2, dims[1]);
            goto error;
        }
    }
    if(dims_cnt > 2){//3D
        if(params->dim_2 > dims[1]){
            if(MY_RANK == 0) printf("Failed: Required dimension_3(%lu) is greater than file dimension(%llu).\n", params->dim_3, dims[2]);
            goto error;
        }
    }

    NUM_PARTICLES = total_particles / NUM_RANKS;

    unsigned long  read_elem_cnt = params->cnt_actual_particles_M * M_VAL;

    if(read_elem_cnt  > NUM_PARTICLES){
        if(MY_RANK == 0) printf("read_elem_cnt_m <= num_particles must hold.\n");
        return 0;
    }

    MPI_Info info  = MPI_INFO_NULL;
    if (MY_RANK == 0) {
        printf("Total particles in the file: %lu\n", total_particles);
        printf ("Number of particles available per rank: %lld \n", NUM_PARTICLES);
    }

    MPI_Barrier (MPI_COMM_WORLD);

    unsigned long t0 = get_time_usec();
    MPI_Allreduce(&NUM_PARTICLES, &TOTAL_PARTICLES, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
    MPI_Scan(&NUM_PARTICLES, &FILE_OFFSET, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
    FILE_OFFSET -= NUM_PARTICLES;

    BUF_STRUCT = prepare_contig_memory_multi_dim(params->dim_1, params->dim_2, params->dim_3);

    unsigned long t1 = get_time_usec();

    if(file_id < 0) {
        if(MY_RANK == 0) printf("Error with opening file [%s]!\n", file_name);
        goto done;
    }

    if (MY_RANK == 0) printf ("Opened HDF5 file ... [%s]\n", file_name);

    unsigned long raw_read_time, local_data_size;
    unsigned long t2 = get_time_usec();
    _run_benchmark_read(file_id, fapl, gapl, filespace, *params, &raw_read_time, &local_data_size);
    unsigned long t3 = get_time_usec();

    MPI_Barrier (MPI_COMM_WORLD);

    H5Pclose(fapl);
    H5Pclose(gapl);
    H5Fclose_async(file_id, 0);

    MPI_Barrier (MPI_COMM_WORLD);
    unsigned long t4 = get_time_usec();
    free_contig_memory(BUF_STRUCT);

    if (MY_RANK == 0) {
        printf("\n =================  Performance results  =================\n");
        int total_sleep_time = sleep_time * (NUM_TIMESTEPS - 1);
        unsigned long total_size_mb = NUM_RANKS * local_data_size/(1024*1024);
        printf("Total sleep time %ds, total read size = %lu MB\n", total_sleep_time, total_size_mb);

        float rrt_s = (float)raw_read_time / (1000*1000);
        float raw_rate_mbs = total_size_mb / rrt_s;
        printf("RR: Raw read time = %.3f sec, RR = %.3f MB/sec \n", rrt_s, raw_rate_mbs);

        unsigned long meta_time_ms = (t3 - t2 - raw_read_time - sleep_time * (NUM_TIMESTEPS - 1) * 1000*1000) / 1000;
        printf("Core metadata time = %lu ms\n", meta_time_ms);

        double or_mbs = (float)total_size_mb/((float)(t4 - t1 - (NUM_TIMESTEPS - 1) * 1000*1000)/(1000 * 1000));
        printf("OR (observed read rate) = %.3f MB/sec\n", or_mbs);

        float oct_s = (float)(t4 - t0) / (1000*1000);
        printf("OCT (observed read completion time) = %.3f sec\n", oct_s);

        if(params->useCSV){
            fprintf(params->csv_fs, "NUM_RANKS, %d\n", NUM_RANKS);
            fprintf(params->csv_fs, "Total_sleep_time, %d, sec\n", total_sleep_time);
            fprintf(params->csv_fs, "Total_read_size, %lu, MB\n", total_size_mb);
            fprintf(params->csv_fs, "Raw_read_time, %.3f, sec\n", rrt_s);
            fprintf(params->csv_fs, "Raw_read_rate, %.3f, MB/sec\n", raw_rate_mbs);
            fprintf(params->csv_fs, "Core_metadata_time, %lu, ms\n", meta_time_ms);
            fprintf(params->csv_fs, "Observed_read_rate, %.3f, MB/sec\n", or_mbs);
            fprintf(params->csv_fs, "Observed_completion_time, %.3f, sec\n", oct_s);
            fclose(params->csv_fs);
        }
    }

error:
    H5E_BEGIN_TRY {
        H5Fclose_async(file_id, 0);
        H5Pclose(fapl);
    } H5E_END_TRY;

done:
    H5close();
    MPI_Finalize();
    return 0;
}
