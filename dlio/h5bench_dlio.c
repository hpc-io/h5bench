// TODO:
// - Add logging
// - Add vol-async support
// - Add subfiling support
// - Add more DLIO features
// - Add more data loaders: Tensorflow & dali
// - Add prefetcher configuration?
// - Add read_threads/computation_threads
// - Add file shuffle configuration
// - Add more compression filters

#include <assert.h>
#include <hdf5.h>
#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "h5bench_dlio.h"
#include "stats.h"
#include "utils.h"

//#ifdef HAVE_SUBFILING
//#include "H5FDsubfiling.h"
//#include "H5FDioc.h"
//#endif

#define GENERATION_BUFFER_SIZE 2 * 1073741824lu

// Global variables
int NUM_RANKS, MY_RANK;
uint32_t GENERATION_SIZE;
uint32_t DIM;
hid_t DCPL, FAPL, DAPL, DXPL;

void generate_labels_dataset(hid_t file_id, hid_t filespace, hid_t memspace) {
    hid_t dataset_id = H5Dcreate(file_id, config.LABELS_DATASET_NAME, H5T_STD_I64LE, filespace, H5P_DEFAULT, H5P_DEFAULT, DAPL);
    assert(dataset_id >= 0);

    uint64_t *data = (uint64_t*)malloc(config.NUM_SAMPLES_PER_FILE * sizeof(uint64_t));
    if (data == NULL) {
        exit(1);
    }
    for (uint32_t i = 0; i < config.NUM_SAMPLES_PER_FILE; i++) {
        data[i] = 0;
    }

    hsize_t offset[1] = {0};
    hsize_t dims[1] = {config.NUM_SAMPLES_PER_FILE};
    H5Sselect_hyperslab(filespace, H5S_SELECT_SET, offset, NULL, dims, NULL);
    herr_t status = H5Dwrite(dataset_id, H5T_STD_I64LE, memspace, filespace, DXPL, data);
    assert(status >= 0);

    free(data);
    H5Dclose(dataset_id);
}

void generate_records_dataset(hid_t file_id, hid_t filespace, hid_t memspace, hid_t extra_memspace) {
    hid_t dataset_id = H5Dcreate(file_id, config.RECORDS_DATASET_NAME, H5T_STD_U8LE, filespace, H5P_DEFAULT, DCPL, DAPL);
    assert(dataset_id >= 0);

    uint8_t *data = (uint8_t*)malloc(GENERATION_SIZE * sizeof(uint8_t));
    if (data == NULL) {
        exit(1);
    }
    for (size_t i = 0; i < GENERATION_SIZE; i++) {
        data[i] = rand() % 255;
    }

    uint32_t num_iterations = (config.RECORD_LENGTH * config.NUM_SAMPLES_PER_FILE) / GENERATION_SIZE;
    uint32_t extra_elements = (config.RECORD_LENGTH * config.NUM_SAMPLES_PER_FILE) % GENERATION_SIZE;

    hsize_t offset[3] = {0, 0, 0};
    hsize_t dims[3] = {config.NUM_SAMPLES_PER_FILE, DIM, DIM};

    for (uint32_t i = 0; i < num_iterations; i++) {
        offset[0] = i * config.RECORD_LENGTH * config.NUM_SAMPLES_PER_FILE;
        H5Sselect_hyperslab(filespace, H5S_SELECT_SET, offset, NULL, dims, NULL);
        herr_t status = H5Dwrite(dataset_id, H5T_STD_U8LE, memspace, filespace, DXPL, data);
        assert(status >= 0);
    }

    if (extra_elements > 0) {
        hsize_t extra_count[3] = {extra_elements, DIM, DIM};
        offset[0] = num_iterations * config.RECORD_LENGTH * config.NUM_SAMPLES_PER_FILE;
        H5Sselect_hyperslab(filespace, H5S_SELECT_SET, offset, NULL, extra_count, NULL);
        herr_t status = H5Dwrite(dataset_id, H5T_STD_U8LE, extra_memspace, filespace, DXPL, data);
        assert(status >= 0);
    }

    free(data);
    H5Dclose(dataset_id);
}

void generate_file(const char *file_name, hid_t labels_filespace, hid_t labels_memspace,
                   hid_t records_filespace, hid_t records_memspace, hid_t extra_records_memspace) {
    hid_t file_id = H5Fcreate(file_name, H5F_ACC_TRUNC, H5P_DEFAULT, FAPL);
    assert(file_id >= 0);

    generate_records_dataset(file_id, records_filespace, records_memspace, extra_records_memspace);
    generate_labels_dataset(file_id, labels_filespace, labels_memspace);

    H5Fclose(file_id);
}

void generate_data() {
    hsize_t labels_dims[1] = {config.NUM_SAMPLES_PER_FILE};
    hid_t labels_filespace = H5Screate_simple(1, labels_dims, NULL);
    assert(labels_filespace >= 0);
    hid_t labels_memspace = H5Screate_simple(1, labels_dims, NULL);
    assert(labels_memspace >= 0);

    hsize_t records_dims[3] = {config.NUM_SAMPLES_PER_FILE, DIM, DIM};
    hid_t records_filespace = H5Screate_simple(3, records_dims, NULL);
    assert(records_filespace >= 0);
    hid_t records_memspace = H5Screate_simple(3, records_dims, NULL);
    assert(records_memspace >= 0);

    hsize_t extra_records_count[3] = {(config.RECORD_LENGTH * config.NUM_SAMPLES_PER_FILE) % GENERATION_SIZE, DIM, DIM};
    hid_t extra_records_memspace = H5Screate_simple(3, extra_records_count, NULL);
    assert(extra_records_memspace >= 0);

    for (uint32_t i = MY_RANK; i < config.NUM_FILES_TRAIN; i += NUM_RANKS) {
        srand(config.RANDOM_SEED + i);

        printf("Generate train file %u / %u\n", i + 1, config.NUM_FILES_TRAIN);
        char file_name[256];
        snprintf(file_name, sizeof(file_name), "%s/%s/%s_%u_of_%u.h5", config.DATA_FOLDER, config.TRAIN_DATA_FOLDER, config.FILE_PREFIX, i + 1, config.NUM_FILES_TRAIN);
        generate_file(file_name, labels_filespace, labels_memspace, records_filespace, records_memspace, extra_records_memspace);
    }

    for (uint32_t i = MY_RANK; i < config.NUM_FILES_EVAL; i += NUM_RANKS) {
        srand(config.RANDOM_SEED + config.NUM_FILES_TRAIN + i);

        printf("Generate valid file %u / %u\n", i + 1, config.NUM_FILES_EVAL);
        char file_name[256];
        snprintf(file_name, sizeof(file_name), "%s/%s/%s_%u_of_%u.h5", config.DATA_FOLDER, config.VALID_DATA_FOLDER, config.FILE_PREFIX, i + 1, config.NUM_FILES_EVAL);
        generate_file(file_name, labels_filespace, labels_memspace, records_filespace, records_memspace, extra_records_memspace);
    }

    H5Sclose(labels_memspace);
    H5Sclose(labels_filespace);
    H5Sclose(records_memspace);
    H5Sclose(extra_records_memspace);
    H5Sclose(records_filespace);
}

void read_sample(const char *file_path, uint32_t sample, uint64_t *metadata_time_out, uint64_t *read_time_out) {
    hsize_t offset[3] = {sample, 0, 0};
    hsize_t count[3] = {1, DIM, DIM};

    uint64_t t1 = get_time_usec();
    hid_t file_id = H5Fopen(file_path, H5F_ACC_RDONLY, FAPL);
    hid_t dataset_id = H5Dopen(file_id, config.RECORDS_DATASET_NAME, DXPL);
    hid_t filespace = H5Dget_space(dataset_id);
    hid_t memspace = H5Screate_simple(3, count, NULL);
    H5Sselect_hyperslab(filespace, H5S_SELECT_SET, offset, NULL, count, NULL);
    uint64_t t2 = get_time_usec();
    assert(file_id >= 0);
    assert(dataset_id >= 0);
    assert(filespace >= 0);
    assert(memspace >= 0);

    uint8_t *data = (uint8_t *)malloc(DIM * DIM * sizeof(uint8_t));
    if (data == NULL) {
        exit(1);
    }

    uint64_t t3 = get_time_usec();
    herr_t status = H5Dread(dataset_id, H5T_STD_U8LE, memspace, filespace, DXPL, data);
    uint64_t t4 = get_time_usec();
    assert(status >= 0);

    free(data);

    uint64_t t5 = get_time_usec();
    H5Sclose(memspace);
    H5Sclose(filespace);
    H5Dclose(dataset_id);
    H5Fclose(file_id);
    uint64_t t6 = get_time_usec();

    *metadata_time_out = (t2 - t1) + (t6 - t5);
    *read_time_out = t4 - t3;
}

uint64_t compute(float time, float time_stdev) {
    if (time != 0.0 || time_stdev != 0.0) {
        int t = (uint64_t)(generate_normal_random(time, time_stdev) * 1000000.0);
        usleep(t > 0 ? t : 0);
        return t;
    }
    return 0;
}

void eval(uint32_t epoch, uint32_t *indices, uint64_t *local_metadata_time_out, uint64_t *local_read_time_out) {
    uint32_t total_samples = config.NUM_FILES_EVAL * config.NUM_SAMPLES_PER_FILE;

    if (config.DO_SHUFFLE) {
        shuffle(indices, total_samples);
    }

    uint32_t samples_per_rank = total_samples / NUM_RANKS;
    uint32_t read_from = MY_RANK * samples_per_rank;
    uint32_t read_to = (MY_RANK + 1) * samples_per_rank;
    uint32_t read_counter = 0;

    uint64_t t0 = get_time_usec();
    for (uint32_t i = read_from; i < read_to; i++) {
        uint32_t file_num = indices[i] / config.NUM_SAMPLES_PER_FILE + 1;
        uint32_t sample_num = indices[i] % config.NUM_SAMPLES_PER_FILE;
        char file_path[256];
        snprintf(file_path, sizeof(file_path), "%s/%s/%s_%u_of_%u.h5", config.DATA_FOLDER, config.VALID_DATA_FOLDER, config.FILE_PREFIX, file_num, config.NUM_FILES_EVAL);

        uint64_t metadata_time = 0, read_time = 0;
        read_sample(file_path, sample_num, &metadata_time, &read_time);
        read_counter++;
        compute(config.PREPROCESS_TIME, config.PREPROCESS_TIME_STDEV);

        *local_metadata_time_out += metadata_time;
        *local_read_time_out += read_time;

        if (read_counter % config.BATCH_SIZE_EVAL == 0 && read_counter != 0) {
            batch_loaded_eval(epoch, t0);

            uint64_t t = compute(config.EVAL_TIME, config.EVAL_TIME_STDEV);
            batch_processed_eval(epoch, t, t0);
            read_counter = 0;

            t0 = get_time_usec();
        }
    }

    for (uint32_t iteration = MY_RANK; iteration < total_samples - NUM_RANKS * samples_per_rank; iteration += NUM_RANKS) {
        uint32_t i = NUM_RANKS * samples_per_rank + iteration;
        uint32_t file_num = indices[i] / config.NUM_SAMPLES_PER_FILE + 1;
        uint32_t sample_num = indices[i] % config.NUM_SAMPLES_PER_FILE;
        char file_path[256];
        snprintf(file_path, sizeof(file_path), "%s/%s/%s_%u_of_%u.h5", config.DATA_FOLDER, config.VALID_DATA_FOLDER, config.FILE_PREFIX, file_num, config.NUM_FILES_EVAL);

        uint64_t metadata_time = 0, read_time = 0;
        read_sample(file_path, sample_num, &metadata_time, &read_time);
        read_counter++;
        compute(config.PREPROCESS_TIME, config.PREPROCESS_TIME_STDEV);

        *local_metadata_time_out += metadata_time;
        *local_read_time_out += read_time;

        if (read_counter % config.BATCH_SIZE_EVAL == 0){
            batch_loaded_eval(epoch, t0);

            uint64_t t = compute(config.EVAL_TIME, config.EVAL_TIME_STDEV);
            batch_processed_eval(epoch, t, t0);
            read_counter = 0;

            t0 = get_time_usec();
        }
    }

    if (read_counter != 0) {
        batch_loaded_eval(epoch, t0);

        uint64_t t = compute(config.EVAL_TIME, config.EVAL_TIME_STDEV);
        batch_processed_eval(epoch, t, t0);
    }
}

void train(uint32_t epoch, uint32_t *indices, uint64_t *local_metadata_time_out, uint64_t *local_read_time_out) {
    if (indices == NULL) return;
    uint32_t total_samples = config.NUM_FILES_TRAIN * config.NUM_SAMPLES_PER_FILE;

    if (config.DO_SHUFFLE) {
        shuffle(indices, total_samples);
    }

    uint32_t samples_per_rank = total_samples / NUM_RANKS;
    uint32_t read_from = MY_RANK * samples_per_rank;
    uint32_t read_to = (MY_RANK + 1) * samples_per_rank;
    uint32_t read_counter = 0;

    uint64_t t0 = get_time_usec();
    for (uint32_t i = read_from; i < read_to; i++) {
        uint32_t file_num = indices[i] / config.NUM_SAMPLES_PER_FILE + 1;
        uint32_t sample_num = indices[i] % config.NUM_SAMPLES_PER_FILE;
        char file_path[256];
        snprintf(file_path, sizeof(file_path), "%s/%s/%s_%u_of_%u.h5", config.DATA_FOLDER, config.TRAIN_DATA_FOLDER, config.FILE_PREFIX, file_num, config.NUM_FILES_TRAIN);

        uint64_t metadata_time = 0, read_time = 0;
        read_sample(file_path, sample_num, &metadata_time, &read_time);
        read_counter++;
        compute(config.PREPROCESS_TIME, config.PREPROCESS_TIME_STDEV);

        *local_metadata_time_out += metadata_time;
        *local_read_time_out += read_time;

        if (read_counter % config.BATCH_SIZE == 0 && read_counter != 0) {
            batch_loaded_train(epoch, t0);

            uint64_t t = compute(config.COMPUTATION_TIME, config.COMPUTATION_TIME_STDEV);
            batch_processed_train(epoch, t, t0);
            MPI_Barrier(MPI_COMM_WORLD);

            read_counter = 0;
            t0 = get_time_usec();
        }
    }

    for (uint32_t iteration = MY_RANK; iteration < total_samples - NUM_RANKS * samples_per_rank; iteration += NUM_RANKS) {
        uint32_t i = NUM_RANKS * samples_per_rank + iteration;
        uint32_t file_num = indices[i] / config.NUM_SAMPLES_PER_FILE + 1;
        uint32_t sample_num = indices[i] % config.NUM_SAMPLES_PER_FILE;
        char file_path[256];
        snprintf(file_path, sizeof(file_path), "%s/%s/%s_%u_of_%u.h5", config.DATA_FOLDER, config.TRAIN_DATA_FOLDER, config.FILE_PREFIX, file_num, config.NUM_FILES_TRAIN);

        uint64_t metadata_time = 0, read_time = 0;
        read_sample(file_path, sample_num, &metadata_time, &read_time);
        read_counter++;
        compute(config.PREPROCESS_TIME, config.PREPROCESS_TIME_STDEV);

        *local_metadata_time_out += metadata_time;
        *local_read_time_out += read_time;

        if (read_counter % config.BATCH_SIZE == 0){
            batch_loaded_train(epoch, t0);

            uint64_t t = compute(config.COMPUTATION_TIME, config.COMPUTATION_TIME_STDEV);
            batch_processed_train(epoch, t, t0);

            read_counter = 0;
            t0 = get_time_usec();
        }
    }

    if (read_counter != 0) {
        batch_loaded_train(epoch, t0);

        uint64_t t = compute(config.COMPUTATION_TIME, config.COMPUTATION_TIME_STDEV);
        batch_processed_train(epoch, t, t0);
    }

    MPI_Barrier(MPI_COMM_WORLD);
}

void run(uint64_t *train_metadata_time, uint64_t *train_read_time,
         uint64_t *eval_metadata_time, uint64_t *eval_read_time) {
    uint32_t total_train_samples = config.NUM_FILES_TRAIN * config.NUM_SAMPLES_PER_FILE;
    uint32_t *indices_train = (uint32_t *)malloc(total_train_samples * sizeof(uint32_t));
    if (indices_train == NULL) {
        exit(1);
    }
    for (uint32_t i = 0; i < total_train_samples; i++) {
        indices_train[i] = i;
    }

    uint32_t total_eval_samples = config.NUM_FILES_EVAL * config.NUM_SAMPLES_PER_FILE;
    uint32_t *indices_eval = (uint32_t *)malloc(total_eval_samples * sizeof(uint32_t));
    if (indices_eval == NULL) {
        exit(1);
    }
    for (unsigned long i = 0; i < total_eval_samples; i++) {
        indices_eval[i] = i;
    }

    uint64_t local_train_metadata_time = 0, local_train_read_time = 0,
             local_eval_metadata_time = 0, local_eval_read_time = 0;
    uint32_t next_eval_epoch = config.EPOCHS_BETWEEN_EVALS;

    MPI_Barrier(MPI_COMM_WORLD);

    for (uint32_t epoch = 0; epoch < config.EPOCHS; epoch++) {
//        if (MY_RANK == 0) printf("New Epoch %u\n", epoch + 1);
        if (config.SEED_CHANGE_EPOCH) srand(config.RANDOM_SEED + epoch);

        start_train(epoch);
        train(epoch, indices_train, &local_train_metadata_time, &local_train_read_time);
        end_train(epoch);

        if (config.DO_EVALUATION && (epoch + 1 >= next_eval_epoch)) {
            next_eval_epoch += config.EPOCHS_BETWEEN_EVALS;
            start_eval(epoch);
            eval(epoch, indices_eval, &local_eval_metadata_time, &local_eval_read_time);
            end_eval(epoch);
        }

        MPI_Barrier(MPI_COMM_WORLD);
    }

    MPI_Reduce(&local_train_metadata_time, train_metadata_time, 1, MPI_UNSIGNED_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_train_read_time, train_read_time, 1, MPI_UNSIGNED_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_eval_metadata_time, eval_metadata_time, 1, MPI_UNSIGNED_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_eval_read_time, eval_read_time, 1, MPI_UNSIGNED_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

    free(indices_train);
    free(indices_eval);
}

void init(int num_ranks) {
    config.NUM_RANKS = num_ranks;

    DIM = (uint32_t)sqrt(config.RECORD_LENGTH);
    config.RECORD_LENGTH = DIM * DIM;

    uint32_t chunk_dimension = (uint32_t)sqrt(config.CHUNK_SIZE);
    chunk_dimension = chunk_dimension > DIM? DIM: chunk_dimension;
    config.CHUNK_SIZE = chunk_dimension * chunk_dimension;

    uint32_t data_length = config.RECORD_LENGTH * config.NUM_SAMPLES_PER_FILE;
    GENERATION_SIZE = data_length > GENERATION_BUFFER_SIZE? GENERATION_BUFFER_SIZE: data_length;

    srand(config.RANDOM_SEED);

    DCPL = H5Pcreate(H5P_DATASET_CREATE);
    if (config.DO_CHUNKING) {
        hsize_t chunk_dims[3] = {1, chunk_dimension, chunk_dimension};
        H5Pset_chunk(DCPL, 3, chunk_dims);
        if (config.DO_COMPRESSION) {
            H5Pset_deflate(DCPL, config.COMPRESSION_LEVEL);
        }
    }

    FAPL = H5Pcreate(H5P_FILE_ACCESS);
//    H5Pset_fapl_mpio(fapl, MPI_COMM_WORLD, MPI_INFO_NULL);
#if H5_VERSION_GE(1, 10, 0)
    H5Pset_all_coll_metadata_ops(FAPL, true);
    H5Pset_coll_metadata_write(FAPL, true);
#endif

    hid_t DAPL = H5Pcreate(H5P_DATASET_ACCESS);
#if H5_VERSION_GE(1, 10, 0)
    H5Pset_all_coll_metadata_ops(DAPL, true);
#endif

    hid_t DXPL = H5Pcreate(H5P_DATASET_XFER);
}

int main(int argc, char *argv[]) {
    int mpi_thread_lvl_provided = -1;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &mpi_thread_lvl_provided);
    assert(MPI_THREAD_MULTIPLE == mpi_thread_lvl_provided);
    MPI_Comm_rank(MPI_COMM_WORLD, &MY_RANK);
    MPI_Comm_size(MPI_COMM_WORLD, &NUM_RANKS);

    parse_args(argc, argv);

    if (MY_RANK == 0) {
        printf("Create directory \"%s\"... ", config.DATA_FOLDER);
        create_directory(config.DATA_FOLDER);
        printf("OK\n");

        printf("Create directory \"%s/%s\"... ", config.DATA_FOLDER, config.TRAIN_DATA_FOLDER);
        char dir_name[256];
        snprintf(dir_name, sizeof(dir_name), "%s/%s", config.DATA_FOLDER, config.TRAIN_DATA_FOLDER);
        create_directory(dir_name);
        printf("OK\n");

        printf("Create directory \"%s/%s\"... ", config.DATA_FOLDER, config.VALID_DATA_FOLDER);
        snprintf(dir_name, sizeof(dir_name), "%s/%s", config.DATA_FOLDER, config.VALID_DATA_FOLDER);
        create_directory(dir_name);
        printf("OK\n");
    }

    init(NUM_RANKS);

    MPI_Barrier(MPI_COMM_WORLD);

    if (config.DO_DATA_GENERATION) {
        generate_data();
    }

    MPI_Barrier(MPI_COMM_WORLD);

    if (config.DO_TRAIN) {
        // TODO: check files dimension if generate=no
        stats_initialize();

        uint64_t train_metadata_time = 0, train_read_time = 0, eval_metadata_time = 0, eval_read_time = 0;
        run(&train_metadata_time, &train_read_time, &eval_metadata_time, &eval_read_time);

        prepare_data();

        MPI_Reduce(MY_RANK == 0? MPI_IN_PLACE: &train_metadata_time, &train_metadata_time, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(MY_RANK == 0? MPI_IN_PLACE: &train_read_time, &train_read_time, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(MY_RANK == 0? MPI_IN_PLACE: &eval_metadata_time, &eval_metadata_time, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(MY_RANK == 0? MPI_IN_PLACE: &eval_read_time, &eval_read_time, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

        train_metadata_time /= NUM_RANKS;
        train_read_time /= NUM_RANKS;
        eval_metadata_time /= NUM_RANKS;
        eval_read_time /= NUM_RANKS;

        MPI_Barrier(MPI_COMM_WORLD);

        if (MY_RANK == 0) {
            print_data(&train_metadata_time, &train_read_time, &eval_metadata_time, &eval_read_time);
        }

        stats_finalize();
    }

    if (!config.KEEP_FILES && MY_RANK == 0) {
        delete_directory(config.DATA_FOLDER);
    }

    H5Pclose(DCPL);
    H5Pclose(DXPL);
    H5Pclose(DAPL);
    H5Pclose(FAPL);
    MPI_Finalize();
    return 0;
}
