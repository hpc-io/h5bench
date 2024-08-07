#include <assert.h>
#include <hdf5.h>
#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include "h5bench_dlio.h"
#include "stats.h"
#include "utils.h"
#include "workers.h"

#ifdef HAVE_SUBFILING
#include "H5FDsubfiling.h"
#include "H5FDioc.h"
#endif

// Maximum size of randomly generated data per file. If the file size is larger than the specified value,
// randomly generated data will be written to the file several times in a row. Default value is 2 GB
#define GENERATION_BUFFER_SIZE 2 * 1073741824lu

// Global variables
int      NUM_RANKS, MY_RANK;
uint32_t GENERATION_SIZE;
uint32_t DIM;
hid_t    DCPL, FAPL, DAPL, DXPL;
MPI_Comm rest_training_steps_comm = MPI_COMM_WORLD;

// Generating a dataset containing training data labels
void
generate_labels_dataset(hid_t file_id, hid_t filespace, hid_t memspace)
{
    hid_t dataset_id = H5Dcreate(file_id, config.LABELS_DATASET_NAME, H5T_STD_I64LE, filespace, H5P_DEFAULT,
                                 H5P_DEFAULT, DAPL);
    assert(dataset_id >= 0);

    uint64_t *data = (uint64_t *)malloc(config.NUM_SAMPLES_PER_FILE * sizeof(uint64_t));
    if (data == NULL) {
        exit(1);
    }
    for (uint32_t i = 0; i < config.NUM_SAMPLES_PER_FILE; i++) {
        data[i] = 0;
    }

    hsize_t offset[1] = {0};
    hsize_t dims[1]   = {config.NUM_SAMPLES_PER_FILE};
    H5Sselect_hyperslab(filespace, H5S_SELECT_SET, offset, NULL, dims, NULL);
    herr_t status = H5Dwrite(dataset_id, H5T_STD_I64LE, memspace, filespace, DXPL, data);
    assert(status >= 0);

    free(data);
    H5Dclose(dataset_id);
}

// Generating a dataset containing random training data
void
generate_records_dataset(hid_t file_id, hid_t filespace, hid_t memspace, hid_t extra_memspace)
{
    hid_t dataset_id =
        H5Dcreate(file_id, config.RECORDS_DATASET_NAME, H5T_STD_U8LE, filespace, H5P_DEFAULT, DCPL, DAPL);
    assert(dataset_id >= 0);

    uint8_t *data = (uint8_t *)malloc(GENERATION_SIZE * sizeof(uint8_t));
    if (data == NULL) {
        exit(1);
    }
    for (size_t i = 0; i < GENERATION_SIZE; i++) {
        data[i] = rand() % 255;
    }

    uint32_t num_iterations = (config.RECORD_LENGTH * config.NUM_SAMPLES_PER_FILE) / GENERATION_SIZE;
    uint32_t extra_elements = (config.RECORD_LENGTH * config.NUM_SAMPLES_PER_FILE) % GENERATION_SIZE;

    hsize_t offset[3] = {0, 0, 0};
    hsize_t dims[3]   = {config.NUM_SAMPLES_PER_FILE, DIM, DIM};

    for (uint32_t i = 0; i < num_iterations; i++) {
        offset[0] = i * config.RECORD_LENGTH * config.NUM_SAMPLES_PER_FILE;
        H5Sselect_hyperslab(filespace, H5S_SELECT_SET, offset, NULL, dims, NULL);
        herr_t status = H5Dwrite(dataset_id, H5T_STD_U8LE, memspace, filespace, DXPL, data);
        assert(status >= 0);
    }

    if (extra_elements > 0) {
        hsize_t extra_count[3] = {extra_elements, DIM, DIM};
        offset[0]              = num_iterations * config.RECORD_LENGTH * config.NUM_SAMPLES_PER_FILE;
        H5Sselect_hyperslab(filespace, H5S_SELECT_SET, offset, NULL, extra_count, NULL);
        herr_t status = H5Dwrite(dataset_id, H5T_STD_U8LE, extra_memspace, filespace, DXPL, data);
        assert(status >= 0);
    }

    free(data);
    H5Dclose(dataset_id);
}

// Generating a hdf5 file containing a dataset with random training data and a dataset with labels
void
generate_file(const char *file_name, hid_t labels_filespace, hid_t labels_memspace, hid_t records_filespace,
              hid_t records_memspace, hid_t extra_records_memspace)
{
    hid_t file_id = H5Fcreate(file_name, H5F_ACC_TRUNC, H5P_DEFAULT, FAPL);
    assert(file_id >= 0);

    generate_records_dataset(file_id, records_filespace, records_memspace, extra_records_memspace);
    generate_labels_dataset(file_id, labels_filespace, labels_memspace);

    H5Fclose(file_id);
}

// Distribution of file generation work among MPI ranks
void
generate_data()
{
    if (MY_RANK == 0) {
        printf("Starting data generation\n");
        printf("Number of files for training dataset: %u\n", config.NUM_FILES_TRAIN);
        printf("Number of files for evaluation dataset: %u\n", config.NUM_FILES_EVAL);
    }

    hsize_t labels_dims[1]   = {config.NUM_SAMPLES_PER_FILE};
    hid_t   labels_filespace = H5Screate_simple(1, labels_dims, NULL);
    assert(labels_filespace >= 0);
    hid_t labels_memspace = H5Screate_simple(1, labels_dims, NULL);
    assert(labels_memspace >= 0);

    hsize_t records_dims[3]   = {config.NUM_SAMPLES_PER_FILE, DIM, DIM};
    hid_t   records_filespace = H5Screate_simple(3, records_dims, NULL);
    assert(records_filespace >= 0);
    hid_t records_memspace = H5Screate_simple(3, records_dims, NULL);
    assert(records_memspace >= 0);

    hsize_t extra_records_count[3] = {(config.RECORD_LENGTH * config.NUM_SAMPLES_PER_FILE) % GENERATION_SIZE,
                                      DIM, DIM};
    hid_t   extra_records_memspace = H5Screate_simple(3, extra_records_count, NULL);
    assert(extra_records_memspace >= 0);

    uint32_t from      = config.SUBFILING ? 0 : MY_RANK;
    uint32_t increment = config.SUBFILING ? 1 : NUM_RANKS;

    for (uint32_t i = from; i < config.NUM_FILES_TRAIN; i += increment) {
        srand(config.RANDOM_SEED + i);

        //        if (!config.SUBFILING || config.SUBFILING && (MY_RANK == 0))
        //            printf("Generate train file %u / %u\n", i + 1, config.NUM_FILES_TRAIN);
        char file_name[256];
        snprintf(file_name, sizeof(file_name), "%s/%s/%s_%u_of_%u.h5", config.DATA_FOLDER,
                 config.TRAIN_DATA_FOLDER, config.FILE_PREFIX, i + 1, config.NUM_FILES_TRAIN);
        generate_file(file_name, labels_filespace, labels_memspace, records_filespace, records_memspace,
                      extra_records_memspace);
    }

    for (uint32_t i = from; i < config.NUM_FILES_EVAL; i += increment) {
        srand(config.RANDOM_SEED + config.NUM_FILES_TRAIN + i);

        //        if (!config.SUBFILING || config.SUBFILING && (MY_RANK == 0))
        //            printf("Generate valid file %u / %u\n", i + 1, config.NUM_FILES_EVAL);
        char file_name[256];
        snprintf(file_name, sizeof(file_name), "%s/%s/%s_%u_of_%u.h5", config.DATA_FOLDER,
                 config.VALID_DATA_FOLDER, config.FILE_PREFIX, i + 1, config.NUM_FILES_EVAL);
        generate_file(file_name, labels_filespace, labels_memspace, records_filespace, records_memspace,
                      extra_records_memspace);
    }

    H5Sclose(labels_memspace);
    H5Sclose(labels_filespace);
    H5Sclose(records_memspace);
    H5Sclose(extra_records_memspace);
    H5Sclose(records_filespace);

    if (MY_RANK == 0) {
        printf("Generation done\n");
    }
}

// Read a given sample from a given hdf5 file
void
read_sample(const char *file_path, uint32_t sample, uint64_t *metadata_time_out, uint64_t *read_time_out)
{
    hsize_t offset[3] = {sample, 0, 0};
    hsize_t count[3]  = {1, DIM, DIM};

    uint64_t t1         = get_time_usec_return_uint64();
    hid_t    file_id    = H5Fopen(file_path, H5F_ACC_RDONLY, FAPL);
    hid_t    dataset_id = H5Dopen(file_id, config.RECORDS_DATASET_NAME, DAPL);
    hid_t    filespace  = H5Dget_space(dataset_id);
    hid_t    memspace   = H5Screate_simple(3, count, NULL);
    H5Sselect_hyperslab(filespace, H5S_SELECT_SET, offset, NULL, count, NULL);
    uint64_t t2 = get_time_usec_return_uint64();
    assert(file_id >= 0);
    assert(dataset_id >= 0);
    assert(filespace >= 0);
    assert(memspace >= 0);

    uint8_t *data = (uint8_t *)malloc(DIM * DIM * sizeof(uint8_t));
    if (data == NULL) {
        exit(1);
    }

    uint64_t t3     = get_time_usec_return_uint64();
    herr_t   status = H5Dread(dataset_id, H5T_STD_U8LE, memspace, filespace, DXPL, data);
    uint64_t t4     = get_time_usec_return_uint64();
    assert(status >= 0);

    free(data); // TODO: free memory only after compute() call?

    uint64_t t5 = get_time_usec_return_uint64();
    H5Sclose(memspace);
    H5Sclose(filespace);
    H5Dclose(dataset_id);
    H5Fclose(file_id);
    uint64_t t6 = get_time_usec_return_uint64();

    *metadata_time_out = (t2 - t1) + (t6 - t5);
    *read_time_out     = t4 - t3;

    compute(config.PREPROCESS_TIME, config.PREPROCESS_TIME_STDEV);
}

// Simulation of computations by means of sleep() function
uint64_t
compute(float time, float time_stdev)
{
    if (time != 0.0 || time_stdev != 0.0) {
        int t = (uint64_t)(generate_normal_random(time, time_stdev) * 1000000.0);
        usleep(t > 0 ? t : 0);
        return t;
    }
    return 0;
}

// Evaluation process simulation without the use of multiprocessing and workers
void
eval_without_workers(uint32_t epoch, uint32_t *indices, uint64_t *local_metadata_time_out,
                     uint64_t *local_read_time_out)
{
    uint32_t offset = MY_RANK * config.NUM_EVAL_BATCHES_PER_RANK;

    uint64_t t0 = get_time_usec_return_uint64();
    for (uint32_t i = 0; i < config.NUM_EVAL_BATCHES_PER_RANK; i++) {
        for (uint32_t j = 0; j < config.BATCH_SIZE_EVAL; j++) {
            uint32_t file_num =
                indices[offset + i * config.BATCH_SIZE_EVAL + j] / config.NUM_SAMPLES_PER_FILE + 1;
            uint32_t sample_num =
                indices[offset + i * config.BATCH_SIZE_EVAL + j] % config.NUM_SAMPLES_PER_FILE;
            char file_path[256];
            snprintf(file_path, sizeof(file_path), "%s/%s/%s_%u_of_%u.h5", config.DATA_FOLDER,
                     config.VALID_DATA_FOLDER, config.FILE_PREFIX, file_num, config.NUM_FILES_EVAL);

            uint64_t metadata_time = 0, read_time = 0;
            read_sample(file_path, sample_num, &metadata_time, &read_time);

            *local_metadata_time_out += metadata_time;
            *local_read_time_out += read_time;
        }

        batch_loaded_eval(epoch, t0);

        uint64_t t = compute(config.EVAL_TIME, config.EVAL_TIME_STDEV);
        batch_processed_eval(epoch, t, t0);
        MPI_Barrier(MPI_COMM_WORLD);

        t0 = get_time_usec_return_uint64();
    }
}

// Evaluation process simulation using multiprocessing and workers
void
eval_using_workers(uint32_t epoch, uint64_t *local_metadata_time_out, uint64_t *local_read_time_out)
{
    force_workers_to_shuffle(get_eval_read_fd(), get_eval_write_fd(), get_eval_system_fd());

    uint32_t offset = MY_RANK * config.NUM_EVAL_BATCHES_PER_RANK;

    for (uint32_t i = 0;
         i < (config.READ_THREADS > config.NUM_EVAL_BATCHES_PER_RANK ? config.NUM_EVAL_BATCHES_PER_RANK
                                                                     : config.READ_THREADS);
         i++) {
        int32_t batch = offset + i;
        write(get_eval_write_fd(), &batch, sizeof(batch));
    }

    for (uint32_t i = config.READ_THREADS; i < config.NUM_EVAL_BATCHES_PER_RANK; i++) {
        execution_time_t data_from_child_process;
        uint64_t         t0 = get_time_usec_return_uint64();
        read(get_eval_read_fd(), &data_from_child_process, sizeof(data_from_child_process));

        batch_loaded_eval(epoch, t0);

        *local_metadata_time_out += data_from_child_process.metadata_time;
        *local_read_time_out += data_from_child_process.read_time;

        int32_t batch = offset + i;
        write(get_eval_write_fd(), &batch, sizeof(batch));

        uint64_t t = compute(config.EVAL_TIME, config.EVAL_TIME_STDEV);
        batch_processed_eval(epoch, t, t0);
        MPI_Barrier(MPI_COMM_WORLD);
    }

    for (uint32_t i = 0;
         i < (config.READ_THREADS > config.NUM_EVAL_BATCHES_PER_RANK ? config.NUM_EVAL_BATCHES_PER_RANK
                                                                     : config.READ_THREADS);
         i++) {
        execution_time_t data_from_child_process;
        uint64_t         t0 = get_time_usec_return_uint64();
        read(get_eval_read_fd(), &data_from_child_process, sizeof(data_from_child_process));

        batch_loaded_eval(epoch, t0);

        *local_metadata_time_out += data_from_child_process.metadata_time;
        *local_read_time_out += data_from_child_process.read_time;

        uint64_t t = compute(config.EVAL_TIME, config.EVAL_TIME_STDEV);
        batch_processed_eval(epoch, t, t0);
        MPI_Barrier(MPI_COMM_WORLD);
    }
}

// Preparing and selecting a way to simulate the evaluation process
void
eval(uint32_t epoch, uint32_t *indices, bool enable_multiprocessing)
{
    uint64_t eval_metadata_time = 0, eval_read_time = 0;
    if (enable_multiprocessing) {
        start_eval(epoch);
        eval_using_workers(epoch, &eval_metadata_time, &eval_read_time);
        end_eval(epoch, eval_metadata_time, eval_read_time);
        return;
    }

    if (config.SEED_CHANGE_EPOCH)
        srand(config.RANDOM_SEED * 2 + epoch);
    if (config.DO_SHUFFLE)
        shuffle(indices, config.NUM_FILES_EVAL * config.NUM_SAMPLES_PER_FILE);

    start_eval(epoch);
    eval_without_workers(epoch, indices, &eval_metadata_time, &eval_read_time);
    end_eval(epoch, eval_metadata_time, eval_read_time);
}

// Training process simulation without the use of multiprocessing and workers
void
train_without_workers(uint32_t epoch, uint32_t *indices, uint64_t *local_metadata_time_out,
                      uint64_t *local_read_time_out)
{
    uint32_t offset = MY_RANK * config.NUM_TRAIN_BATCHES_PER_RANK;

    uint64_t t0 = get_time_usec_return_uint64();
    for (uint32_t i = 0; i < config.NUM_TRAIN_BATCHES_PER_RANK; i++) {
        if (i == config.TOTAL_TRAINING_STEPS_PER_RANK) {
            break;
        }
        for (uint32_t j = 0; j < config.BATCH_SIZE; j++) {
            uint32_t file_num = indices[offset + i * config.BATCH_SIZE + j] / config.NUM_SAMPLES_PER_FILE + 1;
            uint32_t sample_num = indices[offset + i * config.BATCH_SIZE + j] % config.NUM_SAMPLES_PER_FILE;
            char     file_path[256];
            snprintf(file_path, sizeof(file_path), "%s/%s/%s_%u_of_%u.h5", config.DATA_FOLDER,
                     config.TRAIN_DATA_FOLDER, config.FILE_PREFIX, file_num, config.NUM_FILES_TRAIN);

            uint64_t metadata_time = 0, read_time = 0;
            read_sample(file_path, sample_num, &metadata_time, &read_time);

            *local_metadata_time_out += metadata_time;
            *local_read_time_out += read_time;
        }

        batch_loaded_train(epoch, t0);

        uint64_t t = compute(config.COMPUTATION_TIME, config.COMPUTATION_TIME_STDEV);
        batch_processed_train(epoch, t, t0);
        if ((MY_RANK < config.TOTAL_TRAINING_STEPS % NUM_RANKS) &&
            (i + 1 == config.TOTAL_TRAINING_STEPS_PER_RANK)) {
            MPI_Barrier(rest_training_steps_comm);
        }
        else {
            MPI_Barrier(MPI_COMM_WORLD);
        }

        t0 = get_time_usec_return_uint64();
    }
}

// Training process simulation using multiprocessing and workers
void
train_using_workers(uint32_t epoch, uint64_t *local_metadata_time_out, uint64_t *local_read_time_out)
{
    force_workers_to_shuffle(get_train_read_fd(), get_train_write_fd(), get_train_system_fd());
    uint32_t offset = MY_RANK * config.NUM_TRAIN_BATCHES_PER_RANK;

    for (uint32_t i = 0; i < config.NUM_OF_ACTUALLY_USED_PROCESSES_TRAIN; i++) {
        int32_t batch = offset + i;
        write(get_train_write_fd(), &batch, sizeof(batch));
    }

    for (uint32_t i = config.NUM_OF_ACTUALLY_USED_PROCESSES_TRAIN; i < config.NUM_TRAIN_BATCHES_PER_RANK;
         i++) {
        if (i == config.TOTAL_TRAINING_STEPS_PER_RANK) {
            break;
        }
        execution_time_t data_from_child_process;
        uint64_t         t0 = get_time_usec_return_uint64();
        read(get_train_read_fd(), &data_from_child_process, sizeof(data_from_child_process));

        batch_loaded_train(epoch, t0);

        *local_metadata_time_out += data_from_child_process.metadata_time;
        *local_read_time_out += data_from_child_process.read_time;

        int32_t batch = offset + i;
        write(get_train_write_fd(), &batch, sizeof(batch));

        uint64_t t = compute(config.COMPUTATION_TIME, config.COMPUTATION_TIME_STDEV);
        batch_processed_train(epoch, t, t0);

        if ((MY_RANK < config.TOTAL_TRAINING_STEPS % NUM_RANKS) &&
            (i + 1 == config.TOTAL_TRAINING_STEPS_PER_RANK)) {
            MPI_Barrier(rest_training_steps_comm);
        }
        else {
            MPI_Barrier(MPI_COMM_WORLD);
        }
    }

    for (uint32_t i = 0; i < config.NUM_OF_ACTUALLY_USED_PROCESSES_TRAIN; i++) {
        execution_time_t data_from_child_process;
        uint64_t         t0 = get_time_usec_return_uint64();
        read(get_train_read_fd(), &data_from_child_process, sizeof(data_from_child_process));

        batch_loaded_train(epoch, t0);

        *local_metadata_time_out += data_from_child_process.metadata_time;
        *local_read_time_out += data_from_child_process.read_time;

        uint64_t t = compute(config.COMPUTATION_TIME, config.COMPUTATION_TIME_STDEV);
        batch_processed_train(epoch, t, t0);
        if ((MY_RANK < config.TOTAL_TRAINING_STEPS % NUM_RANKS) &&
            (i + 1 == config.TOTAL_TRAINING_STEPS_PER_RANK)) {
            MPI_Barrier(rest_training_steps_comm);
        }
        else {
            MPI_Barrier(MPI_COMM_WORLD);
        }
    }
}

// Preparing and selecting a way to simulate the training process
void
train(uint32_t epoch, uint32_t *indices, bool enable_multiprocessing)
{
    uint64_t train_metadata_time = 0, train_read_time = 0;
    if (enable_multiprocessing) {
        start_train(epoch);
        train_using_workers(epoch, &train_metadata_time, &train_read_time);
        end_train(epoch, train_metadata_time, train_read_time);
        return;
    }

    if (config.SEED_CHANGE_EPOCH)
        srand(config.RANDOM_SEED + epoch);
    if (config.DO_SHUFFLE)
        shuffle(indices, config.NUM_FILES_TRAIN * config.NUM_SAMPLES_PER_FILE);

    start_train(epoch);
    train_without_workers(epoch, indices, &train_metadata_time, &train_read_time);
    end_train(epoch, train_metadata_time, train_read_time);
}

// Starting the benchmark and simulation process of training and evaluation
void
run()
{
    uint32_t  total_train_samples = config.NUM_FILES_TRAIN * config.NUM_SAMPLES_PER_FILE;
    uint32_t *indices_train       = (uint32_t *)malloc(total_train_samples * sizeof(uint32_t));
    if (indices_train == NULL) {
        exit(1);
    }
    for (uint32_t i = 0; i < total_train_samples; i++) {
        indices_train[i] = i;
    }

    uint32_t  total_eval_samples = config.NUM_FILES_EVAL * config.NUM_SAMPLES_PER_FILE;
    uint32_t *indices_eval       = (uint32_t *)malloc(total_eval_samples * sizeof(uint32_t));
    if (indices_eval == NULL) {
        exit(1);
    }
    for (unsigned long i = 0; i < total_eval_samples; i++) {
        indices_eval[i] = i;
    }

    uint32_t next_eval_epoch        = config.EPOCHS_BETWEEN_EVALS;
    bool     enable_multiprocessing = config.READ_THREADS > 0;
    if (enable_multiprocessing) {
        init_workers(indices_train, indices_eval);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    for (uint32_t epoch = 0; epoch < config.EPOCHS; epoch++) {
        if (MY_RANK == 0)
            printf("Starting epoch %u...\n", epoch + 1);

        train(epoch, indices_train, enable_multiprocessing);
        MPI_Barrier(MPI_COMM_WORLD);

        if (config.DO_EVALUATION && (epoch + 1 >= next_eval_epoch)) {
            if (MY_RANK == 0)
                printf("Starting evaluation...\n");
            eval(epoch, indices_eval, enable_multiprocessing);
            next_eval_epoch += config.EPOCHS_BETWEEN_EVALS;
            MPI_Barrier(MPI_COMM_WORLD);
        }
    }
    if (enable_multiprocessing) {
        fin_workers();
    }

    free(indices_train);
    free(indices_eval);

    MPI_Barrier(MPI_COMM_WORLD);
}

// Initialization of some global variables and settings for benchmark operation
void
init_global_variables()
{
    if (MY_RANK == 0) {
        if (config.EPOCHS == 0) {
            printf("The value of parameter \"epochs\" must be greater than 0\n");
        }
        if (config.NUM_SAMPLES_PER_FILE == 0) {
            printf("The value of parameter \"num-samples-per-file\" must be greater than 0\n");
        }
        if (config.BATCH_SIZE == 0) {
            printf("The value of parameter \"batch-size\" must be greater than 0\n");
        }
        if (config.BATCH_SIZE_EVAL == 0) {
            printf("The value of parameter \"batch-size-eval\" must be greater than 0\n");
        }
    }

    DIM                  = (uint32_t)sqrt(config.RECORD_LENGTH);
    config.RECORD_LENGTH = DIM * DIM;

    uint32_t chunk_dimension = (uint32_t)sqrt(config.CHUNK_SIZE);
    chunk_dimension          = chunk_dimension > DIM ? DIM : chunk_dimension;
    config.CHUNK_SIZE        = chunk_dimension * chunk_dimension;

    uint32_t data_length = config.RECORD_LENGTH * config.NUM_SAMPLES_PER_FILE;
    GENERATION_SIZE      = data_length > GENERATION_BUFFER_SIZE ? GENERATION_BUFFER_SIZE : data_length;

    config.NUM_TRAIN_BATCHES_PER_RANK =
        config.NUM_FILES_TRAIN * config.NUM_SAMPLES_PER_FILE / config.BATCH_SIZE / NUM_RANKS;
    config.NUM_EVAL_BATCHES_PER_RANK =
        config.NUM_FILES_EVAL * config.NUM_SAMPLES_PER_FILE / config.BATCH_SIZE_EVAL / NUM_RANKS;

    config.NUM_OF_ACTUALLY_USED_PROCESSES_TRAIN = config.READ_THREADS > config.NUM_TRAIN_BATCHES_PER_RANK
                                                      ? config.NUM_TRAIN_BATCHES_PER_RANK
                                                      : config.READ_THREADS;
    config.NUM_OF_ACTUALLY_USED_PROCESSES_EVAL = config.READ_THREADS > config.NUM_EVAL_BATCHES_PER_RANK
                                                     ? config.NUM_EVAL_BATCHES_PER_RANK
                                                     : config.READ_THREADS;

    if (config.TOTAL_TRAINING_STEPS != -1 &&
        config.TOTAL_TRAINING_STEPS < config.NUM_TRAIN_BATCHES_PER_RANK * NUM_RANKS) {
        config.TOTAL_TRAINING_STEPS_PER_RANK = config.TOTAL_TRAINING_STEPS / NUM_RANKS;
        if (MY_RANK < config.TOTAL_TRAINING_STEPS % NUM_RANKS) {
            config.TOTAL_TRAINING_STEPS_PER_RANK++;
            MPI_Comm_split(MPI_COMM_WORLD, 0, MY_RANK, &rest_training_steps_comm);
        }
        else {
            MPI_Comm_split(MPI_COMM_WORLD, MPI_UNDEFINED, MY_RANK, &rest_training_steps_comm);
        }

        config.NUM_OF_ACTUALLY_USED_PROCESSES_TRAIN =
            config.NUM_OF_ACTUALLY_USED_PROCESSES_TRAIN > config.TOTAL_TRAINING_STEPS_PER_RANK
                ? config.TOTAL_TRAINING_STEPS_PER_RANK
                : config.NUM_OF_ACTUALLY_USED_PROCESSES_TRAIN;
    }
    else {
        config.TOTAL_TRAINING_STEPS = -1;
    }

    srand(config.RANDOM_SEED);

    // drop last warning

#ifndef HAVE_SUBFILING
    config.SUBFILING = false;
#endif

    FAPL = H5Pcreate(H5P_FILE_ACCESS);
    DCPL = H5Pcreate(H5P_DATASET_CREATE);
    DAPL = H5Pcreate(H5P_DATASET_ACCESS);
    DXPL = H5Pcreate(H5P_DATASET_XFER);

    if (config.SUBFILING) {
        if (MY_RANK == 0)
            printf("Using Subfiling VFD\n");
#ifdef HAVE_SUBFILING
        H5Pset_fapl_subfiling(FAPL, NULL);
#endif
        if (config.COLLECTIVE_DATA) {
            if (MY_RANK == 0)
                printf("Warning: Collective mode can't be used with subfiling\n");
            config.COLLECTIVE_DATA = false;
        }
        if (config.DO_CHUNKING) {
            if (MY_RANK == 0)
                printf("Warning: Chunking can't be used with subfiling\n");
            config.DO_CHUNKING = false;
        }
        if (config.READ_THREADS > 0) {
            if (MY_RANK == 0)
                printf(
                    "Warning: Multiprocessing can't be used with subfiling. READ_THREADS is set to 0...\n");
            config.READ_THREADS = 0;
        }
    }
    else if (config.DO_CHUNKING) {
        if (MY_RANK == 0)
            printf("Using chunking with the chunk shape (1, %u, %u)", chunk_dimension, chunk_dimension);
        hsize_t chunk_dims[3] = {1, chunk_dimension, chunk_dimension};
        H5Pset_chunk(DCPL, 3, chunk_dims);
        if (config.DO_COMPRESSION) {
            if (MY_RANK == 0)
                printf(" and compression (level %u)", config.COMPRESSION_LEVEL);
            H5Pset_deflate(DCPL, config.COMPRESSION_LEVEL);
        }
        if (MY_RANK == 0)
            printf("\n");
        if (config.COLLECTIVE_DATA) {
            if (MY_RANK == 0)
                printf("Warning: Collective mode can't be used with subfiling\n");
            config.COLLECTIVE_DATA = false;
        }
    }
    else {
        H5Pset_fapl_mpio(FAPL, MPI_COMM_SELF, MPI_INFO_NULL);
        if (config.COLLECTIVE_DATA) {
            if (MY_RANK == 0)
                printf("Using collective I/O mode\n");
            H5Pset_dxpl_mpio(DXPL, H5FD_MPIO_COLLECTIVE);
        }
        else {
            if (MY_RANK == 0)
                printf("Using independent I/O mode\n");
            H5Pset_dxpl_mpio(DXPL, H5FD_MPIO_INDEPENDENT);
        }
    }

#if H5_VERSION_GE(1, 10, 0)
    if (config.COLLECTIVE_META) {
        if (MY_RANK == 0)
            printf("Using collective meta-data I/O mode\n");
        H5Pset_all_coll_metadata_ops(FAPL, true);
        H5Pset_coll_metadata_write(FAPL, true);
        H5Pset_all_coll_metadata_ops(DAPL, true);
    }
#endif

    if ((MY_RANK == 0) && config.DO_TRAIN) {
        printf("The number of training batches per rank: %u\n", config.NUM_TRAIN_BATCHES_PER_RANK);
        if (config.READ_THREADS > config.NUM_TRAIN_BATCHES_PER_RANK) {
            printf("Warning: The number of requested read threads (%u) is greater than the number of "
                   "training batches per rank (%u)!\n",
                   config.READ_THREADS, config.NUM_TRAIN_BATCHES_PER_RANK);
        }
        printf("The number of evaluation batches per rank: %u\n", config.NUM_EVAL_BATCHES_PER_RANK);
        if (config.READ_THREADS > config.NUM_EVAL_BATCHES_PER_RANK) {
            printf("Warning: The number of requested read threads (%u) is greater than the number of "
                   "evaluation batches per rank (%u)!\n",
                   config.READ_THREADS, config.NUM_EVAL_BATCHES_PER_RANK);
        }
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

        printf("Create directory \"%s\"... ", config.OUTPUT_DATA_FOLDER);
        snprintf(dir_name, sizeof(dir_name), "%s", config.OUTPUT_DATA_FOLDER);
        create_directory(dir_name);
        printf("OK\n");
    }
    init_global_variables();
    MPI_Barrier(MPI_COMM_WORLD);

    if (config.DO_DATA_GENERATION) {
        generate_data();
    }

    if (config.DO_TRAIN) {
        // TODO: check files dimension if generate=no
        stats_initialize();

        run();
        prepare_data();
        MPI_Barrier(MPI_COMM_WORLD);

        if (config.OUTPUT_RANKS_DATA) {
            print_rank_data();
        }

        if (MY_RANK == 0) {
            print_average_data();
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
    H5close();
    MPI_Finalize();
    return 0;
}
