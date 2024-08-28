#include <dirent.h>
#include <math.h>
#include <mpi.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <unistd.h>

#include "utils.h"

// Returns the current time in microseconds
uint64_t
get_time_usec_return_uint64()
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (uint64_t)1000000 * tv.tv_sec + tv.tv_usec;
}

config_datatype_t config = {
    // Workflow
    .DO_DATA_GENERATION = false,
    .DO_TRAIN           = false,
    .DO_EVALUATION      = false,

    // Dataset
    .RECORD_LENGTH = 67108864, // should be a square number
                               //   .RECORD_LENGTH_STDEV = 0.0f,
                               //   .RECORD_LENGTH_RESIZE = 0.0f,
    .NUM_FILES_TRAIN      = 32,
    .NUM_FILES_EVAL       = 8,
    .NUM_SAMPLES_PER_FILE = 4,
    .DATA_FOLDER          = "./data",
    //    .NUM_SUBFOLDERS_TRAIN = 0,
    //    .NUM_SUBFOLDERS_EVAL = 0,
    .FILE_PREFIX       = "img",
    .DO_COMPRESSION    = false,
    .COMPRESSION_LEVEL = 4,
    .DO_CHUNKING       = false,
    .CHUNK_SIZE        = 1024, // should be greater than 120 on CLAIX23
    .KEEP_FILES        = false,
    .COLLECTIVE_META   = false,
    .COLLECTIVE_DATA   = false,
    .SUBFILING         = false,

    // Reader
    .BATCH_SIZE      = 7,
    .BATCH_SIZE_EVAL = 2,
    .READ_THREADS    = 4,
    //    .PREFETCH_SIZE = 0,
    .DO_SHUFFLE = false, // sample shuffle vs file_shuffle
                         //    .TRANSFER_SIZE = 262144,
    .PREPROCESS_TIME       = 0.0f,
    .PREPROCESS_TIME_STDEV = 0.000f,
    .DROP_LAST             = true,

    // Train
    .EPOCHS                        = 5,
    .COMPUTATION_TIME              = 0.323f,
    .COMPUTATION_TIME_STDEV        = 0.000f,
    .TOTAL_TRAINING_STEPS          = -1,
    .TOTAL_TRAINING_STEPS_PER_RANK = -1,
    .SEED_CHANGE_EPOCH             = false,
    .RANDOM_SEED                   = 42,

    // Evaluation
    .EVAL_TIME            = 0.323f,
    .EVAL_TIME_STDEV      = 0.000f,
    .EPOCHS_BETWEEN_EVALS = 1,

    // Output
    .TRAIN_DATA_FOLDER    = "train",
    .VALID_DATA_FOLDER    = "valid",
    .RECORDS_DATASET_NAME = "records",
    .LABELS_DATASET_NAME  = "labels",
    .OUTPUT_DATA_FOLDER   = "results",
    .OUTPUT_CSV_NAME      = "output",
    .OUTPUT_RANKS_DATA    = false,

    // Internal
    .NUM_TRAIN_BATCHES_PER_RANK           = 0,
    .NUM_EVAL_BATCHES_PER_RANK            = 0,
    .NUM_OF_ACTUALLY_USED_PROCESSES_TRAIN = 0,
    .NUM_OF_ACTUALLY_USED_PROCESSES_EVAL  = 0,
};

// Creating a directory with a specified name
void
create_directory(const char *folder)
{
    struct stat st = {0};
    if (stat(folder, &st) == -1) {
        if (mkdir(folder, 0700) != 0) {
            perror("Failed to create directory");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }
}

// Deleting a directory with a specified name
void
delete_directory(const char *dir_path)
{
    struct dirent *entry;
    DIR *          dir = opendir(dir_path);

    if (dir == NULL) {
        perror("Error opening directory");
        return;
    }

    while ((entry = readdir(dir)) != NULL) {
        char path[1024];
        snprintf(path, sizeof(path), "%s/%s", dir_path, entry->d_name);

        if (strcmp(entry->d_name, ".") == 0 || strcmp(entry->d_name, "..") == 0) {
            continue;
        }

        struct stat statbuf;
        if (stat(path, &statbuf) == 0) {
            if (S_ISDIR(statbuf.st_mode)) {
                delete_directory(path);
            }
            else {
                if (remove(path) != 0) {
                    perror("Error deleting file");
                }
            }
        }
    }

    closedir(dir);

    if (rmdir(dir_path) != 0) {
        perror("Error deleting directory");
    }
}

// Shuffle the values in the specified array
void
shuffle(uint32_t *array, size_t n)
{
    if (n > 1 && array != NULL) {
        for (size_t i = n - 1; i > 0; i--) {
            size_t   j    = rand() % (i + 1);
            uint32_t temp = array[i];
            array[i]      = array[j];
            array[j]      = temp;
        }
    }
}

// Generation of normally distributed random number
double
generate_normal_random(float mean, float stdev)
{
    double u1 = (double)rand() / RAND_MAX;
    double u2 = (double)rand() / RAND_MAX;
    double z0 = sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
    return z0 * stdev + mean;
}

// Parsing of arguments that the program receives as input
void
parse_args(int argc, char *argv[])
{
    for (uint32_t i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--generate-data") == 0) {
            config.DO_DATA_GENERATION = true;
        }
        else if (strcmp(argv[i], "--train") == 0) {
            config.DO_TRAIN = true;
        }
        else if (strcmp(argv[i], "--evaluation") == 0) {
            config.DO_EVALUATION = true;
        }
        else if (strcmp(argv[i], "--record-length") == 0) {
            i++;
            config.RECORD_LENGTH = atoi(argv[i]);
        }
        else if (strcmp(argv[i], "--num-files-train") == 0) {
            i++;
            config.NUM_FILES_TRAIN = atoi(argv[i]);
        }
        else if (strcmp(argv[i], "--num-files-eval") == 0) {
            i++;
            config.NUM_FILES_EVAL = atoi(argv[i]);
        }
        else if (strcmp(argv[i], "--num-samples-per-file") == 0) {
            i++;
            config.NUM_SAMPLES_PER_FILE = atoi(argv[i]);
        }
        else if (strcmp(argv[i], "--data-folder") == 0) {
            i++;
            config.DATA_FOLDER = argv[i];
        }
        else if (strcmp(argv[i], "--file-prefix") == 0) {
            i++;
            config.FILE_PREFIX = argv[i];
        }
        else if (strcmp(argv[i], "--chunking") == 0) {
            config.DO_CHUNKING = true;
        }
        else if (strcmp(argv[i], "--chunk-size") == 0) {
            i++;
            config.CHUNK_SIZE = atoi(argv[i]);
        }
        else if (strcmp(argv[i], "--keep-files") == 0) {
            config.KEEP_FILES = true;
        }
        else if (strcmp(argv[i], "--compression") == 0) {
            config.DO_COMPRESSION = true;
        }
        else if (strcmp(argv[i], "--compression-level") == 0) {
            i++;
            config.COMPRESSION_LEVEL = atoi(argv[i]);
        }
        else if (strcmp(argv[i], "--batch-size") == 0) {
            i++;
            config.BATCH_SIZE = atoi(argv[i]);
        }
        else if (strcmp(argv[i], "--batch-size-eval") == 0) {
            i++;
            config.BATCH_SIZE_EVAL = atoi(argv[i]);
        }
        else if (strcmp(argv[i], "--shuffle") == 0) {
            config.DO_SHUFFLE = true;
        }
        else if (strcmp(argv[i], "--preprocess-time") == 0) {
            i++;
            config.PREPROCESS_TIME = atof(argv[i]);
        }
        else if (strcmp(argv[i], "--preprocess-time-stdev") == 0) {
            i++;
            config.PREPROCESS_TIME_STDEV = atof(argv[i]);
        }
        else if (strcmp(argv[i], "--epochs") == 0) {
            i++;
            config.EPOCHS = atoi(argv[i]);
        }
        else if (strcmp(argv[i], "--computation-time") == 0) {
            i++;
            config.COMPUTATION_TIME = atof(argv[i]);
        }
        else if (strcmp(argv[i], "--computation-time-stdev") == 0) {
            i++;
            config.COMPUTATION_TIME_STDEV = atof(argv[i]);
        }
        else if (strcmp(argv[i], "--random-seed") == 0) {
            i++;
            config.RANDOM_SEED = atoi(argv[i]);
        }
        else if (strcmp(argv[i], "--eval-time") == 0) {
            i++;
            config.EVAL_TIME = atof(argv[i]);
        }
        else if (strcmp(argv[i], "--eval-time-stdev") == 0) {
            i++;
            config.EVAL_TIME_STDEV = atof(argv[i]);
        }
        else if (strcmp(argv[i], "--epochs-between-evals") == 0) {
            i++;
            config.EPOCHS_BETWEEN_EVALS = atoi(argv[i]);
        }
        else if (strcmp(argv[i], "--train-data-folder") == 0) {
            i++;
            config.TRAIN_DATA_FOLDER = argv[i];
        }
        else if (strcmp(argv[i], "--valid-data-folder") == 0) {
            i++;
            config.VALID_DATA_FOLDER = argv[i];
        }
        else if (strcmp(argv[i], "--records-dataset-name") == 0) {
            i++;
            config.RECORDS_DATASET_NAME = argv[i];
        }
        else if (strcmp(argv[i], "--labels-dataset-name") == 0) {
            i++;
            config.LABELS_DATASET_NAME = argv[i];
        }
        else if (strcmp(argv[i], "--seed-change-epoch") == 0) {
            config.SEED_CHANGE_EPOCH = true;
        }
        else if (strcmp(argv[i], "--read-threads") == 0) {
            i++;
            config.READ_THREADS = atoi(argv[i]);
        }
        else if (strcmp(argv[i], "--collective-meta") == 0) {
            config.COLLECTIVE_META = true;
        }
        else if (strcmp(argv[i], "--collective-data") == 0) {
            config.COLLECTIVE_DATA = true;
        }
        else if (strcmp(argv[i], "--subfiling") == 0) {
            config.SUBFILING = true;
        }
        else if (strcmp(argv[i], "--drop-last") == 0) {
            config.DROP_LAST = true;
        }
        else if (strcmp(argv[i], "--output-data-folder") == 0) {
            i++;
            config.OUTPUT_DATA_FOLDER = argv[i];
        }
        else if (strcmp(argv[i], "--output-csv-name") == 0) {
            i++;
            config.OUTPUT_CSV_NAME = argv[i];
        }
        else if (strcmp(argv[i], "--output-ranks-data") == 0) {
            config.OUTPUT_RANKS_DATA = true;
        }
        else if (strcmp(argv[i], "--total-training-steps") == 0) {
            i++;
            config.TOTAL_TRAINING_STEPS = atoi(argv[i]);
        }
        else {
            printf("WARNING: %s not found\n", argv[i]);
        }
    }
}