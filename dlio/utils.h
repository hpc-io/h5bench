#ifndef SANDBOX_UTILS_H
#define SANDBOX_UTILS_H

#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>

uint64_t get_time_usec_return_uint64();

typedef struct config_datatype {
    // Workflow
    bool DO_DATA_GENERATION;
    bool DO_TRAIN;
    bool DO_EVALUATION;

    // Dataset
    uint32_t RECORD_LENGTH; // should be a square number
                            //    float RECORD_LENGTH_STDEV;
                            //    float RECORD_LENGTH_RESIZE;
    uint32_t NUM_FILES_TRAIN;
    uint32_t NUM_FILES_EVAL;
    uint32_t NUM_SAMPLES_PER_FILE;
    char *   DATA_FOLDER;
    //    unsigned int NUM_SUBFOLDERS_TRAIN;
    //    unsigned int NUM_SUBFOLDERS_EVAL;
    char *   FILE_PREFIX;
    bool     DO_COMPRESSION;
    uint32_t COMPRESSION_LEVEL;
    bool     DO_CHUNKING;
    uint32_t CHUNK_SIZE; // should be a square number
    bool     KEEP_FILES;
    bool     COLLECTIVE_META;
    bool     COLLECTIVE_DATA;
    bool     SUBFILING;

    // Reader
    //    DATA_LOADER;
    uint32_t BATCH_SIZE;
    uint32_t BATCH_SIZE_EVAL;
    uint32_t READ_THREADS;
    //    int COMPUTATION_THREADS;
    //    unsigned int PREFETCH_SIZE;
    bool DO_SHUFFLE; // sample shuffle vs file_shuffle
                     //    unsigned int TRANSFER_SIZE;
    float PREPROCESS_TIME;
    float PREPROCESS_TIME_STDEV;
    bool  DROP_LAST;

    // Train
    uint32_t EPOCHS;
    float    COMPUTATION_TIME;
    float    COMPUTATION_TIME_STDEV;
    uint32_t TOTAL_TRAINING_STEPS;
    uint32_t TOTAL_TRAINING_STEPS_PER_RANK;
    bool     SEED_CHANGE_EPOCH;
    int      RANDOM_SEED;

    // Evaluation
    float    EVAL_TIME;
    float    EVAL_TIME_STDEV;
    uint32_t EPOCHS_BETWEEN_EVALS;

    // Output
    char *TRAIN_DATA_FOLDER;
    char *VALID_DATA_FOLDER;
    char *RECORDS_DATASET_NAME;
    char *LABELS_DATASET_NAME;
    char *OUTPUT_DATA_FOLDER;
    char *OUTPUT_CSV_NAME;
    bool  OUTPUT_RANKS_DATA;

    // Internal
    uint32_t NUM_TRAIN_BATCHES_PER_RANK;
    uint32_t NUM_EVAL_BATCHES_PER_RANK;
    uint32_t NUM_OF_ACTUALLY_USED_PROCESSES_TRAIN;
    uint32_t NUM_OF_ACTUALLY_USED_PROCESSES_EVAL;
} config_datatype_t;

extern config_datatype_t config;

void shuffle(uint32_t *array, size_t n);

double generate_normal_random(float mean, float stdev);

void create_directory(const char *folder);

void delete_directory(const char *dir_path);

void parse_args(int argc, char *argv[]);

#endif // SANDBOX_UTILS_H
