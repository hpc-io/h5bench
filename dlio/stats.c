#include <mpi.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "../commons/h5bench_util.h"
#include "h5bench_dlio.h"
#include "stats.h"
#include "utils.h"

epoch_data_t *stats;
epoch_data_t *global_stats;

uint32_t *last_load_train;
uint32_t *last_load_eval;
uint32_t *last_proc_train;
uint32_t *last_proc_eval;
uint32_t *last_compute_train;
uint32_t *last_compute_eval;

// Initialization of variables for storing statistics information
void
stats_initialize()
{
    stats = (struct epoch_data *)malloc(config.EPOCHS * sizeof(struct epoch_data));
    if (stats == NULL) {
        exit(1);
    }

    for (uint32_t i = 0; i < config.EPOCHS; i++) {
        stats[i].load.train = (uint64_t *)calloc(config.NUM_TRAIN_BATCHES_PER_RANK, sizeof(uint64_t));
        if (stats[i].load.train == NULL) {
            exit(1);
        }
        stats[i].load.eval = (uint64_t *)calloc(config.NUM_EVAL_BATCHES_PER_RANK, sizeof(uint64_t));
        if (stats[i].load.eval == NULL) {
            exit(1);
        }
        stats[i].proc.train = (uint64_t *)calloc(config.NUM_TRAIN_BATCHES_PER_RANK, sizeof(uint64_t));
        if (stats[i].proc.train == NULL) {
            exit(1);
        }
        stats[i].proc.eval = (uint64_t *)calloc(config.NUM_EVAL_BATCHES_PER_RANK, sizeof(uint64_t));
        if (stats[i].proc.eval == NULL) {
            exit(1);
        }
        stats[i].throughput.train = 0.0;
        stats[i].throughput.eval  = 0.0;
        stats[i].compute.train    = (uint64_t *)calloc(config.NUM_TRAIN_BATCHES_PER_RANK, sizeof(uint64_t));
        if (stats[i].compute.train == NULL) {
            exit(1);
        }
        stats[i].compute.eval = (uint64_t *)calloc(config.NUM_EVAL_BATCHES_PER_RANK, sizeof(uint64_t));
        if (stats[i].compute.eval == NULL) {
            exit(1);
        }
        stats[i].observed_time.train = 0;
        stats[i].observed_time.eval  = 0;
        stats[i].metadata_time.train = 0;
        stats[i].metadata_time.eval  = 0;
        stats[i].raw_read_time.train = 0;
        stats[i].raw_read_time.eval  = 0;
    }

    last_load_train = calloc(config.EPOCHS, sizeof(uint32_t));
    if (last_load_train == NULL) {
        exit(1);
    }
    last_load_eval = calloc(config.EPOCHS, sizeof(uint32_t));
    if (last_load_eval == NULL) {
        exit(1);
    }
    last_proc_train = calloc(config.EPOCHS, sizeof(uint32_t));
    if (last_proc_train == NULL) {
        exit(1);
    }
    last_proc_eval = calloc(config.EPOCHS, sizeof(uint32_t));
    if (last_proc_eval == NULL) {
        exit(1);
    }
    last_compute_train = calloc(config.EPOCHS, sizeof(uint32_t));
    if (last_compute_train == NULL) {
        exit(1);
    }
    last_compute_eval = calloc(config.EPOCHS, sizeof(uint32_t));
    if (last_compute_eval == NULL) {
        exit(1);
    }
}

// Release of resources initialized for storing statistics information
void
stats_finalize()
{
    free(last_load_train);
    free(last_load_eval);
    free(last_proc_train);
    free(last_proc_eval);
    free(last_compute_train);
    free(last_compute_eval);

    for (uint32_t i = 0; i < config.EPOCHS; i++) {
        free(stats[i].load.train);
        free(stats[i].load.eval);
        free(stats[i].proc.train);
        free(stats[i].proc.eval);
        free(stats[i].compute.train);
        free(stats[i].compute.eval);

        free(global_stats[i].load.train);
        free(global_stats[i].load.eval);
        free(global_stats[i].proc.train);
        free(global_stats[i].proc.eval);
        free(global_stats[i].compute.train);
        free(global_stats[i].compute.eval);
    }

    free(stats);
    free(global_stats);
}

//
void
prepare_data()
{
    global_stats = (struct epoch_data *)malloc(config.EPOCHS * sizeof(struct epoch_data));
    if (global_stats == NULL) {
        exit(1);
    }

    for (uint32_t i = 0; i < config.EPOCHS; i++) {
        global_stats[i].load.train = (uint64_t *)calloc(config.NUM_TRAIN_BATCHES_PER_RANK, sizeof(uint64_t));
        if (global_stats[i].load.train == NULL) {
            exit(1);
        }
        global_stats[i].load.eval = (uint64_t *)calloc(config.NUM_EVAL_BATCHES_PER_RANK, sizeof(uint64_t));
        if (global_stats[i].load.eval == NULL) {
            exit(1);
        }
        global_stats[i].proc.train = (uint64_t *)calloc(config.NUM_TRAIN_BATCHES_PER_RANK, sizeof(uint64_t));
        if (global_stats[i].proc.train == NULL) {
            exit(1);
        }
        global_stats[i].proc.eval = (uint64_t *)calloc(config.NUM_EVAL_BATCHES_PER_RANK, sizeof(uint64_t));
        if (global_stats[i].proc.eval == NULL) {
            exit(1);
        }
        global_stats[i].compute.train =
            (uint64_t *)calloc(config.NUM_TRAIN_BATCHES_PER_RANK, sizeof(uint64_t));
        if (global_stats[i].compute.train == NULL) {
            exit(1);
        }
        global_stats[i].compute.eval = (uint64_t *)calloc(config.NUM_EVAL_BATCHES_PER_RANK, sizeof(uint64_t));
        if (global_stats[i].compute.eval == NULL) {
            exit(1);
        }

        MPI_Reduce(stats[i].load.train, global_stats[i].load.train, config.NUM_TRAIN_BATCHES_PER_RANK,
                   MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(stats[i].load.eval, global_stats[i].load.eval, config.NUM_EVAL_BATCHES_PER_RANK,
                   MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(stats[i].proc.train, global_stats[i].proc.train, config.NUM_TRAIN_BATCHES_PER_RANK,
                   MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(stats[i].proc.eval, global_stats[i].proc.eval, config.NUM_EVAL_BATCHES_PER_RANK,
                   MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(&stats[i].throughput.train, &global_stats[i].throughput.train, 1, MPI_DOUBLE, MPI_SUM, 0,
                   MPI_COMM_WORLD);
        MPI_Reduce(&stats[i].throughput.eval, &global_stats[i].throughput.eval, 1, MPI_DOUBLE, MPI_SUM, 0,
                   MPI_COMM_WORLD);
        MPI_Reduce(stats[i].compute.train, global_stats[i].compute.train, config.NUM_TRAIN_BATCHES_PER_RANK,
                   MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(stats[i].compute.eval, global_stats[i].compute.eval, config.NUM_EVAL_BATCHES_PER_RANK,
                   MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(&stats[i].observed_time.train, &global_stats[i].observed_time.train, 1,
                   MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(&stats[i].observed_time.eval, &global_stats[i].observed_time.eval, 1,
                   MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(&stats[i].metadata_time.train, &global_stats[i].metadata_time.train, 1,
                   MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(&stats[i].metadata_time.eval, &global_stats[i].metadata_time.eval, 1,
                   MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(&stats[i].raw_read_time.train, &global_stats[i].raw_read_time.train, 1,
                   MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(&stats[i].raw_read_time.eval, &global_stats[i].raw_read_time.eval, 1,
                   MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

        for (int j = 0; j < config.NUM_TRAIN_BATCHES_PER_RANK; j++) {
            global_stats[i].load.train[j] /= NUM_RANKS;
            global_stats[i].proc.train[j] /= NUM_RANKS;
            global_stats[i].compute.train[j] /= NUM_RANKS;
        }

        for (int j = 0; j < config.NUM_EVAL_BATCHES_PER_RANK; j++) {
            global_stats[i].load.eval[j] /= NUM_RANKS;
            global_stats[i].proc.eval[j] /= NUM_RANKS;
            global_stats[i].compute.eval[j] /= NUM_RANKS;
        }

        global_stats[i].throughput.train /= NUM_RANKS;
        global_stats[i].throughput.eval /= NUM_RANKS;
        global_stats[i].observed_time.train /= NUM_RANKS;
        global_stats[i].observed_time.eval /= NUM_RANKS;
        global_stats[i].metadata_time.train /= NUM_RANKS;
        global_stats[i].metadata_time.eval /= NUM_RANKS;
        global_stats[i].raw_read_time.train /= NUM_RANKS;
        global_stats[i].raw_read_time.eval /= NUM_RANKS;

        //        if (config.NUM_OF_ACTUALLY_USED_PROCESSES_TRAIN > 0) {
        //            global_stats[i].metadata_time.train /= config.NUM_OF_ACTUALLY_USED_PROCESSES_TRAIN;
        //            global_stats[i].raw_read_time.train /= config.NUM_OF_ACTUALLY_USED_PROCESSES_TRAIN;
        //        }
        //        if (config.NUM_OF_ACTUALLY_USED_PROCESSES_EVAL > 0) {
        //            global_stats[i].metadata_time.eval /= config.NUM_OF_ACTUALLY_USED_PROCESSES_EVAL;
        //            global_stats[i].raw_read_time.eval /= config.NUM_OF_ACTUALLY_USED_PROCESSES_EVAL;
        //        }
    }
}

// Preparing data obtained during benchmark execution for output
void
print_average_data()
{
    // Train
    uint64_t train_total_size_bytes =
        (uint64_t)config.BATCH_SIZE *
        (config.TOTAL_TRAINING_STEPS == -1 ? config.NUM_TRAIN_BATCHES_PER_RANK * NUM_RANKS
                                           : config.TOTAL_TRAINING_STEPS) *
        config.RECORD_LENGTH;
    uint64_t train_size_bytes_per_rank = train_total_size_bytes / NUM_RANKS;

    uint64_t  train_total_compute_time     = 0;
    uint64_t *train_compute_time_per_epoch = (uint64_t *)malloc(config.EPOCHS * sizeof(uint64_t));
    for (uint32_t i = 0; i < config.EPOCHS; i++) {
        unsigned long int compute_time = 0;
        for (uint32_t j = 0; j < config.NUM_TRAIN_BATCHES_PER_RANK; j++) {
            compute_time += global_stats[i].compute.train[j];
        }
        train_total_compute_time += compute_time;
        train_compute_time_per_epoch[i] = compute_time;
    }

    uint64_t train_total_metadata_time = 0;
    for (uint32_t i = 0; i < config.EPOCHS; i++) {
        train_total_metadata_time += global_stats[i].metadata_time.train;
    }

    uint64_t train_total_read_time = 0;
    for (uint32_t i = 0; i < config.EPOCHS; i++) {
        train_total_read_time += global_stats[i].raw_read_time.train;
    }

    double train_total_avg_read_rate = 0;
    for (uint32_t i = 0; i < config.EPOCHS; i++) {
        if (global_stats[i].raw_read_time.train == 0) {
            continue;
        }
        train_total_avg_read_rate +=
            (double)train_size_bytes_per_rank / global_stats[i].raw_read_time.train * 1000000.0;
    }
    train_total_avg_read_rate /= config.EPOCHS;

    uint64_t train_total_observed_time = 0;
    for (uint32_t i = 0; i < config.EPOCHS; i++) {
        train_total_observed_time += global_stats[i].observed_time.train;
    }

    double  train_total_avg_observed_rate     = 0.0;
    double *train_avg_observed_rate_per_epoch = (double *)malloc(config.EPOCHS * sizeof(double));
    for (uint32_t i = 0; i < config.EPOCHS; i++) {
        unsigned long int compute_time = 0;
        for (uint32_t j = 0; j < config.NUM_TRAIN_BATCHES_PER_RANK; j++) {
            compute_time += global_stats[i].compute.train[j];
        }
        if ((global_stats[i].observed_time.train - compute_time) == 0) {
            train_avg_observed_rate_per_epoch[i] = NAN;
            continue;
        }
        train_avg_observed_rate_per_epoch[i] = (double)train_size_bytes_per_rank /
                                               (global_stats[i].observed_time.train - compute_time) *
                                               1000000.0;
        train_total_avg_observed_rate += train_avg_observed_rate_per_epoch[i];
    }
    train_total_avg_observed_rate /= config.EPOCHS;

    double train_throughput_mean_samples_per_second = 0.0;
    for (uint32_t i = 0; i < config.EPOCHS; i++) {
        train_throughput_mean_samples_per_second += global_stats[i].throughput.train;
    }
    train_throughput_mean_samples_per_second =
        train_throughput_mean_samples_per_second / (double)config.EPOCHS;

    double train_throughput_stdev_samples_per_second = 0.0;
    for (uint32_t i = 0; i < config.EPOCHS; i++) {
        train_throughput_stdev_samples_per_second +=
            (global_stats[i].throughput.train - train_throughput_mean_samples_per_second) *
            (global_stats[i].throughput.train - train_throughput_mean_samples_per_second);
    }
    train_throughput_stdev_samples_per_second =
        sqrt(train_throughput_stdev_samples_per_second / (double)config.EPOCHS);

    double train_io_mean = train_throughput_mean_samples_per_second * config.RECORD_LENGTH;

    double train_io_stdev = train_throughput_stdev_samples_per_second * config.RECORD_LENGTH;

    // Evaluation
    uint64_t eval_total_size_bytes = (uint64_t)config.NUM_EVAL_BATCHES_PER_RANK * NUM_RANKS *
                                     config.BATCH_SIZE_EVAL * config.RECORD_LENGTH;
    uint64_t eval_size_bytes_per_rank =
        (uint64_t)config.NUM_EVAL_BATCHES_PER_RANK * config.BATCH_SIZE_EVAL * config.RECORD_LENGTH;

    uint64_t  eval_total_compute_time     = 0;
    uint64_t *eval_compute_time_per_epoch = (uint64_t *)malloc(config.EPOCHS * sizeof(uint64_t));
    for (uint32_t i = 0; i < config.EPOCHS; i++) {
        unsigned long int compute_time = 0;
        for (uint32_t j = 0; j < config.NUM_EVAL_BATCHES_PER_RANK; j++) {
            compute_time += global_stats[i].compute.eval[j];
        }
        eval_compute_time_per_epoch[i] = compute_time;
        eval_total_compute_time += compute_time;
    }

    uint64_t eval_total_metadata_time = 0;
    for (uint32_t i = 0; i < config.EPOCHS; i++) {
        eval_total_metadata_time += global_stats[i].metadata_time.eval;
    }

    uint64_t eval_total_read_time = 0;
    for (uint32_t i = 0; i < config.EPOCHS; i++) {
        eval_total_read_time += global_stats[i].raw_read_time.eval;
    }

    double eval_total_avg_read_rate = 0;
    for (uint32_t i = 0; i < config.EPOCHS; i++) {
        if (global_stats[i].raw_read_time.eval == 0) {
            continue;
        }
        eval_total_avg_read_rate +=
            (double)eval_size_bytes_per_rank / global_stats[i].raw_read_time.eval * 1000000.0;
    }
    eval_total_avg_read_rate /= config.EPOCHS;

    uint64_t eval_total_observed_time = 0;
    for (uint32_t i = 0; i < config.EPOCHS; i++) {
        eval_total_observed_time += global_stats[i].observed_time.eval;
    }

    double  eval_total_avg_observed_rate     = 0.0;
    double *eval_avg_observed_rate_per_epoch = (double *)malloc(config.EPOCHS * sizeof(double));
    for (uint32_t i = 0; i < config.EPOCHS; i++) {
        unsigned long compute_time = 0;
        for (uint32_t j = 0; j < config.NUM_EVAL_BATCHES_PER_RANK; j++) {
            compute_time += global_stats[i].compute.eval[j];
        }
        if ((global_stats[i].observed_time.eval - compute_time) == 0) {
            eval_avg_observed_rate_per_epoch[i] = NAN;
            continue;
        }
        eval_avg_observed_rate_per_epoch[i] = (double)eval_size_bytes_per_rank /
                                              (global_stats[i].observed_time.eval - compute_time) * 1000000.0;
        eval_total_avg_observed_rate += eval_avg_observed_rate_per_epoch[i];
    }
    eval_total_avg_observed_rate /= config.EPOCHS;

    double eval_throughput_mean_samples_per_second = 0.0;
    for (uint32_t i = 0; i < config.EPOCHS; i++) {
        eval_throughput_mean_samples_per_second += global_stats[i].throughput.eval;
    }
    eval_throughput_mean_samples_per_second = eval_throughput_mean_samples_per_second / (double)config.EPOCHS;

    double eval_throughput_stdev_samples_per_second = 0.0;
    for (uint32_t i = 0; i < config.EPOCHS; i++) {
        eval_throughput_stdev_samples_per_second +=
            (global_stats[i].throughput.eval - eval_throughput_mean_samples_per_second) *
            (global_stats[i].throughput.eval - eval_throughput_mean_samples_per_second);
    }
    eval_throughput_stdev_samples_per_second =
        sqrt(eval_throughput_stdev_samples_per_second / (double)config.EPOCHS);

    double eval_io_mean = eval_throughput_mean_samples_per_second * config.RECORD_LENGTH;

    double eval_io_stdev = eval_throughput_stdev_samples_per_second * config.RECORD_LENGTH;

    human_readable value;

    printf("\n=================== Performance Results ==================\n");
    printf("Total number of ranks: %d\n", NUM_RANKS);
    printf("The number of read threads per rank: %d\n", config.READ_THREADS);

    value = format_human_readable(train_total_size_bytes);
    printf("Total training set size: %.3lf %cB\n", value.value, value.unit);
    value = format_human_readable(train_size_bytes_per_rank);
    printf("Training set size per rank: %.3lf %cB\n", value.value, value.unit);
    printf("Total training emulated compute time: %.3lf s\n", train_total_compute_time / 1000000.0);
    printf("Training metadata time: %.3lf s\n", train_total_metadata_time / 1000000.0);
    printf("Training raw read time: %.3lf s\n", train_total_read_time / 1000000.0);
    value = format_human_readable(train_total_avg_read_rate);
    printf("Training average raw read rate: %.3f %cB/s\n", value.value, value.unit);
    printf("Observed training completion time: %.3lf s\n", train_total_observed_time / 1000000.0);
    value = format_human_readable(train_total_avg_observed_rate);
    printf("Observed average training rate: %.3f %cB/s\n", value.value, value.unit);
    printf("Training average throughput: %.3lf samples/s\n", train_throughput_mean_samples_per_second);
    printf("Training throughput standard deviation: %.3lf samples/s\n",
           train_throughput_stdev_samples_per_second);
    value = format_human_readable(train_io_mean);
    printf("Training average IO: %.3f %cB/s\n", value.value, value.unit);
    value = format_human_readable(train_io_stdev);
    printf("Training IO standard deviation: %.3f %cB/s\n", value.value, value.unit);

    if (config.DO_EVALUATION) {
        value = format_human_readable(eval_total_size_bytes);
        printf("Total evaluation set size: %.3lf %cB\n", value.value, value.unit);
        value = format_human_readable(eval_size_bytes_per_rank);
        printf("Evaluation set size per rank: %.3lf %cB\n", value.value, value.unit);
        printf("Total evaluation emulated compute time: %.3lf s\n", eval_total_compute_time / 1000000.0);
        printf("Evaluation metadata time: %.3lf s\n", eval_total_metadata_time / 1000000.0);
        printf("Evaluation raw read time: %.3lf s\n", eval_total_read_time / 1000000.0);
        value = format_human_readable(eval_total_avg_read_rate);
        printf("Evaluation average raw read rate: %.3lf %cB/s\n", value.value, value.unit);
        printf("Observed evaluation completion time: %.3lf s\n", eval_total_observed_time / 1000000.0);
        value = format_human_readable(eval_total_avg_observed_rate);
        printf("Observed average evaluation rate: %.3lf %cB/s\n", value.value, value.unit);
        printf("Evaluation average throughput avg: %.3lf samples/s\n",
               eval_throughput_mean_samples_per_second);
        printf("Evaluation throughput standard deviation: %.3lf samples/s\n",
               eval_throughput_stdev_samples_per_second);
        value = format_human_readable(eval_io_mean);
        printf("Evaluation average IO: %.3lf %cB/s\n", value.value, value.unit);
        value = format_human_readable(eval_io_stdev);
        printf("Evaluation IO standard deviation: %.3lf %cB/s\n", value.value, value.unit);
    }

    printf("===========================================================\n");

    char file_name[256];
    snprintf(file_name, sizeof(file_name), "%s/%s.csv", config.OUTPUT_DATA_FOLDER, config.OUTPUT_CSV_NAME);

    FILE *csv_file = fopen(file_name, "w+");

    char *units = (char *)malloc(config.EPOCHS * sizeof(char));

    fprintf(csv_file, "metric, value, unit\n");
    fprintf(csv_file, "operation, dlio,\n");
    fprintf(csv_file, "ranks, %d,\n", NUM_RANKS);
    fprintf(csv_file, "read threads, %d,\n", config.READ_THREADS);
    fprintf(csv_file, "subfiling, %s,\n", config.SUBFILING ? "YES" : "NO");
    fprintf(csv_file, "chunking, %s,\n", config.DO_CHUNKING ? "YES" : "NO");
    fprintf(csv_file, "collective meta, %s,\n", config.COLLECTIVE_META ? "YES" : "NO");
    fprintf(csv_file, "collective data, %s,\n", config.COLLECTIVE_DATA ? "YES" : "NO");

    value = format_human_readable(train_total_size_bytes);
    fprintf(csv_file, "train total size, %.3lf, %cB\n", value.value, value.unit);
    value = format_human_readable(train_size_bytes_per_rank);
    fprintf(csv_file, "train size per rank, %.3lf, %cB\n", value.value, value.unit);
    fprintf(csv_file, "train emulated compute time per epoch, \"");
    for (uint32_t i = 0; i < config.EPOCHS; i++) {
        fprintf(csv_file, "%.3lf", train_compute_time_per_epoch[i] / 1000000.0);
        if (i != config.EPOCHS - 1)
            fprintf(csv_file, ", ");
    }
    fprintf(csv_file, "\", \"");
    for (uint32_t i = 0; i < config.EPOCHS; i++) {
        fprintf(csv_file, "s");
        if (i != config.EPOCHS - 1)
            fprintf(csv_file, ", ");
    }
    fprintf(csv_file, "\"\ntrain emulated compute time, %.3lf, s\n", train_total_compute_time / 1000000.0);
    fprintf(csv_file, "train metadata time per epoch, \"");
    for (uint32_t i = 0; i < config.EPOCHS; i++) {
        fprintf(csv_file, "%.3lf", global_stats[i].metadata_time.train / 1000000.0);
        if (i != config.EPOCHS - 1)
            fprintf(csv_file, ", ");
    }
    fprintf(csv_file, "\", \"");
    for (uint32_t i = 0; i < config.EPOCHS; i++) {
        fprintf(csv_file, "s");
        if (i != config.EPOCHS - 1)
            fprintf(csv_file, ", ");
    }
    fprintf(csv_file, "\"\ntrain metadata time, %.3lf, s\n", train_total_metadata_time / 1000000.0);
    fprintf(csv_file, "train raw read time per epoch, \"");
    for (uint32_t i = 0; i < config.EPOCHS; i++) {
        fprintf(csv_file, "%.3lf", global_stats[i].raw_read_time.train / 1000000.0);
        if (i != config.EPOCHS - 1)
            fprintf(csv_file, ", ");
    }
    fprintf(csv_file, "\", \"");
    for (uint32_t i = 0; i < config.EPOCHS; i++) {
        fprintf(csv_file, "s");
        if (i != config.EPOCHS - 1)
            fprintf(csv_file, ", ");
    }
    fprintf(csv_file, "\"\ntrain total raw read time, %.3lf, s\n", train_total_read_time / 1000000.0);
    fprintf(csv_file, "train raw read rate per epoch, \"");
    for (uint32_t i = 0; i < config.EPOCHS; i++) {
        if (global_stats[i].raw_read_time.train == 0) {
            units[i] = ' ';
            fprintf(csv_file, "NaN");
        }
        else {
            value    = format_human_readable((double)train_size_bytes_per_rank /
                                          global_stats[i].raw_read_time.train * 1000000.0);
            units[i] = value.unit;
            fprintf(csv_file, "%.3lf", value.value);
        }
        if (i != config.EPOCHS - 1)
            fprintf(csv_file, ", ");
    }
    fprintf(csv_file, "\", \"");
    for (uint32_t i = 0; i < config.EPOCHS; i++) {
        fprintf(csv_file, "%cB/s", units[i]);
        if (i != config.EPOCHS - 1)
            fprintf(csv_file, ", ");
    }
    value = format_human_readable(train_total_avg_read_rate);
    fprintf(csv_file, "\"\ntrain avg raw read rate, %.3lf, %cB/s\n", value.value, value.unit);
    fprintf(csv_file, "train observed time per epoch, \"");
    for (uint32_t i = 0; i < config.EPOCHS; i++) {
        fprintf(csv_file, "%.3lf", global_stats[i].observed_time.train / 1000000.0);
        if (i != config.EPOCHS - 1)
            fprintf(csv_file, ", ");
    }
    fprintf(csv_file, "\", \"");
    for (uint32_t i = 0; i < config.EPOCHS; i++) {
        fprintf(csv_file, "s");
        if (i != config.EPOCHS - 1)
            fprintf(csv_file, ", ");
    }
    fprintf(csv_file, "\"\ntrain observed time, %.3lf, s\n", train_total_observed_time / 1000000.0);
    fprintf(csv_file, "train observed rate per epoch, \"");
    for (uint32_t i = 0; i < config.EPOCHS; i++) {
        value    = format_human_readable(train_avg_observed_rate_per_epoch[i]);
        units[i] = value.unit;
        fprintf(csv_file, "%.3lf", value.value);
        if (i != config.EPOCHS - 1)
            fprintf(csv_file, ", ");
    }
    fprintf(csv_file, "\", \"");
    for (uint32_t i = 0; i < config.EPOCHS; i++) {
        fprintf(csv_file, "%cB/s", units[i]);
        if (i != config.EPOCHS - 1)
            fprintf(csv_file, ", ");
    }
    value = format_human_readable(train_total_avg_observed_rate);
    fprintf(csv_file, "\"\ntrain avg observed rate, %.3lf, %cB/s\n", value.value, value.unit);
    fprintf(csv_file, "train throughput samples per second per epoch, \"");
    for (uint32_t i = 0; i < config.EPOCHS; i++) {
        fprintf(csv_file, "%.3lf", global_stats[i].throughput.train);
        if (i != config.EPOCHS - 1)
            fprintf(csv_file, ", ");
    }
    fprintf(csv_file, "\", \"");
    for (uint32_t i = 0; i < config.EPOCHS; i++) {
        fprintf(csv_file, "samples/s");
        if (i != config.EPOCHS - 1)
            fprintf(csv_file, ", ");
    }
    fprintf(csv_file, "\"\ntrain throughput avg samples per second, %.3lf, samples/s\n",
            train_throughput_mean_samples_per_second);
    fprintf(csv_file, "train throughput stdev samples per second, %.3lf, samples/s\n",
            train_throughput_stdev_samples_per_second);
    value = format_human_readable(train_io_mean);
    fprintf(csv_file, "train io avg, %.3lf, %cB/s\n", value.value, value.unit);
    value = format_human_readable(train_io_stdev);
    fprintf(csv_file, "train io stdev, %.3lf, %cB/s\n", value.value, value.unit);

    if (config.DO_EVALUATION) {
        value = format_human_readable(eval_total_size_bytes);
        fprintf(csv_file, "eval total size, %.3lf, %cB\n", value.value, value.unit);
        value = format_human_readable(eval_size_bytes_per_rank);
        fprintf(csv_file, "eval size per rank, %.3lf, %cB\n", value.value, value.unit);
        fprintf(csv_file, "eval emulated compute time per epoch, \"");
        for (uint32_t i = 0; i < config.EPOCHS; i++) {
            fprintf(csv_file, "%.3lf", eval_compute_time_per_epoch[i] / 1000000.0);
            if (i != config.EPOCHS - 1)
                fprintf(csv_file, ", ");
        }
        fprintf(csv_file, "\", \"");
        for (uint32_t i = 0; i < config.EPOCHS; i++) {
            fprintf(csv_file, "s");
            if (i != config.EPOCHS - 1)
                fprintf(csv_file, ", ");
        }
        fprintf(csv_file, "\"\neval emulated compute time, %.3lf, s\n", eval_total_compute_time / 1000000.0);
        fprintf(csv_file, "eval metadata time per epoch, \"");
        for (uint32_t i = 0; i < config.EPOCHS; i++) {
            fprintf(csv_file, "%.3lf", global_stats[i].metadata_time.eval / 1000000.0);
            if (i != config.EPOCHS - 1)
                fprintf(csv_file, ", ");
        }
        fprintf(csv_file, "\", \"");
        for (uint32_t i = 0; i < config.EPOCHS; i++) {
            fprintf(csv_file, "s");
            if (i != config.EPOCHS - 1)
                fprintf(csv_file, ", ");
        }
        fprintf(csv_file, "\"\neval metadata time, %.3lf, s\n", eval_total_metadata_time / 1000000.0);
        fprintf(csv_file, "eval raw read time per epoch, \"");
        for (uint32_t i = 0; i < config.EPOCHS; i++) {
            fprintf(csv_file, "%.3lf", global_stats[i].raw_read_time.eval / 1000000.0);
            if (i != config.EPOCHS - 1)
                fprintf(csv_file, ", ");
        }
        fprintf(csv_file, "\", \"");
        for (uint32_t i = 0; i < config.EPOCHS; i++) {
            fprintf(csv_file, "s");
            if (i != config.EPOCHS - 1)
                fprintf(csv_file, ", ");
        }

        fprintf(csv_file, "\"\neval total raw read time, %.3lf, s\n", eval_total_read_time / 1000000.0);
        fprintf(csv_file, "eval raw read rate per epoch, \"");
        for (uint32_t i = 0; i < config.EPOCHS; i++) {
            if (global_stats[i].raw_read_time.eval == 0) {
                units[i] = ' ';
                fprintf(csv_file, "NaN");
            }
            else {
                value = format_human_readable(eval_size_bytes_per_rank / global_stats[i].raw_read_time.eval *
                                              1000000.0);
                units[i] = value.unit;
                fprintf(csv_file, "%.3lf", value.value);
            }
            if (i != config.EPOCHS - 1)
                fprintf(csv_file, ", ");
        }
        fprintf(csv_file, "\", \"");
        for (uint32_t i = 0; i < config.EPOCHS; i++) {
            fprintf(csv_file, "%cB/s", units[i]);
            if (i != config.EPOCHS - 1)
                fprintf(csv_file, ", ");
        }
        value = format_human_readable(eval_total_avg_read_rate);
        fprintf(csv_file, "\"\neval avg raw read rate, %.3lf, %cB/s\n", value.value, value.unit);
        fprintf(csv_file, "eval observed time per epoch, \"");
        for (uint32_t i = 0; i < config.EPOCHS; i++) {
            fprintf(csv_file, "%.3lf", global_stats[i].observed_time.eval / 1000000.0);
            if (i != config.EPOCHS - 1)
                fprintf(csv_file, ", ");
        }
        fprintf(csv_file, "\", \"");
        for (uint32_t i = 0; i < config.EPOCHS; i++) {
            fprintf(csv_file, "s");
            if (i != config.EPOCHS - 1)
                fprintf(csv_file, ", ");
        }
        fprintf(csv_file, "\"\neval observed time, %.3lf, s\n", eval_total_observed_time / 1000000.0);
        fprintf(csv_file, "eval observed rate per epoch, \"");
        for (uint32_t i = 0; i < config.EPOCHS; i++) {
            value    = format_human_readable(eval_avg_observed_rate_per_epoch[i]);
            units[i] = value.unit;
            fprintf(csv_file, "%.3lf", value.value);
            if (i != config.EPOCHS - 1)
                fprintf(csv_file, ", ");
        }
        fprintf(csv_file, "\", \"");
        for (uint32_t i = 0; i < config.EPOCHS; i++) {
            fprintf(csv_file, "%cB/s", units[i]);
            if (i != config.EPOCHS - 1)
                fprintf(csv_file, ", ");
        }
        value = format_human_readable(eval_total_avg_observed_rate);
        fprintf(csv_file, "\"\neval avg observed rate, %.3lf, %cB/s\n", value.value, value.unit);
        fprintf(csv_file, "eval throughput samples per second per epoch, \"");
        for (uint32_t i = 0; i < config.EPOCHS; i++) {
            fprintf(csv_file, "%.3lf", global_stats[i].throughput.eval);
            if (i != config.EPOCHS - 1)
                fprintf(csv_file, ", ");
        }
        fprintf(csv_file, "\", \"");
        for (uint32_t i = 0; i < config.EPOCHS; i++) {
            fprintf(csv_file, "samples/s");
            if (i != config.EPOCHS - 1)
                fprintf(csv_file, ", ");
        }
        fprintf(csv_file, "\"\neval throughput avg samples per second, %.3lf, samples/s\n",
                eval_throughput_mean_samples_per_second);
        fprintf(csv_file, "eval throughput stdev samples per second, %.3lf, samples/s\n",
                eval_throughput_stdev_samples_per_second);
        value = format_human_readable(eval_io_mean);
        fprintf(csv_file, "eval io avg, %.3lf, %cB/s\n", value.value, value.unit);
        value = format_human_readable(eval_io_stdev);
        fprintf(csv_file, "eval io stdev, %.3lf, %cB/s\n", value.value, value.unit);
    }

    fclose(csv_file);
    free(units);
    free(train_compute_time_per_epoch);
    free(eval_compute_time_per_epoch);
    free(train_avg_observed_rate_per_epoch);
    free(eval_avg_observed_rate_per_epoch);
}

// Output collected statistics on the current MPI rank
void
print_rank_data()
{
    // Train
    uint64_t train_total_size_bytes =
        (uint64_t)config.NUM_TRAIN_BATCHES_PER_RANK * NUM_RANKS * config.BATCH_SIZE * config.RECORD_LENGTH;
    uint64_t train_size_bytes_per_rank =
        (uint64_t)config.NUM_TRAIN_BATCHES_PER_RANK * config.BATCH_SIZE * config.RECORD_LENGTH;

    uint64_t  train_total_compute_time     = 0;
    uint64_t *train_compute_time_per_epoch = (uint64_t *)malloc(config.EPOCHS * sizeof(uint64_t));
    for (uint32_t i = 0; i < config.EPOCHS; i++) {
        unsigned long int compute_time = 0;
        for (uint32_t j = 0; j < config.NUM_TRAIN_BATCHES_PER_RANK; j++) {
            compute_time += stats[i].compute.train[j];
        }
        train_total_compute_time += compute_time;
        train_compute_time_per_epoch[i] = compute_time;
    }

    uint64_t train_total_metadata_time = 0;
    for (uint32_t i = 0; i < config.EPOCHS; i++) {
        train_total_metadata_time += stats[i].metadata_time.train;
    }

    uint64_t train_total_read_time = 0;
    for (uint32_t i = 0; i < config.EPOCHS; i++) {
        train_total_read_time += stats[i].raw_read_time.train;
    }

    double train_total_avg_read_rate = 0;
    for (uint32_t i = 0; i < config.EPOCHS; i++) {
        if (stats[i].raw_read_time.train == 0) {
            continue;
        }
        train_total_avg_read_rate +=
            (double)train_size_bytes_per_rank / stats[i].raw_read_time.train * 1000000.0;
    }
    train_total_avg_read_rate /= config.EPOCHS;

    uint64_t train_total_observed_time = 0;
    for (uint32_t i = 0; i < config.EPOCHS; i++) {
        train_total_observed_time += stats[i].observed_time.train;
    }

    double  train_total_avg_observed_rate     = 0.0;
    double *train_avg_observed_rate_per_epoch = (double *)malloc(config.EPOCHS * sizeof(double));
    for (uint32_t i = 0; i < config.EPOCHS; i++) {
        unsigned long int compute_time = 0;
        for (uint32_t j = 0; j < config.NUM_TRAIN_BATCHES_PER_RANK; j++) {
            compute_time += stats[i].compute.train[j];
        }
        if ((stats[i].observed_time.train - compute_time) == 0) {
            train_avg_observed_rate_per_epoch[i] = NAN;
            continue;
        }
        train_avg_observed_rate_per_epoch[i] =
            (double)train_size_bytes_per_rank / (stats[i].observed_time.train - compute_time) * 1000000.0;
        train_total_avg_observed_rate += train_avg_observed_rate_per_epoch[i];
    }
    train_total_avg_observed_rate /= config.EPOCHS;

    double train_throughput_mean_samples_per_second = 0.0;
    for (uint32_t i = 0; i < config.EPOCHS; i++) {
        train_throughput_mean_samples_per_second += stats[i].throughput.train;
    }
    train_throughput_mean_samples_per_second =
        train_throughput_mean_samples_per_second / (double)config.EPOCHS;

    double train_throughput_stdev_samples_per_second = 0.0;
    for (uint32_t i = 0; i < config.EPOCHS; i++) {
        train_throughput_stdev_samples_per_second +=
            (stats[i].throughput.train - train_throughput_mean_samples_per_second) *
            (stats[i].throughput.train - train_throughput_mean_samples_per_second);
    }
    train_throughput_stdev_samples_per_second =
        sqrt(train_throughput_stdev_samples_per_second / (double)config.EPOCHS);

    double train_io_mean = train_throughput_mean_samples_per_second * config.RECORD_LENGTH;

    double train_io_stdev = train_throughput_stdev_samples_per_second * config.RECORD_LENGTH;

    // Evaluation
    uint64_t eval_total_size_bytes = (uint64_t)config.NUM_EVAL_BATCHES_PER_RANK * NUM_RANKS *
                                     config.BATCH_SIZE_EVAL * config.RECORD_LENGTH;
    uint64_t eval_size_bytes_per_rank =
        (uint64_t)config.NUM_EVAL_BATCHES_PER_RANK * config.BATCH_SIZE_EVAL * config.RECORD_LENGTH;

    uint64_t  eval_total_compute_time     = 0;
    uint64_t *eval_compute_time_per_epoch = (uint64_t *)malloc(config.EPOCHS * sizeof(uint64_t));
    for (uint32_t i = 0; i < config.EPOCHS; i++) {
        unsigned long int compute_time = 0;
        for (uint32_t j = 0; j < config.NUM_EVAL_BATCHES_PER_RANK; j++) {
            compute_time += stats[i].compute.eval[j];
        }
        eval_compute_time_per_epoch[i] = compute_time;
        eval_total_compute_time += compute_time;
    }

    uint64_t eval_total_metadata_time = 0;
    for (uint32_t i = 0; i < config.EPOCHS; i++) {
        eval_total_metadata_time += stats[i].metadata_time.eval;
    }

    uint64_t eval_total_read_time = 0;
    for (uint32_t i = 0; i < config.EPOCHS; i++) {
        eval_total_read_time += stats[i].raw_read_time.eval;
    }

    double eval_total_avg_read_rate = 0;
    for (uint32_t i = 0; i < config.EPOCHS; i++) {
        if (stats[i].raw_read_time.eval == 0) {
            continue;
        }
        eval_total_avg_read_rate +=
            (double)eval_size_bytes_per_rank / stats[i].raw_read_time.eval * 1000000.0;
    }
    eval_total_avg_read_rate /= config.EPOCHS;

    uint64_t eval_total_observed_time = 0;
    for (uint32_t i = 0; i < config.EPOCHS; i++) {
        eval_total_observed_time += stats[i].observed_time.eval;
    }

    double  eval_total_avg_observed_rate     = 0.0;
    double *eval_avg_observed_rate_per_epoch = (double *)malloc(config.EPOCHS * sizeof(double));
    for (uint32_t i = 0; i < config.EPOCHS; i++) {
        unsigned long compute_time = 0;
        for (uint32_t j = 0; j < config.NUM_EVAL_BATCHES_PER_RANK; j++) {
            compute_time += stats[i].compute.eval[j];
        }
        if ((stats[i].observed_time.eval - compute_time) == 0) {
            eval_avg_observed_rate_per_epoch[i] = NAN;
            continue;
        }
        eval_avg_observed_rate_per_epoch[i] =
            (double)eval_size_bytes_per_rank / (stats[i].observed_time.eval - compute_time) * 1000000.0;
        eval_total_avg_observed_rate += eval_avg_observed_rate_per_epoch[i];
    }
    eval_total_avg_observed_rate /= config.EPOCHS;

    double eval_throughput_mean_samples_per_second = 0.0;
    for (uint32_t i = 0; i < config.EPOCHS; i++) {
        eval_throughput_mean_samples_per_second += stats[i].throughput.eval;
    }
    eval_throughput_mean_samples_per_second = eval_throughput_mean_samples_per_second / (double)config.EPOCHS;

    double eval_throughput_stdev_samples_per_second = 0.0;
    for (uint32_t i = 0; i < config.EPOCHS; i++) {
        eval_throughput_stdev_samples_per_second +=
            (stats[i].throughput.eval - eval_throughput_mean_samples_per_second) *
            (stats[i].throughput.eval - eval_throughput_mean_samples_per_second);
    }
    eval_throughput_stdev_samples_per_second =
        sqrt(eval_throughput_stdev_samples_per_second / (double)config.EPOCHS);

    double eval_io_mean = eval_throughput_mean_samples_per_second * config.RECORD_LENGTH;

    double eval_io_stdev = eval_throughput_stdev_samples_per_second * config.RECORD_LENGTH;

    human_readable value;

    char filename[256];
    snprintf(filename, sizeof(filename), "%s/%d_%s.csv", config.OUTPUT_DATA_FOLDER, MY_RANK,
             config.OUTPUT_CSV_NAME);
    FILE *csv_file = fopen(filename, "w+");

    char *units = (char *)malloc(config.EPOCHS * sizeof(char));

    fprintf(csv_file, "metric, value, unit\n");
    fprintf(csv_file, "operation, dlio,\n");
    fprintf(csv_file, "ranks, %d,\n", NUM_RANKS);
    fprintf(csv_file, "read threads, %d,\n", config.READ_THREADS);
    fprintf(csv_file, "subfiling, %s,\n", config.SUBFILING ? "YES" : "NO");
    fprintf(csv_file, "chunking, %s,\n", config.DO_CHUNKING ? "YES" : "NO");
    fprintf(csv_file, "collective meta, %s,\n", config.COLLECTIVE_META ? "YES" : "NO");
    fprintf(csv_file, "collective data, %s,\n", config.COLLECTIVE_DATA ? "YES" : "NO");

    value = format_human_readable(train_total_size_bytes);
    fprintf(csv_file, "train total size, %.3lf, %cB\n", value.value, value.unit);
    value = format_human_readable(train_size_bytes_per_rank);
    fprintf(csv_file, "train size per rank, %.3lf, %cB\n", value.value, value.unit);
    fprintf(csv_file, "train emulated compute time per epoch, \"");
    for (uint32_t i = 0; i < config.EPOCHS; i++) {
        fprintf(csv_file, "%.3lf", train_compute_time_per_epoch[i] / 1000000.0);
        if (i != config.EPOCHS - 1)
            fprintf(csv_file, ", ");
    }
    fprintf(csv_file, "\", \"");
    for (uint32_t i = 0; i < config.EPOCHS; i++) {
        fprintf(csv_file, "s");
        if (i != config.EPOCHS - 1)
            fprintf(csv_file, ", ");
    }
    fprintf(csv_file, "\"\ntrain emulated compute time, %.3lf, s\n", train_total_compute_time / 1000000.0);
    fprintf(csv_file, "train metadata time per epoch, \"");
    for (uint32_t i = 0; i < config.EPOCHS; i++) {
        fprintf(csv_file, "%.3lf", stats[i].metadata_time.train / 1000000.0);
        if (i != config.EPOCHS - 1)
            fprintf(csv_file, ", ");
    }
    fprintf(csv_file, "\", \"");
    for (uint32_t i = 0; i < config.EPOCHS; i++) {
        fprintf(csv_file, "s");
        if (i != config.EPOCHS - 1)
            fprintf(csv_file, ", ");
    }
    fprintf(csv_file, "\"\ntrain metadata time, %.3lf, s\n", train_total_metadata_time / 1000000.0);
    fprintf(csv_file, "train raw read time per epoch, \"");
    for (uint32_t i = 0; i < config.EPOCHS; i++) {
        fprintf(csv_file, "%.3lf", stats[i].raw_read_time.train / 1000000.0);
        if (i != config.EPOCHS - 1)
            fprintf(csv_file, ", ");
    }
    fprintf(csv_file, "\", \"");
    for (uint32_t i = 0; i < config.EPOCHS; i++) {
        fprintf(csv_file, "s");
        if (i != config.EPOCHS - 1)
            fprintf(csv_file, ", ");
    }
    fprintf(csv_file, "\"\ntrain total raw read time, %.3lf, s\n", train_total_read_time / 1000000.0);
    fprintf(csv_file, "train raw read rate per epoch, \"");
    for (uint32_t i = 0; i < config.EPOCHS; i++) {
        if (stats[i].raw_read_time.train == 0) {
            units[i] = ' ';
            fprintf(csv_file, "NaN");
        }
        else {
            value = format_human_readable((double)train_size_bytes_per_rank / stats[i].raw_read_time.train *
                                          1000000.0);
            units[i] = value.unit;
            fprintf(csv_file, "%.3lf", value.value);
        }
        if (i != config.EPOCHS - 1)
            fprintf(csv_file, ", ");
    }
    fprintf(csv_file, "\", \"");
    for (uint32_t i = 0; i < config.EPOCHS; i++) {
        fprintf(csv_file, "%cB/s", units[i]);
        if (i != config.EPOCHS - 1)
            fprintf(csv_file, ", ");
    }
    value = format_human_readable(train_total_avg_read_rate);
    fprintf(csv_file, "\"\ntrain avg raw read rate, %.3lf, %cB/s\n", value.value, value.unit);
    fprintf(csv_file, "train observed time per epoch, \"");
    for (uint32_t i = 0; i < config.EPOCHS; i++) {
        fprintf(csv_file, "%.3lf", stats[i].observed_time.train / 1000000.0);
        if (i != config.EPOCHS - 1)
            fprintf(csv_file, ", ");
    }
    fprintf(csv_file, "\", \"");
    for (uint32_t i = 0; i < config.EPOCHS; i++) {
        fprintf(csv_file, "s");
        if (i != config.EPOCHS - 1)
            fprintf(csv_file, ", ");
    }
    fprintf(csv_file, "\"\ntrain observed time, %.3lf, s\n", train_total_observed_time / 1000000.0);
    fprintf(csv_file, "train observed rate per epoch, \"");
    for (uint32_t i = 0; i < config.EPOCHS; i++) {
        value    = format_human_readable(train_avg_observed_rate_per_epoch[i]);
        units[i] = value.unit;
        fprintf(csv_file, "%.3lf", value.value);
        if (i != config.EPOCHS - 1)
            fprintf(csv_file, ", ");
    }
    fprintf(csv_file, "\", \"");
    for (uint32_t i = 0; i < config.EPOCHS; i++) {
        fprintf(csv_file, "%cB/s", units[i]);
        if (i != config.EPOCHS - 1)
            fprintf(csv_file, ", ");
    }
    value = format_human_readable(train_total_avg_observed_rate);
    fprintf(csv_file, "\"\ntrain avg observed rate, %.3lf, %cB/s\n", value.value, value.unit);
    fprintf(csv_file, "train throughput samples per second per epoch, \"");
    for (uint32_t i = 0; i < config.EPOCHS; i++) {
        fprintf(csv_file, "%.3lf", stats[i].throughput.train);
        if (i != config.EPOCHS - 1)
            fprintf(csv_file, ", ");
    }
    fprintf(csv_file, "\", \"");
    for (uint32_t i = 0; i < config.EPOCHS; i++) {
        fprintf(csv_file, "samples/s");
        if (i != config.EPOCHS - 1)
            fprintf(csv_file, ", ");
    }
    fprintf(csv_file, "\"\ntrain throughput avg samples per second, %.3lf, samples/s\n",
            train_throughput_mean_samples_per_second);
    fprintf(csv_file, "train throughput stdev samples per second, %.3lf, samples/s\n",
            train_throughput_stdev_samples_per_second);
    value = format_human_readable(train_io_mean);
    fprintf(csv_file, "train io avg, %.3lf, %cB/s\n", value.value, value.unit);
    value = format_human_readable(train_io_stdev);
    fprintf(csv_file, "train io stdev, %.3lf, %cB/s\n", value.value, value.unit);

    value = format_human_readable(eval_total_size_bytes);
    fprintf(csv_file, "eval total size, %.3lf, %cB\n", value.value, value.unit);
    value = format_human_readable(eval_size_bytes_per_rank);
    fprintf(csv_file, "eval size per rank, %.3lf, %cB\n", value.value, value.unit);
    fprintf(csv_file, "eval emulated compute time per epoch, \"");
    for (uint32_t i = 0; i < config.EPOCHS; i++) {
        fprintf(csv_file, "%.3lf", eval_compute_time_per_epoch[i] / 1000000.0);
        if (i != config.EPOCHS - 1)
            fprintf(csv_file, ", ");
    }
    fprintf(csv_file, "\", \"");
    for (uint32_t i = 0; i < config.EPOCHS; i++) {
        fprintf(csv_file, "s");
        if (i != config.EPOCHS - 1)
            fprintf(csv_file, ", ");
    }
    fprintf(csv_file, "\"\neval emulated compute time, %.3lf, s\n", eval_total_compute_time / 1000000.0);
    fprintf(csv_file, "eval metadata time per epoch, \"");
    for (uint32_t i = 0; i < config.EPOCHS; i++) {
        fprintf(csv_file, "%.3lf", stats[i].metadata_time.eval / 1000000.0);
        if (i != config.EPOCHS - 1)
            fprintf(csv_file, ", ");
    }
    fprintf(csv_file, "\", \"");
    for (uint32_t i = 0; i < config.EPOCHS; i++) {
        fprintf(csv_file, "s");
        if (i != config.EPOCHS - 1)
            fprintf(csv_file, ", ");
    }
    fprintf(csv_file, "\"\neval metadata time, %.3lf, s\n", eval_total_metadata_time / 1000000.0);
    fprintf(csv_file, "eval raw read time per epoch, \"");
    for (uint32_t i = 0; i < config.EPOCHS; i++) {
        fprintf(csv_file, "%.3lf", stats[i].raw_read_time.eval / 1000000.0);
        if (i != config.EPOCHS - 1)
            fprintf(csv_file, ", ");
    }
    fprintf(csv_file, "\", \"");
    for (uint32_t i = 0; i < config.EPOCHS; i++) {
        fprintf(csv_file, "s");
        if (i != config.EPOCHS - 1)
            fprintf(csv_file, ", ");
    }

    fprintf(csv_file, "\"\neval total raw read time, %.3lf, s\n", eval_total_read_time / 1000000.0);
    fprintf(csv_file, "eval raw read rate per epoch, \"");
    for (uint32_t i = 0; i < config.EPOCHS; i++) {
        if (stats[i].raw_read_time.eval == 0) {
            units[i] = ' ';
            fprintf(csv_file, "NaN");
        }
        else {
            value = format_human_readable(eval_size_bytes_per_rank / stats[i].raw_read_time.eval * 1000000.0);
            units[i] = value.unit;
            fprintf(csv_file, "%.3lf", value.value);
        }
        if (i != config.EPOCHS - 1)
            fprintf(csv_file, ", ");
    }
    fprintf(csv_file, "\", \"");
    for (uint32_t i = 0; i < config.EPOCHS; i++) {
        fprintf(csv_file, "%cB/s", units[i]);
        if (i != config.EPOCHS - 1)
            fprintf(csv_file, ", ");
    }
    value = format_human_readable(eval_total_avg_read_rate);
    fprintf(csv_file, "\"\neval avg raw read rate, %.3lf, %cB/s\n", value.value, value.unit);
    fprintf(csv_file, "eval observed time per epoch, \"");
    for (uint32_t i = 0; i < config.EPOCHS; i++) {
        fprintf(csv_file, "%.3lf", stats[i].observed_time.eval / 1000000.0);
        if (i != config.EPOCHS - 1)
            fprintf(csv_file, ", ");
    }
    fprintf(csv_file, "\", \"");
    for (uint32_t i = 0; i < config.EPOCHS; i++) {
        fprintf(csv_file, "s");
        if (i != config.EPOCHS - 1)
            fprintf(csv_file, ", ");
    }
    fprintf(csv_file, "\"\neval observed time, %.3lf, s\n", eval_total_observed_time / 1000000.0);
    fprintf(csv_file, "eval observed rate per epoch, \"");
    for (uint32_t i = 0; i < config.EPOCHS; i++) {
        value    = format_human_readable(eval_avg_observed_rate_per_epoch[i]);
        units[i] = value.unit;
        fprintf(csv_file, "%.3lf", value.value);
        if (i != config.EPOCHS - 1)
            fprintf(csv_file, ", ");
    }
    fprintf(csv_file, "\", \"");
    for (uint32_t i = 0; i < config.EPOCHS; i++) {
        fprintf(csv_file, "%cB/s", units[i]);
        if (i != config.EPOCHS - 1)
            fprintf(csv_file, ", ");
    }
    value = format_human_readable(eval_total_avg_observed_rate);
    fprintf(csv_file, "\"\neval avg observed rate, %.3lf, %cB/s\n", value.value, value.unit);
    fprintf(csv_file, "eval throughput samples per second per epoch, \"");
    for (uint32_t i = 0; i < config.EPOCHS; i++) {
        fprintf(csv_file, "%.3lf", stats[i].throughput.eval);
        if (i != config.EPOCHS - 1)
            fprintf(csv_file, ", ");
    }
    fprintf(csv_file, "\", \"");
    for (uint32_t i = 0; i < config.EPOCHS; i++) {
        fprintf(csv_file, "samples/s");
        if (i != config.EPOCHS - 1)
            fprintf(csv_file, ", ");
    }
    fprintf(csv_file, "\"\neval throughput avg samples per second, %.3lf, samples/s\n",
            eval_throughput_mean_samples_per_second);
    fprintf(csv_file, "eval throughput stdev samples per second, %.3lf, samples/s\n",
            eval_throughput_stdev_samples_per_second);
    value = format_human_readable(eval_io_mean);
    fprintf(csv_file, "eval io avg, %.3lf, %cB/s\n", value.value, value.unit);
    value = format_human_readable(eval_io_stdev);
    fprintf(csv_file, "eval io stdev, %.3lf, %cB/s\n", value.value, value.unit);

    fclose(csv_file);
    free(units);
    free(train_compute_time_per_epoch);
    free(eval_compute_time_per_epoch);
    free(train_avg_observed_rate_per_epoch);
    free(eval_avg_observed_rate_per_epoch);
}

// Saving the time spent on loading a batch during the training process
void
batch_loaded_train(uint32_t epoch, uint64_t t0)
{
    stats[epoch].load.train[last_load_train[epoch]++] = (get_time_usec_return_uint64() - t0);
}

// Saving the time spent on processing a batch during the trining process
void
batch_processed_train(uint32_t epoch, uint64_t computation_time, uint64_t t0)
{
    stats[epoch].proc.train[last_proc_train[epoch]++]       = (get_time_usec_return_uint64() - t0);
    stats[epoch].compute.train[last_compute_train[epoch]++] = computation_time;
}

// Saving the time spent on loading a batch during the evaluation process
void
batch_loaded_eval(uint32_t epoch, uint64_t t0)
{
    stats[epoch].load.eval[last_load_eval[epoch]++] = (get_time_usec_return_uint64() - t0);
}

// Saving the time spent on processing a batch during the evaluation process
void
batch_processed_eval(uint32_t epoch, uint64_t computation_time, uint64_t t0)
{
    stats[epoch].proc.eval[last_proc_eval[epoch]++]       = (get_time_usec_return_uint64() - t0);
    stats[epoch].compute.eval[last_compute_eval[epoch]++] = computation_time;
}

// Saving the start time of the training process
void
start_train(uint32_t epoch)
{
    stats[epoch].start_time.train = get_time_usec_return_uint64();
}

// Saving data on the training process
void
end_train(uint32_t epoch, uint64_t metadata_time, uint64_t read_time)
{
    uint64_t end_time                = get_time_usec_return_uint64();
    stats[epoch].observed_time.train = end_time - stats[epoch].start_time.train;
    if ((end_time - stats[epoch].start_time.train) == 0) {
        stats[epoch].throughput.train = NAN;
    }
    else {
        stats[epoch].throughput.train =
            (double)config.BATCH_SIZE *
            (config.TOTAL_TRAINING_STEPS_PER_RANK == -1 ? config.NUM_TRAIN_BATCHES_PER_RANK
                                                        : config.TOTAL_TRAINING_STEPS_PER_RANK) *
            1000000.0 / (end_time - stats[epoch].start_time.train);
    }
    stats[epoch].metadata_time.train = metadata_time;
    stats[epoch].raw_read_time.train = read_time;
}

// Saving the start time of the evaluation process
void
start_eval(uint32_t epoch)
{
    stats[epoch].start_time.eval = get_time_usec_return_uint64();
}

// Saving data on the evaluation process
void
end_eval(uint32_t epoch, uint64_t metadata_time, uint64_t read_time)
{
    uint64_t end_time               = get_time_usec_return_uint64();
    stats[epoch].observed_time.eval = end_time - stats[epoch].start_time.eval;
    if ((end_time - stats[epoch].start_time.eval) == 0) {
        stats[epoch].throughput.eval = NAN;
    }
    else {
        stats[epoch].throughput.eval = (double)config.NUM_EVAL_BATCHES_PER_RANK * config.BATCH_SIZE_EVAL *
                                       1000000.0 / (end_time - stats[epoch].start_time.eval);
    }
    stats[epoch].metadata_time.eval = metadata_time;
    stats[epoch].raw_read_time.eval = read_time;
}
