#include <mpi.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "h5bench_dlio.h"
#include "stats.h"
#include "utils.h"

uint32_t TRAIN_MAX_STEPS;
uint32_t EVAL_MAX_STEPS;
epoch_data_t *stats;
epoch_data_t *global_stats;

uint32_t *last_load_train;
uint32_t *last_load_eval;
uint32_t *last_proc_train;
uint32_t *last_proc_eval;
uint32_t *last_compute_train;
uint32_t *last_compute_eval;

double AU;

void stats_initialize() {
    AU = 0.90;

    uint32_t train_steps_count = config.NUM_FILES_TRAIN * config.NUM_SAMPLES_PER_FILE / config.BATCH_SIZE / NUM_RANKS;
    uint32_t train_steps_count_remainder = config.NUM_FILES_TRAIN * config.NUM_SAMPLES_PER_FILE % (config.BATCH_SIZE * NUM_RANKS);
    uint32_t eval_steps_count = config.NUM_FILES_EVAL * config.NUM_SAMPLES_PER_FILE / config.BATCH_SIZE_EVAL / NUM_RANKS;
    uint32_t eval_steps_count_remainder = config.NUM_FILES_EVAL * config.NUM_SAMPLES_PER_FILE % (config.BATCH_SIZE_EVAL * NUM_RANKS);

    TRAIN_MAX_STEPS = train_steps_count;
    EVAL_MAX_STEPS = eval_steps_count;

//    TODO: drop_last = False
//    TRAIN_MAX_STEPS = train_steps_count + (train_steps_count_remainder > 0);
//    EVAL_MAX_STEPS = eval_steps_count + (eval_steps_count_remainder > 0);

    stats = (struct epoch_data *)malloc(config.EPOCHS * sizeof(struct epoch_data));
    if (stats == NULL) {
        exit(1);
    }

    for (uint32_t i = 0; i < config.EPOCHS; i++) {
        stats[i].load.train = (uint64_t*)calloc(TRAIN_MAX_STEPS, sizeof(uint64_t));
        if (stats[i].load.train == NULL) {
            exit(1);
        }
        stats[i].load.eval = (uint64_t*)calloc(EVAL_MAX_STEPS, sizeof(uint64_t));
        if (stats[i].load.eval == NULL) {
            exit(1);
        }
        stats[i].proc.train = (uint64_t*)calloc(TRAIN_MAX_STEPS, sizeof(uint64_t));
        if (stats[i].proc.train == NULL) {
            exit(1);
        }
        stats[i].proc.eval = (uint64_t*)calloc(EVAL_MAX_STEPS, sizeof(uint64_t));
        if (stats[i].proc.eval == NULL) {
            exit(1);
        }
        stats[i].throughput.train = 0.0;
        stats[i].throughput.eval = 0.0;
        stats[i].au.train = 0.0;
        stats[i].au.eval = 0.0;
        stats[i].compute.train = (uint64_t*)calloc(TRAIN_MAX_STEPS, sizeof(uint64_t));
        if (stats[i].compute.train == NULL) {
            exit(1);
        }
        stats[i].compute.eval = (uint64_t*)calloc(EVAL_MAX_STEPS, sizeof(uint64_t));
        if (stats[i].compute.eval == NULL) {
            exit(1);
        }
        stats[i].observed_time.train = 0;
        stats[i].observed_time.eval = 0;
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

void stats_finalize() {
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

void prepare_data() {
    global_stats = (struct epoch_data *)malloc(config.EPOCHS * sizeof(struct epoch_data));
    if (global_stats == NULL) {
        exit(1);
    }

    for (uint32_t i = 0; i < config.EPOCHS; i++) {
        global_stats[i].load.train = (uint64_t*)calloc(TRAIN_MAX_STEPS, sizeof(uint64_t));
        if (global_stats[i].load.train == NULL) {
            exit(1);
        }
        global_stats[i].load.eval = (uint64_t*)calloc(EVAL_MAX_STEPS, sizeof(uint64_t));
        if (global_stats[i].load.eval == NULL) {
            exit(1);
        }
        global_stats[i].proc.train = (uint64_t*)calloc(TRAIN_MAX_STEPS, sizeof(uint64_t));
        if (global_stats[i].proc.train == NULL) {
            exit(1);
        }
        global_stats[i].proc.eval = (uint64_t*)calloc(EVAL_MAX_STEPS, sizeof(uint64_t));
        if (global_stats[i].proc.eval == NULL) {
            exit(1);
        }
        global_stats[i].compute.train = (uint64_t*)calloc(TRAIN_MAX_STEPS, sizeof(uint64_t));
        if (global_stats[i].compute.train == NULL) {
            exit(1);
        }
        global_stats[i].compute.eval = (uint64_t*)calloc(EVAL_MAX_STEPS, sizeof(uint64_t));
        if (global_stats[i].compute.eval == NULL) {
            exit(1);
        }

        MPI_Reduce(stats[i].load.train, global_stats[i].load.train, TRAIN_MAX_STEPS, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(stats[i].load.eval, global_stats[i].load.eval, EVAL_MAX_STEPS, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(stats[i].proc.train, global_stats[i].proc.train, TRAIN_MAX_STEPS, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(stats[i].proc.eval, global_stats[i].proc.eval, EVAL_MAX_STEPS, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(&stats[i].au.train, &global_stats[i].au.train, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(&stats[i].au.eval, &global_stats[i].au.eval, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(&stats[i].throughput.train, &global_stats[i].throughput.train, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(&stats[i].throughput.eval, &global_stats[i].throughput.eval, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(stats[i].compute.train, global_stats[i].compute.train, TRAIN_MAX_STEPS, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(stats[i].compute.eval, global_stats[i].compute.eval, EVAL_MAX_STEPS, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(&stats[i].observed_time.train, &global_stats[i].observed_time.train, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(&stats[i].observed_time.eval, &global_stats[i].observed_time.eval, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

        for (int j = 0; j < TRAIN_MAX_STEPS; j++) {
            global_stats[i].load.train[j] /= NUM_RANKS;
            global_stats[i].proc.train[j] /= NUM_RANKS;
            global_stats[i].compute.train[j] /= NUM_RANKS;
        }

        for (int j = 0; j < EVAL_MAX_STEPS; j++) {
            global_stats[i].load.eval[j] /= NUM_RANKS;
            global_stats[i].proc.eval[j] /= NUM_RANKS;
            global_stats[i].compute.eval[j] /= NUM_RANKS;
        }

        global_stats[i].au.train /= NUM_RANKS;
        global_stats[i].au.eval /= NUM_RANKS;
        global_stats[i].throughput.train /= NUM_RANKS;
        global_stats[i].throughput.eval /= NUM_RANKS;
        global_stats[i].observed_time.train /= NUM_RANKS;
        global_stats[i].observed_time.eval /= NUM_RANKS;
    }
}

void print_data(uint64_t *train_metadata_time, uint64_t *train_read_time,
                uint64_t *eval_metadata_time, uint64_t *eval_read_time) {

    printf("metric, value\n");
    printf("operation, dlio\n");
    printf("ranks, %d\n", NUM_RANKS);
//    printf("collective meta");
//    printf("collective data");
    // Train
    printf("train compute time, \"");
    uint64_t train_total_compute_time = 0;
    for (uint32_t i = 0; i < config.EPOCHS; i++) {
        unsigned long int compute_time = 0;
        for (uint32_t j = 0; j < TRAIN_MAX_STEPS; j++) {
            compute_time += global_stats[i].compute.train[j];
        }
        train_total_compute_time += compute_time;
        printf("%lf", compute_time / 1000000.0);
        if (i != config.EPOCHS - 1) printf(", ");
    }
    printf("\"\ntrain total compute time, %lf\n", train_total_compute_time / 1000000.0);

    // TODO: drop_last = False
    uint64_t train_total_batches = (uint64_t)config.NUM_FILES_TRAIN * config.NUM_SAMPLES_PER_FILE / config.BATCH_SIZE / NUM_RANKS * NUM_RANKS;
    uint64_t train_total_size_bytes = train_total_batches * config.BATCH_SIZE_EVAL * config.NUM_SAMPLES_PER_FILE * config.RECORD_LENGTH;
    printf("train total size, %lu\n", train_total_size_bytes);

    printf("train total metadata time, %lf\n", *train_metadata_time / 1000000.0);
    printf("train total raw read time, %lf\n", *train_read_time / 1000000.0);
    printf("train total raw read rate, %lf\n", (double)train_total_size_bytes / *train_read_time * 1000000.0);

    printf("train observed time, \"");
    double train_total_observed_time = 0;
    for (uint32_t i = 0; i < config.EPOCHS; i++) {
        double observed_time = global_stats[i].observed_time.train / 1000000.0;
        train_total_observed_time += observed_time;
        printf("%lf", observed_time);
        if (i != config.EPOCHS - 1) printf(", ");
    }
    printf("\"\ntrain total observed time, %lf\n", train_total_observed_time);

    printf("train observed rate, \"");
    for (uint32_t i = 0; i < config.EPOCHS; i++) {
        unsigned long int compute_time = 0;
        for (uint32_t j = 0; j < TRAIN_MAX_STEPS; j++) {
            compute_time += global_stats[i].compute.train[j];
        }
        printf("%lf", (double)train_total_size_bytes / (global_stats[i].observed_time.train - compute_time) * 1000000.0);
        if (i != config.EPOCHS - 1) printf(", ");
    }
    printf("\"\n");

    printf("train au percentage, \"");
    double train_au_mean_percentage = 0.0;
    for (uint32_t i = 0; i < config.EPOCHS; i++) {
        train_au_mean_percentage += global_stats[i].au.train;
        printf("%lf", global_stats[i].au.train);
        if (i != config.EPOCHS - 1) printf(", ");
    }
    train_au_mean_percentage = train_au_mean_percentage / (double)config.EPOCHS;
    printf("\"\ntrain au mean percentage, %lf\n", train_au_mean_percentage);
    printf("train au meet expectation, %s\n", train_au_mean_percentage >= 100 * AU? "success": "fail");

    double train_au_stdev_percentage = 0.0;
    for (uint32_t i = 0; i < config.EPOCHS; i++) {
        train_au_stdev_percentage += (global_stats[i].au.train - train_au_mean_percentage) * (global_stats[i].au.train - train_au_mean_percentage);
    }
    train_au_stdev_percentage = sqrt(train_au_stdev_percentage / (double)config.EPOCHS);
    printf("train au stdev percentage, %lf\n", train_au_stdev_percentage);

    printf("train throughput samples per second, \"");
    double train_throughput_mean_samples_per_second = 0.0;
    for (uint32_t i = 0; i < config.EPOCHS; i++) {
        train_throughput_mean_samples_per_second += global_stats[i].throughput.train;
        printf("%lf", global_stats[i].throughput.train);
        if (i != config.EPOCHS - 1) printf(", ");
    }
    train_throughput_mean_samples_per_second = train_throughput_mean_samples_per_second / (double)config.EPOCHS;
    printf("\"\ntrain throughput mean samples per second, %lf\n", train_throughput_mean_samples_per_second);

    double train_throughput_stdev_samples_per_second = 0.0;
    for (uint32_t i = 0; i < config.EPOCHS; i++) {
        train_throughput_stdev_samples_per_second += (global_stats[i].throughput.train - train_throughput_mean_samples_per_second) * (global_stats[i].throughput.train - train_throughput_mean_samples_per_second);
    }
    train_throughput_stdev_samples_per_second = sqrt(train_throughput_stdev_samples_per_second / (double)config.EPOCHS);
    printf("train throughput stdev samples per second, %lf\n", train_throughput_stdev_samples_per_second);

    double train_io_mean_MB_per_second = train_throughput_mean_samples_per_second * config.RECORD_LENGTH / 1024 / 1024;
    printf("train io mean MB per second, %lf\n", train_io_mean_MB_per_second);

    double train_io_stdev_MB_per_second = train_throughput_stdev_samples_per_second * config.RECORD_LENGTH / 1024 / 1024;
    printf("train io stdev MB per second, %lf\n", train_io_stdev_MB_per_second);

    // Evaluation
    printf("eval compute time, \"");
    uint64_t eval_total_compute_time = 0;
    for (uint32_t i = 0; i < config.EPOCHS; i++) {
        unsigned long int compute_time = 0;
        for (uint32_t j = 0; j < EVAL_MAX_STEPS; j++) {
            compute_time += global_stats[i].compute.eval[j];
        }
        eval_total_compute_time += compute_time;
        printf("%lf", compute_time / 1000000.0);
        if (i != config.EPOCHS - 1) printf(", ");
    }
    printf("\"\neval total compute time, %lf\n", eval_total_compute_time / 1000000.0);
    // TODO: drop_last = False
    uint64_t eval_total_batches = (uint64_t)config.NUM_FILES_EVAL * config.NUM_SAMPLES_PER_FILE / config.BATCH_SIZE_EVAL / NUM_RANKS * NUM_RANKS;
    uint64_t eval_total_size_bytes = eval_total_batches * config.BATCH_SIZE_EVAL * config.NUM_SAMPLES_PER_FILE * config.RECORD_LENGTH;
    printf("eval total size, %lu\n", eval_total_size_bytes);

    printf("eval metadata time, %lf\n", *eval_metadata_time / 1000000.0);
    printf("eval raw read time, %lf\n", *eval_read_time / 1000000.0);
    printf("eval raw read rate, %lf\n", (double)eval_total_size_bytes / *eval_read_time * 1000000.0);

    printf("eval observed time, \"");
    double eval_total_observed_time = 0;
    for (uint32_t i = 0; i < config.EPOCHS; i++) {
        double observed_time = global_stats[i].observed_time.eval / 1000000.0;
        eval_total_observed_time += observed_time;
        printf("%lf", observed_time);
        if (i != config.EPOCHS - 1) printf(", ");
    }
    printf("\"\neval total observed time, %lf\n", eval_total_observed_time);

    printf("eval observed rate, \"");
    for (uint32_t i = 0; i < config.EPOCHS; i++) {
        unsigned long compute_time = 0;
        for (uint32_t j = 0; j < EVAL_MAX_STEPS; j++) {
            compute_time += global_stats[i].compute.eval[j];
        }
        printf("%lf", (double)eval_total_size_bytes / (global_stats[i].observed_time.eval - compute_time) * 1000000.0);
        if (i != config.EPOCHS - 1) printf(", ");
    }
    printf("\"\n");

    printf("eval au percentage, \"");
    double eval_au_mean_percentage = 0.0;
    for (uint32_t i = 0; i < config.EPOCHS; i++) {
        eval_au_mean_percentage += global_stats[i].au.eval;
        printf("%lf", global_stats[i].au.eval);
        if (i != config.EPOCHS - 1) printf(", ");
    }
    eval_au_mean_percentage = eval_au_mean_percentage / (double)config.EPOCHS;
    printf("\"\neval au mean percentage, %lf\n", eval_au_mean_percentage);
    printf("eval au meet expectation, %s\n", eval_au_mean_percentage >= 100 * AU? "success": "fail");

    double eval_au_stdev_percentage = 0.0;
    for (uint32_t i = 0; i < config.EPOCHS; i++) {
        eval_au_stdev_percentage += (global_stats[i].au.eval - eval_au_mean_percentage) * (global_stats[i].au.eval - eval_au_mean_percentage);
    }
    eval_au_stdev_percentage = sqrt(eval_au_stdev_percentage / (double)config.EPOCHS);
    printf("eval au stdev percentage, %lf\n", eval_au_stdev_percentage);

    printf("eval throughput samples per second, \"");
    double eval_throughput_mean_samples_per_second = 0.0;
    for (uint32_t i = 0; i < config.EPOCHS; i++) {
        eval_throughput_mean_samples_per_second += global_stats[i].throughput.eval;
        printf("%lf", global_stats[i].throughput.eval);
        if (i != config.EPOCHS - 1) printf(", ");
    }
    eval_throughput_mean_samples_per_second = eval_throughput_mean_samples_per_second / (double)config.EPOCHS;
    printf("\"\neval throughput mean samples per second, %lf\n", eval_throughput_mean_samples_per_second);

    double eval_throughput_stdev_samples_per_second = 0.0;
    for (uint32_t i = 0; i < config.EPOCHS; i++) {
        eval_throughput_stdev_samples_per_second += (global_stats[i].throughput.eval - eval_throughput_mean_samples_per_second) * (global_stats[i].throughput.eval - eval_throughput_mean_samples_per_second);
    }
    eval_throughput_stdev_samples_per_second = sqrt(eval_throughput_stdev_samples_per_second / (double)config.EPOCHS);
    printf("eval throughput stdev samples per second, %lf\n", eval_throughput_stdev_samples_per_second);

    double eval_io_mean_MB_per_second = eval_throughput_mean_samples_per_second * config.RECORD_LENGTH / 1024 / 1024;
    printf("eval io mean MB per second, %lf\n", eval_io_mean_MB_per_second);

    double eval_io_stdev_MB_per_second = eval_throughput_stdev_samples_per_second * config.RECORD_LENGTH / 1024 / 1024;
    printf("eval io stdev MB per second, %lf\n", eval_io_stdev_MB_per_second);
}

void batch_loaded_train(uint32_t epoch, uint64_t t0) {
    stats[epoch].load.train[last_load_train[epoch]++] = (get_time_usec() - t0);
}

void batch_processed_train(uint32_t epoch, uint64_t computation_time, uint64_t t0) {
    stats[epoch].proc.train[last_proc_train[epoch]++] = (get_time_usec() - t0);
    stats[epoch].compute.train[last_compute_train[epoch]++] = computation_time;
}

void batch_loaded_eval(uint32_t epoch, uint64_t t0) {
    stats[epoch].load.eval[last_load_eval[epoch]++] = (get_time_usec() - t0);
}

void batch_processed_eval(uint32_t epoch, uint64_t computation_time, uint64_t t0) {
    stats[epoch].proc.eval[last_proc_eval[epoch]++] = (get_time_usec() - t0);
    stats[epoch].compute.eval[last_compute_eval[epoch]++] = computation_time;
}

void start_train(uint32_t epoch) {
    stats[epoch].start_time.train = get_time_usec();
}

void end_train(uint32_t epoch) {
    uint64_t end_time = get_time_usec();
    uint64_t total_compute_time = 0;
    double au = 0.0;

    for (int i = 0; i < TRAIN_MAX_STEPS; i++) {
        total_compute_time += stats[epoch].compute.train[i];
    }
    if (total_compute_time > 0) {
        stats[epoch].observed_time.train = end_time - stats[epoch].start_time.train;
        au = (double)total_compute_time / stats[epoch].observed_time.train;
    }

    stats[epoch].au.train = au * 100;
    stats[epoch].throughput.train = (double)TRAIN_MAX_STEPS * config.BATCH_SIZE * 1000000.0 / (end_time - stats[epoch].start_time.train);
}

void start_eval(uint32_t epoch) {
    stats[epoch].start_time.eval = get_time_usec();
}

void end_eval(uint32_t epoch) {
    uint64_t end_time = get_time_usec();
    uint64_t total_compute_time = 0;
    double au = 0.0;

    for (int i = 0; i < EVAL_MAX_STEPS; i++) {
        total_compute_time += stats[epoch].compute.eval[i];
    }
    if (total_compute_time > 0) {
        stats[epoch].observed_time.eval = end_time - stats[epoch].start_time.eval;
        au = (double)total_compute_time / stats[epoch].observed_time.eval;
    }
    stats[epoch].au.eval = au * 100;
    stats[epoch].throughput.eval = (double)EVAL_MAX_STEPS * config.BATCH_SIZE_EVAL * 1000000.0 / (end_time - stats[epoch].start_time.eval);
}

