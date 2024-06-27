#include <mpi.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

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

void
stats_initialize()
{
    //    TODO: drop_last = False
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
        global_stats[i].compute.train = (uint64_t *)calloc(config.NUM_TRAIN_BATCHES_PER_RANK, sizeof(uint64_t));
        if (global_stats[i].compute.train == NULL) {
            exit(1);
        }
        global_stats[i].compute.eval = (uint64_t *)calloc(config.NUM_EVAL_BATCHES_PER_RANK, sizeof(uint64_t));
        if (global_stats[i].compute.eval == NULL) {
            exit(1);
        }

        MPI_Reduce(stats[i].load.train, global_stats[i].load.train, config.NUM_TRAIN_BATCHES_PER_RANK, MPI_UNSIGNED_LONG_LONG,
                   MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(stats[i].load.eval, global_stats[i].load.eval, config.NUM_EVAL_BATCHES_PER_RANK, MPI_UNSIGNED_LONG_LONG,
                   MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(stats[i].proc.train, global_stats[i].proc.train, config.NUM_TRAIN_BATCHES_PER_RANK, MPI_UNSIGNED_LONG_LONG,
                   MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(stats[i].proc.eval, global_stats[i].proc.eval, config.NUM_EVAL_BATCHES_PER_RANK, MPI_UNSIGNED_LONG_LONG,
                   MPI_SUM, 0, MPI_COMM_WORLD);
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

void
print_data()
{
    printf("metric, value\n");
    printf("operation, dlio\n");
    printf("ranks, %d\n", NUM_RANKS);
    printf("read threads, %d\n", config.READ_THREADS);
    printf("subfiling, %s\n", config.SUBFILING ? "YES" : "NO");
    printf("chunking, %s\n", config.DO_CHUNKING ? "YES" : "NO");
    printf("collective meta, %s\n", config.COLLECTIVE_META ? "YES" : "NO");
    printf("collective data, %s\n", config.COLLECTIVE_DATA ? "YES" : "NO");

    // Train
    // TODO: drop_last = false
    uint64_t train_total_size_bytes = (uint64_t)config.NUM_TRAIN_BATCHES_PER_RANK * NUM_RANKS * config.BATCH_SIZE * config.RECORD_LENGTH;
    printf("train total size, %" PRId64 "\n", train_total_size_bytes);
    uint64_t train_size_bytes_per_rank = (uint64_t)config.NUM_TRAIN_BATCHES_PER_RANK * config.BATCH_SIZE * config.RECORD_LENGTH;
    printf("train size per rank, %" PRId64 "\n", train_size_bytes_per_rank);

    printf("train emulated compute time per epoch, \"");
    uint64_t train_total_compute_time = 0;
    for (uint32_t i = 0; i < config.EPOCHS; i++) {
        unsigned long int compute_time = 0;
        for (uint32_t j = 0; j < config.NUM_TRAIN_BATCHES_PER_RANK; j++) {
            compute_time += global_stats[i].compute.train[j];
        }
        train_total_compute_time += compute_time;
        printf("%lf", compute_time / 1000000.0);
        if (i != config.EPOCHS - 1)
            printf(", ");
    }
    printf("\"\ntrain emulated compute time, %lf\n", train_total_compute_time / 1000000.0);

    printf("train metadata time per epoch, \"");
    double train_total_metadata_time = 0;
    for (uint32_t i = 0; i < config.EPOCHS; i++) {
        double metadata_time = stats[i].metadata_time.train / 1000000.0;
        train_total_metadata_time += metadata_time;
        printf("%lf", metadata_time);
        if (i != config.EPOCHS - 1)
            printf(", ");
    }

    printf("\"\ntrain metadata time, %lf\n", train_total_metadata_time);

    printf("train raw read time per epoch, \"");
    double train_total_read_time = 0;
    for (uint32_t i = 0; i < config.EPOCHS; i++) {
        double read_time = stats[i].raw_read_time.train / 1000000.0;
        train_total_read_time += read_time;
        printf("%lf", read_time);
        if (i != config.EPOCHS - 1)
            printf(", ");
    }
    printf("\"\ntrain total raw read time, %lf\n", train_total_read_time);

    printf("train raw read rate per epoch, \"");
    double train_total_avg_read_rate = 0;
    for (uint32_t i = 0; i < config.EPOCHS; i++) {
        double read_rate = (double)train_size_bytes_per_rank / stats[i].raw_read_time.train * 1000000.0;
        train_total_avg_read_rate += read_rate;
        printf("%lf", read_rate);
        if (i != config.EPOCHS - 1)
            printf(", ");
    }
    printf("\"\ntrain avg raw read rate, %lf\n", train_total_avg_read_rate / config.EPOCHS);

    printf("train observed time per epoch, \"");
    double train_total_observed_time = 0;
    for (uint32_t i = 0; i < config.EPOCHS; i++) {
        double observed_time = global_stats[i].observed_time.train / 1000000.0;
        train_total_observed_time += observed_time;
        printf("%lf", observed_time);
        if (i != config.EPOCHS - 1)
            printf(", ");
    }
    printf("\"\ntrain observed time, %lf\n", train_total_observed_time);

    printf("train observed rate per epoch, \"");
    double train_total_avg_observed_rate = 0.0;
    for (uint32_t i = 0; i < config.EPOCHS; i++) {
        unsigned long int compute_time = 0;
        for (uint32_t j = 0; j < config.NUM_TRAIN_BATCHES_PER_RANK; j++) {
            compute_time += global_stats[i].compute.train[j];
        }
        double observed_rate = (double)train_size_bytes_per_rank / (global_stats[i].observed_time.train - compute_time) *
                               1000000.0;
        train_total_avg_observed_rate += observed_rate;
        printf("%lf", observed_rate);
        if (i != config.EPOCHS - 1)
            printf(", ");
    }
    printf("\"\ntrain avg observed rate, %lf\n", train_total_avg_observed_rate / config.EPOCHS);

    printf("train throughput samples per second per epoch, \"");
    double train_throughput_mean_samples_per_second = 0.0;
    for (uint32_t i = 0; i < config.EPOCHS; i++) {
        train_throughput_mean_samples_per_second += global_stats[i].throughput.train;
        printf("%lf", global_stats[i].throughput.train);
        if (i != config.EPOCHS - 1)
            printf(", ");
    }
    train_throughput_mean_samples_per_second =
        train_throughput_mean_samples_per_second / (double)config.EPOCHS;
    printf("\"\ntrain throughput avg samples per second, %lf\n", train_throughput_mean_samples_per_second);

    double train_throughput_stdev_samples_per_second = 0.0;
    for (uint32_t i = 0; i < config.EPOCHS; i++) {
        train_throughput_stdev_samples_per_second +=
            (global_stats[i].throughput.train - train_throughput_mean_samples_per_second) *
            (global_stats[i].throughput.train - train_throughput_mean_samples_per_second);
    }
    train_throughput_stdev_samples_per_second =
        sqrt(train_throughput_stdev_samples_per_second / (double)config.EPOCHS);
    printf("train throughput stdev samples per second, %lf\n", train_throughput_stdev_samples_per_second);

    double train_io_mean_MB_per_second =
        train_throughput_mean_samples_per_second * config.RECORD_LENGTH / 1024 / 1024;
    printf("train io avg MB per second, %lf\n", train_io_mean_MB_per_second);

    double train_io_stdev_MB_per_second =
        train_throughput_stdev_samples_per_second * config.RECORD_LENGTH / 1024 / 1024;
    printf("train io stdev MB per second, %lf\n", train_io_stdev_MB_per_second);

    // Evaluation
    // TODO: drop_last = False
    uint64_t eval_total_size_bytes = (uint64_t)config.NUM_EVAL_BATCHES_PER_RANK * NUM_RANKS * config.BATCH_SIZE_EVAL * config.RECORD_LENGTH;
    printf("eval total size, %" PRId64 "\n", eval_total_size_bytes);
    uint64_t eval_size_bytes_per_rank = (uint64_t)config.NUM_EVAL_BATCHES_PER_RANK * config.BATCH_SIZE_EVAL * config.RECORD_LENGTH;
    printf("eval size per rank, %" PRId64 "\n", eval_size_bytes_per_rank);

    printf("eval emulated compute time per epoch, \"");
    uint64_t eval_total_compute_time = 0;
    for (uint32_t i = 0; i < config.EPOCHS; i++) {
        unsigned long int compute_time = 0;
        for (uint32_t j = 0; j < config.NUM_EVAL_BATCHES_PER_RANK; j++) {
            compute_time += global_stats[i].compute.eval[j];
        }
        eval_total_compute_time += compute_time;
        printf("%lf", compute_time / 1000000.0);
        if (i != config.EPOCHS - 1)
            printf(", ");
    }
    printf("\"\neval emulated compute time, %lf\n", eval_total_compute_time / 1000000.0);

    printf("eval metadata time per epoch, \"");
    double eval_total_metadata_time = 0;
    for (uint32_t i = 0; i < config.EPOCHS; i++) {
        double metadata_time = stats[i].metadata_time.eval / 1000000.0;
        eval_total_metadata_time += metadata_time;
        printf("%lf", metadata_time);
        if (i != config.EPOCHS - 1)
            printf(", ");
    }

    printf("\"\neval metadata time, %lf\n", eval_total_metadata_time);

    printf("eval raw read time per epoch, \"");
    double eval_total_read_time = 0;
    for (uint32_t i = 0; i < config.EPOCHS; i++) {
        double read_time = stats[i].raw_read_time.eval / 1000000.0;
        eval_total_read_time += read_time;
        printf("%lf", read_time);
        if (i != config.EPOCHS - 1)
            printf(", ");
    }
    printf("\"\neval total raw read time, %lf\n", eval_total_read_time);

    printf("eval raw read rate per epoch, \"");
    double eval_total_avg_read_rate = 0;
    for (uint32_t i = 0; i < config.EPOCHS; i++) {
        double read_rate = (double)eval_size_bytes_per_rank / stats[i].raw_read_time.eval * 1000000.0;
        eval_total_avg_read_rate += read_rate;
        printf("%lf", read_rate);
        if (i != config.EPOCHS - 1)
            printf(", ");
    }
    printf("\"\neval avg raw read rate, %lf\n", eval_total_avg_read_rate / config.EPOCHS);

    printf("eval observed time per epoch, \"");
    double eval_total_observed_time = 0;
    for (uint32_t i = 0; i < config.EPOCHS; i++) {
        double observed_time = global_stats[i].observed_time.eval / 1000000.0;
        eval_total_observed_time += observed_time;
        printf("%lf", observed_time);
        if (i != config.EPOCHS - 1)
            printf(", ");
    }
    printf("\"\neval observed time, %lf\n", eval_total_observed_time);

    printf("eval observed rate per epoch, \"");
    double eval_total_avg_observed_rate = 0.0;
    for (uint32_t i = 0; i < config.EPOCHS; i++) {
        unsigned long compute_time = 0;
        for (uint32_t j = 0; j < config.NUM_EVAL_BATCHES_PER_RANK; j++) {
            compute_time += global_stats[i].compute.eval[j];
        }
        double observed_rate = (double)eval_size_bytes_per_rank / (global_stats[i].observed_time.eval - compute_time) *
                               1000000.0;
        eval_total_avg_observed_rate += observed_rate;
        printf("%lf", observed_rate);
        if (i != config.EPOCHS - 1)
            printf(", ");
    }
    printf("\"\neval avg observed rate, %lf\n", eval_total_avg_observed_rate / config.EPOCHS);

    printf("eval throughput samples per second per epoch, \"");
    double eval_throughput_mean_samples_per_second = 0.0;
    for (uint32_t i = 0; i < config.EPOCHS; i++) {
        eval_throughput_mean_samples_per_second += global_stats[i].throughput.eval;
        printf("%lf", global_stats[i].throughput.eval);
        if (i != config.EPOCHS - 1)
            printf(", ");
    }
    eval_throughput_mean_samples_per_second = eval_throughput_mean_samples_per_second / (double)config.EPOCHS;
    printf("\"\neval throughput avg samples per second, %lf\n", eval_throughput_mean_samples_per_second);

    double eval_throughput_stdev_samples_per_second = 0.0;
    for (uint32_t i = 0; i < config.EPOCHS; i++) {
        eval_throughput_stdev_samples_per_second +=
            (global_stats[i].throughput.eval - eval_throughput_mean_samples_per_second) *
            (global_stats[i].throughput.eval - eval_throughput_mean_samples_per_second);
    }
    eval_throughput_stdev_samples_per_second =
        sqrt(eval_throughput_stdev_samples_per_second / (double)config.EPOCHS);
    printf("eval throughput stdev samples per second, %lf\n", eval_throughput_stdev_samples_per_second);

    double eval_io_mean_MB_per_second =
        eval_throughput_mean_samples_per_second * config.RECORD_LENGTH / 1024 / 1024;
    printf("eval io avg MB per second, %lf\n", eval_io_mean_MB_per_second);

    double eval_io_stdev_MB_per_second =
        eval_throughput_stdev_samples_per_second * config.RECORD_LENGTH / 1024 / 1024;
    printf("eval io stdev MB per second, %lf\n", eval_io_stdev_MB_per_second);
}

void
batch_loaded_train(uint32_t epoch, uint64_t t0)
{
    stats[epoch].load.train[last_load_train[epoch]++] = (get_time_usec() - t0);
}

void
batch_processed_train(uint32_t epoch, uint64_t computation_time, uint64_t t0)
{
    stats[epoch].proc.train[last_proc_train[epoch]++]       = (get_time_usec() - t0);
    stats[epoch].compute.train[last_compute_train[epoch]++] = computation_time;
}

void
batch_loaded_eval(uint32_t epoch, uint64_t t0)
{
    stats[epoch].load.eval[last_load_eval[epoch]++] = (get_time_usec() - t0);
}

void
batch_processed_eval(uint32_t epoch, uint64_t computation_time, uint64_t t0)
{
    stats[epoch].proc.eval[last_proc_eval[epoch]++]       = (get_time_usec() - t0);
    stats[epoch].compute.eval[last_compute_eval[epoch]++] = computation_time;
}

void
start_train(uint32_t epoch)
{
    stats[epoch].start_time.train = get_time_usec();
}

void
end_train(uint32_t epoch, uint64_t metadata_time, uint64_t read_time)
{
    uint64_t end_time                = get_time_usec();
    stats[epoch].observed_time.train = end_time - stats[epoch].start_time.train;
    stats[epoch].throughput.train =
        (double)config.NUM_TRAIN_BATCHES_PER_RANK * config.BATCH_SIZE * 1000000.0 / (end_time - stats[epoch].start_time.train);
    stats[epoch].metadata_time.train = metadata_time;
    stats[epoch].raw_read_time.train = read_time;
}

void
start_eval(uint32_t epoch)
{
    stats[epoch].start_time.eval = get_time_usec();
}

void
end_eval(uint32_t epoch, uint64_t metadata_time, uint64_t read_time)
{
    uint64_t end_time               = get_time_usec();
    stats[epoch].observed_time.eval = end_time - stats[epoch].start_time.eval;
    stats[epoch].throughput.eval    = (double)config.NUM_EVAL_BATCHES_PER_RANK * config.BATCH_SIZE_EVAL * 1000000.0 /
                                   (end_time - stats[epoch].start_time.eval);
    stats[epoch].metadata_time.eval = metadata_time;
    stats[epoch].raw_read_time.eval = read_time;
}
