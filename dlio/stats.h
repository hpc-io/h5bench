#ifndef SANDBOX_STATS_H
#define SANDBOX_STATS_H

struct load_data {
    uint64_t *train;
    uint64_t *eval;
};

struct proc_data {
    uint64_t *train;
    uint64_t *eval;
};

struct throughput_data {
    double train;
    double eval;
};

struct au_data {
    double train;
    double eval;
};

struct compute_data {
    uint64_t *train;
    uint64_t *eval;
};

struct start_time_data {
    uint64_t train;
    uint64_t eval;
};

struct observed_time_data {
    uint64_t train;
    uint64_t eval;
};

typedef struct epoch_data {
    struct start_time_data start_time;
    struct load_data load;
    struct proc_data proc;
    struct throughput_data throughput;
    struct au_data au;
    struct compute_data compute;
    struct observed_time_data observed_time;
} epoch_data_t;

void stats_initialize();

void stats_finalize();

void prepare_data();

void print_data(uint64_t *train_metadata_time, uint64_t *train_read_time,
                uint64_t *eval_metadata_time, uint64_t *eval_read_time);

void batch_loaded_train(uint32_t epoch, uint64_t start_time);

void batch_processed_train(uint32_t epoch, uint64_t computation_time, uint64_t start_time);

void batch_loaded_eval(uint32_t epoch, uint64_t t0);

void batch_processed_eval(uint32_t epoch, uint64_t computation_time, uint64_t t0);

void start_train(uint32_t epoch);

void end_train(uint32_t epoch);

void start_eval(uint32_t epoch);

void end_eval(uint32_t epoch);

#endif //SANDBOX_STATS_H
