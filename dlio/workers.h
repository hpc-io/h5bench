#ifndef H5BENCH_WORKERS_H
#define H5BENCH_WORKERS_H

#include <stdint.h>
#include <stdbool.h>

typedef struct execution_time {
    uint64_t metadata_time;
    uint64_t read_time;
} execution_time_t;

void init_workers(uint32_t *indices_train, uint32_t *indices_eval);

int get_train_read_fd();

int get_eval_read_fd();

int get_train_write_fd();

int get_eval_write_fd();

int get_train_system_fd();

int get_eval_system_fd();

void fin_workers();

void force_workers_to_shuffle(int read_fd, int write_fd, int system_fd);

void run_worker(uint32_t *indices, int pipe_task_fd[2], int pipe_result_fd[2], int pipe_system_fd[2],
                bool is_train_worker);

#endif // H5BENCH_WORKERS_H
