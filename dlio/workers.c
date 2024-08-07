// TODO: handle errors in child processes

#include <stdint.h>
#include <stdlib.h>
#include <sys/types.h>
#include <unistd.h>
#include <sys/wait.h>

#include "h5bench_dlio.h"
#include "workers.h"
#include "utils.h"

int pipe_train_task_fd[2], pipe_train_result_fd[2], pipe_eval_task_fd[2], pipe_eval_result_fd[2];
int pipe_train_system_fd[2], pipe_eval_system_fd[2];

// Initialization of processes that will be used later on in the simulation of data processing
void
init_workers(uint32_t *indices_train, uint32_t *indices_eval)
{
    if ((pipe(pipe_train_system_fd) == -1) || (pipe(pipe_train_task_fd) == -1) ||
        (pipe(pipe_train_result_fd) == -1)) {
        perror("pipe");
        exit(EXIT_FAILURE);
    }

    for (uint32_t i = 0; i < config.READ_THREADS; i++) {
        pid_t pid = fork();
        if (pid == -1) {
            perror("fork");
            exit(EXIT_FAILURE);
        }
        else if (pid == 0) {
            close(pipe_train_task_fd[1]);
            close(pipe_train_result_fd[0]);
            close(pipe_train_system_fd[1]);

            run_worker(indices_train, pipe_train_task_fd, pipe_train_result_fd, pipe_train_system_fd, true);

            close(pipe_train_task_fd[0]);
            close(pipe_train_result_fd[1]);
            close(pipe_train_system_fd[0]);
            exit(EXIT_SUCCESS);
        }
    }

    if (config.DO_EVALUATION) {
        if ((pipe(pipe_eval_system_fd) == -1) || (pipe(pipe_eval_task_fd) == -1) ||
            (pipe(pipe_eval_result_fd) == -1)) {
            perror("pipe");
            exit(EXIT_FAILURE);
        }

        for (uint32_t i = 0; i < config.READ_THREADS; i++) {
            pid_t pid = fork();
            if (pid == -1) {
                perror("fork");
                exit(EXIT_FAILURE);
            }
            else if (pid == 0) {
                close(pipe_eval_task_fd[1]);
                close(pipe_eval_result_fd[0]);
                close(pipe_eval_system_fd[1]);

                run_worker(indices_eval, pipe_eval_task_fd, pipe_eval_result_fd, pipe_eval_system_fd, false);

                close(pipe_eval_task_fd[0]);
                close(pipe_eval_result_fd[1]);
                close(pipe_eval_system_fd[0]);
                exit(EXIT_SUCCESS);
            }
        }

        close(pipe_eval_task_fd[0]);
        close(pipe_eval_result_fd[1]);
        close(pipe_eval_system_fd[0]);
    }

    close(pipe_train_task_fd[0]);
    close(pipe_train_result_fd[1]);
    close(pipe_train_system_fd[0]);
}

// Returns the file descriptor opened for reading and used to communicate with training workers
int
get_train_read_fd()
{
    return pipe_train_result_fd[0];
}

// Returns the file descriptor opened for reading and used to communicate with evaluation workers
int
get_eval_read_fd()
{
    return pipe_eval_result_fd[0];
}

// Returns the file descriptor opened for writing and used to communicate with training workers
int
get_train_write_fd()
{
    return pipe_train_task_fd[1];
}

// Returns the file descriptor opened for writing and used to communicate with evaluation workers
int
get_eval_write_fd()
{
    return pipe_eval_task_fd[1];
}

// Returns the file descriptor opened for writing and used to manage the training workers
int
get_train_system_fd()
{
    return pipe_train_system_fd[1];
}

// Returns the file descriptor opened for writing and used to manage the evaluation workers
int
get_eval_system_fd()
{
    return pipe_eval_system_fd[1];
}

// Release all resources used by processes and the processes themselves
void
fin_workers()
{
    close(pipe_train_task_fd[1]);
    close(pipe_train_result_fd[0]);
    close(pipe_train_system_fd[1]);

    if (config.DO_TRAIN) {
        close(pipe_eval_task_fd[1]);
        close(pipe_eval_result_fd[0]);
        close(pipe_eval_system_fd[1]);
    }

    for (uint32_t i = 0; i < config.READ_THREADS; i++) {
        wait(NULL);
    }

    if (config.DO_EVALUATION) {
        for (uint32_t i = 0; i < config.READ_THREADS; i++) {
            wait(NULL);
        }
    }
}

// Command all workers to shuffle data files
void
force_workers_to_shuffle(int read_fd, int write_fd, int system_fd)
{
    int32_t shuffle_code = -1;
    for (uint32_t i = 0; i < config.READ_THREADS; i++) {
        write(write_fd, &shuffle_code, sizeof(shuffle_code));
    }

    for (uint32_t i = 0; i < config.READ_THREADS; i++) {
        read(read_fd, &shuffle_code, sizeof(shuffle_code));
    }

    for (uint32_t i = 0; i < config.READ_THREADS; i++) {
        write(system_fd, &shuffle_code, sizeof(shuffle_code));
    }
}

// Starting a worker waiting for commands to read data batches
void
run_worker(uint32_t *indices, int pipe_task_fd[2], int pipe_result_fd[2], int pipe_system_fd[2],
           bool is_train_worker)
{
    int32_t batch = 0, current_epoch = 0;
    while (read(pipe_task_fd[0], &batch, sizeof(batch)) > 0) {
        // A new epoch has begun
        if (batch == -1) {
            if (config.SEED_CHANGE_EPOCH) {
                srand(config.RANDOM_SEED * (is_train_worker ? 1 : 2) + current_epoch);
            }
            if (config.DO_SHUFFLE) {
                shuffle(indices, config.NUM_SAMPLES_PER_FILE *
                                     (is_train_worker ? config.NUM_FILES_TRAIN : config.NUM_FILES_EVAL));
            }
            current_epoch++;
            write(pipe_result_fd[1], &batch, sizeof(batch));
            read(pipe_system_fd[0], &batch, sizeof(batch));
            continue;
        }

        uint32_t read_from = batch * (is_train_worker ? config.BATCH_SIZE : config.BATCH_SIZE_EVAL);
        uint32_t read_to   = (batch + 1) * (is_train_worker ? config.BATCH_SIZE : config.BATCH_SIZE_EVAL);
        uint64_t process_metadata_time = 0, process_read_time = 0;

        for (uint32_t i = read_from; i < read_to; i++) {
            uint32_t file_num   = indices[i] / config.NUM_SAMPLES_PER_FILE + 1;
            uint32_t sample_num = indices[i] % config.NUM_SAMPLES_PER_FILE;
            char     file_path[256];
            snprintf(file_path, sizeof(file_path), "%s/%s/%s_%u_of_%u.h5", config.DATA_FOLDER,
                     is_train_worker ? config.TRAIN_DATA_FOLDER : config.VALID_DATA_FOLDER,
                     config.FILE_PREFIX, file_num,
                     is_train_worker ? config.NUM_FILES_TRAIN : config.NUM_FILES_EVAL);

            uint64_t metadata_time = 0, read_time = 0;
            read_sample(file_path, sample_num, &metadata_time, &read_time);

            process_metadata_time += metadata_time;
            process_read_time += read_time;
        }

        execution_time_t data = {
            .metadata_time = process_metadata_time,
            .read_time     = process_read_time,
        };

        write(pipe_result_fd[1], &data, sizeof(data));
    }
}
