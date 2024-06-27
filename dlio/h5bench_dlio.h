#ifndef SANDBOX_H5BENCH_ML_READ_H
#define SANDBOX_H5BENCH_ML_READ_H

#include <hdf5.h>

extern int NUM_RANKS, MY_RANK;

void generate_labels_dataset(hid_t file_id, hid_t filespace, hid_t memspace);

void generate_records_dataset(hid_t file_id, hid_t filespace, hid_t memspace, hid_t extra_memspace);

void generate_file(const char *file_name, hid_t labels_filespace, hid_t labels_memspace,
                   hid_t records_filespace, hid_t records_memspace, hid_t extra_records_memspace);

void generate_data();

void read_sample(const char *file_path, uint32_t sample, uint64_t *metadata_time_out,
                 uint64_t *read_time_out);

uint64_t compute(float time, float time_stdev);

void eval_without_workers(uint32_t epoch, uint32_t *indices, uint64_t *local_metadata_time_out,
                          uint64_t *local_read_time_out);

void eval_using_workers(uint32_t epoch, uint64_t *local_metadata_time_out, uint64_t *local_read_time_out);

void eval(uint32_t epoch, uint32_t *indices, bool enable_multiprocessing);

void train_without_workers(uint32_t epoch, uint32_t *indices, uint64_t *local_metadata_time_out,
                           uint64_t *local_read_time_out);

void train_using_workers(uint32_t epoch, uint64_t *local_metadata_time_out, uint64_t *local_read_time_out);

void train(uint32_t epoch, uint32_t *indices, bool enable_multiprocessing);

void run();

void init_global_variables();

#endif // SANDBOX_H5BENCH_ML_READ_H
