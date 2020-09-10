/*
 * h5bench_util.h
 *
 *  Created on: Aug 24, 2020
 *      Author: tonglin
 */

#ifndef COMMONS_H5BENCH_UTIL_H_
#define COMMONS_H5BENCH_UTIL_H_

#define DEBUG_PRINT printf("%s:%d\n", __func__, __LINE__);
//Maximal line length of the config file
const int CFG_LINE_LEN_MAX = 510;

const char* CFG_DELIMS = "=\n \t";

typedef enum Access_pattern {
    CONTIG_CONTIG_1D,
    CONTIG_INTERLEAVED_1D,
    INTERLEAVED_CONTIG_1D,
    INTERLEAVED_INTERLEAVED_1D,
    CONTIG_CONTIG_2D,
    CONTIG_INTERLEAVED_2D,
    INTERLEAVED_CONTIG_2D,
    INTERLEAVED_INTERLEAVED_2D,
    CONTIG_CONTIG_3D
} access_pattern;

//For VPIC and BDCATS
typedef struct bench_params{
    access_pattern bench_pattern;
    char* data_file_path;
    char* pattern_name;
    int cnt_time_step;
    int cnt_particle_M;
    int sleep_time;
    int _dim_cnt;
    int dim_1;
    int dim_2;
    int dim_3;
} bench_params;

typedef struct data_md{
    long particle_cnt;
    long dim_1, dim_2, dim_3;
    float *x, *y, *z;
    float *px, *py, *pz;
    int *id_1, *id_2;
}data_contig_md;

// Uniform random number
float uniform_random_number();

data_contig_md* prepare_contig_memory(long particle_cnt, long dim_1, long dim_2, long dim_3);
void free_contig_memory(data_contig_md* data);

unsigned long get_time_usec();

int read_config(const char* file_path, bench_params* params_out);

void print_params(const bench_params* p);

void test_read_config(const char* file_path);


#endif /* COMMONS_H5BENCH_UTIL_H_ */
