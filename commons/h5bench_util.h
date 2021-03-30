/*
 * h5bench_util.h
 *
 *  Created on: Aug 24, 2020
 *      Author: tonglin
 */

#ifndef COMMONS_H5BENCH_UTIL_H_
#define COMMONS_H5BENCH_UTIL_H_

#define DEBUG_PRINT printf("%s:%d\n", __func__, __LINE__); fflush(stdout);
//Maximal line length of the config file
#define CFG_LINE_LEN_MAX 510
#define CFG_DELIMS "=\n \t"
#define G_VAL 1024 * 1024 * 1024
#define M_VAL 1024 * 1024
#define K_VAL 1024
#define PARTICLE_SIZE 7 * sizeof(float) +  sizeof(int)
typedef enum async_mode {
    ASYNC_NON,
    ASYNC_EXPLICIT,
    ASYNC_IMPLICIT
} async_mode;

typedef enum write_pattern {
    CONTIG_CONTIG_1D,
    CONTIG_INTERLEAVED_1D,
    INTERLEAVED_CONTIG_1D,
    INTERLEAVED_INTERLEAVED_1D,
    CONTIG_CONTIG_STRIDED_1D,
    CONTIG_CONTIG_2D,
    CONTIG_INTERLEAVED_2D,
    INTERLEAVED_CONTIG_2D,
    INTERLEAVED_INTERLEAVED_2D,
    CONTIG_CONTIG_3D
} write_pattern;

typedef enum read_pattern{
    CONTIG_1D,
    CONTIG_1D_PART,
    STRIDED_1D,
    CONTIG_2D,
    CONTIG_3D
}read_pattern;

typedef struct bench_params{
    int isWrite;
    int useCompress;
    int useCSV;
    async_mode asyncMode;
    union access_pattern{
        read_pattern pattern_read;
        write_pattern pattern_write;
    }access_pattern;

    //write_pattern bench_pattern;
    char* data_file_path;
    char* pattern_name;
    int meta_coll;//for write only, metadata collective
    int data_coll;//data collective
    int cnt_time_step;
    int cnt_time_step_delay;
    int cnt_particle_M;//total number per rank
    int cnt_try_particles_M;// to read
    int memory_bound_M;//memory usage bound
    int sleep_time;
    int _dim_cnt;
    unsigned long stride;
    unsigned long block_size;
    unsigned long block_cnt;
    unsigned long dim_1;
    unsigned long dim_2;
    unsigned long dim_3;
    unsigned long chunk_dim_1;
    unsigned long chunk_dim_2;
    unsigned long chunk_dim_3;
    char* csv_path;
    char* meta_list_path;
    FILE* csv_fs;
    int file_per_proc;
} bench_params;

typedef struct data_md{
    long particle_cnt;
    long dim_1, dim_2, dim_3;
    float *x, *y, *z;
    float *px, *py, *pz;
    int *id_1;
    float *id_2;
}data_contig_md;

typedef struct csv_hanle{
    int use_csv;
    FILE* fs;
}csv_handle;

typedef enum ts_status{
    TS_INIT,
    TS_DELAY,
    TS_READY,
    TS_DONE
}ts_status;
typedef struct time_step time_step;
struct time_step{
    hid_t es_meta_create;
    hid_t es_meta_close;
    hid_t es_data;
    hid_t grp_id;
    hid_t dset_ids[8];
    ts_status status;
    unsigned long mem_size;
};

typedef struct mem_monitor{
    unsigned int time_step_cnt;
    unsigned int delay_ts_cnt;
    unsigned int ts_open;//check opened ts and close them when reaches a limit.
    unsigned long mem_used;
    unsigned long mem_threshold;
    async_mode mode;
    time_step* time_steps;
}mem_monitor;

void timestep_es_id_close(time_step* ts, async_mode mode);

mem_monitor* mem_monitor_new(int time_step_cnt, async_mode mode,
        unsigned long time_step_size, unsigned long mem_threshold);
int mem_monitor_free(mem_monitor* mon);
int mem_monitor_check_run(mem_monitor* mon, unsigned long *metadata_time_total, unsigned long *data_time_total);
// Uniform random number
float uniform_random_number();

hid_t es_id_set(async_mode mode);
void es_id_close(hid_t es_id, async_mode mode);

data_contig_md* prepare_contig_memory(long particle_cnt, long dim_1, long dim_2, long dim_3);
data_contig_md* prepare_contig_memory_multi_dim(long dim_1, long dim_2, long dim_3);

void free_contig_memory(data_contig_md* data);

unsigned long get_time_usec();

int read_config(const char* file_path, bench_params* params_out, int do_write);

void print_params(const bench_params* p);
void bench_params_free(bench_params* p);
void test_read_config(const char* file_path, int do_write);

int file_create_try(const char* path);
int file_exist(const char* path);
FILE* csv_init(const char* csv_path, const char* metadata_list_file);
int csv_output_line(FILE* fs, char* name, char* val_str);
int record_env_metadata(FILE* fs, const char* metadata_list_file);//set metadata_list_file to NULL if you don't need metadata.

int argv_print(int argc, char* argv[]);

char* get_file_name_from_path( char* path );
char* get_dir_from_path( char* path );

#endif /* COMMONS_H5BENCH_UTIL_H_ */
