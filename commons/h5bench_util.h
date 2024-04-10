/*
 * h5bench_util.h
 *
 *  Created on: Aug 24, 2020
 *      Author: tonglin
 */

#ifndef COMMONS_H5BENCH_UTIL_H_
#define COMMONS_H5BENCH_UTIL_H_

#define DEBUG_PRINT                                                                                          \
    printf("%s:%d\n", __func__, __LINE__);                                                                   \
    fflush(stdout);
// Maximal line length of the config file
#define CFG_LINE_LEN_MAX 510
#define CFG_DELIMS       "=\n"
#define T_VAL            ((unsigned long long)1024 * 1024 * 1024 * 1024)
#define G_VAL            ((unsigned long long)1024 * 1024 * 1024)
#define M_VAL            ((unsigned long long)1024 * 1024)
#define K_VAL            ((unsigned long long)1024)
#define PARTICLE_SIZE    (7 * sizeof(float) + sizeof(int))
typedef enum async_mode { MODE_SYNC, MODE_ASYNC } async_mode;

typedef enum num_unit {
    UNIT_INVALID,
    UNIT_K,
    UNIT_M,
    UNIT_G,
    UNIT_T,
} num_unit;

typedef enum time_unit {
    TIME_INVALID,
    TIME_MIN,
    TIME_SEC,
    TIME_MS,
    TIME_US,
} time_unit;

typedef struct human_readable {
    double value;
    char   unit;
} human_readable;

typedef struct duration {
    unsigned long time_num;
    time_unit     unit;
} duration;

typedef enum write_pattern {
    WRITE_PATTERN_INVALID,
    CONTIG_CONTIG_1D,
    CONTIG_COMPOUND_1D,
    COMPOUND_CONTIG_1D,
    COMPOUND_COMPOUND_1D,
    CONTIG_CONTIG_STRIDED_1D,
    CONTIG_CONTIG_2D,
    CONTIG_COMPOUND_2D,
    COMPOUND_CONTIG_2D,
    COMPOUND_COMPOUND_2D,
    CONTIG_CONTIG_3D,
} write_pattern;

typedef enum read_pattern {
    READ_PATTERN_INVALID,
    CONTIG_1D,
    CONTIG_1D_PART,
    STRIDED_1D,
    CONTIG_2D,
    CONTIG_3D,
    LDC_2D,
    RDC_2D,
    CS_2D,
    PRL_2D,
} read_pattern;

typedef enum pattern {
    PATTERN_INVALID,
    PATTERN_CONTIG,
    PATTERN_INTERLEAVED,
    PATTERN_STRIDED,
} pattern;

typedef enum io_operation {
    IO_INVALID,
    IO_READ,
    IO_WRITE,
    IO_OVERWRITE,
    IO_APPEND,
} io_operation;

typedef enum read_option {
    READ_OPTION_INVALID,
    READ_FULL,
    READ_PARTIAL,
    READ_STRIDED,
    LDC,
    RDC,
    PRL,
    CS
} read_option;

typedef struct bench_params {
    io_operation io_op;
    pattern      mem_pattern;
    pattern      file_pattern;
    read_option  read_option;
    int          useCompress;
    int          useCSV;
    int          useDataDist;
    async_mode   asyncMode;
    int          subfiling;
    union access_pattern {
        read_pattern  pattern_read;
        write_pattern pattern_write;
    } access_pattern;

    // write_pattern bench_pattern;
    char *             data_file_path;
    char *             pattern_name;
    int                meta_coll; // for write only, metadata collective
    int                data_coll; // data collective
    int                cnt_time_step;
    int                cnt_time_step_delay;
    unsigned long long num_particles;     // total number per rank
    unsigned long long try_num_particles; // to read
    unsigned long long io_mem_limit;      // memory usage bound
    //    int sleep_time;
    duration      compute_time;
    int           num_dims;
    unsigned long stride;
    unsigned long stride_2;
    unsigned long stride_3;
    unsigned long block_size;
    unsigned long block_size_2;
    unsigned long block_size_3;
    unsigned long block_cnt;
    unsigned long dim_1;
    unsigned long dim_2;
    unsigned long dim_3;
    unsigned long chunk_dim_1;
    unsigned long chunk_dim_2;
    unsigned long chunk_dim_3;
    char *        csv_path;
    char *        env_meta_path;
    FILE *        csv_fs;
    char *        data_dist_path;
    int           file_per_proc;
    int           align;
    unsigned long align_threshold;
    unsigned long align_len;
    unsigned long stdev_dim_1;
} bench_params;

typedef struct data_md {
    unsigned long long particle_cnt;
    unsigned long long dim_1, dim_2, dim_3;
    float *            x, *y, *z;
    float *            px, *py, *pz;
    int *              id_1;
    float *            id_2;
} data_contig_md;

typedef struct csv_hanle {
    int   use_csv;
    FILE *fs;
} csv_handle;

typedef enum ts_status { TS_INIT, TS_DELAY, TS_READY, TS_DONE } ts_status;

typedef struct time_step time_step;
struct time_step {
    hid_t              es_meta_create;
    hid_t              es_meta_close;
    hid_t              es_data;
    hid_t              grp_id;
    hid_t              dset_ids[8];
    ts_status          status;
    unsigned long long mem_size;
};

typedef struct mem_monitor {
    unsigned int       time_step_cnt;
    unsigned int       delay_ts_cnt; // check opened ts and close them when reaches a limit.
    unsigned long long mem_used;
    unsigned long long mem_threshold;
    async_mode         mode;
    time_step *        time_steps;
} mem_monitor;

unsigned long long read_time_val(duration time, time_unit unit);

void h5bench_sleep(duration sleep_time);
void async_sleep(hid_t es_id, duration sleep_time);

void timestep_es_id_close(time_step *ts, async_mode mode);

mem_monitor *mem_monitor_new(int time_step_cnt, async_mode mode, unsigned long long time_step_size,
                             unsigned long long mem_threshold);
int          mem_monitor_free(mem_monitor *mon);
int          ts_delayed_close(mem_monitor *mon, unsigned long *metadata_time_total, int dset_cnt);
int          mem_monitor_check_run(mem_monitor *mon, unsigned long *metadata_time_total,
                                   unsigned long *data_time_total);
int          mem_monitor_final_run(mem_monitor *mon, unsigned long *metadata_time_total,
                                   unsigned long *data_time_total);
// Uniform random number
float uniform_random_number();

hid_t es_id_set(async_mode mode);
void  es_id_close(hid_t es_id, async_mode mode);

data_contig_md *prepare_contig_memory(long particle_cnt, long dim_1, long dim_2, long dim_3);
data_contig_md *prepare_contig_memory_multi_dim(unsigned long long dim_1, unsigned long long dim_2,
                                                unsigned long long dim_3);

void          free_contig_memory(data_contig_md *data);
unsigned long get_time_usec();

int read_config(const char *file_path, bench_params *params_out, int do_write);

void print_params(const bench_params *p);
void bench_params_free(bench_params *p);
int  has_vol_connector();

extern int has_vol_async;

int   file_create_try(const char *path);
int   file_exist(const char *path);
FILE *csv_init(const char *csv_path, const char *metadata_list_file);
int   csv_output_line(FILE *fs, char *name, char *val_str);
int   record_env_metadata(
      FILE *fs, const char *metadata_list_file); // set metadata_list_file to NULL if you don't need metadata.

int argv_print(int argc, char *argv[]);

char *get_file_name_from_path(char *path);
char *get_dir_from_path(char *path);

human_readable format_human_readable(uint64_t bytes);

#endif /* COMMONS_H5BENCH_UTIL_H_ */
