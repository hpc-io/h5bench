/*
 * h5bench_util.c
 *
 *  Created on: Aug 24, 2020
 *      Author: tonglin
 */

#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <assert.h>
#include <sys/time.h>
#include <hdf5.h>

#include "h5bench_util.h"

unsigned long get_time_usec() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return 1000000 * tv.tv_sec + tv.tv_usec;
}

int metric_msg_print(unsigned long number, char *msg, char *unit) {
    printf("%s %lu %s\n", msg, number, unit);
    return 0;
}

hid_t es_id_set(async_mode mode) {
    hid_t es_id = 0;
    switch (mode) {
        case ASYNC_NON:
            es_id = H5ES_NONE;
            break;
        case ASYNC_EXPLICIT:
            es_id = H5EScreate();
            break;
        case ASYNC_IMPLICIT:
            break;
        default:
            break;
    }
    return es_id;
}

void es_id_close(hid_t es_id, async_mode mode) {
    switch (mode) {
        case ASYNC_NON:
            break;
        case ASYNC_EXPLICIT:
            H5ESclose(es_id);
            break;
        case ASYNC_IMPLICIT:
            break;
        default:
            break;
    }
}

float uniform_random_number() {
    return (((float) rand()) / ((float) (RAND_MAX)));
}

data_contig_md* prepare_contig_memory(long particle_cnt, long dim_1, long dim_2, long dim_3) {
    data_contig_md *buf_struct = (data_contig_md*) malloc(sizeof(data_contig_md));
    buf_struct->particle_cnt = particle_cnt;
    buf_struct->dim_1 = dim_1;
    buf_struct->dim_2 = dim_2;
    buf_struct->dim_3 = dim_3;
    buf_struct->x = (float*) malloc(particle_cnt * sizeof(float));
    buf_struct->y = (float*) malloc(particle_cnt * sizeof(float));
    buf_struct->z = (float*) malloc(particle_cnt * sizeof(float));
    buf_struct->px = (float*) malloc(particle_cnt * sizeof(float));
    buf_struct->py = (float*) malloc(particle_cnt * sizeof(float));
    buf_struct->pz = (float*) malloc(particle_cnt * sizeof(float));
    buf_struct->id_1 = (int*) malloc(particle_cnt * sizeof(int));
    buf_struct->id_2 = (float*) malloc(particle_cnt * sizeof(float));
    return buf_struct;
}

data_contig_md* prepare_contig_memory_multi_dim(long dim_1, long dim_2, long dim_3) {
    data_contig_md *buf_struct = (data_contig_md*) malloc(sizeof(data_contig_md));
    buf_struct->dim_1 = dim_1;
    buf_struct->dim_2 = dim_2;
    buf_struct->dim_3 = dim_3;
    long num_particles = dim_1 * dim_2 * dim_3;

    buf_struct->particle_cnt = num_particles;
    buf_struct->x = (float*) malloc(num_particles * sizeof(float));
    buf_struct->y = (float*) malloc(num_particles * sizeof(float));
    buf_struct->z = (float*) malloc(num_particles * sizeof(float));
    buf_struct->px = (float*) malloc(num_particles * sizeof(float));
    buf_struct->py = (float*) malloc(num_particles * sizeof(float));
    buf_struct->pz = (float*) malloc(num_particles * sizeof(float));
    buf_struct->id_1 = (int*) malloc(num_particles * sizeof(int));
    buf_struct->id_2 = (float*) malloc(num_particles * sizeof(float));
    return buf_struct;
}

void free_contig_memory(data_contig_md *data) {
    free(data->x);
    free(data->y);
    free(data->z);
    free(data->px);
    free(data->py);
    free(data->pz);
    free(data->id_1);
    free(data->id_2);
    free(data);
}

int _set_params(char *key, char *val, bench_params *params_in_out, int do_write) {
    if (!params_in_out)
        return 0;

    if (strcmp(key, "WRITE_PATTERN") == 0) {
        if (do_write < 1) {
            printf("This is NOT a write benchmark, but try to set a WRITE_PATTERN\n");
            return -1;
        }

        if (strcmp(val, "CC") == 0) {
            (*params_in_out).access_pattern.pattern_write = CONTIG_CONTIG_1D;
            (*params_in_out).pattern_name = strdup("CONTIG_CONTIG_1D");
            (*params_in_out)._dim_cnt = 1;
        } else if (strcmp(val, "CC2D") == 0) {
            (*params_in_out).access_pattern.pattern_write = CONTIG_CONTIG_2D;
            (*params_in_out).pattern_name = strdup("CONTIG_CONTIG_2D");
            (*params_in_out)._dim_cnt = 2;
        } else if (strcmp(val, "CI") == 0) {
            (*params_in_out).access_pattern.pattern_write = CONTIG_INTERLEAVED_1D;
            (*params_in_out).pattern_name = strdup("CONTIG_INTERLEAVED_1D");
            (*params_in_out)._dim_cnt = 1;
        } else if (strcmp(val, "CI2D") == 0) {
            (*params_in_out).access_pattern.pattern_write = CONTIG_INTERLEAVED_2D;
            (*params_in_out).pattern_name = strdup("CONTIG_INTERLEAVED_2D");
            (*params_in_out)._dim_cnt = 2;
        } else if (strcmp(val, "II") == 0) {
            (*params_in_out).access_pattern.pattern_write = INTERLEAVED_INTERLEAVED_1D;
            (*params_in_out).pattern_name = strdup("INTERLEAVED_INTERLEAVED_1D");
            (*params_in_out)._dim_cnt = 1;
        } else if (strcmp(val, "II2D") == 0) {
            (*params_in_out).access_pattern.pattern_write = INTERLEAVED_INTERLEAVED_2D;
            (*params_in_out).pattern_name = strdup("INTERLEAVED_INTERLEAVED_2D");
            (*params_in_out)._dim_cnt = 2;
        } else if (strcmp(val, "IC") == 0) {
            (*params_in_out).access_pattern.pattern_write = INTERLEAVED_CONTIG_1D;
            (*params_in_out).pattern_name = strdup("INTERLEAVED_CONTIG_1D");
            (*params_in_out)._dim_cnt = 1;
        } else if (strcmp(val, "IC2D") == 0) {
            (*params_in_out).access_pattern.pattern_write = INTERLEAVED_CONTIG_2D;
            (*params_in_out).pattern_name = strdup("INTERLEAVED_CONTIG_2D");
            (*params_in_out)._dim_cnt = 2;
        } else if (strcmp(val, "CC3D") == 0) {
            (*params_in_out).access_pattern.pattern_write = CONTIG_CONTIG_3D;
            (*params_in_out).pattern_name = strdup("CONTIG_CONTIG_3D");
            (*params_in_out)._dim_cnt = 3;
        } else {
            printf("Unknown WRITE_PATTERN: %s\n", val);
            return -1;
        }
    } else if (strcmp(key, "READ_PATTERN") == 0) {
        if (do_write > 0) {
            printf("This is a write benchmark, but try to set a READ_PATTERN\n");
            return -1;
        }
        if (strcmp(val, "SEQ") == 0) {
            (*params_in_out).access_pattern.pattern_read = CONTIG_1D;
            (*params_in_out).pattern_name = strdup("CONTIG_1D");
            (*params_in_out)._dim_cnt = 1;
        } else if (strcmp(val, "PART") == 0) {
            (*params_in_out).access_pattern.pattern_read = CONTIG_1D_PART;
            (*params_in_out).pattern_name = strdup("CONTIG_1D_PART");
            (*params_in_out)._dim_cnt = 1;
        } else if (strcmp(val, "STRIDED") == 0) {
            (*params_in_out).access_pattern.pattern_read = STRIDED_1D;
            (*params_in_out).pattern_name = strdup("CONTIG_1D");
            (*params_in_out)._dim_cnt = 1;
        } else if (strcmp(val, "2D") == 0) {
            (*params_in_out).access_pattern.pattern_read = CONTIG_2D;
            (*params_in_out).pattern_name = strdup("CONTIG_2D");
            (*params_in_out)._dim_cnt = 2;
        } else if (strcmp(val, "3D") == 0) {
            (*params_in_out).access_pattern.pattern_read = CONTIG_3D;
            (*params_in_out).pattern_name = strdup("CONTIG_3D");
            (*params_in_out)._dim_cnt = 3;
        } else {
            printf("Unknown READ_PATTERN: %s\n", val);
            return -1;
        }

    } else if (strcmp(key, "TO_READ_CNT_M") == 0) {
        if ((*params_in_out).isWrite) {
            printf("TO_READ_CNT_M parameter is only used with READ_PATTERNs, please check config file.\n");
            return -1;
        }
        (*params_in_out).cnt_try_particles_M = atoi(val);
    } else if (strcmp(key, "META_COLL") == 0) {
        if (strcmp(val, "YES") == 0 || strcmp(val, "Y") == 0)
            (*params_in_out).meta_coll = 1;
        else
            (*params_in_out).meta_coll = 0;

    } else if (strcmp(key, "DATA_COLL") == 0) {
        if (strcmp(val, "YES") == 0 || strcmp(val, "Y") == 0)
            (*params_in_out).data_coll = 1;
        else
            (*params_in_out).data_coll = 0;

    } else if (strcmp(key, "COMPRESS") == 0) {
        if (strcmp(val, "YES") == 0 || strcmp(val, "Y") == 0)
            (*params_in_out).useCompress = 1;
        else
            (*params_in_out).useCompress = 0;

    } else if (strcmp(key, "TIME_STEPS_CNT") == 0) {
        int ts_cnt = atoi(val);
        if (ts_cnt >= 1)
            (*params_in_out).cnt_time_step = ts_cnt;
        else {
            printf("TIME_STEPS_CNT must be at least 1.\n");
            return -1;
        }
    } else if (strcmp(key, "PARTICLE_CNT_M") == 0) {
        int ts_cnt = atoi(val);
        if (ts_cnt >= 1)
            (*params_in_out).cnt_particle_M = ts_cnt;
        else {
            printf("PARTICLE_CNT_M must be at least 1.\n");
            return -1;
        }
    } else if (strcmp(key, "SLEEP_TIME") == 0) {
        int sleep_time = atoi(val);
        if (sleep_time >= 0)
            (*params_in_out).sleep_time = sleep_time;
        else {
            printf("SLEEP_TIME must be at least 0.\n");
            return -1;
        }
    } else if (strcmp(key, "DIM_1") == 0) {
        int dim = atoi(val);
        if (dim > 0)
            (*params_in_out).dim_1 = dim;
        else {
            printf("DIM_1 must be at least 1\n");
        }
    } else if (strcmp(key, "DIM_2") == 0) {
        if ((*params_in_out)._dim_cnt == 1)
            return 1;
        int dim = atoi(val);
        if (dim >= 1)
            (*params_in_out).dim_2 = dim;
        else {
            printf("DIM_2 must be at least 1\n");
            return -1;
        }
    } else if (strcmp(key, "DIM_3") == 0) {
        if ((*params_in_out)._dim_cnt == 1 || (*params_in_out)._dim_cnt == 2)
            return 1;
        int dim = atoi(val);
        if (dim >= 1)
            (*params_in_out).dim_3 = dim;
        else {
            printf("DIM_3 must be at least 1\n");
            return -1;
        }
    } else if (strcmp(key, "CHUNK_DIM_1") == 0) {
        int dim = atoi(val);
        if (dim > 0)
            (*params_in_out).chunk_dim_1 = dim;
        else {
            printf("CHUNK_DIM_1 must be at least 1\n");
        }
    } else if (strcmp(key, "CHUNK_DIM_2") == 0) {
        if ((*params_in_out)._dim_cnt == 1)
            return 1;
        int dim = atoi(val);
        if (dim >= 1)
            (*params_in_out).chunk_dim_2 = dim;
        else {
            printf("CHUNK_DIM_2 must be at least 1.\n");
            return -1;
        }
    } else if (strcmp(key, "CHUNK_DIM_3") == 0) {
        if ((*params_in_out)._dim_cnt == 1 || (*params_in_out)._dim_cnt == 2)
            return 1;
        int dim = atoi(val);
        if (dim >= 1)
            (*params_in_out).chunk_dim_3 = dim;
        else {
            printf("CHUNK_DIM_3 must be at least 1.\n");
            return -1;
        }
    } else if (strcmp(key, "STRIDE_SIZE") == 0) {
        (*params_in_out).stride = atoi(val);

    } else if (strcmp(key, "BLOCK_SIZE") == 0) {
        (*params_in_out).block_size = atoi(val);
    } else if (strcmp(key, "CSV_FILE") == 0) {
        (*params_in_out).useCSV = 1;
        (*params_in_out).csv_path = strdup(val);
    } else if (strcmp(key, "META_LIST_FILE") == 0) {
        (*params_in_out).meta_list_path = strdup(val);
    } else if (strcmp(key, "ASYNC_MODE") == 0) {
        if (strcmp(val, "ASYNC_NON") == 0) {
            (*params_in_out).asyncMode = ASYNC_NON;
        } else if (strcmp(val, "ASYNC_EXP") == 0 || strcmp(val, "ASYNC_EXPLICIT") == 0) {
            (*params_in_out).asyncMode = ASYNC_EXPLICIT;
        } else if (strcmp(val, "ASYNC_IMP") == 0 || strcmp(val, "ASYNC_IMPLICIT") == 0) {
            (*params_in_out).asyncMode = ASYNC_IMPLICIT;
        }
    } else if (strcmp(key, "FILE_PER_PROC") == 0) {
        if (strcmp(val, "YES") == 0 || strcmp(val, "Y") == 0)
            (*params_in_out).file_per_proc = 1;
        else
            (*params_in_out).file_per_proc = 0;
    } else {
        printf("Unknown Parameter: %s\n", key);
        return -1;
    }
    if ((*params_in_out).useCSV)
        (*params_in_out).csv_fs = csv_init(params_in_out->csv_path, params_in_out->meta_list_path);

    return 1;
}

//only for vpic
int read_config(const char *file_path, bench_params *params_out, int do_write) {
    char cfg_line[CFG_LINE_LEN_MAX] = "";

    if (!params_out)
        params_out = (bench_params*) calloc(1, sizeof(bench_params));
    else
        memset(params_out, 0, sizeof(bench_params));
    (*params_out).data_file_path = strdup(file_path);
    (*params_out).pattern_name = NULL;
    (*params_out).meta_coll = 0;
    (*params_out).data_coll = 0;

    (*params_out).cnt_time_step = 0;
    (*params_out).cnt_particle_M = 0; //total number per rank
    (*params_out).cnt_try_particles_M = 0; // to read
    (*params_out).sleep_time = 0;
    (*params_out)._dim_cnt = 1;

    (*params_out).stride = 0;
    (*params_out).block_size = 0;
    (*params_out).dim_1 = 1;
    (*params_out).dim_2 = 1;
    (*params_out).dim_3 = 1;
    (*params_out).chunk_dim_1 = 1;
    (*params_out).chunk_dim_2 = 1;
    (*params_out).chunk_dim_3 = 1;
    (*params_out).csv_path = NULL;
    (*params_out).meta_list_path = NULL;

    (*params_out).csv_path = NULL;
    (*params_out).csv_fs = NULL;
    (*params_out).meta_list_path = NULL;
    (*params_out).file_per_proc = 0;

    FILE *csv_fs;

    FILE *file = fopen(file_path, "r");
    char *key, val;
    int parsed = 1;

    //default values
    (*params_out).useCSV = 0;
    if (do_write)
        (*params_out).isWrite = 1;
    else
        (*params_out).isWrite = 0;


    while (fgets(cfg_line, CFG_LINE_LEN_MAX, file) && (parsed == 1)) {
        if (cfg_line[0] == '#') { //skip comment lines
            continue;
        }
        char *tokens[2];
        char *tok = strtok(cfg_line, CFG_DELIMS);
        if (tok) {
            tokens[0] = tok;
            tok = strtok(NULL, CFG_DELIMS);
            if (tok)
                tokens[1] = tok;
            else
                return -1;
        } else
            return -1;
        //printf("key = [%s], val = [%s]\n", tokens[0], tokens[1]);
        parsed = _set_params(tokens[0], tokens[1], params_out, do_write);
    }

    if (params_out->isWrite) {

    } else { //read
        if (params_out->access_pattern.pattern_read == CONTIG_1D) { //read whole file
            params_out->cnt_try_particles_M = params_out->cnt_particle_M;
        }
    }
    if (parsed < 0)
        return -1;
    else
        return 0;
}

void print_params(const bench_params *p) {
    printf("=======================================\n");
    printf("Benchmark parameter list: \nFile: %s\n", p->data_file_path);
    printf("Benchmark pattern = %s\n", p->pattern_name);
    printf("Per rank particles number(in M) = %d M\n", p->cnt_particle_M);
    //printf("Per rank actual read number (in M) = %d M\n", p->cnt_actual_particles_M);
    printf("Time step number = %d\n", p->cnt_time_step);
    printf("Sleep time = %d\n", p->sleep_time);

    if (p->meta_coll == 1)
        printf("Metadata collective ops: YES.\n");
    else
        printf("Metadata collective ops: NO.\n");
    if (p->data_coll == 1)
        printf("Data collective ops: YES.\n");
    else
        printf("Data collective ops: NO.\n");

    printf("Dimension cnt = %d\n", p->_dim_cnt);
    printf("    Dim_1 = %lu\n", p->dim_1);
    if (p->_dim_cnt >= 2) {
        printf("    Dim_2 = %lu\n", p->dim_2);
    } else if (p->_dim_cnt >= 3) {
        printf("    Dim_3 = %lu\n", p->dim_3);
    }
    if (p->useCompress) {
        printf("useCompress = %d\n", p->useCompress);
        printf("chunk_dim1 = %lu\n", p->chunk_dim_1);
        if (p->_dim_cnt >= 2) {
            printf("chunk_dim2 = %lu\n", p->chunk_dim_2);
        } else if (p->_dim_cnt >= 3) {
            printf("chunk_dim3 = %lu\n", p->chunk_dim_3);
        }
    }

    printf("=======================================\n");
}

void bench_params_free(bench_params *p) {
    if (!p)
        return;
    free(p->data_file_path);
    free(p->pattern_name);
}

void test_read_config(const char *file_path, int do_write) {
    bench_params param;

    int ret = read_config(file_path, &param, do_write);

    if (ret != 0) {
        printf("read_config() failed, ret = %d\n", ret);
    } else {
        printf("param->isWrite = %d\n", param.isWrite);
        if (param.isWrite) {
            printf("param->access_pattern = pattern_write, val = %d\n", param.access_pattern.pattern_write);
        } else {
            printf("param->access_pattern = pattern_read, val = %d\n", param.access_pattern.pattern_read);
        }

        printf("param->pattern_name = %s\n", param.pattern_name);
        printf("param->cnt_time_step = %d\n", param.cnt_time_step);
        printf("param->cnt_particle_M = %d\n", param.cnt_particle_M);
        printf("param->sleep_time = %d\n", param.sleep_time);
        printf("param->DIM_1 = %lu\n", param.dim_1);
        printf("param->DIM_2 = %lu\n", param.dim_2);
        printf("param->DIM_3 = %lu\n", param.dim_3);
    }
}

int file_create_try(const char *path) {
    FILE *fs = fopen(path, "w+");
    if (!fs) {
        printf("Failed to create file: %s, Please check permission.\n", path);
        return -1;
    }
    fclose(fs);
    return 0;
}

int file_exist(const char *path) {
    FILE *f = fopen(path, "r");
    if (!f) {
        printf("Failed to open file: %s, Please check if the file exists.\n", path);
        return -1;
    }
    fclose(f);
    return 0;
}

/*  TODO:
 *      - read lines from metadata_list_file, each presents an environment variable name.
 *      - get val from getrnv(), write to fs.
 * */

int record_env_metadata(FILE *fs, const char *metadata_list_file) {
    //read list file line, use each line as a key to search env
    if (!fs)
        return -1;
    FILE *lfs = fopen(metadata_list_file, "r");
    if (!lfs) {
        printf("Can not open metadata list file: %s\n", metadata_list_file);
        return -1;
    }

    fprintf(fs, "======================= Metadata =====================\n");

    char line[10 * CFG_LINE_LEN_MAX]; //some env val could be very large, such as PATH
    while (fgets(line, CFG_LINE_LEN_MAX, lfs)) {
        if (line[0] == '#') //skip comment lines
            continue;
        if (line[0] == '\n')
            continue;

        if (line[strlen(line) - 1] == '\n') {
            line[strlen(line) - 1] = 0;
        }

        char *val = getenv(line);
        //printf("%s = %s\n", line, val);
        fprintf(fs, "%s = %s\n", line, val);

        if (!val) { //null
            printf("    %s not set.\n", line);
            continue;
        }
    }

    fprintf(fs, "======================= Metadata end ====================\n");
    fclose(lfs);
    return 0;
}

FILE* csv_init(const char *csv_path, const char *metadata_list_file) { //, const char* metadata_list_file: should be optional.
    FILE *fs = fopen(csv_path, "w+");

    if (!fs) {
        printf("Failed to create file: %s, Please check permission.\n", csv_path);
        return NULL;
    }

    if (metadata_list_file) {
        if (record_env_metadata(fs, metadata_list_file) < 0)
            return NULL;
    }

    return fs;
}

int csv_output_line(FILE *fs, char *name, char *val_str) {
    fprintf(fs, "%s,", name);
    fprintf(fs, " %s\n", val_str);
    return 0;
}

int argv_print(int argc, char *argv[]) {
    if (argc < 1)
        return -1;
    printf("%d arguments provided.\n", argc);
    for (int i = 0; i < argc; i++) {
        printf("idx = %d, argv = %s\n", i, argv[i]);
    }
    return 0;
}

char* get_file_name_from_path(char *path) {
    if (path == NULL)
        return NULL;

    char *pFileName = path;
    for (char *pCur = path; *pCur != '\0'; pCur++) {
        if (*pCur == '/' || *pCur == '\\')
            pFileName = pCur + 1;
    }

    return pFileName;
}

char* substr(char *src, size_t start, size_t len) {
    if (start + len > strlen(src)) {
        fprintf(stderr, "%s() error: invalid substring index (start+len > length).\n", __func__);
        return NULL;
    }

    char *sub = calloc(1, len + 1);
    if (!sub) {
        fprintf(stderr, "%s() error: memory allocation failed.\n", __func__);
        return NULL;
    }

    memcpy(sub, src + start, len);
    // sub[len] = '\0';  // by using calloc, sub is filled with 0 (null)

    return sub;
}

char* get_dir_from_path(char *path) {
    if (path == NULL)
        return NULL;

    char *pDir = substr(path, 0, strlen(path) - strlen(get_file_name_from_path(path)));

    return pDir;
}
