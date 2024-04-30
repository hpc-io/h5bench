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

#ifdef USE_ASYNC_VOL
#include <H5VLconnector.h>
#include <h5_async_lib.h>
#else
#include "async_adaptor.h"
#endif

#include "h5bench_util.h"

int str_to_ull(char *str_in, unsigned long long *num_out);
int parse_time(char *str_in, duration *time);
int parse_unit(char *str_in, unsigned long long *num, char **unit_str);

int has_vol_async;

unsigned long
get_time_usec()
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return 1000000 * tv.tv_sec + tv.tv_usec;
}

int
metric_msg_print(unsigned long number, char *msg, char *unit)
{
    printf("%s %lu %s\n", msg, number, unit);
    return 0;
}

void
h5bench_sleep(duration sleep_time)
{
    if (sleep_time.unit == TIME_SEC) {
        sleep(sleep_time.time_num);
    }
    else if (sleep_time.unit == TIME_MIN) {

        sleep(60 * sleep_time.time_num);
    }
    else {
        if (sleep_time.unit == TIME_MS)
            usleep(1000 * sleep_time.time_num);
        else if (sleep_time.unit == TIME_US)
            usleep(sleep_time.time_num);
        else
            printf("Invalid sleep time unit.\n");
    }
}

void
async_sleep(hid_t es_id, duration sleep_time)
{
#ifdef USE_ASYNC_VOL
    size_t  num_in_progress;
    hbool_t op_failed;

    H5ESwait(es_id, 0, &num_in_progress, &op_failed);
#endif
    h5bench_sleep(sleep_time);
}

void
timestep_es_id_close(time_step *ts, async_mode mode)
{
    es_id_close(ts->es_meta_create, mode);
    es_id_close(ts->es_data, mode);
    es_id_close(ts->es_meta_close, mode);
}
unsigned long long
read_time_val(duration time, time_unit unit)
{
    unsigned long long t_us     = time.time_num;
    unsigned long long factor   = 1;
    unsigned long long t_output = 1;
    switch (time.unit) {
        case TIME_MIN:
            factor = 60 * 1000 * 1000 * 1000llu;
            break;
        case TIME_SEC:
            factor = 1000 * 1000;
            break;
        case TIME_MS:
            factor = 1000;
            break;
        case TIME_US:
            factor = 1;
            break;
        default:
            factor = 0;
            printf("Invalid time unit: %s\n", __func__);
            break;
    }
    t_us *= factor;

    switch (unit) {
        case TIME_MIN:
            t_output = t_us / (60 * 1000 * 1000 * 1000llu);
            break;
        case TIME_SEC:
            t_output = t_us / (1000 * 1000);
            break;
        case TIME_MS:
            t_output = t_us / 1000;
            break;
        case TIME_US:
            t_output = t_us;
            break;
        default:
            t_output = 0;
            printf("Invalid time unit: %s\n", __func__);
            break;
    }
    return t_output;
}

mem_monitor *
mem_monitor_new(int time_step_cnt, async_mode mode, unsigned long long time_step_size,
                unsigned long long mem_threshold)
{
    mem_monitor *monitor   = calloc(1, sizeof(mem_monitor));
    monitor->mode          = mode;
    monitor->time_step_cnt = time_step_cnt;

    monitor->mem_used      = 0;
    monitor->mem_threshold = mem_threshold;
    monitor->time_steps    = calloc(time_step_cnt, sizeof(time_step));

    for (int i = 0; i < time_step_cnt; i++) {
        monitor->time_steps[i].es_meta_create = es_id_set(mode);
        monitor->time_steps[i].es_data        = es_id_set(mode);
        monitor->time_steps[i].es_meta_close  = es_id_set(mode);

        monitor->time_steps[i].mem_size = time_step_size;
        monitor->time_steps[i].status   = TS_INIT;
        for (int j = 0; j < 8; j++)
            monitor->time_steps[i].dset_ids[j] = 0;
    }
    return monitor;
}

int
mem_monitor_free(mem_monitor *mon)
{
    if (mon) {
        if (mon->time_steps)
            free(mon->time_steps);
        free(mon);
    }
    return 0;
}

int
ts_delayed_close(mem_monitor *mon, unsigned long *metadata_time_total, int dset_cnt)
{
    *metadata_time_total = 0;
    if (!mon || !metadata_time_total)
        return -1;

    time_step *   ts_run;
    size_t        num_in_progress;
    H5ES_status_t op_failed;
    unsigned long t1, t2;
    unsigned long meta_time = 0;

    if (!has_vol_async)
        return 0;

    for (int i = 0; i < mon->time_step_cnt; i++) {
        ts_run = &(mon->time_steps[i]);
        if (mon->time_steps[i].status == TS_DELAY) {
            t1 = get_time_usec();
            for (int j = 0; j < dset_cnt; j++) {
                if (ts_run->dset_ids[j] != 0) {
                    H5Dclose_async(ts_run->dset_ids[j], ts_run->es_meta_close);
                }
            }
            H5Gclose_async(ts_run->grp_id, ts_run->es_meta_close);
            t2 = get_time_usec();
            meta_time += (t2 - t1);
            ts_run->status = TS_READY;
        }
    }
    *metadata_time_total = meta_time;
    return 0;
}

int
mem_monitor_check_run(mem_monitor *mon, unsigned long *metadata_time_total, unsigned long *data_time_total)
{
    *metadata_time_total = 0;
    *data_time_total     = 0;
    if (!mon || !metadata_time_total || !data_time_total)
        return -1;
    if (!has_vol_async)
        return 0;
    time_step *   ts_run;
    size_t        num_in_progress;
    hbool_t       op_failed;
    unsigned long t1, t2, t3, t4;
    unsigned long meta_time = 0, data_time = 0;
    int           dset_cnt = 8;
    if (mon->mem_used >= mon->mem_threshold) { // call ESWait and do ops
        for (int i = 0; i < mon->time_step_cnt; i++) {
            ts_run = &(mon->time_steps[i]);
            if (mon->time_steps[i].status == TS_READY) {
                t1 = get_time_usec();
                H5ESwait(ts_run->es_meta_create, H5ES_WAIT_FOREVER, &num_in_progress, &op_failed);
                t2 = get_time_usec();
                H5ESwait(ts_run->es_data, H5ES_WAIT_FOREVER, &num_in_progress, &op_failed);
                t3 = get_time_usec();
                H5ESwait(ts_run->es_meta_close, H5ES_WAIT_FOREVER, &num_in_progress, &op_failed);
                timestep_es_id_close(ts_run, mon->mode);
                t4 = get_time_usec();
                meta_time += ((t2 - t1) + (t4 - t3));
                data_time += (t3 - t2);
                ts_run->status = TS_DONE;
                mon->mem_used -= ts_run->mem_size;
                if (mon->mem_used >= mon->mem_threshold)
                    continue;
                else
                    break;
            }
            else if (mon->time_steps[i].status == TS_INIT)
                break;
        }
        *metadata_time_total = meta_time;
        *data_time_total     = data_time;
    }
    return 0;
}

int
mem_monitor_final_run(mem_monitor *mon, unsigned long *metadata_time_total, unsigned long *data_time_total)
{
    *metadata_time_total = 0;
    *data_time_total     = 0;
    size_t        num_in_progress;
    hbool_t       op_failed;
    time_step *   ts_run;
    unsigned long t1, t2, t3, t4, t5, t6;
    unsigned long meta_time = 0, data_time = 0;
    int           dset_cnt = 8;

    if (!has_vol_async) {
        for (int i = 0; i < mon->time_step_cnt; i++) {
            ts_run = &(mon->time_steps[i]);
            if (mon->time_steps[i].status == TS_DELAY) {
                for (int j = 0; j < dset_cnt; j++) {
                    if (ts_run->dset_ids[j] != 0) {
                        H5Dclose_async(ts_run->dset_ids[j], ts_run->es_meta_close);
                    }
                }
                H5Gclose_async(ts_run->grp_id, ts_run->es_meta_close);
            }
        }
        return 0;
    }

    if (!mon || !metadata_time_total || !data_time_total)
        return -1;
    t1 = get_time_usec();
    for (int i = 0; i < mon->time_step_cnt; i++) {
        ts_run = &(mon->time_steps[i]);
        if (mon->time_steps[i].status == TS_DELAY) {

            for (int j = 0; j < dset_cnt; j++) {
                if (ts_run->dset_ids[j] != 0) {
                    H5Dclose_async(ts_run->dset_ids[j], ts_run->es_meta_close);
                }
            }
            H5Gclose_async(ts_run->grp_id, ts_run->es_meta_close);

            ts_run->status = TS_READY;
        }
    }

    t2 = get_time_usec();
    meta_time += (t2 - t1);

    if (!has_vol_async)
        return 0;

    for (int i = 0; i < mon->time_step_cnt; i++) {
        ts_run = &(mon->time_steps[i]);
        if (mon->time_steps[i].status == TS_READY) {
            t1 = get_time_usec();
            H5ESwait(ts_run->es_meta_create, H5ES_WAIT_FOREVER, &num_in_progress, &op_failed);
            t2 = get_time_usec();

            H5ESwait(ts_run->es_data, H5ES_WAIT_FOREVER, &num_in_progress, &op_failed);
            t3 = get_time_usec();

            H5ESwait(ts_run->es_meta_close, H5ES_WAIT_FOREVER, &num_in_progress, &op_failed);
            t4 = get_time_usec();

            timestep_es_id_close(ts_run, mon->mode);
            t5 = get_time_usec();

            t6 = get_time_usec();

            meta_time += ((t2 - t1) + (t4 - t3));
            data_time += (t3 - t2);
            ts_run->status = TS_DONE;
        }
    }

    *metadata_time_total = meta_time;
    *data_time_total     = data_time;
    return 0;
}

hid_t
es_id_set(async_mode mode)
{
    hid_t es_id = 0;
    if (has_vol_async) {
        es_id = H5EScreate();
    }
    else {
        es_id = H5ES_NONE;
    }

    return es_id;
}

void
es_id_close(hid_t es_id, async_mode mode)
{
    if (has_vol_async) {
        H5ESclose(es_id);
    }
}

float
uniform_random_number()
{
    return (((float)rand()) / ((float)(RAND_MAX)));
}

data_contig_md *
prepare_contig_memory(long particle_cnt, long dim_1, long dim_2, long dim_3)
{
    data_contig_md *buf_struct = (data_contig_md *)malloc(sizeof(data_contig_md));
    buf_struct->particle_cnt   = particle_cnt;
    buf_struct->dim_1          = dim_1;
    buf_struct->dim_2          = dim_2;
    buf_struct->dim_3          = dim_3;
    buf_struct->x              = (float *)malloc(particle_cnt * sizeof(float));
    buf_struct->y              = (float *)malloc(particle_cnt * sizeof(float));
    buf_struct->z              = (float *)malloc(particle_cnt * sizeof(float));
    buf_struct->px             = (float *)malloc(particle_cnt * sizeof(float));
    buf_struct->py             = (float *)malloc(particle_cnt * sizeof(float));
    buf_struct->pz             = (float *)malloc(particle_cnt * sizeof(float));
    buf_struct->id_1           = (int *)malloc(particle_cnt * sizeof(int));
    buf_struct->id_2           = (float *)malloc(particle_cnt * sizeof(float));
    return buf_struct;
}

data_contig_md *
prepare_contig_memory_multi_dim(unsigned long long dim_1, unsigned long long dim_2, unsigned long long dim_3)
{
    data_contig_md *buf_struct       = (data_contig_md *)malloc(sizeof(data_contig_md));
    buf_struct->dim_1                = dim_1;
    buf_struct->dim_2                = dim_2;
    buf_struct->dim_3                = dim_3;
    unsigned long long num_particles = dim_1 * dim_2 * dim_3;

    buf_struct->particle_cnt = num_particles;
    buf_struct->x            = (float *)malloc(num_particles * sizeof(float));
    buf_struct->y            = (float *)malloc(num_particles * sizeof(float));
    buf_struct->z            = (float *)malloc(num_particles * sizeof(float));
    buf_struct->px           = (float *)malloc(num_particles * sizeof(float));
    buf_struct->py           = (float *)malloc(num_particles * sizeof(float));
    buf_struct->pz           = (float *)malloc(num_particles * sizeof(float));
    buf_struct->id_1         = (int *)malloc(num_particles * sizeof(int));
    buf_struct->id_2         = (float *)malloc(num_particles * sizeof(float));
    return buf_struct;
}

void
free_contig_memory(data_contig_md *data)
{
    if (data) {
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
}

int
parse_unit(char *str_in, unsigned long long *num, char **unit_str)
{
    char *str     = strdup(str_in);
    char *ptr     = NULL;
    ptr           = strtok(str, " ");
    char *num_str = strdup(ptr);
    if (!num_str) {
        printf("Number parsing failed: \"%s\" is not recognized.\n", str_in);
        return -1;
    }
    char *endptr;
    *num = strtoul(num_str, &endptr, 10);
    ptr  = strtok(NULL, " ");
    if (ptr)
        *unit_str = strdup(ptr);
    else
        *unit_str = NULL;
    return 0;
}

int
parse_time(char *str_in, duration *time)
{
    if (!time)
        time = calloc(1, sizeof(duration));
    unsigned long long num = 0;
    char *             unit_str;
    parse_unit(str_in, &num, &unit_str);

    if (!unit_str)
        time->unit = TIME_SEC;
    else if (unit_str[0] == 'S' || unit_str[0] == 's')
        time->unit = TIME_SEC;
    else if (unit_str[0] == 'M' || unit_str[0] == 'm') {
        if (strcmp(unit_str, "ms") == 0 || strcmp(unit_str, "MS") == 0)
            time->unit = TIME_MS;
        else
            time->unit = TIME_MIN;
    }
    else if (unit_str[0] == 'U' || unit_str[0] == 'u')
        time->unit = TIME_US;
    else {
        printf("time parsing failed\n");
        return -1;
    }
    time->time_num = num;
    return 0;
}

int
str_to_ull(char *str_in, unsigned long long *num_out)
{
    if (!str_in) {
        printf("Number parsing failed: \"%s\" is not recognized.\n", str_in);
        return -1;
    }
    unsigned long long num = 0;
    char *             unit_str;
    int                ret = parse_unit(str_in, &num, &unit_str);
    if (ret < 0)
        return -1;
    if (!unit_str)
        num = num * 1;
    else if (unit_str[0] == 'K' || unit_str[0] == 'k')
        num = num * K_VAL;
    else if (unit_str[0] == 'M' || unit_str[0] == 'm')
        num = num * M_VAL;
    else if (unit_str[0] == 'G' || unit_str[0] == 'g')
        num = num * G_VAL;
    else if (unit_str[0] == 'T' || unit_str[0] == 't')
        num = num * T_VAL;

    if (unit_str)
        free(unit_str);
    *num_out = num;
    return 0;
}

int
_set_io_pattern(bench_params *params_in_out)
{
    if (!params_in_out)
        return -1;
    int ret = 0;
    if (params_in_out->io_op == IO_WRITE) { // mem --> file
        if (params_in_out->mem_pattern == PATTERN_CONTIG) {
            if (params_in_out->file_pattern == PATTERN_CONTIG) { // CC
                switch (params_in_out->num_dims) {
                    case 1:
                        (*params_in_out).access_pattern.pattern_write = CONTIG_CONTIG_1D;
                        ret                                           = 0;
                        break;
                    case 2:
                        (*params_in_out).access_pattern.pattern_write = CONTIG_CONTIG_2D;
                        ret                                           = 0;
                        break;
                    case 3:
                        (*params_in_out).access_pattern.pattern_write = CONTIG_CONTIG_3D;
                        ret                                           = 0;
                        break;
                    default:
                        ret = -1;
                        printf("%s() failed on line %d\n", __func__, __LINE__);
                        break;
                }
            }
            else if (params_in_out->file_pattern == PATTERN_INTERLEAVED) { // CI
                if (params_in_out->num_dims == 1) {
                    (*params_in_out).access_pattern.pattern_write = CONTIG_COMPOUND_1D;
                    ret                                           = 0;
                }
                else if (params_in_out->num_dims == 2) {
                    (*params_in_out).access_pattern.pattern_write = CONTIG_COMPOUND_2D;
                    ret                                           = 0;
                }
                else {
                    ret = -1;
                    printf("%s() failed on line %d\n", __func__, __LINE__);
                }
            }
            else if (params_in_out->file_pattern == PATTERN_STRIDED) { // Strided write 1d
                if (params_in_out->num_dims == 1) {
                    (*params_in_out).access_pattern.pattern_write = CONTIG_CONTIG_STRIDED_1D;
                    ret                                           = 0;
                }
                else {
                    ret = -1;
                    printf("%s() failed on line %d\n", __func__, __LINE__);
                }
            }
            else {
                ret = -1;
                printf("%s() failed on line %d\n", __func__, __LINE__);
            }
        }
        else if (params_in_out->mem_pattern == PATTERN_INTERLEAVED) {
            if (params_in_out->file_pattern == PATTERN_CONTIG) { // IC
                if (params_in_out->num_dims == 1) {
                    (*params_in_out).access_pattern.pattern_write = COMPOUND_CONTIG_1D;
                    ret                                           = 0;
                }
                else if (params_in_out->num_dims == 2) {
                    (*params_in_out).access_pattern.pattern_write = COMPOUND_CONTIG_2D;
                    ret                                           = 0;
                }
                else {
                    ret = -1;
                    printf("%s() failed on line %d\n", __func__, __LINE__);
                }
            }
            else if (params_in_out->file_pattern == PATTERN_INTERLEAVED) { // II
                if (params_in_out->num_dims == 1) {
                    (*params_in_out).access_pattern.pattern_write = COMPOUND_COMPOUND_1D;
                    ret                                           = 0;
                }
                else if (params_in_out->num_dims == 2) {
                    (*params_in_out).access_pattern.pattern_write = COMPOUND_COMPOUND_2D;
                    ret                                           = 0;
                }
                else {
                    ret = -1;
                    printf("%s() failed on line %d\n", __func__, __LINE__);
                }
            }
        }
        else {
            ret = -1;
            printf("%s() failed on line %d\n", __func__, __LINE__);
        }
    }
    else if ((params_in_out->io_op == IO_READ) || (params_in_out->io_op == IO_OVERWRITE) ||
             (params_in_out->io_op == IO_APPEND)) { // file --> mem
        if (params_in_out->mem_pattern == PATTERN_CONTIG) {
            if (params_in_out->file_pattern == PATTERN_CONTIG) {
                if (params_in_out->read_option == LDC) {
                    switch (params_in_out->num_dims) {
                        case 2:
                            (*params_in_out).access_pattern.pattern_read = LDC_2D;
                            ret                                          = 0;
                            break;
                        default:
                            ret = -1;
                            printf("%s(). Unexpected Dimensions for LDC. failed on line %d\n", __func__,
                                   __LINE__);
                            break;
                    }
                }
                else if (params_in_out->read_option == RDC) {
                    switch (params_in_out->num_dims) {
                        case 2:
                            (*params_in_out).access_pattern.pattern_read = RDC_2D;
                            ret                                          = 0;
                            break;
                        default:
                            ret = -1;
                            printf("%s(). Unexpected Dimensions for RDC. failed on line %d\n", __func__,
                                   __LINE__);
                            break;
                    }
                }
                else if (params_in_out->read_option == PRL) {
                    switch (params_in_out->num_dims) {
                        case 2:
                            (*params_in_out).access_pattern.pattern_read = PRL_2D;
                            ret                                          = 0;
                            break;
                        default:
                            ret = -1;
                            printf("%s(). Unexpected Dimensions for PRL. failed on line %d\n", __func__,
                                   __LINE__);
                            break;
                    }
                }
                else if (params_in_out->read_option == CS) {
                    switch (params_in_out->num_dims) {
                        case 2:
                            (*params_in_out).access_pattern.pattern_read = CS_2D;
                            ret                                          = 0;
                            break;
                        default:
                            ret = -1;
                            printf("%s(). Unexpected Dimensions for CS. failed on line %d\n", __func__,
                                   __LINE__);
                            break;
                    }
                }
                else {
                    switch (params_in_out->num_dims) {
                        case 1:
                            (*params_in_out).access_pattern.pattern_read = CONTIG_1D;
                            ret                                          = 0;
                            break;
                        case 2:
                            (*params_in_out).access_pattern.pattern_read = CONTIG_2D;
                            ret                                          = 0;
                            break;
                        case 3:
                            (*params_in_out).access_pattern.pattern_read = CONTIG_3D;
                            ret                                          = 0;
                            break;
                        default:
                            ret = -1;
                            printf("%s() failed on line %d\n", __func__, __LINE__);
                            break;
                    }
                }
            }
            else if (params_in_out->file_pattern == PATTERN_STRIDED) {
                (*params_in_out).access_pattern.pattern_read = STRIDED_1D;
                ret                                          = 0;
            }
        }
        else {
            ret = -1;
            printf("%s() failed on line %d\n", __func__, __LINE__);
        }
    }
    else {
        ret = -1;
        printf("%s() failed on line %d\n", __func__, __LINE__);
    }
    if (ret < 0)
        printf("%s() failed, unsupported value/patterns.\n", __func__);
    return ret;
}

char *
_parse_val(char *val_in)
{
    char *val_str = strdup(val_in);
    char *tokens[2];
    char *tok = strtok(val_str, "#");
    char *val = NULL;
    val       = strdup(tok);
    //    printf("_parse_val: val_in = [%s], val = [%s]\n", val_in, val);
    if (val_str)
        free(val_str);
    return val;
}

int
_set_params(char *key, char *val_in, bench_params *params_in_out, int do_write)
{
    if (!params_in_out)
        return 0;
    char *val = _parse_val(val_in);

    if (strcmp(key, "IO_OPERATION") == 0) {
        if (strcmp(val, "READ") == 0) {
            params_in_out->io_op = IO_READ;
        }
        else if (strcmp(val, "WRITE") == 0) {
            params_in_out->io_op = IO_WRITE;
        }
        else if (strcmp(val, "OVERWRITE") == 0) {
            params_in_out->io_op = IO_OVERWRITE;
        }
        else if (strcmp(val, "APPEND") == 0) {
            params_in_out->io_op = IO_APPEND;
        }
        else {
            printf("Unknown value for \"IO_OPERATION\": %s\n", val);
            return -1;
        }
    }
    else if (strcmp(key, "MEM_PATTERN") == 0) {
        if (strcmp(val_in, "CONTIG") == 0) {
            params_in_out->mem_pattern = PATTERN_CONTIG;
        }
        else if (strcmp(val_in, "INTERLEAVED") == 0) {
            params_in_out->mem_pattern = PATTERN_INTERLEAVED;
        }
        else if (strcmp(val_in, "STRIDED") == 0) {
            params_in_out->mem_pattern = PATTERN_STRIDED;
        }
        else {
            params_in_out->mem_pattern = PATTERN_INVALID;
        }
    }
    else if (strcmp(key, "FILE_PATTERN") == 0) {
        if (strcmp(val_in, "CONTIG") == 0) {
            params_in_out->file_pattern = PATTERN_CONTIG;
        }
        else if (strcmp(val_in, "INTERLEAVED") == 0) {
            params_in_out->file_pattern = PATTERN_INTERLEAVED;
        }
        else if (strcmp(val_in, "STRIDED") == 0) {
            params_in_out->file_pattern = PATTERN_STRIDED;
        }
        else {
            params_in_out->file_pattern = PATTERN_INVALID;
        }
    }

    else if (strcmp(key, "TO_READ_NUM_PARTICLES") == 0) {
        if ((*params_in_out).io_op != IO_READ) {
            printf("TO_READ_CNT_M parameter is only used with READ_PATTERNs, please check config file.\n");
            return -1;
        }
        unsigned long long num = 0;
        if (str_to_ull(val, &num) < 0)
            return -1;
        (*params_in_out).try_num_particles = num;
    }
    else if (strcmp(key, "COLLECTIVE_METADATA") == 0) {
        if (val[0] == 'Y' || val[0] == 'y')
            (*params_in_out).meta_coll = 1;
        else
            (*params_in_out).meta_coll = 0;
    }
    else if (strcmp(key, "COLLECTIVE_DATA") == 0) {
        if (val[0] == 'Y' || val[0] == 'y')
            (*params_in_out).data_coll = 1;
        else
            (*params_in_out).data_coll = 0;
    }
    else if (strcmp(key, "COMPRESS") == 0) {
        if (val[0] == 'Y' || val[0] == 'y')
            (*params_in_out).useCompress = 1;
        else
            (*params_in_out).useCompress = 0;
    }
    else if (strcmp(key, "TIMESTEPS") == 0) {
        int ts_cnt = atoi(val);
        if (ts_cnt >= 1)
            (*params_in_out).cnt_time_step = ts_cnt;
        else {
            printf("TIMESTEPS must be at least 1.\n");
            return -1;
        }
    }
    else if (strcmp(key, "DELAYED_CLOSE_TIMESTEPS") == 0) {
        int delay_ts_cnt = atoi(val);
        if (delay_ts_cnt < 0)
            delay_ts_cnt = 0;
        (*params_in_out).cnt_time_step_delay = delay_ts_cnt;
    }
    else if (strcmp(key, "NUM_PARTICLES") == 0) { // 16M, 8K
        unsigned long long num = 0;
        if (str_to_ull(val, &num) < 0)
            return -1;

        if (num >= 1)
            (*params_in_out).num_particles = num;
        else {
            printf("NUM_PARTICLES must be at least 1.\n");
            return -1;
        }
    }
    else if (strcmp(key, "IO_MEM_LIMIT") == 0) {
        unsigned long long num = 0;
        if (str_to_ull(val, &num) < 0)
            return -1;
        if (num >= 0) {
            (*params_in_out).io_mem_limit = num;
        }
        else {
            printf("IO_MEM_LIMIT must be at least 0.\n");
            return -1;
        }
    }
    else if (strcmp(key, "EMULATED_COMPUTE_TIME_PER_TIMESTEP") == 0) {
        duration time;
        if (parse_time(val, &time) < 0)
            return -1;
        if (time.time_num >= 0)
            (*params_in_out).compute_time = time;
        else {
            printf("EMULATED_COMPUTE_TIME_PER_TIMESTEP must be at least 0.\n");
            return -1;
        }
    }
    else if (strcmp(key, "READ_OPTION") == 0) {
        if (strcmp(val_in, "FULL") == 0) { // FULL
            (*params_in_out).read_option = READ_FULL;
        }
        else if (strcmp(val_in, "PARTIAL") == 0) { // PARTIAL
            (*params_in_out).read_option = READ_PARTIAL;
        }
        else if (strcmp(val_in, "STRIDED") == 0) { // STRIDED
            (*params_in_out).read_option = READ_STRIDED;
        }
        else if (strcmp(val_in, "LDC") == 0) {
            (*params_in_out).read_option = LDC;
        }
        else if (strcmp(val_in, "RDC") == 0) {
            (*params_in_out).read_option = RDC;
        }
        else if (strcmp(val_in, "PRL") == 0) {
            (*params_in_out).read_option = PRL;
        }
        else if (strcmp(val_in, "CS") == 0) {
            (*params_in_out).read_option = CS;
        }

        else
            (*params_in_out).read_option = READ_OPTION_INVALID;
    }
    else if (strcmp(key, "NUM_DIMS") == 0) {
        int num = atoi(val);
        if (num > 0)
            (*params_in_out).num_dims = num;
        else {
            printf("NUM_DIMS must be at least 1\n");
            return -1;
        }
    }
    else if (strcmp(key, "DIM_1") == 0) {
        unsigned long long num = 0;
        if (str_to_ull(val, &num) < 0)
            return -1;
        if (num > 0)
            (*params_in_out).dim_1 = num;
        else {
            printf("DIM_1 must be at least 1\n");
            return -1;
        }
    }
    else if (strcmp(key, "DIM_2") == 0) {
        if ((*params_in_out).num_dims == 1)
            return 1;
        unsigned long long num = 0;
        if (str_to_ull(val, &num) < 0)
            return -1;
        if (num >= 1)
            (*params_in_out).dim_2 = num;
        else {
            printf("DIM_2 must be at least 1\n");
            return -1;
        }
    }
    else if (strcmp(key, "DIM_3") == 0) {
        if ((*params_in_out).num_dims == 1 || (*params_in_out).num_dims == 2)
            return 1;
        unsigned long long num = 0;
        if (str_to_ull(val, &num) < 0)
            return -1;
        if (num >= 1)
            (*params_in_out).dim_3 = num;
        else {
            printf("DIM_3 must be at least 1\n");
            return -1;
        }
    }
    else if (strcmp(key, "CHUNK_DIM_1") == 0) {
        unsigned long long dim = 0;
        if (str_to_ull(val, &dim) < 0)
            return -1;
        if (dim > 0)
            (*params_in_out).chunk_dim_1 = dim;
        else {
            printf("CHUNK_DIM_1 must be at least 1\n");
            return -1;
        }
    }
    else if (strcmp(key, "CHUNK_DIM_2") == 0) {
        if ((*params_in_out).num_dims == 1)
            return 1;
        unsigned long long dim = 0;
        if (str_to_ull(val, &dim) < 0)
            return -1;
        if (dim >= 1)
            (*params_in_out).chunk_dim_2 = dim;
        else {
            printf("CHUNK_DIM_2 must be at least 1.\n");
            return -1;
        }
    }
    else if (strcmp(key, "CHUNK_DIM_3") == 0) {
        if ((*params_in_out).num_dims == 1 || (*params_in_out).num_dims == 2)
            return 1;
        unsigned long long dim = 0;
        if (str_to_ull(val, &dim) < 0)
            return -1;
        if (dim >= 1)
            (*params_in_out).chunk_dim_3 = dim;
        else {
            printf("CHUNK_DIM_3 must be at least 1.\n");
            return -1;
        }
    }
    else if (strcmp(key, "STRIDE_SIZE") == 0) {
        unsigned long long num = 0;
        if (str_to_ull(val, &num) < 0)
            return -1;
        (*params_in_out).stride = num;
    }
    else if (strcmp(key, "STRIDE_SIZE_2") == 0) {
        unsigned long long num = 0;
        if (str_to_ull(val, &num) < 0)
            return -1;
        (*params_in_out).stride_2 = num;
    }
    else if (strcmp(key, "STRIDE_SIZE_3") == 0) {
        unsigned long long num = 0;
        if (str_to_ull(val, &num) < 0)
            return -1;
        (*params_in_out).stride_3 = num;
    }
    else if (strcmp(key, "BLOCK_SIZE") == 0) {
        unsigned long long num = 0;
        if (str_to_ull(val, &num) < 0)
            return -1;
        (*params_in_out).block_size = num;
    }
    else if (strcmp(key, "BLOCK_SIZE_2") == 0) {
        unsigned long long num = 0;
        if (str_to_ull(val, &num) < 0)
            return -1;
        (*params_in_out).block_size_2 = num;
    }
    else if (strcmp(key, "BLOCK_SIZE_3") == 0) {
        unsigned long long num = 0;
        if (str_to_ull(val, &num) < 0)
            return -1;
        (*params_in_out).block_size_3 = num;
    }
    else if (strcmp(key, "BLOCK_CNT") == 0) {
        unsigned long long num = 0;
        if (str_to_ull(val, &num) < 0)
            return -1;
        (*params_in_out).block_cnt = num;
    }
    else if (strcmp(key, "CSV_FILE") == 0) {
        (*params_in_out).useCSV   = 1;
        (*params_in_out).csv_path = strdup(val);\
    }
    else if (strcmp(key, "DATA_DIST_PATH") == 0) {
        (*params_in_out).useDataDist   = 1;
        (*params_in_out).data_dist_path = strdup(val);
    }
    else if (strcmp(key, "DATA_DIST_SCALE") == 0) {
      float num = 0.0;
      char *tok;
      tok = strtok(val, "/");
      num = strtof(tok, NULL);
      if (tok = strtok(NULL, "/"))
	num = num / strtof(tok, NULL); // two terms with / delim is fraction
      (*params_in_out).data_dist_scale = num;
    }
    else if (strcmp(key, "ENV_METADATA_FILE") == 0) {
        (*params_in_out).env_meta_path = strdup(val);
    }
    else if (strcmp(key, "FILE_PER_PROC") == 0) {
        if (val[0] == 'Y' || val[0] == 'y')
            (*params_in_out).file_per_proc = 1;
        else
            (*params_in_out).file_per_proc = 0;
    }
    else if (strcmp(key, "SUBFILING") == 0) {
        if (val[0] == 'Y' || val[0] == 'y') {
#ifndef HAVE_SUBFILING
            printf("HDF5 version does not support SUBFILING \n");
            return -1;
#endif
            (*params_in_out).subfiling = 1;
        }
        else
            (*params_in_out).subfiling = 0;
    }
    else if (strcmp(key, "ALIGN") == 0) {
        if (val[0] == 'Y' || val[0] == 'y') {
            (*params_in_out).align = 1;
        }
        else {
            (*params_in_out).align = 0;
        }
    }
    else if (strcmp(key, "ALIGN_THRESHOLD") == 0) {
        int align_threshold = atoi(val);
        if (align_threshold >= 0)
            (*params_in_out).align_threshold = align_threshold;
        else {
            printf("ALIGN_THRESHOLD must be >=0\n");
            return -1;
        }
    }
    else if (strcmp(key, "ALIGN_LEN") == 0) {
        int align_len = atoi(val);
        if (align_len >= 0)
            (*params_in_out).align_len = align_len;
        else {
            printf("ALIGN_LEN must be >=0\n");
            return -1;
        }
    }
    else if (strcmp(key, "STDEV_DIM_1") == 0) {
        unsigned long long stdev_dim_1 = atoi(val);
        if (stdev_dim_1 >= 0)
            (*params_in_out).stdev_dim_1 = stdev_dim_1;
        else {
            printf("STDEV_DIM_1 must be >=0\n");
            return -1;
        }
    }
    else {
        printf("Unknown Parameter: %s\n", key);
        return -1;
    }

    has_vol_async = has_vol_connector();

    if (has_vol_async) {
        (*params_in_out).asyncMode = MODE_ASYNC;
    }
    else {
        (*params_in_out).asyncMode = MODE_SYNC;
    }

    if ((*params_in_out).useCSV)
        (*params_in_out).csv_fs = csv_init(params_in_out->csv_path, params_in_out->env_meta_path);

    if (val)
        free(val);
    return 1;
}
void
bench_params_init(bench_params *params_out)
{
    if (!params_out)
        params_out = (bench_params *)calloc(1, sizeof(bench_params));
    (*params_out).pattern_name = NULL;
    (*params_out).meta_coll    = 0;
    (*params_out).data_coll    = 0;
    (*params_out).asyncMode    = MODE_SYNC;
    (*params_out).subfiling    = 0;
    (*params_out).useDataDist  = 0;

    (*params_out).cnt_time_step         = 0;
    (*params_out).cnt_time_step_delay   = 0;
    (*params_out).num_particles         = 0; // total number per rank
    (*params_out).io_mem_limit          = 0;
    (*params_out).try_num_particles     = 0; // to read
    (*params_out).compute_time.time_num = 0;
    (*params_out).num_dims              = 1;

    (*params_out).stride        = 0;
    (*params_out).stride_2      = 0;
    (*params_out).stride_3      = 0;
    (*params_out).block_size    = 0;
    (*params_out).block_size_2  = 0;
    (*params_out).block_size_3  = 0;
    (*params_out).block_cnt     = 0;
    (*params_out).dim_1         = 1;
    (*params_out).dim_2         = 1;
    (*params_out).dim_3         = 1;
    (*params_out).chunk_dim_1   = 1;
    (*params_out).chunk_dim_2   = 1;
    (*params_out).chunk_dim_3   = 1;
    (*params_out).csv_path      = NULL;
    (*params_out).env_meta_path = NULL;
    (*params_out).stdev_dim_1   = 1;

    (*params_out).csv_path        = NULL;
    (*params_out).csv_fs          = NULL;
    (*params_out).data_dist_path  = NULL;
    (*params_out).data_dist_scale = 1.0;
    (*params_out).env_meta_path   = NULL;
    (*params_out).file_per_proc   = 0;
    (*params_out).align           = 0;
    (*params_out).align_threshold = 0;
    (*params_out).align_len       = 0;
}

int
has_vol_connector()
{
#if H5_VERSION_GE(1, 13, 0)
    char *connector = getenv("HDF5_VOL_CONNECTOR");

    if (connector != NULL && strstr(connector, "async")) {
        return 1;
    }
#endif

    return 0;
}

int
read_config(const char *file_path, bench_params *params_out, int do_write)
{
    char cfg_line[CFG_LINE_LEN_MAX] = "";

    if (!params_out)
        params_out = (bench_params *)calloc(1, sizeof(bench_params));
    else
        memset(params_out, 0, sizeof(bench_params));
    // Default settings
    bench_params_init(params_out);
    (*params_out).data_file_path = strdup(file_path);

    FILE *file = fopen(file_path, "r");

    int parsed = 1;

    // default values
    (*params_out).useCSV = 0;
    if (do_write)
        (*params_out).io_op = IO_WRITE;
    else
        (*params_out).io_op = IO_READ;

    while (fgets(cfg_line, CFG_LINE_LEN_MAX, file) && (parsed == 1)) {
        if (cfg_line[0] == '#') { // skip comment lines
            continue;
        }
        char *tokens[2];
        char *tok = strtok(cfg_line, CFG_DELIMS);
        if (tok) {
            tokens[0] = tok;
            tok       = strtok(NULL, CFG_DELIMS);
            if (tok) {
                tokens[1] = tok;
            }
            else
                return -1;
        }
        else
            return -1;
        //        printf("key = [%s], val = [%s]\n", tokens[0], tokens[1]);
        parsed = _set_params(tokens[0], tokens[1], params_out, do_write);
    }
    if (parsed < 0)
        return -1;

    int ret = _set_io_pattern(params_out);
    if (ret < 0)
        return ret;

    if (params_out->io_op == IO_WRITE || params_out->io_op == IO_OVERWRITE ||
        params_out->io_op == IO_APPEND ||
        (params_out->io_op == IO_READ && params_out->try_num_particles == 0)) {
        (*params_out).num_particles = params_out->dim_1 * params_out->dim_2 * params_out->dim_3;
    }

    if (params_out->io_mem_limit > 0) {
        if (params_out->num_particles * PARTICLE_SIZE >= params_out->io_mem_limit) {
            printf("Requested memory (%llu particles, %llu, PARTICLE_SIZE = %ld) is larger than specified "
                   "memory bound (%llu), "
                   "please check IO_MEM_LIMIT in your config file.\n",
                   params_out->num_particles, params_out->num_particles * PARTICLE_SIZE, PARTICLE_SIZE,
                   params_out->io_mem_limit);
            return -1;
        }
    }
    if (params_out->io_op == IO_WRITE) {
        if (params_out->access_pattern.pattern_write == CONTIG_CONTIG_STRIDED_1D) {
            if (params_out->stride < 1 || params_out->block_size < 1 || params_out->block_cnt < 1) {
                printf("Strided read requires STRIDE_SIZE/BLOCK_SIZE/BLOCK_CNT no less than 1.\n");
                return -1;
            }
        }
    }
    else if ((params_out->io_op == IO_READ) || (params_out->io_op == IO_OVERWRITE) ||
             (params_out->io_op == IO_APPEND)) {                    // read-based operations
        if (params_out->access_pattern.pattern_read == CONTIG_1D) { // read whole file
            if (params_out->num_particles > 1)
                params_out->try_num_particles = params_out->num_particles;
            else
                params_out->num_particles = params_out->try_num_particles;
        }
        if (params_out->access_pattern.pattern_read == STRIDED_1D) {
            if (params_out->stride < 1 || params_out->block_size < 1 || params_out->block_cnt < 1) {
                printf("Strided read requires STRIDE_SIZE/BLOCK_SIZE/BLOCK_CNT no less than 1.\n");
                return -1;
            }
        }
        if (params_out->access_pattern.pattern_read == LDC_2D) {
            if (params_out->block_size < 1 || params_out->block_size_2 < 1) {
                printf("LDC read requires BLOCK_SIZE/BLOCK_SIZE_2 no less than 1.\n");
                return -1;
            }
        }
        if (params_out->access_pattern.pattern_read == RDC_2D) {
            if (params_out->block_size < 1 || params_out->block_size_2 < 1) {
                printf("RDC read requires BLOCK_SIZE/BLOCK_SIZE_2 no less than 1.\n");
                return -1;
            }
        }
        if (params_out->access_pattern.pattern_read == PRL_2D) {
            if (params_out->block_size < 1 || params_out->block_size_2 < 1) {
                printf("PRL read requires BLOCK_SIZE/BLOCK_SIZE_2 no less than 1.\n");
                return -1;
            }
        }
        if (params_out->access_pattern.pattern_read == CS_2D) {
            if (params_out->stride < 1 || params_out->stride_2 < 1) {
                printf("CS read requires STRIDE_SIZE/STRIDE_SIZE_2 no less than 1.\n");
                return -1;
            }
        }
    }
    if (params_out->subfiling > 0 && params_out->data_coll == 1) {
        printf("Subfiling does not support collective data buffering for data.\n");
        return -1;
    }

    return 0;
}

// print all fields of params
void
print_params(const bench_params *p)
{
    printf("\n");
    printf("================ Benchmark Configuration ==================\n");
    printf("File: %s\n", p->data_file_path);
    printf("Number of particles per rank: %llu\n", p->num_particles);
    printf("Number of time steps: %d\n", p->cnt_time_step);
    printf("Emulated compute time per timestep: %lu\n", p->compute_time.time_num);

    printf("Mode: %s\n", p->asyncMode == MODE_SYNC ? "SYNC" : "ASYNC");
    printf("Collective metadata operations: %s\n", p->meta_coll == 1 ? "YES" : "NO");
    printf("Collective buffering for data operations: %s\n", p->data_coll == 1 ? "YES" : "NO");
    if (p->subfiling) {
        printf("Use Subfiling: %s\n", p->subfiling == 1 ? "YES" : "NO");
    }
    printf("Number of dimensions: %d\n", p->num_dims);
    printf("    Dim_1: %lu\n", p->dim_1);
    if (p->num_dims >= 2) {
        printf("    Dim_2: %lu\n", p->dim_2);
    }
    if (p->num_dims >= 3) {
        printf("    Dim_3: %lu\n", p->dim_3);
    }

    if (p->access_pattern.pattern_read == STRIDED_1D ||
        p->access_pattern.pattern_write == CONTIG_CONTIG_STRIDED_1D) {
        printf("Strided access settings:\n");
        printf("    Stride size = %ld\n", p->stride);
        printf("    Block size = %ld\n", p->block_size);
    }

    if (p->useCompress) {
        printf("Use compression: %d\n", p->useCompress);
        printf("    chunk_dim1: %lu\n", p->chunk_dim_1);
        if (p->num_dims >= 2) {
            printf("    chunk_dim2: %lu\n", p->chunk_dim_2);
        }
        else if (p->num_dims >= 3) {
            printf("    chunk_dim3: %lu\n", p->chunk_dim_3);
        }
    }
    if (p->align) {
        printf("Align settings: \n");
        printf("    align  = %d\n", p->align);
        printf("    align threshold = %ld\n", p->align_threshold);
        printf("    align length = %ld\n", p->align_len);
    }
    if (p->stdev_dim_1) {
        printf("Standard deviation for varying particle size in normal distribution = %ld\n", p->stdev_dim_1);
    }
    printf("===========================================================\n");
    printf("\n");
}

void
bench_params_free(bench_params *p)
{
    if (!p)
        return;
    if (p->data_file_path)
        free(p->data_file_path);
    if (p->pattern_name)
        free(p->pattern_name);
}

int
file_create_try(const char *path)
{
    FILE *fs = fopen(path, "w+");
    if (!fs) {
        printf("Failed to create file: %s, Please check permission.\n", path);
        return -1;
    }
    fclose(fs);
    return 0;
}

int
file_exist(const char *path)
{
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

int
record_env_metadata(FILE *fs, const char *metadata_list_file)
{
    // read list file line, use each line as a key to search env
    if (!fs)
        return -1;
    FILE *lfs = fopen(metadata_list_file, "r");
    if (!lfs) {
        printf("Can not open metadata list file: %s\n", metadata_list_file);
        return -1;
    }

    fprintf(fs, "======================= Metadata =====================\n");

    char line[10 * CFG_LINE_LEN_MAX]; // some env val could be very large, such as PATH
    while (fgets(line, CFG_LINE_LEN_MAX, lfs)) {
        if (line[0] == '#') // skip comment lines
            continue;
        if (line[0] == '\n')
            continue;

        if (line[strlen(line) - 1] == '\n') {
            line[strlen(line) - 1] = 0;
        }

        char *val = getenv(line);
        // printf("%s = %s\n", line, val);
        fprintf(fs, "%s = %s\n", line, val);

        if (!val) { // null
            printf("    %s not set.\n", line);
            continue;
        }
    }

    fprintf(fs, "======================= Metadata end ====================\n");
    fclose(lfs);
    return 0;
}

FILE *
csv_init(const char *csv_path, const char *metadata_list_file)
{ //, const char* metadata_list_file: should be optional.
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

int
csv_output_line(FILE *fs, char *name, char *val_str)
{
    fprintf(fs, "%s,", name);
    fprintf(fs, " %s\n", val_str);
    return 0;
}

int
argv_print(int argc, char *argv[])
{
    if (argc < 1)
        return -1;
    printf("%d arguments provided.\n", argc);
    for (int i = 0; i < argc; i++) {
        printf("idx = %d, argv = %s\n", i, argv[i]);
    }
    return 0;
}

char *
get_file_name_from_path(char *path)
{
    if (path == NULL)
        return NULL;

    char *pFileName = path;
    for (char *pCur = path; *pCur != '\0'; pCur++) {
        if (*pCur == '/' || *pCur == '\\')
            pFileName = pCur + 1;
    }

    return pFileName;
}

char *
substr(char *src, size_t start, size_t len)
{
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

char *
get_dir_from_path(char *path)
{
    if (path == NULL)
        return NULL;

    char *pDir = substr(path, 0, strlen(path) - strlen(get_file_name_from_path(path)));

    return pDir;
}

human_readable
format_human_readable(uint64_t bytes)
{
    human_readable value;

    char unit[] = {' ', 'K', 'M', 'G', 'T'};
    char length = sizeof(unit) / sizeof(unit[0]);

    int    i      = 0;
    double format = bytes;

    if (bytes >= 1024) {
        for (i = 0; (bytes / 1024) > 0 && i < length - 1; i++, bytes /= 1024)
            format = bytes / 1024.0;
    }

    value.value = format;
    value.unit  = unit[i];

    return value;
}
