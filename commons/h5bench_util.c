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

#include "h5bench_util.h"

unsigned long get_time_usec(){
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return 1000000 * tv.tv_sec + tv.tv_usec;
}

int metric_msg_print(unsigned long number, char* msg, char* unit){
    printf("%s %lu %s\n", msg, number, unit);
    return 0;
}

int _set_params(char* key, char* val, bench_params* params_in_out){
    if(!params_in_out)
        return 0;

    if(strcmp(key, "PATTERN") == 0){
        if (strcmp(val, "CC") == 0) {
            (*params_in_out).bench_pattern = CONTIG_CONTIG_1D;
            (*params_in_out).pattern_name = strdup("CONTIG_CONTIG_1D");
            (*params_in_out)._dim_cnt = 1;
        } else if (strcmp(val, "CC2D") == 0) {
            (*params_in_out).bench_pattern = CONTIG_CONTIG_2D;
            (*params_in_out).pattern_name = strdup("CONTIG_CONTIG_2D");
            (*params_in_out)._dim_cnt = 2;
        } else if (strcmp(val, "CI") == 0) {
            (*params_in_out).bench_pattern = CONTIG_INTERLEAVED_1D;
            (*params_in_out).pattern_name = strdup("CONTIG_INTERLEAVED_1D");
            (*params_in_out)._dim_cnt = 1;
        } else if (strcmp(val, "CI2D") == 0) {
            (*params_in_out).bench_pattern = CONTIG_INTERLEAVED_2D;
            (*params_in_out).pattern_name = strdup("CONTIG_INTERLEAVED_2D");
            (*params_in_out)._dim_cnt = 2;
        }else if (strcmp(val, "II") == 0) {
            (*params_in_out).bench_pattern = INTERLEAVED_INTERLEAVED_1D;
            (*params_in_out).pattern_name = strdup("INTERLEAVED_INTERLEAVED_1D");
            (*params_in_out)._dim_cnt = 1;
        }else if (strcmp(val, "II2D") == 0) {
            (*params_in_out).bench_pattern = INTERLEAVED_INTERLEAVED_2D;
            (*params_in_out).pattern_name = strdup("INTERLEAVED_INTERLEAVED_2D");
            (*params_in_out)._dim_cnt = 2;
        } else if (strcmp(val, "IC") == 0) {
            (*params_in_out).bench_pattern = INTERLEAVED_CONTIG_1D;
            (*params_in_out).pattern_name = strdup("INTERLEAVED_CONTIG_1D");
            (*params_in_out)._dim_cnt = 1;
        } else if (strcmp(val, "IC2D") == 0) {
            (*params_in_out).bench_pattern = INTERLEAVED_CONTIG_2D;
            (*params_in_out).pattern_name = strdup("INTERLEAVED_CONTIG_2D");
            (*params_in_out)._dim_cnt = 2;
        } else if(strcmp(val, "CC3D") == 0){
            (*params_in_out).bench_pattern = CONTIG_CONTIG_3D;
            (*params_in_out).pattern_name = strdup("CONTIG_CONTIG_3D");
            (*params_in_out)._dim_cnt = 3;
        }
    } else if(strcmp(key, "TIME_STEPS_CNT") == 0){
        int ts_cnt = atoi(val);
        if(ts_cnt >= 1)
            (*params_in_out).cnt_time_step = ts_cnt;
        else {
            printf("TIME_STEPS_CNT must be at least 1.\n");
            return -1;
        }
    } else if(strcmp(key, "PARTICLE_CNT_M") == 0){
        int ts_cnt = atoi(val);
        if(ts_cnt >= 1)
            (*params_in_out).cnt_particle_M = ts_cnt;
        else {
            printf("PARTICLE_CNT_M must be at least 1.\n");
            return -1;
        }
    } else if(strcmp(key, "SLEEP_TIME") == 0){
        int sleep_time = atoi(val);
        if(sleep_time >= 0)
            (*params_in_out).sleep_time = sleep_time;
        else{
            printf("SLEEP_TIME must be at least 0.\n");
            return -1;
        }
    } else if(strcmp(key, "DIM_1") == 0){
        if((*params_in_out)._dim_cnt == 1)
            return 1;
        int dim = atoi(val);
        if(dim > 0)
            (*params_in_out).dim_1 = dim;
        else {
            printf("SLEEP_TIME must be at least 0.\n");
        }
    } else if(strcmp(key, "DIM_2") == 0){
        if((*params_in_out)._dim_cnt == 1)
            return 1;
        int dim = atoi(val);
        if(dim >= 1)
            (*params_in_out).dim_2 = dim;
        else {
            printf("DIM_2 must be at least 1.\n");
            return -1;
        }
    } else if(strcmp(key, "DIM_3") == 0){
        if((*params_in_out)._dim_cnt == 1 || (*params_in_out)._dim_cnt == 2)
            return 1;
        int dim = atoi(val);
        if(dim >= 1)
            (*params_in_out).dim_3 = dim;
        else {
            printf("DIM_3 must be at least 1.\n");
            return -1;
        }
    }
    return 1;
}

int read_config(const char* file_path, bench_params* params_out){
    if(!params_out)
        params_out = (bench_params*)calloc(1, sizeof(bench_params));
    char cfg_line[CFG_LINE_LEN_MAX] = "";
    (*params_out).data_file_path = strdup(file_path);
    FILE* file = fopen(file_path, "r");
    char* key, val;
    int parsed = 1;

    while(fgets(cfg_line, CFG_LINE_LEN_MAX, file) && (parsed == 1)){
        if(cfg_line[0] == '#'){ //skip comment lines
            continue;
        }
        char* tokens[2];
        char* tok = strtok(cfg_line, CFG_DELIMS);
        if(tok){
            tokens[0] = tok;
            tok = strtok(NULL, CFG_DELIMS);
            if(tok)
                tokens[1] = tok;
            else
                return -1;
        } else
            return -1;
        //printf("key = [%s], val = [%s]\n", tokens[0], tokens[1]);
        parsed = _set_params(tokens[0], tokens[1], params_out);
    }

    if(parsed < 0)
        return -1;
    else
        return 0;
}

void print_params(const bench_params* p){
    printf("=======================================\n");
    printf("Benchmark parameters: read from config file: %s\n", p->data_file_path);
    printf("Benchmark pattern = %s\n", p->pattern_name);
    printf("Per rank particles number(in M) = %d M\n", p->cnt_particle_M);
    printf("Time step number = %d\n", p->cnt_time_step);
    printf("Sleep time = %d\n", p->sleep_time);
    printf("Dimension cnt = %d\n", p->_dim_cnt);
    if(p->_dim_cnt == 2){
        printf("    Dim_1 = %d\n", p->dim_1);
        printf("    Dim_2 = %d\n", p->dim_2);
    } else if(p->_dim_cnt == 3){
        printf("    Dim_1 = %d\n", p->dim_1);
        printf("    Dim_2 = %d\n", p->dim_2);
        printf("    Dim_3 = %d\n", p->dim_3);
    }
    printf("=======================================\n");
}

void bench_params_free(bench_params* p){
    if(!p)
        return;
    free(p->data_file_path);
    free(p->pattern_name);
}

void test_read_config(const char* file_path){
     bench_params param;
     DEBUG_PRINT
     int ret = read_config(file_path, &param);
     DEBUG_PRINT
     if(ret != 0){
         printf("read_config() failed, ret = %d\n", ret);
     } else {
         printf("param->bench_pattern = %d\n", param.bench_pattern);
         printf("param->pattern_name = %s\n", param.pattern_name);
         printf("param->cnt_time_step = %d\n", param.cnt_time_step);
         printf("param->cnt_particle_M = %d\n", param.cnt_particle_M);
         printf("param->sleep_time = %d\n", param.sleep_time);
         printf("param->DIM_1 = %d\n", param.dim_1);
         printf("param->DIM_2 = %d\n", param.dim_2);
         printf("param->DIM_3 = %d\n", param.dim_3);
     }
}
