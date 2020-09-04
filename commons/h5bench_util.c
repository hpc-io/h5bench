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
