#pragma once

#include <hdf5.h>

#include <inttypes.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>
#include <stdlib.h>
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime_api.h>

#define RUNTIME_API_CALL(apiFuncCall)                                    \
{                                                                        \
  cudaError_t _status = apiFuncCall;                                     \
  if (_status != cudaSuccess) {                                          \
    fprintf(stderr, "%s:%d: error: function %s failed with error %s.\n", \
      __FILE__, __LINE__, #apiFuncCall, cudaGetErrorString(_status));    \
    exit(-1);                                                            \
  }                                                                      \
}

#ifdef __cplusplus
extern "C" {
#endif

  void kernel_call(data_contig_md *data, volatile int *kernel_flag, cudaStream_t stream_id);

#ifdef __cplusplus
}
#endif

