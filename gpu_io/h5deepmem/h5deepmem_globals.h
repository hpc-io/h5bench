#ifndef H5DEEPMEM_GLOBALS_H
#define H5DEEPMEM_GLOBALS_H

#ifdef HDF5_USE_CUDA
#include <cuda_runtime.h>
#define CUDA_RUNTIME_API_CALL(apiFuncCall)                               \
{                                                                        \
  cudaError_t _status = apiFuncCall;                                     \
  if (_status != cudaSuccess) {                                          \
    fprintf(stderr, "%s:%d: error: function %s failed with error %s.\n", \
      __FILE__, __LINE__, #apiFuncCall, cudaGetErrorString(_status));    \
    exit(-1);                                                            \
  }                                                                      \
}
#endif

typedef enum {
  DEEPMEM_CUDA,
  DEEPMEM_HIP,
  DEEPMEM_OneAPI
} h5deepmem_api_t;

typedef enum {
  H5MEM_CPU_PAGEABLE,
  H5MEM_CPU_PINNED,
  H5MEM_CPU_GPU_MANAGED,
  H5MEM_GPU
} h5mem_type_t;

#endif // H5DEEPMEM_GLOBALS_H
