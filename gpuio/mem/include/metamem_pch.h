#ifndef METAMEM_GLOBALS_H
#define METAMEM_GLOBALS_H

#ifdef METAMEM_USE_CUDA
  #include <cuda.h>
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

#ifdef METAMEM_USE_HIP
  #include <hip/hip_runtime.h>
  #include <hip/hip_runtime_api.h>

  #define HIP_ASSERT(x) (assert((x)==hipSuccess))

  #define HIP_RUNTIME_API_CALL(apiFuncCall)                                    \
  {                                                                        \
    hipError_t _status = apiFuncCall;                                      \
    if (_status != hipSuccess) {                                           \
      fprintf(stderr, "%s:%d: error: function %s failed with error %s.\n", \
        __FILE__, __LINE__, #apiFuncCall, hipGetErrorString(_status));     \
      exit(EXIT_FAILURE);                                                  \
    }                                                                      \
  }
#endif

typedef enum {
  METAMEM_POSIX = 0,
  METAMEM_CUDA,
  METAMEM_HIP,
  METAMEM_OneAPI
} metamem_api_t;

typedef enum {
  MEM_CPU_PAGEABLE = 0,
  MEM_CPU_PINNED,
  MEM_CPU_GPU_MANAGED,
  MEM_GPU
} mem_type_t;

#endif // METAMEM_GLOBALS_H
