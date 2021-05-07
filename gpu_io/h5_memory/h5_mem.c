#include "h5_mem.h"

#include <stdio.h>
#include <string.h> // memset

#include <unistd.h>
#include <string.h>
#include <time.h>
#include <sys/types.h>
#include <sys/mman.h>
#include <fcntl.h>

#include <stdarg.h>
#include <limits.h>

#include <stdint.h>

#include <sys/types.h>
#include <sys/stat.h>

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

static void h5_mem_free(h5_mem *inst)
{
  if(inst)
  {
    if(inst->ptr) free(inst->ptr);
    free(inst);
  }
}

static h5_mem_functions const h5_mem_vtable = {
  &h5_mem_free
};

h5_mem* h5_mem_alloc(size_t nitems, size_t size, h5_mem_type_t mem_type)
{
  h5_mem *inst = malloc( sizeof(h5_mem) );
  if(!inst)
  {
    fprintf(stderr, "failed to create h5_mem instance\n");
    return NULL;
  }

  inst->nitems = nitems;
  inst->size = size;
  inst->mem_type = mem_type;

  switch(mem_type)
  {
    case MEMORY_CPU_PAGEABLE:
      inst->ptr = malloc(nitems*size);
      if(!inst->ptr)
      {
        fprintf(stderr, "failed to allocate memory for buffer\n");
        return NULL;
      }
      break;
    case MEMORY_CPU_PINNED:
      CUDA_RUNTIME_API_CALL(cudaHostAlloc((void **)&inst->ptr, nitems*size, cudaHostAllocDefault));
      break;
    case MEMORY_CPU_GPU_MANAGED:
      CUDA_RUNTIME_API_CALL(cudaMallocManaged((void **)&inst->ptr, nitems*size, cudaMemAttachGlobal));
      break;
    case MEMORY_GPU:
      CUDA_RUNTIME_API_CALL(cudaMalloc((void **)&inst->ptr, nitems*size));
      break;
  }

  return inst;
}

