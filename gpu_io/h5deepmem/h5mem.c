#include "h5mem.h"

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

static void
h5mem_free(h5mem *inst, h5deepmem_api_t device_api)
{
    if (device_api == DEEPMEM_CUDA) {
        switch (inst->mem_type) {
            case H5MEM_CPU_PAGEABLE:
                free(inst->ptr);
                break;
            case H5MEM_CPU_PINNED:
                CUDA_RUNTIME_API_CALL(cudaFreeHost(inst->ptr));
                break;
            case H5MEM_CPU_GPU_MANAGED:
            case H5MEM_GPU:
                CUDA_RUNTIME_API_CALL(cudaFree(inst->ptr));
                break;
        }
    }
    else if (device_api == DEEPMEM_HIP) {
        fprintf(stderr, "not implemented %s:%d\n", __FILE__, __LINE__);
        exit(EXIT_FAILURE);
    }
    else if (device_api == DEEPMEM_OneAPI) {
        fprintf(stderr, "not implemented %s:%d\n", __FILE__, __LINE__);
        exit(EXIT_FAILURE);
    }
    else {
        fprintf(stderr, "not implemented %s:%d\n", __FILE__, __LINE__);
        exit(EXIT_FAILURE);
    }
    free(inst);
}

static h5mem_functions const h5mem_vtable = {&h5mem_free};

h5mem *
h5mem_alloc(size_t nitems, size_t size, h5deepmem_api_t device_api, h5mem_type_t mem_type)
{
    h5mem *inst = malloc(sizeof(h5mem));
    if (!inst) {
        fprintf(stderr, "failed to create h5mem instance\n");
        return NULL;
    }

    inst->nitems   = nitems;
    inst->size     = size;
    inst->mem_type = mem_type;

    if (device_api == DEEPMEM_CUDA) {
        switch (mem_type) {
            case H5MEM_CPU_PAGEABLE:
                inst->ptr = malloc(nitems * size);
                if (!inst->ptr) {
                    fprintf(stderr, "failed to allocate memory for buffer\n");
                    return NULL;
                }
                break;
            case H5MEM_CPU_PINNED:
                CUDA_RUNTIME_API_CALL(
                    cudaHostAlloc((void **)&inst->ptr, nitems * size, cudaHostAllocDefault));
                break;
            case H5MEM_CPU_GPU_MANAGED:
                CUDA_RUNTIME_API_CALL(
                    cudaMallocManaged((void **)&inst->ptr, nitems * size, cudaMemAttachGlobal));
                break;
            case H5MEM_GPU:
                CUDA_RUNTIME_API_CALL(cudaMalloc((void **)&inst->ptr, nitems * size));
                break;
        }
    }
    else if (device_api == DEEPMEM_HIP) {
        fprintf(stderr, "not implemented %s:%d\n", __FILE__, __LINE__);
        exit(EXIT_FAILURE);
    }
    else if (device_api == DEEPMEM_OneAPI) {
        fprintf(stderr, "not implemented %s:%d\n", __FILE__, __LINE__);
        exit(EXIT_FAILURE);
    }
    else {
        fprintf(stderr, "not implemented %s:%d\n", __FILE__, __LINE__);
        exit(EXIT_FAILURE);
    }

    inst->fn = &h5mem_vtable;
    return inst;
}
