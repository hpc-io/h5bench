#include "h5xfer.h"

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
h5xfer_copy(h5xfer *inst, void *dest, void *src, size_t size, h5deepmem_api_t device_api,
            h5mem_type_t dest_type, h5mem_type_t src_type)
{
    if (device_api == DEEPMEM_CUDA) {
        if (dest_type == H5MEM_GPU && src_type == H5MEM_CPU_PAGEABLE) {
            CUDA_RUNTIME_API_CALL(cudaMemcpy(dest, src, size, cudaMemcpyHostToDevice));
        }
        else if (dest_type == H5MEM_CPU_PAGEABLE && src_type == H5MEM_GPU) {
            CUDA_RUNTIME_API_CALL(cudaMemcpy(dest, src, size, cudaMemcpyDeviceToHost));
        }
        else if (dest_type == H5MEM_GPU && src_type == H5MEM_CPU_PINNED) {
            CUDA_RUNTIME_API_CALL(cudaMemcpy(dest, src, size, cudaMemcpyHostToDevice));
        }
        else if (dest_type == H5MEM_CPU_PINNED && src_type == H5MEM_GPU) {
            CUDA_RUNTIME_API_CALL(cudaMemcpy(dest, src, size, cudaMemcpyDeviceToHost));
        }
        else if (dest_type == H5MEM_CPU_GPU_MANAGED && src_type == H5MEM_CPU_GPU_MANAGED) {
            // devptr, size, device
            // CUDA_RUNTIME_API_CALL(cudaMemPrefetchAsync(dest, size, 0, 0));
            fprintf(stderr, "not implemented %s:%d\n", __FILE__, __LINE__);
            exit(EXIT_FAILURE);
        }
        else {
            fprintf(stderr, "not implemented %s:%d\n", __FILE__, __LINE__);
            exit(EXIT_FAILURE);
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
}

static void
h5xfer_free(h5xfer *inst)
{
    if (inst) {
        free(inst);
    }
}

static h5xfer_functions const h5xfer_vtable = {&h5xfer_copy, &h5xfer_free};

h5xfer *
h5xfer_new()
{
    h5xfer *inst = malloc(sizeof(h5xfer));
    if (!inst) {
        fprintf(stderr, "failed to create h5xfer instance\n");
        return NULL;
    }
    inst->fn = &h5xfer_vtable;
    return inst;
}
