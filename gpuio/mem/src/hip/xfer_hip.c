#include "xfer.h"

#include <stdlib.h>
#include <stdio.h>

static void
xfer_copy(xfer *inst, void *dest, void *src, size_t size, metamem_api_t device_api, mem_type_t dest_type,
          mem_type_t src_type)
{
    if (dest_type == MEM_GPU && src_type == MEM_CPU_PAGEABLE) {
        HIP_RUNTIME_API_CALL(hipMemcpy(dest, src, size, hipMemcpyHostToDevice));
    }
    else if (dest_type == MEM_CPU_PAGEABLE && src_type == MEM_GPU) {
        HIP_RUNTIME_API_CALL(hipMemcpy(dest, src, size, hipMemcpyDeviceToHost));
    }
    else if (dest_type == MEM_GPU && src_type == MEM_CPU_PINNED) {
        HIP_RUNTIME_API_CALL(hipMemcpy(dest, src, size, hipMemcpyHostToDevice));
    }
    else if (dest_type == MEM_CPU_PINNED && src_type == MEM_GPU) {
        HIP_RUNTIME_API_CALL(hipMemcpy(dest, src, size, hipMemcpyDeviceToHost));
    }
    else if (dest_type == MEM_CPU_GPU_MANAGED && src_type == MEM_CPU_GPU_MANAGED) {
        // devptr, size, device
        // HIP_RUNTIME_API_CALL(hipMemPrefetchAsync(dest, size, 0, 0));
        fprintf(stderr, "not implemented %s:%d\n", __FILE__, __LINE__);
        exit(EXIT_FAILURE);
    }
    else {
        fprintf(stderr, "not implemented %s:%d\n", __FILE__, __LINE__);
        exit(EXIT_FAILURE);
    }
}

static void
xfer_free(xfer *inst)
{
    if (inst)
        free(inst);
}

static xfer_functions const xfer_vtable = {&xfer_copy, &xfer_free};

xfer *
xfer_new()
{
    xfer *inst = malloc(sizeof(xfer));
    if (!inst) {
        fprintf(stderr, "failed to create xfer instance\n");
        return NULL;
    }
    inst->fn = &xfer_vtable;
    return inst;
}
