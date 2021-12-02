#include "metamem_pch.h"
#include "xfer.h"

#include <stdlib.h>
#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>

// blocking copy 
static void xfer_copy(xfer *inst, void *dest, void *src, size_t size, metamem_api_t device_api, mem_type_t dest_type, mem_type_t src_type)
{
  if(dest_type == MEM_GPU && src_type == MEM_CPU_PAGEABLE)
  {
    CUDA_RUNTIME_API_CALL(cudaMemcpy(dest, src, size, cudaMemcpyHostToDevice));
  }
  else if(dest_type == MEM_CPU_PAGEABLE && src_type == MEM_GPU)
  {
    CUDA_RUNTIME_API_CALL(cudaMemcpy(dest, src, size, cudaMemcpyDeviceToHost));
  }
  else if(dest_type == MEM_GPU && src_type == MEM_CPU_PINNED)
  {
    CUDA_RUNTIME_API_CALL(cudaMemcpy(dest, src, size, cudaMemcpyHostToDevice));
  }
  else if(dest_type == MEM_CPU_PINNED && src_type == MEM_GPU)
  {
    CUDA_RUNTIME_API_CALL(cudaMemcpy(dest, src, size, cudaMemcpyDeviceToHost));
  }
  else if(dest_type == MEM_CPU_GPU_MANAGED && src_type == MEM_CPU_GPU_MANAGED)
  {
    // the point of uvm is that you dont have to do a copy..
    CUDA_RUNTIME_API_CALL(cudaMemPrefetchAsync(dest, size, 0, 0));
    CUDA_RUNTIME_API_CALL(cudaStreamSynchronize(0));
  }
  else
  {
    fprintf(stderr, "not implemented %s:%d\n", __FILE__, __LINE__); exit(EXIT_FAILURE);
  }
}

static void xfer_free(xfer *inst)
{
  if(inst)   free(inst);
}

static xfer_functions const xfer_vtable = {
  &xfer_copy,
  &xfer_free
};

xfer* xfer_new()
{
  xfer *inst = malloc( sizeof(xfer) );
  if(!inst)
  {
    fprintf(stderr, "failed to create xfer instance\n");
    return NULL;
  }
  inst->fn = &xfer_vtable;
  return inst;
}

