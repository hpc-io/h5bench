#include "../include/metamem_api.h"
#include <stdio.h>
#include <stdlib.h>

static metamem* metamem_alloc(metamem *inst, size_t nitems, size_t size, mem_type_t host_mem_type, mem_type_t device_mem_type)
{
  inst->host_ptr = mem_alloc(nitems, size, inst->device_api, host_mem_type);

  if( (inst->device_api == METAMEM_POSIX) || (host_mem_type == MEM_CPU_GPU_MANAGED) )
  {
    inst->device_ptr = inst->host_ptr;
  }
  else 
  {
    inst->device_ptr = mem_alloc(nitems, size, inst->device_api, device_mem_type);
  }
  return inst;
}

static void metamem_copy(metamem *inst, xfer_direction_t direction)
{
  if(direction == H2D)
  {
    inst->xfer_inst->fn->copy(inst->xfer_inst,
      inst->device_ptr->ptr,
      inst->host_ptr->ptr,
      inst->host_ptr->nitems*inst->host_ptr->size,
      inst->device_api,
      inst->device_ptr->mem_type,
      inst->host_ptr->mem_type);
  }
  else if (direction == D2H)
  {
    inst->xfer_inst->fn->copy(inst->xfer_inst,
      inst->host_ptr->ptr,
      inst->device_ptr->ptr,
      inst->host_ptr->nitems*inst->host_ptr->size,
      inst->device_api,
      inst->host_ptr->mem_type,
      inst->device_ptr->mem_type);
  }
  else
  {
    fprintf(stderr, "not implemented %s:%d\n", __FILE__, __LINE__); exit(EXIT_FAILURE);
  }
}

static void metamem_free(metamem *inst)
{
  // if(inst->host_ptr)    inst->host_ptr->fn->free(inst->host_ptr, inst->device_api);
  if(inst->device_ptr)    inst->device_ptr->fn->free(inst->device_ptr, inst->device_api);
}

static metamem_functions const metamem_vtable = {
  &metamem_alloc,
  &metamem_copy,
  &metamem_free
};

metamem* metamem_init(metamem_api_t device_api)
{
  metamem *inst = malloc( sizeof(metamem) );
  if(!inst)
  {
    fprintf(stderr, "failed to create metamem instance\n");
    return NULL;
  }

  inst->device_api = device_api;
  inst->xfer_inst = xfer_new();
  inst->fn = &metamem_vtable;
  return inst;
}

void metamem_shutdown(metamem* inst)
{
  if(inst->xfer_inst)    free(inst->xfer_inst);
  if(inst)    free(inst);
}

