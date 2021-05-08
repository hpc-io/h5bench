#include "h5deepmem.h"

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

static void h5deepmem_copy(h5deepmem *inst, h5xfer_direction_t direction)
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

static void h5deepmem_free(h5deepmem *inst)
{
  if(inst->host_ptr)
  {
    inst->host_ptr->fn->free(inst->host_ptr, inst->device_api);
  }

  if(inst->device_ptr)
  {
    inst->device_ptr->fn->free(inst->device_ptr, inst->device_api);
  }

  free(inst->xfer_inst);
  free(inst);
}

static h5deepmem_functions const h5deepmem_vtable = {
  &h5deepmem_copy,
  &h5deepmem_free
};

h5deepmem* h5deepmem_alloc(size_t nitems, size_t size, h5deepmem_api_t device_api, h5mem_type_t host_mem_type, h5mem_type_t device_mem_type)
{
  h5deepmem *inst = malloc( sizeof(h5deepmem) );
  if(!inst)
  {
    fprintf(stderr, "failed to create h5deepmem instance\n");
    return NULL;
  }
  inst->device_api = device_api;
  inst->host_ptr = h5mem_alloc(nitems, size, inst->device_api, host_mem_type);
  inst->device_ptr = h5mem_alloc(nitems, size, inst->device_api, device_mem_type);
  inst->xfer_inst = h5xfer_new();
  inst->fn = &h5deepmem_vtable;
  return inst;
}

