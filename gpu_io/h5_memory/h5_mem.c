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

h5_mem* h5_mem_alloc(size_t nitems, size_t size)
{
  h5_mem *inst = malloc( sizeof(h5_mem) );
  if(!inst)
  {
    fprintf(stderr, "failed to create h5_mem instance\n");
    return NULL;
  }

  inst->nitems = nitems;
  inst->size = size;
  inst->ptr = malloc(nitems*size);
  if(!inst->ptr)
  {
    fprintf(stderr, "failed to allocate memory for buffer\n");
    return NULL;
  }

  return inst;
}

