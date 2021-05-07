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

static void h5deepmem_free(h5deepmem *inst)
{
  if(inst)
  {
    free(inst);
  }
}

static h5deepmem_functions const h5deepmem_vtable = {
  &h5deepmem_free
};

h5deepmem* h5deepmem_new()
{
  h5deepmem *inst = malloc( sizeof(h5deepmem) );
  if(!inst)
  {
    fprintf(stderr, "failed to create h5deepmem instance\n");
    return NULL;
  }
  return inst;
}

