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

static void h5xfer_free(h5xfer *inst)
{
  if(inst)
  {
    free(inst);
  }
}

static h5xfer_functions const h5xfer_vtable = {
  &h5xfer_free
};

h5xfer* h5xfer_new()
{
  h5xfer *inst = malloc( sizeof(h5xfer) );
  if(!inst)
  {
    fprintf(stderr, "failed to create h5xfer instance\n");
    return NULL;
  }
  return inst;
}

