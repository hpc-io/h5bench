#include "device.h"
#include <stdio.h>
#include <stdlib.h>

static void device_free(device *inst)
{
  if(inst)    free(inst);
}

static device_functions const device_vtable = {
  &device_free
};

device* device_new()
{
  device *inst = malloc( sizeof(device) );
  if(!inst)
  {
    fprintf(stderr, "failed to create device instance\n");
    return NULL;
  }
  inst->fn = &device_vtable;
  return inst;
}

