#include "h5device.h"
#include <stdio.h>

static void
h5device_free(h5device *inst)
{
    if (inst) {
        free(inst);
    }
}

static h5device_functions const h5device_vtable = {&h5device_free};

h5device *
h5device_new()
{
    h5device *inst = malloc(sizeof(h5device));
    if (!inst) {
        fprintf(stderr, "failed to create h5device instance\n");
        return NULL;
    }
    inst->fn = &h5device_vtable;
    return inst;
}
