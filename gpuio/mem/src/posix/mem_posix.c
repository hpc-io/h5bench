#include "mem.h"

#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>

static void
mem_free(mem *inst, metamem_api_t device_api)
{
    switch (inst->mem_type) {
        case MEM_CPU_PAGEABLE:
            free(inst->ptr);
            break;
        case MEM_CPU_PINNED:
            munlock(inst->ptr, inst->nitems * inst->size);
            free(inst->ptr);
            break;
    }

    free(inst);
}

static mem_functions const mem_vtable = {&mem_free};

mem *
mem_alloc(size_t nitems, size_t size, metamem_api_t device_api, mem_type_t mem_type)
{
    mem *inst = malloc(sizeof(mem));
    if (!inst) {
        fprintf(stderr, "failed to create mem instance\n");
        return NULL;
    }

    inst->nitems   = nitems;
    inst->size     = size;
    inst->mem_type = mem_type;

    switch (mem_type) {
        case MEM_CPU_PAGEABLE:
            inst->ptr = malloc(nitems * size);
            if (!inst->ptr) {
                fprintf(stderr, "failed to allocate memory for buffer\n");
                return NULL;
            }
            break;
        case MEM_CPU_PINNED:
            inst->ptr = malloc(nitems * size);
            if (!inst->ptr) {
                fprintf(stderr, "failed to allocate memory for buffer\n");
                return NULL;
            }
            mlock(inst->ptr, nitems * size);
            break;
    }

    inst->fn = &mem_vtable;
    return inst;
}
