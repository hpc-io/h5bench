#ifndef H5DEEPMEM_H
#define H5DEEPMEM_H

#include <stdlib.h>
#include "h5mem.h"
#include "h5xfer.h"

typedef struct h5deepmem_functions_t h5deepmem_functions;
typedef struct h5deepmem_t           h5deepmem;

struct h5deepmem_t {
    h5mem * device_ptr;
    h5mem * host_ptr;
    h5xfer *xfer_inst;

    h5deepmem_api_t            device_api;
    h5deepmem_functions const *fn; // Object-Oriented Programming in C
};

h5deepmem *h5deepmem_alloc(size_t nitems, size_t size, h5deepmem_api_t device_api, h5mem_type_t host_mem_type,
                           h5mem_type_t device_mem_type);

struct h5deepmem_functions_t {
    void (*copy)(h5deepmem *, h5xfer_direction_t);
    void (*free)(h5deepmem *);
};

#endif // H5DEEPMEM_H
