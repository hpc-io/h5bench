#ifndef H5MEM_H
#define H5MEM_H

#include <stdlib.h>
#include "h5deepmem_globals.h"

typedef struct h5mem_functions_t h5mem_functions;
typedef struct h5mem_t h5mem;

struct h5mem_t
{
  size_t nitems; // number of elements to be allocated
  size_t size; // size of elements
  void *ptr; // memory address
  h5mem_type_t mem_type; // memory allocation type
  h5mem_functions const* fn; // Object-Oriented Programming in C
};

h5mem* h5mem_alloc(size_t nitems, size_t size, h5deepmem_api_t device_api, h5mem_type_t mem_type);

struct h5mem_functions_t {
  void (*free)(h5mem*, h5deepmem_api_t);
};

#endif // H5MEM_H
