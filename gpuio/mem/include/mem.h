#ifndef MEM_H
#define MEM_H

#include "metamem_pch.h"
#include <stddef.h>

typedef struct mem_functions_t mem_functions;
typedef struct mem_t mem;

struct mem_t
{
  size_t nitems; // number of elements to be allocated
  size_t size; // size of elements
  void *ptr; // memory address
  mem_type_t mem_type; // memory allocation type
  mem_functions const* fn; // Object-Oriented Programming in C
};

mem* mem_alloc(size_t nitems, size_t size, metamem_api_t device_api, mem_type_t mem_type);

struct mem_functions_t {
  void (*free)(mem*, metamem_api_t);
};

#endif // MEM_H
