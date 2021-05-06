#ifndef H5_MEM_H
#define H5_MEM_H

#include <sys/time.h> // time
#include <stdlib.h>
#include <semaphore.h>
#include <stdio.h>
#include <errno.h>

typedef enum {CPU_PAGED, CPU_PINNED, CPU_GPU_MANAGED, GPU_MEMORY} h5_mem_type_t;

typedef struct h5_mem_functions_t h5_mem_functions;
typedef struct h5_mem_t h5_mem;

struct h5_mem_t
{
  size_t nitems; // number of elements to be allocated
  size_t size; // size of elements
  void *ptr; // memory address
  h5_mem_type_t mem_type; // memory allocation type
  h5_mem_functions const* fn; // Object-Oriented Programming in C
};

h5_mem* h5_mem_alloc(size_t nitems, size_t size);

struct h5_mem_functions_t {
  // float (*query)(h5_mem *);
  void (*free)(h5_mem*);
};

#endif // H5_MEM_H
