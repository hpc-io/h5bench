#ifndef H5DEEPMEM_H
#define H5DEEPMEM_H

#include <sys/time.h> // time
#include <stdlib.h>
#include <semaphore.h>
#include <stdio.h>
#include <errno.h>

typedef struct h5deepmem_functions_t h5deepmem_functions;
typedef struct h5deepmem_t h5deepmem;

struct h5deepmem_t
{
  h5deepmem_functions const* fn; // Object-Oriented Programming in C
};

h5deepmem* h5deepmem_new();

struct h5deepmem_functions_t {
  void (*free)(h5deepmem*);
};

#endif // H5DEEPMEM_H
