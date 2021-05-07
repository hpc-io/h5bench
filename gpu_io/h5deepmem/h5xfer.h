#ifndef H5XFER_H
#define H5XFER_H

#include <sys/time.h> // time
#include <stdlib.h>
#include <semaphore.h>
#include <stdio.h>
#include <errno.h>

typedef enum {
  EXPLICIT,
  PREFETCH,
  GPU_DIRECT
} h5xfer_type_t;

typedef struct h5xfer_functions_t h5xfer_functions;
typedef struct h5xfer_t h5xfer;

struct h5xfer_t
{
  h5xfer_type_t xfer_type; // xfer type
  h5xfer_functions const* fn; // Object-Oriented Programming in C
};

h5xfer* h5xfer_new();

struct h5xfer_functions_t {
  void (*copy)(h5xfer*);
  void (*free)(h5xfer*);
};

#endif // H5XFER_H
