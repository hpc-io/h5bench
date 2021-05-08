#ifndef H5XFER_H
#define H5XFER_H

#include <stdlib.h>
#include "h5deepmem_globals.h"

typedef enum {
  H2D,
  D2H
} h5xfer_direction_t;

// typedef enum {
//   EXPLICIT,
//   PREFETCH,
//   GPU_DIRECT
// } h5xfer_type_t;

typedef struct h5xfer_functions_t h5xfer_functions;
typedef struct h5xfer_t h5xfer;

struct h5xfer_t
{
  // h5xfer_type_t xfer_type; // xfer type
  h5xfer_functions const* fn; // Object-Oriented Programming in C
};

h5xfer* h5xfer_new();

struct h5xfer_functions_t {
  void (*copy)(h5xfer*, void *, void *, size_t size, h5deepmem_api_t, h5mem_type_t, h5mem_type_t);
  void (*free)(h5xfer*);
};

#endif // H5XFER_H
