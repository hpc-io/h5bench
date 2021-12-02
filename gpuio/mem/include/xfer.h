#ifndef XFER_H
#define XFER_H

#include "metamem_pch.h"
#include <stddef.h>

typedef enum {
  H2D,
  D2H
} xfer_direction_t;

// typedef enum {
//   EXPLICIT,
//   PREFETCH,
//   GPU_DIRECT
// } xfer_type_t;

typedef struct xfer_functions_t xfer_functions;
typedef struct xfer_t xfer;

struct xfer_t
{
  // xfer_type_t xfer_type; // xfer type
  xfer_functions const* fn; // Object-Oriented Programming in C
};

xfer* xfer_new();

struct xfer_functions_t {
  void (*copy)(xfer*, void *, void *, size_t size, metamem_api_t, mem_type_t, mem_type_t);
  void (*free)(xfer*);
};

#endif // XFER_H
