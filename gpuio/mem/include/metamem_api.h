#ifndef METAMEM_H
#define METAMEM_H

#include "mem.h"
#include "xfer.h"

typedef struct metamem_functions_t metamem_functions;
typedef struct metamem_t metamem;

struct metamem_t
{
  mem *device_ptr;
  mem *host_ptr;
  xfer *xfer_inst;

  metamem_api_t device_api;
  metamem_functions const* fn; // Object-Oriented Programming in C
};

metamem* metamem_init(metamem_api_t device_api);
void metamem_shutdown(metamem*);

struct metamem_functions_t {
  metamem* (*alloc)(metamem*, size_t, size_t, mem_type_t, mem_type_t);
  void (*copy)(metamem*, xfer_direction_t);
  void (*free)(metamem*);
};

#endif // METAMEM_H
