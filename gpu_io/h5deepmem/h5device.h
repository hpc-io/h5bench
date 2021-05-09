#ifndef H5DEVICE_H
#define H5DEVICE_H

#include <stdlib.h>

typedef struct h5device_functions_t h5device_functions;
typedef struct h5device_t h5device;

struct h5device_t
{
  h5device_functions const* fn; // Object-Oriented Programming in C
};

h5device* h5device_new();

struct h5device_functions_t {
  void (*free)(h5device*);
};

#endif // H5DEVICE_H
