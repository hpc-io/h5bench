#ifndef DEVICE_H
#define DEVICE_H

typedef struct device_functions_t device_functions;
typedef struct device_t device;

struct device_t
{
  device_functions const* fn; // Object-Oriented Programming in C
};

device* device_new();

struct device_functions_t {
  void (*free)(device*);
};

#endif // DEVICE_H
