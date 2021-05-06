#ifndef H5_XFER_H
#define H5_XFER_H

#include <sys/time.h> // time
#include <stdlib.h>
#include <semaphore.h>
#include <stdio.h>
#include <errno.h>

// TODO: ???
typedef enum {EXPLICIT, IMPLICIT, GDS} h5_xfer_type_t;

typedef struct h5_xfer_functions_t h5_xfer_functions;
typedef struct h5_xfer_t h5_xfer;

struct h5_xfer_t
{
  h5_xfer_type_t xfer_type; // xfer type
  h5_xfer_functions const* fn; // Object-Oriented Programming in C
};

// h5_xfer* h5_xfer_open(char *shm_path, char *sem_path);
h5_xfer* h5_xfer_new(char *sem_path, char *buffer_path, unsigned io_size);

struct h5_xfer_functions_t {
  // float (*query)(h5_xfer *);
  void (*copy)(h5_xfer*);
  void (*free)(h5_xfer*);
};

#endif // H5_XFER_H
