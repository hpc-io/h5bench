HDF5_DIR = /Users/tonglin/nersc_dev_sync/hdf5_build/hdf5
H5BENCH_DIR = /Users/tonglin/nersc_dev_sync/h5bench
COMMON_DIR = $(H5BENCH_DIR)/commons
CC = mpicc
# CC = gcc-9
# CC = mpicc

DEBUG = -g -O0
CFLAGS = -I$(HDF5_DIR)/include -I$(COMMON_DIR)
LDFLAGS = -L$(HDF5_DIR)/lib -lhdf5 -L$(COMMON_DIR) -lh5bench_util
UTIL_SRC = $(COMMON_DIR)/h5bench_util.c
LIB = $(COMMON_DIR)/libh5bench_util.so
SRC = bdcatsio.c
BIN = bdcatsio

all: common bin
common:	
	$(CC) $(CFLAGS) $(UTIL_SRC) -shared -fPIC -o $(LIB)
bin:
	$(CC) $(CFLAGS) $(SRC) $(LDFLAGS) -o $(BIN)

.PHONY: clean
clean:
	rm -rf $(BIN) 
