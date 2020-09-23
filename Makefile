HDF5_DIR = /Users/tonglin/nersc_dev_sync/hdf5_build/hdf5

COMMON_DIR = ./commons

CC = mpicc
BASIC_IO_DIR = ./basic_io
DEBUG = -g -O0
CFLAGS = $(DEBUG) -I$(HDF5_DIR)/include -I$(COMMON_DIR)
LDFLAGS = -L$(HDF5_DIR)/lib -lhdf5 -L$(COMMON_DIR) -lh5bench_util
UTIL_SRC = $(COMMON_DIR)/h5bench_util.c
LIB = $(COMMON_DIR)/libh5bench_util.so
SRC_VPIC = $(BASIC_IO_DIR)/h5bench_vpicio.c
SRC_BDCATS = $(BASIC_IO_DIR)/h5bench_bdcatsio.c
BIN_VPIC = $(BASIC_IO_DIR)/h5bench_vpicio
BIN_BDCATS = $(BASIC_IO_DIR)/h5bench_bdcatsio

all: #common vpic bdcats
#common 
	$(CC) $(CFLAGS) $(UTIL_SRC) -shared -fPIC -o $(LIB)
#vpic
	$(CC) $(CFLAGS) $(SRC_VPIC) $(LDFLAGS) -o $(BIN_VPIC)
#bdcats
	$(CC) $(CFLAGS) $(SRC_BDCATS) $(LDFLAGS) -o $(BIN_BDCATS)

.PHONY: clean
clean:
	rm -f $(BIN_VPIC) $(BIN_BDCATS)
