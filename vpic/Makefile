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
SRC = vpicio_uni_h5.c
BIN = vpicio_uni_h5

all: common bin
common:	
	$(CC) $(CFLAGS) $(UTIL_SRC) -shared -fPIC -o $(LIB)
bin:
	$(CC) $(CFLAGS) $(SRC) $(LDFLAGS) -o $(BIN)

check:
	echo "test run..."
	mpirun -n 2 ./vpicio_uni_h5 test_data.h5 5 1 2
	rm test_data.h5
.PHONY: clean
clean:
	rm -rf $(BIN) 
