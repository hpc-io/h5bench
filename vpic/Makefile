HDF5_DIR = /Users/tonglin/nersc_dev_sync/hdf5_build/hdf5
CC = mpicc
# CC = gcc-9
# CC = mpicc

DEBUG = -g -O0
CFLAGS = -I$(HDF5_DIR)/include
LDFLAGS = -L$(HDF5_DIR)/lib -lhdf5

SRC = vpicio_uni_h5.c
BIN = vpicio_uni_h5

all: $(EXE)
	$(CC) $(CFLAGS) $(SRC) $(LDFLAGS) -o $(BIN)

check:
	echo "test run..."
	mpirun -n 2 ./vpicio_uni_h5 test_data.h5 5 1 2
	rm test_data.h5
.PHONY: clean
clean:
	rm -rf $(BIN) 
