# HDF5 Exerciser Benchmark

**Authors:**

- Richard J. Zamora (rzamora@anl.gov)
- Paul Coffman (pcoffman@anl.gov)
- Venkatram Vishwanath (venkat@anl.gov)

**Updates:**

- December 13th 2018 (Version 2.0)

**NOTE: For more-detailed instructions of how to build and run the exerciser code on specific machines (at ALCF), see the `Exerciser/BGQ/VESTA_XL/README.md` and `Exerciser/BGQ/THETA/README.md` directories of this repository. Those README files also include instructions for building the CCIO and `develop` versions of HDF5 for use with this benchmark.**

## Exerciser Overview

The **HDF5 Exerciser Benchmark** creates an HDF5 use case with some ideas/code borrowed from other benchmarks (namely `IOR`, `VPICIO` and `FLASHIO`). Currently, the algorithm does the following in parallel over all MPI ranks:

- For each rank, a local data buffer (with dimensions given by `numDims`) is initialized with `minNEls` double-precision elements in each dimension.
- If the `--derivedtype` flag is used, a second local dataset is also specified with a *derived* data type a-signed to each element.
- For a given number of iterations (hardcoded as `NUM_ITERATIONS`):

	- Open a file, create a top group, set the `MPI-IO` transfer property, and (optionally) add a simple attribute string to the top group
	- Create *memory* and *file* dataspaces with hyperslab selections for simple rank-ordered offsets into the file. The `-rshift` option can be used to specify the number of rank positions to shift the write position in the file (the read will be shifted *twice* this amount to avoid client-side caching effects
	- Write the data and close the file
	- Open the file, read in the data, and check correctness (if dataset is small enough)
	- Close the dataset (but not the file)
	- If the second (derived-type) data set is specified: (1) create a derived type, (2) open a new data set with the same number of elements and dimension, (3) write the data and (4) close everything.
- Each dimension of `curNEls` is then multiplied by each dimension of `bufMult`, and the previous steps (the loop over `NUM_ITERATIONS`) are repeated. This outer loop over local buffer sizes is repeated a total of `nsizes` times.

### Command-line Arguments (Options)

#### Required
- **``--numdims <x>``:** Dimension of the datasets to write to the hdf5 file
- **``--minels <x> ... <x>``:** Min number of double elements to write in each dim of the dataset (one value for each dimension)

#### *Optional*
- **``--nsizes <x>``:** How many buffer sizes to use (Code will start with ``minbuf`` and loop through ``nsizes`` iterations, with the buffer size multiplied by ``bufmult`` in each dim, for each iteration)
- ``--bufmult <x> ... <x>``: Constant, for each dimension, used to multiply the buffer [default: *2* *2* ... ]
- ``--metacoll``:  Whether to set meta data collective usage [default: *False*]
- ``--derivedtype``: Whether to create a second data set containing a derived type [default: *False*]
- ``--addattr``: Whether to add attributes to group 1 [default: *False*]
- ``--indepio``: Whether to use independant I/O (not MPI-IO) [default: *False*]
- ``--keepfile``: Whether to keep the file around after the program ends for futher analysis, otherwise deletes it [default: *False*]
- ``--usechunked``:  Whether to *chunk* the data when reading/writing [default: *False*]
- ``--maxcheck <x>``: Maximum buffer size (in bytes) to validate.  Note that **all** buffers will be vaidated if this option is **not** set by this command-line argument [default: *Inf*]
- ``--memblock <x>``: Define the block size to use in the local memory buffer (local buffer is always 1D for now, *Note*: This currently applies to the 'double' dataset only) [default: *local buffer size*]
- ``--memstride <x>``: Define the stride of the local memory buffer (local buffer is always 1D for now, *Note*: This currently applies to the 'double' dataset only) [default: *local buffer size*]
- ``--fileblocks <x>..<x>``(one value for each dimension): block sizes to use in the file for each dataset dimension (*Note*: This currently applies to the 'double' dataset only) [default: *1* ... *1*]
- ``--filestrides <x>..<x>``(one value for each dimension): stride dist. to use in the file for each dataset dimension (*Note*: This currently applies to the 'double' dataset only) [default: *1* ... *1*]

The exerciser also allows the MPI decomposition to be explicitly defined:

- ``--dimranks <x>...<x>``: (one value for each dimension) mpi-rank division in each dimension. Note that, if not set, decomposition will be in 1st dimension only

### Exerciser Basics

In the simplest case, the Exerciser code will simply write and then read an n-dimensional double-precision dataset in parallel (with all the necessary HDF5 steps in between). At a minimum, the user must specify the **number** of dimensions to use for this dataset (using the `--numdims` flag), and the **size** of each dimension (using the `--minels` flag). By default, the maximum number of dimensions allowed by the code is set by `MAX_DIM` (currently 4, but can be modified easily).  Note that the user is specifying the number of elements to use in each dimension with `--minels`.  Therefore, the local buffer size is the product of the dimension sizes and `sizeof(double)` (and the global dataset in the file is a product of the total MPI ranks and the local buffer size). As illustrated in **Fig. 1**, the mapping of ranks to hyper-slabs in the global dataset can be specified with the `--dimranks` flag (here, *Example 1* is the default decomposition, while *Example 2* corresponds to: `--dimranks 2 2`).  This flag simply allows the user to list the number of spatial decompositions in each dimension of the global dataset, and requires that the product of the input to be equal to the total number of MPI ranks.

**Fig. 1 - Illustration of different local-to-global dataset mapping options:**

![alt text](./dimranks.png "Illustration of local-to-global dataset mapping.")

If the user wants to loop through a range of buffer sizes, the `--nsizes` flag can be used to specify how many sizes measure, and the `--bufmult` flag can be used to specify the multiplication factor for each dimension between each loop. For example, if the user wanted to test 64x64, 128x128, and 256x256-element local datasets on 32 ranks, they could use the following command to run the code:

```
mpirun -np 32 ./hdf5Exerciser --numdims 2 --minels 8 8 --nsizes 3 --bufmult 2 --dimranks 8 4
```

When executed for a single local-buffer size (default), the Exerciser output will look something like this:

```
useMetaDataCollectives: 0 addDerivedTypeDataset: 0 addAttributes: 0 useIndependentIO: 0 numDims: 1 useChunked: 0 rankShift: 4096
Metric      Bufsize   H5DWrite    RawWrBDWTH    H5Dread    RawRdBDWTH    Dataset      Group  Attribute    H5Fopen   H5Fclose   H5Fflush OtherClose
Min           32768   0.134616   3058.154823   0.191049   2534.613015   0.361010   0.551608   0.000001   0.224550   0.127877   0.210821   0.000755
Med           32768   0.143874   3554.180478   0.191684   2670.829718   0.379858   0.612309   0.000001   0.236735   0.132450   0.228889   0.000761
Max           32768   0.167421   3803.418460   0.202003   2679.939135   0.405620   0.679779   0.000002   0.268622   0.138463   0.270188   0.000785
Avg           32768   0.146435   3506.598052   0.192068   2666.021346   0.379799   0.616157   0.000001   0.237219   0.132410   0.233730   0.000763
Std           32768   0.008055    185.366133   0.002090     27.665058   0.010248   0.026048   0.000000   0.008915   0.002650   0.017362   0.000006
```

Using `NUM_ITERATIONS` samples for each local buffer size (`Bufsize`), the minimum, median, maximum, average, and standard deviation of all metrics will be reported in distinct rows of the output. The `Bufsize` values are reported in **bytes**, while the `RawWrBDWTH` and `RawRdBDWTH` are in **MB/s**, and all other metrics are in **seconds**.


## Building Exerciser
H5bench's make process builds the h5bench_exerciser. 

In case, Exerciser needs to be built separately, given the path to a parallel HDF5 installation, building it is straightforward. The following Makefile can be used as a reference:

```
default: hdf5Exerciser

HDF5_INSTALL_DIR=/Users/rzamora/hdf5-install

exerciser.o: exerciser.c
        mpicc  -c -g -DMETACOLOK -I${HDF5_INSTALL_DIR}/include  exerciser.c -o exerciser.o

hdf5Exerciser: exerciser.o
        mpicc exerciser.o -o hdf5Exerciser  -L${HDF5_INSTALL_DIR}/lib -lhdf5 -lz

clean:
        rm -f exerciser.o
        rm -f hdf5Exerciser
```

For more-detailed instructions of how to build and run both HDF5 and the exerciser on specific machines (at ALCF), see the `Exerciser/BGQ/VESTA_XL` and `Exerciser/BGQ/THETA` directories of this repository.
