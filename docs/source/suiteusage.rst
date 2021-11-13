Benchmark Suite Usage
=====================================

-------------------------------------
h5bench_patterns benchmark
-------------------------------------

Major refactoring is in progress, this document may be out of date. All of h5bench_write, h5bench_read, h5bench_write_unlimited, h5bench_overwrite and h5bench_append take config and data file path as command line arguments.

.. code-block:: bash

	./h5bench_write my_config.cfg my_data.h5
	./h5bench_read my_config.cfg my_data.h5
	./h5bench_write_unlimited my_config.cfg my_data.h5
	./h5bench_overwrite my_config.cfg my_data.h5
	./h5bench_append my_config.cfg my_data.h5

This set of benchmarks contains an I/O kernel developed based on a particle physics simulation's I/O pattern (VPIC-IO for writing data in a HDF5 file) and on a big data clustering algorithm (BDCATS-IO for reading the HDF5 file VPIC-IO wrote).


Settings in the Configuration File
-------------------------------------

The h5bench_write, h5bench_read, h5bench_write_unlimited, h5bench_overwrite and h5bench_appendtake parameters in a plain text config file. The content format is **strict**. Unsupported formats :

* Blank/empty lines, including ending lines.
* Comment symbol(#) follows value immediately:
	* TIMESTEPS=5# Not supported
	* TIMESTEPS=5 #This is supported
	* TIMESTEPS=5 # This is supported
* Blank space in assignment
	* TIMESTEPS=5 # This is supported
	* TIMESTEPS = 5 # Not supported
	* TIMESTEPS =5 # Not supported
	* TIMESTEPS= 5 # Not supported


A template of config file can be found basic_io/sample_config/template.cfg:

.. code-block:: none

	#========================================================
	#   General settings
	NUM_PARTICLES=16 M #16 K/G
	TIMESTEPS=5
	EMULATED_COMPUTE_TIME_PER_TIMESTEP=1 s #1 ms, 1 min
	#========================================================
	#   Benchmark data dimensionality
	NUM_DIMS=1
	DIM_1=16777216
	DIM_2=1
	DIM_3=1
	#========================================================
	#   IO pattern settings
	IO_OPERATION=READ
	#IO_OPERATION=WRITE
	MEM_PATTERN=CONTIG
	# INTERLEAVED STRIDED
	FILE_PATTERN=CONTIG # STRIDED
	#========================================================
	#   Options for IO_OPERATION=READ
	READ_OPTION=FULL # PARTIAL STRIDED
	TO_READ_NUM_PARTICLES=4 M
	#========================================================
	#   Strided access parameters, required for strided access
	#STRIDE_SIZE=
	#BLOCK_SIZE=
	#BLOCK_CNT=
	#========================================================
	# Collective data/metadata settings
	#COLLECTIVE_DATA=NO #Optional, default for NO.
	#COLLECTIVE_METADATA=NO #Optional, default for NO.
	#========================================================
	#    Compression, optional, default is NO.
	#COMPRESS=NO
	#CHUNK_DIM_1=1
	#CHUNK_DIM_2=1
	#CHUNK_DIM_3=1
	#========================================================
	#    Async related settings
	DELAYED_CLOSE_TIMESTEPS=2
	IO_MEM_LIMIT=5000 K
	ASYNC_MODE=EXP #EXP IMP NON 
	#========================================================
	#    Output performance results to a CSV file
	#CSV_FILE=perf_write_1d.csv
	#
	#FILE_PER_PROC=

General Settings
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* **IO_OPERATION**: required, chose from **READ**, **WRITE**, **WRITE_UNLIMITED**, **OVERWRITE** and **APPEND**
* **MEM_PATTERN**: required, chose from **CONTIG, INTERLEAVED** and **STRIDED**
* **FILE_PATTERN**: required, chose from **CONTIG**, and **STRIDED**
* **NUM_PARTICLES**: required, the number of particles that each rank needs to process, can be in exact numbers (12345) or in units (format like 16 K, 128 M and 256 G are supported, format like 16K, 128M, 256G is NOT supported).
* TIMESTEPS: required, the number of iterations
* **EMULATED_COMPUTE_TIME_PER_TIMESTEP**: required, must be with units (eg,. 10 s, 100 ms or 5000 us). In each iteration, the same amount of data will be written and the file size will increase correspondingly. After each iteration, the program sleeps for $EMULATED_COMPUTE_TIME_PER_TIMESTEP time to emulate the application computation.
* **NUM_DIMS**: required, the number of dimensions, valid values are 1, 2 and 3.
* **DIM_1, DIM_2, and DIM_3**: required, the dimensionality of the source data. Always set these parameters in ascending order, and set unused dimensions to 1, and remember that NUM_PARTICLES == DIM_1 * DIM_2 * DIM_3 MUST hold. For example, DIM_1=1024, DIM_2=256, DIM_3=1 is a valid setting for a 2D array when NUM_PARTICLES=262144 or NUM_PARTICLES=256 K, because 10242561 = 262144, which is 256 K.


Example for using multi-dimensional array data
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* Using 2D as the example, 3D cases are similar, the file is generated with with 4 ranks, each rank write 8M elements, organized in a 4096 * 2048 array, in total it forms a (4 * 4096) * 2048 2D array. The file should be around 1GB.

Dimensionality part of the Config file:

.. code-block:: bash

	NUM_DIMS=2
	DIM_1=4096
	DIM_2=2048
	DIM_3=64 	# Note: extra dimensions than specified by NUM_DIMS are ignored


Addtional Settings for READ (h5bench_read)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* **READ_OPTION**: required for IO_OPERATION=READ, not allowed for IO_OPERATION=WRITE.

	* FULL: read the whole file
	* PARTIAL: read the first $TO_READ_NUM_PARTICLES particles
	* STRIDED: read in streded pattern

* **TO_READ_NUM_PARTICLES**: required, the number for particles attempt to read.


Async Related Settings
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* **ASYNC_MODE**: optional, the default is NON.

	* NON: the benchmark will run in synchronous mode.
	* EXP: enable the asynchronous mode. An installed async VOL connector and coresponding environment variables are required.

* **IO_MEM_LIMIT**: optional, the default is 0, requires **ASYNC_MODE=EXP**, only works in asynchronous mode. This is the memory threshold used to determine when to actually execute the IO operations. The actual IO operations (data read/write) will not be executed until the timesteps associated memory reachs the threshold, or the application run to the end.

* **DELAYED_CLOSE_TIMESTEPS**: optional, the default is 0. The groups and datasets associated to to the timesteps will be closed later for potential caching.



Compression Settings
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* **COMPRESS**: YES or NO, optional. Only applicable for WRITE(h5bench_write), has no effect for READ. Used to enable compression, when enabled, chunk dimensions(CHUNK_DIM_1, CHUNK_DIM_2, CHUNK_DIM_3) are required. To enable parallel compression feature for VPIC, add following section to the config file, and make sure chunk dimension settings are compatible with the data dimensions: they must have the same rank of dimensions (eg,. 2D array dataset needs 2D chunk dimensions), and chunk dimension size cannot be greater than data dimension size.


.. code-block:: bash

	COMPRESS=YES 	# to enable parallel compression( chunking)
	CHUNK_DIM_1=512 # chunk dimensions
	CHUNK_DIM_2=256
	CHUNK_DIM_3=1 	# extra chunk dimension take no effects

.. attention::

	There is a known bug on HDF5 parallel compression that could cause the system run out of memory when the chunk amount is large (large number of particle and very small chunk sizes). On Cori Hasswell nodes, the setting of 16M particles per rank, 8 nodes (total 256 ranks), 64 * 64 chunk size will crash the system by runing out of memory, on single nodes the minimal chunk size is 4 * 4.


Collective Operation Settings
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* **COLLECTIVE_DATA**: optional, set to "YES" for collective data operations, otherwise and default (not set) cases for independent operations.

* **COLLECTIVE_METADATA**: optional, set to "YES" for collective metadata operations, otherwise and default (not set) cases for independent operations.



Other Settings
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* **CSV_FILE**: optional CSV file output name, performance results will be print to the file and the standard output as well.



Supported Patterns
-------------------------------------


.. attention:: 

	Not every pattern combination is covered, supported benchmark parameter settings are listed below.

Supported Write Patterns (h5bench_write): IO_OPERATION=WRITE
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The I/O patterns include array of structures (AOS) and structure of arrays (SOA) in memory as well as in file. The array dimensions are 1D, 2D, and 3D for the write benchmark. This defines the write access pattern, including CONTIG (contiguous), INTERLEAVED and STRIDED” for the source (the data layout in the memory) and the destination (the data layout in the resulting file). For example, MEM_PATTERN=CONTIG and FILE_PATTERN=INTERLEAVED is a write pattern where the in-memory data layout is contiguous (see the implementation of prepare_data_contig_2D() for details) and file data layout is interleaved by due to its’ compound data structure (see the implementation of data_write_contig_to_interleaved () for details).


4 patterns for both 1D and 2D array write (NUM_DIMS=1 or NUM_DIMS=2)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: none

	MEM_PATTERN=CONTIG, FILE_PATTERN=CONTIG
	MEM_PATTERN=CONTIG, FILE_PATTERN=INTERLEAVED
	MEM_PATTERN=INTERLEAVED, FILE_PATTERN=CONTIG
	MEM_PATTERN=INTERLEAVED, FILE_PATTERN=INTERLEAVED


1 pattern for 3D array (NUM_DIMS=3)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: none

	MEM_PATTERN=CONTIG, FILE_PATTERN=CONTIG


1 strided pattern for 1D array (NUM_DIMS=1)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: none

	MEM_PATTERN=CONTIG, FILE_PATTERN=STRIDED



Supported Read Patterns (h5bench_read): IO_OPERATION=READ
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

1 pattern for 1D, 2D and 3D read (NUM_DIMS=1 or NUM_DIMS=2)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: none

	MEM_PATTERN=CONTIG, FILE_PATTERN=CONTIG, READ_OPTION=FULL, contiguously read through the whole data file.

2 patterns for 1D read
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: none

	MEM_PATTERN=CONTIG, FILE_PATTERN=CONTIG, READ_OPTION=PARTIAL, contiguously read the first $TO_READ_NUM_PARTICLES elements.

	MEM_PATTERN=CONTIG, FILE_PATTERN=STRIDED, READ_OPTION=STRIDED



Supported Unlimited Write Patterns (h5bench_write_unlimited): IO_OPERATION=WRITE. This is a bootstrap for Append.
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


1 pattern for 1D, 2D and 3D write (NUM_DIMS=1 or NUM_DIMS=2 or NUM_DIMS=3)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: none

	MEM_PATTERN=CONTIG, FILE_PATTERN=CONTIG, COMPRESS=YES



Supported Append Patterns (h5bench_append): IO_OPERATION=APPEND. Read the dataset and extend the dataset by doubling it at the end of the first dimension.
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


1 pattern for 1D, 2D and 3D write (NUM_DIMS=1 or NUM_DIMS=2 or NUM_DIMS=3)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: none

	MEM_PATTERN=CONTIG, FILE_PATTERN=CONTIG, COLLECTIVE_DATA=YES


Supported Overwrite Patterns (h5bench_overwrite): IO_OPERATION=OVERWRITE. Overwrite every dataset in a given file. Dimension parameters have to be consistent with the given file's dimension parameters.
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


1 pattern for 1D overwrite (NUM_DIMS=1 or NUM_DIMS=2 or NUM_DIMS=3)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: none

	MEM_PATTERN=CONTIG, FILE_PATTERN=CONTIG




Sample Settings
------------------------------------------------------------------------

The following setting reads 2048 particles from 128 blocks in total, each block consists of the top 16 from every 64 elements. See HDF5 documentation for details of using strided access.


.. code-block:: none

	#   General settings
	NUM_PARTICLES=16 M
	TIMESTEPS=5
	MULATED_COMPUTE_TIME_PER_TIMESTEP=1 s
	#========================================================
	#   Benchmark data dimensionality
	NUM_DIMS=1
	DIM_1=16777216
	DIM_2=1
	DIM_3=1
	#========================================================
	#   IO pattern settings
	IO_OPERATION=READ
	MEM_PATTERN=CONTIG
	FILE_PATTERN=CONTIG
	#========================================================
	#    Options for IO_OPERATION=READ
	READ_OPTION=PARTIAL # FULL PARTIAL STRIDED
	TO_READ_NUM_PARTICLES=2048
	#========================================================
	#    Strided access parameters
	STRIDE_SIZE=64
	BLOCK_SIZE=16
	BLOCK_CNT=128

For more examples, please find the config files and template.cfg in basic_io/sample_config/ directory.



To Run the h5bench_write, h5bench_read, h5bench_write_unlimited, h5bench_append and h5bench_overwrite
------------------------------------------------------------------------

All of them use the same command line arguments:

Single process run:

.. code-block:: bash

	./h5bench_write sample_write_cc1d_es1.cfg my_data.h5


Parallel run (replace mpirun with your system provided command, for example, srun on Cori/NERSC and jsrun on Summit/OLCF):

.. code-block:: bash
	
	mpirun -n 2 ./h5bench_write sample_write_cc1d_es1.cfg output_file


In Cori/NERSC or similar platforms that use Cray-MPICH library, if you encouter a failed assertion regarding support for MPI_THREAD_MULTIPLE you should define the following environment variable:

.. code-block:: bash

	export MPICH_MAX_THREAD_SAFETY="multiple"


Argobots in MacOS
------------------------------------------------------------------------

If you're trying to run the benchmark in a MacOS and are getting segmentation fault (from ABT_thread_create), please try to set the following environment variable:

.. code-block:: bash

	ABT_THREAD_STACKSIZE=100000 ./h5bench_write sample_write_cc1d_es1.cfg my_data.h5


Understanding the Output
------------------------------------------------------------------------

The metadata and raw data operations are timed separately, and overserved time and rate are based on the total time.

Sample output of h5bench_write:

.. code-block:: none

	==================  Performance results  =================
	Total emulated compute time 4000 ms
	Total write size = 2560 MB
	Data preparation time = 739 ms
	Raw write time = 1.012 sec
	Metadata time = 284.990 ms
	H5Fcreate() takes 4.009 ms
	H5Fflush() takes 14.575 ms
	H5Fclose() takes 4.290 ms
	Observed completion time = 6.138 sec
	Raw write rate = 2528.860 MB/sec
	Observed write rate = 1197.592 MB/sec

Sample output of h5bench_read:

.. code-block:: none

	=================  Performance results  =================
	Total emulated compute time = 4 sec
	Total read size = 2560 MB
	Metadata time = 17.523 ms
	Raw read time = 1.201 sec
	Observed read completion time = 5.088 sec
	Raw read rate = 2132.200 MB/sec
	Observed read rate = 2353.605225 MB/sec

Sample output of h5bench_write_unlimited:

.. code-block:: none

	==================  Performance results  =================
	Total emulated compute time 4000 ms
	Total write size = 2 MB
	Raw write time = 6.464 sec
	Metadata time = 994.788 ms
	H5Fcreate() takes 6.111 ms
	H5Fflush() takes 203.762 ms
	H5Fclose() takes 39.520 ms
	Observed completion time = 11.710 sec
	Sync Raw write rate = 0.309 MB/sec 
	Sync Observed write rate = 0.259 MB/sec

Sample output of h5bench_append:

.. code-block:: none

	=================  Performance results  =================
	Total emulated compute time = 4000 ms
	Total modify size = 2 MB
	Raw modify time = 7.466 sec 
	Metadata time = 18.080 ms
	Observed modify completion time = 11.507 sec
	Sync Raw modify rate = 0.268 MB/sec 
	Sync Observed modify rate = 0.266425 MB/sec

Sample output of h5bench_overwrite:

.. code-block:: none

	=================  Performance results  =================
	Total emulated compute time = 4000 ms
	Total modify size = 24 MB
	Raw modify time = 0.008 sec 
	Metadata time = 4.497 ms
	Observed modify completion time = 4.026 sec
	Sync Raw modify rate = 3004.507 MB/sec 
	Sync Observed modify rate = 915.646118 MB/sec


------------------------------------------------------------
h5bench_exerciser
------------------------------------------------------------

We modified this benchmark slightly so to be able to specify a file location that is writable. Except for the first argument $write_file_prefix, it's identical to the original one. Detailed README can be found int source code directory, the original can be found here https://xgitlab.cels.anl.gov/ExaHDF5/BuildAndTest/-/blob/master/Exerciser/README.md

Example run:

.. code-block:: bash

	mpirun -n 8 ./h5bench_exerciser $write_file_prefix -numdims 2 --minels 8 8 --nsizes 3 --bufmult 2 --dimranks 8 4


------------------------------------------------------------
The Metadata Stress Test: h5bench_hdf5_iotest
------------------------------------------------------------

This is the same benchmark as it's originally found at https://github.com/HDFGroup/hdf5-iotest. We modified this benchmark slightly so to be able to specify the config file location, everything else remains untouched.

Example run:

.. code-block:: bash

	mpirun -n 4 ./h5bench_hdf5_iotest hdf5_iotest.ini


------------------------------------------------------------
Streaming operation benchmark: h5bench_vl_stream_hl
------------------------------------------------------------


This benchmark tests the performance of append operation. It supports two types of appends, FIXED and VLEN, represents fixed length data and variable length data respectively. Note: This benchmark doesn't run in parallel mode.



To run benchmarks
-------------------------------------------------------------


.. code-block:: bash

	./h5bench_vl_stream_hl write_file_path FIXED/VLEN num_ops

Example runs:

.. code-block:: bash

	./h5bench_vl_stream_hl here.dat FIXED 1000
	./h5bench_vl_stream_hl here.dat VLEN 1000

