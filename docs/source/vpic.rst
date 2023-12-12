Read / Write
================

This set of benchmarks contains an I/O kernel developed based on a particle physics simulation's I/O pattern (VPIC-IO for writing data in a HDF5 file) and on a big data clustering algorithm (BDCATS-IO for reading the HDF5 file VPIC-IO wrote).

Configuration
-------------

You can configure the ``h5bench_write`` and ``h5bench_read`` benchmarks with the following options. Notice that if you use the ``configuration.json`` approach to define the runs for ``h5bench``, we will automatically generate the final configuration file based on the options you provide in the JSON file. For standalone usage of this benchmark, you can check the input format at the end of this document and refer to its documentation.

======================================= ==========================================================
**Parameter**                           **Description**                                         
======================================= ==========================================================
``MEM_PATTERN``                         Options: ``CONTIG``, ``INTERLEAVED``, and ``STRIDED``   
``FILE_PATTERN``                        Options: ``CONTIG`` and ``STRIDED``                     
``TIMESTEPS``                           The number of iterations                                
``EMULATED_COMPUTE_TIME_PER_TIMESTEP``  Sleeps after each iteration to emulate computation      
``NUM_DIMS``                            The number of dimensions, valid values are 1, 2 and 3   
``DIM_1``                               The dimensionality of the source data                   
``DIM_2``                               The dimensionality of the source data                   
``DIM_3``                               The dimensionality of the source data      
``BLOCK_SIZE``             				Size of the block of data along ``dim_1`` for ``CS``, ``LDC``, ``RDC`` and Size of frame along ``dim_1`` for ``PRL``
``BLOCK_SIZE_2``             			Size of the block of data along ``dim_2`` for ``CS``, ``LDC``, ``RDC`` and Size of frame along ``dim_2`` for ``PRL``
``STRIDE_SIZE``             			Stride of the block of data along ``dim_1``, required for ``PRL``
``STRIDE_SIZE_2``             			Stride of the block of data along ``dim_2``, required for ``PRL``
======================================= ==========================================================

For ``MEM_PATTERN``, ``CONTIG`` represents arrays of basic data types (i.e., int, float, double, etc.); ``INTERLEAVED`` represents an array of structure (AOS) where each array element is a C struct; and ``STRIDED`` represents a few elements in an array of basic data types that are separated by a constant stride. ``STRIDED`` is supported only for 1D arrays. 

For ``FILE_PATTERN``, ``CONTIG`` represents a HDF5 dataset of basic data types (i.e., int, float, double, etc.); ``INTERLEAVED`` represents a dataset of a compound datatype;

For ``EMULATED_COMPUTE_TIME_PER_TIMESTEP``, you `must` provide the time unit (e.g. ``10 s``, ``100 ms``, or ``5000us``) to ensure correct behavior.

For ``DIM_2`` and ``DIM_3`` if **unused**, you should set both as ``1``. Notice that the total number of particles will be given by ``DIM_1 * DIM_2 * DIM_3``. For example, ``DIM_1=1024``, ``DIM_2=256``, ``DIM_3=1`` is a valid setting for a 2D array and it will generate ``262144`` particles.

A set of sample configuration files can be found in the ``samples/`` diretory in GitHub.

READ Settings (``h5bench_read``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

======================================= ==========================================================
**Parameter**                           **Description**                                         
======================================= ==========================================================
``READ_OPTION``                         Options: ``FULL``, ``PARTIAL``, ``STRIDED``, ``PRL``, ``RDC``, ``LDC``, ``CS``         
======================================= ==========================================================

For the ``PARTIAL`` option, the benchmark will read only the first ``TO_READ_NUM_PARTICLES`` particles. ``PRL``, ``LDC``, ``RDC`` and ``CS`` options work with a single MPI process. In case multiple processes are used, only the root performs the read operations and all other processes skip the reads. 


Asynchronous Settings
^^^^^^^^^^^^^^^^^^^^^

======================================= ==========================================================
**Parameter**                           **Description**                                         
======================================= ==========================================================
``MODE``                                Options: ``SYNC`` or ``ASYNC``        
``IO_MEM_LIMIT``                        Memory threshold to determine when to execute I/O       
``DELAYED_CLOSE_TIMESTEPS``             Groups and datasets will be closed later.               
======================================= ==========================================================

The ``IO_MEM_LIMIT`` parameter is optional. Its default value is ``0`` and it requires ``ASYNC``, i.e., it only works in asynchronous mode. This is the memory threshold used to determine when to actually execute the I/O operations. The actual I/O operations (data read/write) will not be executed until the timesteps associated memory reachs the threshold, or the application run to the end.

For the ``ASYNC`` mode to work you **must** define the necessay HDF5 ASYNC-VOL connector. For more information about it, refer to its `documentation <https://hdf5-vol-async.readthedocs.io/en/latest/>`_.

Compression Settings
^^^^^^^^^^^^^^^^^^^^

======================================= ==========================================================
**Parameter**                           **Description**                                         
======================================= ==========================================================
``COMPRESS``                            `YES` or `NO` (optional) enables parralel compression   
``CHUNK_DIM_1``                         Chunk dimension                                         
``CHUNK_DIM_2``                         Chunk dimension                                         
``CHUNK_DIM_3``                         Chunk dimension                                         
======================================= ==========================================================

Compression is only applicable for ``h5bench_write``. It has not effect for ``h5bench_read``. When enabled the chunk dimensions parameters (``CHUNK_DIM_1``, ``CHUNK_DIM_2``, ``CHUNK_DIM_3``) are required. The chunk dimension settings should be compatible with the data dimensions, i.e., they must have the same rank of dimensions, and chunk dimension size cannot be greater than data dimension size. Extra chunk dimensions have no effect and should be set to ``1``.

.. warning::

	There is a known bug on HDF5 parallel compression that could cause the system run out of memory when the chunk amount is large (large number of particle and very small chunk sizes). On Cori Hasswell nodes, the setting of 16M particles per rank, 8 nodes (total 256 ranks), 64 * 64 chunk size will crash the system by runing out of memory, on single nodes the minimal chunk size is 4 * 4.

Collective Operation Settings
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

======================================= ==========================================================
**Parameter**                           **Description**                                         
======================================= ==========================================================
``COLLECTIVE_DATA``                     Enables collective operation (default is ``NO``)        
``COLLECTIVE_METADATA``                 Enables collective HDF5 metadata (default is ``NO``)    
======================================= ==========================================================

Both ``COLLECTIVE_DATA`` and ``COLLECTIVE_METADATA`` parameters are optional.

Subfiling Settings
^^^^^^^^^^^^^^^^^^

======================================= ==========================================================
**Parameter**                           **Description**                                         
======================================= ==========================================================
``SUBFILING``                           Enables HDF5 subfiling (default is ``NO``)  
======================================= ==========================================================

.. attention:: 

	In order to enable this option your HDF5 must have been compiled with support for the HDF5 Subfiling Virtual File Driver (VFD) which was introduced in the HDF5 1.14.0. For CMake you can use the ``-DHDF5_ENABLE_PARALLEL=ON -DHDF5_ENABLE_SUBFILING_VFD=ON`` and for autotools ``--enable-parallel --enable-subfiling-vfd=yes``. Without this support, this parameter has no effect.

CSV Settings
^^^^^^^^^^^^

Performance results will be written to this file and standard output once a file name is provided.

======================================= ==========================================================
**Parameter**                           **Description**                                         
======================================= ==========================================================
``CSV_FILE``                            CSV file name to store benchmark results                
======================================= ==========================================================

Supported Patterns
------------------

.. attention:: 

	Not every pattern combination is covered by the benchmark. Supported benchmark parameter settings are listed below.

Supported Write Patterns (``h5bench_write``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The I/O patterns include array of structures (AOS) and structure of arrays (SOA) in memory as well as in file. The array dimensions are 1D, 2D, and 3D for the write benchmark. This defines the write access pattern, including ``CONTIG`` (contiguous), ``INTERLEAVED`` and ``STRIDED`` for the source (the data layout in the memory) and the destination (the data layout in the resulting file). For example, ``MEM_PATTERN=CONTIG`` and ``FILE_PATTERN=INTERLEAVED`` is a write pattern where the in-memory data layout is contiguous (see the implementation of ``prepare_data_contig_2D()`` for details) and file data layout is interleaved by due to its compound data structure (see the implementation of ``data_write_contig_to_interleaved()`` for details).


- 4 patterns for both 1D and 2D array write (``NUM_DIMS=1`` or ``NUM_DIMS=2``)

.. code-block:: none

	'MEM_PATTERN': 'CONTIG'
	'FILE_PATTERN': 'CONTIG'

.. code-block:: none

	'MEM_PATTERN': 'CONTIG'
	'FILE_PATTERN': 'INTERLEAVED'

.. code-block:: none

	'MEM_PATTERN': 'INTERLEAVED'
	'FILE_PATTERN': 'CONTIG'

.. code-block:: none

	'MEM_PATTERN': 'INTERLEAVED'
	'FILE_PATTERN': 'INTERLEAVED'

- 1 pattern for 3D array (``NUM_DIMS=3``)

.. code-block:: none

	'MEM_PATTERN': 'CONTIG'
	'FILE_PATTERN': 'CONTIG'


- 1 strided pattern for 1D array (``NUM_DIMS=1``)

.. code-block:: none

	'MEM_PATTERN': 'CONTIG'
	'FILE_PATTERN': 'STRIDED'


Supported Read Patterns (``h5bench_read``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- 1 pattern for 1D, 2D and 3D read (``NUM_DIMS=1`` or ``NUM_DIMS=2``)


Contiguously read through the whole data file:

.. code-block:: none

	'MEM_PATTERN': 'CONTIG'
	'FILE_PATTERN': 'CONTIG'
	'READ_OPTION': 'FULL'

- 2 patterns for 1D read

Contiguously read the first ``TO_READ_NUM_PARTICLES`` elements:

.. code-block:: none

	'MEM_PATTERN': 'CONTIG'
	'FILE_PATTERN': 'CONTIG'
	'READ_OPTION': 'PARTIAL'
	
.. code-block:: none

	'MEM_PATTERN': 'CONTIG'
	'FILE_PATTERN': 'STRIDED'
	'READ_OPTION': 'STRIDED'

- 4 patterns for 2D read

1. PRL: Refers to the Peripheral data access pattern. Data is read from the periphery of the 2D dataset, which is a frame of fixed width and height around the dataset.
.. code-block:: none

	'MEM_PATTERN': 'CONTIG'
	'FILE_PATTERN': 'CONTIG'
	'READ_OPTION': 'PRL'

2. RDC: Refers to the Right Diagonal Corner data access pattern. Data is read from two identical blocks of fixed sides, one in the top right corner and the other in the bottom left corner in the 2D HDF5 dataset
.. code-block:: none

	'MEM_PATTERN': 'CONTIG'
	'FILE_PATTERN': 'CONTIG'
	'READ_OPTION': 'RDC'

3. LDC: Refers to the Left Diagonal Corner data access pattern. Data is read from two identical blocks of fixed sides, one in the top left corner and the other in the bottom right corner in the 2D HDF5 dataset
.. code-block:: none

	'MEM_PATTERN': 'CONTIG'
	'FILE_PATTERN': 'CONTIG'
	'READ_OPTION': 'LDC'

4. CS: Refers to the Cross Stencil data access pattern. A block of fixed sides is used to read data from an HDF5 dataset. This block is given a fixed stride in each dimension and data till end of dataset is read.
.. code-block:: none

	'MEM_PATTERN': 'CONTIG'
	'FILE_PATTERN': 'CONTIG'
	'READ_OPTION': 'CS'



Understanding the Output
------------------------

The metadata and raw data operations are timed separately, and the overserved time and I/O rate are based on the total time.

Sample output of ``h5bench_write``:

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

Sample output of ``h5bench_read``:

.. code-block:: none

	=================  Performance results  =================
	Total emulated compute time = 4 sec
	Total read size = 2560 MB
	Metadata time = 17.523 ms
	Raw read time = 1.201 sec
	Observed read completion time = 5.088 sec
	Raw read rate = 2132.200 MB/sec
	Observed read rate = 2353.605225 MB/sec

Supported Special Write Pattern (``h5bench_write_var_normal_dist``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
In ``h5bench_write``, each process writes the same amount of local data. This program ``h5bench_write_var_normal_dist`` demonstrates a prototype for each process writing a varying size local data buffer which 
follows a normal distribution based on the given mean number of particles provided from ``DIM1`` and standard deviation ``STDEV_DIM1`` in the config file. This special benchmark currently supports only ``DIM1``. check ``samples/sync-write-1d-contig-contig-write-full_var_normal_dist.json``

.. code-block:: none

 "benchmarks": [
        {
            "benchmark": "write_var_normal_dist",
            "file": "test.h5",
            "configuration": {
                "MEM_PATTERN": "CONTIG",
                "FILE_PATTERN": "CONTIG",
                "TIMESTEPS": "5",
                "DELAYED_CLOSE_TIMESTEPS": "2",
                "COLLECTIVE_DATA": "YES",
                "COLLECTIVE_METADATA": "YES",
                "EMULATED_COMPUTE_TIME_PER_TIMESTEP": "1 s", 
                "NUM_DIMS": "1",
                "DIM_1": "524288",
                "STDEV_DIM_1":"100000",
                "DIM_2": "1",
                "DIM_3": "1",
                "CSV_FILE": "output.csv",
                "MODE": "SYNC"
            }

Sample output of ``h5bench_write_var_normal_dist``:

.. code-block:: none

	==================  Performance results  =================
	metric, value, unit
	operation, write, 
	ranks, 16, 
	Total number of particles, 8M, 
	Final mean particles, 550199, 
	Final standard deviation, 103187.169653, 
	collective data, YES, 
	collective meta, YES, 
	subfiling, NO, 
	total compute time, 4.000, seconds
	total size, 1.849, GB
	raw time, 17.949, seconds
	raw rate, 105.509, MB/s
	metadata time, 0.001, seconds
	observed rate, 87.519, MB/s
	observed time, 25.639, seconds



Known Issues
------------

.. warning::

	In Cori/NERSC or similar platforms that use Cray-MPICH library, if you encouter a failed assertion regarding support for ``MPI_THREAD_MULTIPLE`` you should define the following environment variable:

	.. code-block:: bash

		export MPICH_MAX_THREAD_SAFETY="multiple"

.. warning::

	If you're trying to run the benchmark with the HDF5 VOL ASYNC connector in MacOS and are getting segmentation fault (from ``ABT_thread_create``), please try to set the following environment variable:

	.. code-block:: bash

		export ABT_THREAD_STACKSIZE=100000
