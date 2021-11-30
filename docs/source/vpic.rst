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
``NUM_PARTICLES``                       The number of particles that each rank needs to process 
``TIMESTEPS``                           The number of iterations                                
``EMULATED_COMPUTE_TIME_PER_TIMESTEP``  Sleeps after each iteration to emulate computation      
``NUM_DIMS``                            The number of dimensions, valid values are 1, 2 and 3   
``DIM_1``                               The dimensionality of the source data                   
``DIM_2``                               The dimensionality of the source data                   
``DIM_3``                               The dimensionality of the source data                   
======================================= ==========================================================

For ``MEM_PATTERN``, ``CONTIG`` represents arrays of basic data types (i.e., int, float, double, etc.); ``INTERLEAVED`` represents an array of structure (AOS) where each array element is a C struct; and ``STRIDED`` represents a few elements in an array of basic data types that are separated by a constant stride. ``STRIDED`` is supported only for 1D arrays. 

For ``FILE_PATTERN``, ``CONTIG`` represents a HDF5 dataset of basic data types (i.e., int, float, double, etc.); ``INTERLEAVED`` represents a dataset of a compound datatype;

For ``NUM_PARTICLES``, you can use absolute numbers (e.g. ``12345``) or in units (e.g. ``16 K``, ``128 M``, or ``256 G``). Notice that you `must` provide a space between the number and unit.

For ``EMULATED_COMPUTE_TIME_PER_TIMESTEP``, you `must` provide the time unit (e.g. ``10 s``, ``100 ms``, or ``5000us``) to ensure correct behavior.

For ``DIM_2`` and ``DIM_3`` if **unused**, you should set both as ``1``. Notice that ``NUM_PARTICLES == DIM_1 * DIM_2 * DIM_3`` **must** hold. For example, ``DIM_1=1024``, ``DIM_2=256``, ``DIM_3=1`` is a valid setting for a 2D array when ``NUM_PARTICLES=262144`` or ``NUM_PARTICLES=256 K``.

A set of sample configuration files can be found in the ``samples/`` diretory in GitHub.

READ Settings (``h5bench_read``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

======================================= ==========================================================
**Parameter**                           **Description**                                         
======================================= ==========================================================
``READ_OPTION``                         Options: ``FULL``, ``PARTIAL``, and ``STRIDED``         
======================================= ==========================================================

For the ``PARTIAL`` option, the benchmark will read only the first ``TO_READ_NUM_PARTICLES`` particles.


Asynchronous Settings
^^^^^^^^^^^^^^^^^^^^^

======================================= ==========================================================
**Parameter**                           **Description**                                         
======================================= ==========================================================
``ASYNC_MODE``                          Options: ``NON`` (default), ``IMP``, and ``EXP``        
``IO_MEM_LIMIT``                        Memory threshold to determine when to execute I/O       
``DELAYED_CLOSE_TIMESTEPS``             Groups and datasets will be closed later.               
======================================= ==========================================================

The ``IO_MEM_LIMIT`` parameter is optional. Its default value is ``0`` and it requires ``ASYNC_MODE=EXP``. It also only works in asynchronous mode. This is the memory threshold used to determine when to actually execute the I/O operations. The actual I/O operations (data read/write) will not be executed until the timesteps associated memory reachs the threshold, or the application run to the end.

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
