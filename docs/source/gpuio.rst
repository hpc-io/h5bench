GPU-IO
=========

These benchmarks extend the :doc:`Read / Write benchmarks <./vpic>` with memory transfers to and from GPU memory. Refer to the :doc:`Build Instructions <./buildinstructions>` for enabling these benchmarks with CUDA support.

Configuration
-------------

You can configure the ``h5bench_cuda_write`` and ``h5bench_cuda_read`` benchmarks with the following options. Notice that if you use the ``samples/sync-cuda-write-1d-contig-contig.json`` approach to define the runs for ``h5bench``, we will automatically generate the final configuration file based on the options you provide in the JSON file. For standalone usage of this benchmark, you can check the input format at the end of this document and refer to its documentation.

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
``DIM_3``                               The dimensionality of the source data
``FILE_PER_PROC``                       ``YES`` to enable file per process mode (``NO`` is not supported with GPU-IO)
======================================= ==========================================================

For ``MEM_PATTERN``, ``CONTIG`` represents arrays of basic data types (i.e., int, float, double, etc.); ``INTERLEAVED`` represents an array of structure (AOS) where each array element is a C struct; and ``STRIDED`` represents a few elements in an array of basic data types that are separated by a constant stride. ``STRIDED`` is supported only for 1D arrays.

For ``FILE_PATTERN``, ``CONTIG`` represents a HDF5 dataset of basic data types (i.e., int, float, double, etc.); ``INTERLEAVED`` represents a dataset of a compound datatype;

For ``EMULATED_COMPUTE_TIME_PER_TIMESTEP``, you `must` provide the time unit (e.g. ``10 s``, ``100 ms``, or ``5000us``) to ensure correct behavior.

For ``DIM_2`` and ``DIM_3`` if **unused**, you should set both as ``1``. Notice that the total number of particles will be given by ``DIM_1 * DIM_2 * DIM_3``. For example, ``DIM_1=1024``, ``DIM_2=256``, ``DIM_3=1`` is a valid setting for a 2D array and it will generate ``262144`` particles.

A set of sample configuration files can be found in the ``samples/`` diretory in GitHub.

READ Settings (``h5bench_cuda_read``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

======================================= ==========================================================
**Parameter**                           **Description**
======================================= ==========================================================
``READ_OPTION``                         Options: ``FULL``, ``PARTIAL``, and ``STRIDED``
======================================= ==========================================================

For the ``PARTIAL`` option, the benchmark will read only the first ``TO_READ_NUM_PARTICLES`` particles.


GPUDirect Storage with GDS VFD
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

With the HDF5 GDS VFD (https://github.com/hpc-io/vfd-gds), you can benchmark GPUDirect Storage. Note that not all NVIDIA GPUs and filesystems support GDS (see https://docs.nvidia.com/gpudirect-storage for more details on supported platforms). Currently, this mode is only supported by running the benchmark with a config file with the follow parameter set, manually.

======================================= ==========================================================
**Parameter**                           **Description**
======================================= ==========================================================
``DYNAMIC_VFD_NAME``                    Options: ``gds``
======================================= ==========================================================

CSV Settings
^^^^^^^^^^^^

Performance results will be written to this file and standard output once a file name is provided.

======================================= ==========================================================
**Parameter**                           **Description**
======================================= ==========================================================
``CSV_FILE``                            CSV file name to store benchmark results
======================================= ==========================================================


Understanding the Output
------------------------

The metadata and raw data operations are timed separately, and the overserved time and I/O rate are based on the total time.

Sample output of ``h5bench_cuda_write``:

.. code-block:: none

	=================== Performance Results ==================
	Total number of ranks: 1
	Total emulated compute time: 4.000 s
	Total write size: 2.500 GB
	Raw h2d time = 82.600 s
	Raw d2h time = 46.584 s
	Raw write time: 2.003 s
	Raw Full write time (inc. d2h) = 246.879 s
	Metadata time: 0.002 s
	H5Fcreate() time: 0.002 s
	H5Fflush() time: 0.000 s
	H5Fclose() time: 0.000 s
	Observed completion time: 7.311 s
	SYNC Raw h2d rate: 30.992 MB/s
	SYNC Raw d2h rate: 54.954 MB/s
	SYNC Raw write rate: 1.248 GB/s
	SYNC Raw Full write rate (inc. d2h): 10.369 MB/s
	SYNC Observed write rate: 773.215 MB/s
	===========================================================


Sample output of ``h5bench_cuda_read``:

.. code-block:: none

	=================== Performance Results ==================
	Total number of ranks: 1
	Total emulated compute time: 4.000 s
	Total read size: 2.500 GB
	Raw h2d time = 0.697 s
	Raw d2h time = 0.494 s
	Raw read time: 1.155 s
	Raw Full read time (inc. h2d) = 1.851 s
	Metadata time: 0.002 s
	Observed read completion time: 6.349 s
	SYNC Raw h2d rate: 3.587 GB/s
	SYNC Raw d2h rate: 5.058 GB/s
	SYNC Raw read rate: 2.165 GB/s
	SYNC Raw Full read rate (inc. d2h) = 1.350 GB/s
	SYNC Observed read rate: 1.063 GB/s
	===========================================================




