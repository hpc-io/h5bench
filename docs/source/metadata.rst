Metadata Stress
===============

The Metadata Stress benchmark (``h5bench_hdf5_iotest``) is a simple I/O performance tester for HDF5. Its purpose is to assess the performance variability of a set of logically equivalent HDF5 representations of a common pattern. The test repeatedly writes (and reads) in parallel a set of 2D array variables in a tiled fashion, over a set of time steps. For more information referer to HDF Group `GitHub repository <https://github.com/HDFGroup/hdf5-iotest>`_. We modified this benchmark slightly so to be able to specify the config file location, everything else remains untouched.

Configuration
-------------

You can configure the Metadata Stress test with the following options. Notice that if you use the ``configuration.json`` approach to define the runs for ``h5bench``, we will automatically generate the final configuration file based on the options you provide in the JSON file. For standalone usage of this benchmark, you can check the input format at the end of this document and refer to its documentation.

====================== ======================================================================================================
 **Parameter**         **Description**                                                                                       
====================== ======================================================================================================
``steps``              Number of steps                                                                                       
``arrays``             Number of arrays                                                                                      
``rows``               Total number of array rows for strong scaling. Number of array rows per block for weak scaling.       
``columns``            Total number of array columns for strong scaling. Number of array columns per block for weak scaling. 
``process-rows``       Number of MPI-process rows: rows % proc-rows == 0 for strong scaling                                  
``process-columns``    Number of MPI-process columns: columns % proc-columns == 0 for strong scaling                         
``scaling``            Scaling ([weak, strong])                                                                              
``dataset-rank``       Rank of the dataset(s) in the file ([2, 3, 4])                                                        
``slowest-dimension``  Slowest changing dimension ([step, array])                                                            
``layout``             HDF5 dataset layout ([contiguous, chunked]                                                            
``mpi-io``             MPI I/O mode ([independent, collective])                                                              
``hdf5-file``          HDF5 output file name                                                                                 
``csv-file``           CSV results file name                                                                                 
====================== ======================================================================================================

JSON Configuration (recomended)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To run an instance of Metadata Stress Test benchmark you need to include the following in the ``benchmarks`` property of your ``configuration.json`` file:

.. code-block::

    {
    	"benchmark": "metadata",
        "file": "hdf5_iotest.h5",
        "configuration": {
            "version": "0",
            "steps": "20",
            "arrays": "500",
            "rows": "100",
            "columns": "200",
            "process-rows": "2",
            "process-columns": "2",
            "scaling": "weak",
            "dataset-rank": "4",
            "slowest-dimension": "step",
            "layout": "contiguous",
            "mpi-io": "independent",       
            "csv-file": "hdf5_iotest.csv"
        }
    }

Standalone Configuration
^^^^^^^^^^^^^^^^^^^^^^^^

For standalone usage of this benchmark, this is the observed input configuration you should provide to the ``h5bench_hdf5_iotest`` executable.

.. code-block::

	[DEFAULT]
	version = 0
	steps = 20
	arrays = 500
	rows = 100
	columns = 200
	process-rows = 1
	process-columns = 1
	scaling = weak
	dataset-rank = 4
	slowest-dimension = step
	layout = contiguous
	mpi-io = independent
	hdf5-file = hdf5_iotest.h5
	csv-file = hdf5_iotest.csv
