MACSio
====

MACSio (Multi-purpose, Application-Centric, Scalable I/O Proxy Application) is being developed to fill a long existing void in co-design proxy applications that allow for I/O performance testing and evaluation of tradeoffs in data models, I/O library interfaces and parallel I/O paradigms for multi-physics, HPC applications.

MACSio in h5bench only supports the HDF5 interface. You need to have the `json-cwx <https://github.com/LLNL/json-cwx>`_ dependency library installed prior to compiling it in h5bench.

You can find more information in MACSio `GitHub repository <https://github.com/Parallel-NetCDF/E3SM-IO>`_.

Configuration
-------------

You can configure the MACSio benchmark with the following options. Notice that if you use the ``configuration.json`` approach to define the runs for ``h5bench``, we will automatically generate the final configuration file based on the options you provide in the JSON file. For standalone usage of this benchmark, you can refer to MACSio repository.

======================== ==========================================================================================
**Parameter**            **Description**                                                             
======================== ==========================================================================================
``parallel_file_mode``   Defines the parallel file mode
``part_size``            Defines the request sized (default per-proc request size is 80,0000 bytes (10K doubles))
``avg_num_parts``        Number of parts per process (default is 2)                                                            
``num_dumps``            Number of dumps (default is 10)
======================== ==========================================================================================

For the ``parallel_file_mode`` you can use Multiple Independent File (``MIF``) or Single Shared File (``SIF``) and specify the number of HDF5 files right after this parameter (see example below).


JSON Configuration (recomended)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To run an instance of MACSio HDF5 benchmark you need to include the following in the ``benchmarks`` property of your ``configuration.json`` file:

.. code-block::

    {
        "benchmark": "macsio",
        "file": "test.h5",
        "configuration": {
            "parallel_file_mode": "MIF 8",
            "part_size": "1M"
        }
    }