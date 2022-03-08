E3SM
====

E3SM-IO is the parallel I/O kernel from the E3SM climate simulation model. It makes use of PIO library which is built on top of PnetCDF.

This benchmark currently has two cases from E3SM, namely F and G cases. The F case uses three unique data decomposition patterns shared by 388 2D and 3D variables (2 sharing Decomposition 1, 323 sharing Decomposition 2, and 63 sharing Decomposition 3). The G case uses 6 data decompositions shared by 52 variables (6 sharing Decomposition 1, 2 sharing Decomposition 2, 25 sharing Decomposition 3, 2 sharing Decomposition 4, 2 sharing Decomposition 5, and 4 sharing Decomposition 6).

You can find more information in Parallel-NetCDF `GitHub repository <https://github.com/Parallel-NetCDF/E3SM-IO>`_.

Configuration
-------------

You can configure the ES3M-IO benchmark with the following options. Notice that if you use the ``configuration.json`` approach to define the runs for ``h5bench``, we will automatically generate the final configuration file based on the options you provide in the JSON file. For standalone usage of this benchmark, you can refer to E3SM-IO repository.

====================== ==============================================================================
**Parameter**          **Description**                                                             
====================== ==============================================================================
``k``                  Keep the output files when program exits                                                                 
``x``                  I/O strategy to write (``canonical``, ``log``, and ``blob``) 
``a``                  I/O library name to perform write operation (``hdf5``, ``hdf5_log``, ``hdf5_md``)                                                            
``r``                  Number of records/time steps for F case h1 file                                       
``o``                  Enable write performance evaluation
``netcdf``             Define the HDF5 compression algorithm to use                                
====================== ==============================================================================

.. warning::

    h5bench temporarily only supports ``-x blob`` and ``-a hdf5``. If you set other options, they will be overwritten to the supported version.

JSON Configuration (recomended)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To run an instance of AMReX HDF5 benchmark you need to include the following in the ``benchmarks`` property of your ``configuration.json`` file:

.. code-block::

    {
        "benchmark": "e3sm",
        "file": "coisa.h5",
        "configuration": {
            "k": "",
            "x": "blob",
            "a": "hdf5",
            "r": "25",
            "o": "ON",
            "netcdf": "../../e3sm/datasets/f_case_866x72_16p.nc"
        }
    }