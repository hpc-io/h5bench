AMReX
=====

AMReX is a software framework for massively parallel, block-structured adaptive mesh refinement (AMR) applications.

You can find more information in AMReX-Codes `GitHub repository <https://amrex-codes.github.io/amrex>`_.

Configuration
-------------

You can configure the AMReX HDF5 benchmark with the following options. Notice that if you use the ``configuration.json`` approach to define the runs for ``h5bench``, we will automatically generate the final configuration file based on the options you provide in the JSON file. For standalone usage of this benchmark, you can check the input format at the end of this document and refer to its documentation.

====================== ==============================================================================
**Parameter**          **Description**                                                             
====================== ==============================================================================
``ncells``             Domain size                                                                 
``max_grid_size``      The maximum allowable size of each subdomain (used for parallel decomposal) 
``nlevs``              Number of levels                                                            
``ncomp``              Number of components in the multifabs                                       
``nppc``               Number of particles per cell                                                
``nplotfile``          Number of plot files to write                                               
``nparticlefile``      Number of particle files to write                                           
``sleeptime``          Time to sleep before each write                                             
``restart_check``      Whether to check the correctness of checkpoint/restart                      
``grids_from_file``    Enable AMReX to read grids from file                                        
``ref_ratio_file``     Refinement ratios for different AMReX refinement levels                     
``hdf5compression``    Define the HDF5 compression algorithm to use                                
====================== ==============================================================================

JSON Configuration (recomended)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To run an instance of AMReX HDF5 benchmark you need to include the following in the ``benchmarks`` property of your ``configuration.json`` file:

.. code-block::

    {
        "amrex": {
            "file": "amrex.h5",
            "configuration": {
                "ncells": "64",
                "max_grid_size": "8",
                "nlevs": "1",
                "ncomp": "6",
                "nppc": "2",
                "nplotfile": "2",
                "nparticlefile": "2",
                "sleeptime": "2",
                "restart_check": "1",
                "hdf5compression": "ZFP_ACCURACY#0.001"
            }
        }
    }

To read grids from file you need to set: ``grids_from_file``, ``nlevels``, and ``ref_ratio_file``.

.. code-block::

    {
        "amrex": {
            "file": "amrex.h5",
            "configuration": {
                "ncells": "64",
                "max_grid_size": "8",
                "nlevs": "1",
                "ncomp": "6",
                "nppc": "2",
                "nplotfile": "2",
                "nparticlefile": "2",
                "sleeptime": "2",
                "restart_check": "1",
                "hdf5compression": "ZFP_ACCURACY#0.001",
                "nlevs": "3",
                "grids_from_file": "1",
                "ref_ratio_file": "4 2"
            }
        }
    }

HDF5 ASYNC VOL Connector
^^^^^^^^^^^^^^^^^^^^^^^^

AMReX supports the `HDF5 ASYNC VOL connector <https://github.com/hpc-io/vol-async>`__. To enable it, you should specify in the ``vol`` property of you ``configuration.json`` file: the required library paths, the VOL ASYNC source path, and the connector setup.

.. code-block::

    "vol": {
        "library": "/vol-async/src:/hdf5-async-vol-register-install/lib:/argobots/install/lib:/hdf5-install/install:",
        "path": "/vol-async/src",
        "connector": "async under_vol=0;under_info={}"
    }


Standalone Configuration
^^^^^^^^^^^^^^^^^^^^^^^^

For standalone usage of this benchmark, this is the observed input configuration you should provide to the ``h5bench_amrex`` executable.

.. code-block::

    ncells = 64
    max_grid_size = 8
    nlevs = 1
    ncomp = 6
    nppc = 2
    nplotfile = 2
    nparticlefile = 2
    sleeptime = 2
    restart_check = 1

    # Uncomment to read grids from file
    # nlevs = 3
    # grids_from_file = 1
    # ref_ratio_file = 4 2

    # Uncomment to enable compression
    # hdf5compression=ZFP_ACCURACY#0.001

    directory = .
