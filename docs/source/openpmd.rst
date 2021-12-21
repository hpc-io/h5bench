OpenPMD
=======

OpenPMD is an open meta-data schema that provides meaning and self-description for data sets in science and engineering.

The openPMD-api library provides a reference API for openPMD data handling. 
In the h5bench Benchmarking Suite we provide support for the write and read parallel benchmarks with HDF5 backend.
You can find more information in `OpenPMD documentation <openpmd-api.readthedocs.io>`_.

Configuration
-------------

You can configure the openPMD write HDF5 benchmark with the following options. Notice that if you use the ``configuration.json`` approach to define the runs for ``h5bench``, we will automatically generate the final configuration file based on the options you provide in the JSON file. For standalone usage of this benchmark, you can check the input format at the end of this document and refer to its documentation.

====================== ==============================================================================
**Parameter**          **Description**                                                             
====================== ==============================================================================
``operation``		   Operation: write or read
``dim``                Number of dimensions                                                                 
``balanced``      	   Should it use a balanced load? 
``ratio``              Particle to mesh ratio                                                            
``steps``              Number of iteration steps                                       
``minBlock``           Meshes are viewed as grid of mini blocks                                                
``grid``               Grid based on the mini block                                               
``fileLocation``       Directory where the file will be written to or read from                    
====================== ==============================================================================

JSON Configuration (recomended)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To run an instance of openPMD HDF5 benchmark you need to include the following in the ``benchmarks`` property of your ``configuration.json`` file:

.. code-block::

	{
        "benchmark": "openpmd",
        "configuration": {
            "operation": "write",
            "dim": "3",
            "balanced": "true",
            "ratio": "1",
            "steps": "1",
            "minBlock": "8 16 16",
            "grid": "16 16 8"
        }
    },
    {
        "benchmark": "openpmd",
        "configuration": {
            "operation": "read",
            "dim": "3",
            "balanced": "true",
            "ratio": "1",
            "steps": "1",
            "minBlock": "8 16 16",
            "grid": "16 16 8"
        }
    }

Standalone Configuration
^^^^^^^^^^^^^^^^^^^^^^^^

For standalone usage of this benchmark, this is the observed input configuration you should provide to the ``h5bench_openpmd_write``.

.. code-block::

	dim=3
	balanced=true
	ratio=1
	steps=10
	minBlock=16 32 32
	grid=32 32 16


For the ``h5bench_openpmd_read``, you need to provide two arguments: the file prefix and the pattern.\
