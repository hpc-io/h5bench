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
``operation``		   Operation: ``write`` or ``read``
``fileLocation``       Directory where the file will be written to or read from                    
====================== ==============================================================================

When running with the ``write`` operation, you have to define the following options:

``dim``                Number of dimensions (``1``, ``2``, or ``3``)                                                     
``balanced``           Should it use a balanced load? (``true`` or ``false``)
``ratio``              Particle to mesh ratio                             
``steps``              Number of iteration steps                                       
``minBlock``           Meshes are viewed as grid of mini blocks
``grid``               Grid based on the mini block                   

When running with the ``read`` operation, you have to define the pattern:

``pattern``            Read access pattern


The ``minBlock`` and ``grid`` parameters must include the values for each of the ``dim`` dimensions. For example, if ``"dim": "3"`` (for a 3D mesh) ``minBlock`` should contain three values, one for each dimension ``"16 32 32"`` and ``grid`` (which is based on the mini block) should also contain three values, one for each dimension ``"32 32 16"``.

For the ``pattern`` attribute for read you can chose:

- ``m``: metadata only
- ``sx``: slice of the 'rho' mesh in the x-axis (eg. ``x=0``)
- ``sy``: slice of the 'rho' mesh in the y-axis (eg. ``y=0``)
- ``sz``: slice of the 'rho' mesh in the z-axis (eg. ``z=0``)
- ``fx``: slice of the 3D magnetic field in the x-axis (eg. ``x=0``)
- ``fy``: slice of the 3D magnetic field in the y-axis (eg. ``y=0``)
- ``fz``: slice of the 3D magnetic field in the z-axis (eg. ``z=0``)

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
            "pattern": "sy"
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
