QMCPACK
========

QMCPACK is an open-source production-level many-body ab initio Quantum Monte Carlo code for computing the electronic structure of atoms, molecules, 2D nanomaterials and solids. The solid-state capabilities include metallic systems as well as insulators.

QMCPACK has a checkpoint/restart miniapp which has been integrated into `h5bench`. It dumps walker configurations and random number seeds to the HDF5 files and then reads them in and check the correctness. To have good performance at large scale, HDF5 version >= 1.10 is needed.

You can find more information in QMCPACK `GitHub repository <https://github.com/QMCPACK/qmcpack>`_.

Configuration
-------------

You can configure QMCPACK benchmark with the following options. Notice that if you use the ``configuration.json`` approach to define the runs for ``h5bench``, we will automatically generate the final configuration file based on the options you provide in the JSON file. For standalone usage of this benchmark, you can refer to E3SM-IO repository.

====================== ==============================================================================
**Parameter**          **Description**                                                             
====================== ==============================================================================
``i``                  Number of Monte Carlo steps
``g``                  Tiling (``tiling_1 tiling_2 tiling_3``)
``s``                  Random seed
``w``                  Number of walkers
``r``                  Maximum pair distance (i.e., ``rmax``)
====================== ==============================================================================

JSON Configuration (recomended)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To run an instance of AMReX HDF5 benchmark you need to include the following in the ``benchmarks`` property of your ``configuration.json`` file:

.. code-block::

    {
        "benchmark": "qmcpack",
        "configuration": {
            "i": "100",
            "g": "4 4 1"
        }
    }