.. H5bench documentation master file, created by
   sphinx-quickstart on Thu Jun 17 08:33:20 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

h5bench
===================================

h5bench is a suite of parallel I/O benchmarks or kernels representing I/O patterns that are commonly used in HDF5 applications on high performance computing systems. H5bench measures I/O performance from various aspects, including the I/O overhead, observed I/O rate, etc. You can check h5bench source-code at https://github.com/hpc-io/h5bench.

|badge-github| |badge-spack| |badge-status| |badge-docs|

.. |badge-github| image:: https://img.shields.io/github/v/release/hpc-io/h5bench?label=h5bench&logo=github
   :alt: GitHub release (latest by date)
   
.. |badge-spack| image:: https://img.shields.io/spack/v/h5bench
   :alt: Spack
   
.. |badge-status| image:: https://img.shields.io/github/checks-status/hpc-io/h5bench/master
   :alt: GitHub branch checks state

.. |badge-docs| image:: https://img.shields.io/readthedocs/h5bench?logo=readthedocs&logoColor=white
   :alt: Read the Docs

These are the benchmarks and kernels currently available in h5bench:

==================== =========================== ==================== ======================== ======================== ========================
**Benchmark**        **Name**                    **SYNC**             **ASYNC**                **CACHE**                **LOG**          
==================== =========================== ==================== ======================== ======================== ========================
h5bench write        ``h5bench_write``           |:white_check_mark:| |:white_check_mark:|     |:white_check_mark:|     |:white_check_mark:|
h5bench read         ``h5bench_read``            |:white_check_mark:| |:white_check_mark:|     |:white_check_mark:|     |:white_check_mark:|
Metadata Stress      ``h5bench_hdf5_iotest``     |:white_check_mark:| |:white_large_square:|   |:white_large_square:|   |:white_large_square:| 
AMReX                ``h5bench_amrex``           |:white_check_mark:| |:white_check_mark:|     |:white_large_square:|   |:white_large_square:| 
Exerciser            ``h5bench_exerciser``       |:white_check_mark:| |:white_large_square:|   |:white_large_square:|   |:white_large_square:| 
OpenPMD (write)      ``h5bench_openpmd_write``   |:white_check_mark:| |:white_large_square:|   |:white_large_square:|   |:white_large_square:| 
OpenPMD (read)       ``h5bench_openpmd_read``    |:white_check_mark:| |:white_large_square:|   |:white_large_square:|   |:white_large_square:| 
E3SM-IO              ``h5bench_e3sm``            |:white_check_mark:| |:white_large_square:|   |:white_large_square:|   |:white_check_mark:| 
MACSio               ``h5bench_macsio``          |:white_check_mark:| |:white_large_square:|   |:white_large_square:|   |:white_check_mark:| 
==================== =========================== ==================== ======================== ======================== ========================

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   buildinstructions
   running

.. toctree::
   :maxdepth: 2
   :caption: Benchmarks

   vpic
   metadata
   amrex
   openpmd
   exerciser
   e3sm
   macsio

.. toctree::
   :maxdepth: 2
   :caption: Contribute

   instructions

.. toctree::
   :maxdepth: 2
   :caption: Legal

   copyright
   license
