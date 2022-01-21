.. H5bench documentation master file, created by
   sphinx-quickstart on Thu Jun 17 08:33:20 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

h5bench
===================================

h5bench is a suite of parallel I/O benchmarks or kernels representing I/O patterns that are commonly used in HDF5 applications on high performance computing systems. H5bench measures I/O performance from various aspects, including the I/O overhead, observed I/O rate, etc.

These are the benchmarks and kernels currently available in h5bench:

==================== =========================== ==================== ========================
**Benchmark**        **Name**                    **SYNC**             **ASYNC VOL**          
==================== =========================== ==================== ========================
h5bench write        ``h5bench_write``           |:white_check_mark:| |:white_check_mark:|   
h5bench read         ``h5bench_read``            |:white_check_mark:| |:white_check_mark:|   
Metadata Stress      ``h5bench_hdf5_iotest``     |:white_check_mark:| |:white_large_square:| 
AMReX                ``h5bench_amrex``           |:white_check_mark:| |:white_check_mark:|   
Exerciser            ``h5bench_exerciser``       |:white_check_mark:| |:white_large_square:| 
OpenPMD (write)      ``h5bench_openpmd_write``   |:white_check_mark:| |:white_large_square:| 
OpenPMD (read)       ``h5bench_openpmd_read``    |:white_check_mark:| |:white_large_square:|
==================== =========================== ==================== ========================

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
   
.. toctree::
   :maxdepth: 2
   :caption: Contribute

   instructions

.. toctree::
   :maxdepth: 2
   :caption: Legal

   copyright
   license
