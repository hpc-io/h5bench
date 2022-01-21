Exerciser
=========

.. attention::

	For more-detailed instructions of how to build and run the exerciser code on specific machines (at ALCF), see the Exerciser/BGQ/VESTA_XL/README.md and Exerciser/BGQ/THETA/README.md directories of this repository. Those README files also include instructions for building the CCIO and develop versions of HDF5 for use with this benchmark.

The **HDF5 Exerciser Benchmark** creates an HDF5 use case with some ideas/code borrowed from other benchmarks (namely IOR, VPICIO and FLASHIO). Currently, the algorithm does the following in parallel over all MPI ranks:

* For each rank, a local data buffer (with dimensions given by ``numdims`` is initialized with minNEls double-precision elements in each dimension

* If the ``derivedtype`` flag is used, a second local dataset is also specified with a derived data type a-signed to each element

* For a given number of iterations (hardcoded as ``NUM_ITERATIONS``):

	* Open a file, create a top group, set the MPI-IO transfer property, and (optionally) add a simple attribute string to the top group
	* Create memory and file dataspaces with hyperslab selections for simple rank-ordered offsets into the file. The ``rshift`` option can be used to specify the number of rank positions to shift the write position in the file (the read will be shifted twice this amount to avoid client-side caching effects
	* Write the data and close the file
	* Open the file, read in the data, and check correctness (if dataset is small enough)
	* Close the dataset (but not the file)
	* If the second (derived-type) data set is specified: (1) create a derived type, (2) open a new data set with the same number of elements and dimension, (3) write the data and (4) close everything

* Each dimension of curNEls is then multiplied by each dimension of bufMult, and the previous steps (the loop over ``NUM_ITERATIONS``) are repeated. This outer loop over local buffer sizes is repeated a total of nsizes times

Configuration
-------------

You can configure the ``h5bench_write`` and ``h5bench_read`` benchmarks with the following options. Notice that if you use the ``configuration.json`` approach to define the runs for ``h5bench``, we will automatically generate the final configuration file based on the options you provide in the JSON file. For standalone usage of this benchmark, you can check the input format at the end of this document and refer to its documentation.

Required
########

======================================= ==========================================================
**Parameter**                           **Description**                                         
======================================= ==========================================================
``numdims <x>``                    	   Dimension of the datasets to write to the HDF5 file
``minels <x> ... <x>``            	    Min number of double elements to write in each dim of the dataset (one value for each dimension)
======================================= ==========================================================

Optional
########

======================================= ==========================================================
**Parameter**                           **Description**                                         
======================================= ==========================================================
``nsizes <x>`` 							How many buffer sizes to use (Code will start with minbuf and loop through nsizes iterations, with the buffer size multiplied by bufmult in each dim, for each iteration)
``bufmult <x> ... <x>`` 				Constant, for each dimension, used to multiply the buffer [default: 2 2 ... ]
``metacoll`` 							Whether to set meta data collective usage [default: False]
``derivedtype`` 						Whether to create a second data set containing a derived type [default: False]
``addattr``								Whether to add attributes to group 1 [default: False]
``indepio`` 							Whether to use independant I/O (not MPI-IO) [default: False]
``keepfile`` 							Whether to keep the file around after the program ends for futher analysis, otherwise deletes it [default: False]
``usechunked`` 							Whether to chunk the data when reading/writing [default: False]
``maxcheck <x>`` 						Maximum buffer size (in bytes) to validate. Note that all buffers will be vaidated if this option is not set by this command-line argument [default: Inf]
``memblock <x>`` 						Define the block size to use in the local memory buffer (local buffer is always 1D for now, Note: This currently applies to the 'double' dataset only) [default: local buffer size]
``memstride <x>`` 						Define the stride of the local memory buffer (local buffer is always 1D for now, Note: This currently applies to the 'double' dataset only) [default: local buffer size]
``fileblocks <x> ...<x>``				Block sizes to use in the file for each dataset dimension (Note: This currently applies to the 'double' dataset only) [default: 1 ... 1]
``filestrides <x> ...<x>``				Stride dist. to use in the file for each dataset dimension (Note: This currently applies to the 'double' dataset only) [default: 1 ... 1]
======================================= ==========================================================

The exerciser also allows the MPI decomposition to be explicitly defined:

======================================= ==========================================================
**Parameter**                           **Description**                                         
======================================= ==========================================================
``dimranks <x> ...<x>``           	    MPI-rank division in each dimension. Note that, if not set, decomposition will be in 1st dimension only.
======================================= ==========================================================

Exerciser Basics
----------------

In the simplest case, the Exerciser code will simply write and then read an n-dimensional double-precision dataset in parallel (with all the necessary HDF5 steps in between). At a minimum, the user must specify the number of dimensions to use for this dataset (using the ``numdims`` flag), and the size of each dimension (using the ``minels`` flag). By default, the maximum number of dimensions allowed by the code is set by ``MAX_DIM`` (currently 4, but can be modified easily). Note that the user is specifying the number of elements to use in each dimension with ``minels``. Therefore, the local buffer size is the product of the dimension sizes and ``sizeof(double)`` (and the global dataset in the file is a product of the total MPI ranks and the local buffer size). As illustrated in Fig. 1, the mapping of ranks to hyper-slabs in the global dataset can be specified with the ``dimranks`` flag (here, Example 1 is the default decomposition, while Example 2 corresponds to: ``"dimranks": "2 2"``). This flag simply allows the user to list the number of spatial decompositions in each dimension of the global dataset, and requires that the product of the input to be equal to the total number of MPI ranks.


.. figure:: ../source/images/dimranks.png
	:width: 600
	:align: center
	:alt: Fig. 1 - Illustration of different local-to-global dataset mapping options.

	Fig. 1 - Illustration of different local-to-global dataset mapping options.


.. note::

	Authors:

	* Richard J. Zamora (rzamora@anl.gov)
	* Paul Coffman (pcoffman@anl.gov)
	* Venkatram Vishwanath (venkat@anl.gov)