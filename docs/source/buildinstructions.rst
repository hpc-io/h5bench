Build Instructions
===================================

-----------------------------------
Build with CMake (recommended)
-----------------------------------

First, clone the h5bench GitHub repository and ensure you are cloning the submodules:

.. code-block:: bash

	git clone --recurse-submodules https://github.com/hpc-io/h5bench

If you are upadting your h5bench, ensure you have the latest submodules that could be included in new releases:

.. code-block:: bash

	git submodule update --init

Dependency and environment variable settings
---------------------------------------------------

H5bench depends on MPI and parallel HDF5.

+++++++++++++++++++++++++++++++++
Use system provided by HDF5 
+++++++++++++++++++++++++++++++++

For instance on the Cori system at NERSC:

.. code-block:: bash
	
	module load cray-hdf5-parallel

You can also load any paralel HDF5 provided on your system, and you are good to go.

+++++++++++++++++++++++++++++++++
Use your own installed HDF5
+++++++++++++++++++++++++++++++++

Make sure to unload any system provided HDF5 version:, and set an environment variable to specify the HDF5 install path:

.. code-block:: bash

	export HDF5_HOME=/path/to/your/hdf5/installation

It should point to a path that contains the ``include/``, ``lib/``, and ``bin/`` subdirectories.

Compile with CMake
---------------------------------------------------

In the source directory of your cloned h5bench repository, run the following:

.. code-block:: bash

	mkdir build
	cd build

	cmake ..

	make
	make install

By default, h5bench will only compile the base write and read benchmarks. To enable the additional benchmarks, you need to explicitly enable them before building h5bench. You can also enable all the benchmarks with ``-DH5BENCH_ALL=ON``. Notice that some of them have additional dependencies.

==================== =========================== ===============================
**Benchmark**        **Name**                    **Build**                     
==================== =========================== ===============================
h5bench write        ``h5bench_write``           Always   
h5bench read         ``h5bench_read``            Always   
Metadata Stress      ``h5bench_hdf5_iotest``     ``-DH5BENCH_METADATA=ON``
AMReX                ``h5bench_amrex``           ``-DH5BENCH_AMREX=ON``   
Exerciser            ``h5bench_exerciser``       ``-DH5BENCH_EXERCISER=ON``
OpenPMD (write)      ``h5bench_openpmd_write``   ``-DH5BENCH_OPENPMD=ON``
OpenPMD (read)       ``h5bench_openpmd_read``    ``-DH5BENCH_OPENPMD=ON``
E3SM-IO              ``h5bench_e3sm``            ``-DH5BENCH_E3SM=ON`` 
==================== =========================== ===============================

.. warning::

	If you want to specify the installation directory, you can pass ``-DCMAKE_INSTALL_PREFIX`` to ``cmake``. If you are not installing it, make sure when you run ``h5bench``, you update your environment variables to include the `build` directory. Otherwise, h5bench will not be able to find all the benchmarks.

Build with HDF5 ASYNC VOL connector support
---------------------------------------------------

To run ``_async`` benchmarks, you need the develop branch of **both** HDF5 and ASYNC-VOL. When building h5bench you need to specify the ``-DWITH_ASYNC_VOL:BOOL=ON`` option and have already compiled the VOL connector in the ``$ASYNC_VOL`` directory:

.. code-block:: bash

	mkdir build
	cd build

	cmake .. -DWITH_ASYNC_VOL=ON -DCMAKE_C_FLAGS="-I/$ASYNC_VOL/src -L/$ASYNC_VOL/src"

	make
	make install

h5bench will automatically set the environment variables required to run the asynchronous versions, as long as you specify them in your JSON configuration file. However, if you run the benchmarks manually, you will need to set the following environment variables:

.. code-block:: bash

	export HDF5_HOME="$YOUR_HDF5_DEVELOP_BRANCH_BUILD/hdf5"
	export ASYNC_HOME="$YOUR_ASYNC_VOL/src"

	export HDF5_VOL_CONNECTOR="async under_vol=0;under_info={}"
	export HDF5_PLUGIN_PATH="$ASYNC_HOME"

	// Linux
	export LD_LIBRARY_PATH="$HDF5_HOME/lib:$ASYNC_HOME"
	// MacOS
	export DYLD_LIBRARY_PATH="$HDF5_HOME/lib:$ASYNC_HOME"

-----------------------------------
Build with Spack
-----------------------------------

You can also use Spack to install h5bench. However, notice that the current versions in Spack do not have support for the HDF5 VOL async/cache connectors yet.

.. code-block:: bash

	spack install h5bench

There are some variants available as described bellow:

.. code-block:: bash

	CMakePackage:   h5bench

	Description:
	    A benchmark suite for measuring HDF5 performance.

	Homepage: https://github.com/hpc-io/h5bench

	Maintainers: @jeanbez @sbyna

	Externally Detectable: 
	    False

	Tags: 
	    None

	Preferred version:  
	    1.1        https://github.com/hpc-io/h5bench/archive/refs/tags/1.1.tar.gz

	Safe versions:  
	    develop    [git] https://github.com/hpc-io/h5bench.git on branch develop
	    master     [git] https://github.com/hpc-io/h5bench.git on branch master
	    1.1        https://github.com/hpc-io/h5bench/archive/refs/tags/1.1.tar.gz
	    1.0        https://github.com/hpc-io/h5bench/archive/refs/tags/1.0.tar.gz

	Deprecated versions:  
	    None

	Variants:
	    Name [Default]                 When    Allowed values          Description
	    ===========================    ====    ====================    ==================================

	    build_type [RelWithDebInfo]    --      Debug, Release,         CMake build type
						   RelWithDebInfo,         
						   MinSizeRel              
	    ipo [off]                      --      on, off                 CMake interprocedural optimization

	Installation Phases:
	    cmake    build    install

	Build Dependencies:
	    cmake  hdf5  mpi

	Link Dependencies:
	    hdf5  mpi

	Run Dependencies:
	    None

	Virtual Packages: 
	    None
