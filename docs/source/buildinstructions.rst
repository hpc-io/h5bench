Build Instructions
===================================

-----------------------------------
Build with CMake (recommended)
-----------------------------------

Dependency and environment variable settings
---------------------------------------------------

H5bench depends on MPI and Parallel HDF5.

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

Make sure to unload any system provided HDF5 version, and set an environment variable to specify the HDF5 install path:

.. code-block:: bash

	HDF5_HOME: the location you installed HDF5. It should point to a path that look like /path_to_my_hdf5_build/hdf5 and contains include/, lib/ and bin/ subdirectories.


Compile with CMake
---------------------------------------------------

Assume that the repo is cloned and now you are in the source directory h5bench, run the following simple steps:

.. code-block:: bash

	mkdir build
	cd build
	cmake ..
	make


Build to run in async
---------------------------------------------------

To run h5bench_vpicio or h5bench_bdcatsio in async mode, you need the develop branchs of BOTH HDF5 and Async-VOL and build H5bench separately.

.. code-block:: bash

	mkdir build
	cd build
	cmake .. -DWITH_ASYNC_VOL:BOOL=ON -DCMAKE_C_FLAGS="-I/$YOUR_ASYNC_VOL/src -L/$YOUR_ASYNC_VOL/src"
	make

Necessary environment variable setting:

.. code-block:: bash

	export HDF5_HOME="$YOUR_HDF5_DEVELOP_BRANCH_BUILD/hdf5"
	export ASYNC_HOME="$YOUR_ASYNC_VOL/src"
	export HDF5_VOL_CONNECTOR="async under_vol=0;under_info={}"
	export HDF5_PLUGIN_PATH="$ASYNC_HOME"
	export DYLD_LIBRARY_PATH="$HDF5_HOME/lib:$ASYNC_HOME"


And all the binaries will be built to the build/directory.

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
