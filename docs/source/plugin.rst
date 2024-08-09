Build and Run External Compression Plugins
===================================

-----------------------------------
SZ3
-----------------------------------

Build with CMake
-----------------------------------

.. code-block:: bash

    git clone https://github.com/szcompressor/SZ3
    cd SZ3
    mkdir build installer
    realpath installer
    cd build
    export CMAKE_INSTALL_PREFIX=.../SZ3/installer
    export PATH=.../hdf5/installer/bin:$PATH
    export HDF5_HOME=.../hdf5/installer
    cmake -DCMAKE_INSTALL_PREFIX=$CMAKE_INSTALL_PREFIX -DBUILD_H5Z_FILTER=ON ..
    make
    make install

Enable SZ3 in benchmark at runtime`
-----------------------------------
In order to make sure HDF5 can find the installed plugin and apply it to the datasets, you **must** either define the macro ``HDF5_PLUGIN_PATH`` using ``export HDF5_PLUGIN_PATH=.../SZ3/installer/lib64`` in every session or giving that as an input in the configuration JSON file and h5bench will set up for you:
.. code-block::

    "vol": {
        "path": ".../SZ3/installer/lib64"
    }

-----------------------------------
ZFP
-----------------------------------

Build with CMake
-----------------------------------
First, clone the ZFP GitHub repository and build ZFP
.. code-block:: bash
    
    git clone https://github.com/LLNL/zfp.git
    cd zfp
    mkdir build installer
    realpath installer
    cd build
    export CMAKE_INSTALL_PREFIX=.../zfp/installer
    cmake -DCMAKE_INSTALL_PREFIX=$CMAKE_INSTALL_PREFIX -DZFP_BIT_STREAM_WORD_SIZE=8 ..
    make
    make install

Second, clone the H5Z-ZFP GitHub repository and build H5Z-ZFP
.. code-block:: bash

    git clone https://github.com/LLNL/H5Z-ZFP.git
    cd H5Z-ZFP
    mkdir build installer
    realpath installer
    cd build
    export CMAKE_INSTALL_PREFIX=.../H5Z-ZFP/installer
    export HDF5_DIR=.../hdf5/installer
    export ZFP_DIR=.../zfp/installer/lib64/cmake/zfp
    cmake -DCMAKE_INSTALL_PREFIX=$CMAKE_INSTALL_PREFIX ..
    make
    make install

Enable ZFP in benchmark at runtime
-----------------------------------
You **must** either define the macro ``HDF5_PLUGIN_PATH`` using ``export HDF5_PLUGIN_PATH=.../H5Z-ZFP/installer/plugin`` in every session or giving that in the JSON file:
.. code-block::

    "vol": {
        "path": ".../H5Z-ZFP/installer/plugin"
    }

