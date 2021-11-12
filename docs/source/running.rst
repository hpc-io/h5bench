Running h5bench
===================================

-----------------------------------
h5bench (recommended)
-----------------------------------

We provide a single script you can use to run the benchmarks available in the h5bench Benchmarking Suite.
You can combine multiple benchmarks into a workflow with distinct configurations.
If you prefer, you can also manually run each benchmark in h5bench. For more details, refer to the Manual Execution section.

.. code-block::

   usage: h5bench [-h] [--debug] setup

   H5bench: a Parallel I/O Benchmark Suite for HDF5:

   positional arguments:
     setup       JSON file with the benchmarks to run

   optional arguments:
     -h, --help  show this help message and exit
     --debug     Enable debug mode


You need to provide a JSON file with the configurations you want to run.
If you're using `h5bench`, you should *not* call `mpirun`, `srun`, or any other parallel launcher on your own. 
Refer to the manual execution section if you want to follow that approach instead. 
The main script will handle setting and unsetting environment variables, launching the benchmarks with the provided configuration and HDF5 VOL connectors.

.. code-block::

   ./h5bench configuration.json

If you run it with the `--debug` option, h5bench will also print log messages `stdout`. The default behavior is to store it in a file. 

Configuration
-------------

The JSON configuration file has five main properties: `mpi`, `vol`, `file-system`, `directory`, `benchmarks`. 


MPI
^^^

You can set the MPI launcher you want to use, e.g. `mpirun`, `mpiexec`, and `srun`,
and provide the number of processes you want to use.
For other methods or a fine grain control on the job configuration, you can define the `configuration` properties that h5bench will use to launch the experiments using the `command` property you provided. If the `configuration` option is defined, h5bench will ignore the `ranks` property.

.. code-block::

   "mpi": {
      "command": "mpirun",
      "ranks": "4",
      "configuration": "-np 8 --oversubscribe"
   }

VOL
^^^

You can use HDF5 VOL connectors (async, cache, etc) for `h5bench_write` and `h5bench_read`.
Because some benchmarks inside h5bench do not have support for VOL connectors yet, you need to provide the necessary information in the configuration file to handle the VOL setup during runtime.

.. code-block::

   "vol": {
      "library": "/vol-async/src:/hdf5-async-vol-register-install/lib:/argobots/install/lib:/hdf5-install/install:",
      "path": "/vol-async/src",
      "connector": "async under_vol=0;under_info={}"
   }

You should provide the absolute path for all the libraries required by the VOL connector using the `library` property, the `path` of the VOL connector, and the configuration in `connector`. The provided example depicts how to configure the HDF5 VOL async connector.

Directory
^^^^^^^^^

h5bench will create a directory for the given execution workflow, where it will store all the generated files and logs. 
Additional options such as data striping for Lustre, if configured, will be applied to this directory.

.. code-block::

   "directory": "hdf5-output-directory"

File System
^^^^^^^^^^^

You can use this property to configure some file system options. For now, you can use it for Lustre to define the striping count and size that should be applied to the `directory` that will store all the generated data from `h5bench`.

.. code-block::

   "file-system": {
      "lustre": {
         "stripe-size": "1M",
         "stripe-count": "4"
      }
   }

Benchmarks
^^^^^^^^^^

You can specify which benchmarks `h5bench` should run in this property, their order, and configuration.
You can choose between: `write`, `read`, `metadata`, and `exerciser`. 

For the `write` pattern of `h5bench`, you should provide the `file` and the `configuration`:

.. code-block::

   {
      "write": {
         "file": "test.h5",
         "configuration": {
            "MEM_PATTERN": "CONTIG",
            "FILE_PATTERN": "CONTIG",
            "NUM_PARTICLES": "16 M",
            "TIMESTEPS": "5",
            "DELAYED_CLOSE_TIMESTEPS": "2",
            "COLLECTIVE_DATA": "NO",
            "COLLECTIVE_METADATA": "NO",
            "EMULATED_COMPUTE_TIME_PER_TIMESTEP": "1 s", 
            "NUM_DIMS": "1",
            "DIM_1": "16777216",
            "DIM_2": "1",
            "DIM_3": "1",
            "ASYNC_MODE": "NON",
            "CSV_FILE": "output.csv"
         }
      }
   }

For the `read` pattern of `h5bench`, you should provide the `file` and the `configuration`. 
If you provide the same `file` name used for a previous `write` execution, it will read from that file.
This way, you can configure a workflow with multiple interleaving files, e.g., `write` file-01, `write` file-02, `read` file-02, `read` file-01.

.. code-block::

   {
      "read": {
         "file": "test.h5",
         "configuration": {
            "MEM_PATTERN": "CONTIG",
            "FILE_PATTERN": "CONTIG",
            "NUM_PARTICLES": "16 M",
            "TIMESTEPS": "5",
            "DELAYED_CLOSE_TIMESTEPS": "2",
            "COLLECTIVE_DATA": "NO",
            "COLLECTIVE_METADATA": "NO",
            "EMULATED_COMPUTE_TIME_PER_TIMESTEP": "1 s", 
            "NUM_DIMS": "1",
            "DIM_1": "16777216",
            "DIM_2": "1",
            "DIM_3": "1",
            "ASYNC_MODE": "NON",
            "CSV_FILE": "output.csv"
         }
      }
   }

For the `metadata stress benchmark, `file` and `configuration` properties must be defined:

.. code-block::

   {
      "metadata": {
         "file": "hdf5_iotest.h5",
         "configuration": {
            "version": "0",
            "steps": "20",
            "arrays": "500",
            "rows": "100",
            "columns": "200",
            "process-rows": "2",
            "process-columns": "2",
            "scaling": "weak",
            "dataset-rank": "4",
            "slowest-dimension": "step",
            "layout": "contiguous",
            "mpi-io": "independent",       
            "csv-file": "hdf5_iotest.csv"
         }
      }
   }

For the `exerciser` benchmark, you need to provide the required runtime options in the JSON file inside the `configuration` property.

.. code-block::

   {
      "exerciser": {
      "configuration": {
         "numdims": "2",
         "minels": "8 8",
         "nsizes": "3",
         "bufmult": "2 2",
         "dimranks": "8 4"
      }
   }

You can refer to this sample of a complete `configuration.json` file that defined the workflow of the execution of multiple benchmarks from h5bench Suite:

.. literalinclude:: ../../configuration.json
   :language: json

For a description of all the options available in each benchmark, please refer to their entries in the documentation.

When the `--debug` option is enabled, you can expect an output similar to:

.. code-block::

   2021-10-25 16:31:24,866 h5bench - INFO - Starting h5bench Suite
   2021-10-25 16:31:24,889 h5bench - INFO - Lustre support detected
   2021-10-25 16:31:24,889 h5bench - DEBUG - Lustre stripping configuration: lfs setstripe -S 1M -c 4 full-teste
   2021-10-25 16:31:24,903 h5bench - INFO - h5bench [write] - Starting
   2021-10-25 16:31:24,903 h5bench - INFO - h5bench [write] - DIR: full-teste/504fc233/
   2021-10-25 16:31:24,904 h5bench - INFO - Parallel setup: srun --cpu_bind=cores -n 4
   2021-10-25 16:31:24,908 h5bench - INFO - srun --cpu_bind=cores -n 4 build/h5bench_write full-teste/504fc233/h5bench.cfg full-teste/test.h5
   2021-10-25 16:31:41,670 h5bench - INFO - SUCCESS
   2021-10-25 16:31:41,754 h5bench - INFO - Runtime: 16.8505464 seconds (elapsed time, includes allocation wait time)
   2021-10-25 16:31:41,755 h5bench - INFO - h5bench [write] - Complete
   2021-10-25 16:31:41,755 h5bench - INFO - h5bench [exerciser] - Starting
   2021-10-25 16:31:41,755 h5bench - INFO - h5bench [exerciser] - DIR: full-teste/247659d1/
   2021-10-25 16:31:41,755 h5bench - INFO - Parallel setup: srun --cpu_bind=cores -n 4
   2021-10-25 16:31:41,756 h5bench - INFO - srun --cpu_bind=cores -n 4 build/h5bench_exerciser --numdims 2  --minels 8 8  --nsizes 3  --bufmult 2 2  --dimranks 8 4 
   2021-10-25 16:31:49,174 h5bench - INFO - SUCCESS
   2021-10-25 16:31:49,174 h5bench - INFO - Finishing h5bench Suite

Cori
^^^^

In case you are running on Cori and the benchmark fails with an MPI message indicating no support for multiple threads, make sure you define:

.. code-block::

   export MPICH_MAX_THREAD_SAFETY="multiple" 

-----------------------------------
Manual Execution
-----------------------------------

If you prefer, you can execute each benchmark manually. In this scenario, you will be responsible for generating the input configuration file needed for each benchmark in the suite, ensuring it follows the pre-defined format unique for each one. 

If you want to use HDF5 VOL connectors or tune the file system configuration, `h5bench` will *not* take care of that. Remember that not all benchmarks in the suite have support for VOL connectors yet.
