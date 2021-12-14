Ways to contribute
===================

We appreciate your interest in **h5bench**, and thank you for taking the time to contribute!

We have compiled a set of instructions to help us make h5bench even better.

Reporting bugs
--------------

You can open a new issue using our GitHub `issue tracker <https://github.com/hpc-io/h5bench/issues/new/choose>`_. If you run into an issue, please search first to ensure the issue has not been reported before. Open a new issue only if you have not found anything similar to your issue. Please, try to provide as much information as possible to reproduce your bug quickly.

Suggesting enhancements
-----------------------

You can use our GitHub `issue tracker <https://github.com/hpc-io/h5bench/issues/new/choose>`_ to describe your proposed feature. Please, provide the necessary context, covering why it is needed and what problem does it solve.

Adding new benchmarks
---------------------

We provide a set of instructions on how to add new benchmarks to the h5bench Benchmarking Suite. However, please notice that you might require some changes depending on how your benchmark work. You can contribute in two ways:

- **Adding existing benchmarks as submodules**: We plan to support only the version included in the original PR, based on its commit hash. Updates on the submodule require the contributor's help to ensure we can smoothly upgrade the available version without breaking existing features (both in the benchmark and in h5bench).

- **Adding newly developed benchmarks**: The community may perform maintenance, requiring you to provide comprehensive documentation (in code and usage) and examples to understand the benchmark module.

Example
^^^^^^^^^

To illustrate how you can add a new benchmark using the submodule aprroach we will use AMReX:

1. You need to include the AMReX repository as a submodule:

.. code-block:: bash

    git submodule add https://github.com/AMReX-Codes/amrex amrex

2. For this benchmark, we need some libraries to be compiled and available as well, so we will need to modify our ``CMakeLists.txt``, so it builds that subdirectory:

.. code-block:: bash

    set(AMReX_HDF5 YES)
    set(AMReX_PARTICLES YES)
    set(AMReX_MPI_THREAD_MULTIPLE YES)
    add_subdirectory(amrex)

3. AMReX comes with several other benchmarks. Still, since we are only interested in the HDF5 one, we will only compile that code. For that, we will need to add the following to our ``CMakeLists.txt``. This is based on how that benchmark is normally compiled within AMReX.

.. code-block:: bash

    set(amrex_src amrex/Tests/HDF5Benchmark/main.cpp)
    add_executable(h5bench_amrex ${amrex_src})

4. Be sure to follow the convention of naming the executable as ``h5bench_`` plus the benchmark name, e.g. ``h5bench_amrex``.

5. If you are going to provide support for the HDF5 async VOL connector with explicit implementation (which require changes in the original code), make sure you link the required libraries (``asynchdf5`` and ``h5async``):

.. code-block:: bash

    if(WITH_ASYNC_VOL)
            set(AMREX_USE_HDF5_ASYNC YES)
            target_link_libraries(h5bench_amrex hdf5 z m amrex asynchdf5 h5async MPI::MPI_C)
    else()
            target_link_libraries(h5bench_amrex hdf5 z m amrex MPI::MPI_C)
    endif()

6. The last step is to update the ``h5bench`` Python-based script to handle the new benchmark. On the top of the file, add the path of your benchmark:

.. code-block:: python

    H5BENCH_AMREX = 'h5bench_amrex'

Update the ``run()`` function that iterates over the ``benchmarks`` property list defined by the user in the ``configuration.json`` file to accept the new benchmark name:

.. code-block:: python

    elif name == 'amrex':
        self.run_amrex(id, benchmark[name], setup['vol'])

You then need to define the ``run_`` function for the benchmark you’re adding. The most important part is translating the configuration defined in the ``configuration.json`` file into a format accepted by your benchmark (e.g., a file, a JSON, command line). For AMReX, it requires an ``amrex.ini`` file with key-value configurations defined in the format ``key = value``, one per line:

.. code-block:: python

    # Create the configuration file for this benchmark
    with open(configuration_file, 'w+') as f:
        for key in configuration:
            f.write('{} = {}\n'.format(key, configuration[key]))

        f.write('directory = {}\n'.format(file))

If you plan to support the HDF5 async VOL connector, make sure you can ``enable_vol()`` and ``disable_vol()`` at the beginning and end of this ``run_`` function.

Here you can check an example of the complete ``run_amrex`` function:

.. code-block:: python

    def run_amrex(self, id, setup, vol):
        """Run the AMReX benchmark."""
        self.enable_vol(vol)

        try:
            start = time.time()

            file = '{}/{}'.format(self.directory, setup['file'])
            configuration = setup['configuration']

            configuration_file = '{}/{}/amrex.ini'.format(self.directory, id)

            # Create the configuration file for this benchmark
            with open(configuration_file, 'w+') as f:
                for key in configuration:
                    f.write('{} = {}\n'.format(key, configuration[key]))

                f.write('directory = {}\n'.format(file))

            command = '{} {} {}'.format(
                self.mpi,
                self.H5BENCH_AMREX,
                configuration_file
            )

            self.logger.info(command)

            # Make sure the command line is in the correct format
            arguments = shlex.split(command)

            stdout_file_name = 'stdout'
            stderr_file_name = 'stderr'

            with open(stdout_file_name, mode='w') as stdout_file, open(stderr_file_name, mode='w') as stderr_file:
                s = subprocess.Popen(arguments, stdout=stdout_file, stderr=stderr_file, env=self.vol_environment)
                sOutput, sError = s.communicate()

                if s.returncode == 0:
                    self.logger.info('SUCCESS')
                else:
                    self.logger.error('Return: %s (check %s for detailed log)', s.returncode, stderr_file_name)

                    if self.abort:
                        self.logger.critical('h5bench execution aborted upon first error')

                        exit(-1)

            end = time.time()

            self.logger.info('Runtime: {:.7f} seconds (elapsed time, includes allocation wait time)'.format(end - start))
        except Exception as e:
            self.logger.error('Unable to run the benchmark: %s', e)

        self.disable_vol(vol)

7. Make sure you provide some sample JSON configuration files in the ``configurations`` directory.

Please, feel free to reach us if you have questions!

Testing
-------

h5bench constantly receives updates and improvements. If you can run the latest version, please consider helping us by reporting your findings, including bugs and performance regressions. Running the benchmarks contained in the h5bench Benchmarking Suite with different configurations and platforms helps us a lot in making it more robust by quickly identifying and solving issues.