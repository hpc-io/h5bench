#!/usr/bin/env python3

import os
import sys
import json
import time
import uuid
import shlex
import errno
import distutils.spawn
import argparse
import collections
import subprocess
import h5bench_version
import h5bench_configuration
import logging
import logging.handlers


class H5bench:
    """H5bench benchmark suite."""

    H5BENCH_PATTERNS_WRITE = 'h5bench_write'
    H5BENCH_PATTERNS_WRITE_UNLIMITED = 'h5bench_write_unlimited'
    H5BENCH_PATTERNS_WRITE_VAR_NORMAL_DIST = 'h5bench_write_var_normal_dist'
    H5BENCH_PATTERNS_WRITE_VAR_DATA_DIST = 'h5bench_write_var_data_dist'
    H5BENCH_PATTERNS_APPEND = 'h5bench_append'
    H5BENCH_PATTERNS_OVERWRITE = 'h5bench_overwrite'
    H5BENCH_PATTERNS_READ = 'h5bench_read'
    H5BENCH_EXERCISER = 'h5bench_exerciser'
    H5BENCH_METADATA = 'h5bench_hdf5_iotest'
    H5BENCH_AMREX_SYNC = 'h5bench_amrex_sync'
    H5BENCH_AMREX_ASYNC = 'h5bench_amrex_async'
    H5BENCH_OPENPMD_WRITE = 'h5bench_openpmd_write'
    H5BENCH_OPENPMD_READ = 'h5bench_openpmd_read'
    H5BENCH_E3SM = 'h5bench_e3sm'
    H5BENCH_MACSIO = 'h5bench_macsio'

    def __init__(self, setup, prefix=None, debug=None, abort=None, validate=None, filter=None):
        """Initialize the suite."""
        self.LOG_FILENAME = '{}-h5bench.log'.format(setup.replace('.json', ''))

        self.check_parallel()

        self.configure_log(debug)

        self.prefix = prefix
        self.setup = setup
        self.abort = abort
        self.validate = validate

        if filter:
            self.filter = filter.split(',')
        else:
            self.filter = None

    def check_parallel(self):
        """Check for parallel overwrite command."""
        mpi = [
            'mpirun', 'mpiexec',
            'srun'
        ]

        # Get user defined shell
        if 'SHELL' in os.environ:
            shell = os.environ['SHELL']

            for m in mpi:
                if m in shell:
                    print('You should not call MPI directly when running h5bench.')

                    sys.exit(os.EX_USAGE)
        else:
            shell = None

    def configure_log(self, debug):
        """Configure the logging system."""
        self.logger = logging.getLogger('h5bench')

        if debug:
            self.logger.setLevel(logging.DEBUG)
        else:
            self.logger.setLevel(logging.INFO)

        # Defines the format of the logger
        formatter = logging.Formatter(
            '%(asctime)s %(module)s - %(levelname)s - %(message)s'
        )

        # Configure the log rotation
        handler = logging.handlers.RotatingFileHandler(
            self.LOG_FILENAME,
            maxBytes=268435456,
            backupCount=50,
            encoding='utf8'
        )

        handler.setFormatter(formatter)

        self.logger.addHandler(handler)

        if debug:
            console = logging.StreamHandler()
            console.setFormatter(formatter)

            self.logger.addHandler(console)

    def prepare(self, setup):
        """Create a directory to store all the results of the benchmark."""
        self.directory = setup['directory']

        try:
            # Create a temporary directory to store all configurations
            os.makedirs(self.directory)
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise

            self.logger.warning('Base directory already exists: {}'.format(self.directory))

            pass
        except Exception as e:
            self.logger.debug('Unable to create {}: {}'.format(self.directory, e))

        # Check for Lustre support to set the data stripping configuration
        try:
            command = 'lfs getstripe {}'.format(self.directory)

            arguments = shlex.split(command)

            s = subprocess.Popen(arguments, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            sOutput, sError = s.communicate()

            if s.returncode == 0:
                self.logger.info('Lustre support detected')

                if 'file-system' in setup:
                    if 'lustre' in setup['file-system']:
                        command = 'lfs setstripe'

                        if 'stripe-size' in setup['file-system']['lustre']:
                            command += ' -S {}'.format(setup['file-system']['lustre']['stripe-size'])

                        if 'stripe-count' in setup['file-system']['lustre']:
                            command += ' -c {}'.format(setup['file-system']['lustre']['stripe-count'])

                        command += ' {}'.format(self.directory)

                        self.logger.debug('Lustre stripping configuration: {}'.format(command))

                        arguments = shlex.split(command)

                        s = subprocess.Popen(arguments, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                        sOutput, sError = s.communicate()

            else:
                self.logger.info('Lustre support not detected')
        except Exception:
            self.logger.info('Lustre support not detected')

    def validate_json(self, setup):
        """Make sure JSON contains all the necessary properties."""
        properties = [
            'mpi',
            'vol',
            'file-system',
            'directory',
            'benchmarks'
        ]

        for p in properties:
            if p not in setup:
                self.logger.critical('JSON configuration file is invalid: "{}" property is missing'.format(p))

                sys.exit(os.EX_DATAERR)

    def run(self):
        """Run all the benchmarks/kernels."""
        self.logger.info('Starting h5bench Suite')

        try:
            with open(self.setup) as file:
                setup = json.load(file, object_pairs_hook=collections.OrderedDict)
        except FileNotFoundError:
            self.logger.critical('Unable to open the input configuration file')

            sys.exit(os.EX_NOINPUT)
        except Exception as e:
            self.logger.critical('Unable to parse the input configuration file)')
            self.logger.critical(e)

            sys.exit(os.EX_NOINPUT)

        self.validate_json(setup)
        self.prepare(setup)

        self.vol_environment = os.environ.copy()
        self.prepare_vol(setup['vol'])

        benchmarks = setup['benchmarks']

        for benchmark in benchmarks:
            name = benchmark['benchmark']

            # Check if filters were enabled
            if self.filter:
                if name not in self.filter:
                    self.logger.warning('Skipping "{}" due to active filters'.format(name))

                    continue

            id = str(uuid.uuid4()).split('-')[0]
            if 'SLURM_JOB_ID' in os.environ:
                jobid = os.environ['SLURM_JOB_ID']  # nersc
            elif 'COBALT_JOBID' in os.environ:
                jobid = os.environ['COBALT_JOBID']  # alcf_theta
            elif 'PBS_JOBID' in os.environ:
                jobid = os.environ['PBS_JOBID']     # alcf_polaris
            elif 'LSB_JOBID' in os.environ:
                jobid = os.environ['LSB_JOBID']     # olcf
            else:
                jobid = None

            if jobid is not None:
                id = id + "-" + jobid
                self.logger.info('JOBID: {}'.format(jobid))

            self.logger.info('h5bench [{}] - Starting'.format(name))
            self.logger.info('h5bench [{}] - DIR: {}/{}/'.format(name, setup['directory'], id))

            os.makedirs('{}/{}'.format(setup['directory'], id))

            self.prepare_parallel(setup['mpi'])

            if name in ['write', 'write-unlimited', 'overwrite', 'append', 'read', 'write_var_normal_dist', 'write_var_data_dist']:
                self.run_pattern(id, name, benchmark, setup['vol'])
            elif name == 'exerciser':
                self.run_exerciser(id, benchmark)
            elif name == 'metadata':
                self.run_metadata(id, benchmark)
            elif name == 'amrex':
                self.run_amrex(id, benchmark, setup['vol'])
            elif name == 'openpmd':
                self.run_openpmd(id, benchmark)
            elif name == 'e3sm':
                self.run_e3sm(id, benchmark)
            elif name == 'macsio':
                self.run_macsio(id, benchmark)
            else:
                self.logger.critical('{} - Unsupported benchmark/kernel')

            self.logger.info('h5bench [{}] - Complete'.format(name))

        self.logger.info('Finishing h5bench Suite')

    def prepare_parallel(self, mpi):
        """Prepare the MPI launch command."""
        if 'configuration' in mpi:
            self.mpi = '{} {}'.format(mpi['command'], mpi['configuration'])
        else:
            if mpi['command'] in ['mpirun', 'mpiexec']:
                self.mpi = '{} -np {}'.format(mpi['command'], mpi['ranks'])
            elif mpi['command'] == 'srun':
                self.mpi = '{} --cpu_bind=cores -n {}'.format(mpi['command'], mpi['ranks'])
            else:
                self.logger.warning('Unknown MPI launcher selected!')

                self.mpi = ''

                return

        self.logger.info('Parallel setup: {}'.format(self.mpi))

    def prepare_vol(self, vol):
        """Prepare the environment variables for the VOL."""

        if vol is not None:
            if 'LD_LIBRARY_PATH' not in self.vol_environment:
                self.vol_environment['LD_LIBRARY_PATH'] = ''

            if 'DYLD_LIBRARY_PATH' not in self.vol_environment:
                self.vol_environment['DYLD_LIBRARY_PATH'] = ''

            if 'LD_PRELOAD' not in self.vol_environment:
                self.vol_environment['LD_PRELOAD'] = ''

            if 'library' in vol:
                self.vol_environment['LD_LIBRARY_PATH'] += ':' + vol['library']
                self.vol_environment['DYLD_LIBRARY_PATH'] += ':' + vol['library']
            if 'path' in vol:
                self.vol_environment['HDF5_PLUGIN_PATH'] = vol['path']
            if 'preload' in vol:
                self.vol_environment['LD_PRELOAD'] += vol['preload']

            self.vol_environment['ABT_THREAD_STACKSIZE'] = '100000'

            if 'HDF5_PLUGIN_PATH' in self.vol_environment:
                self.logger.debug('HDF5_PLUGIN_PATH: %s', self.vol_environment['HDF5_PLUGIN_PATH'])

        if 'LD_LIBRARY_PATH' in self.vol_environment:
            self.logger.debug('LD_LIBRARY_PATH: %s', self.vol_environment['LD_LIBRARY_PATH'])

        if 'DYLD_LIBRARY_PATH' in self.vol_environment:
            self.logger.debug('DYLD_LIBRARY_PATH: %s', self.vol_environment['DYLD_LIBRARY_PATH'])

        if 'LD_PRELOAD' in self.vol_environment:
            self.logger.debug('LD_PRELOAD: %s', self.vol_environment['LD_PRELOAD'])

    def enable_vol(self, vol):
        """Enable VOL by setting the connector."""
        if 'connector' in vol:
            self.vol_environment['HDF5_VOL_CONNECTOR'] = vol['connector']

            self.logger.info('HDF5 VOL connector: %s', vol['connector'])

    def disable_vol(self, vol):
        """Disable VOL by setting the connector."""
        if 'HDF5_VOL_CONNECTOR' in self.vol_environment:
            del self.vol_environment['HDF5_VOL_CONNECTOR']

    def reset_vol(self):
        """Reset the environment variables for the VOL."""
        if self.vol_environment is not None:
            if 'HDF5_PLUGIN_PATH' in self.vol_environment:
                del self.vol_environment['HDF5_PLUGIN_PATH']
            if 'HDF5_VOL_CONNECTOR' in self.vol_environment:
                del self.vol_environment['HDF5_VOL_CONNECTOR']

            if 'ABT_THREAD_STACKSIZE' in self.vol_environment:
                del self.vol_environment['ABT_THREAD_STACKSIZE']

    def check_for_hdf5_error(self, stderr_file_name):
        has_error_message = False

        with open(stderr_file_name, mode='r') as stderr_file:
            lines = stderr_file.readlines()

            for line in lines:
                if 'Error detected in HDF5' in line:
                    has_error_message = True

                    self.logger.error(line.strip())
                    self.logger.error('Check %s for detailed log', stderr_file_name)

                    sys.exit(os.EX_IOERR)

        return has_error_message

    def run_pattern(self, id, operation, setup, vol):
        """Run the h5bench_patterns (write/read) benchmarks."""
        try:
            start = time.time()

            # Define the output file (should be a .h5 file)
            file = '{}/{}'.format(self.directory, setup['file'])
            configuration = setup['configuration']

            # Disable any user-defined VOL connectors as we will be handling that
            self.disable_vol(vol)

            if configuration['MODE'] in ['ASYNC', 'LOG']:
                self.enable_vol(vol)

            configuration_file = '{}/{}/h5bench.cfg'.format(self.directory, id)

            # Create the configuration file for this benchmark
            with open(configuration_file, 'w+') as f:
                for key in configuration:
                    # Make sure the CSV file is generated in the temporary path
                    if key == 'CSV_FILE':
                        configuration[key] = '{}/{}/{}'.format(self.directory, id, configuration[key])

                    if key == 'MODE':
                        continue

                    f.write('{}={}\n'.format(key, configuration[key]))

                if operation == 'append':
                    f.write('IO_OPERATION=APPEND\n')

                if operation == 'overwrite':
                    f.write('IO_OPERATION=OVERWRITE\n')

            if operation == 'write':
                benchmark_path = self.H5BENCH_PATTERNS_WRITE

            if operation == 'write-unlimited':
                benchmark_path = self.H5BENCH_PATTERNS_WRITE_UNLIMITED

            if operation == 'write_var_normal_dist':
                benchmark_path = self.H5BENCH_PATTERNS_WRITE_VAR_NORMAL_DIST

            if operation == 'write_var_data_dist':
                benchmark_path = self.H5BENCH_PATTERNS_WRITE_VAR_DATA_DIST

            if operation == 'overwrite':
                benchmark_path = self.H5BENCH_PATTERNS_OVERWRITE

            if operation == 'append':
                benchmark_path = self.H5BENCH_PATTERNS_APPEND

            if operation == 'read':
                benchmark_path = self.H5BENCH_PATTERNS_READ

            if self.prefix:
                benchmark_path = self.prefix + '/' + benchmark_path
            else:
                if os.path.isfile(h5bench_configuration.__install__ + '/' + benchmark_path):
                    benchmark_path = h5bench_configuration.__install__ + '/' + benchmark_path
                else:
                    benchmark_path = benchmark_path

            command = '{} {} {} {}'.format(
                self.mpi,
                benchmark_path,
                configuration_file,
                file
            )

            self.logger.info(command)

            # Make sure the command line is in the correct format
            arguments = shlex.split(command)

            stdout_file_name = '{}/{}/stdout'.format(self.directory, id)
            stderr_file_name = '{}/{}/stderr'.format(self.directory, id)

            with open(stdout_file_name, mode='w') as stdout_file, open(stderr_file_name, mode='w') as stderr_file:
                s = subprocess.Popen(arguments, stdout=stdout_file, stderr=stderr_file, env=self.vol_environment)
                sOutput, sError = s.communicate()

                if s.returncode == 0 and not self.check_for_hdf5_error(stderr_file_name):
                    self.logger.info('SUCCESS (all output files are located at %s/%s)', self.directory, id)
                else:
                    self.logger.error('Return: %s (check %s for detailed log)', s.returncode, stderr_file_name)

                    if self.abort:
                        self.logger.critical('h5bench execution aborted upon first error')

                        sys.exit(os.EX_SOFTWARE)

            end = time.time()

            if configuration['MODE'] in ['ASYNC', 'LOG']:
                self.disable_vol(vol)

            if self.validate:
                used_async = False

                with open(stdout_file_name, mode='r') as stdout_file:
                    lines = stdout_file.readlines()

                    for line in lines:
                        if 'Mode: ASYNC' in line:
                            used_async = True
                            break

                if (configuration['MODE'] == 'ASYNC' and used_async) or (configuration['MODE'] == 'SYNC' and not used_async):
                    self.logger.info('Requested and ran in %s mode', 'ASYNC' if used_async else 'SYNC')
                else:
                    self.logger.warning('Requested %s mode but ran in %s mode', configuration['MODE'], 'ASYNC' if used_async else 'SYNC')

            self.logger.info('Runtime: {:.7f} seconds (elapsed time, includes allocation wait time)'.format(end - start))
        except Exception as e:
            self.logger.error('Unable to run the benchmark: %s', e)

            sys.exit(os.EX_SOFTWARE)

    def is_available(self, executable):
        """Check if binary is available."""
        return distutils.spawn.find_executable(
            executable,
            path=os.environ['PATH'] + ':.'
        )

    def run_exerciser(self, id, setup):
        """Run the exerciser benchmark."""
        if not self.is_available(self.H5BENCH_EXERCISER):
            self.logger.critical('{} is not available'.format(self.H5BENCH_EXERCISER))

            sys.exit(os.EX_UNAVAILABLE)

        try:
            start = time.time()

            configuration = setup['configuration']

            parameters = []

            parameters_binary = [
                'keepfile',
                'usechunked',
                'indepio',
                'addattr',
                'derivedtype'
            ]

            # Create the configuration parameter list
            for key in configuration:
                if key in parameters_binary:
                    if configuration[key].lower() == 'true':
                        parameters.append('--{} '.format(key))
                else:
                    parameters.append('--{} {} '.format(key, configuration[key]))

            if self.prefix:
                benchmark_path = self.prefix + '/' + self.H5BENCH_EXERCISER
            else:
                if os.path.isfile(h5bench_configuration.__install__ + '/' + self.H5BENCH_EXERCISER):
                    benchmark_path = h5bench_configuration.__install__ + '/' + self.H5BENCH_EXERCISER
                else:
                    benchmark_path = self.H5BENCH_EXERCISER

            command = '{} {} {}'.format(
                self.mpi,
                benchmark_path,
                ' '.join(parameters)
            )

            self.logger.info(command)

            # Make sure the command line is in the correct format
            arguments = shlex.split(command)

            stdout_file_name = '{}/{}/stdout'.format(self.directory, id)
            stderr_file_name = '{}/{}/stderr'.format(self.directory, id)

            with open(stdout_file_name, mode='w') as stdout_file, open(stderr_file_name, mode='w') as stderr_file:
                s = subprocess.Popen(arguments, stdout=stdout_file, stderr=stderr_file, env=self.vol_environment)
                sOutput, sError = s.communicate()

                if s.returncode == 0 and not self.check_for_hdf5_error(stderr_file_name):
                    self.logger.info('SUCCESS (all output files are located at %s/%s)', self.directory, id)
                else:
                    self.logger.error('Return: %s (check %s for detailed log)', s.returncode, stderr_file_name)

                    if self.abort:
                        self.logger.critical('h5bench execution aborted upon first error')

                        sys.exit(os.EX_SOFTWARE)

            end = time.time()

            self.logger.info('Runtime: {:.7f} seconds (elapsed time, includes allocation wait time)'.format(end - start))
        except Exception as e:
            self.logger.error('Unable to run the benchmark: %s', e)

            sys.exit(os.EX_SOFTWARE)

    def run_metadata(self, id, setup):
        """Run the metadata stress benchmark."""
        if not self.is_available(self.H5BENCH_METADATA):
            self.logger.critical('{} is not available'.format(self.H5BENCH_METADATA))

            sys.exit(os.EX_UNAVAILABLE)

        try:
            start = time.time()

            # Define the output file (should be a .h5 file)
            file = '{}/{}'.format(self.directory, setup['file'])
            configuration = setup['configuration']

            configuration_file = '{}/{}/hdf5_iotest.ini'.format(self.directory, id)

            # Create the configuration file for this benchmark
            with open(configuration_file, 'w+') as f:
                f.write('[DEFAULT]\n')

                for key in configuration:
                    # Make sure the CSV file is generated in the temporary path
                    if key == 'csv-file':
                        configuration[key] = '{}/{}/{}'.format(self.directory, id, configuration[key])

                    f.write('{} = {}\n'.format(key, configuration[key]))

                f.write('hdf5-file = {}\n'.format(file))

            if self.prefix:
                benchmark_path = self.prefix + '/' + self.H5BENCH_METADATA
            else:
                if os.path.isfile(h5bench_configuration.__install__ + '/' + self.H5BENCH_METADATA):
                    benchmark_path = h5bench_configuration.__install__ + '/' + self.H5BENCH_METADATA
                else:
                    benchmark_path = self.H5BENCH_METADATA

            command = '{} {} {}'.format(
                self.mpi,
                benchmark_path,
                configuration_file
            )

            self.logger.info(command)

            # Make sure the command line is in the correct format
            arguments = shlex.split(command)

            stdout_file_name = '{}/{}/stdout'.format(self.directory, id)
            stderr_file_name = '{}/{}/stderr'.format(self.directory, id)

            with open(stdout_file_name, mode='w') as stdout_file, open(stderr_file_name, mode='w') as stderr_file:
                s = subprocess.Popen(arguments, stdout=stdout_file, stderr=stderr_file, env=self.vol_environment)
                sOutput, sError = s.communicate()

                if s.returncode == 0 and not self.check_for_hdf5_error(stderr_file_name):
                    self.logger.info('SUCCESS (all output files are located at %s/%s)', self.directory, id)
                else:
                    self.logger.error('Return: %s (check %s for detailed log)', s.returncode, stderr_file_name)

                    if self.abort:
                        self.logger.critical('h5bench execution aborted upon first error')

                        sys.exit(os.EX_SOFTWARE)

            end = time.time()

            self.logger.info('Runtime: {:.7f} seconds (elapsed time, includes allocation wait time)'.format(end - start))
        except Exception as e:
            self.logger.error('Unable to run the benchmark: %s', e)

            sys.exit(os.EX_SOFTWARE)

    def run_amrex(self, id, setup, vol):
        """Run the AMReX benchmark."""
        if not self.is_available(self.H5BENCH_AMREX_SYNC):
            self.logger.critical('{} is not available'.format(self.H5BENCH_AMREX_SYNC))

            sys.exit(os.EX_UNAVAILABLE)

        try:
            start = time.time()

            directory = '{}/{}/{}'.format(self.directory, id, setup['file'])
            configuration = setup['configuration']

            # Disable any user-defined VOL connectors as we will be handling that
            self.disable_vol(vol)

            if configuration['mode'] in ['ASYNC', 'LOG']:
                self.enable_vol(vol)

                binary = self.H5BENCH_AMREX_ASYNC
            else:
                binary = self.H5BENCH_AMREX_SYNC

            configuration_file = '{}/{}/amrex.ini'.format(self.directory, id)

            try:
                # Create a temporary directory to store all configurations
                os.makedirs(directory)
            except OSError as exc:
                if exc.errno != errno.EEXIST:
                    raise

                self.logger.warning('Base directory already exists: {}'.format(self.directory))

                pass
            except Exception as e:
                self.logger.debug('Unable to create {}: {}'.format(self.directory, e))

            # Create the configuration file for this benchmark
            with open(configuration_file, 'w+') as f:
                for key in configuration:
                    f.write('{} = {}\n'.format(key, configuration[key]))

                # f.write('directory = {}\n'.format(directory))

            if self.prefix:
                benchmark_path = self.prefix + '/' + self.binary
            else:
                if os.path.isfile(h5bench_configuration.__install__ + '/' + binary):
                    benchmark_path = h5bench_configuration.__install__ + '/' + binary
                else:
                    benchmark_path = binary

            command = '{} {} {}'.format(
                self.mpi,
                benchmark_path,
                configuration_file
            )

            self.logger.info(command)

            # Make sure the command line is in the correct format
            arguments = shlex.split(command)

            stdout_file_name = '{}/{}/stdout'.format(self.directory, id)
            stderr_file_name = '{}/{}/stderr'.format(self.directory, id)

            with open(stdout_file_name, mode='w') as stdout_file, open(stderr_file_name, mode='w') as stderr_file:
                s = subprocess.Popen(arguments, stdout=stdout_file, stderr=stderr_file, env=self.vol_environment)
                sOutput, sError = s.communicate()

                if s.returncode == 0 and not self.check_for_hdf5_error(stderr_file_name):
                    self.logger.info('SUCCESS (all output files are located at %s/%s)', self.directory, id)
                else:
                    self.logger.error('Return: %s (check %s for detailed log)', s.returncode, stderr_file_name)

                    if self.abort:
                        self.logger.critical('h5bench execution aborted upon first error')

                        sys.exit(os.EX_SOFTWARE)

            end = time.time()

            if configuration['mode'] in ['ASYNC', 'LOG']:
                self.disable_vol(vol)

            self.logger.info('Runtime: {:.7f} seconds (elapsed time, includes allocation wait time)'.format(end - start))
        except Exception as e:
            self.logger.error('Unable to run the benchmark: %s', e)

            sys.exit(os.EX_SOFTWARE)

    def run_openpmd(self, id, setup):
        """Run the OpenPMD kernel benchmark."""
        if not self.is_available(self.H5BENCH_OPENPMD_WRITE):
            self.logger.critical('{} is not available'.format(self.H5BENCH_OPENPMD_WRITE))

            sys.exit(os.EX_UNAVAILABLE)

        if not self.is_available(self.H5BENCH_OPENPMD_READ):
            self.logger.critical('{} is not available'.format(self.H5BENCH_OPENPMD_READ))

            sys.exit(os.EX_UNAVAILABLE)

        try:
            start = time.time()

            # Define the output file (should be a .h5 file)
            if 'file' in setup:
                self.logger.warning('OpenPMD does not take an input file name, only the directory')

            configuration = setup['configuration']

            configuration_file = '{}/{}/openpmd.input'.format(self.directory, id)

            # Create the configuration file for this benchmark
            with open(configuration_file, 'w+') as f:
                for key in configuration:
                    if 'operation' in key:
                        continue

                    f.write('{}={}\n'.format(key, configuration[key]))

                f.write('fileLocation={}\n'.format(self.directory))

            if configuration['operation'] == 'write':
                binary = self.H5BENCH_OPENPMD_WRITE

                if self.prefix:
                    benchmark_path = self.prefix + '/' + binary
                else:
                    if os.path.isfile(h5bench_configuration.__install__ + '/' + binary):
                        benchmark_path = h5bench_configuration.__install__ + '/' + binary
                    else:
                        benchmark_path = binary

                command = '{} {} {}'.format(
                    self.mpi,
                    benchmark_path,
                    configuration_file
                )
            elif configuration['operation'] == 'read':
                binary = self.H5BENCH_OPENPMD_READ

                if self.prefix:
                    benchmark_path = self.prefix + '/' + binary
                else:
                    if os.path.isfile(h5bench_configuration.__install__ + '/' + binary):
                        benchmark_path = h5bench_configuration.__install__ + '/' + binary
                    else:
                        benchmark_path = binary

                file_path = '{}/8a_parallel_3Db'.format(self.directory)

                command = '{} {} {} {}'.format(
                    self.mpi,
                    benchmark_path,
                    file_path,
                    configuration['pattern']
                )
            else:
                self.logger.error('Unsupported operation for OpenPMD benchmark')

                sys.exit(os.EX_SOFTWARE)

            self.logger.info(command)

            # Make sure the command line is in the correct format
            arguments = shlex.split(command)

            stdout_file_name = '{}/{}/stdout'.format(self.directory, id)
            stderr_file_name = '{}/{}/stderr'.format(self.directory, id)

            with open(stdout_file_name, mode='w') as stdout_file, open(stderr_file_name, mode='w') as stderr_file:
                s = subprocess.Popen(arguments, stdout=stdout_file, stderr=stderr_file, env=self.vol_environment)
                sOutput, sError = s.communicate()

                if s.returncode == 0 and not self.check_for_hdf5_error(stderr_file_name):
                    self.logger.info('SUCCESS (all output files are located at %s/%s)', self.directory, id)
                else:
                    self.logger.error('Return: %s (check %s for detailed log)', s.returncode, stderr_file_name)

                    if self.abort:
                        self.logger.critical('h5bench execution aborted upon first error')

                        sys.exit(os.EX_SOFTWARE)

            end = time.time()

            self.logger.info('Runtime: {:.7f} seconds (elapsed time, includes allocation wait time)'.format(end - start))
        except Exception as e:
            self.logger.error('Unable to run the benchmark: %s', e)

    def run_e3sm(self, id, setup):
        """Run the E3SM benchmark."""
        if not self.is_available(self.H5BENCH_E3SM):
            self.logger.critical('{} is not available'.format(self.H5BENCH_E3SM))

            sys.exit(os.EX_UNAVAILABLE)

        try:
            start = time.time()

            configuration = setup['configuration']

            parameters = []

            # Create the configuration parameter list
            for key in configuration:
                if key not in ['i', 'o', 'map'] and configuration[key]:
                    parameters.append('-{} {} '.format(key, configuration[key]))

            # Temporarily overwrite -x and -a to only supported patterns
            parameters.append('-{} {}'.format('a', 'hdf5'))
            parameters.append('-{} {}'.format('x', 'blob'))

            parameters.append('-o {}/{}/{} '.format(self.directory, id, setup['file']))

            file = '{}/{}'.format(self.directory, configuration['map'])

            if self.prefix:
                benchmark_path = self.prefix + '/' + self.H5BENCH_E3SM
            else:
                if os.path.isfile(h5bench_configuration.__install__ + '/' + self.H5BENCH_E3SM):
                    benchmark_path = h5bench_configuration.__install__ + '/' + self.H5BENCH_E3SM
                else:
                    benchmark_path = self.H5BENCH_E3SM

            command = '{} {} {} {}'.format(
                self.mpi,
                benchmark_path,
                ' '.join(parameters),
                file
            )

            self.logger.info(command)

            # Make sure the command line is in the correct format
            arguments = shlex.split(command)

            stdout_file_name = '{}/{}/stdout'.format(self.directory, id)
            stderr_file_name = '{}/{}/stderr'.format(self.directory, id)

            with open(stdout_file_name, mode='w') as stdout_file, open(stderr_file_name, mode='w') as stderr_file:
                s = subprocess.Popen(arguments, stdout=stdout_file, stderr=stderr_file, env=self.vol_environment)
                sOutput, sError = s.communicate()

                if s.returncode == 0 and not self.check_for_hdf5_error(stderr_file_name):
                    self.logger.info('SUCCESS (all output files are located at %s/%s)', self.directory, id)
                else:
                    self.logger.error('Return: %s (check %s for detailed log)', s.returncode, stderr_file_name)

                    if self.abort:
                        self.logger.critical('h5bench execution aborted upon first error')

                        sys.exit(os.EX_SOFTWARE)

            end = time.time()

            self.logger.info('Runtime: {:.7f} seconds (elapsed time, includes allocation wait time)'.format(end - start))
        except Exception as e:
            self.logger.error('Unable to run the benchmark: %s', e)

    def run_macsio(self, id, setup):
        """Run the MACSIO benchmark."""
        if not self.is_available(self.H5BENCH_MACSIO):
            self.logger.critical('{} is not available'.format(self.H5BENCH_MACSIO))

            sys.exit(os.EX_UNAVAILABLE)

        try:
            start = time.time()

            configuration = setup['configuration']

            parameters = []

            # Create the configuration parameter list
            for key in configuration:
                if key not in ['filebase', 'interface'] and configuration[key]:
                    parameters.append('--{} {} '.format(key, configuration[key]))

            parameters.append('--interface {} '.format('hdf5'))
            parameters.append('--filebase {}/{}/{} '.format(self.directory, id, setup['file'].replace('.h5', '')))
            parameters.append('--log_file_name {}/{}/macsio.log '.format(self.directory, id))
            parameters.append('--timings_file_name {}/{}/timings.log '.format(self.directory, id))

            if self.prefix:
                benchmark_path = self.prefix + '/' + self.H5BENCH_MACSIO
            else:
                if os.path.isfile(h5bench_configuration.__install__ + '/' + self.H5BENCH_MACSIO):
                    benchmark_path = h5bench_configuration.__install__ + '/' + self.H5BENCH_MACSIO
                else:
                    benchmark_path = self.H5BENCH_MACSIO

            command = '{} {} {}'.format(
                self.mpi,
                benchmark_path,
                ' '.join(parameters)
            )

            self.logger.info(command)

            # Make sure the command line is in the correct format
            arguments = shlex.split(command)

            stdout_file_name = '{}/{}/stdout'.format(self.directory, id)
            stderr_file_name = '{}/{}/stderr'.format(self.directory, id)

            with open(stdout_file_name, mode='w') as stdout_file, open(stderr_file_name, mode='w') as stderr_file:
                s = subprocess.Popen(arguments, stdout=stdout_file, stderr=stderr_file, env=self.vol_environment)
                sOutput, sError = s.communicate()

                if s.returncode == 0 and not self.check_for_hdf5_error(stderr_file_name):
                    self.logger.info('SUCCESS (all output files are located at %s/%s)', self.directory, id)
                else:
                    self.logger.error('Return: %s (check %s for detailed log)', s.returncode, stderr_file_name)

                    if self.abort:
                        self.logger.critical('h5bench execution aborted upon first error')

                        sys.exit(os.EX_SOFTWARE)

                # Move the files if they were generated
                if os.path.isfile('{}-macsio-log.log'.format(id)):
                    os.rename('{}-macsio-log.log'.format(id), '{}/{}/macsio-log.log'.format(self.directory, id))

                if os.path.isfile('{}-macsio-timings.log'.format(id)):
                    os.rename('{}-macsio-timings.log'.format(id), '{}/{}/macsio-timings.log'.format(self.directory, id))

            end = time.time()

            self.logger.info('Runtime: {:.7f} seconds (elapsed time, includes allocation wait time)'.format(end - start))
        except Exception as e:
            self.logger.error('Unable to run the benchmark: %s', e)


def main():
    PARSER = argparse.ArgumentParser(
        description='H5bench: a Parallel I/O Benchmark Suite for HDF5: '
    )

    PARSER.add_argument(
        'setup',
        action='store',
        help='JSON file with the benchmarks to run'
    )

    PARSER.add_argument(
        '-a',
        '--abort-on-failure',
        action='store_true',
        dest='abort',
        help='Stop h5bench if a benchmark failed'
    )

    PARSER.add_argument(
        '-d',
        '--debug',
        action='store_true',
        dest='debug',
        help='Enable debug mode'
    )

    PARSER.add_argument(
        '-v',
        '--validate-mode',
        action='store_true',
        dest='validate',
        help='Validated if the requested mode (async/sync) was run'
    )

    PARSER.add_argument(
        '-p',
        '--prefix',
        action='store',
        dest='prefix',
        help='Prefix where all h5bench binaries were installed'
    )

    PARSER.add_argument(
        '-f',
        '--filter',
        action='store',
        dest='filter',
        help='Execute only filtered benchmarks'
    )

    PARSER.add_argument(
        '-V',
        '--version',
        action='version',
        version='%(prog)s ' + h5bench_version.__version__
    )

    ARGS = PARSER.parse_args()

    BENCH = H5bench(ARGS.setup, ARGS.prefix, ARGS.debug, ARGS.abort, ARGS.validate, ARGS.filter)
    BENCH.run()


if __name__ == '__main__':
    main()
