# Copyright 2013-2022 Lawrence Livermore National Security, LLC and other
# Spack Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

from spack import *


class H5bench(CMakePackage):
    """A benchmark suite for measuring HDF5 performance."""

    homepage = 'https://github.com/hpc-io/h5bench'
    git      = 'https://github.com/hpc-io/h5bench.git'

    maintainers = ['jeanbez', 'sbyna']

    version('master', branch='master', submodules=True)
    version('develop', branch='develop', submodules=True)

    version('1.2', tag='1.2', submodules=True)
    version('1.1', tag='1.1', submodules=True, depecrated=True)
    version('1.0', tag='1.0', submodules=True, depecrated=True)

    variant('async', default=False, description='Enable the HDF5 VOL-ASYNC connector support')

    depends_on('cmake@3.10:', type='build')
    depends_on('mpi')
    depends_on('hdf5@develop-1.13 +mpi +threadsafe')
    depends_on('hdf5-vol-async@test', when='+async')

    def setup_build_environment(self, env):
        env.set('HDF5_HOME', self.spec['hdf5'].prefix)
        
        if '+async' in self.spec:
            env.set('ASYNC_HOME', self.spec['hdf5-vol-async'].prefix)

    def setup_run_environment(self, env):
        env.set('HDF5_HOME', self.spec['hdf5'].prefix)

        if '+async' in self.spec:
            env.set('ASYNC_HOME', self.spec['hdf5-vol-async'].prefix)
            env.set('HDF5_PLUGIN_PATH', self.spec['hdf5-vol-async'].prefix.lib)

    def cmake_args(self):
        args = []

        if '+async' in self.spec:
            args.append('-DWITH_ASYNC_VOL:BOOL=ON')

        args.append('-DCMAKE_C_COMPILER=h5pcc')

        return args