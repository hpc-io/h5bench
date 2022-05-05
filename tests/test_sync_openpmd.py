#!/usr/bin/env python3

import os
import glob
import pytest

from src import h5bench

DEBUG = True
ABORT = True
VALIDATE = True

PREFIX = '../build'
BINARY_WRITE = 'h5bench_openpmd_write'
BINARY_READ = 'h5bench_openpmd_read'

samples = glob.glob('sync-openpmd*.json')

@pytest.mark.parametrize('configuration', samples)
@pytest.mark.skipif(
	os.path.isfile('{}/{}'.format(PREFIX, BINARY_WRITE)) == False or
	os.path.isfile('{}/{}'.format(PREFIX, BINARY_READ)) == False,
	reason="OpenPMD is disabled"
)
def test_benchmark(configuration):
	assert os.path.isfile(configuration) is True

	benchmark = h5bench.H5bench(
		configuration,
		PREFIX,
		DEBUG,
		ABORT,
		VALIDATE
	)

	benchmark.run()
