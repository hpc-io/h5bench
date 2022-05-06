#!/usr/bin/env python3

import os
import glob
import pytest

from src import h5bench

DEBUG = True
ABORT = True
VALIDATE = True

BINARY_WRITE = 'h5bench_write'
BINARY_APPEND = 'h5bench_append'
BINARY_OVERWRITE = 'h5bench_overwrite'
BINARY_UNLIMITED = 'h5bench_write_unlimited'

samples = \
	glob.glob('async-write-*d-*.json') + \
	glob.glob('async-append*.json') + \
	glob.glob('async-overwrite*.json') + \
	glob.glob('async-write-unlimited*.json')

@pytest.mark.parametrize('configuration', samples)
@pytest.mark.skipif(
	os.path.isfile(BINARY_WRITE) == False or
	os.path.isfile(BINARY_APPEND) == False or
	os.path.isfile(BINARY_OVERWRITE) == False or
	os.path.isfile(BINARY_UNLIMITED) == False,
	reason="Benchmarks (ASYNC) are disabled"
)
def test_benchmark(configuration):
	assert os.path.isfile(configuration) is True

	benchmark = h5bench.H5bench(
		configuration,
		None,
		DEBUG,
		ABORT,
		VALIDATE
	)

	benchmark.run()
