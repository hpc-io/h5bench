#!/usr/bin/env python3

import os
import glob
import pytest

from src import h5bench

DEBUG = True
ABORT = True
VALIDATE = True

PREFIX = '../build'
BINARY = 'h5bench_amrex_sync'

samples = glob.glob('samples/async-amrex*.json')

@pytest.mark.parametrize('configuration', samples)
@pytest.mark.skipif(
	os.path.isfile('{}/{}'.format(PREFIX, BINARY)) == False,
	reason="AMReX (SYNC) is disabled"
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
