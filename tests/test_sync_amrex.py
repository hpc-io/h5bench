#!/usr/bin/env python3

import os
import glob
import pytest

from src import h5bench

DEBUG = True
ABORT = True
VALIDATE = True

BINARY = 'h5bench_amrex_sync'

samples = glob.glob('sync-amrex*.json')

@pytest.mark.parametrize('configuration', samples)
@pytest.mark.skipif(
	os.path.isfile(BINARY) == False,
	reason="AMReX (SYNC) is disabled"
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
