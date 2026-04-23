#!/usr/bin/env python3

import glob
import os

import pytest

from src import h5bench
from tests.helpers.build_config import requires_amrex

DEBUG = True
ABORT = True
VALIDATE = True

BINARY = 'h5bench_amrex_sync'

samples = glob.glob('sync-amrex*.json')


@pytest.mark.parametrize('configuration', samples)
@requires_amrex
def test_benchmark(configuration):
	# The CMake option says AMReX was enabled, so the binary must exist.
	# If it doesn't, that's a build regression — fail loudly rather than
	# quietly skipping.
	assert os.path.isfile(BINARY), (
		f"H5BENCH_AMREX=ON but {BINARY!r} is not in the build dir"
	)
	assert os.path.isfile(configuration) is True

	benchmark = h5bench.H5bench(
		configuration,
		None,
		DEBUG,
		ABORT,
		VALIDATE
	)

	benchmark.run()
