#!/usr/bin/env python3

import glob
import os

import pytest

from src import h5bench
from tests.helpers.build_config import requires_openpmd

DEBUG = True
ABORT = True
VALIDATE = True

BINARY_WRITE = 'h5bench_openpmd_write'
BINARY_READ = 'h5bench_openpmd_read'

samples = glob.glob('sync-openpmd*.json')


@pytest.mark.parametrize('configuration', samples)
@requires_openpmd
def test_benchmark(configuration):
	# OpenPMD was enabled — both binaries should have been produced.
	assert os.path.isfile(BINARY_WRITE), (
		f"H5BENCH_OPENPMD=ON but {BINARY_WRITE!r} is not in the build dir"
	)
	assert os.path.isfile(BINARY_READ), (
		f"H5BENCH_OPENPMD=ON but {BINARY_READ!r} is not in the build dir"
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
