#!/usr/bin/env python3

import glob
import os

import pytest

from src import h5bench
from tests.helpers.build_config import requires_exerciser

DEBUG = True
ABORT = True
VALIDATE = True

BINARY = 'h5bench_exerciser'

samples = glob.glob('sync-exerciser*.json')


@pytest.mark.parametrize('configuration', samples)
@requires_exerciser
def test_benchmark(configuration):
	assert os.path.isfile(BINARY), (
		f"H5BENCH_EXERCISER=ON but {BINARY!r} is not in the build dir"
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
