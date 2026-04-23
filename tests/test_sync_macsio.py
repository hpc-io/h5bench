#!/usr/bin/env python3

import glob
import os

import pytest

from src import h5bench
from tests.helpers.build_config import requires_macsio

DEBUG = True
ABORT = True
VALIDATE = True

BINARY = 'h5bench_macsio'

samples = glob.glob('sync-macsio*.json')


@pytest.mark.parametrize('configuration', samples)
@requires_macsio
def test_benchmark(configuration):
	assert os.path.isfile(BINARY), (
		f"H5BENCH_MACSIO=ON but {BINARY!r} is not in the build dir"
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
