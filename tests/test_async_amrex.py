#!/usr/bin/env python3

import glob
import os

import pytest

from src import h5bench
from tests.helpers.build_config import requires_amrex, requires_async_vol

DEBUG = True
ABORT = True
VALIDATE = True

BINARY = 'h5bench_amrex_async'

samples = glob.glob('async-amrex*.json')


@pytest.mark.parametrize('configuration', samples)
@requires_amrex
@requires_async_vol
def test_benchmark(configuration):
	# Both H5BENCH_AMREX and WITH_ASYNC_VOL were ON at configure time;
	# a missing binary here means the build produced inconsistent output.
	assert os.path.isfile(BINARY), (
		f"H5BENCH_AMREX and WITH_ASYNC_VOL are ON but {BINARY!r} "
		f"is not in the build dir"
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
