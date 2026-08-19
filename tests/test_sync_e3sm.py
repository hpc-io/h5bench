#!/usr/bin/env python3

import glob
import os

import pytest

from src import h5bench
from tests.helpers.build_config import requires_e3sm

DEBUG = True
ABORT = True
VALIDATE = True

BINARY = 'h5bench_e3sm'

# Production samples live at the top of the build dir after conftest's copy.
samples = glob.glob('sync-e3sm*.json')


@pytest.mark.parametrize('configuration', samples)
@requires_e3sm
def test_benchmark(configuration):
	# The E3SM wrapper (ExternalProject_Add) places the binary in the build
	# root via its INSTALL_COMMAND + the root CMakeLists install(PROGRAMS ...)
	# rule; if H5BENCH_E3SM=ON, the binary must be there.
	assert os.path.isfile(BINARY), (
		f"H5BENCH_E3SM=ON but {BINARY!r} is not in the build dir"
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
