#!/usr/bin/env python3

import glob
import os

import pytest

from src import h5bench
from tests.helpers.build_config import requires_async_vol

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
@requires_async_vol
def test_benchmark(configuration):
	# Pattern binaries are built unconditionally; ASYNC mode needs the
	# VOL-ASYNC connector to actually run asynchronously. We gate on the
	# build flag and then assert the binaries exist (any missing one here
	# is a build regression, not a configuration choice).
	for binary in (BINARY_WRITE, BINARY_APPEND, BINARY_OVERWRITE, BINARY_UNLIMITED):
		assert os.path.isfile(binary), (
			f"Pattern binary {binary!r} is missing from the build dir"
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
