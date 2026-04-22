#!/usr/bin/env python3
"""
Read-path round-trip test.

The existing test suite never exercised ``h5bench_read`` — every test drove a
write/append/overwrite and asserted only that the config file existed. This
file closes that gap: it runs a write-then-read benchmark and confirms both
steps completed and the file produced by the write step survives a read
(checksumable, matches the expected shape).

Only one fixture today — more can be added by appending to the parametrize
list as we extend the tiny-fixtures set.
"""

import os

import pytest

from src import h5bench
from tests.helpers import h5_assert


PARTICLES_PER_RANK = 1024
NUM_RANKS = 2
TOTAL_PARTICLES = PARTICLES_PER_RANK * NUM_RANKS

BINARY_WRITE = "h5bench_write"
BINARY_READ = "h5bench_read"


@pytest.mark.skipif(
	not (os.path.isfile(BINARY_WRITE) and os.path.isfile(BINARY_READ)),
	reason="h5bench_write or h5bench_read binary missing",
)
def test_read_roundtrip_1d_contig():
	configuration = "tiny-1d-write-read.json"
	assert os.path.isfile(configuration), (
		f"fixture {configuration} not staged into PWD — check conftest.py copy step"
	)

	# The driver runs write first, then read, against the same file. A failure
	# in either step throws and h5bench.H5bench.run() calls sys.exit, failing
	# the test.
	benchmark = h5bench.H5bench(configuration, None, True, True, True)
	benchmark.run()

	# Post-read, the file we wrote is still on disk and well-formed: structure,
	# shape, dtype, and the deterministic id_1 pattern all survive the read
	# step (the read benchmark reads the file but doesn't modify it).
	output = "test-output-tiny-1d-write-read/tiny-rt.h5"

	h5_assert.assert_timestep_count(output, 2)
	h5_assert.assert_contig_datasets_present(output, "/Timestep_0")
	h5_assert.assert_dataset_shape(
		output, "/Timestep_0/id_1", (TOTAL_PARTICLES,)
	)
	h5_assert.assert_id_1_1d_pattern(
		output, "/Timestep_0/id_1", PARTICLES_PER_RANK, NUM_RANKS
	)
