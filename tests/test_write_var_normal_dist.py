#!/usr/bin/env python3
"""
Output-correctness test for ``h5bench_write_var_normal_dist``.

The existing ``test_sync_h5bench_normal_dist.py`` was misconfigured (it
probes for a binary name ``h5bench_write_normal_dist`` that no CMake target
produces, and globs a sample filename that does not exist), so the variable-
distribution benchmark has had zero pytest coverage. This module replaces
it with a working minimum-viable test that drives the binary via the Python
driver and confirms the output has the expected shape and dataset set.

The fixture ``tiny-write-var-normal-dist.json`` uses DIM_1=1024,
STDEV_DIM_1=128 so each rank draws a particle count around 1024. Because
the count is per-rank random, the total file extent can vary run-to-run;
we assert shape only loosely (within ±4σ of 2 * DIM_1) and focus on
structure instead.
"""

import os

import pytest

from src import h5bench
from tests.helpers import h5_assert


BINARY = "h5bench_write_var_normal_dist"
MEAN_PER_RANK = 1024
STDEV_PER_RANK = 128
NUM_RANKS = 2


@pytest.mark.skipif(
	not os.path.isfile(BINARY),
	reason="h5bench_write_var_normal_dist binary missing",
)
def test_write_var_normal_dist_structure():
	configuration = "tiny-write-var-normal-dist.json"
	assert os.path.isfile(configuration), (
		f"fixture {configuration} not staged into PWD — check conftest.py copy step"
	)

	benchmark = h5bench.H5bench(configuration, None, True, True, True)
	benchmark.run()

	output = "test-output-tiny-write-var-normal-dist/tiny-var-normal.h5"

	# Structure: one Timestep group, eight datasets under it (same layout as
	# regular 1D contig write — the distribution only affects the per-rank
	# particle count, not the dataset names or shapes' dimensionality).
	h5_assert.assert_timestep_count(output, 1)
	h5_assert.assert_contig_datasets_present(output, "/Timestep_0")

	# The file extent is the sum of per-rank particle counts (each drawn from
	# N(MEAN_PER_RANK, STDEV_PER_RANK^2)). With 2 ranks, 4-sigma-around-mean
	# gives a very generous envelope of [mean - 4σ, mean + 4σ] per rank, so
	# the file axis-0 extent should land inside 2*(mean ± 4σ).
	expected_lo = NUM_RANKS * max(0, MEAN_PER_RANK - 4 * STDEV_PER_RANK)
	expected_hi = NUM_RANKS * (MEAN_PER_RANK + 4 * STDEV_PER_RANK)
	import h5py
	with h5py.File(output, "r") as f:
		actual = f["/Timestep_0/id_1"].shape[0]
	assert expected_lo <= actual <= expected_hi, (
		f"variable-count file extent {actual} outside 4σ envelope "
		f"[{expected_lo}, {expected_hi}]"
	)
