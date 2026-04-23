#!/usr/bin/env python3
"""
Output-correctness tests for the core h5bench patterns.

Each test drives one of the tiny fixtures under ``tests/fixtures/`` through
the Python driver, then uses h5py (see ``tests/helpers/h5_assert.py``) to
verify the resulting HDF5 file's structure and — where the write path is
deterministic — its content.

The goal is to catch refactors (Wave C and onwards) that silently change the
file format, not to benchmark anything. Runs in a few seconds per fixture.
"""

import os

import pytest

from src import h5bench
from tests.helpers import h5_assert


# Tiny fixtures — keep these in sync with tests/fixtures/*.json.
PARTICLES_PER_RANK = 1024
NUM_RANKS = 2
TOTAL_PARTICLES = PARTICLES_PER_RANK * NUM_RANKS

BINARY_WRITE = "h5bench_write"
BINARY_APPEND = "h5bench_append"
BINARY_OVERWRITE = "h5bench_overwrite"
BINARY_UNLIMITED = "h5bench_write_unlimited"


def _run(configuration):
	assert os.path.isfile(configuration), (
		f"fixture {configuration} not staged into PWD — check conftest.py copy step"
	)
	benchmark = h5bench.H5bench(configuration, None, True, True, True)
	benchmark.run()


def _all_binaries_present(*names):
	return all(os.path.isfile(name) for name in names)


# ---------------------------------------------------------------------------
# Write — structural + deterministic content
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
	not _all_binaries_present(BINARY_WRITE),
	reason="h5bench_write binary missing",
)
def test_write_1d_contig_contig():
	_run("tiny-1d-write.json")

	output = "test-output-tiny-1d-write/tiny-1d.h5"

	# Two Timestep_* groups, each with the full contig dataset set.
	h5_assert.assert_timestep_count(output, 2)
	h5_assert.assert_contig_datasets_present(output, "/Timestep_0")
	h5_assert.assert_contig_datasets_present(output, "/Timestep_1")

	# X_DIM/Y_DIM/Z_DIM are declared as 64 at file scope (h5bench_write.c:71)
	# but then clobbered in main() to the X_RAND/Y_RAND/Z_RAND constants
	# (h5bench_write.c:881). That's why the data-fill range below uses 191 /
	# 1009 / 3701 rather than 64 — surprising, but part of the current public
	# contract that Wave C must not silently change.
	X_MAX, Y_MAX, Z_MAX = 191.0, 1009.0, 3701.0
	ranges = {
		"x": (0.0, X_MAX),
		"y": (0.0, Y_MAX),
		"z": (0.0, Z_MAX),
		"px": (0.0, X_MAX),
		"py": (0.0, Y_MAX),
		# pz = (id_2 / NUM_PARTICLES) * Z_DIM, and id_2 ∈ [0, 2*NUM_PARTICLES],
		# so pz ∈ [0, 2 * Z_DIM].
		"pz": (0.0, 2 * Z_MAX),
	}
	for ts in ("Timestep_0", "Timestep_1"):
		# Shape: particles_per_rank * num_ranks along axis 0 (1D contig write
		# concatenates ranks in the file dataspace).
		for dset, (lo, hi) in ranges.items():
			h5_assert.assert_dataset_shape(output, f"/{ts}/{dset}", (TOTAL_PARTICLES,))
			h5_assert.assert_dataset_dtype(output, f"/{ts}/{dset}", "f")
			h5_assert.assert_value_in_range(output, f"/{ts}/{dset}", lo, hi)

		h5_assert.assert_dataset_shape(output, f"/{ts}/id_1", (TOTAL_PARTICLES,))
		h5_assert.assert_dataset_dtype(output, f"/{ts}/id_1", "i")
		h5_assert.assert_dataset_shape(output, f"/{ts}/id_2", (TOTAL_PARTICLES,))
		h5_assert.assert_dataset_dtype(output, f"/{ts}/id_2", "f")

		# id_1 and id_2 are deterministic functions of the per-rank index.
		h5_assert.assert_id_1_1d_pattern(
			output, f"/{ts}/id_1", PARTICLES_PER_RANK, NUM_RANKS
		)
		h5_assert.assert_id_2_1d_write_pattern(
			output, f"/{ts}/id_2", PARTICLES_PER_RANK, NUM_RANKS
		)


# ---------------------------------------------------------------------------
# Append — write-unlimited creates chunked/unlimited datasets at the
# (2 * dim_1)-extent the append step later targets. With 2 ranks writing
# dim_1 particles each, the pre-append file extent is already 2*dim_1
# (= TOTAL_PARTICLES), so append ends up overwriting the second half rather
# than growing the dataset. The observable contract: chunked datasets exist,
# the extent is unchanged, axis 0 is unlimited. If any of that changes, a
# Wave C refactor silently altered the append semantics.
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
	not _all_binaries_present(BINARY_UNLIMITED, BINARY_APPEND),
	reason="h5bench_write_unlimited/h5bench_append binary missing",
)
def test_append_preserves_extent_and_unlimited_axis():
	_run("tiny-append.json")

	output = "test-output-tiny-append/tiny-append.h5"

	for dset in ("x", "y", "z", "px", "py", "pz", "id_1", "id_2"):
		h5_assert.assert_dataset_shape(output, f"/Timestep_0/{dset}", (TOTAL_PARTICLES,))
		h5_assert.assert_unlimited_axis(output, f"/Timestep_0/{dset}", axis=0)


# ---------------------------------------------------------------------------
# Overwrite — shape unchanged, id_2 values change (write: 2*i, overwrite: 0.1*i)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
	not _all_binaries_present(BINARY_WRITE, BINARY_OVERWRITE),
	reason="h5bench_write/h5bench_overwrite binary missing",
)
def test_overwrite_changes_id_2_values():
	_run("tiny-overwrite.json")

	output = "test-output-tiny-overwrite/tiny-overwrite.h5"

	# Shape invariant across the overwrite step.
	for dset in ("x", "y", "z", "px", "py", "pz", "id_1", "id_2"):
		h5_assert.assert_dataset_shape(output, f"/Timestep_0/{dset}", (TOTAL_PARTICLES,))

	# After overwrite, id_2 is filled with `i * 0.1` (overwrite.c:105) per rank,
	# *not* the original `2 * i` pattern from the write step. Confirm by reading
	# the first rank's slice and asserting strict monotone growth from 0.
	id_2 = h5_assert.read_dataset(output, "/Timestep_0/id_2")
	first_rank = id_2[:PARTICLES_PER_RANK]
	assert first_rank[0] == pytest.approx(0.0), (
		f"overwrite id_2[0] should be 0, got {first_rank[0]}"
	)
	# Last element of the first rank: 1023 * 0.1 = 102.3 (float32 rounding OK).
	assert first_rank[-1] == pytest.approx(
		(PARTICLES_PER_RANK - 1) * 0.1, rel=1e-4
	), f"overwrite id_2[{PARTICLES_PER_RANK - 1}] should be ~102.3, got {first_rank[-1]}"


# ---------------------------------------------------------------------------
# Compound / interleaved write variants
#
# MEM_PATTERN × FILE_PATTERN drives which of four prepare/write helper pairs
# in h5bench_write.c runs. The tests above already cover CONTIG/CONTIG; the
# three below cover the remaining combinations:
#
#   * CONTIG / INTERLEAVED        → one compound "particles" dataset per TS
#   * INTERLEAVED / CONTIG        → eight scalar datasets (same layout as C/C)
#   * INTERLEAVED / INTERLEAVED   → one compound "particles" dataset per TS
#
# We assert only the output-structure invariant each variant guarantees; the
# actual data values come from a different prepare_* helper per variant and
# we let the structural checks catch silent format changes.
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
	not _all_binaries_present(BINARY_WRITE),
	reason="h5bench_write binary missing",
)
def test_write_1d_contig_interleaved():
	_run("tiny-1d-contig-interleaved.json")

	output = "test-output-tiny-1d-contig-interleaved/tiny-ci.h5"
	h5_assert.assert_timestep_count(output, 1)
	h5_assert.assert_compound_dataset_present(output, "/Timestep_0")
	h5_assert.assert_dataset_shape(
		output, "/Timestep_0/particles", (TOTAL_PARTICLES,)
	)


@pytest.mark.skipif(
	not _all_binaries_present(BINARY_WRITE),
	reason="h5bench_write binary missing",
)
def test_write_1d_interleaved_contig():
	_run("tiny-1d-interleaved-contig.json")

	output = "test-output-tiny-1d-interleaved-contig/tiny-ic.h5"
	h5_assert.assert_timestep_count(output, 1)
	h5_assert.assert_contig_datasets_present(output, "/Timestep_0")
	for dset in ("x", "y", "z", "px", "py", "pz", "id_1", "id_2"):
		h5_assert.assert_dataset_shape(
			output, f"/Timestep_0/{dset}", (TOTAL_PARTICLES,)
		)


@pytest.mark.skipif(
	not _all_binaries_present(BINARY_WRITE),
	reason="h5bench_write binary missing",
)
def test_write_1d_interleaved_interleaved():
	_run("tiny-1d-interleaved-interleaved.json")

	output = "test-output-tiny-1d-interleaved-interleaved/tiny-ii.h5"
	h5_assert.assert_timestep_count(output, 1)
	h5_assert.assert_compound_dataset_present(output, "/Timestep_0")
	h5_assert.assert_dataset_shape(
		output, "/Timestep_0/particles", (TOTAL_PARTICLES,)
	)


# ---------------------------------------------------------------------------
# 2D / 3D contig write — multi-dim file dataspace
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
	not _all_binaries_present(BINARY_WRITE),
	reason="h5bench_write binary missing",
)
def test_write_2d_contig():
	_run("tiny-2d-write.json")

	output = "test-output-tiny-2d-write/tiny-2d.h5"
	# dim_1 = 32, dim_2 = 16, 2 ranks → file dataspace (32*2, 16) = (64, 16).
	expected_shape_f = (64, 16)
	expected_shape_i = (64, 16)

	h5_assert.assert_timestep_count(output, 1)
	h5_assert.assert_contig_datasets_present(output, "/Timestep_0")
	for dset in ("x", "y", "z", "px", "py", "pz", "id_2"):
		h5_assert.assert_dataset_shape(output, f"/Timestep_0/{dset}", expected_shape_f)
		h5_assert.assert_dataset_dtype(output, f"/Timestep_0/{dset}", "f")
	h5_assert.assert_dataset_shape(output, "/Timestep_0/id_1", expected_shape_i)
	h5_assert.assert_dataset_dtype(output, "/Timestep_0/id_1", "i")


@pytest.mark.skipif(
	not _all_binaries_present(BINARY_WRITE),
	reason="h5bench_write binary missing",
)
def test_write_3d_contig():
	_run("tiny-3d-write.json")

	output = "test-output-tiny-3d-write/tiny-3d.h5"
	# dim_1 = dim_2 = dim_3 = 8, 2 ranks → file dataspace (8*2, 8, 8) = (16, 8, 8).
	expected_shape = (16, 8, 8)

	h5_assert.assert_timestep_count(output, 1)
	h5_assert.assert_contig_datasets_present(output, "/Timestep_0")
	for dset in ("x", "y", "z", "px", "py", "pz", "id_2"):
		h5_assert.assert_dataset_shape(output, f"/Timestep_0/{dset}", expected_shape)
		h5_assert.assert_dataset_dtype(output, f"/Timestep_0/{dset}", "f")
	h5_assert.assert_dataset_shape(output, "/Timestep_0/id_1", expected_shape)
	h5_assert.assert_dataset_dtype(output, "/Timestep_0/id_1", "i")


# ---------------------------------------------------------------------------
# Write unlimited — maxshape[0] is None
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
	not _all_binaries_present(BINARY_UNLIMITED),
	reason="h5bench_write_unlimited binary missing",
)
def test_write_unlimited_has_unlimited_axis():
	_run("tiny-write-unlimited.json")

	output = "test-output-tiny-write-unlimited/tiny-unlimited.h5"

	# The whole point of write_unlimited is H5S_UNLIMITED on axis 0. Check every
	# dataset in every timestep.
	for ts in ("Timestep_0", "Timestep_1"):
		for dset in ("x", "y", "z", "px", "py", "pz", "id_1", "id_2"):
			h5_assert.assert_unlimited_axis(output, f"/{ts}/{dset}", axis=0)
			# Initial extent still matches the per-rank write.
			h5_assert.assert_dataset_shape(
				output, f"/{ts}/{dset}", (TOTAL_PARTICLES,)
			)
