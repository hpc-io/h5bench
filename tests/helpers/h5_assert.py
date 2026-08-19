"""
HDF5 output assertions used by the Wave C-prep test suite.

These helpers let a test verify the *content* of a benchmark's HDF5 output —
shape, dtype, group structure, deterministic-id patterns, range of random
values — instead of only confirming that the binary exited with status 0.

Requires `h5py` (see tests/requirements.txt).
"""

import hashlib

import h5py
import numpy as np


# ---------------------------------------------------------------------------
# Low-level accessors
# ---------------------------------------------------------------------------


def read_dataset(path, dset_path):
    """Return the dataset at `dset_path` from `path` as a NumPy array."""
    with h5py.File(path, "r") as f:
        return f[dset_path][()]


def list_timestep_groups(path):
    """Return the sorted list of top-level `/Timestep_<N>` group names."""
    with h5py.File(path, "r") as f:
        return sorted(k for k in f.keys() if k.startswith("Timestep_"))


def dataset_sha256(path, dset_path):
    """Stable SHA-256 of a dataset's raw bytes — useful for before/after comparisons."""
    data = read_dataset(path, dset_path)
    return hashlib.sha256(np.ascontiguousarray(data).tobytes()).hexdigest()


# ---------------------------------------------------------------------------
# Structural assertions
# ---------------------------------------------------------------------------


def assert_timestep_count(path, expected):
    groups = list_timestep_groups(path)
    assert len(groups) == expected, (
        f"{path}: expected {expected} Timestep_* groups, found {len(groups)}: {groups}"
    )


# The h5bench contig patterns write these eight 1D-per-particle datasets.
CONTIG_DATASETS = ("x", "y", "z", "px", "py", "pz", "id_1", "id_2")


def assert_contig_datasets_present(path, group="/Timestep_0"):
    with h5py.File(path, "r") as f:
        actual = set(f[group].keys())
    missing = set(CONTIG_DATASETS) - actual
    assert not missing, f"{path}{group}: missing datasets {sorted(missing)} (have {sorted(actual)})"


def assert_dataset_shape(path, dset_path, expected_shape):
    with h5py.File(path, "r") as f:
        actual = f[dset_path].shape
    assert actual == expected_shape, (
        f"{path}{dset_path}: shape {actual} != expected {expected_shape}"
    )


def assert_dataset_dtype(path, dset_path, expected_kind):
    """`expected_kind` is a NumPy dtype kind char: 'f' for float, 'i' for signed int.

    We compare kinds rather than exact dtypes because HDF5's native types (e.g.
    H5T_NATIVE_FLOAT) materialize as numpy float32 but the underlying precision
    is an implementation detail we don't want to hard-code per platform.
    """
    with h5py.File(path, "r") as f:
        actual = f[dset_path].dtype
    assert actual.kind == expected_kind, (
        f"{path}{dset_path}: dtype {actual} (kind '{actual.kind}') != expected kind '{expected_kind}'"
    )


def assert_unlimited_axis(path, dset_path, axis=0):
    with h5py.File(path, "r") as f:
        maxshape = f[dset_path].maxshape
    assert maxshape[axis] is None, (
        f"{path}{dset_path}: expected axis {axis} to be unlimited (maxshape[axis]==None), got {maxshape}"
    )


# ---------------------------------------------------------------------------
# Content assertions
# ---------------------------------------------------------------------------


def assert_value_in_range(path, dset_path, lo, hi):
    """Bounds check for the random-valued datasets (x/y/z/px/py/pz).

    h5bench fills these with `uniform_random_number() * X_DIM` etc., so every
    element must land in [0, dim]. Catches corrupt writes and type-pun bugs
    without requiring a deterministic seed.
    """
    data = read_dataset(path, dset_path)
    data_min = data.min() if data.size else 0
    data_max = data.max() if data.size else 0
    assert lo <= data_min and data_max <= hi, (
        f"{path}{dset_path}: values outside [{lo}, {hi}] (min={data_min}, max={data_max})"
    )


def assert_id_1_1d_pattern(path, dset_path, particles_per_rank, num_ranks):
    """For the 1D contig write pattern, `id_1[i] = i` on each rank. Ranks are
    concatenated along axis 0, so the whole dataset should be
    `[0..N-1, 0..N-1, ...]` repeated num_ranks times.
    """
    data = read_dataset(path, dset_path)
    expected = np.tile(np.arange(particles_per_rank, dtype=data.dtype), num_ranks)
    np.testing.assert_array_equal(
        data,
        expected,
        err_msg=f"{path}{dset_path}: id_1 pattern mismatch (expected tiled arange).",
    )


def assert_id_2_1d_write_pattern(path, dset_path, particles_per_rank, num_ranks):
    """Write path: `id_2[i] = (float)(i * 2)`. Same tiling across ranks as id_1."""
    data = read_dataset(path, dset_path)
    per_rank = (np.arange(particles_per_rank, dtype=np.float64) * 2).astype(data.dtype)
    expected = np.tile(per_rank, num_ranks)
    np.testing.assert_array_equal(
        data,
        expected,
        err_msg=f"{path}{dset_path}: id_2 write pattern mismatch (expected 2*i tiled).",
    )


def assert_datasets_equal(path_a, path_b, dset_path):
    """Byte-exact equality check — primary tool for read round-trip tests."""
    a = read_dataset(path_a, dset_path)
    b = read_dataset(path_b, dset_path)
    np.testing.assert_array_equal(
        a,
        b,
        err_msg=f"{dset_path} differs between {path_a} and {path_b}",
    )


# The compound-type write paths emit a single "particles" dataset per timestep
# with 8 named fields in the HDF5 compound type (see make_compound_type() in
# h5bench_write.c). These helpers let the compound-variant tests express the
# same invariants that CONTIG_DATASETS expresses for the 8-dataset layout.
COMPOUND_FIELDS = ("x", "y", "z", "px", "py", "pz", "id_1", "id_2")


def assert_compound_dataset_present(path, group="/Timestep_0", name="particles"):
    """The file has exactly one compound dataset per timestep, named `particles`,
    whose HDF5 compound type carries the 8 named fields above."""
    dset_path = f"{group}/{name}"
    with h5py.File(path, "r") as f:
        assert name in f[group], (
            f"{path}{group}: expected a '{name}' compound dataset, have {sorted(f[group].keys())}"
        )
        dtype = f[dset_path].dtype
    assert dtype.kind == "V", (
        f"{path}{dset_path}: expected compound/void dtype, got {dtype} (kind '{dtype.kind}')"
    )
    field_names = set(dtype.names or ())
    missing = set(COMPOUND_FIELDS) - field_names
    assert not missing, (
        f"{path}{dset_path}: compound type missing fields {sorted(missing)} "
        f"(have {sorted(field_names)})"
    )
