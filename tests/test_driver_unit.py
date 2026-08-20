#!/usr/bin/env python3
"""
Pure-Python unit tests for the h5bench.H5bench driver class.

These exercise the parts of ``src/h5bench.py`` that don't require a built
benchmark binary — JSON validation, MPI launcher assembly, VOL environment
preparation, and the subprocess-independent helpers. The existing
integration tests in ``test_sync_*.py`` drive the same class end-to-end but
only cover the happy-path dispatcher; these tests fill in the negative
paths and argument-shaping logic that were 0% covered before.

No HDF5 binary is invoked. Tests are fast (< 1s total) and run without
MPI, HDF5, or network access.
"""

import os
import sys
import pytest

from src import h5bench as _h5bench


# ---------------------------------------------------------------------------
# Shared fixture — a dummy H5bench whose constructor side-effects all
# target files under tmp_path so tests don't pollute the repo.
# ---------------------------------------------------------------------------


@pytest.fixture
def bench(tmp_path, monkeypatch):
    """Return an H5bench instance rooted in tmp_path.

    ``check_parallel()`` reads ``$SHELL`` and may exit if the shell name
    contains 'mpirun'/'mpiexec'/'srun'; we override it to /bin/bash so the
    instantiation is deterministic regardless of the test host.
    """
    monkeypatch.setenv("SHELL", "/bin/bash")
    cfg = tmp_path / "dummy.json"
    cfg.write_text("{}")
    monkeypatch.chdir(tmp_path)  # log file lands here, not in repo
    return _h5bench.H5bench(str(cfg), debug=False)


# ---------------------------------------------------------------------------
# validate_json
# ---------------------------------------------------------------------------


_REQUIRED_KEYS = ("mpi", "vol", "file-system", "directory", "benchmarks")


def _full_setup():
    """A minimal setup dict that both the legacy top-level-keys check and
    the JSON schema (`schemas/h5bench-config.schema.json`) accept. The
    opaque-benchmark branch is the least-restrictive way to satisfy the
    `benchmarks` oneOf, so we use it here."""
    return {
        "mpi": {"command": "mpirun"},
        "vol": {},
        "file-system": {},
        "directory": "storage",
        "benchmarks": [{"benchmark": "amrex", "configuration": {}}],
    }


def test_validate_json_accepts_all_required_keys(bench):
    # Should not raise / exit.
    bench.validate_json(_full_setup())


@pytest.mark.parametrize("missing_key", _REQUIRED_KEYS)
def test_validate_json_exits_when_key_missing(bench, missing_key):
    setup = _full_setup()
    del setup[missing_key]
    with pytest.raises(SystemExit) as excinfo:
        bench.validate_json(setup)
    assert excinfo.value.code == os.EX_DATAERR


def test_validate_json_tolerates_extra_keys(bench):
    setup = _full_setup()
    setup["this-is-fine"] = {"nested": True}
    # The current contract is permissive about extras — lock that in.
    bench.validate_json(setup)


# ---------------------------------------------------------------------------
# _load_config_schema and validate_json's legacy-fallback branches
# ---------------------------------------------------------------------------


def test_load_config_schema_returns_none_when_jsonschema_missing(bench, monkeypatch):
    # Force `import jsonschema` inside _load_config_schema to raise so
    # the ImportError branch (and the (None, None) return) is exercised.
    monkeypatch.setitem(sys.modules, "jsonschema", None)
    schema, mod = bench._load_config_schema()
    assert schema is None
    assert mod is None


def test_load_config_schema_returns_none_when_no_file_found(bench, monkeypatch):
    # jsonschema itself imports fine, but no candidate path exists on disk.
    monkeypatch.setattr(os.path, "isfile", lambda p: False)
    schema, mod = bench._load_config_schema()
    assert schema is None
    assert mod is not None


def test_load_config_schema_honors_env_var(bench, monkeypatch, tmp_path):
    fake = tmp_path / "custom-schema.json"
    fake.write_text('{"type": "object"}')
    monkeypatch.setenv("H5BENCH_SCHEMA", str(fake))
    schema, mod = bench._load_config_schema()
    assert schema == {"type": "object"}
    assert mod is not None


def test_validate_json_falls_back_when_schema_unavailable(bench, monkeypatch):
    # When _load_config_schema returns (None, None), validate_json must
    # still accept a setup that has all five legacy required keys.
    monkeypatch.setattr(bench, "_load_config_schema", lambda: (None, None))
    legacy_setup = {k: {} if k != "benchmarks" else [] for k in _REQUIRED_KEYS}
    bench.validate_json(legacy_setup)


@pytest.mark.parametrize("missing_key", _REQUIRED_KEYS)
def test_validate_json_fallback_exits_when_key_missing(bench, monkeypatch, missing_key):
    monkeypatch.setattr(bench, "_load_config_schema", lambda: (None, None))
    legacy_setup = {k: {} if k != "benchmarks" else [] for k in _REQUIRED_KEYS}
    del legacy_setup[missing_key]
    with pytest.raises(SystemExit) as excinfo:
        bench.validate_json(legacy_setup)
    assert excinfo.value.code == os.EX_DATAERR


# ---------------------------------------------------------------------------
# prepare_parallel — builds the MPI launch command prefix
# ---------------------------------------------------------------------------


def test_prepare_parallel_uses_explicit_configuration(bench):
    bench.prepare_parallel(
        {"command": "mpirun", "ranks": "4", "configuration": "-np 2 --oversubscribe"}
    )
    assert bench.mpi == "mpirun -np 2 --oversubscribe"


def test_prepare_parallel_mpirun_without_configuration_uses_ranks(bench):
    bench.prepare_parallel({"command": "mpirun", "ranks": "4"})
    assert bench.mpi == "mpirun -np 4"


def test_prepare_parallel_mpiexec_without_configuration_uses_ranks(bench):
    bench.prepare_parallel({"command": "mpiexec", "ranks": "8"})
    assert bench.mpi == "mpiexec -np 8"


def test_prepare_parallel_srun_uses_cpu_bind(bench):
    bench.prepare_parallel({"command": "srun", "ranks": "16"})
    assert bench.mpi == "srun --cpu_bind=cores -n 16"


@pytest.mark.parametrize("launcher", ["ibrun", "aprun", "flux run", "prun"])
def test_prepare_parallel_extra_launchers_use_dash_n(bench, launcher):
    bench.prepare_parallel({"command": launcher, "ranks": "8"})
    assert bench.mpi == "{} -n 8".format(launcher)


def test_prepare_parallel_unknown_command_warns_and_blanks(bench):
    bench.prepare_parallel({"command": "not-a-launcher", "ranks": "2"})
    # The legacy contract: unknown command -> blank mpi prefix, no exception.
    assert bench.mpi == ""


# ---------------------------------------------------------------------------
# is_available — shutil.which wrapper that also searches PWD
# ---------------------------------------------------------------------------


def test_is_available_returns_path_when_found(bench, tmp_path, monkeypatch):
    fake = tmp_path / "fake-h5bench"
    fake.write_text("#!/bin/sh\nexit 0\n")
    fake.chmod(0o755)
    # With PWD = tmp_path (set by the fixture), and is_available appending
    # ':.' to PATH, the file must resolve.
    assert bench.is_available("fake-h5bench") is not None


def test_is_available_returns_none_when_missing(bench, monkeypatch):
    # Clear PATH so even system binaries aren't resolvable; PWD (tmp_path)
    # also doesn't contain the fake name.
    monkeypatch.setenv("PATH", "/nonexistent")
    assert bench.is_available("definitely-not-a-real-binary") is None


# ---------------------------------------------------------------------------
# prepare_vol / enable_vol / disable_vol / reset_vol
# ---------------------------------------------------------------------------


def _prime_vol_environment(bench):
    # run() normally copies os.environ into self.vol_environment; the VOL
    # helpers expect the attribute to exist.
    bench.vol_environment = {}


def test_prepare_vol_none_leaves_environment_alone(bench):
    _prime_vol_environment(bench)
    bench.prepare_vol(None)
    # None means "no VOL configured" — the helper should not inject
    # ABT_THREAD_STACKSIZE or library paths.
    assert "ABT_THREAD_STACKSIZE" not in bench.vol_environment
    assert "LD_LIBRARY_PATH" not in bench.vol_environment


def test_prepare_vol_populates_library_preload_and_path(bench):
    _prime_vol_environment(bench)
    bench.prepare_vol(
        {
            "library": "/opt/hdf5/lib",
            "preload": "/opt/hdf5/lib/libhdf5.so",
            "path": "/opt/vol-async/lib",
        }
    )
    assert "/opt/hdf5/lib" in bench.vol_environment["LD_LIBRARY_PATH"]
    assert "/opt/hdf5/lib" in bench.vol_environment["DYLD_LIBRARY_PATH"]
    assert bench.vol_environment["LD_PRELOAD"].endswith("/opt/hdf5/lib/libhdf5.so")
    assert bench.vol_environment["HDF5_PLUGIN_PATH"] == "/opt/vol-async/lib"
    assert bench.vol_environment["ABT_THREAD_STACKSIZE"] == "100000"


def test_enable_vol_sets_connector(bench):
    _prime_vol_environment(bench)
    bench.enable_vol({"connector": "async under_vol=0;under_info={}"})
    assert bench.vol_environment["HDF5_VOL_CONNECTOR"] == "async under_vol=0;under_info={}"


def test_enable_vol_without_connector_key_is_noop(bench):
    _prime_vol_environment(bench)
    bench.enable_vol({"library": "/x"})  # no 'connector'
    assert "HDF5_VOL_CONNECTOR" not in bench.vol_environment


def test_disable_vol_removes_connector(bench):
    _prime_vol_environment(bench)
    bench.vol_environment["HDF5_VOL_CONNECTOR"] = "async"
    bench.disable_vol({})
    assert "HDF5_VOL_CONNECTOR" not in bench.vol_environment


def test_disable_vol_when_not_set_is_noop(bench):
    _prime_vol_environment(bench)
    bench.disable_vol({})  # nothing to remove — must not raise
    assert "HDF5_VOL_CONNECTOR" not in bench.vol_environment


def test_reset_vol_clears_all_vol_variables(bench):
    _prime_vol_environment(bench)
    bench.vol_environment.update(
        {
            "HDF5_PLUGIN_PATH": "/x",
            "HDF5_VOL_CONNECTOR": "async",
            "ABT_THREAD_STACKSIZE": "100000",
        }
    )
    bench.reset_vol()
    for k in ("HDF5_PLUGIN_PATH", "HDF5_VOL_CONNECTOR", "ABT_THREAD_STACKSIZE"):
        assert k not in bench.vol_environment


# ---------------------------------------------------------------------------
# check_for_hdf5_error — scans a stderr log for the HDF5 error banner
# ---------------------------------------------------------------------------


def test_check_for_hdf5_error_returns_false_on_clean_stderr(bench, tmp_path):
    err = tmp_path / "stderr"
    err.write_text("no errors here\nall good\n")
    assert bench.check_for_hdf5_error(str(err)) is False


def test_check_for_hdf5_error_exits_on_hdf5_banner(bench, tmp_path):
    err = tmp_path / "stderr"
    err.write_text("some log\nError detected in HDF5 (2.1.1) something\nmore\n")
    with pytest.raises(SystemExit) as excinfo:
        bench.check_for_hdf5_error(str(err))
    assert excinfo.value.code == os.EX_IOERR


# ---------------------------------------------------------------------------
# check_parallel — exits if the user's $SHELL implies they're inside mpirun
# ---------------------------------------------------------------------------


def test_check_parallel_accepts_plain_shell(tmp_path, monkeypatch):
    monkeypatch.setenv("SHELL", "/bin/bash")
    cfg = tmp_path / "dummy.json"
    cfg.write_text("{}")
    monkeypatch.chdir(tmp_path)
    # Constructor calls check_parallel internally — no exit expected.
    _h5bench.H5bench(str(cfg), debug=False)


@pytest.mark.parametrize("tool", ["mpirun", "mpiexec", "srun"])
def test_check_parallel_exits_when_shell_names_an_mpi_launcher(
    tmp_path, monkeypatch, tool
):
    monkeypatch.setenv("SHELL", f"/usr/bin/{tool}")
    cfg = tmp_path / "dummy.json"
    cfg.write_text("{}")
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit) as excinfo:
        _h5bench.H5bench(str(cfg), debug=False)
    assert excinfo.value.code == os.EX_USAGE


def test_check_parallel_without_shell_env_does_not_exit(tmp_path, monkeypatch):
    monkeypatch.delenv("SHELL", raising=False)
    cfg = tmp_path / "dummy.json"
    cfg.write_text("{}")
    monkeypatch.chdir(tmp_path)
    _h5bench.H5bench(str(cfg), debug=False)


# ---------------------------------------------------------------------------
# run() — error paths that don't need real binaries
# ---------------------------------------------------------------------------


def test_run_exits_when_setup_file_missing(tmp_path, monkeypatch):
    monkeypatch.setenv("SHELL", "/bin/bash")
    monkeypatch.chdir(tmp_path)
    # Constructor only reads setup on run(), so we can point at a
    # non-existent file here.
    bench = _h5bench.H5bench(str(tmp_path / "nope.json"), debug=False)
    with pytest.raises(SystemExit) as excinfo:
        bench.run()
    assert excinfo.value.code == os.EX_NOINPUT


def test_run_exits_when_setup_file_malformed(tmp_path, monkeypatch):
    monkeypatch.setenv("SHELL", "/bin/bash")
    monkeypatch.chdir(tmp_path)
    cfg = tmp_path / "broken.json"
    cfg.write_text("not-json-at-all")
    bench = _h5bench.H5bench(str(cfg), debug=False)
    with pytest.raises(SystemExit) as excinfo:
        bench.run()
    assert excinfo.value.code == os.EX_NOINPUT
