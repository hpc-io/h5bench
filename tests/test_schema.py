#!/usr/bin/env python3
"""
JSON schema validation tests for the h5bench configuration file.

Drives ``schemas/h5bench-config.schema.json`` directly with ``jsonschema``
(no benchmark binary, no MPI). Confirms that every production sample under
``samples/`` still validates, that each pattern-benchmark enum (``MODE``,
``MEM_PATTERN``, ``FILE_PATTERN``, ``READ_OPTION``, ``COLLECTIVE_DATA``,
``COMPRESS``, etc.) rejects garbage, and that the driver's ``validate_json``
falls back to the legacy five-key check when ``jsonschema`` is unavailable.
"""

import copy
import glob
import json
import os

import pytest

# jsonschema is the one hard dependency of this module. CI installs it from
# tests/requirements.txt; if it is somehow absent the whole module skips
# (rather than erroring at collection) so a missing dev dependency degrades
# gracefully instead of failing the ctest run.
jsonschema = pytest.importorskip("jsonschema")

from src import h5bench as _h5bench


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SCHEMA_PATH = os.path.join(REPO_ROOT, 'schemas', 'h5bench-config.schema.json')


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope='module')
def schema():
    with open(SCHEMA_PATH) as f:
        return json.load(f)


@pytest.fixture
def base_pattern_config():
    """A minimal-but-complete write benchmark config that validates cleanly.
    Tests deep-copy and mutate this to construct each failure case."""
    return {
        'mpi': {
            'command': 'mpirun',
            'ranks': '2',
            'configuration': '-np 2 --oversubscribe',
        },
        'vol': {},
        'file-system': {},
        'directory': 'storage',
        'benchmarks': [
            {
                'benchmark': 'write',
                'file': 'out.h5',
                'configuration': {
                    'MEM_PATTERN': 'CONTIG',
                    'FILE_PATTERN': 'CONTIG',
                    'TIMESTEPS': '1',
                    'DELAYED_CLOSE_TIMESTEPS': '0',
                    'COLLECTIVE_DATA': 'YES',
                    'COLLECTIVE_METADATA': 'YES',
                    'EMULATED_COMPUTE_TIME_PER_TIMESTEP': '0 s',
                    'NUM_DIMS': '1',
                    'DIM_1': '1024',
                    'DIM_2': '1',
                    'DIM_3': '1',
                    'MODE': 'SYNC',
                },
            }
        ],
    }


@pytest.fixture
def bench(tmp_path, monkeypatch):
    """A real H5bench instance that we can call validate_json on; mirrors
    the helper in test_driver_unit.py."""
    monkeypatch.setenv('SHELL', '/bin/bash')
    cfg = tmp_path / 'dummy.json'
    cfg.write_text('{}')
    monkeypatch.chdir(tmp_path)
    return _h5bench.H5bench(str(cfg), debug=False)


# ---------------------------------------------------------------------------
# Every shipped sample validates
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    'sample_path',
    sorted(glob.glob(os.path.join(REPO_ROOT, 'samples', '*.json'))),
    ids=lambda p: os.path.basename(p),
)
def test_production_sample_validates(schema, sample_path):
    """Every sample under samples/ has to keep validating cleanly. If a new
    sample lands and the schema is too tight, this fires first."""
    with open(sample_path) as f:
        setup = json.load(f)
    jsonschema.validate(setup, schema)


# ---------------------------------------------------------------------------
# Top-level required keys
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    'key', ['mpi', 'vol', 'file-system', 'directory', 'benchmarks']
)
def test_top_level_key_missing_rejected(schema, base_pattern_config, key):
    setup = copy.deepcopy(base_pattern_config)
    del setup[key]
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(setup, schema)


def test_top_level_extra_key_tolerated(schema, base_pattern_config):
    # The root has additionalProperties: true so callers can ship extra
    # documentation keys without tripping validation.
    setup = copy.deepcopy(base_pattern_config)
    setup['x-comment'] = 'arbitrary note'
    jsonschema.validate(setup, schema)


# ---------------------------------------------------------------------------
# mpi.command must be a known launcher
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    'launcher',
    ['mpirun', 'mpiexec', 'srun', 'ibrun', 'aprun', 'flux run', 'prun'],
)
def test_mpi_known_launcher_accepted(schema, base_pattern_config, launcher):
    setup = copy.deepcopy(base_pattern_config)
    setup['mpi']['command'] = launcher
    jsonschema.validate(setup, schema)


@pytest.mark.parametrize('launcher', ['jsrun', 'runjob', 'lrun'])
def test_mpi_legacy_launcher_rejected(schema, base_pattern_config, launcher):
    # jsrun (Summit/Sierra), runjob (BlueGene/Q), lrun (Sierra/Lassen
    # wrapper) were dropped because their systems are decommissioned or
    # winding down. Confirm the enum still rejects them so nobody
    # accidentally re-adds without updating this test.
    setup = copy.deepcopy(base_pattern_config)
    setup['mpi']['command'] = launcher
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(setup, schema)


def test_mpi_unknown_launcher_rejected(schema, base_pattern_config):
    setup = copy.deepcopy(base_pattern_config)
    setup['mpi']['command'] = 'horovodrun'
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(setup, schema)


# ---------------------------------------------------------------------------
# Benchmark dispatcher enum
# ---------------------------------------------------------------------------


_VALID_BENCHMARKS = [
    'write', 'write-unlimited', 'overwrite', 'append', 'read',
    'write_var_normal_dist',
]


@pytest.mark.parametrize('name', _VALID_BENCHMARKS)
def test_pattern_benchmark_names_accepted(schema, base_pattern_config, name):
    setup = copy.deepcopy(base_pattern_config)
    setup['benchmarks'][0]['benchmark'] = name
    jsonschema.validate(setup, schema)


def test_bogus_benchmark_name_rejected(schema, base_pattern_config):
    setup = copy.deepcopy(base_pattern_config)
    setup['benchmarks'][0]['benchmark'] = 'BOGUS'
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(setup, schema)


def test_legacy_write_normal_dist_name_rejected(schema, base_pattern_config):
    """The legacy `write_normal_dist` name (without `var_`) is not recognised
    by the driver dispatcher; the schema must reject it so the typo doesn't
    get reintroduced (samples/sync-write-1d-contig-contig-normal-dist.json
    used it historically and silently fell through to 'Unsupported')."""
    setup = copy.deepcopy(base_pattern_config)
    setup['benchmarks'][0]['benchmark'] = 'write_normal_dist'
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(setup, schema)


# ---------------------------------------------------------------------------
# Pattern config enums
# ---------------------------------------------------------------------------


@pytest.mark.parametrize('mode', ['SYNC', 'ASYNC', 'LOG'])
def test_mode_enum_accepted(schema, base_pattern_config, mode):
    setup = copy.deepcopy(base_pattern_config)
    setup['benchmarks'][0]['configuration']['MODE'] = mode
    jsonschema.validate(setup, schema)


def test_mode_enum_rejects_garbage(schema, base_pattern_config):
    setup = copy.deepcopy(base_pattern_config)
    setup['benchmarks'][0]['configuration']['MODE'] = 'sync'  # case-sensitive
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(setup, schema)


@pytest.mark.parametrize('pattern', ['CONTIG', 'INTERLEAVED', 'STRIDED'])
def test_mem_file_pattern_enum_accepted(schema, base_pattern_config, pattern):
    setup = copy.deepcopy(base_pattern_config)
    setup['benchmarks'][0]['configuration']['MEM_PATTERN'] = pattern
    setup['benchmarks'][0]['configuration']['FILE_PATTERN'] = pattern
    jsonschema.validate(setup, schema)


def test_mem_pattern_rejects_garbage(schema, base_pattern_config):
    setup = copy.deepcopy(base_pattern_config)
    setup['benchmarks'][0]['configuration']['MEM_PATTERN'] = 'WACKY'
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(setup, schema)


def test_collective_data_rejects_garbage(schema, base_pattern_config):
    setup = copy.deepcopy(base_pattern_config)
    setup['benchmarks'][0]['configuration']['COLLECTIVE_DATA'] = 'maybe'
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(setup, schema)


# ---------------------------------------------------------------------------
# Numeric-string discipline
# ---------------------------------------------------------------------------


def test_dim_1_actual_number_rejected(schema, base_pattern_config):
    """h5bench config uses string-quoted integers historically; an actual
    JSON number is rejected so config writers don't get silent surprises
    if h5bench's parser ever tightens up."""
    setup = copy.deepcopy(base_pattern_config)
    setup['benchmarks'][0]['configuration']['DIM_1'] = 1024
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(setup, schema)


def test_emulated_compute_time_pattern(schema, base_pattern_config):
    setup = copy.deepcopy(base_pattern_config)
    # Valid forms accepted by h5bench_util parse_time.
    for value in ('5 s', '0 s', '10 ms', '2 min', '500 us'):
        setup['benchmarks'][0]['configuration']['EMULATED_COMPUTE_TIME_PER_TIMESTEP'] = value
        jsonschema.validate(setup, schema)
    # Garbage rejected.
    setup['benchmarks'][0]['configuration']['EMULATED_COMPUTE_TIME_PER_TIMESTEP'] = 'soonish'
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(setup, schema)


# ---------------------------------------------------------------------------
# Non-pattern benchmarks pass through opaquely
# ---------------------------------------------------------------------------


def test_exerciser_opaque_configuration_passes(schema, base_pattern_config):
    setup = copy.deepcopy(base_pattern_config)
    setup['benchmarks'][0] = {
        'benchmark': 'exerciser',
        'configuration': {
            'numdims': '2',
            'minels': '8 8',
            'nsizes': '3',
            'arbitrary-future-flag': 'yes',
        },
    }
    jsonschema.validate(setup, schema)


def test_amrex_opaque_configuration_passes(schema, base_pattern_config):
    setup = copy.deepcopy(base_pattern_config)
    setup['benchmarks'][0] = {
        'benchmark': 'amrex',
        'file': 'amrex.h5',
        'configuration': {
            'mode': 'MULTIPLE',
            'ncomp': '6',
            'something-new': True,
        },
    }
    jsonschema.validate(setup, schema)


# ---------------------------------------------------------------------------
# Driver-level wiring — validate_json calls jsonschema when available
# and falls back gracefully when it isn't.
# ---------------------------------------------------------------------------


def test_driver_validate_json_rejects_bad_mode(bench, base_pattern_config):
    setup = copy.deepcopy(base_pattern_config)
    setup['benchmarks'][0]['configuration']['MODE'] = 'NOPE'
    with pytest.raises(SystemExit) as excinfo:
        bench.validate_json(setup)
    assert excinfo.value.code == os.EX_DATAERR


def test_driver_validate_json_accepts_valid_config(bench, base_pattern_config):
    # Should return cleanly without raising.
    bench.validate_json(base_pattern_config)


def test_driver_falls_back_when_jsonschema_missing(
    bench, base_pattern_config, monkeypatch
):
    """Simulate a runtime where jsonschema isn't installed: the driver
    must still reject obviously broken configs via the legacy check
    (missing required top-level key)."""
    # Force the loader to return (None, None), simulating ImportError.
    monkeypatch.setattr(
        bench, '_load_config_schema', lambda: (None, None)
    )
    # Valid config still passes — falls through legacy checks.
    bench.validate_json(base_pattern_config)
    # Missing required key still rejected.
    broken = copy.deepcopy(base_pattern_config)
    del broken['mpi']
    with pytest.raises(SystemExit) as excinfo:
        bench.validate_json(broken)
    assert excinfo.value.code == os.EX_DATAERR
