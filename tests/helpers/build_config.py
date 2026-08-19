"""
pytest skip markers keyed to the CMake configure-time build flags.

Every optional h5bench benchmark (``H5BENCH_METADATA``, ``H5BENCH_EXERCISER``,
``H5BENCH_AMREX``, ``H5BENCH_OPENPMD``, ``H5BENCH_E3SM``, ``H5BENCH_MACSIO``)
and the ``WITH_ASYNC_VOL`` VOL connector have a ``requires_*`` marker below.
Decorate a test with one of them and it will skip with a human-readable
reason when the corresponding option was OFF at configure time.

The underlying flags come from ``h5bench_configuration.__options__`` — the
module that CMake generates into the build directory from
``src/h5bench_configuration.py.in`` — so the skip reason reflects what the
build actually shipped, not a guess based on binary presence.
"""

import pytest

try:
    from h5bench_configuration import __options__ as _options
except ImportError:  # pragma: no cover - shouldn't happen inside ctest
    # h5bench_configuration is generated into the build dir and placed on
    # the driver's import path. If it's missing (e.g. running pytest out-of-
    # tree without cmake), fall back to "everything off" so tests skip
    # rather than erroring out.
    _options = {
        "H5BENCH_METADATA": False,
        "H5BENCH_EXERCISER": False,
        "H5BENCH_AMREX": False,
        "H5BENCH_OPENPMD": False,
        "H5BENCH_E3SM": False,
        "H5BENCH_MACSIO": False,
        "WITH_ASYNC_VOL": False,
    }


def _marker(option_name, label):
    enabled = _options.get(option_name, False)
    return pytest.mark.skipif(
        not enabled,
        reason=f"{option_name}={'ON' if enabled else 'OFF'} at build time ({label})",
    )


requires_metadata   = _marker("H5BENCH_METADATA",  "metadata benchmark")
requires_exerciser  = _marker("H5BENCH_EXERCISER", "exerciser benchmark")
requires_amrex      = _marker("H5BENCH_AMREX",     "AMReX benchmark")
requires_openpmd    = _marker("H5BENCH_OPENPMD",   "OpenPMD benchmark")
requires_e3sm       = _marker("H5BENCH_E3SM",      "E3SM-IO benchmark")
requires_macsio     = _marker("H5BENCH_MACSIO",    "MACSio benchmark")
requires_async_vol  = _marker("WITH_ASYNC_VOL",    "VOL-ASYNC connector")


def is_enabled(option_name):
    """Programmatic lookup for tests that need to branch rather than skip."""
    return bool(_options.get(option_name, False))
