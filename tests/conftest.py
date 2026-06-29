import os
import json
import glob
import shutil

HDF5_DIR = os.getenv('HDF5_DIR')
ABT_DIR = os.getenv('ABT_DIR')
ASYNC_DIR = os.getenv('ASYNC_DIR')


def _inject_vol_paths(file_path):
	"""Rewrite the config's vol.library / vol.preload / vol.path fields with
	locally-resolved HDF5 / Argobots / VOL-ASYNC paths. Shared between the
	production samples and the test-scoped fixtures so both go through the
	exact same VOL wiring."""
	with open(file_path, 'r') as f:
		data = json.load(f)

	paths = []
	preloads = []
	has_async_vol = False

	if ASYNC_DIR:
		paths.append('/'.join([ASYNC_DIR, 'lib']))
		preloads.append('/'.join([ASYNC_DIR, 'lib', 'libh5async.so']))
		has_async_vol = True

	if ABT_DIR:
		paths.append('/'.join([ABT_DIR, 'lib']))
		preloads.append('/'.join([ABT_DIR, 'lib', 'libabt.so']))

	if HDF5_DIR:
		paths.append('/'.join([HDF5_DIR, 'lib']))
		preloads.append('/'.join([HDF5_DIR, 'lib', 'libhdf5.so']))

	data.setdefault('vol', {})
	data['vol']['library'] = ':'.join(paths)
	data['vol']['preload'] = ':'.join(preloads)

	if has_async_vol:
		data['vol']['path'] = '/'.join([ASYNC_DIR, 'lib'])

	with open(file_path, 'w') as f:
		json.dump(data, f, indent=4, sort_keys=False)


def pytest_configure(config):
	# Legacy: every production sample is copied into the build dir and has
	# its VOL paths injected. Existing tests glob for the resulting files in
	# PWD, so this must continue to run unchanged.
	for sample in glob.glob('../samples/*.json'):
		dest = os.path.basename(sample)
		shutil.copyfile(sample, dest)
		_inject_vol_paths(dest)

	# Test-scoped tiny fixtures (1024 particles, 2 ranks, 2 timesteps). These
	# live under tests/fixtures/ and feed the Wave C-prep output-correctness
	# assertions in test_output_validation.py / test_read_roundtrip.py.
	for fixture in glob.glob('../tests/fixtures/*.json'):
		dest = os.path.basename(fixture)
		shutil.copyfile(fixture, dest)
		_inject_vol_paths(dest)
