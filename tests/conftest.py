import os
import json
import glob
import uuid
import shutil

HDF5_DIR = os.getenv('HDF5_DIR')
ABT_DIR = os.getenv('ABT_DIR')
ASYNC_DIR = os.getenv('ASYNC_DIR')

def pytest_configure(config):
    samples = glob.glob('../samples/*.json')

    for sample in samples:
        file = os.path.basename(sample)

        shutil.copyfile(sample, file)

        with open(file, 'r') as f:
            data = json.load(f)

            data['vol']['library'] = '{}:{}:{}'.format(
                '/'.join([ASYNC_DIR, 'lib']),
                '/'.join([ABT_DIR, 'lib']),
                '/'.join([HDF5_DIR, 'lib'])
            )

            data['vol']['preload'] = '{}:{}:{}'.format(
                '/'.join([ASYNC_DIR, 'lib', 'libh5async.so']),
                '/'.join([ABT_DIR, 'lib', 'libabt.so']),
                '/'.join([HDF5_DIR, 'lib', 'libhdf5.so'])
            )

            data['vol']['path'] = '/'.join([ASYNC_DIR, 'lib'])

            with open(file, 'w') as f:
                json.dump(data, f, indent=4, sort_keys=False)