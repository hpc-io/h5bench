import os
import json
import glob
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

            data['vol']['library'] = ':'.join(paths)
            data['vol']['preload'] = ':'.join(preloads)

            if has_async_vol:
                data['vol']['path'] = '/'.join([ASYNC_DIR, 'lib'])

            with open(file, 'w') as f:
                json.dump(data, f, indent=4, sort_keys=False)
