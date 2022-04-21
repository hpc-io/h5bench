import os
import json
import argparse
import collections

PARSER = argparse.ArgumentParser(
    description='H5bench: update configuration.json with enviroment variables '
)

PARSER.add_argument(
    'setup',
    action='store',
    help='JSON file with the benchmarks to run'
)

ARGS = PARSER.parse_args()

HDF5_DIR = os.getenv('HDF5_DIR')
ABT_DIR = os.getenv('ABT_DIR')
ASYNC_DIR = os.getenv('ASYNC_DIR')

if HDF5_DIR is None:
	print('HDF5_DIR enviroment variable is not set!')
	exit(-1)

if ABT_DIR is None:
	print('ABT_DIR enviroment variable is not set!')
	exit(-1)

if ASYNC_DIR is None:
	print('ASYNC_DIR enviroment variable is not set!')
	exit(-1)

with open(ARGS.setup, 'r') as f:
    data = json.load(f, object_pairs_hook=collections.OrderedDict)

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

with open(ARGS.setup, 'w') as f:
    json.dump(data, f, indent=4, sort_keys=False)