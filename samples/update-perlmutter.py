import os
import json
import argparse
import collections

PARSER = argparse.ArgumentParser(
    description='H5bench: update configuration.json with SLRUM commands for Perlmutter'
)

PARSER.add_argument(
    'setup',
    action='store',
    help='JSON file with the benchmarks to run'
)

ARGS = PARSER.parse_args()

with open(ARGS.setup, 'r') as f:
    data = json.load(f, object_pairs_hook=collections.OrderedDict)

data['mpi']['command'] = ''
del data['mpi']['ranks']

data['mpi']['configuration'] = 'srun -A m2621 --qos=debug --constraint=cpu --tasks-per-node=64 -N 1 -n 4 -t 00:30:00'

if 'library' in data['vol']:
    data['vol']['library'] = '{}:{}'.format(
        data['vol']['library'],
        os.getenv('LD_LIBRARY_PATH')
    )

with open(ARGS.setup, 'w') as f:
    json.dump(data, f, indent=4, sort_keys=False)