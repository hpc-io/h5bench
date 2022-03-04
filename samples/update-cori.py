import os
import json
import argparse
import collections

PARSER = argparse.ArgumentParser(
    description='H5bench: update configuration.json with SLRUM commands for Cori'
)

PARSER.add_argument(
    'setup',
    action='store',
    help='JSON file with the benchmarks to run'
)

ARGS = PARSER.parse_args()

with open(ARGS.setup, 'r') as f:
    data = json.load(f, object_pairs_hook=collections.OrderedDict)

data['mpi']['command'] = 'srun'
data['mpi']['ranks'] = '4'

del data['mpi']['configuration']

with open(ARGS.setup, 'w') as f:
    json.dump(data, f, indent=4, sort_keys=False)