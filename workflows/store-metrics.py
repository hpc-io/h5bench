import os
import csv
import sys
import glob
import gspread

import numpy as np
import pandas as pd

from gspread_dataframe import set_with_dataframe
from google.oauth2.service_account import Credentials
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive

SPREADSHEET_ID = os.getenv('GOOGLE_SPREADSHEET_ID')
CREDENTIALS = os.getenv('GOOGLE_CREDENTIALS')

dataset = None

experiments = [
    'contiguous-contiguous',
    'interleaved-interleaved'
]

scopes = [
    'https://www.googleapis.com/auth/spreadsheets',
    'https://www.googleapis.com/auth/drive'
]

credentials = Credentials.from_service_account_file(
    CREDENTIALS,
    scopes=scopes
)

gc = gspread.authorize(credentials)

gauth = GoogleAuth()
drive = GoogleDrive(gauth)

gs = gc.open_by_key(SPREADSHEET_ID)

for test in glob.glob('h5bench-suite-sync*.err'):
    print(test)

    tmp = None
    found = False
    dimensions = None
    connector = None

    with open(test, 'r') as f:
        lines = f.readlines()

        index = -1

        for line in lines:
            if 'Starting' in line and 'Suite' not in line:
                dir_hash = None
                status = None
                filename = None

                start_date = line.split(' ')[0].strip()
                start_time = line.split(' ')[1].strip()

                index += 1

            if 'VOL connector' in line:
                connector = line.split(' ')[9].strip()

            if 'DIR:' in line:
                dir_hash = line.split(' ')[10].strip()

            if 'srun' in line and 'Parallel setup' not in line:
                dimensions = line.split(' ')[-1].strip().split('-')[-2]

                filename = line.split(' ')[-1].strip().split('/')[-1]

            if 'SUCCESS' in line:
                for file in glob.glob('{}*.csv'.format(dir_hash)):
                    try:
                        df = pd.read_csv(file, index_col=0).T.head(1)
                        units = pd.read_csv(file, index_col=0).T.tail(1)
                        
                        unit = units['total size'].iloc[0].strip()
                        if unit not in ('MB', 'GB', 'TB'):
                            print('CRITICAL - Unhandled unit: ', unit)
                            exit()
                        if unit == 'MB':
                            print('-> Converting `total size` from MB to GB')
                            df['total size'].iloc[0] = float(df['total size'].iloc[0]) / 1024
                        if unit == 'TB':
                            print('-> Converting `total size` from TB to GB')
                            df['total size'].iloc[0] = float(df['total size'].iloc[0]) * 1024 * 1024

                        unit = units['raw rate'].iloc[0].strip()
                        if unit not in ('MB/s', 'GB/s', 'TB/s'):
                            print('CRITICAL - Unhandled unit: ', unit)
                            exit()
                        if unit == 'MB/s':
                            print('-> Converting `raw rate` from MB/s to GB/s')
                            df['raw rate'].iloc[0] = float(df['raw rate'].iloc[0]) / 1024
                        if unit == 'TB/s':
                            print('-> Converting `raw rate` from TB/s to GB/s')
                            df['raw rate'].iloc[0] = float(df['raw rate'].iloc[0]) * 1024 * 1024

                        unit = units['observed rate'].iloc[0].strip()
                        if unit not in ('MB/s', 'GB/s', 'TB/s'):
                            print('CRITICAL - Unhandled unit: ', unit)
                            exit()
                        if unit == 'MB/s':
                            print('-> Converting `observed rate` from MB/s to GB/s')
                            df['observed rate'].iloc[0] = float(df['observed rate'].iloc[0]) / 1024
                        if unit == 'TB/s':
                            print('-> Converting `observed rate` from TB/s to GB/s')
                            df['observed rate'].iloc[0] = float(df['observed rate'].iloc[0]) * 1024 * 1024

                        df['status'] = 'SUCCESS'
                        df['date'] = start_date
                        df['time'] = start_time
                        df['pattern'] = experiments[index]
                        df['dimensions'] = dimensions
                        df['connector'] = connector

                        h5_stdout = '{}stdout'.format(dir_hash)

                        with open(h5_stdout, 'r') as h5_f:
                            h5_lines = h5_f.readlines()
                            
                            tags = [
                                'H5Fcreate',
                                'H5Fflush',
                                'H5Fclose'
                            ]

                            h5_timers = {}

                            for tag in tags:
                                h5_timers[tag] = None

                            for h5_line in h5_lines:
                                for tag in tags:
                                    if tag in h5_line:
                                        h5_info = h5_line.split()
                                        h5_timers[tag] = h5_info[2]

                            for tag in tags:
                                df[tag] = h5_timers[tag]

                        if tmp is None:
                            tmp = df
                        else:
                            tmp = pd.concat([tmp, df])

                        found = True
                    except Exception as e:
                        print(e)
                        print('Unable to parse: {}'.format(file))

                dimensions = None
                connector = None

    if found:
        tmp.fillna('', inplace=True)

        if dataset is None:
            dataset = tmp
        else:
            dataset = pd.concat([dataset, tmp]) 

dataset['ranks'] = dataset['ranks'].str.strip()
dataset['operation'] = dataset['operation'].str.strip()
dataset['collective data'] = dataset['collective data'].str.strip()
dataset['collective meta'] = dataset['collective meta'].str.strip()
dataset['subfiling'] = dataset['subfiling'].str.strip()

dataset['collective data'] = dataset['collective data'].replace('', 'NO')
dataset['collective meta'] = dataset['collective meta'].replace('', 'NO')

print(dataset)

dataset.applymap(lambda x: x.strip() if isinstance(x, str) else x)

df_values = dataset.values.tolist()

gs.values_append(
    sys.argv[1],
    {
        'valueInputOption': 'USER_ENTERED'
    },
    {
        'values': df_values
    }
)
