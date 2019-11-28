'''
Here i'll be writting utility functions if needed

@AUTHOR: Régis Faria
@EMAIL: regisprogramming@gmail.com
'''
import os
import time
from pathlib import Path

def get_project_root() -> Path:
    return Path(__file__).parent.parent

#for a [0, 1] problem
'''
def dataset_menu():
    print('-------------------------')
    print('available training sets:')
    print('0. aolp-le')
    print('1. cars-dataset')
    print('2. ssig')
    print('-------------------------')

    return int(input('insert one dataset value: '))
'''

# not needed anymore
'''
def concatenate_datasets(path, dataset_index):
    # this is different for each dataset
    # the below is for 'aolp-le' dataset
    filenames = []
    if dataset_index == 0:
        for n in range(1, 52):
            filenames.append(path+str(n)+'.txt')
    elif dataset_index == 1:
        pass
    elif dataset_index == 2:
        pass
    else:
        print('INVALID DATASET INDEX INPUT')
        return -1

    with open(path+'concatened_dataset.csv', 'w') as outfile:
        for fname in filenames:
            try:
                with open(fname) as infile:
                    for line in infile:
                        outfile.write(line)
            except FileNotFoundError:
                print('The following file:', fname, 'is missing.\nSkipping...')
                continue
'''
