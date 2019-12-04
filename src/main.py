'''
MLP for car plate recognition 

@AUTHOR: Régis Faria
@EMAIL: regisprogramming@gmail.com
'''

#######  DEVELOPMENT STEPS  #######
# [X] from a car image, extract it's plate
# [ ] implement a way to check if the extract plate was a success
# [X] segmentation of the car plate
# [X] save the segmented plate as a new img for each char
# [ ] implement the network
# [ ] send the segmented imgs to the network
# [ ] output info
# [ ] finish github repo

##########  OPTIONAL LIST  ###########
# [ ] find a training digit and char dataset
# [ ] find a way to get how many files are in a folder
# [ ] threading for plate recognition
# [ ] check this out later - dataset
# / mnist / emnist /

import sys
import utils
import os
import time
from tqdm import tqdm

import logging
from logging import handlers

from PIL import Image

# Setting directories 
project_directory = str(utils.get_project_root())
output_path = project_directory + '/output/'
datasets_path = project_directory + '/datasets/'
script_name = os.path.basename(__file__)

# Output folder setup
if not os.path.exists(output_path):
    os.makedirs(output_path)

# Logs setup
logs_directory = os.path.join(project_directory, 'logs')
if not os.path.exists(logs_directory):
    os.makedirs(logs_directory)

# LOGGING
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
log_template = '%(asctime)s %(module)s %(levelname)s: %(message)s'
formatter = logging.Formatter(log_template)

# Logging - File Handler
log_file_size_in_mb = 10
count_of_backups = 5  # example.log example.log.1 example.log.2
log_file_size_in_bytes = log_file_size_in_mb * 1024 * 1024
log_filename = os.path.join(logs_directory, os.path.splitext(script_name)[0]) + '.log'
file_handler = handlers.RotatingFileHandler(log_filename, maxBytes=log_file_size_in_bytes,
                                            backupCount=count_of_backups)
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# Logging - STDOUT Handler
stdout_handler = logging.StreamHandler(sys.stdout)
stdout_handler.setFormatter(formatter)
logger.addHandler(stdout_handler)

def select_dataset(path):
    '''
    This will return a list containing both test and train dataset location
    where list[0] == train and list[1] == test
    '''
    
    print('-----------------------------------------------------------------')
    logger.debug('Please, select one dataset:')
    logger.debug('Dataset 1: Turkish car plates')
    logger.debug('Dataset 2: Random car plates')
    logger.debug('Insert dataset value:')
    selected = int(input())
    
    path_list = []
    if selected == 1:
        path_list.append(path + '/datasets/1/train/')
        path_list.append(path + '/datasets/1/test/')
    elif selected == 2:
        path_list.append(project_directory + '/datasets/2/train/')
        path_list.append(project_directory + '/datasets/2/test/')
    else:
        logger.debug('Invalid option, make sure to pick one of the listed numbers.')
    print('-----------------------------------------------------------------')

    return path_list
#######################################################

if __name__ == '__main__':
    #[ ] here i'll create a menu
    ############################
    dataset_path = select_dataset(project_directory)

    '''
    # test for multiple imgs
    try:    
        # here i will make a quick test for img extract
        for i in tqdm(range(1, 21)):
            if i < 10:
                image = dataset_path[0] + '0' + str(i) + '.jpg'
            else:
                image = dataset_path[0] + str(i) + '.jpg'
            utils.extract_car_plate(image, output_path+'plate_out/')

            utils.plate_segmentation(image, output_path+'plate_out/')
    except Exception as e:
        logger.debug(e)
    '''
    
    # test for 1 img only 
    try:
        image = dataset_path[0] + '08.jpg'
        imageout = output_path + '08_plate.jpg'
        # Extract car's plate from a car img
        utils.extract_car_plate(image, output_path)
        # Improvements on the car extracted plate, removing some blank sides
        utils.pre_segmentation_improvements(imageout, output_path)
        # Segmentation on each character of the plate
        utils.plate_segmentation(imageout, output_path)
    except Exception as e:
        logger.debug(e)
    