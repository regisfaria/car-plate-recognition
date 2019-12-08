'''
MLP for car plate recognition 

@AUTHOR: Régis Faria
@EMAIL: regisprogramming@gmail.com
'''

#######  DEVELOPMENT STEPS  #######
# [X] from a car image, extract it's plate
# [X] implement a way to check if the extract plate was a success if not skip
# [X] segmentation of the car plate
# [X] save the segmented plate as a new img for each char
# [X] make a list return with path for the segmented chars/digits 
# [X] create a function to re-scale the digits/chars. maybe 28x28 *pick same as dataset
# [ ] implement the network
# [ ] send the segmented imgs to the network
# [ ] output info
# [ ] finish github repo

##########  OPTIONAL STEPS  ###########
# [ ] find a way to get how many files are in a folder and delete the invalid ones

import sys
import utils
import os
import time
from tqdm import tqdm

import logging
from logging import handlers

import numpy as np
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.datasets import mnist
from keras.utils import np_utils

# random seed
np.random.seed(9)

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

if __name__ == '__main__':

    # image dataset selection
    # list[0] == train and list[1] == test
    dataset_path = []
    dataset_path.append(project_directory + '/datasets/1/train/')
    dataset_path.append(project_directory + '/datasets/1/test/')

    '''
    while True:
        print('\n\n---------------------------------------------------------------------------')
        # I'll add more options if needed
        logger.debug("----- CAR'S PLATE RECOGNITION MLP -----")
        logger.debug("NAVIGATION MENU")
        logger.debug("1. Train network")
        logger.debug("2. Use the network")
        logger.debug("3. Quit\n")
        user_choice = int(input())
        if user_choice == 1:
            pass
        elif user_choice == 2:
            pass
        elif user_choice == 3:
            logger.debug("Tks for using")
            print('---------------------------------------------------------------------------')
            break
        else:
            logger.debug('Invalid choice, try again.')
        print('---------------------------------------------------------------------------')
    '''
    # Imgs with problem: 
    # 03, 04, 13, 14, 15, 16
    # 17, 20
    
    #'''
    # test for multiple imgs
    try:    
        # here i will make a quick test for img extract
        for i in tqdm(range(1, 21)):
            if i < 10:
                image = dataset_path[0] + '0' + str(i) + '.jpg'
            else:
                image = dataset_path[0] + str(i) + '.jpg'
            # Extract car's plate from a car img
            success = utils.extract_car_plate(image, output_path)
            if not success:
                logger.debug('Invalid plate extraction for "{}". Skipping...'.format(image.split('/')[-1]))
                continue

            if i < 10:
                image_extracted = output_path + '0' + str(i) + '_plate.jpg'
            else:
                image_extracted = output_path + str(i) + '_plate.jpg'
            
            # Improvements on the car extracted plate, removing some blank sides
            success = utils.pre_segmentation_improvements(image_extracted, output_path)
            if not success:
                logger.debug('Invalid pre segmentation for "{}". Skipping...'.format(image_extracted.split('/')[-1]))
                # [ ] find a way to list and then delete the file created in output
                continue
            
            # Segmentation on each character of the plate
            success, char_imgs = utils.plate_segmentation(image_extracted, output_path)
            if not success:
                logger.debug('Invalid segmentation for "{}". Skipping...'.format(image_extracted.split('/')[-1]))
                # [ ] find a way to list and then delete the file created in output
                continue
            # After reshaping the img to the same amount of pixels from my dataset, i'll now need
            # to make a posprocessing, because the first row and collunm of the image is a black line
            # and in some cases that could cause a problem
            for img in char_imgs:
                utils.posprocessing(img)
            
            # Now i'll send the chars to the network

            # read development steps
    except Exception as e:
        logger.debug(e)
    #'''
    '''
    # test for 1 img only 
    # Imgs with problem: 03, 04, 13, 14, 15, 16
    try:
        image = dataset_path[0] + '05.jpg'
        imageout = output_path + '05_plate.jpg'
        # Extract car's plate from a car img
        utils.extract_car_plate(image, output_path)
        # Improvements on the car extracted plate, removing some blank sides
        utils.pre_segmentation_improvements(imageout, output_path)
        # Segmentation on each character of the plate
        utils.plate_segmentation(imageout, output_path)
    except Exception as e:
        logger.debug(e)
    '''