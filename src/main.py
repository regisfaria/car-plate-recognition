#######  DEVELOPMENT STEPS  #######
# [X] from a car image, extract it's plate
# [X] implement a way to check if the extract plate was a success if not skip
# [X] segmentation of the car plate
# [X] save the segmented plate as a new img for each char
# [X] make a list return with path for the segmented chars/digits 
# [X] create a function to re-scale the digits/chars. maybe 28x28 *pick same as dataset
# [X] implement the network
# [X] find vehicle imgs
# [X] send the segmented imgs to the network
# [X] output info
# [X] finish github repo

###### ISSUES ######
# The plate's characters are not being recognized in it's order, so the output isn't the same as plate
# The network mistake a lot

import utils
import network
import sys
import os
import time
from tqdm import tqdm
import logging
from logging import handlers

#######################
# Setting directories #
#######################
project_directory = str(utils.get_project_root())
output_path = project_directory + '/images/output/'
# Here you should set where you car images are
vehicles_path = project_directory + '/images/vehicles/'
# Change this to match your images extention
img_extention = '.png'
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

def print_folder_files(path):
    names = []
    for f in os.listdir(path):
        logger.debug(f)
        names.append(f.split('.')[0])
    return names


if __name__ == '__main__':    
    while True:
        print('\n\n---------------------------------------------------------------------------')
        
        logger.debug("----- CAR'S PLATE RECOGNITION MLP -----")
        logger.debug("NAVIGATION MENU")
        logger.debug("1. Train network")
        logger.debug("2. Load trained network")
        logger.debug("3. Test network with test data")
        logger.debug("4. Recognize car's plate")
        logger.debug("5. Quit\n")
        logger.debug('NOTE: you must train or load the network before option 3 and 4')
        logger.debug('NOTE: training takes ~1min per epoch.')
        user_choice = int(input())
        
        # train
        if user_choice == 1:
            logger.debug('Please, state how many epochs you want to train the model:\n')
            ep = int(input('epochs = '))
            
            logger.debug('[INFO] Training will begin soon...')
            model = network.train_emnist(ep)
            logger.debug('[INFO] Training completed')
        
        # load pre trained
        elif user_choice == 2:
            model = network.load_trained_model()
            logger.debug('[INFO] Model loaded')
        
        # test model with emnist data
        elif user_choice == 3:
            try:
                if model:
                    logger.debug("[INFO] Model is loaded")
            except Exception as e:
                logger.debug(e)
                logger.debug("Please load(op2) or train(op1) the network before you chose this option")
                continue

            logger.debug('Before testing, give two numbers to pick samples from the dataset')
            logger.debug('NOTE: first number should be lower then the first one. (we will do n1 until n2 samples)')
            logger.debug('NOTE: you must pick a 4 digits difference. (i.e. n1 = 1; n2 = 5)')

            user_n1 = int(input('N1 = '))
            user_n2 = int(input('N2 = '))
            network.test_emnist(model, user_n1, user_n2)
        # extract car's plate and send to NN
        elif user_choice == 4:
            try:
                if model:
                    logger.debug("[INFO] Model is loaded")
            except Exception as e:
                logger.debug(e)
                logger.debug("Please load(op2) or train(op1) the network before you chose this option")
                continue
            
            logger.debug('Listing available car images...')
            choices = print_folder_files(vehicles_path)
            
            logger.debug('Chose one. NOTE: only the name')
            img_name = input('Img name: ')
            valid_choice = False
            for c in choices:
                if img_name == c:
                    valid_choice = True
            
            if valid_choice:
                try:
                    img_path = vehicles_path + img_name + img_extention
                    
                    # Extract the plate
                    success = utils.extract_car_plate(img_path, output_path)
                    if not success:
                        logger.debug('Invalid plate extraction for "{}".'.format(img_path.split('/')[-1]))
                        continue

                    # set up new img path
                    extracted_img_path = output_path + img_name + '_plate' + img_extention

                    # Improvements on the car extracted plate, removing some blank sides
                    success = utils.pre_segmentation_improvements(extracted_img_path, output_path)
                    if not success:
                        logger.debug('Invalid pre segmentation for "{}"'.format(extracted_img_path.split('/')[-1]))
                        continue

                    # Segmentation on each character of the plate
                    success, char_imgs = utils.plate_segmentation(extracted_img_path, output_path)
                    if not success:
                        logger.debug('Invalid segmentation for "{}". Skipping...'.format(extracted_img_path.split('/')[-1]))
                        continue

                    # After reshaping the img to the same amount of pixels from my dataset, i'll now need
                    # to make a posprocessing, because the first row and collunm of the image is a black line
                    # and in some cases that could cause a problem
                    # I'm making the blur and erode optional, because some imgs get wrecked using erode
                    # so sometimes it's not needed, or it just need a lower value, so it's best like this
                    logger.debug('Do you wish to apply erode function on each char? (y/n): ')
                    erode = input()
                    if erode == 'y':
                        logger.debug('Please choose a erode size. (ideal value is 3): ')
                        erode_size = int(input())
                    else:
                        erode_size = None
                    
                    logger.debug('Do you wish to apply blur function on each char? (y/n): ')
                    blur = input()
                    if blur == 'y':
                        logger.debug('Please choose a blur itensity. (ideal value is 3): ')
                        blur_size = int(input())
                    else:
                        blur_size = None
                    
                    for img in char_imgs:
                        utils.posprocessing(img, 50, erode, erode_kernel, blur, blur_kernel)

                    # now its time to send it to the NN
                    network.identify_plate(model, char_imgs, len(char_imgs))
                except Exception as e:
                    logger.debug(e)
                    continue
            else:
                logger.debug('You have pick a wrong option. Try again later.')
        # quit
        elif user_choice == 5:
            logger.debug("Tks for using")
            print('---------------------------------------------------------------------------')
            break
        
        # error input
        else:
            logger.debug('Invalid choice, try again.')
        print('---------------------------------------------------------------------------')