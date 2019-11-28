'''
MLP for recognize car plates 

@AUTHOR: Régis Faria
@EMAIL: regisprogramming@gmail.com
'''
import sys
import utils
import os
import time
import logging
from logging import handlers

# Setting directories 
project_directory = utils.get_project_root()
samples_path = project_directory + '/output/' 
output_path = project_directory + '/samples/'
script_name = os.path.basename(__file__)

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

