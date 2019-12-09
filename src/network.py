# TO-DO
# [ ] Find a way to train a model with 2 datasets
# [ ] Implement a training to the model for characters

import sys
import utils
import os
import logging
from logging import handlers
import tensorflow as tf
import emnist
import h5py
import cv2
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout, Flatten
from keras.utils import np_utils

# Setting directories 
project_directory = str(utils.get_project_root())
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

def train_emnist(u_epochs):
    #######################################################
    ################### Network setup #####################

    # batch_size - Number of images given to the model at a particular instance
    # v_length - Dimension of flattened input image size i.e. if input image size is [28x28], then v_length = 784
    # network inputs
    epochs = u_epochs
    n_classes = 62
    batch_size = 256
    train_size = 697932
    test_size = 116323
    v_length = 784

    # split the emnist data into train and test
    trainData, trainLabels = emnist.extract_training_samples('byclass') 
    testData, testLabels = emnist.extract_test_samples('byclass')

    # print shapes
    logger.debug("[INFO] train data shape: {}".format(trainData.shape))
    logger.debug("[INFO] test data shape: {}".format(testData.shape))
    logger.debug("[INFO] train samples: {}".format(trainData.shape[0]))
    logger.debug("[INFO] test samples: {}".format(testData.shape[0]))

    # reshape the dataset
    trainData = trainData.reshape(train_size, v_length)
    testData = testData.reshape(test_size, v_length)

    trainData = trainData.astype("float32")
    testData = testData.astype("float32")

    trainData /= 255
    testData /= 255

    logger.debug("[INFO] after re-shape")
    # print new shape
    logger.debug("[INFO] train data shape: {}".format(trainData.shape))
    logger.debug("[INFO] test data shape: {}".format(testData.shape))
    logger.debug("[INFO] train samples: {}".format(trainData.shape[0]))
    logger.debug("[INFO] test samples: {}".format(testData.shape[0]))

    # convert class vectors to binary class matrices --> one-hot encoding
    mTrainLabels = np_utils.to_categorical(trainLabels, n_classes)
    mTestLabels = np_utils.to_categorical(testLabels, n_classes)

    # create the model
    model = Sequential()
    model.add(Dense(512, input_shape=(784,)))
    model.add(Activation("relu"))
    model.add(Dropout(0.2))
    
    model.add(Dense(256))
    model.add(Activation("relu"))
    model.add(Dropout(0.2))
    
    model.add(Dense(n_classes))
    model.add(Activation("softmax"))

    # summarize the model
    model.summary()
    # compile
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    # fit the model
    history = model.fit(trainData, mTrainLabels, validation_data=(testData, mTestLabels), batch_size=batch_size, epochs=epochs, verbose=2)

    # print the history keys
    logger.debug(history.history.keys())

    
    # evaluate the model
    scores = model.evaluate(testData, mTestLabels, verbose=0)

    # history plot for accuracy
    plt.plot(history.history["accuracy"])
    plt.plot(history.history["val_accuracy"])
    plt.title("Model Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend(["train", "test"], loc="upper left")
    plt.show()

    # history plot for accuracy
    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    plt.title("Model Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(["train", "test"], loc="upper left")
    plt.show()

    # print the results
    logger.debug("[INFO] test score - {}".format(scores[0]))
    logger.debug("[INFO] test accuracy - {}".format(scores[1]))
    
    #model.save_weights('first_try.h5')

    return model

def load_trained_model():
    #######################################################
    ################### Network setup #####################
    n_classes = 62
    train_size = 697932
    test_size = 116323
    v_length = 784

    # split the emnist data into train and test
    trainData, trainLabels = emnist.extract_training_samples('byclass') 
    testData, testLabels = emnist.extract_test_samples('byclass')

    # reshape the dataset
    trainData = trainData.reshape(train_size, v_length)
    testData = testData.reshape(test_size, v_length)

    trainData = trainData.astype("float32")
    testData = testData.astype("float32")

    trainData /= 255
    testData /= 255

    # convert class vectors to binary class matrices --> one-hot encoding
    mTrainLabels = np_utils.to_categorical(trainLabels, n_classes)
    mTestLabels = np_utils.to_categorical(testLabels, n_classes)

    # create the model
    model = Sequential()
    model.add(Dense(512, input_shape=(784,)))
    model.add(Activation("relu"))
    model.add(Dropout(0.2))
    
    model.add(Dense(256))
    model.add(Activation("relu"))
    model.add(Dropout(0.2))
    
    model.add(Dense(n_classes))
    model.add(Activation("softmax"))

    # load weights
    model.load_weights('model_weights.h5')

    # summarize the model
    model.summary()
    # compile
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    # evaluate the model
    scores = model.evaluate(testData, mTestLabels, verbose=0)

    return model

def test_emnist(model, n1, n2):
    ##################### TEST #####################
    # EMNIST output infos as numbers, so I created a label dict, so it will output it respective class
    label_value = {'0':'0', '1':'1', '2':'2', '3':'3', '4':'4', '5':'5', '6':'6', '7':'7', '8':'8', '9':'9', 
                   '10':'A', '11':'B', '12':'C', '13':'D', '14':'E', '15':'F', '16':'G', '17':'H', '18':'I', '19':'J', 
                   '20':'K', '21':'L', '22':'M', '23':'N', '24':'O', '25':'P', '26':'Q', '27':'R', '28':'S', '29':'T', 
                   '30':'U', '31':'V', '32':'W', '33':'X', '34':'Y', '35':'Z', '36':'a', '37':'b', '38':'c', '39':'d',
                   '40':'e', '41':'f', '42':'g', '43':'h', '44':'i', '45':'j', '46':'k', '47':'l', '48':'m', '49':'n',
                   '50':'o', '51':'p', '52':'q', '53':'r', '54':'s', '55':'t', '56':'u', '57':'v', '58':'w', '59':'x',
                   '60':'y', '61':'z'}
    
    # grab some test images from the test data
    a = n1
    b = n2
    v_length = 784
    test_size = 116323
    
    # load train data
    testData, testLabels = emnist.extract_test_samples('byclass')
    
    # reshape
    testData = testData.reshape(test_size, v_length)
    testData = testData.astype("float32")
    testData /= 255

    test_images = testData[a:b]

    # reshape the test images to standard 28x28 format
    test_images = test_images.reshape(test_images.shape[0], 28, 28)

    # loop over each of the test images
    for i, test_image in enumerate(test_images, start=1):
        # grab a copy of test image for viewing
        org_image = test_image
        
        # reshape the test image to [1x784] format so that our model understands
        test_image = test_image.reshape(1,784)
        
        # make prediction on test image using our trained model
        prediction = model.predict_classes(test_image, verbose=0)
        
        # display the prediction and image
        logger.debug("I think the character is - {}".format(label_value[str(prediction[0])]))
        plt.subplot(220+i)
        plt.imshow(org_image, cmap=plt.get_cmap('gray'))

    logger.debug('Press Q to close')
    plt.show()

# params: 1- mlmodel, 2- root path to the prediction imgs, 3- how many imgs we have in imgs_path
def identify_plate(model, imgs_path, test_size):
    # EMNIST output infos as numbers, so I created a label dict, so it will output it respective class
    label_value = {'0':'0', '1':'1', '2':'2', '3':'3', '4':'4', '5':'5', '6':'6', '7':'7', '8':'8', '9':'9', 
                   '10':'A', '11':'B', '12':'C', '13':'D', '14':'E', '15':'F', '16':'G', '17':'H', '18':'I', '19':'J', 
                   '20':'K', '21':'L', '22':'M', '23':'N', '24':'O', '25':'P', '26':'Q', '27':'R', '28':'S', '29':'T', 
                   '30':'U', '31':'V', '32':'W', '33':'X', '34':'Y', '35':'Z', '36':'a', '37':'b', '38':'c', '39':'d',
                   '40':'e', '41':'f', '42':'g', '43':'h', '44':'i', '45':'j', '46':'k', '47':'l', '48':'m', '49':'n',
                   '50':'o', '51':'p', '52':'q', '53':'r', '54':'s', '55':'t', '56':'u', '57':'v', '58':'w', '59':'x',
                   '60':'y', '61':'z'}
    # 28*28
    v_length = 784
    
    # open imgs
    testData = []
    for img in imgs_path:
        testData.append(cv2.imread(img, 0))

    # normalize the img data
    testData = testData.reshape(test_size, v_length)
    testData = testData.astype('float32')
    testData /= 255

    test_images = testData
    test_images = test_images.reshape(test_images.shape[0], 28, 28)

    # in pos_predict i will store the original img and its respective prediction
    plate_predictions, predict_result = [], []
    for test_image in test_images:
        # grab a copy of test image for viewing
        original_img = test_image
        
        # reshape the test image to [1x784] format so that our model understands
        test_image = test_image.reshape(1,784)
        
        # make prediction on test image using our trained model
        prediction = model.predict_classes(test_image, verbose=0)
        
        # make prediction on test image using our trained model
        prediction = model.predict_classes(image, verbose=0)
        plate_predictions.append(label_value[str(prediction[0])])
        
        predict_result.append([original_img, label_value[str(prediction[0])]])
        #plt.imshow(org_image, cmap=plt.get_cmap('gray'))

    plate = ''
    for char in plate_predictions:
        plate += char
    
    # output the recognized plate
    logger.debug("The car plate is: {}".format(plate))
    logger.debug("Now the system will show each predicted picture and it's respective prediction")
    logger.debug("NOTE: Press any key to close img window")

    for i in range(predict_result):
        logger.debug('This character is: {}'.format(predict_result[i][1]))
        cv2.imshow(predict_result[i][0])
        cv2.waitKey(0)


if __name__ == '__main__':
    model = load_trained_model()
    test_emnist(model, 5, 9)