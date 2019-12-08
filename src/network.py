# TO-DO
# [ ] Find a way to train a model with 2 datasets
# [ ] Implement a training to the model for characters

import sys
import utils
import os
# The below 2 lines is to hide keras warnings

import logging
from logging import handlers

import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
#maybe my problem is this .core [ ] check it out later
from keras.layers.core import Dense, Activation, Dropout, Flatten
from keras.datasets import mnist
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from tflearn.data_utils import image_preloader as preloader

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

'''
def train_digit():
    #######################################################
    ################### Network setup #####################

    # batch_size - Number of images given to the model at a particular instance
    # v_length - Dimension of flattened input image size i.e. if input image size is [28x28], then v_length = 784
    # network inputs
    epochs = 25
    n_classes = 10
    batch_size = 128
    train_size = 60000
    test_size = 10000
    v_length = 784

    # split the mnist data into train and test
    (trainData, trainLabels), (testData, testLabels) = mnist.load_data()
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
    history = model.fit(trainData, mTrainLabels, validation_data=(testData, mTestLabels), batch_size=batch_size, nb_epoch=epochs, verbose=2)

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
    
    return model
'''


'''
def test_digit(model):
    ##################### TEST #####################

    q = 'c'
    while q != 'q':
        # grab some test images from the test data
        a = 1
        b = 5
        v_length = 784
        test_size = 10000
        (trainData, trainLabels), (testData, testLabels) = mnist.load_data()
        testData = testData.reshape(test_size, v_length)
        testData = testData.astype("float32")
        testData /= 255

        test_images = testData[a:b]

        # reshape the test images to standard 28x28 format
        test_images = test_images.reshape(test_images.shape[0], 28, 28)
        logger.debug("[INFO] test images shape - {}".format(test_images.shape))

        # loop over each of the test images
        for i, test_image in enumerate(test_images, start=1):
            # grab a copy of test image for viewing
            org_image = test_image
            
            # reshape the test image to [1x784] format so that our model understands
            test_image = test_image.reshape(1,784)
            
            # make prediction on test image using our trained model
            prediction = model.predict_classes(test_image, verbose=0)
            
            # display the prediction and image
            logger.debug("[INFO] I think the digit is - {}".format(prediction[0]))
            plt.subplot(220+i)
            plt.imshow(org_image, cmap=plt.get_cmap('gray'))

        plt.show()
        q = input('write "q" to quit or anything to test again')
'''

def train_character():
    #######################################################
    ################### Network input #####################

    train_path = '/home/regisf/Documents/dev/python/neural-networks/car_plate_recognition/datasets/ch74k/Fnt/train'
    test_path = '/home/regisf/Documents/dev/python/neural-networks/car_plate_recognition/datasets/ch74k/Fnt/test'
    validation_path = '/home/regisf/Documents/dev/python/neural-networks/car_plate_recognition/datasets/ch74k/Fnt/validation'
    
    epochs = 20
    
    # letters and digits(alphabet + digits = 36, but we have some upper and lower case letters)
    n_classes = 62
    
    # if stays to slow change it to '254'
    batch_size = 254
    batch_size2 = 2
    
    # total imgs on dataset: 62992
    # each folder have: 1016
    train_size = 62992
    
    v_length = 128*128
    input_shape = (128,128)

    #######################################################
    #######################################################

    # create the model
    model = Sequential()
    model.add(Dense(508, input_shape=input_shape, activation='relu'))
    model.add(Dense(254, activation='relu'))
    model.add(Dense(n_classes, activation='softmax'))

    #logger.debug(model.summary())
    # compile ||categorical_crossentropy  binary_crossentropy|| adam  rmsprop ||
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    
    # [ ] try changing categorical labels to false after || " normalize=True " change this as well
    train_x, train_y = preloader(train_path, image_shape=(128,128), mode='folder', categorical_labels=True, normalize=True)
    test_x, test_y = preloader(test_path, image_shape=(128,128), mode='folder', categorical_labels=True, normalize=True)
    
    trainData = np.array(train_x)
    testData = np.array(test_y)
    
    mTrainLabels = np_utils.to_categorical(train_y, n_classes)
    mTestLabels = np_utils.to_categorical(test_y, n_classes)
    
    model.fit(trainData, mTrainLabels, validation_data=(testData, mTestLabels), batch_size=batch_size, epochs=epochs)
    '''
    # create a data generator
    datagen = ImageDataGenerator(
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

    # load and iterate training dataset
    train_it = datagen.flow_from_directory(train_path, target_size=(128, 128), class_mode='categorical', batch_size=batch_size)
    # load and iterate validation dataset
    #val_it = datagen.flow_from_directory(validation_path, target_size=(128, 128), class_mode='categorical', batch_size=batch_size2)
    # load and iterate test dataset
    #test_it = datagen.flow_from_directory(test_path, target_size=(128, 128), class_mode='categorical', batch_size=batch_size2)
    batchX, batchy = train_it.next()
    print('Batch shape=%s, min=%.3f, max=%.3f' % (batchX.shape, batchX.min(), batchX.max()))
    print("batch y: {}".format(batchy))
    
    # fit model
    model.fit_generator(train_it, epochs=epochs, verbose=1, steps_per_epoch=2000)
    #model.save_weights('first_try.h5')  # always save your weights after training or during training
    '''

    return model

# [ ] 
def identify_plate(model, imgs_path, size):
    v_length = 784
    # send how many segmentations you have
    test_size = size
    imgs = []
    for img in imgs_path:
        imgs.append(cv2.imread(img, 0))

    # omiting "testLabel" bcause i still dont know if i will have this
    #testData = imgs
    # [ ] test this on local dataset
    #testData = testData.reshape(test_size, v_length)
    #testData = testData.astype("float32")
    #testData /= 255

    #test_images = testData[a:b]

    # reshape the test images to standard 128x128 format
    #test_images = imgs.reshape(imgs.shape[0], 128, 128)
    #logger.debug("[INFO] test images shape - {}".format(test_images.shape))

    # loop over each of the test images
    #for i, test_image in enumerate(test_images, start=1):
    plate_prediction = []
    for image in imgs:
        # reshape the test image to [1x16384] format so that our model understands
        # try comenting below line
        image = image.reshape(1,128*128)
        
        # make prediction on test image using our trained model
        prediction = model.predict_classes(image, verbose=0)
        
        plate_prediction.append(prediction[0])

    '''
    plate = ''
    for char in plate_prediction:
        plate += char
    '''
    # output the recognized plate
    logger.debug("The car plate is: {}".format(plate))

if __name__ == '__main__':
    '''
    # this works for digit network
    # NOTE: to test uncomment respectives functions as well
    model = train_digit()
    test_digit(model)
    '''
    p = '/home/regisf/Documents/dev/python/neural-networks/car_plate_recognition/output/01_plate_char_1.jpg'

    model = train_character()
    identify_plate(model, p, 1)