import os
import time
import sys
import cv_functions as Functions
from pathlib import Path
import logging
from logging import handlers
import cv2
import numpy as np
import math

def get_project_root() -> Path:
    return Path(__file__).parent.parent

#######################
# Setting directories #
#######################
project_directory = str(get_project_root())
output_path = project_directory + '/images/output/'
# Change this to match your image extention
img_extention = '.png'
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
credit of this function goes for: Link009
https://github.com/Link009/LPEX
'''
def extract_car_plate(img_path, *args):
    try:    
        # define folder to save the imgs
        if len(args) > 0:
            folder_to_save = args[0]
        else:
            folder_to_save = output_path
        img = cv2.imread(img_path)
        img_name = img_path.split('/')[-1].split('.')

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hue, saturation, value = cv2.split(hsv)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

        # applying topHat/blackHat operations
        topHat = cv2.morphologyEx(value, cv2.MORPH_TOPHAT, kernel)
        blackHat = cv2.morphologyEx(value, cv2.MORPH_BLACKHAT, kernel)

        # add and subtract between morphological operations
        add = cv2.add(value, topHat)
        subtract = cv2.subtract(add, blackHat)

        # applying gaussian blur on subtract image
        blur = cv2.GaussianBlur(subtract, (5, 5), 0)

        # thresholding
        thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 19, 9)

        # cv2.findCountours() function changed from OpenCV3 to OpenCV4: now it have only two parameters instead of 3
        cv2MajorVersion = cv2.__version__.split(".")[0]
        # check for contours on thresh
        if int(cv2MajorVersion) >= 4:
            contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        else:
            imageContours, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        # get height and width
        height, width = thresh.shape

        # create a numpy array with shape given by threshed image value dimensions
        imageContours = np.zeros((height, width, 3), dtype=np.uint8)

        # list and counter of possible chars
        possibleChars = []
        countOfPossibleChars = 0

        # loop to check if any (possible) char is found
        for i in range(0, len(contours)):

            # draw contours based on actual found contours of thresh image
            cv2.drawContours(imageContours, contours, i, (255, 255, 255))

            # retrieve a possible char by the result ifChar class give us
            possibleChar = Functions.ifChar(contours[i])

            # by computing some values (area, width, height, aspect ratio) possibleChars list is being populated
            if Functions.checkIfChar(possibleChar) is True:
                countOfPossibleChars = countOfPossibleChars + 1
                possibleChars.append(possibleChar)

        imageContours = np.zeros((height, width, 3), np.uint8)

        ctrs = []

        # populating ctrs list with each char of possibleChars
        for char in possibleChars:
            ctrs.append(char.contour)

        # using values from ctrs to draw new contours
        cv2.drawContours(imageContours, ctrs, -1, (255, 255, 255))

        plates_list = []
        listOfListsOfMatchingChars = []

        for possibleC in possibleChars:

            # the purpose of this function is, given a possible char and a big list of possible chars,
            # find all chars in the big list that are a match for the single possible char, and return those matching chars as a list
            def matchingChars(possibleC, possibleChars):
                listOfMatchingChars = []

                # if the char we attempting to find matches for is the exact same char as the char in the big list we are currently checking
                # then we should not include it in the list of matches b/c that would end up double including the current char
                # so do not add to list of matches and jump back to top of for loop
                for possibleMatchingChar in possibleChars:
                    if possibleMatchingChar == possibleC:
                        continue

                    # compute stuff to see if chars are a match
                    distanceBetweenChars = Functions.distanceBetweenChars(possibleC, possibleMatchingChar)

                    angleBetweenChars = Functions.angleBetweenChars(possibleC, possibleMatchingChar)

                    changeInArea = float(abs(possibleMatchingChar.boundingRectArea - possibleC.boundingRectArea)) / float(
                        possibleC.boundingRectArea)

                    changeInWidth = float(abs(possibleMatchingChar.boundingRectWidth - possibleC.boundingRectWidth)) / float(
                        possibleC.boundingRectWidth)

                    changeInHeight = float(abs(possibleMatchingChar.boundingRectHeight - possibleC.boundingRectHeight)) / float(
                        possibleC.boundingRectHeight)

                    # check if chars match
                    if distanceBetweenChars < (possibleC.diagonalSize * 5) and \
                            angleBetweenChars < 12.0 and \
                            changeInArea < 0.5 and \
                            changeInWidth < 0.8 and \
                            changeInHeight < 0.2:
                        listOfMatchingChars.append(possibleMatchingChar)

                return listOfMatchingChars

            # here we are re-arranging the one big list of chars into a list of lists of matching chars
            # the chars that are not found to be in a group of matches do not need to be considered further
            listOfMatchingChars = matchingChars(possibleC, possibleChars)

            listOfMatchingChars.append(possibleC)

            # if current possible list of matching chars is not long enough to constitute a possible plate
            # jump back to the top of the for loop and try again with next char
            if len(listOfMatchingChars) < 3:
                continue

            # here the current list passed test as a "group" or "cluster" of matching chars
            listOfListsOfMatchingChars.append(listOfMatchingChars)

            # remove the current list of matching chars from the big list so we don't use those same chars twice,
            # make sure to make a new big list for this since we don't want to change the original big list
            listOfPossibleCharsWithCurrentMatchesRemoved = list(set(possibleChars) - set(listOfMatchingChars))

            recursiveListOfListsOfMatchingChars = []

            for recursiveListOfMatchingChars in recursiveListOfListsOfMatchingChars:
                listOfListsOfMatchingChars.append(recursiveListOfMatchingChars)

            break

        imageContours = np.zeros((height, width, 3), np.uint8)

        for listOfMatchingChars in listOfListsOfMatchingChars:
            contoursColor = (255, 0, 255)

            contours = []

            for matchingChar in listOfMatchingChars:
                contours.append(matchingChar.contour)

            cv2.drawContours(imageContours, contours, -1, contoursColor)

        for listOfMatchingChars in listOfListsOfMatchingChars:
            possiblePlate = Functions.PossiblePlate()

            # sort chars from left to right based on x position
            listOfMatchingChars.sort(key=lambda matchingChar: matchingChar.centerX)

            # calculate the center point of the plate
            plateCenterX = (listOfMatchingChars[0].centerX + listOfMatchingChars[len(listOfMatchingChars) - 1].centerX) / 2.0
            plateCenterY = (listOfMatchingChars[0].centerY + listOfMatchingChars[len(listOfMatchingChars) - 1].centerY) / 2.0

            plateCenter = plateCenterX, plateCenterY

            # calculate plate width and height
            plateWidth = int((listOfMatchingChars[len(listOfMatchingChars) - 1].boundingRectX + listOfMatchingChars[
                len(listOfMatchingChars) - 1].boundingRectWidth - listOfMatchingChars[0].boundingRectX) * 1.3)

            totalOfCharHeights = 0

            for matchingChar in listOfMatchingChars:
                totalOfCharHeights = totalOfCharHeights + matchingChar.boundingRectHeight

            averageCharHeight = totalOfCharHeights / len(listOfMatchingChars)

            plateHeight = int(averageCharHeight * 1.5)

            # calculate correction angle of plate region
            opposite = listOfMatchingChars[len(listOfMatchingChars) - 1].centerY - listOfMatchingChars[0].centerY

            hypotenuse = Functions.distanceBetweenChars(listOfMatchingChars[0],
                                                        listOfMatchingChars[len(listOfMatchingChars) - 1])
            correctionAngleInRad = math.asin(opposite / hypotenuse)
            correctionAngleInDeg = correctionAngleInRad * (180.0 / math.pi)

            # pack plate region center point, width and height, and correction angle into rotated rect member variable of plate
            possiblePlate.rrLocationOfPlateInScene = (tuple(plateCenter), (plateWidth, plateHeight), correctionAngleInDeg)

            # get the rotation matrix for our calculated correction angle
            rotationMatrix = cv2.getRotationMatrix2D(tuple(plateCenter), correctionAngleInDeg, 1.0)

            height, width, numChannels = img.shape

            # rotate the entire image
            imgRotated = cv2.warpAffine(img, rotationMatrix, (width, height))

            # crop the image/plate detected
            imgCropped = cv2.getRectSubPix(imgRotated, (plateWidth, plateHeight), tuple(plateCenter))

            # copy the cropped plate image into the applicable member variable of the possible plate
            possiblePlate.Plate = imgCropped

            # populate plates_list with the detected plate
            if possiblePlate.Plate is not None:
                plates_list.append(possiblePlate)

            # draw a ROI on the original image
            for i in range(0, len(plates_list)):
                # finds the four vertices of a rotated rect - it is useful to draw the rectangle.
                p2fRectPoints = cv2.boxPoints(plates_list[i].rrLocationOfPlateInScene)

                # roi rectangle colour
                rectColour = (0, 255, 0)

                cv2.line(imageContours, tuple(p2fRectPoints[0]), tuple(p2fRectPoints[1]), rectColour, 2)
                cv2.line(imageContours, tuple(p2fRectPoints[1]), tuple(p2fRectPoints[2]), rectColour, 2)
                cv2.line(imageContours, tuple(p2fRectPoints[2]), tuple(p2fRectPoints[3]), rectColour, 2)
                cv2.line(imageContours, tuple(p2fRectPoints[3]), tuple(p2fRectPoints[0]), rectColour, 2)

                cv2.line(img, tuple(p2fRectPoints[0]), tuple(p2fRectPoints[1]), rectColour, 2)
                cv2.line(img, tuple(p2fRectPoints[1]), tuple(p2fRectPoints[2]), rectColour, 2)
                cv2.line(img, tuple(p2fRectPoints[2]), tuple(p2fRectPoints[3]), rectColour, 2)
                cv2.line(img, tuple(p2fRectPoints[3]), tuple(p2fRectPoints[0]), rectColour, 2)

                
                cv2.imwrite(folder_to_save + img_name[0] + '_plate'+ img_extention, plates_list[i].Plate)
                return True
    
    except Exception as e:
        logger.debug('Error: {} in car plate extraction'.format(e))

def pre_segmentation_improvements(img_path, *args):
    # define folder to save the imgs
    if len(args) > 0:
        folder_to_save = args[0]
    else:
        folder_to_save = output_path
    try:
        #0 flag = cv2.IMREAD_GRAYSCALE:
        img = cv2.imread(img_path, 0)
        img_name = img_path.split('/')[-1].split('.')

        final_img = None
        # open outputed plate as gray
        thresh = 130
        img_bw = cv2.threshold(img.copy(), thresh, 255, cv2.THRESH_BINARY)[1]
        
        #cv2.imshow('test1', img_bw)
        #cv2.waitKey(0)

        contours, hier = cv2.findContours(img_bw, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        max_h, max_w = 0, 0
        for cts in contours:
            x, y, w, h = cv2.boundingRect(cts)

            segmented_plate_img = cv2.rectangle(img_bw.copy(),(x,y),(x+w,y+h),(0,255,0),1)
            segmented_plate_img = segmented_plate_img[y:y+h, x:x+w]
            

            if w >= max_w and h >= max_h:
                max_w, max_h = w, h
                final_img = segmented_plate_img
        # now i will resize the final img, so we can have a standard of the chars
        #logger.debug('img shape: {}'.format(final_img.shape))
        new_w = 250
        new_h = 50
        dim = (new_w, new_h)
        final_img = cv2.resize(final_img.copy(), dim, interpolation = cv2.INTER_AREA)

        # added dilation
        # Try changing to (4,4)
        #kernel = np.ones((3,3), np.uint8)
        #final_img = cv2.dilate(final_img.copy(), kernel, iterations=1)

        # sve the final img
        cv2.imwrite(folder_to_save + img_name[0] + img_extention, final_img)
        return True
    except Exception as e:
        logger.debug('Exception: {} found in file "{}"'.format(e, img_name[0]))
        return False

def reshape_img(img, width, height):
    dim = (width, height)
    reshaped_img = cv2.resize(img.copy(), dim, interpolation = cv2.INTER_AREA)

    return reshaped_img

def plate_segmentation(img_path, *args):
    # define folder to save the imgs
    if len(args) > 0:
        folder_to_save = args[0]
    else:
        folder_to_save = output_path
    try:
        #0 flag = cv2.IMREAD_GRAYSCALE:
        img = cv2.imread(img_path, 0)
        img_name = img_path.split('/')[-1].split('.')
        
        #I have set a random value to thresh
        thresh = 130
        img_bw = cv2.threshold(img, thresh, 255, cv2.THRESH_BINARY)[1]

        contours, hier = cv2.findContours(img_bw, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        char_count = 0
        segmented_paths = []
        for cts in contours:
            x, y, w, h = cv2.boundingRect(cts)
            
            #logger.debug('\nx:{}\ny:{}\nw:{}\nh:{}'.format(x, y, w, h))
            possiblechar = Functions.ifChar(cts)
            # change h values if problems
            # if is a a char, append it to the char paths and save in output folder
            if Functions.checkIfChar(possiblechar) is True and h >= 35 and h <= 42:
                segmented_char_img = cv2.rectangle(img_bw.copy(),(x,y),(x+w,y+h),(0,255,0),1)
                segmented_char_img = segmented_char_img[y:y+h, x:x+w]
                #logger.debug('img "{}_char_{}" shape: {}'.format(img_name[0], char_count, segmented_char_img.shape))

                img_char_path = folder_to_save+img_name[0]+'_char_{}'.format(char_count) + img_extention
                segmented_paths.append(img_char_path)
                
                # before saving, i will reshape the imgs
                segmented_char_img = reshape_img(segmented_char_img, 28, 28)
                
                # save img
                cv2.imwrite(img_char_path, segmented_char_img)
                char_count += 1
        
        return True, segmented_paths
    
    except Exception as e:
        logger.debug('Exception: {} found in file "{}_char_{}"'.format(e, img_name[0], char_count))
        return False

def posprocessing(img_path_list, thresh, erode, erode_kernel, blur, blur_kernel):
    new_img_path_list = []
    for img_path in img_path_list:
        img = cv2.imread(img_path, 0)
        img_name = img_path.split('/')[-1].split('.')

        rows,cols = img.shape
        # remove unnecessary lines that appears on reshaping
        for i in range(0, rows):
            for j in range(0, cols):
                if i == 0 or i == 1:
                    img[i, j] = 255
                if j == 0 or j == 1:
                    img[i, j] = 255
        # the EMNIST dataset used have a black background and white characters
        # my segmented files are white background and black characters
        # below code is to invert that based on a thresh pixel value
        for i in range(0, rows):
            for j in range(0, cols):
                if img[i, j] < thresh:
                    img[i, j] = (img[i, j]-255)*-1
                elif img[i, j] > thresh:
                    img[i, j] = (img[i, j]+255)/255
        
        if erode == 'y':
            # erode to make the digit more thin
            kernel = np.ones((erode_kernel,erode_kernel), np.uint8)
            img = cv2.erode(img.copy(), kernel, iterations=1)
            # update img path
            img_path = output_path + img_name[0] + '_erode' + img_extention
        if blur == 'y':
            # gausian blur may help
            img = cv2.GaussianBlur(img.copy(),(blur_kernel,blur_kernel),cv2.BORDER_DEFAULT)
            # update img path
            img_path = output_path + img_name[0] + '_blur' + img_extention
        # update img path
        if erode == 'y' and blur == 'y':
            img_path = output_path + img_name[0] + '_erode_and_blur' + img_extention
        # save new img
        cv2.imwrite(img_path, img)
        new_img_path_list.append(img_path)
    return new_img_path_list