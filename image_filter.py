import cv2
import numpy as np


def filter_by_white_background(images, filenames=None):
    # create an empty list to store filtered images
    filtered_images = []
    filtered_images_filenames = []

    # loop through images
    # for image in images:
    for i in range(len(images)):
        # convert the image to HSV color space
        hsv = cv2.cvtColor(images[i], cv2.COLOR_BGR2HSV)

        # define lower and upper bounds for white
        lower_white = np.array([0, 0, 100])
        upper_white = np.array([360, 25, 255])

        # create mask for white color
        mask = cv2.inRange(hsv, lower_white, upper_white)

        # count number of white pixels in mask
        white_pixels = np.sum(mask == 255)

        # calculate percentage of white pixels in mask
        white_percentage = white_pixels / (mask.shape[0] * mask.shape[1])

        # if percentage is above a certain threshold, add image to filtered list
        if white_percentage > 0.5:
            filtered_images.append(images[i])
            if filenames is not None:
                filtered_images_filenames.append(filenames[i])

    # return filtered list
    if len(filtered_images_filenames) > 0 and len(filtered_images) > 0:
        return filtered_images, filtered_images_filenames
    elif len(filtered_images) > 0:
        return filtered_images
    elif filenames is None:
        return None
    else:
        return None, None
