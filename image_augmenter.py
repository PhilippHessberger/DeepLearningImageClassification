import cv2
import numpy as np
import random


# shapes an image and rotates it by a random amount
def rotate_image_random(image, shape=None):
    if shape is not None:
        image = reshape_image(image, shape)
    height, width = image.shape[:2]
    angle = random.randint(0, 360)
    rotation_matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)
    rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))
    mask = cv2.inRange(rotated_image, (0, 0, 0), (0, 0, 0))
    rotated_image[mask == 255] = (255, 255, 255)

    return rotated_image

def change_white_background(image, background_image=None, shape=None):

    # Convert image to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define lower and upper bounds for white color
    lower_white = np.array([0, 0, 100])
    upper_white = np.array([360, 25, 255])

    # Create a mask for white pixels
    mask = cv2.inRange(hsv, lower_white, upper_white)

    # Perform closing operation on white pixels
    kernel = np.ones((3, 3), np.uint8)
    closed_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Invert the mask
    inverted_mask = cv2.bitwise_not(closed_mask)

    # Apply the mask to the image
    result = cv2.bitwise_and(image, image, mask=inverted_mask)

    if background_image is not None:
        background_image = reshape_image(background_image, shape)
        random_color_image = background_image
    else:
        # Generate a random color in BGR format with the same shape and dtype as the image
        random_color_image = np.full(image.shape, np.random.randint(0, 256, size=3, dtype=np.uint8))

    # Replace the white pixels with the random color
    background = cv2.bitwise_and(random_color_image, random_color_image, mask=closed_mask)

    # Combine the result and the background
    final_image = cv2.add(result, background)

    cv2.waitKey()
    cv2.destroyAllWindows()

    return final_image

def closing_operation(image, rgb_color):
    black_image = np.zeros(image.shape, dtype=np.uint8)
    # create a mask that has 1 for every pixel of the color rgb_color
    mask = cv2.inRange(image, rgb_color - 1, rgb_color + 1)

    # reverse mask, so foreground is 1
    reversed_mask = cv2.bitwise_not(mask)

    kernel = np.ones((3, 3), np.uint8)

    closed_mask = cv2.morphologyEx(reversed_mask, cv2.MORPH_CLOSE, kernel)

    result = cv2.bitwise_and(image, image, mask=closed_mask)

    return result

def reshape_image(image, shape):
    height, width = image.shape[:2]

    ratio = min(shape[0] / width, shape[1] / height)
    new_size = (int(width * ratio), int(height * ratio))

    resized_img = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)

    top = ((shape[1] - new_size[1]) // 2)
    bottom = ((shape[1] - new_size[1]) // 2)
    left = ((shape[0] - new_size[0]) // 2)
    right = ((shape[0] - new_size[0]) // 2)

    if (shape[1] - new_size[1]) % 2 == 1:
        top = top + 1
    if (shape[0] - new_size[0]) % 2 == 1:
        right = right + 1

    result = cv2.copyMakeBorder(resized_img,
                                top,
                                bottom,
                                left,
                                right,
                                cv2.BORDER_CONSTANT,
                                value=(255, 255, 255))

    return result
