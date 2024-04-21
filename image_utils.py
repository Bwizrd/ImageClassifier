import cv2
import numpy as np

def preprocess_image(image):

    # Apply Gaussian blur to reduce noise
    blurred_image = cv2.GaussianBlur(image, (5, 5), 0)

    # Apply morphological operations to clean the image
    kernel = np.ones((3, 3), np.uint8)  # Adjust kernel size as needed
    opened_image = cv2.morphologyEx(blurred_image, cv2.MORPH_OPEN, kernel)  # Erosion followed by dilation

    return opened_image