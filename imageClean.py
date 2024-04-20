import cv2
import numpy as np
import matplotlib.pyplot as plt

def display_images(image, mask, result):
    # Convert BGR images to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(15, 5))

    plt.subplot(131)
    plt.imshow(image_rgb)
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(132)
    plt.imshow(mask, cmap='gray')
    plt.title('Mask')
    plt.axis('off')

    plt.subplot(133)
    plt.imshow(result_rgb)
    plt.title('Result Image')
    plt.axis('off')

    plt.show()

# Load the image
image = cv2.imread('longTest2.jpg')
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Adjust these values
lower_green = np.array([40, 40, 40])  # Example: adjust these values
upper_green = np.array([80, 255, 255])  # Example: adjust these values

# Create mask
mask = cv2.inRange(hsv, lower_green, upper_green)
result = cv2.bitwise_and(image, image, mask=mask)

# Display images
display_images(image, mask, result)
