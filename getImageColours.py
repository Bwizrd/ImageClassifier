import cv2
import numpy as np
import matplotlib.pyplot as plt

def display_image(image, title='Original Image'):
    # Convert BGR images to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(15, 5))

    plt.subplot(131)
    # plt.imshow(image)
    plt.imshow(image_rgb)
    plt.title(title)
    plt.axis('off')

    plt.show()

image = cv2.imread('longColours.jpg')
# hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

height = image.shape[0]
half_height = height // 2

# Get the upper and lower halves of the image
upper_half = image[:half_height, :]
lower_half = image[half_height:, :]

# Calculate the average color of each half
average_color_upper = np.mean(upper_half, axis=(0, 1))
average_color_lower = np.mean(lower_half, axis=(0, 1))

average_color_upper = np.uint8(average_color_upper)
average_color_lower = np.uint8(average_color_lower)

# Create an image of the average color for each half
average_upper_img = np.ones((100, 100, 3), dtype=np.uint8) * average_color_upper[np.newaxis, np.newaxis, :]
average_lower_img = np.ones((100, 100, 3), dtype=np.uint8) * average_color_lower[np.newaxis, np.newaxis, :]


display_image(image)
display_image(average_upper_img, title='Average Color of Upper Half')
display_image(average_lower_img, title='Average Color of Lower Half')

# Print out the average colors
print(f'Average color of the upper half (BGR): {average_color_upper}')
print(f'Average color of the lower half (BGR): {average_color_lower}')