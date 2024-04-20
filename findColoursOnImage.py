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


# Function to find the color within the specified range in the image
def find_color(image, target_color, tolerance):
    # Convert the target color to a numpy array
    target_color = np.array(target_color, dtype=np.uint8)

    # Calculate the lower and upper bounds of the color range
    lower_bound = target_color - tolerance
    upper_bound = target_color + tolerance

    # Clip the bounds to be within valid color range
    lower_bound = np.clip(lower_bound, 0, 255)
    upper_bound = np.clip(upper_bound, 0, 255)

    # Convert image to the HSV color space
    # hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Find the colors within the specified range
    # mask = cv2.inRange(hsv_image, lower_bound, upper_bound)
    mask = cv2.inRange(image, lower_bound, upper_bound)

    # Optionally, apply the mask to get the result image
    result = cv2.bitwise_and(image, image, mask=mask)

    return mask, result

# Your target color from the upper half average
target_color_bgr = [230, 234, 205]

# Load another image to find this color in
# target_image = cv2.imread('longColours.jpg')
target_image = cv2.imread('longTest2.jpg')

# Define a tolerance level for color matching (you may need to adjust this)
tolerance = np.array([15, 15, 15])  # Example tolerance

# Find the color in the other image
mask, result = find_color(target_image, target_color_bgr, tolerance)

assert cv2.countNonZero(mask) > 0, "The mask is completely black."
# Display the mask and result
# ... (you can use the display_image function here)
display_image(target_image)
display_image(mask, 'Mask')
display_image(result, 'Result')