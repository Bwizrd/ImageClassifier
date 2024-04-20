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


def find_colors(image, target_colors, tolerances):
    # Ensure input is in list format for multiple colors
    if not isinstance(target_colors[0], (list, np.ndarray)):
        target_colors = [target_colors]

    # Initialize an empty mask
    combined_mask = np.zeros(image.shape[:2], dtype="uint8")

    for target_color, tolerance in zip(target_colors, tolerances):
        # Convert the target color to a numpy array
        target_color = np.array(target_color, dtype=np.uint8)

        # Calculate the lower and upper bounds of the color range
        lower_bound = target_color - tolerance
        upper_bound = target_color + tolerance

        # Clip the bounds to be within valid color range
        lower_bound = np.clip(lower_bound, 0, 255)
        upper_bound = np.clip(upper_bound, 0, 255)

        # Find the colors within the specified range
        mask = cv2.inRange(image, lower_bound, upper_bound)

        # Combine the masks
        combined_mask = cv2.bitwise_or(combined_mask, mask)

    # Optionally, apply the combined mask to get the result image
    result = cv2.bitwise_and(image, image, mask=combined_mask)

    return combined_mask, result

# Define your target colors and tolerances
target_colors_bgr = [
    [230, 234, 205],  # Average color of the upper half
    [219, 215, 247]   # Average color of the lower half
]

# Define a tolerance level for color matching
tolerances = [
    np.array([15, 15, 15]),  # Tolerance for the first color
    np.array([15, 15, 15])   # Tolerance for the second color
]

# Load the image where you want to find these colors
target_image = cv2.imread('longTest2.jpg')


# Find the colors in the image
combined_mask, result = find_colors(target_image, target_colors_bgr, tolerances)

assert cv2.countNonZero(combined_mask) > 0, "The mask is completely black."
# Display the mask and result
# ... (you can use the display_image function here)
display_image(target_image)
display_image(combined_mask, 'Mask')
display_image(result, 'Result')