import cv2
import numpy as np
import matplotlib.pyplot as plt
import uuid
import os
import string

def sanitize_filename(filename):
    """ Sanitize the string to be safe for filenames. """
    valid_chars = "-_.() %s%s" % (string.ascii_letters, string.digits)
    filename = ''.join(c for c in filename if c in valid_chars)
    return filename.strip()

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

    sanitized_title = sanitize_filename(title)
    save_path='saved_images'
    # Generate a random filename and save the image
    random_filename = f"{sanitized_title}_{uuid.uuid4()}.png"
    full_path = os.path.join(save_path, random_filename)
    cv2.imwrite(full_path, image)  # Save the original BGR image

    print(f"Image saved as {full_path}")


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
# target_image = cv2.imread('longTest.jpg')
# target_image = cv2.imread('shortTest.jpg')


# Find the colors in the image
combined_mask, result = find_colors(target_image, target_colors_bgr, tolerances)

assert cv2.countNonZero(combined_mask) > 0, "The mask is completely black."
# Display the mask and result
# ... (you can use the display_image function here)
# display_image(target_image)
display_image(combined_mask, 'Mask')
# display_image(result, 'Result')

# Use morphological opening to remove noise: erosion followed by dilation
kernel = np.ones((5, 5), np.uint8)  # The size of the kernel affects the strength of the noise removal
cleaned_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)

# Use morphological closing to close small holes: dilation followed by erosion
cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_CLOSE, kernel)

contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# for i, contour in enumerate(contours):
    # print(f"Contour #{i}: {contour}")

# Optionally, draw the contours over the image to visualize them
# contour_image = target_image.copy()
# cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 3)  # Draw in green

# # display_image(contour_image, 'Contours')

# Initialize an image with black background
shape_image = np.zeros_like(target_image)


# # Fill the contours with the color of the mask
# for contour in contours:
#     # Create a mask for the current contour
#     contour_mask = np.zeros_like(combined_mask)  # Make sure the mask is the same size as the combined_mask
#     cv2.drawContours(contour_mask, [contour], -1, 255, -1)  # Draw the contour in white

#     # Get the mean color of the contour from the original image using the contour mask
#     mean_val = cv2.mean(target_image, mask=contour_mask)
#     mean_color = (int(mean_val[0]), int(mean_val[1]), int(mean_val[2]))
    
#     # Fill the contour with the mean color on the shape image
#     cv2.drawContours(shape_image, [contour], -1, mean_color, -1)


# # Display the new image
# display_image(shape_image, 'Shape Image')

debug_image = target_image.copy()

# Draw each contour with a unique color and label
for i, contour in enumerate(contours):
    # Random color
    color = np.random.randint(0, 255, size=(3,)).tolist()
    cv2.drawContours(debug_image, [contour], -1, color, 3)
    
    # Label contours
    x, y, w, h = cv2.boundingRect(contour)
    cv2.putText(debug_image, f"#{i}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

# Display the image with labeled contours
display_image(debug_image, 'Debug Contours')

# Now fill contours and show the result
shape_image = np.zeros_like(target_image)
for contour in sorted(contours, key=cv2.contourArea, reverse=True):  # Draw larger contours first
    mask = np.zeros_like(combined_mask)
    cv2.drawContours(mask, [contour], -1, 255, -1)
    mean_val = cv2.mean(target_image, mask=mask)
    mean_color = (int(mean_val[0]), int(mean_val[1]), int(mean_val[2]))
    cv2.drawContours(shape_image, [contour], -1, mean_color, -1)

# Display the shape image
display_image(shape_image, 'Shape Image')


# Initialize a black image
labeled_image = np.zeros_like(target_image)

# Iterate over all the contours
for i, contour in enumerate(contours):
    # Create a mask for the current contour
    contour_mask = np.zeros_like(combined_mask)
    cv2.drawContours(contour_mask, [contour], -1, 255, -1)
    
    # Get the mean color of the contour from the original image using the contour mask
    mean_val = cv2.mean(target_image, mask=contour_mask)
    
    # Get the bounding box of the contour
    x, y, w, h = cv2.boundingRect(contour)

    # Identify if the contour is red or green based on mean color values
    label = "Green" if mean_val[1] > mean_val[2] else "Red"
    label += f"[{i}]"

    # Calculate the position to put the label (roughly at the center of the box)
    label_position = (x + w // 2, y + h // 2)

    # Draw the bounding box on the image
    color = (0, 255, 0) if "Green" in label else (0, 0, 255)
    cv2.rectangle(labeled_image, (x, y), (x + w, y + h), color, 2)
    
    # Draw the label in white for visibility
    cv2.putText(labeled_image, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, (255, 255, 255), 1)

# Display the labeled image
display_image(labeled_image, 'Labeled Contours on Black Background')

# Initialize a black image
filtered_image = np.zeros_like(target_image)

# Tolerance for width difference in pixels
width_tolerance = 10

# Contour pairs that match the criteria
matching_pairs = []

# Iterate over all the contours to find matching pairs
for i, contour_a in enumerate(contours):
    x_a, y_a, w_a, h_a = cv2.boundingRect(contour_a)
    
    for j, contour_b in enumerate(contours):
        # Avoid comparing the same contour
        if i == j:
            continue
        
        x_b, y_b, w_b, h_b = cv2.boundingRect(contour_b)
        
        # Check if contours a and b have similar widths and one is above the other
        if abs(w_a - w_b) <= width_tolerance:
            # Check if contour_a is above contour_b
            if y_a + h_a < y_b:
                matching_pairs.append((contour_a, contour_b))
            # Or if contour_b is above contour_a
            elif y_b + h_b < y_a:
                matching_pairs.append((contour_b, contour_a))

# Draw the matching pairs onto the filtered image
for pair in matching_pairs:
    for contour in pair:
        # Get the mean color to determine if it's red or green
        mask = np.zeros_like(combined_mask)
        cv2.drawContours(mask, [contour], -1, 255, -1)
        mean_val = cv2.mean(target_image, mask=mask)
        color = (0, 255, 0) if mean_val[1] > mean_val[2] else (0, 0, 255)
        
        # Draw the contour
        cv2.drawContours(filtered_image, [contour], -1, color, thickness=cv2.FILLED)

# Display the filtered image with only the appropriate boxes
display_image(filtered_image, 'Filtered Contours')