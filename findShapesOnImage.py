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

    # Initialize masks for each color
    masks = []

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
        masks.append(mask)

    return masks


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
# target_image = cv2.imread('longTest2.jpg')
# target_image = cv2.imread('longTest.jpg')
# target_image = cv2.imread('shortTest.jpg')
# target_image = cv2.imread('data/long/2e639843.jpg')
target_image = cv2.imread('data/long/ef8dc3d6.jpg')


# Find the colors in the image
masks = find_colors(target_image, target_colors_bgr, tolerances)

# Now we have two masks, one for each color
mask_green, mask_red = masks

kernel = np.ones((5, 5), np.uint8)

# Apply morphological opening and closing to the red mask
cleaned_mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_OPEN, kernel)
cleaned_mask_red = cv2.morphologyEx(cleaned_mask_red, cv2.MORPH_CLOSE, kernel)

# Apply morphological opening and closing to the green mask
cleaned_mask_green = cv2.morphologyEx(mask_green, cv2.MORPH_OPEN, kernel)
cleaned_mask_green = cv2.morphologyEx(cleaned_mask_green, cv2.MORPH_CLOSE, kernel)

# Now find contours on the cleaned masks
contours_red, _ = cv2.findContours(cleaned_mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours_green, _ = cv2.findContours(cleaned_mask_green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Process each set of contours here...
# (e.g., draw them on separate images or analyze them as needed)

# Create a black image
combined_contour_image = np.zeros_like(target_image)

# Draw red contours
cv2.drawContours(combined_contour_image, contours_red, -1, (0, 0, 255), 1)

# Draw green contours
cv2.drawContours(combined_contour_image, contours_green, -1, (0, 255, 0), 1)

# Display the combined contour image
display_image(combined_contour_image, 'Combined Red and Green Contours')

# Function to check if one contour is above another within a given distance
def is_above_within_distance(above_contour, below_contour, max_distance=200):
    _, y_above, _, h_above = cv2.boundingRect(above_contour)
    x_below, y_below, w_below, _ = cv2.boundingRect(below_contour)

    # The bottom edge of the 'above' contour
    bottom_above = y_above + h_above

    # Check if the 'above' contour is indeed above the 'below' contour and within max_distance
    if bottom_above < y_below and (y_below - bottom_above) <= max_distance:
        return True
    return False

# Combine red and green contours for processing
all_contours = contours_red + contours_green
debug_image = target_image.copy()
for i, contour in enumerate(all_contours):
    # Random color
    color = np.random.randint(0, 255, size=(3,)).tolist()
    cv2.drawContours(debug_image, [contour], -1, color, 3)
    
    # Label contours
    x, y, w, h = cv2.boundingRect(contour)
    cv2.putText(debug_image, f"#{i}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

# Display the image with labeled contours
display_image(debug_image, 'Debug Contours')

vertically_aligned_contours = []

def check_vertical_alignment(index_above, index_below, contours, max_vertical_distance=200):
    xa, ya, wa, ha = cv2.boundingRect(contours[index_above])
    xb, yb, wb, hb = cv2.boundingRect(contours[index_below])

    # Bottom of the contour above
    bottom_above = ya + ha
    # Top of the contour below
    top_below = yb

    # Check vertical proximity
    if not (top_below > bottom_above and (top_below - bottom_above) < max_vertical_distance):
        return False

    # Check horizontal overlap by finding the horizontal range intersection
    # The horizontal range of the contour above
    horizontal_range_above = set(range(xa, xa + wa))
    # The horizontal range of the contour below
    horizontal_range_below = set(range(xb, xb + wb))
    
    # If there's no intersection in the horizontal ranges, they are not aligned
    if not horizontal_range_above.intersection(horizontal_range_below):
        return False

    return True


def auto_check_vertical_contours(contours, max_vertical_distance=200):
    contour_relationships = {index: False for index in range(len(contours))}

    for i in range(len(contours)):
        for j in range(len(contours)):
            if i != j:  # Ensure we're not comparing the same contour
                if check_vertical_alignment(i, j, contours, max_vertical_distance):
                    contour_relationships[i] = True  # Mark as having a valid contour beneath
                    print(f"Contour {i} is above Contour {j}")

    return [index for index, has_beneath in contour_relationships.items() if has_beneath]

# Run the automatic checking function and capture the result
vertically_aligned_contours = auto_check_vertical_contours(all_contours, 200)

# Create a black image to draw the final contours
final_contours_image = np.zeros_like(target_image)

# Draw only the contours that have been verified to have another directly beneath them
for index in vertically_aligned_contours:
    color = (0, 255, 0) if index < len(contours_green) else (0, 0, 255)
    cv2.drawContours(final_contours_image, [all_contours[index]], -1, color, 3)

# Display the final contours image
# display_image(final_contours_image, 'Filtered Contours')

def find_vertically_aligned_contours(all_contours, max_vertical_distance=200):
    vertically_aligned_pairs = []
    for i, contour_a in enumerate(all_contours[:-1]):  # Exclude the last contour to prevent index out of range
        for j, contour_b in enumerate(all_contours[i+1:]):  # Only check contours below contour_a
            if check_vertical_alignment(i, j, all_contours, max_vertical_distance):
                vertically_aligned_pairs.append((i, j))

    return vertically_aligned_pairs

def include_additional_contours(vertically_aligned_pairs, all_contours):
    # Initialize a set to keep track of individual indices
    horizontally_included_contours = set()
    for index_above, index_below in vertically_aligned_pairs:
        xa, ya, wa, ha = cv2.boundingRect(all_contours[index_above])
        xb, yb, wb, hb = cv2.boundingRect(all_contours[index_below])

        # Calculate horizontal range from the leftmost to the rightmost edges
        left_bound = min(xa, xb)
        right_bound = max(xa + wa, xb + wb)

        # Check for contours below or above the index_below contour that are within the horizontal bounds
        for k, contour_c in enumerate(all_contours):
            if k != index_above and k != index_below:
                xc, yc, wc, hc = cv2.boundingRect(contour_c)
                # Check if contour_c is horizontally within the bounds of the aligned pair
                if xc < right_bound and (xc + wc) > left_bound:
                    # If contour_c is below index_below and within horizontal bounds
                    if yc > yb + hb:
                        horizontally_included_contours.add(k)
                    # If contour_c is above index_above and within horizontal bounds
                    if yc + hc < ya:
                        horizontally_included_contours.add(k)

    return list(horizontally_included_contours)


# Apply the functions
vertically_aligned_pairs = find_vertically_aligned_contours(all_contours, max_vertical_distance=200)
additional_contours = include_additional_contours(vertically_aligned_pairs, all_contours)

vertically_aligned_indices = [index for pair in vertically_aligned_pairs for index in pair]

assert all(isinstance(idx, int) for idx in additional_contours), "additional_contours must be a list of integers"
# Combine the indices from vertically aligned contours with additional contours
final_indices = set(vertically_aligned_indices + list(additional_contours))
final_indices = sorted(final_indices)  # Now final_indices should only contain integers, which can be sorted

# # Combine the indices from vertically aligned pairs and additional contours
# final_indices = set(index for pair in vertically_aligned_pairs for index in pair) | set(additional_contours)
# final_indices = sorted(final_indices)  # Sort to maintain order

# Draw the final contours
final_contours_image = np.zeros_like(target_image)
for index in final_indices:
    color = (0, 255, 0) if index < len(contours_green) else (0, 0, 255)
    cv2.drawContours(final_contours_image, [all_contours[index]], -1, color, 3)

# Display the final image
display_image(final_contours_image, 'Final Contours Including Additional Contours')

# Create an image with a black background
shaded_contours_image = np.zeros_like(target_image)

# Draw and fill the contours with their original colors on the black background
for index in final_indices:
    # Create a mask for the current contour
    contour_mask = np.zeros_like(mask_red)
    cv2.drawContours(contour_mask, [all_contours[index]], -1, 255, -1)

    # Compute the mean color of the contour area in the original image
    mean_val = cv2.mean(target_image, mask=contour_mask)
    mean_color = (int(mean_val[0]), int(mean_val[1]), int(mean_val[2]))

    # Fill the contour with the mean color on the black background image
    cv2.drawContours(shaded_contours_image, [all_contours[index]], -1, mean_color, cv2.FILLED)

# Display the shaded contours image with black background
display_image(shaded_contours_image, 'Shaded Contours with Black Background')


