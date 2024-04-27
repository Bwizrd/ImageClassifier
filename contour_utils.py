import cv2
import numpy as np

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

# Function to clean and combine masks
def clean_and_combine_masks(masks):
    # Unpack the masks
    mask_green_lighter, mask_green_darker, mask_red = masks

    # Define a kernel for morphological operations
    kernel = np.ones((5, 5), np.uint8)

    # Combine the green masks
    combined_green_mask = cv2.bitwise_or(mask_green_lighter, mask_green_darker)

    # Apply morphological opening and closing to the red mask
    cleaned_mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_OPEN, kernel)
    cleaned_mask_red = cv2.morphologyEx(cleaned_mask_red, cv2.MORPH_CLOSE, kernel)

    # Apply morphological opening and closing to the green mask
    cleaned_mask_green = cv2.morphologyEx(combined_green_mask, cv2.MORPH_OPEN, kernel)
    cleaned_mask_green = cv2.morphologyEx(cleaned_mask_green, cv2.MORPH_CLOSE, kernel)

    return cleaned_mask_green, cleaned_mask_red 



def find_contours_from_masks(mask):
    """ Find contours from a given mask. """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def filter_contours_by_area(contours, min_area=100):
    """ Filter contours based on minimum area. """
    return [contour for contour in contours if cv2.contourArea(contour) >= min_area]


def filter_contours_by_aspect_ratio(contours, max_aspect_ratio=4):
    """ Filter contours based on maximum aspect ratio (width/height). """
    return [contour for contour in contours if (cv2.boundingRect(contour)[2] / cv2.boundingRect(contour)[3]) <= max_aspect_ratio]


def check_vertical_alignment(contour1, contour2, max_vertical_distance=200):
    """ Check if one contour is above another within a specified distance. """
    _, y1, _, h1 = cv2.boundingRect(contour1)
    _, y2, _, h2 = cv2.boundingRect(contour2)

    return (y1 + h1) < y2 and (y2 - (y1 + h1)) <= max_vertical_distance


def find_vertically_aligned_contours(contours, max_vertical_distance=200):
    """ Find pairs of contours where one is vertically aligned with the other. """
    aligned_pairs = []
    for i, contour1 in enumerate(contours):
        for j, contour2 in enumerate(contours):
            if i != j and check_vertical_alignment(contour1, contour2, max_vertical_distance):
                aligned_pairs.append((i, j))
    return aligned_pairs

def classify_long_short_or_unknown(contours, image):
    red_contours, green_contours = [], []

    for contour in contours:
        mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, -1)
        mean_color = cv2.mean(image, mask=mask)

        if mean_color[2] > mean_color[1]:  # Red dominance
            red_contours.append(contour)
        else:  # Green dominance
            green_contours.append(contour)

    if not red_contours or not green_contours:
        return "unknown"
    
    print("RED", red_contours)
    print("GREEN", green_contours)

    # Determine vertical position
    red_top = min(cv2.boundingRect(c)[1] for c in red_contours)
    green_top = min(cv2.boundingRect(c)[1] for c in green_contours)

    if red_top < green_top:
        return "short"
    elif green_top < red_top:
        return "long"
    return "unknown"

def classify_long_short_or_unknown_black_background(contours, image):
    red_contours = []
    green_contours = []

    # Adjust tolerance for considering red or green dominance
    color_threshold = 20  # Adjust as needed
    
    for contour in contours:
        # Create a mask to get the mean color of the contour
        mask = np.zeros_like(image[:, :, 0], dtype=np.uint8)  # Only need 1 channel for the mask
        cv2.drawContours(mask, [contour], -1, 255, -1)
        
        # Compute the mean color
        mean_color = cv2.mean(image, mask=mask)
        
        # Consider pixels only above a certain brightness level to avoid black influences
        if mean_color[2] > mean_color[1] + color_threshold:
            red_contours.append(contour)
        elif mean_color[1] > mean_color[2] + color_threshold:
            green_contours.append(contour)
    
    # Validate that there are at least one red and one green contour
    if not red_contours or not green_contours:
        return "unknown"

    # Determine the topmost position for red and green contours
    red_topmost = min(cv2.boundingRect(c)[1] for c in red_contours)
    green_topmost = min(cv2.boundingRect(c)[1] for c in green_contours)
    
    # Classify based on the vertical position
    if red_topmost < green_topmost:
        return "short"
    elif green_topmost < red_topmost:
        return "long"
    
    return "unknown"