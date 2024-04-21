import cv2
import numpy as np
import os
from contour_utils import find_colors, clean_and_combine_masks, classify_long_short_or_unknown
from display import display_image, display_mask
from file_utils import sanitize_filename, create_save_path
from image_utils import preprocess_image
import argparse


def preprocess_and_find_contours(image, target_colors_bgr, tolerances):
    # Find the colors in the image
    masks = find_colors(image, target_colors_bgr, tolerances)

    mask_green_light, mask_green_dark, mask_red_light, mask_red_dark = masks

    # Combine green masks and red masks
    combined_green_mask = cv2.bitwise_or(mask_green_light, mask_green_dark)
    combined_red_mask = cv2.bitwise_or(mask_red_light, mask_red_dark)

    # Find contours from the masks
    contours_green, _ = cv2.findContours(
        combined_green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_red, _ = cv2.findContours(
        combined_red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Combine all contours
    all_contours = contours_red + contours_green

    return all_contours


# Define your target colors and tolerances
target_colors_bgr = [
    [230, 234, 205],  # Light green
    [190, 200, 160],  # Dark green
    [219, 215, 247]   # Red
]

tolerances = [
    np.array([20, 20, 20]),  # Tolerance for the first color
    np.array([20, 20, 20]),
    np.array([15, 15, 15])   # Tolerance for the third color
]

# Process and classify the image


def process_and_classify_image(image_path):
    base_name = os.path.basename(image_path)
    file_name, _ = os.path.splitext(base_name)

    target_image = cv2.imread(image_path)
    if target_image is None:
        raise ValueError("Invalid image path or unable to read image.")

    display_image(target_image, 'Original Image', save_image=False, save_path=None, filename=None)
    preprocessed_image = preprocess_image(target_image)

    # Find colors
    masks = find_colors(preprocessed_image, target_colors_bgr, tolerances)

    cleaned_mask_green, cleaned_mask_red =  clean_and_combine_masks(masks)
        # Display the cleaned masks
    display_mask(cleaned_mask_red, 'Red Mask')
    display_mask(cleaned_mask_green, 'Green Mask')
    contours_red, _ = cv2.findContours(cleaned_mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_green, _ = cv2.findContours(cleaned_mask_green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    all_contours = contours_red + contours_green
    # Classify as long or short
    position = classify_long_short_or_unknown(all_contours, preprocessed_image)

    # Create folder and save image
    save_path = create_save_path("saved_images", position)
    sanitized_filename = sanitize_filename(file_name)

    save_file_path = os.path.join(
        save_path, f"{sanitized_filename}_classified.png")
    cv2.imwrite(save_file_path, preprocessed_image)

    print(f"Image saved as {save_file_path}")

    return position


# Command-line interface
if __name__ == "__main__":
    # Use argparse to parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Process an image to classify it as 'long', 'short', or 'unknown'.")
    parser.add_argument("image_path", type=str,
                        help="Path to the image to be processed.")

    # Parse the arguments
    args = parser.parse_args()

    # Process and classify the image based on the provided image path
    result = process_and_classify_image(args.image_path)

    print(result)  # Output the result to the console
