import os
import cv2
import numpy as np
import shutil  # For moving files

# Define a function to check if an image has a black or white background
def check_background_color(image):
    # Flatten the image into a single array of pixels
    flattened_pixels = image.reshape((-1, 3))

    # Calculate the mean color across the image
    mean_color = np.mean(flattened_pixels, axis=0)

    # Set a threshold to distinguish between black and white
    threshold = 127  # The midpoint for 8-bit colors

    # Determine if the mean color is predominantly black or white
    if np.all(mean_color < threshold):
        return "Black"
    elif np.all(mean_color > threshold):
        return "White"
    else:
        return "Mixed"

# Define the folder containing the image files
folder_path = 'data/downloaded_images/landscape'

# Define paths to light and dark subfolders
light_folder = os.path.join(folder_path, 'light')
dark_folder = os.path.join(folder_path, 'dark')

# Create the subfolders if they don't already exist
os.makedirs(light_folder, exist_ok=True)
os.makedirs(dark_folder, exist_ok=True)

# Loop through all image files in the folder
for filename in os.listdir(folder_path):
    # Check if the file is an image
    if filename.endswith(('.jpg', '.jpeg', '.png')):
        # Read the image
        image_path = os.path.join(folder_path, filename)
        image = cv2.imread(image_path)

        # Determine the background color
        background_color = check_background_color(image)

        # Determine the new path based on the background color
        if background_color == "Black":
            new_path = os.path.join(dark_folder, filename)
        elif background_color == "White":
            new_path = os.path.join(light_folder, filename)
        else:
            continue  # Skip if background is mixed or undefined

        # Move the image to the appropriate subfolder
        shutil.move(image_path, new_path)
        print(f"Moved {filename} to {new_path}")
