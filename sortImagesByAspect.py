import os
import cv2
import shutil  # For moving files

# Function to classify aspect ratio
def classify_aspect_ratio(width, height):
    aspect_ratio = width / height
    
    # Define aspect ratio categories
    if aspect_ratio > 1.1:
        return "Landscape"  # Wider than tall
    elif aspect_ratio < 0.9:
        return "Portrait"  # Taller than wide
    else:
        return "Square"  # Approximately equal width and height

# Folder containing the image files
folder_path = 'data/downloaded_images'

# Loop through all image files in the folder
for filename in os.listdir(folder_path):
    if filename.endswith(('.jpg', '.jpeg', '.png')):
        # Read the image
        image_path = os.path.join(folder_path, filename)
        image = cv2.imread(image_path)
        
        # Get image dimensions
        height, width, _ = image.shape
        
        # Classify aspect ratio
        aspect_class = classify_aspect_ratio(width, height)
        
        # Define the destination folder based on aspect ratio classification
        destination_folder = os.path.join(folder_path, aspect_class)
        
        # Create the subfolder if it doesn't exist
        os.makedirs(destination_folder, exist_ok=True)
        
        # Define the new path for the image
        new_image_path = os.path.join(destination_folder, filename)
        
        # Move the image to the appropriate folder
        shutil.move(image_path, new_image_path)
        
        # Output the movement result
        print(f"Moved {filename} to {new_image_path}")
