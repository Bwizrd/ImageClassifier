import cv2
import matplotlib.pyplot as plt
import os
from file_utils import sanitize_filename, ensure_directory

def display_image(image, title='Image', save_image=False, save_path=None, filename=None):
    """ Display an image and optionally save it. """
    # Convert BGR to RGB for display
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Display the image
    plt.figure(figsize=(8, 6))
    plt.imshow(image_rgb)
    plt.title(title)
    plt.axis('off')
    plt.show()

    if save_image and save_path and filename:
        # Sanitize the filename and ensure the save directory exists
        ensure_directory(save_path)

        # Construct the full save path
        sanitized_title = sanitize_filename(title)
        save_filename = f"{filename}_{sanitized_title}.png"
        full_path = os.path.join(save_path, save_filename)

        # Save the image
        cv2.imwrite(full_path, cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
        print(f"Image saved as {full_path}")

def display_mask(mask, title='Mask'):
    plt.figure(figsize=(8, 8))
    plt.imshow(mask,)
    plt.title(title)
    plt.axis('off')
    plt.show()