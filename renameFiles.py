import os

# Define the folder containing the files to be renamed
folder_path = 'data/downloaded_images/Landscape/dark/'

# Initialize a counter for the sequential numbering
counter = 1

# Loop through all files in the folder
for filename in os.listdir(folder_path):
    # Get the file extension to preserve it during renaming
    file_extension = os.path.splitext(filename)[1]

    # Define the new filename with a unique number
    new_filename = f"dark{counter}{file_extension}"

    # Get the full paths for the original and new filenames
    old_path = os.path.join(folder_path, filename)
    new_path = os.path.join(folder_path, new_filename)

    # Rename the file
    os.rename(old_path, new_path)

    # Increment the counter
    counter += 1

    # Output the renaming result
    print(f"Renamed {filename} to {new_filename}")
