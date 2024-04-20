import os
import uuid

def rename_files_in_directory(directory):
    # Loop through all files in the directory and its subdirectories
    for subdir, _, files in os.walk(directory):
        for file in files:
            # Ensure the file is a JPG image
            if file.lower().endswith(('.jpg', '.jpeg')):
                # Generate a new file name using UUID and maintain the .jpg extension
                file_extension = '.jpg' if file.lower().endswith('.jpg') else '.jpeg'
                new_name = f"{uuid.uuid4().hex[:8]}{file_extension}"
                # new_name = f"{uuid.uuid4().hex[:8]}.jpg"
                old_file_path = os.path.join(subdir, file)
                new_file_path = os.path.join(subdir, new_name)
                
                # Rename the file
                os.rename(old_file_path, new_file_path)
                print(f"Renamed {old_file_path} to {new_file_path}")

# Specify the path to your data folder
data_folder_path = 'data'

# Call the function for each category folder
for category in ['long', 'short', 'other']:
    category_path = os.path.join(data_folder_path, category)
    rename_files_in_directory(category_path)
