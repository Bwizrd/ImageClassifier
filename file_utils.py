import os
import string

def sanitize_filename(filename):
    """ Sanitize a string to be safe for filenames. """
    valid_chars = "-_.() %s%s" % (string.ascii_letters, string.digits)
    return ''.join(c for c in filename if c in valid_chars).strip()

def ensure_directory(path):
    """ Ensure the specified directory exists, creating it if necessary. """
    if not os.path.exists(path):
        os.makedirs(path)
        
# Create a save path and ensure the directory exists
def create_save_path(base_path, subfolder):
    # Join the base path with the subfolder
    full_path = os.path.join(base_path, subfolder)
    
    # If the directory doesn't exist, create it
    if not os.path.exists(full_path):
        os.makedirs(full_path)
        
    return full_path