from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# Define the directory containing your original images
original_dir = '/path/to/original/images'

# Define the directory where augmented images will be saved
augmented_dir = '/path/to/augmented/images'

# Create a data generator with augmentation parameters
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Iterate over each image in the original directory and generate augmented images
for filename in os.listdir(original_dir):
    img = load_image(os.path.join(original_dir, filename))
    img = img.reshape((1,) + img.shape)  # Reshape to (1, height, width, channels) for flow method
    i = 0
    for batch in datagen.flow(img, batch_size=1, save_to_dir=augmented_dir, save_prefix='aug', save_format='jpg'):
        i += 1
        if i >= 5:  # Generate 5 augmented images per original image
            break  # Break the loop to prevent infinite loop
