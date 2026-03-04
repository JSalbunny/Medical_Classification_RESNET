from PIL import Image
import numpy as np
import os

images_directory = "Images/enhanced"
vectors_directory = "Files"

def extract_number_from_filename(filename):
    # Extract the numeric part from the filename, which indicates the index of the image in the y-vector
    return int(''.join(filter(str.isdigit, filename)))

def images_to_vectors(images_directory, vectors_directory):
    x_data_enhanced = []

    # Get a list of image filenames and sort them based on the numeric part
    image_filenames = [filename for filename in os.listdir(images_directory) if filename.endswith(".png")]
    image_filenames.sort(key=extract_number_from_filename)  # Sort based on the numeric part

    # Import the images as vectors in the desired order
    for filename in image_filenames:
        file_path = os.path.join(images_directory, filename)
        print(file_path)

        image = Image.open(file_path)
        image_vector = np.asarray(image)

        x_data_enhanced.append(image_vector)

    np.save(f'{vectors_directory}/Xtrain_Classification2_enhanced.npy', x_data_enhanced)

images_to_vectors(images_directory, vectors_directory)
