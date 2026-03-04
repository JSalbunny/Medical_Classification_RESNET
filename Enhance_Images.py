import cv2, os
from Data_2 import enhance_output_directory, original_image_directory
import numpy as np

def interpolation():
    return cv2.INTER_CUBIC

# scales the width and height of each image by factor 2
# the upscaled images are not used in the submission because of the slow execution time when training the CNN


scale = 2

image_file_names = [f for f in os.listdir(original_image_directory) if f.endswith('.png')]

enhanced_images = []

for image_file_name in image_file_names:

    # read images
    image_path = os.path.join(original_image_directory, image_file_name)
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)

    image_resized = cv2.resize(image, None, fx=scale, fy=scale, interpolation=interpolation())
    
    enhanced_images.append(image_resized)
    image_resized_file_path = os.path.join(enhance_output_directory, image_file_name)

    # Save the enlarged image to a file
    cv2.imwrite(image_resized_file_path, image_resized)
    
print("Enhancing completed.")
x_enhanced = np.array(enhanced_images)    

x_data_resized = x_enhanced.astype('float32') / 255.0   

   