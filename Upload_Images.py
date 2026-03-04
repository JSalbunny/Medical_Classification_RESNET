import os
import numpy as np 
from PIL import Image
from Data_2 import x_train, original_image_directory

# upload images from vectors to pil images (image.png)

i = 0
for pixel_array in x_train:
    image = np.array(pixel_array).reshape(28, 28, 3)

    # Create a PIL Image
    pil_image = Image.fromarray(image.astype('uint8'))
    image_file_path = os.path.join(original_image_directory, f"image_{i}.png")
    pil_image.save(image_file_path)
    i += 1
    print(f"Saved image {i}")
    
 


