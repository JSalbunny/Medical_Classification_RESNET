import os
import numpy as np 
from PIL import Image


""" import tensorflow as tf
from tf.keras import Dense, layers, Sequential
from tensorflow.keras.utils import to_categorical """

""" dermoscopy and blood cell microscopy (input data either comes from one or another)
and that usually when the prediction is wrong, it think it's another class from the same input 
that means that for example the cnn will always predict correctly if it's in the dermoscopy or in the blood cell microscopy.
"""
#   classes : 
#   Nevu                (dermoscopy)
#   Melanoma            (dermoscopy)
#   Vascular lesions    (dermoscopy)
#   granulocytes        (Blood cell miscroscopy)
#   basophils           (Blood cell miscroscopy)
#   lymphocytes         (Blood cell miscroscopy)

x_train = np.load(r'C:\Users\Salbunny\Desktop\class_problem\Files\Xtrain_classification2.npy')
y_train = np.load(r'C:\Users\Salbunny\Desktop\class_problem\Files\Ytrain_classification2.npy')

enhance_output_directory = r"C:\Users\Salbunny\Desktop\class_problem\Images\enhanced"
original_image_directory = r"C:\Users\Salbunny\Desktop\class_problem\Images\original"

classification_index = {1 :'Nevu' , 2 :'Melanoma', 3 :'Vascular Lesions' , 4 :'Basophils' , 5 :'Lympocytes'}

classes = len(classification_index)

class_counts = {class_name: 0 for class_name in classification_index.values()}

data_count = enumerate(x_train)


nevu_count = 0
melanoma_count = 0
vascular_lesions_count = 0 
basophils_count = 0
lympocytes_count = 0
granulocytes_count = 0 

y_train = np.ravel(y_train).astype('int')

print(y_train)

for value in y_train:
    
    if value == 0:
        nevu_count += 1
        
    elif value == 1:
        melanoma_count += 1

    elif value == 2:
        vascular_lesions_count += 1 
    
    elif value == 3:
        granulocytes_count += 1     
        
    elif value == 4: 
        basophils_count += 1
    
    elif value == 5: 
        lympocytes_count += 1


print(f"The numbers are as followed:  \nData : {len(x_train)}\n")

print(f"Nevu : {nevu_count}\n")                          #5362
print(f"Melanoma : {melanoma_count}\n")                  #890
print(f"Vascular lesions : {vascular_lesions_count}\n")  #116
print(f"granulocytes : {granulocytes_count}\n")          #2305
print(f"Basophils : {basophils_count}\n")                #990
print(f"lympocytes : {lympocytes_count}\n")              #966 






    