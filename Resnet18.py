import numpy as np
import tensorflow as tf
from sklearn.metrics import balanced_accuracy_score
from keras.callbacks import EarlyStopping
from Data_2 import y_train, x_train
from Enhance_Images import x_data_resized
from keras.utils import to_categorical
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
from keras.models import load_model

x_new_data = np.load(r'class_problem/Files/Xtrain_Classification2_enhanced.npy')

loaded_model = load_model('test_1')

# do a random split on the data to divide in train and test data, and normalize it
x_train_2, x_test_2, y_train_2, y_test_2 = train_test_split(x_train, y_train, test_size=0.2, random_state = 10)

x_train_2 = x_train_2.reshape(-1, 28, 28, 3)
x_train_2 = x_train_2 / 255.0  
print(x_train_2.shape)

x_test_2 = x_test_2.reshape(-1, 28, 28, 3)
x_test_2 = x_test_2 / 255.0

class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = dict(enumerate(class_weights))

# Convert labels to one-hot encoded format
num_classes = len(np.unique(y_train))

y_train_categorical = to_categorical(y_train_2, num_classes=num_classes)
y_test_categorical = to_categorical(y_test_2, num_classes=num_classes)

def residual_block(model, filters, stride):

    model = tf.keras.layers.Conv2D(filters, (3, 3), strides=stride, padding='same', use_bias=True)(model)
    model = tf.keras.layers.BatchNormalization()(model)
    model = tf.keras.layers.ReLU()(model)
    shortcut = model

    model = tf.keras.layers.Conv2D(filters, (3, 3), strides=stride, padding='same', use_bias=True)(model)
    model = tf.keras.layers.BatchNormalization()(model)
    model = tf.keras.layers.ReLU()(model)

    # shortcut
    if stride != 1 or model.shape[-1] != shortcut.shape[-1]:
        shortcut = tf.keras.layers.Conv2D(
            filters, (1, 1), strides=stride)(shortcut)

    model = tf.keras.layers.Add()([model, shortcut])
    model = tf.keras.layers.ReLU()(model)

    return model

def ResNet18(input_shape=(28, 28, 3)):
    input_tensor = tf.keras.layers.Input(shape=input_shape)

    model = tf.keras.layers.Conv2D(64, (3, 3), strides=(1, 1), padding='same', use_bias=True)(input_tensor)
    model = tf.keras.layers.BatchNormalization()(model)
    model = tf.keras.layers.ReLU()(model)
    model = tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(model)

    # Architecture of the resnet 18

    model = residual_block(model, 64, 1)
    model = residual_block(model, 64, 1)
    model = residual_block(model, 128, 2)
    model = residual_block(model, 128, 1)
    model = residual_block(model, 256, 2)
    model = residual_block(model, 256, 1)
    model = residual_block(model, 512, 2)
    model = residual_block(model, 512, 1)

    model = tf.keras.layers.GlobalAveragePooling2D()(model)

    # Dropout layer to deal with overfitting

    model = tf.keras.layers.Dropout(0.5)(model)

    # Activation Function

    model = tf.keras.layers.Dense(6, activation='softmax')(model)
    model = tf.keras.models.Model(inputs=input_tensor, outputs=model)

    return model


model = ResNet18(input_shape=(28, 28, 3))

loaded_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

callback = EarlyStopping(monitor = "val_loss", patience = 5, mode = "min" , restore_best_weights = True)

epochs = 20
batch_size = 64

model.fit(x_train_2, y_train_categorical, epochs=epochs, batch_size=batch_size, validation_data=(x_test_2, y_test_categorical), callbacks= callback, class_weight= class_weight_dict)

model.save('class_problem/trained_model')

# evaluating the training of the model
y_pred = model.predict(x_test_2)
y_pred_2 = (y_pred > 0.5).astype(int)

y_pred_arranged = y_pred_2.flatten()
y_test_categorical = y_test_categorical.reshape(-1, )

balanced_acc = balanced_accuracy_score(y_test_categorical, y_pred_arranged)

print(f'{balanced_acc}')

#load test data
x_test = np.load('class_problem/Files/Xtest_Classification2.npy')

x_test = x_test.reshape(-1, 28, 28, 3)
x_test = x_test / 255

y_pred_final = loaded_model.predict(x_test)
y_pred_final = (y_pred_final > 0.5).astype(int)
y_pred_final = np.argmax(y_pred_final, axis=1)

print(y_pred_final)

# save prediction
output_prediction = 'class_problem/Ytest_Classification2.npy'

np.save(output_prediction, y_pred_final)


balanced_acc = balanced_accuracy_score(y_test_categorical, y_pred_arranged)

print(f'{balanced_acc}')



