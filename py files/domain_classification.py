import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv1D, BatchNormalization, Activation, Dropout
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import os

## train algorithm to distinguish images in the three clusters
import timeit

start = timeit.default_timer()

############### retry wiht new version pip

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

tf.debugging.set_log_device_placement(True)

gpus = tf.config.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only use the first GPU
   
    tf.config.set_visible_devices(gpus[0], 'GPU')
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")



print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

new_model = Sequential()
new_model.add(ResNet50(include_top=False, pooling='avg', weights='imagenet'))

#new_model.add(Dense(1000, activation='relu', name='layer_1000'))
#new_model.add(BatchNormalization())
#new_model.add(Dense(3, activation='softmax', name='predictions'))  #0.938

#######################################################3
new_model.add(Dense(units = 1000, kernel_initializer = 'he_normal',
               use_bias = False))
new_model.add(BatchNormalization())
new_model.add(Activation('relu'))
new_model.add(Dropout(rate = 0.2))
new_model.add(Dense(units = 3, activation = 'softmax',
                kernel_initializer = 'he_normal'))


# Say not to train first layer (ResNet) model. It is already trained
new_model.layers[0].trainable = False

new_model.compile(optimizer = 'Adam', loss = 'categorical_crossentropy', metrics = ["accuracy"])

# Generators (train, val, test)
image_size = 224
data_generator = ImageDataGenerator(preprocessing_function=preprocess_input)

train_generator = data_generator.flow_from_directory(
        r'/mnt/c/Users/pietr/Desktop/ImageNet/DATA/domain/train_clusters',
        target_size=(image_size, image_size),  # try to add 3rd dimension (color)
        batch_size=24,
        class_mode='categorical')

validation_generator = data_generator.flow_from_directory(
        r'/mnt/c/Users/pietr/Desktop/ImageNet/DATA/domain/val_clusters',
        target_size=(image_size, image_size),
        class_mode='categorical')

test_generator = data_generator.flow_from_directory(
        r'/mnt/c/Users/pietr/Desktop/ImageNet/DATA/domain/test_clusters',
        target_size=(image_size, image_size),
        class_mode='categorical')



callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)

history=new_model.fit(
        train_generator,
        steps_per_epoch = train_generator.samples // 64,
        validation_data = validation_generator, 
        validation_steps = validation_generator.samples // 64,
        callbacks=[callback],
        epochs = 7) 


######################################################### Testing fitted resnet50
scores=new_model.evaluate_generator(generator=test_generator)

print("Accuracy ResNet TL= ", scores[1])


stop = timeit.default_timer()

print('Time: ', stop - start)  

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

model_history = pd.DataFrame(history.history)
model_history['epoch'] = history.epoch

fig, ax = plt.subplots(1, figsize=(8,6))
num_epochs = model_history.shape[0]

ax.plot(np.arange(0, num_epochs), model_history["accuracy"], 
        label="Training accuracy")
ax.plot(np.arange(0, num_epochs), model_history["val_accuracy"], 
        label="Validation accuracy")
ax.legend()

plt.tight_layout()
plt.show()