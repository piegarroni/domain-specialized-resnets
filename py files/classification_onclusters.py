import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, BatchNormalization, Dropout, Activation
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator


import os
import timeit

start = timeit.default_timer()


# train directly on cluster
# try with cleaned clusters (eliminate misclassification)
# find more efficient algoriothm to divide clusters


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

tf.debugging.set_log_device_placement(True)

gpus = tf.config.list_physical_devices('GPU')
if gpus:   
    tf.config.set_visible_devices(gpus[0], 'GPU')
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")



print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

new_model = Sequential()
new_model.add(ResNet50(include_top=False, pooling='avg', weights='imagenet'))

new_model.add(Dense(units = 1000, kernel_initializer = 'he_normal',
               use_bias = False))
new_model.add(BatchNormalization())
new_model.add(Activation('relu'))
new_model.add(Dropout(rate = 0.2))
new_model.add(Dense(units = 84, activation = 'softmax',
                kernel_initializer = 'he_normal'))


#new_model.add(Dense(84, activation='softmax', name='predictions'))

# Say not to train first layer (ResNet) model. It is already trained
new_model.layers[0].trainable = False

new_model.compile(optimizer = 'Adam', loss = 'categorical_crossentropy', metrics = ["accuracy"])

# Generators (train, val, test)
image_size = 224
data_generator = ImageDataGenerator(preprocessing_function=preprocess_input)


train_generator = data_generator.flow_from_directory(
        r'/mnt/c/Users/pietr/Desktop/ImageNet/DATA/train252_clust/2',
        target_size=(image_size, image_size),  # try to add 3rd dimension (color)
        batch_size=24,
        class_mode='categorical')

validation_generator = data_generator.flow_from_directory(
        r'/mnt/c/Users/pietr/Desktop/ImageNet/DATA/val252_clust/2',
        target_size=(image_size, image_size),
        class_mode='categorical')

accuracy_generator = data_generator.flow_from_directory(
        r'/mnt/c/Users/pietr/Desktop/ImageNet/DATA/testing_clust/val252/2',
        target_size=(image_size, image_size),
        class_mode='categorical')


darker_generator = data_generator.flow_from_directory(
        r'/mnt/c/Users/pietr/Desktop/ImageNet/DATA/testing_clust/val_dark/2',
        target_size=(image_size, image_size),
        class_mode='categorical')

brighter_generator = data_generator.flow_from_directory(
        r'/mnt/c/Users/pietr/Desktop/ImageNet/DATA/testing_clust/val_bright/2',
        target_size=(image_size, image_size),
        class_mode='categorical')

cropped_generator = data_generator.flow_from_directory(
        r'/mnt/c/Users/pietr/Desktop/ImageNet/DATA/testing_clust/val_crop/2',
        target_size=(image_size, image_size),
        class_mode='categorical')

noisy_generator = data_generator.flow_from_directory(
        r'/mnt/c/Users/pietr/Desktop/ImageNet/DATA/testing_clust/val_noisy/2',
        target_size=(image_size, image_size),
        class_mode='categorical')



callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)


'''
history = new_model.fit(
    train_generator,
    steps_per_epoch = train_generator.samples // 256,
    validation_data = validation_generator, 
    validation_steps = validation_generator.samples // 256,
    epochs = 15) # 2200-2300 seconds, 0.84 on val 0.76 on test
'''
history = new_model.fit(
    train_generator,
    steps_per_epoch = train_generator.samples // 128,
    validation_data = validation_generator, 
    validation_steps = validation_generator.samples // 64,
    epochs = 17) # 


####################################### Testing fitted resnet50
accuracy=new_model.evaluate_generator(generator=accuracy_generator)
bright=new_model.evaluate_generator(generator=brighter_generator)
dark=new_model.evaluate_generator(generator=darker_generator)
crop=new_model.evaluate_generator(generator=cropped_generator)
noisy=new_model.evaluate_generator(generator=noisy_generator)

stop = timeit.default_timer()

new_model.save(r'/mnt/c/Users/pietr/Thesis/models/baseline/')

print("Accuracy test ResNet TL= ", accuracy[1])
print("bright test ResNet TL= ", bright[1])
print("dark test ResNet TL= ", dark[1])
print("crop test ResNet TL= ", crop[1])
print("noisy test ResNet TL= ", noisy[1])

print('Time: ', stop - start)  


#################### plotting accuracy at every epoch

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


plt.style.use('ggplot')
model_history = pd.DataFrame(history.history)

model_history.to_csv(r'/mnt/c/Users/pietr/Thesis/history_cluster2.csv')

model_history['epoch'] = history.epoch

fig, ax = plt.subplots(1, figsize=(8,6))
num_epochs = model_history.shape[0]

ax.plot(np.arange(0, num_epochs), model_history["accuracy"], 
        label="Training accuracy")
ax.plot(np.arange(0, num_epochs), model_history["val_accuracy"], 
        label="Validation accuracy")
ax.legend()

plt.tight_layout()

plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.show()


