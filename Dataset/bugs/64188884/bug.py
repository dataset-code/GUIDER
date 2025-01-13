import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.optimizers import Adam, SGD
from keras.utils import to_categorical
from keras.layers import Conv2D, Dropout, Dense, Flatten
import matplotlib.pyplot as plt
import cv2
import os

NUM_CLASSES = 5
IMG_SIZE = 150
DAISY = 'flowers/daisy'
DANDELION = 'flowers/dandelion'
ROSE = 'flowers/rose'
SUNFLOWER = 'flowers/sunflower'
TULIP = 'flowers/tulip'

x = []
y = []

def train_data_gen(DIR, ID):
    for img in os.listdir(DIR):
        try:
            path = DIR + '/' + img
            img = plt.imread(path)
            img = cv2.resize(img,(IMG_SIZE,IMG_SIZE))
            x.append(img)
            y.append(ID)
        except:
            None

train_data_gen(DAISY, 0)
train_data_gen(DANDELION, 1)
train_data_gen(ROSE, 2)
train_data_gen(SUNFLOWER, 3)
train_data_gen(TULIP, 4)

x = np.array(x)
y = to_categorical(y,num_classes = 5)

x_train,x_test,y_train,y_test = train_test_split(x, y, test_size = 0.15)
x_train,x_val,y_train,y_val = train_test_split(x_train, y_train, test_size = 0.15)

datagen = ImageDataGenerator(
    featurewise_center=False,
    samplewise_center=False,
    samplewise_std_normalization=False,
    rotation_range=60,
    zoom_range = 0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    fill_mode = "reflect"
)

datagen.fit(x_train)

model = Sequential()
model.add(Conv2D(64, kernel_size=(3, 3), strides=2, activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)))
model.add(Dropout(0.5))
model.add(Conv2D(128, kernel_size=(3, 3), strides=2, activation='relu'))
model.add(Dropout(0.5))
model.add(Conv2D(128, kernel_size=(3, 3), strides=2, activation='relu'))
model.add(Dropout(0.5))
model.add(Conv2D(128, kernel_size=(3, 3), strides=2, activation='relu'))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dense(5, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit_generator(datagen.flow(x_train,y_train,batch_size=16), epochs=10, steps_per_epoch=x_train.shape[0]//16, validation_data=(x_val, y_val), verbose=1)
