from keras.datasets import mnist
from keras import models
from keras import layers
from keras.utils import to_categorical
import time
import numpy as np

train_images = np.array([[0,0,0],[0,0,1],[0,1,0],[0,1,1],[1,0,0],[1,0,1],[1,1,0],[1,1,1]])
train_labels = np.array([[0],[1],[1],[1],[1],[1],[1],[1]])

train_images = train_images.reshape((8, 3))

model = models.Sequential()
model.add(layers.Dense(6, input_dim=3, activation='relu'))
model.add(layers.Dense(6, activation='relu'))
model.add(layers.Dense(1, activation='softmax'))

model.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=5, batch_size=4)
