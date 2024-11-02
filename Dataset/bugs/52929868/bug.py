import tensorflow as tf
import numpy as np
import random
from matplotlib import pyplot as plt

img_height = 150
img_width = 150
batch_size = 8

class MyDataset(object):

    def __init__(self):
        placeholder = 0

    def generator(self):
        is_black = True
        X, y = [], []
        while True:
            if is_black:
                img = np.full((img_height, img_width, 3), 255)
            else:
                img = np.zeros((img_height, img_width, 3))
            img = img / 255.
            X.append(img)
            y.append(is_black)
            is_black = not is_black

            if len(X) >= batch_size:
                c = list(zip(X, y))
                random.shuffle(c)
                X, y = zip(*c)
                yield np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.float32)
                X, y = [], []

dataset = MyDataset()
sample_gen = dataset.generator()


# print(sample_gen)

model = tf.keras.Sequential()
model.add(tf.keras.layers.Conv2D(filters=1, kernel_size=(1,1), padding='same', 
                                 activation='relu', input_shape=(img_height, img_width, 3)))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(img_height//2,img_height//2)))

model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(1, activation='softmax'))

model.summary()

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit_generator(
    sample_gen, 
    steps_per_epoch = 100//batch_size , 
    epochs=30)
