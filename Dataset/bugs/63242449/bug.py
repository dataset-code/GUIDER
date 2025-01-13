import numpy as np
import tensorflow as tf
import keras

train_x = np.array([1] * 1000 + [2] * 1000 + [3] * 1000)
train_x = tf.keras.utils.to_categorical(train_x - 1)
train_y = np.zeros((3000, 3))
train_y[:1000,0] = 1
train_y[1000:2000,1] = 1
train_y[2000:3000,2] = 1
val_x = train_x
val_y = train_y

model = keras.Sequential()
model.add(keras.layers.Dense(3, activation='relu'))
model.add(keras.layers.Dense(3, activation='relu'))
model.compile(optimizer=keras.optimizers.Adam(0.1),
             loss=keras.losses.categorical_crossentropy,
             metrics=[keras.metrics.categorical_accuracy])

model.fit(train_x, train_y, epochs = 10, batch_size = 32, verbose = 1,
          shuffle = False,
          validation_data=(val_x, val_y))
