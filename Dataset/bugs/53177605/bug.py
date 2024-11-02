from keras import models
from keras import layers
import numpy as np
from random import randint

def get_output(a, b): return 0 if a == b else 1

def get_data ():
    data = []
    targets = []

    for _ in range(40000):
        a, b = randint(0, 1), randint(0, 1)

        targets.append(get_output(a, b))
        data.append([a, b])

    return data, targets

data, targets = get_data()

data = np.array(data).astype("float32")
targets = np.array(targets).astype("float32")

test_x = data[30000:]
test_y = targets[30000:]

train_x = data[:30000]
train_y = targets[:30000]

model = models.Sequential()

# input
model.add(layers.Dense(2, activation='relu', input_shape=(2,)))
# model.add(layers.Dense(8, activation="relu", input_dim=2))

# hidden
# model.add(layers.Dropout(0.3, noise_shape=None, seed=None))
model.add(layers.Dense(2, activation='relu'))
# model.add(layers.Dropout(0.2, noise_shape=None, seed=None))
model.add(layers.Dense(2, activation='relu'))

# output
model.add(layers.Dense(1, activation='sigmoid')) # sigmoid puts between 0 and 1

model.summary() # print out summary of model

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

res = model.fit(train_x, train_y, epochs=10, batch_size=200, validation_data=(test_x, test_y)) # train

