import time

import numpy as np

from keras.datasets import cifar10
from keras.models import Model, Sequential
from keras.layers import Dense, Dropout, Flatten, Convolution2D, MaxPooling2D, Input
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint, EarlyStopping

batch_size = 32
num_classes = 10
epochs = 2
STAMP = 'simple_cnn'

(x_train, y_train), (x_val, y_val) = cifar10.load_data()
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_val.shape[0], 'test samples')

y_train = np_utils.to_categorical(y_train, num_classes)
y_val = np_utils.to_categorical(y_val, num_classes)

x_train = x_train.astype('float32')
x_val = x_val.astype('float32')
x_train /= 255
x_val /= 255

model = Sequential()

# Block 1
model.add(Convolution2D(32, (7, 7), padding='same', activation='relu', input_shape=(x_train.shape[1:])))
model.add(Convolution2D(32, (7, 7), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# Block 2
model.add(Convolution2D(64, (3, 3), padding='same', activation='relu'))
model.add(Convolution2D(64, (3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# Block 3
model.add(Convolution2D(128, (1, 1), padding='same', activation='relu'))
model.add(Convolution2D(128, (1, 1), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# Dense layers
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))


model.summary()
model.compile(loss='categorical_crossentropy',
	optimizer='adam',
	metrics=['accuracy'])

# model_json = model.to_json()
# with open('model/' + STAMP + ".json", "w") as json_file:
#     json_file.write(model_json)
    
# early_stopping = EarlyStopping(monitor='val_loss',
# 	patience=10,
# 	verbose=1)

# bst_model_path = 'model/' + STAMP + '.h5'
# model_checkpoint = ModelCheckpoint(bst_model_path,
#     monitor='val_acc',
#     verbose=1,
#     save_best_only=True,
#     save_weights_only=True)

#================================================== Train the Model =============================================================
print('Start training.')
# start_time = time.time()

model.fit(x_train, y_train,
	batch_size=batch_size,
	epochs=epochs,
	validation_data=(x_val, y_val),
	shuffle=True)
