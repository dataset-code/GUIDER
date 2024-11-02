from keras.layers import Dense, Dropout, LSTM, Embedding
from keras.models import Sequential
from keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
import numpy as np

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=10000)
x_train = sequence.pad_sequences(x_train, maxlen=300)
x_test = sequence.pad_sequences(x_test, maxlen=300)

model = Sequential()
model.add(Dense(12, activation='sigmoid'))  # hidden layer
# model.add(Embedding(300, 12))
# model.add(LSTM(12))
model.add(Dense(2, activation='softmax'))  # output layer

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=["accuracy"])
model.fit(x_train, y_train, epochs=1, validation_data=(x_test, y_test))
