from keras.layers import Input, Dense
from keras.models import Sequential
import keras
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import numpy as np

X, y = make_classification(1000, 24, random_state=42, n_classes=5, n_informative=20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=40)

padding_len = 24 # len of each tokenized sentence
neurons = 16 # 2/3 the length of the text that is padded
model = Sequential()
model.add(Dense(neurons, input_dim = padding_len, activation = 'relu', name = 'hidden-1'))
model.add(Dense(neurons, activation = 'relu', name = 'hidden-2'))
model.add(Dense(neurons, activation = 'relu', name = 'hidden-3'))
model.add(Dense(1, activation = 'sigmoid', name = 'output_layer'))

model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs = 10, batch_size = 64)
