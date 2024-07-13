from keras.models import Sequential
from keras.layers import Activation
from keras.layers import Dense
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import numpy as np

X_train, y_train = make_classification(1000, 7, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, train_size=.75, random_state=2)

model = Sequential()
model.add(Dense(16, input_shape=(7,)))
model.add(Activation('relu'))
model.add(Activation('relu'))
model.add(Dense(1))  # 2 outputs possible (ok or nok)
model.add(Activation('relu'))
model.compile(loss='mse', optimizer='sgd', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=5)
