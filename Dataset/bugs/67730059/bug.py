from keras.layers import Dense, Input
from keras.models import Sequential, Model
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
import numpy as np
import time


X, y = make_regression(n_samples=2011, n_features=3, n_targets=5, random_state=42)
# print(X.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)


# model = Sequential()
# model.add(Dense(78, activation='relu', input_shape = (3,)))
# model.add(Dense(54, activation='relu'))
# model.add(Dense(54, activation='relu'))
# model.add(Dense(5, activation='relu'))

inputs = Input(shape=3) #(X.shape[1],)
out = Dense(78, activation='relu')(inputs)
out = Dense(54, activation='relu')(out)
out = Dense(54, activation='relu')(out)
out = Dense(5, activation='relu')(out)
model = Model(inputs=inputs, outputs=out)

model.compile(optimizer='rmsprop', loss='mse', metrics=['mean_squared_error'])
model.fit(X_train, y_train, epochs=5, verbose=1, validation_data=(X_test, y_test))