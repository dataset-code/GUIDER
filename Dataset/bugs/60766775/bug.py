from keras.models import Sequential
from keras import layers
from keras.layers import Dropout
from sklearn.datasets import make_regression
from sklearn.preprocessing import StandardScaler
from keras import optimizers
import numpy as np
from sklearn.model_selection import train_test_split

X_train, y_train = make_regression(1000, 8, n_targets=4,random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.25, random_state=42)

scaler = StandardScaler()
input_shape = X_train[0].shape
x_train_std = scaler.fit_transform(X_train)
x_test_std = scaler.fit_transform(X_test)

model = Sequential()
model.add(layers.Dense(32, activation='sigmoid', input_shape=input_shape))
model.add(Dropout(0.1))
model.add(layers.Dense(20, activation='sigmoid'))
model.add(Dropout(0.1))
model.add(layers.Dense(15, activation='sigmoid'))
model.add(Dropout(0.1))

model.add(layers.Dense(4, activation='softmax'))
sgd = optimizers.SGD(learning_rate=0.01, momentum=0.87, nesterov=True)
model.compile(loss='mean_squared_error',
              optimizer=sgd)
history = model.fit(x_train_std, y_train, validation_split=0.1, epochs=100, batch_size=1)