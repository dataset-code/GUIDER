from keras.layers import Dense
from keras.models import Sequential
import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

X, y = make_regression(n_samples=5008, n_features=6, n_targets=1, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = Sequential()
model.add(Dense(units = 64, input_dim = 6, kernel_initializer = 'uniform', activation='relu'))
model.add(Dense(units = 32, activation='relu'))
model.add(Dense(units = 16, activation='relu'))
model.add(Dense(1, activation='sigmoid')) #output layer

model.compile(optimizer = 'rmsprop', loss = 'mean_absolute_error')

history = model.fit(X_train, y_train, batch_size = 2048, epochs = 20, verbose=1)
