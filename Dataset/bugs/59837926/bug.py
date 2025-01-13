from keras.layers import Dense
from keras.models import Sequential
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
import numpy as np

X, y = make_regression(n_samples=1000, n_features=3, n_targets=1, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)


model = Sequential([
     Dense(32, activation='relu', input_shape=(3,)),
     Dense(32, activation='relu'),
     Dense(1, activation='sigmoid'),
     ])
model.compile(loss='mse',
              optimizer='adam',
              metrics=['acc'])
hist = model.fit(X_train, y_train,
          batch_size=32, epochs=40,
          validation_data=(X_test, y_test)) 
