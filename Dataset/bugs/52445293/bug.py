from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from keras.layers import Dense, Dropout, LSTM
from keras.models import Sequential
import numpy as np

X, y = make_regression(n_samples=1000, n_features=13, n_targets=1, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

model = Sequential()
model.add(Dense(13, activation="tanh", input_dim = 13))
model.add(Dense(10, activation="tanh"))
model.add(Dropout(0.2))
model.add(Dense(6, activation="tanh"))
model.add(Dense(3, activation="tanh"))
model.add(Dense(1))
print(model.summary())
model.compile(loss='mean_squared_error',
    optimizer='adam',
    metrics=['accuracy'])
model.fit(X_train,y_train, epochs = 20 , batch_size= 10, validation_data=(X_test, y_test))
