from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split


X, y = make_regression(n_samples=100, n_features=14, n_targets=1, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

input_layer = Input(shape=(14,))
hidden1 = Dense(128, activation='relu')(input_layer)
hidden2 = Dense(128, activation='relu')(hidden1)
output_layer = Dense(1, activation='relu')(hidden2)

model = Model(inputs=input_layer, outputs=output_layer)
model.compile(loss='mse', optimizer='Adam')


model.fit(X_train, y_train, epochs=100, batch_size=14)
