from keras.models import Model
from keras.layers import Input
from keras.layers import Dense, Dropout
import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

X, y = make_regression(n_samples=1000, n_features=3, n_targets=1, random_state=420)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

input_layer = Input(shape=(3,))
x = Dense(32, activation="sigmoid")(input_layer)
x = Dropout(0.5)(x)
output = Dense(1, activation="relu")(x)
model = Model(input_layer, output)
model.compile("adam", "mse", metrics=["mse"])

model.fit(X_train, y_train, epochs=10, verbose=1)